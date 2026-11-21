use memchr::{memchr, memchr2_iter};
use memmap2::Mmap;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering};
use mimalloc::MiMalloc;
use std::sync::Mutex;

mod parquet_writer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const MS_PER_DAY: i64 = 86_400_000;
const MS_30_MIN: i64 = 1_800_000;
const MS_7_DAYS: i64 = 604_800_000;
const PERSON_SHARDS: usize = 256;
const DEDUP_SHARDS: usize = 256;
const META_TAG: u32 = 1 << 31;

#[derive(Clone, Copy)]
struct DedupSlot {
    key: u64,
    ingest_sec: i32,
    loc: u32,
}
const _: [(); 16] = [(); std::mem::size_of::<DedupSlot>()];

#[inline]
fn parse_numeric_id(b: &[u8], prefix: &[u8], max: usize) -> Option<usize> {
    if !b.starts_with(prefix)
        || b.len() == prefix.len()
        || (b.len() > prefix.len() + 1 && b[prefix.len()] == b'0')
    {
        return None;
    }
    let mut value = 0usize;
    for &c in &b[prefix.len()..] {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(d as usize)?;
        if value > max {
            return None;
        }
    }
    (value > 0).then_some(value)
}

#[derive(Deserialize)]
struct Manifest {
    window_start: String,
    window_end: String,
}

#[derive(Clone, Copy)]
struct DateCache {
    entries: [(u64, i64); 32],
}

impl DateCache {
    fn new(window_start: i64, window_end: i64) -> Self {
        let mut entries = [(0u64, 0i64); 32];
        let lo = (window_start - 9 * MS_PER_DAY).div_euclid(MS_PER_DAY);
        let hi = (window_end + 8 * MS_PER_DAY).div_euclid(MS_PER_DAY);
        for day in lo..=hi {
            let ms = day * MS_PER_DAY;
            let (year, month, day_of_month) = civil_from_days(day);
            let prefix: [u8; 8] = format!("{year:04}-{month:02}-")
                .into_bytes()
                .try_into()
                .expect("date prefix length");
            {
                let idx = day_of_month as usize & 31;
                entries[idx] = (u64::from_ne_bytes(prefix), ms);
            }
        }
        Self { entries }
    }

    fn get_with_day(&self, b: &[u8], d: u32) -> Option<i64> {
        if d < 1 || d > 31 { return None; }
        let key = u64::from_ne_bytes(b[0..8].try_into().ok()?);
        let entry = self.entries[d as usize & 31];
        if entry.0 == key { Some(entry.1) } else { None }
    }

}

#[inline]
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    if m <= 2 { y += 1; }
    (y, m as u32, d as u32)
}

#[inline]
fn fast_shard(key: u64, shards: usize) -> usize {
    ((key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - shards.trailing_zeros())) as usize)
        & (shards - 1)
}

#[inline]
fn mix_key(mut key: u64) -> u64 {
    key ^= key >> 30;
    key = key.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    key ^= key >> 27;
    key = key.wrapping_mul(0x94d0_49bb_1331_11eb);
    key ^ (key >> 31)
}

#[inline]
fn dedup_hash(key: u64) -> u64 {
    if key >> 63 != 0 {
        return mix_key(key);
    }
    let folded = key ^ (key >> 18) ^ (key >> 36) ^ (key >> 54);
    folded.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

const fn make_base36_table(upper: bool) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0usize;
    while i < 10 {
        table[b'0' as usize + i] = i as u8 + 1;
        i += 1;
    }
    i = 0;
    while i < 26 {
        table[(if upper { b'A' } else { b'a' }) as usize + i] = i as u8 + 11;
        i += 1;
    }
    table
}

const BASE36_LOWER: [u8; 256] = make_base36_table(false);
const BASE36_UPPER: [u8; 256] = make_base36_table(true);

#[inline]
fn pack_fast_id(b: &[u8], lo_prefix: u8, hi_prefix: u8) -> Option<u64> {
    if !(2..=11).contains(&b.len()) {
        return None;
    }
    let lower = b[0] == lo_prefix;
    if !lower && b[0] != hi_prefix {
        return None;
    }
    let mut packed = if lower { 1u64 << 62 } else { 0 };
    let table = if lower { &BASE36_LOWER } else { &BASE36_UPPER };
    unsafe {
        let bytes = b.as_ptr();
        let lookup = table.as_ptr();
        for i in 1..b.len() {
            let value = *lookup.add(*bytes.add(i) as usize);
            if value == 0 {
                return None;
            }
            packed |= (value as u64) << (54 - (i - 1) * 6);
        }
    }
    Some(packed)
}

#[inline]
fn unpack_fast_id<'a>(val: u64, lo_prefix: u8, hi_prefix: u8, buf: &'a mut [u8; 11]) -> &'a [u8] {
    let lower = val & (1u64 << 62) != 0;
    buf[0] = if lower { lo_prefix } else { hi_prefix };
    let mut len = 1usize;
    for i in 0..10 {
        let code = ((val >> (54 - i * 6)) & 0x3f) as u8;
        if code == 0 { break; }
        buf[len] = match code {
            1..=10 => b'0' + code - 1,
            11..=36 if lower => b'a' + code - 11,
            11..=36 => b'A' + code - 11,
            _ => b'?',
        };
        len += 1;
    }
    &buf[..len]
}

struct IdPool {
    fallback: Mutex<(Vec<Box<[u8]>>, FxHashMap<Box<[u8]>, u64>)>,
}

impl IdPool {
    fn new() -> Self {
        Self { fallback: Mutex::new((Vec::new(), FxHashMap::default())) }
    }

    fn key(&self, bytes: &[u8], lo: u8, hi: u8) -> u64 {
        if let Some(key) = pack_fast_id(bytes, lo, hi) {
            return key;
        }
        let mut pool = self.fallback.lock().expect("fallback ID pool");
        if let Some(&key) = pool.1.get(bytes) {
            return key;
        }
        let idx = pool.0.len() as u64;
        assert!(idx < (1u64 << 63), "too many fallback IDs");
        let key = (1u64 << 63) | idx;
        let owned: Box<[u8]> = bytes.into();
        pool.0.push(owned.clone());
        pool.1.insert(owned, key);
        key
    }

    fn fallback_snapshot(&self) -> Vec<Box<[u8]>> {
        self.fallback.lock().expect("fallback ID pool").0.clone()
    }

    fn cmp_user(&self, a: u64, b: u64) -> std::cmp::Ordering {
        self.cmp_ids(a, b, b'u', b'U')
    }

    fn cmp_ids(&self, a: u64, b: u64, lo: u8, hi: u8) -> std::cmp::Ordering {
        if a == b { return std::cmp::Ordering::Equal; }
        if a >> 63 == 0 && b >> 63 == 0 {
            return a.cmp(&b);
        }
        let pool = self.fallback.lock().expect("fallback ID pool");
        let mut ab = [0u8; 11];
        let mut bb = [0u8; 11];
        let a_bytes: &[u8] = if a >> 63 != 0 {
            pool.0[(a & !(1u64 << 63)) as usize].as_ref()
        } else {
            unpack_fast_id(a, lo, hi, &mut ab)
        };
        let b_bytes: &[u8] = if b >> 63 != 0 {
            pool.0[(b & !(1u64 << 63)) as usize].as_ref()
        } else {
            unpack_fast_id(b, lo, hi, &mut bb)
        };
        a_bytes.cmp(b_bytes)
    }

}

#[inline]
fn parse_csv_line<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    parse_csv_line_fast::<N>(line)
}

#[inline]
fn parse_csv_line_fast<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    if line.is_empty() {
        return None;
    }
    let mut out = [&[][..]; N];
    let mut field = 0usize;
    let mut start = 0usize;
    let mut state = 0u8;

    let mut i = 0usize;
    while i < line.len() {
        match line[i] {
            b',' => {
                if field >= N || state == 1 {
                    return None;
                }
                if state == 2 {
                    if i < start + 2 || line[i - 1] != b'"' {
                        return None;
                    }
                    out[field] = &line[start + 1..i - 1];
                } else {
                    out[field] = &line[start..i];
                }
                field += 1;
                start = i + 1;
                state = 0;
            }
            b'"' => {
                if state == 0 {
                    if i != start {
                        return None;
                    }
                    state = 1;
                } else if state == 1 {
                    state = 2;
                } else {
                    return None;
                }
            }
            _ => {
                if state == 2 {
                    return None;
                }
            }
        }
        i += 1;
    }

    if field >= N {
        return None;
    }
    match state {
        0 => out[field] = &line[start..],
        2 => {
            if line.len() < start + 2 || line[line.len() - 1] != b'"' {
                return None;
            }
            out[field] = &line[start + 1..line.len() - 1];
        }
        _ => return None,
    }
    field += 1;
    if field == N { Some(out) } else { None }
}

#[inline]
fn finish_csv_line<const N: usize, F>(
    data: &[u8],
    start: usize,
    end: usize,
    commas: &[usize; N],
    comma_count: usize,
    has_quote: bool,
    callback: &mut F,
)
where
    F: for<'a> FnMut([&'a [u8]; N]),
{
    if start == end {
        return;
    }
    if comma_count != N.saturating_sub(1) {
        return;
    }

    if has_quote {
        let mut fields = [&[][..]; N];
        let mut field_start = start;
        for field in 0..N {
            let field_end = if field + 1 == N { end } else { commas[field] };
            let mut value = &data[field_start..field_end];
            if value.first() == Some(&b'"') {
                if value.len() < 2 || value.last() != Some(&b'"') {
                    return;
                }
                value = &value[1..value.len() - 1];
                if memchr(b'"', value).is_some() {
                    return;
                }
            } else if memchr(b'"', value).is_some() {
                return;
            }
            fields[field] = value;
            field_start = field_end + 1;
        }
        callback(fields);
        return;
    }

    let mut fields = std::mem::MaybeUninit::<[&[u8]; N]>::uninit();
    let out = fields.as_mut_ptr().cast::<&[u8]>();
    let mut field_start = start;
    for (field, &comma) in commas[..comma_count].iter().enumerate() {
        unsafe { out.add(field).write(&data[field_start..comma]); }
        field_start = comma + 1;
    }
    unsafe {
        out.add(comma_count).write(&data[field_start..end]);
        callback(fields.assume_init());
    }
}

fn scan_csv_morsel<const N: usize, F>(data: &[u8], mut callback: F)
where
    F: for<'a> FnMut([&'a [u8]; N]),
{
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    unsafe {
        scan_csv_morsel_avx512::<N, F>(data, &mut callback);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
    scan_csv_morsel_scalar::<N, F>(data, &mut callback);
}

fn scan_csv_morsel_scalar<const N: usize, F>(data: &[u8], callback: &mut F)
where
    F: for<'a> FnMut([&'a [u8]; N]),
{
    let mut line_start = 0usize;
    let mut commas = [0usize; N];
    let mut comma_count = 0usize;
    let mut comma_overflow = false;
    for pos in memchr2_iter(b'\n', b',', data) {
        if data[pos] == b',' {
            if comma_count < N {
                commas[comma_count] = pos;
                comma_count += 1;
            } else {
                comma_overflow = true;
            }
        } else {
            if !comma_overflow {
                finish_csv_line::<N, F>(data, line_start, pos, &commas, comma_count, true, callback);
            }
            line_start = pos + 1;
            comma_count = 0;
            comma_overflow = false;
        }
    }
    if !comma_overflow {
        finish_csv_line::<N, F>(data, line_start, data.len(), &commas, comma_count, true, callback);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
unsafe fn scan_csv_morsel_avx512<const N: usize, F>(data: &[u8], callback: &mut F)
where
    F: for<'a> FnMut([&'a [u8]; N]),
{
    use std::arch::x86_64::{
        __m512i, _mm512_cmpeq_epi8_mask, _mm512_loadu_si512, _mm512_set1_epi8,
    };
    let newline = _mm512_set1_epi8(b'\n' as i8);
    let comma = _mm512_set1_epi8(b',' as i8);
    let quote = _mm512_set1_epi8(b'"' as i8);
    let mut line_start = 0usize;
    let mut commas = [0usize; N];
    let mut comma_count = 0usize;
    let mut comma_overflow = false;
    let mut has_quote = false;
    let mut block_start = 0usize;
    while block_start + 64 <= data.len() {
        let block = _mm512_loadu_si512(data.as_ptr().add(block_start) as *const __m512i);
        let mut newline_mask = _mm512_cmpeq_epi8_mask(block, newline);
        let mut comma_mask = _mm512_cmpeq_epi8_mask(block, comma);
        let mut quote_mask = _mm512_cmpeq_epi8_mask(block, quote);
        let mut mask = newline_mask | comma_mask;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            let pos = block_start + bit;
            let bit_mask = 1u64 << bit;
            if newline_mask & bit_mask != 0 {
                let through_line = if bit == 0 { 0 } else { (1u64 << bit) - 1 };
                has_quote |= quote_mask & through_line != 0;
                if !comma_overflow {
                    finish_csv_line::<N, F>(data, line_start, pos, &commas, comma_count, has_quote, callback);
                }
                line_start = pos + 1;
                comma_count = 0;
                comma_overflow = false;
                has_quote = false;
                let through_newline = if bit == 63 { u64::MAX } else { (1u64 << (bit + 1)) - 1 };
                quote_mask &= !through_newline;
                newline_mask &= !bit_mask;
            } else if comma_mask & bit_mask != 0 {
                if comma_count < N {
                    commas[comma_count] = pos;
                    comma_count += 1;
                } else {
                    comma_overflow = true;
                }
                comma_mask &= !bit_mask;
            }
            mask &= mask - 1;
        }
        has_quote |= quote_mask != 0;
        block_start += 64;
    }
    for pos in block_start..data.len() {
        match data[pos] {
            b'"' => has_quote = true,
            b',' => {
                if comma_count < N {
                    commas[comma_count] = pos;
                    comma_count += 1;
                } else {
                    comma_overflow = true;
                }
            }
            b'\n' => {
                if !comma_overflow {
                    finish_csv_line::<N, F>(data, line_start, pos, &commas, comma_count, has_quote, callback);
                }
                line_start = pos + 1;
                comma_count = 0;
                comma_overflow = false;
                has_quote = false;
            }
            _ => {}
        }
    }
    if !comma_overflow {
        finish_csv_line::<N, F>(data, line_start, data.len(), &commas, comma_count, has_quote, callback);
    }
}

#[inline]
fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

#[inline]
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let m_adj = if m > 2 { m - 3 } else { m + 9 } as u64;
    let doy = (153 * m_adj + 2) / 5 + (d as u64) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + (doe as i64) - 719468
}

#[inline]
fn parse_2d(b: &[u8]) -> Option<u32> {
    let d0 = b[0].wrapping_sub(b'0');
    let d1 = b[1].wrapping_sub(b'0');
    if d0 > 9 || d1 > 9 {
        return None;
    }
    Some((d0 as u32) * 10 + (d1 as u32))
}

#[inline]
unsafe fn parse_2d_at_unchecked(b: &[u8], offset: usize) -> Option<u32> {
    let d0 = (*b.get_unchecked(offset)).wrapping_sub(b'0');
    let d1 = (*b.get_unchecked(offset + 1)).wrapping_sub(b'0');
    if d0 > 9 || d1 > 9 {
        return None;
    }
    Some(d0 as u32 * 10 + d1 as u32)
}

#[inline]
fn parse_4d(b: &[u8]) -> Option<i64> {
    let hi = parse_2d(&b[0..2])? as i64;
    let lo = parse_2d(&b[2..4])? as i64;
    Some(hi * 100 + lo)
}

#[inline]
fn parse_iso_base(b: &[u8]) -> Option<i64> {
    if b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let y = parse_4d(&b[0..4])?;
    let m = parse_2d(&b[5..7])?;
    let d = parse_2d(&b[8..10])?;
    let hh = parse_2d(&b[11..13])?;
    let mm = parse_2d(&b[14..16])?;
    let ss = parse_2d(&b[17..19])?;

    if m < 1 || m > 12 || d < 1 || hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    let max_d = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap(y) { 29 } else { 28 },
        _ => unreachable!(),
    };
    if d > max_d {
        return None;
    }
    let days = days_from_civil(y, m, d);
    Some((days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64)) * 1000)
}

#[inline]
fn parse_iso_base_cached(b: &[u8], cache: &DateCache) -> Option<i64> {
    if b.len() < 19 {
        return None;
    }
    let (year_dash, month_dash, t, hour_colon, minute_colon) = unsafe {
        (
            *b.get_unchecked(4),
            *b.get_unchecked(7),
            *b.get_unchecked(10),
            *b.get_unchecked(13),
            *b.get_unchecked(16),
        )
    };
    if year_dash != b'-' || month_dash != b'-' || t != b'T' || hour_colon != b':' || minute_colon != b':' {
        return None;
    }
    let (d, hh, mm, ss) = unsafe {
        (
            parse_2d_at_unchecked(b, 8)?,
            parse_2d_at_unchecked(b, 11)?,
            parse_2d_at_unchecked(b, 14)?,
            parse_2d_at_unchecked(b, 17)?,
        )
    };
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    if let Some(day_ms) = cache.get_with_day(b, d) {
        return Some(day_ms + (hh as i64 * 3600 + mm as i64 * 60 + ss as i64) * 1000);
    }
    parse_iso_base(b)
}

#[inline]
fn parse_f1_ingest(b: &[u8], cache: &DateCache) -> Option<i64> {
    if b.len() != 20 {
        return None;
    }
    let (z, year_dash, month_dash, t, hour_colon, minute_colon) = unsafe {
        (
            *b.get_unchecked(19),
            *b.get_unchecked(4),
            *b.get_unchecked(7),
            *b.get_unchecked(10),
            *b.get_unchecked(13),
            *b.get_unchecked(16),
        )
    };
    if z != b'Z' || year_dash != b'-' || month_dash != b'-' || t != b'T'
        || hour_colon != b':' || minute_colon != b':'
    {
        return None;
    }
    let (d, hh, mm, ss) = unsafe {
        (
            parse_2d_at_unchecked(b, 8)?,
            parse_2d_at_unchecked(b, 11)?,
            parse_2d_at_unchecked(b, 14)?,
            parse_2d_at_unchecked(b, 17)?,
        )
    };
    if hh <= 23 && mm <= 59 && ss <= 59 {
        if let Some(day_ms) = cache.get_with_day(b, d) {
            return Some(day_ms + (hh as i64 * 3600 + mm as i64 * 60 + ss as i64) * 1000);
        }
    }
    parse_iso_base(b)
}

#[inline]
fn parse_ts(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    if b.len() == 20 && b[19] == b'Z' {
        return parse_iso_base(b);
    }
    if b[0] >= b'0' && b[0] <= b'9' && b.len() != 20 && b.len() != 25 {
        let mut val: i64 = 0;
        for &c in b {
            let d = c.wrapping_sub(b'0');
            if d > 9 {
                return None;
            }
            val = val.checked_mul(10)?.checked_add(d as i64)?;
        }
        return Some(val);
    }
    if b.len() == 25 && (b[19] == b'+' || b[19] == b'-') && b[22] == b':' {
        let base = parse_iso_base(b)?;
        let oh = parse_2d(&b[20..22])?;
        let om = parse_2d(&b[23..25])?;
        if oh > 23 || om > 59 {
            return None;
        }
        let off_ms = ((oh as i64) * 3600 + (om as i64) * 60) * 1000;
        return Some(if b[19] == b'+' { base - off_ms } else { base + off_ms });
    }
    // F3: ASCII digits only (overflow means outside window -> None)
    let mut val: i64 = 0;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        val = val.checked_mul(10)?.checked_add(d as i64)?;
    }
    Some(val)
}

#[inline]
fn parse_ts_cached(b: &[u8], cache: &DateCache) -> Option<i64> {
    if b.len() == 13 {
        let (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) = unsafe {
            (
                (*b.get_unchecked(0)).wrapping_sub(b'0'),
                (*b.get_unchecked(1)).wrapping_sub(b'0'),
                (*b.get_unchecked(2)).wrapping_sub(b'0'),
                (*b.get_unchecked(3)).wrapping_sub(b'0'),
                (*b.get_unchecked(4)).wrapping_sub(b'0'),
                (*b.get_unchecked(5)).wrapping_sub(b'0'),
                (*b.get_unchecked(6)).wrapping_sub(b'0'),
                (*b.get_unchecked(7)).wrapping_sub(b'0'),
                (*b.get_unchecked(8)).wrapping_sub(b'0'),
                (*b.get_unchecked(9)).wrapping_sub(b'0'),
                (*b.get_unchecked(10)).wrapping_sub(b'0'),
                (*b.get_unchecked(11)).wrapping_sub(b'0'),
                (*b.get_unchecked(12)).wrapping_sub(b'0'),
            )
        };
        if (d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12) <= 9 {
            return Some(
                d0 as i64 * 1_000_000_000_000
                    + d1 as i64 * 100_000_000_000
                    + d2 as i64 * 10_000_000_000
                    + d3 as i64 * 1_000_000_000
                    + d4 as i64 * 100_000_000
                    + d5 as i64 * 10_000_000
                    + d6 as i64 * 1_000_000
                    + d7 as i64 * 100_000
                    + d8 as i64 * 10_000
                    + d9 as i64 * 1_000
                    + d10 as i64 * 100
                    + d11 as i64 * 10
                    + d12 as i64,
            );
        }
    }
    if b.len() == 20 && b[19] == b'Z' {
        return parse_iso_base_cached(b, cache);
    }
    if b.len() == 25 && (b[19] == b'+' || b[19] == b'-') && b[22] == b':' {
        let base = parse_iso_base_cached(b, cache)?;
        let oh = parse_2d(&b[20..22])?;
        let om = parse_2d(&b[23..25])?;
        if oh > 23 || om > 59 { return None; }
        let offset = (oh as i64 * 3600 + om as i64 * 60) * 1000;
        return Some(if b[19] == b'+' { base - offset } else { base + offset });
    }
    parse_ts(b)
}

#[inline]
fn parse_event_fields(
    fields: [&[u8]; 7],
    window_start: i64,
    window_end: i64,
    prod_fast: &[u16],
    prod_fallback: &FxHashMap<Box<[u8]>, u16>,
    ids: &IdPool,
    date_cache: &mut DateCache,
    buckets: &mut [Vec<RawEvent>],
    staging: &mut EventStaging,
    wide_ingest: &mut FxHashMap<u64, i64>,
    wide_ts: &mut FxHashMap<u64, i64>,
    wide_category: &mut FxHashMap<u64, u16>,
) {
    let [eid, uid, ts_b, ing_b, etype, pid, price_b] = fields;
    if eid.is_empty() {
        return;
    }
    let Some(ingest_ts) = parse_f1_ingest(ing_b, date_cache) else {
        return;
    };
    let mut price_cat_flags = 0u32;
    let mut category_id = 0u16;
    let mut timestamp = None;

    if !uid.is_empty() && !pid.is_empty() {
        if let Some(parsed_ts) = parse_ts_cached(ts_b, date_cache) {
            if parsed_ts >= window_start
                && parsed_ts < window_end
                && ingest_ts - parsed_ts <= MS_PER_DAY
            {
                match etype {
                    b"view" | b"cart" => {
                        price_cat_flags = 1 << 30;
                        timestamp = Some(parsed_ts);
                    }
                    b"purchase" => {
                        if let Some(p) = parse_price(price_b) {
                            price_cat_flags = (1 << 30) | (1 << 31) | p;
                            category_id = parse_numeric_id(pid, b"p", 65_535)
                                .and_then(|idx| prod_fast.get(idx).copied())
                                .unwrap_or_else(|| *prod_fallback.get(pid).unwrap_or(&0));
                            timestamp = Some(parsed_ts);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    let eid_key = ids.key(eid, b'e', b'E');
    if price_cat_flags & (1 << 31) != 0 {
        let category_code = if category_id < 63 { category_id } else { 63 };
        price_cat_flags |= (category_code as u32) << 24;
    }
    let user_key = if price_cat_flags & (1 << 30) != 0 { ids.key(uid, b'u', b'U') } else { 0 };
    let shard = fast_shard(eid_key, DEDUP_SHARDS);
    let ingest_offset_sec = ingest_ts.div_euclid(1_000) - window_start.div_euclid(1_000);
    let ingest_sec = i32::try_from(ingest_offset_sec).unwrap_or(i32::MIN);
    let ts_off = timestamp
        .and_then(|ts| u32::try_from(ts - window_start).ok())
        .unwrap_or(u32::MAX);
    let idx = staging.push(buckets, shard, RawEvent {
        eid_key,
        user_key,
        ingest_sec,
        ts_off,
        price_cat_flags,
    });
    let loc = ((shard as u64) << 32) | idx as u64;
    if ingest_sec == i32::MIN {
        wide_ingest.insert(loc, ingest_offset_sec);
    }
    if ts_off == u32::MAX {
        if let Some(ts) = timestamp { wide_ts.insert(loc, ts); }
    }
    if category_id >= 63 {
        wide_category.insert(loc, category_id);
    }
}

#[inline]
fn parse_click_fields(
    fields: [&[u8]; 4],
    click_start: i64,
    window_end: i64,
    camp_fast: &[u16],
    camp_fallback: &FxHashMap<Box<[u8]>, u16>,
    ids: &IdPool,
    date_cache: &mut DateCache,
    buckets: &mut [Vec<RawClick>],
    wide_ts: &mut FxHashMap<u64, i64>,
) {
    let [cid, uid, camp_b, ts_b] = fields;
    if cid.is_empty() || uid.is_empty() {
        return;
    }
    let camp_idx = if let Some(idx) = parse_numeric_id(camp_b, b"c_", 4_095) {
        camp_fast.get(idx).copied().unwrap_or(0)
    } else {
        camp_fallback.get(camp_b).copied().unwrap_or(0)
    };
    if camp_idx == 0 {
        return;
    }
    let Some(ts) = parse_ts_cached(ts_b, date_cache) else {
        return;
    };
    if ts < click_start || ts >= window_end {
        return;
    }
    let click_key = ids.key(cid, b'k', b'K');
    let user_key = ids.key(uid, b'u', b'U');
    let shard = fast_shard(click_key, DEDUP_SHARDS);
    let ts_off = u32::try_from(ts - click_start).unwrap_or(u32::MAX);
    let idx = buckets[shard].len() as u32;
    buckets[shard].push(RawClick {
        click_key,
        user_key,
        ts_off,
        campaign_idx: camp_idx,
    });
    if ts_off == u32::MAX {
        wide_ts.insert(((shard as u64) << 32) | idx as u64, ts);
    }
}

#[inline]
fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() {
        return None;
    }
    let mut val: u64 = 0;
    for &c in b {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        val = val.saturating_mul(10).saturating_add(d as u64);
        if val > 10_000_000 {
            return None;
        }
    }
    if val >= 1 && val <= 10_000_000 {
        Some(val as u32)
    } else {
        None
    }
}

fn list_csvs(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("csv") {
                files.push(p);
            }
        }
    }
    files.sort();
    files
}

fn for_each_data_line<F: FnMut(&[u8])>(path: &Path, mut f: F) {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let meta = match file.metadata() {
        Ok(m) => m,
        Err(_) => return,
    };
    if meta.len() == 0 {
        return;
    }
    let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };
    #[cfg(unix)]
    unsafe {
        let _ = libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            libc::MADV_SEQUENTIAL,
        );
    }
    let mmap = Box::leak(Box::new(mmap));
    let data: &[u8] = &mmap[..];
    // Skip header line
    let mut pos = match memchr(b'\n', data) {
        Some(i) => i + 1,
        None => return,
    };
    while pos < data.len() {
        match memchr(b'\n', &data[pos..]) {
            Some(rel) => {
                f(&data[pos..pos + rel]);
                pos += rel + 1;
            }
            None => {
                f(&data[pos..]);
                break;
            }
        }
    }
}

fn mmap_data(path: &Path) -> Option<&'static [u8]> {
    let file = File::open(path).ok()?;
    if file.metadata().ok()?.len() == 0 {
        return None;
    }
    let mmap = unsafe { Mmap::map(&file).expect("mmap input CSV") };
    #[cfg(unix)]
    unsafe {
        let _ = libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            libc::MADV_SEQUENTIAL,
        );
    }
    let mmap = Box::leak(Box::new(mmap));
    Some(&mmap[..])
}

fn mmap_morsels(data: &'static [u8]) -> Vec<&'static [u8]> {
    const MORSEL_BYTES: usize = 4 * 1024 * 1024;
    let mut morsels = Vec::new();
    let mut start = match memchr(b'\n', data) {
        Some(header_end) => header_end + 1,
        None => return morsels,
    };
    while start < data.len() {
        let target = start.saturating_add(MORSEL_BYTES);
        let end = if target >= data.len() {
            data.len()
        } else {
            target + memchr(b'\n', &data[target..]).map_or(data.len() - target, |p| p + 1)
        };
        morsels.push(&data[start..end]);
        start = end;
    }
    morsels
}

fn mmap_csv_morsels(dir: &Path) -> Vec<&'static [u8]> {
    list_csvs(dir)
        .iter()
        .filter_map(|path| mmap_data(path))
        .flat_map(mmap_morsels)
        .collect()
}

// Union-Find tracking byte-wise smallest user_id per component
struct Dsu {
    parent: Vec<u32>,
    names: Vec<u64>,
    index_keys: Vec<u64>,
    index_values: Vec<u32>,
    index_mask: usize,
}

impl Dsu {
    fn new(total_pairs: usize) -> Self {
        let cap = total_pairs.saturating_mul(2).next_power_of_two().max(16);
        Self {
            parent: Vec::new(),
            names: Vec::new(),
            index_keys: vec![0; cap],
            index_values: vec![u32::MAX; cap],
            index_mask: cap - 1,
        }
    }

    fn get_or_insert(&mut self, id: u64) -> u32 {
        let mut slot = mix_key(id) as usize & self.index_mask;
        loop {
            let key = self.index_keys[slot];
            if key == id {
                return self.index_values[slot];
            }
            if key == 0 {
                let idx = self.parent.len() as u32;
                self.parent.push(idx);
                self.names.push(id);
                self.index_keys[slot] = id;
                self.index_values[slot] = idx;
                return idx;
            }
            slot = (slot + 1) & self.index_mask;
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        while self.parent[x as usize] != root {
            let next = self.parent[x as usize];
            self.parent[x as usize] = root;
            x = next;
        }
        root
    }

    fn union(&mut self, a: u32, b: u32, ids: &IdPool) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Keep the byte-wise smaller name as the root
        if ids.cmp_user(self.names[ra as usize], self.names[rb as usize]).is_lt() {
            self.parent[rb as usize] = ra;
        } else {
            self.parent[ra as usize] = rb;
        }
    }

    fn finalize(mut self) -> CanonicalMap {
        let n = self.parent.len();
        let mut non_roots = 0usize;
        for i in 0..n {
            let r = self.find(i as u32);
            self.parent[i] = r;
            if r as usize != i {
                non_roots += 1;
            }
        }
        let mut canon = CanonicalMap::new(non_roots);
        for i in 0..n {
            let r = self.parent[i] as usize;
            if i != r {
                canon.insert(self.names[i], self.names[r]);
            }
        }
        canon
    }
}

struct CanonicalMap {
    entries: Vec<(u64, u64)>,
    mask: usize,
    bloom: Vec<u64>,
}

impl CanonicalMap {
    fn new(expected: usize) -> Self {
        let cap = expected.saturating_mul(2).next_power_of_two().max(16);
        Self {
            entries: vec![(0, 0); cap],
            mask: cap - 1,
            bloom: vec![0; 1 << 18],
        }
    }

    fn insert(&mut self, key: u64, value: u64) {
        let hash = mix_key(key);
        let bit = hash as usize & ((1usize << 24) - 1);
        self.bloom[bit >> 6] |= 1u64 << (bit & 63);
        let mut slot = hash as usize & self.mask;
        loop {
            if self.entries[slot].0 == 0 || self.entries[slot].0 == key {
                self.entries[slot] = (key, value);
                return;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    #[inline]
    fn bloom_index(hash: u64) -> usize {
        let bit = hash as usize & ((1usize << 24) - 1);
        bit >> 6
    }

    #[inline]
    fn bloom_contains(&self, hash: u64) -> bool {
        let bit = hash as usize & ((1usize << 24) - 1);
        unsafe { *self.bloom.get_unchecked(bit >> 6) & (1u64 << (bit & 63)) != 0 }
    }

    #[inline]
    fn initial_slot(&self, hash: u64) -> usize {
        hash as usize & self.mask
    }

    #[inline]
    fn canonical_hashed(&self, key: u64, hash: u64) -> u64 {
        if !self.bloom_contains(hash) {
            return key;
        }
        let mut slot = self.initial_slot(hash);
        loop {
            let (found, value) = unsafe { *self.entries.get_unchecked(slot) };
            if found == 0 {
                return key;
            }
            if found == key {
                return value;
            }
            slot = (slot + 1) & self.mask;
        }
    }

}

fn build_canon_map(pairs_by_worker: Vec<Vec<(u64, u64)>>, ids: &IdPool) -> CanonicalMap {
    let total_pairs = pairs_by_worker.iter().map(Vec::len).sum();
    let mut dsu = Dsu::new(total_pairs);
    for pairs in pairs_by_worker {
        for (ka, kb) in pairs {
            let ia = dsu.get_or_insert(ka);
            let ib = dsu.get_or_insert(kb);
            dsu.union(ia, ib, ids);
        }
    }
    dsu.finalize()
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawEvent {
    eid_key: u64,
    user_key: u64,
    ingest_sec: i32,
    ts_off: u32,
    price_cat_flags: u32,
}
const _: [(); 32] = [(); std::mem::size_of::<RawEvent>()];

struct EventWorkerBuckets {
    buckets: Vec<Vec<RawEvent>>,
    wide_ingest: FxHashMap<u64, i64>,
    wide_ts: FxHashMap<u64, i64>,
    wide_category: FxHashMap<u64, u16>,
}

struct EventStaging {
    rows: [[RawEvent; 4]; DEDUP_SHARDS],
    counts: [u8; DEDUP_SHARDS],
}

impl EventStaging {
    fn new() -> Self {
        Self {
            rows: [[RawEvent::default(); 4]; DEDUP_SHARDS],
            counts: [0; DEDUP_SHARDS],
        }
    }

    fn push(&mut self, buckets: &mut [Vec<RawEvent>], shard: usize, event: RawEvent) -> u32 {
        let count = self.counts[shard] as usize;
        let idx = buckets[shard].len() + count;
        self.rows[shard][count] = event;
        let next = count + 1;
        if next == 4 {
            buckets[shard].extend_from_slice(&self.rows[shard]);
            self.counts[shard] = 0;
        } else {
            self.counts[shard] = next as u8;
        }
        idx as u32
    }

    fn flush(&mut self, buckets: &mut [Vec<RawEvent>]) {
        for shard in 0..DEDUP_SHARDS {
            let count = self.counts[shard] as usize;
            if count != 0 {
                buckets[shard].extend_from_slice(&self.rows[shard][..count]);
                self.counts[shard] = 0;
            }
        }
    }
}

#[inline]
fn event_ingest_sec(worker: &EventWorkerBuckets, shard: usize, idx: usize) -> i64 {
    let ev = &worker.buckets[shard][idx];
    if ev.ingest_sec == i32::MIN {
        worker.wide_ingest[&(((shard as u64) << 32) | idx as u64)]
    } else {
        ev.ingest_sec as i64
    }
}

#[inline]
fn event_timestamp(ev: &CleanEvent, window_start: i64, wide: &[WideEventMeta]) -> i64 {
    if ev.ts_off & META_TAG != 0 {
        wide[(ev.ts_off & !META_TAG) as usize].ts
    } else {
        window_start + ev.ts_off as i64
    }
}

#[inline]
fn event_category(ev: &CleanEvent, wide: &[WideEventMeta]) -> usize {
    if ev.ts_off & META_TAG != 0 {
        wide[(ev.ts_off & !META_TAG) as usize].category_id as usize
    } else {
        ((ev.price_cat_flags >> 24) & 0x3f) as usize
    }
}

#[derive(Clone, Copy)]
struct CleanEvent {
    person_key: u64,
    ts_off: u32,
    price_cat_flags: u32,
}
const _: [(); 16] = [(); std::mem::size_of::<CleanEvent>()];

struct CleanEventStaging {
    rows: [[CleanEvent; 8]; PERSON_SHARDS],
    counts: [u8; PERSON_SHARDS],
}

impl CleanEventStaging {
    fn new() -> Self {
        Self {
            rows: [[CleanEvent { person_key: 0, ts_off: 0, price_cat_flags: 0 }; 8]; PERSON_SHARDS],
            counts: [0; PERSON_SHARDS],
        }
    }

    fn push(&mut self, buckets: &mut [Vec<CleanEvent>], shard: usize, event: CleanEvent) {
        let count = self.counts[shard] as usize;
        self.rows[shard][count] = event;
        let next = count + 1;
        if next == 8 {
            buckets[shard].extend_from_slice(&self.rows[shard]);
            self.counts[shard] = 0;
        } else {
            self.counts[shard] = next as u8;
        }
    }

    fn flush(&mut self, buckets: &mut [Vec<CleanEvent>]) {
        for shard in 0..PERSON_SHARDS {
            let count = self.counts[shard] as usize;
            if count != 0 {
                buckets[shard].extend_from_slice(&self.rows[shard][..count]);
                self.counts[shard] = 0;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WideEventMeta {
    ts: i64,
    category_id: u16,
}

#[derive(Clone, Copy)]
struct CleanClick {
    person_key: u64,
    click_key: u64,
    ts_off: u32,
    campaign_idx: u16,
}
const _: [(); 24] = [(); std::mem::size_of::<CleanClick>()];

struct CleanClickStaging {
    rows: [[CleanClick; 4]; PERSON_SHARDS],
    counts: [u8; PERSON_SHARDS],
}

impl CleanClickStaging {
    fn new() -> Self {
        Self {
            rows: [[CleanClick { person_key: 0, click_key: 0, ts_off: 0, campaign_idx: 0 }; 4]; PERSON_SHARDS],
            counts: [0; PERSON_SHARDS],
        }
    }

    fn push(&mut self, buckets: &mut [Vec<CleanClick>], shard: usize, click: CleanClick) {
        let count = self.counts[shard] as usize;
        self.rows[shard][count] = click;
        let next = count + 1;
        if next == 4 {
            buckets[shard].extend_from_slice(&self.rows[shard]);
            self.counts[shard] = 0;
        } else {
            self.counts[shard] = next as u8;
        }
    }

    fn flush(&mut self, buckets: &mut [Vec<CleanClick>]) {
        for shard in 0..PERSON_SHARDS {
            let count = self.counts[shard] as usize;
            if count != 0 {
                buckets[shard].extend_from_slice(&self.rows[shard][..count]);
                self.counts[shard] = 0;
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawClick {
    click_key: u64,
    user_key: u64,
    ts_off: u32,
    campaign_idx: u16,
}
const _: [(); 24] = [(); std::mem::size_of::<RawClick>()];

struct ClickWorkerBuckets {
    buckets: Vec<Vec<RawClick>>,
    wide_ts: FxHashMap<u64, i64>,
}

#[derive(Clone, Copy, Debug)]
struct WideClickMeta(i64);

#[inline]
fn click_timestamp(click: &CleanClick, click_start: i64, wide: &[WideClickMeta]) -> i64 {
    if click.ts_off & META_TAG != 0 {
        wide[(click.ts_off & !META_TAG) as usize].0
    } else {
        click_start + click.ts_off as i64
    }
}

#[derive(Default, Clone)]
struct CampaignDayAgg {
    sessions: i64,
    purchasing_sessions: i64,
    revenue_cents: i64,
    assisted_sessions: i64,
}

struct UserDailyBuffers {
    rows: usize,
    user_id: Vec<u8>,
    day: Vec<u8>,
    sessions: Vec<u8>,
    events: Vec<u8>,
    revenue: Vec<u8>,
    category_def: Vec<u8>,
    category_values: Vec<u8>,
}

impl UserDailyBuffers {
    fn with_capacity(rows: usize) -> Self {
        Self {
            rows: 0,
            user_id: Vec::with_capacity(rows.saturating_mul(12)),
            day: Vec::with_capacity(rows.saturating_mul(4)),
            sessions: Vec::with_capacity(rows.saturating_mul(8)),
            events: Vec::with_capacity(rows.saturating_mul(8)),
            revenue: Vec::with_capacity(rows.saturating_mul(8)),
            category_def: Vec::with_capacity(rows),
            category_values: Vec::with_capacity(rows.saturating_mul(12)),
        }
    }

    fn push(
        &mut self,
        user_id: &[u8],
        day: i32,
        sessions: i64,
        events: i64,
        revenue: i64,
        category: Option<&[u8]>,
    ) {
        append_plain_byte_array(&mut self.user_id, user_id);
        self.day.extend_from_slice(&day.to_le_bytes());
        self.sessions.extend_from_slice(&sessions.to_le_bytes());
        self.events.extend_from_slice(&events.to_le_bytes());
        self.revenue.extend_from_slice(&revenue.to_le_bytes());
        if let Some(category) = category {
            self.category_def.push(1);
            append_plain_byte_array(&mut self.category_values, category);
        } else {
            self.category_def.push(0);
        }
        self.rows += 1;
    }

    fn columns(&self) -> [parquet_writer::Column<'_>; 6] {
        use parquet_writer::PhysicalType::{ByteArray, Int32, Int64};
        [
            parquet_writer::Column {
                name: "user_id",
                physical_type: ByteArray,
                converted_type: Some(0),
                values: &self.user_id,
                definition_levels: None,
            },
            parquet_writer::Column {
                name: "day",
                physical_type: Int32,
                converted_type: Some(6),
                values: &self.day,
                definition_levels: None,
            },
            parquet_writer::Column {
                name: "sessions",
                physical_type: Int64,
                converted_type: None,
                values: &self.sessions,
                definition_levels: None,
            },
            parquet_writer::Column {
                name: "events",
                physical_type: Int64,
                converted_type: None,
                values: &self.events,
                definition_levels: None,
            },
            parquet_writer::Column {
                name: "revenue_cents",
                physical_type: Int64,
                converted_type: None,
                values: &self.revenue,
                definition_levels: None,
            },
            parquet_writer::Column {
                name: "top_category",
                physical_type: ByteArray,
                converted_type: Some(0),
                values: &self.category_values,
                definition_levels: Some(&self.category_def),
            },
        ]
    }
}

fn append_plain_byte_array(output: &mut Vec<u8>, value: &[u8]) {
    let len = u32::try_from(value.len()).expect("Parquet byte array length");
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(value);
}

struct ParseWorkerData {
    events: EventWorkerBuckets,
    clicks: ClickWorkerBuckets,
}

struct ParsedData {
    workers: Vec<ParseWorkerData>,
    fallback_ids: Vec<Box<[u8]>>,
    total_event_rows: usize,
    total_click_rows: usize,
}

struct DedupWorkerData {
    events: Vec<Vec<CleanEvent>>,
    clicks: Vec<Vec<CleanClick>>,
}

struct DedupData {
    workers: Vec<DedupWorkerData>,
    wide_event_meta: Vec<WideEventMeta>,
    wide_click_meta: Vec<WideClickMeta>,
    fallback_ids: Vec<Box<[u8]>>,
    total_event_rows: usize,
    total_click_rows: usize,
}

struct CompletionCounter {
    remaining: AtomicUsize,
    finished_at: Mutex<Option<std::time::Instant>>,
}

impl CompletionCounter {
    fn new(total: usize) -> Self {
        Self {
            remaining: AtomicUsize::new(total),
            finished_at: Mutex::new(if total == 0 { Some(std::time::Instant::now()) } else { None }),
        }
    }

    fn complete_one(&self) {
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            *self.finished_at.lock().expect("completion time") = Some(std::time::Instant::now());
        }
    }

    fn finished_at(&self) -> std::time::Instant {
        self.finished_at
            .lock()
            .expect("completion time")
            .as_ref()
            .copied()
            .unwrap_or_else(std::time::Instant::now)
    }
}

#[derive(Clone, Copy, Default)]
struct StageTiming {
    elapsed: std::time::Duration,
    cumulative: std::time::Duration,
}

#[derive(Default)]
struct PipelineTimings {
    links: StageTiming,
    events_parse: StageTiming,
    events_dedup: StageTiming,
    clicks: StageTiming,
    sessionize_write: StageTiming,
}

fn stage_timing(start: std::time::Instant, finish: std::time::Instant, overall: std::time::Instant) -> StageTiming {
    StageTiming {
        elapsed: finish.saturating_duration_since(start),
        cumulative: finish.saturating_duration_since(overall),
    }
}

fn deduplicate_worker(
    parsed: &ParsedData,
    canon_map: &CanonicalMap,
    worker_count: usize,
    window_start: i64,
    click_start: i64,
    event_shard_index: &AtomicUsize,
    click_shard_index: &AtomicUsize,
    event_complete: &CompletionCounter,
    click_complete: &CompletionCounter,
    event_meta: &Mutex<Vec<WideEventMeta>>,
    click_meta: &Mutex<Vec<WideClickMeta>>,
    phase_barrier: &Barrier,
) -> DedupWorkerData {
    let mut table = Vec::<DedupSlot>::new();
    let mut occupied = Vec::<u32>::new();
    let output_event_capacity = parsed.total_event_rows / worker_count / PERSON_SHARDS + 64;
    let output_click_capacity = parsed.total_click_rows / worker_count / PERSON_SHARDS + 64;
    let mut event_outputs = (0..PERSON_SHARDS)
        .map(|_| Vec::with_capacity(output_event_capacity))
        .collect::<Vec<Vec<CleanEvent>>>();
    let mut click_outputs = (0..PERSON_SHARDS)
        .map(|_| Vec::with_capacity(output_click_capacity))
        .collect::<Vec<Vec<CleanClick>>>();
    let mut event_staging = CleanEventStaging::new();
    let mut click_staging = CleanClickStaging::new();
    let empty_slot = DedupSlot { key: 0, ingest_sec: 0, loc: u32::MAX };

    loop {
        let shard = event_shard_index.fetch_add(1, Ordering::Relaxed);
        if shard >= DEDUP_SHARDS {
            break;
        }
        let total: usize = parsed.workers.iter().map(|producer| producer.events.buckets[shard].len()).sum();
        let cap = ((total * 3 / 2).next_power_of_two()).max(64);
        table.resize(cap, empty_slot);
        let mask = cap - 1;
        for (worker_index, producer) in parsed.workers.iter().enumerate() {
            let bucket = &producer.events.buckets[shard];
            for (group_index, group) in bucket.chunks(64).enumerate() {
                let base_idx = group_index * 64;
                let mut initial_positions = [0usize; 64];
                for (j, event) in group.iter().enumerate() {
                    let pos = dedup_hash(event.eid_key) as usize & mask;
                    initial_positions[j] = pos;
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            table.as_ptr().add(pos) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                for (j, event) in group.iter().enumerate() {
                    let index = base_idx + j;
                    let loc = ((worker_index as u32) << 27) | index as u32;
                    let mut pos = initial_positions[j];
                    loop {
                        let slot = unsafe { table.get_unchecked_mut(pos) };
                        if slot.key == 0 {
                            *slot = DedupSlot { key: event.eid_key, ingest_sec: event.ingest_sec, loc };
                            occupied.push(pos as u32);
                            break;
                        }
                        if slot.key == event.eid_key {
                            let earlier = if event.ingest_sec != i32::MIN && slot.ingest_sec != i32::MIN {
                                event.ingest_sec < slot.ingest_sec
                            } else {
                                let current = slot.loc;
                                let current_worker = (current >> 27) as usize;
                                let current_index = (current & ((1u32 << 27) - 1)) as usize;
                                event_ingest_sec(
                                    &parsed.workers[current_worker].events,
                                    shard,
                                    current_index,
                                ) > event_ingest_sec(&producer.events, shard, index)
                            };
                            if earlier {
                                *slot = DedupSlot { key: event.eid_key, ingest_sec: event.ingest_sec, loc };
                            }
                            break;
                        }
                        pos = (pos + 1) & mask;
                    }
                }
            }
        }

        for group in occupied.chunks(64) {
            let mut keys = [0u64; 64];
            let mut hashes = [0u64; 64];
            let mut bloom_words = [0usize; 64];
            let mut valid = [false; 64];
            for (j, &position) in group.iter().enumerate() {
                let slot = unsafe { *table.get_unchecked(position as usize) };
                let producer_index = (slot.loc >> 27) as usize;
                let event_index = (slot.loc & ((1u32 << 27) - 1)) as usize;
                let event = parsed.workers[producer_index].events.buckets[shard][event_index];
                if event.price_cat_flags & (1 << 30) != 0 {
                    let hash = mix_key(event.user_key);
                    let bloom_word = CanonicalMap::bloom_index(hash);
                    keys[j] = event.user_key;
                    hashes[j] = hash;
                    bloom_words[j] = bloom_word;
                    valid[j] = true;
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            canon_map.bloom.as_ptr().add(bloom_word) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }
            for j in 0..group.len() {
                if valid[j]
                    && canon_map.bloom[bloom_words[j]] & (1u64 << (hashes[j] as usize & 63)) != 0
                {
                    let slot = canon_map.initial_slot(hashes[j]);
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            canon_map.entries.as_ptr().add(slot) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }
            for (j, &position) in group.iter().enumerate() {
                if !valid[j] {
                    continue;
                }
                let slot = unsafe { *table.get_unchecked(position as usize) };
                let producer_index = (slot.loc >> 27) as usize;
                let event_index = (slot.loc & ((1u32 << 27) - 1)) as usize;
                let producer = &parsed.workers[producer_index].events;
                let event = producer.buckets[shard][event_index];
                let person = canon_map.canonical_hashed(keys[j], hashes[j]);
                let person_shard = fast_shard(mix_key(person), PERSON_SHARDS);
                let loc_key = ((shard as u64) << 32) | event_index as u64;
                let timestamp = if event.ts_off == u32::MAX {
                    producer.wide_ts[&loc_key]
                } else {
                    window_start + event.ts_off as i64
                };
                let category_code = ((event.price_cat_flags >> 24) & 0x3f) as u16;
                let category_id = if category_code == 63 {
                    producer.wide_category[&loc_key]
                } else {
                    category_code
                };
                let ts_off = if event.ts_off < META_TAG && category_id < 63 {
                    event.ts_off
                } else {
                    let mut wide = event_meta.lock().expect("wide event metadata");
                    let index = wide.len();
                    assert!(index < META_TAG as usize, "too many wide events");
                    wide.push(WideEventMeta { ts: timestamp, category_id });
                    META_TAG | index as u32
                };
                event_staging.push(
                    &mut event_outputs,
                    person_shard,
                    CleanEvent { person_key: person, ts_off, price_cat_flags: event.price_cat_flags },
                );
            }
        }
        for &position in &occupied {
            table[position as usize] = empty_slot;
        }
        occupied.clear();
        event_complete.complete_one();
    }
    event_staging.flush(&mut event_outputs);
    phase_barrier.wait();

    loop {
        let shard = click_shard_index.fetch_add(1, Ordering::Relaxed);
        if shard >= DEDUP_SHARDS {
            break;
        }
        let total: usize = parsed.workers.iter().map(|producer| producer.clicks.buckets[shard].len()).sum();
        let cap = ((total * 3 / 2).next_power_of_two()).max(64);
        table.resize(cap, empty_slot);
        let mask = cap - 1;
        for (worker_index, producer) in parsed.workers.iter().enumerate() {
            let bucket = &producer.clicks.buckets[shard];
            for (group_index, group) in bucket.chunks(64).enumerate() {
                let base_idx = group_index * 64;
                let mut initial_positions = [0usize; 64];
                for (j, click) in group.iter().enumerate() {
                    let pos = dedup_hash(click.click_key) as usize & mask;
                    initial_positions[j] = pos;
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            table.as_ptr().add(pos) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                for (j, click) in group.iter().enumerate() {
                    let index = base_idx + j;
                    let loc = ((worker_index as u32) << 27) | index as u32;
                    let mut pos = initial_positions[j];
                    loop {
                        let slot = unsafe { table.get_unchecked_mut(pos) };
                        if slot.key == 0 {
                            *slot = DedupSlot { key: click.click_key, ingest_sec: 0, loc };
                            occupied.push(pos as u32);
                            break;
                        }
                        if slot.key == click.click_key {
                            break;
                        }
                        pos = (pos + 1) & mask;
                    }
                }
            }
        }

        for group in occupied.chunks(64) {
            let mut keys = [0u64; 64];
            let mut hashes = [0u64; 64];
            let mut bloom_words = [0usize; 64];
            for (j, &position) in group.iter().enumerate() {
                let slot = unsafe { *table.get_unchecked(position as usize) };
                let producer_index = (slot.loc >> 27) as usize;
                let click_index = (slot.loc & ((1u32 << 27) - 1)) as usize;
                let click = parsed.workers[producer_index].clicks.buckets[shard][click_index];
                let hash = mix_key(click.user_key);
                keys[j] = click.user_key;
                hashes[j] = hash;
                bloom_words[j] = CanonicalMap::bloom_index(hash);
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        canon_map.bloom.as_ptr().add(bloom_words[j]) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
            for j in 0..group.len() {
                if canon_map.bloom[bloom_words[j]] & (1u64 << (hashes[j] as usize & 63)) != 0 {
                    let slot = canon_map.initial_slot(hashes[j]);
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            canon_map.entries.as_ptr().add(slot) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }
            for (j, &position) in group.iter().enumerate() {
                let slot = unsafe { *table.get_unchecked(position as usize) };
                let producer_index = (slot.loc >> 27) as usize;
                let click_index = (slot.loc & ((1u32 << 27) - 1)) as usize;
                let producer = &parsed.workers[producer_index].clicks;
                let click = producer.buckets[shard][click_index];
                let person = canon_map.canonical_hashed(keys[j], hashes[j]);
                let person_shard = fast_shard(mix_key(person), PERSON_SHARDS);
                let loc_key = ((shard as u64) << 32) | click_index as u64;
                let timestamp = if click.ts_off == u32::MAX {
                    producer.wide_ts[&loc_key]
                } else {
                    click_start + click.ts_off as i64
                };
                let ts_off = if click.ts_off < META_TAG {
                    click.ts_off
                } else {
                    let mut wide = click_meta.lock().expect("wide click metadata");
                    let index = wide.len();
                    assert!(index < META_TAG as usize, "too many wide clicks");
                    wide.push(WideClickMeta(timestamp));
                    META_TAG | index as u32
                };
                click_staging.push(
                    &mut click_outputs,
                    person_shard,
                    CleanClick {
                        person_key: person,
                        click_key: click.click_key,
                        ts_off,
                        campaign_idx: click.campaign_idx,
                    },
                );
            }
        }
        for &position in &occupied {
            table[position as usize] = empty_slot;
        }
        occupied.clear();
        click_complete.complete_one();
    }
    click_staging.flush(&mut click_outputs);

    DedupWorkerData { events: event_outputs, clicks: click_outputs }
}

fn sessionize_write_worker(
    worker_id: usize,
    dedup: &DedupData,
    worker_count: usize,
    ids: &IdPool,
    next_person_shard: &AtomicUsize,
    window_start: i64,
    click_start: i64,
    start_day: i32,
    num_days: usize,
    num_cats: usize,
    num_camps: usize,
    cat_names: &[String],
    out_dir: &Path,
) -> Vec<CampaignDayAgg> {
    let mut events = Vec::<CleanEvent>::with_capacity(dedup.total_event_rows / PERSON_SHARDS + 64);
    let mut clicks = Vec::<CleanClick>::with_capacity(dedup.total_click_rows / PERSON_SHARDS + 64);
    let mut user_rows =
        UserDailyBuffers::with_capacity(dedup.total_event_rows / worker_count / 8 + 64);
    let mut camp_agg: Vec<CampaignDayAgg> =
        (0..num_camps * num_days).map(|_| CampaignDayAgg::default()).collect();
    let mut assist_stamp = vec![0u32; num_camps];
    let mut cur_stamp = 0u32;
    let mut day_acc = vec![(0i64, 0i64, 0i64, false); num_days];
    let mut day_cat_rev = vec![0i64; num_days * num_cats];
    let mut touched_days = Vec::with_capacity(num_days);
    let fast_event_sort = dedup.wide_event_meta.is_empty();
    let fast_click_sort = dedup.wide_click_meta.is_empty() && dedup.fallback_ids.is_empty();

    loop {
        let shard = next_person_shard.fetch_add(1, Ordering::Relaxed);
        if shard >= PERSON_SHARDS {
            break;
        }
        events.clear();
        clicks.clear();
        for producer in &dedup.workers {
            events.extend_from_slice(&producer.events[shard]);
            clicks.extend_from_slice(&producer.clicks[shard]);
        }
        if fast_event_sort {
            events.sort_unstable_by_key(|event| (event.person_key, event.ts_off));
        } else {
            events.sort_unstable_by(|a, b| {
                a.person_key.cmp(&b.person_key).then_with(|| {
                    event_timestamp(a, window_start, &dedup.wide_event_meta)
                        .cmp(&event_timestamp(b, window_start, &dedup.wide_event_meta))
                })
            });
        }
        if fast_click_sort {
            clicks.sort_unstable_by_key(|click| (click.person_key, click.ts_off, click.click_key));
        } else {
            clicks.sort_unstable_by(|a, b| {
                a.person_key
                    .cmp(&b.person_key)
                    .then_with(|| {
                        click_timestamp(a, click_start, &dedup.wide_click_meta)
                            .cmp(&click_timestamp(b, click_start, &dedup.wide_click_meta))
                    })
                    .then_with(|| ids.cmp_ids(a.click_key, b.click_key, b'k', b'K'))
            });
        }

        let mut event_index = 0usize;
        let mut click_index = 0usize;
        while event_index < events.len() {
            let person = events[event_index].person_key;
            let mut event_end = event_index + 1;
            while event_end < events.len() && events[event_end].person_key == person {
                event_end += 1;
            }
            while click_index < clicks.len() && clicks[click_index].person_key < person {
                click_index += 1;
            }
            let mut click_end = click_index;
            while click_end < clicks.len() && clicks[click_end].person_key == person {
                click_end += 1;
            }
            let person_clicks = &clicks[click_index..click_end];
            let mut click_left = 0usize;
            let mut click_right = 0usize;
            let mut i = event_index;
            while i < event_end {
                let session_start = event_timestamp(&events[i], window_start, &dedup.wide_event_meta);
                let day = session_start.div_euclid(MS_PER_DAY) as i32;
                let day_index = (day - start_day) as usize;
                if day_acc[day_index].0 == 0 {
                    touched_days.push(day_index);
                }
                day_acc[day_index].0 += 1;
                let mut session_events = 0i64;
                let mut session_revenue = 0i64;
                let mut has_purchase = false;
                let mut previous_ts = session_start;
                while i < event_end
                    && event_timestamp(&events[i], window_start, &dedup.wide_event_meta) - previous_ts
                        <= MS_30_MIN
                {
                    let event = &events[i];
                    previous_ts = event_timestamp(event, window_start, &dedup.wide_event_meta);
                    session_events += 1;
                    if event.price_cat_flags & (1 << 31) != 0 {
                        let price = (event.price_cat_flags & 0x00ff_ffff) as i64;
                        let category = event_category(event, &dedup.wide_event_meta);
                        session_revenue += price;
                        has_purchase = true;
                        day_cat_rev[day_index * num_cats + category] += price;
                        day_acc[day_index].3 = true;
                    }
                    i += 1;
                }
                day_acc[day_index].1 += session_events;
                day_acc[day_index].2 += session_revenue;

                let window_lo = session_start - MS_7_DAYS;
                while click_right < person_clicks.len()
                    && click_timestamp(&person_clicks[click_right], click_start, &dedup.wide_click_meta)
                        <= session_start
                {
                    click_right += 1;
                }
                while click_left < click_right
                    && click_timestamp(&person_clicks[click_left], click_start, &dedup.wide_click_meta)
                        < window_lo
                {
                    click_left += 1;
                }
                if click_left < click_right {
                    let best_ts = click_timestamp(
                        &person_clicks[click_right - 1],
                        click_start,
                        &dedup.wide_click_meta,
                    );
                    let mut best_index = click_right - 1;
                    while best_index > click_left
                        && click_timestamp(
                            &person_clicks[best_index - 1],
                            click_start,
                            &dedup.wide_click_meta,
                        ) == best_ts
                    {
                        best_index -= 1;
                    }
                    let credited_campaign = person_clicks[best_index].campaign_idx;
                    let credited = &mut camp_agg[credited_campaign as usize * num_days + day_index];
                    credited.sessions += 1;
                    if has_purchase {
                        credited.purchasing_sessions += 1;
                        credited.revenue_cents += session_revenue;
                    }
                    cur_stamp = cur_stamp.wrapping_add(1);
                    if cur_stamp == 0 {
                        assist_stamp.fill(0);
                        cur_stamp = 1;
                    }
                    for click in &person_clicks[click_left..click_right] {
                        let campaign = click.campaign_idx as usize;
                        if assist_stamp[campaign] != cur_stamp {
                            assist_stamp[campaign] = cur_stamp;
                            camp_agg[campaign * num_days + day_index].assisted_sessions += 1;
                        }
                    }
                } else {
                    let organic = &mut camp_agg[day_index];
                    organic.sessions += 1;
                    if has_purchase {
                        organic.purchasing_sessions += 1;
                        organic.revenue_cents += session_revenue;
                    }
                }
            }

            for &day_index in &touched_days {
                let day = start_day + day_index as i32;
                let top_category = if day_acc[day_index].3 {
                    let mut best_category = 0usize;
                    let mut best_revenue = -1i64;
                    for category in 0..num_cats {
                        let revenue = day_cat_rev[day_index * num_cats + category];
                        if revenue > 0
                            && (revenue > best_revenue
                                || (revenue == best_revenue
                                    && cat_names[category] < cat_names[best_category]))
                        {
                            best_revenue = revenue;
                            best_category = category;
                        }
                    }
                    Some(best_category as u16)
                } else {
                    None
                };
                let mut user_id_buffer = [0u8; 11];
                let user_id = if person >> 63 != 0 {
                    dedup.fallback_ids[(person & !(1u64 << 63)) as usize].as_ref()
                } else {
                    unpack_fast_id(person, b'u', b'U', &mut user_id_buffer)
                };
                user_rows.push(
                    user_id,
                    day,
                    day_acc[day_index].0,
                    day_acc[day_index].1,
                    day_acc[day_index].2,
                    top_category.map(|category| cat_names[category as usize].as_bytes()),
                );
                day_acc[day_index] = (0, 0, 0, false);
                day_cat_rev[day_index * num_cats..(day_index + 1) * num_cats].fill(0);
            }
            touched_days.clear();
            event_index = event_end;
            click_index = click_end;
        }
    }

    let columns = user_rows.columns();
    parquet_writer::write_table(
        &out_dir.join("user_daily").join(format!("part-{worker_id}.parquet")),
        user_rows.rows,
        &columns,
    )
    .expect("write user_daily parquet");
    camp_agg
}

struct PipelineConfig<'a> {
    data_dir: &'a Path,
    out_dir: &'a Path,
    worker_count: usize,
    overall_start: std::time::Instant,
    window_start: i64,
    window_end: i64,
    click_start: i64,
    date_cache: DateCache,
    prod_fast: &'a [u16],
    prod_fallback: &'a FxHashMap<Box<[u8]>, u16>,
    camp_fast: &'a [u16],
    camp_fallback: &'a FxHashMap<Box<[u8]>, u16>,
    ids: &'a IdPool,
    cat_names: &'a [String],
    camp_names: &'a [String],
    camp_channels: &'a [String],
}

fn run_persistent_pipeline(config: PipelineConfig<'_>) -> PipelineTimings {
    let worker_count = config.worker_count;
    let link_morsels = mmap_csv_morsels(&config.data_dir.join("links"));
    let event_morsels = mmap_csv_morsels(&config.data_dir.join("events"));
    let click_morsels = mmap_csv_morsels(&config.data_dir.join("clicks"));
    let event_bytes = event_morsels.iter().map(|m| m.len()).sum::<usize>();
    let click_bytes = click_morsels.iter().map(|m| m.len()).sum::<usize>();
    let event_bucket_capacity = event_bytes / worker_count / DEDUP_SHARDS / 70;
    let click_bucket_capacity = click_bytes / worker_count / DEDUP_SHARDS / 70;

    let link_index = AtomicUsize::new(0);
    let event_index = AtomicUsize::new(0);
    let click_index = AtomicUsize::new(0);
    let event_dedup_index = AtomicUsize::new(0);
    let click_dedup_index = AtomicUsize::new(0);
    let person_index = AtomicUsize::new(0);
    let event_parse_complete = CompletionCounter::new(event_morsels.len());
    let click_parse_complete = CompletionCounter::new(click_morsels.len());
    let event_dedup_complete = CompletionCounter::new(DEDUP_SHARDS);
    let click_dedup_complete = CompletionCounter::new(DEDUP_SHARDS);

    let link_pairs: Mutex<Vec<Option<Vec<(u64, u64)>>>> =
        Mutex::new((0..worker_count).map(|_| None).collect());
    let parsed_workers: Mutex<Vec<Option<ParseWorkerData>>> =
        Mutex::new((0..worker_count).map(|_| None).collect());
    let canonical_shared: Mutex<Option<Arc<CanonicalMap>>> = Mutex::new(None);
    let parsed_shared: Mutex<Option<Arc<ParsedData>>> = Mutex::new(None);
    let dedup_workers: Mutex<Vec<Option<DedupWorkerData>>> =
        Mutex::new((0..worker_count).map(|_| None).collect());
    let dedup_shared: Mutex<Option<Arc<DedupData>>> = Mutex::new(None);
    let campaign_results: Mutex<Vec<Option<Vec<CampaignDayAgg>>>> =
        Mutex::new((0..worker_count).map(|_| None).collect());
    let wide_event_meta = Mutex::new(Vec::<WideEventMeta>::new());
    let wide_click_meta = Mutex::new(Vec::<WideClickMeta>::new());
    let link_timing = Mutex::new(None::<StageTiming>);
    let event_dedup_start = Mutex::new(None::<std::time::Instant>);
    let sessionize_start = Mutex::new(None::<std::time::Instant>);
    let sessionize_finish = Mutex::new(None::<std::time::Instant>);
    let parse_start = std::time::Instant::now();
    let link_start = std::time::Instant::now();
    let barrier1 = Barrier::new(worker_count);
    let barrier2 = Barrier::new(worker_count);
    let barrier3 = Barrier::new(worker_count);
    let event_click_barrier = Barrier::new(worker_count);
    let session_barrier = Barrier::new(worker_count);
    let num_cats = config.cat_names.len();
    let num_camps = config.camp_names.len();
    let start_day = config.window_start.div_euclid(MS_PER_DAY) as i32;
    let end_day = (config.window_end - 1).div_euclid(MS_PER_DAY) as i32;
    let num_days = (end_day - start_day + 1).max(1) as usize;

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_id in 0..worker_count {
            let link_morsels = &link_morsels;
            let event_morsels = &event_morsels;
            let click_morsels = &click_morsels;
            let link_pairs = &link_pairs;
            let parsed_workers = &parsed_workers;
            let canonical_shared = &canonical_shared;
            let parsed_shared = &parsed_shared;
            let dedup_workers = &dedup_workers;
            let dedup_shared = &dedup_shared;
            let campaign_results = &campaign_results;
            let wide_event_meta = &wide_event_meta;
            let wide_click_meta = &wide_click_meta;
            let link_timing = &link_timing;
            let event_dedup_start = &event_dedup_start;
            let sessionize_start = &sessionize_start;
            let sessionize_finish = &sessionize_finish;
            let ids = config.ids;
            let prod_fast = config.prod_fast;
            let prod_fallback = config.prod_fallback;
            let camp_fast = config.camp_fast;
            let camp_fallback = config.camp_fallback;
            let cat_names = config.cat_names;
            let camp_names = config.camp_names;
            let camp_channels = config.camp_channels;
            let out_dir = config.out_dir;
            let overall_start = config.overall_start;
            let window_start = config.window_start;
            let window_end = config.window_end;
            let click_start = config.click_start;
            let date_cache = config.date_cache;
            let worker_count = config.worker_count;
            let event_bucket_capacity = event_bucket_capacity;
            let click_bucket_capacity = click_bucket_capacity;
            let event_parse_complete = &event_parse_complete;
            let click_parse_complete = &click_parse_complete;
            let event_dedup_complete = &event_dedup_complete;
            let click_dedup_complete = &click_dedup_complete;
            let event_index = &event_index;
            let click_index = &click_index;
            let link_index = &link_index;
            let event_dedup_index = &event_dedup_index;
            let click_dedup_index = &click_dedup_index;
            let person_index = &person_index;
            let barrier1 = &barrier1;
            let barrier2 = &barrier2;
            let barrier3 = &barrier3;
            let event_click_barrier = &event_click_barrier;
            let session_barrier = &session_barrier;
            handles.push(scope.spawn(move || {
                let mut local_link_pairs = Vec::new();
                loop {
                    let chunk_index = link_index.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = link_morsels.get(chunk_index) else { break };
                    scan_csv_morsel::<2, _>(chunk, |[a, b]| {
                        if !a.is_empty() && !b.is_empty() {
                            local_link_pairs.push((ids.key(a, b'u', b'U'), ids.key(b, b'u', b'U')));
                        }
                    });
                }
                link_pairs.lock().expect("link result slots")[worker_id] = Some(local_link_pairs);
                barrier1.wait();

                if worker_id == 0 {
                    let pairs = {
                        let mut slots = link_pairs.lock().expect("link result slots");
                        slots.iter_mut().map(|slot| slot.take().expect("link worker result")).collect()
                    };
                    let canonical = Arc::new(build_canon_map(pairs, ids));
                    *canonical_shared.lock().expect("canonical map") = Some(canonical);
                    let finish = std::time::Instant::now();
                    *link_timing.lock().expect("links timing") =
                        Some(stage_timing(link_start, finish, overall_start));
                }

                let mut event_buckets = (0..DEDUP_SHARDS)
                    .map(|_| Vec::with_capacity(event_bucket_capacity))
                    .collect::<Vec<Vec<RawEvent>>>();
                let mut event_staging = EventStaging::new();
                let mut wide_ingest = FxHashMap::default();
                let mut wide_ts = FxHashMap::default();
                let mut wide_category = FxHashMap::default();
                let mut click_buckets = (0..DEDUP_SHARDS)
                    .map(|_| Vec::with_capacity(click_bucket_capacity))
                    .collect::<Vec<Vec<RawClick>>>();
                let mut click_wide_ts = FxHashMap::default();
                let mut local_date_cache = date_cache;
                let mut buffer_events = true;
                let mut buffer_clicks = true;
                while buffer_events || buffer_clicks {
                    if buffer_events {
                        let index = event_index.fetch_add(1, Ordering::Relaxed);
                        if let Some(morsel) = event_morsels.get(index) {
                            scan_csv_morsel::<7, _>(morsel, |fields| {
                                parse_event_fields(
                                    fields,
                                    window_start,
                                    window_end,
                                    prod_fast,
                                    prod_fallback,
                                    ids,
                                    &mut local_date_cache,
                                    &mut event_buckets,
                                    &mut event_staging,
                                    &mut wide_ingest,
                                    &mut wide_ts,
                                    &mut wide_category,
                                );
                            });
                            event_parse_complete.complete_one();
                        } else {
                            buffer_events = false;
                        }
                    }
                    if buffer_clicks {
                        let index = click_index.fetch_add(1, Ordering::Relaxed);
                        if let Some(morsel) = click_morsels.get(index) {
                            scan_csv_morsel::<4, _>(morsel, |fields| {
                                parse_click_fields(
                                    fields,
                                    click_start,
                                    window_end,
                                    camp_fast,
                                    camp_fallback,
                                    ids,
                                    &mut local_date_cache,
                                    &mut click_buckets,
                                    &mut click_wide_ts,
                                );
                            });
                            click_parse_complete.complete_one();
                        } else {
                            buffer_clicks = false;
                        }
                    }
                }
                event_staging.flush(&mut event_buckets);
                parsed_workers.lock().expect("parse result slots")[worker_id] = Some(ParseWorkerData {
                    events: EventWorkerBuckets {
                        buckets: event_buckets,
                        wide_ingest,
                        wide_ts,
                        wide_category,
                    },
                    clicks: ClickWorkerBuckets { buckets: click_buckets, wide_ts: click_wide_ts },
                });
                barrier2.wait();

                if worker_id == 0 {
                    let producers = {
                        let mut slots = parsed_workers.lock().expect("parse result slots");
                        slots.iter_mut().map(|slot| slot.take().expect("parse worker result")).collect::<Vec<_>>()
                    };
                    let total_event_rows = producers
                        .iter()
                        .flat_map(|producer| producer.events.buckets.iter())
                        .map(Vec::len)
                        .sum();
                    let total_click_rows = producers
                        .iter()
                        .flat_map(|producer| producer.clicks.buckets.iter())
                        .map(Vec::len)
                        .sum();
                    *parsed_shared.lock().expect("parsed data") = Some(Arc::new(ParsedData {
                        workers: producers,
                        fallback_ids: ids.fallback_snapshot(),
                        total_event_rows,
                        total_click_rows,
                    }));
                    *event_dedup_start.lock().expect("event dedup start") = Some(std::time::Instant::now());
                }
                barrier2.wait();

                let parsed = parsed_shared.lock().expect("parsed data").as_ref().expect("parsed data").clone();
                let canonical = canonical_shared
                    .lock()
                    .expect("canonical map")
                    .as_ref()
                    .expect("canonical map")
                    .clone();
                let local_dedup = deduplicate_worker(
                    &parsed,
                    &canonical,
                    worker_count,
                    window_start,
                    click_start,
                    event_dedup_index,
                    click_dedup_index,
                    event_dedup_complete,
                    click_dedup_complete,
                    wide_event_meta,
                    wide_click_meta,
                    event_click_barrier,
                );
                dedup_workers.lock().expect("dedup result slots")[worker_id] = Some(local_dedup);
                barrier3.wait();

                if worker_id == 0 {
                    let producers = {
                        let mut slots = dedup_workers.lock().expect("dedup result slots");
                        slots.iter_mut().map(|slot| slot.take().expect("dedup worker result")).collect()
                    };
                    let event_meta = std::mem::take(&mut *wide_event_meta.lock().expect("wide event metadata"));
                    let click_meta = std::mem::take(&mut *wide_click_meta.lock().expect("wide click metadata"));
                    *dedup_shared.lock().expect("dedup data") = Some(Arc::new(DedupData {
                        workers: producers,
                        wide_event_meta: event_meta,
                        wide_click_meta: click_meta,
                        fallback_ids: parsed.fallback_ids.clone(),
                        total_event_rows: parsed.total_event_rows,
                        total_click_rows: parsed.total_click_rows,
                    }));
                    *sessionize_start.lock().expect("sessionize start") = Some(std::time::Instant::now());
                }
                barrier3.wait();

                let dedup = dedup_shared.lock().expect("dedup data").as_ref().expect("dedup data").clone();
                let camp_agg = sessionize_write_worker(
                    worker_id,
                    &dedup,
                    worker_count,
                    ids,
                    person_index,
                    window_start,
                    click_start,
                    start_day,
                    num_days,
                    num_cats,
                    num_camps,
                    cat_names,
                    out_dir,
                );
                campaign_results.lock().expect("campaign result slots")[worker_id] = Some(camp_agg);
                session_barrier.wait();

                if worker_id == 0 {
                    let mut final_campaigns: FxHashMap<(u16, i32), CampaignDayAgg> = FxHashMap::default();
                    let mut slots = campaign_results.lock().expect("campaign result slots");
                    for campaign_agg in slots.iter_mut() {
                        let values = campaign_agg.take().expect("campaign worker result");
                        for (index, value) in values.into_iter().enumerate() {
                            if value.sessions != 0 || value.assisted_sessions != 0 {
                                let campaign = index / num_days;
                                let day = start_day + (index % num_days) as i32;
                                let entry = final_campaigns.entry((campaign as u16, day)).or_default();
                                entry.sessions += value.sessions;
                                entry.purchasing_sessions += value.purchasing_sessions;
                                entry.revenue_cents += value.revenue_cents;
                                entry.assisted_sessions += value.assisted_sessions;
                            }
                        }
                    }
                    write_campaign_daily(
                        &out_dir.join("campaign_daily"),
                        &final_campaigns,
                        camp_names,
                        camp_channels,
                    );
                    *sessionize_finish.lock().expect("sessionize finish") = Some(std::time::Instant::now());
                }
            }));
        }
        for handle in handles {
            handle.join().expect("pipeline worker");
        }
    });

    let mut timings = PipelineTimings::default();
    timings.links = link_timing.lock().expect("links timing").unwrap_or_default();
    let event_parse_finish = event_parse_complete.finished_at();
    timings.events_parse = stage_timing(parse_start, event_parse_finish, config.overall_start);
    let event_dedup_finish = event_dedup_complete.finished_at();
    let event_dedup_begin = event_dedup_start.lock().expect("event dedup start").unwrap_or(event_dedup_finish);
    timings.events_dedup = stage_timing(event_dedup_begin, event_dedup_finish, config.overall_start);
    let click_parse_finish = click_parse_complete.finished_at();
    let click_dedup_finish = click_dedup_complete.finished_at();
    let click_dedup_elapsed = click_dedup_finish.saturating_duration_since(event_dedup_finish);
    timings.clicks.elapsed =
        click_parse_finish.saturating_duration_since(parse_start) + click_dedup_elapsed;
    timings.clicks.cumulative = click_dedup_finish.saturating_duration_since(config.overall_start);
    let sessionize_begin = sessionize_start.lock().expect("sessionize start").unwrap_or(click_dedup_finish);
    let sessionize_end = sessionize_finish.lock().expect("sessionize finish").unwrap_or_else(std::time::Instant::now);
    timings.sessionize_write = stage_timing(sessionize_begin, sessionize_end, config.overall_start);
    timings
}

#[cfg(any())]
fn legacy_main() {
    let t_start = std::time::Instant::now();
    let mut t = std::time::Instant::now();
    let data_dir = PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out_dir = PathBuf::from(env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let worker_count = env::var("NCPU")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |count| count.get()));

    // 1. Manifest
    let manifest_str = fs::read_to_string(data_dir.join("manifest.json")).expect("read manifest");
    let manifest: Manifest = serde_json::from_str(&manifest_str).expect("parse manifest");
    let window_start = parse_ts(manifest.window_start.as_bytes()).expect("window_start");
    let window_end = parse_ts(manifest.window_end.as_bytes()).expect("window_end");
    let click_start = window_start - MS_7_DAYS;
    let date_cache = DateCache::new(window_start, window_end);

    // 2. Products -> category_id (0 = "unknown")
    let mut cat_names: Vec<String> = vec!["unknown".to_string()];
    let mut cat_map: FxHashMap<String, u16> = FxHashMap::default();
    cat_map.insert("unknown".to_string(), 0);
    let mut prod_fast = vec![0u16; 65_536];
    let mut prod_fallback: FxHashMap<Box<[u8]>, u16> = FxHashMap::default();
    for_each_data_line(&data_dir.join("products.csv"), |line| {
        if let Some([pid, cat]) = parse_csv_line::<2>(line) {
            let cat_str = std::str::from_utf8(cat).unwrap_or("unknown");
            let cid = if let Some(&id) = cat_map.get(cat_str) {
                id
            } else {
                let id = cat_names.len() as u16;
                cat_names.push(cat_str.to_string());
                cat_map.insert(cat_str.to_string(), id);
                id
            };
            if let Some(idx) = parse_numeric_id(pid, b"p", 65_535) {
                prod_fast[idx] = cid;
            } else {
                prod_fallback.insert(pid.into(), cid);
            }
        }
    });

    // 3. Campaigns -> campaign_idx (0 = "organic")
    let mut camp_names: Vec<String> = vec!["organic".to_string()];
    let mut camp_channels: Vec<String> = vec!["organic".to_string()];
    let mut camp_fast = vec![0u16; 4_096];
    let mut camp_fallback: FxHashMap<Box<[u8]>, u16> = FxHashMap::default();
    for_each_data_line(&data_dir.join("campaigns.csv"), |line| {
        if let Some([cid, ch]) = parse_csv_line::<2>(line) {
            let idx = camp_names.len() as u16;
            camp_names.push(String::from_utf8_lossy(cid).into_owned());
            camp_channels.push(String::from_utf8_lossy(ch).into_owned());
            if let Some(id) = parse_numeric_id(cid, b"c_", 4_095) {
                camp_fast[id] = idx;
            } else {
                camp_fallback.insert(cid.into(), idx);
            }
        }
    });

    let ids = Arc::new(IdPool::new());
    // 4. Parse links and events together; event parsing does not need the canonical map.
    let link_chunks = read_chunks(&data_dir.join("links"));
    let event_chunks = read_chunks(&data_dir.join("events"));
    let (canon_map, worker_event_buckets, links_elapsed, event_parse_elapsed) = if worker_count > 1 {
        std::thread::scope(|scope| {
            let links_start = std::time::Instant::now();
            let link_chunks_ref = &link_chunks;
            let ids_ref = ids.as_ref();
            let link_handle = scope.spawn(move || {
                let map = build_canon_map_from_chunks(link_chunks_ref, ids_ref, 1);
                (map, links_start.elapsed())
            });
            let event_parse_start = std::time::Instant::now();
            let workers = parse_event_chunks(
                &event_chunks,
                worker_count - 1,
                window_start,
                window_end,
                &prod_fast,
                &prod_fallback,
                &ids,
                date_cache,
            );
            let event_elapsed = event_parse_start.elapsed();
            let (map, link_elapsed) = link_handle.join().expect("links worker");
            (map, workers, link_elapsed, event_elapsed)
        })
    } else {
        let links_start = std::time::Instant::now();
        let map = build_canon_map_from_chunks(&link_chunks, &ids, 1);
        let link_elapsed = links_start.elapsed();
        let event_parse_start = std::time::Instant::now();
        let workers = parse_event_chunks(
            &event_chunks,
            1,
            window_start,
            window_end,
            &prod_fast,
            &prod_fallback,
            &ids,
            date_cache,
        );
        (map, workers, link_elapsed, event_parse_start.elapsed())
    };
    eprintln!("links: {:?} cumulative: {:?}", links_elapsed, t.elapsed());
    t = std::time::Instant::now();
    eprintln!("events_parse: {:?} cumulative: {:?}", event_parse_elapsed, t_start.elapsed());
    let _events_parse_wall_since_link_log = t.elapsed();
    t = std::time::Instant::now();
    let total_event_rows: usize = worker_event_buckets
        .iter()
        .flat_map(|worker| worker.buckets.iter())
        .map(Vec::len)
        .sum();

    // Dedup events per 256-shard, then repartition by canonical person.
    let shard_index = AtomicUsize::new(0);
    let wide_event_meta = Arc::new(Mutex::new(Vec::<WideEventMeta>::new()));
    let person_event_shards: Vec<Vec<Vec<CleanEvent>>> =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for _ in 0..worker_count {
                handles.push(scope.spawn(|| {
                    let mut table = Vec::<DedupSlot>::new();
                    let mut occupied = Vec::<u32>::new();
                    let output_capacity = total_event_rows / worker_count / PERSON_SHARDS + 64;
                    let mut outputs = (0..PERSON_SHARDS)
                        .map(|_| Vec::with_capacity(output_capacity))
                        .collect::<Vec<Vec<CleanEvent>>>();
                    let mut staging = CleanEventStaging::new();
                    loop {
                        let s = shard_index.fetch_add(1, Ordering::Relaxed);
                        if s >= DEDUP_SHARDS {
                            break;
                        }
                        let total: usize = worker_event_buckets.iter().map(|w| w.buckets[s].len()).sum();
                        let cap = ((total * 3 / 2).next_power_of_two()).max(64);
                        if table.len() < cap {
                            table.resize(cap, DedupSlot { key: 0, ingest_sec: 0, loc: u32::MAX });
                        }
                        for (worker, bucket) in worker_event_buckets.iter().enumerate() {
                            for (group_index, group) in bucket.buckets[s].chunks(64).enumerate() {
                                let mut initial_positions = [0usize; 64];
                                let base_idx = group_index * 64;
                                for (j, ev) in group.iter().enumerate() {
                                    let pos = dedup_hash(ev.eid_key) as usize & (cap - 1);
                                    initial_positions[j] = pos;
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        std::arch::x86_64::_mm_prefetch(
                                            table.as_ptr().add(pos) as *const i8,
                                            std::arch::x86_64::_MM_HINT_T0,
                                        );
                                    }
                                }
                                for (j, ev) in group.iter().enumerate() {
                                let idx = base_idx + j;
                                let key = ev.eid_key;
                                let ingest_sec = ev.ingest_sec;
                                let loc = ((worker as u32) << 27) | idx as u32;
                                let mut pos = initial_positions[j];
                                loop {
                                    if table[pos].key == 0 {
                                        table[pos] = DedupSlot { key, ingest_sec, loc };
                                        occupied.push(pos as u32);
                                        break;
                                    }
                                    if table[pos].key == key {
                                        let is_earlier = if ingest_sec != i32::MIN && table[pos].ingest_sec != i32::MIN {
                                            ingest_sec < table[pos].ingest_sec
                                        } else {
                                            let cur_loc = table[pos].loc;
                                            let cur_w = (cur_loc >> 27) as usize;
                                            let cur_i = (cur_loc & ((1u32 << 27) - 1)) as usize;
                                            event_ingest_sec(&worker_event_buckets[cur_w], s, cur_i)
                                                > event_ingest_sec(bucket, s, idx)
                                        };
                                        if is_earlier {
                                            table[pos] = DedupSlot { key, ingest_sec, loc };
                                        }
                                        break;
                                    }
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        let ahead = (pos + 8) & (cap - 1);
                                        std::arch::x86_64::_mm_prefetch(
                                            table.as_ptr().add(ahead) as *const i8,
                                            std::arch::x86_64::_MM_HINT_T0,
                                        );
                                    }
                                    pos = (pos + 1) & (cap - 1);
                                }
                                }
                            }
                        }
                        for group in occupied.chunks(64) {
                            let mut user_keys = [0u64; 64];
                            let mut user_hashes = [0u64; 64];
                            let mut bloom_words = [0usize; 64];
                            let mut valid = [false; 64];
                            for (j, &pos) in group.iter().enumerate() {
                                let slot = table[pos as usize];
                                let w = (slot.loc >> 27) as usize;
                                let i = (slot.loc & ((1u32 << 27) - 1)) as usize;
                                let ev = worker_event_buckets[w].buckets[s][i];
                                if ev.price_cat_flags & (1 << 30) != 0 {
                                    let hash = mix_key(ev.user_key);
                                    let bloom_word = CanonicalMap::bloom_index(hash);
                                    user_keys[j] = ev.user_key;
                                    user_hashes[j] = hash;
                                    bloom_words[j] = bloom_word;
                                    valid[j] = true;
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        std::arch::x86_64::_mm_prefetch(
                                            canon_map.bloom.as_ptr().add(bloom_word) as *const i8,
                                            std::arch::x86_64::_MM_HINT_T0,
                                        );
                                    }
                                }
                            }
                            for j in 0..group.len() {
                                if valid[j]
                                    && canon_map.bloom[bloom_words[j]]
                                        & (1u64 << (user_hashes[j] as usize & 63))
                                        != 0
                                {
                                    let slot = canon_map.initial_slot(user_hashes[j]);
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        std::arch::x86_64::_mm_prefetch(
                                            canon_map.entries.as_ptr().add(slot) as *const i8,
                                            std::arch::x86_64::_MM_HINT_T0,
                                        );
                                    }
                                }
                            }
                            for (j, &pos) in group.iter().enumerate() {
                                if !valid[j] {
                                    continue;
                                }
                                let slot = table[pos as usize];
                                let w = (slot.loc >> 27) as usize;
                                let i = (slot.loc & ((1u32 << 27) - 1)) as usize;
                                let worker = &worker_event_buckets[w];
                                let ev = worker.buckets[s][i];
                                let hash = user_hashes[j];
                                let person = canon_map.canonical_hashed(user_keys[j], hash);
                                let ps = fast_shard(mix_key(person), PERSON_SHARDS);
                                let loc_key = ((s as u64) << 32) | i as u64;
                                let ts = if ev.ts_off == u32::MAX {
                                    worker.wide_ts[&loc_key]
                                } else {
                                    window_start + ev.ts_off as i64
                                };
                                let category_code = ((ev.price_cat_flags >> 24) & 0x3f) as u16;
                                let category_id = if category_code == 63 {
                                    worker.wide_category[&loc_key]
                                } else {
                                    category_code
                                };
                                let ts_off = if ev.ts_off < META_TAG && category_id < 63 {
                                    ev.ts_off
                                } else {
                                    let mut meta = wide_event_meta.lock().expect("wide event metadata");
                                    let index = meta.len();
                                    assert!(index < META_TAG as usize, "too many wide events");
                                    meta.push(WideEventMeta { ts, category_id });
                                    META_TAG | index as u32
                                };
                                staging.push(&mut outputs, ps, CleanEvent {
                                    person_key: person,
                                    ts_off,
                                    price_cat_flags: ev.price_cat_flags,
                                });
                            }
                        }
                        for pos in occupied.iter().copied() {
                            table[pos as usize] = DedupSlot { key: 0, ingest_sec: 0, loc: u32::MAX };
                        }
                        occupied.clear();
                    }
                    staging.flush(&mut outputs);
                    outputs
                }));
            }
            handles.into_iter().map(|h| h.join().expect("event dedup worker")).collect()
        });
    let wide_event_meta = Arc::try_unwrap(wide_event_meta)
        .expect("wide event metadata references")
        .into_inner()
        .expect("wide event metadata");

    eprintln!("events_dedup: {:?} cumulative: {:?}", t.elapsed(), t_start.elapsed());
    t = std::time::Instant::now();

    // 6. Clicks: parallel parse, dedup & repartition by canonical person
    let click_chunks = read_chunks(&data_dir.join("clicks"));
    let worker_click_buckets = parse_click_chunks(
        &click_chunks,
        worker_count,
        click_start,
        window_end,
        &camp_fast,
        &camp_fallback,
        &ids,
        date_cache,
    );
    let total_click_rows: usize = worker_click_buckets
        .iter()
        .flat_map(|worker| worker.buckets.iter())
        .map(Vec::len)
        .sum();
    let click_shard_index = AtomicUsize::new(0);
    let wide_click_meta = Arc::new(Mutex::new(Vec::<WideClickMeta>::new()));
    let person_click_shards: Vec<Vec<Vec<CleanClick>>> =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for _ in 0..worker_count {
                handles.push(scope.spawn(|| {
                    let mut table = Vec::<DedupSlot>::new();
                    let mut occupied = Vec::<u32>::new();
                    let output_capacity = total_click_rows / worker_count / PERSON_SHARDS + 64;
                    let mut outputs = (0..PERSON_SHARDS)
                        .map(|_| Vec::with_capacity(output_capacity))
                        .collect::<Vec<Vec<CleanClick>>>();
                    let mut staging = CleanClickStaging::new();
                    loop {
                        let s = click_shard_index.fetch_add(1, Ordering::Relaxed);
                        if s >= DEDUP_SHARDS {
                            break;
                        }
                        let total: usize = worker_click_buckets.iter().map(|w| w.buckets[s].len()).sum();
                        let cap = ((total * 3 / 2).next_power_of_two()).max(64);
                        if table.len() < cap {
                            table.resize(cap, DedupSlot { key: 0, ingest_sec: 0, loc: u32::MAX });
                        }
                        for (worker, bucket) in worker_click_buckets.iter().enumerate() {
                            for (group_index, group) in bucket.buckets[s].chunks(64).enumerate() {
                                let mut initial_positions = [0usize; 64];
                                let base_idx = group_index * 64;
                                for (j, cl) in group.iter().enumerate() {
                                    let pos = dedup_hash(cl.click_key) as usize & (cap - 1);
                                    initial_positions[j] = pos;
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        std::arch::x86_64::_mm_prefetch(
                                            table.as_ptr().add(pos) as *const i8,
                                            std::arch::x86_64::_MM_HINT_T0,
                                        );
                                    }
                                }
                                for (j, cl) in group.iter().enumerate() {
                                let idx = base_idx + j;
                                let key = cl.click_key;
                                let loc = ((worker as u32) << 27) | idx as u32;
                                let mut pos = initial_positions[j];
                                loop {
                                    if table[pos].key == 0 {
                                        table[pos] = DedupSlot { key, ingest_sec: 0, loc };
                                        occupied.push(pos as u32);
                                        break;
                                    }
                                    if table[pos].key == key {
                                        break;
                                    }
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        let ahead = (pos + 8) & (cap - 1);
                                        std::arch::x86_64::_mm_prefetch(
                                            table.as_ptr().add(ahead) as *const i8,
                                            std::arch::x86_64::_MM_HINT_T0,
                                        );
                                    }
                                    pos = (pos + 1) & (cap - 1);
                                }
                                }
                            }
                        }
                        for group in occupied.chunks(64) {
                            let mut user_keys = [0u64; 64];
                            let mut user_hashes = [0u64; 64];
                            let mut bloom_words = [0usize; 64];
                            for (j, &pos) in group.iter().enumerate() {
                                let slot = table[pos as usize];
                                let w = (slot.loc >> 27) as usize;
                                let i = (slot.loc & ((1u32 << 27) - 1)) as usize;
                                let cl = worker_click_buckets[w].buckets[s][i];
                                let hash = mix_key(cl.user_key);
                                let bloom_word = CanonicalMap::bloom_index(hash);
                                user_keys[j] = cl.user_key;
                                user_hashes[j] = hash;
                                bloom_words[j] = bloom_word;
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    std::arch::x86_64::_mm_prefetch(
                                        canon_map.bloom.as_ptr().add(bloom_word) as *const i8,
                                        std::arch::x86_64::_MM_HINT_T0,
                                    );
                                }
                            }
                            for j in 0..group.len() {
                                if canon_map.bloom[bloom_words[j]]
                                    & (1u64 << (user_hashes[j] as usize & 63))
                                    != 0
                                {
                                    let slot = canon_map.initial_slot(user_hashes[j]);
                                    #[cfg(target_arch = "x86_64")]
                                    unsafe {
                                        std::arch::x86_64::_mm_prefetch(
                                            canon_map.entries.as_ptr().add(slot) as *const i8,
                                            std::arch::x86_64::_MM_HINT_T0,
                                        );
                                    }
                                }
                            }
                            for (j, &pos) in group.iter().enumerate() {
                            let slot = table[pos as usize];
                            let w = (slot.loc >> 27) as usize;
                            let i = (slot.loc & ((1u32 << 27) - 1)) as usize;
                            let worker = &worker_click_buckets[w];
                            let cl = worker.buckets[s][i];
                            let person = canon_map.canonical_hashed(user_keys[j], user_hashes[j]);
                            let ps = fast_shard(mix_key(person), PERSON_SHARDS);
                            let loc_key = ((s as u64) << 32) | i as u64;
                            let ts = if cl.ts_off == u32::MAX {
                                worker.wide_ts[&loc_key]
                            } else {
                                click_start + cl.ts_off as i64
                            };
                            let ts_off = if cl.ts_off < META_TAG {
                                cl.ts_off
                            } else {
                                let mut meta = wide_click_meta.lock().expect("wide click metadata");
                                let index = meta.len();
                                assert!(index < META_TAG as usize, "too many wide clicks");
                                meta.push(WideClickMeta(ts));
                                META_TAG | index as u32
                            };
                            staging.push(&mut outputs, ps, CleanClick {
                                person_key: person,
                                ts_off,
                                campaign_idx: cl.campaign_idx,
                                click_key: cl.click_key,
                            });
                            }
                        }
                        for pos in occupied.iter().copied() {
                            table[pos as usize] = DedupSlot { key: 0, ingest_sec: 0, loc: u32::MAX };
                        }
                        occupied.clear();
                    }
                    staging.flush(&mut outputs);
                    outputs
                }));
            }
            handles.into_iter().map(|h| h.join().expect("click dedup worker")).collect()
        });
    let wide_click_meta = Arc::try_unwrap(wide_click_meta)
        .expect("wide click metadata references")
        .into_inner()
        .expect("wide click metadata");

    eprintln!("clicks: {:?} cumulative: {:?}", t.elapsed(), t_start.elapsed());
    t = std::time::Instant::now();

    // 7. Sessionize, attribute, and aggregate per person shard in parallel
    let num_cats = cat_names.len();
    let num_camps = camp_names.len();

    let start_day = window_start.div_euclid(MS_PER_DAY) as i32;
    let end_day = (window_end - 1).div_euclid(MS_PER_DAY) as i32;
    let num_days = (end_day - start_day + 1).max(1) as usize;
    let fast_event_sort = wide_event_meta.is_empty();
    let fast_click_sort = wide_click_meta.is_empty() && !ids.has_fallback();
    let next_person_shard = AtomicUsize::new(0);
    let shard_results: Vec<(Vec<UserDayRow>, Vec<CampaignDayAgg>)> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(scope.spawn(|| {
                let mut events = Vec::<CleanEvent>::with_capacity(total_event_rows / PERSON_SHARDS + 64);
                let mut clicks = Vec::<CleanClick>::with_capacity(total_click_rows / PERSON_SHARDS + 64);
                let mut user_rows = Vec::new();
                let mut camp_agg: Vec<CampaignDayAgg> =
                    (0..num_camps * num_days).map(|_| CampaignDayAgg::default()).collect();
                let mut assist_stamp = vec![0u32; num_camps];
                let mut cur_stamp: u32 = 0;
                let mut day_acc = vec![(0i64, 0i64, 0i64, false); num_days];
                let mut day_cat_rev = vec![0i64; num_days * num_cats];
                let mut touched_days = Vec::with_capacity(num_days);

                loop {
                    let s = next_person_shard.fetch_add(1, Ordering::Relaxed);
                    if s >= PERSON_SHARDS { break; }
                    events.clear();
                    clicks.clear();
                    for producer in &person_event_shards { events.extend_from_slice(&producer[s]); }
                    for producer in &person_click_shards { clicks.extend_from_slice(&producer[s]); }
                    if fast_event_sort {
                        events.sort_unstable_by_key(|ev| (ev.person_key, ev.ts_off));
                    } else {
                        events.sort_unstable_by(|a, b| {
                            a.person_key.cmp(&b.person_key).then_with(|| {
                                event_timestamp(a, window_start, &wide_event_meta)
                                    .cmp(&event_timestamp(b, window_start, &wide_event_meta))
                            })
                        });
                    }
                    if fast_click_sort {
                        clicks.sort_unstable_by_key(|cl| (cl.person_key, cl.ts_off, cl.click_key));
                    } else {
                        clicks.sort_unstable_by(|a, b| {
                            a.person_key.cmp(&b.person_key)
                                .then_with(|| click_timestamp(a, click_start, &wide_click_meta)
                                    .cmp(&click_timestamp(b, click_start, &wide_click_meta)))
                                .then_with(|| ids.cmp_ids(a.click_key, b.click_key, b'k', b'K'))
                        });
                    }

                    let mut ei = 0usize;
                    let mut ci = 0usize;
                    while ei < events.len() {
                        let person = events[ei].person_key;
                        let mut event_end = ei + 1;
                        while event_end < events.len() && events[event_end].person_key == person { event_end += 1; }
                        while ci < clicks.len() && clicks[ci].person_key < person { ci += 1; }
                        let mut click_end = ci;
                        while click_end < clicks.len() && clicks[click_end].person_key == person { click_end += 1; }
                        let person_clicks = &clicks[ci..click_end];

                        let mut click_l = 0usize;
                        let mut click_r = 0usize;
                        let mut i = ei;
                        while i < event_end {
                            let sess_start = event_timestamp(&events[i], window_start, &wide_event_meta);
                            let day = sess_start.div_euclid(MS_PER_DAY) as i32;
                            let d_idx = (day - start_day) as usize;
                            if day_acc[d_idx].0 == 0 { touched_days.push(d_idx); }
                            day_acc[d_idx].0 += 1;
                            let mut sess_events = 0i64;
                            let mut sess_rev = 0i64;
                            let mut sess_has_purchase = false;

                            let mut prev_ts = sess_start;
                            while i < event_end && event_timestamp(&events[i], window_start, &wide_event_meta) - prev_ts <= MS_30_MIN {
                                let ev = &events[i];
                                prev_ts = event_timestamp(ev, window_start, &wide_event_meta);
                                sess_events += 1;
                                if ev.price_cat_flags & (1 << 31) != 0 {
                                    let price = (ev.price_cat_flags & 0x00ff_ffff) as i64;
                                    let category = event_category(ev, &wide_event_meta);
                                    sess_rev += price;
                                    sess_has_purchase = true;
                                    day_cat_rev[d_idx * num_cats + category] += price;
                                    day_acc[d_idx].3 = true;
                                }
                                i += 1;
                            }
                            day_acc[d_idx].1 += sess_events;
                            day_acc[d_idx].2 += sess_rev;

                            let win_lo = sess_start - MS_7_DAYS;
                            while click_r < person_clicks.len()
                                && click_timestamp(&person_clicks[click_r], click_start, &wide_click_meta) <= sess_start
                            { click_r += 1; }
                            while click_l < click_r
                                && click_timestamp(&person_clicks[click_l], click_start, &wide_click_meta) < win_lo
                            { click_l += 1; }
                            if click_l < click_r {
                                let best_ts = click_timestamp(&person_clicks[click_r - 1], click_start, &wide_click_meta);
                                let mut best_idx = click_r - 1;
                                while best_idx > click_l
                                    && click_timestamp(&person_clicks[best_idx - 1], click_start, &wide_click_meta) == best_ts
                                { best_idx -= 1; }
                                let credited_campaign = person_clicks[best_idx].campaign_idx;
                                let entry = &mut camp_agg[credited_campaign as usize * num_days + d_idx];
                                entry.sessions += 1;
                                if sess_has_purchase {
                                    entry.purchasing_sessions += 1;
                                    entry.revenue_cents += sess_rev;
                                }
                                cur_stamp = cur_stamp.wrapping_add(1);
                                if cur_stamp == 0 {
                                    assist_stamp.fill(0);
                                    cur_stamp = 1;
                                }
                                for click in &person_clicks[click_l..click_r] {
                                    let campaign = click.campaign_idx as usize;
                                    if assist_stamp[campaign] != cur_stamp {
                                        assist_stamp[campaign] = cur_stamp;
                                        camp_agg[campaign * num_days + d_idx].assisted_sessions += 1;
                                    }
                                }
                            } else {
                                let entry = &mut camp_agg[d_idx];
                                entry.sessions += 1;
                                if sess_has_purchase {
                                    entry.purchasing_sessions += 1;
                                    entry.revenue_cents += sess_rev;
                                }
                            }
                        }

                        for &d_idx in &touched_days {
                            let day = start_day + d_idx as i32;
                            let top_category = if day_acc[d_idx].3 {
                                let mut best_category = 0usize;
                                let mut best_revenue = -1i64;
                                for category in 0..num_cats {
                                    let revenue = day_cat_rev[d_idx * num_cats + category];
                                    if revenue > 0 && (revenue > best_revenue
                                        || (revenue == best_revenue && cat_names[category] < cat_names[best_category]))
                                    {
                                        best_revenue = revenue;
                                        best_category = category;
                                    }
                                }
                                Some(best_category as u16)
                            } else { None };
                            user_rows.push(UserDayRow {
                                user_key: person,
                                day,
                                sessions: day_acc[d_idx].0,
                                events: day_acc[d_idx].1,
                                revenue_cents: day_acc[d_idx].2,
                                top_category,
                            });
                            day_acc[d_idx] = (0, 0, 0, false);
                            day_cat_rev[d_idx * num_cats..(d_idx + 1) * num_cats].fill(0);
                        }
                        touched_days.clear();
                        ei = event_end;
                        ci = click_end;
                    }
                }
                (user_rows, camp_agg)
            }));
        }
        handles.into_iter().map(|handle| handle.join().expect("sessionize worker")).collect()
    });

    // 8. Merge & Write Parquet
    fs::create_dir_all(out_dir.join("user_daily")).expect("create user_daily dir");
    let fallback_ids = ids.fallback_snapshot();
    for part in 0..8 {
        let start = part * shard_results.len() / 8;
        let end = ((part + 1) * shard_results.len() / 8).min(shard_results.len());
        let slices: Vec<&[UserDayRow]> = shard_results[start..end]
            .iter()
            .map(|(rows, _)| rows.as_slice())
            .collect();
        write_user_daily_shard(&out_dir.join("user_daily"), part, &slices, &cat_names, &fallback_ids);
    }

    let mut final_camp: FxHashMap<(u16, i32), CampaignDayAgg> = FxHashMap::default();
    for (_, ca) in shard_results {
        for (idx, v) in ca.into_iter().enumerate() {
            if v.sessions != 0 || v.assisted_sessions != 0 {
                let cidx = idx / num_days;
                let day = start_day + (idx % num_days) as i32;
                let e = final_camp.entry((cidx as u16, day)).or_default();
                e.sessions += v.sessions;
                e.purchasing_sessions += v.purchasing_sessions;
                e.revenue_cents += v.revenue_cents;
                e.assisted_sessions += v.assisted_sessions;
            }
        }
    }

    write_campaign_daily(&out_dir.join("campaign_daily"), &final_camp, &camp_names, &camp_channels);
    eprintln!("sessionize_write: {:?} cumulative: {:?}", t.elapsed(), t_start.elapsed());
    eprintln!("total: {:?}", t_start.elapsed());
    std::process::exit(0);
}

fn main() {
    let overall_start = std::time::Instant::now();
    let data_dir = PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out_dir = PathBuf::from(env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    let worker_count = env::var("NCPU")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |count| count.get()));

    let manifest_str = fs::read_to_string(data_dir.join("manifest.json")).expect("read manifest");
    let manifest: Manifest = serde_json::from_str(&manifest_str).expect("parse manifest");
    let window_start = parse_ts(manifest.window_start.as_bytes()).expect("window_start");
    let window_end = parse_ts(manifest.window_end.as_bytes()).expect("window_end");
    let click_start = window_start - MS_7_DAYS;
    let date_cache = DateCache::new(window_start, window_end);

    let mut cat_names = vec!["unknown".to_string()];
    let mut cat_map: FxHashMap<String, u16> = FxHashMap::default();
    cat_map.insert("unknown".to_string(), 0);
    let mut prod_fast = vec![0u16; 65_536];
    let mut prod_fallback: FxHashMap<Box<[u8]>, u16> = FxHashMap::default();
    for_each_data_line(&data_dir.join("products.csv"), |line| {
        if let Some([product, category]) = parse_csv_line::<2>(line) {
            let category_name = std::str::from_utf8(category).unwrap_or("unknown");
            let category_id = if let Some(&known) = cat_map.get(category_name) {
                known
            } else {
                let id = cat_names.len() as u16;
                cat_names.push(category_name.to_string());
                cat_map.insert(category_name.to_string(), id);
                id
            };
            if let Some(index) = parse_numeric_id(product, b"p", 65_535) {
                prod_fast[index] = category_id;
            } else {
                prod_fallback.insert(product.into(), category_id);
            }
        }
    });

    let mut camp_names = vec!["organic".to_string()];
    let mut camp_channels = vec!["organic".to_string()];
    let mut camp_fast = vec![0u16; 4_096];
    let mut camp_fallback: FxHashMap<Box<[u8]>, u16> = FxHashMap::default();
    for_each_data_line(&data_dir.join("campaigns.csv"), |line| {
        if let Some([campaign, channel]) = parse_csv_line::<2>(line) {
            let index = camp_names.len() as u16;
            camp_names.push(String::from_utf8_lossy(campaign).into_owned());
            camp_channels.push(String::from_utf8_lossy(channel).into_owned());
            if let Some(id) = parse_numeric_id(campaign, b"c_", 4_095) {
                camp_fast[id] = index;
            } else {
                camp_fallback.insert(campaign.into(), index);
            }
        }
    });

    let ids = IdPool::new();
    let timings = run_persistent_pipeline(PipelineConfig {
        data_dir: &data_dir,
        out_dir: &out_dir,
        worker_count,
        overall_start,
        window_start,
        window_end,
        click_start,
        date_cache,
        prod_fast: &prod_fast,
        prod_fallback: &prod_fallback,
        camp_fast: &camp_fast,
        camp_fallback: &camp_fallback,
        ids: &ids,
        cat_names: &cat_names,
        camp_names: &camp_names,
        camp_channels: &camp_channels,
    });
    eprintln!("links: {:?} cumulative: {:?}", timings.links.elapsed, timings.links.cumulative);
    eprintln!(
        "events_parse: {:?} cumulative: {:?}",
        timings.events_parse.elapsed, timings.events_parse.cumulative
    );
    eprintln!(
        "events_dedup: {:?} cumulative: {:?}",
        timings.events_dedup.elapsed, timings.events_dedup.cumulative
    );
    eprintln!("clicks: {:?} cumulative: {:?}", timings.clicks.elapsed, timings.clicks.cumulative);
    eprintln!(
        "sessionize_write: {:?} cumulative: {:?}",
        timings.sessionize_write.elapsed, timings.sessionize_write.cumulative
    );
    eprintln!("total: {:?}", overall_start.elapsed());
    std::process::exit(0);
}

fn write_campaign_daily(
    dir: &Path,
    agg: &FxHashMap<(u16, i32), CampaignDayAgg>,
    camp_names: &[String],
    camp_channels: &[String],
) {
    let n = agg.len();
    let mut campaign_ids = Vec::with_capacity(n * 12);
    let mut channels = Vec::with_capacity(n * 12);
    let mut days = Vec::with_capacity(n * 4);
    let mut sessions = Vec::with_capacity(n * 8);
    let mut purchasing_sessions = Vec::with_capacity(n * 8);
    let mut revenue = Vec::with_capacity(n * 8);
    let mut assisted_sessions = Vec::with_capacity(n * 8);

    for (&(cidx, day), v) in agg {
        append_plain_byte_array(&mut campaign_ids, camp_names[cidx as usize].as_bytes());
        append_plain_byte_array(&mut channels, camp_channels[cidx as usize].as_bytes());
        days.extend_from_slice(&day.to_le_bytes());
        sessions.extend_from_slice(&v.sessions.to_le_bytes());
        purchasing_sessions.extend_from_slice(&v.purchasing_sessions.to_le_bytes());
        revenue.extend_from_slice(&v.revenue_cents.to_le_bytes());
        assisted_sessions.extend_from_slice(&v.assisted_sessions.to_le_bytes());
    }

    use parquet_writer::PhysicalType::{ByteArray, Int32, Int64};
    let columns = [
        parquet_writer::Column {
            name: "campaign_id",
            physical_type: ByteArray,
            converted_type: Some(0),
            values: &campaign_ids,
            definition_levels: None,
        },
        parquet_writer::Column {
            name: "channel",
            physical_type: ByteArray,
            converted_type: Some(0),
            values: &channels,
            definition_levels: None,
        },
        parquet_writer::Column {
            name: "day",
            physical_type: Int32,
            converted_type: Some(6),
            values: &days,
            definition_levels: None,
        },
        parquet_writer::Column {
            name: "sessions",
            physical_type: Int64,
            converted_type: None,
            values: &sessions,
            definition_levels: None,
        },
        parquet_writer::Column {
            name: "purchasing_sessions",
            physical_type: Int64,
            converted_type: None,
            values: &purchasing_sessions,
            definition_levels: None,
        },
        parquet_writer::Column {
            name: "revenue_cents",
            physical_type: Int64,
            converted_type: None,
            values: &revenue,
            definition_levels: None,
        },
        parquet_writer::Column {
            name: "assisted_sessions",
            physical_type: Int64,
            converted_type: None,
            values: &assisted_sessions,
            definition_levels: None,
        },
    ];
    parquet_writer::write_table(
        &dir.join("part-0.parquet"),
        n,
        &columns,
    )
    .expect("write campaign_daily parquet");
}