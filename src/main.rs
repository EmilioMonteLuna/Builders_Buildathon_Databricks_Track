use arrow::array::{Date32Builder, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use memchr::memchr;
use memmap2::Mmap;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::env;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use mimalloc::MiMalloc;
use std::sync::Mutex;

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

    fn get(&self, b: &[u8]) -> Option<i64> {
        if b.len() < 10 { return None; }
        self.get_with_day(b, parse_2d(&b[8..10])?)
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
    for (i, &c) in b[1..].iter().enumerate() {
        let value = table[c as usize];
        if value == 0 { return None; }
        packed |= (value as u64) << (54 - i * 6);
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

    fn has_fallback(&self) -> bool {
        !self.fallback.lock().expect("fallback ID pool").0.is_empty()
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
    if has_quote {
        if let Some(fields) = parse_csv_line_fast::<N>(&data[start..end]) {
            callback(fields);
        }
        return;
    }
    if comma_count != N.saturating_sub(1) {
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
    let mut has_quote = false;
    for (pos, &byte) in data.iter().enumerate() {
        match byte {
            b',' => {
                if comma_count < N {
                    commas[comma_count] = pos;
                    comma_count += 1;
                } else {
                    comma_overflow = true;
                }
            }
            b'"' => has_quote = true,
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
        let mut mask = newline_mask | comma_mask | quote_mask;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            let pos = block_start + bit;
            let bit_mask = 1u64 << bit;
            if newline_mask & bit_mask != 0 {
                if !comma_overflow {
                    finish_csv_line::<N, F>(data, line_start, pos, &commas, comma_count, has_quote, callback);
                }
                line_start = pos + 1;
                comma_count = 0;
                comma_overflow = false;
                has_quote = false;
                newline_mask &= !bit_mask;
            } else if comma_mask & bit_mask != 0 {
                if comma_count < N {
                    commas[comma_count] = pos;
                    comma_count += 1;
                } else {
                    comma_overflow = true;
                }
                comma_mask &= !bit_mask;
            } else {
                has_quote = true;
                quote_mask &= !bit_mask;
            }
            mask &= mask - 1;
        }
        block_start += 64;
    }
    for pos in block_start..data.len() {
        match data[pos] {
            b',' => {
                if comma_count < N {
                    commas[comma_count] = pos;
                    comma_count += 1;
                } else {
                    comma_overflow = true;
                }
            }
            b'"' => has_quote = true,
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
    if b.len() < 19
        || b[4] != b'-' || b[7] != b'-' || b[10] != b'T'
        || b[13] != b':' || b[16] != b':'
    {
        return None;
    }
    let d = parse_2d(&b[8..10])?;
    let hh = parse_2d(&b[11..13])?;
    let mm = parse_2d(&b[14..16])?;
    let ss = parse_2d(&b[17..19])?;
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
    if b.len() != 20 || b[19] != b'Z'
        || b[4] != b'-' || b[7] != b'-' || b[10] != b'T'
        || b[13] != b':' || b[16] != b':' {
        return None;
    }
    if let Some(day_ms) = cache.get(b) {
        let hh = parse_2d(&b[11..13])?;
        let mm = parse_2d(&b[14..16])?;
        let ss = parse_2d(&b[17..19])?;
        if hh <= 23 && mm <= 59 && ss <= 59 {
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
        let d0 = b[0].wrapping_sub(b'0');
        let d1 = b[1].wrapping_sub(b'0');
        let d2 = b[2].wrapping_sub(b'0');
        let d3 = b[3].wrapping_sub(b'0');
        let d4 = b[4].wrapping_sub(b'0');
        let d5 = b[5].wrapping_sub(b'0');
        let d6 = b[6].wrapping_sub(b'0');
        let d7 = b[7].wrapping_sub(b'0');
        let d8 = b[8].wrapping_sub(b'0');
        let d9 = b[9].wrapping_sub(b'0');
        let d10 = b[10].wrapping_sub(b'0');
        let d11 = b[11].wrapping_sub(b'0');
        let d12 = b[12].wrapping_sub(b'0');
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
            libc::MADV_SEQUENTIAL | libc::MADV_WILLNEED,
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

const FILE_CHUNK_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone)]
struct FileChunk {
    path: PathBuf,
    start: u64,
    end: u64,
    content_start: u64,
    file_len: u64,
}

fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        file.seek_read(buf, offset)
    }
}

fn fill_at(
    file: &File,
    buffer: &mut Vec<u8>,
    file_start: u64,
    target_len: usize,
    file_len: u64,
) -> io::Result<()> {
    let target = target_len.min(file_len.saturating_sub(file_start) as usize);
    while buffer.len() < target {
        let start = buffer.len();
        let end = (start + 256 * 1024).min(target);
        buffer.resize(end, 0);
        let mut read = 0usize;
        while read < end - start {
            let n = read_at(file, &mut buffer[start + read..end], file_start + (start + read) as u64)?;
            if n == 0 {
                buffer.truncate(start + read);
                return Ok(());
            }
            read += n;
        }
    }
    Ok(())
}

fn read_header_end(file: &File, file_len: u64) -> io::Result<Option<u64>> {
    let mut offset = 0u64;
    let mut buffer = [0u8; 4096];
    while offset < file_len {
        let len = (file_len - offset).min(buffer.len() as u64) as usize;
        let n = read_at(file, &mut buffer[..len], offset)?;
        if n == 0 {
            break;
        }
        if let Some(pos) = memchr(b'\n', &buffer[..n]) {
            return Ok(Some(offset + pos as u64 + 1));
        }
        offset += n as u64;
    }
    Ok(None)
}

fn read_chunks(dir: &Path) -> Vec<FileChunk> {
    let mut chunks = Vec::new();
    for path in list_csvs(dir) {
        let file = File::open(&path).expect("open input CSV");
        let file_len = file.metadata().expect("stat input CSV").len();
        let Some(content_start) = read_header_end(&file, file_len).expect("read CSV header") else {
            continue;
        };
        let mut start = content_start;
        while start < file_len {
            let end = (start + FILE_CHUNK_BYTES as u64).min(file_len);
            chunks.push(FileChunk {
                path: path.clone(),
                start,
                end,
                content_start,
                file_len,
            });
            start = end;
        }
    }
    chunks
}

fn read_file_chunk(chunk: &FileChunk, buffer: &mut Vec<u8>) -> io::Result<(usize, usize)> {
    let file = File::open(&chunk.path)?;
    let nominal_len = (chunk.end - chunk.start) as usize;
    buffer.clear();
    fill_at(&file, buffer, chunk.start, nominal_len, chunk.file_len)?;

    let mut line_start = 0usize;
    if chunk.start > chunk.content_start {
        let mut previous = [0u8; 1];
        read_at(&file, &mut previous, chunk.start - 1)?;
        if previous[0] != b'\n' {
            let mut search_from = 0usize;
            loop {
                if let Some(relative) = memchr(b'\n', &buffer[search_from..]) {
                    line_start = search_from + relative + 1;
                    break;
                }
                search_from = buffer.len();
                if chunk.start + buffer.len() as u64 >= chunk.file_len {
                    return Ok((0, 0));
                }
                fill_at(
                    &file,
                    buffer,
                    chunk.start,
                    buffer.len().saturating_add(64 * 1024),
                    chunk.file_len,
                )?;
            }
        }
    }
    if line_start >= nominal_len {
        return Ok((0, 0));
    }

    let search_from = nominal_len.saturating_sub(1);
    loop {
        if let Some(relative) = memchr(b'\n', &buffer[search_from.min(buffer.len())..]) {
            return Ok((line_start, (search_from.min(buffer.len()) + relative + 1).min(buffer.len())));
        }
        if chunk.start + buffer.len() as u64 >= chunk.file_len {
            return Ok((line_start, buffer.len()));
        }
        fill_at(
            &file,
            buffer,
            chunk.start,
            buffer.len().saturating_add(64 * 1024),
            chunk.file_len,
        )?;
    }
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
        self.bloom[bit >> 6] & (1u64 << (bit & 63)) != 0
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
            let (found, value) = self.entries[slot];
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

fn build_canon_map_from_chunks(
    link_chunks: &[FileChunk],
    ids: &IdPool,
    worker_count: usize,
) -> CanonicalMap {
    let next_chunk = AtomicUsize::new(0);
    let pairs_by_worker = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(scope.spawn(|| {
                let mut pairs = Vec::new();
                let mut buffer = Vec::with_capacity(FILE_CHUNK_BYTES);
                loop {
                    let chunk_index = next_chunk.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = link_chunks.get(chunk_index) else { break };
                    let (start, end) = read_file_chunk(chunk, &mut buffer).expect("read links chunk");
                    scan_csv_morsel::<2, _>(&buffer[start..end], |[a, b]| {
                        if !a.is_empty() && !b.is_empty() {
                            pairs.push((ids.key(a, b'u', b'U'), ids.key(b, b'u', b'U')));
                        }
                    });
                }
                pairs
            }));
        }
        handles.into_iter().map(|h| h.join().expect("link parser")).collect::<Vec<_>>()
    });
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

fn parse_event_chunks(
    chunks: &[FileChunk],
    worker_count: usize,
    window_start: i64,
    window_end: i64,
    prod_fast: &[u16],
    prod_fallback: &FxHashMap<Box<[u8]>, u16>,
    ids: &IdPool,
    date_cache: DateCache,
) -> Vec<EventWorkerBuckets> {
    let event_index = AtomicUsize::new(0);
    let estimated_bytes: usize = chunks.iter().map(|chunk| (chunk.end - chunk.start) as usize).sum();
    let per_worker_bucket = estimated_bytes / worker_count / DEDUP_SHARDS / 70;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(scope.spawn(|| {
                let mut buckets = (0..DEDUP_SHARDS)
                    .map(|_| Vec::with_capacity(per_worker_bucket))
                    .collect::<Vec<Vec<RawEvent>>>();
                let mut staging = EventStaging::new();
                let mut wide_ingest = FxHashMap::default();
                let mut wide_ts = FxHashMap::default();
                let mut wide_category = FxHashMap::default();
                let mut date_cache = date_cache;
                let mut buffer = Vec::with_capacity(FILE_CHUNK_BYTES);
                loop {
                    let chunk_index = event_index.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(chunk_index) else { break };
                    let (start, end) = read_file_chunk(chunk, &mut buffer).expect("read events chunk");
                    scan_csv_morsel::<7, _>(&buffer[start..end], |fields| {
                        parse_event_fields(
                            fields,
                            window_start,
                            window_end,
                            prod_fast,
                            prod_fallback,
                            ids,
                            &mut date_cache,
                            &mut buckets,
                            &mut staging,
                            &mut wide_ingest,
                            &mut wide_ts,
                            &mut wide_category,
                        );
                    });
                }
                staging.flush(&mut buckets);
                EventWorkerBuckets { buckets, wide_ingest, wide_ts, wide_category }
            }));
        }
        handles.into_iter().map(|handle| handle.join().expect("event worker")).collect()
    })
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

fn parse_click_chunks(
    chunks: &[FileChunk],
    worker_count: usize,
    click_start: i64,
    window_end: i64,
    camp_fast: &[u16],
    camp_fallback: &FxHashMap<Box<[u8]>, u16>,
    ids: &IdPool,
    date_cache: DateCache,
) -> Vec<ClickWorkerBuckets> {
    let click_index = AtomicUsize::new(0);
    let estimated_bytes: usize = chunks.iter().map(|chunk| (chunk.end - chunk.start) as usize).sum();
    let per_worker_bucket = estimated_bytes / worker_count / DEDUP_SHARDS / 70;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(scope.spawn(|| {
                let mut buckets = (0..DEDUP_SHARDS)
                    .map(|_| Vec::with_capacity(per_worker_bucket))
                    .collect::<Vec<Vec<RawClick>>>();
                let mut wide_ts = FxHashMap::default();
                let mut date_cache = date_cache;
                let mut buffer = Vec::with_capacity(FILE_CHUNK_BYTES);
                loop {
                    let chunk_index = click_index.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(chunk_index) else { break };
                    let (start, end) = read_file_chunk(chunk, &mut buffer).expect("read clicks chunk");
                    scan_csv_morsel::<4, _>(&buffer[start..end], |fields| {
                        parse_click_fields(
                            fields,
                            click_start,
                            window_end,
                            camp_fast,
                            camp_fallback,
                            ids,
                            &mut date_cache,
                            &mut buckets,
                            &mut wide_ts,
                        );
                    });
                }
                ClickWorkerBuckets { buckets, wide_ts }
            }));
        }
        handles.into_iter().map(|handle| handle.join().expect("click worker")).collect()
    })
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

#[derive(Clone)]
struct UserDayRow {
    user_key: u64,
    day: i32,
    sessions: i64,
    events: i64,
    revenue_cents: i64,
    top_category: Option<u16>,
}

fn main() {
    let t_start = std::time::Instant::now();
    let mut t = std::time::Instant::now();
    let data_dir = PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let out_dir = PathBuf::from(env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    if let Ok(ncpu) = env::var("NCPU").and_then(|s| s.parse::<usize>().map_err(|_| env::VarError::NotPresent)) {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(ncpu).build_global();
    }
    let worker_count = env::var("NCPU")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| rayon::current_num_threads());

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
    (0..8).into_par_iter().for_each(|part| {
        let start = part * shard_results.len() / 8;
        let end = ((part + 1) * shard_results.len() / 8).min(shard_results.len());
        let slices: Vec<&[UserDayRow]> = shard_results[start..end]
            .iter()
            .map(|(rows, _)| rows.as_slice())
            .collect();
        write_user_daily_shard(&out_dir.join("user_daily"), part, &slices, &cat_names, &fallback_ids);
    });

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

fn write_user_daily_shard(
    dir: &Path,
    shard: usize,
    row_slices: &[&[UserDayRow]],
    cat_names: &[String],
    fallback_ids: &[Box<[u8]>],
) {
    fs::create_dir_all(dir).expect("create user_daily dir");
    let rows_len: usize = row_slices.iter().map(|rows| rows.len()).sum();
    let schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("events", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", DataType::Utf8, true),
    ]));

    let mut uid_b = StringBuilder::with_capacity(rows_len, rows_len * 12);
    let mut day_b = Date32Builder::with_capacity(rows_len);
    let mut ses_b = Int64Builder::with_capacity(rows_len);
    let mut ev_b = Int64Builder::with_capacity(rows_len);
    let mut rev_b = Int64Builder::with_capacity(rows_len);
    let mut cat_b = StringBuilder::with_capacity(rows_len, rows_len * 8);

    for rows in row_slices {
        for r in *rows {
            let mut uid_buf = [0u8; 11];
            let uid = if r.user_key >> 63 != 0 {
                fallback_ids[(r.user_key & !(1u64 << 63)) as usize].as_ref()
            } else {
                unpack_fast_id(r.user_key, b'u', b'U', &mut uid_buf)
            };
            uid_b.append_value(std::str::from_utf8(uid).expect("utf8 uid"));
            day_b.append_value(r.day);
            ses_b.append_value(r.sessions);
            ev_b.append_value(r.events);
            rev_b.append_value(r.revenue_cents);
            match r.top_category {
                Some(c) => cat_b.append_value(&cat_names[c as usize]),
                None => cat_b.append_null(),
            }
        }
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(uid_b.finish()),
            Arc::new(day_b.finish()),
            Arc::new(ses_b.finish()),
            Arc::new(ev_b.finish()),
            Arc::new(rev_b.finish()),
            Arc::new(cat_b.finish()),
        ],
    )
    .expect("user_daily batch");

    let file = File::create(dir.join(format!("part-{shard}.parquet"))).expect("create user_daily parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_statistics_enabled(EnabledStatistics::None)
        .set_dictionary_enabled(false)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("arrow writer");
    writer.write(&batch).expect("write user_daily");
    writer.close().expect("close user_daily");
}

fn write_campaign_daily(
    dir: &Path,
    agg: &FxHashMap<(u16, i32), CampaignDayAgg>,
    camp_names: &[String],
    camp_channels: &[String],
) {
    fs::create_dir_all(dir).expect("create campaign_daily dir");
    let schema = Arc::new(Schema::new(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("channel", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("purchasing_sessions", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("assisted_sessions", DataType::Int64, false),
    ]));

    let n = agg.len();
    let mut cid_b = StringBuilder::with_capacity(n, n * 8);
    let mut ch_b = StringBuilder::with_capacity(n, n * 8);
    let mut day_b = Date32Builder::with_capacity(n);
    let mut ses_b = Int64Builder::with_capacity(n);
    let mut pses_b = Int64Builder::with_capacity(n);
    let mut rev_b = Int64Builder::with_capacity(n);
    let mut ast_b = Int64Builder::with_capacity(n);

    for (&(cidx, day), v) in agg {
        cid_b.append_value(&camp_names[cidx as usize]);
        ch_b.append_value(&camp_channels[cidx as usize]);
        day_b.append_value(day);
        ses_b.append_value(v.sessions);
        pses_b.append_value(v.purchasing_sessions);
        rev_b.append_value(v.revenue_cents);
        ast_b.append_value(v.assisted_sessions);
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(cid_b.finish()),
            Arc::new(ch_b.finish()),
            Arc::new(day_b.finish()),
            Arc::new(ses_b.finish()),
            Arc::new(pses_b.finish()),
            Arc::new(rev_b.finish()),
            Arc::new(ast_b.finish()),
        ],
    )
    .expect("campaign_daily batch");

    let file = File::create(dir.join("part-0.parquet")).expect("create campaign_daily parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_statistics_enabled(EnabledStatistics::None)
        .set_dictionary_enabled(false)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("arrow writer");
    writer.write(&batch).expect("write campaign_daily");
    writer.close().expect("close campaign_daily");
}