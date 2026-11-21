use memchr::memchr;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
use memchr::memchr2_iter;
use memmap2::Mmap;
use mimalloc::MiMalloc;
use rustc_hash::FxHashMap;
use std::env;
use std::fs::{self, File};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

mod parquet_writer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const MS_PER_DAY: i64 = 86_400_000;
const MS_30_MIN: i64 = 1_800_000;
const MS_7_DAYS: i64 = 604_800_000;
const MS_PER_DAY_U32: u32 = 86_400_000;
const MS_30_MIN_U32: u32 = 1_800_000;
const MS_7_DAYS_U32: u32 = 604_800_000;
const DEDUP_SHARDS: usize = 512;
const PERSON_SHARDS: usize = 512;
const META_TAG: u32 = 1 << 31;
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
const PEXT_MASK_7BIT: u64 = 0x7F7F_7F7F_7F7F_7F7F;

#[derive(Clone, Copy)]
struct DedupSlot {
    key: u64,
    ingest_sec: i32,
    loc: u32,
}
const _: [(); 16] = [(); std::mem::size_of::<DedupSlot>()];

// Zero-dependency manifest.json parser
fn extract_json_string<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\"");
    let mut rest = json;
    while let Some(pos) = rest.find(&needle) {
        let after = rest[pos + needle.len()..].trim_start();
        if let Some(after_colon) = after.strip_prefix(':') {
            let after_ws = after_colon.trim_start();
            if let Some(after_quote) = after_ws.strip_prefix('"') {
                if let Some(end) = after_quote.find('"') {
                    return Some(&after_quote[..end]);
                }
            }
        }
        rest = &rest[pos + needle.len()..];
    }
    None
}

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

// 64-entry zero-collision DateCache for any 32-day window (2024..2030+)
#[derive(Clone, Copy)]
struct DateCache {
    entries: [(u64, i64); 64],
}

impl DateCache {
    fn new(window_start: i64, window_end: i64) -> Self {
        let mut entries = [(0u64, 0i64); 64];
        let lo = (window_start - 9 * MS_PER_DAY).div_euclid(MS_PER_DAY);
        let hi = (window_end + 8 * MS_PER_DAY).div_euclid(MS_PER_DAY);
        for day in lo..=hi {
            let ms = day * MS_PER_DAY;
            let (year, month, day_of_month) = civil_from_days(day);
            let prefix: [u8; 8] = format!("{year:04}-{month:02}-")
                .into_bytes()
                .try_into()
                .expect("date prefix length");
            let idx = (((prefix[6] & 1) as usize) << 5) | (day_of_month as usize);
            entries[idx] = (u64::from_ne_bytes(prefix), ms);
        }
        Self { entries }
    }

    #[inline(always)]
    unsafe fn get_unchecked_day(&self, b: &[u8], d: u32) -> Option<i64> {
        if d < 1 || d > 31 {
            return None;
        }
        let key = (b.as_ptr() as *const u64).read_unaligned();
        let idx = (((*b.get_unchecked(6) & 1) as usize) << 5) | (d as usize);
        let entry = *self.entries.get_unchecked(idx);
        if entry.0 == key {
            Some(entry.1)
        } else {
            None
        }
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
    if m <= 2 {
        y += 1;
    }
    (y, m as u32, d as u32)
}

#[inline(always)]
fn fast_shard(key: u64, shards: usize) -> usize {
    ((key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - shards.trailing_zeros())) as usize)
        & (shards - 1)
}

#[inline(always)]
fn mix_key(mut key: u64) -> u64 {
    key ^= key >> 30;
    key = key.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    key ^= key >> 27;
    key = key.wrapping_mul(0x94d0_49bb_1331_11eb);
    key ^ (key >> 31)
}

#[inline(always)]
fn dedup_hash(key: u64) -> u64 {
    mix_key(key)
}

// 1-instruction BMI2 _pext_u64 / _pdep_u64 packing for any 2..=9 byte non-null ASCII ID
// Preserves exact byte-wise lexicographical ordering and is guaranteed non-zero.
#[inline(always)]
fn pack_fast_id(b: &[u8], lo_prefix: u8, hi_prefix: u8) -> Option<u64> {
    let len = b.len();
    if len < 2 || len > 9 {
        return None;
    }
    let first = unsafe { *b.get_unchecked(0) };
    let case_bit = if first == lo_prefix {
        1u64 << 62
    } else if first == hi_prefix {
        0u64
    } else {
        return None;
    };

    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        let (raw_be, fill_mask) = if len == 9 {
            let v = (b.as_ptr().add(1) as *const u64).read_unaligned();
            (u64::from_be(v), 0u64)
        } else {
            let mut tmp = [0u8; 8];
            let tail_len = len - 1;
            std::ptr::copy_nonoverlapping(b.as_ptr().add(1), tmp.as_mut_ptr(), tail_len);
            (
                u64::from_be_bytes(tmp),
                0x0101_0101_0101_0101u64 >> (tail_len * 8),
            )
        };
        if (raw_be & 0x8080_8080_8080_8080) != 0 {
            return None;
        }
        let chk = raw_be | fill_mask;
        if (chk.wrapping_sub(0x0101_0101_0101_0101) & !chk & 0x8080_8080_8080_8080) != 0 {
            return None;
        }
        let packed56 = std::arch::x86_64::_pext_u64(raw_be, PEXT_MASK_7BIT);
        Some(case_bit | packed56)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut packed = case_bit;
        let mut shift = 49u32;
        for i in 1..len {
            let c = unsafe { *b.get_unchecked(i) };
            if c == 0 || c >= 128 {
                return None;
            }
            packed |= (c as u64) << shift;
            shift = shift.wrapping_sub(7);
        }
        Some(packed)
    }
}

#[inline(always)]
fn unpack_fast_id<'a>(val: u64, lo_prefix: u8, hi_prefix: u8, buf: &'a mut [u8; 9]) -> &'a [u8] {
    buf[0] = if (val & (1u64 << 62)) != 0 {
        lo_prefix
    } else {
        hi_prefix
    };

    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        let raw_be = std::arch::x86_64::_pdep_u64(val & ((1u64 << 56) - 1), PEXT_MASK_7BIT);
        let bytes = raw_be.to_be_bytes();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr().add(1), 8);
        let tail_len = 8 - ((raw_be.trailing_zeros() as usize) >> 3);
        &buf[..1 + tail_len]
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut len = 1usize;
        let mut shift = 49u32;
        for _ in 0..8 {
            let c = ((val >> shift) & 0x7F) as u8;
            if c == 0 {
                break;
            }
            buf[len] = c;
            len += 1;
            shift = shift.wrapping_sub(7);
        }
        &buf[..len]
    }
}

struct IdPool {
    fallback: Mutex<(Vec<Box<[u8]>>, FxHashMap<Box<[u8]>, u64>)>,
}

impl IdPool {
    fn new() -> Self {
        Self {
            fallback: Mutex::new((Vec::new(), FxHashMap::default())),
        }
    }

    #[inline(always)]
    fn key(&self, bytes: &[u8], lo: u8, hi: u8) -> u64 {
        if let Some(key) = pack_fast_id(bytes, lo, hi) {
            return key;
        }
        self.key_slow(bytes)
    }

    #[cold]
    fn key_slow(&self, bytes: &[u8]) -> u64 {
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

    #[inline(always)]
    fn cmp_user(&self, a: u64, b: u64) -> std::cmp::Ordering {
        self.cmp_ids(a, b, b'u', b'U')
    }

    #[inline(always)]
    fn cmp_ids(&self, a: u64, b: u64, lo: u8, hi: u8) -> std::cmp::Ordering {
        if a == b {
            return std::cmp::Ordering::Equal;
        }
        if (a | b) >> 63 == 0 {
            return a.cmp(&b);
        }
        self.cmp_ids_slow(a, b, lo, hi)
    }

    #[cold]
    fn cmp_ids_slow(&self, a: u64, b: u64, lo: u8, hi: u8) -> std::cmp::Ordering {
        let pool = self.fallback.lock().expect("fallback ID pool");
        let mut ab = [0u8; 9];
        let mut bb = [0u8; 9];
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
    if line.is_empty() {
        return None;
    }
    let mut out = [&[][..]; N];
    let mut field = 0usize;
    let mut start = 0usize;
    let mut state = 0u8;
    for (i, &b) in line.iter().enumerate() {
        match b {
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
    if field + 1 == N {
        Some(out)
    } else {
        None
    }
}

// Fast finisher for AVX-512 path: internal quotes are ALREADY rejected by 4-instruction bitmask!
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[inline(always)]
unsafe fn finish_csv_line_prechecked<const N: usize, F>(
    data: &[u8],
    start: usize,
    end: usize,
    commas: &[usize; N],
    comma_count: usize,
    callback: &mut F,
) where
    F: for<'a> FnMut([&'a [u8]; N]),
{
    if start == end || comma_count != N - 1 {
        return;
    }
    let mut fields = MaybeUninit::<[&[u8]; N]>::uninit();
    let out = fields.as_mut_ptr().cast::<&[u8]>();
    let ptr = data.as_ptr();
    let mut fs = start;
    for field in 0..N {
        let fe = if field + 1 == N {
            end
        } else {
            *commas.get_unchecked(field)
        };
        if fs == fe {
            out.add(field).write(&[]);
        } else {
            let open_q = *ptr.add(fs) == b'"';
            let close_q = *ptr.add(fe - 1) == b'"';
            if open_q != close_q {
                return;
            }
            if open_q {
                if fe - fs < 2 {
                    return;
                }
                out.add(field)
                    .write(std::slice::from_raw_parts(ptr.add(fs + 1), fe - fs - 2));
            } else {
                out.add(field)
                    .write(std::slice::from_raw_parts(ptr.add(fs), fe - fs));
            }
        }
        fs = fe + 1;
    }
    callback(fields.assume_init());
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
#[inline(always)]
fn finish_csv_line_scalar<const N: usize, F>(
    data: &[u8],
    start: usize,
    end: usize,
    commas: &[usize; N],
    comma_count: usize,
    callback: &mut F,
) where
    F: for<'a> FnMut([&'a [u8]; N]),
{
    if start == end || comma_count != N - 1 {
        return;
    }
    let mut fields = MaybeUninit::<[&[u8]; N]>::uninit();
    let out = fields.as_mut_ptr().cast::<&[u8]>();
    let mut fs = start;
    for field in 0..N {
        let fe = if field + 1 == N {
            end
        } else {
            unsafe { *commas.get_unchecked(field) }
        };
        let mut value = unsafe { data.get_unchecked(fs..fe) };
        if !value.is_empty() {
            if unsafe { *value.get_unchecked(0) } == b'"' {
                if value.len() < 2 || unsafe { *value.get_unchecked(value.len() - 1) } != b'"' {
                    return;
                }
                value = unsafe { value.get_unchecked(1..value.len() - 1) };
                if memchr(b'"', value).is_some() {
                    return;
                }
            } else if memchr(b'"', value).is_some() {
                return;
            }
        }
        unsafe { out.add(field).write(value) };
        fs = fe + 1;
    }
    callback(unsafe { fields.assume_init() });
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

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
fn scan_csv_morsel_scalar<const N: usize, F>(data: &[u8], callback: &mut F)
where
    F: for<'a> FnMut([&'a [u8]; N]),
{
    let mut line_start = 0usize;
    let mut commas = [0usize; N];
    let mut comma_count = 0usize;
    let mut comma_overflow = false;
    for pos in memchr2_iter(b'\n', b',', data) {
        if unsafe { *data.get_unchecked(pos) } == b',' {
            if comma_count < N {
                unsafe { *commas.get_unchecked_mut(comma_count) = pos };
                comma_count += 1;
            } else {
                comma_overflow = true;
            }
        } else {
            if !comma_overflow {
                finish_csv_line_scalar::<N, F>(data, line_start, pos, &commas, comma_count, callback);
            }
            line_start = pos + 1;
            comma_count = 0;
            comma_overflow = false;
        }
    }
    if !comma_overflow {
        finish_csv_line_scalar::<N, F>(data, line_start, data.len(), &commas, comma_count, callback);
    }
}

// Zero-memchr AVX-512 single-pass CSV scanner: validates internal quotes across 64 bytes in 4 ALU ops
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
unsafe fn scan_csv_morsel_avx512<const N: usize, F>(data: &[u8], callback: &mut F)
where
    F: for<'a> FnMut([&'a [u8]; N]),
{
    use std::arch::x86_64::{
        __m512i, _mm512_cmpeq_epi8_mask, _mm512_loadu_si512, _mm512_set1_epi8,
    };
    let n = data.len();
    let ptr = data.as_ptr();
    let v_nl = _mm512_set1_epi8(b'\n' as i8);
    let v_comma = _mm512_set1_epi8(b',' as i8);
    let v_quote = _mm512_set1_epi8(b'"' as i8);

    let mut line_start = 0usize;
    let mut commas = [0usize; N];
    let mut comma_count = 0usize;
    let mut comma_overflow = false;
    let mut bad_quote = false;
    let mut prev_ended_delim = 1u64;
    let mut block_start = 0usize;

    while block_start + 64 <= n {
        let block = _mm512_loadu_si512(ptr.add(block_start) as *const __m512i);
        let nl_mask = _mm512_cmpeq_epi8_mask(block, v_nl);
        let comma_mask = _mm512_cmpeq_epi8_mask(block, v_comma);
        let quote_mask = _mm512_cmpeq_epi8_mask(block, v_quote);

        let delim_mask = nl_mask | comma_mask;
        let next_starts_delim = u64::from(
            block_start + 64 == n || matches!(*ptr.add(block_start + 64), b',' | b'\n'),
        );
        let after_delim = (delim_mask << 1) | prev_ended_delim;
        let before_delim = (delim_mask >> 1) | (next_starts_delim << 63);
        let mut block_bad_quotes = quote_mask & !(after_delim | before_delim);
        prev_ended_delim = delim_mask >> 63;

        let mut mask = delim_mask;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            let bit_mask = 1u64 << bit;
            let pos = block_start + bit;
            if (nl_mask & bit_mask) != 0 {
                let before_bit = bit_mask - 1;
                let line_bad = bad_quote || ((block_bad_quotes & before_bit) != 0);
                if !comma_overflow && !line_bad {
                    finish_csv_line_prechecked::<N, F>(
                        data,
                        line_start,
                        pos,
                        &commas,
                        comma_count,
                        callback,
                    );
                }
                line_start = pos + 1;
                comma_count = 0;
                comma_overflow = false;
                bad_quote = false;
                let through_bit = if bit == 63 {
                    u64::MAX
                } else {
                    (1u64 << (bit + 1)) - 1
                };
                block_bad_quotes &= !through_bit;
            } else if comma_count < N {
                *commas.get_unchecked_mut(comma_count) = pos;
                comma_count += 1;
            } else {
                comma_overflow = true;
            }
            mask &= mask - 1;
        }
        if block_bad_quotes != 0 {
            bad_quote = true;
        }
        block_start += 64;
    }

    for pos in block_start..n {
        match *ptr.add(pos) {
            b'"' => {
                let after_d = pos == 0 || matches!(*ptr.add(pos - 1), b',' | b'\n');
                let before_d = pos + 1 == n || matches!(*ptr.add(pos + 1), b',' | b'\n');
                if !(after_d || before_d) {
                    bad_quote = true;
                }
            }
            b',' => {
                if comma_count < N {
                    *commas.get_unchecked_mut(comma_count) = pos;
                    comma_count += 1;
                } else {
                    comma_overflow = true;
                }
            }
            b'\n' => {
                if !comma_overflow && !bad_quote {
                    finish_csv_line_prechecked::<N, F>(
                        data,
                        line_start,
                        pos,
                        &commas,
                        comma_count,
                        callback,
                    );
                }
                line_start = pos + 1;
                comma_count = 0;
                comma_overflow = false;
                bad_quote = false;
            }
            _ => {}
        }
    }
    if !comma_overflow && !bad_quote {
        finish_csv_line_prechecked::<N, F>(data, line_start, n, &commas, comma_count, callback);
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

#[inline(always)]
fn parse_2d(b: &[u8]) -> Option<u32> {
    let d0 = b[0].wrapping_sub(b'0');
    let d1 = b[1].wrapping_sub(b'0');
    if (d0 | d1) > 9 {
        return None;
    }
    Some((d0 as u32) * 10 + (d1 as u32))
}

#[inline(always)]
unsafe fn parse_2d_at_unchecked(b: &[u8], offset: usize) -> Option<u32> {
    let d0 = (*b.get_unchecked(offset)).wrapping_sub(b'0');
    let d1 = (*b.get_unchecked(offset + 1)).wrapping_sub(b'0');
    if (d0 | d1) > 9 {
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

// SWAR HH:MM:SS parser (8 bytes starting at ptr)
#[inline(always)]
unsafe fn swar_hms(ptr: *const u8) -> Option<i64> {
    let v = u64::from_le((ptr as *const u64).read_unaligned());
    if (v & 0x0000_FF00_00FF_0000) != 0x0000_3A00_003A_0000 {
        return None;
    }
    let dv = v ^ 0x0000_0A00_000A_0000;
    const M: u64 = 0xF0F0_F0F0_F0F0_F0F0;
    const T: u64 = 0x3030_3030_3030_3030;
    if (((dv ^ T) | (dv.wrapping_add(0x0606_0606_0606_0606) ^ T)) & M) != 0 {
        return None;
    }
    let d = dv - T;
    let hh = (d & 0xFF) * 10 + ((d >> 8) & 0xFF);
    let mm = ((d >> 24) & 0xFF) * 10 + ((d >> 32) & 0xFF);
    let ss = ((d >> 48) & 0xFF) * 10 + ((d >> 56) & 0xFF);
    if hh > 23 || mm > 59 || ss > 59 {
        return None;
    }
    Some(((hh * 3600 + mm * 60 + ss) * 1000) as i64)
}

// SWAR 8-digit & 4-digit decimal parsers for 13-digit F3 timestamps
#[inline(always)]
unsafe fn swar_8_digits(ptr: *const u8) -> Option<u64> {
    let v = u64::from_le((ptr as *const u64).read_unaligned());
    const M: u64 = 0xF0F0_F0F0_F0F0_F0F0;
    const T: u64 = 0x3030_3030_3030_3030;
    if (((v ^ T) | (v.wrapping_add(0x0606_0606_0606_0606) ^ T)) & M) != 0 {
        return None;
    }
    let mut val = v - T;
    val = (val.wrapping_mul(10) + (val >> 8)) & 0x00FF_00FF_00FF_00FF;
    val = (val.wrapping_mul(100) + (val >> 16)) & 0x0000_FFFF_0000_FFFF;
    val = (val.wrapping_mul(10_000) + (val >> 32)) & 0x0000_0000_FFFF_FFFF;
    Some(val)
}

#[inline(always)]
unsafe fn swar_4_digits(ptr: *const u8) -> Option<u64> {
    let v = u32::from_le((ptr as *const u32).read_unaligned());
    const M: u32 = 0xF0F0_F0F0;
    const T: u32 = 0x3030_3030;
    if (((v ^ T) | (v.wrapping_add(0x0606_0606) ^ T)) & M) != 0 {
        return None;
    }
    let mut val = v - T;
    val = (val.wrapping_mul(10) + (val >> 8)) & 0x00FF_00FF;
    val = (val.wrapping_mul(100) + (val >> 16)) & 0x0000_FFFF;
    Some(val as u64)
}

#[inline]
fn parse_iso_base(b: &[u8]) -> Option<i64> {
    if b.len() < 19
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
    {
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
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => unreachable!(),
    };
    if d > max_d {
        return None;
    }
    let days = days_from_civil(y, m, d);
    Some((days * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + (ss as i64)) * 1000)
}

#[inline(always)]
fn parse_iso_base_cached(b: &[u8], cache: &DateCache) -> Option<i64> {
    unsafe {
        if *b.get_unchecked(10) != b'T' {
            return None;
        }
        let d = parse_2d_at_unchecked(b, 8)?;
        if let Some(day_ms) = cache.get_unchecked_day(b, d) {
            let hms_ms = swar_hms(b.as_ptr().add(11))?;
            return Some(day_ms + hms_ms);
        }
    }
    parse_iso_base(b)
}

#[inline(always)]
fn parse_f1_ingest(b: &[u8], cache: &DateCache) -> Option<i64> {
    if b.len() != 20 {
        return None;
    }
    unsafe {
        if *b.get_unchecked(19) != b'Z' || *b.get_unchecked(10) != b'T' {
            return None;
        }
        let d = parse_2d_at_unchecked(b, 8)?;
        if let Some(day_ms) = cache.get_unchecked_day(b, d) {
            let hms_ms = swar_hms(b.as_ptr().add(11))?;
            return Some(day_ms + hms_ms);
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
    if b.len() == 25 && (b[19] == b'+' || b[19] == b'-') && b[22] == b':' {
        let base = parse_iso_base(b)?;
        let oh = parse_2d(&b[20..22])?;
        let om = parse_2d(&b[23..25])?;
        if oh > 23 || om > 59 {
            return None;
        }
        let off_ms = ((oh as i64) * 3600 + (om as i64) * 60) * 1000;
        return Some(if b[19] == b'+' {
            base - off_ms
        } else {
            base + off_ms
        });
    }
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

// Ordered by frequency: F1 (60%) -> F3 13-digit (30%) -> F2 offset (10%) -> general fallback
#[inline(always)]
fn parse_ts_cached(b: &[u8], cache: &DateCache) -> Option<i64> {
    match b.len() {
        20 => unsafe {
            if *b.get_unchecked(19) == b'Z' {
                parse_iso_base_cached(b, cache)
            } else {
                parse_ts(b)
            }
        },
        13 => unsafe {
            if let (Some(hi), Some(mid)) = (
                swar_8_digits(b.as_ptr()),
                swar_4_digits(b.as_ptr().add(8)),
            ) {
                let d12 = (*b.get_unchecked(12)).wrapping_sub(b'0');
                if d12 <= 9 {
                    return Some((hi * 100_000 + mid * 10 + d12 as u64) as i64);
                }
            }
            None
        },
        25 => unsafe {
            let sign = *b.get_unchecked(19);
            if (sign == b'+' || sign == b'-') && *b.get_unchecked(22) == b':' {
                let base = parse_iso_base_cached(b, cache)?;
                let oh = parse_2d_at_unchecked(b, 20)?;
                let om = parse_2d_at_unchecked(b, 23)?;
                if oh > 23 || om > 59 {
                    return None;
                }
                let offset = (oh as i64 * 3600 + om as i64 * 60) * 1000;
                Some(if sign == b'+' {
                    base - offset
                } else {
                    base + offset
                })
            } else {
                parse_ts(b)
            }
        },
        _ => parse_ts(b),
    }
}

#[inline(always)]
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

#[inline(always)]
fn parse_event_fields(
    fields: [&[u8]; 7],
    window_start: i64,
    window_start_sec: i64,
    window_end: i64,
    prod_fast: &[u16],
    prod_fallback: &FxHashMap<Box<[u8]>, u16>,
    ids: &IdPool,
    date_cache: &DateCache,
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
    let user_key = if price_cat_flags & (1 << 30) != 0 {
        ids.key(uid, b'u', b'U')
    } else {
        0
    };
    let shard = fast_shard(eid_key, DEDUP_SHARDS);
    let ingest_offset_sec = ingest_ts.div_euclid(1_000) - window_start_sec;
    let ingest_sec = i32::try_from(ingest_offset_sec).unwrap_or(i32::MIN);
    let ts_off = timestamp
        .and_then(|ts| u32::try_from(ts - window_start).ok())
        .unwrap_or(u32::MAX);
    let idx = staging.push(
        buckets,
        shard,
        RawEvent {
            eid_key,
            user_key,
            ingest_sec,
            ts_off,
            price_cat_flags,
        },
    );
    if ingest_sec == i32::MIN {
        wide_ingest.insert(((shard as u64) << 32) | idx as u64, ingest_offset_sec);
    }
    if ts_off == u32::MAX {
        if let Some(ts) = timestamp {
            wide_ts.insert(((shard as u64) << 32) | idx as u64, ts);
        }
    }
    if category_id >= 63 {
        wide_category.insert(((shard as u64) << 32) | idx as u64, category_id);
    }
}

#[inline(always)]
fn parse_click_fields(
    fields: [&[u8]; 4],
    click_start: i64,
    window_end: i64,
    camp_fast: &[u16],
    camp_fallback: &FxHashMap<Box<[u8]>, u16>,
    ids: &IdPool,
    date_cache: &DateCache,
    buckets: &mut [Vec<RawClick>],
    staging: &mut RawClickStaging,
    wide_ts: &mut FxHashMap<u64, i64>,
) {
    let [cid, uid, camp_b, ts_b] = fields;
    if cid.is_empty() || uid.is_empty() {
        return;
    }
    let camp_idx = if let Some(idx) = parse_numeric_id(camp_b, b"c_", 4_095) {
        unsafe { *camp_fast.get_unchecked(idx) }
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
    let idx = staging.push(
        buckets,
        shard,
        RawClick {
            click_key,
            user_key,
            ts_off,
            campaign_idx: camp_idx,
        },
    );
    if ts_off == u32::MAX {
        wide_ts.insert(((shard as u64) << 32) | idx as u64, ts);
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

fn mmap_morsels(data: &'static [u8], out: &mut Vec<&'static [u8]>) {
    const MORSEL_BYTES: usize = 4 * 1024 * 1024;
    let mut start = match memchr(b'\n', data) {
        Some(header_end) => header_end + 1,
        None => return,
    };
    while start < data.len() {
        let target = start.saturating_add(MORSEL_BYTES);
        let end = if target >= data.len() {
            data.len()
        } else {
            target + memchr(b'\n', &data[target..]).map_or(data.len() - target, |p| p + 1)
        };
        out.push(&data[start..end]);
        start = end;
    }
}

// Flat interleaved Union-Find (load factor <= 0.67 guaranteed)
struct Dsu {
    parent: Vec<u32>,
    names: Vec<u64>,
    entries: Vec<(u64, u32)>,
    index_mask: usize,
}

impl Dsu {
    fn new(total_pairs: usize) -> Self {
        let cap = total_pairs.saturating_mul(3).next_power_of_two().max(32);
        Self {
            parent: Vec::with_capacity(total_pairs.saturating_mul(2).min(1 << 22)),
            names: Vec::with_capacity(total_pairs.saturating_mul(2).min(1 << 22)),
            entries: vec![(0, u32::MAX); cap],
            index_mask: cap - 1,
        }
    }

    #[inline(always)]
    fn get_or_insert(&mut self, id: u64) -> u32 {
        let mut slot = mix_key(id) as usize & self.index_mask;
        loop {
            let (key, val) = unsafe { *self.entries.get_unchecked(slot) };
            if key == id {
                return val;
            }
            if key == 0 {
                let idx = self.parent.len() as u32;
                self.parent.push(idx);
                self.names.push(id);
                unsafe { *self.entries.get_unchecked_mut(slot) = (id, idx) };
                return idx;
            }
            slot = (slot + 1) & self.index_mask;
        }
    }

    #[inline(always)]
    fn find(&mut self, mut x: u32) -> u32 {
        let mut root = x;
        unsafe {
            while *self.parent.get_unchecked(root as usize) != root {
                root = *self.parent.get_unchecked(root as usize);
            }
            while *self.parent.get_unchecked(x as usize) != root {
                let next = *self.parent.get_unchecked(x as usize);
                *self.parent.get_unchecked_mut(x as usize) = root;
                x = next;
            }
        }
        root
    }

    #[inline(always)]
    fn union(&mut self, a: u32, b: u32, ids: &IdPool) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if ids
            .cmp_user(self.names[ra as usize], self.names[rb as usize])
            .is_lt()
        {
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

    #[inline(always)]
    fn bloom_index(hash: u64) -> usize {
        (hash as usize & ((1usize << 24) - 1)) >> 6
    }

    #[inline(always)]
    fn bloom_contains(&self, hash: u64) -> bool {
        let bit = hash as usize & ((1usize << 24) - 1);
        unsafe { (*self.bloom.get_unchecked(bit >> 6) & (1u64 << (bit & 63))) != 0 }
    }

    #[inline(always)]
    fn initial_slot(&self, hash: u64) -> usize {
        hash as usize & self.mask
    }

    #[inline(always)]
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

// 512 * 2 * 32B = 32 KB (exact L1d fit, 64B cache-line flushes)
struct EventStaging {
    rows: [[RawEvent; 2]; DEDUP_SHARDS],
    counts: [u8; DEDUP_SHARDS],
}

impl EventStaging {
    fn new() -> Self {
        Self {
            rows: [[RawEvent::default(); 2]; DEDUP_SHARDS],
            counts: [0; DEDUP_SHARDS],
        }
    }

    #[inline(always)]
    fn push(&mut self, buckets: &mut [Vec<RawEvent>], shard: usize, event: RawEvent) -> u32 {
        unsafe {
            let count = *self.counts.get_unchecked(shard) as usize;
            let bucket = buckets.get_unchecked_mut(shard);
            let idx = bucket.len() + count;
            *self.rows.get_unchecked_mut(shard).get_unchecked_mut(count) = event;
            if count == 1 {
                bucket.extend_from_slice(self.rows.get_unchecked(shard));
                *self.counts.get_unchecked_mut(shard) = 0;
            } else {
                *self.counts.get_unchecked_mut(shard) = 1;
            }
            idx as u32
        }
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

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawClick {
    click_key: u64,
    user_key: u64,
    ts_off: u32,
    campaign_idx: u16,
}
const _: [(); 24] = [(); std::mem::size_of::<RawClick>()];

struct RawClickStaging {
    rows: [[RawClick; 2]; DEDUP_SHARDS],
    counts: [u8; DEDUP_SHARDS],
}

impl RawClickStaging {
    fn new() -> Self {
        Self {
            rows: [[RawClick::default(); 2]; DEDUP_SHARDS],
            counts: [0; DEDUP_SHARDS],
        }
    }

    #[inline(always)]
    fn push(&mut self, buckets: &mut [Vec<RawClick>], shard: usize, click: RawClick) -> u32 {
        unsafe {
            let count = *self.counts.get_unchecked(shard) as usize;
            let bucket = buckets.get_unchecked_mut(shard);
            let idx = bucket.len() + count;
            *self.rows.get_unchecked_mut(shard).get_unchecked_mut(count) = click;
            if count == 1 {
                bucket.extend_from_slice(self.rows.get_unchecked(shard));
                *self.counts.get_unchecked_mut(shard) = 0;
            } else {
                *self.counts.get_unchecked_mut(shard) = 1;
            }
            idx as u32
        }
    }

    fn flush(&mut self, buckets: &mut [Vec<RawClick>]) {
        for shard in 0..DEDUP_SHARDS {
            let count = self.counts[shard] as usize;
            if count != 0 {
                buckets[shard].extend_from_slice(&self.rows[shard][..count]);
                self.counts[shard] = 0;
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct CleanEvent {
    person_key: u64,
    ts_off: u32,
    price_cat_flags: u32,
}
const _: [(); 16] = [(); std::mem::size_of::<CleanEvent>()];

// 512 * 4 * 16B = 32 KB (exact L1d fit, 64B cache-line flushes)
struct CleanEventStaging {
    rows: [[CleanEvent; 4]; PERSON_SHARDS],
    counts: [u8; PERSON_SHARDS],
}

impl CleanEventStaging {
    fn new() -> Self {
        Self {
            rows: [[CleanEvent::default(); 4]; PERSON_SHARDS],
            counts: [0; PERSON_SHARDS],
        }
    }

    #[inline(always)]
    fn push(&mut self, buckets: &mut [Vec<CleanEvent>], shard: usize, event: CleanEvent) {
        unsafe {
            let count = *self.counts.get_unchecked(shard) as usize;
            *self.rows.get_unchecked_mut(shard).get_unchecked_mut(count) = event;
            let next = count + 1;
            if next == 4 {
                buckets
                    .get_unchecked_mut(shard)
                    .extend_from_slice(self.rows.get_unchecked(shard));
                *self.counts.get_unchecked_mut(shard) = 0;
            } else {
                *self.counts.get_unchecked_mut(shard) = next as u8;
            }
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

#[derive(Clone, Copy, Default)]
struct CleanClick {
    person_key: u64,
    click_key: u64,
    ts_off: u32,
    campaign_idx: u16,
}
const _: [(); 24] = [(); std::mem::size_of::<CleanClick>()];

struct CleanClickStaging {
    rows: [[CleanClick; 2]; PERSON_SHARDS],
    counts: [u8; PERSON_SHARDS],
}

impl CleanClickStaging {
    fn new() -> Self {
        Self {
            rows: [[CleanClick::default(); 2]; PERSON_SHARDS],
            counts: [0; PERSON_SHARDS],
        }
    }

    #[inline(always)]
    fn push(&mut self, buckets: &mut [Vec<CleanClick>], shard: usize, click: CleanClick) {
        unsafe {
            let count = *self.counts.get_unchecked(shard) as usize;
            *self.rows.get_unchecked_mut(shard).get_unchecked_mut(count) = click;
            if count == 1 {
                buckets
                    .get_unchecked_mut(shard)
                    .extend_from_slice(self.rows.get_unchecked(shard));
                *self.counts.get_unchecked_mut(shard) = 0;
            } else {
                *self.counts.get_unchecked_mut(shard) = 1;
            }
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

struct ClickWorkerBuckets {
    buckets: Vec<Vec<RawClick>>,
    wide_ts: FxHashMap<u64, i64>,
}

#[derive(Clone, Copy, Debug)]
struct WideEventMeta {
    ts: i64,
    category_id: u16,
}

#[derive(Clone, Copy, Debug)]
struct WideClickMeta(i64);

#[inline]
fn event_ingest_sec(worker: &EventWorkerBuckets, shard: usize, idx: usize) -> i64 {
    let ev = &worker.buckets[shard][idx];
    if ev.ingest_sec == i32::MIN {
        worker.wide_ingest[&(((shard as u64) << 32) | idx as u64)]
    } else {
        ev.ingest_sec as i64
    }
}

#[inline(always)]
fn event_timestamp(ev: &CleanEvent, window_start: i64, wide: &[WideEventMeta]) -> i64 {
    if ev.ts_off & META_TAG != 0 {
        wide[(ev.ts_off & !META_TAG) as usize].ts
    } else {
        window_start + ev.ts_off as i64
    }
}

#[inline(always)]
fn event_category(ev: &CleanEvent, wide: &[WideEventMeta]) -> usize {
    if ev.ts_off & META_TAG != 0 {
        wide[(ev.ts_off & !META_TAG) as usize].category_id as usize
    } else {
        ((ev.price_cat_flags >> 24) & 0x3f) as usize
    }
}

#[inline(always)]
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
            user_id: Vec::with_capacity(rows.saturating_mul(13)),
            day: Vec::with_capacity(rows.saturating_mul(4)),
            sessions: Vec::with_capacity(rows.saturating_mul(8)),
            events: Vec::with_capacity(rows.saturating_mul(8)),
            revenue: Vec::with_capacity(rows.saturating_mul(8)),
            category_def: Vec::with_capacity(rows),
            category_values: Vec::with_capacity(rows.saturating_mul(3)),
        }
    }

    #[inline(always)]
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

#[inline(always)]
fn append_plain_byte_array(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u32).to_le_bytes());
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
            finished_at: Mutex::new(if total == 0 {
                Some(std::time::Instant::now())
            } else {
                None
            }),
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

fn stage_timing(
    start: std::time::Instant,
    finish: std::time::Instant,
    overall: std::time::Instant,
) -> StageTiming {
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
    let empty_slot = DedupSlot {
        key: 0,
        ingest_sec: 0,
        loc: u32::MAX,
    };

    // Event dedup (L2-resident 512 shards, stack-cached batches, single-pass hash reuse)
    loop {
        let shard = event_shard_index.fetch_add(1, Ordering::Relaxed);
        if shard >= DEDUP_SHARDS {
            break;
        }
        let total: usize = parsed
            .workers
            .iter()
            .map(|producer| producer.events.buckets[shard].len())
            .sum();
        let cap = ((total * 3 / 2).next_power_of_two()).max(64);
        if table.len() < cap {
            table.resize(cap, empty_slot);
        }
        let mask = cap - 1;

        for (worker_index, producer) in parsed.workers.iter().enumerate() {
            let bucket = &producer.events.buckets[shard];
            let worker_hi = (worker_index as u32) << 27;
            for (group_index, group) in bucket.chunks(64).enumerate() {
                let base_idx = group_index * 64;
                let mut initial_positions = [0usize; 64];
                for (j, event) in group.iter().enumerate() {
                    let pos = dedup_hash(event.eid_key) as usize & mask;
                    unsafe {
                        *initial_positions.get_unchecked_mut(j) = pos;
                        #[cfg(target_arch = "x86_64")]
                        std::arch::x86_64::_mm_prefetch(
                            table.as_ptr().add(pos) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                for (j, event) in group.iter().enumerate() {
                    let index = base_idx + j;
                    let loc = worker_hi | index as u32;
                    let mut pos = unsafe { *initial_positions.get_unchecked(j) };
                    loop {
                        let slot = unsafe { table.get_unchecked_mut(pos) };
                        if slot.key == 0 {
                            *slot = DedupSlot {
                                key: event.eid_key,
                                ingest_sec: event.ingest_sec,
                                loc,
                            };
                            occupied.push(pos as u32);
                            break;
                        }
                        if slot.key == event.eid_key {
                            let earlier =
                                if event.ingest_sec != i32::MIN && slot.ingest_sec != i32::MIN {
                                    event.ingest_sec < slot.ingest_sec
                                } else {
                                    let current = slot.loc;
                                    let cw = (current >> 27) as usize;
                                    let ci = (current & ((1u32 << 27) - 1)) as usize;
                                    event_ingest_sec(&parsed.workers[cw].events, shard, ci)
                                        > event_ingest_sec(&producer.events, shard, index)
                                };
                            if earlier {
                                *slot = DedupSlot {
                                    key: event.eid_key,
                                    ingest_sec: event.ingest_sec,
                                    loc,
                                };
                            }
                            break;
                        }
                        pos = (pos + 1) & mask;
                    }
                }
            }
        }

        // Stack-cached 64-wide group prefetch over canon_map (zero second read of table/buckets!)
        for group in occupied.chunks(64) {
            let mut batch_ev = [RawEvent::default(); 64];
            let mut batch_loc = [0u32; 64];
            let mut hashes = [0u64; 64];
            let mut bloom_words = [0usize; 64];
            let mut valid_count = 0usize;

            for &position in group {
                let slot = unsafe { *table.get_unchecked(position as usize) };
                let p_idx = (slot.loc >> 27) as usize;
                let e_idx = (slot.loc & ((1u32 << 27) - 1)) as usize;
                let event = unsafe {
                    *parsed
                        .workers
                        .get_unchecked(p_idx)
                        .events
                        .buckets
                        .get_unchecked(shard)
                        .get_unchecked(e_idx)
                };
                if (event.price_cat_flags & (1 << 30)) != 0 {
                    let hash = mix_key(event.user_key);
                    let bw = CanonicalMap::bloom_index(hash);
                    unsafe {
                        *batch_ev.get_unchecked_mut(valid_count) = event;
                        *batch_loc.get_unchecked_mut(valid_count) = slot.loc;
                        *hashes.get_unchecked_mut(valid_count) = hash;
                        *bloom_words.get_unchecked_mut(valid_count) = bw;
                        #[cfg(target_arch = "x86_64")]
                        std::arch::x86_64::_mm_prefetch(
                            canon_map.bloom.as_ptr().add(bw) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                    valid_count += 1;
                }
            }

            let mut maybe_linked = [false; 64];
            for j in 0..valid_count {
                let hash = unsafe { *hashes.get_unchecked(j) };
                let bw = unsafe { *bloom_words.get_unchecked(j) };
                let hit = unsafe {
                    (*canon_map.bloom.get_unchecked(bw) & (1u64 << (hash as usize & 63))) != 0
                };
                unsafe { *maybe_linked.get_unchecked_mut(j) = hit };
                if hit {
                    let cslot = canon_map.initial_slot(hash);
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            canon_map.entries.as_ptr().add(cslot) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }

            for j in 0..valid_count {
                let event = unsafe { *batch_ev.get_unchecked(j) };
                let hash = unsafe { *hashes.get_unchecked(j) };
                let (person, person_hash) = if unsafe { *maybe_linked.get_unchecked(j) } {
                    let p = canon_map.canonical_hashed(event.user_key, hash);
                    let ph = if p == event.user_key {
                        hash
                    } else {
                        mix_key(p)
                    };
                    (p, ph)
                } else {
                    (event.user_key, hash)
                };
                let person_shard = fast_shard(person_hash, PERSON_SHARDS);
                let category_code = ((event.price_cat_flags >> 24) & 0x3f) as u16;
                let ts_off = if event.ts_off < META_TAG && category_code < 63 {
                    event.ts_off
                } else {
                    let loc = unsafe { *batch_loc.get_unchecked(j) };
                    let p_idx = (loc >> 27) as usize;
                    let e_idx = (loc & ((1u32 << 27) - 1)) as usize;
                    let producer = &parsed.workers[p_idx].events;
                    let loc_key = ((shard as u64) << 32) | e_idx as u64;
                    let ts = if event.ts_off == u32::MAX {
                        producer.wide_ts[&loc_key]
                    } else {
                        window_start + event.ts_off as i64
                    };
                    let cat_id = if category_code == 63 {
                        producer.wide_category[&loc_key]
                    } else {
                        category_code
                    };
                    let mut wide = event_meta.lock().expect("wide event metadata");
                    let index = wide.len();
                    assert!(index < META_TAG as usize, "too many wide events");
                    wide.push(WideEventMeta {
                        ts,
                        category_id: cat_id,
                    });
                    META_TAG | index as u32
                };
                event_staging.push(
                    &mut event_outputs,
                    person_shard,
                    CleanEvent {
                        person_key: person,
                        ts_off,
                        price_cat_flags: event.price_cat_flags,
                    },
                );
            }
        }
        for &position in &occupied {
            unsafe { *table.get_unchecked_mut(position as usize) = empty_slot };
        }
        occupied.clear();
        event_complete.complete_one();
    }
    event_staging.flush(&mut event_outputs);
    phase_barrier.wait();

    // Click dedup (stack-cached batches, single-pass hash reuse)
    loop {
        let shard = click_shard_index.fetch_add(1, Ordering::Relaxed);
        if shard >= DEDUP_SHARDS {
            break;
        }
        let total: usize = parsed
            .workers
            .iter()
            .map(|producer| producer.clicks.buckets[shard].len())
            .sum();
        let cap = ((total * 3 / 2).next_power_of_two()).max(64);
        if table.len() < cap {
            table.resize(cap, empty_slot);
        }
        let mask = cap - 1;

        for (worker_index, producer) in parsed.workers.iter().enumerate() {
            let bucket = &producer.clicks.buckets[shard];
            let worker_hi = (worker_index as u32) << 27;
            for (group_index, group) in bucket.chunks(64).enumerate() {
                let base_idx = group_index * 64;
                let mut initial_positions = [0usize; 64];
                for (j, click) in group.iter().enumerate() {
                    let pos = dedup_hash(click.click_key) as usize & mask;
                    unsafe {
                        *initial_positions.get_unchecked_mut(j) = pos;
                        #[cfg(target_arch = "x86_64")]
                        std::arch::x86_64::_mm_prefetch(
                            table.as_ptr().add(pos) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                for (j, click) in group.iter().enumerate() {
                    let index = base_idx + j;
                    let loc = worker_hi | index as u32;
                    let mut pos = unsafe { *initial_positions.get_unchecked(j) };
                    loop {
                        let slot = unsafe { table.get_unchecked_mut(pos) };
                        if slot.key == 0 {
                            *slot = DedupSlot {
                                key: click.click_key,
                                ingest_sec: 0,
                                loc,
                            };
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
            let mut batch_cl = [RawClick::default(); 64];
            let mut batch_loc = [0u32; 64];
            let mut hashes = [0u64; 64];
            let mut bloom_words = [0usize; 64];
            let count = group.len();

            for (j, &position) in group.iter().enumerate() {
                let slot = unsafe { *table.get_unchecked(position as usize) };
                let p_idx = (slot.loc >> 27) as usize;
                let c_idx = (slot.loc & ((1u32 << 27) - 1)) as usize;
                let click = unsafe {
                    *parsed
                        .workers
                        .get_unchecked(p_idx)
                        .clicks
                        .buckets
                        .get_unchecked(shard)
                        .get_unchecked(c_idx)
                };
                let hash = mix_key(click.user_key);
                let bw = CanonicalMap::bloom_index(hash);
                unsafe {
                    *batch_cl.get_unchecked_mut(j) = click;
                    *batch_loc.get_unchecked_mut(j) = slot.loc;
                    *hashes.get_unchecked_mut(j) = hash;
                    *bloom_words.get_unchecked_mut(j) = bw;
                    #[cfg(target_arch = "x86_64")]
                    std::arch::x86_64::_mm_prefetch(
                        canon_map.bloom.as_ptr().add(bw) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            let mut maybe_linked = [false; 64];
            for j in 0..count {
                let hash = unsafe { *hashes.get_unchecked(j) };
                let bw = unsafe { *bloom_words.get_unchecked(j) };
                let hit = unsafe {
                    (*canon_map.bloom.get_unchecked(bw) & (1u64 << (hash as usize & 63))) != 0
                };
                unsafe { *maybe_linked.get_unchecked_mut(j) = hit };
                if hit {
                    let cslot = canon_map.initial_slot(hash);
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(
                            canon_map.entries.as_ptr().add(cslot) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }

            for j in 0..count {
                let click = unsafe { *batch_cl.get_unchecked(j) };
                let hash = unsafe { *hashes.get_unchecked(j) };
                let (person, person_hash) = if unsafe { *maybe_linked.get_unchecked(j) } {
                    let p = canon_map.canonical_hashed(click.user_key, hash);
                    let ph = if p == click.user_key {
                        hash
                    } else {
                        mix_key(p)
                    };
                    (p, ph)
                } else {
                    (click.user_key, hash)
                };
                let person_shard = fast_shard(person_hash, PERSON_SHARDS);
                let ts_off = if click.ts_off < META_TAG {
                    click.ts_off
                } else {
                    let loc = unsafe { *batch_loc.get_unchecked(j) };
                    let p_idx = (loc >> 27) as usize;
                    let c_idx = (loc & ((1u32 << 27) - 1)) as usize;
                    let producer = &parsed.workers[p_idx].clicks;
                    let loc_key = ((shard as u64) << 32) | c_idx as u64;
                    let ts = if click.ts_off == u32::MAX {
                        producer.wide_ts[&loc_key]
                    } else {
                        click_start + click.ts_off as i64
                    };
                    let mut wide = click_meta.lock().expect("wide click metadata");
                    let index = wide.len();
                    assert!(index < META_TAG as usize, "too many wide clicks");
                    wide.push(WideClickMeta(ts));
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
            unsafe { *table.get_unchecked_mut(position as usize) = empty_slot };
        }
        occupied.clear();
        click_complete.complete_one();
    }
    click_staging.flush(&mut click_outputs);

    DedupWorkerData {
        events: event_outputs,
        clicks: click_outputs,
    }
}

fn sessionize_write_worker(
    worker_id: usize,
    dedup: &DedupData,
    worker_count: usize,
    ids: &IdPool,
    next_person_shard: &AtomicUsize,
    window_start: i64,
    window_end: i64,
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
    let fast_u32_session = dedup.wide_event_meta.is_empty()
        && dedup.wide_click_meta.is_empty()
        && window_start.rem_euclid(MS_PER_DAY) == 0
        && (window_end - click_start) < (META_TAG as i64);

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

            if fast_u32_session {
                // Pure u32 fast path: 3-cycle unsigned day division, zero META_TAG checks
                while i < event_end {
                    let sess_start = unsafe { events.get_unchecked(i).ts_off };
                    let day_index = (sess_start / MS_PER_DAY_U32) as usize;
                    let d_acc = unsafe { day_acc.get_unchecked_mut(day_index) };
                    if d_acc.0 == 0 {
                        touched_days.push(day_index);
                    }
                    d_acc.0 += 1;
                    let mut session_events = 0i64;
                    let mut session_revenue = 0i64;
                    let mut has_purchase = false;
                    let mut prev_ts = sess_start;
                    let cat_base = day_index * num_cats;

                    while i < event_end {
                        let ev = unsafe { *events.get_unchecked(i) };
                        if ev.ts_off - prev_ts > MS_30_MIN_U32 {
                            break;
                        }
                        prev_ts = ev.ts_off;
                        session_events += 1;
                        if (ev.price_cat_flags & (1 << 31)) != 0 {
                            let price = (ev.price_cat_flags & 0x00ff_ffff) as i64;
                            let category = ((ev.price_cat_flags >> 24) & 0x3f) as usize;
                            session_revenue += price;
                            has_purchase = true;
                            unsafe { *day_cat_rev.get_unchecked_mut(cat_base + category) += price };
                            d_acc.3 = true;
                        }
                        i += 1;
                    }
                    d_acc.1 += session_events;
                    d_acc.2 += session_revenue;

                    let sess_click_hi = sess_start + MS_7_DAYS_U32;
                    let pcl_len = person_clicks.len();
                    while click_right < pcl_len
                        && unsafe { person_clicks.get_unchecked(click_right).ts_off }
                            <= sess_click_hi
                    {
                        click_right += 1;
                    }
                    while click_left < click_right
                        && unsafe { person_clicks.get_unchecked(click_left).ts_off } < sess_start
                    {
                        click_left += 1;
                    }
                    if click_left < click_right {
                        let best_ts =
                            unsafe { person_clicks.get_unchecked(click_right - 1).ts_off };
                        let mut best_index = click_right - 1;
                        while best_index > click_left
                            && unsafe { person_clicks.get_unchecked(best_index - 1).ts_off }
                                == best_ts
                        {
                            best_index -= 1;
                        }
                        let credited_campaign =
                            unsafe { person_clicks.get_unchecked(best_index).campaign_idx } as usize;
                        let credited = unsafe {
                            camp_agg.get_unchecked_mut(credited_campaign * num_days + day_index)
                        };
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
                        for click in unsafe { person_clicks.get_unchecked(click_left..click_right) } {
                            let campaign = click.campaign_idx as usize;
                            unsafe {
                                if *assist_stamp.get_unchecked(campaign) != cur_stamp {
                                    *assist_stamp.get_unchecked_mut(campaign) = cur_stamp;
                                    camp_agg
                                        .get_unchecked_mut(campaign * num_days + day_index)
                                        .assisted_sessions += 1;
                                }
                            }
                        }
                    } else {
                        let organic = unsafe { camp_agg.get_unchecked_mut(day_index) };
                        organic.sessions += 1;
                        if has_purchase {
                            organic.purchasing_sessions += 1;
                            organic.revenue_cents += session_revenue;
                        }
                    }
                }
            } else {
                // General i64 fallback for arbitrary fixture windows
                while i < event_end {
                    let session_start =
                        event_timestamp(&events[i], window_start, &dedup.wide_event_meta);
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
                        && event_timestamp(&events[i], window_start, &dedup.wide_event_meta)
                            - previous_ts
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
                        && click_timestamp(
                            &person_clicks[click_right],
                            click_start,
                            &dedup.wide_click_meta,
                        ) <= session_start
                    {
                        click_right += 1;
                    }
                    while click_left < click_right
                        && click_timestamp(
                            &person_clicks[click_left],
                            click_start,
                            &dedup.wide_click_meta,
                        ) < window_lo
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
                        let credited =
                            &mut camp_agg[credited_campaign as usize * num_days + day_index];
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
            }

            let mut user_id_buffer = [0u8; 9];
            let user_id: &[u8] = if person >> 63 != 0 {
                dedup.fallback_ids[(person & !(1u64 << 63)) as usize].as_ref()
            } else {
                unpack_fast_id(person, b'u', b'U', &mut user_id_buffer)
            };

            for &day_index in &touched_days {
                let day = start_day + day_index as i32;
                let acc = day_acc[day_index];
                let top_category = if acc.3 {
                    let mut best_category = 0usize;
                    let mut best_revenue = -1i64;
                    let base = day_index * num_cats;
                    for category in 0..num_cats {
                        let revenue = day_cat_rev[base + category];
                        if revenue > 0
                            && (revenue > best_revenue
                                || (revenue == best_revenue
                                    && cat_names[category] < cat_names[best_category]))
                        {
                            best_revenue = revenue;
                            best_category = category;
                        }
                    }
                    day_cat_rev[base..base + num_cats].fill(0);
                    Some(cat_names[best_category].as_bytes())
                } else {
                    None
                };
                user_rows.push(user_id, day, acc.0, acc.1, acc.2, top_category);
                day_acc[day_index] = (0, 0, 0, false);
            }
            touched_days.clear();
            event_index = event_end;
            click_index = click_end;
        }
    }

    let columns = user_rows.columns();
    parquet_writer::write_table(
        &out_dir
            .join("user_daily")
            .join(format!("part-{worker_id}.parquet")),
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
    let link_paths = list_csvs(&config.data_dir.join("links"));
    let event_paths = list_csvs(&config.data_dir.join("events"));
    let click_paths = list_csvs(&config.data_dir.join("clicks"));

    // Parallel mmap + morsel splitting across all worker threads
    let link_file_idx = AtomicUsize::new(0);
    let event_file_idx = AtomicUsize::new(0);
    let click_file_idx = AtomicUsize::new(0);

    let link_morsels_shared: Mutex<Vec<&'static [u8]>> = Mutex::new(Vec::new());
    let event_morsels_shared: Mutex<Vec<&'static [u8]>> = Mutex::new(Vec::new());
    let click_morsels_shared: Mutex<Vec<&'static [u8]>> = Mutex::new(Vec::new());

    let link_morsels_ready: Mutex<Option<Arc<Vec<&'static [u8]>>>> = Mutex::new(None);
    let event_morsels_ready: Mutex<Option<Arc<Vec<&'static [u8]>>>> = Mutex::new(None);
    let click_morsels_ready: Mutex<Option<Arc<Vec<&'static [u8]>>>> = Mutex::new(None);
    let parse_caps: Mutex<(usize, usize)> = Mutex::new((64, 64));
    let event_parse_complete_shared: Mutex<Option<Arc<CompletionCounter>>> = Mutex::new(None);
    let click_parse_complete_shared: Mutex<Option<Arc<CompletionCounter>>> = Mutex::new(None);

    let link_index = AtomicUsize::new(0);
    let event_index = AtomicUsize::new(0);
    let click_index = AtomicUsize::new(0);
    let event_dedup_index = AtomicUsize::new(0);
    let click_dedup_index = AtomicUsize::new(0);
    let person_index = AtomicUsize::new(0);
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

    let barrier0 = Barrier::new(worker_count);
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
    let window_start_sec = config.window_start.div_euclid(1_000);

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for worker_id in 0..worker_count {
            let link_paths = &link_paths;
            let event_paths = &event_paths;
            let click_paths = &click_paths;
            let link_file_idx = &link_file_idx;
            let event_file_idx = &event_file_idx;
            let click_file_idx = &click_file_idx;
            let link_morsels_shared = &link_morsels_shared;
            let event_morsels_shared = &event_morsels_shared;
            let click_morsels_shared = &click_morsels_shared;
            let link_morsels_ready = &link_morsels_ready;
            let event_morsels_ready = &event_morsels_ready;
            let click_morsels_ready = &click_morsels_ready;
            let parse_caps = &parse_caps;
            let event_parse_complete_shared = &event_parse_complete_shared;
            let click_parse_complete_shared = &click_parse_complete_shared;
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
            let event_dedup_complete = &event_dedup_complete;
            let click_dedup_complete = &click_dedup_complete;
            let event_index = &event_index;
            let click_index = &click_index;
            let link_index = &link_index;
            let event_dedup_index = &event_dedup_index;
            let click_dedup_index = &click_dedup_index;
            let person_index = &person_index;
            let barrier0 = &barrier0;
            let barrier1 = &barrier1;
            let barrier2 = &barrier2;
            let barrier3 = &barrier3;
            let event_click_barrier = &event_click_barrier;
            let session_barrier = &session_barrier;

            handles.push(scope.spawn(move || {
                // Phase 0: Parallel mmap & morsel discovery across all 8 threads
                let mut local_lm = Vec::new();
                loop {
                    let i = link_file_idx.fetch_add(1, Ordering::Relaxed);
                    let Some(p) = link_paths.get(i) else { break };
                    if let Some(d) = mmap_data(p) {
                        mmap_morsels(d, &mut local_lm);
                    }
                }
                if !local_lm.is_empty() {
                    link_morsels_shared.lock().unwrap().extend(local_lm);
                }

                let mut local_em = Vec::new();
                loop {
                    let i = event_file_idx.fetch_add(1, Ordering::Relaxed);
                    let Some(p) = event_paths.get(i) else { break };
                    if let Some(d) = mmap_data(p) {
                        mmap_morsels(d, &mut local_em);
                    }
                }
                if !local_em.is_empty() {
                    event_morsels_shared.lock().unwrap().extend(local_em);
                }

                let mut local_cm = Vec::new();
                loop {
                    let i = click_file_idx.fetch_add(1, Ordering::Relaxed);
                    let Some(p) = click_paths.get(i) else { break };
                    if let Some(d) = mmap_data(p) {
                        mmap_morsels(d, &mut local_cm);
                    }
                }
                if !local_cm.is_empty() {
                    click_morsels_shared.lock().unwrap().extend(local_cm);
                }
                barrier0.wait();

                if worker_id == 0 {
                    let lm = std::mem::take(&mut *link_morsels_shared.lock().unwrap());
                    let em = std::mem::take(&mut *event_morsels_shared.lock().unwrap());
                    let cm = std::mem::take(&mut *click_morsels_shared.lock().unwrap());
                    let eb: usize = em.iter().map(|m| m.len()).sum();
                    let cb: usize = cm.iter().map(|m| m.len()).sum();
                    *parse_caps.lock().unwrap() = (
                        eb / worker_count / DEDUP_SHARDS / 70 + 16,
                        cb / worker_count / DEDUP_SHARDS / 70 + 16,
                    );
                    *event_parse_complete_shared.lock().unwrap() =
                        Some(Arc::new(CompletionCounter::new(em.len())));
                    *click_parse_complete_shared.lock().unwrap() =
                        Some(Arc::new(CompletionCounter::new(cm.len())));
                    *link_morsels_ready.lock().unwrap() = Some(Arc::new(lm));
                    *event_morsels_ready.lock().unwrap() = Some(Arc::new(em));
                    *click_morsels_ready.lock().unwrap() = Some(Arc::new(cm));
                }
                barrier0.wait();

                let link_morsels = link_morsels_ready.lock().unwrap().as_ref().unwrap().clone();
                let event_morsels = event_morsels_ready.lock().unwrap().as_ref().unwrap().clone();
                let click_morsels = click_morsels_ready.lock().unwrap().as_ref().unwrap().clone();
                let event_parse_complete = event_parse_complete_shared
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .clone();
                let click_parse_complete = click_parse_complete_shared
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .clone();
                let (event_bucket_capacity, click_bucket_capacity) = *parse_caps.lock().unwrap();

                // Phase 1a: Parallel link parsing
                let mut local_link_pairs = Vec::new();
                loop {
                    let chunk_index = link_index.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = link_morsels.get(chunk_index) else {
                        break;
                    };
                    scan_csv_morsel::<2, _>(chunk, |[a, b]| {
                        if !a.is_empty() && !b.is_empty() {
                            local_link_pairs.push((ids.key(a, b'u', b'U'), ids.key(b, b'u', b'U')));
                        }
                    });
                }
                link_pairs.lock().expect("link result slots")[worker_id] = Some(local_link_pairs);
                barrier1.wait();

                // Thread 0 builds CanonicalMap while Threads 1..N immediately start parsing events
                if worker_id == 0 {
                    let pairs = {
                        let mut slots = link_pairs.lock().expect("link result slots");
                        slots
                            .iter_mut()
                            .map(|slot| slot.take().expect("link worker result"))
                            .collect()
                    };
                    let canonical = Arc::new(build_canon_map(pairs, ids));
                    *canonical_shared.lock().expect("canonical map") = Some(canonical);
                    let finish = std::time::Instant::now();
                    *link_timing.lock().expect("links timing") =
                        Some(stage_timing(link_start, finish, overall_start));
                }

                // Phase 1b: Drain all events first (keeps L1 EventStaging hot), then drain clicks
                let mut event_buckets = (0..DEDUP_SHARDS)
                    .map(|_| Vec::with_capacity(event_bucket_capacity))
                    .collect::<Vec<Vec<RawEvent>>>();
                let mut event_staging = EventStaging::new();
                let mut wide_ingest = FxHashMap::default();
                let mut wide_ts = FxHashMap::default();
                let mut wide_category = FxHashMap::default();

                loop {
                    let index = event_index.fetch_add(1, Ordering::Relaxed);
                    let Some(morsel) = event_morsels.get(index) else {
                        break;
                    };
                    scan_csv_morsel::<7, _>(morsel, |fields| {
                        parse_event_fields(
                            fields,
                            window_start,
                            window_start_sec,
                            window_end,
                            prod_fast,
                            prod_fallback,
                            ids,
                            &date_cache,
                            &mut event_buckets,
                            &mut event_staging,
                            &mut wide_ingest,
                            &mut wide_ts,
                            &mut wide_category,
                        );
                    });
                    event_parse_complete.complete_one();
                }
                event_staging.flush(&mut event_buckets);

                let mut click_buckets = (0..DEDUP_SHARDS)
                    .map(|_| Vec::with_capacity(click_bucket_capacity))
                    .collect::<Vec<Vec<RawClick>>>();
                let mut click_staging = RawClickStaging::new();
                let mut click_wide_ts = FxHashMap::default();

                loop {
                    let index = click_index.fetch_add(1, Ordering::Relaxed);
                    let Some(morsel) = click_morsels.get(index) else {
                        break;
                    };
                    scan_csv_morsel::<4, _>(morsel, |fields| {
                        parse_click_fields(
                            fields,
                            click_start,
                            window_end,
                            camp_fast,
                            camp_fallback,
                            ids,
                            &date_cache,
                            &mut click_buckets,
                            &mut click_staging,
                            &mut click_wide_ts,
                        );
                    });
                    click_parse_complete.complete_one();
                }
                click_staging.flush(&mut click_buckets);

                parsed_workers.lock().expect("parse result slots")[worker_id] =
                    Some(ParseWorkerData {
                        events: EventWorkerBuckets {
                            buckets: event_buckets,
                            wide_ingest,
                            wide_ts,
                            wide_category,
                        },
                        clicks: ClickWorkerBuckets {
                            buckets: click_buckets,
                            wide_ts: click_wide_ts,
                        },
                    });
                barrier2.wait();

                if worker_id == 0 {
                    let producers = {
                        let mut slots = parsed_workers.lock().expect("parse result slots");
                        slots
                            .iter_mut()
                            .map(|slot| slot.take().expect("parse worker result"))
                            .collect::<Vec<_>>()
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
                    *event_dedup_start.lock().expect("event dedup start") =
                        Some(std::time::Instant::now());
                }
                barrier2.wait();

                // Phase 2: Parallel dedup + canonical resolution
                let parsed = parsed_shared.lock().unwrap().as_ref().unwrap().clone();
                let canonical = canonical_shared.lock().unwrap().as_ref().unwrap().clone();
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
                        slots
                            .iter_mut()
                            .map(|slot| slot.take().expect("dedup worker result"))
                            .collect()
                    };
                    let event_meta =
                        std::mem::take(&mut *wide_event_meta.lock().expect("wide event metadata"));
                    let click_meta =
                        std::mem::take(&mut *wide_click_meta.lock().expect("wide click metadata"));
                    *dedup_shared.lock().expect("dedup data") = Some(Arc::new(DedupData {
                        workers: producers,
                        wide_event_meta: event_meta,
                        wide_click_meta: click_meta,
                        fallback_ids: parsed.fallback_ids.clone(),
                        total_event_rows: parsed.total_event_rows,
                        total_click_rows: parsed.total_click_rows,
                    }));
                    *sessionize_start.lock().expect("sessionize start") =
                        Some(std::time::Instant::now());
                }
                barrier3.wait();

                // Phase 3: Parallel sessionize + direct Parquet write
                let dedup = dedup_shared.lock().unwrap().as_ref().unwrap().clone();
                let camp_agg = sessionize_write_worker(
                    worker_id,
                    &dedup,
                    worker_count,
                    ids,
                    person_index,
                    window_start,
                    window_end,
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
                    let mut final_campaigns: FxHashMap<(u16, i32), CampaignDayAgg> =
                        FxHashMap::default();
                    let mut slots = campaign_results.lock().expect("campaign result slots");
                    for campaign_agg in slots.iter_mut() {
                        let values = campaign_agg.take().expect("campaign worker result");
                        for (index, value) in values.into_iter().enumerate() {
                            if value.sessions != 0 || value.assisted_sessions != 0 {
                                let campaign = index / num_days;
                                let day = start_day + (index % num_days) as i32;
                                let entry =
                                    final_campaigns.entry((campaign as u16, day)).or_default();
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
                    *sessionize_finish.lock().expect("sessionize finish") =
                        Some(std::time::Instant::now());
                }
            }));
        }
        for handle in handles {
            handle.join().expect("pipeline worker");
        }
    });

    let mut timings = PipelineTimings::default();
    timings.links = link_timing.lock().unwrap().unwrap_or_default();
    let event_parse_finish = event_parse_complete_shared
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .finished_at();
    timings.events_parse = stage_timing(parse_start, event_parse_finish, config.overall_start);
    let event_dedup_finish = event_dedup_complete.finished_at();
    let event_dedup_begin = event_dedup_start
        .lock()
        .unwrap()
        .unwrap_or(event_dedup_finish);
    timings.events_dedup =
        stage_timing(event_dedup_begin, event_dedup_finish, config.overall_start);
    let click_parse_finish = click_parse_complete_shared
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .finished_at();
    let click_dedup_finish = click_dedup_complete.finished_at();
    let click_dedup_elapsed = click_dedup_finish.saturating_duration_since(event_dedup_finish);
    timings.clicks.elapsed =
        click_parse_finish.saturating_duration_since(event_parse_finish) + click_dedup_elapsed;
    timings.clicks.cumulative = click_dedup_finish.saturating_duration_since(config.overall_start);
    let sessionize_begin = sessionize_start
        .lock()
        .unwrap()
        .unwrap_or(click_dedup_finish);
    let sessionize_end = sessionize_finish
        .lock()
        .unwrap()
        .unwrap_or_else(std::time::Instant::now);
    timings.sessionize_write =
        stage_timing(sessionize_begin, sessionize_end, config.overall_start);
    timings
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
    let ws_str = extract_json_string(&manifest_str, "window_start").expect("window_start");
    let we_str = extract_json_string(&manifest_str, "window_end").expect("window_end");
    let window_start = parse_ts(ws_str.as_bytes()).expect("parse window_start");
    let window_end = parse_ts(we_str.as_bytes()).expect("parse window_end");
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
    eprintln!(
        "links: {:?} cumulative: {:?}",
        timings.links.elapsed, timings.links.cumulative
    );
    eprintln!(
        "events_parse: {:?} cumulative: {:?}",
        timings.events_parse.elapsed, timings.events_parse.cumulative
    );
    eprintln!(
        "events_dedup: {:?} cumulative: {:?}",
        timings.events_dedup.elapsed, timings.events_dedup.cumulative
    );
    eprintln!(
        "clicks: {:?} cumulative: {:?}",
        timings.clicks.elapsed, timings.clicks.cumulative
    );
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
    parquet_writer::write_table(&dir.join("part-0.parquet"), n, &columns)
        .expect("write campaign_daily parquet");
}