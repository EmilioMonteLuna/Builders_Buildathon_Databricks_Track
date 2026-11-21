use arrow::array::{Date32Builder, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use memchr::{memchr, memchr_iter};
use memmap2::Mmap;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::hash::Hasher;
use mimalloc::MiMalloc;
use std::sync::Mutex;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const MS_PER_DAY: i64 = 86_400_000;
const MS_30_MIN: i64 = 1_800_000;
const MS_7_DAYS: i64 = 604_800_000;
const NUM_SHARDS: usize = 64;
const DEDUP_SHARDS: usize = 256;

#[derive(Clone, Copy)]
struct DedupSlot {
    key: u64,
    loc: u32,
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

    fn get(&self, b: &[u8]) -> Option<i64> {
        if b.len() < 10 { return None; }
        let d = parse_2d(&b[8..10])?;
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
fn fx_hash_key(key: u64) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    h.write_u64(key);
    h.finish()
}

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
    for (i, &c) in b[1..].iter().enumerate() {
        let value = match c {
            b'0'..=b'9' => c - b'0' + 1,
            b'a'..=b'z' if lower => c - b'a' + 11,
            b'A'..=b'Z' if !lower => c - b'A' + 11,
            _ => return None,
        };
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
    fallback: Mutex<(Vec<&'static [u8]>, FxHashMap<&'static [u8], u64>)>,
}

impl IdPool {
    fn new() -> Self {
        Self { fallback: Mutex::new((Vec::new(), FxHashMap::default())) }
    }

    fn key(&self, bytes: &'static [u8], lo: u8, hi: u8) -> u64 {
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
        pool.0.push(bytes);
        pool.1.insert(bytes, key);
        key
    }

    fn fallback_snapshot(&self) -> Vec<&'static [u8]> {
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
        let a_bytes = if a >> 63 != 0 { pool.0[(a & !(1u64 << 63)) as usize] } else { unpack_fast_id(a, lo, hi, &mut ab) };
        let b_bytes = if b >> 63 != 0 { pool.0[(b & !(1u64 << 63)) as usize] } else { unpack_fast_id(b, lo, hi, &mut bb) };
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
    if let Some(day_ms) = cache.get(b) {
        return Some(day_ms + (hh as i64 * 3600 + mm as i64 * 60 + ss as i64) * 1000);
    }
    let _ = d;
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
fn parse_event_line(
    line: &'static [u8],
    window_start: i64,
    window_end: i64,
    prod_fast: &[u16],
    prod_fallback: &FxHashMap<Box<[u8]>, u16>,
    ids: &IdPool,
    date_cache: &mut DateCache,
    buckets: &mut [Vec<RawEvent>],
) {
    let Some([eid, uid, ts_b, ing_b, etype, pid, price_b]) = parse_csv_line_fast::<7>(line) else {
        return;
    };
    if eid.is_empty() {
        return;
    }
    let Some(ingest_ts) = parse_f1_ingest(ing_b, date_cache) else {
        return;
    };
    let mut flags = 0u8;
    let mut ts = 0i64;
    let mut price_cents = 0u32;
    let mut category_id = 0u16;

    if !uid.is_empty() && !pid.is_empty() {
        if let Some(parsed_ts) = parse_ts_cached(ts_b, date_cache) {
            if parsed_ts >= window_start
                && parsed_ts < window_end
                && ingest_ts - parsed_ts <= MS_PER_DAY
            {
                match etype {
                    b"view" | b"cart" => {
                        flags = 1;
                        ts = parsed_ts;
                    }
                    b"purchase" => {
                        if let Some(p) = parse_price(price_b) {
                            flags = 3;
                            ts = parsed_ts;
                            price_cents = p;
                            category_id = parse_numeric_id(pid, b"p", 65_535)
                                .and_then(|idx| prod_fast.get(idx).copied())
                                .unwrap_or_else(|| *prod_fallback.get(pid).unwrap_or(&0));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    let eid_key = ids.key(eid, b'e', b'E');
    let user_key = if flags & 1 != 0 { ids.key(uid, b'u', b'U') } else { 0 };
    let shard = fast_shard(eid_key, DEDUP_SHARDS);
    buckets[shard].push(RawEvent {
        eid_key,
        ingest_ts,
        ts,
        user_key,
        price_cents,
        category_id,
        flags,
    });
}

#[inline]
fn parse_click_line(
    line: &'static [u8],
    click_start: i64,
    window_end: i64,
    camp_fast: &[u16],
    camp_fallback: &FxHashMap<Box<[u8]>, u16>,
    ids: &IdPool,
    date_cache: &mut DateCache,
    buckets: &mut [Vec<RawClick>],
) {
    let Some([cid, uid, camp_b, ts_b]) = parse_csv_line_fast::<4>(line) else {
        return;
    };
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
    buckets[shard].push(RawClick {
        click_key,
        ts,
        user_key,
        campaign_idx: camp_idx,
    });
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
    Some(&mmap[..])
}

fn morsels(data: &'static [u8]) -> Vec<(&'static [u8], bool)> {
    const MORSEL: usize = 4 * 1024 * 1024;
    let mut out = Vec::new();
    let mut pos = match memchr(b'\n', data) {
        Some(i) => i + 1,
        None => return out,
    };
    let mut first = true;
    while pos < data.len() {
        let target = pos.saturating_add(MORSEL);
        let end = if target >= data.len() {
            data.len()
        } else if let Some(rel) = memchr(b'\n', &data[target..]) {
            target + rel + 1
        } else {
            data.len()
        };
        out.push((&data[pos..end], first));
        first = false;
        pos = end;
    }
    out
}

// Union-Find tracking byte-wise smallest user_id per component
struct Dsu {
    parent: Vec<u32>,
    names: Vec<u64>,
    id_to_idx: FxHashMap<u64, u32>,
}

impl Dsu {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            names: Vec::new(),
            id_to_idx: FxHashMap::default(),
        }
    }

    fn get_or_insert(&mut self, id: u64) -> u32 {
        if let Some(&idx) = self.id_to_idx.get(&id) {
            return idx;
        }
        let idx = self.parent.len() as u32;
        self.parent.push(idx);
        self.names.push(id);
        self.id_to_idx.insert(id, idx);
        idx
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

    fn finalize(mut self) -> FxHashMap<u64, u64> {
        let n = self.parent.len();
        let mut canon = FxHashMap::with_capacity_and_hasher(n, Default::default());
        for i in 0..n {
            let r = self.find(i as u32) as usize;
            if i != r {
                canon.insert(self.names[i], self.names[r]);
            }
        }
        canon
    }
}

fn build_canon_map(links_dir: &Path, ids: &IdPool) -> FxHashMap<u64, u64> {
    let pairs_by_file: Vec<Vec<(u64, u64)>> = list_csvs(links_dir)
        .par_iter()
        .map(|path| {
            let mut pairs = Vec::new();
            if let Some(data) = mmap_data(path) {
                let mut pos = match memchr(b'\n', data) {
                    Some(i) => i + 1,
                    None => return pairs,
                };
                while pos < data.len() {
                    let rel = memchr(b'\n', &data[pos..]).unwrap_or(data.len() - pos);
                    let line = &data[pos..pos + rel];
                    pos += rel + 1;
                    if let Some([a, b]) = parse_csv_line::<2>(line) {
                        if !a.is_empty() && !b.is_empty() {
                            pairs.push((ids.key(a, b'u', b'U'), ids.key(b, b'u', b'U')));
                        }
                    }
                }
            }
            pairs
        })
        .collect();
    let mut dsu = Dsu::new();
    for pairs in pairs_by_file {
        for (ka, kb) in pairs {
            let ia = dsu.get_or_insert(ka);
            let ib = dsu.get_or_insert(kb);
            dsu.union(ia, ib, ids);
        }
    }
    dsu.finalize()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawEvent {
    eid_key: u64,
    ingest_ts: i64,
    ts: i64,
    user_key: u64,
    price_cents: u32,
    category_id: u16,
    flags: u8,
}

#[derive(Clone, Copy)]
struct CleanEvent {
    ts: i64,
    price_cents: u32,
    category_id: u16,
    flags: u8,
}

#[derive(Clone, Copy)]
struct CleanClick {
    ts: i64,
    campaign_idx: u16,
    click_key: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawClick {
    click_key: u64,
    ts: i64,
    user_key: u64,
    campaign_idx: u16,
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
    let links_dir = data_dir.join("links");
    let links_ids = Arc::clone(&ids);
    let links_handle = std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let canon = build_canon_map(&links_dir, &links_ids);
        (canon, started.elapsed())
    });

    // 4. Events: parallel parse & local shard, then parallel dedup
    let event_morsels: Vec<&'static [u8]> = list_csvs(&data_dir.join("events"))
        .iter()
        .filter_map(|path| mmap_data(path))
        .flat_map(|data| morsels(data).into_iter().map(|(m, _)| m))
        .collect();
    let worker_count = env::var("NCPU")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(rayon::current_num_threads);
    let estimated_bytes: usize = event_morsels.iter().map(|m| m.len()).sum();
    let per_worker_bucket = estimated_bytes / worker_count / DEDUP_SHARDS / 70;
    let event_index = AtomicUsize::new(0);
    let event_parse_start = std::time::Instant::now();
    let worker_event_buckets = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(scope.spawn(|| {
                let mut buckets = (0..DEDUP_SHARDS)
                    .map(|_| Vec::with_capacity(per_worker_bucket))
                    .collect::<Vec<Vec<RawEvent>>>();
                let mut date_cache = date_cache;
                loop {
                    let chunk = event_index.fetch_add(1, Ordering::Relaxed);
                    if chunk >= event_morsels.len() {
                        break;
                    }
                    let morsel = event_morsels[chunk];
                    let mut start = 0usize;
                    for end in memchr_iter(b'\n', morsel) {
                        if end > start {
                            let line = &morsel[start..end];
                            parse_event_line(line, window_start, window_end, &prod_fast, &prod_fallback, &ids, &mut date_cache, &mut buckets);
                        }
                        start = end + 1;
                    }
                    if start < morsel.len() {
                        parse_event_line(&morsel[start..], window_start, window_end, &prod_fast, &prod_fallback, &ids, &mut date_cache, &mut buckets);
                    }
                }
                buckets
            }));
        }
        handles.into_iter().map(|h| h.join().expect("event worker")).collect::<Vec<_>>()
    });

    let event_parse_elapsed = event_parse_start.elapsed();
    let (canon_map, links_elapsed) = links_handle.join().expect("links worker");
    let mut canon_bloom = vec![0u64; 1 << 18];
    for key in canon_map.keys() {
        let h = fx_hash_key(*key);
        let bit = h as usize & ((1usize << 24) - 1);
        canon_bloom[bit >> 6] |= 1u64 << (bit & 63);
    }
    eprintln!("links: {:?} cumulative: {:?}", links_elapsed, t_start.elapsed());
    eprintln!("events_parse: {:?} cumulative: {:?}", event_parse_elapsed, t_start.elapsed());
    t = std::time::Instant::now();

    // Dedup events per 256-shard, then repartition by canonical person.
    let shard_index = AtomicUsize::new(0);
    let person_event_shards: Vec<Vec<Vec<(u64, CleanEvent)>>> =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for _ in 0..worker_count {
                handles.push(scope.spawn(|| {
                    let mut table = Vec::<DedupSlot>::new();
                    let mut occupied = Vec::<u32>::new();
                    let mut outputs = (0..NUM_SHARDS)
                        .map(|_| Vec::new())
                        .collect::<Vec<Vec<(u64, CleanEvent)>>>();
                    loop {
                        let s = shard_index.fetch_add(1, Ordering::Relaxed);
                        if s >= DEDUP_SHARDS {
                            break;
                        }
                        let total: usize = worker_event_buckets.iter().map(|w| w[s].len()).sum();
                        let cap = ((total * 3 / 2).next_power_of_two()).max(64);
                        if table.len() < cap {
                            table.resize(cap, DedupSlot { key: 0, loc: u32::MAX });
                        }
                        for pos in occupied.iter().copied() {
                            table[pos as usize].loc = u32::MAX;
                        }
                        occupied.clear();
                        for (worker, bucket) in worker_event_buckets.iter().enumerate() {
                            for (idx, ev) in bucket[s].iter().enumerate() {
                                let key = ev.eid_key;
                                let loc = ((worker as u32) << 27) | idx as u32;
                                let mut pos = mix_key(key) as usize & (cap - 1);
                                loop {
                                    if table[pos].key == 0 {
                                        table[pos] = DedupSlot { key, loc };
                                        occupied.push(pos as u32);
                                        break;
                                    }
                                    if table[pos].key == key {
                                        let cur_loc = table[pos].loc;
                                        let cur_w = (cur_loc >> 27) as usize;
                                        let cur_i = (cur_loc & ((1u32 << 27) - 1)) as usize;
                                        let cur = &worker_event_buckets[cur_w][s][cur_i];
                                        if ev.ingest_ts < cur.ingest_ts {
                                            table[pos] = DedupSlot { key, loc };
                                        }
                                        break;
                                    }
                                    pos = (pos + 1) & (cap - 1);
                                }
                            }
                        }
                        for pos in occupied.iter().copied() {
                            let slot = table[pos as usize];
                            let w = (slot.loc >> 27) as usize;
                            let i = (slot.loc & ((1u32 << 27) - 1)) as usize;
                            let ev = worker_event_buckets[w][s][i];
                            if ev.flags & 1 != 0 {
                                let h = fx_hash_key(ev.user_key);
                                let bit = h as usize & ((1usize << 24) - 1);
                                let person = if canon_bloom[bit >> 6] & (1u64 << (bit & 63)) == 0 {
                                    ev.user_key
                                } else {
                                    canon_map.get(&ev.user_key).copied().unwrap_or(ev.user_key)
                                };
                                let person_hash = if person == ev.user_key { h } else { fx_hash_key(person) };
                                let ps = fast_shard(person_hash, NUM_SHARDS);
                                outputs[ps].push((person, CleanEvent {
                                    ts: ev.ts,
                                    price_cents: ev.price_cents,
                                    category_id: ev.category_id,
                                    flags: ev.flags,
                                }));
                            }
                        }
                        for pos in occupied.iter().copied() {
                            table[pos as usize] = DedupSlot { key: 0, loc: u32::MAX };
                        }
                        occupied.clear();
                    }
                    outputs
                }));
            }
            handles.into_iter().map(|h| h.join().expect("event dedup worker")).collect()
        });

    eprintln!("events_dedup: {:?} cumulative: {:?}", t.elapsed(), t_start.elapsed());
    t = std::time::Instant::now();

    // 6. Clicks: parallel parse, dedup & repartition by canonical person
    let click_morsels: Vec<&'static [u8]> = list_csvs(&data_dir.join("clicks"))
        .iter()
        .filter_map(|path| mmap_data(path))
        .flat_map(|data| morsels(data).into_iter().map(|(m, _)| m))
        .collect();
    let estimated_click_bytes: usize = click_morsels.iter().map(|m| m.len()).sum();
    let per_worker_click_bucket = estimated_click_bytes / worker_count / DEDUP_SHARDS / 70;
    let click_index = AtomicUsize::new(0);
    let worker_click_buckets = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            handles.push(scope.spawn(|| {
                let mut buckets = (0..DEDUP_SHARDS)
                    .map(|_| Vec::with_capacity(per_worker_click_bucket))
                    .collect::<Vec<Vec<RawClick>>>();
                let mut date_cache = date_cache;
                loop {
                    let chunk = click_index.fetch_add(1, Ordering::Relaxed);
                    if chunk >= click_morsels.len() {
                        break;
                    }
                    let morsel = click_morsels[chunk];
                    let mut start = 0usize;
                    for end in memchr_iter(b'\n', morsel) {
                        if end > start {
                            parse_click_line(&morsel[start..end], click_start, window_end, &camp_fast, &camp_fallback, &ids, &mut date_cache, &mut buckets);
                        }
                        start = end + 1;
                    }
                    if start < morsel.len() {
                        parse_click_line(&morsel[start..], click_start, window_end, &camp_fast, &camp_fallback, &ids, &mut date_cache, &mut buckets);
                    }
                }
                buckets
            }));
        }
        handles.into_iter().map(|h| h.join().expect("click worker")).collect::<Vec<_>>()
    });
    let click_shard_index = AtomicUsize::new(0);
    let person_click_shards: Vec<Vec<Vec<(u64, CleanClick)>>> =
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for _ in 0..worker_count {
                handles.push(scope.spawn(|| {
                    let mut table = Vec::<DedupSlot>::new();
                    let mut occupied = Vec::<u32>::new();
                    let mut outputs = (0..NUM_SHARDS)
                        .map(|_| Vec::new())
                        .collect::<Vec<Vec<(u64, CleanClick)>>>();
                    loop {
                        let s = click_shard_index.fetch_add(1, Ordering::Relaxed);
                        if s >= DEDUP_SHARDS {
                            break;
                        }
                        let total: usize = worker_click_buckets.iter().map(|w| w[s].len()).sum();
                        let cap = ((total * 3 / 2).next_power_of_two()).max(64);
                        if table.len() < cap {
                            table.resize(cap, DedupSlot { key: 0, loc: u32::MAX });
                        }
                        for pos in occupied.iter().copied() {
                            table[pos as usize].loc = u32::MAX;
                        }
                        occupied.clear();
                        for (worker, bucket) in worker_click_buckets.iter().enumerate() {
                            for (idx, cl) in bucket[s].iter().enumerate() {
                                let key = cl.click_key;
                                let loc = ((worker as u32) << 27) | idx as u32;
                                let mut pos = mix_key(key) as usize & (cap - 1);
                                loop {
                                    if table[pos].key == 0 {
                                        table[pos] = DedupSlot { key, loc };
                                        occupied.push(pos as u32);
                                        break;
                                    }
                                    if table[pos].key == key {
                                        break;
                                    }
                                    pos = (pos + 1) & (cap - 1);
                                }
                            }
                        }
                        for pos in occupied.iter().copied() {
                            let slot = table[pos as usize];
                            let w = (slot.loc >> 27) as usize;
                            let i = (slot.loc & ((1u32 << 27) - 1)) as usize;
                            let cl = worker_click_buckets[w][s][i];
                            let h = fx_hash_key(cl.user_key);
                            let bit = h as usize & ((1usize << 24) - 1);
                            let person = if canon_bloom[bit >> 6] & (1u64 << (bit & 63)) == 0 {
                                cl.user_key
                            } else {
                                canon_map.get(&cl.user_key).copied().unwrap_or(cl.user_key)
                            };
                            let person_hash = if person == cl.user_key { h } else { fx_hash_key(person) };
                            let ps = fast_shard(person_hash, NUM_SHARDS);
                            outputs[ps].push((person, CleanClick {
                                ts: cl.ts,
                                campaign_idx: cl.campaign_idx,
                                click_key: cl.click_key,
                            }));
                        }
                        for pos in occupied.iter().copied() {
                            table[pos as usize] = DedupSlot { key: 0, loc: u32::MAX };
                        }
                        occupied.clear();
                    }
                    outputs
                }));
            }
            handles.into_iter().map(|h| h.join().expect("click dedup worker")).collect()
        });

    eprintln!("clicks: {:?} cumulative: {:?}", t.elapsed(), t_start.elapsed());
    t = std::time::Instant::now();

    // 7. Sessionize, attribute, and aggregate per person shard in parallel
    let num_cats = cat_names.len();
    let num_camps = camp_names.len();

    let start_day = window_start.div_euclid(MS_PER_DAY) as i32;
    let end_day = (window_end - 1).div_euclid(MS_PER_DAY) as i32;
    let num_days = (end_day - start_day + 1).max(1) as usize;
    let shard_results: Vec<(Vec<UserDayRow>, Vec<CampaignDayAgg>)> =
        (0..NUM_SHARDS)
            .into_par_iter()
            .map(|s| {
                let ev_len: usize = person_event_shards.iter().map(|w| w[s].len()).sum();
                let cl_len: usize = person_click_shards.iter().map(|w| w[s].len()).sum();
                let mut events = Vec::<(u64, CleanEvent)>::with_capacity(ev_len);
                let mut clicks = Vec::<(u64, CleanClick)>::with_capacity(cl_len);
                for producer in &person_event_shards {
                    events.extend_from_slice(&producer[s]);
                }
                for producer in &person_click_shards {
                    clicks.extend_from_slice(&producer[s]);
                }
                events.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.ts.cmp(&b.1.ts)));
                clicks.sort_unstable_by(|a, b| {
                    a.0.cmp(&b.0)
                        .then_with(|| a.1.ts.cmp(&b.1.ts))
                        .then_with(|| ids.cmp_ids(a.1.click_key, b.1.click_key, b'k', b'K'))
                });

                let mut user_rows = Vec::new();
                let mut camp_agg: Vec<CampaignDayAgg> =
                    (0..num_camps * num_days).map(|_| CampaignDayAgg::default()).collect();
                let mut assist_stamp = vec![0u32; num_camps];
                let mut cur_stamp: u32 = 0;
                let mut day_acc = vec![(0i64, 0i64, 0i64, false); num_days];
                let mut day_cat_rev = vec![0i64; num_days * num_cats];
                let mut touched_days = Vec::with_capacity(num_days);

                let mut ei = 0usize;
                let mut ci = 0usize;
                while ei < events.len() {
                    let person = events[ei].0;
                    let mut event_end = ei + 1;
                    while event_end < events.len() && events[event_end].0 == person {
                        event_end += 1;
                    }
                    while ci < clicks.len() && clicks[ci].0 < person {
                        ci += 1;
                    }
                    let mut click_end = ci;
                    while click_end < clicks.len() && clicks[click_end].0 == person {
                        click_end += 1;
                    }
                    let person_clicks = &clicks[ci..click_end];

                    let mut click_l = 0usize;
                    let mut click_r = 0usize;
                    let mut i = ei;

                    while i < event_end {
                        let sess_start = events[i].1.ts;
                        let day = sess_start.div_euclid(MS_PER_DAY) as i32;
                        let d_idx = (day - start_day) as usize;
                        if day_acc[d_idx].0 == 0 { touched_days.push(d_idx); }
                        day_acc[d_idx].0 += 1;
                        let mut sess_events = 0i64;
                        let mut sess_rev = 0i64;
                        let mut sess_has_purchase = false;

                        let mut prev_ts = sess_start;
                        while i < event_end && (events[i].1.ts - prev_ts <= MS_30_MIN) {
                            let ev = &events[i].1;
                            prev_ts = ev.ts;
                            sess_events += 1;
                            if ev.flags & 2 != 0 {
                                let p = ev.price_cents as i64;
                                sess_rev += p;
                                sess_has_purchase = true;
                                day_cat_rev[d_idx * num_cats + ev.category_id as usize] += p;
                                day_acc[d_idx].3 = true;
                            }
                            i += 1;
                        }
                        day_acc[d_idx].1 += sess_events;
                        day_acc[d_idx].2 += sess_rev;

                        // Sliding window over clicks: [sess_start - 7d, sess_start]
                        let win_lo = sess_start - MS_7_DAYS;
                        while click_r < person_clicks.len() && person_clicks[click_r].1.ts <= sess_start {
                            click_r += 1;
                        }
                        while click_l < click_r && person_clicks[click_l].1.ts < win_lo {
                            click_l += 1;
                        }

                        if click_l < click_r {
                            // Latest ts; tie-break by smallest click_id
                            // Since clicks are sorted by (ts ASC, click_id ASC),
                            // the latest ts is clicks[click_r - 1].ts; scan back for smallest click_id at that ts.
                            let best_ts = person_clicks[click_r - 1].1.ts;
                            let mut best_idx = click_r - 1;
                            while best_idx > click_l && person_clicks[best_idx - 1].1.ts == best_ts {
                                best_idx -= 1;
                            }
                            let cred_camp = person_clicks[best_idx].1.campaign_idx;
                            let entry = &mut camp_agg[(cred_camp as usize) * num_days + (day - start_day) as usize];
                            entry.sessions += 1;
                            if sess_has_purchase {
                                entry.purchasing_sessions += 1;
                                entry.revenue_cents += sess_rev;
                            }

                            // Assists: each unique campaign in [click_l, click_r)
                            cur_stamp = cur_stamp.wrapping_add(1);
                            if cur_stamp == 0 {
                                assist_stamp.fill(0);
                                cur_stamp = 1;
                            }
                            for cl in &person_clicks[click_l..click_r] {
                                let c = cl.1.campaign_idx as usize;
                                if assist_stamp[c] != cur_stamp {
                                    assist_stamp[c] = cur_stamp;
                                    let idx = (cl.1.campaign_idx as usize) * num_days + (day - start_day) as usize;
                                    camp_agg[idx].assisted_sessions += 1;
                                }
                            }
                        } else {
                            // Organic (campaign_idx = 0)
                            let entry = &mut camp_agg[(day - start_day) as usize];
                            entry.sessions += 1;
                            if sess_has_purchase {
                                entry.purchasing_sessions += 1;
                                entry.revenue_cents += sess_rev;
                            }
                        }
                    }

                    for &d_idx in &touched_days {
                        let day = start_day + d_idx as i32;
                        let top_cat = if day_acc[d_idx].3 {
                            let mut best_c = 0usize;
                            let mut best_rev = -1i64;
                            for c in 0..num_cats {
                                let r = day_cat_rev[d_idx * num_cats + c];
                                if r > 0 {
                                    if r > best_rev || (r == best_rev && cat_names[c] < cat_names[best_c]) {
                                        best_rev = r;
                                        best_c = c;
                                    }
                                }
                            }
                            Some(best_c as u16)
                        } else {
                            None
                        };
                        user_rows.push(UserDayRow {
                            user_key: person,
                            day,
                            sessions: day_acc[d_idx].0,
                            events: day_acc[d_idx].1,
                            revenue_cents: day_acc[d_idx].2,
                            top_category: top_cat,
                        });
                        day_acc[d_idx] = (0, 0, 0, false);
                        day_cat_rev[d_idx * num_cats..(d_idx + 1) * num_cats].fill(0);
                    }
                    touched_days.clear();
                    ei = event_end;
                    ci = click_end;
                }

                (user_rows, camp_agg)
            })
            .collect();

    // 8. Merge & Write Parquet
    fs::create_dir_all(out_dir.join("user_daily")).expect("create user_daily dir");
    let fallback_ids = ids.fallback_snapshot();
    (0..8).into_par_iter().for_each(|part| {
        let start = part * NUM_SHARDS / 8;
        let end = ((part + 1) * NUM_SHARDS / 8).min(NUM_SHARDS);
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
    fallback_ids: &[&'static [u8]],
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
                fallback_ids[(r.user_key & !(1u64 << 63)) as usize]
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