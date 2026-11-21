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
use std::path::{Path, PathBuf};
use std::sync::Arc;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const MS_PER_DAY: i64 = 86_400_000;
const MS_30_MIN: i64 = 1_800_000;
const MS_7_DAYS: i64 = 604_800_000;
const NUM_SHARDS: usize = 64;

#[derive(Deserialize)]
struct Manifest {
    window_start: String,
    window_end: String,
}

#[inline]
fn fast_hash(bytes: &[u8]) -> u128 {
    xxhash_rust::xxh3::xxh3_128(bytes)
}

#[inline]
fn unquote(b: &[u8]) -> Option<&[u8]> {
    if b.is_empty() {
        return Some(b);
    }
    if b[0] == b'"' {
        if b.len() < 2 || b[b.len() - 1] != b'"' {
            return None;
        }
        let inner = &b[1..b.len() - 1];
        if memchr(b'"', inner).is_some() {
            return None;
        }
        Some(inner)
    } else {
        if memchr(b'"', b).is_some() {
            return None;
        }
        Some(b)
    }
}

#[inline]
fn parse_csv_line<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    if line.is_empty() {
        return None;
    }
    let mut out = [&[][..]; N];
    let mut start = 0;
    for i in 0..(N - 1) {
        let rel = memchr(b',', &line[start..])?;
        let end = start + rel;
        out[i] = unquote(&line[start..end])?;
        start = end + 1;
    }
    if memchr(b',', &line[start..]).is_some() {
        return None;
    }
    out[N - 1] = unquote(&line[start..])?;
    Some(out)
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
fn parse_f1_unchecked(b: &[u8]) -> i64 {
    let y = parse_4d(&b[0..4]).unwrap_or(0);
    let m = parse_2d(&b[5..7]).unwrap_or(0);
    let d = parse_2d(&b[8..10]).unwrap_or(0);
    let hh = parse_2d(&b[11..13]).unwrap_or(0);
    let mm = parse_2d(&b[14..16]).unwrap_or(0);
    let ss = parse_2d(&b[17..19]).unwrap_or(0);
    (days_from_civil(y, m, d) * 86_400 + (hh as i64) * 3600 + (mm as i64) * 60 + ss as i64) * 1000
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
    names: Vec<&'static [u8]>,
    id_to_idx: FxHashMap<&'static [u8], u32>,
}

impl Dsu {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            names: Vec::new(),
            id_to_idx: FxHashMap::default(),
        }
    }

    fn get_or_insert(&mut self, id: &'static [u8]) -> u32 {
        if let Some(&idx) = self.id_to_idx.get(id) {
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

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Keep the byte-wise smaller name as the root
        if self.names[ra as usize] < self.names[rb as usize] {
            self.parent[rb as usize] = ra;
        } else {
            self.parent[ra as usize] = rb;
        }
    }

    fn finalize(mut self) -> FxHashMap<&'static [u8], &'static [u8]> {
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

#[repr(C)]
struct RawEvent {
    eid_hash: u128,
    ingest_ts: i64,
    ts: i64,
    event_id: &'static [u8],
    user_id: &'static [u8],
    price_cents: u32,
    category_id: u16,
    flags: u8,
}

struct CleanEvent {
    ts: i64,
    price_cents: u32,
    category_id: u16,
    flags: u8,
}

struct CleanClick {
    ts: i64,
    campaign_idx: u16,
    click_id: &'static [u8],
}

#[repr(C)]
struct RawClick {
    cid_hash: u128,
    ts: i64,
    click_id: &'static [u8],
    user_id: &'static [u8],
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
    user_id: Box<[u8]>,
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

    // 2. Products -> category_id (0 = "unknown")
    let mut cat_names: Vec<String> = vec!["unknown".to_string()];
    let mut cat_map: FxHashMap<String, u16> = FxHashMap::default();
    cat_map.insert("unknown".to_string(), 0);
    let mut prod_to_cat: FxHashMap<Box<[u8]>, u16> = FxHashMap::default();
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
            prod_to_cat.insert(pid.into(), cid);
        }
    });

    // 3. Campaigns -> campaign_idx (0 = "organic")
    let mut camp_names: Vec<String> = vec!["organic".to_string()];
    let mut camp_channels: Vec<String> = vec!["organic".to_string()];
    let mut camp_map: FxHashMap<Box<[u8]>, u16> = FxHashMap::default();
    for_each_data_line(&data_dir.join("campaigns.csv"), |line| {
        if let Some([cid, ch]) = parse_csv_line::<2>(line) {
            let idx = camp_names.len() as u16;
            camp_names.push(String::from_utf8_lossy(cid).into_owned());
            camp_channels.push(String::from_utf8_lossy(ch).into_owned());
            camp_map.insert(cid.into(), idx);
        }
    });

    // 4. Links -> canonical user_id map
    let mut dsu = Dsu::new();
    for path in list_csvs(&data_dir.join("links")) {
        if let Some(data) = mmap_data(&path) {
            let mut pos = match memchr(b'\n', data) {
                Some(i) => i + 1,
                None => continue,
            };
            while pos < data.len() {
                let rel = memchr(b'\n', &data[pos..]).unwrap_or(data.len() - pos);
                let line = &data[pos..pos + rel];
                pos += rel + 1;
                if let Some([a, b]) = parse_csv_line::<2>(line) {
                    if !a.is_empty() && !b.is_empty() {
                        let ia = dsu.get_or_insert(a);
                        let ib = dsu.get_or_insert(b);
                        dsu.union(ia, ib);
                    }
                }
            }
        }
    }
    let canon_map = dsu.finalize();
    eprintln!("links: {:?} cumulative: {:?}", t.elapsed(), t_start.elapsed());
    t = std::time::Instant::now();

    // 5. Events: parallel parse & local shard, then parallel dedup
    let event_morsels: Vec<&'static [u8]> = list_csvs(&data_dir.join("events"))
        .iter()
        .filter_map(|path| mmap_data(path))
        .flat_map(|data| morsels(data).into_iter().map(|(m, _)| m))
        .collect();
    let event_shards: Vec<Vec<(u128, RawEvent)>> = event_morsels
        .par_iter()
        .fold(
            Vec::new,
            |mut acc, &morsel| {
                if acc.is_empty() {
                    let cap = morsel.len() / (NUM_SHARDS * 70).max(1);
                    acc = (0..NUM_SHARDS).map(|_| Vec::with_capacity(cap)).collect();
                }
                let mut pos = 0;
                while pos < morsel.len() {
                    let rel = memchr(b'\n', &morsel[pos..]).unwrap_or(morsel.len() - pos);
                    let line = &morsel[pos..pos + rel];
                    pos += rel + 1;
                    let Some([eid, uid, ts_b, ing_b, etype, pid, price_b]) = parse_csv_line::<7>(line) else {
                        continue;
                    };
                    if eid.is_empty() {
                        continue;
                    }
                    let ingest_ts = if ing_b.len() == 20 && ing_b[19] == b'Z' {
                        parse_f1_unchecked(ing_b)
                    } else if let Some(v) = parse_ts(ing_b) {
                        v
                    } else {
                        continue;
                    };
                    let mut flags = 0u8;
                    let mut ts = 0;
                    let mut price_cents = 0;
                    let mut category_id = 0;

                    if !uid.is_empty() && !pid.is_empty() {
                        if let Some(parsed_ts) = parse_ts(ts_b) {
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
                                            category_id = *prod_to_cat.get(pid).unwrap_or(&0);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    let key = fast_hash(eid);
                    let shard = (key as usize) & (NUM_SHARDS - 1);
                    acc[shard].push((
                        key,
                        RawEvent {
                            eid_hash: key,
                            ingest_ts,
                            ts,
                            event_id: eid,
                            user_id: if flags & 1 != 0 { uid } else { &[] },
                            price_cents,
                            category_id,
                            flags,
                        },
                    ));
                }
                acc
            },
        )
        .reduce(
            || (0..NUM_SHARDS).map(|_| Vec::new()).collect::<Vec<_>>(),
            |mut a, b| {
                for (sa, mut sb) in a.iter_mut().zip(b.into_iter()) {
                    sa.append(&mut sb);
                }
                a
            },
        );

    eprintln!("events_parse: {:?} cumulative: {:?}", t.elapsed(), t_start.elapsed());
    t = std::time::Instant::now();

    // Dedup events per shard & repartition by canonical person
    let person_event_shards: Vec<Vec<(Box<[u8]>, CleanEvent)>> = event_shards
        .into_par_iter()
        .map(|shard_vec| {
            let mut shard_vec = shard_vec;
            shard_vec.sort_unstable_by(|a, b| {
                a.1.eid_hash.cmp(&b.1.eid_hash)
                    .then_with(|| a.1.event_id.cmp(b.1.event_id))
                    .then_with(|| a.1.ingest_ts.cmp(&b.1.ingest_ts))
            });
            let mut out = (0..NUM_SHARDS)
                .map(|_| Vec::with_capacity(shard_vec.len() / NUM_SHARDS))
                .collect::<Vec<_>>();
            let mut i = 0usize;
            while i < shard_vec.len() {
                let hash = shard_vec[i].1.eid_hash;
                let eid = shard_vec[i].1.event_id;
                let mut j = i + 1;
                while j < shard_vec.len()
                    && shard_vec[j].1.eid_hash == hash
                    && shard_vec[j].1.event_id == eid
                {
                    j += 1;
                }
                let ev = &shard_vec[i].1;
                if ev.flags & 1 != 0 {
                    let person: Box<[u8]> = canon_map
                        .get(ev.user_id)
                        .cloned()
                        .map(|p| p.into())
                        .unwrap_or_else(|| ev.user_id.into());
                    let s = (fast_hash(&person) as usize) & (NUM_SHARDS - 1);
                    out[s].push((
                        person,
                        CleanEvent {
                            ts: ev.ts,
                            price_cents: ev.price_cents,
                            category_id: ev.category_id,
                            flags: ev.flags,
                        },
                    ));
                }
                i = j;
            }
            out
        })
        .reduce(
            || (0..NUM_SHARDS).map(|_| Vec::new()).collect::<Vec<_>>(),
            |mut a, b| {
                for (sa, mut sb) in a.iter_mut().zip(b.into_iter()) {
                    sa.append(&mut sb);
                }
                a
            },
        );

    eprintln!("events_dedup: {:?} cumulative: {:?}", t.elapsed(), t_start.elapsed());
    t = std::time::Instant::now();

    // 6. Clicks: parallel parse, dedup & repartition by canonical person
    let click_morsels: Vec<&'static [u8]> = list_csvs(&data_dir.join("clicks"))
        .iter()
        .filter_map(|path| mmap_data(path))
        .flat_map(|data| morsels(data).into_iter().map(|(m, _)| m))
        .collect();
    let click_shards: Vec<Vec<RawClick>> = click_morsels
        .par_iter()
        .fold(
            Vec::new,
            |mut acc, &morsel| {
                if acc.is_empty() {
                    let cap = morsel.len() / (NUM_SHARDS * 70).max(1);
                    acc = (0..NUM_SHARDS).map(|_| Vec::with_capacity(cap)).collect();
                }
                let mut pos = 0;
                while pos < morsel.len() {
                    let rel = memchr(b'\n', &morsel[pos..]).unwrap_or(morsel.len() - pos);
                    let line = &morsel[pos..pos + rel];
                    pos += rel + 1;
                    let Some([cid, uid, camp_b, ts_b]) = parse_csv_line::<4>(line) else {
                        continue;
                    };
                    if cid.is_empty() || uid.is_empty() {
                        continue;
                    }
                    let Some(&camp_idx) = camp_map.get(camp_b) else {
                        continue;
                    };
                    let Some(ts) = parse_ts(ts_b) else {
                        continue;
                    };
                    if ts < click_start || ts >= window_end {
                        continue;
                    }
                    let key = fast_hash(cid);
                    let s = (key as usize) & (NUM_SHARDS - 1);
                    acc[s].push(RawClick {
                        cid_hash: key,
                        ts,
                        click_id: cid,
                        user_id: uid,
                        campaign_idx: camp_idx,
                    });
                }
                acc
            },
        )
        .reduce(
            || (0..NUM_SHARDS).map(|_| Vec::new()).collect::<Vec<_>>(),
            |mut a, b| {
                for (sa, mut sb) in a.iter_mut().zip(b.into_iter()) {
                    sa.append(&mut sb);
                }
                a
            },
        );

    let person_click_shards: Vec<Vec<(Box<[u8]>, CleanClick)>> = click_shards
        .into_par_iter()
        .map(|shard_vec| {
            let mut shard_vec = shard_vec;
            shard_vec.sort_unstable_by(|a, b| {
                a.cid_hash.cmp(&b.cid_hash)
                    .then_with(|| a.click_id.cmp(b.click_id))
            });
            let mut out = (0..NUM_SHARDS)
                .map(|_| Vec::with_capacity(shard_vec.len() / NUM_SHARDS))
                .collect::<Vec<_>>();
            let mut i = 0usize;
            while i < shard_vec.len() {
                let hash = shard_vec[i].cid_hash;
                let cid = shard_vec[i].click_id;
                let mut j = i + 1;
                while j < shard_vec.len()
                    && shard_vec[j].cid_hash == hash
                    && shard_vec[j].click_id == cid
                {
                    j += 1;
                }
                let cl = &shard_vec[i];
                let person: Box<[u8]> = canon_map
                    .get(cl.user_id)
                    .cloned()
                    .map(|p| p.into())
                    .unwrap_or_else(|| cl.user_id.into());
                let s = (fast_hash(&person) as usize) & (NUM_SHARDS - 1);
                out[s].push((
                    person,
                    CleanClick {
                        ts: cl.ts,
                        campaign_idx: cl.campaign_idx,
                        click_id: cl.click_id,
                    },
                ));
                i = j;
            }
            out
        })
        .reduce(
            || (0..NUM_SHARDS).map(|_| Vec::new()).collect::<Vec<_>>(),
            |mut a, b| {
                for (sa, mut sb) in a.iter_mut().zip(b.into_iter()) {
                    sa.append(&mut sb);
                }
                a
            },
        );

    eprintln!("clicks: {:?} cumulative: {:?}", t.elapsed(), t_start.elapsed());
    t = std::time::Instant::now();

    // 7. Sessionize, attribute, and aggregate per person shard in parallel
    let num_cats = cat_names.len();
    let num_camps = camp_names.len();

    let start_day = window_start.div_euclid(MS_PER_DAY) as i32;
    let end_day = (window_end - 1).div_euclid(MS_PER_DAY) as i32;
    let num_days = (end_day - start_day + 1).max(1) as usize;
    let shard_results: Vec<(Vec<UserDayRow>, Vec<CampaignDayAgg>)> =
        person_event_shards
            .into_par_iter()
            .zip(person_click_shards.into_par_iter())
            .map(|(ev_vec, cl_vec)| {
                let mut ev_by_person: FxHashMap<Box<[u8]>, Vec<CleanEvent>> = FxHashMap::default();
                for (p, ev) in ev_vec {
                    ev_by_person.entry(p).or_default().push(ev);
                }
                let mut cl_by_person: FxHashMap<Box<[u8]>, Vec<CleanClick>> = FxHashMap::default();
                for (p, cl) in cl_vec {
                    cl_by_person.entry(p).or_default().push(cl);
                }

                let mut user_rows = Vec::new();
                let mut camp_agg: Vec<CampaignDayAgg> =
                    (0..num_camps * num_days).map(|_| CampaignDayAgg::default()).collect();
                let mut assist_stamp = vec![0u32; num_camps];
                let mut cur_stamp: u32 = 0;

                for (person, mut events) in ev_by_person {
                    events.sort_unstable_by_key(|e| e.ts);
                    let mut clicks = cl_by_person.remove(&person).unwrap_or_default();
                    clicks.sort_unstable_by(|a, b| a.ts.cmp(&b.ts).then_with(|| a.click_id.cmp(&b.click_id)));

                    // Per-day accumulator for this person
                    struct DayAcc {
                        sessions: i64,
                        events: i64,
                        revenue_cents: i64,
                        cat_rev: Vec<i64>,
                        has_purchase: bool,
                    }
                    let mut per_day: FxHashMap<i32, DayAcc> = FxHashMap::default();

                    let mut click_l = 0usize;
                    let mut click_r = 0usize;
                    let mut i = 0usize;
                    let n = events.len();

                    while i < n {
                        let sess_start = events[i].ts;
                        let day = sess_start.div_euclid(MS_PER_DAY) as i32;
                        let mut sess_events = 0i64;
                        let mut sess_rev = 0i64;
                        let mut sess_has_purchase = false;

                        let day_acc = per_day.entry(day).or_insert_with(|| DayAcc {
                            sessions: 0,
                            events: 0,
                            revenue_cents: 0,
                            cat_rev: vec![0; num_cats],
                            has_purchase: false,
                        });
                        day_acc.sessions += 1;

                        let mut prev_ts = sess_start;
                        while i < n && (events[i].ts - prev_ts <= MS_30_MIN) {
                            let ev = &events[i];
                            prev_ts = ev.ts;
                            sess_events += 1;
                            if ev.flags & 2 != 0 {
                                let p = ev.price_cents as i64;
                                sess_rev += p;
                                sess_has_purchase = true;
                                day_acc.cat_rev[ev.category_id as usize] += p;
                                day_acc.has_purchase = true;
                            }
                            i += 1;
                        }
                        day_acc.events += sess_events;
                        day_acc.revenue_cents += sess_rev;

                        // Sliding window over clicks: [sess_start - 7d, sess_start]
                        let win_lo = sess_start - MS_7_DAYS;
                        while click_r < clicks.len() && clicks[click_r].ts <= sess_start {
                            click_r += 1;
                        }
                        while click_l < click_r && clicks[click_l].ts < win_lo {
                            click_l += 1;
                        }

                        if click_l < click_r {
                            // Latest ts; tie-break by smallest click_id
                            // Since clicks are sorted by (ts ASC, click_id ASC),
                            // the latest ts is clicks[click_r - 1].ts; scan back for smallest click_id at that ts.
                            let best_ts = clicks[click_r - 1].ts;
                            let mut best_idx = click_r - 1;
                            while best_idx > click_l && clicks[best_idx - 1].ts == best_ts {
                                best_idx -= 1;
                            }
                            let cred_camp = clicks[best_idx].campaign_idx;
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
                            for cl in &clicks[click_l..click_r] {
                                let c = cl.campaign_idx as usize;
                                if assist_stamp[c] != cur_stamp {
                                    assist_stamp[c] = cur_stamp;
                                    let idx = (cl.campaign_idx as usize) * num_days + (day - start_day) as usize;
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

                    for (day, acc) in per_day {
                        let top_cat = if acc.has_purchase {
                            let mut best_c = 0usize;
                            let mut best_rev = -1i64;
                            for c in 0..num_cats {
                                let r = acc.cat_rev[c];
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
                            user_id: person.clone(),
                            day,
                            sessions: acc.sessions,
                            events: acc.events,
                            revenue_cents: acc.revenue_cents,
                            top_category: top_cat,
                        });
                    }
                }

                (user_rows, camp_agg)
            })
            .collect();

    // 8. Merge & Write Parquet
    fs::create_dir_all(out_dir.join("user_daily")).expect("create user_daily dir");
    shard_results
        .par_iter()
        .enumerate()
        .chunks(8)
        .enumerate()
        .for_each(|(part, batch)| {
            let mut rows = Vec::new();
            for (_, (shard_rows, _)) in batch {
                rows.extend_from_slice(shard_rows);
            }
            write_user_daily_shard(&out_dir.join("user_daily"), part, &rows, &cat_names);
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

fn write_user_daily_shard(dir: &Path, shard: usize, rows: &[UserDayRow], cat_names: &[String]) {
    fs::create_dir_all(dir).expect("create user_daily dir");
    let schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("events", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", DataType::Utf8, true),
    ]));

    let mut uid_b = StringBuilder::with_capacity(rows.len(), rows.len() * 12);
    let mut day_b = Date32Builder::with_capacity(rows.len());
    let mut ses_b = Int64Builder::with_capacity(rows.len());
    let mut ev_b = Int64Builder::with_capacity(rows.len());
    let mut rev_b = Int64Builder::with_capacity(rows.len());
    let mut cat_b = StringBuilder::with_capacity(rows.len(), rows.len() * 8);

    for r in rows {
        uid_b.append_value(std::str::from_utf8(&r.user_id).expect("utf8 uid"));
        day_b.append_value(r.day);
        ses_b.append_value(r.sessions);
        ev_b.append_value(r.events);
        rev_b.append_value(r.revenue_cents);
        match r.top_category {
            Some(c) => cat_b.append_value(&cat_names[c as usize]),
            None => cat_b.append_null(),
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