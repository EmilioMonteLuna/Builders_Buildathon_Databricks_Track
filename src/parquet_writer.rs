use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

const STOP: u8 = 0;
const I32: u8 = 5;
const I64: u8 = 6;
const BINARY: u8 = 8;
const LIST: u8 = 9;
const STRUCT: u8 = 12;

#[derive(Clone, Copy)]
pub enum PhysicalType {
    Int32,
    Int64,
    ByteArray,
}

impl PhysicalType {
    fn parquet(self) -> i32 {
        match self {
            Self::Int32 => 1,
            Self::Int64 => 2,
            Self::ByteArray => 6,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Column<'a> {
    pub name: &'a str,
    pub physical_type: PhysicalType,
    pub converted_type: Option<i32>,
    pub values: &'a [u8],
    pub definition_levels: Option<&'a [u8]>,
}

struct Compact {
    bytes: Vec<u8>,
    last_field: i16,
}

impl Compact {
    fn new() -> Self {
        Self { bytes: Vec::new(), last_field: 0 }
    }

    fn varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.bytes.push((value as u8) | 0x80);
            value >>= 7;
        }
        self.bytes.push(value as u8);
    }

    fn field(&mut self, id: i16, kind: u8) {
        let delta = id - self.last_field;
        if (1..=15).contains(&delta) {
            self.bytes.push(((delta as u8) << 4) | kind);
        } else {
            self.bytes.push(kind);
            self.varint(((id << 1) ^ (id >> 15)) as u16 as u64);
        }
        self.last_field = id;
    }

    fn int32(&mut self, id: i16, value: i32) {
        self.field(id, I32);
        self.varint(((value << 1) ^ (value >> 31)) as u32 as u64);
    }

    fn int64(&mut self, id: i16, value: i64) {
        self.field(id, I64);
        self.varint(((value << 1) ^ (value >> 63)) as u64);
    }

    fn binary(&mut self, id: i16, value: &[u8]) {
        self.field(id, BINARY);
        self.varint(value.len() as u64);
        self.bytes.extend_from_slice(value);
    }

    fn struct_value(&mut self, id: i16, value: &[u8]) {
        self.field(id, STRUCT);
        self.bytes.extend_from_slice(value);
    }

    fn list(&mut self, id: i16, element_kind: u8, count: usize, values: &[u8]) {
        self.field(id, LIST);
        write_list_header(&mut self.bytes, element_kind, count);
        self.bytes.extend_from_slice(values);
    }

    fn finish(mut self) -> Vec<u8> {
        self.bytes.push(STOP);
        self.bytes
    }
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn write_list_header(out: &mut Vec<u8>, element_kind: u8, count: usize) {
    if count <= 14 {
        out.push(((count as u8) << 4) | element_kind);
    } else {
        out.push(0xf0 | element_kind);
        write_varint(out, count as u64);
    }
}

fn schema_element(
    name: &str,
    physical: Option<PhysicalType>,
    repetition: Option<i32>,
    converted: Option<i32>,
    children: Option<i32>,
) -> Vec<u8> {
    let mut t = Compact::new();
    if let Some(physical) = physical {
        t.int32(1, physical.parquet());
    }
    if let Some(repetition) = repetition {
        t.int32(3, repetition);
    }
    t.binary(4, name.as_bytes());
    if let Some(children) = children {
        t.int32(5, children);
    }
    if let Some(converted) = converted {
        t.int32(6, converted);
    }
    t.finish()
}

fn data_page_header(rows: usize, body_size: usize) -> Vec<u8> {
    let mut data = Compact::new();
    data.int32(1, rows as i32);
    data.int32(2, 0);
    data.int32(3, 3);
    data.int32(4, 3);
    let data = data.finish();

    let mut page = Compact::new();
    page.int32(1, 0);
    page.int32(2, body_size as i32);
    page.int32(3, body_size as i32);
    page.struct_value(5, &data);
    page.finish()
}

fn encode_definition_levels(levels: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(levels.len().div_ceil(8) + levels.len().div_ceil(504));
    let mut row = 0usize;
    while row < levels.len() {
        let groups = (levels.len() - row).div_ceil(8).min(63);
        write_varint(&mut encoded, ((groups << 1) | 1) as u64);
        for group in 0..groups {
            let mut packed = 0u8;
            for bit in 0..8 {
                let index = row + group * 8 + bit;
                if index < levels.len() && levels[index] != 0 {
                    packed |= 1 << bit;
                }
            }
            encoded.push(packed);
        }
        row += groups * 8;
    }
    encoded
}

struct ColumnLocation<'a> {
    column: Column<'a>,
    offset: i64,
    total_size: i64,
}

fn column_metadata(location: &ColumnLocation<'_>, rows: usize) -> Vec<u8> {
    let mut encodings = Vec::new();
    write_varint(&mut encodings, 0);
    if location.column.definition_levels.is_some() {
        write_varint(&mut encodings, 6);
    }

    let mut paths = Vec::new();
    write_varint(&mut paths, location.column.name.len() as u64);
    paths.extend_from_slice(location.column.name.as_bytes());

    let mut metadata = Compact::new();
    metadata.int32(1, location.column.physical_type.parquet());
    metadata.list(2, I32, if location.column.definition_levels.is_some() { 2 } else { 1 }, &encodings);
    metadata.list(3, BINARY, 1, &paths);
    metadata.int32(4, 0);
    metadata.int64(5, rows as i64);
    metadata.int64(6, location.total_size);
    metadata.int64(7, location.total_size);
    metadata.int64(9, location.offset);
    metadata.finish()
}

fn column_chunk(location: &ColumnLocation<'_>, rows: usize) -> Vec<u8> {
    let metadata = column_metadata(location, rows);
    let mut chunk = Compact::new();
    chunk.int64(2, location.offset);
    chunk.struct_value(3, &metadata);
    chunk.finish()
}

fn row_group(locations: &[ColumnLocation<'_>], rows: usize) -> Vec<u8> {
    let chunks: Vec<Vec<u8>> = locations.iter().map(|column| column_chunk(column, rows)).collect();
    let mut chunk_values = Vec::new();
    for chunk in &chunks {
        chunk_values.extend_from_slice(chunk);
    }
    let total_size: i64 = locations.iter().map(|column| column.total_size).sum();
    let mut group = Compact::new();
    group.list(1, STRUCT, chunks.len(), &chunk_values);
    group.int64(2, total_size);
    group.int64(3, rows as i64);
    group.finish()
}

fn file_metadata(columns: &[Column<'_>], locations: &[ColumnLocation<'_>], rows: usize) -> Vec<u8> {
    let mut schema = Vec::new();
    schema.extend_from_slice(&schema_element("schema", None, None, None, Some(columns.len() as i32)));
    for column in columns {
        let repetition = if column.definition_levels.is_some() { 1 } else { 0 };
        schema.extend_from_slice(&schema_element(
            column.name,
            Some(column.physical_type),
            Some(repetition),
            column.converted_type,
            None,
        ));
    }
    let groups = if rows == 0 { Vec::new() } else { row_group(locations, rows) };
    let mut metadata = Compact::new();
    metadata.int32(1, 1);
    metadata.list(2, STRUCT, columns.len() + 1, &schema);
    metadata.int64(3, rows as i64);
    metadata.list(4, STRUCT, usize::from(rows != 0), &groups);
    metadata.binary(6, b"shopping-sessions custom parquet v1");
    metadata.finish()
}

pub fn write_table(path: &Path, rows: usize, columns: &[Column<'_>]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut output = BufWriter::with_capacity(1024 * 1024, file);
    output.write_all(b"PAR1")?;
    let mut offset = 4i64;
    let mut locations = Vec::with_capacity(columns.len());

    if rows != 0 {
        for column in columns {
            let levels = column.definition_levels.map(encode_definition_levels);
            let body_size = column.values.len() + levels.as_ref().map_or(0, |v| v.len() + 4);
            let header = data_page_header(rows, body_size);
            let total_size = (header.len() + body_size) as i64;
            locations.push(ColumnLocation {
                column: *column,
                offset,
                total_size,
            });
            output.write_all(&header)?;
            if let Some(levels) = levels {
                output.write_all(&(levels.len() as u32).to_le_bytes())?;
                output.write_all(&levels)?;
            }
            output.write_all(column.values)?;
            offset += total_size;
        }
    }

    let footer = file_metadata(columns, &locations, rows);
    output.write_all(&footer)?;
    let footer_len = u32::try_from(footer.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Parquet footer exceeds 4 GiB"))?;
    output.write_all(&footer_len.to_le_bytes())?;
    output.write_all(b"PAR1")?;
    output.flush()
}
