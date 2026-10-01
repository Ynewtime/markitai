//! markitai: embedded OLE objects ([MS-PPT] `ExEmbed`), read as the data
//! they hold. A shape showing an object names it with an `ExObjRefAtom`; the
//! document's `ExObjList` maps that id to an `ExOleObjStg` record through the
//! persist directory, and the record holds the object's compound file,
//! zlib-compressed when its instance is 1. Three kinds of storage are read:
//!
//! - a BIFF workbook (`Workbook` or `Book` stream): an Excel worksheet or
//!   chart, or an MS Graph chart, through the spreadsheet reader;
//! - LibreOffice's `package_stream`, an OpenDocument package: a chart reads
//!   as its title and the data table it keeps, a spreadsheet as its tables;
//! - an OOXML `Package` stream holding a workbook.
//!
//! Anything else (an equation, a document, a picture) keeps upstream's
//! behavior of contributing no text, with a debug log naming it.

use super::children;
use crate::error::ConvertError;
use crate::model::{Block, Cell, CellSlot, ImageSource, Inline, Style, Table, TableKind};
use crate::package::Package;
use crate::package::limits;
use crate::package::xml::{Element, ns};
use crate::shared::binary::{get_u32, read_ole_stream, utf16le_units};
use crate::shared::officeart::record_at;
use crate::shared::text::{clean_text, collapse_ws};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Cursor, Read};

/// `ExEmbedContainer`: one embedded object of the `ExObjList`.
const EX_EMBED: u16 = 0x0FCC;
/// `ExOleObjAtom`: exObjId at 8, persistIdRef at 16.
const EX_OLE_OBJ_ATOM: u16 = 0x0FC3;
/// `ExOleObjStg`, compressed (instance 1) or not (instance 0).
const EX_OLE_OBJ_STG: u16 = 0x1011;
/// `CString`; instance 2 in an `ExEmbed` is the object's ProgID.
const CSTRING: u16 = 0x0FBA;

/// An embedded object the deck lists: where its storage record is, and the
/// ProgID it was saved with (empty when the writer left it out).
struct Listed {
    offset: usize,
    prog_id: String,
}

/// The deck's embedded objects, decoded on first reference.
#[derive(Default)]
pub(super) struct Objects<'a> {
    stream: &'a [u8],
    listed: HashMap<u32, Listed>,
    /// Blocks by exObjId, empty when the object holds nothing readable.
    decoded: HashMap<u32, Vec<Block>>,
    /// Bytes decompressed so far, against `MAX_TOTAL_BYTES`.
    inflated: u64,
    /// Table positions written so far, against `MAX_GRID_SLOTS`: an object
    /// a thousand shapes name is written a thousand times.
    written: u64,
}

impl<'a> Objects<'a> {
    /// The embedded objects of `list` (the `ExObjList` body), resolved
    /// through the persist directory into `stream`. Linked objects and
    /// controls are not embedded data and are not listed.
    pub(super) fn new(
        list: Option<&[u8]>,
        persist: &HashMap<u32, usize>,
        stream: &'a [u8],
    ) -> Objects<'a> {
        let mut listed = HashMap::new();
        for (_, rec_type, body) in list.into_iter().flat_map(children) {
            if rec_type != EX_EMBED {
                continue;
            }
            let Some((.., atom)) = children(body).find(|&(_, t, _)| t == EX_OLE_OBJ_ATOM) else {
                continue;
            };
            let (Some(id), Some(persist_ref)) = (get_u32(atom, 8), get_u32(atom, 16)) else {
                continue;
            };
            let Some(&offset) = persist.get(&persist_ref) else {
                continue;
            };
            let prog_id = children(body)
                .find(|&(ver_inst, t, _)| t == CSTRING && ver_inst >> 4 == 2)
                .map(|(.., text)| {
                    String::from_utf16_lossy(&utf16le_units(text).collect::<Vec<_>>())
                })
                .unwrap_or_default();
            listed.entry(id).or_insert(Listed { offset, prog_id });
        }
        Objects { stream, listed, ..Objects::default() }
    }

    /// The blocks of the object `id` names, read the first time it is asked
    /// for. Empty when there is no such embedded object or it holds nothing
    /// this reader can read.
    pub(super) fn blocks(&mut self, id: u32) -> Result<Vec<Block>, ConvertError> {
        if !self.decoded.contains_key(&id) {
            let blocks = self.decode(id)?;
            self.decoded.insert(id, blocks);
        }
        let blocks = &self.decoded[&id];
        self.written = self.written.saturating_add(positions(blocks));
        if self.written > limits::MAX_GRID_SLOTS {
            return Err(ConvertError::ResourceLimit {
                limit: "max_grid_slots",
                detail: format!("embedded objects write {} table positions", self.written),
            });
        }
        Ok(blocks.clone())
    }

    fn decode(&mut self, id: u32) -> Result<Vec<Block>, ConvertError> {
        let Some(Listed { offset, prog_id }) = self.listed.get(&id) else {
            log::debug!("ppt shape shows object {id}, which is not an embedded object; not read");
            return Ok(Vec::new());
        };
        let name = if prog_id.is_empty() { "no ProgID".to_string() } else { prog_id.clone() };
        let stream = self.stream;
        let storage = match record_at(stream, *offset) {
            Some((ver_inst, EX_OLE_OBJ_STG, body)) if ver_inst >> 4 == 0 => Cow::Borrowed(body),
            Some((ver_inst, EX_OLE_OBJ_STG, body)) if ver_inst >> 4 == 1 => {
                match self.inflate(body)? {
                    Some(bytes) => Cow::Owned(bytes),
                    None => {
                        log::debug!("ppt embedded object {id} ({name}) does not decompress");
                        return Ok(Vec::new());
                    }
                }
            }
            _ => {
                log::debug!("ppt embedded object {id} ({name}) has no storage record");
                return Ok(Vec::new());
            }
        };
        let blocks = read_storage(&storage)?;
        if blocks.is_empty() {
            log::debug!(
                "ppt embedded object {id} ({name}) holds no data this reader reads; left out"
            );
        }
        Ok(blocks)
    }

    /// A compressed storage, its bytes charged against `MAX_TOTAL_BYTES`.
    fn inflate(&mut self, body: &[u8]) -> Result<Option<Vec<u8>>, ConvertError> {
        let Some(out) = inflate(body, limits::MAX_ENTRY_BYTES)? else {
            return Ok(None);
        };
        self.inflated = self.inflated.saturating_add(out.len() as u64);
        if self.inflated > limits::MAX_TOTAL_BYTES {
            return Err(ConvertError::ResourceLimit {
                limit: "max_total_bytes",
                detail: format!("embedded objects decompress to {} bytes", self.inflated),
            });
        }
        Ok(Some(out))
    }
}

/// A compressed storage: the decompressed size, then a zlib stream. The
/// declared size is not trusted; the output stops past `cap` instead, which
/// is a resource limit. `None` when the stream is corrupt.
fn inflate(body: &[u8], cap: u64) -> Result<Option<Vec<u8>>, ConvertError> {
    let Some(compressed) = body.get(4..) else {
        return Ok(None);
    };
    let mut out = Vec::new();
    let read = flate2::read::ZlibDecoder::new(compressed)
        .take(cap.saturating_add(1))
        .read_to_end(&mut out);
    if let Err(e) = read {
        log::debug!("ppt embedded object storage: {e}");
        return Ok(None);
    }
    if out.len() as u64 > cap {
        return Err(ConvertError::ResourceLimit {
            limit: "max_entry_bytes",
            detail: "an embedded object decompresses past the read cap".to_string(),
        });
    }
    Ok(Some(out))
}

/// Table positions and blocks in `blocks`, charged each time they are
/// written.
fn positions(blocks: &[Block]) -> u64 {
    blocks
        .iter()
        .map(|block| match block {
            Block::Table(table) => table.grid.iter().map(|row| row.len() as u64).sum::<u64>(),
            _ => 1,
        })
        .sum()
}

/// The data of one object's compound file. Readers that fail on the
/// object's own content degrade to nothing; resource limits propagate.
fn read_storage(bytes: &[u8]) -> Result<Vec<Block>, ConvertError> {
    let Ok(mut ole) = cfb::CompoundFile::open(Cursor::new(bytes)) else {
        log::debug!("ppt embedded object storage is not a compound file");
        return Ok(Vec::new());
    };
    let streams: Vec<String> = ole
        .read_root_storage()
        .filter(|entry| entry.is_stream())
        .map(|entry| entry.name().to_string())
        .collect();
    let named = |name: &str| streams.iter().find(|s| s.eq_ignore_ascii_case(name)).cloned();
    let read = if named("Workbook").or_else(|| named("Book")).is_some() {
        crate::formats::sheet::embedded_workbook(bytes)
    } else if let Some(stream) = named("package_stream") {
        read_ole_stream(&mut ole, &stream).and_then(|package| odf_object(&package))
    } else if let Some(stream) = named("Package") {
        read_ole_stream(&mut ole, &stream)
            .and_then(|package| crate::formats::sheet::parse(&package))
            .map(|doc| doc.blocks)
    } else {
        log::debug!("ppt embedded object storage holds streams {streams:?}");
        return Ok(Vec::new());
    };
    match read {
        Ok(mut blocks) => {
            detach(&mut blocks);
            Ok(sheet_names_as_paragraphs(blocks))
        }
        Err(e @ ConvertError::ResourceLimit { .. }) => Err(e),
        Err(e) => {
            log::debug!("ppt embedded object unreadable: {e}");
            Ok(Vec::new())
        }
    }
}

/// Blocks read from an object's own document, made to stand in the deck:
/// that document's assets and notes are not the deck's, so an image keeps
/// only its alt text and a note reference is dropped.
fn detach(blocks: &mut [Block]) {
    for block in blocks {
        match block {
            Block::Paragraph(inlines) | Block::Heading { content: inlines, .. } => {
                detach_inlines(inlines);
            }
            Block::List(list) => list.items.iter_mut().for_each(|item| detach(&mut item.blocks)),
            Block::Table(table) => {
                for slot in table.grid.iter_mut().flatten() {
                    if let CellSlot::Origin(cell) = slot {
                        detach(&mut cell.blocks);
                    }
                }
            }
            Block::BlockQuote(blocks) => detach(blocks),
            _ => {}
        }
    }
}

fn detach_inlines(inlines: &mut Vec<Inline>) {
    inlines.retain(|inline| !matches!(inline, Inline::NoteRef(_)));
    for inline in inlines {
        match inline {
            Inline::Image { source: source @ ImageSource::Asset(_), .. } => {
                *source = ImageSource::Unavailable;
            }
            Inline::Link { content, .. } => detach_inlines(content),
            _ => {}
        }
    }
}

/// A workbook of several sheets names each sheet in a heading, which on a
/// slide would read as another slide title: the name of the one sheet with
/// data is dropped, and several names become paragraphs.
fn sheet_names_as_paragraphs(blocks: Vec<Block>) -> Vec<Block> {
    let tables = blocks.iter().filter(|b| matches!(b, Block::Table(_))).count();
    blocks
        .into_iter()
        .filter_map(|block| match block {
            Block::Heading { content, .. } => (tables > 1).then_some(Block::Paragraph(content)),
            other => Some(other),
        })
        .collect()
}

/// An OpenDocument object: a chart's title and data table, or a
/// spreadsheet's tables. Other documents are not data and are left out.
fn odf_object(package: &[u8]) -> Result<Vec<Block>, ConvertError> {
    let content = Package::open(package)?.optional_xml_part("content.xml")?;
    let Some(body) = content.as_ref().and_then(|tree| {
        tree.find(ns::OFFICE, "document-content").and_then(|doc| doc.find(ns::OFFICE, "body"))
    }) else {
        return Ok(Vec::new());
    };
    if let Some(chart) = body.find(ns::OFFICE, "chart").and_then(|c| c.find(ns::ODF_CHART, "chart"))
    {
        return odf_chart(chart);
    }
    if body.find(ns::OFFICE, "spreadsheet").is_some() {
        return crate::formats::odf::parse(package).map(|doc| doc.blocks);
    }
    Ok(Vec::new())
}

/// A chart document's title, then the table it plots from (`chart:chart`'s
/// `table:table`: series names across the first row, categories down the
/// first column), the first row as the header, as the ODF reader reads a
/// chart embedded in a text or presentation document.
fn odf_chart(chart: &Element) -> Result<Vec<Block>, ConvertError> {
    let mut blocks = Vec::new();
    if let Some(title) = chart.find(ns::ODF_CHART, "title") {
        let text = paragraphs(title);
        if !text.is_empty() {
            blocks.push(Block::Paragraph(vec![Inline::Text { text, style: Style::PLAIN }]));
        }
    }
    if let Some(table) = chart.find(ns::TABLE, "table") {
        let mut rows = Rows::default();
        rows.read(table)?;
        let mut table = Table::from_rows(rows.cells, 0, TableKind::Data);
        if !table.grid.is_empty() {
            table.header_rows = 1;
            blocks.push(Block::Table(table));
        }
    }
    Ok(blocks)
}

/// The text of an element's `text:p` paragraphs, whitespace collapsed and
/// joined by spaces.
fn paragraphs(element: &Element) -> String {
    element
        .find_all(ns::TEXT, "p")
        .map(|p| collapse_ws(&clean_text(&p.text())).trim().to_string())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A chart table's rows. Repeated cells and rows expand only up to the
/// last one with content, so a row padded to the sheet's width costs
/// nothing; what does expand is charged against `MAX_GRID_SLOTS`.
#[derive(Default)]
struct Rows {
    cells: Vec<Vec<Cell>>,
    /// Empty rows waiting for a row with content after them.
    empty_rows: u64,
    slots: u64,
}

impl Rows {
    fn read(&mut self, container: &Element) -> Result<(), ConvertError> {
        for child in container.child_elems() {
            if child.ns.as_deref() != Some(ns::TABLE) {
                continue;
            }
            match child.local.as_str() {
                "table-header-rows" | "table-rows" | "table-row-group" => self.read(child)?,
                "table-row" => self.row(child)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn row(&mut self, row: &Element) -> Result<(), ConvertError> {
        let mut cells = Vec::new();
        let mut empty_cells = 0u64;
        for cell in row.child_elems() {
            if !(cell.is(ns::TABLE, "table-cell") || cell.is(ns::TABLE, "covered-table-cell")) {
                continue;
            }
            let repeat = repeated(cell, "number-columns-repeated");
            let text = cell_text(cell);
            if text.is_empty() {
                empty_cells = empty_cells.saturating_add(repeat);
                continue;
            }
            self.charge(empty_cells.saturating_add(repeat))?;
            cells.extend((0..empty_cells).map(|_| Cell::new(Vec::new())));
            empty_cells = 0;
            let content = vec![Block::Paragraph(vec![Inline::plain(text)])];
            cells.extend((0..repeat).map(|_| Cell::new(content.clone())));
        }
        let repeat = repeated(row, "number-rows-repeated");
        if cells.is_empty() {
            self.empty_rows = self.empty_rows.saturating_add(repeat);
            return Ok(());
        }
        let width = cells.len() as u64;
        self.charge(self.empty_rows.saturating_add(repeat).saturating_mul(width))?;
        self.cells.extend((0..self.empty_rows).map(|_| Vec::new()));
        self.empty_rows = 0;
        for _ in 1..repeat {
            self.cells.push(cells.clone());
        }
        self.cells.push(cells);
        Ok(())
    }

    fn charge(&mut self, positions: u64) -> Result<(), ConvertError> {
        self.slots = self.slots.saturating_add(positions);
        if self.slots > limits::MAX_GRID_SLOTS {
            return Err(ConvertError::ResourceLimit {
                limit: "max_grid_slots",
                detail: format!("embedded chart table covers {} positions", self.slots),
            });
        }
        Ok(())
    }
}

/// A repeat count attribute: at least 1.
fn repeated(element: &Element, attribute: &str) -> u64 {
    element.attr(ns::TABLE, attribute).and_then(|n| n.parse::<u64>().ok()).unwrap_or(1).max(1)
}

/// A chart cell's text: its paragraphs, else the value it stores.
fn cell_text(cell: &Element) -> String {
    let text = paragraphs(cell);
    if !text.is_empty() {
        return text;
    }
    ["value", "date-value", "time-value", "boolean-value", "string-value"]
        .iter()
        .find_map(|name| cell.attr(ns::OFFICE, name))
        .map(|value| clean_text(value.trim()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AssetId, LinkTarget, List, ListItem, MarkerKind, inlines_to_plain_text};
    use std::io::Write;

    fn record(ver: u16, instance: u16, rec_type: u16, body: &[u8]) -> Vec<u8> {
        let mut out = (ver | instance << 4).to_le_bytes().to_vec();
        out.extend(rec_type.to_le_bytes());
        out.extend((body.len() as u32).to_le_bytes());
        out.extend(body);
        out
    }

    fn container(rec_type: u16, parts: &[Vec<u8>]) -> Vec<u8> {
        record(0xF, 0, rec_type, &parts.concat())
    }

    /// A shape with a text box of the given text type.
    fn text_shape(tx_type: u32, chars: &str) -> Vec<u8> {
        let units: Vec<u8> = chars.encode_utf16().flat_map(u16::to_le_bytes).collect();
        let text = [record(0, 0, 0x0F9F, &tx_type.to_le_bytes()), record(0, 0, 0x0FA0, &units)];
        container(0xF004, &[record(2, 1, 0xF00A, &[0; 8]), container(0xF00D, &text)])
    }

    /// A shape showing embedded object `id`.
    fn object_shape(id: u32) -> Vec<u8> {
        let data = container(0xF011, &[record(0, 0, 0x0BC1, &id.to_le_bytes())]);
        container(0xF004, &[record(2, 75, 0xF00A, &[0, 0, 0, 0, 0x10, 0x0A, 0, 0]), data])
    }

    fn compound(streams: &[(&str, &[u8])]) -> Vec<u8> {
        let mut ole = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
        for (name, bytes) in streams {
            ole.create_stream(name).unwrap().write_all(bytes).unwrap();
        }
        ole.into_inner().into_inner()
    }

    /// An object's storage record, compressed or not.
    fn storage(bytes: &[u8], compressed: bool) -> Vec<u8> {
        if !compressed {
            return record(0, 0, EX_OLE_OBJ_STG, bytes);
        }
        let mut body = (bytes.len() as u32).to_le_bytes().to_vec();
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), Default::default());
        encoder.write_all(bytes).unwrap();
        body.extend(encoder.finish().unwrap());
        record(0, 1, EX_OLE_OBJ_STG, &body)
    }

    /// A one-slide deck: `shapes` on the slide, and `objects` (exObjId,
    /// persisted record) listed in the ExObjList.
    fn deck(shapes: &[Vec<u8>], objects: &[(u32, Vec<u8>)]) -> Vec<u8> {
        // Persist ids: 1 the document, 2 the slide, 3.. the storages.
        let mut embeds = vec![record(0, 0, 0x040A, &(objects.len() as u32).to_le_bytes())];
        let prog_id: Vec<u8> = "Test.Object".encode_utf16().flat_map(u16::to_le_bytes).collect();
        for (index, (id, _)) in objects.iter().enumerate() {
            let mut atom = [0u8; 24];
            atom[8..12].copy_from_slice(&id.to_le_bytes());
            atom[16..20].copy_from_slice(&(3 + index as u32).to_le_bytes());
            embeds.push(container(
                EX_EMBED,
                &[
                    record(0, 0, 0x0FCD, &[0; 8]),
                    record(1, 0, EX_OLE_OBJ_ATOM, &atom),
                    record(0, 2, CSTRING, &prog_id),
                ],
            ));
        }
        let mut persist = [0u8; 20];
        persist[0] = 2;
        persist[12..16].copy_from_slice(&256u32.to_le_bytes());
        let slide_list = record(0xF, 0, 0x0FF0, &record(0, 0, 0x03F3, &persist));
        let document = container(0x03E8, &[container(0x0409, &embeds), slide_list]);
        let own = [record(1, 0, 0xF009, &[0; 16]), record(2, 0, 0xF00A, &[0; 8])];
        let mut group = vec![container(0xF004, &own)];
        group.extend(shapes.iter().cloned());
        let drawing = container(0x040C, &[container(0xF002, &[container(0xF003, &group)])]);
        let slide = container(0x03EE, &[record(2, 0, 0x03EF, &[0; 24]), drawing]);

        let mut stream = document;
        let mut offsets = vec![0u32, stream.len() as u32];
        stream.extend(slide);
        for (_, persisted) in objects {
            offsets.push(stream.len() as u32);
            stream.extend(persisted);
        }
        let mut directory = (1u32 | (offsets.len() as u32) << 20).to_le_bytes().to_vec();
        directory.extend(offsets.iter().flat_map(|o| o.to_le_bytes()));
        let directory_at = stream.len() as u32;
        stream.extend(record(0, 0, 0x1772, &directory));
        let mut edit = [0u8; 28];
        edit[12..16].copy_from_slice(&directory_at.to_le_bytes());
        edit[16..20].copy_from_slice(&1u32.to_le_bytes());
        let edit_at = stream.len() as u32;
        stream.extend(record(0, 0, 0x0FF5, &edit));
        let mut user = 20u32.to_le_bytes().to_vec();
        user.extend(0xE391_C05Fu32.to_le_bytes());
        user.extend(edit_at.to_le_bytes());
        let user = record(0, 0, 0x0FF6, &user);
        compound(&[("PowerPoint Document", &stream), ("Current User", &user)])
    }

    fn biff(rec_type: u16, body: &[u8]) -> Vec<u8> {
        let mut out = rec_type.to_le_bytes().to_vec();
        out.extend((body.len() as u16).to_le_bytes());
        out.extend(body);
        out
    }

    fn bof(dt: u16) -> Vec<u8> {
        let mut body = vec![0u8; 16];
        body[..2].copy_from_slice(&0x0600u16.to_le_bytes());
        body[2..4].copy_from_slice(&dt.to_le_bytes());
        biff(0x0809, &body)
    }

    /// A BIFF8 Excel worksheet object whose one sheet holds `rows` of text.
    fn worksheet_object(rows: &[&[&str]]) -> Vec<u8> {
        let mut stream = bof(0x0005);
        stream.extend(biff(0x00E0, &[0; 20]));
        let patch = stream.len() + 4;
        stream.extend(biff(0x0085, &[0, 0, 0, 0, 0, 0, 1, 0, b'S']));
        stream.extend(biff(0x000A, &[]));
        let at = (stream.len() as u32).to_le_bytes();
        stream[patch..patch + 4].copy_from_slice(&at);
        stream.extend(bof(0x0010));
        for (row, cells) in rows.iter().enumerate() {
            for (col, text) in cells.iter().enumerate() {
                let mut body: Vec<u8> =
                    [row as u16, col as u16, 0].iter().flat_map(|v| v.to_le_bytes()).collect();
                body.extend((text.len() as u16).to_le_bytes());
                body.push(0);
                body.extend(text.as_bytes());
                stream.extend(biff(0x0204, &body));
            }
        }
        stream.extend(biff(0x000A, &[]));
        compound(&[("\u{1}CompObj", &[0; 28]), ("Workbook", &stream)])
    }

    fn zip(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, body) in parts {
            zip.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    const ODF: &str = concat!(
        r#"xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" "#,
        r#"xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" "#,
        r#"xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" "#,
        r#"xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0" "#,
        r#"xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" "#,
        r#"xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" "#,
        r#"xmlns:xlink="http://www.w3.org/1999/xlink""#
    );

    /// A LibreOffice object: an ODF package in `package_stream`.
    fn odf_object(body: &str) -> Vec<u8> {
        odf_object_with(body, &[])
    }

    /// A LibreOffice object whose package holds further `parts`.
    fn odf_object_with(body: &str, parts: &[(&str, &str)]) -> Vec<u8> {
        let content = format!(
            r#"<?xml version="1.0"?><office:document-content {ODF}><office:body>{body}</office:body></office:document-content>"#
        );
        let mut all = vec![("content.xml", content.as_str())];
        all.extend_from_slice(parts);
        let package = zip(&all);
        compound(&[("\u{1}CompObj", &[0; 28]), ("package_stream", &package)])
    }

    fn chart(table: &str) -> String {
        format!(
            r#"<office:chart><chart:chart><chart:title><text:p>Sales</text:p><text:p>by  quarter</text:p></chart:title><table:table>{table}</table:table></chart:chart></office:chart>"#
        )
    }

    fn row(cells: &str) -> String {
        format!("<table:table-row>{cells}</table:table-row>")
    }

    fn cell(text: &str) -> String {
        format!("<table:table-cell><text:p>{text}</text:p></table:table-cell>")
    }

    fn blocks_of(bytes: &[u8]) -> Vec<Block> {
        super::super::parse(bytes).unwrap().blocks
    }

    fn text(block: &Block) -> String {
        match block {
            Block::Paragraph(inlines) | Block::Heading { content: inlines, .. } => {
                inlines_to_plain_text(inlines)
            }
            other => format!("{other:?}"),
        }
    }

    fn grid(block: &Block) -> Vec<Vec<String>> {
        let Block::Table(table) = block else {
            panic!("not a table: {block:?}");
        };
        table
            .grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|slot| match slot {
                        CellSlot::Origin(cell) => cell.blocks.iter().map(text).collect(),
                        CellSlot::Covered { .. } => "<covered>".to_string(),
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn an_embedded_worksheet_reads_as_its_table_where_its_shape_is() {
        let object = worksheet_object(&[&["Item", "Price"], &["Tea", "3"]]);
        for compressed in [false, true] {
            let bytes = deck(
                &[text_shape(0, "Prices"), object_shape(7), text_shape(4, "After")],
                &[(7, storage(&object, compressed))],
            );
            let blocks = blocks_of(&bytes);
            let [title, table, after] = &blocks[..] else {
                panic!("unexpected blocks: {blocks:?}");
            };
            assert!(matches!(title, Block::Heading { .. }));
            assert_eq!(grid(table), [["Item", "Price"], ["Tea", "3"]]);
            assert_eq!(text(after), "After");
        }
    }

    #[test]
    fn a_libreoffice_chart_reads_as_its_title_and_data_table() {
        // The header row's first cell is empty; a row ends in padding
        // repeated to the sheet's width, a value is stored without display
        // text, and the table ends in a million empty rows.
        let padding = r#"<table:table-cell table:number-columns-repeated="1000000"/>"#;
        let stored = r#"<table:table-cell office:value-type="float" office:value="5.1"/>"#;
        let empty_rows = r#"<table:table-row table:number-rows-repeated="1000000"><table:table-cell/></table:table-row>"#;
        let table = format!(
            "<table:table-header-rows>{}</table:table-header-rows><table:table-rows>{}{}{empty_rows}</table:table-rows>",
            row(&format!("<table:table-cell/>{}{}", cell("North"), cell("South"))),
            row(&format!("{}{}{}{padding}", cell("Q1"), cell("4.5"), cell("3.25"))),
            row(&format!("{}{stored}{}", cell("Q2"), cell("3.9"))),
        );
        let bytes = deck(&[object_shape(1)], &[(1, storage(&odf_object(&chart(&table)), true))]);
        let blocks = blocks_of(&bytes);
        let [title, table] = &blocks[..] else {
            panic!("unexpected blocks: {blocks:?}");
        };
        assert_eq!(text(title), "Sales by quarter");
        assert_eq!(
            grid(table),
            [["", "North", "South"], ["Q1", "4.5", "3.25"], ["Q2", "5.1", "3.9"]]
        );
        let Block::Table(table) = table else { unreachable!() };
        assert_eq!(table.header_rows, 1);
    }

    #[test]
    fn repeated_chart_cells_with_content_are_charged_against_the_grid_budget() {
        let table = row(&format!(
            r#"{}<table:table-cell table:number-columns-repeated="5000000"><text:p>x</text:p></table:table-cell>"#,
            cell("a")
        ));
        let bytes = deck(&[object_shape(1)], &[(1, storage(&odf_object(&chart(&table)), false))]);
        assert!(matches!(
            super::super::parse(&bytes),
            Err(ConvertError::ResourceLimit { limit: "max_grid_slots", .. })
        ));
    }

    #[test]
    fn spreadsheet_objects_read_as_their_tables() {
        // A LibreOffice Calc object.
        let calc = odf_object(&format!(
            r#"<office:spreadsheet><table:table table:name="S">{}{}</table:table></office:spreadsheet>"#,
            row(&format!("{}{}", cell("Name"), cell("Qty"))),
            row(&format!("{}{}", cell("Pen"), cell("2")))
        ));
        // An Excel 2007 object: an OOXML workbook in a `Package` stream.
        const SML: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
        const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
        const RELS: &str = "http://schemas.openxmlformats.org/package/2006/relationships";
        let workbook = format!(
            r#"<workbook xmlns="{SML}" xmlns:r="{R}"><sheets><sheet name="A" sheetId="1" r:id="rId1"/></sheets></workbook>"#
        );
        let rels = format!(
            r#"<Relationships xmlns="{RELS}"><Relationship Id="rId1" Type="{R}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#
        );
        let sheet = format!(
            r#"<worksheet xmlns="{SML}"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Size</t></is></c><c r="B1"><v>12</v></c></row></sheetData></worksheet>"#
        );
        let xlsx = zip(&[
            ("xl/workbook.xml", &workbook),
            ("xl/_rels/workbook.xml.rels", &rels),
            ("xl/worksheets/sheet1.xml", &sheet),
        ]);
        let excel = compound(&[("\u{1}CompObj", &[0; 28]), ("Package", &xlsx)]);
        let bytes = deck(
            &[object_shape(1), object_shape(2)],
            &[(1, storage(&calc, true)), (2, storage(&excel, false))],
        );
        let blocks = blocks_of(&bytes);
        let [calc, excel] = &blocks[..] else {
            panic!("unexpected blocks: {blocks:?}");
        };
        assert_eq!(grid(calc), [["Name", "Qty"], ["Pen", "2"]]);
        assert_eq!(grid(excel), [["Size", "12"]]);
    }

    #[test]
    fn objects_without_readable_data_add_nothing() {
        let equation = compound(&[("\u{1}CompObj", &[0; 28]), ("Equation Native", &[0; 40])]);
        let writer = odf_object("<office:text><text:p>Not data</text:p></office:text>");
        let mut corrupt = storage(&worksheet_object(&[&["x"]]), true);
        let end = corrupt.len();
        corrupt[end - 20..].fill(0xAB);
        let shapes = [
            text_shape(0, "Title"),
            object_shape(1),
            object_shape(2),
            object_shape(3),
            object_shape(4),
            object_shape(5),
            // No object 9 is listed.
            object_shape(9),
        ];
        let objects = [
            (1, storage(&equation, false)),
            (2, storage(&writer, true)),
            (3, corrupt),
            (4, storage(b"not a compound file", false)),
            // Listed, but its persisted record is not a storage.
            (5, record(0, 0, 0x0FBA, &[0; 4])),
        ];
        let blocks = blocks_of(&deck(&shapes, &objects));
        assert_eq!(blocks.iter().map(text).collect::<Vec<_>>(), ["Title"]);
    }

    #[test]
    fn an_object_shown_by_two_shapes_is_written_at_both() {
        let object = worksheet_object(&[&["One"]]);
        let bytes = deck(&[object_shape(3), object_shape(3)], &[(3, storage(&object, true))]);
        let blocks = blocks_of(&bytes);
        assert_eq!(blocks.iter().map(grid).collect::<Vec<_>>(), [[["One"]], [["One"]]]);
    }

    #[test]
    fn every_showing_of_an_object_counts_against_the_grid_budget() {
        let cell = || Cell::from_inlines(vec![Inline::plain("x")]);
        let table = Block::Table(Table::from_rows(vec![vec![cell(), cell()]], 0, TableKind::Data));
        let mut objects = Objects::default();
        objects.decoded.insert(1, vec![table]);
        objects.written = limits::MAX_GRID_SLOTS - 4;
        assert_eq!(objects.blocks(1).unwrap().len(), 1);
        assert_eq!(objects.blocks(1).unwrap().len(), 1);
        assert!(matches!(
            objects.blocks(1),
            Err(ConvertError::ResourceLimit { limit: "max_grid_slots", .. })
        ));
    }

    #[test]
    fn decompression_stops_at_the_cap() {
        let persisted = storage(&[0; 4096], true);
        let (.., body) = record_at(&persisted, 0).unwrap();
        assert_eq!(inflate(body, 4096).unwrap().map(|bytes| bytes.len()), Some(4096));
        assert!(matches!(
            inflate(body, 1024),
            Err(ConvertError::ResourceLimit { limit: "max_entry_bytes", .. })
        ));
    }

    #[test]
    fn an_objects_own_images_and_notes_stay_with_it() {
        // A Calc object whose cell holds an image: the image is the
        // object's asset, not one of the deck's pictures.
        let calc = odf_object_with(
            &format!(
                r#"<office:spreadsheet><table:table table:name="S"><table:table-row><table:table-cell><text:p>Logo<draw:frame><draw:image xlink:href="Pictures/logo.png"/><svg:title>Company logo</svg:title></draw:frame></text:p></table:table-cell>{}</table:table-row></table:table></office:spreadsheet>"#,
                cell("2")
            ),
            &[("Pictures/logo.png", "not really a png")],
        );
        let blocks = blocks_of(&deck(&[object_shape(1)], &[(1, storage(&calc, false))]));
        let [Block::Table(table)] = &blocks[..] else {
            panic!("unexpected blocks: {blocks:?}");
        };
        let CellSlot::Origin(first) = &table.grid[0][0] else { unreachable!() };
        let [Block::Paragraph(inlines)] = &first.blocks[..] else {
            panic!("unexpected cell: {first:?}");
        };
        assert!(
            inlines.iter().any(|inline| matches!(
                inline,
                Inline::Image { alt, source: ImageSource::Unavailable } if alt == "Company logo"
            )),
            "{inlines:?}"
        );

        // Every place a block can hold inlines is reached.
        let image = Inline::Image { alt: "a".into(), source: ImageSource::Asset(AssetId(0)) };
        let link = Inline::Link {
            content: vec![image.clone()],
            target: LinkTarget::External("https://example.com".into()),
        };
        let item = ListItem {
            blocks: vec![Block::Paragraph(vec![link, Inline::NoteRef("1".into())])],
            marker_label: None,
        };
        let list = Block::List(List { marker: MarkerKind::Bullet, start: 1, items: vec![item] });
        let table = Table::from_rows(
            vec![vec![Cell::from_inlines(vec![image.clone()])]],
            0,
            TableKind::Data,
        );
        let mut blocks = vec![
            Block::BlockQuote(vec![list]),
            Block::Table(table),
            Block::heading(2, vec![image, Inline::NoteRef("2".into())]),
        ];
        detach(&mut blocks);
        let text = format!("{blocks:?}");
        assert!(!text.contains("Asset(") && !text.contains("NoteRef"), "{text}");
        assert_eq!(text.matches("Unavailable").count(), 3, "{text}");
    }

    #[test]
    fn sheet_names_stay_as_paragraphs_only_between_several_tables() {
        let table = || {
            Block::Table(Table::from_rows(vec![vec![Cell::new(Vec::new())]], 0, TableKind::Data))
        };
        let heading = |name: &str| Block::heading(2, vec![Inline::plain(name)]);
        let one = sheet_names_as_paragraphs(vec![heading("A"), table()]);
        assert!(matches!(&one[..], [Block::Table(_)]), "{one:?}");
        let two = sheet_names_as_paragraphs(vec![heading("A"), table(), heading("B"), table()]);
        assert!(
            matches!(
                &two[..],
                [Block::Paragraph(_), Block::Table(_), Block::Paragraph(_), Block::Table(_)]
            ),
            "{two:?}"
        );
    }
}
