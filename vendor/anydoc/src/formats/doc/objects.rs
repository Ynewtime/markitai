//! Stored data of a legacy Word OLE field. Identity comes only from its
//! separator's character properties; ObjectPool orphans are never read.

use super::TextStream;
use crate::error::ConvertError;
use crate::model::{Block, CellSlot, ImageSource, Inline, LinkTarget};
use crate::package::limits;
use crate::shared::binary::{get_u32, read_ole_stream};
use std::collections::HashMap;
use std::io::Cursor;

const MAX_OBJECTS: usize = 1024;
const MAX_REFERENCES: u64 = 100_000;

#[derive(Clone, Copy, Default)]
pub(super) struct Field {
    pub(super) kind: Option<u8>,
    pub(super) end_flags: Option<u8>,
    pub(super) malformed_metadata: bool,
    pub(super) missing_metadata: bool,
}

/// New stored-data reads use whole-story field visibility, independently of
/// the paragraph renderer's local state. Sorted disjoint intervals make both
/// separator and floating-anchor checks logarithmic, including across CRs.
#[derive(Default)]
pub(super) struct Fields {
    pub(super) at_separator: HashMap<usize, Field>,
    blocked: Vec<(usize, usize)>,
}

impl Fields {
    #[cfg(test)]
    pub(super) fn from_separators(at_separator: HashMap<usize, Field>) -> Self {
        Self { at_separator, ..Default::default() }
    }

    pub(super) fn exposes(&self, position: usize) -> bool {
        let next = self.blocked.partition_point(|&(_, end)| end <= position);
        self.blocked.get(next).is_none_or(|&(start, _)| position < start)
    }
}

struct Span {
    begin: usize,
    separator: Option<usize>,
    end: Option<usize>,
    nested: bool,
    repeated_separator: bool,
}

/// Pair fields for the entire document part, with UTF-16 CPs from its PLC.
/// Missing entries can be recovered; known contradictions cannot. A private
/// or instruction ancestor continues to suppress data across paragraph marks.
pub(super) fn fields(
    word: &[u8],
    table: &[u8],
    text: &TextStream,
    parts: &[(usize, usize, usize)],
) -> Result<Fields, ConvertError> {
    let mut result = Fields::default();
    let mut total_fields = 0u64;
    let mut indexed_chars = 0u64;
    for &(fib_offset, base, count) in parts {
        let mut metadata = HashMap::new();
        let mut malformed = false;
        let offset = get_u32(word, fib_offset).unwrap_or(0) as usize;
        let size = get_u32(word, fib_offset + 4).unwrap_or(0) as usize;
        if size > 0 {
            if let Some(plc) = table.get(offset..).and_then(|part| part.get(..size))
                && size >= 4
                && (size - 4).is_multiple_of(6)
            {
                let entries = (size - 4) / 6;
                indexed_chars += entries as u64;
                if indexed_chars > MAX_REFERENCES * 3 {
                    return Err(ConvertError::ResourceLimit {
                        limit: "max_field_count",
                        detail: "Word field index exceeds 300,000 characters".into(),
                    });
                }
                let mut previous = None;
                for n in 0..entries {
                    let cp = get_u32(plc, 4 * n).unwrap() as usize;
                    let at = text.index_of_cp(base.saturating_add(cp));
                    let character = plc[4 * (entries + 1) + 2 * n] & 0x1F;
                    let flags = plc[4 * (entries + 1) + 2 * n + 1];
                    if cp >= count
                        || previous.is_some_and(|old| cp <= old)
                        || text.cps.get(at).copied().map(|cp| cp as usize) != base.checked_add(cp)
                        || !matches!(character, 0x13..=0x15)
                        || text.chars.get(at).copied().map(|c| c as u32) != Some(character as u32)
                    {
                        malformed = true;
                        break;
                    }
                    metadata.insert(at, flags);
                    previous = Some(cp);
                }
                // The final CP is a sorting sentinel, not a character index.
                // Its other meaning/value is deliberately not interpreted.
                let last = get_u32(plc, 4 * entries).unwrap() as usize;
                malformed |= previous.is_some_and(|old| last <= old);
            } else {
                malformed = true;
            }
        }
        let lo = text.index_of_cp(base);
        let hi = text.index_of_cp(base.saturating_add(count)).min(text.chars.len());
        let mut stack: Vec<usize> = Vec::new();
        let mut spans: Vec<Span> = Vec::new();
        for at in lo..hi {
            match text.chars[at] {
                '\u{13}' => {
                    if stack.len() >= limits::MAX_XML_DEPTH {
                        return Err(ConvertError::ResourceLimit {
                            limit: "max_field_depth",
                            detail: "nested Word fields exceed 256".into(),
                        });
                    }
                    total_fields += 1;
                    if total_fields > MAX_REFERENCES {
                        return Err(ConvertError::ResourceLimit {
                            limit: "max_field_count",
                            detail: "Word fields exceed 100,000".into(),
                        });
                    }
                    let index = spans.len();
                    spans.push(Span {
                        begin: at,
                        separator: None,
                        end: None,
                        nested: !stack.is_empty(),
                        repeated_separator: false,
                    });
                    stack.push(index);
                }
                '\u{14}' => {
                    if let Some(&index) = stack.last() {
                        let span = &mut spans[index];
                        if span.separator.is_some() {
                            span.repeated_separator = true;
                        } else {
                            span.separator = Some(at);
                        }
                    }
                }
                '\u{15}' => {
                    if let Some(index) = stack.pop() {
                        spans[index].end = Some(at);
                    }
                }
                _ => {}
            }
        }
        for span in spans {
            let Some(end) = span.end else {
                result.blocked.push((span.begin, hi));
                continue;
            };
            let flags = metadata.get(&end).copied();
            let conflicting_flags = flags.is_some_and(|flags| {
                (flags & 0x80 != 0) != span.separator.is_some()
                    || (flags & 0x40 != 0) != span.nested
            });
            let bad = malformed || span.repeated_separator || conflicting_flags;
            if bad || flags.is_some_and(|flags| flags & 0x22 != 0) {
                result.blocked.push((span.begin, end + 1));
            }
            // The separator belongs to the result, so it is not included in
            // the instruction interval. A field without one is all instruction.
            result.blocked.push((span.begin, span.separator.unwrap_or(end)));
            if let Some(separator) = span.separator
                && !span.repeated_separator
            {
                result.at_separator.insert(
                    separator,
                    Field {
                        kind: metadata.get(&span.begin).copied(),
                        end_flags: flags,
                        malformed_metadata: bad,
                        missing_metadata: !metadata.contains_key(&span.begin)
                            || !metadata.contains_key(&separator)
                            || !metadata.contains_key(&end),
                    },
                );
            }
        }
    }
    result.blocked.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in result.blocked {
        if start >= end {
            continue;
        }
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    result.blocked = merged;
    Ok(result)
}

/// Only single-chain textboxes with an unambiguous live main-story anchor
/// are recovered. Spare ranges, header stories and unbound storage stay out.
#[derive(Default)]
pub(super) struct Textboxes {
    pub(super) ranges: HashMap<usize, (usize, usize)>,
    pub(super) warnings: Vec<String>,
}

impl Textboxes {
    pub(super) fn read(
        word: &[u8],
        table: &[u8],
        text: &TextStream,
        main_count: usize,
        textbox_base: usize,
        textbox_count: usize,
    ) -> Result<Self, ConvertError> {
        let mut output = Self::default();
        if textbox_count == 0 {
            return Ok(output);
        }
        let Some(boxes) = textbox_plc(word, table, 0x25A, 22)? else {
            output.reject("textbox ranges are missing or malformed");
            return Ok(output);
        };
        let Some(anchors) = textbox_plc(word, table, 0x1DA, 26)? else {
            output.reject("main shape anchors are missing or malformed");
            return Ok(output);
        };
        let Some(shapes) = main_shapes(word, table)? else {
            output.reject("main drawing shape identities are missing or malformed");
            return Ok(output);
        };
        let box_count = (boxes.len() - 4) / 26;
        let anchor_count = (anchors.len() - 4) / 30;
        if box_count > MAX_OBJECTS {
            return Err(ConvertError::ResourceLimit {
                limit: "max_textboxes",
                detail: "Word textboxes exceed 1,024 ranges".into(),
            });
        }
        // The last FTXBXS range is the ignored spare; it is never exposed.
        if box_count < 2
            || !increasing_cps(boxes, box_count, Some(textbox_count))
            || !increasing_cps(anchors, anchor_count, None)
        {
            output.reject("textbox or main shape CP boundaries are inconsistent");
            return Ok(output);
        }
        let mut by_shape = HashMap::new();
        for n in 0..anchor_count {
            let cp = get_u32(anchors, 4 * n).unwrap() as usize;
            let lid = get_u32(anchors, 4 * (anchor_count + 1) + 26 * n).unwrap();
            let index = text.index_of_cp(cp);
            let valid = cp < main_count
                && text.cps.get(index).copied() == Some(cp as u32)
                && text.chars.get(index) == Some(&'\u{8}');
            // A repeated lid has no unique anchor, even if only one of those
            // CPs is otherwise valid. Never repair by choosing the first one.
            if by_shape.insert(lid, valid.then_some(index)).is_some() {
                by_shape.insert(lid, None);
            }
        }
        let mut claimed = std::collections::HashSet::new();
        let mut duplicates = std::collections::HashSet::new();
        for n in 0..box_count - 1 {
            let row = 4 * (box_count + 1) + 22 * n;
            let lid = get_u32(boxes, row + 14).unwrap();
            if !claimed.insert(lid) {
                duplicates.insert(lid);
            }
        }
        for n in 0..box_count - 1 {
            let row = 4 * (box_count + 1) + 22 * n;
            let lid = get_u32(boxes, row + 14).unwrap();
            let reusable = u16::from_le_bytes(boxes[row + 8..row + 10].try_into().unwrap());
            let chain_count = get_u32(boxes, row).unwrap();
            let relative_lo = get_u32(boxes, 4 * n).unwrap() as usize;
            let relative_hi = get_u32(boxes, 4 * (n + 1)).unwrap() as usize;
            let lo_cp = textbox_base.saturating_add(relative_lo);
            let hi_cp = textbox_base.saturating_add(relative_hi);
            let lo = text.index_of_cp(lo_cp);
            let hi = text.index_of_cp(hi_cp);
            let Some(&Some(anchor)) = by_shape.get(&lid) else {
                output.reject("textbox has no unique main shape anchor; its data was not read");
                continue;
            };
            if reusable != 0
                || chain_count != 1
                || duplicates.contains(&lid)
                || shapes.get(&lid) != Some(&true)
                || text.cps.get(lo).copied().map(|cp| cp as usize) != Some(lo_cp)
                || hi > text.chars.len()
                || lo >= hi
                || text.cps.get(hi).copied().map(|cp| cp as usize).unwrap_or(usize::MAX) != hi_cp
                || text.chars.get(hi - 1) != Some(&'\r')
            {
                output.reject("textbox identity, range or chain is unsupported or inconsistent; its data was not read");
                continue;
            }
            output.ranges.insert(anchor, (lo, hi));
        }
        Ok(output)
    }

    fn reject(&mut self, why: &str) {
        self.warnings.push(format!("Word textbox: {why}."));
    }
}

fn textbox_plc<'a>(
    word: &[u8],
    table: &'a [u8],
    fib: usize,
    record: usize,
) -> Result<Option<&'a [u8]>, ConvertError> {
    let offset = get_u32(word, fib).unwrap_or(0) as usize;
    let size = get_u32(word, fib + 4).unwrap_or(0) as usize;
    if size < 4 || !(size - 4).is_multiple_of(record + 4) {
        return Ok(None);
    }
    if (size - 4) / (record + 4) > MAX_REFERENCES as usize {
        return Err(ConvertError::ResourceLimit {
            limit: "max_shape_records",
            detail: "Word shape indexes exceed 100,000 records".into(),
        });
    }
    Ok(table.get(offset..).and_then(|tail| tail.get(..size)))
}

fn increasing_cps(data: &[u8], count: usize, bound: Option<usize>) -> bool {
    let mut previous = None;
    for n in 0..=count {
        let cp = get_u32(data, 4 * n).unwrap() as usize;
        if previous.is_some_and(|old| cp <= old) || bound.is_some_and(|bound| cp > bound) {
            return false;
        }
        previous = Some(cp);
    }
    true
}

fn main_shapes(word: &[u8], table: &[u8]) -> Result<Option<HashMap<u32, bool>>, ConvertError> {
    use crate::shared::officeart::record_at;
    let offset = get_u32(word, 0x22A).unwrap_or(0) as usize;
    let length = get_u32(word, 0x22E).unwrap_or(0) as usize;
    let Some(data) = table.get(offset..).and_then(|tail| tail.get(..length)) else {
        return Ok(None);
    };
    let Some((version, 0xF000, group)) = record_at(data, 0) else {
        return Ok(None);
    };
    if version & 0xF != 0xF {
        return Ok(None);
    }
    let mut cursor = 8 + group.len();
    let mut main = None;
    let mut drawing_count = 0;
    while cursor < data.len() {
        let selector = data[cursor];
        cursor += 1;
        let Some((version, 0xF002, body)) = record_at(data, cursor) else {
            return Ok(None);
        };
        if version & 0xF != 0xF || selector > 1 {
            return Ok(None);
        }
        drawing_count += 1;
        if drawing_count > 2 {
            return Ok(None);
        }
        if selector == 0 && main.replace(body).is_some() {
            return Ok(None);
        }
        cursor += 8 + body.len();
    }
    let Some(main) = main else {
        return Ok(None);
    };
    let mut shapes = HashMap::new();
    let mut visited = 0;
    if !drawing_shapes(main, 0, &mut visited, &mut shapes)? {
        return Ok(None);
    }
    Ok(Some(shapes))
}

fn shape_identity(data: &[u8], visited: &mut usize) -> Result<Option<(u32, u32)>, ConvertError> {
    use crate::shared::officeart::record_at;
    let mut cursor = 0;
    let mut identity = None;
    while cursor < data.len() {
        count_drawing_record(visited)?;
        let Some((version, kind, body)) = record_at(data, cursor) else {
            return Ok(None);
        };
        if kind == 0xF00A {
            if version & 0xF != 2 || body.len() != 8 || identity.is_some() {
                return Ok(None);
            }
            identity = Some((get_u32(body, 0).unwrap(), get_u32(body, 4).unwrap()));
        }
        cursor += 8 + body.len();
    }
    Ok(identity)
}

fn count_drawing_record(visited: &mut usize) -> Result<(), ConvertError> {
    *visited += 1;
    if *visited > 10_000 {
        return Err(ConvertError::ResourceLimit {
            limit: "max_drawing_records",
            detail: "Word drawings exceed 10,000 inspected records".into(),
        });
    }
    Ok(())
}

fn drawing_shapes(
    data: &[u8],
    depth: usize,
    visited: &mut usize,
    shapes: &mut HashMap<u32, bool>,
) -> Result<bool, ConvertError> {
    use crate::shared::officeart::record_at;
    if depth > 16 {
        return Err(ConvertError::ResourceLimit {
            limit: "max_drawing_depth",
            detail: "Word drawings exceed 16 container levels".into(),
        });
    }
    let mut cursor = 0;
    while cursor < data.len() {
        count_drawing_record(visited)?;
        let Some((version, kind, body)) = record_at(data, cursor) else {
            return Ok(false);
        };
        match kind {
            0xF004 if version & 0xF == 0xF => {
                let Some((id, flags)) = shape_identity(body, visited)? else {
                    return Ok(false);
                };
                // OLE is not required: a live ordinary textbox is also text.
                let live = flags & 0x8 == 0 && flags & 0x200 != 0;
                if shapes.insert(id, live).is_some() {
                    shapes.insert(id, false);
                }
            }
            0xF003 if version & 0xF == 0xF => {
                // The drawing's patriarch group is supported. Chained or
                // nested non-patriarch group geometry is not guessed.
                let Some((_, 0xF004, group)) = record_at(body, 0) else {
                    return Ok(false);
                };
                let Some((_, flags)) = shape_identity(group, visited)? else {
                    return Ok(false);
                };
                if flags & 0xD == 0x5 && !drawing_shapes(body, depth + 1, visited, shapes)? {
                    return Ok(false);
                }
            }
            _ => {}
        }
        cursor += 8 + body.len();
    }
    Ok(true)
}

pub(super) struct Objects<'a> {
    ole: cfb::CompoundFile<Cursor<&'a [u8]>>,
    decoded: HashMap<i32, Vec<Block>>,
    package_bytes: u64,
    cached: Cost,
    written: Cost,
    references: u64,
    pub(super) warnings: Vec<String>,
}

impl<'a> Objects<'a> {
    pub(super) fn new(ole: cfb::CompoundFile<Cursor<&'a [u8]>>) -> Self {
        Self {
            ole,
            decoded: HashMap::new(),
            package_bytes: 0,
            cached: Cost::default(),
            written: Cost::default(),
            references: 0,
            warnings: Vec::new(),
        }
    }

    pub(super) fn blocks(
        &mut self,
        id: i32,
        instruction: &str,
        field: Field,
        cp: u32,
    ) -> Result<Vec<Block>, ConvertError> {
        let kind = instruction.split_whitespace().next().unwrap_or("");
        let expected = if kind.eq_ignore_ascii_case("EMBED") {
            0x3A
        } else if kind.eq_ignore_ascii_case("LINK") {
            0x38
        } else if kind.eq_ignore_ascii_case("CONTROL") {
            0x57
        } else {
            return Ok(Vec::new());
        };
        self.references += 1;
        if self.references > MAX_REFERENCES {
            return Err(ConvertError::ResourceLimit {
                limit: "max_object_references",
                detail: "Word OLE references exceed 100,000".into(),
            });
        }
        if field.malformed_metadata || field.kind.is_some_and(|kind| kind != expected) {
            self.warn(
                cp,
                "field index is malformed or disagrees with its object type; data not read",
            );
            return Ok(Vec::new());
        }
        if field.end_flags.is_some_and(|flags| flags & 0x22 != 0) {
            self.warn(cp, "field result is marked zombie or private; stored data not read");
            return Ok(Vec::new());
        }
        if field.kind.is_none() || field.end_flags.is_none() || field.missing_metadata {
            self.warn(cp, "field index omits this object; data recovered from its field text and separator properties");
        }
        if !self.decoded.contains_key(&id) {
            if self.decoded.len() >= MAX_OBJECTS {
                return Err(ConvertError::ResourceLimit {
                    limit: "max_embedded_objects",
                    detail: "Word stored objects exceed 1,024".into(),
                });
            }
            let path = format!("/ObjectPool/_{id}/package_stream");
            let blocks = if self.ole.is_stream(&path) {
                let package = read_ole_stream(&mut self.ole, &path)?;
                self.package_bytes = self.package_bytes.saturating_add(package.len() as u64);
                if self.package_bytes > limits::MAX_TOTAL_BYTES {
                    return Err(ConvertError::ResourceLimit {
                        limit: "max_total_bytes",
                        detail: "Word object packages exceed 512 MiB".into(),
                    });
                }
                match crate::formats::ppt::embedded_object(&package) {
                    Ok(blocks) => blocks,
                    Err(error @ ConvertError::ResourceLimit { .. }) => return Err(error),
                    Err(_) => Vec::new(),
                }
            } else {
                Vec::new()
            };
            if blocks.is_empty() {
                self.warn(cp, "stored object data is missing, unsupported or unreadable; any existing picture is retained");
            }
            self.cached.add(Cost::of(&blocks))?;
            self.decoded.insert(id, blocks);
        }
        let blocks = &self.decoded[&id];
        self.written.add(Cost::of(blocks))?;
        Ok(blocks.clone())
    }

    #[cfg(test)]
    pub(super) fn decoded_count(&self) -> usize {
        self.decoded.len()
    }

    pub(super) fn warn(&mut self, cp: u32, message: &str) {
        self.warnings.push(format!("Word embedded object at character {cp}: {message}."));
    }
}

#[derive(Clone, Copy, Default)]
struct Cost {
    slots: u64,
    text: u64,
}

impl Cost {
    fn of(blocks: &[Block]) -> Self {
        let mut cost = Self::default();
        for block in blocks {
            cost.slots += 1;
            match block {
                Block::Table(table) => {
                    cost.slots += table.grid.iter().map(|row| row.len() as u64).sum::<u64>();
                    for slot in table.grid.iter().flatten() {
                        if let CellSlot::Origin(cell) = slot {
                            cost.sum(Self::of(&cell.blocks));
                        }
                    }
                }
                Block::Paragraph(text) => cost.sum(inline_cost(text)),
                Block::Heading { content, anchor, .. } => {
                    cost.sum(inline_cost(content));
                    cost.text += anchor.as_ref().map_or(0, |s| s.len() as u64);
                }
                Block::CodeBlock { text, lang } => {
                    cost.text += text.len() as u64 + lang.as_ref().map_or(0, |s| s.len() as u64);
                }
                Block::Math(text) => cost.text += text.len() as u64,
                Block::BlockQuote(blocks) => cost.sum(Self::of(blocks)),
                Block::List(list) => {
                    for item in &list.items {
                        cost.sum(Self::of(&item.blocks));
                        cost.text += item.marker_label.as_ref().map_or(0, |s| s.len() as u64);
                    }
                }
                Block::Rule => {}
            }
        }
        cost
    }

    fn sum(&mut self, other: Self) {
        self.slots = self.slots.saturating_add(other.slots);
        self.text = self.text.saturating_add(other.text);
    }

    fn add(&mut self, other: Self) -> Result<(), ConvertError> {
        self.sum(other);
        if self.slots > limits::MAX_GRID_SLOTS {
            return Err(ConvertError::ResourceLimit {
                limit: "max_grid_slots",
                detail: "Word embedded objects exceed the table/block budget".into(),
            });
        }
        if self.text > limits::MAX_EXPANSION_TEXT_BYTES {
            return Err(ConvertError::ResourceLimit {
                limit: "max_expansion_text_bytes",
                detail: "Word embedded objects exceed the text budget".into(),
            });
        }
        Ok(())
    }
}

fn inline_cost(inlines: &[Inline]) -> Cost {
    let mut cost = Cost { slots: inlines.len() as u64, text: 0 };
    for inline in inlines {
        match inline {
            Inline::Text { text, .. }
            | Inline::Math(text)
            | Inline::NoteRef(text)
            | Inline::Anchor(text) => cost.text += text.len() as u64,
            Inline::Link { content, target } => {
                cost.sum(inline_cost(content));
                let (LinkTarget::External(target)
                | LinkTarget::Relative(target)
                | LinkTarget::Anchor(target)) = target;
                cost.text += target.len() as u64;
            }
            Inline::Image { alt, source } => {
                cost.text += alt.len() as u64;
                if let ImageSource::External(url) = source {
                    cost.text += url.len() as u64;
                }
            }
            Inline::LineBreak | Inline::Checkbox(_) => {}
        }
    }
    cost
}
