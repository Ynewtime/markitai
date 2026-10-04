//! Legacy PowerPoint 97-2003 binary (.ppt): OLE2 container, record stream.
//! Slides resolve through the persist directory (the only default path);
//! text comes from TextHeaderAtom + TextCharsAtom/TextBytesAtom with
//! StyleTextPropAtom runs and TxMasterStyleAtom defaults applied. Raw
//! stream-order scanning exists only as an explicitly labelled recovery for
//! files whose persist directory is unusable. Speaker notes are included
//! (fixed policy), rendered as a quote after their slide.

// markitai: embedded objects read as their data.
mod ole;
pub(crate) use ole::object_file as embedded_object;
pub(crate) mod pictures;
mod styletext;

use crate::error::ConvertError;
use crate::model::{
    Block, Cell, Document, GridBuilder, Inline, Style, Table, TableKind, inlines_are_empty,
};
use crate::package::limits;
use crate::shared::binary::{get_u32, read_ole_stream, utf16le_units};
use crate::shared::delta::{StyleDelta, rebase_emphasis};
use crate::shared::header::resolve_header_rows;
use crate::shared::list::{ListEntry, ListKey, MarkerKind, flush_list};
use crate::shared::officeart::record_at;
use crate::shared::text::clean_text;
use std::collections::HashMap;
use std::io::Cursor;
use styletext::{CharProps, MasterLevel, StyleRuns};

/// One master's per-text-type level defaults, keyed by TxMasterStyleAtom
/// instance (the text type).
type MasterStyles = HashMap<u16, Vec<MasterLevel>>;

pub fn parse(bytes: &[u8]) -> Result<Document, ConvertError> {
    let cursor = Cursor::new(bytes);
    let mut ole = cfb::CompoundFile::open(cursor)
        .map_err(|e| ConvertError::malformed(format!("not an OLE2 compound file: {e}")))?;
    let data = read_ole_stream(&mut ole, "PowerPoint Document")?;
    let current_user = read_ole_stream(&mut ole, "Current User").unwrap_or_default();
    if get_u32(&current_user, 12) == Some(0xF3D1_C4DF) {
        return Err(ConvertError::Encrypted);
    }

    let delay = match read_ole_stream(&mut ole, "Pictures") {
        Ok(bytes) => bytes,
        Err(e @ ConvertError::ResourceLimit { .. }) => return Err(e),
        Err(_) => Vec::new(),
    };
    let mut ex = Extractor::default();
    if !ex.parse_slides(&data, &current_user, &delay)? {
        // Labelled recovery path: the persist directory is unusable, so text
        // is taken in raw stream order (may include superseded edits).
        log::warn!("ppt persist directory unusable; recovering text in raw stream order");
        ex = Extractor::default();
        ex.recovering = true;
        ex.walk(&data)?;
        ex.end_segment(None);
    }
    if ex.encrypted {
        return Err(ConvertError::Encrypted);
    }
    let assets = std::mem::take(&mut ex.pictures.assets.assets);
    let warnings = std::mem::take(&mut ex.pictures.warnings);
    let (blocks, slide_starts) = ex.into_blocks();
    Ok(Document { blocks, notes: Vec::new(), assets, slide_starts, warnings })
}

/// Iterate the records laid out back to back in `data`.
fn children(data: &[u8]) -> impl Iterator<Item = (u16, u16, &[u8])> {
    let mut pos = 0;
    std::iter::from_fn(move || {
        let (ver_inst, rec_type, body) = record_at(data, pos)?;
        pos += 8 + body.len();
        Some((ver_inst, rec_type, body))
    })
}

/// A text shape being accumulated: header type, text, then styling.
struct PendingShape {
    tx_type: u8,
    text: String,
    styles: Option<StyleRuns>,
}

// markitai: segments record what they belong to, so slide boundaries reach
// `Document::slide_starts`.
/// What a finished segment of text belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SegmentKind {
    /// One slide of the slide list, kept even when it holds no text.
    Slide,
    Notes,
    /// Text that no slide of the slide list owns (raw-order recovery).
    Loose,
}

#[derive(Default)]
struct Extractor<'a> {
    /// Finished segments: (blocks, pairing id, kind).
    segments: Vec<(Vec<Block>, Option<u32>, SegmentKind)>,
    current: Vec<Block>,
    current_is_notes: bool,
    list_run: Vec<ListEntry>,
    pending: Option<PendingShape>,
    /// Master style tables in master-list order: (masterId, styles).
    masters: Vec<(u32, MasterStyles)>,
    /// Index into `masters` for the slide being extracted (0 fallback).
    active_master: usize,
    shape_counter: u64,
    encrypted: bool,
    /// Raw-stream recovery: no persist lists, so notes descend inline.
    recovering: bool,
    /// Records visited across the whole extraction, capped.
    records: u64,
    /// markitai: the cells of a table group are being read.
    in_table: bool,
    /// markitai: the embedded objects shapes may show.
    objects: ole::Objects<'a>,
    pictures: pictures::Bank<'a>,
}

/// The persist-resolved layout of the presentation: slide/notes lists from
/// the current DocumentContainer, persist id -> offset for every container.
struct DocLayout<'a> {
    persist: HashMap<u32, usize>,
    slide_list: &'a [u8],
    notes_list: Option<&'a [u8]>,
    master_list: Option<&'a [u8]>,
    /// markitai: the ExObjList, when the document has one.
    objects: Option<&'a [u8]>,
    drawings: Option<&'a [u8]>,
    drawings_conflict: bool,
}

/// Resolve the UserEditAtom chain into the persist directory and find the
/// DocumentContainer's SlideListWithText instances. `None` means the persist
/// directory is unusable and the caller falls back to raw-order recovery.
fn locate_document<'a>(data: &'a [u8], current_user: &[u8]) -> Option<DocLayout<'a>> {
    let mut persist: HashMap<u32, usize> = HashMap::new();
    let mut doc_persist: Option<u32> = None;
    let mut edit_off = get_u32(current_user, 16)? as usize;
    for _ in 0..100 {
        if edit_off == 0 {
            break;
        }
        let (_, rec_type, body) = record_at(data, edit_off)?;
        if rec_type != 0x0FF5 {
            return None;
        }
        if doc_persist.is_none() {
            doc_persist = get_u32(body, 16);
        }
        let dir_off = get_u32(body, 12)? as usize;
        if let Some((_, 0x1772, dir)) = record_at(data, dir_off) {
            let mut pos = 0;
            while pos + 4 <= dir.len() {
                let head = get_u32(dir, pos)?;
                let id = head & 0xF_FFFF;
                let count = (head >> 20) as usize;
                pos += 4;
                for k in 0..count {
                    // Newer edits win: keep the first offset seen.
                    persist.entry(id + k as u32).or_insert(get_u32(dir, pos)? as usize);
                    pos += 4;
                }
            }
        }
        let prev = get_u32(body, 8)? as usize;
        if prev == edit_off {
            break;
        }
        edit_off = prev;
    }

    let doc_off = *persist.get(&doc_persist?)?;
    let Some((_, 0x03E8, doc)) = record_at(data, doc_off) else {
        return None;
    };
    let (.., slide_list) =
        children(doc).find(|&(ver_inst, rec_type, _)| rec_type == 0x0FF0 && ver_inst >> 4 == 0)?;
    let notes_list = children(doc)
        .find(|&(ver_inst, rec_type, _)| rec_type == 0x0FF0 && ver_inst >> 4 == 2)
        .map(|(.., body)| body);
    let master_list = children(doc)
        .find(|&(ver_inst, rec_type, _)| rec_type == 0x0FF0 && ver_inst >> 4 == 1)
        .map(|(.., body)| body);
    let objects = children(doc).find(|&(_, rec_type, _)| rec_type == 0x0409).map(|(.., body)| body);
    let mut drawing_groups = children(doc).filter(|&(_, t, _)| t == 0x040B);
    let drawings = drawing_groups.next().map(|(.., body)| body);
    let drawings_conflict = drawing_groups.next().is_some();
    let drawings = if drawings_conflict { None } else { drawings };
    Some(DocLayout {
        persist,
        slide_list,
        notes_list,
        master_list,
        objects,
        drawings,
        drawings_conflict,
    })
}

/// One master's TxMasterStyleAtoms, keyed by text-type instance.
fn master_styles(master: &[u8]) -> MasterStyles {
    let mut styles = MasterStyles::new();
    for (ver_inst, rec_type, body) in children(master) {
        if rec_type == 0x0FA3 {
            let instance = ver_inst >> 4;
            styles.entry(instance).or_insert_with(|| styletext::parse_master_style(body, instance));
        }
    }
    styles
}

/// Masters in master-list order (MasterPersistAtoms: persistIdRef at 0,
/// masterId at 12); falls back to a persist-directory scan when the list is
/// absent so single-master decks still get their defaults.
fn collect_masters(
    master_list: Option<&[u8]>,
    persist: &HashMap<u32, usize>,
    data: &[u8],
) -> Vec<(u32, MasterStyles)> {
    let mut out = Vec::new();
    if let Some(list) = master_list {
        for (_, rec_type, body) in children(list) {
            if rec_type != 0x03F3 {
                continue;
            }
            let (Some(persist_ref), Some(master_id)) = (get_u32(body, 0), get_u32(body, 12)) else {
                continue;
            };
            if let Some(&off) = persist.get(&persist_ref)
                && let Some((_, 0x03F8, master)) = record_at(data, off)
            {
                out.push((master_id, master_styles(master)));
            }
        }
    }
    if out.is_empty() {
        let mut offs: Vec<usize> = persist.values().copied().collect();
        offs.sort_unstable();
        for off in offs {
            if let Some((_, 0x03F8, master)) = record_at(data, off) {
                out.push((0, master_styles(master)));
            }
        }
    }
    out
}

impl<'a> Extractor<'a> {
    /// Walk slides in presentation order: the UserEditAtom chain yields the
    /// persist directory, the DocumentContainer's SlideListWithText yields
    /// slide order and outline text, each slide container its own textboxes.
    /// `Ok(false)` means the persist directory was unusable.
    fn parse_slides(
        &mut self,
        data: &'a [u8],
        current_user: &[u8],
        delay: &'a [u8],
    ) -> Result<bool, ConvertError> {
        let Some(layout) = locate_document(data, current_user) else {
            return Ok(false);
        };
        self.pictures = pictures::Bank::read(layout.drawings.unwrap_or_default(), delay)?;
        if layout.drawings_conflict {
            self.pictures.warn("Conflicting drawing groups in the current presentation document prevent picture-bank resolution; its figures are omitted.".into());
        }
        self.masters = collect_masters(layout.master_list, &layout.persist, data);
        self.objects = ole::Objects::new(layout.objects, &layout.persist, data);
        self.walk_slide_list(layout.slide_list, &layout.persist, data, false, 0x03EE)?;
        if let Some(notes_list) = layout.notes_list {
            self.walk_slide_list(notes_list, &layout.persist, data, true, 0x03F0)?;
        }
        Ok(true)
    }

    fn walk_slide_list(
        &mut self,
        list: &[u8],
        persist: &HashMap<u32, usize>,
        data: &[u8],
        is_notes: bool,
        container_type: u16,
    ) -> Result<(), ConvertError> {
        // (persistIdRef, slideId) of the page whose container is pending.
        let mut pending: Option<(u32, u32)> = None;
        for (ver_inst, rec_type, body) in children(list) {
            match rec_type {
                // SlidePersistAtom: the next slide/notes page begins.
                0x03F3 => {
                    let page = pending.is_some();
                    let id =
                        self.finish_slide(pending.take(), persist, data, container_type, is_notes)?;
                    self.close_segment(id, page);
                    self.current_is_notes = is_notes;
                    pending = get_u32(body, 0).map(|p| (p, get_u32(body, 12).unwrap_or(0)));
                    if !is_notes {
                        self.select_master(pending.map(|(p, _)| p), persist, data);
                    }
                }
                _ => self.record(ver_inst, rec_type, body)?,
            }
        }
        let page = pending.is_some();
        let id = self.finish_slide(pending, persist, data, container_type, is_notes)?;
        self.close_segment(id, page);
        Ok(())
    }

    /// Emit a slide's own textboxes after its outline text. Returns the
    /// segment's pairing id: the slideId for slides, or the owning slide's
    /// id (NotesAtom slideIdRef) for notes pages.
    fn finish_slide(
        &mut self,
        pending: Option<(u32, u32)>,
        persist: &HashMap<u32, usize>,
        data: &[u8],
        container_type: u16,
        is_notes: bool,
    ) -> Result<Option<u32>, ConvertError> {
        let Some((persist_ref, slide_id)) = pending else {
            return Ok(None);
        };
        let mut id = if is_notes { None } else { (slide_id != 0).then_some(slide_id) };
        if let Some(off) = persist.get(&persist_ref)
            && let Some((_, t, body)) = record_at(data, *off)
            && t == container_type
        {
            if is_notes {
                // NotesAtom.slideIdRef names the owning slide (0 = none).
                id = children(body)
                    .find(|&(_, t, _)| t == 0x03F1)
                    .and_then(|(.., atom)| get_u32(atom, 0))
                    .filter(|&v| v != 0);
            }
            self.walk(body)?;
        }
        Ok(id)
    }

    fn end_segment(&mut self, id: Option<u32>) {
        self.close_segment(id, false);
    }

    /// Finish the segment being accumulated. `slide` says a page of the slide
    /// list ends here: unless it is a notes page, the segment is kept even
    /// when empty (markitai: a blank slide still counts); any other empty
    /// segment is dropped.
    fn close_segment(&mut self, id: Option<u32>, slide: bool) {
        self.flush_shape();
        flush_list(&mut self.current, &mut self.list_run);
        let kind = if self.current_is_notes {
            SegmentKind::Notes
        } else if slide {
            SegmentKind::Slide
        } else {
            SegmentKind::Loose
        };
        if !self.current.is_empty() || kind == SegmentKind::Slide {
            let blocks = std::mem::take(&mut self.current);
            self.segments.push((blocks, id, kind));
        }
    }

    /// The blocks in presentation order and, for each slide of the slide
    /// list, the index of its first block (markitai).
    fn into_blocks(mut self) -> (Vec<Block>, Vec<usize>) {
        self.end_segment(None);
        // (pairing id, blocks, whether the segment is a slide of the list)
        let mut slides: Vec<(Option<u32>, Vec<Block>, bool)> = Vec::new();
        let mut notes: Vec<(Option<u32>, Vec<Block>)> = Vec::new();
        for (blocks, id, kind) in self.segments {
            match kind {
                SegmentKind::Notes => notes.push((id, blocks)),
                SegmentKind::Slide => slides.push((id, blocks, true)),
                SegmentKind::Loose => slides.push((id, blocks, false)),
            }
        }
        // Notes pages pair to slides by their stored slide id, not by list
        // position: the notes list may be sparse (notes on only some
        // slides), which order-based zipping would misattribute.
        let mut used = vec![false; notes.len()];
        let mut out = Vec::new();
        let mut slide_starts = Vec::new();
        for (sid, blocks, is_slide) in slides {
            if is_slide {
                slide_starts.push(out.len());
            }
            out.extend(blocks);
            for (i, (nid, nblocks)) in notes.iter_mut().enumerate() {
                if !used[i] && sid.is_some() && *nid == sid {
                    used[i] = true;
                    out.push(Block::BlockQuote(std::mem::take(nblocks)));
                }
            }
        }
        // Notes without a resolvable owner keep document order at the end.
        for (used, (_, nblocks)) in used.into_iter().zip(notes) {
            if !used && !nblocks.is_empty() {
                out.push(Block::BlockQuote(nblocks));
            }
        }
        (out, slide_starts)
    }

    /// Pick the master the slide references (SlideAtom.masterIdRef at
    /// offset 12); the first listed master is the deterministic fallback.
    fn select_master(&mut self, slide: Option<u32>, persist: &HashMap<u32, usize>, data: &[u8]) {
        self.active_master = slide
            .and_then(|id| persist.get(&id))
            .and_then(|&off| record_at(data, off))
            .filter(|&(_, t, _)| t == 0x03EE)
            .and_then(|(_, _, body)| children(body).find(|&(_, rec_type, _)| rec_type == 0x03EF))
            .and_then(|(.., atom)| get_u32(atom, 12))
            .and_then(|mid| self.masters.iter().position(|&(id, _)| id == mid))
            .unwrap_or(0);
    }

    /// Iterative container walk over an explicit stack with fixed depth and
    /// record-count bounds — nesting or record counts beyond any real
    /// presentation are attack shapes and hard-fail.
    fn walk(&mut self, data: &[u8]) -> Result<(), ConvertError> {
        self.walk_hidden(data, false)
    }

    fn walk_hidden(&mut self, data: &[u8], hidden: bool) -> Result<(), ConvertError> {
        let mut stack: Vec<(&[u8], usize, bool)> = vec![(data, 0, hidden)];
        while let Some((buf, pos, inherited_hidden)) = stack.last_mut() {
            let Some((ver_inst, rec_type, body)) = record_at(buf, *pos) else {
                stack.pop();
                continue;
            };
            *pos += 8 + body.len();
            let inherited_hidden = *inherited_hidden;
            self.charge_record()?;
            if rec_type == 0xF004 && !inherited_hidden && !self.recovering {
                self.figure(body)?;
            }
            // markitai: ExObjRefAtom, in a shape's client data: the shape
            // shows an embedded object, read as its data where the shape is.
            if rec_type == 0x0BC1 && ver_inst & 0xF != 0xF {
                self.object(body)?;
                continue;
            }
            if ver_inst & 0xF != 0xF {
                self.atom(rec_type, body);
                continue;
            }
            match rec_type {
                // CryptSession10Container: the stream is encrypted.
                0x2F14 => self.encrypted = true,
                // Notes containers are walked via the notes list; recovery
                // has no lists, so their text is taken inline (as notes).
                // A NotesAtom slideIdRef in the masters range (high bit
                // set) marks the notes master: template chrome, excluded.
                0x03F0 if self.recovering => {
                    let master = children(body)
                        .find(|&(_, t, _)| t == 0x03F1)
                        .and_then(|(.., atom)| get_u32(atom, 0))
                        .is_some_and(|id| id & 0x8000_0000 != 0);
                    if !master {
                        self.end_segment(None);
                        self.current_is_notes = true;
                        self.walk(body)?;
                        self.end_segment(None);
                        self.current_is_notes = false;
                    }
                }
                // Notes, master, and handout containers are walked via their
                // own lists, not inline.
                0x03F0 | 0x03F8 | 0x0FC9 => {}
                // Only instance 0 of SlideListWithText holds slide text here.
                0x0FF0 if ver_inst >> 4 != 0 => {}
                // markitai: a group shape marked as a table is read as one
                // table; one nested in a cell is walked as plain shapes.
                0xF003 if !self.in_table && is_table_group(body) => {
                    self.table(body, inherited_hidden || pictures::group_hidden(body))?;
                }
                _ => {
                    if stack.len() >= limits::MAX_RECORD_DEPTH {
                        return Err(ConvertError::ResourceLimit {
                            limit: "max_record_depth",
                            detail: format!("record nesting exceeds {}", limits::MAX_RECORD_DEPTH),
                        });
                    }
                    let hidden = inherited_hidden
                        || (rec_type == 0xF003 && pictures::group_hidden(body))
                        || (rec_type == 0xF004 && pictures::shape(body).hidden);
                    stack.push((body, 0, hidden));
                }
            }
        }
        Ok(())
    }

    /// One record outside `walk` (slide-list traversal): containers descend
    /// through the bounded walk, atoms extract directly.
    fn record(&mut self, ver_inst: u16, rec_type: u16, body: &[u8]) -> Result<(), ConvertError> {
        self.charge_record()?;
        if ver_inst & 0xF == 0xF {
            match rec_type {
                0x2F14 => self.encrypted = true,
                0x03F0 | 0x03F8 | 0x0FC9 => {}
                0x0FF0 if ver_inst >> 4 != 0 => {}
                _ => self.walk(body)?,
            }
        } else {
            self.atom(rec_type, body);
        }
        Ok(())
    }

    fn charge_record(&mut self) -> Result<(), ConvertError> {
        self.records += 1;
        if self.records > limits::MAX_RECORDS {
            return Err(ConvertError::ResourceLimit {
                limit: "max_records",
                detail: format!("record stream exceeds {} records", limits::MAX_RECORDS),
            });
        }
        Ok(())
    }

    fn atom(&mut self, rec_type: u16, body: &[u8]) {
        match rec_type {
            // TextHeaderAtom: a new text shape begins.
            0x0F9F => {
                self.flush_shape();
                self.pending = Some(PendingShape {
                    tx_type: body.first().copied().unwrap_or(1),
                    text: String::new(),
                    styles: None,
                });
            }
            // TextCharsAtom: UTF-16LE.
            0x0FA0 => {
                let units = utf16le_units(body);
                let text: String =
                    char::decode_utf16(units).map(|r| r.unwrap_or('\u{fffd}')).collect();
                self.push_text(text);
            }
            // TextBytesAtom: low bytes of UTF-16 code units.
            0x0FA8 => {
                let text: String = body.iter().map(|&b| b as char).collect();
                self.push_text(text);
            }
            // StyleTextPropAtom: styling for the pending shape's text.
            0x0FA1 => {
                if let Some(pending) = &mut self.pending {
                    let len = pending.text.chars().map(char::len_utf16).sum();
                    pending.styles = Some(styletext::parse_style_text(body, len));
                }
            }
            // ExHyperlinkAtom: explicit degradation — hyperlink targets in
            // the legacy record stream are not resolved to link inlines.
            0x0FD3 => {
                log::debug!("ppt hyperlink records present; targets are not resolved");
            }
            _ => {}
        }
    }

    /// markitai: the data of the embedded object an `ExObjRefAtom` names,
    /// after the text read so far.
    fn object(&mut self, atom: &[u8]) -> Result<(), ConvertError> {
        let Some(id) = get_u32(atom, 0) else {
            return Ok(());
        };
        let blocks = self.objects.blocks(id)?;
        if !blocks.is_empty() {
            self.flush_shape();
            flush_list(&mut self.current, &mut self.list_run);
            self.current.extend(blocks);
        }
        Ok(())
    }

    /// markitai: read a table group. Each cell is a shape of the group with
    /// its own text box, read like any other text shape; the cells' anchors
    /// draw the grid. A group whose cells cannot be placed keeps their text
    /// as the blocks it had before tables were read.
    fn figure(&mut self, body: &[u8]) -> Result<(), ConvertError> {
        if let Some(pib) = pictures::shape(body).pib
            && let Some(image) = self.pictures.image(pib)?
        {
            self.flush_shape();
            flush_list(&mut self.current, &mut self.list_run);
            self.current.push(Block::Paragraph(vec![image]));
        }
        Ok(())
    }

    fn table(&mut self, group: &[u8], hidden: bool) -> Result<(), ConvertError> {
        self.flush_shape();
        flush_list(&mut self.current, &mut self.list_run);
        let outer = std::mem::take(&mut self.current);
        self.in_table = true;
        let mut cells = Vec::new();
        let mut placed = true;
        for (_, rec_type, shape) in children(group) {
            self.charge_record()?;
            // The group's own shape holds no cell.
            if rec_type != 0xF004 || children(shape).any(|(_, t, _)| t == 0xF009) {
                continue;
            }
            let anchor = children(shape).find(|&(_, t, _)| t == 0xF00F).and_then(|(.., body)| {
                let edge = |at| get_u32(body, at).map(|v| v as i32);
                Some([edge(0)?, edge(4)?, edge(8)?, edge(12)?])
            });
            if !hidden && !self.recovering {
                self.figure(shape)?;
            }
            self.walk_hidden(shape, hidden || pictures::shape(shape).hidden)?;
            self.flush_shape();
            flush_list(&mut self.current, &mut self.list_run);
            let blocks = std::mem::take(&mut self.current);
            match anchor {
                Some([left, top, right, bottom]) if left < right && top < bottom => {
                    cells.push(([left, top, right, bottom], blocks));
                }
                // A zero-size shape is a border line, not a cell.
                Some(_) if blocks.is_empty() => {}
                _ => {
                    placed = false;
                    cells.push(([0; 4], blocks));
                }
            }
        }
        self.in_table = false;
        self.current = outer;
        match placed.then(|| table_grid(&mut cells)).flatten() {
            Some(table) => {
                let table = table?;
                if !table.grid.is_empty() {
                    self.current.push(Block::Table(table));
                }
            }
            None => self.current.extend(cells.into_iter().flat_map(|(_, blocks)| blocks)),
        }
        Ok(())
    }

    fn push_text(&mut self, text: String) {
        match &mut self.pending {
            Some(pending) => pending.text.push_str(&text),
            None => {
                self.pending = Some(PendingShape { tx_type: 1, text, styles: None });
            }
        }
    }

    /// Emit the pending shape: paragraphs split on CR, styled by the
    /// character runs, listed by paragraph depth/bullet with master defaults.
    fn flush_shape(&mut self) {
        let Some(shape) = self.pending.take() else {
            return;
        };
        if shape.text.is_empty() {
            return;
        }
        self.shape_counter += 1;
        let shape_id = self.shape_counter;
        let is_title = matches!(shape.tx_type, 0 | 6);
        let styles = shape.styles.unwrap_or_default();
        // The active master's per-level defaults for this text type; local
        // exceptions are tri-state and resolve over these.
        let master_levels: Vec<MasterLevel> = self
            .masters
            .get(self.active_master)
            .and_then(|(_, m)| m.get(&(shape.tx_type as u16)))
            .cloned()
            .unwrap_or_default();
        let level_default =
            |depth: u16| master_levels.get(depth as usize).copied().unwrap_or_default();

        // Cursors over the style runs, counted in UTF-16 units.
        let mut char_runs = styles.chars.iter();
        let mut char_run: Option<&CharProps> = char_runs.next();
        let mut char_left = char_run.map(|r| r.count).unwrap_or(usize::MAX);
        let mut para_runs = styles.paragraphs.iter();
        let mut para_run = para_runs.next();
        let mut para_left = para_run.map(|r| r.count).unwrap_or(usize::MAX);

        let mut paragraphs: Vec<(Vec<Inline>, u16, Option<bool>)> = Vec::new();
        let mut inlines: Vec<Inline> = Vec::new();
        let mut run_text = String::new();
        let mut run_style = Style::PLAIN;
        let para_props = |r: Option<&styletext::ParaProps>| match r {
            Some(p) => (p.depth, p.bullet),
            None => (0, None),
        };

        for c in shape.text.chars() {
            let d = level_default(para_props(para_run).0);
            let style = Style {
                bold: char_run.and_then(|r| r.bold).or(d.bold).unwrap_or(false),
                italic: char_run.and_then(|r| r.italic).or(d.italic).unwrap_or(false),
                strike: false,
                underline: false,
                code: false,
            };
            if c == '\r' {
                if !run_text.is_empty() {
                    let text = clean_text(&std::mem::take(&mut run_text));
                    if !text.is_empty() {
                        inlines.push(Inline::Text { text, style: run_style });
                    }
                }
                let (depth, bullet) = para_props(para_run);
                paragraphs.push((std::mem::take(&mut inlines), depth, bullet));
            } else if c == '\u{b}' {
                if !run_text.is_empty() {
                    let text = clean_text(&std::mem::take(&mut run_text));
                    if !text.is_empty() {
                        inlines.push(Inline::Text { text, style: run_style });
                    }
                }
                inlines.push(Inline::LineBreak);
            } else {
                if style != run_style && !run_text.is_empty() {
                    let text = clean_text(&std::mem::take(&mut run_text));
                    if !text.is_empty() {
                        inlines.push(Inline::Text { text, style: run_style });
                    }
                }
                run_style = style;
                run_text.push(c);
            }
            // Advance run cursors by the character's UTF-16 width.
            let width = c.len_utf16();
            char_left = char_left.saturating_sub(width);
            if char_left == 0 {
                char_run = char_runs.next();
                char_left = char_run.map(|r| r.count).unwrap_or(usize::MAX);
            }
            para_left = para_left.saturating_sub(width);
            if para_left == 0 {
                para_run = para_runs.next();
                para_left = para_run.map(|r| r.count).unwrap_or(usize::MAX);
            }
        }
        if !run_text.is_empty() {
            let text = clean_text(&run_text);
            if !text.is_empty() {
                inlines.push(Inline::Text { text, style: run_style });
            }
        }
        if !inlines.is_empty() {
            let (depth, bullet) = para_props(para_run);
            paragraphs.push((inlines, depth, bullet));
        }

        for (mut inlines, depth, bullet) in paragraphs {
            if inlines_are_empty(&inlines) {
                flush_list(&mut self.current, &mut self.list_run);
                continue;
            }
            if is_title {
                flush_list(&mut self.current, &mut self.list_run);
                let d = level_default(depth);
                let base = StyleDelta { bold: d.bold, italic: d.italic, ..Default::default() };
                rebase_emphasis(&mut inlines, base.resolve());
                let anchor = Some(crate::model::inlines_to_plain_text(&inlines));
                self.current.push(Block::Heading { level: 2, anchor, content: inlines });
                continue;
            }
            let bullet = bullet.or(level_default(depth).bullet).unwrap_or(false);
            if bullet {
                self.list_run.push(ListEntry {
                    level: depth as usize,
                    key: ListKey { instance: shape_id, marker: MarkerKind::Bullet },
                    number: 0,
                    label: None,
                    blocks: vec![Block::Paragraph(inlines)],
                    indent: None,
                    continues: false,
                });
            } else {
                flush_list(&mut self.current, &mut self.list_run);
                self.current.push(Block::Paragraph(inlines));
            }
        }
    }
}

/// markitai: whether a group shape is a table: its own shape's
/// tertiary options set bit 0 of `tableProperties` (0x039F).
fn is_table_group(group: &[u8]) -> bool {
    let Some((.., shape)) = children(group).next().filter(|&(_, t, _)| t == 0xF004) else {
        return false;
    };
    children(shape).filter(|&(_, t, _)| t == 0xF122).any(|(ver_inst, _, options)| {
        (0..usize::from(ver_inst >> 4)).any(|i| {
            let (Some(id), Some(value)) =
                (options.get(i * 6..i * 6 + 2), get_u32(options, i * 6 + 2))
            else {
                return false;
            };
            let id = u16::from_le_bytes([id[0], id[1]]);
            id & 0x3FFF == 0x039F && value & 1 == 1
        })
    })
}

/// markitai: edges closer than this, in master units (576 to the inch), are
/// one grid line; the cells of a table share their edges.
const EDGE_TOLERANCE: i32 = 8;

/// markitai: the grid lines along one axis: the cells' edges in order, an
/// edge within the tolerance of the line before it merged into that line.
fn grid_lines(mut edges: Vec<i32>) -> Vec<i32> {
    edges.sort_unstable();
    let mut lines: Vec<i32> = Vec::new();
    for edge in edges {
        if lines.last().is_none_or(|&last| edge.saturating_sub(last) > EDGE_TOLERANCE) {
            lines.push(edge);
        }
    }
    lines
}

/// markitai: the table that cells (left, top, right, bottom, then content)
/// draw. A cell starts at the grid lines of its left and top edges and spans
/// to those of its right and bottom; a cell landing where another already
/// is adds its text to that one. `None` when the cells draw no grid or
/// more positions than any grid may hold.
fn table_grid(cells: &mut [([i32; 4], Vec<Block>)]) -> Option<Result<Table, ConvertError>> {
    let columns = grid_lines(cells.iter().flat_map(|&([l, _, r, _], _)| [l, r]).collect());
    let rows = grid_lines(cells.iter().flat_map(|&([_, t, _, b], _)| [t, b]).collect());
    let (width, height) = (columns.len().checked_sub(1)?, rows.len().checked_sub(1)?);
    if width == 0 || height == 0 || width as u64 * height as u64 > limits::MAX_GRID_SLOTS {
        return None;
    }
    // Each edge belongs to the last line at or before it.
    let line = |lines: &[i32], edge: i32| lines.partition_point(|&l| l <= edge).saturating_sub(1);
    // (row, column, rows spanned, columns spanned, cell) in reading order.
    let mut origins: Vec<[usize; 5]> = cells
        .iter()
        .enumerate()
        .map(|(index, &([left, top, right, bottom], _))| {
            let row = line(&rows, top).min(height - 1);
            let column = line(&columns, left).min(width - 1);
            let depth = line(&rows, bottom).saturating_sub(row).max(1);
            [row, column, depth, line(&columns, right).saturating_sub(column).max(1), index]
        })
        .collect();
    origins.sort_unstable();
    // The kept origin owning each position: a span claims the positions no
    // earlier cell holds, and the grid builder cuts it short of the others.
    const FREE: u32 = u32::MAX;
    let mut owner = vec![FREE; width * height];
    let mut kept: Vec<[usize; 5]> = Vec::new();
    for [row, column, depth, span, index] in origins {
        let holder = owner[row * width + column];
        if holder != FREE {
            let blocks = std::mem::take(&mut cells[index].1);
            cells[kept[holder as usize][4]].1.extend(blocks);
            continue;
        }
        for r in row..row + depth {
            for slot in &mut owner[r * width + column..r * width + column + span] {
                if *slot == FREE {
                    *slot = kept.len() as u32;
                }
            }
        }
        kept.push([row, column, depth, span, index]);
    }
    let mut builder = GridBuilder::new();
    for row in 0..height {
        builder.next_row();
        for column in 0..width {
            let placed = match owner[row * width + column] {
                FREE => builder.place(Cell::new(Vec::new())),
                holder => match kept[holder as usize] {
                    [r, c, depth, span, index] if (r, c) == (row, column) => {
                        builder.place(Cell::spanning(
                            std::mem::take(&mut cells[index].1),
                            span as u32,
                            depth as u32,
                        ))
                    }
                    _ => {
                        builder.covered();
                        Ok(())
                    }
                },
            };
            if let Err(error) = placed {
                return Some(Err(error));
            }
        }
    }
    let mut table = builder.finish(TableKind::Data);
    // PowerPoint styles a table's first row as its header by default.
    table.header_rows = resolve_header_rows(&table, 1);
    Some(Ok(table))
}

// markitai: tests for the slide boundaries recorded in `Document::slide_starts`.
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn record(ver: u16, instance: u16, rec_type: u16, body: &[u8]) -> Vec<u8> {
        let mut out = (ver | instance << 4).to_le_bytes().to_vec();
        out.extend(rec_type.to_le_bytes());
        out.extend((body.len() as u32).to_le_bytes());
        out.extend(body);
        out
    }

    /// A text shape: TextHeaderAtom of the given text type, then its text.
    fn text(tx_type: u32, chars: &str) -> Vec<u8> {
        let units: Vec<u8> = chars.encode_utf16().flat_map(u16::to_le_bytes).collect();
        let mut out = record(0, 0, 0x0F9F, &tx_type.to_le_bytes());
        out.extend(record(0, 0, 0x0FA0, &units));
        out
    }

    /// A SlidePersistAtom: persistIdRef at 0 and slideId at 12.
    fn slide_atom(persist_ref: u32, slide_id: u32) -> Vec<u8> {
        let mut body = [0u8; 20];
        body[..4].copy_from_slice(&persist_ref.to_le_bytes());
        body[12..16].copy_from_slice(&slide_id.to_le_bytes());
        record(0, 0, 0x03F3, &body)
    }

    /// A deck of five slides: a titled one, a blank one, a blank one with
    /// speaker notes, one with a body, and a blank last one. Slide text is
    /// outline text in the slide list. With `usable_directory` false the
    /// "Current User" stream names no edit, which leaves the persist
    /// directory unusable.
    fn deck(usable_directory: bool) -> Vec<u8> {
        // The notes container of slide 3 comes first, persisted as id 2; its
        // NotesAtom names slide id 258.
        let mut notes_body = record(0, 0, 0x03F1, &[2, 1, 0, 0, 0, 0, 0, 0]);
        notes_body.extend(text(2, "Note three"));
        let notes = record(0xF, 0, 0x03F0, &notes_body);

        let mut slides = Vec::new();
        slides.extend(slide_atom(10, 256));
        slides.extend(text(0, "First slide"));
        slides.extend(text(1, "First body"));
        slides.extend(slide_atom(11, 257));
        slides.extend(slide_atom(12, 258));
        slides.extend(slide_atom(13, 259));
        slides.extend(text(1, "Fourth body"));
        slides.extend(slide_atom(14, 260));
        let notes_list = slide_atom(2, 0);
        let mut document_body = record(0xF, 0, 0x0FF0, &slides);
        document_body.extend(record(0xF, 2, 0x0FF0, &notes_list));
        let document = record(0xF, 0, 0x03E8, &document_body);

        let mut stream = notes;
        let document_at = stream.len() as u32;
        stream.extend(document);
        // Persist ids 1 (the document) and 2 (the notes container).
        let mut directory = (1u32 | 2 << 20).to_le_bytes().to_vec();
        directory.extend(document_at.to_le_bytes());
        directory.extend(0u32.to_le_bytes());
        let directory_at = stream.len() as u32;
        stream.extend(record(0, 0, 0x1772, &directory));
        let mut edit = [0u8; 28];
        edit[12..16].copy_from_slice(&directory_at.to_le_bytes());
        edit[16..20].copy_from_slice(&1u32.to_le_bytes());
        let edit_at = stream.len() as u32;
        stream.extend(record(0, 0, 0x0FF5, &edit));

        let mut user = Vec::new();
        user.extend(20u32.to_le_bytes());
        user.extend(0xE391_C05Fu32.to_le_bytes());
        user.extend(if usable_directory { edit_at } else { 0 }.to_le_bytes());
        let current_user = record(0, 0, 0x0FF6, &user);

        let mut ole = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
        ole.create_stream("PowerPoint Document").unwrap().write_all(&stream).unwrap();
        ole.create_stream("Current User").unwrap().write_all(&current_user).unwrap();
        ole.into_inner().into_inner()
    }

    #[test]
    fn every_slide_starts_at_its_first_block_and_blank_slides_keep_their_place() {
        let doc = parse(&deck(true)).unwrap();
        let [
            Block::Heading { .. },
            Block::Paragraph(_),
            Block::BlockQuote(notes),
            Block::Paragraph(fourth),
        ] = &doc.blocks[..]
        else {
            panic!("unexpected blocks: {:?}", doc.blocks);
        };
        assert_eq!(notes.len(), 1);
        assert_eq!(crate::model::inlines_to_plain_text(fourth), "Fourth body");
        // Slide 2 has no blocks, so it starts where slide 3 does, and the
        // notes of the otherwise blank slide 3 stay in it; slide 5 has none
        // and is last, so it starts at the end.
        assert_eq!(doc.slide_starts, [0, 2, 2, 3, 4]);
    }

    // markitai: tables read from table groups.

    /// A shape of the given type at `anchor` (left, top, right, bottom),
    /// with a text box holding `chars` when there is one.
    fn shape(shape_type: u16, anchor: [i32; 4], chars: Option<&str>) -> Vec<u8> {
        let mut body = record(2, shape_type, 0xF00A, &[0; 8]);
        if anchor != [0; 4] {
            let edges: Vec<u8> = anchor.iter().flat_map(|e| e.to_le_bytes()).collect();
            body.extend(record(0, 0, 0xF00F, &edges));
        }
        if let Some(chars) = chars {
            body.extend(record(0xF, 0, 0xF00D, &text(4, chars)));
        }
        record(0xF, 0, 0xF004, &body)
    }

    fn cell(anchor: [i32; 4], chars: &str) -> Vec<u8> {
        shape(1, anchor, Some(chars))
    }

    /// A group of `shapes` whose own shape sets `tableProperties` to `flags`.
    fn group(flags: u32, shapes: &[Vec<u8>]) -> Vec<u8> {
        let mut own = record(1, 0, 0xF009, &[0; 16]);
        own.extend(record(2, 0, 0xF00A, &[0; 8]));
        let mut option = 0x039Fu16.to_le_bytes().to_vec();
        option.extend(flags.to_le_bytes());
        own.extend(record(3, 1, 0xF122, &option));
        let mut body = record(0xF, 0, 0xF004, &own);
        for shape in shapes {
            body.extend(shape);
        }
        record(0xF, 0, 0xF003, &body)
    }

    fn walked(data: &[u8]) -> Vec<Block> {
        let mut ex = Extractor::default();
        ex.walk(data).unwrap();
        ex.flush_shape();
        flush_list(&mut ex.current, &mut ex.list_run);
        ex.current
    }

    fn plain(blocks: &[Block]) -> String {
        let text = |block: &Block| match block {
            Block::Paragraph(inlines) => crate::model::inlines_to_plain_text(inlines),
            other => format!("{other:?}"),
        };
        blocks.iter().map(text).collect::<Vec<_>>().join("/")
    }

    /// Each slot as `text:columns x rows`, or `^row,column` when covered.
    fn grid(block: &Block) -> Vec<Vec<String>> {
        let Block::Table(table) = block else {
            panic!("not a table: {block:?}");
        };
        let slot = |slot: &crate::model::CellSlot| match slot {
            crate::model::CellSlot::Origin(cell) => {
                format!("{}:{}x{}", plain(&cell.blocks), cell.col_span, cell.row_span)
            }
            crate::model::CellSlot::Covered { origin_row, origin_col } => {
                format!("^{origin_row},{origin_col}")
            }
        };
        table.grid.iter().map(|row| row.iter().map(slot).collect()).collect()
    }

    #[test]
    fn a_table_group_reads_as_one_table_laid_out_by_its_cells_anchors() {
        // Stored out of reading order, with a border rule among the cells
        // and one cell's edges a few units off the shared grid lines.
        let shapes = [
            cell([203, 102, 300, 150], "9"),
            cell([100, 100, 200, 150], "7"),
            shape(20, [0, 50, 300, 50], None),
            cell([100, 50, 300, 100], "Both quarters"),
            cell([0, 50, 100, 150], "North"),
            cell([200, 0, 300, 50], "Q2"),
            cell([100, 0, 200, 50], "Q1"),
            cell([0, 0, 100, 50], "Region"),
        ];
        let blocks = walked(&group(0x11, &shapes));
        let [table] = &blocks[..] else {
            panic!("unexpected blocks: {blocks:?}");
        };
        assert_eq!(
            grid(table),
            [
                ["Region:1x1", "Q1:1x1", "Q2:1x1"],
                ["North:1x2", "Both quarters:2x1", "^1,1"],
                ["^1,0", "7:1x1", "9:1x1"],
            ]
        );
        let Block::Table(table) = table else { unreachable!() };
        assert_eq!(table.header_rows, 1);
    }

    #[test]
    fn a_group_not_marked_as_a_table_keeps_its_shapes_as_paragraphs() {
        let shapes = [cell([0, 0, 100, 50], "Left"), cell([100, 0, 200, 50], "Right")];
        assert_eq!(plain(&walked(&group(0x10, &shapes))), "Left/Right");
    }

    #[test]
    fn cells_that_cannot_be_placed_keep_their_text_in_stored_order() {
        let shapes = [
            cell([100, 0, 200, 50], "Second"),
            shape(1, [0; 4], Some("Unanchored")),
            cell([0, 0, 100, 50], "First"),
        ];
        assert_eq!(plain(&walked(&group(1, &shapes))), "Second/Unanchored/First");
    }

    #[test]
    fn a_cell_stored_over_another_adds_its_text_to_that_cell() {
        let shapes = [
            cell([0, 0, 100, 50], "Name"),
            cell([0, 0, 100, 50], "continued"),
            cell([100, 0, 200, 50], "Value"),
            cell([0, 50, 100, 100], "a"),
            cell([100, 50, 200, 100], "b"),
        ];
        let blocks = walked(&group(1, &shapes));
        assert_eq!(grid(&blocks[0]), [["Name/continued:1x1", "Value:1x1"], ["a:1x1", "b:1x1"]]);
    }

    #[test]
    fn a_cell_reaching_into_another_cells_span_stops_short_of_it() {
        let shapes = [
            cell([0, 0, 100, 50], "A"),
            cell([100, 0, 200, 100], "Tall"),
            cell([0, 50, 200, 100], "Wide"),
            cell([100, 50, 200, 100], "Under"),
        ];
        let blocks = walked(&group(1, &shapes));
        assert_eq!(grid(&blocks[0]), [["A:1x1", "Tall/Under:1x2"], ["Wide:1x1", "^0,1"]]);
    }

    #[test]
    fn a_table_whose_rows_below_the_first_are_empty_keeps_its_header() {
        let shapes = [
            cell([0, 0, 100, 50], "Column 1"),
            cell([100, 0, 200, 50], "Column 2"),
            cell([0, 50, 100, 100], ""),
            cell([100, 50, 200, 100], ""),
        ];
        let blocks = walked(&group(1, &shapes));
        assert_eq!(grid(&blocks[0]), [["Column 1:1x1", "Column 2:1x1"]]);
        let Block::Table(table) = &blocks[0] else { unreachable!() };
        assert_eq!(table.header_rows, 1);
    }

    #[test]
    fn slivers_within_the_tolerance_of_a_grid_line_join_the_cell_they_touch() {
        // Grid lines at x 0, 100, 200 and y 0, 50, 100; the first cell's
        // corner sits a few units inside them.
        let shapes = [
            cell([3, 3, 100, 50], "a"),
            cell([100, 0, 200, 50], "b"),
            cell([0, 50, 100, 100], "c"),
            cell([100, 50, 200, 100], "d"),
            cell([100, 0, 104, 50], "narrow"),
            cell([0, 50, 100, 54], "flat"),
            cell([202, 50, 206, 100], "right"),
            cell([0, 102, 200, 106], "low"),
        ];
        let blocks = walked(&group(1, &shapes));
        assert_eq!(
            grid(&blocks[0]),
            [["a:1x1", "b/narrow:1x1"], ["c/flat/low:1x1", "d/right:1x1"]]
        );
        // Cells that all collapse onto one point draw no grid.
        assert_eq!(plain(&walked(&group(1, &[cell([0, 0, 4, 4], "Dot")]))), "Dot");
    }

    #[test]
    fn a_table_group_inside_a_cell_is_read_as_plain_shapes() {
        let inner = group(1, &[cell([0, 0, 10, 10], "x"), cell([10, 0, 20, 10], "y")]);
        let mut nested = record(2, 1, 0xF00A, &[0; 8]);
        let edges: Vec<u8> = [0i32, 0, 100, 50].iter().flat_map(|e| e.to_le_bytes()).collect();
        nested.extend(record(0, 0, 0xF00F, &edges));
        nested.extend(inner);
        let shapes = [record(0xF, 0, 0xF004, &nested), cell([100, 0, 200, 50], "z")];
        let blocks = walked(&group(1, &shapes));
        assert_eq!(grid(&blocks[0]), [["x/y:1x1", "z:1x1"]]);
    }

    #[test]
    fn a_deck_read_in_raw_stream_order_has_no_slide_boundaries() {
        let doc = parse(&deck(false)).unwrap();
        assert!(!doc.blocks.is_empty());
        assert!(doc.slide_starts.is_empty(), "{:?}", doc.slide_starts);
    }
}
