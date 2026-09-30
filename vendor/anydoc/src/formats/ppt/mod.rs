//! Legacy PowerPoint 97-2003 binary (.ppt): OLE2 container, record stream.
//! Slides resolve through the persist directory (the only default path);
//! text comes from TextHeaderAtom + TextCharsAtom/TextBytesAtom with
//! StyleTextPropAtom runs and TxMasterStyleAtom defaults applied. Raw
//! stream-order scanning exists only as an explicitly labelled recovery for
//! files whose persist directory is unusable. Speaker notes are included
//! (fixed policy), rendered as a quote after their slide.

mod styletext;

use crate::error::ConvertError;
use crate::model::{Block, Document, Inline, Style, inlines_are_empty};
use crate::package::limits;
use crate::shared::binary::{get_u32, read_ole_stream, utf16le_units};
use crate::shared::delta::{StyleDelta, rebase_emphasis};
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

    let mut ex = Extractor::default();
    if !ex.parse_slides(&data, &current_user)? {
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
    let assets = collect_pictures(&mut ole)?;
    let (blocks, slide_starts) = ex.into_blocks();
    Ok(Document { blocks, notes: Vec::new(), assets, slide_starts })
}

/// Retain the deck's embedded pictures from the `Pictures` stream (OfficeArt
/// BStore file blocks). Pictures are document-level assets; per-slide
/// placement is not resolved. Unsupported formats degrade with a log.
fn collect_pictures<R: std::io::Read + std::io::Seek>(
    ole: &mut cfb::CompoundFile<R>,
) -> Result<Vec<crate::model::Asset>, ConvertError> {
    use crate::shared::officeart;
    let Ok(pictures) = read_ole_stream(ole, "Pictures") else {
        return Ok(Vec::new());
    };
    let mut sink = crate::shared::assets::AssetSink::new();
    let mut pos = 0usize;
    let mut index = 0u32;
    while let Some((ver_inst, rec_type, body)) = officeart::record_at(&pictures, pos) {
        pos += 8 + body.len();
        index += 1;
        if index > 100_000 {
            break;
        }
        let cap = limits::MAX_ENTRY_BYTES as usize;
        let blip = match rec_type {
            0xF007 => officeart::fbse_blip(body, cap),
            _ => officeart::decode_blip(ver_inst, rec_type, body, cap),
        };
        match blip {
            Some(blip) => {
                sink.add(
                    blip.media_type.to_string(),
                    format!("pictures/{index}.{}", blip.extension),
                    &blip.bytes,
                )?;
            }
            None => log::debug!("skipping unsupported Pictures record 0x{rec_type:04X}"),
        }
    }
    Ok(sink.assets)
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
struct Extractor {
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
}

/// The persist-resolved layout of the presentation: slide/notes lists from
/// the current DocumentContainer, persist id -> offset for every container.
struct DocLayout<'a> {
    persist: HashMap<u32, usize>,
    slide_list: &'a [u8],
    notes_list: Option<&'a [u8]>,
    master_list: Option<&'a [u8]>,
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
    Some(DocLayout { persist, slide_list, notes_list, master_list })
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

impl Extractor {
    /// Walk slides in presentation order: the UserEditAtom chain yields the
    /// persist directory, the DocumentContainer's SlideListWithText yields
    /// slide order and outline text, each slide container its own textboxes.
    /// `Ok(false)` means the persist directory was unusable.
    fn parse_slides(&mut self, data: &[u8], current_user: &[u8]) -> Result<bool, ConvertError> {
        let Some(layout) = locate_document(data, current_user) else {
            return Ok(false);
        };
        self.masters = collect_masters(layout.master_list, &layout.persist, data);
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
        let mut stack: Vec<(&[u8], usize)> = vec![(data, 0)];
        while let Some((buf, pos)) = stack.last_mut() {
            let Some((ver_inst, rec_type, body)) = record_at(buf, *pos) else {
                stack.pop();
                continue;
            };
            *pos += 8 + body.len();
            self.charge_record()?;
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
                _ => {
                    if stack.len() >= limits::MAX_RECORD_DEPTH {
                        return Err(ConvertError::ResourceLimit {
                            limit: "max_record_depth",
                            detail: format!("record nesting exceeds {}", limits::MAX_RECORD_DEPTH),
                        });
                    }
                    stack.push((body, 0));
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
                });
            } else {
                flush_list(&mut self.current, &mut self.list_run);
                self.current.push(Block::Paragraph(inlines));
            }
        }
    }
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

    #[test]
    fn a_deck_read_in_raw_stream_order_has_no_slide_boundaries() {
        let doc = parse(&deck(false)).unwrap();
        assert!(!doc.blocks.is_empty());
        assert!(doc.slide_starts.is_empty(), "{:?}", doc.slide_starts);
    }
}
