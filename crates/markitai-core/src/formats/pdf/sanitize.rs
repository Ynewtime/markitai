//! A policy for extracted page text, never a rewrite of the input PDF.
//!
//! Filtering operates on show operators before the layout reader, retaining
//! text advances and graphics state. Only suspect documents incur a second
//! reading; the original assets, comments and scan-layer verdict remain owned
//! by the first reading.

use super::{
    GraphicsState, MAX_CONTENT_OPERATIONS, MAX_STREAM_BYTES, PdfPages, decoded, matrix_scale,
    named_resource,
};
use crate::{Error, Result};
use lopdf::{
    Dictionary, Document, Object, ObjectId, Stream,
    content::{Content, Operation},
};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

const MAX_REWRITE_BYTES: usize = 128 * 1024 * 1024;
const UNKNOWN: &str = "hidden-text inspection could not determine";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    Off,
    Warn,
    Remove,
}

impl Mode {
    pub(super) fn from_config(cfg: &serde_json::Value) -> Result<Self> {
        match cfg.pointer("/security/pdf_sanitize") {
            None => Ok(Self::Warn),
            Some(serde_json::Value::String(mode)) => match mode.as_str() {
                "off" => Ok(Self::Off),
                "warn" => Ok(Self::Warn),
                "remove" => Ok(Self::Remove),
                _ => Err(Error::Config(
                    "security.pdf_sanitize must be off, warn or remove".into(),
                )),
            },
            _ => Err(Error::Config(
                "security.pdf_sanitize must be off, warn or remove".into(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ColorSpace {
    Gray,
    Rgb,
    Cmyk,
    Unknown,
}

impl ColorSpace {
    fn named(pdf: &Document, resources: &[&Dictionary], name: &[u8]) -> Self {
        let name = match name {
            b"DeviceGray" | b"DeviceRGB" | b"DeviceCMYK" => Some(name),
            _ => named_resource(pdf, resources, b"ColorSpace", name)
                .and_then(|v| super::resolved(pdf, v).ok())
                .and_then(|v| v.as_name().ok()),
        };
        match name {
            Some(b"DeviceGray") => Self::Gray,
            Some(b"DeviceRGB") => Self::Rgb,
            Some(b"DeviceCMYK") => Self::Cmyk,
            _ => Self::Unknown,
        }
    }

    fn white(self, operands: &[Object]) -> bool {
        let values: Option<Vec<_>> = operands
            .iter()
            .map(|v| v.as_float().ok().filter(|v| v.is_finite()))
            .collect();
        let Some(values) = values else {
            return false;
        };
        match self {
            Self::Gray => values.len() == 1 && values[0] >= 0.98,
            Self::Rgb => values.len() == 3 && values.iter().all(|v| *v >= 0.98),
            Self::Cmyk => values.len() == 4 && values.iter().all(|v| *v <= 0.02),
            Self::Unknown => false,
        }
    }
}

impl GraphicsState {
    /// State used by both inspection and filtering. Missing ExtGState entries
    /// inherit their values; fill and stroke have separate alpha channels.
    pub(super) fn apply_paint(
        &mut self,
        pdf: &Document,
        resources: &[&Dictionary],
        op: &Operation,
    ) {
        let last = op.operands.last();
        let number = || {
            last.and_then(|v| v.as_float().ok())
                .filter(|v| v.is_finite())
        };
        match op.operator.as_str() {
            "Tr" => {
                if let Some(mode) = last
                    .and_then(|v| v.as_i64().ok())
                    .filter(|v| (0..=7).contains(v))
                {
                    self.render_mode = mode;
                }
            }
            "Tf" => {
                if let Some(size) = number() {
                    self.font_size = size;
                }
            }
            "cm" => {
                if let Some(scale) = matrix_scale(&op.operands) {
                    self.ctm_scale *= scale;
                }
            }
            "BT" => self.text_scale = 1.0,
            "Tm" => {
                if let Some(scale) = matrix_scale(&op.operands) {
                    self.text_scale = scale;
                }
            }
            "g" | "rg" | "k" => {
                self.fill_space = match op.operator.as_str() {
                    "g" => ColorSpace::Gray,
                    "rg" => ColorSpace::Rgb,
                    _ => ColorSpace::Cmyk,
                };
                self.white_fill = self.fill_space.white(&op.operands);
            }
            "G" | "RG" | "K" => {
                self.stroke_space = match op.operator.as_str() {
                    "G" => ColorSpace::Gray,
                    "RG" => ColorSpace::Rgb,
                    _ => ColorSpace::Cmyk,
                };
                self.white_stroke = self.stroke_space.white(&op.operands);
            }
            "cs" | "CS" => {
                let space = last
                    .and_then(|v| v.as_name().ok())
                    .map(|name| ColorSpace::named(pdf, resources, name))
                    .unwrap_or(ColorSpace::Unknown);
                if op.operator == "cs" {
                    self.fill_space = space;
                    self.white_fill = false;
                } else {
                    self.stroke_space = space;
                    self.white_stroke = false;
                }
            }
            "sc" | "scn" => self.white_fill = self.fill_space.white(&op.operands),
            "SC" | "SCN" => self.white_stroke = self.stroke_space.white(&op.operands),
            "gs" => {
                let state = last
                    .and_then(|v| v.as_name().ok())
                    .and_then(|name| named_resource(pdf, resources, b"ExtGState", name))
                    .and_then(|v| super::resolved(pdf, v).ok())
                    .and_then(|v| v.as_dict().ok());
                let Some(state) = state else {
                    self.unknown_compositing = true;
                    return;
                };
                for (key, alpha) in [
                    (b"ca".as_slice(), &mut self.fill_alpha),
                    (b"CA".as_slice(), &mut self.stroke_alpha),
                ] {
                    if let Ok(value) = state.get(key) {
                        match value
                            .as_float()
                            .ok()
                            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
                        {
                            Some(value) => *alpha = value,
                            None => self.unknown_compositing = true,
                        }
                    }
                }
                if state
                    .get(b"SMask")
                    .is_ok_and(|v| v.as_name().ok() != Some(b"None"))
                    || state
                        .get(b"BM")
                        .is_ok_and(|v| !matches!(v.as_name().ok(), Some(b"Normal" | b"Compatible")))
                {
                    self.unknown_compositing = true;
                }
                if let Ok(font) = state.get(b"Font").and_then(Object::as_array)
                    && let Some(size) = font
                        .get(1)
                        .and_then(|v| v.as_float().ok())
                        .filter(|v| v.is_finite())
                {
                    self.font_size = size;
                }
            }
            _ => {}
        }
    }

    pub(super) fn unknown(self) -> Option<&'static str> {
        if matches!(self.render_mode, 3 | 7)
            || self.suspicious() == Some("transparent text graphics state")
        {
            None
        } else if self.unknown_compositing {
            Some("soft masks, blending or an unresolved graphics state")
        } else if (matches!(self.render_mode, 0 | 2 | 4 | 6)
            && self.fill_space == ColorSpace::Unknown)
            || (matches!(self.render_mode, 1 | 2 | 5 | 6)
                && self.stroke_space == ColorSpace::Unknown)
        {
            Some("text paint in an unsupported color space")
        } else {
            None
        }
    }
}

pub(super) fn unknown_warning(page: u32, reasons: &BTreeSet<&'static str>) -> String {
    format!(
        "PDF page {page}: {UNKNOWN} visibility for {}; some hidden text may remain. This bounded inspection does not establish PDF security sanitization.",
        reasons.iter().copied().collect::<Vec<_>>().join(", ")
    )
}

fn visibility_warning(text: &str) -> bool {
    text.contains("complete hidden-text filtering is not established") || text.contains(UNKNOWN)
}

/// Leave hidden text out of the extracted body. `warn` (the default) and
/// `remove` both leave out text a reader cannot see: transparent text, text
/// at one point or smaller, and white text over a white or near-white
/// background (what the page shows under each white run is sampled from a
/// render, so white text on a dark panel stays). They differ where the
/// answer is not known: `remove` also leaves out white text whose background
/// could not be checked and looks at pages whose visibility could not be
/// determined, `warn` keeps that text and says so. One notice quotes the
/// text left out. The original PDF and its image payloads are never changed;
/// a page read from its OCR layer is not touched.
pub(super) fn apply(
    bytes: &[u8],
    max_pages: Option<usize>,
    mode: Mode,
    pages: &mut PdfPages,
    trust: &super::Trust,
) -> Result<()> {
    if mode == Mode::Off {
        remove_warnings(pages, visibility_warning);
        return Ok(());
    }
    let selected: BTreeSet<_> = pages
        .pages
        .iter()
        .filter(|page| {
            (page.visibility_suspect
                || (mode == Mode::Remove
                    && pages.unverified_visibility.contains(&(page.number as u32))))
                && page.ocr_layer.is_none()
        })
        .map(|page| page.number as u32)
        .collect();
    if selected.is_empty() {
        return Ok(());
    }
    let name = match mode {
        Mode::Remove => "remove",
        _ => "warn",
    };
    let source = match lopdf::Document::load_mem(bytes) {
        Ok(source) => source,
        Err(_) => {
            pages.document.warnings.push(format!("PDF hidden-text removal (pdf_sanitize={name}) was not completed: the document could not be loaded for bounded operator filtering; the original reader output is retained."));
            return Ok(());
        }
    };
    let mut global_bytes = 0usize;
    let mut reports = BTreeMap::new();
    // First pass: every show operator painted in a colour that names it, so
    // the positioned reader says where each suspicious run is and what it
    // reads.
    let mut probe_document = source.clone();
    let mut probed = BTreeMap::new();
    let mut ids = BTreeMap::new();
    for (number, id) in source
        .get_pages()
        .into_iter()
        .filter(|(number, _)| selected.contains(number))
    {
        let (inspection, _) = super::inspect_page(&source, id);
        let mut report = Report::default();
        if inspection.incomplete {
            report
                .notes
                .insert("content inspection is incomplete; this page's original body is retained");
            reports.insert(number, report);
            continue;
        }
        let resources = match page_resources(&source, id) {
            Ok(Some(resources)) => resources.clone(),
            Ok(None) => Dictionary::new(),
            Err(_) => {
                report.notes.insert("page resources could not be resolved");
                reports.insert(number, report);
                continue;
            }
        };
        let content = match source.get_page_content_with_limit(id, MAX_STREAM_BYTES) {
            Ok(content) => content,
            Err(_) => {
                report.notes.insert("page content could not be decoded");
                reports.insert(number, report);
                continue;
            }
        };
        let mut probe_resources = resources.clone();
        let mut rewriter = Rewriter::new(
            &source,
            &mut probe_document,
            &mut global_bytes,
            &mut report,
            Pass::Probe,
        );
        let outcome = rewriter.content(
            &content,
            &mut probe_resources,
            GraphicsState::default(),
            &mut BTreeSet::new(),
            0,
        );
        let kinds = std::mem::take(&mut rewriter.kinds);
        match outcome {
            Ok((probe, _)) => {
                let probe = probe.encode().map_err(super::conversion)?;
                let stream = probe_document.add_object(Stream::new(Dictionary::new(), probe));
                if let Ok(page) = probe_document.get_dictionary_mut(id) {
                    page.set("Contents", stream);
                    page.set("Resources", probe_resources);
                }
                probed.insert(number, (content, resources, kinds));
                ids.insert(number, id);
            }
            Err(why) => {
                report.notes.insert(why);
            }
        }
        reports.insert(number, report);
    }
    let runs = probe_runs(probe_document, bytes.len(), &probed, &ids);
    // Second pass: the runs judged hidden become invisible (`Tr` 3 or 7, as
    // the reader leaves out), keeping every advance.
    let mut rewritten = source.clone();
    let mut changed = BTreeSet::new();
    // Pages read to the end with nothing left unknown: filtered, or with
    // nothing to filter (white text that is visible where it is drawn). A
    // page whose only suspicious runs are invisible ones keeps its signal.
    let mut checked = BTreeSet::new();
    let mut quoted: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    let mut background = super::Background::new(bytes);
    // A page that goes to OCR is not read from its text layer: its
    // invisible text (a rejected scan layer) is not quoted.
    let to_ocr: BTreeSet<u32> = pages
        .pages
        .iter()
        .filter(|page| page.needs_ocr)
        .map(|page| page.number as u32)
        .collect();
    for (number, (content, mut resources, kinds)) in probed {
        let id = ids[&number];
        let report = reports
            .get_mut(&number)
            .expect("every probed page has a report");
        let frame = super::geometry::frame(&source, id);
        let page_runs = runs.as_ref().and_then(|runs| runs.get(&number));
        let painted = has_background(&source, id, &mut BTreeSet::new(), 0);
        let mut hidden = BTreeSet::new();
        let mut texts = Vec::new();
        for (index, kind) in kinds.iter().enumerate() {
            let run = page_runs.and_then(|runs| runs.get(&index));
            let hides = match *kind {
                super::INVISIBLE_RENDERING => true,
                "white text" => {
                    let under = if painted {
                        run.zip(frame.as_ref())
                            .and_then(|(run, frame)| background.under(run, frame))
                    } else {
                        // Nothing is painted on the page: white on white.
                        Some(1.0)
                    };
                    match under {
                        Some(under) => under > super::Background::WHITE,
                        None if mode == Mode::Remove => true,
                        None => {
                            report.notes.insert(
                                "white text whose background could not be checked was retained",
                            );
                            false
                        }
                    }
                }
                _ => true,
            };
            if hides {
                hidden.insert(index);
                if *kind == super::INVISIBLE_RENDERING && to_ocr.contains(&number) {
                    continue;
                }
                if let Some(text) = run
                    .map(|run| run.text.trim())
                    .filter(|text| !text.is_empty())
                {
                    texts.push(text.to_owned());
                }
            }
        }
        let outcome = Rewriter::new(
            &source,
            &mut rewritten,
            &mut global_bytes,
            report,
            Pass::Filter(&hidden),
        )
        .content(
            &content,
            &mut resources,
            GraphicsState::default(),
            &mut BTreeSet::new(),
            0,
        );
        match outcome {
            Ok((content, true)) => {
                let content = content.encode().map_err(super::conversion)?;
                let stream = rewritten.add_object(Stream::new(Dictionary::new(), content));
                if let Ok(page) = rewritten.get_dictionary_mut(id) {
                    page.set("Contents", stream);
                    page.set("Resources", resources);
                    changed.insert(number);
                    checked.insert(number);
                }
            }
            Ok((_, false)) => {
                if !kinds.contains(&super::INVISIBLE_RENDERING) {
                    checked.insert(number);
                }
            }
            Err(why) => {
                report.notes.insert(why);
                report.filtered = 0;
                texts.clear();
            }
        }
        if !texts.is_empty() {
            quoted.insert(number, texts);
        }
    }
    if !changed.is_empty() {
        // The cap includes the original streams plus bounded additions. Stop
        // serialization before a malformed file can allocate an unbounded Vec.
        let mut buffer = BoundedBuffer {
            bytes: Vec::new(),
            limit: bytes.len().saturating_add(MAX_REWRITE_BYTES),
        };
        let reread = rewritten
            .save_to(&mut buffer)
            .ok()
            .and_then(|_| super::extract_pages_inner(&buffer.bytes, max_pages, false, trust).ok());
        if let Some(mut reread) = reread {
            let mut by_page: BTreeMap<_, _> = reread
                .pages
                .drain(..)
                .map(|page| (page.number as u32, page))
                .collect();
            for page in &mut pages.pages {
                let number = page.number as u32;
                if !changed.contains(&number) {
                    continue;
                }
                if let Some(mut body) = by_page.remove(&number) {
                    // A forced Tr3 mask must never become newly trusted OCR.
                    if body.ocr_layer.is_some() {
                        body.markdown.clear();
                        body.needs_ocr = true;
                        body.ocr_reason =
                            Some("filtered hidden text is not an accepted OCR layer".into());
                    }
                    page.markdown = body.markdown;
                    page.needs_ocr = body.needs_ocr;
                    page.ocr_reason = body.ocr_reason;
                    page.omitted_text = body.omitted_text;
                    page.layer_failure = body.layer_failure;
                    page.layer_kept = body.layer_kept;
                } else {
                    let report = reports.get_mut(&number).unwrap();
                    report.filtered = 0;
                    report.notes.insert(
                        "the filtered page could not be read; original reader output retained",
                    );
                    quoted.remove(&number);
                }
            }
        } else {
            for number in &changed {
                let report = reports.get_mut(number).unwrap();
                report.filtered = 0;
                report.notes.insert("the filtered document could not be read within its bound; original reader output retained");
                quoted.remove(number);
            }
        }
    }
    // A page filtered completely needs no "filtering is not established"
    // signal warning: the notice below says what was left out.
    let complete: BTreeSet<u32> = checked
        .into_iter()
        .filter(|number| reports[number].notes.is_empty())
        .collect();
    for page in &mut pages.pages {
        if complete.contains(&(page.number as u32)) {
            page.visibility_suspect = false;
        }
    }
    remove_warnings(pages, |warning| {
        complete.iter().any(|number| {
            warning.starts_with(&format!("PDF page {number}: contains "))
                && warning.contains("complete hidden-text filtering is not established")
        })
    });
    if !quoted.is_empty() {
        let spans: Vec<&str> = quoted.values().flatten().map(String::as_str).collect();
        let mut excerpt = spans.join("; ");
        if excerpt.chars().count() > EXCERPT_CHARS {
            excerpt = excerpt.chars().take(EXCERPT_CHARS).collect::<String>() + "...";
        }
        let numbers: Vec<usize> = quoted.keys().map(|&number| number as usize).collect();
        pages.document.warnings.push(format!(
            "PDF hidden text: {} span(s) on page(s) {} left out of the Markdown (possible prompt injection; excerpt: {excerpt:?}).",
            spans.len(),
            crate::formats::page_ranges(&numbers),
        ));
    }
    for (number, report) in reports {
        if mode == Mode::Remove && report.filtered > 0 {
            pages.document.warnings.push(format!("PDF page {number}: pdf_sanitize=remove filtered {} suspicious text show operator(s) from the extracted body; the original PDF and assets are unchanged. This is bounded output filtering, not a complete PDF security sanitizer.", report.filtered));
        }
        if !report.notes.is_empty() {
            let advice = if mode == Mode::Warn
                && report
                    .notes
                    .contains("white text whose background could not be checked was retained")
            {
                " Set security.pdf_sanitize to remove to leave such text out."
            } else {
                ""
            };
            pages.document.warnings.push(format!("PDF page {number}: pdf_sanitize={name} was not complete ({}); some text may remain. The original PDF and assets are unchanged.{advice}", report.notes.into_iter().collect::<Vec<_>>().join("; ")));
        }
    }
    Ok(())
}

/// The longest excerpt of hidden text a notice quotes, as in Python.
const EXCERPT_CHARS: usize = 80;

/// Drops the document warnings `drop` names, keeping each page's place
/// among the remaining ones.
fn remove_warnings(pages: &mut PdfPages, drop: impl Fn(&str) -> bool) {
    let removed: Vec<_> = pages
        .document
        .warnings
        .iter()
        .enumerate()
        .filter_map(|(i, w)| drop(w).then_some(i))
        .collect();
    if removed.is_empty() {
        return;
    }
    for page in &mut pages.pages {
        page.warning_index -= removed.iter().filter(|&&i| i < page.warning_index).count();
    }
    pages.document.warnings.retain(|w| !drop(w));
}

/// The colour the probe paints suspicious run `index` with: its number plus
/// one in the three 8-bit channels, never black (the colour of every other
/// run).
fn probe_color(index: usize) -> Option<[u8; 3]> {
    let value = u32::try_from(index).ok()?.checked_add(1)?;
    (value < 1 << 24).then_some([(value >> 16) as u8, (value >> 8) as u8, value as u8])
}

/// Each probed page's suspicious runs as the positioned reader reads them,
/// by their index: the first item painted in a run's colour. A run merged
/// into another item has no entry. `None` when the probe cannot be read.
/// Only the probed pages are written and read: the probe document's page
/// tree holds just them, so a long document is not read twice for a page.
fn probe_runs(
    mut probe: Document,
    input_bytes: usize,
    probed: &BTreeMap<u32, (Vec<u8>, Dictionary, Vec<&'static str>)>,
    ids: &BTreeMap<u32, ObjectId>,
) -> Option<BTreeMap<u32, BTreeMap<usize, pdf_inspector::TextItem>>> {
    if probed.is_empty() {
        return None;
    }
    let root = probe
        .catalog()
        .ok()?
        .get(b"Pages")
        .ok()?
        .as_reference()
        .ok()?;
    // Page numbers in the probe, in page order, to the original numbers.
    let original: Vec<u32> = ids.keys().copied().collect();
    for &id in ids.values() {
        // What the page inherits from the nodes it leaves behind.
        let mut inherited = Vec::new();
        let mut node = probe
            .get_dictionary(id)
            .ok()?
            .get(b"Parent")
            .and_then(Object::as_reference)
            .ok();
        let mut seen = BTreeSet::new();
        while let Some(parent) = node.filter(|parent| seen.insert(*parent) && seen.len() <= 64) {
            let Ok(dictionary) = probe.get_dictionary(parent) else {
                break;
            };
            for key in [b"MediaBox".as_slice(), b"CropBox", b"Rotate"] {
                if let Ok(value) = dictionary.get(key) {
                    inherited.push((key.to_vec(), value.clone()));
                }
            }
            node = dictionary
                .get(b"Parent")
                .and_then(Object::as_reference)
                .ok();
        }
        let page = probe.get_dictionary_mut(id).ok()?;
        for (key, value) in inherited {
            if !page.has(&key) {
                page.set(key, value);
            }
        }
        page.set("Parent", root);
    }
    let pages = probe.get_dictionary_mut(root).ok()?;
    pages.set(
        "Kids",
        ids.values()
            .map(|&id| Object::Reference(id))
            .collect::<Vec<_>>(),
    );
    pages.set("Count", ids.len() as i64);
    pages.remove(b"Parent");
    probe.prune_objects();
    let mut buffer = BoundedBuffer {
        bytes: Vec::new(),
        limit: input_bytes.saturating_add(MAX_REWRITE_BYTES),
    };
    probe.save_to(&mut buffer).ok()?;
    let loaded = pdf_inspector::LoadedPdf::load_mem(&buffer.bytes).ok()?;
    let (items, rotations) = loaded
        .text_with_positions_and_rotations(None, pdf_inspector::PositionOptions::new())
        .ok()?;
    let mut runs: BTreeMap<u32, BTreeMap<usize, pdf_inspector::TextItem>> = BTreeMap::new();
    for mut item in items {
        // A turned page's items are not in the frame its render is sampled in.
        if rotations.contains_key(&item.page) {
            continue;
        }
        let Some(&number) = original.get((item.page as usize).wrapping_sub(1)) else {
            continue;
        };
        let Some([r, g, b]) = item.fill_color else {
            continue;
        };
        let value = (usize::from(r) << 16) | (usize::from(g) << 8) | usize::from(b);
        let Some((_, _, kinds)) = probed.get(&number) else {
            continue;
        };
        if value == 0 || value > kinds.len() {
            continue;
        }
        item.page = number;
        runs.entry(number)
            .or_default()
            .entry(value - 1)
            .or_insert(item);
    }
    Some(runs)
}

#[derive(Default)]
struct Report {
    filtered: usize,
    notes: BTreeSet<&'static str>,
}

/// The nearest complete Resources entry wins. A page-local dictionary never
/// borrows missing categories/names from an older ancestor.
pub(super) fn page_resources(
    pdf: &Document,
    id: ObjectId,
) -> std::result::Result<Option<&Dictionary>, &'static str> {
    let mut id = id;
    let mut seen = BTreeSet::new();
    for _ in 0..64 {
        if !seen.insert(id) {
            return Err("a cycle was found in the page Parent chain");
        }
        let node = pdf
            .get_dictionary(id)
            .map_err(|_| "a page Parent is not a dictionary")?;
        if let Ok(resource) = node.get(b"Resources") {
            return super::resolved(pdf, resource)
                .ok()
                .and_then(|v| v.as_dict().ok())
                .map(Some)
                .ok_or("the nearest Resources entry is invalid");
        }
        match node.get(b"Parent") {
            Ok(parent) => {
                id = parent
                    .as_reference()
                    .map_err(|_| "a page Parent reference is invalid")?
            }
            Err(_) => return Ok(None),
        }
    }
    Err("the page Parent chain exceeds its inspection limit")
}

/// A direct inherited Resources dictionary is valid PDF. The pinned resource
/// helper reads only referenced ancestors. Materialize just these entries at
/// their pages before extraction; image streams and the input remain unchanged.
pub(super) fn normalize_inherited_resources(
    pdf: &Document,
    input_bytes: usize,
) -> Result<Option<Vec<u8>>> {
    let mut normalized = None;
    for (_, page_id) in pdf.get_pages() {
        let Ok(page) = pdf.get_dictionary(page_id) else {
            continue;
        };
        if page.has(b"Resources") {
            continue;
        }
        let mut ancestor = page.get(b"Parent").ok().and_then(|v| v.as_reference().ok());
        let mut seen = BTreeSet::new();
        for _ in 0..64 {
            let Some(id) = ancestor else {
                break;
            };
            if !seen.insert(id) {
                break;
            }
            let Ok(node) = pdf.get_dictionary(id) else {
                break;
            };
            if let Ok(resource) = node.get(b"Resources") {
                if let Ok(resource) = resource.as_dict() {
                    let copy = normalized.get_or_insert_with(|| pdf.clone());
                    if let Ok(page) = copy.get_dictionary_mut(page_id) {
                        page.set("Resources", resource.clone());
                    }
                }
                break;
            }
            ancestor = node.get(b"Parent").ok().and_then(|v| v.as_reference().ok());
        }
    }
    let Some(mut normalized) = normalized else {
        return Ok(None);
    };
    let mut buffer = BoundedBuffer {
        bytes: Vec::new(),
        limit: input_bytes.saturating_add(MAX_REWRITE_BYTES),
    };
    normalized.save_to(&mut buffer).map_err(super::conversion)?;
    Ok(Some(buffer.bytes))
}

/// Whiteness alone proves nothing on a page with a painted background.
/// Abstain for any path/image/shading/pattern paint, including Forms. This is
/// intentionally conservative rather than approximating coverage rectangles.
fn has_background(
    pdf: &Document,
    id: ObjectId,
    seen: &mut BTreeSet<ObjectId>,
    depth: usize,
) -> bool {
    let Ok(resources) = page_resources(pdf, id) else {
        return true;
    };
    let Ok(bytes) = pdf.get_page_content_with_limit(id, MAX_STREAM_BYTES) else {
        return true;
    };
    Background {
        pdf,
        bytes: 0,
        streams: 0,
    }
    .content(
        &bytes,
        &resources.into_iter().collect::<Vec<_>>(),
        seen,
        depth,
    )
}

struct Background<'a> {
    pdf: &'a Document,
    bytes: usize,
    streams: usize,
}
impl Background<'_> {
    fn content(
        &mut self,
        bytes: &[u8],
        resources: &[&Dictionary],
        seen: &mut BTreeSet<ObjectId>,
        depth: usize,
    ) -> bool {
        if depth > 32
            || self.streams >= 256
            || bytes.len() > MAX_STREAM_BYTES.saturating_sub(self.bytes)
        {
            return true;
        }
        self.streams += 1;
        self.bytes += bytes.len();
        let Ok(content) = Content::decode(bytes) else {
            return true;
        };
        if content.operations.len() > MAX_CONTENT_OPERATIONS {
            return true;
        }
        for op in content.operations {
            if matches!(
                op.operator.as_str(),
                "S" | "s"
                    | "f"
                    | "f*"
                    | "F"
                    | "B"
                    | "B*"
                    | "b"
                    | "b*"
                    | "sh"
                    | "BI"
                    | "ID"
                    | "scn"
                    | "SCN"
            ) {
                return true;
            }
            if op.operator != "Do" {
                continue;
            }
            let object = op
                .operands
                .first()
                .and_then(|v| v.as_name().ok())
                .and_then(|name| named_resource(self.pdf, resources, b"XObject", name));
            let Some(id) = object.and_then(|v| v.as_reference().ok()) else {
                return true;
            };
            let Ok(stream) = self.pdf.get_object(id).and_then(Object::as_stream) else {
                return true;
            };
            if stream.dict.get(b"Subtype").and_then(Object::as_name).ok() != Some(b"Form")
                || !seen.insert(id)
            {
                return true;
            }
            let local;
            let form_resources = if stream.dict.has(b"Resources") {
                let Ok(resource) = self.pdf.get_dict_in_dict(&stream.dict, b"Resources") else {
                    return true;
                };
                local = vec![resource];
                local.as_slice()
            } else {
                resources
            };
            let painted = match decoded(stream) {
                Ok(bytes) => self.content(&bytes, form_resources, seen, depth + 1),
                Err(_) => true,
            };
            seen.remove(&id);
            if painted {
                return true;
            }
        }
        false
    }
}

/// What a [`Rewriter`] pass does with the page's show operators.
#[derive(Clone, Copy)]
enum Pass<'a> {
    /// Paints each suspicious run in its [`probe_color`] (visible, so the
    /// positioned reader reads it) and every other run black.
    Probe,
    /// Makes the suspicious runs with these indices invisible.
    Filter(&'a BTreeSet<usize>),
}

struct Rewriter<'a> {
    source: &'a Document,
    output: &'a mut Document,
    global_bytes: &'a mut usize,
    bytes: usize,
    streams: usize,
    serial: usize,
    report: &'a mut Report,
    pass: Pass<'a>,
    /// Why each suspicious run met so far is suspicious, by its index in
    /// content order (Forms walked per invocation), the same in both passes.
    kinds: Vec<&'static str>,
}

impl<'a> Rewriter<'a> {
    fn new(
        source: &'a Document,
        output: &'a mut Document,
        global_bytes: &'a mut usize,
        report: &'a mut Report,
        pass: Pass<'a>,
    ) -> Self {
        Self {
            source,
            output,
            global_bytes,
            bytes: 0,
            streams: 0,
            serial: 0,
            report,
            pass,
            kinds: Vec::new(),
        }
    }

    fn content(
        &mut self,
        bytes: &[u8],
        resources: &mut Dictionary,
        mut state: GraphicsState,
        seen: &mut BTreeSet<ObjectId>,
        depth: usize,
    ) -> std::result::Result<(Content, bool), &'static str> {
        if depth > 32
            || self.streams >= 256
            || bytes.len() > MAX_STREAM_BYTES.saturating_sub(self.bytes)
            || bytes.len() > MAX_REWRITE_BYTES.saturating_sub(*self.global_bytes)
        {
            return Err("operator filtering exceeded its stream, nesting or byte budget");
        }
        self.streams += 1;
        self.bytes += bytes.len();
        *self.global_bytes += bytes.len();
        let content =
            Content::decode(bytes).map_err(|_| "content operators could not be decoded")?;
        if content.operations.len() > MAX_CONTENT_OPERATIONS {
            return Err("operator filtering exceeded its operation budget");
        }
        let mut operations = Vec::with_capacity(content.operations.len());
        let mut states = Vec::new();
        let mut changed = false;
        for mut op in content.operations {
            match op.operator.as_str() {
                "q" => {
                    if states.len() >= 1024 {
                        return Err("graphics-state nesting exceeded its limit");
                    }
                    states.push(state);
                }
                "Q" => {
                    if let Some(saved) = states.pop() {
                        state = saved;
                    } else {
                        self.report
                            .notes
                            .insert("an unmatched graphics-state restore was encountered");
                    }
                }
                "Tj" | "TJ" | "'" | "\"" => {
                    if let Some(unknown) = state.unknown() {
                        self.report.notes.insert(unknown);
                    }
                    let Some(reason) = state.suspicious() else {
                        if let Pass::Probe = self.pass {
                            operations.push(Operation::new("g", vec![0.into()]));
                            operations.push(op);
                            changed = true;
                            continue;
                        }
                        operations.push(op);
                        continue;
                    };
                    let index = self.kinds.len();
                    self.kinds.push(reason);
                    match self.pass {
                        Pass::Probe => {
                            let color = probe_color(index)
                                .map_or([0.0; 3], |c| c.map(|v| f32::from(v) / 255.0));
                            operations.push(Operation::new(
                                "rg",
                                color.iter().map(|&v| Object::Real(v)).collect(),
                            ));
                            if matches!(state.render_mode, 3 | 7) {
                                operations.push(Operation::new("Tr", vec![0.into()]));
                                operations.push(op);
                                operations
                                    .push(Operation::new("Tr", vec![state.render_mode.into()]));
                            } else {
                                operations.push(op);
                            }
                            changed = true;
                            continue;
                        }
                        Pass::Filter(hidden) => {
                            if matches!(state.render_mode, 3 | 7) {
                                // Already omitted by the native reader.
                            } else if reason == "white text" && state.unknown().is_some() {
                                self.report
                                    .notes
                                    .insert("white text in an unknown paint was retained");
                            } else if hidden.contains(&index) {
                                let mode = if state.render_mode >= 4 { 7 } else { 3 };
                                operations.push(Operation::new("Tr", vec![mode.into()]));
                                operations.push(op);
                                operations
                                    .push(Operation::new("Tr", vec![state.render_mode.into()]));
                                self.report.filtered += 1;
                                changed = true;
                                continue;
                            }
                        }
                    }
                }
                "Do" => {
                    let name = op.operands.first().and_then(|v| v.as_name().ok());
                    let object = name.and_then(|name| {
                        named_resource(self.source, &[resources], b"XObject", name)
                    });
                    if let Some(id) = object.and_then(|v| v.as_reference().ok())
                        && let Ok(stream) = self.source.get_object(id).and_then(Object::as_stream)
                        && stream.dict.get(b"Subtype").and_then(Object::as_name).ok()
                            == Some(b"Form")
                    {
                        if !seen.insert(id) {
                            return Err("a recursive Form invocation could not be filtered");
                        }
                        let mut form_resources = if stream.dict.has(b"Resources") {
                            self.source
                                .get_dict_in_dict(&stream.dict, b"Resources")
                                .map_err(|_| "the Form Resources entry is invalid")?
                                .clone()
                        } else {
                            resources.clone()
                        };
                        let mut inherited = state;
                        if let Some(scale) = stream
                            .dict
                            .get(b"Matrix")
                            .and_then(Object::as_array)
                            .ok()
                            .and_then(|v| matrix_scale(v))
                        {
                            inherited.ctm_scale *= scale;
                        }
                        let bytes = decoded(stream).map_err(|_| "a Form could not be decoded")?;
                        let (content, form_changed) =
                            self.content(&bytes, &mut form_resources, inherited, seen, depth + 1)?;
                        seen.remove(&id);
                        if form_changed {
                            let mut dictionary = stream.dict.clone();
                            dictionary.remove(b"Filter");
                            dictionary.remove(b"DecodeParms");
                            dictionary.set("Resources", form_resources);
                            let stream = Stream::new(
                                dictionary,
                                content
                                    .encode()
                                    .map_err(|_| "a Form could not be encoded")?,
                            );
                            let id = self.output.add_object(stream);
                            let mut xobjects = resources
                                .get(b"XObject")
                                .ok()
                                .and_then(|v| super::resolved(self.source, v).ok())
                                .and_then(|v| v.as_dict().ok())
                                .cloned()
                                .unwrap_or_default();
                            let name = loop {
                                self.serial += 1;
                                let name = format!("MarkitaiSanitize{}", self.serial);
                                if !xobjects.has(name.as_bytes()) {
                                    break name;
                                }
                            };
                            xobjects.set(name.clone(), id);
                            resources.set("XObject", xobjects);
                            op.operands[0] = Object::Name(name.into_bytes());
                            changed = true;
                        }
                    }
                }
                _ => state.apply_paint(self.source, &[resources], &op),
            }
            operations.push(op);
        }
        if !states.is_empty() {
            self.report
                .notes
                .insert("an unmatched graphics-state save was encountered");
        }
        Ok((Content { operations }, changed))
    }
}

struct BoundedBuffer {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other(
                "temporary filtered PDF exceeds its byte limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod position_tests {
    use super::*;
    #[test]
    fn hidden_show_operators_preserve_following_visible_glyph_geometry_and_font() {
        let content=b"BT /F1 12 Tf 1 0 0 1 40 650 Tm (Before ) Tj /Zero gs [(SECRET) -150 (OFFSET)] TJ /Opaque gs /F2 12 Tf (VISIBLE AFTER) Tj ET";
        let bytes = super::super::sanitize_tests::fixture(&[content.to_vec()], &[]);
        let source = Document::load_mem(&bytes).unwrap();
        let page = source.get_pages()[&1];
        let normalized = normalize_inherited_resources(&source, bytes.len())
            .unwrap()
            .unwrap();
        let original = pdf_inspector::extract_text_with_positions_mem(&normalized).unwrap();
        let mut output = source.clone();
        let mut resources = page_resources(&source, page).unwrap().unwrap().clone();
        let mut total = 0;
        let mut report = Report::default();
        let hidden = BTreeSet::from([0, 1]);
        let (rewritten, changed) = Rewriter::new(
            &source,
            &mut output,
            &mut total,
            &mut report,
            Pass::Filter(&hidden),
        )
        .content(
            content,
            &mut resources,
            GraphicsState::default(),
            &mut BTreeSet::new(),
            0,
        )
        .unwrap();
        assert!(changed);
        let stream = output.add_object(Stream::new(Dictionary::new(), rewritten.encode().unwrap()));
        let dictionary = output.get_dictionary_mut(page).unwrap();
        dictionary.set("Contents", stream);
        dictionary.set("Resources", resources);
        let mut filtered = Vec::new();
        output.save_to(&mut filtered).unwrap();
        let result = pdf_inspector::extract_text_with_positions_mem(&filtered).unwrap();
        assert!(
            !result
                .iter()
                .any(|v| v.text.contains("SECRET") || v.text.contains("OFFSET"))
        );
        let before = original.iter().find(|v| v.text == "VISIBLE AFTER").unwrap();
        let after = result.iter().find(|v| v.text == "VISIBLE AFTER").unwrap();
        assert!((before.x - after.x).abs() < 0.001 && (before.y - after.y).abs() < 0.001);
        assert!(
            (before.width - after.width).abs() < 0.001
                && (before.height - after.height).abs() < 0.001
        );
        assert_eq!(before.font, after.font);
        assert!(after.font.contains("Bold"));
    }
}
