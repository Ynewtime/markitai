use crate::{Asset, Document, Error, Result};
use lopdf::{Dictionary, Object, ObjectId, Stream, content::Content};
use std::collections::{BTreeMap, BTreeSet, HashSet};

#[path = "pdf/continued.rs"]
mod continued;
#[path = "pdf/geometry.rs"]
mod geometry;
#[path = "pdf/layout.rs"]
mod layout;
#[cfg(test)]
#[path = "pdf/page_tests.rs"]
mod page_tests;
#[cfg(test)]
#[path = "pdf/policy_tests.rs"]
mod policy_tests;

const MAX_STREAM_BYTES: usize = 64 * 1024 * 1024;
const MAX_ASSET_BYTES: usize = 128 * 1024 * 1024;
const MAX_IMAGE_PIXELS: usize = 32 * 1024 * 1024;

fn conversion(message: impl std::fmt::Display) -> Error {
    Error::Conversion(format!("Native PDF conversion failed: {message}"))
}

fn decoded(stream: &Stream) -> std::result::Result<Vec<u8>, String> {
    if !stream.dict.has(b"Filter") {
        if stream.content.len() > MAX_STREAM_BYTES {
            return Err("image stream exceeds the size limit".into());
        }
        return Ok(stream.content.clone());
    }
    stream
        .decompressed_content_with_limit(MAX_STREAM_BYTES)
        .map_err(|e| e.to_string())
}

/// Colour spaces encoded without colour conversion. ICC-based and calibrated
/// spaces use their component count, as their alternate device space would;
/// the profile itself is not embedded.
#[derive(Debug, PartialEq)]
enum Space {
    Gray,
    Rgb,
    /// RGB palette of `hival + 1` entries.
    Indexed(Vec<u8>),
}

fn resolved<'a>(
    pdf: &'a lopdf::Document,
    value: &'a Object,
) -> std::result::Result<&'a Object, String> {
    pdf.dereference(value)
        .map(|(_, value)| value)
        .map_err(|e| e.to_string())
}

fn color_space(
    pdf: &lopdf::Document,
    value: &Object,
    indexed: bool,
) -> std::result::Result<Space, String> {
    let value = resolved(pdf, value)?;
    if let Ok(name) = value.as_name() {
        return match name {
            b"DeviceRGB" => Ok(Space::Rgb),
            b"DeviceGray" => Ok(Space::Gray),
            _ => Err("image color space requires color conversion".into()),
        };
    }
    let items = value
        .as_array()
        .map_err(|_| "image color space is malformed")?;
    let family = items
        .first()
        .and_then(|family| family.as_name().ok())
        .ok_or("image color space is malformed")?;
    match (family, items.len()) {
        (b"ICCBased", 2) => {
            let profile = resolved(pdf, &items[1])?
                .as_stream()
                .map_err(|_| "ICC profile is not a stream")?;
            match profile.dict.get(b"N").and_then(Object::as_i64) {
                Ok(1) => Ok(Space::Gray),
                Ok(3) => Ok(Space::Rgb),
                _ => Err("image color space requires color conversion".into()),
            }
        }
        (b"CalGray", 2) => Ok(Space::Gray),
        (b"CalRGB", 2) => Ok(Space::Rgb),
        (b"Indexed", 4) if !indexed => {
            let base = color_space(pdf, &items[1], true)?;
            let hival = resolved(pdf, &items[2])?
                .as_i64()
                .ok()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|n| *n <= 255)
                .ok_or("indexed color space has an invalid maximum index")?;
            let lookup = match resolved(pdf, &items[3])? {
                Object::String(bytes, _) => bytes.clone(),
                Object::Stream(stream) => decoded(stream)?,
                _ => return Err("indexed color space lookup is malformed".into()),
            };
            let channels = if base == Space::Gray { 1 } else { 3 };
            let entries = lookup
                .get(..(hival + 1) * channels)
                .ok_or("indexed color space lookup is shorter than its entries")?;
            Ok(Space::Indexed(if channels == 1 {
                entries.iter().flat_map(|&v| [v, v, v]).collect()
            } else {
                entries.to_vec()
            }))
        }
        _ => Err("image color space requires color conversion".into()),
    }
}

fn image_bytes(
    pdf: &lopdf::Document,
    stream: &Stream,
) -> std::result::Result<(&'static str, Vec<u8>), String> {
    let dict = &stream.dict;
    if dict.has(b"SMask")
        || dict.has(b"Mask")
        || dict
            .get(b"ImageMask")
            .and_then(Object::as_bool)
            .unwrap_or(false)
    {
        return Err("image transparency or stencil masks require compositing".into());
    }
    if dict.has(b"Decode") {
        return Err("image sample remapping is not implemented".into());
    }
    let filters = if dict.has(b"Filter") {
        stream.filters().map_err(|e| e.to_string())?
    } else {
        Vec::new()
    };
    if filters.as_slice() == [b"DCTDecode".as_slice()] {
        if stream.content.len() > MAX_STREAM_BYTES {
            return Err("JPEG stream exceeds the size limit".into());
        }
        return Ok(("jpg", stream.content.clone()));
    }
    if filters.iter().any(|f| {
        !matches!(
            *f,
            b"FlateDecode"
                | b"ASCII85Decode"
                | b"ASCIIHexDecode"
                | b"LZWDecode"
                | b"RunLengthDecode"
        )
    }) {
        return Err(format!(
            "unsupported image filters: {}",
            filters
                .iter()
                .map(|name| String::from_utf8_lossy(name))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let width = dict
        .get(b"Width")
        .and_then(Object::as_i64)
        .map_err(|e| e.to_string())?;
    let height = dict
        .get(b"Height")
        .and_then(Object::as_i64)
        .map_err(|e| e.to_string())?;
    let width = u32::try_from(width)
        .ok()
        .filter(|n| *n > 0)
        .ok_or("invalid image width")?;
    let height = u32::try_from(height)
        .ok()
        .filter(|n| *n > 0)
        .ok_or("invalid image height")?;
    (width as usize)
        .checked_mul(height as usize)
        .filter(|n| *n <= MAX_IMAGE_PIXELS)
        .ok_or("image dimensions exceed the pixel limit")?;
    let bits = dict
        .get(b"BitsPerComponent")
        .and_then(Object::as_i64)
        .unwrap_or(0);
    let space = color_space(
        pdf,
        dict.get(b"ColorSpace")
            .map_err(|_| "image color space is missing")?,
        false,
    )?;
    let (color, channels, depth) = match (&space, bits) {
        (Space::Rgb, 8) => (png::ColorType::Rgb, 3, png::BitDepth::Eight),
        (Space::Gray, 8) => (png::ColorType::Grayscale, 1, png::BitDepth::Eight),
        (Space::Indexed(_), 1) => (png::ColorType::Indexed, 1, png::BitDepth::One),
        (Space::Indexed(_), 2) => (png::ColorType::Indexed, 1, png::BitDepth::Two),
        (Space::Indexed(_), 4) => (png::ColorType::Indexed, 1, png::BitDepth::Four),
        (Space::Indexed(_), 8) => (png::ColorType::Indexed, 1, png::BitDepth::Eight),
        _ => {
            return Err(
                "only 8-bit direct and 1/2/4/8-bit indexed image samples are currently encoded"
                    .into(),
            );
        }
    };
    let samples = decoded(stream)?;
    // PDF and PNG rows both start on a byte boundary.
    let row = (width as usize)
        .checked_mul(channels * bits as usize)
        .map(|bits| bits.div_ceil(8))
        .ok_or("image sample count overflow")?;
    let expected = row
        .checked_mul(height as usize)
        .ok_or("image sample count overflow")?;
    if samples.len() != expected {
        return Err(format!(
            "image sample length {} does not match expected {expected}",
            samples.len()
        ));
    }
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(color);
        encoder.set_depth(depth);
        if let Space::Indexed(palette) = &space {
            // Indices above the maximum take its colour, as PDF clamps them;
            // entries the sample depth cannot address are not written.
            let mut palette = palette.clone();
            let last = palette[palette.len() - 3..].to_vec();
            while palette.len() < 3 << bits {
                palette.extend_from_slice(&last);
            }
            palette.truncate(3 << bits);
            encoder.set_palette(palette);
        }
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer
            .write_image_data(&samples)
            .map_err(|e| e.to_string())?;
    }
    Ok(("png", bytes))
}

#[derive(Clone, Copy)]
struct GraphicsState {
    render_mode: i64,
    font_size: f32,
    /// Area scale (square root of the determinant) of the transformation
    /// in force and of the text matrix: a producer that sets `1 Tf` and
    /// scales with `Tm` (Quartz) or `cm` still prints at the product.
    ctm_scale: f32,
    text_scale: f32,
    white_fill: bool,
    white_stroke: bool,
    invisible_alpha: bool,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            render_mode: 0,
            font_size: 12.0,
            ctm_scale: 1.0,
            text_scale: 1.0,
            white_fill: false,
            white_stroke: false,
            invisible_alpha: false,
        }
    }
}

impl GraphicsState {
    fn suspicious(self) -> Option<&'static str> {
        if matches!(self.render_mode, 3 | 7) {
            Some("invisible text rendering mode")
        } else if self.invisible_alpha {
            Some("transparent text graphics state")
        } else if (self.font_size * self.ctm_scale * self.text_scale).abs() <= 1.0 {
            Some("text at one point or smaller")
        } else if (matches!(self.render_mode, 0 | 4) && self.white_fill)
            || (matches!(self.render_mode, 1 | 5) && self.white_stroke)
            || (matches!(self.render_mode, 2 | 6) && self.white_fill && self.white_stroke)
        {
            Some("white text")
        } else {
            None
        }
    }
}

#[derive(Default)]
struct PageInspection {
    images: Vec<ObjectId>,
    signals: BTreeSet<&'static str>,
    warnings: Vec<String>,
    inspected_bytes: usize,
    inspected_streams: usize,
}

fn named_resource<'a>(
    pdf: &'a lopdf::Document,
    resources: &[&'a Dictionary],
    kind: &[u8],
    name: &[u8],
) -> Option<&'a Object> {
    resources
        .iter()
        .find_map(|resources| pdf.get_dict_in_dict(resources, kind).ok()?.get(name).ok())
}

fn inspect_content(
    pdf: &lopdf::Document,
    bytes: &[u8],
    resources: &[&Dictionary],
    state: GraphicsState,
    seen_forms: &mut BTreeSet<ObjectId>,
    depth: usize,
    out: &mut PageInspection,
) -> Option<Content> {
    if out.inspected_streams >= 256
        || bytes.len() > MAX_STREAM_BYTES.saturating_sub(out.inspected_bytes)
    {
        out.warnings
            .push("Expanded page/Form content exceeds the inspection budget.".into());
        return None;
    }
    out.inspected_bytes += bytes.len();
    out.inspected_streams += 1;
    if depth > 32 {
        out.warnings
            .push("Form nesting exceeds the inspection limit.".into());
        return None;
    }
    let content = match Content::decode(bytes) {
        Ok(content) => content,
        Err(error) => {
            out.warnings
                .push(format!("Content stream inspection failed: {error}"));
            return None;
        }
    };
    inspect_operations(pdf, &content, resources, state, seen_forms, depth, out);
    Some(content)
}

fn inspect_operations(
    pdf: &lopdf::Document,
    content: &Content,
    resources: &[&Dictionary],
    mut state: GraphicsState,
    seen_forms: &mut BTreeSet<ObjectId>,
    depth: usize,
    out: &mut PageInspection,
) {
    let mut states = Vec::new();
    for operation in &content.operations {
        let last = operation.operands.last();
        let name = last.and_then(|obj| obj.as_name().ok());
        match operation.operator.as_str() {
            "q" => {
                if states.len() >= 1024 {
                    out.warnings
                        .push("Graphics-state nesting exceeds the inspection limit.".into());
                    return;
                }
                states.push(state);
            }
            "Q" => {
                if let Some(saved) = states.pop() {
                    state = saved;
                }
            }
            "Tr" => {
                if let Some(mode) = last.and_then(|obj| obj.as_i64().ok()) {
                    state.render_mode = mode;
                }
            }
            "Tf" => {
                if let Some(size) = last.and_then(|obj| obj.as_float().ok()) {
                    state.font_size = size;
                }
            }
            "cm" => {
                if let Some(scale) = matrix_scale(&operation.operands) {
                    state.ctm_scale *= scale;
                }
            }
            "BT" => state.text_scale = 1.0,
            "Tm" => {
                if let Some(scale) = matrix_scale(&operation.operands) {
                    state.text_scale = scale;
                }
            }
            "g" => {
                state.white_fill = last
                    .and_then(|obj| obj.as_float().ok())
                    .is_some_and(|v| v >= 0.98)
            }
            "G" => {
                state.white_stroke = last
                    .and_then(|obj| obj.as_float().ok())
                    .is_some_and(|v| v >= 0.98)
            }
            "rg" => {
                state.white_fill = operation.operands.len() == 3
                    && operation
                        .operands
                        .iter()
                        .all(|obj| obj.as_float().is_ok_and(|v| v >= 0.98))
            }
            "RG" => {
                state.white_stroke = operation.operands.len() == 3
                    && operation
                        .operands
                        .iter()
                        .all(|obj| obj.as_float().is_ok_and(|v| v >= 0.98))
            }
            "k" => {
                state.white_fill = operation.operands.len() == 4
                    && operation
                        .operands
                        .iter()
                        .all(|obj| obj.as_float().is_ok_and(|v| v <= 0.02))
            }
            "K" => {
                state.white_stroke = operation.operands.len() == 4
                    && operation
                        .operands
                        .iter()
                        .all(|obj| obj.as_float().is_ok_and(|v| v <= 0.02))
            }
            "scn" | "SCN" if name.is_some() => {
                out.warnings.push(
                    "Pattern paint may contain raster content; pattern streams are not inspected."
                        .into(),
                );
            }
            "gs" => {
                if let Some(obj) =
                    name.and_then(|name| named_resource(pdf, resources, b"ExtGState", name))
                {
                    let resolved = match obj {
                        Object::Reference(id) => pdf.get_object(*id).ok(),
                        other => Some(other),
                    };
                    if let Some(dict) = resolved.and_then(|obj| obj.as_dict().ok()) {
                        state.invisible_alpha = dict
                            .get(b"ca")
                            .and_then(Object::as_float)
                            .is_ok_and(|v| v <= 0.01);
                    }
                }
            }
            "Tj" | "TJ" | "'" | "\"" => {
                if let Some(reason) = state.suspicious() {
                    out.signals.insert(reason);
                }
            }
            "BI" | "ID" => {
                out.warnings
                    .push("Inline PDF images are not extracted.".into());
            }
            "Do" => {
                let Some(obj) =
                    name.and_then(|name| named_resource(pdf, resources, b"XObject", name))
                else {
                    continue;
                };
                let Ok(id) = obj.as_reference() else {
                    out.warnings
                        .push("A direct XObject cannot be extracted.".into());
                    continue;
                };
                let Ok(stream) = pdf.get_object(id).and_then(Object::as_stream) else {
                    continue;
                };
                match stream.dict.get(b"Subtype").and_then(Object::as_name).ok() {
                    Some(b"Image") => {
                        if !out.images.contains(&id) {
                            out.images.push(id);
                        }
                    }
                    Some(b"Form") if seen_forms.insert(id) => {
                        let nested = pdf.get_dict_in_dict(&stream.dict, b"Resources").ok();
                        let form_resources: Vec<_> = nested
                            .into_iter()
                            .chain(resources.iter().copied())
                            .collect();
                        let mut form_state = state;
                        if let Some(scale) = stream
                            .dict
                            .get(b"Matrix")
                            .and_then(Object::as_array)
                            .ok()
                            .and_then(|matrix| matrix_scale(matrix))
                        {
                            form_state.ctm_scale *= scale;
                        }
                        match decoded(stream) {
                            Ok(bytes) => {
                                let _ = inspect_content(
                                    pdf,
                                    &bytes,
                                    &form_resources,
                                    form_state,
                                    seen_forms,
                                    depth + 1,
                                    out,
                                );
                            }
                            Err(error) => out
                                .warnings
                                .push(format!("Form stream inspection failed: {error}")),
                        }
                        seen_forms.remove(&id);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

/// The area scale of a six-number matrix: the square root of its 2×2
/// determinant's magnitude.
fn matrix_scale(operands: &[Object]) -> Option<f32> {
    let [a, b, c, d, ..] = operands else {
        return None;
    };
    let [a, b, c, d] = [a, b, c, d].map(|n| n.as_float().ok());
    let determinant = a? * d? - b? * c?;
    determinant.is_finite().then(|| determinant.abs().sqrt())
}

fn inspect_page(pdf: &lopdf::Document, id: ObjectId) -> (PageInspection, Option<Content>) {
    let mut out = PageInspection::default();
    let (direct, ids) = match pdf.get_page_resources(id) {
        Ok(resources) => resources,
        Err(error) => {
            out.warnings
                .push(format!("Page resource inspection failed: {error}"));
            return (out, None);
        }
    };
    let resources = direct
        .into_iter()
        .chain(ids.iter().filter_map(|id| pdf.get_dictionary(*id).ok()))
        .collect::<Vec<_>>();
    let content = match pdf.get_page_content_with_limit(id, MAX_STREAM_BYTES) {
        Ok(bytes) => inspect_content(
            pdf,
            &bytes,
            &resources,
            GraphicsState::default(),
            &mut BTreeSet::new(),
            0,
            &mut out,
        ),
        Err(error) => {
            out.warnings
                .push(format!("Page content inspection failed: {error}"));
            None
        }
    };
    (out, content)
}

/// A page's native body and reliability verdict, before generated output wrappers.
#[derive(Debug)]
pub(crate) struct PdfPage {
    pub number: usize,
    pub markdown: String,
    pub needs_ocr: bool,
    pub ocr_reason: Option<String>,
    pub asset_names: Vec<String>,
    pub asset_ocr: BTreeMap<String, String>,
    pub screenshot_name: Option<String>,
    pub visibility_suspect: bool,
    pub ocr_completed: bool,
    // Deferred missing-text diagnostics retain their original position among
    // inspection/image warnings. Callers may append warnings before finishing.
    warning_index: usize,
    /// The native body starts with a table that layout geometry found to
    /// continue the table ending the previous page's body.
    continues_table: bool,
}

#[derive(Debug)]
pub(crate) struct PdfPages {
    pub pages: Vec<PdfPage>,
    pub document: Document,
}

impl PdfPages {
    /// Assemble the default reader output and reject a document with no content.
    pub(crate) fn finish(self) -> Result<Document> {
        self.assemble(false)
    }

    /// The caller has rendered pages or completed OCR, including blank results.
    pub(crate) fn finish_with_media(self) -> Result<Document> {
        self.assemble(true)
    }

    fn assemble(self, allow_empty: bool) -> Result<Document> {
        let Self {
            mut pages,
            mut document,
        } = self;
        let readable: Vec<bool> = pages
            .iter()
            .map(|page| !page.needs_ocr && !page.markdown.trim().is_empty())
            .collect();
        continued::join_tables(&mut pages);
        let mut sections = Vec::with_capacity(pages.len());
        let mut readable_pages = 0;
        let mut deferred = Vec::new();
        for (page, readable) in pages.into_iter().zip(readable) {
            readable_pages += usize::from(readable);
            let mut warning = Vec::new();
            // A page whose text all continued the previous page's table
            // keeps only its marker.
            let mut section = if readable && page.markdown.trim().is_empty() {
                format!("<!-- Page number: {} -->", page.number)
            } else {
                page_markdown(&page, &mut warning)
            };
            if let Some(warning) = warning.pop() {
                deferred.push((page.warning_index, warning));
            }
            for name in page.asset_names {
                section.push_str(&format!(
                    "\n\n![Image on page {}](.markitai/assets/{name})",
                    page.number
                ));
                if let Some(text) = page
                    .asset_ocr
                    .get(&name)
                    .filter(|text| !text.trim().is_empty())
                {
                    section.push_str("\n\n");
                    section.push_str(text.trim());
                }
            }
            if let Some(name) = page.screenshot_name {
                section.push_str("\n\n");
                section.push_str(&screenshot_reference(page.number, &name));
            }
            sections.push(section);
        }
        // Stable ordering also handles multiple page diagnostics sharing an
        // offset. The merge moves strings once instead of repeated Vec inserts.
        deferred.sort_by_key(|(index, _)| *index);
        let mut deferred = deferred.into_iter().peekable();
        let mut warnings = Vec::with_capacity(document.warnings.len() + deferred.len());
        for (index, warning) in document.warnings.into_iter().enumerate() {
            while deferred.peek().is_some_and(|(offset, _)| *offset <= index) {
                warnings.push(deferred.next().expect("peeked diagnostic").1);
            }
            warnings.push(warning);
        }
        warnings.extend(deferred.map(|(_, warning)| warning));
        document.warnings = warnings;
        if !allow_empty && readable_pages == 0 && document.assets.is_empty() {
            return Err(conversion(format!(
                "no reliable native text or extractable images; {}",
                document.warnings.join(" ")
            )));
        }
        document.markdown = sections.join("\n\n");
        // Only a document with extracted images has images out of place.
        if !document.assets.is_empty() {
            document
                .warnings
                .push(crate::pdf_media::IMAGE_PLACEMENT.into());
        }
        Ok(document)
    }
}

pub(crate) fn screenshot_reference(page: usize, name: &str) -> String {
    format!(
        "<!-- ![Page {page}](.markitai/screenshots/{}) -->",
        screenshot_destination(name)
    )
}

// Screenshot names are filesystem basenames, not pre-escaped URLs. Encode the
// raw UTF-8 bytes once, including URI delimiters and HTML comment terminators.
fn screenshot_destination(name: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut output = String::with_capacity(name.len());
    for byte in name.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            output.push(byte as char);
        } else {
            output.push('%');
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 15) as usize] as char);
        }
    }
    output
}

/// One bold span across a line wrap: the page reader closes and reopens bold
/// at each physical line (`**wrapped** **text**`). Bold-italic markers,
/// whitespace-adjacent markers and fenced code are left alone.
fn join_bold_runs(markdown: &str) -> std::borrow::Cow<'_, str> {
    if !markdown.contains("** **") {
        return std::borrow::Cow::Borrowed(markdown);
    }
    let mut output = String::with_capacity(markdown.len());
    let mut fenced = false;
    for (index, line) in markdown.split('\n').enumerate() {
        if index > 0 {
            output.push('\n');
        }
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
        }
        if fenced || !line.contains("** **") {
            output.push_str(line);
            continue;
        }
        let mut rest = line;
        while let Some(at) = rest.find("** **") {
            let before = rest[..at].chars().next_back();
            let after = rest[at + 5..].chars().next();
            let joinable = |ch: Option<char>| ch.is_some_and(|ch| !ch.is_whitespace() && ch != '*');
            output.push_str(&rest[..at]);
            output.push_str(if joinable(before) && joinable(after) {
                " "
            } else {
                "** **"
            });
            rest = &rest[at + 5..];
        }
        output.push_str(rest);
    }
    std::borrow::Cow::Owned(output)
}

fn page_markdown(page: &PdfPage, warnings: &mut Vec<String>) -> String {
    let number = page.number;
    let marker = format!("<!-- Page number: {number} -->");
    if page.ocr_completed && page.markdown.trim().is_empty() {
        return marker;
    }
    if page.needs_ocr || page.markdown.trim().is_empty() {
        let reason = page
            .ocr_reason
            .as_deref()
            .unwrap_or("empty native text extraction");
        warnings.push(format!("PDF page {number}: native text was not recovered ({reason}); OCR is required for this page."));
        marker
    } else {
        format!("{marker}\n\n{}", join_bold_runs(page.markdown.trim()))
    }
}

fn readable_fallback(text: &str) -> bool {
    if text.len() > 1024 * 1024
        || text.chars().any(|ch| {
            ch == '\u{fffd}'
                || (ch.is_control() && !matches!(ch, '\n' | '\r' | '\t'))
                || matches!(ch as u32, 0xe000..=0xf8ff | 0xf0000..=0xffffd | 0x100000..=0x10fffd)
        })
    {
        return false;
    }
    let visible = text.chars().filter(|ch| !ch.is_whitespace()).count();
    let alphanumeric = text.chars().filter(|ch| ch.is_alphanumeric()).count();
    alphanumeric >= 20 && alphanumeric * 100 >= visible * 55
}

fn recover_plain_text(
    pdf: &lopdf::Document,
    number: u32,
    page_id: ObjectId,
    page: &mut pdf_inspector::PageMarkdown,
    inspection: &PageInspection,
    warnings: &mut Vec<String>,
) {
    // The layout API also rejects pages for unnamed formatting/GID heuristics.
    // An unused image resource can also cause a false scan verdict. Only recover
    // when executed content has no raster, hidden text or uninspected streams.
    if !page.needs_ocr
        || page
            .ocr_reason
            .as_deref()
            .is_some_and(|reason| reason != "scanned")
        || !inspection.signals.is_empty()
        || !inspection.warnings.is_empty()
        || !inspection.images.is_empty()
    {
        return;
    }
    let Ok(fonts) = pdf.get_page_fonts(page_id) else {
        return;
    };
    if fonts.values().any(|font| {
        font.get(b"Subtype")
            .and_then(Object::as_name)
            .is_ok_and(|name| name == b"Type3")
    }) {
        return;
    }
    let Ok(text) = pdf.extract_text_with_limit(&[number], MAX_STREAM_BYTES) else {
        return;
    };
    if !readable_fallback(&text) {
        return;
    }
    page.markdown = super::escape(text.trim());
    page.needs_ocr = false;
    let reason = if page.ocr_reason.as_deref() == Some("scanned") {
        "a scan verdict from declared image resources, with no executed raster content"
    } else {
        "an unnamed layout/GID verdict"
    };
    warnings.push(format!("PDF page {number}: recovered bounded font-decoded text after {reason}. Reading order, paragraph boundaries and text styling may differ."));
}

/// A page's bullet-sized marks as the page reader is given them, in the
/// page's own coordinates (the reader's text is not moved to the page box).
/// A mark inside a ruled table is a status dot or an icon in a cell, not a
/// list bullet.
fn reader_marks(
    frame: geometry::Frame,
    grids: &[geometry::Grid],
    marks: &[geometry::Mark],
) -> Vec<geometry::Mark> {
    let in_table = |mark: &geometry::Mark| {
        grids.iter().any(|grid| {
            mark.x0 >= grid.xs[0]
                && mark.x1 <= grid.xs[grid.xs.len() - 1]
                && mark.y0 >= grid.ys[0]
                && mark.y1 <= grid.ys[grid.ys.len() - 1]
        })
    };
    marks
        .iter()
        .filter(|mark| !in_table(mark))
        .map(|mark| geometry::Mark {
            x0: mark.x0 + frame.x,
            y0: mark.y0 + frame.y,
            x1: mark.x1 + frame.x,
            y1: mark.y1 + frame.y,
            ..*mark
        })
        .collect()
}

/// Pages holding cells of a table in the structure tree: the page reader reads
/// those tables from the tags, which layout geometry does not override.
fn tagged_table_pages(pdf: &lopdf::Document, pages: &BTreeMap<u32, ObjectId>) -> HashSet<u32> {
    pdf_inspector::structure_tree::StructTree::from_doc(pdf)
        .map(|tree| tree.extract_tables(pages))
        .unwrap_or_default()
        .iter()
        .flat_map(|table| &table.rows)
        .flat_map(|row| &row.cells)
        .flat_map(|cell| cell.mcids.iter().map(|&(_, page)| page))
        .collect()
}

pub(super) fn extract(bytes: &[u8]) -> Result<Document> {
    extract_pages(bytes)?.finish()
}

pub(crate) fn extract_pages(bytes: &[u8]) -> Result<PdfPages> {
    extract_pages_inner(bytes, None)
}

/// Bound page processing before invoking the native text/layout reader.
pub(crate) fn extract_pages_bounded(bytes: &[u8], max_pages: usize) -> Result<PdfPages> {
    extract_pages_inner(bytes, Some(max_pages))
}

fn extract_pages_inner(bytes: &[u8], max_pages: Option<usize>) -> Result<PdfPages> {
    // One parse of the file serves the page reader, the layout reader and,
    // when the reader's document is what lopdf alone makes of the bytes,
    // this module's own inspection; otherwise that inspection loads the
    // bytes itself, unrepaired, as it always has.
    let loaded = pdf_inspector::LoadedPdf::load_mem(bytes);
    let unrepaired;
    let pdf = match loaded
        .as_ref()
        .ok()
        .and_then(pdf_inspector::LoadedPdf::as_loaded_by_lopdf)
    {
        Some(pdf) => pdf,
        None => {
            unrepaired = lopdf::Document::load_mem(bytes).map_err(conversion)?;
            &unrepaired
        }
    };
    let page_ids = pdf.get_pages();
    if page_ids.is_empty() {
        return Err(conversion("document contains no pages"));
    }
    if let Some(limit) = max_pages.filter(|&limit| page_ids.len() > limit) {
        return Err(Error::InvalidInput(format!(
            "PDF page count exceeds the {limit}-page limit"
        )));
    }
    let mut document = Document::default();
    // Each page's content is inspected first: the shapes of a page whose
    // inspection is clean (its rule grids and bullet-sized marks) serve the
    // layout reader, and its painted list bullets the page reader too.
    let mut inspections = BTreeMap::new();
    let mut shapes = BTreeMap::new();
    for (&number, &id) in &page_ids {
        let (inspection, content) = inspect_page(pdf, id);
        if inspection.signals.is_empty()
            && inspection.warnings.is_empty()
            && let Some(frame) = geometry::frame(pdf, id)
            && let Some(content) = content.as_ref()
        {
            let resources = geometry::rule_resources(pdf, id);
            let (grids, marks) = geometry::page_shapes(content, frame, &resources);
            shapes.insert(number, (frame, grids, marks));
        }
        // Retain only bounded table coordinates across pages, never their
        // expanded streams or parsed operation trees.
        inspections.insert(number, inspection);
    }
    let painted = |page: u32| {
        shapes
            .get(&page)
            .map(|(frame, grids, marks)| reader_marks(*frame, grids, marks))
            .unwrap_or_default()
    };
    // A file the reader cannot load fails each reading with the load's error.
    let whole = match &loaded {
        Ok(loaded) => loaded
            .pages_markdown_with_marks(None, &painted)
            .map_err(|error| error.to_string()),
        Err(error) => Err(error.to_string()),
    };
    let extracted = match whole {
        Ok(result) => result.pages,
        Err(error) => {
            document.warnings.push(format!(
                "Whole-document extraction failed ({error}); pages were retried independently."
            ));
            page_ids
                .keys()
                .map(|&page| {
                    loaded
                        .as_ref()
                        .ok()
                        .and_then(|loaded| {
                            loaded
                                .pages_markdown_with_marks(Some(&[page - 1]), &painted)
                                .ok()
                        })
                        .and_then(|mut result| result.pages.pop())
                        .unwrap_or(pdf_inspector::PageMarkdown {
                            page: page - 1,
                            markdown: String::new(),
                            needs_ocr: true,
                            ocr_reason: Some("page extraction failed".into()),
                        })
                })
                .collect()
        }
    };
    let mut pages = extracted
        .into_iter()
        .map(|page| (page.page + 1, page))
        .collect::<BTreeMap<_, _>>();
    let mut layout_pages = HashSet::new();
    let mut page_geometry = BTreeMap::new();
    for (&number, &id) in &page_ids {
        if let Some(page) = pages.get_mut(&number) {
            let inspection = &inspections[&number];
            recover_plain_text(pdf, number, id, page, inspection, &mut document.warnings);
            if !page.needs_ocr
                && !page.markdown.trim().is_empty()
                && let Some(geometry) = shapes.remove(&number)
            {
                layout_pages.insert(number);
                page_geometry.insert(number, geometry);
            }
        }
    }
    let mut layout = if layout_pages.is_empty() {
        None
    } else {
        match layout::Layout::read(loaded.as_ref().ok(), &layout_pages, &page_geometry) {
            Ok(layout) => Some(layout),
            Err(reason) => {
                document.warnings.push(format!("PDF layout refinement was skipped ({reason}); the original page reader's output is retained."));
                None
            }
        }
    };
    // The layout reader was the last to read the pages' text.
    if let Ok(loaded) = &loaded {
        loaded.forget_page_runs();
    }
    // Read only when a page shows a table drawn without rules.
    let tagged_tables = std::cell::OnceCell::new();
    let mut image_names = BTreeMap::<ObjectId, Option<String>>::new();
    let mut total_asset_bytes = 0;
    let mut extracted_pages = Vec::with_capacity(page_ids.len());
    for &number in page_ids.keys() {
        let mut page = pages
            .remove(&number)
            .unwrap_or(pdf_inspector::PageMarkdown {
                page: number - 1,
                markdown: String::new(),
                needs_ocr: true,
                ocr_reason: Some("page absent from extraction result".into()),
            });
        let inspection = inspections
            .remove(&number)
            .expect("every page was inspected");
        let tagged = || {
            tagged_tables
                .get_or_init(|| tagged_table_pages(pdf, &page_ids))
                .contains(&number)
        };
        let mut continues_table = false;
        if let Some((frame, grids, marks)) = page_geometry.remove(&number)
            && let Some((refined, continues)) = layout.as_mut().and_then(|layout| {
                layout.page(number, frame, grids, &marks, &page.markdown, &tagged)
            })
        {
            page.markdown = refined;
            continues_table = continues;
        }
        let warning_index = document.warnings.len();
        let visibility_suspect = !inspection.signals.is_empty();
        let mut asset_names = Vec::new();
        for warning in inspection.warnings {
            document
                .warnings
                .push(format!("PDF page {number}: {warning}"));
        }
        if !inspection.signals.is_empty() {
            document.warnings.push(format!("PDF page {number}: contains {}; the native reader applies its own visibility heuristics, and complete hidden-text filtering is not established.", inspection.signals.into_iter().collect::<Vec<_>>().join(", ")));
        }
        for image_id in inspection.images {
            let name = image_names.entry(image_id).or_insert_with(|| {
                let extracted = pdf
                    .get_object(image_id)
                    .and_then(Object::as_stream)
                    .map_err(|e| e.to_string())
                    .and_then(|stream| image_bytes(pdf, stream));
                match extracted {
                    Ok((extension, bytes))
                        if total_asset_bytes + bytes.len() <= MAX_ASSET_BYTES =>
                    {
                        total_asset_bytes += bytes.len();
                        let name = format!("pdf-image-{}-{}.{}", image_id.0, image_id.1, extension);
                        document.assets.push(Asset {
                            name: name.clone(),
                            bytes,
                        });
                        Some(name)
                    }
                    Ok(_) => {
                        document.warnings.push(format!(
                            "PDF page {number}: image {} exceeds the total asset size limit.",
                            image_id.0
                        ));
                        None
                    }
                    Err(error) => {
                        document.warnings.push(format!(
                            "PDF page {number}: image {} was not extracted ({error}).",
                            image_id.0
                        ));
                        None
                    }
                }
            });
            if let Some(name) = name {
                asset_names.push(name.clone());
            }
        }
        extracted_pages.push(PdfPage {
            number: number as usize,
            markdown: page.markdown,
            needs_ocr: page.needs_ocr,
            ocr_reason: page.ocr_reason,
            asset_names,
            asset_ocr: BTreeMap::new(),
            screenshot_name: None,
            visibility_suspect,
            ocr_completed: false,
            warning_index,
            continues_table,
        });
    }
    document
        .metadata
        .insert("converter".into(), "pdf-inspector".into());
    if let Ok(info) = pdf
        .trailer
        .get(b"Info")
        .and_then(Object::as_reference)
        .and_then(|id| pdf.get_dictionary(id))
        && let Ok(title) = info.get(b"Title").and_then(lopdf::decode_text_string)
        && !title.trim().is_empty()
    {
        document
            .metadata
            .insert("title".into(), title.trim().into());
    }
    Ok(PdfPages {
        pages: extracted_pages,
        document,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{content::Operation, dictionary};

    #[test]
    fn compressed_page_streams_share_state_and_keep_complete_grid_geometry() {
        let mut pdf = lopdf::Document::with_version("1.7");
        let chunks: &[&[u8]] = &[
            b"q 1 0 0 1 10 20 cm 0 g 0 G 20 50 m 220 50 l S 20 100 m 220 100 l S",
            b"20 150 m 220 150 l S 20 50 m 20 150 l S 100 50 m 100 150 l S 220 50 m 220 150 l S Q",
        ];
        let mut expanded_bytes = 0;
        let contents = chunks
            .iter()
            .map(|chunk| {
                let mut bytes = b"% repeated padding makes compression worthwhile\n".repeat(16);
                bytes.extend_from_slice(chunk);
                expanded_bytes += bytes.len() + 1;
                let mut stream = Stream::new(Dictionary::new(), bytes);
                stream.compress().unwrap();
                assert_eq!(
                    stream.dict.get(b"Filter").unwrap().as_name().unwrap(),
                    b"FlateDecode"
                );
                Object::Reference(pdf.add_object(stream))
            })
            .collect::<Vec<_>>();
        let page = pdf.add_object(dictionary! {
            "Type" => "Page", "Contents" => contents,
            "Resources" => dictionary! {},
            "MediaBox" => vec![10.into(), 20.into(), 310.into(), 420.into()]
        });
        let (inspection, content) = inspect_page(&pdf, page);
        assert!(inspection.warnings.is_empty());
        assert!(inspection.signals.is_empty());
        assert_eq!(inspection.inspected_bytes, expanded_bytes);
        assert_eq!(inspection.inspected_streams, 1);
        let grids = geometry::grids(
            &content.unwrap(),
            geometry::frame(&pdf, page).unwrap(),
            &geometry::rule_resources(&pdf, page),
        );
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0].xs, [20., 100., 220.]);
        assert_eq!(grids[0].ys, [50., 100., 150.]);
    }

    #[test]
    fn inspection_budget_accepts_the_boundary_and_rejects_further_content() {
        let pdf = lopdf::Document::with_version("1.7");
        let bytes = b"3 Tr (hidden) Tj";
        let mut exact = PageInspection {
            inspected_bytes: MAX_STREAM_BYTES - bytes.len(),
            inspected_streams: 255,
            ..Default::default()
        };
        assert!(
            inspect_content(
                &pdf,
                bytes,
                &[],
                GraphicsState::default(),
                &mut BTreeSet::new(),
                0,
                &mut exact
            )
            .is_some()
        );
        assert_eq!(exact.inspected_bytes, MAX_STREAM_BYTES);
        assert_eq!(exact.inspected_streams, 256);
        assert!(exact.signals.contains("invisible text rendering mode"));
        assert!(exact.warnings.is_empty());
        assert!(
            inspect_content(
                &pdf,
                b"q Q",
                &[],
                GraphicsState::default(),
                &mut BTreeSet::new(),
                0,
                &mut exact
            )
            .is_none()
        );
        assert_eq!(exact.inspected_bytes, MAX_STREAM_BYTES);
        assert_eq!(exact.inspected_streams, 256);
        assert_eq!(
            exact.warnings,
            ["Expanded page/Form content exceeds the inspection budget."]
        );

        for (inspected_bytes, inspected_streams) in
            [(MAX_STREAM_BYTES - bytes.len() + 1, 0), (0, 256)]
        {
            let mut limited = PageInspection {
                inspected_bytes,
                inspected_streams,
                ..Default::default()
            };
            assert!(
                inspect_content(
                    &pdf,
                    bytes,
                    &[],
                    GraphicsState::default(),
                    &mut BTreeSet::new(),
                    0,
                    &mut limited
                )
                .is_none()
            );
            assert!(limited.signals.is_empty());
            assert_eq!(limited.inspected_bytes, inspected_bytes);
            assert_eq!(limited.inspected_streams, inspected_streams);
            assert_eq!(
                limited.warnings,
                ["Expanded page/Form content exceeds the inspection budget."]
            );
        }
    }

    #[test]
    fn forms_at_depth_limit_are_inspected_and_deeper_forms_warn_without_geometry() {
        for count in [32, 33] {
            let mut pdf = lopdf::Document::with_version("1.7");
            let mut form = pdf.add_object(Stream::new(
                dictionary! { "Subtype" => "Form" },
                b"3 Tr (hidden) Tj".to_vec(),
            ));
            for _ in 1..count {
                form = pdf.add_object(Stream::new(dictionary! {
                    "Subtype" => "Form", "Resources" => dictionary! { "XObject" => dictionary! { "Fm" => form } }
                }, b"/Fm Do".to_vec()));
            }
            let content = pdf.add_object(Stream::new(Dictionary::new(), b"/Fm Do".to_vec()));
            let page = pdf.add_object(dictionary! {
                "Type" => "Page", "Contents" => content,
                "Resources" => dictionary! { "XObject" => dictionary! { "Fm" => form } },
                "MediaBox" => vec![0.into(), 0.into(), 300.into(), 400.into()]
            });
            let (inspection, content) = inspect_page(&pdf, page);
            assert_eq!(inspection.inspected_streams, count + 1);
            assert_eq!(
                inspection.signals.contains("invisible text rendering mode"),
                count == 32
            );
            if count == 32 {
                assert!(inspection.warnings.is_empty());
            } else {
                assert_eq!(
                    inspection.warnings,
                    ["Form nesting exceeds the inspection limit."]
                );
            }
            // Root geometry never treats a Form's partial graphics as a table.
            assert!(
                geometry::grids(
                    &content.unwrap(),
                    geometry::frame(&pdf, page).unwrap(),
                    &geometry::rule_resources(&pdf, page),
                )
                .is_empty()
            );
        }
    }

    #[test]
    fn unreadable_form_preserves_warning_and_prevents_text_recovery() {
        let mut pdf = lopdf::Document::with_version("1.7");
        let pages = pdf.new_object_id();
        let font = pdf.add_object(
            dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica" },
        );
        let form = pdf.add_object(Stream::new(
            dictionary! {
                "Subtype" => "Form", "Filter" => "UnsupportedFilter"
            },
            b"3 Tr (hidden) Tj".to_vec(),
        ));
        let contents = pdf.add_object(Stream::new(Dictionary::new(), b"BT /F1 12 Tf 40 700 Td (Readable native words alone must not bypass an uninspected form.) Tj ET /Fm Do".to_vec()));
        let id = pdf.add_object(dictionary! {
            "Type" => "Page", "Parent" => pages, "Contents" => contents,
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font }, "XObject" => dictionary! { "Fm" => form } }
        });
        pdf.objects.insert(pages, dictionary! { "Type" => "Pages", "Kids" => vec![id.into()], "Count" => 1, "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()] }.into());
        let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
        pdf.trailer.set("Root", catalog);
        let (inspection, content) = inspect_page(&pdf, id);
        assert!(content.is_some());
        assert_eq!(inspection.warnings.len(), 1);
        assert!(inspection.warnings[0].starts_with("Form stream inspection failed:"));
        let mut page = pdf_inspector::PageMarkdown {
            page: 0,
            markdown: "Retain the dependency verdict without speculative recovery.".into(),
            needs_ocr: true,
            ocr_reason: Some("scanned".into()),
        };
        let before = page.markdown.clone();
        let mut warnings = Vec::new();
        recover_plain_text(&pdf, 1, id, &mut page, &inspection, &mut warnings);
        assert!(page.needs_ocr);
        assert_eq!(page.markdown, before);
        assert!(warnings.is_empty());
        // The words are recoverable: it is specifically the incomplete Form
        // inspection that must keep this page on its warned OCR path.
        recover_plain_text(
            &pdf,
            1,
            id,
            &mut page,
            &PageInspection::default(),
            &mut warnings,
        );
        assert!(!page.needs_ocr);
        assert!(page.markdown.contains("Readable native words"));
    }

    #[test]
    fn text_size_is_judged_after_the_text_matrix_and_transformation() {
        // Quartz writes `1 Tf` and scales with the text matrix: 12pt text.
        // A 12pt font under a 0.05 transformation prints at 0.6pt.
        let signals = |operations: Vec<Operation>| {
            let mut pdf = lopdf::Document::with_version("1.7");
            let pages_id = pdf.new_object_id();
            let font = pdf.add_object(
                dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica" },
            );
            let resources = pdf.add_object(dictionary! { "Font" => dictionary! { "F1" => font } });
            let content = Content { operations }.encode().unwrap();
            let text_id = pdf.add_object(Stream::new(Dictionary::new(), content));
            let page = pdf.add_object(dictionary! { "Type" => "Page", "Parent" => pages_id, "Contents" => text_id, "Resources" => resources });
            pdf.objects.insert(pages_id, dictionary! { "Type" => "Pages", "Kids" => vec![page.into()], "Count" => 1, "MediaBox" => vec![0.into(),0.into(),612.into(),792.into()] }.into());
            inspect_page(&pdf, page).0.signals
        };
        let text = |size: Object, matrix: Vec<Object>| {
            vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), size]),
                Operation::new("Tm", matrix),
                Operation::new("Tj", vec![Object::string_literal("Visible words")]),
                Operation::new("ET", vec![]),
            ]
        };
        let quartz = text(
            1.into(),
            vec![
                12.into(),
                0.into(),
                0.into(),
                Object::Real(-12.0),
                72.into(),
                700.into(),
            ],
        );
        assert!(signals(quartz).is_empty());
        let mut shrunk = vec![Operation::new(
            "cm",
            vec![
                Object::Real(0.05),
                0.into(),
                0.into(),
                Object::Real(0.05),
                0.into(),
                0.into(),
            ],
        )];
        shrunk.extend(text(
            12.into(),
            vec![
                1.into(),
                0.into(),
                0.into(),
                1.into(),
                72.into(),
                700.into(),
            ],
        ));
        assert!(signals(shrunk).contains("text at one point or smaller"));
        let tiny = text(
            Object::Real(0.5),
            vec![
                1.into(),
                0.into(),
                0.into(),
                1.into(),
                72.into(),
                700.into(),
            ],
        );
        assert!(signals(tiny).contains("text at one point or smaller"));
    }

    #[test]
    fn real_pdf_recovers_text_despite_unused_image_and_keeps_blank_page_warning() {
        let mut pdf = lopdf::Document::with_version("1.7");
        let pages_id = pdf.new_object_id();
        let font = pdf.add_object(
            dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica" },
        );
        let unused_image = pdf.add_object(Stream::new(dictionary! { "Type" => "XObject", "Subtype" => "Image", "Width" => 2000, "Height" => 2000, "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8, "Filter" => "DCTDecode" }, vec![0xff, 0xd8, 0xff, 0xd9]));
        let resources = pdf.add_object(dictionary! { "Font" => dictionary! { "F1" => font }, "XObject" => dictionary! { "Unused" => unused_image } });
        let content = Content { operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), 12.into()]),
            Operation::new("Td", vec![40.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal("This page has readable native text which must survive a blank second page.")]),
            Operation::new("ET", vec![]),
        ] }.encode().unwrap();
        let text_id = pdf.add_object(Stream::new(Dictionary::new(), content));
        let first = pdf.add_object(
            dictionary! { "Type" => "Page", "Parent" => pages_id, "Contents" => text_id, "Resources" => resources },
        );
        let second = pdf.add_object(dictionary! { "Type" => "Page", "Parent" => pages_id });
        pdf.objects.insert(pages_id, dictionary! { "Type" => "Pages", "Kids" => vec![first.into(), second.into()], "Count" => 2, "Resources" => resources, "MediaBox" => vec![0.into(),0.into(),612.into(),792.into()] }.into());
        let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        pdf.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        let initial = pdf_inspector::extract_pages_markdown_mem(&bytes, Some(&[0])).unwrap();
        assert!(initial.pages[0].needs_ocr);
        assert_eq!(initial.pages[0].ocr_reason.as_deref(), Some("scanned"));
        let result = extract(&bytes).unwrap();
        assert!(result.markdown.contains("readable native text"));
        assert!(result.assets.is_empty());
        assert!(result.warnings.iter().any(|warning| {
            warning.contains("declared image resources, with no executed raster content")
        }));
        assert!(result.markdown.contains("<!-- Page number: 2 -->"));
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.starts_with("PDF page 2: native text was not recovered"))
        );
    }
    #[test]
    fn partial_text_keeps_page_markers_and_precise_ocr_warning() {
        let mut warnings = Vec::new();
        let good = page_markdown(
            &PdfPage {
                number: 1,
                markdown: "Readable text".into(),
                needs_ocr: false,
                ocr_reason: None,
                asset_names: Vec::new(),
                asset_ocr: BTreeMap::new(),
                screenshot_name: None,
                visibility_suspect: false,
                ocr_completed: false,
                warning_index: 0,
                continues_table: false,
            },
            &mut warnings,
        );
        let missing = page_markdown(
            &PdfPage {
                number: 3,
                markdown: "unreliable".into(),
                needs_ocr: true,
                ocr_reason: Some("image-only".into()),
                asset_names: Vec::new(),
                asset_ocr: BTreeMap::new(),
                screenshot_name: None,
                visibility_suspect: false,
                ocr_completed: false,
                warning_index: 0,
                continues_table: false,
            },
            &mut warnings,
        );
        assert_eq!(good, "<!-- Page number: 1 -->\n\nReadable text");
        assert_eq!(missing, "<!-- Page number: 3 -->");
        assert_eq!(
            warnings,
            [
                "PDF page 3: native text was not recovered (image-only); OCR is required for this page."
            ]
        );
    }

    #[test]
    fn the_page_reader_gets_marks_outside_ruled_tables_in_page_coordinates() {
        let frame = geometry::Frame {
            x: 100.,
            y: 50.,
            width: 400.,
            height: 600.,
        };
        let grid = geometry::Grid {
            xs: vec![10., 60., 200.],
            ys: vec![300., 320., 340.],
        };
        let mark = |x0: f32, y0: f32| geometry::Mark {
            x0,
            y0,
            x1: x0 + 4.,
            y1: y0 + 4.,
            color: [0; 3],
        };
        // A dot in a cell stays out; a bullet beside the table moves into
        // the page's coordinates.
        assert_eq!(
            reader_marks(frame, &[grid], &[mark(14., 304.), mark(14., 360.)]),
            [mark(114., 410.)]
        );
    }

    #[test]
    fn decoded_fallback_rejects_empty_control_and_unmapped_glyphs() {
        assert!(readable_fallback(
            "Visible text with enough letters to recover a rejected page."
        ));
        assert!(readable_fallback(
            "这是具有足够原生文本的页面，恢复文本内容需要保留全部字符和清晰警告。"
        ));
        assert!(!readable_fallback(""));
        assert!(!readable_fallback("Short"));
        assert!(!readable_fallback(
            "A long but damaged text with \u{fffd} replacement glyph"
        ));
        assert!(!readable_fallback(
            "A long but damaged text with \u{e042} private glyph"
        ));
        assert!(!readable_fallback(
            "A long but damaged text with \u{1} control glyph"
        ));
    }
    #[test]
    fn rgb_image_preserves_pixels_and_masks_are_explicit() {
        let mut stream = Stream::new(
            dictionary! { "Width" => 2, "Height" => 1, "BitsPerComponent" => 8, "ColorSpace" => "DeviceRGB" },
            vec![255, 0, 0, 0, 0, 255],
        );
        let (extension, bytes) = image_bytes(&lopdf::Document::new(), &stream).unwrap();
        assert_eq!(extension, "png");
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut decoded).unwrap();
        assert_eq!(decoded, stream.content);
        stream.dict.set("SMask", Object::Reference((99, 0)));
        assert!(
            image_bytes(&lopdf::Document::new(), &stream)
                .unwrap_err()
                .contains("compositing")
        );
    }
    fn decode_png(bytes: Vec<u8>) -> (png::ColorType, Vec<u8>) {
        let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        decoder.set_transformations(png::Transformations::EXPAND);
        let mut reader = decoder.read_info().unwrap();
        let mut decoded = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut decoded).unwrap();
        decoded.truncate(info.buffer_size());
        (reader.info().color_type, decoded)
    }

    #[test]
    fn bold_spans_split_at_line_wraps_are_rejoined() {
        assert_eq!(
            join_bold_runs("Text **Vivamus ipsum cursus** **convallis. Maecenas.** More"),
            "Text **Vivamus ipsum cursus convallis. Maecenas.** More"
        );
        for kept in [
            "***bold italic*** ***next***",
            "**a** ** b**",
            "```\n**code** **stays**\n```",
            "plain text",
        ] {
            assert_eq!(join_bold_runs(kept), kept);
        }
    }

    #[test]
    fn icc_calibrated_and_indexed_images_keep_their_samples_without_conversion() {
        let mut pdf = lopdf::Document::with_version("1.7");
        let rgb_profile = pdf.add_object(Stream::new(dictionary! { "N" => 3 }, vec![0; 4]));
        let gray_profile = pdf.add_object(Stream::new(dictionary! { "N" => 1 }, vec![0; 4]));
        let cmyk_profile = pdf.add_object(Stream::new(dictionary! { "N" => 4 }, vec![0; 4]));
        let icc = |profile| Object::Array(vec!["ICCBased".into(), Object::Reference(profile)]);
        let image = |space: Object, bits: i64, width: i64, samples: Vec<u8>| {
            Stream::new(
                dictionary! { "Width" => width, "Height" => 1, "BitsPerComponent" => bits, "ColorSpace" => space },
                samples,
            )
        };
        // Indirect ICC profiles use their component count.
        let (kind, bytes) = image_bytes(
            &pdf,
            &image(icc(rgb_profile), 8, 2, vec![200, 60, 40, 1, 2, 3]),
        )
        .unwrap();
        assert_eq!(kind, "png");
        assert_eq!(
            decode_png(bytes),
            (png::ColorType::Rgb, vec![200, 60, 40, 1, 2, 3])
        );
        let (_, bytes) = image_bytes(&pdf, &image(icc(gray_profile), 8, 2, vec![7, 9])).unwrap();
        assert_eq!(decode_png(bytes), (png::ColorType::Grayscale, vec![7, 9]));
        let calibrated = Object::Array(vec!["CalRGB".into(), Object::Dictionary(dictionary! {})]);
        let (_, bytes) = image_bytes(&pdf, &image(calibrated, 8, 1, vec![4, 5, 6])).unwrap();
        assert_eq!(decode_png(bytes), (png::ColorType::Rgb, vec![4, 5, 6]));
        // A two-bit RGB palette of two colours: index 3 exceeds the maximum
        // index and is clamped to it; rows are byte-aligned.
        let palette = Object::Array(vec![
            "Indexed".into(),
            "DeviceRGB".into(),
            1.into(),
            Object::string_literal(vec![10, 20, 30, 40, 50, 60]),
        ]);
        let (_, bytes) = image_bytes(&pdf, &image(palette, 2, 3, vec![0b0001_1100])).unwrap();
        assert_eq!(
            decode_png(bytes).1,
            vec![10, 20, 30, 40, 50, 60, 40, 50, 60]
        );
        // A gray base expands to RGB entries; the lookup may be a stream.
        let lookup = pdf.add_object(Stream::new(dictionary! {}, vec![0, 255]));
        let gray_palette = Object::Array(vec![
            "Indexed".into(),
            icc(gray_profile),
            1.into(),
            Object::Reference(lookup),
        ]);
        let (_, bytes) = image_bytes(&pdf, &image(gray_palette, 8, 2, vec![1, 0])).unwrap();
        assert_eq!(decode_png(bytes).1, vec![255, 255, 255, 0, 0, 0]);
        // A palette longer than the sample depth can address keeps only the
        // addressable entries.
        let long = Object::Array(vec![
            "Indexed".into(),
            "DeviceRGB".into(),
            3.into(),
            Object::string_literal((0..12).collect::<Vec<u8>>()),
        ]);
        let (_, bytes) = image_bytes(&pdf, &image(long, 1, 2, vec![0b0100_0000])).unwrap();
        let reader = png::Decoder::new(std::io::Cursor::new(bytes.clone()))
            .read_info()
            .unwrap();
        assert_eq!(
            reader.info().palette.as_deref(),
            Some(&[0, 1, 2, 3, 4, 5][..])
        );
        assert_eq!(decode_png(bytes).1, vec![0, 1, 2, 3, 4, 5]);
        // Colour conversion and malformed palettes remain explicit errors.
        for (space, bits, samples) in [
            (icc(cmyk_profile), 8, vec![0; 8]),
            ("DeviceCMYK".into(), 8, vec![0; 8]),
            (icc(rgb_profile), 4, vec![0; 3]),
            (
                Object::Array(vec![
                    "Indexed".into(),
                    "DeviceRGB".into(),
                    2.into(),
                    Object::string_literal(vec![0; 6]),
                ]),
                8,
                vec![0, 1],
            ),
            (
                Object::Array(vec![
                    "Indexed".into(),
                    Object::Array(vec![
                        "Indexed".into(),
                        "DeviceRGB".into(),
                        0.into(),
                        Object::string_literal(vec![0; 3]),
                    ]),
                    0.into(),
                    Object::string_literal(vec![0]),
                ]),
                8,
                vec![0, 0],
            ),
        ] {
            assert!(image_bytes(&pdf, &image(space, bits, 2, samples)).is_err());
        }
    }

    /// One page written by hand with a cross-reference stream, its Info
    /// dictionary packed with 9 MB of unreferenced filler into one object
    /// stream: past the page reader's 8 MB load bound, within lopdf's own.
    fn info_in_a_large_object_stream() -> Vec<u8> {
        fn object(body: &mut Vec<u8>, offsets: &mut Vec<u32>, number: u32, data: &[u8]) {
            offsets.push(body.len() as u32);
            body.extend_from_slice(format!("{number} 0 obj\n").as_bytes());
            body.extend_from_slice(data);
            body.extend_from_slice(b"\nendobj\n");
        }
        let (mut body, mut offsets) = (b"%PDF-1.5\n".to_vec(), Vec::new());
        let content =
            b"BT /F1 12 Tf 72 720 Td (The page reader loads every object of this page.) Tj ET";
        object(
            &mut body,
            &mut offsets,
            1,
            b"<< /Type /Catalog /Pages 2 0 R >>",
        );
        object(
            &mut body,
            &mut offsets,
            2,
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 612 792] >>",
        );
        object(&mut body, &mut offsets, 3, b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>");
        let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
        stream.extend_from_slice(content);
        stream.extend_from_slice(b"\nendstream");
        object(&mut body, &mut offsets, 4, &stream);
        let mut header = String::new();
        let mut members = b"<< /Title (Kept by lopdf) >> ".to_vec();
        header.push_str("6 0 ");
        for index in 0..9 {
            header.push_str(&format!("{} {} ", 7 + index, members.len()));
            members.extend_from_slice(b"<< /Filler (");
            members.extend(std::iter::repeat_n(b'y', 1024 * 1024));
            members.extend_from_slice(b") >> ");
        }
        let mut packed = header.clone().into_bytes();
        packed.extend_from_slice(&members);
        let mut packed = Stream::new(Dictionary::new(), packed);
        packed.compress().unwrap();
        let mut data = format!(
            "<< /Type /ObjStm /N 10 /First {} /Filter /FlateDecode /Length {} >>\nstream\n",
            header.len(),
            packed.content.len()
        )
        .into_bytes();
        data.extend_from_slice(&packed.content);
        data.extend_from_slice(b"\nendstream");
        object(&mut body, &mut offsets, 5, &data);
        let mut rows = vec![(0u8, 0u32, 65535u16)];
        rows.extend(offsets.iter().map(|&offset| (1, offset, 0)));
        rows.extend((0..10).map(|index| (2, 5, index)));
        let xref = rows.len();
        rows.push((1, body.len() as u32, 0));
        let table: Vec<u8> = rows
            .iter()
            .flat_map(|&(kind, field, index)| {
                std::iter::once(kind)
                    .chain(field.to_be_bytes())
                    .chain(index.to_be_bytes())
            })
            .collect();
        let start = body.len();
        body.extend_from_slice(format!("{xref} 0 obj\n<< /Type /XRef /Size {} /W [1 4 2] /Root 1 0 R /Info 6 0 R /Length {} >>\nstream\n", xref + 1, table.len()).as_bytes());
        body.extend_from_slice(&table);
        body.extend_from_slice(
            format!("\nendstream\nendobj\nstartxref\n{start}\n%%EOF\n").as_bytes(),
        );
        body
    }

    #[test]
    fn a_file_the_page_reader_loads_differently_is_inspected_as_lopdf_loads_it() {
        let bytes = info_in_a_large_object_stream();
        let loaded = pdf_inspector::LoadedPdf::load_mem(&bytes).unwrap();
        assert!(loaded.as_loaded_by_lopdf().is_none());
        assert!(loaded.document().trailer.get(b"Info").is_ok());
        assert!(loaded.document().get_object((6, 0)).is_err());
        // The page reader's text and this module's own load's metadata.
        let result = extract(&bytes).unwrap();
        assert!(
            result
                .markdown
                .contains("The page reader loads every object of this page."),
            "{}",
            result.markdown
        );
        assert_eq!(
            result
                .metadata
                .get("title")
                .and_then(|title| title.as_str()),
            Some("Kept by lopdf")
        );
    }

    #[test]
    fn invisible_text_and_used_images_are_inspected_inside_forms() {
        let mut pdf = lopdf::Document::with_version("1.7");
        let image = pdf.add_object(Stream::new(dictionary! { "Subtype" => "Image" }, vec![1]));
        let content = Content {
            operations: vec![
                Operation::new("Tr", vec![3.into()]),
                Operation::new("Tj", vec![Object::string_literal("hidden")]),
                Operation::new("Do", vec![Object::Name(b"Im".to_vec())]),
            ],
        }
        .encode()
        .unwrap();
        let form = pdf.add_object(Stream::new(dictionary! { "Subtype" => "Form", "Resources" => dictionary! { "XObject" => dictionary! { "Im" => image } } }, content));
        let resources =
            dictionary! { "XObject" => dictionary! { "Fm" => form, "Unused" => image } };
        let content = Content {
            operations: vec![Operation::new("Do", vec![Object::Name(b"Fm".to_vec())])],
        }
        .encode()
        .unwrap();
        let mut output = PageInspection {
            images: Vec::new(),
            signals: BTreeSet::new(),
            warnings: Vec::new(),
            inspected_bytes: 0,
            inspected_streams: 0,
        };
        let _ = inspect_content(
            &pdf,
            &content,
            &[&resources],
            GraphicsState::default(),
            &mut BTreeSet::new(),
            0,
            &mut output,
        );
        assert_eq!(output.images, [image]);
        assert!(output.signals.contains("invisible text rendering mode"));
        assert!(output.warnings.is_empty());
    }
}
