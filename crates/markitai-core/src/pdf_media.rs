//! In-memory PDF page media planning; publication stays with the output layer.

use crate::formats::{PdfPages, extract_pdf_pages_bounded_with_config};
use crate::{Asset, Document, Error, Result, config, ocr, pdf_raster::PdfRasterSession};
use image::{ImageEncoder, ImageReader, RgbImage};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::Write;

const DPI: f64 = 150.0;
const MAX_PAGES: usize = 1_000;
const MAX_PAGE_PIXELS: u64 = 32_000_000;
const MAX_DOCUMENT_PIXELS: u64 = 2_000_000_000;
const MAX_SHOT_BYTES: usize = 5 * 1024 * 1024;
const MAX_SHOTS_BYTES: usize = 100 * 1024 * 1024;
const MIN_PICTURE_PIXELS: u64 = 40_000;
/// The extraction warning of a PDF with images: they follow their page's
/// text rather than sit where the page shows them.
pub(crate) const IMAGE_PLACEMENT: &str = "PDF images are placed after their page's text; their exact position and vector graphics are not reconstructed.";

fn failure(message: impl std::fmt::Display) -> Error {
    Error::Conversion(format!("PDF page media: {message}"))
}

pub(crate) struct PreparedPdf {
    pub screenshots: Vec<Asset>,
    pub has_reliable_text: bool,
    pages: PdfPages,
    screenshot_pages: Vec<usize>,
    completed_media: bool,
    media_requested: bool,
}

impl PreparedPdf {
    /// Use final published names, then return the same owned screenshot payloads.
    pub(crate) fn finish(mut self) -> Result<(Document, Vec<Asset>)> {
        if self.screenshots.len() != self.screenshot_pages.len() {
            return Err(failure("published screenshot count changed"));
        }
        for (asset, &index) in self.screenshots.iter().zip(&self.screenshot_pages) {
            validate_name(&asset.name)?;
            self.pages.pages[index].screenshot_name = Some(asset.name.clone());
        }
        let mut document = if self.completed_media {
            self.pages.finish_with_media()?
        } else {
            self.pages.finish()?
        };
        // This replaces the image placement statement, never a routing signal. All page-specific extraction and inspection diagnostics remain.
        // It names the screenshots only when some were written.
        if self.media_requested && !self.screenshots.is_empty() {
            for warning in &mut document.warnings {
                if warning == IMAGE_PLACEMENT {
                    *warning = "PDF images are placed after their page's text and vector graphics are not reconstructed; the page screenshots keep each page's appearance.".into();
                }
            }
        }
        Ok((document, self.screenshots))
    }
}

#[derive(Debug)]
struct Plan {
    local_ocr: bool,
    vlm_ocr: bool,
    screenshots: bool,
    recognize: Vec<bool>,
}

impl Plan {
    fn new(pages: &PdfPages, cfg: &Value, vlm_optout: bool) -> Result<Self> {
        let count = pages.pages.len();
        if count == 0 || count > MAX_PAGES {
            return Err(failure("documents must contain between 1 and 1000 pages"));
        }
        if pages
            .pages
            .iter()
            .enumerate()
            .any(|(index, page)| page.number != index + 1)
        {
            return Err(failure(
                "page extraction did not preserve contiguous page identity",
            ));
        }
        let ocr = config::enabled(cfg, "/ocr/enabled");
        let llm = config::enabled(cfg, "/llm/enabled");
        let local_ocr = ocr && (!llm || vlm_optout);
        let vlm_ocr = ocr && llm && !vlm_optout;
        let screenshots = config::enabled(cfg, "/screenshot/enabled") || vlm_ocr;
        let sends_images = llm
            && screenshots
            && (!config::enabled(cfg, "/llm/pure")
                || config::enabled(cfg, "/screenshot/screenshot_only"));
        let limit = cfg
            .pointer("/llm/max_vision_pages_per_document")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if sends_images && limit > 0 && count as u64 > limit {
            return Err(Error::InvalidInput(
                "PDF pages exceed llm.max_vision_pages_per_document; no pages were rendered or sent".into(),
            ));
        }
        let routing = cfg
            .pointer("/ocr/per_page_routing")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let recognize = pages
            .pages
            .iter()
            .map(|page| {
                local_ocr
                    && (!routing
                        || page.needs_ocr
                        || page.visibility_suspect
                        || page.omitted_text.is_some()
                        || page.markdown.trim().is_empty())
            })
            .collect();
        Ok(Self {
            local_ocr,
            vlm_ocr,
            screenshots,
            recognize,
        })
    }
}

#[derive(Default)]
struct Budget {
    pixels: u64,
    screenshots: usize,
}

impl Budget {
    fn pixels(&mut self, width: u32, height: u32) -> Result<()> {
        let count = u64::from(width) * u64::from(height);
        if count == 0 || count > MAX_PAGE_PIXELS {
            return Err(failure("rendered page exceeds the 32-million-pixel limit"));
        }
        self.pixels = self
            .pixels
            .checked_add(count)
            .filter(|&total| total <= MAX_DOCUMENT_PIXELS)
            .ok_or_else(|| failure("document exceeds the two-billion-pixel media budget"))?;
        Ok(())
    }

    fn screenshot(&mut self, bytes: usize) -> Result<()> {
        if bytes == 0 || bytes > MAX_SHOT_BYTES {
            return Err(failure("a screenshot must be nonempty and at most 5 MiB"));
        }
        self.screenshots = self
            .screenshots
            .checked_add(bytes)
            .filter(|&total| total <= MAX_SHOTS_BYTES)
            .ok_or_else(|| failure("document screenshots exceed 100 MiB"))?;
        Ok(())
    }
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || matches!(name, "." | "..")
        || name.contains(['/', '\\'])
        || name.chars().any(char::is_control)
    {
        return Err(Error::InvalidInput(
            "PDF screenshot prefix/name must be one filename".into(),
        ));
    }
    Ok(())
}

struct Bounded {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl Write for Bounded {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(std::io::Error::other(
                "PDF screenshot encoded byte limit exceeded",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encoding {
    Jpeg,
    Png,
    Webp,
}

impl Encoding {
    fn extension(self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
            Self::Webp => "webp",
        }
    }
}

fn encode_attempt(
    image: &RgbImage,
    format: Encoding,
    quality: u8,
    limit: usize,
) -> Result<Option<Vec<u8>>> {
    let mut out = Bounded {
        bytes: Vec::new(),
        limit,
        exceeded: false,
    };
    let result = match format {
        Encoding::Jpeg => image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgb8,
            ),
        Encoding::Png => image::codecs::png::PngEncoder::new_with_quality(
            &mut out,
            image::codecs::png::CompressionType::Fast,
            image::codecs::png::FilterType::Adaptive,
        )
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::Rgb8,
        ),
        Encoding::Webp => image::codecs::webp::WebPEncoder::new_lossless(&mut out).write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::Rgb8,
        ),
    };
    if out.exceeded {
        return Ok(None);
    }
    result.map_err(|error| failure(format!("cannot encode screenshot: {error}")))?;
    Ok(Some(out.bytes))
}

fn fit(width: u32, height: u32, max_width: u32, max_height: u32) -> (u32, u32) {
    if width <= max_width && height <= max_height {
        return (width, height);
    }
    if u64::from(max_width) * u64::from(height) <= u64::from(max_height) * u64::from(width) {
        (
            max_width,
            (u64::from(height) * u64::from(max_width) / u64::from(width)).max(1) as u32,
        )
    } else {
        (
            (u64::from(width) * u64::from(max_height) / u64::from(height)).max(1) as u32,
            max_height,
        )
    }
}

fn positive_dimension(cfg: &Value, key: &str, default: u32) -> Result<u32> {
    cfg.pointer(key)
        .and_then(Value::as_u64)
        .unwrap_or(u64::from(default))
        .try_into()
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| Error::Config(format!("{key} must be a positive 32-bit dimension")))
}

struct EncodedShot {
    bytes: Vec<u8>,
    format: Encoding,
    fallback: bool,
}

fn encode_screenshot(rgb: &RgbImage, cfg: &Value, limit: usize) -> Result<EncodedShot> {
    let max_width = positive_dimension(cfg, "/image/max_width", 1920)?;
    let max_height = positive_dimension(cfg, "/image/max_height", 99_999)?;
    let quality = cfg
        .pointer("/image/quality")
        .and_then(Value::as_u64)
        .unwrap_or(75);
    if !(1..=100).contains(&quality) {
        return Err(Error::Config(
            "image.quality must be between 1 and 100".into(),
        ));
    }
    let format = match cfg
        .pointer("/image/format")
        .and_then(Value::as_str)
        .unwrap_or("jpeg")
    {
        "jpeg" | "jpg" => Encoding::Jpeg,
        "png" => Encoding::Png,
        "webp" => Encoding::Webp,
        _ => {
            return Err(Error::Config(
                "PDF screenshots require JPEG, PNG or WebP format".into(),
            ));
        }
    };
    let size = fit(rgb.width(), rgb.height(), max_width, max_height);
    let resized;
    let image = if size != rgb.dimensions() {
        resized =
            image::imageops::resize(rgb, size.0, size.1, image::imageops::FilterType::Lanczos3);
        &resized
    } else {
        rgb
    };
    if let Some(bytes) = encode_attempt(image, format, quality as u8, limit)? {
        return Ok(EncodedShot {
            bytes,
            format,
            fallback: false,
        });
    }
    // Lossless encoders ignore quality. Once their size limit is reached, switch
    // to JPEG rather than repeat the same unbounded or ineffective attempt.
    let qualities = [quality as u8, 70, 55, 40, 25];
    let mut previous = 101;
    for candidate in qualities {
        if candidate >= previous || (format == Encoding::Jpeg && candidate == quality as u8) {
            previous = previous.min(candidate);
            continue;
        }
        previous = candidate;
        if let Some(bytes) = encode_attempt(image, Encoding::Jpeg, candidate, limit)? {
            return Ok(EncodedShot {
                bytes,
                format: Encoding::Jpeg,
                fallback: true,
            });
        }
    }
    let size = fit(image.width(), image.height(), 1024, 1024);
    let reduced;
    let image = if size != image.dimensions() {
        reduced =
            image::imageops::resize(image, size.0, size.1, image::imageops::FilterType::Lanczos3);
        &reduced
    } else {
        image
    };
    let bytes = encode_attempt(image, Encoding::Jpeg, (quality as u8).min(20), limit)?
        .ok_or_else(|| failure("screenshot still exceeds 5 MiB after bounded JPEG fallback"))?;
    Ok(EncodedShot {
        bytes,
        format: Encoding::Jpeg,
        fallback: true,
    })
}

#[derive(Default)]
struct OcrCounts {
    attempted: usize,
    nonempty: usize,
    blank: usize,
}

impl OcrCounts {
    fn successful(&self) -> usize {
        self.nonempty + self.blank
    }
}

fn picture_result(
    name: &str,
    result: Result<String>,
    backend_available: bool,
    counts: &mut OcrCounts,
    warnings: &mut Vec<String>,
) -> Result<Option<String>> {
    counts.attempted += 1;
    match result {
        // A photo or figure without text is ordinary; only the count records it.
        Ok(text) if text.trim().is_empty() => {
            counts.blank += 1;
            Ok(None)
        }
        Ok(text) => {
            counts.nonempty += 1;
            Ok(Some(text))
        }
        Err(error @ Error::Config(_)) => Err(error),
        Err(error @ Error::Unsupported(_)) if backend_available => Err(error),
        Err(error) => {
            warnings.push(format!("PDF embedded image {name}: local OCR failed ({error}); the native page and image reference were preserved."));
            Ok(None)
        }
    }
}

fn recognize_native_pictures(
    pages: &mut PdfPages,
    plan: &Plan,
    cfg: &Value,
    budget: &mut Budget,
) -> Result<OcrCounts> {
    let indices: HashMap<_, _> = pages
        .document
        .assets
        .iter()
        .enumerate()
        .map(|(index, asset)| (asset.name.as_str(), index))
        .collect();
    let mut recognized: HashMap<usize, Option<String>> = HashMap::new();
    let mut counts = OcrCounts::default();
    for (page_index, page) in pages.pages.iter_mut().enumerate() {
        // A page read from the OCR layer over its scan already holds the
        // scan's text: reading the scan again as a picture would repeat it.
        if plan.recognize[page_index]
            || page.needs_ocr
            || page.visibility_suspect
            || page.ocr_layer.is_some()
            || page.markdown.trim().is_empty()
        {
            continue;
        }
        for name in &page.asset_names {
            let Some(&index) = indices.get(name.as_str()) else {
                continue;
            };
            if let std::collections::hash_map::Entry::Vacant(entry) = recognized.entry(index) {
                let asset = &pages.document.assets[index];
                // One reader type for every image decode keeps one copy of
                // the decoders (`ImageBytes`); a `Cursor` here compiled them
                // all again.
                let dimensions = ImageReader::new(crate::images::ImageBytes::new(&asset.bytes))
                    .with_guessed_format()
                    .map_err(|error| error.to_string())
                    .and_then(|reader| reader.into_dimensions().map_err(|error| error.to_string()));
                let (width, height) = match dimensions {
                    Ok(size) => size,
                    Err(_) => {
                        pages.document.warnings.push(format!("PDF embedded image {name}: local OCR could not read its dimensions; the image reference was preserved."));
                        entry.insert(None);
                        continue;
                    }
                };
                if u64::from(width) * u64::from(height) < MIN_PICTURE_PIXELS {
                    entry.insert(None);
                    continue;
                }
                if u64::from(width) * u64::from(height) > MAX_PAGE_PIXELS {
                    pages.document.warnings.push(format!("PDF embedded image {name}: local OCR exceeds the 32-million-pixel limit; the native page and image reference were preserved."));
                    entry.insert(None);
                    continue;
                }
                budget.pixels(width, height)?;
                let text = picture_result(
                    name,
                    ocr::recognize(&asset.bytes, cfg).map(|result| result.text),
                    ocr::available(),
                    &mut counts,
                    &mut pages.document.warnings,
                )?;
                entry.insert(text);
            }
            if let Some(Some(text)) = recognized.get(&index) {
                page.asset_ocr.insert(name.clone(), text.clone());
            }
        }
    }
    Ok(counts)
}

pub(crate) struct CapturedPages {
    pub screenshots: Vec<Asset>,
    pub ocr: Vec<String>,
    pub warnings: Vec<String>,
}

/// Rasterize an already validated Office export without replacing its native text.
pub(crate) fn capture_external_pdf(
    bytes: &[u8],
    count: usize,
    prefix: &str,
    cfg: &Value,
    screenshots: bool,
    local_ocr: bool,
) -> Result<CapturedPages> {
    validate_name(prefix)?;
    if count == 0 || count > MAX_PAGES {
        return Err(failure("documents must contain between 1 and 1000 pages"));
    }
    let sends_images = screenshots
        && config::enabled(cfg, "/llm/enabled")
        && (!config::enabled(cfg, "/llm/pure")
            || config::enabled(cfg, "/screenshot/screenshot_only"));
    let limit = cfg
        .pointer("/llm/max_vision_pages_per_document")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if sends_images && limit > 0 && count as u64 > limit {
        return Err(Error::InvalidInput(
            "Office pages exceed llm.max_vision_pages_per_document; no pages were rendered or sent"
                .into(),
        ));
    }
    let mut captured = CapturedPages {
        screenshots: Vec::new(),
        ocr: Vec::new(),
        warnings: Vec::new(),
    };
    if !screenshots && !local_ocr {
        return Ok(captured);
    }
    let session = PdfRasterSession::open(bytes)?;
    if session.pages() != count {
        return Err(failure("renderer and Office export disagree on page count"));
    }
    let mut budget = Budget::default();
    let mut sizes = Vec::with_capacity(count);
    for page in 1..=count {
        let size = session.dimensions(page, DPI)?;
        budget.pixels(size.0, size.1)?;
        sizes.push(size);
    }
    for (index, size) in sizes.into_iter().enumerate() {
        let page = index + 1;
        let pixels = session.render(page, DPI)?;
        if pixels.dimensions() != size {
            return Err(failure(
                "rendered dimensions changed after budget validation",
            ));
        }
        if screenshots {
            let encoded = encode_screenshot(&pixels, cfg, MAX_SHOT_BYTES)?;
            budget.screenshot(encoded.bytes.len())?;
            if encoded.fallback {
                captured.warnings.push(format!(
                    "Office page {page}: screenshot size required JPEG compression fallback."
                ));
            } else if encoded.format == Encoding::Webp && captured.screenshots.is_empty() {
                captured.warnings.push("Office screenshots use lossless WebP; image.quality does not affect this encoder.".into());
            }
            captured.screenshots.push(Asset {
                name: format!("{prefix}.page{page:04}.{}", encoded.format.extension()),
                bytes: encoded.bytes,
            });
        }
        if local_ocr {
            let recognized = ocr::recognize_rgb(pixels, cfg)?;
            if recognized.unread {
                captured
                    .warnings
                    .push(ocr::unread_warning(&format!("Office page {page}")));
            }
            let text = recognized.text;
            if text.trim().is_empty() {
                captured.warnings.push(format!(
                    "Office page {page}: local OCR completed with no recognized text."
                ));
            }
            captured.ocr.push(text);
        }
    }
    Ok(captured)
}

pub(crate) fn prepare(
    bytes: &[u8],
    prefix: &str,
    cfg: &Value,
    vlm_optout: bool,
) -> Result<PreparedPdf> {
    validate_name(prefix)?;
    let mut pages = extract_pdf_pages_bounded_with_config(bytes, MAX_PAGES, cfg)?;
    let plan = Plan::new(&pages, cfg, vlm_optout)?;
    let render_indices = plan
        .recognize
        .iter()
        .enumerate()
        .filter_map(|(index, &recognize)| (plan.screenshots || recognize).then_some(index))
        .collect::<Vec<_>>();
    let mut budget = Budget::default();
    let session = if render_indices.is_empty() {
        None
    } else {
        Some(PdfRasterSession::open(bytes)?)
    };
    let mut sizes = Vec::with_capacity(render_indices.len());
    if let Some(session) = &session {
        if session.pages() != pages.pages.len() {
            return Err(failure("renderer and extractor disagree on page count"));
        }
        for &index in &render_indices {
            let size = session.dimensions(pages.pages[index].number, DPI)?;
            budget.pixels(size.0, size.1)?;
            sizes.push(size);
        }
    }
    let mut screenshots = Vec::new();
    let mut screenshot_pages = Vec::new();
    let mut counts = OcrCounts::default();
    if plan.local_ocr && config::enabled(cfg, "/llm/enabled") && vlm_optout {
        let sends_images = plan.screenshots
            && (!config::enabled(cfg, "/llm/pure")
                || config::enabled(cfg, "/screenshot/screenshot_only"));
        pages.document.warnings.push(if sends_images {
            "VLM OCR is disabled; OCR stays local, but explicitly requested screenshot enhancement still sends page images to the model. Disable screenshots to keep page images local."
        } else {
            "VLM OCR is disabled; only locally extracted text is used for PDF LLM enhancement."
        }.into());
    }
    for (&index, &size) in render_indices.iter().zip(&sizes) {
        let page = &mut pages.pages[index];
        let pixels = session
            .as_ref()
            .expect("render indices require a session")
            .render(page.number, DPI)?;
        if pixels.dimensions() != size {
            return Err(failure(
                "rendered dimensions changed after budget validation",
            ));
        }
        if plan.screenshots {
            let encoded = encode_screenshot(&pixels, cfg, MAX_SHOT_BYTES)?;
            budget.screenshot(encoded.bytes.len())?;
            if encoded.fallback {
                pages.document.warnings.push(format!("PDF page {}: screenshot size required JPEG compression fallback.", page.number));
            } else if encoded.format == Encoding::Webp && !pages.document.warnings.iter().any(|warning| warning == "PDF screenshots use lossless WebP; image.quality does not affect this encoder.") {
                pages.document.warnings.push("PDF screenshots use lossless WebP; image.quality does not affect this encoder.".into());
            }
            screenshots.push(Asset {
                name: format!(
                    "{prefix}.page{:04}.{}",
                    page.number,
                    encoded.format.extension()
                ),
                bytes: encoded.bytes,
            });
            screenshot_pages.push(index);
        }
        if plan.recognize[index] {
            counts.attempted += 1;
            let result = ocr::recognize_rgb(pixels, cfg).map_err(|error| {
                failure(format!("local OCR failed on page {}: {error}", page.number))
            })?;
            if result.unread {
                pages
                    .document
                    .warnings
                    .push(ocr::unread_warning(&format!("PDF page {}", page.number)));
            }
            page.markdown = result.text;
            page.needs_ocr = false;
            page.ocr_reason = None;
            page.ocr_completed = true;
            // The page's text is now this recognition's, not its OCR layer's.
            page.ocr_layer = None;
            if page.markdown.trim().is_empty() {
                counts.blank += 1;
                pages.document.warnings.push(format!(
                    "PDF page {}: local OCR completed with no recognized text.",
                    page.number
                ));
            } else {
                counts.nonempty += 1;
            }
        }
    }
    let picture_counts = if plan.local_ocr {
        recognize_native_pictures(&mut pages, &plan, cfg, &mut budget)?
    } else {
        OcrCounts::default()
    };
    let native_pages = pages
        .pages
        .iter()
        .filter(|page| {
            !page.ocr_completed
                && !page.needs_ocr
                && !page.visibility_suspect
                && !page.markdown.trim().is_empty()
        })
        .count();
    let has_reliable_text = pages.pages.iter().any(|page| {
        !page.needs_ocr
            && (page.ocr_completed || !page.visibility_suspect)
            && !page.markdown.trim().is_empty()
    });
    pages
        .document
        .metadata
        .insert("pages".into(), json!(pages.pages.len()));
    pages
        .document
        .metadata
        .insert("native_pages".into(), json!(native_pages));
    if plan.local_ocr {
        for (name, value) in [
            ("ocr_pages_attempted", counts.attempted),
            ("ocr_pages_nonempty", counts.nonempty),
            ("ocr_pages_blank", counts.blank),
            ("ocr_images_attempted", picture_counts.attempted),
            ("ocr_images_nonempty", picture_counts.nonempty),
            ("ocr_images_blank", picture_counts.blank),
        ] {
            pages.document.metadata.insert(name.into(), json!(value));
        }
        pages.document.metadata.insert(
            "ocr_used".into(),
            json!(counts.successful() + picture_counts.successful() > 0),
        );
        pages
            .document
            .metadata
            .insert("ocr_path".into(), json!(ocr::backend()));
    } else if plan.vlm_ocr {
        pages
            .document
            .metadata
            .insert("ocr_path".into(), json!("vlm"));
    }
    let completed_media = !screenshots.is_empty() || counts.attempted > 0;
    let media_requested = plan.local_ocr || plan.screenshots;
    Ok(PreparedPdf {
        screenshots,
        has_reliable_text,
        pages,
        screenshot_pages,
        completed_media,
        media_requested,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::extract_pdf_pages;
    use image::{ImageFormat, Rgb};

    fn pages() -> PdfPages {
        let mut pages = extract_pdf_pages(include_bytes!(
            "pdf_raster/fixtures/mixed-native-scanned-blank.pdf"
        ))
        .unwrap();
        assert_eq!(pages.pages.len(), 3);
        pages.pages[0].markdown = "# Reliable native heading\n\nA native paragraph with **structure** and [link](https://example.test).".into();
        pages.pages[0].needs_ocr = false;
        pages.pages[0].visibility_suspect = false;
        pages.pages[1].needs_ocr = true;
        pages.pages[2].markdown.clear();
        pages
    }

    #[test]
    fn local_routes_use_typed_page_evidence_and_preserve_native_selection() {
        let mut pages = pages();
        let cfg = json!({"ocr":{"enabled":true}});
        let plan = Plan::new(&pages, &cfg, false).unwrap();
        assert_eq!(plan.recognize, [false, true, true]);
        assert!(plan.local_ocr && !plan.vlm_ocr && !plan.screenshots);
        pages.pages[0].visibility_suspect = true;
        assert_eq!(Plan::new(&pages, &cfg, false).unwrap().recognize, [true; 3]);
        pages.pages[0].visibility_suspect = false;
        let cfg = json!({"ocr":{"enabled":true,"per_page_routing":false}});
        assert_eq!(Plan::new(&pages, &cfg, false).unwrap().recognize, [true; 3]);
    }

    #[test]
    fn native_pages_with_omitted_text_are_recognized_when_ocr_is_on() {
        let mut pages = pages();
        pages.pages[0].omitted_text = Some(pdf_inspector::PageOmittedText {
            page: 1,
            runs: 1,
            chars: 20,
            replacement_chars: 0,
            unidentified_glyphs: false,
        });
        let cfg = json!({"ocr":{"enabled":true}});
        assert_eq!(Plan::new(&pages, &cfg, false).unwrap().recognize, [true; 3]);
        assert_eq!(
            Plan::new(&pages, &json!({}), false).unwrap().recognize,
            [false; 3]
        );
    }

    #[test]
    fn vlm_optout_and_explicit_screenshots_are_independent_requests() {
        let pages = pages();
        let mut cfg = json!({"ocr":{"enabled":true},"llm":{"enabled":true}});
        let vlm = Plan::new(&pages, &cfg, false).unwrap();
        assert!(vlm.vlm_ocr && vlm.screenshots && !vlm.local_ocr);
        assert_eq!(vlm.recognize, [false; 3]);
        let local = Plan::new(&pages, &cfg, true).unwrap();
        assert!(local.local_ocr && !local.vlm_ocr && !local.screenshots);
        cfg["screenshot"] = json!({"enabled":true});
        let explicit = Plan::new(&pages, &cfg, true).unwrap();
        assert!(explicit.local_ocr && explicit.screenshots);
        assert_eq!(explicit.recognize, [false, true, true]);
    }

    #[test]
    fn file_only_does_not_enable_capture_but_overrides_pure_when_images_exist() {
        let pages = pages();
        let cfg = json!({"screenshot":{"screenshot_only":true}});
        let plan = Plan::new(&pages, &cfg, false).unwrap();
        assert!(!plan.screenshots && !plan.local_ocr);
        let mut cfg = json!({"ocr":{"enabled":true},"llm":{"enabled":true,"pure":true,"max_vision_pages_per_document":1}});
        // OCR+LLM still renders in pure mode, but its text-only request is not
        // rejected by a vision cap. File screenshot-only changes the request.
        assert!(Plan::new(&pages, &cfg, false).unwrap().screenshots);
        cfg["screenshot"] = json!({"screenshot_only":true});
        assert!(matches!(
            Plan::new(&pages, &cfg, false),
            Err(Error::InvalidInput(_))
        ));
        cfg["ocr"]["enabled"] = json!(false);
        assert!(!Plan::new(&pages, &cfg, false).unwrap().screenshots);
    }

    #[test]
    fn budgets_fail_at_boundaries_without_allocating_large_buffers() {
        let mut budget = Budget::default();
        for _ in 0..62 {
            budget.pixels(8_000, 4_000).unwrap();
        }
        budget.pixels(4_000, 4_000).unwrap();
        assert_eq!(budget.pixels, MAX_DOCUMENT_PIXELS);
        assert!(budget.pixels(1, 1).is_err());
        assert!(Budget::default().pixels(0, 1).is_err());
        assert!(Budget::default().pixels(8_000, 4_001).is_err());
        let mut budget = Budget::default();
        for _ in 0..20 {
            budget.screenshot(MAX_SHOT_BYTES).unwrap();
        }
        assert!(budget.screenshot(1).is_err());
        assert!(Budget::default().screenshot(MAX_SHOT_BYTES + 1).is_err());
        let mut writer = Bounded {
            bytes: Vec::new(),
            limit: 3,
            exceeded: false,
        };
        writer.write_all(b"abc").unwrap();
        assert!(writer.write_all(b"d").is_err());
        assert_eq!(writer.bytes, b"abc");
        assert!(writer.exceeded);
    }

    #[test]
    fn screenshot_formats_match_real_bytes_and_leave_ocr_pixels_unchanged() {
        let rgb = RgbImage::from_fn(80, 40, |x, y| Rgb([x as u8, y as u8, 180]));
        let original = rgb.clone();
        for (format, expected) in [
            ("jpeg", ImageFormat::Jpeg),
            ("png", ImageFormat::Png),
            ("webp", ImageFormat::WebP),
        ] {
            let cfg =
                json!({"image":{"format":format,"max_width":20,"max_height":20,"quality":80}});
            let encoded = encode_screenshot(&rgb, &cfg, MAX_SHOT_BYTES).unwrap();
            assert_eq!(image::guess_format(&encoded.bytes).unwrap(), expected);
            let decoded = image::load_from_memory(&encoded.bytes).unwrap();
            assert_eq!((decoded.width(), decoded.height()), (20, 10));
            assert!(!encoded.fallback);
        }
        assert_eq!(rgb, original);
    }

    #[test]
    fn oversized_lossless_capture_changes_extension_and_stays_bounded() {
        let mut seed = 17u32;
        let noisy = RgbImage::from_fn(96, 96, |_, _| {
            let channels = std::array::from_fn(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 24) as u8
            });
            Rgb(channels)
        });
        let cfg = json!({"image":{"format":"png","quality":75}});
        let encoded = encode_screenshot(&noisy, &cfg, 12_000).unwrap();
        assert!(encoded.fallback);
        assert_eq!(encoded.format.extension(), "jpg");
        assert_eq!(
            image::guess_format(&encoded.bytes).unwrap(),
            ImageFormat::Jpeg
        );
        assert!(encoded.bytes.len() <= 12_000);
        assert!(encode_screenshot(&noisy, &cfg, 4).is_err());
    }

    #[test]
    fn final_publication_names_attach_to_their_pages_without_changing_native_body() {
        let pages = pages();
        let native = pages.pages[0].markdown.clone();
        let mut prepared = PreparedPdf {
            pages,
            screenshots: vec![Asset {
                name: "old.page0003.jpg".into(),
                bytes: vec![1, 2, 3],
            }],
            screenshot_pages: vec![2],
            has_reliable_text: true,
            completed_media: true,
            media_requested: true,
        };
        prepared.screenshots[0].name = "report.pdf.v2.page0003.jpg".into();
        let (document, shots) = prepared.finish().unwrap();
        assert!(document.markdown.contains(&native));
        assert!(
            document
                .markdown
                .ends_with("<!-- ![Page 3](.markitai/screenshots/report.pdf.v2.page0003.jpg) -->")
        );
        assert_eq!(document.markdown.matches("<!-- Page number:").count(), 3);
        assert!(!document.markdown.contains("old.page0003"));
        assert_eq!(shots[0].bytes, [1, 2, 3]);
        assert!(
            !document
                .warnings
                .iter()
                .any(|warning| warning == IMAGE_PLACEMENT)
        );
    }

    #[test]
    fn placement_names_screenshots_only_when_some_were_written() {
        // Local OCR alone renders pages but publishes no screenshot.
        let prepared = PreparedPdf {
            pages: pages(),
            screenshots: Vec::new(),
            screenshot_pages: Vec::new(),
            has_reliable_text: true,
            completed_media: false,
            media_requested: true,
        };
        let (document, _) = prepared.finish().unwrap();
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning == IMAGE_PLACEMENT),
            "{:?}",
            document.warnings
        );
        assert!(
            !document
                .warnings
                .iter()
                .any(|warning| warning.contains("screenshots"))
        );
    }

    #[test]
    fn empty_native_without_completed_media_keeps_the_original_failure_guard() {
        let mut pages = pages();
        for page in &mut pages.pages {
            page.markdown.clear();
            page.needs_ocr = true;
            page.asset_names.clear();
        }
        pages.document.assets.clear();
        let prepared = PreparedPdf {
            pages,
            screenshots: Vec::new(),
            screenshot_pages: Vec::new(),
            has_reliable_text: false,
            completed_media: false,
            media_requested: false,
        };
        assert!(prepared.finish().is_err());
        assert!(validate_name("../escape").is_err());
        assert!(validate_name("bad\nname").is_err());
        assert!(validate_name("报告 1.pdf.v2").is_ok());
    }

    #[test]
    fn picture_configuration_errors_are_fatal_without_becoming_content_or_notices() {
        for error in [
            Error::Config("invalid OCR language identifier".into()),
            Error::Unsupported("language is not installed".into()),
        ] {
            let mut counts = OcrCounts::default();
            let mut warnings = Vec::new();
            assert!(
                picture_result("picture.png", Err(error), true, &mut counts, &mut warnings)
                    .is_err()
            );
            assert_eq!(counts.successful(), 0);
            assert!(warnings.is_empty());
        }
    }

    #[test]
    fn unreadable_pictures_preserve_native_content_and_do_not_count_as_ocr_success() {
        for (error, available) in [
            (Error::Unsupported("backend unavailable".into()), false),
            (Error::Conversion("recognition failed".into()), true),
        ] {
            let mut counts = OcrCounts::default();
            let mut warnings = Vec::new();
            let text = picture_result(
                "picture.png",
                Err(error),
                available,
                &mut counts,
                &mut warnings,
            )
            .unwrap();
            assert!(text.is_none());
            assert_eq!(counts.attempted, 1);
            assert_eq!(counts.successful(), 0);
            assert_eq!(warnings.len(), 1);
            assert!(warnings[0].contains("native page and image reference were preserved"));
        }
    }

    #[test]
    fn successful_blank_picture_is_ocr_use_but_not_recovered_text() {
        let mut counts = OcrCounts::default();
        let mut warnings = Vec::new();
        assert!(
            picture_result(
                "blank.png",
                Ok(" \n".into()),
                true,
                &mut counts,
                &mut warnings
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(
            (
                counts.attempted,
                counts.blank,
                counts.nonempty,
                counts.successful()
            ),
            (1, 1, 0, 1)
        );
        let text = picture_result(
            "words.png",
            Ok("Recovered words".into()),
            true,
            &mut counts,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(text.as_deref(), Some("Recovered words"));
        assert_eq!(
            (
                counts.attempted,
                counts.blank,
                counts.nonempty,
                counts.successful()
            ),
            (2, 1, 1, 2)
        );
        // A picture without text is ordinary and only counted.
        assert!(warnings.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mixed_pdf_retains_native_body_and_accounts_for_real_ocr_and_blank_pages() {
        let config = json!({"ocr":{"enabled":true,"lang":"en","per_page_routing":true},"llm":{"enabled":false},"screenshot":{"enabled":false}});
        let prepared = prepare(
            include_bytes!("pdf_raster/fixtures/mixed-native-scanned-blank.pdf"),
            "mixed.pdf",
            &config,
            false,
        );
        if crate::ocr::tests::vision_unavailable_under_rosetta(&prepared) {
            return;
        }
        let prepared = prepared.unwrap();
        assert!(prepared.has_reliable_text);
        assert!(prepared.screenshots.is_empty());
        let (document, screenshots) = prepared.finish().unwrap();
        assert!(screenshots.is_empty());
        assert_eq!(document.metadata["native_pages"], 1);
        assert_eq!(document.metadata["ocr_pages_attempted"], 2);
        assert_eq!(document.metadata["ocr_pages_nonempty"], 1);
        assert_eq!(document.metadata["ocr_pages_blank"], 1);
        assert_eq!(document.metadata["ocr_used"], true);
        assert_eq!(document.metadata["ocr_path"], "vision");
        assert_eq!(document.markdown.matches("<!-- Page number:").count(), 3);
        assert!(document.warnings.iter().any(|warning| {
            warning.contains("page 3: local OCR completed with no recognized text")
        }));
    }

    /// A searchable scan's page (a raster with strokes where its lines are
    /// printed, and Tesseract-style invisible text on them) and a native
    /// text page.
    fn searchable_and_native() -> Vec<u8> {
        use lopdf::{Dictionary, Object, Stream, dictionary};
        let lines: Vec<(String, usize)> = (0..8)
            .map(|row| {
                (
                    format!("Line {row} of the scanned report reads as its print."),
                    700 - row * 16,
                )
            })
            .collect();
        let (width, height) = (612usize, 792usize);
        let mut pixels = vec![245u8; width * height];
        let mut layer = String::from("q 612 0 0 792 0 0 cm /Scan Do Q\n");
        for (text, baseline) in &lines {
            for y in *baseline..baseline + 8 {
                for x in (72..72 + text.len() * 11 / 2).step_by(3) {
                    pixels[(height - 1 - y) * width + x] = 20;
                }
            }
            layer.push_str(&format!(
                "BT 3 Tr 1 0 0 1 72 {baseline} Tm /F1 11 Tf ({text}) Tj ET\n"
            ));
        }
        let mut doc = lopdf::Document::with_version("1.5");
        let tree = doc.new_object_id();
        let font = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"
        });
        let mut scan = Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image", "Width" => 612, "Height" => 792,
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8
            },
            pixels,
        );
        scan.compress().unwrap();
        let scan = doc.add_object(scan);
        let mut kids = Vec::new();
        for (content, xobjects) in [
            (layer, dictionary! { "Scan" => scan }),
            (
                "BT /F1 12 Tf 1 0 0 1 72 700 Tm (A native page of visible text.) Tj ET".into(),
                dictionary! {},
            ),
        ] {
            let content = doc.add_object(Stream::new(Dictionary::new(), content.into_bytes()));
            kids.push(Object::Reference(doc.add_object(dictionary! {
                "Type" => "Page", "Parent" => tree, "Contents" => content,
                "Resources" => dictionary! { "Font" => dictionary! { "F1" => font }, "XObject" => xobjects }
            })));
        }
        doc.objects.insert(
            tree,
            dictionary! {
                "Type" => "Pages", "Count" => 2, "Kids" => kids,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
            }
            .into(),
        );
        let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
        doc.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn a_page_read_from_its_ocr_layer_is_not_recognized_again_unless_routing_is_off() {
        let bytes = searchable_and_native();
        let pages = extract_pdf_pages(&bytes).unwrap();
        assert!(pages.pages[0].ocr_layer.is_some());
        assert!(!pages.pages[0].needs_ocr && !pages.pages[0].visibility_suspect);
        let cfg = json!({"ocr":{"enabled":true}});
        let plan = Plan::new(&pages, &cfg, false).unwrap();
        assert_eq!(plan.recognize, [false, false]);
        // The scan is not read again as a picture of the page either.
        let mut pages = pages;
        let counts =
            recognize_native_pictures(&mut pages, &plan, &cfg, &mut Budget::default()).unwrap();
        assert_eq!(counts.attempted, 0);
        assert!(pages.pages[0].asset_ocr.is_empty());
        let cfg = json!({"ocr":{"enabled":true,"per_page_routing":false}});
        assert_eq!(Plan::new(&pages, &cfg, false).unwrap().recognize, [true; 2]);
    }

    #[test]
    fn ocr_without_pages_to_recognize_keeps_the_layer_and_counts_it_native() {
        let prepared = prepare(
            &searchable_and_native(),
            "scan.pdf",
            &json!({"ocr":{"enabled":true},"llm":{"enabled":false},"screenshot":{"enabled":false}}),
            false,
        )
        .unwrap();
        assert!(prepared.has_reliable_text);
        let (document, screenshots) = prepared.finish().unwrap();
        assert!(screenshots.is_empty());
        assert_eq!(document.metadata["native_pages"], 2);
        assert_eq!(document.metadata["ocr_pages_attempted"], 0);
        assert_eq!(document.metadata["ocr_images_attempted"], 0);
        assert_eq!(document.metadata["ocr_layer_pages"], json!([1]));
        assert!(document.markdown.contains("Line 7 of the scanned report"));
        assert!(document.warnings.iter().any(|warning| warning.starts_with(
            "PDF page 1: the text was read from the invisible OCR text layer laid over the page image;"
        )));
    }
}
