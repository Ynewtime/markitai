//! In-memory PDF page media planning; publication stays with the output layer.

use crate::formats::{PdfPages, check_pdf_glyphs, extract_pdf_pages_bounded_with_config};
use crate::{Asset, Document, Error, Result, config, ocr, pdf_raster::PdfRasterSession};
use image::{ImageEncoder, ImageReader, RgbImage};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::Write;

const DPI: f64 = 150.0;
const MAX_PAGES: usize = 1_000;
use crate::images::MAX_PIXELS as MAX_PAGE_PIXELS;
const MAX_DOCUMENT_PIXELS: u64 = 2_000_000_000;
const MAX_SHOT_BYTES: usize = 5 * 1024 * 1024;
const MAX_SHOTS_BYTES: usize = 100 * 1024 * 1024;
const MIN_PICTURE_PIXELS: u64 = 40_000;
/// The extraction warning of a PDF with images: they follow their page's
/// text rather than sit where the page shows them.
pub(crate) const IMAGE_PLACEMENT: &str = "Embedded PDF images are placed after their page's text; their exact position is not reconstructed. Add --screenshot to keep each page's appearance.";

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
    /// The page images go to a vision model that reads them beside the
    /// text layer (`--ocr --llm`).
    vision_reads: bool,
}

impl PreparedPdf {
    /// Use final published names, then return the same owned screenshot payloads.
    pub(crate) fn finish(mut self) -> Result<(Document, Vec<Asset>)> {
        if self.screenshots.len() != self.screenshot_pages.len() {
            return Err(failure("published screenshot count changed"));
        }
        for (asset, &index) in self.screenshots.iter().zip(&self.screenshot_pages) {
            validate_name(&asset.name)?;
            let page = &mut self.pages.pages[index];
            page.screenshot_name = Some(asset.name.clone());
            page.vision_read = self.vision_reads;
        }
        let mut document = if self.completed_media {
            self.pages.finish_with_media()?
        } else {
            self.pages.finish()?
        };
        // The placement notice offers page screenshots; once some were
        // written there is nothing left for the user to do, so it goes.
        if self.media_requested && !self.screenshots.is_empty() {
            document
                .warnings
                .retain(|warning| warning != IMAGE_PLACEMENT);
        }
        Ok((document, self.screenshots))
    }
}

#[derive(Debug)]
struct Plan {
    local_ocr: bool,
    vlm_ocr: bool,
    screenshots: bool,
    /// The vision model is sent the page images (see [`PreparedPdf`]).
    vision_reads: bool,
    /// Only the pages needing it are recognized (`ocr.per_page_routing`).
    routing: bool,
    recognize: Vec<bool>,
}

impl Plan {
    /// OCR runs here, not in a vision model.
    fn local_ocr(cfg: &Value, vlm_optout: bool) -> bool {
        config::enabled(cfg, "/ocr/enabled")
            && (!config::enabled(cfg, "/llm/enabled") || vlm_optout)
    }

    /// Only the pages needing it are recognized (`ocr.per_page_routing`).
    fn routing(cfg: &Value) -> bool {
        cfg.pointer("/ocr/per_page_routing")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

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
        let local_ocr = Self::local_ocr(cfg, vlm_optout);
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
        let routing = Self::routing(cfg);
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
            vision_reads: vlm_ocr && sends_images,
            routing,
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

/// Protect the original point-space page and the repair's existing text margin.
/// Quartz centres less than one pixel of ceil rounding; hayro leaves it at the
/// right. One extra column conservatively covers both existing backends.
fn screenshot_floor(width_pt: Option<f64>) -> Option<u32> {
    let points = width_pt?;
    if !points.is_finite() || points <= 0. {
        return None;
    }
    let pixels = (points * DPI / 72.).ceil() + 1.;
    if !pixels.is_finite() || pixels > f64::from(u32::MAX) {
        return None;
    }
    Some(pixels as u32)
}

/// White support required by the current Lanczos3 pipeline, in original pixels.
/// Account for both the configured fit and the possible 1024px encoding fallback.
/// The integer guard covers each resampler's floor/ceil sample endpoints.
fn screenshot_white_support(width: u32, height: u32, max_width: u32, max_height: u32) -> u32 {
    let first = fit(width, height, max_width, max_height);
    let second = fit(first.0, first.1, 1024, 1024);
    let ratio = f64::from(width) / f64::from(first.0);
    let mut support = 0.;
    if first != (width, height) {
        support += 3. * ratio.max(1.) + 1.;
    }
    if second != first {
        support += (3. * (f64::from(first.0) / f64::from(second.0)).max(1.) + 1.) * ratio;
    }
    support.ceil().min(f64::from(u32::MAX)) as u32
}

/// Return a smaller width only if every removed RGB sample is exactly white.
/// A single faint sample, graphic or antialiased edge keeps its entire column.
/// The floor is Office repair provenance, never inferred from input PDF text.
fn screenshot_white_tail(
    rgb: &RgbImage,
    floor: u32,
    max_width: u32,
    max_height: u32,
) -> Option<u32> {
    let (width, height) = rgb.dimensions();
    if floor == 0 || floor >= width || height == 0 || max_width == 0 || max_height == 0 {
        return None;
    }
    let mut content_end = width;
    while content_end > floor {
        let x = content_end - 1;
        if (0..height).any(|y| rgb.get_pixel(x, y).0 != [255, 255, 255]) {
            break;
        }
        content_end -= 1;
    }
    if content_end == width {
        return None;
    }
    let support = screenshot_white_support(width, height, max_width, max_height);
    let retained = content_end.saturating_add(support).min(width);
    if retained >= width
        || retained - content_end
            < screenshot_white_support(retained, height, max_width, max_height)
    {
        // Integer fit rounding can change the support after a crop. If the
        // first bound is insufficient, preserve the full page rather than loop.
        return None;
    }
    Some(retained)
}

fn encode_screenshot(
    rgb: &RgbImage,
    cfg: &Value,
    limit: usize,
    repaired_floor: Option<u32>,
) -> Result<EncodedShot> {
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
    let retained_width = repaired_floor
        .and_then(|floor| screenshot_white_tail(rgb, floor, max_width, max_height))
        .unwrap_or(rgb.width());
    let size = fit(retained_width, rgb.height(), max_width, max_height);
    let prepared;
    let image = if retained_width != rgb.width() {
        // A borrowed view avoids an additional full-size RGB allocation before
        // shrinking. OCR below still owns the original complete page pixels.
        let view = image::imageops::crop_imm(rgb, 0, 0, retained_width, rgb.height());
        prepared = if size != (retained_width, rgb.height()) {
            image::imageops::resize(
                &*view,
                size.0,
                size.1,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            view.to_image()
        };
        &prepared
    } else if size != rgb.dimensions() {
        prepared =
            image::imageops::resize(rgb, size.0, size.1, image::imageops::FilterType::Lanczos3);
        &prepared
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
    /// The rendered pages local OCR found no text on, by number.
    blank_pages: Vec<usize>,
}

/// The one warning for the rendered pages of a `kind` document ("PDF",
/// "Office") of `total` pages that local OCR read without finding text,
/// their numbers as ranges; none when there are none. A PDF page is blank in
/// the output; an Office page keeps its native text without an OCR supplement.
fn no_text_warning(kind: &str, pages: &[usize], total: usize) -> Option<String> {
    if pages.is_empty() {
        return None;
    }
    let one = pages.len() == 1;
    let outcome = match (kind, one) {
        ("PDF", true) => "it is blank in the output",
        ("PDF", false) => "they are blank in the output",
        (_, true) => "the OCR supplement leaves it out",
        (_, false) => "the OCR supplement leaves them out",
    };
    Some(format!(
        "{kind} {} {} ({} of {}): OCR found no text; {outcome}.",
        if one { "page" } else { "pages" },
        crate::formats::page_ranges(pages),
        pages.len(),
        total.max(pages.len())
    ))
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
                let result = ocr::recognize(&asset.bytes, cfg);
                if result.as_ref().is_ok_and(|result| result.capped) {
                    pages
                        .document
                        .warnings
                        .push(ocr::capped_warning(&format!("PDF embedded image {name}")));
                }
                let text = picture_result(
                    name,
                    result.map(|result| result.text),
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
    repaired_min_widths_pt: &[Option<f64>],
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
    let mut blank_pages = Vec::new();
    let mut sizes = Vec::with_capacity(count);
    for page in 1..=count {
        let dpi = session.fitting_dpi(page, DPI)?;
        let size = session.dimensions(page, dpi)?;
        budget.pixels(size.0, size.1)?;
        sizes.push((size, dpi));
    }
    for (index, (size, dpi)) in sizes.into_iter().enumerate() {
        let page = index + 1;
        let pixels = session.render(page, dpi)?;
        if pixels.dimensions() != size {
            return Err(failure(
                "rendered dimensions changed after budget validation",
            ));
        }
        if screenshots {
            let encoded = encode_screenshot(
                &pixels,
                cfg,
                MAX_SHOT_BYTES,
                screenshot_floor(repaired_min_widths_pt.get(index).copied().flatten()),
            )?;
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
            if recognized.capped {
                captured
                    .warnings
                    .push(ocr::capped_warning(&format!("Office page {page}")));
            }
            let text = recognized.text;
            if text.trim().is_empty() {
                blank_pages.push(page);
            }
            captured.ocr.push(text);
        }
    }
    captured
        .warnings
        .extend(no_text_warning("Office", &blank_pages, count));
    Ok(captured)
}

const WEBP_NOTE: &str =
    "PDF screenshots use lossless WebP; image.quality does not affect this encoder.";

/// Scanned pages recognized at once. The portable engine spreads a page's
/// lines over several threads, but a page alone leaves processors idle
/// between its steps and while its last lines are read, so a quarter of the
/// processors' worth of pages run at once (two to four). Its detections, which
/// hold about 300 bytes per page pixel each, take turns separately, which
/// bounds the memory. Vision reads two pages at a time, and so does qppocr,
/// which spreads one image over all of its own threads and whose four-page
/// concurrency was not measured.
fn ocr_pages_at_once() -> usize {
    if ocr::backend() == "paddle" && ocr::portable_engine() == "tract" {
        ocr::thread_budget().div_euclid(4).clamp(2, 4)
    } else {
        2
    }
}

/// Recognized pages kept waiting for an earlier page, past which the next
/// page waits too: their results are small, but a page that takes long must
/// not let the queue grow with the document.
const OCR_PAGES_QUEUED: usize = 16;

/// Tells the page loop that a recognition ended, also when it panicked.
struct Ended(std::sync::mpsc::Sender<()>);

impl Drop for Ended {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

/// A rendered page: its screenshot's diagnostics and its recognition, if any,
/// and the resolution it was rendered at.
struct Pending<'scope> {
    index: usize,
    warnings: Vec<String>,
    recognition: Option<std::thread::ScopedJoinHandle<'scope, Result<ocr::OcrResult>>>,
    dpi: f64,
}

/// A page local OCR read: its blocks, the assets its pictures were written
/// to, and where its lines and pictures are ([`OCR_LAYOUT`]).
struct ReadPage {
    index: usize,
    blocks: Vec<ocr::Block>,
    figures: Vec<Option<String>>,
    layout: Value,
}

/// What the pages local OCR read add up to while they come in: the pages,
/// and the bytes of the pictures cut out of them so far.
#[derive(Default)]
struct ReadPages {
    pages: Vec<ReadPage>,
    figure_bytes: usize,
}

/// The pictures cut out of OCR'd pages take this many bytes at most; the
/// others are left out with a warning.
const MAX_FIGURE_BYTES: usize = 64 * 1024 * 1024;

/// The document metadata entry with where the recognized lines and the cut
/// out pictures of each OCR'd page are, in PDF points from the page's
/// top-left corner (`{"version": 1, "pages": [{"page", "width", "height",
/// "lines": [{"text", "bbox"}], "images": [{"bbox"}]}]}`); a result field
/// ([`crate::ConversionOutput::ocr_layout`]), never front matter.
pub(crate) const OCR_LAYOUT: &str = "ocr_layout";

/// A picture cut out of an OCR'd page, as its Markdown shows it.
fn figure_reference(page: usize, name: &str) -> String {
    format!("![Image on page {page}](.markitai/assets/{name})")
}

/// Write a rendered page's diagnostics and recognized text, in page order:
/// the pictures OCR cut out of it become assets, and its headings are
/// levelled by the page alone until the document's are known
/// ([`level_headings`]).
fn apply_page(
    pages: &mut PdfPages,
    counts: &mut OcrCounts,
    done: Pending<'_>,
    cfg: &Value,
    read: &mut ReadPages,
) -> Result<()> {
    pages.document.warnings.extend(done.warnings);
    let Some(recognition) = done.recognition else {
        return Ok(());
    };
    let page = &mut pages.pages[done.index];
    let result = recognition
        .join()
        .unwrap_or_else(|_| Err(failure("a recognition thread failed")))
        .map_err(|error| failure(format!("local OCR failed on page {}: {error}", page.number)))?;
    if result.unread {
        pages
            .document
            .warnings
            .push(ocr::unread_warning(&format!("PDF page {}", page.number)));
    }
    if result.capped {
        pages
            .document
            .warnings
            .push(ocr::capped_warning(&format!("PDF page {}", page.number)));
    }
    let number = page.number;
    let mut figures = Vec::with_capacity(result.figures.len());
    for (index, figure) in result.figures.iter().enumerate() {
        let encoded = encode_screenshot(&figure.image, cfg, MAX_SHOT_BYTES, None)?;
        if read.figure_bytes + encoded.bytes.len() > MAX_FIGURE_BYTES {
            pages.document.warnings.push(format!(
                "PDF page {number}: a picture OCR found was left out, as the pictures cut out of scanned pages exceed 64 MiB."
            ));
            figures.push(None);
            continue;
        }
        read.figure_bytes += encoded.bytes.len();
        let name = format!(
            "pdf-ocr-page{number}-figure{}.{}",
            index + 1,
            encoded.format.extension()
        );
        pages.document.assets.push(Asset {
            name: name.clone(),
            bytes: encoded.bytes,
        });
        figures.push(Some(name));
    }
    // Points from the page's top-left corner, as the page reader's lines.
    let scale = 72.0 / done.dpi;
    let bbox = |b: [f32; 4]| b.map(|v| ((f64::from(v) * scale) * 100.0).round() / 100.0);
    let images: Vec<Value> = result
        .blocks
        .iter()
        .filter_map(|block| match block {
            ocr::Block::Figure(index) if figures[*index].is_some() => {
                Some(json!({"bbox": bbox(result.figures[*index].bounds)}))
            }
            _ => None,
        })
        .collect();
    let (width, height) = result.size;
    let layout = json!({
        "page": number,
        "width": (f64::from(width) * scale * 100.0).round() / 100.0,
        "height": (f64::from(height) * scale * 100.0).round() / 100.0,
        "lines": result
            .lines
            .iter()
            .map(|line| json!({"text": line.text, "bbox": bbox(line.bounds)}))
            .collect::<Vec<_>>(),
        "images": images,
    });
    let tiers = ocr::tiers(&result.blocks);
    page.markdown = ocr::render(&result.blocks, &tiers, &|index| {
        figures[index]
            .as_deref()
            .map(|name| figure_reference(number, name))
    });
    read.pages.push(ReadPage {
        index: done.index,
        blocks: result.blocks,
        figures,
        layout,
    });
    page.needs_ocr = false;
    page.ocr_reason = None;
    page.ocr_completed = true;
    // The page's text is now this recognition's, not its OCR layer's.
    page.ocr_layer = None;
    if page.markdown.trim().is_empty() {
        counts.blank += 1;
        counts.blank_pages.push(page.number);
    } else {
        counts.nonempty += 1;
    }
    Ok(())
}

/// The OCR'd pages' headings levelled across the document (a heading as
/// large as the title's on another page is of the first level, not the
/// largest of its own page's), and where their lines and pictures are, for
/// the result ([`OCR_LAYOUT`]).
fn level_headings(pages: &mut PdfPages, read: Vec<ReadPage>) {
    if read.is_empty() {
        return;
    }
    let mut sizes: Vec<f32> = read
        .iter()
        .flat_map(|page| &page.blocks)
        .filter_map(|block| match block {
            ocr::Block::Heading { size, .. } => Some(*size),
            _ => None,
        })
        .collect();
    let tiers = ocr::tiers_of(&mut sizes);
    let mut layouts = Vec::with_capacity(read.len());
    for page in read {
        let target = &mut pages.pages[page.index];
        let number = target.number;
        target.markdown = ocr::render(&page.blocks, &tiers, &|index| {
            page.figures[index]
                .as_deref()
                .map(|name| figure_reference(number, name))
        });
        layouts.push(page.layout);
    }
    pages
        .document
        .metadata
        .insert(OCR_LAYOUT.into(), json!({"version": 1, "pages": layouts}));
}

pub(crate) fn prepare(
    bytes: &[u8],
    prefix: &str,
    cfg: &Value,
    vlm_optout: bool,
) -> Result<PreparedPdf> {
    prepare_with(
        bytes,
        prefix,
        cfg,
        vlm_optout,
        &ocr::recognize_page,
        &ocr::read_lines,
    )
}

/// What recognizes a rendered page; [`ocr::recognize_page`] outside tests.
type Recognize<'a> = dyn Fn(RgbImage, &Value) -> Result<ocr::OcrResult> + Sync + 'a;
/// What reads the glyph check's line crops; [`ocr::read_lines`] outside tests.
type ReadLines<'a> = dyn Fn(Vec<RgbImage>, &Value) -> Result<Vec<String>> + Sync + 'a;

fn prepare_with(
    bytes: &[u8],
    prefix: &str,
    cfg: &Value,
    vlm_optout: bool,
    recognize: &Recognize<'_>,
    read_lines: &ReadLines<'_>,
) -> Result<PreparedPdf> {
    validate_name(prefix)?;
    let mut pages = extract_pdf_pages_bounded_with_config(bytes, MAX_PAGES, cfg)?;
    // With local OCR routing pages, a text layer the text-only checks
    // trusted is also compared with what its glyphs draw (a few crops per
    // font, read by the recognizer); a page that fails is read by OCR.
    // A page that failed is drawn from the copy the check drew it from.
    let mut drawn = None;
    if Plan::local_ocr(cfg, vlm_optout) && Plan::routing(cfg) {
        let mut lap = ocr::trace::Lap::start();
        let report = check_pdf_glyphs(bytes, &mut pages, cfg, &|images| read_lines(images, cfg));
        ocr::trace::line(format_args!(
            "glyph-check crops={} {:.0}ms",
            report.crops,
            lap.ms()
        ));
        drawn = report.render;
    }
    let plan = Plan::new(&pages, cfg, vlm_optout)?;
    // Why each page OCR reads is read, before its recognition replaces the
    // reader's verdict: its page source.
    for (page, &recognize) in pages.pages.iter_mut().zip(&plan.recognize) {
        if recognize {
            page.ocr_note = Some(if plan.routing {
                page.layer_reason()
            } else {
                crate::formats::PDF_OCR_FORCED.into()
            });
        }
    }
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
        Some(PdfRasterSession::open(drawn.as_deref().unwrap_or(bytes))?)
    };
    let mut sizes = Vec::with_capacity(render_indices.len());
    if let Some(session) = &session {
        if session.pages() != pages.pages.len() {
            return Err(failure("renderer and extractor disagree on page count"));
        }
        for &index in &render_indices {
            // A poster-sized page is drawn smaller, to fit the pixel limit.
            let number = pages.pages[index].number;
            let dpi = session.fitting_dpi(number, DPI)?;
            let size = session.dimensions(number, dpi)?;
            budget.pixels(size.0, size.1)?;
            sizes.push((size, dpi));
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
    // Pages are rendered here, in order (a session stays on its thread), and
    // recognized on up to `ocr_pages_at_once` threads at once: the next page
    // starts as soon as any recognition ends, not only the oldest. Each page's
    // diagnostics are written in page order once its recognition is in.
    let mut webp_noted = false;
    let mut read = ReadPages::default();
    let at_once = ocr_pages_at_once();
    std::thread::scope(|scope| -> Result<()> {
        let mut pending: std::collections::VecDeque<Pending<'_>> =
            std::collections::VecDeque::new();
        let (ended, endings) = std::sync::mpsc::channel::<()>();
        let mut running = 0usize;
        for (&index, &(size, dpi)) in render_indices.iter().zip(&sizes) {
            let number = pages.pages[index].number;
            let mut lap = ocr::trace::Lap::start();
            let pixels = session
                .as_ref()
                .expect("render indices require a session")
                .render(number, dpi)?;
            ocr::trace::set_page(number);
            ocr::trace::line(format_args!(
                "render={:.0}ms size={}x{}",
                lap.ms(),
                pixels.width(),
                pixels.height()
            ));
            // This thread goes on to other documents' images.
            ocr::trace::clear_page();
            if pixels.dimensions() != size {
                return Err(failure(
                    "rendered dimensions changed after budget validation",
                ));
            }
            let mut warnings = Vec::new();
            if plan.screenshots {
                let encoded = encode_screenshot(&pixels, cfg, MAX_SHOT_BYTES, None)?;
                budget.screenshot(encoded.bytes.len())?;
                if encoded.fallback {
                    warnings.push(format!(
                        "PDF page {number}: screenshot size required JPEG compression fallback."
                    ));
                } else if encoded.format == Encoding::Webp
                    && !webp_noted
                    && !pages
                        .document
                        .warnings
                        .iter()
                        .any(|warning| warning == WEBP_NOTE)
                {
                    webp_noted = true;
                    warnings.push(WEBP_NOTE.into());
                }
                screenshots.push(Asset {
                    name: format!("{prefix}.page{number:04}.{}", encoded.format.extension()),
                    bytes: encoded.bytes,
                });
                screenshot_pages.push(index);
            }
            // Wait for a recognition to end only when another would exceed
            // the limit: the next page renders while the others run.
            if plan.recognize[index] {
                running -= endings.try_iter().count();
                while running >= at_once {
                    // A sender stays here, so this waits for an ending.
                    let _ = endings.recv();
                    running -= 1;
                }
                while pending.len() >= OCR_PAGES_QUEUED {
                    let next = pending.pop_front().expect("a page is pending");
                    apply_page(&mut pages, &mut counts, next, cfg, &mut read)?;
                }
            }
            let recognition = plan.recognize[index].then(|| {
                counts.attempted += 1;
                running += 1;
                let ended = Ended(ended.clone());
                scope.spawn(move || {
                    let _ended = ended;
                    ocr::trace::set_page(number);
                    recognize(pixels, cfg)
                })
            });
            pending.push_back(Pending {
                index,
                warnings,
                recognition,
                dpi,
            });
            while pending.front().is_some_and(|p| {
                p.recognition
                    .as_ref()
                    .is_none_or(std::thread::ScopedJoinHandle::is_finished)
            }) {
                let next = pending.pop_front().expect("a page is pending");
                apply_page(&mut pages, &mut counts, next, cfg, &mut read)?;
            }
        }
        while let Some(next) = pending.pop_front() {
            apply_page(&mut pages, &mut counts, next, cfg, &mut read)?;
        }
        Ok(())
    })?;
    level_headings(&mut pages, read.pages);
    let total = pages.pages.len();
    pages
        .document
        .warnings
        .extend(no_text_warning("PDF", &counts.blank_pages, total));
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
        vision_reads: plan.vision_reads,
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
            let encoded = encode_screenshot(&rgb, &cfg, MAX_SHOT_BYTES, None).unwrap();
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
        let encoded = encode_screenshot(&noisy, &cfg, 12_000, None).unwrap();
        assert!(encoded.fallback);
        assert_eq!(encoded.format.extension(), "jpg");
        assert_eq!(
            image::guess_format(&encoded.bytes).unwrap(),
            ImageFormat::Jpeg
        );
        assert!(encoded.bytes.len() <= 12_000);
        assert!(encode_screenshot(&noisy, &cfg, 4, None).is_err());
    }

    #[test]
    fn repaired_tail_removes_only_added_white_columns_and_keeps_ocr_original() {
        let mut rgb = RgbImage::from_pixel(160, 12, Rgb([255, 255, 255]));
        rgb.put_pixel(20, 6, Rgb([0, 0, 0]));
        // User's existing blank canvas through column 79 must not be trimmed.
        let original = rgb.clone();
        let cfg = json!({"image":{"format":"png","max_width":160,"max_height":100}});
        let encoded = encode_screenshot(&rgb, &cfg, MAX_SHOT_BYTES, Some(80)).unwrap();
        let decoded = image::load_from_memory(&encoded.bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (80, 12));
        for y in 0..12 {
            for x in 0..80 {
                assert_eq!(decoded.get_pixel(x, y), rgb.get_pixel(x, y));
            }
        }
        assert_eq!(rgb, original);
        let untouched = encode_screenshot(&rgb, &cfg, MAX_SHOT_BYTES, None).unwrap();
        assert_eq!(
            image::load_from_memory(&untouched.bytes).unwrap().width(),
            160
        );
        let no_repair = encode_screenshot(&rgb, &cfg, MAX_SHOT_BYTES, Some(160)).unwrap();
        assert_eq!(untouched.bytes, no_repair.bytes);
    }

    #[test]
    fn repaired_tail_keeps_faint_graphics_beyond_text_floor_and_never_cuts_white_holes() {
        let mut rgb = RgbImage::from_pixel(160, 12, Rgb([255, 255, 255]));
        rgb.put_pixel(20, 0, Rgb([0, 0, 0]));
        rgb.put_pixel(120, 11, Rgb([255, 254, 255]));
        let cfg = json!({"image":{"format":"png","max_width":160,"max_height":100}});
        let encoded = encode_screenshot(&rgb, &cfg, MAX_SHOT_BYTES, Some(80)).unwrap();
        let decoded = image::load_from_memory(&encoded.bytes).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (121, 12));
        assert_eq!(decoded.get_pixel(120, 11).0, [255, 254, 255]);
        assert_eq!(decoded.get_pixel(20, 0).0, [0, 0, 0]);
        rgb.put_pixel(159, 3, Rgb([254, 255, 255]));
        assert_eq!(screenshot_white_tail(&rgb, 80, 160, 100), None);
    }

    #[test]
    fn repaired_tail_filter_support_scales_with_downsampling_and_fallback() {
        let mut rgb = RgbImage::from_pixel(4000, 80, Rgb([255, 255, 255]));
        for y in 0..80 {
            rgb.put_pixel(2500, y, Rgb([0, 180, 0]));
        }
        let retained = screenshot_white_tail(&rgb, 2000, 200, 100).unwrap();
        assert!(
            retained > 2504 && retained < 4000,
            "not a fixed three-pixel guard"
        );
        let cfg = json!({"image":{"format":"png","max_width":200,"max_height":100}});
        let encoded = encode_screenshot(&rgb, &cfg, MAX_SHOT_BYTES, Some(2000)).unwrap();
        let decoded = image::load_from_memory(&encoded.bytes).unwrap().to_rgb8();
        assert_eq!(decoded.width(), 200);
        assert!(
            decoded.pixels().any(|p| p.0 != [255, 255, 255]),
            "thin graphic survives"
        );
        assert!(
            (0..decoded.height()).all(|y| decoded.get_pixel(199, y).0 == [255, 255, 255]),
            "white filter support survives at the edge"
        );
        // Even when the first fit is disabled, a later 1024px size fallback
        // needs source-space support greater than three columns.
        let fallback = screenshot_white_tail(&rgb, 2000, 4000, 100).unwrap();
        assert!(fallback > 2504 && fallback < 4000);
    }

    #[test]
    fn repaired_floor_covers_fractional_geometry_and_invalid_metadata_is_noop() {
        assert_eq!(screenshot_floor(Some(10.)), Some(22));
        assert_eq!(screenshot_floor(Some(36.)), Some(76));
        for width in [
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(-1.),
            Some(0.),
            Some(f64::MAX),
        ] {
            assert_eq!(screenshot_floor(width), None);
        }
        let white = RgbImage::from_pixel(100, 4, Rgb([255, 255, 255]));
        assert_eq!(screenshot_white_tail(&white, 80, 100, 100), Some(80));
        assert_eq!(screenshot_white_tail(&white, 0, 100, 100), None);
        assert_eq!(screenshot_white_tail(&white, 100, 100, 100), None);
        assert_eq!(screenshot_white_tail(&white, u32::MAX, 100, 100), None);
        // Cropping metadata cannot bypass the complete raster's existing budget.
        assert!(Budget::default().pixels(8_000, 4_001).is_err());
    }

    #[test]
    fn repaired_tail_height_limited_fit_keeps_color_and_original_height() {
        let mut rgb = RgbImage::from_pixel(400, 800, Rgb([255, 255, 255]));
        for y in 0..800 {
            rgb.put_pixel(250, y, Rgb([0, 0, 0]));
        }
        let retained = screenshot_white_tail(&rgb, 200, 1920, 200).unwrap();
        assert!(retained > 254 && retained < 400);
        let cfg = json!({"image":{"format":"png","max_width":1920,"max_height":200}});
        let encoded = encode_screenshot(&rgb, &cfg, MAX_SHOT_BYTES, Some(200)).unwrap();
        let decoded = image::load_from_memory(&encoded.bytes).unwrap().to_rgb8();
        assert_eq!(decoded.height(), 200);
        assert!(decoded.pixels().any(|p| p.0 != [255, 255, 255]));
        assert_eq!(rgb.height(), 800);
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
            vision_reads: false,
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
    fn placement_notice_stays_only_without_screenshots() {
        // Local OCR alone renders pages but publishes no screenshot. The
        // picture is on the page of text: a scan's picture is no notice's.
        let mut pages = pages();
        pages.pages[0].asset_names = std::mem::take(&mut pages.pages[1].asset_names);
        assert!(!pages.pages[0].asset_names.is_empty());
        let prepared = PreparedPdf {
            pages,
            screenshots: Vec::new(),
            screenshot_pages: Vec::new(),
            has_reliable_text: true,
            completed_media: false,
            media_requested: true,
            vision_reads: false,
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
            vision_reads: false,
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

    /// The glyph check's reader for documents without a text layer.
    fn no_lines(_: Vec<RgbImage>, _: &Value) -> Result<Vec<String>> {
        unreachable!("a page without a text layer has no glyphs to check")
    }

    /// Image-only pages, each one gray of its own (page `n` at `40 * n`).
    fn scans(count: u8) -> Vec<u8> {
        scans_sized(count, 72, 96)
    }

    /// [`scans`] on pages `width` × `height` points large.
    fn scans_sized(count: u8, width: i64, height: i64) -> Vec<u8> {
        use lopdf::{Dictionary, Object, Stream, dictionary};
        let mut pdf = lopdf::Document::with_version("1.7");
        let tree = pdf.new_object_id();
        let mut kids = Vec::new();
        for page in 1..=count {
            let gray = 40 * page;
            let image = pdf.add_object(Stream::new(
                dictionary! {
                    "Type" => "XObject", "Subtype" => "Image", "Width" => 1, "Height" => 1,
                    "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8
                },
                vec![gray; 3],
            ));
            let content = pdf.add_object(Stream::new(
                Dictionary::new(),
                format!("q {width} 0 0 {height} 0 0 cm /Scan Do Q").into_bytes(),
            ));
            kids.push(Object::Reference(pdf.add_object(dictionary! {
                "Type" => "Page", "Parent" => tree, "Contents" => content,
                "Resources" => dictionary! { "XObject" => dictionary! { "Scan" => image } }
            })));
        }
        pdf.objects.insert(
            tree,
            dictionary! {
                "Type" => "Pages", "Count" => i64::from(count), "Kids" => kids,
                "MediaBox" => vec![0.into(), 0.into(), width.into(), height.into()]
            }
            .into(),
        );
        let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
        pdf.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn scanned_pages_are_recognized_several_at_a_time_and_written_in_page_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (running, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let recognize = |pixels: RgbImage, _: &Value| -> Result<ocr::OcrResult> {
            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
            most.fetch_max(now, Ordering::SeqCst);
            // Wait (a bounded while) for the pages that may run beside this
            // one, so on a loaded machine a page that started late still
            // overlaps the others rather than missing them.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while most.load(Ordering::SeqCst) < ocr_pages_at_once()
                && std::time::Instant::now() < deadline
            {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            let (width, height) = pixels.dimensions();
            let page = (u64::from(pixels.get_pixel(width / 2, height / 2).0[0]) + 20) / 40;
            // Earlier pages take longer, so later ones finish first.
            std::thread::sleep(std::time::Duration::from_millis(80 * (5 - page)));
            running.fetch_sub(1, Ordering::SeqCst);
            let mut result = ocr::OcrResult::of_text(if page == 2 {
                String::new()
            } else {
                format!("Recognized page {page}.")
            });
            result.unread = page == 3;
            Ok(result)
        };
        let cfg =
            json!({"ocr":{"enabled":true},"llm":{"enabled":false},"screenshot":{"enabled":false}});
        let prepared =
            prepare_with(&scans(4), "scan.pdf", &cfg, false, &recognize, &no_lines).unwrap();
        assert_eq!(most.load(Ordering::SeqCst), ocr_pages_at_once());
        let (document, _) = prepared.finish().unwrap();
        let markdown = &document.markdown;
        let at = |text: &str| {
            markdown
                .find(text)
                .unwrap_or_else(|| panic!("{text}: {markdown}"))
        };
        assert!(
            at("Recognized page 1.") < at("Recognized page 3.")
                && at("Recognized page 3.") < at("Recognized page 4.")
        );
        assert_eq!(document.metadata["ocr_pages_attempted"], 4);
        assert_eq!(document.metadata["ocr_pages_blank"], 1);
        // Each page's notices in page order, though page 3 finished first.
        let warning = |text: &str| {
            document
                .warnings
                .iter()
                .position(|warning| warning.contains(text))
                .unwrap_or_else(|| panic!("{text}: {:?}", document.warnings))
        };
        assert!(warning("PDF page 3") < warning("PDF page 2 (1 of 4): OCR found no text"));
        assert_eq!(
            document
                .warnings
                .iter()
                .filter(|warning| warning.contains("OCR found no text"))
                .count(),
            1
        );
    }

    /// A scan's headings are levelled across its pages, the pictures OCR cut
    /// out are written as assets where they stand, the scans themselves give
    /// way to the text read from them, and where the lines and pictures are
    /// is kept in points for the result.
    #[test]
    fn ocr_pages_level_headings_across_the_document_and_keep_their_pictures() {
        let recognize = |pixels: RgbImage, _: &Value| -> Result<ocr::OcrResult> {
            let (width, height) = pixels.dimensions();
            let page = (u64::from(pixels.get_pixel(width / 2, height / 2).0[0]) + 20) / 40;
            let mut result = ocr::OcrResult::default();
            result.size = (width, height);
            result.confidence = 1.0;
            let heading = |size: f32, text: &str| ocr::Block::Heading {
                size,
                text: text.into(),
            };
            if page == 1 {
                result.blocks = vec![
                    heading(2.2, "Title"),
                    ocr::Block::Text("Read text.".into()),
                    ocr::Block::Figure(0),
                ];
                result.figures = vec![ocr::Figure {
                    bounds: [15.0, 30.0, 75.0, 90.0],
                    image: RgbImage::from_pixel(60, 60, image::Rgb([200, 40, 0])),
                }];
                result.lines = vec![ocr::Recognized {
                    text: "Title".into(),
                    bounds: [15.0, 0.0, 75.0, 15.0],
                }];
            } else {
                // The largest heading of its own page, but no title.
                result.blocks = vec![heading(1.4, "Section"), ocr::Block::Text("More.".into())];
            }
            result.text = "unused".into();
            Ok(result)
        };
        let cfg =
            json!({"ocr":{"enabled":true},"llm":{"enabled":false},"screenshot":{"enabled":false}});
        let prepared =
            prepare_with(&scans(2), "scan.pdf", &cfg, false, &recognize, &no_lines).unwrap();
        let (document, _) = prepared.finish().unwrap();
        let markdown = &document.markdown;
        assert!(markdown.contains("# Title\n\nRead text.\n\n![Image on page 1](.markitai/assets/pdf-ocr-page1-figure1.jpg)"), "{markdown}");
        assert!(markdown.contains("## Section\n\nMore."), "{markdown}");
        assert!(!markdown.contains("pdf-image-"), "{markdown}");
        assert_eq!(
            document
                .assets
                .iter()
                .map(|asset| asset.name.as_str())
                .collect::<Vec<_>>(),
            ["pdf-ocr-page1-figure1.jpg"]
        );
        let layout = &document.metadata[OCR_LAYOUT];
        assert_eq!(layout["version"], 1);
        let first = &layout["pages"][0];
        assert_eq!(
            (first["page"].clone(), first["width"].clone()),
            (json!(1), json!(72.0))
        );
        assert_eq!(first["images"], json!([{"bbox": [7.2, 14.4, 36.0, 43.2]}]));
        assert_eq!(
            first["lines"],
            json!([{"text": "Title", "bbox": [7.2, 0.0, 36.0, 7.2]}])
        );
        assert_eq!(layout["pages"][1]["page"], 2);
    }

    /// One page of Helvetica lines whose ToUnicode map reads each lower-case
    /// letter as the next one: the glyphs say `the committee`, the text
    /// layer `uif dpnnjuuff`.
    fn shifted_pdf() -> Vec<u8> {
        use lopdf::{Object, Stream, dictionary};
        let mut cmap = String::from(
            "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n\
             /CMapName /Shifted def /CMapType 2 def\n\
             1 begincodespacerange <00> <FF> endcodespacerange\n95 beginbfchar\n",
        );
        for byte in 0x20u8..0x7F {
            let read = if byte.is_ascii_lowercase() {
                b'a' + (byte - b'a' + 1) % 26
            } else {
                byte
            };
            cmap.push_str(&format!("<{byte:02X}> <{read:04X}>\n"));
        }
        cmap.push_str("endbfchar\nendcmap CMapName currentdict /CMap defineresource pop end end\n");
        let mut pdf = lopdf::Document::with_version("1.4");
        let map = pdf.add_object(Stream::new(dictionary! {}, cmap.into_bytes()));
        let font = pdf.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
            "Encoding" => "WinAnsiEncoding", "ToUnicode" => map,
        });
        let content: String = (0..6)
            .map(|index| {
                format!(
                    "BT /F1 12 Tf 72 {} Td (the committee met on tuesday to review the budget) Tj ET\n",
                    720 - 20 * index
                )
            })
            .collect();
        let content = pdf.add_object(Stream::new(dictionary! {}, content.into_bytes()));
        let tree = pdf.new_object_id();
        let page = pdf.add_object(dictionary! {
            "Type" => "Page", "Parent" => tree, "Contents" => content,
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font } },
        });
        pdf.objects.insert(
            tree,
            dictionary! {
                "Type" => "Pages", "Count" => 1, "Kids" => vec![Object::Reference(page)],
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            }
            .into(),
        );
        let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
        pdf.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn a_layer_that_is_not_what_its_glyphs_draw_is_read_by_ocr() {
        let recognize = |_: RgbImage, _: &Value| -> Result<ocr::OcrResult> {
            Ok(ocr::OcrResult::of_text(
                "Recognized: the committee met.".into(),
            ))
        };
        let read_lines = |images: Vec<RgbImage>, _: &Value| -> Result<Vec<String>> {
            Ok(images
                .iter()
                .map(|_| "the committee met on".to_owned())
                .collect())
        };
        let cfg =
            json!({"ocr":{"enabled":true},"llm":{"enabled":false},"screenshot":{"enabled":false}});
        let prepared = prepare_with(
            &shifted_pdf(),
            "s.pdf",
            &cfg,
            false,
            &recognize,
            &read_lines,
        )
        .unwrap();
        let (document, _) = prepared.finish().unwrap();
        assert!(document.markdown.contains("Recognized: the committee met."));
        assert!(!document.markdown.contains("uif dpnnjuuff"));
        let source = &document.metadata[crate::formats::PDF_PAGE_SOURCES][0];
        assert_eq!(source["source"], "ocr");
        assert_eq!(
            source["reason"],
            "text layer failed the script check: the layer's text is not what its glyphs draw (font Helvetica)"
        );
        // Without OCR routing pages, or with a vision model reading them,
        // nothing is cropped.
        let never =
            |_: Vec<RgbImage>, _: &Value| -> Result<Vec<String>> { unreachable!("no glyph check") };
        for cfg in [
            json!({"ocr":{"enabled":false}}),
            json!({"ocr":{"enabled":true,"per_page_routing":false},"llm":{"enabled":false}}),
            json!({"ocr":{"enabled":true},"llm":{"enabled":true}}),
        ] {
            let prepared = prepare_with(&shifted_pdf(), "s.pdf", &cfg, false, &recognize, &never);
            assert!(prepared.is_ok());
        }
    }

    #[test]
    fn a_poster_page_is_recognized_at_a_lower_resolution_instead_of_failing() {
        // 200 × 200 inches: 900 million pixels at 150 dpi.
        let poster = scans_sized(1, 14_400, 14_400);
        let seen = std::sync::Mutex::new(None);
        let recognize = |pixels: RgbImage, _: &Value| -> Result<ocr::OcrResult> {
            *seen.lock().unwrap() = Some(pixels.dimensions());
            Ok(ocr::OcrResult::of_text("Recognized poster.".into()))
        };
        let cfg =
            json!({"ocr":{"enabled":true},"llm":{"enabled":false},"screenshot":{"enabled":true}});
        let prepared =
            prepare_with(&poster, "poster.pdf", &cfg, false, &recognize, &no_lines).unwrap();
        let (width, height) = seen.lock().unwrap().expect("the page was recognized");
        assert_eq!(width, height);
        let pixels = u64::from(width) * u64::from(height);
        assert!((31_000_000..=MAX_PAGE_PIXELS).contains(&pixels), "{pixels}");
        assert_eq!(prepared.screenshots.len(), 1);
        let (document, _) = prepared.finish().unwrap();
        assert!(document.markdown.contains("Recognized poster."));
    }

    #[test]
    fn a_slow_page_does_not_hold_up_the_pages_after_it() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (running, most, started) = (
            AtomicUsize::new(0),
            AtomicUsize::new(0),
            AtomicUsize::new(0),
        );
        let started_while_first_ran = AtomicUsize::new(0);
        let recognize = |pixels: RgbImage, _: &Value| -> Result<ocr::OcrResult> {
            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
            most.fetch_max(now, Ordering::SeqCst);
            started.fetch_add(1, Ordering::SeqCst);
            let (width, height) = pixels.dimensions();
            let page = (u64::from(pixels.get_pixel(width / 2, height / 2).0[0]) + 20) / 40;
            let pause = if page == 1 { 900 } else { 30 };
            std::thread::sleep(std::time::Duration::from_millis(pause));
            if page == 1 {
                started_while_first_ran.store(started.load(Ordering::SeqCst), Ordering::SeqCst);
            }
            running.fetch_sub(1, Ordering::SeqCst);
            Ok(ocr::OcrResult::of_text(format!("Recognized page {page}.")))
        };
        let cfg =
            json!({"ocr":{"enabled":true},"llm":{"enabled":false},"screenshot":{"enabled":false}});
        let prepared =
            prepare_with(&scans(6), "scan.pdf", &cfg, false, &recognize, &no_lines).unwrap();
        assert!(most.load(Ordering::SeqCst) <= ocr_pages_at_once());
        // The other five pages were read while the first one still ran.
        assert_eq!(started_while_first_ran.load(Ordering::SeqCst), 6);
        let (document, _) = prepared.finish().unwrap();
        let markdown = &document.markdown;
        let positions: Vec<usize> = (1..=6)
            .map(|page| markdown.find(&format!("Recognized page {page}.")).unwrap())
            .collect();
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "{markdown}"
        );
    }

    #[test]
    fn pages_without_text_are_one_warning_with_ranges_and_the_page_count() {
        assert_eq!(no_text_warning("PDF", &[], 10), None);
        assert_eq!(
            no_text_warning("PDF", &[57], 318).unwrap(),
            "PDF page 57 (1 of 318): OCR found no text; it is blank in the output."
        );
        assert_eq!(
            no_text_warning("PDF", &[57, 212], 318).unwrap(),
            "PDF pages 57, 212 (2 of 318): OCR found no text; they are blank in the output."
        );
        assert_eq!(
            no_text_warning("PDF", &[3, 4, 5, 9, 40, 41, 42, 43, 44], 120).unwrap(),
            "PDF pages 3\u{2013}5, 9, 40\u{2013}44 (9 of 120): OCR found no text; they are blank in the output."
        );
        assert_eq!(
            no_text_warning("Office", &[2], 3).unwrap(),
            "Office page 2 (1 of 3): OCR found no text; the OCR supplement leaves it out."
        );
        assert_eq!(
            no_text_warning("Office", &[2, 3], 3).unwrap(),
            "Office pages 2\u{2013}3 (2 of 3): OCR found no text; the OCR supplement leaves them out."
        );
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
        assert_eq!(document.metadata["ocr_path"], crate::ocr::backend());
        assert_eq!(document.markdown.matches("<!-- Page number:").count(), 3);
        assert!(document.warnings.iter().any(|warning| {
            warning == "PDF page 3 (1 of 3): OCR found no text; it is blank in the output."
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
