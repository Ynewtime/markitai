//! Pure-Rust page rendering with hayro, for every platform without CoreGraphics.
//!
//! The session keeps one parsed document and draws a page at a time into an
//! opaque white pixmap, then drops the alpha channel. Geometry, limits and the
//! returned pixels follow the CoreGraphics backend: CropBox intersected with
//! MediaBox, the page's quarter-turn rotation, ceil-rounded dimensions at the
//! requested DPI. The content is scaled uniformly from the top-left corner, so
//! rounding leaves at most one white pixel column or row at the right or
//! bottom edge where CoreGraphics centres it.

use super::{RasterSize, Result, failure, raster_size};
use crate::Error;
use hayro::hayro_interpret::{InterpreterSettings, InterpreterWarning};
use hayro::hayro_syntax::page::{Page, Rotation};
use hayro::hayro_syntax::{DecryptionError, LoadPdfError, Pdf};
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::{RenderCache, RenderSettings};
use image::RgbImage;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

mod fonts;

#[cfg(test)]
pub(crate) use fonts::host_has_cjk_face;

/// hayro sizes its pixmaps with 16-bit sides.
const MAX_SIDE: u32 = u16::MAX as u32;

/// The parsed document owns a copy of the input (or of its decrypted form),
/// so the caller's buffer may be dropped. Caches hold `Rc`s, so the session
/// stays on its caller's thread.
pub(crate) struct PdfRasterSession {
    pdf: Pdf,
    pages: usize,
    settings: InterpreterSettings,
    warnings: Arc<Warnings>,
    _thread: PhantomData<Rc<()>>,
}

/// What hayro reported it could not draw: a font it does not support, or
/// an image it could not decode. The page is still drawn without it.
#[derive(Default)]
struct Warnings {
    fonts: AtomicUsize,
    images: AtomicUsize,
}

/// hayro reads page boxes as `f32`, so a box written `593.76` arrives as
/// 593.760009765625 and a 150 DPI page would be one pixel wider than the
/// CoreGraphics frame. The shortest decimal that reads back as the same
/// `f32` is the number the file wrote (up to `f32`'s seven digits); the pixel
/// size is computed from that.
fn written(value: f64) -> f64 {
    let single = value as f32;
    if f64::from(single) != value || !single.is_finite() {
        return value;
    }
    single.to_string().parse().unwrap_or(value)
}

fn locked() -> Error {
    failure("PDF is locked; password-protected rendering is not supported")
}

/// The document hayro should read. An encrypted document is decrypted by the
/// vendored lopdf, the reader text extraction uses, so text and pixels come
/// from one decryption: it opens a document whose user password is empty
/// (owner-password protection only) and leaves one that needs a password
/// encrypted. If lopdf cannot read the file at all, hayro gets the original
/// bytes and applies its own empty-password decryption.
fn plaintext(bytes: &[u8]) -> Result<Vec<u8>> {
    // `/Encrypt` must appear in a trailer or cross-reference stream
    // dictionary, neither of which is compressed. A match inside content only
    // costs one unnecessary lopdf load.
    if memchr::memmem::find(bytes, b"/Encrypt").is_none() {
        return Ok(bytes.to_vec());
    }
    match lopdf::Document::load_mem(bytes) {
        Ok(document) if document.is_encrypted() => Err(locked()),
        Ok(mut document) if document.was_encrypted() => {
            let mut decrypted = Vec::new();
            document
                .save_to(&mut decrypted)
                .map_err(|_| failure("cannot prepare the decrypted document"))?;
            Ok(decrypted)
        }
        _ => Ok(bytes.to_vec()),
    }
}

impl PdfRasterSession {
    /// `bytes` has passed the shared input checks (size bound, PDF header).
    pub(crate) fn open(bytes: &[u8]) -> Result<Self> {
        let data = plaintext(bytes)?;
        let pdf = catch_unwind(move || Pdf::new(Arc::new(data)))
            .map_err(|_| failure("the portable renderer failed while reading the document"))?
            .map_err(|error| match error {
                LoadPdfError::Decryption(DecryptionError::PasswordProtected) => locked(),
                LoadPdfError::Decryption(_) => {
                    failure("PDF encryption is invalid or uses an unsupported algorithm")
                }
                LoadPdfError::Invalid => failure("cannot open the PDF document"),
            })?;
        let pages = pdf.pages().len();
        if pages == 0 {
            return Err(failure("document contains no readable pages"));
        }
        let warnings = Arc::new(Warnings::default());
        let sink = Arc::clone(&warnings);
        let settings = InterpreterSettings {
            font_resolver: Arc::new(fonts::resolve),
            warning_sink: Arc::new(move |warning| {
                let counter = match warning {
                    InterpreterWarning::UnsupportedFont => &sink.fonts,
                    InterpreterWarning::ImageDecodeFailure => &sink.images,
                };
                counter.fetch_add(1, Ordering::Relaxed);
            }),
            ..InterpreterSettings::default()
        };
        Ok(Self {
            pdf,
            pages,
            settings,
            warnings,
            _thread: PhantomData,
        })
    }

    pub(crate) fn pages(&self) -> usize {
        self.pages
    }

    /// Fonts and images hayro has skipped in this session so far.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn warnings(&self) -> (usize, usize) {
        (
            self.warnings.fonts.load(Ordering::Relaxed),
            self.warnings.images.load(Ordering::Relaxed),
        )
    }

    fn frame(&self, page_1_based: usize, dpi: f64) -> Result<(&Page<'_>, RasterSize)> {
        if page_1_based == 0 || page_1_based > self.pages {
            return Err(failure("page number is outside the document"));
        }
        let page = &self.pdf.pages()[page_1_based - 1];
        let mut boxes = [[0.; 4]; 2];
        for (target, rect) in boxes.iter_mut().zip([page.media_box(), page.crop_box()]) {
            *target = [rect.x0, rect.y0, rect.x1, rect.y1].map(written);
            let [x0, y0, x1, y1] = *target;
            if target.iter().any(|value| !value.is_finite()) || x1 - x0 <= 0. || y1 - y0 <= 0. {
                return Err(failure("page box is empty or has invalid coordinates"));
            }
        }
        let [[mx0, my0, mx1, my1], [cx0, cy0, cx1, cy1]] = boxes;
        let visible = (mx1.min(cx1) - mx0.max(cx0), my1.min(cy1) - my0.max(cy0));
        if visible.0 <= 0. || visible.1 <= 0. {
            return Err(failure("page box is empty or has invalid coordinates"));
        }
        let (width, height) = match page.rotation() {
            Rotation::Horizontal | Rotation::FlippedHorizontal => (visible.1, visible.0),
            Rotation::None | Rotation::Flipped => visible,
        };
        let size = raster_size(width, height, dpi)?;
        if size.width > MAX_SIDE || size.height > MAX_SIDE {
            return Err(failure(
                "page side exceeds the portable renderer's 65,535 pixel limit",
            ));
        }
        Ok((page, size))
    }

    /// Checks the complete page geometry without allocating its pixel buffers.
    pub(crate) fn dimensions(&self, page_1_based: usize, dpi: f64) -> Result<(u32, u32)> {
        let (_, size) = self.frame(page_1_based, dpi)?;
        Ok((size.width, size.height))
    }

    pub(crate) fn render(&self, page_1_based: usize, dpi: f64) -> Result<RgbImage> {
        let (page, size) = self.frame(page_1_based, dpi)?;
        let scale = (dpi / 72.) as f32;
        let target = RenderSettings {
            x_scale: scale,
            y_scale: scale,
            width: Some(size.width as u16),
            height: Some(size.height as u16),
            bg_color: WHITE,
        };
        // hayro is young; a page it cannot handle must fail this conversion,
        // not the process. Nothing the closure touches outlives it except the
        // immutable document, so a caught panic leaves the session usable.
        let pixmap = catch_unwind(AssertUnwindSafe(|| {
            let cache = RenderCache::new();
            hayro::render(page, &cache, &self.settings, &target)
        }))
        .map_err(|_| failure("the portable renderer failed on this page"))?;
        if (u32::from(pixmap.width()), u32::from(pixmap.height())) != (size.width, size.height) {
            return Err(failure("rendered dimensions are inconsistent"));
        }
        let mut rgb = Vec::new();
        rgb.try_reserve_exact(size.rgb_bytes)
            .map_err(|_| failure("cannot allocate the bounded RGB result"))?;
        // The white background makes every pixel opaque already; compositing
        // the premultiplied value over white keeps that true at any alpha.
        for pixel in pixmap.data() {
            let white = 255 - pixel.a;
            rgb.extend_from_slice(&[
                pixel.r.saturating_add(white),
                pixel.g.saturating_add(white),
                pixel.b.saturating_add(white),
            ]);
        }
        RgbImage::from_raw(size.width, size.height, rgb)
            .ok_or_else(|| failure("rendered RGB dimensions are inconsistent"))
    }
}
