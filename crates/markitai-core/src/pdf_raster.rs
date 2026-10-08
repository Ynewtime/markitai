//! In-process PDF page rasterization, independent of OCR and disk publication.
//!
//! macOS draws pages with CoreGraphics. Every other platform uses the
//! pure-Rust `hayro` renderer (`portable`). A macOS build with the
//! `portable-media` feature compiles both, keeps CoreGraphics as the default
//! and lets `MARKITAI_PDF_RENDERER=portable` select hayro, so the two can be
//! compared on one machine. Both backends share the input, page and pixel
//! limits below and return upright RGB8 pixels composited onto white.

use crate::{Error, Result};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
mod portable;

/// Selects a compiled backend for a session; unset means the platform default.
const RENDERER_VARIABLE: &str = "MARKITAI_PDF_RENDERER";
const MAX_INPUT: usize = 500 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backend {
    #[cfg(target_os = "macos")]
    CoreGraphics,
    #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
    Portable,
}

impl Backend {
    #[cfg(target_os = "macos")]
    const DEFAULT: Self = Self::CoreGraphics;
    #[cfg(not(target_os = "macos"))]
    const DEFAULT: Self = Self::Portable;

    /// The backends this build contains, the default first.
    #[cfg(test)]
    pub(crate) fn compiled() -> Vec<Self> {
        vec![
            #[cfg(target_os = "macos")]
            Self::CoreGraphics,
            #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
            Self::Portable,
        ]
    }

    #[cfg(test)]
    pub(crate) fn name(self) -> &'static str {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreGraphics => "coregraphics",
            #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
            Self::Portable => "portable",
        }
    }

    /// Reads the selection variable. A backend this build lacks, or an
    /// unknown name, is an error rather than a silent use of the default.
    fn requested() -> Result<Self> {
        match std::env::var_os(RENDERER_VARIABLE) {
            None => Ok(Self::DEFAULT),
            // A value that is not Unicode names no backend.
            Some(value) => Self::parse(Some(value.to_str().unwrap_or("\u{fffd}"))),
        }
    }

    fn parse(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim) {
            None | Some("") => Ok(Self::DEFAULT),
            Some("coregraphics") => Self::coregraphics(),
            Some("portable") => Self::portable(),
            _ => Err(Error::InvalidInput(format!(
                "{RENDERER_VARIABLE} must be coregraphics or portable"
            ))),
        }
    }

    #[cfg(target_os = "macos")]
    fn coregraphics() -> Result<Self> {
        Ok(Self::CoreGraphics)
    }

    #[cfg(not(target_os = "macos"))]
    fn coregraphics() -> Result<Self> {
        Err(Error::Unsupported(format!(
            "{RENDERER_VARIABLE}=coregraphics: CoreGraphics page rendering exists only on macOS"
        )))
    }

    #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
    fn portable() -> Result<Self> {
        Ok(Self::Portable)
    }

    #[cfg(not(any(not(target_os = "macos"), feature = "portable-media")))]
    fn portable() -> Result<Self> {
        Err(Error::Unsupported(format!(
            "{RENDERER_VARIABLE}=portable: this macOS build was made without the portable-media feature"
        )))
    }
}

/// One open document. Like both backends, it stays on its caller's thread.
pub(crate) enum PdfRasterSession {
    #[cfg(target_os = "macos")]
    CoreGraphics(macos::PdfRasterSession),
    #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
    Portable(portable::PdfRasterSession),
}

impl PdfRasterSession {
    pub(crate) fn open(bytes: &[u8]) -> Result<Self> {
        Self::open_with(bytes, Backend::requested()?)
    }

    pub(crate) fn open_with(bytes: &[u8], backend: Backend) -> Result<Self> {
        check_input(bytes)?;
        match backend {
            #[cfg(target_os = "macos")]
            Backend::CoreGraphics => macos::PdfRasterSession::open(bytes).map(Self::CoreGraphics),
            #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
            Backend::Portable => portable::PdfRasterSession::open(bytes).map(Self::Portable),
        }
    }

    pub(crate) fn pages(&self) -> usize {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreGraphics(session) => session.pages(),
            #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
            Self::Portable(session) => session.pages(),
        }
    }

    /// Checks the complete page geometry without allocating its pixel buffers.
    pub(crate) fn dimensions(&self, page_1_based: usize, dpi: f64) -> Result<(u32, u32)> {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreGraphics(session) => session.dimensions(page_1_based, dpi),
            #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
            Self::Portable(session) => session.dimensions(page_1_based, dpi),
        }
    }

    pub(crate) fn render(&self, page_1_based: usize, dpi: f64) -> Result<image::RgbImage> {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreGraphics(session) => session.render(page_1_based, dpi),
            #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
            Self::Portable(session) => session.render(page_1_based, dpi),
        }
    }

    /// Fonts and images hayro skipped so far; CoreGraphics reports none.
    #[cfg(all(test, any(not(target_os = "macos"), feature = "portable-media")))]
    pub(crate) fn portable_warnings(&self) -> Option<(usize, usize)> {
        match self {
            #[cfg(target_os = "macos")]
            Self::CoreGraphics(_) => None,
            #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
            Self::Portable(session) => Some(session.warnings()),
        }
    }
}

/// Input checks both backends share, made before either parses a byte.
fn check_input(bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() || bytes.len() > MAX_INPUT {
        return Err(failure("input is empty or exceeds 500 MiB"));
    }
    if !bytes[..bytes.len().min(1024)]
        .windows(5)
        .any(|window| window == b"%PDF-")
    {
        return Err(failure("input has no PDF header"));
    }
    Ok(())
}

fn failure(message: &str) -> Error {
    Error::Conversion(format!("Native PDF rendering: {message}"))
}

const MAX_PIXELS: usize = crate::images::MAX_PIXELS as usize;

struct RasterSize {
    width: u32,
    height: u32,
    /// The RGBX drawing buffer only CoreGraphics allocates.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    row_bytes: usize,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    rgba_bytes: usize,
    rgb_bytes: usize,
}

fn raster_size(width: f64, height: f64, dpi: f64) -> Result<RasterSize> {
    if [width, height, dpi]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.)
    {
        return Err(failure(
            "page dimensions and DPI must be positive and finite",
        ));
    }
    // Multiply before dividing so exact point/DPI products such as 792*150
    // remain the integer 1650 rather than rounding just above it before ceil.
    let width = (width * dpi / 72.).ceil();
    let height = (height * dpi / 72.).ceil();
    if !width.is_finite()
        || !height.is_finite()
        || width < 1.
        || height < 1.
        || width > MAX_PIXELS as f64
        || height > MAX_PIXELS as f64
    {
        return Err(failure("page dimensions exceed the 32 million pixel limit"));
    }
    let width = width as usize;
    let height = height as usize;
    let pixels = width
        .checked_mul(height)
        .filter(|pixels| *pixels <= MAX_PIXELS)
        .ok_or_else(|| failure("page exceeds the 32 million pixel limit"))?;
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| failure("row size overflow"))?;
    let rgba_bytes = row_bytes
        .checked_mul(height)
        .ok_or_else(|| failure("bitmap size overflow"))?;
    let rgb_bytes = pixels
        .checked_mul(3)
        .ok_or_else(|| failure("RGB size overflow"))?;
    Ok(RasterSize {
        width: width as u32,
        height: height as u32,
        row_bytes,
        rgba_bytes,
        rgb_bytes,
    })
}

#[cfg(test)]
mod limits_tests {
    use super::*;

    #[test]
    fn dimensions_reject_invalid_or_oversized_frames_before_allocation() {
        for (width, height, dpi) in [
            (0., 100., 150.),
            (-1., 100., 150.),
            (100., 0., 150.),
            (100., 100., 0.),
            (100., 100., f64::NAN),
            (f64::INFINITY, 100., 150.),
            (f64::MAX, 100., 150.),
            (8001., 4000., 72.),
        ] {
            assert!(raster_size(width, height, dpi).is_err());
        }
        let maximum = raster_size(8000., 4000., 72.).unwrap();
        assert_eq!((maximum.width, maximum.height), (8000, 4000));
        assert_eq!(maximum.row_bytes, 32_000);
        assert_eq!(maximum.rgba_bytes, 128_000_000);
        assert_eq!(maximum.rgb_bytes, 96_000_000);
        let a4 = raster_size(600., 800., 150.).unwrap();
        assert_eq!((a4.width, a4.height), (1250, 1667));
        let letter = raster_size(612., 792., 150.).unwrap();
        assert_eq!((letter.width, letter.height), (1275, 1650));
    }

    #[test]
    fn every_build_lists_its_default_backend_first() {
        let compiled = Backend::compiled();
        assert_eq!(compiled.first(), Some(&Backend::DEFAULT));
        assert!(compiled.iter().all(|backend| !backend.name().is_empty()));
    }

    #[test]
    fn the_selection_variable_names_a_compiled_backend_or_fails() {
        for unset in [None, Some(""), Some("  ")] {
            assert_eq!(Backend::parse(unset).unwrap(), Backend::DEFAULT);
        }
        for unknown in ["hayro", "CoreGraphics", "quartz", "pdfium"] {
            assert!(matches!(
                Backend::parse(Some(unknown)),
                Err(Error::InvalidInput(_))
            ));
        }
        let coregraphics = Backend::parse(Some("coregraphics"));
        let portable = Backend::parse(Some("portable"));
        #[cfg(target_os = "macos")]
        assert_eq!(coregraphics.unwrap(), Backend::CoreGraphics);
        #[cfg(not(target_os = "macos"))]
        assert!(matches!(coregraphics, Err(Error::Unsupported(_))));
        #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
        assert_eq!(portable.unwrap(), Backend::Portable);
        #[cfg(not(any(not(target_os = "macos"), feature = "portable-media")))]
        assert!(matches!(portable, Err(Error::Unsupported(_))));
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, target_os = "macos", feature = "portable-media"))]
mod comparison;
