//! Lossy WebP as Python markitai 1.2 wrote it with Pillow's
//! `save(format="WEBP", quality=q)`: libwebp 1.6.0 (the version Pillow 12 and
//! 13 bundle, vendored in `vendor/libwebp`) with Pillow's settings, so a
//! picture's file is byte for byte the one Pillow writes from the same pixels.
//! An alpha channel is kept, losslessly (Pillow's `alpha_quality` 100); one
//! that is opaque everywhere is left out by libwebp. No metadata is written.

use image::DynamicImage;
use markitai_webp_sys::{ERROR_BAD_DIMENSION, ERROR_BAD_WRITE, markitai_webp_encode};
use std::ffi::{c_int, c_void};
use std::io::{Error, Result, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;

/// The longest side a WebP picture can have.
pub(crate) const MAX_SIDE: u32 = 16383;

/// Writes `image` to `out` as lossy WebP at `quality` (1-100): RGBA when it
/// has an alpha channel, RGB otherwise.
pub(crate) fn encode(image: &DynamicImage, quality: u8, out: &mut impl Write) -> Result<()> {
    let (width, height) = (image.width(), image.height());
    match image {
        DynamicImage::ImageRgb8(rgb) => {
            encode_pixels(rgb.as_raw(), width, height, false, quality, out)
        }
        DynamicImage::ImageRgba8(rgba) => {
            encode_pixels(rgba.as_raw(), width, height, true, quality, out)
        }
        _ if image.color().has_alpha() => {
            encode_pixels(image.to_rgba8().as_raw(), width, height, true, quality, out)
        }
        _ => encode_pixels(image.to_rgb8().as_raw(), width, height, false, quality, out),
    }
}

/// Writes packed 8-bit RGB pixels to `out` as lossy WebP at `quality`.
pub(crate) fn encode_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    out: &mut impl Write,
) -> Result<()> {
    encode_pixels(rgb, width, height, false, quality, out)
}

struct Sink<'a, W> {
    out: &'a mut W,
    error: Option<Error>,
}

unsafe extern "C" fn sink<W: Write>(data: *const u8, size: usize, context: *mut c_void) -> c_int {
    // SAFETY: `context` is the `Sink` `encode_pixels` passes, alive and not
    // otherwise borrowed while libwebp encodes; `data` holds `size` bytes.
    let sink = unsafe { &mut *context.cast::<Sink<W>>() };
    let bytes = if size == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, size) }
    };
    // A panic must not unwind into C.
    match catch_unwind(AssertUnwindSafe(|| sink.out.write_all(bytes))) {
        Ok(Ok(())) => 1,
        Ok(Err(error)) => {
            sink.error = Some(error);
            0
        }
        Err(_) => {
            sink.error = Some(Error::other("WebP output writer panicked"));
            0
        }
    }
}

fn encode_pixels<W: Write>(
    pixels: &[u8],
    width: u32,
    height: u32,
    alpha: bool,
    quality: u8,
    out: &mut W,
) -> Result<()> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(Error::other(format!(
            "{width}x{height} pixels: a WebP picture has 1 to {MAX_SIDE} pixels on each side"
        )));
    }
    let channels = if alpha { 4 } else { 3 };
    let stride = width as usize * channels;
    if pixels.len() != stride * height as usize {
        return Err(Error::other(
            "WebP input pixels do not match the picture size",
        ));
    }
    set_up_once();
    libwebp(pixels, width, height, stride, alpha, quality, out)
}

/// libwebp fills its function and lookup tables on first use, guarded only
/// by a flag: never locked on Windows, and elsewhere only because
/// `vendor/libwebp/build.rs` turns on `WEBP_USE_THREAD`. Two first encodes at
/// once (`convert_ahead` runs several) could read a table the other is still
/// filling. One small RGB and one RGBA encode with graded transparency reach
/// every table the encoder fills with these settings, so they run alone
/// before any other encode.
fn set_up_once() {
    static SET_UP: Once = Once::new();
    SET_UP.call_once(|| {
        const SIDE: u32 = 16;
        let rgba: Vec<u8> = (0..SIDE * SIDE)
            .flat_map(|i| {
                let (x, y) = (i % SIDE, i / SIDE);
                [x * 16, y * 16, (x ^ y) * 16, (x + y) * 8].map(|v| v as u8)
            })
            .collect();
        let rgb: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[r, g, b, _]| [r, g, b])
            .collect();
        let mut sink = std::io::sink();
        for (pixels, alpha) in [(&rgb, false), (&rgba, true)] {
            let stride = SIDE as usize * if alpha { 4 } else { 3 };
            // Only the set-up matters; a failure here shows in the real encode.
            let _ = libwebp(pixels, SIDE, SIDE, stride, alpha, 75, &mut sink);
        }
    });
}

/// One call of libwebp's encoder for pixels `encode_pixels` has checked.
fn libwebp<W: Write>(
    pixels: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    alpha: bool,
    quality: u8,
    out: &mut W,
) -> Result<()> {
    let mut context = Sink { out, error: None };
    // SAFETY: `pixels` holds `height` rows of `stride` bytes (checked above;
    // both sides fit in `c_int`), and `context` outlives the call.
    let status = unsafe {
        markitai_webp_encode(
            pixels.as_ptr(),
            width as c_int,
            height as c_int,
            stride as c_int,
            c_int::from(alpha),
            f32::from(quality.clamp(1, 100)),
            sink::<W>,
            (&raw mut context).cast(),
        )
    };
    match (status, context.error) {
        (0, _) => Ok(()),
        (ERROR_BAD_WRITE, Some(error)) => Err(error),
        (ERROR_BAD_DIMENSION, _) => Err(Error::other(format!(
            "{width}x{height} pixels exceed the WebP size limit"
        ))),
        (code, _) => Err(Error::other(format!(
            "WebP encoding failed (libwebp error {code})"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GenericImageView, ImageFormat, Rgb, RgbImage, Rgba, RgbaImage};

    fn photo(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| {
            Rgb([
                (x * 255 / width) as u8,
                (y * 255 / height) as u8,
                ((x * 7 + y * 13) % 256) as u8,
            ])
        })
    }

    fn webp(image: &DynamicImage, quality: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode(image, quality, &mut bytes).unwrap();
        bytes
    }

    /// The RIFF chunks after the WebP header: `VP8 ` alone for an opaque
    /// picture, `VP8X`, `ALPH` and `VP8 ` with transparency.
    fn chunks(bytes: &[u8]) -> Vec<String> {
        assert_eq!((&bytes[..4], &bytes[8..12]), (&b"RIFF"[..], &b"WEBP"[..]));
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize,
            bytes.len() - 8
        );
        let mut found = Vec::new();
        let mut at = 12;
        while at + 8 <= bytes.len() {
            found.push(String::from_utf8_lossy(&bytes[at..at + 4]).into_owned());
            let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
            at += 8 + size + size % 2;
        }
        found
    }

    #[test]
    fn opaque_pictures_are_lossy_vp8_without_metadata_and_quality_sets_the_size() {
        let image = DynamicImage::ImageRgb8(photo(160, 120));
        let sizes: Vec<_> = [10, 50, 75, 90, 100]
            .into_iter()
            .map(|quality| {
                let bytes = webp(&image, quality);
                // Lossy (`VP8 `), never lossless (`VP8L`), even at 100, as
                // Pillow; no ICCP, EXIF or XMP chunk.
                assert_eq!(chunks(&bytes), ["VP8 "], "quality {quality}");
                assert_eq!(image::guess_format(&bytes).unwrap(), ImageFormat::WebP);
                let decoded = image::load_from_memory(&bytes).unwrap();
                assert_eq!(decoded.dimensions(), (160, 120));
                bytes.len()
            })
            .collect();
        assert!(sizes.windows(2).all(|pair| pair[0] < pair[1]), "{sizes:?}");
        // An alpha channel that is opaque everywhere is left out: the same
        // file as from the RGB pixels.
        let rgba = DynamicImage::ImageRgba8(image.to_rgba8());
        assert_eq!(webp(&rgba, 75), webp(&image, 75));
        let mut rgb = Vec::new();
        encode_rgb(image.as_bytes(), 160, 120, 75, &mut rgb).unwrap();
        assert_eq!(rgb, webp(&image, 75));
    }

    #[test]
    fn transparency_is_kept_exactly_beside_lossy_colour() {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_fn(96, 64, |x, y| {
            let alpha = match x {
                0..32 => 0,
                32..64 => (y * 4) as u8,
                _ => 255,
            };
            Rgba([200, (x * 2) as u8, (y * 3) as u8, alpha])
        }));
        for quality in [20, 75, 100] {
            let bytes = webp(&image, quality);
            assert_eq!(chunks(&bytes), ["VP8X", "ALPH", "VP8 "]);
            let decoded = image::load_from_memory(&bytes).unwrap().to_rgba8();
            for (decoded, original) in decoded.pixels().zip(image.to_rgba8().pixels()) {
                assert_eq!(decoded.0[3], original.0[3]);
            }
        }
        // Grey with alpha is encoded as RGBA too.
        let grey = DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_fn(40, 40, |x, _| {
            image::LumaA([128, if x < 20 { 0 } else { 255 }])
        }));
        assert_eq!(chunks(&webp(&grey, 75)), ["VP8X", "ALPH", "VP8 "]);
    }

    /// Pillow 13.0.0 (libwebp 1.6.0) `Image.save(format="WEBP", quality=75)`
    /// of this gradient and of its RGBA form with a clear left half; the
    /// files are compared by SHA-256.
    #[test]
    fn files_are_byte_for_byte_pillows() {
        use sha2::{Digest, Sha256};
        let image = DynamicImage::ImageRgb8(photo(64, 48));
        let digest = |bytes: Vec<u8>| {
            Sha256::digest(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        assert_eq!(
            digest(webp(&image, 75)),
            "1a28b3b9553eb811297f629260d61cc12306d0f3f20bc638984b5d00206147bf"
        );
        let mut rgba = image.to_rgba8();
        for (x, _, pixel) in rgba.enumerate_pixels_mut() {
            if x < 32 {
                pixel.0[3] = 0;
            }
        }
        assert_eq!(
            digest(webp(&DynamicImage::ImageRgba8(rgba), 75)),
            "7ecef35b7164e98a1c6663c9cab6e5326992063c20bdb23dd8f1d8a7d745f26b"
        );
    }

    #[test]
    fn oversized_pictures_and_writer_errors_are_errors() {
        let wide = DynamicImage::ImageRgb8(RgbImage::new(MAX_SIDE + 1, 1));
        let error = encode(&wide, 75, &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("16383"), "{error}");
        struct Full;
        impl Write for Full {
            fn write(&mut self, _: &[u8]) -> Result<usize> {
                Err(Error::other("budget exceeded"))
            }
            fn flush(&mut self) -> Result<()> {
                Ok(())
            }
        }
        let image = DynamicImage::ImageRgb8(photo(32, 32));
        let error = encode(&image, 75, &mut Full).unwrap_err();
        assert_eq!(error.to_string(), "budget exceeded");
    }
}
