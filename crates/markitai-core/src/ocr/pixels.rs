// Off macOS the portable engine decodes images here; the PNG hand-over and
// enlarged copies serve Vision.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use super::{Result, failure};
use crate::images::ImageBytes;
use image::{DynamicImage, ImageDecoder, ImageEncoder, ImageFormat, ImageReader, Rgb, RgbImage};
use std::io::Write;

const MAX_INPUT: usize = 64 * 1024 * 1024;
pub(super) use crate::images::MAX_PIXELS;
const MAX_DECODED: u64 = 256 * 1024 * 1024;
const MAX_ENCODED: usize = 128 * 1024 * 1024;

pub(super) struct Prepared {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

struct Bounded(Vec<u8>);

impl Write for Bounded {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_ENCODED.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other(
                "OCR image exceeds the encoded size limit",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(any(target_os = "macos", test))]
pub(super) fn prepare(bytes: &[u8]) -> Result<Prepared> {
    prepare_rgb(upright(bytes)?)
}

/// The pixels of an encoded image, within the input, pixel and decoder
/// allocation limits, with its EXIF orientation applied and alpha
/// composited over white.
pub(super) fn upright(bytes: &[u8]) -> Result<RgbImage> {
    if bytes.is_empty() || bytes.len() > MAX_INPUT {
        return Err(failure("image input is empty or exceeds 64 MiB"));
    }
    let mut reader = ImageReader::new(ImageBytes::new(bytes))
        .with_guessed_format()
        .map_err(|_| failure("cannot identify image encoding"))?;
    if reader.format() == Some(ImageFormat::Tiff) {
        let tiff = tiff::decoder::Decoder::new(ImageBytes::new(bytes))
            .map_err(|_| failure("cannot decode TIFF image"))?;
        if tiff.more_images() {
            return Err(crate::Error::Unsupported(
                "Local OCR of multi-page TIFF images is not implemented".into(),
            ));
        }
    }
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_PIXELS as u32);
    limits.max_image_height = Some(MAX_PIXELS as u32);
    limits.max_alloc = Some(MAX_DECODED);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .map_err(|_| failure("cannot decode image"))?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(failure("decoded image exceeds 32 million pixels"));
    }
    let orientation = decoder
        .orientation()
        .map_err(|_| failure("cannot read image orientation"))?;
    let mut image =
        DynamicImage::from_decoder(decoder).map_err(|_| failure("cannot decode image pixels"))?;
    image.apply_orientation(orientation);
    let rgb = if image.color().has_alpha() {
        let rgba = image.into_rgba8();
        RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
            let pixel = rgba.get_pixel(x, y).0;
            let alpha = u32::from(pixel[3]);
            Rgb([0, 1, 2].map(|channel| {
                ((u32::from(pixel[channel]) * alpha + 255 * (255 - alpha) + 127) / 255) as u8
            }))
        })
    } else {
        image.into_rgb8()
    };
    Ok(rgb)
}

/// Check pixels a renderer passes as `prepare_rgb` does.
#[cfg_attr(target_os = "macos", allow(dead_code))] // Only the portable engine checks alone.
pub(super) fn check_rgb(rgb: &RgbImage) -> Result<()> {
    let (width, height) = rgb.dimensions();
    validate_rgb_layout(width, height, rgb.as_raw().len())
}

fn validate_rgb_layout(width: u32, height: u32, bytes: usize) -> Result<()> {
    let pixels = u64::from(width) * u64::from(height);
    if width == 0 || height == 0 || pixels > MAX_PIXELS {
        return Err(failure("decoded image exceeds 32 million pixels"));
    }
    let expected = pixels
        .checked_mul(3)
        .and_then(|length| usize::try_from(length).ok())
        .ok_or_else(|| failure("RGB image byte length overflow"))?;
    if bytes != expected {
        return Err(failure(
            "RGB image byte length does not match its dimensions",
        ));
    }
    Ok(())
}

/// Accept upright RGB pixels already composited on white by the renderer.
pub(super) fn prepare_rgb(rgb: RgbImage) -> Result<Prepared> {
    let (width, height) = rgb.dimensions();
    // ImageBuffer permits backing storage longer than the declared dimensions.
    // Check it explicitly instead of relying on an encoder assertion or slicing.
    validate_rgb_layout(width, height, rgb.as_raw().len())?;
    let mut png = Bounded(Vec::new());
    image::codecs::png::PngEncoder::new_with_quality(
        &mut png,
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::Adaptive,
    )
    .write_image(rgb.as_raw(), width, height, image::ExtendedColorType::Rgb8)
    .map_err(|_| failure("cannot prepare bounded OCR image"))?;
    Ok(Prepared {
        png: png.0,
        width,
        height,
    })
}

/// A copy of prepared pixels enlarged `factor` times with Lanczos filtering,
/// within the same pixel limit, for a second reading of small text.
pub(super) fn enlarge(image: &Prepared, factor: f32) -> Result<Prepared> {
    let scale = |side: u32| (f64::from(side) * f64::from(factor)).round();
    let (width, height) = (scale(image.width), scale(image.height));
    if !factor.is_finite() || factor < 1.0 {
        return Err(failure("invalid OCR enlargement factor"));
    }
    if width * height > MAX_PIXELS as f64 {
        return Err(failure("enlarged OCR image exceeds 32 million pixels"));
    }
    let rgb = decode(image)?;
    let larger = image::imageops::resize(
        &rgb,
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    drop(rgb);
    prepare_rgb(larger)
}

/// The pixels of a prepared image.
pub(super) fn decode(image: &Prepared) -> Result<RgbImage> {
    let mut reader = ImageReader::with_format(ImageBytes::new(&image.png), ImageFormat::Png);
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODED);
    reader.limits(limits);
    Ok(reader
        .decode()
        .map_err(|_| failure("cannot decode prepared OCR image"))?
        .into_rgb8())
}

/// A copy of the pixel `region` `[left, top, right, bottom]` of `rgb`,
/// enlarged `factor` times with Lanczos filtering, inside a white `margin`
/// of that many enlarged pixels, for a second reading of a table cell or a
/// number. The copy's region starts at the region's floored top-left pixel.
pub(super) fn crop(rgb: &RgbImage, region: [f32; 4], factor: f32, margin: u32) -> Result<Prepared> {
    let (width, height) = rgb.dimensions();
    let [left, top, right, bottom] = region;
    let x0 = (left.max(0.0).floor() as u32).min(width);
    let y0 = (top.max(0.0).floor() as u32).min(height);
    let x1 = (right.max(0.0).ceil() as u32).min(width);
    let y1 = (bottom.max(0.0).ceil() as u32).min(height);
    if x1 <= x0 || y1 <= y0 || !factor.is_finite() || factor < 1.0 {
        return Err(failure("invalid OCR region"));
    }
    let scale = |side: u32| (f64::from(side) * f64::from(factor)).round() as u32;
    let (w, h) = (scale(x1 - x0), scale(y1 - y0));
    let (outer_w, outer_h) = (
        u64::from(w) + 2 * u64::from(margin),
        u64::from(h) + 2 * u64::from(margin),
    );
    if outer_w * outer_h > MAX_PIXELS || w == 0 || h == 0 {
        return Err(failure("enlarged OCR region exceeds 32 million pixels"));
    }
    // Rows of `width` pixels from `source` (rows `stride` bytes apart, from
    // byte `start`), placed at `offset` bytes in `target` rows `pitch` apart.
    let copy = |source: &[u8],
                start: usize,
                stride: usize,
                rows: u32,
                width: u32,
                target: &mut [u8],
                offset: usize,
                pitch: usize| {
        let length = width as usize * 3;
        for row in 0..rows as usize {
            let from = start + row * stride;
            let to = offset + row * pitch;
            target[to..to + length].copy_from_slice(&source[from..from + length]);
        }
    };
    let (part_w, part_h) = (x1 - x0, y1 - y0);
    let mut part = vec![0; part_w as usize * part_h as usize * 3];
    let stride = width as usize * 3;
    let start = y0 as usize * stride + x0 as usize * 3;
    copy(
        rgb.as_raw(),
        start,
        stride,
        part_h,
        part_w,
        &mut part,
        0,
        part_w as usize * 3,
    );
    let part =
        RgbImage::from_raw(part_w, part_h, part).ok_or_else(|| failure("invalid OCR region"))?;
    let larger = image::imageops::resize(&part, w, h, image::imageops::FilterType::Lanczos3);
    if margin == 0 {
        return prepare_rgb(larger);
    }
    let pitch = outer_w as usize * 3;
    let mut canvas = vec![255; pitch * outer_h as usize];
    let offset = margin as usize * pitch + margin as usize * 3;
    copy(
        larger.as_raw(),
        0,
        w as usize * 3,
        h,
        w,
        &mut canvas,
        offset,
        pitch,
    );
    let canvas = RgbImage::from_raw(outer_w as u32, outer_h as u32, canvas)
        .ok_or_else(|| failure("invalid OCR region"))?;
    prepare_rgb(canvas)
}

#[cfg(test)]
pub(super) fn encode_test_image(image: DynamicImage) -> Vec<u8> {
    let mut buffer = std::io::Cursor::new(Vec::new());
    image.write_to(&mut buffer, ImageFormat::Png).unwrap();
    buffer.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_preserves_resolution_and_composites_alpha_on_white() {
        let rgba = image::RgbaImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                image::Rgba([0, 0, 0, 0])
            } else {
                image::Rgba([200, 10, 20, 128])
            }
        });
        let result = prepare(&encode_test_image(DynamicImage::ImageRgba8(rgba))).unwrap();
        assert_eq!((result.width, result.height), (2, 1));
        let decoded = image::load_from_memory(&result.png).unwrap().into_rgb8();
        assert_eq!(decoded.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(decoded.get_pixel(1, 0).0, [227, 132, 137]);
        let text = prepare(include_bytes!("fixtures/english.png")).unwrap();
        assert_eq!((text.width, text.height), (819, 301));
    }

    #[test]
    fn malformed_and_oversized_images_fail_before_pixel_allocation() {
        assert!(prepare(&[]).is_err());
        assert!(prepare(b"no image").is_err());
        let mut header = vec![0u8; 54];
        header[..2].copy_from_slice(b"BM");
        header[10..14].copy_from_slice(&54u32.to_le_bytes());
        header[14..18].copy_from_slice(&40u32.to_le_bytes());
        header[18..22].copy_from_slice(&100_000i32.to_le_bytes());
        header[22..26].copy_from_slice(&100_000i32.to_le_bytes());
        header[26..28].copy_from_slice(&1u16.to_le_bytes());
        header[28..30].copy_from_slice(&24u16.to_le_bytes());
        assert!(prepare(&header).is_err());
    }

    #[test]
    fn rgb_layout_rechecks_dimensions_and_exact_backing_length() {
        assert!(validate_rgb_layout(8_000, 4_000, 96_000_000).is_ok());
        for (width, height, length) in [
            (0, 1, 0),
            (1, 0, 0),
            (8_000, 4_001, 96_024_000),
            (u32::MAX, u32::MAX, 0),
            (1, 1, 2),
            (1, 1, 4),
        ] {
            assert!(validate_rgb_layout(width, height, length).is_err());
        }
        assert!(prepare_rgb(RgbImage::new(0, 1)).is_err());
        let extra_storage = RgbImage::from_raw(1, 1, vec![255; 4]).unwrap();
        assert!(prepare_rgb(extra_storage).is_err());
    }

    #[test]
    fn rendered_rgb_and_encoded_fixture_deliver_identical_normalized_pixels() {
        let fixture = include_bytes!("fixtures/english.png");
        let rgb = image::load_from_memory(fixture).unwrap().into_rgb8();
        let expected = rgb.clone();
        let encoded = prepare(fixture).unwrap();
        let rendered = prepare_rgb(rgb).unwrap();
        assert_eq!((rendered.width, rendered.height), (819, 301));
        assert_eq!(rendered.png, encoded.png);
        assert_eq!(
            image::load_from_memory(&rendered.png).unwrap().into_rgb8(),
            expected
        );
    }

    #[test]
    fn enlargement_scales_both_sides_and_keeps_the_pixel_limit() {
        // A black left half and a white right half.
        let rgb = RgbImage::from_fn(40, 10, |x, _| Rgb([if x < 20 { 0 } else { 255 }; 3]));
        let prepared = prepare_rgb(rgb).unwrap();
        let larger = enlarge(&prepared, 2.5).unwrap();
        assert_eq!((larger.width, larger.height), (100, 25));
        let decoded = image::load_from_memory(&larger.png).unwrap().into_rgb8();
        assert_eq!(decoded.dimensions(), (100, 25));
        assert_eq!(decoded.get_pixel(10, 12).0, [0; 3]);
        assert_eq!(decoded.get_pixel(90, 12).0, [255; 3]);
        let edge = decoded.get_pixel(50, 12).0[0];
        assert!(edge > 0 && edge < 255, "{edge}");
        // Sides are rounded, not truncated.
        let odd = prepare_rgb(RgbImage::from_pixel(3, 3, Rgb([255; 3]))).unwrap();
        let rounded = enlarge(&odd, 1.5).unwrap();
        assert_eq!((rounded.width, rounded.height), (5, 5));
        for factor in [0.5, f32::NAN, f32::INFINITY] {
            assert!(enlarge(&prepared, factor).is_err(), "{factor}");
        }
        let wide = Prepared {
            png: prepared.png.clone(),
            width: 8_000,
            height: 4_000,
        };
        // Rejected before decoding or allocating the enlarged copy.
        let error = enlarge(&wide, 1.01).err().unwrap().to_string();
        assert!(error.contains("enlarged OCR image exceeds"), "{error}");
    }

    #[test]
    fn a_region_is_copied_enlarged_inside_a_white_margin() {
        // A black square at (10, 10)..(20, 20) of a white 40 by 30 image.
        let rgb = RgbImage::from_fn(40, 30, |x, y| {
            Rgb([if (10..20).contains(&x) && (10..20).contains(&y) {
                0
            } else {
                255
            }; 3])
        });
        let copy = crop(&rgb, [9.5, 9.0, 21.0, 21.0], 2.0, 0).unwrap();
        // The region starts at its floored corner: (9, 9)..(21, 21).
        assert_eq!((copy.width, copy.height), (24, 24));
        let decoded = decode(&copy).unwrap();
        assert_eq!(decoded.get_pixel(12, 12).0, [0; 3]);
        assert!(decoded.get_pixel(0, 0).0[0] > 240);
        let framed = decode(&crop(&rgb, [9.0, 9.0, 21.0, 21.0], 2.0, 5).unwrap()).unwrap();
        assert_eq!(framed.dimensions(), (34, 34));
        assert_eq!(framed.get_pixel(4, 17).0, [255; 3]);
        assert_eq!(framed.get_pixel(17, 17).0, [0; 3]);
        for region in [[5.0, 5.0, 5.0, 9.0], [50.0, 0.0, 60.0, 9.0]] {
            assert!(crop(&rgb, region, 2.0, 0).is_err(), "{region:?}");
        }
        assert!(crop(&rgb, [0.0, 0.0, 9.0, 9.0], f32::NAN, 0).is_err());
    }

    #[test]
    fn rendered_rgb_preserves_upright_rows_and_colors_without_rescaling() {
        let rgb = RgbImage::from_fn(2, 3, |x, y| Rgb([x as u8 * 100, y as u8 * 80, 255]));
        let expected = rgb.clone();
        let rendered = prepare_rgb(rgb).unwrap();
        assert_eq!((rendered.width, rendered.height), (2, 3));
        assert_eq!(
            image::load_from_memory(&rendered.png).unwrap().into_rgb8(),
            expected
        );
    }
}
