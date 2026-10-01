use super::{Decoded, Info, unavailable};
use crate::Result;
use crate::system_frameworks::{self, Framework};
use image::{DynamicImage, RgbaImage, metadata::Orientation};
use objc2_core_foundation::{
    CFBoolean, CFData, CFDictionary, CFNumber, CFString, CFType, CGPoint, CGRect, CGSize,
};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGContext, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo, kCGColorSpaceSRGB,
};
use objc2_image_io::{
    CGImageSource, CGImageSourceStatus, kCGImagePropertyOrientation, kCGImagePropertyPixelHeight,
    kCGImagePropertyPixelWidth, kCGImageSourceShouldAllowFloat, kCGImageSourceShouldCache,
};

fn failure(message: &str) -> crate::Error {
    super::super::error(message)
}

fn number(properties: &CFDictionary, key: &CFString) -> Option<i64> {
    // SAFETY: ImageIO property dictionaries use CoreFoundation string keys
    // and CFType values. The individual value's dynamic type is checked below.
    let properties = unsafe { properties.cast_unchecked::<CFString, CFType>() };
    properties.get(key)?.downcast_ref::<CFNumber>()?.as_i64()
}

fn dimensions(width: i64, height: i64) -> Result<(u32, u32, usize, usize)> {
    let width = u32::try_from(width).map_err(|_| failure("invalid HEIF/AVIF image width"))?;
    let height = u32::try_from(height).map_err(|_| failure("invalid HEIF/AVIF image height"))?;
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > super::super::MAX_PIXELS {
        return Err(failure(
            "HEIF/AVIF image exceeds 32 million pixels or is empty",
        ));
    }
    let row = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| failure("HEIF/AVIF row size overflow"))?;
    let bytes = row
        .checked_mul(height as usize)
        .ok_or_else(|| failure("HEIF/AVIF pixel size overflow"))?;
    Ok((width, height, row, bytes))
}

pub(super) fn decode(bytes: &[u8]) -> Result<Decoded> {
    // Some system codecs synthesize a CGImage even when all pixel payloads
    // were truncated. Validate the declared container before trusting it.
    super::validate_container(bytes)?;
    system_frameworks::open(Framework::ImageIO).map_err(|message| failure(&message))?;
    // SAFETY: CFData copies the checked slice; it remains alive until after
    // ImageIO and the CGImage have been released. Options have CFBoolean values.
    let data = unsafe { CFData::new(None, bytes.as_ptr(), bytes.len() as isize) }
        .ok_or_else(|| failure("cannot retain HEIF/AVIF input"))?;
    let options = CFDictionary::<CFString, CFBoolean>::from_slices(
        &[unsafe { kCGImageSourceShouldCache }, unsafe {
            kCGImageSourceShouldAllowFloat
        }],
        &[CFBoolean::new(false), CFBoolean::new(false)],
    );
    let source = unsafe { CGImageSource::with_data(&data, Some(options.as_opaque())) }
        .ok_or_else(unavailable)?;
    let kind = unsafe { source.r#type() }
        .ok_or_else(unavailable)?
        .to_string();
    let format = match kind.as_str() {
        "public.avif" => "AVIF",
        "public.heic" | "public.heif" | "public.heics" | "public.heifs" => "HEIF",
        _ => return Err(unavailable()),
    };
    let images = unsafe { source.count() };
    let primary = unsafe { source.primary_image_index() };
    if images == 0 || primary >= images {
        return Err(failure("HEIF/AVIF container has no valid primary image"));
    }
    let properties = unsafe { source.properties_at_index(primary, Some(options.as_opaque())) }
        .ok_or_else(|| failure("HEIF/AVIF primary image metadata is unavailable"))?;
    let width = number(&properties, unsafe { kCGImagePropertyPixelWidth })
        .ok_or_else(|| failure("HEIF/AVIF image width is missing"))?;
    let height = number(&properties, unsafe { kCGImagePropertyPixelHeight })
        .ok_or_else(|| failure("HEIF/AVIF image height is missing"))?;
    let (width, height, row, length) = dimensions(width, height)?;
    let orientation = match number(&properties, unsafe { kCGImagePropertyOrientation }) {
        None => Orientation::NoTransforms,
        Some(value) => u8::try_from(value)
            .ok()
            .and_then(Orientation::from_exif)
            .ok_or_else(|| failure("HEIF/AVIF orientation is invalid"))?,
    };
    let decoded = unsafe { source.image_at_index(primary, Some(options.as_opaque())) }
        .ok_or_else(unavailable)?;
    if unsafe { source.status() } != CGImageSourceStatus::StatusComplete
        || unsafe { source.status_at_index(primary) } != CGImageSourceStatus::StatusComplete
    {
        return Err(failure("HEIF/AVIF image data is incomplete or invalid"));
    }
    if CGImage::width(Some(&decoded)) != width as usize
        || CGImage::height(Some(&decoded)) != height as usize
    {
        return Err(failure(
            "HEIF/AVIF decoded dimensions disagree with metadata",
        ));
    }
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(length)
        .map_err(|_| failure("cannot allocate HEIF/AVIF bitmap"))?;
    pixels.resize(length, 0);
    let space = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }))
        .ok_or_else(|| failure("cannot create sRGB color space"))?;
    // SAFETY: This allocation has checked row * height bytes, cannot move
    // during drawing, and outlives the context. RGBA order is explicit.
    let context = unsafe {
        CGBitmapContextCreate(
            pixels.as_mut_ptr().cast(),
            width as usize,
            height as usize,
            8,
            row,
            Some(&space),
            CGImageByteOrderInfo::Order32Big.0 | CGImageAlphaInfo::PremultipliedLast.0,
        )
    }
    .ok_or_else(|| failure("cannot create HEIF/AVIF bitmap context"))?;
    CGContext::draw_image(
        Some(&context),
        CGRect::new(
            CGPoint::new(0., 0.),
            CGSize::new(f64::from(width), f64::from(height)),
        ),
        Some(&decoded),
    );
    CGContext::flush(Some(&context));
    drop(context);
    // image::RgbaImage stores straight alpha; Quartz's supported RGBA bitmap
    // format is premultiplied. Fully transparent RGB has no visible meaning.
    for pixel in pixels.as_chunks_mut::<4>().0 {
        let alpha = u32::from(pixel[3]);
        for channel in &mut pixel[..3] {
            *channel = (u32::from(*channel) * 255 + alpha / 2)
                .checked_div(alpha)
                .unwrap_or(0)
                .min(255) as u8;
        }
    }
    let image = RgbaImage::from_raw(width, height, pixels)
        .ok_or_else(|| failure("invalid HEIF/AVIF pixel buffer"))?;
    let mut image = DynamicImage::ImageRgba8(image);
    image.apply_orientation(orientation);
    Ok(Decoded {
        image,
        info: Info {
            format,
            images,
            primary,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_heic_pixels_and_orientation_are_decoded_once() {
        let regular = decode(include_bytes!(
            "../fixtures/heif/quadrants-orientation1.heic"
        ))
        .unwrap();
        let rotated = decode(include_bytes!(
            "../fixtures/heif/quadrants-orientation6.heic"
        ))
        .unwrap();
        assert_eq!((regular.image.width(), regular.image.height()), (120, 80));
        assert_eq!((rotated.image.width(), rotated.image.height()), (80, 120));
        let expected = regular.image.rotate90();
        assert_eq!(expected, rotated.image);
        let rgba = regular.image.to_rgba8();
        let red = rgba.get_pixel(30, 20).0;
        assert!(
            red[0] > 240 && red[1] < 10 && red[2] < 10 && red[3] == 255,
            "{red:?}"
        );
        assert_eq!((regular.info.images, regular.info.primary), (1, 0));
    }

    #[test]
    fn actual_avif_fixtures_preserve_white_pixels_and_transparency() {
        let white = decode(include_bytes!("../fixtures/heif/white_1x1.avif")).unwrap();
        assert_eq!(white.info.format, "AVIF");
        assert_eq!((white.image.width(), white.image.height()), (1, 1));
        assert!(
            white.image.to_rgba8().get_pixel(0, 0).0[..3]
                .iter()
                .all(|v| *v >= 250)
        );
        let circle = decode(include_bytes!(
            "../fixtures/heif/circle_custom_properties.avif"
        ))
        .unwrap()
        .image
        .to_rgba8();
        let reference = image::load_from_memory(include_bytes!(
            "../fixtures/heif/circle-trns-after-plte.png"
        ))
        .unwrap()
        .to_rgba8();
        assert_eq!(circle.dimensions(), reference.dimensions());
        let (width, height) = circle.dimensions();
        assert_eq!(circle.get_pixel(0, 0)[3], 0);
        let middle = circle.get_pixel(width / 2, height / 2).0;
        assert!(
            middle[2] > 220 && middle[0] < 30 && middle[1] < 30 && middle[3] > 240,
            "{middle:?}"
        );
    }

    #[test]
    fn the_primary_image_is_selected_instead_of_the_first_item() {
        let primary = decode(include_bytes!("../fixtures/heif/primary-second.heic")).unwrap();
        assert_eq!((primary.info.images, primary.info.primary), (2, 1));
        let pixel = primary.image.to_rgba8().get_pixel(30, 20).0;
        assert!(
            pixel[0] < 10 && pixel[1] > 240 && pixel[2] < 10 && pixel[3] == 255,
            "{pixel:?}"
        );
    }

    #[test]
    fn malformed_container_and_dimensions_fail_instead_of_blank_success() {
        assert!(decode(b"not an image").is_err());
        let bytes = include_bytes!("../fixtures/heif/quadrants-orientation1.heic");
        assert!(decode(&bytes[..bytes.len() / 2]).is_err());
        let mut huge = bytes.to_vec();
        for offset in (0..huge.len().saturating_sub(16)).filter(|i| &bytes[*i..*i + 4] == b"ispe") {
            huge[offset + 8..offset + 12].copy_from_slice(&100_000u32.to_be_bytes());
            huge[offset + 12..offset + 16].copy_from_slice(&100_000u32.to_be_bytes());
        }
        assert!(decode(&huge).is_err());
        for (width, height) in [(0, 1), (-1, 20), (100_000, 100_000), (i64::MAX, 1)] {
            assert!(dimensions(width, height).is_err());
        }
        assert!(dimensions(8000, 4000).is_ok());
    }
}
