//! HEIF/HEIC decoding without the operating system, on Windows and Linux:
//! the pure-Rust `heifer` decoder (ISO BMFF container, HEVC intra pictures,
//! grids, overlays, alpha planes, clean aperture, rotation and mirror). AVIF
//! (AV1) images go to the rav1d-based decoder in `avif`.

use super::{Decoded, Info};
use crate::{Error, Result};
use heifer::heifer_isobmff::HeifFile;
use image::{DynamicImage, RgbImage, RgbaImage};
use std::collections::HashSet;

fn failure(message: impl std::fmt::Display) -> Error {
    super::super::error(format!("HEIF: {message}"))
}

pub(super) fn unsupported(what: &str) -> Error {
    Error::Unsupported(format!("{what}; no image has been sent to a model"))
}

/// The displayable images of the container, in item order: coded or derived
/// image items that are neither hidden (grid tiles) nor a thumbnail or an
/// auxiliary image (alpha, depth) of another, as ImageIO counts them.
fn images(file: &HeifFile<'_>) -> Vec<u32> {
    let dependent: HashSet<u32> = file
        .references
        .iter()
        .filter(|reference| matches!(&reference.ref_type.0, b"thmb" | b"auxl"))
        .map(|reference| reference.from)
        .collect();
    file.items
        .iter()
        .filter(|item| {
            matches!(
                &item.item_type.0,
                b"hvc1" | b"grid" | b"iovl" | b"iden" | b"av01" | b"unci"
            ) && !item.hidden
                && !dependent.contains(&item.id)
        })
        .map(|item| item.id)
        .collect()
}

pub(super) fn decode(bytes: &[u8]) -> Result<Decoded> {
    super::validate_container(bytes)?;
    // heifer is young and rav1d is a port of C; an image either cannot handle
    // must fail this conversion, not the process. The closure only reads the
    // caller's bytes.
    std::panic::catch_unwind(|| decode_checked(bytes))
        .unwrap_or_else(|_| Err(failure("the portable decoder failed on this image")))
}

fn decode_checked(bytes: &[u8]) -> Result<Decoded> {
    let file = HeifFile::parse(bytes).map_err(failure)?;
    file.primary_item().map_err(failure)?;
    let images = images(&file);
    let index = images
        .iter()
        .position(|&id| id == file.primary_id)
        .ok_or_else(|| failure("the container has no valid primary image"))?;
    if super::avif::is_av1(&file, file.primary_id) {
        let image = super::avif::decode(&file)?;
        return pixels(image, "AVIF", images.len(), index);
    }
    let options = heifer::Options {
        max_threads: 0,
        max_pixels: super::super::MAX_PIXELS,
    };
    let decoded = heifer::decode_with_options(bytes, &options).map_err(|error| match error {
        heifer::Error::Unsupported(what) => unsupported(&format!(
            "This HEIF image uses a feature the portable decoder does not read ({what})"
        )),
        heifer::Error::Hevc(heifer::heifer_hevc_dec::Error::Unimplemented(what)) => {
            unsupported(&format!(
                "This HEIF image uses HEVC coding the portable decoder does not read ({what})"
            ))
        }
        error => failure(error),
    })?;
    pixels(decoded, "HEIF", images.len(), index)
}

fn pixels(
    decoded: heifer::Image,
    format: &'static str,
    images: usize,
    primary: usize,
) -> Result<Decoded> {
    let (width, height) = (decoded.width, decoded.height);
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > super::super::MAX_PIXELS
    {
        return Err(failure(
            "the decoded image is empty or exceeds 32 million pixels",
        ));
    }
    let image = if decoded.has_alpha {
        RgbaImage::from_raw(width, height, decoded.to_rgba8()).map(DynamicImage::ImageRgba8)
    } else {
        RgbImage::from_raw(width, height, decoded.to_rgb8()).map(DynamicImage::ImageRgb8)
    }
    .ok_or_else(|| failure("invalid pixel buffer"))?;
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

    fn rgb(decoded: &Decoded, x: u32, y: u32) -> [u8; 3] {
        decoded.image.to_rgb8().get_pixel(x, y).0
    }

    fn near(actual: [u8; 3], expected: [u8; 3]) -> bool {
        actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 4)
    }

    #[test]
    fn heic_quadrants_orientation_and_the_primary_image_decode_without_the_os() {
        // Red, green / magenta, blue quadrants of 120 by 80 (generate-quadrants.m).
        let upright = decode(include_bytes!(
            "../fixtures/heif/quadrants-orientation1.heic"
        ))
        .unwrap();
        assert_eq!((upright.image.width(), upright.image.height()), (120, 80));
        assert_eq!(
            (
                upright.info.format,
                upright.info.images,
                upright.info.primary
            ),
            ("HEIF", 1, 0)
        );
        for ((x, y), color) in [
            ((30, 20), [255, 0, 0]),
            ((90, 20), [0, 255, 0]),
            ((30, 60), [255, 0, 255]),
            ((90, 60), [0, 0, 255]),
        ] {
            assert!(
                near(rgb(&upright, x, y), color),
                "{x},{y}: {:?}",
                rgb(&upright, x, y)
            );
        }
        // Turned a quarter clockwise for display.
        let turned = decode(include_bytes!(
            "../fixtures/heif/quadrants-orientation6.heic"
        ))
        .unwrap();
        assert_eq!((turned.image.width(), turned.image.height()), (80, 120));
        assert!(near(rgb(&turned, 20, 30), [255, 0, 255]));
        assert!(near(rgb(&turned, 60, 30), [255, 0, 0]));
        // The second of two images is the primary one: all green.
        let second = decode(include_bytes!("../fixtures/heif/primary-second.heic")).unwrap();
        assert_eq!((second.info.images, second.info.primary), (2, 1));
        assert!(near(rgb(&second, 30, 20), [0, 255, 0]));
    }

    #[test]
    fn damaged_files_are_errors_not_empty_images() {
        let whole = include_bytes!("../fixtures/heif/english.heic");
        assert!(decode(&whole[..whole.len() / 2]).is_err());
    }
}
