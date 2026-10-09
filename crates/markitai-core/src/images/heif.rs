//! In-memory HEIF/AVIF primary-image decoding: through the operating system
//! on macOS, with the pure-Rust HEIF and AV1 decoders elsewhere.

use crate::Result;
use image::DynamicImage;

#[cfg(not(target_os = "macos"))]
mod avif;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(not(target_os = "macos"))]
mod portable;

#[derive(Debug, Clone, Copy)]
pub(super) struct Info {
    pub format: &'static str,
    pub images: usize,
    pub primary: usize,
}

pub(super) struct Decoded {
    pub image: DynamicImage,
    pub info: Info,
}

/// Inspect only a bounded ISO BMFF file-type box. Compatible brands matter:
/// generic `mif1` files can carry either HEVC or AV1 image items.
pub(super) fn signature(bytes: &[u8]) -> bool {
    if bytes.len() < 16 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    let size = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
    let (header, size) = if size == 1 {
        if bytes.len() < 24 {
            return false;
        }
        let Ok(size) = usize::try_from(u64::from_be_bytes(bytes[8..16].try_into().unwrap())) else {
            return false;
        };
        (16, size)
    } else {
        (8, size)
    };
    if size < header + 8 || size > bytes.len() || size > 4096 {
        return false;
    }
    let brand = |value: &[u8]| {
        matches!(
            value,
            b"avif"
                | b"avis"
                | b"heic"
                | b"heix"
                | b"hevc"
                | b"hevx"
                | b"heim"
                | b"heis"
                | b"hevm"
                | b"hevs"
                | b"mif1"
                | b"msf1"
        )
    };
    brand(&bytes[header..header + 4])
        || bytes[header + 8..size]
            .as_chunks::<4>()
            .0
            .iter()
            .any(|value| brand(value))
}

fn validate_container(bytes: &[u8]) -> Result<()> {
    let mut offset = 0usize;
    let mut boxes = 0usize;
    while offset < bytes.len() {
        boxes += 1;
        if boxes > 100_000 || bytes.len() - offset < 8 {
            return Err(super::error(
                "HEIF/AVIF container has too many boxes or a truncated box header",
            ));
        }
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let (header, length) = match size {
            0 => (8, bytes.len() - offset),
            1 => {
                if bytes.len() - offset < 16 {
                    return Err(super::error(
                        "HEIF/AVIF container has a truncated large box header",
                    ));
                }
                let length = u64::from_be_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
                (
                    16,
                    usize::try_from(length)
                        .map_err(|_| super::error("HEIF/AVIF box length overflows"))?,
                )
            }
            value => (8, value as usize),
        };
        if length < header || length > bytes.len() - offset {
            return Err(super::error(
                "HEIF/AVIF container has an invalid or truncated box",
            ));
        }
        offset += length;
    }
    if boxes == 0 {
        return Err(super::error("HEIF/AVIF container is empty"));
    }
    Ok(())
}

pub(super) fn decode(bytes: &[u8]) -> Result<Decoded> {
    if bytes.is_empty() || bytes.len() > 500 * 1024 * 1024 {
        return Err(super::error("HEIF/AVIF input is empty or exceeds 500 MiB"));
    }
    if !signature(bytes) {
        return Err(super::error("invalid HEIF/AVIF file-type box"));
    }
    #[cfg(target_os = "macos")]
    {
        macos::decode(bytes)
    }
    #[cfg(not(target_os = "macos"))]
    {
        portable::decode(bytes)
    }
}

#[cfg(target_os = "macos")]
fn unavailable() -> crate::Error {
    crate::Error::Unsupported("The macOS ImageIO runtime cannot decode this HEIF/AVIF image; no image has been sent to a model".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_type_detection_is_bounded_and_uses_compatible_brands() {
        let mut bytes = Vec::from(24u32.to_be_bytes());
        bytes.extend_from_slice(b"ftypzzzz\0\0\0\0avifmif1");
        assert!(signature(&bytes));
        bytes[..4].copy_from_slice(&8000u32.to_be_bytes());
        assert!(!signature(&bytes));
        assert!(!signature(b"not an image"));
        assert!(signature(include_bytes!(
            "fixtures/heif/quadrants-orientation1.heic"
        )));
        assert!(signature(include_bytes!("fixtures/heif/white_1x1.avif")));
    }

    #[test]
    fn declared_box_lengths_reject_partial_payloads_and_allow_large_or_eof_boxes() {
        let original = include_bytes!("fixtures/heif/quadrants-orientation1.heic");
        assert!(validate_container(original).is_ok());
        for cut in [1, 7, original.len() / 2, original.len() - 1] {
            assert!(validate_container(&original[..cut]).is_err(), "cut {cut}");
        }
        let mut large = Vec::from(1u32.to_be_bytes());
        large.extend_from_slice(b"free");
        large.extend_from_slice(&16u64.to_be_bytes());
        assert!(validate_container(&large).is_ok());
        large[8..16].copy_from_slice(&u64::MAX.to_be_bytes());
        assert!(validate_container(&large).is_err());
        assert!(validate_container(b"\0\0\0\0mdatpayload").is_ok());
        assert!(validate_container(b"\0\0\0\x04free").is_err());
    }
}
