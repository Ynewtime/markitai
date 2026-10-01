//! TIFF page discovery and bounded decoding, with one pixel buffer at a time.

use super::{ImageBytes, MAX_DECODED, MAX_PIXELS, error};
use crate::{Error, Result};
use ::tiff::decoder::Decoder;
use image::{DynamicImage, ImageDecoder};

pub(super) const MAX_PAGES: usize = 1_000;
pub(super) const MAX_TOTAL_PIXELS: u64 = 2_000_000_000;
pub(super) const MAX_INPUT: usize = 100 * 1024 * 1024;
pub(super) const MAX_ENCODED: usize = 256 * 1024 * 1024;

pub(super) fn signature(bytes: &[u8]) -> bool {
    [b"II*\0".as_slice(), b"MM\0*", b"II+\0", b"MM\0+"]
        .iter()
        .any(|magic| bytes.starts_with(magic))
}

pub(super) fn multiple(bytes: &[u8]) -> Result<bool> {
    Ok(Decoder::new(ImageBytes::new(bytes))
        .map_err(error)?
        .more_images())
}

pub(super) struct Pages<'a> {
    bytes: &'a [u8],
    offsets: Vec<u64>,
}

impl<'a> Pages<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Result<Self> {
        let mut directory = Decoder::new(ImageBytes::new(bytes)).map_err(error)?;
        if directory.more_images() && bytes.len() > MAX_INPUT {
            return Err(error("multi-page TIFF input exceeds 100 MiB"));
        }
        let mut offsets = Vec::new();
        let mut total_pixels = 0_u64;
        loop {
            if offsets.len() == MAX_PAGES {
                return Err(error(
                    "TIFF exceeds the 1000-page limit; no pages were processed",
                ));
            }
            let offset = directory
                .ifd_pointer()
                .ok_or_else(|| error("missing TIFF page directory"))?
                .0;
            let mut decoder = page_decoder(bytes, offset)?;
            let (width, height) = decoder.dimensions();
            let pixels = u64::from(width) * u64::from(height);
            total_pixels = total_pixels
                .checked_add(pixels)
                .ok_or_else(|| error("TIFF pixel count overflow"))?;
            if total_pixels > MAX_TOTAL_PIXELS {
                return Err(error("TIFF exceeds the cumulative 2 billion pixel limit"));
            }
            // Parse every page's orientation before any OCR or encoding occurs.
            decoder.orientation().map_err(error)?;
            offsets.push(offset);
            if !directory.more_images() {
                break;
            }
            directory.next_image().map_err(error)?;
        }
        Ok(Self { bytes, offsets })
    }

    pub(super) fn len(&self) -> usize {
        self.offsets.len()
    }

    pub(super) fn check_vision_limit(&self, cfg: &serde_json::Value) -> Result<()> {
        let limit = cfg
            .pointer("/llm/max_vision_pages_per_document")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if limit > 0 && self.len() as u64 > limit {
            return Err(Error::InvalidInput("TIFF pages exceed llm.max_vision_pages_per_document; no pages were decoded or sent".into()));
        }
        Ok(())
    }

    pub(super) fn decode(&self, index: usize) -> Result<DynamicImage> {
        let offset = *self
            .offsets
            .get(index)
            .ok_or_else(|| error("TIFF page is out of range"))?;
        let mut decoder = page_decoder(self.bytes, offset)?;
        let orientation = decoder.orientation().map_err(error)?;
        let mut image = DynamicImage::from_decoder(decoder).map_err(error)?;
        image.apply_orientation(orientation);
        Ok(image)
    }
}

fn page_decoder(
    bytes: &[u8],
    offset: u64,
) -> Result<image::codecs::tiff::TiffDecoder<ImageBytes<'_>>> {
    let mut decoder =
        image::codecs::tiff::TiffDecoder::new(page_reader(bytes, offset)?).map_err(error)?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(error(
            "TIFF page is empty or exceeds the 32 million pixel limit",
        ));
    }
    if decoder.total_bytes() > MAX_DECODED {
        return Err(error("TIFF decoded page exceeds 256 MiB"));
    }
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODED);
    decoder.set_limits(limits).map_err(error)?;
    Ok(decoder)
}

// The image crate handles sample formats, planar data and orientation. Give its
// single-page decoder a virtual first-IFD pointer instead of copying a whole TIFF
// for every page or implementing a second set of color conversions.
fn page_reader(bytes: &[u8], offset: u64) -> Result<ImageBytes<'_>> {
    let little = bytes.starts_with(b"II");
    let big = bytes.starts_with(b"II+\0") || bytes.starts_with(b"MM\0+");
    let header_len = if big { 16 } else { 8 };
    if bytes.len() < header_len || !signature(bytes) {
        return Err(error("invalid TIFF header"));
    }
    let mut header = [0; 16];
    header[..header_len].copy_from_slice(&bytes[..header_len]);
    if big {
        header[8..16].copy_from_slice(&if little {
            offset.to_le_bytes()
        } else {
            offset.to_be_bytes()
        });
    } else {
        let offset = u32::try_from(offset).map_err(error)?;
        header[4..8].copy_from_slice(&if little {
            offset.to_le_bytes()
        } else {
            offset.to_be_bytes()
        });
    }
    Ok(ImageBytes {
        bytes,
        header,
        header_len,
        position: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::tiff::{
        encoder::{TiffEncoder, colortype},
        tags::Tag,
    };
    use image::Rgb;
    use std::io::Cursor;

    fn rgb_pages(orientations: &[u16]) -> Vec<u8> {
        let mut buffer = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut buffer).unwrap();
        for &orientation in orientations {
            let mut page = encoder.new_image::<colortype::RGB8>(3, 2).unwrap();
            page.encoder()
                .write_tag(Tag::Orientation, orientation)
                .unwrap();
            page.write_data(&[1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0])
                .unwrap();
        }
        buffer.into_inner()
    }

    fn change_tag(bytes: &mut [u8], offset: u64, tag: u16, value: u32) {
        let start = offset as usize;
        let count = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap()) as usize;
        let entry = (0..count)
            .map(|index| start + 2 + 12 * index)
            .find(|&entry| u16::from_le_bytes(bytes[entry..entry + 2].try_into().unwrap()) == tag)
            .unwrap();
        bytes[entry + 8..entry + 12].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn every_ifd_has_its_own_orientation_and_pixel_order() {
        let bytes = rgb_pages(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let original = bytes.clone();
        let pages = Pages::new(&bytes).unwrap();
        let expected = [
            vec![1, 2, 3, 4, 5, 6],
            vec![3, 2, 1, 6, 5, 4],
            vec![6, 5, 4, 3, 2, 1],
            vec![4, 5, 6, 1, 2, 3],
            vec![1, 4, 2, 5, 3, 6],
            vec![4, 1, 5, 2, 6, 3],
            vec![6, 3, 5, 2, 4, 1],
            vec![3, 6, 2, 5, 1, 4],
        ];
        assert_eq!(pages.len(), 8);
        for (index, values) in expected.iter().enumerate() {
            let image = pages.decode(index).unwrap().to_rgb8();
            assert_eq!(image.dimensions(), if index < 4 { (3, 2) } else { (2, 3) });
            assert_eq!(
                image
                    .pixels()
                    .map(|Rgb(pixel)| pixel[0])
                    .collect::<Vec<_>>(),
                *values
            );
        }
        assert_eq!(bytes, original);
    }

    #[test]
    fn bigtiff_and_sixteen_bit_grayscale_use_all_pages() {
        let mut buffer = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new_big(&mut buffer).unwrap();
        encoder
            .write_image::<colortype::Gray16>(2, 1, &[0, 65535])
            .unwrap();
        encoder
            .write_image::<colortype::Gray16>(2, 1, &[65535, 0])
            .unwrap();
        let bytes = buffer.into_inner();
        assert!(signature(&bytes));
        let pages = Pages::new(&bytes).unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(pages.decode(0).unwrap().to_luma16().into_raw(), [0, 65535]);
        assert_eq!(pages.decode(1).unwrap().to_luma16().into_raw(), [65535, 0]);
    }

    /// A page reader reads the file with only its first-directory pointer
    /// replaced, whichever way it is read or sought.
    #[test]
    fn a_page_reader_reads_the_file_with_the_page_pointer() {
        use std::io::{BufRead, Read, Seek, SeekFrom};
        let body: Vec<u8> = (0..40u8).collect();
        for (magic, offset, at, pointer) in [
            (&b"II*\0"[..], 0x0102_0304_u64, 4..8, vec![4, 3, 2, 1]),
            (&b"MM\0*"[..], 0x0102_0304, 4..8, vec![1, 2, 3, 4]),
            (
                &b"II+\0"[..],
                0x0102_0304_0506,
                8..16,
                vec![6, 5, 4, 3, 2, 1, 0, 0],
            ),
            (
                &b"MM\0+"[..],
                0x0102_0304_0506,
                8..16,
                vec![0, 0, 1, 2, 3, 4, 5, 6],
            ),
        ] {
            let file: Vec<u8> = magic.iter().copied().chain(body.iter().copied()).collect();
            let mut expected = file.clone();
            expected[at].copy_from_slice(&pointer);

            let mut whole = Vec::new();
            page_reader(&file, offset)
                .unwrap()
                .read_to_end(&mut whole)
                .unwrap();
            assert_eq!(whole, expected);

            let mut reader = page_reader(&file, offset).unwrap();
            let mut pieces = Vec::new();
            let mut piece = [0; 3];
            loop {
                let count = reader.read(&mut piece).unwrap();
                if count == 0 {
                    break;
                }
                pieces.extend_from_slice(&piece[..count]);
            }
            assert_eq!(pieces, expected);

            assert_eq!(reader.seek(SeekFrom::Start(2)).unwrap(), 2);
            assert_eq!(reader.fill_buf().unwrap()[0], expected[2]);
            reader.consume(5);
            let mut rest = Vec::new();
            reader.read_to_end(&mut rest).unwrap();
            assert_eq!(rest, expected[7..]);
            assert_eq!(
                reader.seek(SeekFrom::End(-4)).unwrap(),
                file.len() as u64 - 4
            );
            let mut tail = [0; 4];
            reader.read_exact(&mut tail).unwrap();
            assert_eq!(tail, expected[file.len() - 4..]);
            let error = reader.seek(SeekFrom::Current(-1000)).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
            assert_eq!(error.to_string(), "invalid TIFF seek");
        }
        // A classic TIFF cannot point past 4 GiB, and a short file has no header.
        assert!(page_reader(b"II*\0\0\0\0\0", 1 << 32).is_err());
        assert!(page_reader(b"II*\0", 8).is_err());
    }

    #[test]
    fn a_bad_later_directory_and_cycles_fail_discovery() {
        let mut bytes = rgb_pages(&[1, 1]);
        let offsets = Pages::new(&bytes).unwrap().offsets;
        bytes.truncate(offsets[1] as usize + 1);
        assert!(Pages::new(&bytes).is_err());
        let mut bytes = rgb_pages(&[1, 1]);
        let start = offsets[1] as usize;
        let count = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap()) as usize;
        bytes[start + 2 + count * 12..start + 6 + count * 12]
            .copy_from_slice(&(offsets[0] as u32).to_le_bytes());
        assert!(Pages::new(&bytes).is_err());
    }

    #[test]
    fn budgets_reject_the_whole_document_before_decoding_pixels() {
        let too_many = rgb_pages(&vec![1; MAX_PAGES + 1]);
        assert!(
            Pages::new(&too_many)
                .err()
                .unwrap()
                .to_string()
                .contains("1000-page")
        );
        let mut bytes = rgb_pages(&[1, 1]);
        let offset = Pages::new(&bytes).unwrap().offsets[1];
        change_tag(&mut bytes, offset, 256, 100_000);
        change_tag(&mut bytes, offset, 257, 100_000);
        assert!(
            Pages::new(&bytes)
                .err()
                .unwrap()
                .to_string()
                .contains("32 million")
        );
        let mut bytes = rgb_pages(&[1; 81]);
        let offsets = Pages::new(&bytes).unwrap().offsets;
        for offset in offsets {
            change_tag(&mut bytes, offset, 256, 5000);
            change_tag(&mut bytes, offset, 257, 5000);
        }
        assert!(
            Pages::new(&bytes)
                .err()
                .unwrap()
                .to_string()
                .contains("2 billion")
        );
        let bytes = rgb_pages(&[1, 1, 1]);
        let pages = Pages::new(&bytes).unwrap();
        assert!(
            pages
                .check_vision_limit(&serde_json::json!({"llm":{"max_vision_pages_per_document":2}}))
                .is_err()
        );
        assert!(
            pages
                .check_vision_limit(&serde_json::json!({"llm":{"max_vision_pages_per_document":3}}))
                .is_ok()
        );
    }

    #[test]
    fn malformed_last_pixel_data_is_not_returned_as_a_complete_document() {
        let mut bytes = rgb_pages(&[1, 1]);
        let offset = Pages::new(&bytes).unwrap().offsets[1];
        let end = bytes.len() as u32;
        change_tag(&mut bytes, offset, 273, end + 100);
        let pages = Pages::new(&bytes).unwrap();
        assert!(pages.decode(0).is_ok());
        assert!(pages.decode(1).is_err());
        assert!(
            super::super::prepare_vision(&bytes, "scan.tif", &crate::config::defaults()).is_err()
        );
    }
    #[test]
    fn embedded_documents_preserve_original_and_single_page_output_stays_compatible() {
        let bytes = rgb_pages(&[1, 6]);
        let mut document = crate::Document {
            markdown: "![scan](.markitai/assets/scan.tiff)\n".into(),
            assets: vec![crate::Asset {
                name: "scan.tiff".into(),
                bytes: bytes.clone(),
            }],
            ..Default::default()
        };
        super::super::prepare_assets(&mut document, &crate::config::defaults());
        assert_eq!(document.assets.len(), 1);
        assert_eq!(document.assets[0].bytes, bytes);
        assert_eq!(document.markdown, "![scan](.markitai/assets/scan.tiff)\n");
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("scan.tiff");
        std::fs::write(&path, rgb_pages(&[1])).unwrap();
        let cfg =
            crate::config::normalize(&serde_json::json!({"image":{"compress":false}})).unwrap();
        let (document, vision) = super::super::extract(&path, &cfg, false).unwrap();
        assert_eq!(
            document.markdown,
            "# scan\n\n![scan](.markitai/assets/image.png)\n"
        );
        assert_eq!(document.assets.len(), 1);
        assert_eq!(vision.len(), 1);
        assert_eq!(vision[0].mime, "image/png");
        assert_eq!(document.assets[0].bytes, vision[0].bytes);
    }

    #[test]
    fn payload_encoding_budget_fails_instead_of_returning_truncated_png() {
        let image = DynamicImage::new_rgb8(100, 100);
        let cfg = crate::config::defaults();
        let error = super::super::encode_limited(&image, &cfg, true, 16)
            .err()
            .unwrap();
        assert!(error.to_string().contains("budget"), "{error}");
        let (bytes, _, mime) = super::super::encode_limited(&image, &cfg, true, 4096).unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(image::load_from_memory(&bytes).unwrap().width(), 100);
    }

    #[test]
    fn mixed_color_models_are_converted_without_losing_page_identity() {
        let mut buffer = Cursor::new(Vec::new());
        let mut encoder = TiffEncoder::new(&mut buffer).unwrap();
        encoder
            .write_image::<colortype::CMYK8>(1, 1, &[0, 255, 255, 0])
            .unwrap();
        encoder
            .write_image::<colortype::RGBA8>(1, 1, &[0, 0, 255, 0])
            .unwrap();
        let bytes = buffer.into_inner();
        let pages = Pages::new(&bytes).unwrap();
        assert_eq!(
            pages.decode(0).unwrap().to_rgb8().get_pixel(0, 0).0,
            [255, 0, 0]
        );
        assert_eq!(
            super::super::rgb_on_white(&pages.decode(1).unwrap())
                .get_pixel(0, 0)
                .0,
            [255, 255, 255]
        );
    }

    #[test]
    fn malformed_embedded_tiff_keeps_original_bytes_with_an_explicit_warning() {
        let bytes = b"II*\0broken".to_vec();
        let mut document = crate::Document {
            markdown: "![scan](.markitai/assets/broken.tiff)\n".into(),
            assets: vec![crate::Asset {
                name: "broken.tiff".into(),
                bytes: bytes.clone(),
            }],
            ..Default::default()
        };
        super::super::prepare_assets(&mut document, &crate::config::defaults());
        assert_eq!(document.assets.len(), 1);
        assert_eq!(document.assets[0].bytes, bytes);
        assert_eq!(document.markdown, "![scan](.markitai/assets/broken.tiff)\n");
        assert_eq!(document.warnings.len(), 1);
        assert!(document.warnings[0].contains("preserved unchanged"));
    }
}
