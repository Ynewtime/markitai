//! `TSP.DataInfo` — the package's media registry, and the bytes it points at.
//!
//! A drawable never carries pixels. It carries a `TSP.DataReference`, which is
//! an index into `TSP.PackageMetadata.datas`, which is a list of `DataInfo`
//! records, each naming a file under `Data/`. The registry behaves exactly like
//! the interned string lists Phase 2 found inside tables: **it is refcounted**.
//! Replacing an image in Keynote and saving twice left the first replacement's
//! `DataInfo` *and* its `Data/` entry gone the moment nothing referred to them
//! any more, and allocated a new identifier for the next one.
//!
//! What a `DataInfo` holds, all of it confirmed against documents the apps
//! wrote:
//!
//! | Field | Contents |
//! |---|---|
//! | 1 | identifier — the number a `DataReference` names |
//! | 2 | **a raw 20-byte SHA-1 of the file's bytes** |
//! | 3 | `preferred_file_name` — the name the file had outside the document |
//! | 4 | `file_name` — the entry under `Data/`, `<stem>-<identifier>.<ext>` |
//! | 5 | path inside the app's theme bundle, for assets not copied in |
//! | 10 | attributes; `100` is `TSD.ImageDataAttributes`, whose field 1 is the pixel size |
//! | 18 | `materialized_length` — the file's byte length |
//!
//! The digest was checked rather than assumed: `shasum Data/probe-9077.png`
//! gives `9b157e9efb914ffdc2d4d1263ec0893283f53b56`, and so does field 2.
//!
//! A theme asset — a picture that came with the template — has an empty
//! `file_name` and a `document_resource_locator` instead, and its bytes are not
//! in the package at all. There is nothing to replace in place.

use crate::pb::{decode_nested, Message, Value};

/// Field numbers of `TSP.DataInfo`.
pub mod field {
    pub const IDENTIFIER: u32 = 1;
    pub const DIGEST: u32 = 2;
    pub const PREFERRED_FILE_NAME: u32 = 3;
    pub const FILE_NAME: u32 = 4;
    pub const RESOURCE_LOCATOR: u32 = 5;
    pub const ATTRIBUTES: u32 = 10;
    pub const MATERIALIZED_LENGTH: u32 = 18;
    /// `TSD.ImageDataAttributes`, grafted onto `TSP.DataAttributes` as proto2
    /// extension 100.
    pub const IMAGE_ATTRIBUTES: u32 = 100;
    /// `TSD.ImageDataAttributes.pixel_size`.
    pub const PIXEL_SIZE: u32 = 1;
}

/// RFC 3174 SHA-1, because `TSP.DataInfo.digest` is one and this crate takes no
/// dependencies.
///
/// Sixty lines is cheaper than a dependency for a hash whose only job here is
/// to reproduce a value the app already computed — and which is checked against
/// `shasum` over every stored media file in the corpus by
/// `tests/drawables.rs::every_digest_is_the_sha1_of_the_bytes`.
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut message = data.to_vec();
    let bits = (data.len() as u64) * 8;
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bits.to_be_bytes());

    for chunk in message.chunks(64) {
        let mut w = [0u32; 80];
        for (i, word) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e]) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut out = [0u8; 20];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Pixel dimensions of a PNG or a JPEG, read from its header.
///
/// The app records the size in two places that must agree — the `DataInfo`'s
/// image attributes and the drawable's `naturalSize` — so a replacement that
/// does not know the new picture's size cannot maintain either. Only the two
/// formats the corpus contains are decoded; anything else asks the caller to
/// say, rather than guessing from a format nobody here has checked.
pub fn pixel_size(bytes: &[u8]) -> Option<(f32, f32)> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") && bytes.len() >= 24 {
        // IHDR is always the first chunk: 8 bytes of signature, 4 of length,
        // 4 of type, then width and height as big-endian u32.
        if &bytes[12..16] != b"IHDR" {
            return None;
        }
        let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
        let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
        return Some((width as f32, height as f32));
    }
    if bytes.starts_with(b"\xff\xd8") {
        let mut at = 2usize;
        while at + 9 < bytes.len() {
            if bytes[at] != 0xFF {
                at += 1;
                continue;
            }
            let marker = bytes[at + 1];
            // SOF0..SOF15, skipping the four that are not frame headers.
            if (0xC0..=0xCF).contains(&marker)
                && !matches!(marker, 0xC4 | 0xC8 | 0xCC)
                && at + 9 < bytes.len()
            {
                let height = u16::from_be_bytes([bytes[at + 5], bytes[at + 6]]);
                let width = u16::from_be_bytes([bytes[at + 7], bytes[at + 8]]);
                return Some((f32::from(width), f32::from(height)));
            }
            if matches!(marker, 0xD8 | 0xD9 | 0x01) || (0xD0..=0xD7).contains(&marker) {
                at += 2;
                continue;
            }
            let length = u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]) as usize;
            if length < 2 {
                return None;
            }
            at += 2 + length;
        }
    }
    None
}

/// The name a file takes under `Data/`: its stem, the data identifier, and its
/// extension — `probe-a.png` placed as data 9084 becomes `probe-a-9084.png`,
/// which is what Keynote wrote.
pub fn stored_name(preferred: &str, identifier: u64) -> String {
    let (stem, extension) = match preferred.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() => (stem, Some(extension)),
        _ => (preferred, None),
    };
    match extension {
        Some(extension) => format!("{stem}-{identifier}.{extension}"),
        None => format!("{stem}-{identifier}"),
    }
}

/// What [`crate::Document::replace_media`] did.
#[derive(Debug, Clone)]
pub struct MediaReplacement {
    /// The `DataInfo` identifier whose bytes were swapped.
    pub data: u64,
    /// Package entry before and after — the name changes with the file's.
    pub was: String,
    pub now: String,
    pub digest: [u8; 20],
    pub bytes: usize,
    /// The old picture's size: the registry's recorded pixel size when it has
    /// one, otherwise the drawable's `naturalSize`. Only four of the corpus's
    /// 203 stored files record a pixel size, so the `naturalSize` fallback is
    /// what lets the stretched-aspect check see the old shape at all.
    pub old_size: Option<(f32, f32)>,
    pub new_pixel_size: (f32, f32),
    /// Whether the new pixel size could be written into the registry entry. It
    /// cannot be when the `DataInfo` carries no image attributes to hold it —
    /// this crate does not invent the message — and a caller that relied on the
    /// recorded size being current should be told.
    pub pixel_size_recorded: bool,
    /// Drawables whose `naturalSize` and traced outline were brought into step.
    pub drawables: Vec<u64>,
    /// True when the new picture is a different shape from the old one, so it
    /// is drawn stretched into a frame that was chosen for the old one.
    pub aspect_changed: bool,
}

impl MediaReplacement {
    pub fn aspect_note(&self) -> Option<String> {
        if !self.aspect_changed {
            return None;
        }
        let (ow, oh) = self.old_size?;
        Some(format!(
            "the picture was {ow:.0} × {oh:.0} and is now {:.0} × {:.0}: the frame did not \
             change, so it is drawn stretched. `iwork set-geometry` fixes the frame.",
            self.new_pixel_size.0, self.new_pixel_size.1
        ))
    }
}

/// Read the pixel size a `DataInfo` records, if it records one.
pub fn attribute_pixel_size(info: &Message) -> Option<(f32, f32)> {
    let attributes = info.bytes(field::ATTRIBUTES).and_then(decode_nested)?;
    let image = attributes
        .bytes(field::IMAGE_ATTRIBUTES)
        .and_then(decode_nested)?;
    let point = image.bytes(field::PIXEL_SIZE).and_then(decode_nested)?;
    match (point.get(1), point.get(2)) {
        (Some(Value::Fixed32(w)), Some(Value::Fixed32(h))) => {
            Some((f32::from_le_bytes(*w), f32::from_le_bytes(*h)))
        }
        _ => None,
    }
}

/// Write the pixel size into a `DataInfo`'s image attributes, if it has them.
///
/// The attributes are not invented when they are absent: a `DataAttributes`
/// this crate composed would carry only the fields it happened to know, and
/// the whole discipline here is not to write a message it has never seen.
pub fn set_attribute_pixel_size(info: &mut Message, width: f32, height: f32) -> bool {
    let Some(mut attributes) = info.bytes(field::ATTRIBUTES).and_then(decode_nested) else {
        return false;
    };
    let Some(mut image) = attributes
        .bytes(field::IMAGE_ATTRIBUTES)
        .and_then(decode_nested)
    else {
        return false;
    };
    let mut point = image
        .bytes(field::PIXEL_SIZE)
        .and_then(decode_nested)
        .unwrap_or_default();
    if point.get(1).is_none() && point.get(2).is_none() {
        return false;
    }
    point.set_in_order(1, Value::Fixed32(width.to_le_bytes()));
    point.set_in_order(2, Value::Fixed32(height.to_le_bytes()));
    image.set_in_order(field::PIXEL_SIZE, Value::Bytes(point.encode()));
    attributes.set_in_order(field::IMAGE_ATTRIBUTES, Value::Bytes(image.encode()));
    info.set_in_order(field::ATTRIBUTES, Value::Bytes(attributes.encode()));
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three vectors from RFC 3174, plus the empty string.
    #[test]
    fn sha1_matches_the_published_vectors() {
        let hex = |bytes: [u8; 20]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(
            hex(sha1(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(
            hex(sha1(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
        assert_eq!(
            hex(sha1(&[b'a'; 1_000_000])),
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f"
        );
    }

    /// The message length is padded to a multiple of 64 with the bit count in
    /// the last eight bytes; the boundary cases are where hand-rolled SHA-1
    /// goes wrong.
    #[test]
    fn sha1_handles_every_padding_boundary() {
        let hex = |bytes: [u8; 20]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        // 55, 56, 63, 64 bytes: the four lengths where the block count
        // changes. Values from `head -c N /dev/zero | shasum`.
        assert_eq!(
            hex(sha1(&[0u8; 55])),
            "8e8832c642a6a38c74c17fc92ccedc266c108e6c"
        );
        assert_eq!(
            hex(sha1(&[0u8; 63])),
            "0b8bf9fc37ad802cefa6733ec62b09d5f43a1b75"
        );
        assert_eq!(
            hex(sha1(&[0u8; 56])),
            "9438e360f578e12c0e0e8ed28e2c125c1cefee16"
        );
        assert_eq!(
            hex(sha1(&[0u8; 64])),
            "c8d7d0ef0eedfa82d2ea1aa592845b9a6d4b02b7"
        );
    }

    #[test]
    fn png_and_jpeg_sizes_come_out_of_the_header() {
        // A minimal PNG header: signature, IHDR length, "IHDR", 32 × 24.
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&13u32.to_be_bytes());
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&32u32.to_be_bytes());
        png.extend_from_slice(&24u32.to_be_bytes());
        assert_eq!(pixel_size(&png), Some((32.0, 24.0)));

        // JPEG: SOI, an APP0 segment to skip, then SOF0 with height 200,
        // width 300.
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00];
        jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
        jpeg.extend_from_slice(&200u16.to_be_bytes());
        jpeg.extend_from_slice(&300u16.to_be_bytes());
        jpeg.extend_from_slice(&[0u8; 8]);
        assert_eq!(pixel_size(&jpeg), Some((300.0, 200.0)));

        assert_eq!(pixel_size(b"not an image at all"), None);
        assert_eq!(pixel_size(b""), None);
    }

    #[test]
    fn a_stored_name_carries_the_data_identifier() {
        assert_eq!(stored_name("probe-a.png", 9084), "probe-a-9084.png");
        assert_eq!(stored_name("Sommer.Foto.jpeg", 12), "Sommer.Foto-12.jpeg");
        assert_eq!(stored_name("nameless", 3), "nameless-3");
    }
}
