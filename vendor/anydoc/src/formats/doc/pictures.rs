//! Floating DOC pictures follow PlcfSpaMom -> FSP.spid -> FOPT.pib -> FBSE.
//! The delayed BLIP stream is WordDocument (MS-DOC 2.1.1), not Data. Only
//! live main-story anchors are exposed; unrelated BLIPs are never decoded.

use super::{TextStream, objects};
use crate::error::ConvertError;
use crate::model::{ImageSource, Inline};
use crate::package::limits;
use crate::shared::assets::AssetSink;
use crate::shared::binary::{get_u16, get_u32};
use crate::shared::officeart::{Blip, decode_blip, record_at};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Default)]
pub(super) struct Pictures<'a> {
    anchors: HashMap<usize, u32>,
    blips: Vec<Option<&'a [u8]>>,
    decoded: RefCell<HashMap<u32, Option<Inline>>>,
    pub(super) warnings: Vec<String>,
}

impl<'a> Pictures<'a> {
    pub(super) fn read(
        word: &'a [u8],
        table: &'a [u8],
        text: &TextStream,
        main_count: usize,
    ) -> Result<Self, ConvertError> {
        let mut output = Self::default();
        if get_u32(word, 0x1DE).unwrap_or(0) == 0 {
            return Ok(output);
        }
        let Some(anchors) = objects::textbox_plc(word, table, 0x1DA, 26)? else {
            output.reject("main shape anchors are malformed");
            return Ok(output);
        };
        let Some((group, main)) = objects::drawing_parts(word, table) else {
            output.reject("drawing containers are missing or malformed");
            return Ok(output);
        };
        let count = (anchors.len() - 4) / 30;
        if !objects::increasing_cps(anchors, count, None) {
            output.reject("main shape anchor positions are inconsistent");
            return Ok(output);
        }
        let mut visited = 0;
        let mut shapes = HashMap::new();
        if !picture_shapes(main, 0, true, &mut visited, &mut shapes)? {
            output.reject("picture shape identities or properties are malformed");
            return Ok(output);
        }
        let mut by_shape = HashMap::new();
        for n in 0..count {
            let cp = get_u32(anchors, 4 * n).unwrap() as usize;
            let id = get_u32(anchors, 4 * (count + 1) + 26 * n).unwrap();
            let index = text.index_of_cp(cp);
            let valid = cp < main_count
                && text.cps.get(index).copied() == Some(cp as u32)
                && text.chars.get(index) == Some(&'\u{8}');
            if by_shape.insert(id, valid.then_some(index)).is_some() {
                by_shape.insert(id, None);
            }
        }
        for (id, anchor) in by_shape {
            if let (Some(index), Some(&Some(pib))) = (anchor, shapes.get(&id)) {
                output.anchors.insert(index, pib);
            }
        }
        if output.anchors.is_empty() {
            return Ok(output);
        }
        let Some(blips) = blip_store(group, word, &mut visited)? else {
            output.reject("picture store is missing or malformed");
            return Ok(output);
        };
        if blips.count_mismatch {
            output.warnings.push("Word picture metadata had an inconsistent count; images were recovered from complete records.".into());
        }
        output.blips = blips.records;
        Ok(output)
    }

    pub(super) fn contains(&self, index: usize) -> bool {
        self.anchors.contains_key(&index)
    }

    /// Called only after field visibility and CFSpec have been checked.
    pub(super) fn image_at(
        &self,
        index: usize,
        assets: &RefCell<AssetSink>,
    ) -> Result<Option<Inline>, ConvertError> {
        let Some(&pib) = self.anchors.get(&index) else { return Ok(None) };
        if let Some(image) = self.decoded.borrow().get(&pib) {
            return Ok(image.clone());
        }
        let record = pib.checked_sub(1).and_then(|n| self.blips.get(n as usize)).copied().flatten();
        let blip = record
            .and_then(|record| record_at(record, 0))
            .and_then(|(version, kind, body)| decode_picture(version, kind, body));
        let image = if let Some(blip) = blip {
            let part = format!("word/drawing{pib}.{}", blip.extension);
            let id = assets.borrow_mut().add(blip.media_type.into(), part, &blip.bytes)?;
            Some(Inline::Image { alt: String::new(), source: ImageSource::Asset(id) })
        } else {
            None
        };
        self.decoded.borrow_mut().insert(pib, image.clone());
        Ok(image)
    }

    fn reject(&mut self, reason: &str) {
        self.warnings.push(format!("Word floating picture: {reason}; image data was not read."));
    }
}

/// Preserve every BStore slot, including empty/unsupported entries, because
/// pib is a one-based slot number, not an index of successfully decoded images.
struct BlipStore<'a> {
    records: Vec<Option<&'a [u8]>>,
    count_mismatch: bool,
}

fn blip_store<'a>(
    group: &'a [u8],
    word: &'a [u8],
    visited: &mut usize,
) -> Result<Option<BlipStore<'a>>, ConvertError> {
    let mut cursor = 0;
    let mut store = None;
    while cursor < group.len() {
        objects::count_drawing_record(visited)?;
        let Some((version, kind, body)) = record_at(group, cursor) else { return Ok(None) };
        if kind == 0xF001 {
            if version & 0xF != 0xF || store.is_some() {
                return Ok(None);
            }
            let mut entries = Vec::new();
            let mut at = 0;
            while at < body.len() {
                objects::count_drawing_record(visited)?;
                let Some((v, kind, fbse)) = record_at(body, at) else { return Ok(None) };
                let image = match kind {
                    0xF007 if v & 0xF == 2 => fbse_record(v, fbse, word),
                    0xF018..=0xF117 if v & 0xF == 0 => (fbse.len()
                        <= limits::MAX_ENTRY_BYTES as usize)
                        .then(|| &body[at..at + 8 + fbse.len()]),
                    _ => return Ok(None),
                };
                entries.push(image);
                at += 8 + fbse.len();
            }
            // Some producers leave recInstance stale. Complete records inside
            // the declared container still define unambiguous numbered slots.
            store = Some(BlipStore {
                count_mismatch: entries.len() != (version >> 4) as usize,
                records: entries,
            });
        }
        cursor += 8 + body.len();
    }
    Ok(store)
}

/// Follow exactly the FBSE's embedded payload or delay offset and size. A
/// wrong type, truncated record, or UID mismatch cannot resolve to a neighbor.
fn fbse_record<'a>(version: u16, fbse: &'a [u8], word: &'a [u8]) -> Option<&'a [u8]> {
    let header = fbse.get(..36)?;
    let kind = version >> 4;
    if kind != header[0] as u16 && kind != header[1] as u16 {
        return None;
    }
    let name_len = header[33] as usize;
    if !name_len.is_multiple_of(2) || get_u32(header, 24)? == 0 {
        return None;
    }
    let size = get_u32(header, 20)? as usize;
    if size > limits::MAX_ENTRY_BYTES as usize || size < 8 {
        return None;
    }
    let tail = fbse.get(36 + name_len..)?;
    let record = if tail.is_empty() {
        let offset = get_u32(header, 28)? as usize;
        word.get(offset..)?.get(..size)?
    } else {
        if tail.len() != size {
            return None;
        }
        tail
    };
    let (version, record_type, body) = record_at(record, 0)?;
    if body.len() + 8 != size || version & 0xF != 0 {
        return None;
    }
    let instance = version >> 4;
    let expected = match record_type {
        0xF01A if matches!(instance, 0x3D4 | 0x3D5) => 2,
        0xF01B if matches!(instance, 0x216 | 0x217) => 3,
        0xF01D if matches!(instance, 0x46A | 0x46B | 0x6E2 | 0x6E3) => 5,
        0xF01E if matches!(instance, 0x6E0 | 0x6E1) => 6,
        _ => return None,
    };
    if kind != expected || body.get(..16)? != &header[2..18] {
        return None;
    }
    Some(record)
}

fn picture_shapes(
    data: &[u8],
    depth: usize,
    supported_group: bool,
    visited: &mut usize,
    shapes: &mut HashMap<u32, Option<u32>>,
) -> Result<bool, ConvertError> {
    if depth > 16 {
        return Err(ConvertError::ResourceLimit {
            limit: "max_drawing_depth",
            detail: "Word pictures exceed 16 container levels".into(),
        });
    }
    let mut cursor = 0;
    while cursor < data.len() {
        objects::count_drawing_record(visited)?;
        let Some((version, kind, body)) = record_at(data, cursor) else { return Ok(false) };
        match kind {
            0xF004 if version & 0xF == 0xF => {
                let Some((id, flags)) = objects::shape_identity(body, visited)? else {
                    return Ok(false);
                };
                let pib = if supported_group && flags & 0x209 == 0x200 {
                    picture_index(body)
                } else {
                    None
                };
                // Duplicate identities are ambiguous even if one is deleted.
                if shapes.insert(id, pib).is_some() {
                    shapes.insert(id, None);
                }
            }
            0xF003 if version & 0xF == 0xF => {
                let Some((_, 0xF004, group)) = record_at(body, 0) else { return Ok(false) };
                let Some((_, flags)) = objects::shape_identity(group, visited)? else {
                    return Ok(false);
                };
                let supported = supported_group && flags & 0xD == 0x5;
                if !picture_shapes(body, depth + 1, supported, visited, shapes)? {
                    return Ok(false);
                }
            }
            _ => {}
        }
        cursor += 8 + body.len();
    }
    Ok(true)
}

fn picture_index(shape: &[u8]) -> Option<u32> {
    let mut cursor = 0;
    let mut pib = None;
    let mut seen_fopt = false;
    while cursor < shape.len() {
        let (version, kind, body) = record_at(shape, cursor)?;
        if kind == 0xF00B {
            if version & 0xF != 3 || seen_fopt {
                return None;
            }
            seen_fopt = true;
            let count = (version >> 4) as usize;
            let properties = body.get(..count * 6)?;
            let mut complex_size = 0usize;
            for property in properties.chunks_exact(6) {
                let opid = get_u16(property, 0)?;
                let value = get_u32(property, 2)?;
                if opid & 0x8000 != 0 {
                    complex_size = complex_size.checked_add(value as usize)?;
                }
                if opid & 0x3FFF == 0x104 {
                    if opid != 0x4104 || value == 0 || pib.is_some() {
                        return None;
                    }
                    pib = Some(value);
                }
            }
            if complex_size != body.len() - properties.len() {
                return None;
            }
        }
        cursor += 8 + body.len();
    }
    pib
}

/// Metafiles must decode completely within their declared size, never return
/// a truncated prefix. Bitmap records are bounded before reaching this point.
fn decode_picture(version: u16, kind: u16, body: &[u8]) -> Option<Blip<'_>> {
    let max = limits::MAX_ENTRY_BYTES as usize;
    if body.len() > max || version & 0xF != 0 {
        return None;
    }
    let instance = version >> 4;
    match kind {
        0xF01D if matches!(instance, 0x46A | 0x46B | 0x6E2 | 0x6E3) => {
            decode_blip(version, kind, body, max)
        }
        0xF01E if matches!(instance, 0x6E0 | 0x6E1) => decode_blip(version, kind, body, max),
        0xF01A | 0xF01B => {
            let doubled = match (kind, instance) {
                (0xF01A, 0x3D4) | (0xF01B, 0x216) => false,
                (0xF01A, 0x3D5) | (0xF01B, 0x217) => true,
                _ => return None,
            };
            let header_offset = if doubled { 32 } else { 16 };
            let header = body.get(header_offset..)?;
            let size = get_u32(header, 0)? as usize;
            let saved = get_u32(header, 28)? as usize;
            let data = header.get(34..)?;
            if size > max || data.len() != saved || header[33] != 0xFE {
                return None;
            }
            let bytes = match header[32] {
                0 => {
                    // MS-ODRAW compression 0 is RFC1950 (zlib), not raw deflate.
                    let mut decoder = flate2::Decompress::new(true);
                    let mut decoded = Vec::with_capacity(size + 1);
                    let status = decoder
                        .decompress_vec(data, &mut decoded, flate2::FlushDecompress::Finish)
                        .ok()?;
                    if status != flate2::Status::StreamEnd
                        || decoded.len() != size
                        || decoder.total_in() != data.len() as u64
                    {
                        return None;
                    }
                    Cow::Owned(decoded)
                }
                0xFE if data.len() == size => Cow::Borrowed(data),
                _ => return None,
            };
            let (media_type, extension) =
                if kind == 0xF01A { ("image/emf", "emf") } else { ("image/wmf", "wmf") };
            Some(Blip { media_type, extension, bytes })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn metafile() -> Vec<u8> {
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(b"SYNTHETIC EMF").unwrap();
        let compressed = encoder.finish().unwrap();
        let mut body = vec![0; 16 + 34];
        body[16..20].copy_from_slice(&13u32.to_le_bytes());
        body[44..48].copy_from_slice(&(compressed.len() as u32).to_le_bytes());
        body[49] = 0xFE;
        body.extend(compressed);
        body
    }

    #[test]
    fn metafile_requires_a_complete_zlib_stream_and_exact_bounded_sizes() {
        let body = metafile();
        assert_eq!(decode_picture(0x3D40, 0xF01A, &body).unwrap().bytes.as_ref(), b"SYNTHETIC EMF");
        for size in [0, 12, 14, u32::MAX] {
            let mut bad = body.clone();
            bad[16..20].copy_from_slice(&size.to_le_bytes());
            assert!(decode_picture(0x3D40, 0xF01A, &bad).is_none());
        }
        let mut truncated = body.clone();
        truncated.pop();
        let saved = (truncated.len() - 50) as u32;
        truncated[44..48].copy_from_slice(&saved.to_le_bytes());
        assert!(decode_picture(0x3D40, 0xF01A, &truncated).is_none());
        let mut trailing = body;
        trailing.push(0);
        let saved = (trailing.len() - 50) as u32;
        trailing[44..48].copy_from_slice(&saved.to_le_bytes());
        assert!(decode_picture(0x3D40, 0xF01A, &trailing).is_none());
    }
}
