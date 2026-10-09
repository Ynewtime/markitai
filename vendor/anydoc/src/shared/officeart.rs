//! OfficeArt (MS-ODRAW) blip extraction shared by the legacy binary
//! formats: DOC picture data (`PICF` + OfficeArt records in the Data
//! stream) and the PPT `Pictures` stream (a sequence of BStore file
//! blocks). Only the picture payloads are extracted; drawing geometry is
//! out of scope.

use crate::package::limits;
use crate::shared::binary::get_u32;
use std::borrow::Cow;
use std::io::Read;

/// (verAndInstance, recType, body) of the OfficeArt record at `off` — the
/// same 8-byte header the PPT record stream uses.
pub(crate) fn record_at(data: &[u8], off: usize) -> Option<(u16, u16, &[u8])> {
    let rest = data.get(off..)?;
    let hdr = rest.get(..8)?;
    let ver_inst = u16::from_le_bytes([hdr[0], hdr[1]]);
    let rec_type = u16::from_le_bytes([hdr[2], hdr[3]]);
    let len = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let body = rest.get(8..)?.get(..len)?;
    Some((ver_inst, rec_type, body))
}

/// An extracted picture payload.
pub(crate) struct Blip<'a> {
    pub(crate) media_type: &'static str,
    pub(crate) extension: &'static str,
    pub(crate) bytes: Cow<'a, [u8]>,
}

/// Decode one blip record (`recType` 0xF01A–0xF01F). Metafile blips may be
/// deflate-compressed; the output is bounded by the declared uncompressed
/// size, capped at `max_bytes`.
pub(crate) fn decode_blip(
    ver_inst: u16,
    rec_type: u16,
    body: &[u8],
    max_bytes: usize,
) -> Option<Blip<'_>> {
    let instance = ver_inst >> 4;
    match rec_type {
        // Bitmap blips: rgbUid1 (16), + rgbUid2 (16) for the doubled
        // instance, then the picture bytes, with one tag byte first.
        0xF01D | 0xF01E => {
            let doubled = matches!(instance, 0x46B | 0x6E3 | 0x6E1);
            let start = if doubled { 32 } else { 16 } + 1;
            let bytes = body.get(start..)?;
            let (media_type, extension) =
                if rec_type == 0xF01D { ("image/jpeg", "jpg") } else { ("image/png", "png") };
            Some(Blip { media_type, extension, bytes: Cow::Borrowed(bytes) })
        }
        // Metafile blips: rgbUid (16/32), then a 34-byte metafile header
        // (cbSize, bounds, ptSize, cbSave, compression, filter).
        0xF01A | 0xF01B => {
            let doubled = matches!(instance, 0x3D5 | 0x217);
            let header = if doubled { 32 } else { 16 };
            let header_and_data = body.get(header..)?;
            let cb_size = u32::from_le_bytes(header_and_data.get(..4)?.try_into().ok()?);
            let compression = *header_and_data.get(32)?;
            let data = header_and_data.get(34..)?;
            let (media_type, extension) =
                if rec_type == 0xF01A { ("image/emf", "emf") } else { ("image/wmf", "wmf") };
            let bytes = match compression {
                // 0x00 = deflate-compressed; 0xFE = uncompressed.
                0x00 => {
                    let limit = (cb_size as usize).min(max_bytes);
                    let mut out = Vec::new();
                    let mut decoder = flate2::read::DeflateDecoder::new(data).take(limit as u64);
                    decoder.read_to_end(&mut out).ok()?;
                    Cow::Owned(out)
                }
                _ => Cow::Borrowed(data),
            };
            Some(Blip { media_type, extension, bytes })
        }
        _ => None,
    }
}

/// Find and decode the first blip in a run of OfficeArt records (a
/// `Pictures` stream block sequence, or an inline shape container),
/// descending into containers. Bounded traversal: record counts and
/// nesting beyond any real drawing abort the search.
pub(crate) fn first_blip(data: &[u8], max_bytes: usize) -> Option<Blip<'_>> {
    // (cursor, end) ranges into `data`.
    let mut stack: Vec<(usize, usize)> = vec![(0, data.len())];
    let mut visited = 0u32;
    loop {
        let &(cursor, end) = stack.last()?;
        if cursor >= end {
            stack.pop();
            continue;
        }
        let Some((ver_inst, rec_type, body)) = record_at(&data[..end], cursor) else {
            stack.pop();
            continue;
        };
        let body_start = cursor.checked_add(8)?;
        let body_end = body_start.checked_add(body.len())?;
        stack.last_mut().unwrap().0 = body_end;
        visited += 1;
        if visited > 10_000 || stack.len() > 16 {
            return None;
        }
        if let Some(blip) = decode_blip(ver_inst, rec_type, body, max_bytes) {
            return Some(blip);
        }
        if rec_type == 0xF007 {
            let inner_start =
                body_start.checked_add(fbse_blip_offset(body).unwrap_or(body.len()))?;
            if inner_start < body_end {
                stack.push((inner_start, body_end));
            }
            continue;
        }
        if ver_inst & 0xF == 0xF {
            stack.push((body_start, body_end));
        }
    }
}

/// Offset of the embedded blip record inside an FBSE (0xF007) body: the
/// 36-byte header plus the entry's name.
fn fbse_blip_offset(body: &[u8]) -> Option<usize> {
    let cb_name = *body.get(33)? as usize;
    36usize.checked_add(cb_name)
}

/// markitai: a JPEG, PNG, EMF or WMF blip decoded completely, never a
/// truncated prefix: metafiles must inflate (zlib, MS-ODRAW compression 0)
/// or be stored to exactly their declared size. Shared by the DOC and PPT
/// picture readers; the record is bounded by `MAX_ENTRY_BYTES`.
pub(crate) fn complete_blip(version: u16, kind: u16, body: &[u8]) -> Option<Blip<'_>> {
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
