//! Referenced OfficeArt pictures. Bank ordinals are never compacted: an
//! unsupported or empty slot still occupies its one-based `pib` index.

use crate::error::ConvertError;
use crate::model::{ImageSource, Inline};
use crate::package::limits;
use crate::shared::assets::AssetSink;
use crate::shared::binary::{get_u16, get_u32};
use crate::shared::officeart::{self, record_at};
use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct Bank<'a> {
    slots: Vec<Option<(u16, u16, &'a [u8])>>,
    pub(crate) assets: AssetSink,
    pub(crate) warnings: Vec<String>,
}

impl<'a> Bank<'a> {
    /// The current document's Dgg, with the host's optional delay stream.
    /// FBSE embedded data wins over foDelay. Delay pointers must identify a
    /// complete record boundary, never bytes inside another picture.
    pub(crate) fn read(group: &'a [u8], delay: &'a [u8]) -> Result<Self, ConvertError> {
        let mut bank = Self::default();
        let mut delayed = HashMap::new();
        let mut off = 0;
        let mut visited = 0;
        while off < delay.len() {
            charge(&mut visited)?;
            let Some(rec @ (_, kind, body)) = record_at(delay, off) else {
                bank.warn("A truncated picture delay stream was only partially readable.".into());
                break;
            };
            if kind == 0xF007 || (0xF018..=0xF117).contains(&kind) {
                delayed.insert(off as u32, rec);
            }
            off += 8 + body.len();
        }
        let mut stack = vec![(group, 0)];
        let mut store = None;
        while let Some((data, off)) = stack.last_mut() {
            if *off == data.len() {
                stack.pop();
                continue;
            }
            charge(&mut visited)?;
            let Some((vi, kind, body)) = record_at(data, *off) else {
                bank.warn(
                    "The drawing picture bank is truncated; its unresolved pictures are omitted."
                        .into(),
                );
                return Ok(bank);
            };
            *off += 8 + body.len();
            if kind == 0xF001 && vi & 15 == 15 {
                if store.replace((vi, body)).is_some() {
                    bank.warn("Conflicting drawing picture banks were omitted.".into());
                    return Ok(bank);
                }
            } else if kind == 0xF000 && vi & 15 == 15 {
                if stack.len() >= limits::MAX_RECORD_DEPTH {
                    return Err(depth_limit());
                }
                stack.push((body, 0));
            }
        }
        let Some((vi, body)) = store else {
            return Ok(bank);
        };
        off = 0;
        while off < body.len() {
            charge(&mut visited)?;
            if bank.slots.len() >= 100_000 {
                return Err(ConvertError::ResourceLimit {
                    limit: "max_picture_slots",
                    detail: "picture bank exceeds 100000 slots".into(),
                });
            }
            let Some((entry_vi, kind, entry)) = record_at(body, off) else {
                bank.slots.clear();
                bank.warn(
                    "The picture bank has a truncated slot; bank references are omitted.".into(),
                );
                return Ok(bank);
            };
            off += 8 + entry.len();
            if kind != 0xF007 && !(0xF018..=0xF117).contains(&kind) {
                bank.slots.clear();
                bank.warn("An invalid record in the picture bank prevents ordinal recovery; bank references are omitted.".into());
                return Ok(bank);
            }
            let slot = if kind == 0xF007 {
                fbse_start(entry_vi, entry).and_then(|start| {
                    if get_u32(entry, 24) == Some(0) {
                        None
                    } else if entry.len() > start {
                        // Embedded data has priority even when malformed:
                        // never fall back to a delay pointer in that case.
                        embedded_blip(entry_vi, entry).map(|_| (entry_vi, kind, entry))
                    } else {
                        get_u32(entry, 28)
                            .and_then(|p| delayed.get(&p).copied())
                            .and_then(|rec @ (vi, kind, body)| {
                                if kind == 0xF007 { embedded_blip(vi, body) } else { Some(rec) }
                            })
                            // FBSE.size measures the inner BLIP, including
                            // its 8-byte header, not a delayed FBSE wrapper.
                            .filter(|(_, _, b)| get_u32(entry, 20) == Some((8 + b.len()) as u32))
                    }
                })
            } else {
                // Direct BLIP file blocks also occupy their original
                // rgfb ordinal; only referenced, supported data is decoded.
                Some((entry_vi, kind, entry))
            };
            bank.slots.push(slot);
        }
        if bank.slots.len() != usize::from(vi >> 4) {
            // LibreOffice's PPT writer may understate this count. Every
            // slot above still comes from a complete bounded rgfb record;
            // ordinal references do not depend on the advisory count.
            bank.warn("The picture bank's declared slot count differs from its complete records; referenced figures were recovered by their actual record ordinals.".into());
        }
        Ok(bank)
    }

    pub(crate) fn warn(&mut self, message: String) {
        if !self.warnings.contains(&message) {
            self.warnings.push(message);
        }
    }

    pub(crate) fn image(&mut self, pib: u32) -> Result<Option<Inline>, ConvertError> {
        let slot = pib.checked_sub(1).and_then(|p| self.slots.get(p as usize)).copied().flatten();
        let Some((vi, kind, body)) = slot else {
            self.warn("A drawing refers to an unavailable picture bank slot.".into());
            return Ok(None);
        };
        // The existing decoder's metafile geometry/compression is outside
        // this reader's picture contract. Do not turn a partial decode into
        // an allegedly preserved figure.
        let actual = if kind == 0xF007 { embedded_blip(vi, body) } else { Some((vi, kind, body)) };
        let blip = actual.and_then(|(actual_vi, actual_kind, actual_body)| {
            if actual_vi & 15 != 0
                || !matches!(
                    (actual_kind, actual_vi >> 4),
                    (0xF01D, 0x46A | 0x46B | 0x6E2 | 0x6E3) | (0xF01E, 0x6E0 | 0x6E1)
                )
            {
                return None;
            }
            if kind == 0xF007 {
                officeart::fbse_blip(body, limits::MAX_ENTRY_BYTES as usize)
            } else {
                officeart::decode_blip(
                    actual_vi,
                    actual_kind,
                    actual_body,
                    limits::MAX_ENTRY_BYTES as usize,
                )
            }
        });
        let Some(blip) = blip else {
            self.warn(
                "A referenced OfficeArt picture format is unsupported and was omitted.".into(),
            );
            return Ok(None);
        };
        let id = self.assets.add(
            blip.media_type.into(),
            format!("pictures/{pib}.{}", blip.extension),
            &blip.bytes,
        )?;
        Ok(Some(Inline::Image { alt: String::new(), source: ImageSource::Asset(id) }))
    }
}

/// Validate the FBSE envelope without searching for a picture signature.
fn fbse_start(vi: u16, body: &[u8]) -> Option<usize> {
    if vi & 15 != 2
        || body.len() < 36
        || !(vi >> 4 == u16::from(body[0]) || vi >> 4 == u16::from(body[1]))
    {
        return None;
    }
    let name_len = usize::from(body[33]);
    let start = 36 + name_len;
    if name_len % 2 != 0
        || body.len() < start
        || (name_len > 0 && body.get(start - 2..start)? != [0, 0])
    {
        return None;
    }
    Some(start)
}

fn embedded_blip(vi: u16, body: &[u8]) -> Option<(u16, u16, &[u8])> {
    let start = fbse_start(vi, body)?;
    let rec @ (_, kind, payload) = record_at(body, start)?;
    ((0xF018..=0xF117).contains(&kind)
        && start + 8 + payload.len() == body.len()
        && get_u32(body, 20) == Some((8 + payload.len()) as u32))
    .then_some(rec)
}

#[derive(Default)]
pub(crate) struct Shape {
    pub(crate) pib: Option<u32>,
    pub(crate) hidden: bool,
    pub(crate) anchor: Option<(u32, u32)>,
}

/// Inspect only this shape's own records, not nested textboxes/children.
/// The picture property is a non-complex BLIP id, with a live FSP.
pub(crate) fn shape(body: &[u8]) -> Shape {
    let mut shape = Shape::default();
    let mut off = 0;
    let mut live = None;
    let mut pib_seen = false;
    let mut script_anchor = false;
    let mut really_hidden = false;
    let mut valid = true;
    while off < body.len() {
        let Some((vi, kind, data)) = record_at(body, off) else {
            valid = false;
            break;
        };
        off += 8 + data.len();
        match kind {
            0xF00A => {
                if live.is_some() || vi & 15 != 2 || data.len() != 8 {
                    valid = false;
                }
                live = get_u32(data, 4).map(|flags| flags & 8 == 0);
            }
            0xF00B | 0xF122 => {
                let count = usize::from(vi >> 4);
                if vi & 15 != 3 || data.len() < count * 6 {
                    valid = false;
                    continue;
                }
                for n in 0..count {
                    let opid = get_u16(data, n * 6).unwrap();
                    let value = get_u32(data, n * 6 + 2).unwrap();
                    match opid & 0x3FFF {
                        0x104 if kind == 0xF00B => {
                            if pib_seen || opid & 0xC000 != 0x4000 || value == 0 {
                                valid = false;
                            }
                            pib_seen = true;
                            shape.pib = Some(value);
                        }
                        0x3BF if opid & 0xC000 == 0 => {
                            shape.hidden |= value & 0x0002_0002 == 0x0002_0002;
                            script_anchor |= value & 0x0080_0080 == 0x0080_0080;
                            really_hidden |= value & 0x0100_0100 == 0x0100_0100;
                        }
                        _ => {}
                    }
                }
            }
            0xF010 if data.len() == 18 => {
                shape.anchor = get_u16(data, 6)
                    .zip(get_u16(data, 2))
                    .map(|(r, c)| (u32::from(r), u32::from(c)));
            }
            _ => {}
        }
    }
    // ReallyHidden is meaningful only for an enabled ScriptAnchor;
    // absent use bits select the specification's false defaults.
    shape.hidden |= live != Some(true) || !valid || (script_anchor && really_hidden);
    if shape.hidden {
        shape.pib = None;
    }
    shape
}

/// An Spgr's first SpContainer represents the group, so known hidden or
/// deleted group state also suppresses picture references in descendants.
pub(crate) fn group_hidden(body: &[u8]) -> bool {
    match record_at(body, 0) {
        Some((_, 0xF004, data)) => shape(data).hidden,
        _ => true,
    }
}

pub(crate) fn charge(n: &mut u64) -> Result<(), ConvertError> {
    *n += 1;
    if *n > limits::MAX_RECORDS {
        return Err(ConvertError::ResourceLimit {
            limit: "max_records",
            detail: "drawing traversal exceeds the record-count cap".into(),
        });
    }
    Ok(())
}

pub(crate) fn depth_limit() -> ConvertError {
    ConvertError::ResourceLimit {
        limit: "max_record_depth",
        detail: "drawing containers nested too deeply".into(),
    }
}

#[cfg(test)]
#[path = "pictures_tests.rs"]
mod tests;
