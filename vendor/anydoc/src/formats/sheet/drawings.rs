//! Sheet-level figures, read only through the visible sheet's drawing
//! references. Figures follow that sheet's grid; they are not cell values.

use super::xlsx::{MAX_COLS, MAX_ROWS, SheetContent};
use crate::error::ConvertError;
use crate::formats::ppt::pictures::{self, Bank};
use crate::model::{ImageSource, Inline};
use crate::package::relationships::{Relationships, TargetMode, read_rels, rels_part_for};
use crate::package::xml::{Element, ns};
use crate::package::{Package, limits, path};
use crate::shared::assets::{AssetSink, media_type_for};

const XDR: &str = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing";
const DRAWING: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing";
const IMAGE: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image";

fn hidden_cell(content: &SheetContent, row: u32, col: u32) -> bool {
    content.hidden_rows.contains(&row)
        || content.hidden_cols.iter().any(|&(a, b)| a <= col && col <= b)
}

pub(super) fn xml(
    pkg: &mut Package,
    worksheet: &Element,
    part: &str,
    rels: &Relationships,
    content: &SheetContent,
    assets: &mut AssetSink,
    warnings: &mut Vec<String>,
) -> Result<Vec<Inline>, ConvertError> {
    let mut images = Vec::new();
    for drawing in worksheet.find_all(ns::SML, "drawing") {
        let Some(rel) = drawing.attr_qualified(ns::R, "id").and_then(|id| rels.get(id)) else {
            continue;
        };
        if rel.rel_type != DRAWING || rel.mode != TargetMode::Internal {
            continue;
        }
        let Ok(target) = path::resolve(part, &rel.target) else {
            warnings
                .push("A worksheet drawing has an invalid package path and was omitted.".into());
            continue;
        };
        let Some(root) = pkg.optional_xml_part(&target.path)? else {
            continue;
        };
        let Some(root) = root.find(XDR, "wsDr") else {
            continue;
        };
        let drawing_rels = read_rels(pkg, &rels_part_for(&target.path))?;
        for anchor in root.child_elems() {
            if anchor.ns.as_deref() != Some(XDR) {
                continue;
            }
            let positioned = match anchor.local.as_str() {
                "oneCellAnchor" | "twoCellAnchor" => anchor
                    .find(XDR, "from")
                    .and_then(|from| {
                        Some((
                            from.find(XDR, "row")?.text().parse::<u32>().ok()?,
                            from.find(XDR, "col")?.text().parse::<u32>().ok()?,
                        ))
                    })
                    .is_some_and(|(row, col)| {
                        row < MAX_ROWS && col < MAX_COLS && !hidden_cell(content, row, col)
                    }),
                "absoluteAnchor" => true,
                _ => false,
            };
            if !positioned {
                continue;
            }
            // Only picture/group children of this anchor. The blip in an
            // extension or a chart's cache is not a displayed picture.
            let mut stack: Vec<(&Element, bool)> = anchor
                .child_elems()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|e| (e, false))
                .collect();
            while let Some((shape, inherited_hidden)) = stack.pop() {
                if shape.ns.as_deref() != Some(XDR) {
                    continue;
                }
                let nv = match shape.local.as_str() {
                    "pic" => shape.find(XDR, "nvPicPr"),
                    "grpSp" => shape.find(XDR, "nvGrpSpPr"),
                    _ => continue,
                };
                let prop = nv.and_then(|nv| nv.find(XDR, "cNvPr"));
                let hidden = inherited_hidden
                    || prop
                        .and_then(|p| p.attr_unqualified("hidden"))
                        .is_some_and(|v| matches!(v, "1" | "true" | "on"));
                if hidden {
                    continue;
                }
                if shape.is(XDR, "grpSp") {
                    for child in shape.child_elems().collect::<Vec<_>>().into_iter().rev() {
                        stack.push((child, hidden));
                    }
                    continue;
                }
                let Some(blip) =
                    shape.find(XDR, "blipFill").and_then(|fill| fill.find(ns::A, "blip"))
                else {
                    continue;
                };
                let rid = blip
                    .attr_qualified(ns::R, "embed")
                    .or_else(|| blip.attr_qualified(ns::R, "link"));
                let Some(rel) =
                    rid.and_then(|id| drawing_rels.get(id)).filter(|r| r.rel_type == IMAGE)
                else {
                    continue;
                };
                let source = if rel.mode == TargetMode::External {
                    if rel.target.is_empty() {
                        continue;
                    }
                    ImageSource::External(rel.target.clone())
                } else {
                    let Ok(target) = path::resolve(&target.path, &rel.target) else {
                        warnings.push(
                            "A worksheet image has an invalid package path and was omitted.".into(),
                        );
                        continue;
                    };
                    let Some(bytes) = pkg.optional_part(&target.path)? else {
                        continue;
                    };
                    ImageSource::Asset(assets.add(
                        media_type_for(&target.path),
                        target.path,
                        &bytes,
                    )?)
                };
                let alt = prop
                    .and_then(|p| {
                        p.attr_unqualified("descr").or_else(|| p.attr_unqualified("name"))
                    })
                    .unwrap_or_default()
                    .to_string();
                images.push(Inline::Image { alt, source });
            }
        }
    }
    Ok(images)
}

pub(super) fn binary(
    data: &[u8],
    content: &SheetContent,
    bank: &mut Bank,
) -> Result<Vec<Inline>, ConvertError> {
    let mut images = Vec::new();
    let mut stack = vec![(data, 0, false)];
    let mut visited = 0;
    while let Some((body, off, hidden)) = stack.last_mut() {
        if *off == body.len() {
            stack.pop();
            continue;
        }
        pictures::charge(&mut visited)?;
        let Some((vi, kind, child)) = crate::shared::officeart::record_at(body, *off) else {
            bank.warn(
                "A worksheet drawing is truncated; its unresolved figures were omitted.".into(),
            );
            break;
        };
        *off += 8 + child.len();
        let hidden = *hidden;
        if kind == 0xF004 {
            let shape = pictures::shape(child);
            if !hidden
                && !shape.hidden
                && let Some((row, col)) = shape.anchor
                && row < 65_536
                && col < 256
                && !hidden_cell(content, row, col)
                && let Some(pib) = shape.pib
                && let Some(image) = bank.image(pib)?
            {
                images.push(image);
            }
        } else if vi & 15 == 15 {
            if stack.len() >= limits::MAX_RECORD_DEPTH {
                return Err(pictures::depth_limit());
            }
            stack.push((child, 0, hidden || (kind == 0xF003 && pictures::group_hidden(child))));
        }
    }
    Ok(images)
}

#[cfg(test)]
#[path = "drawings_tests.rs"]
mod tests;
