//! Repair measured right-edge overflow only in a private LibreOffice workbook export.
//! Never called for an input PDF; all authored cell clips remain LibreOffice's job.
use super::{Result, failure};
use std::time::Instant;
mod anchor;
mod package;
#[cfg(test)]
mod tests;
mod xml;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Extension {
    pub page: usize,
    pub width: f64,
    pub height: f64,
    pub right: f64,
}
#[derive(Default)]
pub(super) struct Plan {
    pub extensions: Vec<Extension>,
    pub warnings: Vec<String>,
    uncertain_pages: Vec<usize>,
}
pub(super) fn check_deadline(deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        Err(failure("Office export timed out"))
    } else {
        Ok(())
    }
}
fn dimensions(width: f64, height: f64) -> Result<()> {
    let w = (width * 150. / 72.).ceil();
    let h = (height * 150. / 72.).ceil();
    if !w.is_finite()
        || !h.is_finite()
        || w < 1.
        || h < 1.
        || w > 65535.
        || h > 65535.
        || w * h > 32_000_000.
    {
        return Err(failure(
            "workbook page exceeds native 150 DPI page-pixel limits",
        ));
    }
    Ok(())
}
fn inherited<'a>(
    pdf: &'a lopdf::Document,
    id: lopdf::ObjectId,
    key: &[u8],
) -> Option<&'a lopdf::Object> {
    let mut current = id;
    for _ in 0..128 {
        let dictionary = pdf.get_dictionary(current).ok()?;
        if let Ok(value) = dictionary.get(key) {
            return Some(value);
        }
        current = dictionary.get(b"Parent").ok()?.as_reference().ok()?;
    }
    None
}
fn page_box(pdf: &lopdf::Document, id: lopdf::ObjectId) -> Result<Option<(f64, f64)>> {
    let Some(media) = inherited(pdf, id, b"MediaBox") else {
        return Err(failure("workbook PDF page has no MediaBox"));
    };
    let media = media
        .as_array()
        .map_err(|_| failure("invalid workbook PDF page box"))?;
    let values = media
        .iter()
        .map(|v| v.as_float().map(f64::from))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| failure("invalid workbook PDF page box"))?;
    if values.len() != 4 || values.iter().any(|v| !v.is_finite()) {
        return Err(failure("invalid workbook PDF page box"));
    }
    let width = values[2] - values[0];
    let height = values[3] - values[1];
    dimensions(width, height)?;
    if let Some(crop) = inherited(pdf, id, b"CropBox")
        && crop != &lopdf::Object::Array(media.clone())
    {
        return Ok(None);
    }
    let rotated = inherited(pdf, id, b"Rotate")
        .and_then(|v| v.as_i64().ok())
        .unwrap_or(0)
        .rem_euclid(360)
        != 0;
    let unit = inherited(pdf, id, b"UserUnit")
        .and_then(|v| v.as_float().ok())
        .unwrap_or(1.);
    if rotated || unit != 1. || values[0] != 0. || values[1] != 0. {
        return Ok(None);
    }
    Ok(Some((width, height)))
}
/// Empty plans preserve the original PDF bytes, including in occupied-neighbor cases.
pub(super) fn inspect(bytes: &[u8], expected_pages: usize, deadline: Instant) -> Result<Plan> {
    check_deadline(deadline)?;
    let pdf = lopdf::Document::load_mem(bytes)
        .map_err(|_| failure("invalid workbook PDF for overflow measurement"))?;
    let pages = pdf.get_pages();
    if pages.is_empty() || pages.len() != expected_pages || pages.len() > super::MAX_PAGES {
        return Err(failure("workbook overflow measurement page count mismatch"));
    }
    let boxes = pages
        .values()
        .map(|id| page_box(&pdf, *id))
        .collect::<Result<Vec<_>>>()?;
    let items = pdf_inspector::extract_text_with_positions_mem(bytes)
        .map_err(|_| failure("workbook text bounds could not be measured"))?;
    check_deadline(deadline)?;
    let mut right: Vec<f64> = boxes.iter().map(|v| v.map_or(0., |b| b.0)).collect();
    let mut uncertain = boxes.iter().map(Option::is_none).collect::<Vec<_>>();
    for item in &items {
        check_deadline(deadline)?;
        if item.text.trim().is_empty() || matches!(item.render_mode, Some(3 | 7)) {
            continue;
        }
        let Some(index) = item
            .page
            .checked_sub(1)
            .map(|v| v as usize)
            .filter(|v| *v < boxes.len())
        else {
            return Err(failure("invalid workbook positioned-text page"));
        };
        let Some((width, height)) = boxes[index] else {
            uncertain[index] = true;
            continue;
        };
        let (x, y, w, h) = (
            f64::from(item.x),
            f64::from(item.y),
            f64::from(item.width),
            f64::from(item.height),
        );
        if [x, y, w, h].iter().any(|v| !v.is_finite()) || w < 0. || h < 0. {
            return Err(failure("non-finite workbook text bounds"));
        }
        // This API gives a horizontal advance and baseline, not rotated glyph
        // outlines or descenders. Do not invent a lower/left extension from it.
        if !item.advance_known
            || item.rotation != 0.
            || x < -0.5
            || y < -0.5
            || y + h > height + 0.5
        {
            uncertain[index] = true;
            continue;
        }
        if x + w > width + 0.5 {
            right[index] = right[index].max(x + w + 2.);
        }
    }
    let mut plan = Plan::default();
    for (index, bounds) in boxes.into_iter().enumerate() {
        if uncertain[index] {
            plan.uncertain_pages.push(index + 1);
            plan.warnings.push(format!("Workbook sheet {} has text geometry that cannot be safely expanded (rotation, lower/left overflow, unmeasured advance, or a nonstandard page box); its original LibreOffice layout is retained", index + 1));
            continue;
        }
        if let Some((width, height)) = bounds
            && right[index] > width
        {
            // Very distant clipped text is not evidence that the user wants
            // a huge viewport. Explicitly reject rather than guess or crop.
            if right[index] > width * 2. {
                return Err(failure(
                    "workbook right-edge text is too distant for bounded layout repair",
                ));
            }
            dimensions(right[index], height)?;
            plan.extensions.push(Extension {
                page: index + 1,
                width,
                height,
                right: right[index],
            });
        }
    }
    Ok(plan)
}
pub(super) fn rewrite(
    bytes: &[u8],
    extensions: &[Extension],
    deadline: Instant,
    limit: u64,
) -> Result<Vec<u8>> {
    package::rewrite(bytes, extensions, deadline, limit)
}

/// An unchanged unsupported sheet must not make another sheet's repair fail.
/// New uncertainty (including on a repaired sheet) is a regression, as is any
/// measured overflow remaining after the only retry.
pub(super) fn verify_repair(before: &Plan, after: &Plan) -> Result<()> {
    if !after.extensions.is_empty()
        || after
            .uncertain_pages
            .iter()
            .any(|page| !before.uncertain_pages.contains(page))
    {
        return Err(failure(
            "workbook overflow remains or text geometry changed after its single bounded layout repair",
        ));
    }
    Ok(())
}
