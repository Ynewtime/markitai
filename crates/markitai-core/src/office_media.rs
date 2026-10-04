//! Keep native Office text and add complete rendered-page evidence as an appendix.

use crate::{Asset, Document, Error, Result, config, office_render, output, pdf_media};
use office_render::OfficeKind;
use serde_json::{Value, json};
use std::path::Path;

pub(crate) fn reference(kind: OfficeKind, page: usize, name: &str) -> String {
    let reference = crate::formats::pdf_screenshot_reference(page, name);
    if kind == OfficeKind::Presentation {
        reference.replacen("![Page ", "![Slide ", 1)
    } else {
        reference
    }
}

pub(crate) fn prepare(
    document: &mut Document,
    path: &Path,
    kind: OfficeKind,
    prefix: &str,
    output_dir: Option<&Path>,
    cfg: &Value,
    vlm_disabled: bool,
) -> Result<(Vec<Asset>, bool)> {
    if !crate::pdf_raster::available() {
        return Err(Error::Unsupported(
            "Office page capture requires an available native PDF page renderer on this platform"
                .into(),
        ));
    }
    let llm = config::enabled(cfg, "/llm/enabled");
    let ocr = config::enabled(cfg, "/ocr/enabled");
    let local_ocr = ocr && (!llm || vlm_disabled);
    let vlm_ocr = ocr && llm && !vlm_disabled;
    let screenshots = config::enabled(cfg, "/screenshot/enabled") || vlm_ocr;
    if screenshots
        && !llm
        && output_dir.is_none()
        && config::enabled(cfg, "/screenshot/screenshot_only")
    {
        return Err(Error::InvalidInput(
            "Screenshot-only conversion without LLM requires output_dir to retain captured images"
                .into(),
        ));
    }
    let exported = office_render::export_pdf(path, kind)?;
    let mut media = pdf_media::capture_external_pdf(
        &exported.bytes,
        exported.pages,
        prefix,
        cfg,
        screenshots,
        local_ocr,
        &exported.screenshot_min_widths_pt,
    )?;
    if let Some(dir) = output_dir {
        output::publish_page_screenshots(dir, &mut media.screenshots, cfg)?;
    }
    document.warnings.extend(exported.warnings);
    document.warnings.extend(media.warnings);
    if local_ocr && llm && vlm_disabled {
        document.warnings.push(if screenshots && (!config::enabled(cfg, "/llm/pure")
            || config::enabled(cfg, "/screenshot/screenshot_only")) {
            "VLM OCR is disabled; OCR stays local, but explicitly requested screenshot enhancement still sends page images to the model. Disable screenshots to keep page images local."
        } else {
            "VLM OCR is disabled; only locally extracted text is used for Office LLM enhancement."
        }.into());
    }
    let has_reliable_text = !document.markdown.trim().is_empty()
        || media.ocr.iter().any(|text| !text.trim().is_empty());
    let label = if kind == OfficeKind::Presentation {
        "Slide"
    } else {
        "Page"
    };
    if local_ocr {
        let nonempty = media
            .ocr
            .iter()
            .filter(|text| !text.trim().is_empty())
            .count();
        document
            .metadata
            .insert("ocr_pages_attempted".into(), json!(media.ocr.len()));
        document
            .metadata
            .insert("ocr_pages_nonempty".into(), json!(nonempty));
        document
            .metadata
            .insert("ocr_pages_blank".into(), json!(media.ocr.len() - nonempty));
        document.metadata.insert("ocr_used".into(), json!(true));
        document
            .metadata
            .insert("ocr_path".into(), json!(crate::ocr::backend()));
        if nonempty > 0 {
            document
                .markdown
                .push_str("\n\n## Rendered-page OCR supplement\n");
            for (index, text) in media.ocr.iter().enumerate() {
                if !text.trim().is_empty() {
                    document.markdown.push_str(&format!(
                        "\n### {label} {}\n\n{}\n",
                        index + 1,
                        text.trim()
                    ));
                }
            }
        }
    } else if vlm_ocr {
        document.metadata.insert("ocr_path".into(), json!("vlm"));
    }
    document
        .metadata
        .insert("rendered_pages".into(), json!(exported.pages));
    if !media.screenshots.is_empty() {
        document
            .markdown
            .push_str("\n\n<!-- Rendered Office pages for reference -->\n");
        for (index, shot) in media.screenshots.iter().enumerate() {
            document
                .markdown
                .push_str(&reference(kind, index + 1, &shot.name));
            document.markdown.push('\n');
        }
    }
    Ok((media.screenshots, has_reliable_text))
}
