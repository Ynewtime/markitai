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
    let llm = config::enabled(cfg, "/llm/enabled");
    // `--ocr` adds no OCR of the rendered pages, as in Python markitai: they
    // show the document's own text. With a vision model it still sends them.
    let vlm_ocr = config::enabled(cfg, "/ocr/enabled") && llm && !vlm_disabled;
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
        &exported.screenshot_min_widths_pt,
    )?;
    if let Some(dir) = output_dir {
        output::publish_page_screenshots(dir, &mut media.screenshots, cfg)?;
    }
    document.warnings.extend(exported.warnings);
    document.warnings.extend(media.warnings);
    if config::enabled(cfg, "/ocr/enabled") && llm && vlm_disabled {
        document.warnings.push(if screenshots && (!config::enabled(cfg, "/llm/pure")
            || config::enabled(cfg, "/screenshot/screenshot_only")) {
            "VLM OCR is disabled; explicitly requested screenshot enhancement still sends page images to the model. Disable screenshots to keep page images local."
        } else {
            "VLM OCR is disabled; only the document's own text is used for Office LLM enhancement."
        }.into());
    }
    let has_reliable_text = !document.markdown.trim().is_empty();
    if vlm_ocr {
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
