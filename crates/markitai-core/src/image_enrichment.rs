//! Optional image analysis uses owned image bytes and real document references.

mod resources;

use crate::{
    ConversionOutput, Document, Error, LlmRuntime, Result, config, images, llm, output_profiles,
};
use chrono::Local;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub(crate) fn enabled(source: &str, cfg: &Value) -> bool {
    let standalone = !crate::is_url(source)
        && Path::new(source)
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(images::is_image_extension);
    config::enabled(cfg, "/llm/enabled")
        && (config::enabled(cfg, "/image/alt_enabled")
            || config::enabled(cfg, "/image/desc_enabled"))
        && !(config::enabled(cfg, "/llm/pure") && !crate::is_url(source) && !standalone)
}

/// Run before shared embedded-image filtering/compression, so downloaded and
/// already-owned images obey the same policy and final reference spelling.
/// With an output directory, inline data images always become owned assets,
/// as in the reference workflow; without one they stay self-contained rather
/// than naming unwritten files. Other references are localized only for image
/// enrichment.
pub(crate) fn prepare(
    doc: &mut Document,
    source: &str,
    cfg: &Value,
    persistent: bool,
) -> Result<()> {
    let scope = if enabled(source, cfg) {
        resources::Scope::All
    } else if persistent && doc.markdown.contains("data:image/") {
        resources::Scope::Data
    } else {
        return Ok(());
    };
    resources::prepare(doc, source, cfg, scope)
}

/// Run after document enhancement and before profile/publication transforms.
/// Public `asset` paths still name owned assets; publication resolves them once.
pub(crate) fn analyze(
    doc: &Document,
    result: &mut ConversionOutput,
    source: &str,
    standalone: bool,
    cfg: &Value,
    runtime: Option<&LlmRuntime>,
) -> Result<()> {
    if !enabled(source, cfg) {
        return Ok(());
    }
    if !standalone && result.llm_markdown.is_none() {
        return Ok(());
    }
    let body = result.llm_markdown.as_deref().unwrap_or(&doc.markdown);
    let (_, body_context) = crate::output::split_frontmatter(body);
    let context: String = body_context.chars().take(200).collect();
    let references = output_profiles::image_references(&doc.markdown);
    let referenced: HashSet<_> = references
        .iter()
        .filter_map(|target| resources::asset_name(target))
        .collect();
    let mut responses: HashMap<String, std::result::Result<Value, String>> = HashMap::new();
    let mut captions = HashMap::new();
    let mut first_answer = None;
    let selected: Vec<_> = doc
        .assets
        .iter()
        .filter(|asset| referenced.contains(&asset.name))
        .collect();
    // Multipage TIFF previews are one standalone image document. Analyze all
    // previews together rather than discarding the later pages' descriptions.
    let standalone_answer = if standalone && !selected.is_empty() {
        Some(analyze_assets(&selected, &context, source, cfg, runtime)?)
    } else {
        None
    };
    let mut attempted = 0;
    for (index, asset) in selected.iter().enumerate() {
        let digest = crate::hex(Sha256::digest(&asset.bytes));
        let answer = if let Some(answer) = &standalone_answer {
            let mut answer = answer.clone();
            if index > 0 {
                answer["llm_usage"] = json!({});
            }
            Ok(answer)
        } else if let Some(cached) = responses.get(&digest) {
            cached.clone().map(|mut entry| {
                entry["llm_usage"] = json!({});
                entry
            })
        } else {
            attempted += 1;
            let answer = analyze_assets(&[*asset], &context, source, cfg, runtime)
                .map_err(|error| error.to_string());
            responses.insert(digest, answer.clone());
            answer
        };
        match answer {
            Ok(mut answer) => {
                let target = format!(".markitai/assets/{}", asset.name);
                for reference in &references {
                    if resources::asset_name(reference).as_deref() == Some(asset.name.as_str()) {
                        captions.insert(reference.clone(), answer["alt"].as_str().unwrap_or("Image").to_owned());
                    }
                }
                if first_answer.is_none() { first_answer = Some(answer.clone()); }
                answer["asset"] = json!(target);
                if !(standalone && config::enabled(cfg, "/llm/pure")) { result.images.push(answer); }
            }
            Err(message) => result.warnings.push(format!("Image analysis failed for {}: {message}; existing alt text was retained and no analysis entry was produced.", asset.name)),
        }
    }
    if standalone {
        let Some(answer) = first_answer else {
            return Err(Error::Conversion(format!(
                "Standalone image analysis failed ({attempted} distinct image request plans); base image reference was retained"
            )));
        };
        let stem = Path::new(source)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let caption = answer["alt"].as_str().unwrap_or("Image");
        let description = answer["desc"].as_str().unwrap_or("");
        let text = answer["text"].as_str().unwrap_or("");
        let mut markdown = format!("# {}\n\n", stem.replace(['\n', '\r'], " "));
        if config::enabled(cfg, "/llm/pure") {
            if !description.trim().is_empty() {
                markdown.push_str(description.trim());
                markdown.push_str("\n\n");
            }
            if !text.trim().is_empty() {
                markdown.push_str(text.trim());
                markdown.push('\n');
            }
        } else {
            let preview = output_profiles::replace_image_alts(&doc.markdown, &captions);
            let heading = format!("# {}", stem.replace(['\n', '\r'], " "));
            let preview = preview
                .strip_prefix(&heading)
                .filter(|tail| tail.starts_with('\n'))
                .unwrap_or(&preview);
            markdown.push_str(preview.trim());
            markdown.push_str("\n\n");
            if !description.trim().is_empty() {
                if !description.trim_start().starts_with('#') {
                    markdown.push_str("## Image Description\n\n");
                }
                markdown.push_str(description.trim());
                markdown.push_str("\n\n");
            }
            if !text.trim().is_empty() {
                markdown.push_str("## Extracted Text\n\n");
                let fence = "`".repeat(longest_ticks(text).max(2) + 1);
                markdown.push_str(&fence);
                markdown.push('\n');
                markdown.push_str(text);
                if !text.ends_with('\n') {
                    markdown.push('\n');
                }
                markdown.push_str(&fence);
                markdown.push('\n');
            }
            result.frontmatter.insert("title".into(), json!(stem));
            result
                .frontmatter
                .insert("description".into(), json!(caption));
            result
                .frontmatter
                .insert("tags".into(), json!(["image", "analysis"]));
        }
        result.llm_markdown = Some(markdown);
    } else if config::enabled(cfg, "/image/alt_enabled")
        && let Some(markdown) = &mut result.llm_markdown
    {
        *markdown = output_profiles::replace_image_alts(markdown, &captions);
    }
    Ok(())
}

fn analyze_assets(
    assets: &[&crate::Asset],
    context: &str,
    source: &str,
    cfg: &Value,
    runtime: Option<&LlmRuntime>,
) -> Result<Value> {
    let mut frames = Vec::new();
    let mut total = 0usize;
    let cap = cfg
        .pointer("/llm/max_vision_pages_per_document")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    for asset in assets {
        let prepared = images::prepare_vision(&asset.bytes, &asset.name, cfg)?;
        for frame in prepared {
            total = total
                .checked_add(frame.bytes.len())
                .ok_or_else(|| Error::InvalidInput("Image analysis payload overflow".into()))?;
            if total > 100 * 1024 * 1024 || cap > 0 && frames.len() as u64 >= cap {
                return Err(Error::InvalidInput(
                    "Image analysis exceeds the document page or 100 MiB payload budget".into(),
                ));
            }
            frames.push(frame);
        }
    }
    let input: Vec<_> = frames
        .iter()
        .map(|frame| (frame.mime, frame.bytes.as_slice()))
        .collect();
    let answer = llm::analyze_images_with_runtime(
        context,
        &crate::output::redact_url(source),
        &input,
        cfg,
        runtime,
    )?;
    Ok(
        json!({"alt":answer.caption,"desc":answer.description,"text":answer.extracted_text,
        "llm_usage":answer.usage.by_model,"created":Local::now().to_rfc3339()}),
    )
}

fn longest_ticks(text: &str) -> usize {
    let mut longest = 0;
    let mut run = 0;
    for ch in text.chars() {
        if ch == '`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    longest
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pure_nonimage_local_preserves_reference_priority_but_url_images_remain_eligible() {
        let cfg = json!({"llm":{"enabled":true,"pure":true},"image":{"alt_enabled":true}});
        assert!(!enabled("report.md", &cfg));
        assert!(enabled("photo.png", &cfg));
        assert!(enabled("https://example.test/page", &cfg));
        assert!(!enabled(
            "photo.png",
            &json!({"llm":{"enabled":false},"image":{"alt_enabled":true}})
        ));
    }
    #[test]
    fn extracted_text_fence_can_retain_embedded_fences() {
        assert_eq!(longest_ticks("one ` two\n```\nfour ```` and end"), 4);
    }
}
