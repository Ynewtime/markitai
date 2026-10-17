//! Optional image analysis uses owned image bytes and real document references.

mod resources;

pub(crate) fn asset_name(target: &str) -> Option<String> {
    resources::asset_name(target)
}

use crate::{
    ConversionOutput, Document, Error, LlmRuntime, Result, config, images, llm, output_profiles,
};
use chrono::Local;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

type Answers = HashMap<String, std::result::Result<Value, String>>;

/// Image analysis of a document, started beside its enhancement: the image
/// prompts read the base Markdown, not the enhanced one. Dropping it stops
/// requests not yet sent and waits for those in flight, so no request
/// outlives the conversion that pays for it.
/// What the analysis of a document's images brings from before its enhancement:
/// the requests already running and the assets that are not analysed.
pub(crate) struct Analysis<'a> {
    pub pending: Option<Pending>,
    pub skip: &'a HashSet<String>,
}

pub(crate) struct Pending {
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<Answers>>,
}
impl Pending {
    fn finish(mut self) -> Answers {
        let worker = self.worker.take().expect("pending analysis is joined once");
        worker
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }
    /// Stops requests not yet sent, waits for those in flight like a drop,
    /// and tells whether any image was answered; the answers are discarded.
    fn abandon(self) -> bool {
        self.stop.store(true, Ordering::Release);
        self.finish().values().any(std::result::Result::is_ok)
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            self.stop.store(true, Ordering::Release);
            let _ = worker.join();
        }
    }
}

/// The opening of the document body given to image prompts as context.
fn context(markdown: &str) -> String {
    let (_, body) = crate::output::split_frontmatter(markdown);
    body.chars().take(200).collect()
}

/// The document's assets that its Markdown references, in asset order,
/// except the `skip`ped ones and Windows metafiles (EMF, WMF: a Word chart),
/// which no decoder here reads: they stay files with their own alt text, as
/// a picture that cannot be analysed is no failure of the document.
fn referenced<'a>(doc: &'a Document, skip: &HashSet<String>) -> Vec<&'a crate::Asset> {
    let referenced: HashSet<_> = output_profiles::image_references(&doc.markdown)
        .iter()
        .filter_map(|target| resources::asset_name(target))
        .collect();
    doc.assets
        .iter()
        .filter(|asset| {
            referenced.contains(&asset.name)
                && !skip.contains(&asset.name)
                && !metafile(&asset.bytes)
        })
        .collect()
}

/// Whether `bytes` are an Enhanced Metafile (`EMR_HEADER` with the " EMF"
/// signature) or a Windows Metafile (placeable, or a standard header).
fn metafile(bytes: &[u8]) -> bool {
    let emf = bytes.len() >= 44 && bytes[..4] == [1, 0, 0, 0] && &bytes[40..44] == b" EMF";
    let placeable = bytes.starts_with(&[0xD7, 0xCD, 0xC6, 0x9A]);
    let standard = bytes.len() >= 18
        && matches!(bytes[..2], [1, 0] | [2, 0])
        && bytes[2..4] == [9, 0]
        && matches!(bytes[4..6], [0, 1] | [0, 3]);
    emf || placeable || standard
}

/// The pictures of the `unread` pages of a PDF whose Markdown has page markers
/// (`<!-- Page number: N -->`): a scanned page's picture is the page itself,
/// which a model that reads the page image transcribes, so it is not analysed
/// as a picture of its own.
pub(crate) fn scan_assets(markdown: &str, unread: &[usize]) -> HashSet<String> {
    let mut names = HashSet::new();
    let mut page = None;
    let mut text = String::new();
    let mut flush = |page: Option<usize>, text: &mut String| {
        if page.is_some_and(|page| unread.contains(&page)) {
            names.extend(
                output_profiles::image_references(text)
                    .iter()
                    .filter_map(|target| resources::asset_name(target)),
            );
        }
        text.clear();
    };
    for line in markdown.split_inclusive('\n') {
        let marker = line
            .trim()
            .strip_prefix("<!-- Page number:")
            .and_then(|rest| rest.strip_suffix("-->"))
            .and_then(|number| number.trim().parse::<usize>().ok());
        if let Some(number) = marker {
            flush(page, &mut text);
            page = Some(number);
        } else {
            text.push_str(line);
        }
    }
    flush(page, &mut text);
    names
}

/// Starts analysing each distinct referenced image of a (non-image) document
/// while its enhancement runs. Requests share the document's budget and the
/// runtime's `llm.concurrency`.
pub(crate) fn start(
    doc: &Document,
    base: &str,
    source: &str,
    cfg: &Value,
    runtime: &LlmRuntime,
    skip: &HashSet<String>,
) -> Option<Pending> {
    if !enabled(source, cfg) {
        return None;
    }
    let mut seen = HashSet::new();
    let images: Vec<_> = referenced(doc, skip)
        .into_iter()
        .filter_map(|asset| {
            let digest = crate::hex(Sha256::digest(&asset.bytes));
            seen.insert(digest.clone()).then(|| {
                let asset = crate::Asset {
                    name: asset.name.clone(),
                    bytes: asset.bytes.clone(),
                };
                (digest, asset)
            })
        })
        .collect();
    if images.is_empty() {
        return None;
    }
    let stop = Arc::new(AtomicBool::new(false));
    let document = llm::DocumentHandle::current();
    if let Some(document) = &document {
        document.reserve();
    }
    let (context, source, cfg, runtime, flag) = (
        context(base),
        source.to_owned(),
        cfg.clone(),
        runtime.clone(),
        stop.clone(),
    );
    let worker = std::thread::spawn(move || {
        let next = AtomicUsize::new(0);
        let answers = Mutex::new(Answers::new());
        std::thread::scope(|scope| {
            for _ in 0..images.len().min(runtime.concurrency()) {
                scope.spawn(|| {
                    let _document = document
                        .as_ref()
                        .map(llm::DocumentHandle::enter_image_worker);
                    while let Some((digest, asset)) =
                        images.get(next.fetch_add(1, Ordering::Relaxed))
                    {
                        if flag.load(Ordering::Acquire) {
                            break;
                        }
                        let answer = analyze_assets(
                            &[asset],
                            &context,
                            &source,
                            &cfg,
                            Some(&runtime),
                            Some(&flag),
                        )
                        .map_err(|error| error.to_string());
                        answers
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(digest.clone(), answer);
                    }
                });
            }
        });
        answers.into_inner().unwrap_or_else(|e| e.into_inner())
    });
    Some(Pending {
        stop,
        worker: Some(worker),
    })
}

pub(crate) fn enabled(source: &str, cfg: &Value) -> bool {
    let standalone = !crate::is_url(source)
        && Path::new(source)
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(images::is_image_extension);
    // Text only reads a standalone image's text with the same request, unless
    // images must not be uploaded (local OCR supplies the text then).
    let text_only = standalone
        && config::enabled(cfg, "/output/text_only")
        && !config::environment()
            .get("MARKITAI_NO_VLM_OCR")
            .is_some_and(|value| config::env_opt_out(value));
    config::enabled(cfg, "/llm/enabled")
        && (config::enabled(cfg, "/image/alt_enabled")
            || config::enabled(cfg, "/image/desc_enabled")
            || text_only)
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
/// A document's images were analysed by `pending` beside its enhancement; an
/// enhancement that failed discards them like the enhanced body.
/// Returns whether a model answered for at least one image, discarded or not.
pub(crate) fn analyze(
    doc: &Document,
    result: &mut ConversionOutput,
    source: &str,
    standalone: bool,
    cfg: &Value,
    runtime: Option<&LlmRuntime>,
    analysis: Analysis<'_>,
) -> Result<bool> {
    let Analysis { pending, skip } = analysis;
    if !enabled(source, cfg) {
        return Ok(false);
    }
    if !standalone && result.llm_markdown.is_none() {
        return Ok(pending.is_some_and(Pending::abandon));
    }
    let references = output_profiles::image_references(&doc.markdown);
    // An enhancement served from the cache never spent its reserved request.
    if let Some(document) = llm::DocumentHandle::current() {
        document.release();
    }
    let mut responses = pending.map(Pending::finish).unwrap_or_default();
    let mut captions = HashMap::new();
    let mut first_answer = None;
    let selected = referenced(doc, skip);
    // Multipage TIFF previews are one standalone image document. Analyze all
    // previews together rather than discarding the later pages' descriptions.
    let standalone_answer = if standalone && !selected.is_empty() {
        let context = context(result.llm_markdown.as_deref().unwrap_or(&doc.markdown));
        Some(analyze_assets(
            &selected, &context, source, cfg, runtime, None,
        )?)
    } else {
        None
    };
    let mut attempted = 0;
    let mut answered = HashSet::new();
    for (index, asset) in selected.iter().enumerate() {
        let digest = crate::hex(Sha256::digest(&asset.bytes));
        let answer = if let Some(answer) = &standalone_answer {
            let mut answer = answer.clone();
            if index > 0 {
                answer["llm_usage"] = json!({});
            }
            Ok(answer)
        } else if !answered.insert(digest.clone()) {
            responses[&digest].clone().map(|mut entry| {
                entry["llm_usage"] = json!({});
                entry
            })
        } else {
            attempted += 1;
            responses
                .entry(digest)
                .or_insert_with(|| {
                    Err(
                        "analysis stopped after an earlier fatal LLM failure of this document"
                            .into(),
                    )
                })
                .clone()
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
                if !(standalone && (config::enabled(cfg, "/llm/pure") || config::enabled(cfg, "/output/text_only"))) { result.images.push(answer); }
            }
            Err(message) => result.warnings.push(format!("Image analysis failed for {}: {message}; existing alt text was retained and no analysis entry was produced.", asset.name)),
        }
    }
    let answered = first_answer.is_some();
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
        if config::enabled(cfg, "/output/text_only") {
            // Only what the model read in the image: no title, preview,
            // description or fence.
            let text = text.trim_matches(['\n', '\r']).trim_end();
            if text.trim().is_empty() {
                result.warnings.push(
                    "No text was recognised in the image; the text-only document is empty.".into(),
                );
            }
            result.llm_markdown = Some(if text.trim().is_empty() {
                String::new()
            } else {
                format!("{text}\n")
            });
            result.text_only = true;
            return Ok(answered);
        }
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
    Ok(answered)
}

fn analyze_assets(
    assets: &[&crate::Asset],
    context: &str,
    source: &str,
    cfg: &Value,
    runtime: Option<&LlmRuntime>,
    stop: Option<&AtomicBool>,
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
        stop,
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
    fn the_pictures_of_unread_pages_are_found_by_their_page_markers() {
        let markdown = "<!-- Page number: 1 -->\n\n![Image on page 1](.markitai/assets/scan1.jpg)\n\n<!-- Page number: 2 -->\n\nText and a ![chart](.markitai/assets/chart.png) here.\n\n<!-- Page number: 3 -->\n\n![Image on page 3](.markitai/assets/scan3.jpg)\n";
        let scans = scan_assets(markdown, &[1, 3]);
        assert_eq!(
            scans,
            HashSet::from(["scan1.jpg".to_owned(), "scan3.jpg".to_owned()])
        );
        assert!(scan_assets(markdown, &[]).is_empty());
        // Without page markers nothing can be placed on a page.
        assert!(scan_assets("![x](.markitai/assets/a.jpg)", &[1]).is_empty());
    }

    #[test]
    fn windows_metafiles_are_kept_but_not_analysed() {
        let mut emf = vec![0u8; 88];
        emf[0] = 1;
        emf[40..44].copy_from_slice(b" EMF");
        let mut wmf = vec![1, 0, 9, 0, 0, 3];
        wmf.resize(18, 0);
        let doc = Document {
            markdown: "![chart](.markitai/assets/chart.emf)\n\n![old](.markitai/assets/old.wmf)\n\n![p](.markitai/assets/placeable.wmf)\n\n![photo](.markitai/assets/photo.png)\n".into(),
            assets: vec![
                crate::Asset { name: "chart.emf".into(), bytes: emf },
                crate::Asset { name: "old.wmf".into(), bytes: wmf },
                crate::Asset { name: "placeable.wmf".into(), bytes: vec![0xD7, 0xCD, 0xC6, 0x9A, 0, 0] },
                crate::Asset { name: "photo.png".into(), bytes: b"\x89PNG\r\n\x1a\n".to_vec() },
            ],
            ..Default::default()
        };
        let names: Vec<_> = referenced(&doc, &HashSet::new())
            .iter()
            .map(|asset| asset.name.clone())
            .collect();
        assert_eq!(names, ["photo.png"]);
    }

    #[test]
    fn skipped_assets_are_not_analysed() {
        let doc = Document {
            markdown: "![a](.markitai/assets/a.jpg)\n\n![b](.markitai/assets/b.jpg)\n".into(),
            assets: vec![
                crate::Asset {
                    name: "a.jpg".into(),
                    bytes: vec![1],
                },
                crate::Asset {
                    name: "b.jpg".into(),
                    bytes: vec![2],
                },
            ],
            ..Default::default()
        };
        let names = |skip: &HashSet<String>| {
            referenced(&doc, skip)
                .iter()
                .map(|a| a.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&HashSet::new()), ["a.jpg", "b.jpg"]);
        assert_eq!(names(&HashSet::from(["a.jpg".to_owned()])), ["b.jpg"]);
    }
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
