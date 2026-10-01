use crate::{Asset, ConversionOutput, Document, Error, Result, config};
use chrono::{Local, SecondsFormat};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Component, Path};
use std::sync::Mutex;

static OUTPUT_LOCK: Mutex<()> = Mutex::new(());
pub(crate) mod fence;
mod image_metadata;
pub(crate) mod stdout_assets;

/// Relocate complete asset destinations when archiving Markdown and its assets.
#[doc(hidden)]
pub fn rewrite_asset_references(
    markdown: &str,
    replacements: &std::collections::HashMap<String, String>,
) -> String {
    crate::output_profiles::rewrite_asset_references(markdown, replacements)
}

pub fn check_path(path: &Path, allow_symlinks: bool) -> Result<()> {
    check_paths(&[path], allow_symlinks)
}

/// `check_path` for each path in order, under one observation: an ancestor
/// that several of them share (the common parent of a document family) is
/// examined once, never the leaf of another path. Every call observes the
/// filesystem afresh; nothing is retained after it returns.
pub(crate) fn check_paths(paths: &[&Path], allow_symlinks: bool) -> Result<()> {
    if allow_symlinks {
        return Ok(());
    }
    // (ancestor, symbolic link owned by root); `None` is an ordinary entry or
    // a failed lookup, which this policy has always ignored.
    let mut observed: Vec<(std::path::PathBuf, Option<bool>)> = Vec::new();
    let shared = paths.len() > 1;
    for path in paths {
        let absolute = std::path::absolute(path)?;
        for ancestor in absolute.ancestors() {
            let link = match observed.iter().find(|(seen, _)| seen == ancestor) {
                Some((_, link)) => *link,
                None => {
                    let link = std::fs::symlink_metadata(ancestor)
                        .ok()
                        .filter(|metadata| metadata.file_type().is_symlink())
                        .map(|metadata| root_owned(&metadata));
                    if shared {
                        observed.push((ancestor.to_owned(), link));
                    }
                    link
                }
            };
            match link {
                None => (),
                Some(true) if ancestor != absolute => (),
                Some(_) => {
                    return Err(Error::InvalidInput(format!(
                        "Symlink access is disabled: {} (set output.allow_symlinks to true to follow it)",
                        ancestor.display()
                    )));
                }
            }
        }
    }
    Ok(())
}

fn root_owned(metadata: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        metadata.uid() == 0
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        false
    }
}

pub fn split_frontmatter(text: &str) -> (Map<String, Value>, &str) {
    if let Some(rest) = text.strip_prefix("---\n") {
        let mut offset = 0;
        for line in rest.split_inclusive('\n') {
            if line == "---\n" || line == "---" {
                if let Ok(Value::Object(meta)) = serde_yaml::from_str::<Value>(&rest[..offset]) {
                    return (meta, rest[offset + line.len()..].trim_start_matches('\n'));
                }
                break;
            }
            offset += line.len();
        }
    }
    (Map::new(), text)
}

/// Whether a YAML 1.1 loader (the reference's PyYAML) would resolve this plain
/// scalar as a boolean, number, null, timestamp, merge or value key instead of
/// a string. The reference writer quotes these, so readers keep strings.
fn yaml11_implicit(text: &str) -> bool {
    static PATTERN: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(concat!(
            r"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF",
            r"|[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?",
            r"|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)",
            r"|[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+",
            r"|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+|~|null|Null|NULL|<<|=",
            r"|[0-9]{4}-[0-9]{2}-[0-9]{2}",
            r"|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]*)?",
            r"(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?)$",
        ))
        .expect("valid YAML 1.1 resolver pattern")
    });
    text.is_empty() || PATTERN.is_match(text)
}

fn yaml_entry(key: &str, value: &Value) -> Result<String> {
    let failure = |e: serde_yaml::Error| Error::Conversion(e.to_string());
    let quoted = |text: &str| format!("'{}'", text.replace('\'', "''"));
    let plain_key = key
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        && !key.is_empty()
        && !yaml11_implicit(key);
    match value {
        Value::String(text) if plain_key && yaml11_implicit(text) => {
            Ok(format!("{key}: {}\n", quoted(text)))
        }
        Value::Array(items)
            if plain_key
                && items
                    .iter()
                    .all(|item| item.as_str().is_some_and(|text| !text.contains('\n')))
                && items
                    .iter()
                    .any(|item| item.as_str().is_some_and(yaml11_implicit)) =>
        {
            let mut entry = format!("{key}:\n");
            for text in items.iter().filter_map(Value::as_str) {
                if yaml11_implicit(text) {
                    entry.push_str(&format!("- {}\n", quoted(text)));
                } else {
                    entry.push_str("- ");
                    entry.push_str(&serde_yaml::to_string(text).map_err(failure)?);
                }
            }
            Ok(entry)
        }
        _ => serde_yaml::to_string(&json!({key:value})).map_err(failure),
    }
}

pub fn render(frontmatter: &Map<String, Value>, markdown: &str) -> Result<String> {
    let mut yaml = String::new();
    for key in [
        "title",
        "source",
        "description",
        "tags",
        "markitai_processed",
        "fetch_strategy",
    ] {
        if let Some(value) = frontmatter.get(key) {
            yaml.push_str(&yaml_entry(key, value)?);
        }
    }
    for (key, value) in frontmatter {
        if ![
            "title",
            "source",
            "description",
            "tags",
            "markitai_processed",
            "fetch_strategy",
        ]
        .contains(&key.as_str())
        {
            yaml.push_str(&yaml_entry(key, value)?);
        }
    }
    Ok(format!("---\n{yaml}---\n\n{markdown}"))
}

/// Render the same document bytes for CLI stdout and persisted files.
pub fn content(result: &ConversionOutput, cfg: &Value, enhanced: bool) -> Result<String> {
    let (body, prefix) = if enhanced {
        (
            result.llm_markdown.as_deref().unwrap_or(&result.markdown),
            result.pure_llm_prefix.as_deref(),
        )
    } else {
        (result.markdown.as_str(), result.pure_prefix.as_deref())
    };
    if config::enabled(cfg, "/llm/pure") && cfg["output"]["profile"] != "okf" {
        Ok(format!("{}{body}", prefix.unwrap_or("")))
    } else {
        let metadata = if enhanced {
            &result.frontmatter
        } else {
            result
                .base_frontmatter
                .as_ref()
                .unwrap_or(&result.frontmatter)
        };
        render(metadata, body)
    }
}

pub(crate) fn apply_profiles(result: &mut ConversionOutput, cfg: &Value) {
    if matches!(
        cfg.pointer("/output/profile").and_then(Value::as_str),
        Some("rag" | "obsidian")
    ) {
        for image in &mut result.images {
            if let Some(asset) = image.get("asset").and_then(Value::as_str)
                && let Some(suffix) = asset.strip_prefix(".markitai/assets/")
            {
                image["asset"] = format!("assets/{suffix}").into();
            }
        }
    }
    if let Some(markdown) = &mut result.llm_markdown {
        if let Some(base) = &mut result.base_frontmatter {
            crate::output_profiles::apply(&mut result.markdown, base, cfg);
        }
        crate::output_profiles::apply(markdown, &mut result.frontmatter, cfg);
        if result.frontmatter.is_empty() && config::enabled(cfg, "/llm/keep_base") {
            result.frontmatter = result.base_frontmatter.clone().unwrap_or_default();
        }
    } else {
        crate::output_profiles::apply(&mut result.markdown, &mut result.frontmatter, cfg);
        result.base_frontmatter = None;
    }
    if cfg["output"]["profile"] == "rag" {
        let mut notices = crate::output_profiles::table_warnings(&result.markdown);
        if let Some(markdown) = &result.llm_markdown {
            notices.extend(crate::output_profiles::table_warnings(markdown));
        }
        for warning in notices {
            if !result.warnings.contains(&warning) {
                result.warnings.push(warning);
            }
        }
    }
}

pub fn prepare(source: &str, name: &str, doc: &mut Document, cfg: &Value) -> ConversionOutput {
    let pure = config::enabled(cfg, "/llm/pure");
    let (mut meta, markdown, pure_prefix) = if pure {
        let (meta, body) = split_frontmatter(&doc.markdown);
        let prefix_len = doc.markdown.len() - body.len();
        (
            meta,
            body.to_owned(),
            (prefix_len > 0).then(|| doc.markdown[..prefix_len].to_owned()),
        )
    } else {
        (Map::new(), crate::markdown::normalize(&doc.markdown), None)
    };
    if !pure {
        let low_confidence = Path::new(name)
            .extension()
            .and_then(|v| v.to_str())
            .is_some_and(|v| ["csv", "tsv", "xml"].contains(&v.to_ascii_lowercase().as_str()));
        // Basic workflow titles use the first heading. This deliberately keeps
        // its historical permissive syntax, including headings in code blocks.
        let heading = (!low_confidence)
            .then(|| {
                markdown.trim_start().lines().find_map(|line| {
                    if line.starts_with("# ") || line.starts_with("##") {
                        let title = line.trim_start_matches('#').trim().replace("**", "");
                        (!title.trim().is_empty()).then(|| title.trim().to_owned())
                    } else {
                        None
                    }
                })
            })
            .flatten();
        let title = doc
            .metadata
            .get("title")
            .and_then(Value::as_str)
            .filter(|title| !title.is_empty())
            .map(normalize_title)
            .or(heading)
            .unwrap_or_else(|| {
                if low_confidence {
                    name.to_owned()
                } else {
                    Path::new(name)
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                }
            });
        meta.insert(
            "title".into(),
            json!(title.split_whitespace().collect::<Vec<_>>().join(" ")),
        );
        let source_value = if crate::is_url(source) {
            redact_url(source)
        } else {
            name.to_string()
        };
        meta.insert("source".into(), json!(source_value));
        meta.insert(
            "markitai_processed".into(),
            json!(Local::now().to_rfc3339_opts(SecondsFormat::Millis, false)),
        );
        if crate::is_url(source) {
            for (key, value) in &doc.metadata {
                if ![
                    "title",
                    "source",
                    "description",
                    "tags",
                    "markitai_processed",
                    "language",
                    "format",
                    // Internal reader identity, not user-facing page metadata.
                    "converter",
                ]
                .contains(&key.as_str())
                    && !value.is_null()
                {
                    meta.insert(key.clone(), value.clone());
                }
            }
        }
    }
    ConversionOutput {
        source: source.into(),
        markdown,
        pure_prefix,
        frontmatter: meta,
        warnings: std::mem::take(&mut doc.warnings),
        ..Default::default()
    }
}

fn normalize_title(title: &str) -> String {
    use std::sync::LazyLock;
    static LINKS: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"!?\[([^\]]*)\]\([^)]+\)").unwrap());
    static TAGS: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"(?i)</?(?:strong|em|b|i|code)>").unwrap());
    static HEADING: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"^#{1,6}\s+").unwrap());
    let text = title.split_whitespace().collect::<Vec<_>>().join(" ");
    let text = HEADING.replace(&text, "");
    let text = LINKS.replace_all(&text, "$1");
    let text = TAGS.replace_all(&text, "");
    let mut text = text.trim();
    loop {
        let stripped = ["**", "__", "~~", "`", "*", "_"]
            .into_iter()
            .find_map(|wrapper| {
                text.strip_prefix(wrapper)?
                    .strip_suffix(wrapper)
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
            });
        match stripped {
            Some(value) => text = value,
            None => break,
        }
    }
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn redact_url(source: &str) -> String {
    let Ok(mut url) = url::Url::parse(source) else {
        return source.to_owned();
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);
    let pairs: Vec<_> = url
        .query_pairs()
        .map(|(key, value)| {
            let sensitive = [
                "token",
                "key",
                "secret",
                "password",
                "signature",
                "credential",
            ]
            .iter()
            .any(|part| key.to_lowercase().contains(part));
            (
                key.into_owned(),
                if sensitive {
                    "REDACTED".into()
                } else {
                    value.into_owned()
                },
            )
        })
        .collect();
    if !pairs.is_empty() {
        url.query_pairs_mut().clear().extend_pairs(pairs);
    }
    url.to_string()
}

pub fn url_name(source: &str, _meta: &Map<String, Value>) -> String {
    let parsed = url::Url::parse(source).ok();
    let host = parsed
        .as_ref()
        .and_then(|u| u.host_str())
        .unwrap_or("page")
        .replace('.', "_");
    let host = match parsed.as_ref().and_then(|u| u.port()) {
        Some(port) => format!("{host}_{port}"),
        None => host,
    };
    let segment = parsed
        .as_ref()
        .and_then(|u| u.path_segments()?.rfind(|s| !s.is_empty()));
    let raw = match segment {
        Some(path) if parsed.as_ref().is_some_and(|u| u.query().is_some()) => {
            format!("{host}_{path}")
        }
        Some(path) => path.to_owned(),
        None => host,
    };
    let safe: String = raw
        .chars()
        .map(|c| if "<>:\"/\\|?*".contains(c) { '_' } else { c })
        .collect();
    let safe: String = safe.trim_matches([' ', '.']).chars().take(200).collect();
    if safe.is_empty() {
        "unnamed".into()
    } else {
        safe
    }
}

pub fn should_skip(dir: &Path, name: &str, cfg: &Value) -> Result<bool> {
    if cfg.pointer("/output/on_conflict").and_then(Value::as_str) != Some("skip") {
        return Ok(false);
    }
    let name = cfg
        .pointer("/output/filename")
        .and_then(Value::as_str)
        .map(|s| s.strip_suffix(".md").unwrap_or(s))
        .or_else(|| cfg.pointer("/output/reserved_stem").and_then(Value::as_str))
        .unwrap_or(name);
    if Path::new(name).components().count() != 1 {
        return Err(Error::InvalidInput("Output name must be a filename".into()));
    }
    let base = dir.join(format!("{name}.md"));
    let llm = dir.join(format!("{name}.llm.md"));
    check_paths(
        &[&base, &llm],
        config::enabled(cfg, "/output/allow_symlinks"),
    )?;
    Ok(base.exists() || llm.exists())
}

/// Staging for user-facing outputs: documents, assets, sidecars and reports.
/// Like an ordinary new file and the reference writer, the final mode follows
/// the process umask rather than tempfile's private 0600. Ownership records,
/// receipts, locks and service state keep their own restrictive modes.
#[doc(hidden)]
pub fn deliverable_builder<'a, 'b>() -> tempfile::Builder<'a, 'b> {
    #[cfg_attr(not(unix), allow(unused_mut))] // Only Unix sets a mode.
    let mut builder = tempfile::Builder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(0o666));
    }
    builder
}

fn atomic_write(path: &Path, bytes: &[u8], overwrite: bool) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::InvalidInput("Output has no parent".into()))?;
    let mut temp = deliverable_builder().tempfile_in(parent)?;
    temp.write_all(bytes)?;
    // The bytes precede the name; neither this write nor its rename was ever
    // followed by a directory synchronization.
    fence::order_staged(temp.as_file())?;
    if overwrite {
        temp.persist(path).map_err(|e| Error::Io(e.error))?;
    } else {
        temp.persist_noclobber(path)
            .map_err(|e| Error::Io(e.error))?;
    }
    fence::note("published");
    Ok(())
}

/// Explicit native publication authority; no process-global callback is used.
#[doc(hidden)]
pub trait Publication: Send + Sync {
    fn skip_existing(&self) -> bool;
    fn publish(&self, path: &Path, bytes: &[u8]) -> Result<()>;
}

/// Exact final document bytes awaiting an explicitly owned publication.
#[doc(hidden)]
pub struct RenderedMember {
    pub path: std::path::PathBuf,
    pub bytes: Vec<u8>,
}

#[derive(Default)]
pub(crate) struct PreparedOutput {
    pub(crate) members: Vec<RenderedMember>,
    metadata: Option<PreparedImageMetadata>,
}
struct PreparedImageMetadata {
    directory: std::path::PathBuf,
    images: Vec<Value>,
    source: String,
    allow_symlinks: bool,
}
impl PreparedOutput {
    pub(crate) fn finalize(self) -> Result<()> {
        if let Some(metadata) = self.metadata {
            let _guard = OUTPUT_LOCK
                .lock()
                .map_err(|_| Error::Conversion("Output lock poisoned".into()))?;
            image_metadata::publish(
                &metadata.directory,
                &metadata.images,
                &metadata.source,
                metadata.allow_symlinks,
            )?;
        }
        Ok(())
    }
}

pub fn write(
    dir: &Path,
    name: &str,
    result: &mut ConversionOutput,
    assets: &[Asset],
    cfg: &Value,
) -> Result<()> {
    write_with_publication(dir, name, result, assets, cfg, None)
}

#[doc(hidden)]
pub fn write_with_publication(
    dir: &Path,
    name: &str,
    result: &mut ConversionOutput,
    assets: &[Asset],
    cfg: &Value,
    publication: Option<&dyn Publication>,
) -> Result<()> {
    write_document(
        dir,
        name,
        result,
        assets,
        Screenshots::New(&[]),
        cfg,
        publication,
    )
}

pub(crate) enum Screenshots<'a> {
    New(&'a [Asset]),
    /// Document captures have frozen names and retain their native Markdown.
    PublishedPages(&'a [Asset]),
}

pub(crate) fn write_document(
    dir: &Path,
    name: &str,
    result: &mut ConversionOutput,
    assets: &[Asset],
    screenshots: Screenshots<'_>,
    cfg: &Value,
    publication: Option<&dyn Publication>,
) -> Result<()> {
    write_document_mode(
        dir,
        name,
        result,
        assets,
        screenshots,
        cfg,
        WritePolicy {
            publication,
            prepared: None,
        },
    )
}

pub(crate) struct WritePolicy<'a, 'b> {
    pub(crate) publication: Option<&'a dyn Publication>,
    pub(crate) prepared: Option<&'b mut PreparedOutput>,
}

pub(crate) fn write_document_mode(
    dir: &Path,
    name: &str,
    result: &mut ConversionOutput,
    assets: &[Asset],
    screenshots: Screenshots<'_>,
    cfg: &Value,
    policy: WritePolicy<'_, '_>,
) -> Result<()> {
    let WritePolicy {
        publication,
        mut prepared,
    } = policy;
    let _guard = OUTPUT_LOCK
        .lock()
        .map_err(|_| Error::Conversion("Output lock poisoned".into()))?;
    let allow_symlinks = config::enabled(cfg, "/output/allow_symlinks");
    check_path(dir, allow_symlinks)?;
    let explicit_name = cfg.pointer("/output/filename").and_then(Value::as_str);
    let name = explicit_name
        .map(|name| name.strip_suffix(".md").unwrap_or(name))
        .or_else(|| cfg.pointer("/output/reserved_stem").and_then(Value::as_str))
        .unwrap_or(name);
    if Path::new(name).components().count() != 1
        || !matches!(
            Path::new(name).components().next(),
            Some(Component::Normal(_))
        )
    {
        return Err(Error::InvalidInput("Output name must be a filename".into()));
    }
    std::fs::create_dir_all(dir)?;
    let mode = cfg
        .pointer("/output/on_conflict")
        .and_then(Value::as_str)
        .unwrap_or("rename");
    let mut stem = name.to_owned();
    let mut revision = 2;
    loop {
        let base = dir.join(format!("{stem}.md"));
        let enhanced = dir.join(format!("{stem}.llm.md"));
        check_paths(&[&base, &enhanced], allow_symlinks)?;
        if publication.is_some() || !base.exists() && !enhanced.exists() || mode == "overwrite" {
            break;
        }
        if mode == "skip" {
            result.skip_reason = Some("exists".into());
            result.markdown.clear();
            return Ok(());
        }
        stem = format!("{name}.v{revision}");
        revision += 1;
    }
    let visible = matches!(
        cfg.pointer("/output/profile").and_then(Value::as_str),
        Some("rag" | "obsidian")
    );
    let asset_prefix = if visible {
        "assets"
    } else {
        ".markitai/assets"
    };
    let mut replacements = HashMap::with_capacity(assets.len());
    for asset in assets {
        let extension = Path::new(&asset.name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("bin");
        let safe_extension: String = extension
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .take(12)
            .collect();
        let digest = crate::hex(Sha256::digest(&asset.bytes));
        let filename = format!("{}.{}", &digest[..24], safe_extension);
        let path = dir.join(asset_prefix).join(&filename);
        check_path(&path, allow_symlinks)?;
        std::fs::create_dir_all(path.parent().unwrap())?;
        crate::asset_store::insert_or_verify(&path, &asset.bytes)?;
        let before = format!("{asset_prefix}/{}", asset.name);
        let after = format!("{asset_prefix}/{filename}");
        replacements.entry(before).or_insert(after);
        if !result.assets.contains(&path) {
            result.assets.push(path);
        }
    }
    if !replacements.is_empty() {
        result.markdown = rewrite_asset_references(&result.markdown, &replacements);
        if let Some(md) = &mut result.llm_markdown {
            *md = rewrite_asset_references(md, &replacements);
        }
    }
    for image in &mut result.images {
        let asset = image
            .get("asset")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Conversion("Image analysis has no associated asset".into()))?;
        let published = replacements.get(asset).ok_or_else(|| {
            Error::Conversion("Image analysis refers to an asset that was not published".into())
        })?;
        image["asset"] = std::path::absolute(dir.join(published))?
            .to_string_lossy()
            .as_ref()
            .into();
    }
    let (screenshots, published) = match screenshots {
        Screenshots::New(shots) => (shots, false),
        Screenshots::PublishedPages(shots) => (shots, true),
    };
    for screenshot in screenshots {
        let path = if published {
            let path = dir.join(".markitai/screenshots").join(&screenshot.name);
            check_path(&path, allow_symlinks)?;
            if screenshot_matches(&path, &screenshot.bytes)? != Some(true) {
                return Err(Error::Conversion("A published page screenshot changed during conversion; document publication stopped to preserve its references".into()));
            }
            path
        } else {
            publish_screenshot(dir, screenshot, allow_symlinks)?
        };
        if !result.screenshots.contains(&path) {
            result.screenshots.push(path);
        }
    }
    let capture_only = crate::is_url(&result.source)
        && !published
        && config::enabled(cfg, "/screenshot/screenshot_only")
        && !(config::enabled(cfg, "/llm/enabled") && config::enabled(cfg, "/llm/pure"));
    if !capture_only && (result.llm_markdown.is_none() || config::enabled(cfg, "/llm/keep_base")) {
        let path = dir.join(format!("{stem}.md"));
        let content = content(result, cfg, false)?;
        if let Some(prepared) = prepared.as_deref_mut() {
            prepared.members.push(RenderedMember {
                path: path.clone(),
                bytes: content.into_bytes(),
            });
        } else if let Some(publication) = publication {
            publication.publish(&path, content.as_bytes())?;
        } else {
            atomic_write(&path, content.as_bytes(), mode == "overwrite")?;
        }
        result.output_path = Some(path);
    }
    if result.llm_markdown.is_some() {
        let path = if explicit_name.is_some() && !config::enabled(cfg, "/llm/keep_base") {
            dir.join(format!("{stem}.md"))
        } else {
            dir.join(format!("{stem}.llm.md"))
        };
        let content = content(result, cfg, true)?;
        if let Some(prepared) = prepared.as_deref_mut() {
            prepared.members.push(RenderedMember {
                path: path.clone(),
                bytes: content.into_bytes(),
            });
        } else if let Some(publication) = publication {
            publication.publish(&path, content.as_bytes())?;
        } else {
            atomic_write(&path, content.as_bytes(), mode == "overwrite")?;
        }
        result.llm_output_path = Some(path);
    }
    if config::enabled(cfg, "/image/desc_enabled") && !result.images.is_empty() {
        let source = if crate::is_url(&result.source) {
            result
                .llm_output_path
                .as_ref()
                .or(result.output_path.as_ref())
                .and_then(|path| path.file_stem())
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        } else {
            result.source.clone()
        };
        if let Some(prepared) = prepared {
            prepared.metadata = Some(PreparedImageMetadata {
                directory: dir.join(asset_prefix),
                images: result.images.clone(),
                source,
                allow_symlinks,
            });
        } else {
            image_metadata::publish(
                &dir.join(asset_prefix),
                &result.images,
                &source,
                allow_symlinks,
            )?;
        }
    }
    Ok(())
}

pub(crate) fn publish_page_screenshots(
    dir: &Path,
    screenshots: &mut [Asset],
    cfg: &Value,
) -> Result<()> {
    let _guard = OUTPUT_LOCK
        .lock()
        .map_err(|_| Error::Conversion("Output publication lock poisoned".into()))?;
    let allow_symlinks = config::enabled(cfg, "/output/allow_symlinks");
    check_path(dir, allow_symlinks)?;
    for screenshot in screenshots {
        let path = publish_screenshot(dir, screenshot, allow_symlinks)?;
        screenshot.name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::Conversion("Screenshot filename is not UTF-8".into()))?
            .to_owned();
    }
    Ok(())
}

fn publish_screenshot(
    dir: &Path,
    screenshot: &Asset,
    allow_symlinks: bool,
) -> Result<std::path::PathBuf> {
    let name = Path::new(&screenshot.name);
    if name.components().count() != 1
        || !matches!(name.components().next(), Some(Component::Normal(_)))
    {
        return Err(Error::InvalidInput(
            "Screenshot name must be a filename".into(),
        ));
    }
    let directory = dir.join(".markitai/screenshots");
    check_path(&directory, allow_symlinks)?;
    std::fs::create_dir_all(&directory)?;
    let stem = name.file_stem().unwrap_or_default().to_string_lossy();
    let extension = name.extension().unwrap_or_default().to_string_lossy();
    for revision in 1u64.. {
        let filename = if revision == 1 {
            screenshot.name.clone()
        } else {
            format!("{stem}.v{revision}.{extension}")
        };
        let path = directory.join(filename);
        check_path(&path, allow_symlinks)?;
        if screenshot_matches(&path, &screenshot.bytes)? == Some(false) {
            continue;
        }
        // Capture names remain recognizable. Changed captures get new names,
        // preserving images referenced by earlier results and history entries.
        crate::asset_store::insert_or_verify(&path, &screenshot.bytes)?;
        return Ok(path);
    }
    Err(Error::Conversion(
        "Screenshot version counter exhausted".into(),
    ))
}

fn screenshot_matches(path: &Path, expected: &[u8]) -> Result<Option<bool>> {
    use std::io::Read;
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() {
        return Err(Error::InvalidInput(
            "Screenshot destination is not a regular file".into(),
        ));
    }
    if metadata.len() != expected.len() as u64 {
        return Ok(Some(false));
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Do not block if a concurrently changed destination becomes a FIFO.
        options.custom_flags(libc::O_NONBLOCK);
    }
    let mut file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(Error::InvalidInput(
            "Screenshot destination changed file type".into(),
        ));
    }
    let mut buffer = [0u8; 32 * 1024];
    for bytes in expected.chunks(buffer.len()) {
        match file.read_exact(&mut buffer[..bytes.len()]) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(Some(false));
            }
            Err(error) => return Err(error.into()),
        }
        if &buffer[..bytes.len()] != bytes {
            return Ok(Some(false));
        }
    }
    Ok(Some(file.read(&mut buffer[..1])? == 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frontmatter_quotes_yaml11_implicit_strings_like_the_reference_writer() {
        // Expected spellings were produced by PyYAML safe_dump, the reference writer.
        for (value, expected) in [
            (
                "2026-09-29T22:30:13.781+08:00",
                "'2026-09-29T22:30:13.781+08:00'",
            ),
            ("2026-09-29", "'2026-09-29'"),
            ("2026-9-9 1:02:03", "'2026-9-9 1:02:03'"),
            ("yes", "'yes'"),
            ("Off", "'Off'"),
            ("1:30", "'1:30'"),
            ("0x1F", "'0x1F'"),
            ("1_000", "'1_000'"),
            ("0755", "'0755'"),
            ("+1", "'+1'"),
            ("3.", "'3.'"),
            (".inf", "'.inf'"),
            ("~", "'~'"),
            ("null", "'null'"),
            ("", "''"),
            ("=", "'='"),
            ("<<", "'<<'"),
            ("plain text", "plain text"),
            ("y", "y"),
        ] {
            let mut frontmatter = Map::new();
            frontmatter.insert("markitai_processed".into(), json!(value));
            let rendered = render(&frontmatter, "Body\n").unwrap();
            assert_eq!(
                rendered,
                format!("---\nmarkitai_processed: {expected}\n---\n\nBody\n"),
                "{value:?}"
            );
            let parsed: serde_json::Value =
                serde_yaml::from_str(rendered.split("---\n").nth(1).unwrap()).unwrap();
            assert_eq!(parsed["markitai_processed"], json!(value), "{value:?}");
        }
        let mut frontmatter = Map::new();
        frontmatter.insert("tags".into(), json!(["2026", "o'clock", "on", "topic"]));
        frontmatter.insert("title".into(), json!("It's 2026-01-02"));
        let rendered = render(&frontmatter, "").unwrap();
        assert_eq!(
            rendered,
            "---\ntitle: It's 2026-01-02\ntags:\n- '2026'\n- o'clock\n- 'on'\n- topic\n---\n\n"
        );
    }
    #[test]
    fn screenshot_comparison_rejects_special_files_and_large_existing_payloads() {
        let root = tempfile::tempdir().unwrap();
        let large = root.path().join("large.jpg");
        std::fs::File::create(&large)
            .unwrap()
            .set_len(8 * 1024 * 1024 * 1024)
            .unwrap();
        assert_eq!(screenshot_matches(&large, b"small").unwrap(), Some(false));
        assert!(screenshot_matches(root.path(), b"x").is_err());
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let fifo = root.path().join("capture.jpg");
            let name = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
            assert!(screenshot_matches(&fifo, b"x").is_err());
        }
    }

    #[test]
    fn published_pdf_screenshots_fail_if_changed_before_document_write() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config::normalize(&json!({"llm":{"enabled":false}})).unwrap();
        let mut shots = [Asset {
            name: "report.page0001.jpg".into(),
            bytes: b"original capture".to_vec(),
        }];
        publish_page_screenshots(dir.path(), &mut shots, &cfg).unwrap();
        let capture = dir
            .path()
            .join(".markitai/screenshots")
            .join(&shots[0].name);
        std::fs::write(&capture, b"concurrent changed capture").unwrap();
        let mut result = ConversionOutput {
            source: "report.pdf".into(),
            markdown: "PDF body".into(),
            ..Default::default()
        };
        let error = write_document(
            dir.path(),
            "report.pdf",
            &mut result,
            &[],
            Screenshots::PublishedPages(&shots),
            &cfg,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("changed during conversion"));
        assert!(!dir.path().join("report.pdf.md").exists());
        assert!(!capture.with_file_name("report.page0001.v2.jpg").exists());
        assert_eq!(
            std::fs::read(&capture).unwrap(),
            b"concurrent changed capture"
        );
    }

    #[test]
    fn screenshot_only_publishes_all_tiles_and_preserves_previous_captures() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config::normalize(
            &json!({"screenshot":{"screenshot_only":true},"llm":{"enabled":false}}),
        )
        .unwrap();
        let mut result = ConversionOutput {
            source: "https://example.test/page".into(),
            ..Default::default()
        };
        let shots = [
            Asset {
                name: "example.test_page.full.jpg".into(),
                bytes: b"capture zero".to_vec(),
            },
            Asset {
                name: "example.test_page.full--1.jpg".into(),
                bytes: b"capture one".to_vec(),
            },
        ];
        write_document(
            dir.path(),
            "page",
            &mut result,
            &[],
            Screenshots::New(&shots),
            &cfg,
            None,
        )
        .unwrap();
        assert!(result.output_path.is_none() && result.llm_output_path.is_none());
        assert!(!dir.path().join("page.md").exists());
        assert_eq!(result.screenshots.len(), 2);
        for (path, shot) in result.screenshots.iter().zip(&shots) {
            assert_eq!(std::fs::read(path).unwrap(), shot.bytes);
        }
        let original = result.screenshots[0].clone();
        let mut second = ConversionOutput {
            source: result.source.clone(),
            ..Default::default()
        };
        let changed = [Asset {
            name: shots[0].name.clone(),
            bytes: b"updated capture".to_vec(),
        }];
        write_document(
            dir.path(),
            "page",
            &mut second,
            &[],
            Screenshots::New(&changed),
            &cfg,
            None,
        )
        .unwrap();
        assert_eq!(second.screenshots.len(), 1);
        assert_ne!(second.screenshots[0], original);
        assert_eq!(std::fs::read(original).unwrap(), shots[0].bytes);
        assert_eq!(
            std::fs::read(&second.screenshots[0]).unwrap(),
            changed[0].bytes
        );
        let invalid = Asset {
            name: "../escape.jpg".into(),
            bytes: vec![1],
        };
        assert!(publish_screenshot(dir.path(), &invalid, false).is_err());
        assert!(!dir.path().join("escape.jpg").exists());
    }

    #[test]
    fn css_resources_publish_exact_assets_in_both_outputs_and_visible_profiles() {
        let first = b"first css resource".to_vec();
        let second = b"second css resource".to_vec();
        let first_digest = crate::hex(Sha256::digest(&first));
        let second_digest = crate::hex(Sha256::digest(&second));
        let first_name = format!("{}.bin", &first_digest[..24]);
        let second_name = format!("{}.bin", &second_digest[..24]);
        let source = format!(
            "<div style=\"background:url(.markitai/assets/first.bin?x=1&amp;y=2#part);content:'url(.markitai/assets/first.bin)'\"></div>\n\
<style>\n\
@import '.markitai/assets/{first_name}';\n\
.image {{ background:image-set('.markitai/assets/first.bin' 1x, url(.markitai/assets/duplicate.bin) 2x); }}\n\
@font-face {{ src:url('.markitai/assets/{first_name}?v=1&x=2#face'); }}\n\
/* url(.markitai/assets/first.bin) */\n\
</style>\n\
`<style>.literal{{background:url(.markitai/assets/first.bin)}}</style>`\n"
        );
        for profile in ["default", "rag", "obsidian"] {
            let root = tempfile::tempdir().unwrap();
            let mut cfg = config::defaults();
            if profile != "default" {
                cfg["output"]["profile"] = json!(profile);
            }
            cfg["llm"]["keep_base"] = json!(true);
            cfg["llm"]["pure"] = json!(true);
            let prefix = if profile == "default" {
                ".markitai/assets"
            } else {
                "assets"
            };
            let expected = format!(
                "<div style=\"background:url({prefix}/{first_name}?x=1&amp;y=2#part);content:'url(.markitai/assets/first.bin)'\"></div>\n\
<style>\n\
@import '{prefix}/{second_name}';\n\
.image {{ background:image-set('{prefix}/{first_name}' 1x, url({prefix}/{first_name}) 2x); }}\n\
@font-face {{ src:url('{prefix}/{second_name}?v=1&x=2#face'); }}\n\
/* url(.markitai/assets/first.bin) */\n\
</style>\n\
`<style>.literal{{background:url(.markitai/assets/first.bin)}}</style>`\n"
            );
            let mut result = ConversionOutput {
                markdown: source.clone(),
                llm_markdown: Some(source.clone()),
                base_frontmatter: Some(Map::new()),
                ..Default::default()
            };
            apply_profiles(&mut result, &cfg);
            let assets = [
                Asset {
                    name: "first.bin".into(),
                    bytes: first.clone(),
                },
                Asset {
                    name: first_name.clone(),
                    bytes: second.clone(),
                },
                Asset {
                    name: "duplicate.bin".into(),
                    bytes: first.clone(),
                },
            ];
            write(root.path(), "document", &mut result, &assets, &cfg).unwrap();
            assert_eq!(result.markdown, expected, "{profile}");
            assert_eq!(result.llm_markdown.as_deref(), Some(expected.as_str()));
            assert_eq!(result.assets.len(), 2);
            for (name, bytes) in [(&first_name, &first), (&second_name, &second)] {
                assert_eq!(
                    std::fs::read(root.path().join(prefix).join(name)).unwrap(),
                    *bytes
                );
            }
            for path in [result.output_path.unwrap(), result.llm_output_path.unwrap()] {
                assert_eq!(std::fs::read_to_string(path).unwrap(), expected);
            }
        }
    }

    #[test]
    fn asset_publication_rewrites_original_paths_once_in_base_and_enhanced_outputs() {
        let first = b"first asset".to_vec();
        let second = b"different asset".to_vec();
        let first_digest = crate::hex(Sha256::digest(&first));
        let second_digest = crate::hex(Sha256::digest(&second));
        let first_name = format!("{}.bin", &first_digest[..24]);
        let second_name = format!("{}.bin", &second_digest[..24]);
        for profile in ["default", "rag", "obsidian"] {
            let root = tempfile::tempdir().unwrap();
            let mut cfg = config::defaults();
            cfg["output"]["profile"] = json!(profile);
            cfg["llm"]["keep_base"] = json!(true);
            cfg["llm"]["pure"] = json!(true);
            let prefix = if profile == "default" {
                ".markitai/assets"
            } else {
                "assets"
            };
            let source = format!(
                "[first]({prefix}/first.bin)\n[second]({prefix}/{first_name})\n[duplicate]({prefix}/duplicate.bin)\n`[literal]({prefix}/first.bin)`\n"
            );
            let expected = format!(
                "[first]({prefix}/{first_name})\n[second]({prefix}/{second_name})\n[duplicate]({prefix}/{first_name})\n`[literal]({prefix}/first.bin)`\n"
            );
            let mut result = ConversionOutput {
                markdown: source.clone(),
                llm_markdown: Some(source),
                ..Default::default()
            };
            let assets = [
                Asset {
                    name: "first.bin".into(),
                    bytes: first.clone(),
                },
                Asset {
                    name: first_name.clone(),
                    bytes: second.clone(),
                },
                Asset {
                    name: "duplicate.bin".into(),
                    bytes: first.clone(),
                },
                Asset {
                    name: "first.bin".into(),
                    bytes: second.clone(),
                },
            ];
            write(root.path(), "document", &mut result, &assets, &cfg).unwrap();
            assert_eq!(result.markdown, expected);
            assert_eq!(result.llm_markdown.as_deref(), Some(expected.as_str()));
            assert_eq!(result.assets.len(), 2);
            assert_eq!(
                std::fs::read(root.path().join(prefix).join(&first_name)).unwrap(),
                first
            );
            assert_eq!(
                std::fs::read(root.path().join(prefix).join(&second_name)).unwrap(),
                second
            );
            for path in [result.output_path.unwrap(), result.llm_output_path.unwrap()] {
                assert_eq!(std::fs::read_to_string(path).unwrap(), expected);
            }
        }
    }

    #[test]
    fn url_frontmatter_keeps_page_facts_but_not_reader_identity() {
        let mut document = Document {
            markdown: "# Page\n\nBody.".into(),
            ..Default::default()
        };
        for (key, value) in [
            ("domain", json!("example.test")),
            ("word_count", json!(3)),
            ("converter", json!("native-html")),
        ] {
            document.metadata.insert(key.into(), value);
        }
        let cfg = config::defaults();
        let output = prepare("https://example.test/page", "page", &mut document, &cfg);
        assert_eq!(output.frontmatter["domain"], "example.test");
        assert_eq!(output.frontmatter["word_count"], 3);
        assert!(!output.frontmatter.contains_key("converter"));
    }

    #[test]
    fn local_output_uses_workflow_title_and_preserves_original_frontmatter_as_body() {
        let input = "---\ntitle: Existing\ncustom: kept\n---\n\n# Heading\nBody  ";
        let mut document = Document {
            markdown: input.into(),
            ..Default::default()
        };
        document
            .metadata
            .insert("author".into(), json!("not a local frontmatter field"));
        let mut cfg = config::defaults();
        let normal = prepare("existing.md", "existing.md", &mut document, &cfg);
        assert_eq!(normal.frontmatter["title"], "Heading");
        assert!(!normal.frontmatter.contains_key("author"));
        assert_eq!(
            normal.markdown,
            "---\ntitle: Existing\ncustom: kept\n---\n\n# Heading\n\nBody\n"
        );
        cfg["llm"]["pure"] = json!(true);
        let mut pure = prepare("existing.md", "existing.md", &mut document, &cfg);
        assert_eq!(
            pure.frontmatter,
            json!({"title":"Existing","custom":"kept"})
                .as_object()
                .unwrap()
                .clone()
        );
        assert_eq!(pure.markdown, "# Heading\nBody  ");
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "existing.md", &mut pure, &[], &cfg).unwrap();
        assert_eq!(
            std::fs::read_to_string(pure.output_path.unwrap()).unwrap(),
            input
        );
    }

    #[test]
    fn structured_data_uses_full_name_and_explicit_titles_remain_authoritative() {
        let cfg = config::defaults();
        for name in ["data.csv", "data.tsv", "data.XML"] {
            let mut doc = Document {
                markdown: "# Untrusted data\nbody".into(),
                ..Default::default()
            };
            assert_eq!(
                prepare(name, name, &mut doc, &cfg).frontmatter["title"],
                name
            );
            doc.metadata.insert(
                "title".into(),
                json!("**[Explicit](https://example.test)**"),
            );
            assert_eq!(
                prepare(name, name, &mut doc, &cfg).frontmatter["title"],
                "Explicit"
            );
        }
    }

    #[test]
    fn okf_uses_utc_and_adds_generated_identity_in_pure_mode() {
        let mut cfg = config::defaults();
        cfg["output"]["profile"] = json!("okf");
        let mut doc = Document {
            markdown: "# Heading".into(),
            ..Default::default()
        };
        let mut normal = prepare("file.txt", "file.txt", &mut doc, &cfg);
        apply_profiles(&mut normal, &cfg);
        assert!(
            normal.frontmatter["generated"]["at"]
                .as_str()
                .unwrap()
                .ends_with('Z')
        );
        assert!(!normal.frontmatter.contains_key("source"));
        cfg["llm"]["pure"] = json!(true);
        let mut pure = prepare("file.txt", "file.txt", &mut doc, &cfg);
        apply_profiles(&mut pure, &cfg);
        assert!(pure.frontmatter["generated"].get("at").is_none());
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "file.txt", &mut pure, &[], &cfg).unwrap();
        let written = std::fs::read_to_string(pure.output_path.unwrap()).unwrap();
        assert_eq!(split_frontmatter(&written).0["type"], "Document");
    }
    #[test]
    fn paired_conflicts_and_skip_preserve_existing_bytes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("report.pdf.llm.md"), "keep").unwrap();
        let mut result = ConversionOutput {
            markdown: "body".into(),
            ..Default::default()
        };
        let mut cfg = config::defaults();
        write(dir.path(), "report.pdf", &mut result, &[], &cfg).unwrap();
        assert!(result.output_path.unwrap().ends_with("report.pdf.v2.md"));
        cfg["output"]["on_conflict"] = json!("skip");
        let mut result = ConversionOutput::default();
        write(dir.path(), "report.pdf", &mut result, &[], &cfg).unwrap();
        assert_eq!(result.skip_reason.as_deref(), Some("exists"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("report.pdf.llm.md")).unwrap(),
            "keep"
        );
    }
    #[test]
    fn immediate_writes_order_each_assets_and_documents_bytes_before_its_name() {
        let root = tempfile::tempdir().unwrap();
        let cfg = config::defaults();
        let mut result = ConversionOutput {
            markdown: "![a](.markitai/assets/a.bin)\n".into(),
            ..Default::default()
        };
        let assets = [Asset {
            name: "a.bin".into(),
            bytes: b"asset bytes".to_vec(),
        }];
        fence::take();
        write(root.path(), "doc", &mut result, &assets, &cfg).unwrap();
        let ordered = fence::expected(root.path());
        // Asset, then document: each fence precedes its own rename.
        assert_eq!(fence::take(), [ordered, "published", ordered, "published"]);
        let mut again = ConversionOutput {
            markdown: "![a](.markitai/assets/a.bin)\n".into(),
            ..Default::default()
        };
        write(root.path(), "doc", &mut again, &assets, &cfg).unwrap();
        // The verified asset is reused; only the renamed document is staged.
        assert_eq!(fence::take(), [ordered, "published"]);
        assert!(again.output_path.unwrap().ends_with("doc.v2.md"));
    }
    #[test]
    fn malformed_frontmatter_remains_content() {
        let text = "---\ninvalid: [\n---\nbody";
        assert_eq!(split_frontmatter(text).1, text);
        assert_eq!(
            split_frontmatter("---\n---extra: kept\n---\nbody").0["---extra"],
            "kept"
        );
        assert_eq!(
            split_frontmatter("---\ntitle: 中文\n---\n\n# Body").0["title"],
            "中文"
        );
    }
    #[test]
    fn output_paths_cannot_escape_directory() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            write(
                dir.path(),
                "../bad",
                &mut ConversionOutput::default(),
                &[],
                &config::defaults()
            )
            .is_err()
        );
    }
}

#[cfg(all(test, unix))]
mod path_check_tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn outcome(result: Result<()>) -> std::result::Result<(), String> {
        result.map_err(|error| error.to_string())
    }

    #[test]
    fn family_check_equals_separate_member_checks() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("real/sub")).unwrap();
        std::fs::write(root.join("real/sub/file.md"), b"x").unwrap();
        symlink(root.join("real"), root.join("link")).unwrap();
        symlink(root.join("real/sub/file.md"), root.join("real/sub/leaf.md")).unwrap();
        let paths = [
            root.join("real/sub/file.md"),
            root.join("real/sub/file.llm.md"),
            root.join("real/sub/leaf.md"),
            root.join("link/sub/file.md"),
            root.join("missing/deeper/file.md"),
            root.join("real/sub/file.md/below"),
        ];
        for first in &paths {
            for second in &paths {
                for allow in [false, true] {
                    let separate =
                        check_path(first, allow).and_then(|()| check_path(second, allow));
                    assert_eq!(
                        outcome(check_paths(&[first, second], allow)),
                        outcome(separate),
                        "{first:?} {second:?} {allow}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_member_leaf_is_never_taken_from_its_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("doc.md");
        let enhanced = dir.path().join("doc.llm.md");
        std::fs::write(&base, b"base").unwrap();
        symlink(&base, &enhanced).unwrap();
        let error = check_paths(&[&base, &enhanced], false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("doc.llm.md"), "{error}");
        let error = check_paths(&[&enhanced, &base], false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("doc.llm.md"), "{error}");
    }

    #[test]
    fn each_family_check_observes_a_replaced_parent_afresh() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("out");
        std::fs::create_dir(&parent).unwrap();
        let members = [parent.join("doc.md"), parent.join("doc.llm.md")];
        let members = [members[0].as_path(), members[1].as_path()];
        check_paths(&members, false).unwrap();
        // Substitute the shared parent with a link to another directory.
        std::fs::rename(&parent, dir.path().join("moved")).unwrap();
        symlink(dir.path().join("moved"), &parent).unwrap();
        let error = check_paths(&members, false).unwrap_err().to_string();
        assert!(error.contains("Symlink access is disabled"), "{error}");
        assert!(check_path(members[1], false).is_err());
        // Restore an ordinary directory: the next call accepts it again.
        std::fs::remove_file(&parent).unwrap();
        std::fs::rename(dir.path().join("moved"), &parent).unwrap();
        check_paths(&members, false).unwrap();
    }

    /// A root-owned system link is accepted above a path but not as its leaf,
    /// even when an earlier member of the same check walked through it.
    #[test]
    fn root_owned_link_exception_is_decided_for_each_member() {
        use std::os::unix::fs::MetadataExt;
        let system = ["/var", "/tmp", "/etc"]
            .into_iter()
            .map(Path::new)
            .find(|path| {
                std::fs::symlink_metadata(path)
                    .is_ok_and(|metadata| metadata.file_type().is_symlink() && metadata.uid() == 0)
            });
        let Some(system) = system else {
            return; // No root-owned top-level link on this host.
        };
        let below = system.join("markitai-absent-member.md");
        check_path(&below, false).unwrap();
        assert!(check_path(system, false).is_err());
        assert!(check_paths(&[&below, system], false).is_err());
        check_paths(&[&below, &below], false).unwrap();
    }
}

#[cfg(test)]
mod prepared_tests {
    use super::*;
    struct NoPublish;
    impl Publication for NoPublish {
        fn skip_existing(&self) -> bool {
            false
        }
        fn publish(&self, _: &Path, _: &[u8]) -> Result<()> {
            panic!("preparation must not acknowledge immediate publication");
        }
    }
    #[test]
    fn prepared_images_wait_for_document_commit_and_keep_current_sidecar_on_failure() {
        for corrupt in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut cfg = config::defaults();
            cfg["image"]["desc_enabled"] = json!(true);
            let mut result = ConversionOutput {
                source: "authored.txt".into(),
                markdown: "![authored](.markitai/assets/image.bin)".into(),
                images: vec![
                    json!({"asset":".markitai/assets/image.bin","description":"authored description"}),
                ],
                ..Default::default()
            };
            let mut plan = PreparedOutput::default();
            write_document_mode(
                root.path(),
                "authored",
                &mut result,
                &[Asset {
                    name: "image.bin".into(),
                    bytes: b"authored pixels".to_vec(),
                }],
                Screenshots::New(&[]),
                &cfg,
                WritePolicy {
                    publication: Some(&NoPublish),
                    prepared: Some(&mut plan),
                },
            )
            .unwrap();
            assert_eq!(plan.members.len(), 1);
            assert!(!plan.members[0].path.exists());
            assert_eq!(result.assets.len(), 1);
            assert!(result.assets[0].is_file());
            let sidecar = root.path().join(".markitai/assets/images.json");
            assert!(!sidecar.exists());
            for member in std::mem::take(&mut plan.members) {
                std::fs::write(member.path, member.bytes).unwrap();
            }
            if corrupt {
                std::fs::write(&sidecar, b"authored invalid prior index").unwrap();
            }
            let finalized = plan.finalize();
            if corrupt {
                assert!(finalized.is_err());
                assert_eq!(
                    std::fs::read(&sidecar).unwrap(),
                    b"authored invalid prior index"
                );
                assert!(result.output_path.unwrap().is_file());
            } else {
                finalized.unwrap();
                let index: Value =
                    serde_json::from_slice(&std::fs::read(sidecar).unwrap()).unwrap();
                assert_eq!(index["images"][0]["description"], "authored description");
                assert_eq!(index["images"][0]["source"], "authored.txt");
            }
        }
    }
}
