//! The document JSON: one conversion result as a versioned object with stable
//! fields ([guide](../../../docs/json-output.md), schema
//! `docs/schema/document-v1.schema.json`).
//!
//! Rules the tests hold it to: every key is always present (a value that does
//! not apply or is unknown is `null`, an empty list `[]`), keys keep the order
//! of the structs below, and within schema version 1 keys are only added.
//! The struct order is the serialization order, so the document is built from
//! structs rather than from `serde_json` maps, which sort their keys.

use crate::notices::{self, Level};
use crate::{ConversionOutput, config};
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use regex::Regex;
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;

/// The schema version every document carries. Minor versions add keys;
/// a removed, renamed or retyped key would be version 2.
pub const SCHEMA_VERSION: &str = "1.0";

/// What the result alone does not say.
#[derive(Clone, Copy, Debug, Default)]
pub struct DocumentJsonContext<'a> {
    /// The source as it should be shown, instead of the path converted (a
    /// server's upload path is not the user's file name).
    pub source_label: Option<&'a str>,
    /// The folder the JSON is written to: output paths are given relative to
    /// it when they are inside it.
    pub base_dir: Option<&'a Path>,
    /// The item's whole duration, when the caller measured more than the
    /// conversion itself.
    pub duration_s: Option<f64>,
}

/// One conversion as the document JSON.
#[derive(Debug, Serialize)]
pub struct DocumentJson {
    schema_version: &'static str,
    tool: Tool,
    source: Source,
    status: &'static str,
    skip_reason: Option<String>,
    title: Option<String>,
    properties: Properties,
    markdown: Bodies,
    text: String,
    pages: Vec<Page>,
    images: Vec<Image>,
    tables: Vec<Table>,
    notices: Vec<Notice>,
    usage: Usage,
    timings: Timings,
    outputs: Outputs,
}

#[derive(Debug, Serialize)]
struct Tool {
    name: &'static str,
    version: &'static str,
    commit: &'static str,
}

#[derive(Debug, Serialize)]
struct Source {
    kind: &'static str,
    path: Option<String>,
    url: Option<String>,
    name: String,
    #[serde(rename = "type")]
    kind_of_content: Option<String>,
    size: Option<u64>,
    sha256: Option<String>,
}

/// Front matter in the order the Markdown file writes it.
#[derive(Debug)]
struct Properties(Map<String, Value>);

impl Serialize for Properties {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for key in crate::output::FIELD_ORDER {
            if let Some(value) = self.0.get(key) {
                map.serialize_entry(key, value)?;
            }
        }
        for (key, value) in &self.0 {
            if !crate::output::FIELD_ORDER.contains(&key.as_str()) {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

#[derive(Debug, Serialize)]
struct Bodies {
    base: String,
    llm: Option<String>,
}

#[derive(Debug, Serialize)]
struct Page {
    number: u64,
    text: String,
    markdown: String,
    images: Vec<String>,
    screenshot: Option<String>,
}

#[derive(Debug, Serialize)]
struct Image {
    path: String,
    alt: String,
    description: Option<String>,
    text: Option<String>,
    page: Option<u64>,
}

#[derive(Debug, Serialize)]
struct Table {
    page: Option<u64>,
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct Notice {
    code: &'static str,
    level: Level,
    message: String,
    items: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Usage {
    requests: u64,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
    cost_complete: bool,
    models: Vec<ModelUsage>,
}

#[derive(Debug, Serialize)]
struct ModelUsage {
    model: String,
    requests: u64,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
}

#[derive(Debug, Serialize)]
struct Timings {
    duration_s: f64,
}

#[derive(Debug, Serialize)]
struct Outputs {
    markdown: Option<String>,
    llm_markdown: Option<String>,
    assets: Vec<String>,
    screenshots: Vec<String>,
}

impl DocumentJson {
    /// The document as written to a file or stdout: indented, with a final
    /// newline.
    pub fn render(&self) -> String {
        let mut text = serde_json::to_string_pretty(self).expect("document JSON serializes");
        text.push('\n');
        text
    }
}

/// Build the document JSON of a finished conversion.
pub fn document_json(result: &ConversionOutput, context: &DocumentJsonContext<'_>) -> DocumentJson {
    let final_body = result.llm_markdown.as_deref().unwrap_or(&result.markdown);
    let paged = result.paged_markdown.as_deref().unwrap_or(final_body);
    let screenshot_names: HashSet<String> = result
        .screenshots
        .iter()
        .filter_map(|path| path.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .collect();
    let segments = segments(paged);
    let mut pages = Vec::new();
    let mut images: Vec<Image> = Vec::new();
    let mut tables = Vec::new();
    let mut shots = BTreeMap::new();
    for segment in &segments {
        for target in screenshot_targets(segment.text, &screenshot_names) {
            let number = page_of_screenshot(&target).or(segment.page);
            if let Some(number) = number {
                shots.entry(number).or_insert(target);
            }
        }
    }
    for segment in &segments {
        let body = without_screenshot_comments(segment.text);
        let scan = scan(&body, &screenshot_names);
        for (path, alt) in &scan.images {
            if images.iter().all(|image| &image.path != path) {
                let analysis = analysis_for(result, path);
                images.push(Image {
                    path: path.clone(),
                    alt: alt.clone(),
                    description: analysis.and_then(|entry| text_field(entry, "desc")),
                    text: analysis.and_then(|entry| text_field(entry, "text")),
                    page: segment.page,
                });
            }
        }
        tables.extend(scan.tables.into_iter().map(|(header, rows)| Table {
            page: segment.page,
            header,
            rows,
        }));
        if let Some(number) = segment.page {
            let markdown = body.trim().to_owned();
            pages.push(Page {
                number,
                text: plain_text(&markdown),
                images: scan.images.into_iter().map(|(path, _)| path).collect(),
                screenshot: shots.get(&number).cloned(),
                markdown,
            });
        }
    }
    let usage = &result.usage;
    let mut models = usage
        .by_model
        .iter()
        .map(|(model, row)| {
            let count = |key: &str| row.get(key).and_then(Value::as_u64).unwrap_or(0);
            ModelUsage {
                model: model.clone(),
                requests: count("requests"),
                input_tokens: count("input_tokens"),
                output_tokens: count("output_tokens"),
                cost_usd: round(
                    row.get("cost_usd").and_then(Value::as_f64).unwrap_or(0.0),
                    1e6,
                ),
            }
        })
        .collect::<Vec<_>>();
    models.sort_by(|a, b| a.model.cmp(&b.model));
    let relative = |path: &Path| shown_path(path, context.base_dir);
    DocumentJson {
        schema_version: SCHEMA_VERSION,
        tool: Tool {
            name: "markitai",
            version: crate::VERSION,
            commit: crate::build_info::COMMIT,
        },
        source: source(&result.source, context.source_label),
        status: if result.skip_reason.is_some() {
            "skipped"
        } else {
            "completed"
        },
        skip_reason: result.skip_reason.clone(),
        title: result
            .frontmatter
            .get("title")
            .and_then(Value::as_str)
            .map(str::to_owned),
        properties: Properties(result.frontmatter.clone()),
        markdown: Bodies {
            base: result.markdown.clone(),
            llm: result.llm_markdown.clone(),
        },
        text: plain_text(&without_screenshot_comments(final_body)),
        pages,
        images,
        tables,
        notices: result
            .warnings
            .iter()
            .map(|message| {
                let found = notices::classify(message);
                Notice {
                    code: found.code,
                    level: found.level,
                    message: message.clone(),
                    items: found.items,
                }
            })
            .collect(),
        usage: Usage {
            requests: usage.requests,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cost_usd: round(usage.cost_usd, 1e6),
            cost_complete: usage.cost_complete(),
            models,
        },
        timings: Timings {
            duration_s: round(context.duration_s.unwrap_or(result.duration), 1e3),
        },
        outputs: Outputs {
            markdown: result.output_path.as_deref().map(relative),
            llm_markdown: result.llm_output_path.as_deref().map(relative),
            assets: result.assets.iter().map(|path| relative(path)).collect(),
            screenshots: result
                .screenshots
                .iter()
                .map(|path| relative(path))
                .collect(),
        },
    }
}

fn round(value: f64, factor: f64) -> f64 {
    let rounded = (value * factor).round() / factor;
    // `-0.0` would print as such.
    if rounded == 0.0 { 0.0 } else { rounded }
}

fn shown_path(path: &Path, base: Option<&Path>) -> String {
    let relative = base
        .and_then(|base| path.strip_prefix(base).ok())
        .filter(|relative| !relative.as_os_str().is_empty());
    match relative {
        Some(relative) => relative
            .components()
            .map(|part| part.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/"),
        None => path.to_string_lossy().into_owned(),
    }
}

fn source(source: &str, label: Option<&str>) -> Source {
    if crate::is_url(source) {
        return Source {
            kind: "url",
            path: None,
            url: Some(crate::output::redact_url(source)),
            name: crate::output::url_name(source),
            kind_of_content: None,
            size: None,
            sha256: None,
        };
    }
    let path = config::expand_home(Path::new(source));
    let shown = label.unwrap_or(source);
    let name = Path::new(shown)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| shown.to_owned());
    let extension = Path::new(&name)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    let content = extension.as_deref().map(|extension| {
        crate::formats::real_extension(&path, extension)
            .map(str::to_owned)
            .unwrap_or_else(|| extension.to_owned())
    });
    let (size, sha256) = measure(&path).unzip();
    Source {
        kind: "file",
        path: Some(shown.to_owned()),
        url: None,
        name,
        kind_of_content: content,
        size,
        sha256,
    }
}

/// Size and SHA-256 of a regular file (a `.numbers` package folder has none).
fn measure(path: &Path) -> Option<(u64, String)> {
    let mut file = std::fs::File::open(path).ok()?;
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 64 * 1024];
    let mut size = 0u64;
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        size += read as u64;
        hasher.update(&buffer[..read]);
    }
    Some((size, crate::hex(hasher.finalize())))
}

fn text_field(entry: &Value, key: &str) -> Option<String> {
    entry
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

/// The image analysis record of the image the Markdown links as `target`.
fn analysis_for<'a>(result: &'a ConversionOutput, target: &str) -> Option<&'a Value> {
    let target = Path::new(target.strip_prefix("./").unwrap_or(target));
    result.images.iter().find(|entry| {
        entry
            .get("asset")
            .and_then(Value::as_str)
            .is_some_and(|asset| Path::new(asset).ends_with(target))
    })
}

/// A run of the document belonging to one page, or to none before the first
/// page marker.
struct Segment<'a> {
    page: Option<u64>,
    text: &'a str,
}

/// The page and slide markers, and the `rag` profile's page provenance
/// (`<!-- page: N -->`) they become in its output.
static MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*<!--\s*(?:(?:Page|Slide) number|page):\s*(\d+)\s*-->\s*$")
        .expect("marker compiles")
});

/// Split at page and slide markers outside fenced code.
fn segments(markdown: &str) -> Vec<Segment<'_>> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut page = None;
    let mut fence: Option<(u8, usize)> = None;
    let mut offset = 0;
    for line in markdown.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let ticks = trimmed.bytes().take_while(|byte| *byte == b'`').count();
        let tildes = trimmed.bytes().take_while(|byte| *byte == b'~').count();
        let run = if ticks >= 3 {
            Some((b'`', ticks))
        } else if tildes >= 3 {
            Some((b'~', tildes))
        } else {
            None
        };
        match (fence, run) {
            (None, Some(open)) => fence = Some(open),
            (Some((kind, length)), Some((found, count)))
                if kind == found && count >= length && trimmed[count..].trim().is_empty() =>
            {
                fence = None
            }
            _ => {}
        }
        if fence.is_none()
            && let Some(found) = MARKER.captures(line.trim_end_matches(['\n', '\r']))
            && let Ok(number) = found[1].parse::<u64>()
        {
            if offset > start || page.is_some() {
                segments.push(Segment {
                    page,
                    text: &markdown[start..offset],
                });
            }
            page = Some(number);
            start = offset + line.len();
        }
        offset += line.len();
    }
    if start < markdown.len() || page.is_some() {
        segments.push(Segment {
            page,
            text: &markdown[start..],
        });
    }
    segments
}

static COMMENT_PICTURE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<!--\s*!\[[^\]]*\]\(([^)\s]+)\)\s*-->").expect("comment picture compiles")
});
static SHOT_PAGE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\.page(\d+)\.[A-Za-z0-9]+$").expect("page name compiles"));

fn is_screenshot(target: &str, names: &HashSet<String>) -> bool {
    target.contains(".markitai/screenshots/")
        || target.starts_with("screenshots/")
        || target
            .rsplit('/')
            .next()
            .is_some_and(|name| names.contains(name))
}

/// Screenshots a page links, in comments (`<!-- ![Page 1](…) -->`) or
/// visibly.
fn screenshot_targets(text: &str, names: &HashSet<String>) -> Vec<String> {
    let mut targets: Vec<String> = COMMENT_PICTURE
        .captures_iter(text)
        .map(|found| found[1].to_owned())
        .filter(|target| is_screenshot(target, names))
        .collect();
    for target in scan(text, names).screenshots {
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    targets
}

fn page_of_screenshot(target: &str) -> Option<u64> {
    SHOT_PAGE
        .captures(target)
        .and_then(|found| found[1].parse().ok())
}

/// A page's Markdown without the comments that only link its screenshot.
fn without_screenshot_comments(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let line = line.trim();
            line != "<!-- Page images for reference -->"
                && !(COMMENT_PICTURE
                    .find(line)
                    .is_some_and(|found| found.start() == 0 && found.end() == line.len()))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn options() -> Options {
    Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS
}

#[derive(Default)]
struct Scan {
    /// `(target, alt)` of each picture, screenshots excluded.
    images: Vec<(String, String)>,
    screenshots: Vec<String>,
    tables: Vec<(Vec<String>, Vec<Vec<String>>)>,
}

/// Pictures and tables of a run of Markdown.
fn scan(markdown: &str, names: &HashSet<String>) -> Scan {
    let mut found = Scan::default();
    let mut picture: Option<(String, String)> = None;
    let mut table: Option<(Vec<String>, Vec<Vec<String>>)> = None;
    let mut row: Vec<String> = Vec::new();
    let mut cell: Option<String> = None;
    for event in Parser::new_ext(markdown, options()) {
        match event {
            Event::Start(Tag::Image { dest_url, .. }) => {
                picture = Some((dest_url.into_string(), String::new()));
            }
            Event::End(TagEnd::Image) => {
                if let Some((target, alt)) = picture.take() {
                    if let Some(cell) = cell.as_mut() {
                        cell.push_str(&alt);
                    }
                    if is_screenshot(&target, names) {
                        found.screenshots.push(target);
                    } else {
                        found.images.push((target, alt));
                    }
                }
            }
            Event::Start(Tag::Table(_)) => table = Some((Vec::new(), Vec::new())),
            Event::End(TagEnd::Table) => {
                if let Some(table) = table.take() {
                    found.tables.push(table);
                }
            }
            Event::End(TagEnd::TableHead) => {
                if let Some((header, _)) = table.as_mut() {
                    *header = std::mem::take(&mut row);
                }
            }
            Event::End(TagEnd::TableRow) => {
                if let Some((_, rows)) = table.as_mut() {
                    rows.push(std::mem::take(&mut row));
                }
            }
            Event::Start(Tag::TableCell) => cell = Some(String::new()),
            Event::End(TagEnd::TableCell) => {
                row.push(cell.take().unwrap_or_default().trim().to_owned());
            }
            Event::Text(text) | Event::Code(text) => {
                if let Some((_, alt)) = picture.as_mut() {
                    alt.push_str(&text);
                } else if let Some(cell) = cell.as_mut() {
                    cell.push_str(&text);
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some(cell) = cell.as_mut() {
                    cell.push(' ');
                }
            }
            _ => {}
        }
    }
    found
}

static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<[^>]*>").expect("tag compiles"));

/// The text of Markdown without its markup: no pictures, links reduced to
/// their text, table cells separated by tabs, blocks by a blank line.
pub(crate) fn plain_text(markdown: &str) -> String {
    let mut text = String::new();
    let mut picture = 0usize;
    let mut first_cell = true;
    let block_end = |text: &mut String| {
        let kept = text.trim_end_matches([' ', '\t']).len();
        text.truncate(kept);
        if !text.is_empty() && !text.ends_with("\n\n") {
            text.push_str(if text.ends_with('\n') { "\n" } else { "\n\n" });
        }
    };
    for event in Parser::new_ext(markdown, options()) {
        match event {
            Event::Start(Tag::Image { .. }) => picture += 1,
            Event::End(TagEnd::Image) => picture = picture.saturating_sub(1),
            _ if picture > 0 => {}
            Event::Text(part) | Event::Code(part) => text.push_str(&part),
            Event::SoftBreak | Event::HardBreak => text.push('\n'),
            Event::Html(html) | Event::InlineHtml(html) => {
                let html = html.trim_start();
                if !html.starts_with("<!--") {
                    let stripped = TAG.replace_all(html, "");
                    let stripped = stripped.trim();
                    if !stripped.is_empty() {
                        text.push_str(stripped);
                        text.push(' ');
                    }
                }
            }
            Event::Start(Tag::TableRow | Tag::TableHead) => first_cell = true,
            Event::Start(Tag::TableCell) => {
                if !first_cell {
                    text.push('\t');
                }
                first_cell = false;
            }
            Event::End(TagEnd::TableRow | TagEnd::TableHead) => text.push('\n'),
            Event::End(TagEnd::Item) => {
                if !text.ends_with('\n') {
                    text.push('\n');
                }
            }
            Event::End(
                TagEnd::Paragraph
                | TagEnd::Heading(_)
                | TagEnd::CodeBlock
                | TagEnd::BlockQuote(_)
                | TagEnd::List(_)
                | TagEnd::Table
                | TagEnd::HtmlBlock,
            ) => block_end(&mut text),
            Event::Rule => block_end(&mut text),
            _ => {}
        }
    }
    let mut lines: Vec<&str> = Vec::new();
    for line in text.lines().map(str::trim_end) {
        if !(line.is_empty() && lines.last().is_none_or(|last| last.is_empty())) {
            lines.push(line);
        }
    }
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests;
