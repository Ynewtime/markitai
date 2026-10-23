//! Originals for the workbench's Compare view: what can be shown of an item's
//! source (its kept upload, a URL item's page snapshot, or the conversion's
//! page screenshots, else the item's compare screenshots), its stored bytes,
//! page images, a PDF page's text and a PDF's outline, and where the
//! screenshots show the page's blocks. Pages are drawn on blocking threads,
//! at most two at a time.
use super::{
    State, files,
    jobs::{self, Job},
    snapshots, store,
    types::{ApiError, ApiResult, Item},
};
use axum::{
    Json,
    body::Bytes,
    extract::{Path, Query, State as ExtractState, rejection::QueryRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use markitai_core::preview::{self, PageSize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    fs::{self, File},
    io::Read,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Pages drawn at once; another request is 429 `render_busy`.
const RENDERS: usize = 2;
/// Page boxes a descriptor lists at most; `page_count` has the total.
const MAX_PAGES: usize = 2000;
const MIN_WIDTH: u32 = 160;
const MAX_WIDTH: u32 = 2400;
const DEFAULT_WIDTH: u32 = 1200;
/// Rendered pages and page texts kept in memory, in bytes.
const CACHE_BYTES: usize = 32 * 1024 * 1024;
/// The largest original read for drawing; uploads are at most 100 MiB.
const MAX_READ: u64 = 500 * 1024 * 1024;
/// Bytes of a screenshot read to measure it from its header.
const HEADER_BYTES: u64 = 256 * 1024;
/// A stored original never runs in the service's origin, even if visited.
const SANDBOX: &str = "sandbox; default-src 'none'";

/// Page qualities computed for kept pages without a record, kept in memory.
const QUALITIES: usize = 256;

/// Render slots and a small cache of drawn pages, shared by every job.
pub(super) struct Previews {
    renders: Arc<Semaphore>,
    cache: Mutex<VecDeque<(String, Bytes, &'static str)>>,
    /// How older jobs' kept pages read, by job, item and page version.
    qualities: Mutex<VecDeque<(String, Value)>>,
}
impl Default for Previews {
    fn default() -> Self {
        Self {
            renders: Arc::new(Semaphore::new(RENDERS)),
            cache: Mutex::new(VecDeque::new()),
            qualities: Mutex::new(VecDeque::new()),
        }
    }
}
impl Previews {
    /// How a kept page reads, judged once per page version.
    fn page_quality(&self, key: String, judge: impl FnOnce() -> Value) -> Value {
        if let Some((_, value)) = self
            .qualities
            .lock()
            .unwrap()
            .iter()
            .find(|(entry, _)| *entry == key)
        {
            return value.clone();
        }
        let value = judge();
        let mut qualities = self.qualities.lock().unwrap();
        qualities.retain(|(entry, _)| *entry != key);
        qualities.push_back((key, value.clone()));
        while qualities.len() > QUALITIES {
            qualities.pop_front();
        }
        value
    }
    fn cached(&self, key: &str) -> Option<(Bytes, &'static str)> {
        let mut cache = self.cache.lock().unwrap();
        let index = cache.iter().position(|(entry, _, _)| entry == key)?;
        let entry = cache.remove(index)?;
        let found = (entry.1.clone(), entry.2);
        cache.push_back(entry);
        Some(found)
    }
    fn keep(&self, key: String, bytes: Bytes, media: &'static str) {
        if bytes.len() > CACHE_BYTES / 4 {
            return;
        }
        let mut cache = self.cache.lock().unwrap();
        cache.retain(|(entry, _, _)| *entry != key);
        cache.push_back((key, bytes, media));
        while cache.iter().map(|(_, bytes, _)| bytes.len()).sum::<usize>() > CACHE_BYTES {
            cache.pop_front();
        }
    }
    fn slot(&self) -> Option<OwnedSemaphorePermit> {
        self.renders.clone().try_acquire_owned().ok()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Pdf,
    /// A raster image; `viewable` when browsers show it as stored.
    Image {
        viewable: bool,
    },
    Html,
    Text,
    Other,
}

/// What an original of this name is, and its media type.
fn format(name: &str) -> (Format, &'static str) {
    let extension = FsPath::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let image = |viewable, media| (Format::Image { viewable }, media);
    match extension.as_str() {
        "pdf" => (Format::Pdf, "application/pdf"),
        "png" => image(true, "image/png"),
        "jpg" | "jpeg" => image(true, "image/jpeg"),
        "gif" => image(true, "image/gif"),
        "webp" => image(true, "image/webp"),
        "avif" => image(true, "image/avif"),
        "bmp" => image(true, "image/bmp"),
        "tif" | "tiff" => image(false, "image/tiff"),
        "heic" => image(false, "image/heic"),
        "heif" => image(false, "image/heif"),
        "html" | "htm" => (Format::Html, "text/html"),
        "xhtml" => (Format::Html, "application/xhtml+xml"),
        "md" | "markdown" => (Format::Text, "text/markdown"),
        "txt" | "log" => (Format::Text, "text/plain"),
        "csv" => (Format::Text, "text/csv"),
        "tsv" => (Format::Text, "text/tab-separated-values"),
        "json" => (Format::Text, "application/json"),
        "xml" => (Format::Text, "application/xml"),
        "yaml" | "yml" => (Format::Text, "application/yaml"),
        "svg" => (Format::Other, "image/svg+xml"),
        "docx" => (
            Format::Other,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        "xlsx" => (
            Format::Other,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        "pptx" => (
            Format::Other,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        "doc" => (Format::Other, "application/msword"),
        "xls" => (Format::Other, "application/vnd.ms-excel"),
        "ppt" => (Format::Other, "application/vnd.ms-powerpoint"),
        "epub" => (Format::Other, "application/epub+zip"),
        "eml" => (Format::Other, "message/rfc822"),
        "msg" => (Format::Other, "application/vnd.ms-outlook"),
        "ipynb" => (Format::Other, "application/x-ipynb+json"),
        "zip" => (Format::Other, "application/zip"),
        _ => (Format::Other, "application/octet-stream"),
    }
}

/// The type a stored original is sent with: markup never as itself, so a
/// direct visit cannot render it.
fn sent_type(format: Format, media: &'static str) -> &'static str {
    match (format, media) {
        (Format::Html, _) | (_, "image/svg+xml") => "text/plain; charset=utf-8",
        (_, "text/markdown") => "text/markdown; charset=utf-8",
        (_, "text/plain") => "text/plain; charset=utf-8",
        (_, "text/csv") => "text/csv; charset=utf-8",
        (_, "text/tab-separated-values") => "text/tab-separated-values; charset=utf-8",
        _ => media,
    }
}

/// The stored original: an item's kept upload, or a URL item's page snapshot.
#[derive(Clone)]
struct Stored {
    path: PathBuf,
    size: u64,
    /// Changes whenever a retry replaces it; part of cache keys and ETags.
    version: String,
    format: Format,
    media: &'static str,
    filename: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Pdf,
    Image,
    Html,
    Text,
    Screenshots,
    None,
}
impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Image => "image",
            Self::Html => "html",
            Self::Text => "text",
            Self::Screenshots => "screenshots",
            Self::None => "none",
        }
    }
}

/// What an item's Compare view can show, before anything is decoded.
struct Original {
    job_id: String,
    item_id: String,
    name: String,
    /// A URL item's address; `None` for an upload.
    url: Option<String>,
    stored: Option<Stored>,
    /// The conversion's page screenshots, else a URL item's compare screenshots.
    screenshots: Vec<(PathBuf, String)>,
    kind: Kind,
    reason: Option<&'static str>,
    /// A URL item's `capture.json`.
    capture: Option<PathBuf>,
    /// A URL item's `blocks.json`.
    blocks: Option<PathBuf>,
    /// The job's folder.
    folder: PathBuf,
}

fn version(metadata: &fs::Metadata) -> String {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |time| time.as_nanos());
    format!("{}-{modified}", metadata.len())
}

fn stored(path: PathBuf, name: &str, filename: String) -> Option<Stored> {
    let metadata = fs::metadata(&path).ok()?;
    let (format, media) = format(name);
    Some(Stored {
        size: metadata.len(),
        version: version(&metadata),
        path,
        format,
        media,
        filename,
    })
}

fn item_missing() -> ApiError {
    ApiError::new(404, "item_not_found", "item not found")
}

/// The stored original of `item`: its kept upload, or a URL item's page
/// snapshot. The caller holds the job's access lock.
fn stored_original(job: &Job, data: &jobs::JobData, item: &Item) -> Option<Stored> {
    if item.kind == "file" {
        store::safe_file(&job.folder.join("uploads"), &item.name)
            .ok()
            .and_then(|path| stored(path, &item.name, item.name.clone()))
    } else {
        let base = files::item_base(data, item).unwrap_or_else(|_| "page".into());
        snapshot_name(&item.item_id)
            .and_then(|name| store::safe_file(&job.folder.join("snapshots"), &name).ok())
            .and_then(|path| stored(path, "page.html", format!("{base}.html")))
    }
}

/// The kept original of an item, as `…/original/file` sends it: its path
/// and file name. The caller holds the job's access lock.
pub(super) fn kept(job: &Job, item_id: &str) -> Option<(PathBuf, String)> {
    let data = job.data.lock().unwrap().clone();
    let item = data.items.iter().find(|item| item.item_id == item_id)?;
    stored_original(job, &data, item).map(|stored| (stored.path, stored.filename))
}

/// Find an item's original. Holds the job's access lock, as every reader of
/// its files does, so a retry's publication is never seen half done.
fn resolve(job: &Job, item_id: &str) -> ApiResult<Original> {
    let _access = job.access.lock().unwrap();
    let (data, item) = {
        let data = job.data.lock().unwrap();
        let item = data
            .items
            .iter()
            .find(|item| item.item_id == item_id)
            .ok_or_else(item_missing)?
            .clone();
        (data.clone(), item)
    };
    let stored = stored_original(job, &data, &item);
    let out = job.folder.join("out");
    let screenshot = |name: &String| name.starts_with(".markitai/screenshots/");
    let names: Vec<String> = match data.assets.get(item_id) {
        Some(assets) => assets
            .iter()
            .filter(|name| screenshot(name))
            .cloned()
            .collect(),
        None => {
            let mut owned: Vec<String> = files::owned_files(&job.folder, &data, &item)
                .map(|names| names.into_iter().filter(screenshot).collect())
                .unwrap_or_default();
            owned.sort();
            owned
        }
    };
    let mut screenshots = names
        .into_iter()
        .filter_map(|name| Some((store::safe_file(&out, &name).ok()?, name)))
        .collect::<Vec<_>>();
    let url = item.kind == "url";
    if screenshots.is_empty() && url {
        screenshots = snapshots::shots(&job.folder, item_id);
    }
    let kept = |pick| {
        url.then(|| snapshots::kept(&job.folder, item_id, pick))
            .flatten()
    };
    let capture = kept(|names| &names.capture);
    let blocks = kept(|names| &names.blocks);
    let (kind, reason) = match stored.as_ref().map(|stored| stored.format) {
        Some(Format::Pdf) => (Kind::Pdf, None),
        Some(Format::Image { .. }) => (Kind::Image, None),
        Some(Format::Html) => (Kind::Html, None),
        Some(Format::Text) => (Kind::Text, None),
        _ if !screenshots.is_empty() => (Kind::Screenshots, None),
        Some(Format::Other) => (Kind::None, Some("unsupported_format")),
        None => (Kind::None, Some("not_kept")),
    };
    Ok(Original {
        job_id: data.id,
        item_id: item_id.to_owned(),
        url: url.then(|| item.name.clone()),
        name: item.name,
        stored,
        screenshots,
        kind,
        reason,
        capture,
        blocks,
        folder: job.folder.clone(),
    })
}

/// A URL item's page snapshot inside the job's `snapshots/`; none for an
/// item ID (from an older record) that is not a plain file name.
pub(super) fn snapshot_name(item_id: &str) -> Option<String> {
    snapshots::names(item_id).map(|names| names.html)
}

fn read(path: &FsPath, limit: u64) -> ApiResult<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|file| file.take(limit).read_to_end(&mut bytes))
        .map_err(ApiError::internal)?;
    Ok(bytes)
}

fn unavailable() -> ApiError {
    ApiError::new(
        404,
        "original_unavailable",
        "no original of this item is kept",
    )
}
fn no_page() -> ApiError {
    ApiError::new(404, "page_not_found", "page not found")
}
fn render_failed(error: impl std::fmt::Display) -> ApiError {
    eprintln!("Warning: Serve: page preview failed: {error}");
    ApiError::new(422, "render_failed", "the page cannot be drawn")
}
fn busy() -> Response {
    let mut response = ApiError::new(
        429,
        "render_busy",
        "pages are being drawn; try again shortly",
    )
    .into_response();
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    response
}

/// The headers every original response carries.
fn guard(response: &mut Response, filename: &str) {
    let headers = response.headers_mut();
    if let Ok(value) = HeaderValue::from_str(&format!(
        "attachment; filename*=UTF-8''{}",
        files::encoded(filename)
    )) {
        headers.insert(header::CONTENT_DISPOSITION, value);
    }
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(SANDBOX),
    );
}

fn download_path(original: &Original) -> String {
    original_path(original, "file")
}
fn original_path(original: &Original, route: &str) -> String {
    format!(
        "/api/jobs/{}/items/{}/original/{route}",
        files::encoded(&original.job_id),
        files::encoded(&original.item_id)
    )
}

impl Original {
    /// Whether the page screenshots stand in for pages: kind `screenshots`,
    /// or a URL item's kept page that also has screenshots to fall back to.
    fn screenshot_pages(&self) -> bool {
        self.kind == Kind::Screenshots
            || self.kind == Kind::Html && self.url.is_some() && !self.screenshots.is_empty()
    }
}

fn page(size: PageSize) -> Value {
    json!({"width":size.width,"height":size.height})
}

/// A screenshot's size from its header, else from the whole file.
fn screenshot_size(path: &FsPath) -> PageSize {
    read(path, HEADER_BYTES)
        .ok()
        .and_then(|bytes| preview::image_size(&bytes).ok())
        .or_else(|| {
            read(path, MAX_READ)
                .ok()
                .and_then(|bytes| preview::image_size(&bytes).ok())
        })
        .unwrap_or(PageSize {
            width: 0.,
            height: 0.,
        })
}

/// The largest kept page judged for an older job without a record.
const MAX_JUDGED: u64 = 10 * 1024 * 1024;

/// How a URL item's views read when they were captured: the job's record,
/// else (an older job) its kept page judged now, without a screenshot.
fn capture(original: &Original, previews: &Previews) -> Value {
    let Some(url) = &original.url else {
        return Value::Null;
    };
    if let Some((page, screenshot)) = original.capture.as_deref().and_then(snapshots::recorded) {
        return json!({"page": page, "screenshot": screenshot});
    }
    let page = match &original.stored {
        Some(stored) if stored.size <= MAX_JUDGED => {
            let key = format!(
                "{}\u{0}{}\u{0}{}",
                original.job_id, original.item_id, stored.version
            );
            previews.page_quality(key, || {
                read(&stored.path, MAX_JUDGED)
                    .ok()
                    .map(|bytes| {
                        json!(markitai_core::page_capture::page_quality(
                            url,
                            &String::from_utf8_lossy(&bytes)
                        ))
                    })
                    .unwrap_or(Value::Null)
            })
        }
        _ => Value::Null,
    };
    json!({"page": page, "screenshot": null})
}

/// The descriptor; reads page boxes, never draws a page.
fn describe(original: Original, previews: &Previews) -> Value {
    let mut kind = original.kind;
    let mut reason = original.reason;
    let mut pages = Vec::new();
    let mut count = 0;
    match (kind, &original.stored) {
        (Kind::Pdf, Some(stored)) => {
            let opened = read(&stored.path, MAX_READ)
                .ok()
                .and_then(|bytes| preview::PdfPreview::open(&bytes).ok());
            match opened {
                Some(pdf) => {
                    count = pdf.page_count();
                    for number in 1..=count.min(MAX_PAGES) {
                        match pdf.page_size(number) {
                            Ok(size) => pages.push(page(size)),
                            Err(_) => pages.push(json!({"width":0,"height":0})),
                        }
                    }
                }
                None => (kind, reason) = (Kind::None, Some("render_unavailable")),
            }
        }
        (Kind::Image, Some(stored)) => {
            match read(&stored.path, MAX_READ)
                .ok()
                .and_then(|bytes| preview::image_size(&bytes).ok())
            {
                Some(size) => {
                    count = 1;
                    pages.push(page(size));
                }
                None => (kind, reason) = (Kind::None, Some("render_unavailable")),
            }
        }
        _ if original.screenshot_pages() => {
            count = original.screenshots.len();
            pages = original
                .screenshots
                .iter()
                .take(MAX_PAGES)
                .map(|(path, _)| page(screenshot_size(path)))
                .collect();
        }
        _ => {}
    }
    let capture = capture(&original, previews);
    let blocks = (original.blocks.is_some() && original.screenshot_pages())
        .then(|| original_path(&original, "blocks"));
    let stored = original.stored.as_ref();
    json!({
        "capture": capture,
        "blocks": blocks,
        "kind": kind.name(),
        "name": original.name,
        "media_type": stored.map(|stored| stored.media),
        "size": stored.map(|stored| stored.size),
        "download": stored.map(|_| download_path(&original)),
        "view": (kind == Kind::Html).then(|| original_path(&original, "view")),
        "pages": pages,
        "page_count": count,
        "text_layer": kind == Kind::Pdf,
        "reason": reason,
    })
}

async fn original(state: &State, id: &str, item_id: String) -> ApiResult<Original> {
    let job = jobs::get(state, id)?;
    crate::task::blocking(move || resolve(&job, &item_id))
        .await
        .map_err(ApiError::internal)?
}

/// `GET /api/jobs/{job_id}/items/{item_id}/original`.
pub(super) async fn descriptor(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
) -> ApiResult<Response> {
    let original = original(&state, &id, item_id).await?;
    // Measuring a PDF or a HEIF image parses it: one of the render slots,
    // waited for rather than refused.
    let _slot = state
        .previews
        .renders
        .clone()
        .acquire_owned()
        .await
        .map_err(ApiError::internal)?;
    let shared = state.clone();
    let value = crate::task::blocking(move || describe(original, &shared.previews))
        .await
        .map_err(ApiError::internal)?;
    let mut response = Json(value).into_response();
    guard(&mut response, "original.json");
    Ok(response)
}

/// `GET /api/jobs/{job_id}/items/{item_id}/original/file`.
pub(super) async fn file(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
) -> ApiResult<Response> {
    let original = original(&state, &id, item_id).await?;
    let stored = original.stored.ok_or_else(unavailable)?;
    let opened = File::open(&stored.path).map_err(|_| unavailable())?;
    let mut response = files::body(
        opened,
        None,
        sent_type(stored.format, stored.media),
        &stored.filename,
    )?;
    guard(&mut response, &stored.filename);
    Ok(response)
}

/// `GET /api/jobs/{job_id}/items/{item_id}/original/blocks`: where the
/// item's screenshots show its page's blocks, as kept when they were taken.
pub(super) async fn blocks(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
) -> ApiResult<Response> {
    let original = original(&state, &id, item_id).await?;
    let path = original.blocks.ok_or_else(unavailable)?;
    let limit = markitai_core::page_capture::MAX_BLOCKS_BYTES as u64;
    let bytes = crate::task::blocking(move || read(&path, limit + 1))
        .await
        .map_err(ApiError::internal)?
        .map_err(|_| unavailable())?;
    if bytes.len() as u64 > limit {
        return Err(unavailable());
    }
    let mut response = ([(header::CONTENT_TYPE, "application/json")], bytes).into_response();
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    guard(&mut response, "blocks.json");
    Ok(response)
}

/// The largest HTML original `view` frames.
const MAX_VIEW: u64 = 10 * 1024 * 1024;
/// The policy of a framed HTML original: no script and no form; the site's
/// own pictures, style sheets, fonts and media load over HTTPS (inline styles
/// and data URLs too), nothing else is fetched; links open outside it, and
/// only the workbench may frame it. The frame keeps the workbench's origin
/// (`allow-same-origin`, so the workbench can read the page it shows), which
/// is safe only because nothing in it can run: there is no `allow-scripts`
/// and no `script-src`, so `default-src 'none'` refuses every script, inline
/// handler and `javascript:` URL.
const VIEW_POLICY: &str = "sandbox allow-same-origin allow-popups allow-popups-to-escape-sandbox; default-src 'none'; img-src https: data:; style-src https: 'unsafe-inline'; font-src https: data:; media-src https:; form-action 'none'; frame-ancestors 'self'";

/// HTML-attribute escaping of `&`, `<`, `>`, `"` and `'`.
fn attribute(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Where the first `<name…>` start tag ends (just after its `>`): `<name`
/// matched ASCII case-insensitively and followed by whitespace, `>` or `/`.
fn after_start_tag(html: &str, name: &str) -> Option<usize> {
    let bytes = html.as_bytes();
    let open = format!("<{name}");
    let mut from = 0;
    while let Some(found) = find_caseless(&bytes[from..], open.as_bytes()) {
        let start = from + found;
        let next = start + open.len();
        match bytes.get(next) {
            Some(byte) if byte.is_ascii_whitespace() || matches!(byte, b'>' | b'/') => {
                return bytes[next..]
                    .iter()
                    .position(|byte| *byte == b'>')
                    .map(|end| next + end + 1);
            }
            Some(_) => from = next,
            None => return None,
        }
    }
    None
}
fn find_caseless(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle))
}

/// The kept page with one insertion: no DNS prefetching, and a base that
/// resolves relative links against the page's address (for a URL item) and
/// opens them outside the frame. It goes after the first `<head>` tag, else
/// after the first `<html>` tag, else at the start after a leading doctype.
fn framed(html: &str, url: Option<&str>) -> String {
    let base = match url {
        Some(url) => format!("<base href=\"{}\" target=\"_blank\">", attribute(url)),
        None => "<base target=\"_blank\">".to_owned(),
    };
    let insertion = format!("<meta http-equiv=\"x-dns-prefetch-control\" content=\"off\">{base}");
    let at = after_start_tag(html, "head")
        .or_else(|| after_start_tag(html, "html"))
        .unwrap_or_else(|| {
            let lead = html.len()
                - html
                    .trim_start_matches(['\u{feff}', ' ', '\t', '\n', '\r', '\x0c'])
                    .len();
            let rest = &html.as_bytes()[lead..];
            if rest.len() >= 9 && rest[..9].eq_ignore_ascii_case(b"<!doctype") {
                rest.iter()
                    .position(|byte| *byte == b'>')
                    .map_or(lead, |end| lead + end + 1)
            } else {
                html.len() - html.trim_start_matches('\u{feff}').len()
            }
        });
    let mut framed = String::with_capacity(html.len() + insertion.len());
    framed.push_str(&html[..at]);
    framed.push_str(&insertion);
    framed.push_str(&html[at..]);
    framed
}

/// `GET /api/jobs/{job_id}/items/{item_id}/original/view`: a kept HTML
/// original for a sandboxed frame of the workbench.
pub(super) async fn view(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
) -> ApiResult<Response> {
    let original = original(&state, &id, item_id).await?;
    let stored = match (original.kind, original.stored) {
        (Kind::Html, Some(stored)) => stored,
        _ => return Err(unavailable()),
    };
    if stored.size > MAX_VIEW {
        return Err(ApiError::new(
            413,
            "result_too_large",
            "the page is too large to show; download the original instead",
        ));
    }
    let url = original.url;
    let page = crate::task::blocking(move || -> ApiResult<String> {
        let bytes = read(&stored.path, MAX_VIEW + 1).map_err(|_| unavailable())?;
        if bytes.len() as u64 > MAX_VIEW {
            return Err(ApiError::new(
                413,
                "result_too_large",
                "the page is too large to show; download the original instead",
            ));
        }
        Ok(framed(&String::from_utf8_lossy(&bytes), url.as_deref()))
    })
    .await
    .map_err(ApiError::internal)??;
    let mut response = page.into_response();
    let headers = response.headers_mut();
    for (name, value) in [
        ("content-type", "text/html; charset=utf-8"),
        ("content-security-policy", VIEW_POLICY),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
        ("x-dns-prefetch-control", "off"),
        ("cache-control", "no-store"),
        ("content-disposition", "inline"),
    ] {
        headers.insert(name, HeaderValue::from_static(value));
    }
    Ok(response)
}

#[derive(serde::Deserialize)]
pub(super) struct PageQuery {
    width: Option<String>,
}

fn invalid_page() -> ApiError {
    ApiError::new(
        422,
        "invalid_page_request",
        "the page number and width must be whole numbers",
    )
}
fn page_number(text: &str) -> ApiResult<usize> {
    let number = text.parse::<usize>().map_err(|_| invalid_page())?;
    if number == 0 {
        return Err(no_page());
    }
    Ok(number)
}
fn page_width(query: Result<Query<PageQuery>, QueryRejection>) -> ApiResult<u32> {
    match query.map_err(|_| invalid_page())?.0.width {
        None => Ok(DEFAULT_WIDTH),
        Some(text) => text
            .parse::<u64>()
            .map(|width| width.clamp(u64::from(MIN_WIDTH), u64::from(MAX_WIDTH)) as u32)
            .map_err(|_| invalid_page()),
    }
}

fn etag(key: &str) -> String {
    format!(
        "\"{}\"",
        &markitai_core::hex(Sha256::digest(key.as_bytes()))[..32]
    )
}
fn matches(headers: &HeaderMap, tag: &str) -> bool {
    headers
        .get_all(header::IF_NONE_MATCH)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|candidate| candidate.trim().trim_start_matches("W/") == tag)
}
fn cacheable(bytes: Bytes, media: &str, tag: &str, filename: &str) -> Response {
    let mut response = ([(header::CONTENT_TYPE, media.to_owned())], bytes).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, max-age=86400"),
    );
    if let Ok(value) = HeaderValue::from_str(tag) {
        headers.insert(header::ETAG, value);
    }
    guard(&mut response, filename);
    response
}
fn not_modified(tag: &str) -> Response {
    let mut response = StatusCode::NOT_MODIFIED.into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, max-age=86400"),
    );
    if let Ok(value) = HeaderValue::from_str(tag) {
        headers.insert(header::ETAG, value);
    }
    response
}

/// What to send for one page, decided without drawing it.
enum Plan {
    Pdf(Stored),
    Image(Stored),
    Screenshot(PathBuf, &'static str),
}

/// `GET /api/jobs/{job_id}/items/{item_id}/original/pages/{n}?width=`.
pub(super) async fn page_image(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id, number)): Path<(String, String, String)>,
    query: Result<Query<PageQuery>, QueryRejection>,
    headers: HeaderMap,
) -> ApiResult<Response> {
    let number = page_number(&number)?;
    let width = page_width(query)?;
    let original = original(&state, &id, item_id).await?;
    let (plan, version) = match (original.kind, original.stored.clone()) {
        (Kind::Pdf, Some(stored)) => {
            let version = stored.version.clone();
            (Plan::Pdf(stored), version)
        }
        (Kind::Image, Some(stored)) if number == 1 => {
            let version = stored.version.clone();
            (Plan::Image(stored), version)
        }
        _ if original.screenshot_pages() => {
            let (path, name) = original
                .screenshots
                .get(number - 1)
                .cloned()
                .ok_or_else(no_page)?;
            let metadata = fs::metadata(&path).map_err(|_| no_page())?;
            let (_, media) = format(&name);
            (Plan::Screenshot(path, media), version(&metadata))
        }
        (Kind::None, _) => return Err(unavailable()),
        _ => return Err(no_page()),
    };
    let key = format!(
        "{}\u{0}{}\u{0}{}\u{0}page\u{0}{number}\u{0}{width}",
        id, original.item_id, version
    );
    let tag = etag(&key);
    let filename = format!("page-{number}.png");
    if matches(&headers, &tag) {
        return Ok(not_modified(&tag));
    }
    if let Some((bytes, media)) = state.previews.cached(&key) {
        return Ok(cacheable(bytes, media, &tag, &filename));
    }
    let Some(slot) = state.previews.slot() else {
        return Ok(busy());
    };
    let drawn = crate::task::blocking(move || -> ApiResult<(Vec<u8>, &'static str)> {
        let _slot = slot;
        match plan {
            Plan::Pdf(stored) => {
                let bytes = read(&stored.path, MAX_READ)?;
                let pdf = preview::PdfPreview::open(&bytes).map_err(render_failed)?;
                if number > pdf.page_count() {
                    return Err(no_page());
                }
                let png = pdf.render_png(number, width).map_err(render_failed)?;
                Ok((png, "image/png"))
            }
            Plan::Image(stored) => {
                let bytes = read(&stored.path, MAX_READ)?;
                let size = preview::image_size(&bytes).map_err(render_failed)?;
                let viewable = matches!(stored.format, Format::Image { viewable: true });
                if viewable && size.width <= f64::from(width) {
                    return Ok((bytes, stored.media));
                }
                let png = preview::image_png(&bytes, width).map_err(render_failed)?;
                Ok((png, "image/png"))
            }
            Plan::Screenshot(path, media) => {
                let bytes = read(&path, MAX_READ)?;
                let size = preview::image_size(&bytes).map_err(render_failed)?;
                if size.width <= f64::from(width) {
                    return Ok((bytes, media));
                }
                let png = preview::image_png(&bytes, width).map_err(render_failed)?;
                Ok((png, "image/png"))
            }
        }
    })
    .await
    .map_err(ApiError::internal)??;
    let (bytes, media) = drawn;
    let bytes = Bytes::from(bytes);
    state.previews.keep(key, bytes.clone(), media);
    Ok(cacheable(bytes, media, &tag, &filename))
}

/// `GET /api/jobs/{job_id}/items/{item_id}/original/pages/{n}/text`.
pub(super) async fn page_text(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id, number)): Path<(String, String, String)>,
) -> ApiResult<Response> {
    let number = page_number(&number)?;
    let original = original(&state, &id, item_id).await?;
    let stored = match (original.kind, original.stored) {
        (Kind::Pdf, Some(stored)) => stored,
        _ => return Err(unavailable()),
    };
    // A page local OCR read answers with the recognizer's lines.
    let ocr = snapshots::ocr_version(&original.folder, &original.item_id);
    let key = format!(
        "{}\u{0}{}\u{0}{}\u{0}text\u{0}{number}\u{0}{}",
        id,
        original.item_id,
        stored.version,
        ocr.as_deref().unwrap_or("")
    );
    let filename = format!("page-{number}.json");
    let respond = |bytes: Bytes| {
        let mut response = ([(header::CONTENT_TYPE, "application/json")], bytes).into_response();
        guard(&mut response, &filename);
        response
    };
    if let Some((bytes, _)) = state.previews.cached(&key) {
        return Ok(respond(bytes));
    }
    let Some(slot) = state.previews.slot() else {
        return Ok(busy());
    };
    let (folder, item_id) = (original.folder.clone(), original.item_id.clone());
    let text = crate::task::blocking(move || -> ApiResult<Vec<u8>> {
        let _slot = slot;
        if ocr.is_some()
            && let Some(mut page) = snapshots::ocr_page(&folder, &item_id, number)
        {
            page["source"] = json!("ocr");
            return serde_json::to_vec(&page).map_err(ApiError::internal);
        }
        let bytes = read(&stored.path, MAX_READ)?;
        let pdf = preview::PdfPreview::open(&bytes).map_err(render_failed)?;
        if number > pdf.page_count() {
            return Err(no_page());
        }
        let size = pdf.page_size(number).map_err(render_failed)?;
        let layout = preview::pdf_page_layout(&bytes, number, size).map_err(render_failed)?;
        let lines: Vec<Value> = layout
            .lines
            .into_iter()
            .map(|line| json!({"text":line.text,"bbox":line.bbox}))
            .collect();
        let images: Vec<Value> = layout
            .images
            .into_iter()
            .map(|bbox| json!({"bbox":bbox}))
            .collect();
        serde_json::to_vec(&json!({
            "page": number,
            "width": size.width,
            "height": size.height,
            "source": "text",
            "lines": lines,
            "images": images,
        }))
        .map_err(ApiError::internal)
    })
    .await
    .map_err(ApiError::internal)??;
    let bytes = Bytes::from(text);
    state.previews.keep(key, bytes.clone(), "application/json");
    Ok(respond(bytes))
}

/// `GET /api/jobs/{job_id}/items/{item_id}/original/outline`: a PDF's own
/// outline (bookmarks) as titles, 1-based pages and levels; `items` is empty
/// when it has none. Read once per stored version and kept with the pages.
pub(super) async fn outline(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
) -> ApiResult<Response> {
    let original = original(&state, &id, item_id).await?;
    let stored = match (original.kind, original.stored) {
        (Kind::Pdf, Some(stored)) => stored,
        _ => return Err(unavailable()),
    };
    let key = format!(
        "{}\u{0}{}\u{0}{}\u{0}outline",
        id, original.item_id, stored.version
    );
    let respond = |bytes: Bytes| {
        let mut response = ([(header::CONTENT_TYPE, "application/json")], bytes).into_response();
        guard(&mut response, "outline.json");
        response
    };
    if let Some((bytes, _)) = state.previews.cached(&key) {
        return Ok(respond(bytes));
    }
    // Parsing a large PDF is work like measuring it: a render slot, waited for.
    let slot = state
        .previews
        .renders
        .clone()
        .acquire_owned()
        .await
        .map_err(ApiError::internal)?;
    let body = crate::task::blocking(move || -> ApiResult<Vec<u8>> {
        let _slot = slot;
        let bytes = read(&stored.path, MAX_READ)?;
        let outline = preview::pdf_outline(&bytes);
        let items: Vec<Value> = outline
            .entries
            .into_iter()
            .map(|entry| json!({"title":entry.title,"page":entry.page,"level":entry.level}))
            .collect();
        serde_json::to_vec(&json!({"items":items,"truncated":outline.truncated}))
            .map_err(ApiError::internal)
    })
    .await
    .map_err(ApiError::internal)??;
    let bytes = Bytes::from(body);
    state.previews.keep(key, bytes.clone(), "application/json");
    Ok(respond(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn originals_are_classified_by_name_and_markup_is_never_sent_as_itself() {
        assert_eq!(format("Report.PDF").0, Format::Pdf);
        assert_eq!(format("scan.tiff").0, Format::Image { viewable: false });
        assert_eq!(format("photo.HEIC").0, Format::Image { viewable: false });
        assert_eq!(format("photo.avif").0, Format::Image { viewable: true });
        assert_eq!(format("page.xhtml").0, Format::Html);
        assert_eq!(format("data.yml").0, Format::Text);
        assert_eq!(format("deck.pptx").0, Format::Other);
        assert_eq!(format("drawing.svg").0, Format::Other);
        assert_eq!(format("no-extension").1, "application/octet-stream");
        assert_eq!(
            format("Analysis.IPYNB"),
            (Format::Other, "application/x-ipynb+json")
        );
        for name in ["page.html", "page.htm", "page.xhtml", "drawing.svg"] {
            let (format, media) = format(name);
            assert_eq!(
                sent_type(format, media),
                "text/plain; charset=utf-8",
                "{name}"
            );
        }
        let (format, media) = format("notes.md");
        assert_eq!(sent_type(format, media), "text/markdown; charset=utf-8");
    }

    #[test]
    fn a_framed_page_gets_one_insertion_after_its_head_or_html_or_doctype() {
        const META: &str = "<meta http-equiv=\"x-dns-prefetch-control\" content=\"off\">";
        let upload = format!("{META}<base target=\"_blank\">");
        for (page, expected) in [
            (
                "<!doctype html><html lang=en><head><title>T</title></head><body>x</body></html>",
                format!(
                    "<!doctype html><html lang=en><head>{upload}<title>T</title></head><body>x</body></html>"
                ),
            ),
            (
                "<HTML><HEAD\n data-x=1><TITLE>T</TITLE></HEAD></HTML>",
                format!("<HTML><HEAD\n data-x=1>{upload}<TITLE>T</TITLE></HEAD></HTML>"),
            ),
            ("<head/><p>x", format!("<head/>{upload}<p>x")),
            (
                "<html><header>h</header><p>x</p></html>",
                format!("<html>{upload}<header>h</header><p>x</p></html>"),
            ),
            (
                "<header>h</header><heading><head>",
                format!("<header>h</header><heading><head>{upload}"),
            ),
            (
                "<htmlx><HTML\tdir=ltr><p>x",
                format!("<htmlx><HTML\tdir=ltr>{upload}<p>x"),
            ),
            (
                "  <!DOCTYPE html>\n<p>x</p>",
                format!("  <!DOCTYPE html>{upload}\n<p>x</p>"),
            ),
            ("<p>x</p>", format!("{upload}<p>x</p>")),
            ("\u{feff}<p>x</p>", format!("\u{feff}{upload}<p>x</p>")),
            ("", upload.clone()),
            ("<head", upload.clone() + "<head"),
        ] {
            assert_eq!(framed(page, None), expected, "{page}");
        }
        assert_eq!(
            framed(
                "<head><title>T</title>",
                Some("https://example.test/a?b=1&c=\"<'>")
            ),
            format!(
                "<head>{META}<base href=\"https://example.test/a?b=1&amp;c=&quot;&lt;&#39;&gt;\" target=\"_blank\"><title>T</title>"
            )
        );
    }

    #[test]
    fn only_plain_item_ids_name_a_page_snapshot() {
        assert_eq!(snapshot_name("i12").as_deref(), Some("i12.html"));
        for unsafe_id in ["", "..", "../meta", "a/b", "a\\b", ".hidden", "i1.html"] {
            assert_eq!(snapshot_name(unsafe_id), None, "{unsafe_id}");
        }
    }

    #[test]
    fn page_numbers_and_widths_are_whole_numbers_with_a_bounded_width() {
        assert_eq!(page_number("3").unwrap(), 3);
        assert_eq!(page_number("0").unwrap_err().reason, "page_not_found");
        for bad in ["x", "-1", "1.5", ""] {
            assert_eq!(page_number(bad).unwrap_err().reason, "invalid_page_request");
        }
        let width = |text: Option<&str>| {
            page_width(Ok(Query(PageQuery {
                width: text.map(str::to_owned),
            })))
        };
        assert_eq!(width(None).unwrap(), 1200);
        assert_eq!(width(Some("10")).unwrap(), 160);
        assert_eq!(width(Some("99999999999")).unwrap(), 2400);
        assert_eq!(width(Some("800")).unwrap(), 800);
        assert_eq!(
            width(Some("wide")).unwrap_err().reason,
            "invalid_page_request"
        );
    }

    #[test]
    fn the_page_cache_is_bounded_and_keeps_recent_entries() {
        let previews = Previews::default();
        let entry = || Bytes::from(vec![0u8; CACHE_BYTES / 5]);
        for index in 0..8 {
            previews.keep(format!("k{index}"), entry(), "image/png");
            if index == 4 {
                // Reading an entry makes it recent.
                assert!(previews.cached("k1").is_some());
            }
        }
        assert!(previews.cached("k0").is_none());
        assert!(previews.cached("k1").is_some());
        assert!(previews.cached("k7").is_some());
        let total: usize = previews
            .cache
            .lock()
            .unwrap()
            .iter()
            .map(|(_, bytes, _)| bytes.len())
            .sum();
        assert!(total <= CACHE_BYTES);
        // An entry larger than a quarter of the cache is not kept.
        previews.keep(
            "big".into(),
            Bytes::from(vec![0; CACHE_BYTES / 2]),
            "image/png",
        );
        assert!(previews.cached("big").is_none());
        // Two renders at a time.
        let first = previews.slot().unwrap();
        let _second = previews.slot().unwrap();
        assert!(previews.slot().is_none());
        drop(first);
        assert!(previews.slot().is_some());
    }
}
