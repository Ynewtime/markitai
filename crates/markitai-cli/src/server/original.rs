//! Originals for the workbench's Compare view: what can be shown of an item's
//! source (its kept upload, a URL item's page snapshot, or the conversion's
//! page screenshots), its stored bytes, page images and a PDF page's text.
//! Pages are drawn on blocking threads, at most two at a time.
use super::{
    State, files,
    jobs::{self, Job},
    store,
    types::{ApiError, ApiResult},
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

/// Render slots and a small cache of drawn pages, shared by every job.
pub(super) struct Previews {
    renders: Arc<Semaphore>,
    cache: Mutex<VecDeque<(String, Bytes, &'static str)>>,
}
impl Default for Previews {
    fn default() -> Self {
        Self {
            renders: Arc::new(Semaphore::new(RENDERS)),
            cache: Mutex::new(VecDeque::new()),
        }
    }
}
impl Previews {
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
    stored: Option<Stored>,
    screenshots: Vec<(PathBuf, String)>,
    kind: Kind,
    reason: Option<&'static str>,
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
    let stored = if item.kind == "file" {
        store::safe_file(&job.folder.join("uploads"), &item.name)
            .ok()
            .and_then(|path| stored(path, &item.name, item.name.clone()))
    } else {
        let base = files::item_base(&data, &item).unwrap_or_else(|_| "page".into());
        store::safe_file(&job.folder.join("snapshots"), &snapshot_name(item_id))
            .ok()
            .and_then(|path| stored(path, "page.html", format!("{base}.html")))
    };
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
    let screenshots = names
        .into_iter()
        .filter_map(|name| Some((store::safe_file(&out, &name).ok()?, name)))
        .collect::<Vec<_>>();
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
        name: item.name,
        stored,
        screenshots,
        kind,
        reason,
    })
}

/// A URL item's page snapshot inside the job's `snapshots/`.
pub(super) fn snapshot_name(item_id: &str) -> String {
    format!("{item_id}.html")
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
    eprintln!("Serve: page preview failed: {error}");
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
    format!(
        "/api/jobs/{}/items/{}/original/file",
        files::encoded(&original.job_id),
        files::encoded(&original.item_id)
    )
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

/// The descriptor; reads page boxes, never draws a page.
fn describe(original: Original) -> Value {
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
        (Kind::Screenshots, _) => {
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
    let stored = original.stored.as_ref();
    json!({
        "kind": kind.name(),
        "name": original.name,
        "media_type": stored.map(|stored| stored.media),
        "size": stored.map(|stored| stored.size),
        "download": stored.map(|_| download_path(&original)),
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
    let value = crate::task::blocking(move || describe(original))
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
        (Kind::Screenshots, _) => {
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
    let key = format!(
        "{}\u{0}{}\u{0}{}\u{0}text\u{0}{number}",
        id, original.item_id, stored.version
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
    let text = crate::task::blocking(move || -> ApiResult<Vec<u8>> {
        let _slot = slot;
        let bytes = read(&stored.path, MAX_READ)?;
        let pdf = preview::PdfPreview::open(&bytes).map_err(render_failed)?;
        if number > pdf.page_count() {
            return Err(no_page());
        }
        let size = pdf.page_size(number).map_err(render_failed)?;
        let lines = preview::pdf_page_text(&bytes, number, size).map_err(render_failed)?;
        let lines: Vec<Value> = lines
            .into_iter()
            .map(|line| json!({"text":line.text,"bbox":line.bbox}))
            .collect();
        serde_json::to_vec(
            &json!({"page":number,"width":size.width,"height":size.height,"lines":lines}),
        )
        .map_err(ApiError::internal)
    })
    .await
    .map_err(ApiError::internal)??;
    let bytes = Bytes::from(text);
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
