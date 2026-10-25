use super::{
    State, http,
    jobs::{self, Job},
    store,
    types::{ApiError, ApiResult, Item},
};
use axum::{
    Json,
    body::Body,
    extract::{Path, Query, State as ExtractState, rejection::QueryRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use futures_util::stream;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::Read,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::SystemTime,
};
use tokio::io::AsyncReadExt;

const MAX_RESULT: u64 = 64 * 1024 * 1024;
pub(super) fn encoded(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes())
        .collect::<String>()
        .replace('+', "%20")
}
pub(super) fn body(
    file: File,
    temporary: Option<tempfile::NamedTempFile>,
    content_type: &str,
    filename: &str,
) -> ApiResult<Response> {
    send(file, temporary, content_type, "attachment", filename)
}
/// Streams `file` with `Content-Disposition: <disposition>` (`attachment` or
/// `inline`) naming `filename`.
fn send(
    file: File,
    temporary: Option<tempfile::NamedTempFile>,
    content_type: &str,
    disposition: &str,
    filename: &str,
) -> ApiResult<Response> {
    let size = file.metadata().map_err(ApiError::internal)?.len();
    let stream = stream::try_unfold(
        (tokio::fs::File::from_std(file), temporary),
        |(mut file, temporary)| async move {
            let mut bytes = vec![0u8; 64 * 1024];
            let count = file.read(&mut bytes).await?;
            if count == 0 {
                Ok::<_, std::io::Error>(None)
            } else {
                bytes.truncate(count);
                Ok(Some((bytes, (file, temporary))))
            }
        },
    );
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(content_type).unwrap(),
    );
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, HeaderValue::from(size));
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "{disposition}; filename*=UTF-8''{}",
            encoded(filename)
        ))
        .map_err(ApiError::internal)?,
    );
    response.headers_mut().insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    Ok(response)
}
/// The lowercased extension of a path's last segment.
fn extension(path: &str) -> String {
    path.rsplit('/')
        .next()
        .and_then(|name| name.rsplit_once('.'))
        .map_or_else(String::new, |(_, extension)| extension.to_ascii_lowercase())
}
/// The media type of a raster image every browser shows, by extension: the
/// set the workbench's Markdown view loads, which is all an item asset may be.
fn raster_type(path: &str) -> Option<&'static str> {
    Some(match extension(path).as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "avif" => "image/avif",
        _ => return None,
    })
}
fn content_type(path: &str) -> &'static str {
    raster_type(path).unwrap_or_else(|| match extension(path).as_str() {
        "md" | "txt" => "text/plain; charset=utf-8",
        "svg" => "image/svg+xml",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    })
}
fn missing() -> ApiError {
    ApiError::new(404, "file_not_found", "file not found")
}
fn unavailable() -> ApiError {
    ApiError::new(404, "result_unavailable", "item result not available")
}

pub(super) async fn download(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, relative)): Path<(String, String)>,
) -> ApiResult<Response> {
    let job = jobs::get(&state, &id)?;
    let filename = relative.clone();
    let file = crate::task::blocking(move || {
        let _guard = job.access.lock().unwrap();
        if !public_member(&relative) {
            return Err(missing());
        }
        let path = store::safe_file(&job.folder.join("out"), &relative)?;
        File::open(path).map_err(ApiError::internal)
    })
    .await
    .map_err(ApiError::internal)??;
    body(
        file,
        None,
        content_type(&filename),
        filename.rsplit('/').next().unwrap_or("download"),
    )
}

/// A weak validator of a file's bytes: its size and modification time.
fn version_tag(metadata: &fs::Metadata) -> String {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |time| time.as_nanos());
    format!("W/\"{:x}-{modified:x}\"", metadata.len())
}
/// Whether `If-None-Match` names `tag` (weak comparison) or any version.
fn current(headers: &HeaderMap, tag: &str) -> bool {
    let ours = tag.trim_start_matches("W/");
    headers
        .get_all(header::IF_NONE_MATCH)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .any(|candidate| candidate == "*" || candidate.trim_start_matches("W/") == ours)
}
/// Item assets say nothing beyond a missing job or item: a result that is
/// unavailable, inconsistent or too large to search owns nothing to serve.
fn not_owned(error: ApiError) -> ApiError {
    if error.reason == "item_not_found" || error.status.is_server_error() {
        error
    } else {
        missing()
    }
}

/// `GET /api/jobs/{job_id}/items/{item_id}/assets/{*relpath}`: one image of
/// the item's own result, inline, for the workbench's Markdown view. Only a
/// raster image among the artifacts `result` lists for this item is served,
/// and only from inside the job's output.
pub(super) async fn asset(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id, relative)): Path<(String, String, String)>,
    request: HeaderMap,
) -> ApiResult<Response> {
    let job = jobs::get(&state, &id)?;
    let filename = relative.rsplit('/').next().unwrap_or_default().to_owned();
    let (file, media) = crate::task::blocking(move || {
        let _guard = job.access.lock().unwrap();
        let shown = shown(&job, &item_id).map_err(not_owned)?;
        let media = raster_type(&relative).ok_or_else(missing)?;
        if !shown
            .artifacts(None)
            .map_err(not_owned)?
            .contains(&relative)
        {
            return Err(missing());
        }
        let path = store::safe_file(&shown.out, &relative)?;
        Ok((File::open(path).map_err(ApiError::internal)?, media))
    })
    .await
    .map_err(ApiError::internal)??;
    let tag = version_tag(&file.metadata().map_err(ApiError::internal)?);
    let mut response = if current(&request, &tag) {
        StatusCode::NOT_MODIFIED.into_response()
    } else {
        send(file, None, media, "inline", &filename)?
    };
    let headers = response.headers_mut();
    headers.insert(
        header::ETAG,
        HeaderValue::from_str(&tag).map_err(ApiError::internal)?,
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-cache"),
    );
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("default-src 'none'; sandbox"),
    );
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    Ok(response)
}

#[derive(serde::Deserialize)]
pub(super) struct ResultQuery {
    max_bytes: Option<String>,
}
/// The smallest Markdown excerpt a caller may ask for.
const MIN_EXCERPT: u64 = 1024;

fn excerpt_limit(query: Result<Query<ResultQuery>, QueryRejection>) -> ApiResult<Option<u64>> {
    let invalid = || {
        ApiError::new(
            422,
            "invalid_max_bytes",
            format!("max_bytes must be an integer of at least {MIN_EXCERPT}"),
        )
    };
    match query.map_err(|_| invalid())?.0.max_bytes {
        None => Ok(None),
        Some(text) => text
            .parse::<u64>()
            .ok()
            .filter(|limit| *limit >= MIN_EXCERPT)
            .map(Some)
            .ok_or_else(invalid),
    }
}

/// The start of `bytes` as text, at most `limit` bytes: never inside a UTF-8
/// sequence, and after the last line break in its final quarter when there
/// is one. `None` when the bytes are not UTF-8 text.
fn excerpt(bytes: &[u8], limit: usize) -> Option<&str> {
    let bytes = &bytes[..bytes.len().min(limit)];
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        // Only a sequence the cut split is acceptable.
        Err(error) if error.error_len().is_none() => {
            std::str::from_utf8(&bytes[..error.valid_up_to()]).ok()?
        }
        Err(_) => return None,
    };
    let floor = text.len() - text.len() / 4;
    match text.as_bytes()[floor..]
        .iter()
        .rposition(|byte| *byte == b'\n')
    {
        Some(position) => Some(&text[..floor + position + 1]),
        None => Some(text),
    }
}

/// The lines of a text read in chunks: its line breaks, plus a last line
/// without one (an empty text has none).
fn line_count(mut reader: impl Read) -> std::io::Result<u64> {
    let mut buffer = vec![0; 64 * 1024];
    let (mut lines, mut last) = (0u64, b'\n');
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        lines += buffer[..read].iter().filter(|byte| **byte == b'\n').count() as u64;
        last = buffer[read - 1];
    }
    Ok(lines + u64::from(last != b'\n'))
}

/// Line counts of whole results by path, size, modification time and a
/// fingerprint of their first and last bytes: an excerpt of a large result is
/// read only to its limit, and its count is read once instead of on every
/// request. The fingerprint tells apart two versions of the same size written
/// within one tick of a coarse clock (HFS+ and exFAT keep seconds). Bounded; a
/// full table starts over.
static LINE_COUNTS: OnceLock<Mutex<HashMap<PathBuf, Counted>>> = OnceLock::new();
/// A file's size, modification time and fingerprint, and its lines.
type Counted = (u64, SystemTime, u64, u64);
const LINE_COUNTS_MAX: usize = 256;
/// Bytes read from each end of a file for its fingerprint.
const FINGERPRINT_BYTES: u64 = 4096;

/// A hash of the file's first and last [`FINGERPRINT_BYTES`].
fn fingerprint(file: &mut File, size: u64) -> std::io::Result<u64> {
    use std::hash::{Hash, Hasher};
    use std::io::{Seek, SeekFrom};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut buffer = vec![0u8; FINGERPRINT_BYTES.min(size) as usize];
    file.read_exact(&mut buffer)?;
    buffer.hash(&mut hasher);
    if size > FINGERPRINT_BYTES {
        file.seek(SeekFrom::Start(size - FINGERPRINT_BYTES))?;
        file.read_exact(&mut buffer)?;
        buffer.hash(&mut hasher);
    }
    Ok(hasher.finish())
}

/// The lines of the file at `path`, counted once per version (size,
/// modification time and fingerprint).
fn file_lines(path: &FsPath) -> std::io::Result<u64> {
    let mut file = File::open(path)?;
    let meta = file.metadata()?;
    let Ok(modified) = meta.modified() else {
        return line_count(file);
    };
    let print = fingerprint(&mut file, meta.len())?;
    let counts = LINE_COUNTS.get_or_init(Default::default);
    if let Some((size, time, seen, lines)) = counts.lock().unwrap().get(path)
        && (*size, *time, *seen) == (meta.len(), modified, print)
    {
        return Ok(*lines);
    }
    let lines = line_count(File::open(path)?)?;
    let mut counts = counts.lock().unwrap();
    if counts.len() >= LINE_COUNTS_MAX {
        counts.clear();
    }
    counts.insert(path.to_owned(), (meta.len(), modified, print, lines));
    Ok(lines)
}

/// A done item's result as `result` reads it, resolved under `job.access`:
/// the Markdown it shows and where its artifacts come from.
struct Shown {
    item: Item,
    out: PathBuf,
    variant: &'static str,
    /// The shown Markdown file.
    path: PathBuf,
    /// `<base>.md` and `<base>.llm.md`, with their files when they exist.
    base: (String, Option<PathBuf>),
    enhanced: (String, Option<PathBuf>),
    /// The item's asset index; older histories have none.
    assets: Option<Vec<String>>,
}

fn shown(job: &Job, item_id: &str) -> ApiResult<Shown> {
    // Copy this item's row; reading its files must not hold the data lock.
    let (item, base, assets) = {
        let data = job.data.lock().unwrap();
        let item = data
            .items
            .iter()
            .find(|item| item.item_id == item_id)
            .ok_or_else(|| ApiError::new(404, "item_not_found", "item not found"))?
            .clone();
        let base = item_base(&data, &item);
        (item, base, data.assets.get(item_id).cloned())
    };
    let selected = item
        .output
        .as_deref()
        .filter(|_| item.status == "done")
        .ok_or_else(unavailable)?;
    let out = job.folder.join("out");
    store::safe_file(&out, selected)?;
    let base = base?;
    let base_name = format!("{base}.md");
    let enhanced_name = format!("{base}.llm.md");
    let base_path = store::safe_file(&out, &base_name).ok();
    let enhanced_path = store::safe_file(&out, &enhanced_name).ok();
    let (variant, path) = match (&base_path, &enhanced_path) {
        (_, Some(path)) if item.llm_enhanced => ("llm", path.clone()),
        (Some(path), _) => ("base", path.clone()),
        (None, Some(path)) => ("llm", path.clone()),
        (None, None) => return Err(unavailable()),
    };
    Ok(Shown {
        item,
        out,
        variant,
        path,
        base: (base_name, base_path),
        enhanced: (enhanced_name, enhanced_path),
        assets,
    })
}

impl Shown {
    /// The members of `out/` the result lists as this item's own, in order and
    /// not yet opened: its Markdown, then its indexed assets, or for a history
    /// without an index the assets its Markdown references. `markdown` is the
    /// shown Markdown when the caller has already read it whole.
    fn artifacts(&self, markdown: Option<&str>) -> ApiResult<Vec<String>> {
        let mut names = Vec::new();
        let mut seen = HashSet::new();
        let mut add = |name: &str| {
            if public_member(name) && seen.insert(name.to_owned()) {
                names.push(name.to_owned());
            }
        };
        if self.base.1.is_some() {
            add(&self.base.0);
        }
        if self.item.llm_enhanced && self.enhanced.1.is_some() {
            add(&self.enhanced.0);
        }
        match &self.assets {
            Some(assets) => assets.iter().for_each(|name| add(name)),
            None => self.referenced(markdown)?.iter().for_each(|name| add(name)),
        }
        Ok(names)
    }

    /// Older CLI/server histories lack an item asset index. Recover ownership
    /// through actual destinations rather than exposing unrelated job assets.
    fn referenced(&self, markdown: Option<&str>) -> ApiResult<Vec<String>> {
        let candidates = store::files(&self.out)
            .map_err(ApiError::internal)?
            .into_iter()
            .filter(|(name, _)| {
                name.starts_with(".markitai/assets/")
                    || name.starts_with(".markitai/screenshots/")
                    || name.starts_with("assets/")
            })
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok(candidates);
        }
        let mut text = match markdown {
            Some(text) => text.to_owned(),
            None => {
                if fs::metadata(&self.path).map_err(ApiError::internal)?.len() > MAX_RESULT {
                    return Err(too_large());
                }
                fs::read_to_string(&self.path).map_err(ApiError::internal)?
            }
        };
        if self.item.llm_enhanced
            && let Some(base) = &self.base.1
            && *base != self.path
            && fs::metadata(base).map_err(ApiError::internal)?.len() <= MAX_RESULT
        {
            text.push_str(&fs::read_to_string(base).map_err(ApiError::internal)?);
        }
        let marker = format!("MARKITAI-ARTIFACT-{}-", uuid::Uuid::new_v4().simple());
        let replacements = candidates
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), format!("{marker}{i}-END")))
            .collect::<HashMap<_, _>>();
        let rewritten = markitai_core::output::rewrite_asset_references(&text, &replacements);
        Ok(candidates
            .into_iter()
            .enumerate()
            .filter(|(index, _)| rewritten.contains(&format!("{marker}{index}-END")))
            .map(|(_, name)| name)
            .collect())
    }
}

fn too_large() -> ApiError {
    ApiError::new(
        413,
        "result_too_large",
        "result is too large for JSON; use the file download endpoint",
    )
}

pub(super) async fn result(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
    query: Result<Query<ResultQuery>, QueryRejection>,
) -> ApiResult<Json<Value>> {
    let limit = excerpt_limit(query)?;
    let job = jobs::get(&state, &id)?;
    crate::task::blocking(move || {
        let _guard = job.access.lock().unwrap();
        let shown = shown(&job, &item_id)?;
        let size = fs::metadata(&shown.path).map_err(ApiError::internal)?.len();
        // Older histories find their artifacts in the whole text; an indexed
        // item's excerpt needs only its first bytes.
        let excerpted = limit.filter(|limit| size > *limit && shown.assets.is_some());
        if size > MAX_RESULT && excerpted.is_none() {
            return Err(too_large());
        }
        let markdown = match excerpted {
            Some(limit) => {
                let mut bytes = Vec::new();
                File::open(&shown.path)
                    .map_err(ApiError::internal)?
                    .take(limit)
                    .read_to_end(&mut bytes)
                    .map_err(ApiError::internal)?;
                excerpt(&bytes, limit as usize)
                    .ok_or_else(|| ApiError::internal("result is not UTF-8 text"))?
                    .to_owned()
            }
            None => fs::read_to_string(&shown.path).map_err(ApiError::internal)?,
        };
        // The whole Markdown's lines, which an excerpt alone cannot tell.
        let lines = match excerpted {
            Some(_) => file_lines(&shown.path),
            None => line_count(markdown.as_bytes()),
        }
        .map_err(ApiError::internal)?;
        let mut artifacts = Vec::new();
        for relative in shown.artifacts(excerpted.is_none().then_some(markdown.as_str()))? {
            if let Ok(path) = store::safe_file(&shown.out, &relative) {
                let size = fs::metadata(path).map_err(ApiError::internal)?.len();
                artifacts.push(json!({"relpath":relative,"size":size}));
            }
        }
        let (markdown, truncated) = match limit {
            Some(limit) if markdown.len() as u64 > limit => (
                excerpt(markdown.as_bytes(), limit as usize)
                    .unwrap_or_default()
                    .to_owned(),
                true,
            ),
            _ => (markdown, excerpted.is_some()),
        };
        Ok(Json(
            json!({"name":shown.item.name,"variant":shown.variant,"markdown":markdown,
            "artifacts":artifacts,"truncated":truncated,"markdown_bytes":size,
            "markdown_lines":lines}),
        ))
    })
    .await
    .map_err(ApiError::internal)?
}

/// Files an archive holds at most; more is 413 `archive_too_large`.
const MAX_ARCHIVE_FILES: usize = 100_000;

/// A ZIP archive being written to a private temporary file under `root`.
struct Archive {
    temporary: tempfile::NamedTempFile,
    writer: zip::ZipWriter<File>,
    count: usize,
}
impl Archive {
    fn new(root: &std::path::Path) -> ApiResult<Self> {
        let temporary = tempfile::Builder::new()
            .prefix(".archive-")
            .suffix(".tmp")
            .tempfile_in(root)
            .map_err(ApiError::internal)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            temporary
                .as_file()
                .set_permissions(fs::Permissions::from_mode(0o600))
                .map_err(ApiError::internal)?;
        }
        let writer = zip::ZipWriter::new(temporary.reopen().map_err(ApiError::internal)?);
        Ok(Self {
            temporary,
            writer,
            count: 0,
        })
    }
    /// Copy the file at `path` into the archive as `name`.
    fn add(&mut self, name: &str, path: &std::path::Path) -> ApiResult<()> {
        self.count += 1;
        if self.count > MAX_ARCHIVE_FILES {
            return Err(ApiError::new(
                413,
                "archive_too_large",
                "archive has too many files",
            ));
        }
        self.writer
            .start_file(
                name,
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated)
                    .unix_permissions(0o600),
            )
            .map_err(ApiError::internal)?;
        let mut input = File::open(path).map_err(ApiError::internal)?;
        std::io::copy(&mut input, &mut self.writer).map_err(ApiError::internal)?;
        Ok(())
    }
    fn finish(self) -> ApiResult<(File, tempfile::NamedTempFile)> {
        self.writer
            .finish()
            .map_err(ApiError::internal)?
            .sync_all()
            .map_err(ApiError::internal)?;
        let file = self.temporary.reopen().map_err(ApiError::internal)?;
        Ok((file, self.temporary))
    }
}

fn zip_jobs(
    root: &std::path::Path,
    jobs: Vec<Arc<Job>>,
) -> ApiResult<(File, tempfile::NamedTempFile)> {
    let mut archive = Archive::new(root)?;
    let multiple = jobs.len() > 1;
    let mut directories = HashSet::new();
    for job in jobs {
        let _guard = job.access.lock().unwrap();
        let data = job.data.lock().unwrap();
        if data.status == "running" {
            return Err(ApiError::new(409, "job_running", "job is still running"));
        }
        let prefix = if multiple {
            let raw = data
                .items
                .iter()
                .find_map(|item| item.output.as_deref())
                .map(|name| {
                    let filename = name.rsplit('/').next().unwrap_or(name);
                    filename
                        .strip_suffix(".llm.md")
                        .or_else(|| filename.strip_suffix(".md"))
                        .unwrap_or(filename)
                        .to_owned()
                })
                .unwrap_or_else(|| format!("job-{}", data.id));
            let base = jobs::sanitize_name(&raw);
            let mut name = base.clone();
            let mut count = 2;
            while !directories.insert(caseless::default_case_fold_str(&name)) {
                name = format!("{base} ({count})");
                count += 1;
            }
            format!("{name}/")
        } else {
            String::new()
        };
        drop(data);
        for (relative, path) in store::files(&job.folder.join("out")).map_err(ApiError::internal)? {
            if !public_member(&relative) {
                continue;
            }
            archive.add(&format!("{prefix}{relative}"), &path)?;
        }
    }
    archive.finish()
}

/// One item's archive under `job.access`: the files its result lists, at
/// their relpaths, and its kept original under `original/`. Returns the
/// archive and the item's output base name.
fn zip_item(
    root: &std::path::Path,
    job: &Job,
    item_id: &str,
) -> ApiResult<(File, tempfile::NamedTempFile, String)> {
    let _guard = job.access.lock().unwrap();
    {
        let data = job.data.lock().unwrap();
        let item = data
            .items
            .iter()
            .find(|item| item.item_id == item_id)
            .ok_or_else(|| ApiError::new(404, "item_not_found", "item not found"))?;
        if !["done", "error"].contains(&item.status.as_str())
            || job.retry_pending.lock().unwrap().contains(item_id)
        {
            return Err(ApiError::new(
                409,
                "item_busy",
                "item has not reached a terminal state yet; download it when done",
            ));
        }
    }
    let shown = shown(job, item_id)?;
    let base = shown
        .base
        .0
        .strip_suffix(".md")
        .unwrap_or(&shown.base.0)
        .to_owned();
    let mut archive = Archive::new(root)?;
    for relative in shown.artifacts(None)? {
        if let Ok(path) = store::safe_file(&shown.out, &relative) {
            archive.add(&relative, &path)?;
        }
    }
    if let Some((path, filename)) = super::original::kept(job, item_id) {
        let leaf = filename
            .rsplit(['/', '\\'])
            .next()
            .filter(|leaf| !leaf.is_empty() && !matches!(*leaf, "." | ".."))
            .unwrap_or("original");
        archive.add(&format!("original/{leaf}"), &path)?;
    }
    let (file, temporary) = archive.finish()?;
    Ok((file, temporary, base))
}

/// `GET /api/jobs/{job_id}/items/{item_id}/archive`: a ZIP of one item's
/// result files and its kept original.
pub(super) async fn item_archive(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
) -> ApiResult<Response> {
    let job = jobs::get(&state, &id)?;
    let root = state.root.clone();
    let (file, temp, base) = crate::task::blocking(move || zip_item(&root, &job, &item_id))
        .await
        .map_err(ApiError::internal)??;
    body(file, Some(temp), "application/zip", &format!("{base}.zip"))
}
pub(super) async fn job_archive(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    let job = jobs::get(&state, &id)?;
    let root = state.root.clone();
    let (file, temp) = crate::task::blocking(move || zip_jobs(&root, vec![job]))
        .await
        .map_err(ApiError::internal)??;
    body(
        file,
        Some(temp),
        "application/zip",
        &format!("markitai-{id}.zip"),
    )
}
pub(super) async fn history_archive(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: super::security::Principal,
) -> ApiResult<Response> {
    http::refresh(&state).await?;
    let mut jobs = http::registered_jobs(&state);
    jobs.retain(|job| principal.sees(job) && job.data.lock().unwrap().status == "done");
    crate::sort::by_key(&mut jobs, |job| {
        super::types::instant(&job.data.lock().unwrap().created_at)
    });
    if jobs.is_empty() {
        return Err(ApiError::new(404, "history_empty", "history is empty"));
    }
    let root = state.root.clone();
    let (file, temp) = crate::task::blocking(move || zip_jobs(&root, jobs))
        .await
        .map_err(ApiError::internal)??;
    body(file, Some(temp), "application/zip", "markitai-all.zip")
}

pub(super) fn item_base(
    data: &super::jobs::JobData,
    item: &super::types::Item,
) -> ApiResult<String> {
    let invalid = || {
        ApiError::new(
            409,
            "output_identity_conflict",
            "saved output identity is inconsistent or unsafe",
        )
    };
    let base = if let Some(base) = data.bases.get(&item.item_id) {
        base.clone()
    } else if let Some(name) = &item.output_name {
        name.strip_suffix(".md").unwrap_or(name).to_owned()
    } else if let Some(output) = &item.output {
        let name = if item.llm_enhanced {
            output.strip_suffix(".llm.md")
        } else {
            output.strip_suffix(".md")
        };
        name.or_else(|| output.strip_suffix(".md"))
            .unwrap_or(output)
            .to_owned()
    } else {
        match item.kind.as_str() {
            "file" => item
                .name
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or("")
                .to_owned(),
            "url" => markitai_core::output::url_name(&item.name),
            _ => return Err(invalid()),
        }
    };
    if base.is_empty() || base.contains(['/', '\\', '\0']) || matches!(base.as_str(), "." | "..") {
        return Err(invalid());
    }
    let base_name = format!("{base}.md");
    let enhanced_name = format!("{base}.llm.md");
    if item
        .output_name
        .as_deref()
        .is_some_and(|name| name != base_name && name != base)
        || item
            .output
            .as_deref()
            .is_some_and(|name| name.ends_with(".md") && name != base_name && name != enhanced_name)
    {
        return Err(invalid());
    }
    Ok(base)
}

/// A filename match may locate a result, but cannot grant two items the same
/// mutable Markdown member. Asset sharing is handled separately by owned_files.
pub(super) fn exclusive_item_base(
    data: &super::jobs::JobData,
    item: &super::types::Item,
) -> ApiResult<String> {
    let base = item_base(data, item)?;
    let family = [format!("{base}.md"), format!("{base}.llm.md")]
        .map(|name| caseless::default_case_fold_str(&name));
    for sibling in data
        .items
        .iter()
        .filter(|other| other.item_id != item.item_id)
    {
        let other = item_base(data, sibling)?;
        if [format!("{other}.md"), format!("{other}.llm.md")]
            .iter()
            .any(|name| family.contains(&caseless::default_case_fold_str(name)))
        {
            return Err(ApiError::new(
                409,
                "output_identity_conflict",
                "saved output family overlaps another item",
            ));
        }
    }
    Ok(base)
}

/// Native indexes are authoritative. Legacy histories additionally use exact
/// Markdown destinations and converter filename suffixes, never substring globs.
pub(super) fn owned_files(
    folder: &std::path::Path,
    data: &super::jobs::JobData,
    item: &super::types::Item,
) -> ApiResult<HashSet<String>> {
    let base = item_base(data, item)?;
    let out = folder.join("out");
    let mut names = HashSet::new();
    let mut text = String::new();
    for name in [format!("{base}.md"), format!("{base}.llm.md")] {
        if let Ok(path) = store::safe_file(&out, &name) {
            names.insert(name);
            if fs::metadata(&path).map_err(ApiError::internal)?.len() <= MAX_RESULT
                && let Ok(content) = fs::read_to_string(path)
            {
                text.push_str(&content);
                text.push('\n');
            }
        }
    }
    if let Some(output) = &item.output
        && store::safe_file(&out, output).is_ok()
    {
        names.insert(output.clone());
    }
    if let Some(assets) = data.assets.get(&item.item_id) {
        for asset in assets {
            if store::safe_file(&out, asset).is_ok() {
                names.insert(asset.clone());
            }
        }
    } else {
        let candidates = store::files(&out)
            .map_err(ApiError::internal)?
            .into_iter()
            .filter(|(name, _)| {
                name.starts_with(".markitai/assets/")
                    || name.starts_with(".markitai/screenshots/")
                    || name.starts_with("assets/")
            })
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        let marker = format!("MARKITAI-OWNED-{}-", uuid::Uuid::new_v4().simple());
        let replacements = candidates
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), format!("{marker}{i}-END")))
            .collect();
        let rewritten = markitai_core::output::rewrite_asset_references(&text, &replacements);
        for (i, name) in candidates.into_iter().enumerate() {
            if rewritten.contains(&format!("{marker}{i}-END"))
                || legacy_asset(&base, name.rsplit('/').next().unwrap_or(""))
            {
                names.insert(name);
            }
        }
    }
    names.retain(|name| public_member(name));
    Ok(names)
}
fn legacy_asset(base: &str, name: &str) -> bool {
    let Some((stem, extension)) = name.rsplit_once('.') else {
        return false;
    };
    if ![
        "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif", "svg", "ico", "avif", "heic",
        "heif",
    ]
    .contains(&extension.to_ascii_lowercase().as_str())
    {
        return false;
    }
    let digits = |s: &str, max: usize| {
        !s.is_empty() && s.len() <= max && s.bytes().all(|b| b.is_ascii_digit())
    };
    if let Some(suffix) = stem.strip_prefix(base).and_then(|s| s.strip_prefix('.')) {
        if suffix == "full" || suffix.strip_prefix("full--").is_some_and(|s| digits(s, 4)) {
            return true;
        }
        return digits(suffix, 6)
            || suffix
                .strip_prefix("page")
                .or_else(|| suffix.strip_prefix("slide"))
                .is_some_and(|s| digits(s, 6));
    }
    if let Some(suffix) = stem.strip_prefix(base).and_then(|s| s.strip_prefix('-')) {
        return suffix
            .split_once('-')
            .map_or_else(|| digits(suffix, 6), |(a, b)| digits(a, 6) && digits(b, 6));
    }
    false
}

pub(super) fn public_member(name: &str) -> bool {
    name.rsplit('/').next() != Some(".images.lock")
}

#[cfg(test)]
mod excerpt_tests {
    use super::*;

    #[test]
    fn an_excerpt_ends_at_a_line_in_its_last_quarter_and_never_inside_a_character() {
        // A line break in the final quarter ends the excerpt.
        let text = format!("{}\n{}", "a".repeat(900), "b".repeat(500));
        assert_eq!(excerpt(text.as_bytes(), 1024), Some(&text[..901]));
        // One earlier than that does not: the cut falls on a whole character.
        let text = format!("{}\n{}", "a".repeat(100), "€".repeat(400));
        let cut = excerpt(text.as_bytes(), 1024).unwrap();
        assert_eq!(cut.len(), 1022);
        assert!(text.starts_with(cut) && cut.ends_with('€'));
        // A sequence the read itself split is dropped, not rejected.
        assert_eq!(
            excerpt(&"€".repeat(400).as_bytes()[..1024], 1024)
                .unwrap()
                .len(),
            1023
        );
        assert_eq!(excerpt(b"short\n", 1024), Some("short\n"));
        assert_eq!(excerpt(b"ab\xffcd", 1024), None);
    }

    #[test]
    fn a_line_count_counts_a_last_line_without_its_break() {
        for (text, lines) in [
            ("", 0),
            ("a", 1),
            ("a\n", 1),
            ("a\n\nb", 3),
            ("a\r\nb\r\n", 2),
        ] {
            assert_eq!(line_count(text.as_bytes()).unwrap(), lines, "{text:?}");
            assert_eq!(text.lines().count() as u64, lines, "{text:?}");
        }
        // Chunks end anywhere: a long text counts the same as its lines.
        let long = "line of the notes\n".repeat(10_000) + "end";
        assert_eq!(line_count(long.as_bytes()).unwrap(), 10_001);
    }

    #[test]
    fn a_files_line_count_is_kept_until_the_file_changes() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("notes.md");
        fs::write(&path, "one\ntwo\n").unwrap();
        assert_eq!(file_lines(&path).unwrap(), 2);
        // The second request is answered from the table: the same size and time.
        let key = LINE_COUNTS
            .get()
            .unwrap()
            .lock()
            .unwrap()
            .get(&path)
            .copied();
        assert_eq!(key.map(|(size, _, _, lines)| (size, lines)), Some((8, 2)));
        assert_eq!(file_lines(&path).unwrap(), 2);
        // Another size is counted again.
        fs::write(&path, "one\ntwo\nthree").unwrap();
        assert_eq!(file_lines(&path).unwrap(), 3);
        // The same size written within the same modification time (a coarse
        // clock: HFS+ and exFAT keep seconds) is counted again too.
        let time = fs::metadata(&path).unwrap().modified().unwrap();
        fs::write(&path, "a\nb\nc\nd\ne\nfff").unwrap();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(time)
            .unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 13);
        assert_eq!(file_lines(&path).unwrap(), 6);
    }

    /// A result file is never sent as a script or style sheet the
    /// workbench's `script-src 'self'` would run from this origin.
    #[test]
    fn result_files_are_never_sent_as_scripts() {
        for name in [
            "out/app.js",
            "out/.markitai/assets/x.MJS",
            "out/a.wasm",
            "out/a.css",
        ] {
            assert_eq!(content_type(name), "application/octet-stream", "{name}");
        }
    }
}
