use super::{
    State,
    jobs::{self, Job, JobData},
    security::Trusted,
    store,
    types::{ApiError, ApiResult, Item, JobOptions, MAX_ITEMS, MAX_UPLOAD, instant, now},
};
use axum::{
    Json,
    extract::{FromRequest, Multipart, Path, Request, State as ExtractState},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures_util::stream;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};
use tokio::io::AsyncWriteExt;

pub(super) async fn missing() -> ApiError {
    ApiError::new(404, "route_not_found", "route not found")
}
pub(super) async fn method_not_allowed() -> ApiError {
    ApiError::new(405, "method_not_allowed", "method not allowed")
}
pub(super) async fn capabilities(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
) -> Json<Value> {
    let trusted = request.extensions().get::<Trusted>().is_some_and(|v| v.0);
    let cfg = state.settings.snapshot();
    let llm = markitai_core::llm_capabilities(&cfg);
    // Do not even resolve credentials for an untrusted caller.
    let cloudflare = if trusted {
        markitai_core::cloudflare_capabilities(&cfg)
    } else {
        json!({"configured":false,"available":false,"reason":"client_not_trusted",
            "browser_rendering":false,"file_conversion":false,"file_extensions":[]})
    };
    let mut presets = json!({"minimal":{"llm":false,"ocr":false,"alt":false,"desc":false,"screenshot":false},"standard":{"llm":true,"ocr":false,"alt":true,"desc":true,"screenshot":false},"rich":{"llm":true,"ocr":false,"alt":true,"desc":true,"screenshot":true}});
    if let Some(overrides) = cfg["presets"].as_object() {
        for (name, value) in overrides {
            if presets.get(name).is_some() {
                presets[name] = value.clone();
            }
        }
    }
    Json(
        json!({"version":markitai_core::VERSION,"llm":llm,"remote_services":{"cloudflare":cloudflare},"presets":["minimal","standard","rich"],"preset_options":presets,"extras":{"browser":markitai_core::browser_available(),"svg":true},"limits":{"max_job_items":MAX_ITEMS}}),
    )
}

/// Reads an uploaded `.urls` list; its entries become URL items of this job.
/// A hand-written list may carry a comment, a blank line or a line that is not
/// an HTTP(S) URL, which is skipped the way the CLI skips it; a list that holds
/// no usable URL is refused rather than silently converting nothing.
async fn read_url_list(
    field: &mut axum::extract::multipart::Field<'_>,
    filename: &str,
) -> ApiResult<Vec<(String, Option<String>)>> {
    const MAX_URL_LIST: usize = 1024 * 1024;
    let mut bytes = Vec::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|e| ApiError::multipart(e.status()))?
    {
        if bytes.len() + chunk.len() > MAX_URL_LIST {
            return Err(ApiError::new(
                413,
                "url_list_too_large",
                format!("{filename} exceeds the URL list limit"),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    let text = String::from_utf8(bytes).map_err(|_| {
        ApiError::new(
            422,
            "invalid_url_list",
            format!("{filename} is not UTF-8 text"),
        )
    })?;
    let parsed = markitai_core::url_list::parse(&text).map_err(|error| {
        ApiError::new(
            422,
            "invalid_url_list",
            format!("{filename} is not a JSON URL list: {error}"),
        )
    })?;
    let mut entries = Vec::new();
    for entry in parsed.entries {
        let url = entry.url.trim().to_owned();
        let usable = url::Url::parse(&url).is_ok_and(|parsed| {
            ["http", "https"].contains(&parsed.scheme()) && parsed.host_str().is_some()
        });
        if !usable {
            continue;
        }
        let name = match entry.output_name.map(|name| name.trim().to_owned()) {
            None => None,
            Some(name) if name.is_empty() => None,
            Some(name) => {
                let base = name.strip_suffix(".md").unwrap_or(&name).to_owned();
                if name.contains(['/', '\\']) || base == "." || base == ".." {
                    return Err(ApiError::new(
                        422,
                        "invalid_output_name",
                        format!(
                            "{filename}: a URL output name must be a filename without directory components"
                        ),
                    ));
                }
                Some(base)
            }
        };
        entries.push((url, name));
    }
    if entries.is_empty() {
        return Err(ApiError::new(
            422,
            "empty_url_list",
            format!("{filename} holds no HTTP(S) URLs"),
        ));
    }
    Ok(entries)
}

pub(super) async fn create(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
) -> ApiResult<impl IntoResponse> {
    if state.closing.load(Ordering::SeqCst) {
        return Err(ApiError::new(
            503,
            "shutting_down",
            "server is shutting down",
        ));
    }
    let trusted = request.extensions().get::<Trusted>().is_some_and(|v| v.0);
    let stage = tempfile::Builder::new()
        .prefix(".upload-")
        .tempdir_in(&state.root)
        .map_err(ApiError::internal)?;
    store::private_dir(stage.path()).map_err(ApiError::internal)?;
    store::mark_upload(stage.path()).map_err(ApiError::internal)?;
    store::private_dir(&stage.path().join("uploads")).map_err(ApiError::internal)?;
    store::private_dir(&stage.path().join("out")).map_err(ApiError::internal)?;
    let mut items = Vec::new();
    let mut names = HashSet::new();
    // URL entries from either the `urls` field or an uploaded `.urls` list.
    let mut urls: Vec<(String, Option<String>)> = Vec::new();
    let mut url_field = None;
    let mut options = None;
    if request
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';').next().is_some_and(|v| {
                v.trim()
                    .eq_ignore_ascii_case("application/x-www-form-urlencoded")
            })
        })
    {
        let body = axum::body::to_bytes(request.into_body(), 1024 * 1024)
            .await
            .map_err(|_| ApiError::new(413, "form_field_too_large", "form field exceeds limit"))?;
        for (name, value) in url::form_urlencoded::parse(&body) {
            if name == "urls" {
                url_field = Some(value.as_bytes().to_vec());
            } else if name == "options" {
                options = Some(value.as_bytes().to_vec());
            }
        }
    } else {
        // A request with no content type and no body is an empty job, not a
        // malformed upload.
        let headers = request.headers();
        if !headers.contains_key("content-type")
            && !headers.contains_key("transfer-encoding")
            && headers
                .get("content-length")
                .is_none_or(|length| length.as_bytes() == b"0")
        {
            return Err(ApiError::new(
                422,
                "empty_job",
                "provide at least one file or URL",
            ));
        }
        let mut multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|e| ApiError::multipart(e.status()))?;
        while let Some(mut field) = multipart
            .next_field()
            .await
            .map_err(|e| ApiError::multipart(e.status()))?
        {
            let name = field.name().unwrap_or("").to_owned();
            if name == "files"
                && let Some(filename) = field.file_name()
            {
                if items.len() >= MAX_ITEMS {
                    return Err(ApiError::new(422, "too_many_items", "too many job items"));
                }
                let filename = jobs::unique_name(&jobs::sanitize_name(filename), &mut names);
                if filename.to_ascii_lowercase().ends_with(".urls") {
                    for entry in read_url_list(&mut field, &filename).await? {
                        urls.push(entry);
                    }
                    continue;
                }
                let path = stage.path().join("uploads").join(&filename);
                let mut options = tokio::fs::OpenOptions::new();
                options.write(true).create_new(true);
                #[cfg(unix)]
                options.mode(0o600);
                let mut file = options.open(path).await.map_err(ApiError::internal)?;
                let mut size = 0usize;
                while let Some(bytes) = field
                    .chunk()
                    .await
                    .map_err(|e| ApiError::multipart(e.status()))?
                {
                    size = size.checked_add(bytes.len()).ok_or_else(|| {
                        ApiError::new(413, "file_too_large", "file exceeds upload limit")
                    })?;
                    if size > MAX_UPLOAD {
                        return Err(ApiError::new(
                            413,
                            "file_too_large",
                            "file exceeds upload limit",
                        ));
                    }
                    file.write_all(&bytes).await.map_err(ApiError::internal)?;
                }
                // Flushing waits until the whole job is staged; see `store::sync_uploads`.
                file.flush().await.map_err(ApiError::internal)?;
                items.push(Item::new(items.len() + 1, filename, "file", None));
            } else {
                let mut bytes = Vec::new();
                while let Some(chunk) = field
                    .chunk()
                    .await
                    .map_err(|e| ApiError::multipart(e.status()))?
                {
                    if bytes.len() + chunk.len() > 1024 * 1024 {
                        return Err(ApiError::new(
                            413,
                            "form_field_too_large",
                            "form field exceeds limit",
                        ));
                    }
                    bytes.extend_from_slice(&chunk);
                }
                if name == "urls" {
                    url_field = Some(bytes);
                } else if name == "options" {
                    options = Some(bytes);
                }
            }
        }
    }
    let listed: Vec<String> =
        match url_field.filter(|bytes| !bytes.iter().all(u8::is_ascii_whitespace)) {
            None => Vec::new(),
            Some(bytes) => serde_json::from_slice(&bytes).map_err(|_| {
                ApiError::new(422, "invalid_urls", "urls must be a JSON array of strings")
            })?,
        };
    if listed.iter().any(|url| url.trim().is_empty()) {
        return Err(ApiError::new(
            422,
            "invalid_urls",
            "urls must be a JSON array of non-empty strings",
        ));
    }
    for url in listed {
        urls.push((url.trim().to_owned(), None));
    }
    if !urls.is_empty() && !trusted {
        return Err(ApiError::new(
            403,
            "remote_url_forbidden",
            "URL conversion requires loopback or token authentication; safe remote URL fetching is not yet available",
        ));
    }
    if items.len() + urls.len() > MAX_ITEMS {
        return Err(ApiError::new(422, "too_many_items", "too many job items"));
    }
    for (url, output_name) in urls {
        let parsed =
            url::Url::parse(&url).map_err(|_| ApiError::new(422, "invalid_url", "invalid URL"))?;
        if !["http", "https"].contains(&parsed.scheme()) || parsed.host_str().is_none() {
            return Err(ApiError::new(
                422,
                "unsupported_url_scheme",
                "URLs must use http or https",
            ));
        }
        // A named entry keeps its name as the base it reserves below. The list
        // reader has already refused a name that is not a plain filename.
        let name = output_name.map(|name| jobs::sanitize_name(&name));
        items.push(Item::new(items.len() + 1, url, "url", name));
    }
    if items.is_empty() {
        return Err(ApiError::new(
            422,
            "empty_job",
            "provide at least one file or URL",
        ));
    }
    let options: JobOptions = match options.filter(|bytes| !bytes.is_empty()) {
        None => JobOptions::default(),
        Some(bytes) => JobOptions::parse(&bytes)?,
    };
    let configuration = state.settings.snapshot();
    let cfg = options.config_for_request(&configuration, trusted)?;
    // Without a routable model every item would fail with the same error, so a
    // request for model processing is refused before anything is stored.
    options.require_model(&configuration, &cfg)?;
    let bases = jobs::reserve_outputs(&items);
    for item in &mut items {
        item.remote_processing = options.remote_disclosure();
        if item.kind == "url" {
            item.output_name = Some(format!("{}.md", bases[&item.item_id]));
        }
    }
    let options = options.saved();
    let item_options = items
        .iter()
        .map(|item| (item.item_id.clone(), options.clone()))
        .collect();
    let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_owned();
    let folder = state.root.join(&id);
    let data = JobData {
        id: id.clone(),
        created_at: now(),
        finished_at: None,
        status: "running".into(),
        persistence_error: None,
        options,
        items,
        size: 0,
        bases,
        assets: HashMap::new(),
        item_options,
        transactions: Vec::new(),
    };
    let publication_state = state.clone();
    let job = crate::task::blocking(move || {
        let uploaded: Vec<&str> = data
            .items
            .iter()
            .filter(|item| item.kind == "file")
            .map(|item| item.name.as_str())
            .collect();
        store::sync_uploads(&stage.path().join("uploads"), &uploaded)
            .map_err(ApiError::internal)?;
        store::persist(stage.path(), &data).map_err(ApiError::internal)?;
        // The stable OS lock is shared with CLI history writers. Waiting for it
        // must not occupy a runtime worker or hold the in-memory registry lock.
        let mut lock_options = std::fs::OpenOptions::new();
        lock_options
            .read(true)
            .write(true)
            .create(true)
            .truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            lock_options.mode(0o600);
        }
        let lockpath = publication_state.root.join(".publish.lock");
        markitai_core::output::check_path(&lockpath, false).map_err(ApiError::internal)?;
        let lock = lock_options.open(lockpath).map_err(ApiError::internal)?;
        lock.lock().map_err(ApiError::internal)?;
        let _publication = PublicationLock(lock);
        let mut registry = publication_state.jobs.lock().unwrap();
        if publication_state.closing.load(Ordering::SeqCst) {
            return Err(ApiError::new(
                503,
                "shutting_down",
                "server is shutting down",
            ));
        }
        if folder.exists() {
            return Err(ApiError::new(
                409,
                "job_id_collision",
                "job identifier collision; retry the request",
            ));
        }
        markitai_core::platform::rename(stage.path(), &folder).map_err(ApiError::internal)?;
        // Windows cannot flush the jobs directory; flushing the renamed job's
        // metadata commits the rename instead.
        if let Err(error) = store::unmark_upload(&folder)
            .and_then(|()| markitai_core::platform::sync_renamed_path(&folder.join("meta.json")))
            .and_then(|()| markitai_core::platform::sync_directory(&publication_state.root))
        {
            // Only this transaction's newly created UUID directory is removed.
            // Existing jobs were excluded before rename while holding the lock.
            if let Err(cleanup) = std::fs::remove_dir_all(&folder) {
                eprintln!("Serve: rejected upload cleanup failed: {cleanup}");
            }
            let _ = markitai_core::platform::sync_directory(&publication_state.root);
            return Err(ApiError::internal(error));
        }
        let job = Arc::new(Job::new(folder, data));
        registry.insert(job.data.lock().unwrap().id.clone(), job.clone());
        Ok(job)
    })
    .await
    .map_err(ApiError::internal)??;
    let created = json!({"job_id":id,"items":job.data.lock().unwrap().items.iter().map(Item::created).collect::<Vec<_>>()});
    // Start on the async runtime. Graceful shutdown awaits this HTTP handler
    // before taking task handles, including any final queued cancellation work.
    if let Err(error) = jobs::start(state.clone(), job.clone(), cfg) {
        state.jobs.lock().unwrap().remove(&id);
        let folder = job.folder.clone();
        match crate::task::blocking(move || std::fs::remove_dir_all(folder)).await {
            Ok(Ok(())) => {}
            Ok(Err(cleanup)) => eprintln!("Serve: rejected job cleanup failed: {cleanup}"),
            Err(cleanup) => eprintln!("Serve: rejected job cleanup task failed: {cleanup}"),
        }
        return Err(error);
    }
    Ok((StatusCode::CREATED, Json(created)))
}

/// Stop the original items that are still waiting for a conversion slot.
/// Items already converting finish; queued retries keep their own lifecycle.
pub(super) async fn stop(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    if state.closing.load(Ordering::SeqCst) {
        return Err(ApiError::new(
            503,
            "shutting_down",
            "server is shutting down",
        ));
    }
    let job = jobs::get(&state, &id)?;
    let waiting = {
        let data = job.data.lock().unwrap();
        if data.status != "running" {
            return Err(ApiError::new(409, "job_not_running", "job is not running"));
        }
        data.items
            .iter()
            .filter(|item| item.status == "queued" && item.operation == "convert")
            .count()
    };
    if waiting == 0 {
        return Err(ApiError::new(
            409,
            "nothing_to_stop",
            "no queued items to stop",
        ));
    }
    job.stop.send_replace(true);
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({"job_id":id,"stopping":waiting})),
    ))
}

pub(super) async fn snapshot(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    Ok(Json(
        jobs::get(&state, &id)?.data.lock().unwrap().snapshot(),
    ))
}

pub(super) async fn events(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    let job = jobs::get(&state, &id)?;
    let (snapshot, receiver) = {
        let data = job.data.lock().unwrap();
        (data.snapshot(), job.events.subscribe())
    };
    let terminal = snapshot["status"] != "running";
    let stream = stream::unfold(
        (
            job,
            receiver,
            state.shutdown.subscribe(),
            Some(snapshot),
            terminal,
            false,
        ),
        |(job, mut receiver, mut shutdown, initial, terminal, ended)| async move {
            if ended {
                return None;
            }
            if let Some(snapshot) = initial {
                return Some((
                    Ok::<Event, Infallible>(
                        Event::default()
                            .event("snapshot")
                            .json_data(snapshot)
                            .unwrap(),
                    ),
                    (job, receiver, shutdown, None, terminal, false),
                ));
            }
            if terminal {
                let progress = job.data.lock().unwrap().progress();
                return Some((
                    Ok(Event::default().event("job").json_data(progress).unwrap()),
                    (job, receiver, shutdown, None, true, true),
                ));
            }
            if *shutdown.borrow() {
                return None;
            }
            tokio::select! {
                _=shutdown.changed()=>None,
                message=receiver.recv()=>{
                    let (kind,mut payload)=match message {Ok(value)=>value,Err(tokio::sync::broadcast::error::RecvError::Lagged(_))=>("snapshot",job.data.lock().unwrap().snapshot()),Err(_)=>return None};
                    if kind=="item" { payload=job.data.lock().unwrap().with_item_options(payload); }
                    let ended=kind=="job"&&payload["status"]!="running";
                    let terminal=kind=="snapshot"&&payload["status"]!="running";
                    Some((Ok(Event::default().event(kind).json_data(payload).unwrap()),(job,receiver,shutdown,None,terminal,ended)))
                }
            }
        },
    );
    let mut response = Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("ping"),
        )
        .into_response();
    response
        .headers_mut()
        .insert("x-accel-buffering", "no".parse().unwrap());
    Ok(response)
}

pub(super) async fn refresh(state: &Arc<State>) -> ApiResult<()> {
    let state = state.clone();
    crate::task::blocking(move || store::rehydrate(&state.root, &state.jobs))
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::internal)
}
/// A registered job, else one another local process saved since the last
/// scan; a known job needs no scan of every job folder.
pub(super) async fn registered(state: &Arc<State>, id: &str) -> ApiResult<Arc<Job>> {
    if let Ok(job) = jobs::get(state, id) {
        return Ok(job);
    }
    refresh(state).await?;
    jobs::get(state, id)
}
pub(super) async fn history(
    ExtractState(state): ExtractState<Arc<State>>,
) -> ApiResult<Json<Value>> {
    refresh(&state).await?;
    let mut entries = state
        .jobs
        .lock()
        .unwrap()
        .values()
        .filter_map(|job| {
            let data = job.data.lock().unwrap();
            (data.status == "done").then(|| data.history())
        })
        .collect::<Vec<_>>();
    crate::sort::by_key(&mut entries, |entry| {
        std::cmp::Reverse(entry["created_at"].as_str().and_then(instant))
    });
    Ok(Json(json!(entries)))
}
pub(super) async fn delete(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    let job = jobs::get(&state, &id)?;
    let state = state.clone();
    crate::task::blocking(move || remove_registered(job, &state.jobs, &id))
        .await
        .map_err(ApiError::internal)?
}

fn remove_registered(
    job: Arc<Job>,
    registry: &std::sync::Mutex<HashMap<String, Arc<Job>>>,
    id: &str,
) -> ApiResult<StatusCode> {
    let _guard = job.access.lock().unwrap();
    if !registry
        .lock()
        .unwrap()
        .get(id)
        .is_some_and(|current| Arc::ptr_eq(current, &job))
    {
        return Err(ApiError::new(404, "job_not_found", "job not found"));
    }
    if job.data.lock().unwrap().status == "running" {
        return Err(ApiError::new(409, "job_running", "job is still running"));
    }
    markitai_core::output::check_path(&job.folder, false).map_err(ApiError::internal)?;
    std::fs::remove_dir_all(&job.folder).map_err(ApiError::internal)?;
    registry.lock().unwrap().remove(id);
    Ok(StatusCode::NO_CONTENT)
}

struct PublicationLock(std::fs::File);
impl Drop for PublicationLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

#[cfg(test)]
mod deletion_tests {
    use super::*;

    #[test]
    fn stale_job_reference_returns_not_found_and_cannot_delete_a_replacement() {
        let temporary = tempfile::tempdir().unwrap();
        let folder = temporary.path().join("job");
        std::fs::create_dir(&folder).unwrap();
        let data = JobData {
            id: "job".into(),
            created_at: now(),
            finished_at: Some(now()),
            status: "done".into(),
            persistence_error: None,
            options: json!({}),
            items: Vec::new(),
            size: 0,
            bases: HashMap::new(),
            assets: HashMap::new(),
            item_options: HashMap::new(),
            transactions: Vec::new(),
        };
        let first = Arc::new(Job::new(folder.clone(), data.clone()));
        let stale = first.clone();
        let registry = std::sync::Mutex::new(HashMap::from([("job".into(), first.clone())]));
        assert!(matches!(
            remove_registered(first, &registry, "job"),
            Ok(StatusCode::NO_CONTENT)
        ));
        assert_eq!(
            remove_registered(stale.clone(), &registry, "job")
                .unwrap_err()
                .status,
            StatusCode::NOT_FOUND
        );
        std::fs::create_dir(&folder).unwrap();
        std::fs::write(folder.join("keep"), b"replacement job").unwrap();
        registry
            .lock()
            .unwrap()
            .insert("job".into(), Arc::new(Job::new(folder.clone(), data)));
        assert_eq!(
            remove_registered(stale, &registry, "job")
                .unwrap_err()
                .status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            std::fs::read(folder.join("keep")).unwrap(),
            b"replacement job"
        );
    }
}

#[cfg(test)]
mod error_code_tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        http::Request as HttpRequest,
        routing::{get, post},
    };
    use std::sync::{Mutex, atomic::AtomicBool};
    use tokio::sync::{Semaphore, watch};
    use tower::ServiceExt;

    fn service(root: &std::path::Path) -> (Arc<State>, Router) {
        store::private_dir(root).unwrap();
        let (shutdown, _) = watch::channel(false);
        let cfg = markitai_core::config::normalize(
            &json!({"llm":{"enabled":false},"cache":{"enabled":false},"log":{"dir":null}}),
        )
        .unwrap();
        let state = Arc::new(State {
            settings: super::super::settings::Store::new(
                cfg,
                super::super::SettingsSource {
                    path: root.join("config.json"),
                    origin: "default".into(),
                    overrides: None,
                },
            )
            .unwrap(),
            root: root.into(),
            jobs: Mutex::new(HashMap::new()),
            file_slots: Arc::new(Semaphore::new(1)),
            url_slots: Arc::new(Semaphore::new(1)),
            closing: AtomicBool::new(false),
            persistence_failed: AtomicBool::new(false),
            shutdown,
            tasks: Mutex::new(Vec::new()),
            token: None,
            allowed_hosts: HashSet::new(),
            tickets: Default::default(),
        });
        let router = Router::new()
            .route("/api/jobs", post(create))
            .route("/api/jobs/{job_id}", get(snapshot))
            .route("/api/history", get(history))
            .route(
                "/api/history/archive",
                get(super::super::files::history_archive),
            )
            .route("/api/jobs/{job_id}/cancel", post(stop))
            .route(
                "/api/jobs/{job_id}/items/{item_id}/retry",
                post(super::super::rerun::retry),
            )
            .fallback(missing)
            .with_state(state.clone());
        (state, router)
    }
    // Job creation gets a multipart body with the named files; other routes an empty one.
    async fn call(router: &Router, method: &str, path: &str, files: &[&str]) -> (u16, Value) {
        let mut request = HttpRequest::builder().method(method).uri(path);
        let mut body = String::new();
        if path == "/api/jobs" {
            for name in files {
                body.push_str(&format!("--edge\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{name}\"\r\nContent-Type: application/octet-stream\r\n\r\nsample\r\n"));
            }
            body.push_str("--edge--\r\n");
            request = request.header("content-type", "multipart/form-data; boundary=edge");
        }
        let request = request.body(Body::from(body)).unwrap();
        let response = router.clone().oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }
    // Any body and content type, for requests the service must refuse in its own words.
    async fn send(
        router: &Router,
        method: &str,
        path: &str,
        content_type: Option<&str>,
        body: String,
    ) -> (u16, Value) {
        let mut request = HttpRequest::builder().method(method).uri(path);
        if let Some(content_type) = content_type {
            request = request.header("content-type", content_type);
        }
        let response = router
            .clone()
            .oneshot(request.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        let status = response.status().as_u16();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }
    fn job_form(file: &str, options: &str) -> String {
        format!(
            "--edge\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{file}\"\r\n\r\nsample\r\n--edge\r\nContent-Disposition: form-data; name=\"options\"\r\n\r\n{options}\r\n--edge--\r\n"
        )
    }
    async fn settled(router: &Router, id: &str) -> Value {
        for _ in 0..2000 {
            let (status, value) = call(router, "GET", &format!("/api/jobs/{id}"), &[]).await;
            assert_eq!(status, 200);
            if value["status"] != "running" {
                return value;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("job {id} did not finish");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_items_carry_the_core_code_and_retries_keep_it() {
        let temp = tempfile::tempdir().unwrap();
        let (state, router) = service(temp.path());
        let (status, created) = call(&router, "POST", "/api/jobs", &["notes.xyz"]).await;
        assert_eq!(status, 201, "{created}");
        let id = created["job_id"].as_str().unwrap().to_owned();
        let done = settled(&router, &id).await;
        let item = &done["items"][0];
        assert_eq!(item["status"], "error");
        assert_eq!(item["error_code"], "unsupported", "{item}");
        assert!(
            item["error"]
                .as_str()
                .unwrap()
                .starts_with("Unsupported file format")
        );
        // While the retry waits for the only slot, the old failure no longer describes it.
        let slot = state.file_slots.clone().acquire_owned().await.unwrap();
        let retry = format!("/api/jobs/{id}/items/i1/retry");
        let (status, _) = call(&router, "POST", &retry, &[]).await;
        assert_eq!(status, 202);
        let (_, queued) = call(&router, "GET", &format!("/api/jobs/{id}"), &[]).await;
        assert_eq!(queued["items"][0]["status"], "queued");
        assert!(queued["items"][0].get("error_code").is_none(), "{queued}");
        assert!(queued["items"][0]["error"].is_null());
        drop(slot);
        let retried = settled(&router, &id).await;
        assert_eq!(retried["items"][0]["operation"], "retry");
        assert_eq!(retried["items"][0]["error_code"], "unsupported");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stopped_and_shutdown_items_are_told_apart_by_code() {
        let temp = tempfile::tempdir().unwrap();
        let (state, router) = service(temp.path());
        // Holding the only file slot keeps the items waiting.
        let slot = state.file_slots.clone().acquire_owned().await.unwrap();
        let (_, created) = call(&router, "POST", "/api/jobs", &["a.txt", "b.txt"]).await;
        let id = created["job_id"].as_str().unwrap().to_owned();
        let cancel = format!("/api/jobs/{id}/cancel");
        let (status, reply) = call(&router, "POST", &cancel, &[]).await;
        assert_eq!((status, reply["stopping"].as_u64()), (202, Some(2)));
        drop(slot);
        let stopped = settled(&router, &id).await;
        for item in stopped["items"].as_array().unwrap() {
            assert_eq!(item["error_code"], "cancelled", "{item}");
            assert_eq!(item["error"], "cancelled (stopped by request)");
        }
        let (status, reply) = call(&router, "POST", &cancel, &[]).await;
        assert_eq!(
            (status, reply["reason"].as_str()),
            (409, Some("job_not_running"))
        );

        let slot = state.file_slots.clone().acquire_owned().await.unwrap();
        let (_, created) = call(&router, "POST", "/api/jobs", &["c.txt"]).await;
        let id = created["job_id"].as_str().unwrap().to_owned();
        state.closing.store(true, Ordering::SeqCst);
        state.shutdown.send_replace(true);
        drop(slot);
        let closed = settled(&router, &id).await;
        assert_eq!(closed["items"][0]["error_code"], "shutdown");
        assert_eq!(closed["items"][0]["error"], "cancelled (server shutdown)");
        let (status, reply) = call(&router, "POST", "/api/jobs", &["d.txt"]).await;
        assert_eq!(
            (status, reply["reason"].as_str()),
            (503, Some("shutting_down"))
        );
    }

    #[tokio::test]
    async fn malformed_requests_are_refused_in_the_services_own_words() {
        let temp = tempfile::tempdir().unwrap();
        let (_state, router) = service(temp.path());
        const FORM: &str = "multipart/form-data; boundary=edge";
        let leaks = |value: &Value| {
            let text = value["detail"].as_str().unwrap_or_default().to_owned();
            for framework in ["boundary", "`", "line 1", "column", "expected", "serde"] {
                assert!(!text.contains(framework), "{framework} in {text}");
            }
            text
        };
        // Not a multipart body at all, and a multipart header without a boundary.
        for (content_type, body) in [
            (Some("application/json"), r#"{"urls":[]}"#),
            (Some("multipart/form-data"), "x"),
            (Some("text/plain"), "files"),
        ] {
            let (status, value) =
                send(&router, "POST", "/api/jobs", content_type, body.into()).await;
            assert_eq!(
                (status, value["reason"].as_str()),
                (400, Some("invalid_multipart")),
                "{value}"
            );
            assert!(leaks(&value).contains("multipart/form-data"));
            assert_eq!(value["code"], "bad_request");
        }
        // No body and no content type is an empty job, as for an empty form.
        let (status, value) = send(&router, "POST", "/api/jobs", None, String::new()).await;
        assert_eq!(
            (status, value["reason"].as_str()),
            (422, Some("empty_job")),
            "{value}"
        );
        // Options are named, never quoted from the parser.
        for (options, expected) in [
            (
                r#"{"bogus":true}"#,
                "unknown option 'bogus'; supported options: ",
            ),
            (r#"{"llm":"yes"}"#, "option 'llm' must be true or false"),
            ("{broken", "options must be a JSON object"),
            (
                r#"{"profile":"zz"}"#,
                "option 'profile' must be one of: rag, obsidian, okf",
            ),
        ] {
            let (status, value) = send(
                &router,
                "POST",
                "/api/jobs",
                Some(FORM),
                job_form("a.txt", options),
            )
            .await;
            assert_eq!(
                (status, value["reason"].as_str()),
                (422, Some("invalid_options")),
                "{options}: {value}"
            );
            assert!(leaks(&value).starts_with(expected), "{value}");
        }
        // The rejected uploads left nothing behind.
        let (_, jobs) = send(
            &router,
            "GET",
            "/api/jobs/000000000000",
            None,
            String::new(),
        )
        .await;
        assert_eq!(jobs["reason"], "job_not_found");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retry_bodies_are_validated_by_field_name() {
        let temp = tempfile::tempdir().unwrap();
        let (_state, router) = service(temp.path());
        let (status, created) = call(&router, "POST", "/api/jobs", &["notes.xyz"]).await;
        assert_eq!(status, 201, "{created}");
        let id = created["job_id"].as_str().unwrap().to_owned();
        settled(&router, &id).await;
        let retry = format!("/api/jobs/{id}/items/i1/retry");
        for (body, reason, expected) in [
            (
                r#"{"bogus":1}"#,
                "invalid_retry_body",
                "unknown field 'bogus' in the retry body; supported fields: operation, options",
            ),
            (
                r#"{"operation":"fly"}"#,
                "invalid_retry_body",
                "operation must be 'retry' or 'enhance'",
            ),
            (
                "[1]",
                "invalid_retry_body",
                "retry body must be a JSON object",
            ),
            (
                "{oops",
                "invalid_retry_body",
                "retry body must be a JSON object",
            ),
            (
                r#"{"options":{"llm":1}}"#,
                "invalid_options",
                "option 'llm' must be true or false",
            ),
            (
                r#"{"options":{"nope":1}}"#,
                "invalid_options",
                "unknown option 'nope'; supported options: ",
            ),
        ] {
            let (status, value) = send(
                &router,
                "POST",
                &retry,
                Some("application/json"),
                body.into(),
            )
            .await;
            assert_eq!(
                (status, value["reason"].as_str()),
                (422, Some(reason)),
                "{body}: {value}"
            );
            assert!(
                value["detail"].as_str().unwrap().starts_with(expected),
                "{value}"
            );
            assert!(!value["detail"].as_str().unwrap().contains("line 1"));
        }
        // A well-formed body and the JSON null body are still accepted.
        for body in [r#"{"operation":"retry","options":{"llm":false}}"#, "null"] {
            let (status, value) = send(
                &router,
                "POST",
                &retry,
                Some("application/json"),
                body.into(),
            )
            .await;
            assert_eq!(status, 202, "{body}: {value}");
            settled(&router, &id).await;
        }
    }

    #[tokio::test]
    async fn request_errors_name_their_reason() {
        let temp = tempfile::tempdir().unwrap();
        let (_state, router) = service(temp.path());
        for (method, path, status, code, reason) in [
            (
                "GET",
                "/api/jobs/000000000000",
                404,
                "not_found",
                "job_not_found",
            ),
            ("POST", "/api/jobs", 422, "invalid_request", "empty_job"),
            ("GET", "/api/elsewhere", 404, "not_found", "route_not_found"),
        ] {
            let (actual, value) = call(&router, method, path, &[]).await;
            assert_eq!(actual, status, "{path}: {value}");
            assert_eq!(value["code"], code, "{path}");
            assert_eq!(value["reason"], reason, "{path}");
            assert!(value["detail"].is_string(), "{path}");
        }
    }

    #[tokio::test]
    async fn history_is_newest_first_across_time_zones() {
        let temp = tempfile::tempdir().unwrap();
        let (_state, router) = service(temp.path());
        // A CLI history records local time with its offset, the server UTC.
        for (id, created) in [
            ("00000000000a", "2026-10-09T09:30:00.000+08:00"),
            ("00000000000b", "2026-10-09T02:00:00.000Z"),
            ("00000000000c", "2026-10-08T23:00:00.000-05:00"),
        ] {
            let folder = temp.path().join(id);
            store::private_dir(&folder.join("out")).unwrap();
            std::fs::write(folder.join("out/notes.md"), id).unwrap();
            let item = json!({"item_id":"1","name":"notes.txt","kind":"url","status":"done",
                "output":"notes.md","output_name":"notes.md"});
            std::fs::write(
                folder.join("meta.json"),
                json!({"job_id":id,"created_at":created,"finished_at":created,
                    "status":"done","options":{"origin":"cli"},"items":[item]})
                .to_string(),
            )
            .unwrap();
        }
        let (status, entries) = call(&router, "GET", "/api/history", &[]).await;
        assert_eq!(status, 200, "{entries}");
        let order: Vec<_> = entries
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["job_id"].as_str().unwrap())
            .collect();
        // 04:00Z, 02:00Z, 01:30Z: text order would put the +08:00 job first.
        assert_eq!(order, ["00000000000c", "00000000000b", "00000000000a"]);
        // The archive names same-named jobs oldest first.
        let response = router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/api/history/archive")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        let mut folders = Vec::new();
        for index in 0..archive.len() {
            let mut file = archive.by_index(index).unwrap();
            let mut text = String::new();
            std::io::Read::read_to_string(&mut file, &mut text).unwrap();
            folders.push((file.name().to_owned(), text));
        }
        folders.sort();
        assert_eq!(
            folders,
            [
                ("notes (2)/notes.md".to_owned(), "00000000000b".to_owned()),
                ("notes (3)/notes.md".to_owned(), "00000000000c".to_owned()),
                ("notes/notes.md".to_owned(), "00000000000a".to_owned()),
            ]
        );
    }
}
