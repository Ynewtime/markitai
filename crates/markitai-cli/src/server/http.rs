use super::{
    State,
    jobs::{self, Job, JobData},
    security::Trusted,
    store,
    types::{ApiError, ApiResult, Item, JobOptions, MAX_ITEMS, MAX_UPLOAD, now},
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
    ApiError::new(404, "route not found")
}
pub(super) async fn method_not_allowed() -> ApiError {
    ApiError::new(405, "method not allowed")
}
pub(super) async fn capabilities(ExtractState(state): ExtractState<Arc<State>>) -> Json<Value> {
    let cfg = state.settings.snapshot();
    let llm = markitai_core::llm_capabilities(&cfg);
    let mut presets = json!({"minimal":{"llm":false,"ocr":false,"alt":false,"desc":false,"screenshot":false},"standard":{"llm":true,"ocr":false,"alt":true,"desc":true,"screenshot":false},"rich":{"llm":true,"ocr":false,"alt":true,"desc":true,"screenshot":true}});
    if let Some(overrides) = cfg["presets"].as_object() {
        for (name, value) in overrides {
            if presets.get(name).is_some() {
                presets[name] = value.clone();
            }
        }
    }
    Json(
        json!({"version":markitai_core::VERSION,"llm":llm,"presets":["minimal","standard","rich"],"preset_options":presets,"extras":{"browser":markitai_core::browser_available(),"svg":true},"limits":{"max_job_items":MAX_ITEMS}}),
    )
}

pub(super) async fn create(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
) -> ApiResult<impl IntoResponse> {
    if state.closing.load(Ordering::SeqCst) {
        return Err(ApiError::new(503, "server is shutting down"));
    }
    let trusted = request.extensions().get::<Trusted>().is_some_and(|v| v.0);
    let stage = tempfile::Builder::new()
        .prefix(".upload-")
        .tempdir_in(&state.root)
        .map_err(ApiError::internal)?;
    store::private_dir(stage.path()).map_err(ApiError::internal)?;
    store::private_dir(&stage.path().join("uploads")).map_err(ApiError::internal)?;
    store::private_dir(&stage.path().join("out")).map_err(ApiError::internal)?;
    let mut items = Vec::new();
    let mut names = HashSet::new();
    let mut urls = None;
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
            .map_err(|_| ApiError::new(413, "form field exceeds limit"))?;
        for (name, value) in url::form_urlencoded::parse(&body) {
            if name == "urls" {
                urls = Some(value.as_bytes().to_vec());
            } else if name == "options" {
                options = Some(value.as_bytes().to_vec());
            }
        }
    } else {
        let mut multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|e| ApiError::new(e.status().as_u16(), e.body_text()))?;
        while let Some(mut field) = multipart
            .next_field()
            .await
            .map_err(|e| ApiError::new(e.status().as_u16(), "invalid multipart body"))?
        {
            let name = field.name().unwrap_or("").to_owned();
            if name == "files"
                && let Some(filename) = field.file_name()
            {
                if items.len() >= MAX_ITEMS {
                    return Err(ApiError::new(422, "too many job items"));
                }
                let filename = jobs::unique_name(&jobs::sanitize_name(filename), &mut names);
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
                    .map_err(|e| ApiError::new(e.status().as_u16(), "invalid multipart body"))?
                {
                    size = size
                        .checked_add(bytes.len())
                        .ok_or_else(|| ApiError::new(413, "file exceeds upload limit"))?;
                    if size > MAX_UPLOAD {
                        return Err(ApiError::new(413, "file exceeds upload limit"));
                    }
                    file.write_all(&bytes).await.map_err(ApiError::internal)?;
                }
                file.sync_all().await.map_err(ApiError::internal)?;
                items.push(Item::new(items.len() + 1, filename, "file", None));
            } else {
                let mut bytes = Vec::new();
                while let Some(chunk) = field
                    .chunk()
                    .await
                    .map_err(|e| ApiError::new(e.status().as_u16(), "invalid multipart body"))?
                {
                    if bytes.len() + chunk.len() > 1024 * 1024 {
                        return Err(ApiError::new(413, "form field exceeds limit"));
                    }
                    bytes.extend_from_slice(&chunk);
                }
                if name == "urls" {
                    urls = Some(bytes);
                } else if name == "options" {
                    options = Some(bytes);
                }
            }
        }
    }
    let urls: Vec<String> = match urls.filter(|bytes| !bytes.iter().all(u8::is_ascii_whitespace)) {
        None => Vec::new(),
        Some(bytes) => serde_json::from_slice(&bytes)
            .map_err(|_| ApiError::new(422, "urls must be a JSON array of strings"))?,
    };
    if urls.iter().any(|url| url.trim().is_empty()) {
        return Err(ApiError::new(
            422,
            "urls must be a JSON array of non-empty strings",
        ));
    }
    let urls = urls
        .into_iter()
        .map(|v| v.trim().to_owned())
        .collect::<Vec<_>>();
    if !urls.is_empty() && !trusted {
        return Err(ApiError::new(
            403,
            "URL conversion requires loopback or token authentication; safe remote URL fetching is not yet available",
        ));
    }
    if items.len() + urls.len() > MAX_ITEMS {
        return Err(ApiError::new(422, "too many job items"));
    }
    for url in urls {
        let parsed = url::Url::parse(&url).map_err(|_| ApiError::new(422, "invalid URL"))?;
        if !["http", "https"].contains(&parsed.scheme()) || parsed.host_str().is_none() {
            return Err(ApiError::new(422, "URLs must use http or https"));
        }
        items.push(Item::new(items.len() + 1, url, "url", None));
    }
    if items.is_empty() {
        return Err(ApiError::new(422, "provide at least one file or URL"));
    }
    let options: JobOptions = match options.filter(|bytes| !bytes.is_empty()) {
        None => JobOptions::default(),
        Some(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| ApiError::new(422, format!("invalid options: {e}")))?,
    };
    let cfg = options.config(&state.settings.snapshot())?;
    let bases = jobs::reserve_outputs(&items);
    for item in &mut items {
        if item.kind == "url" {
            item.output_name = Some(format!("{}.md", bases[&item.item_id]));
        }
    }
    let options = serde_json::to_value(options).unwrap();
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
    let job = tokio::task::spawn_blocking(move || {
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
            return Err(ApiError::new(503, "server is shutting down"));
        }
        if folder.exists() {
            return Err(ApiError::new(
                409,
                "job identifier collision; retry the request",
            ));
        }
        std::fs::rename(stage.path(), &folder).map_err(ApiError::internal)?;
        if let Err(error) =
            std::fs::File::open(&publication_state.root).and_then(|file| file.sync_all())
        {
            // Only this transaction's newly created UUID directory is removed.
            // Existing jobs were excluded before rename while holding the lock.
            if let Err(cleanup) = std::fs::remove_dir_all(&folder) {
                eprintln!("Serve: rejected upload cleanup failed: {cleanup}");
            }
            let _ = std::fs::File::open(&publication_state.root).and_then(|file| file.sync_all());
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
        match tokio::task::spawn_blocking(move || std::fs::remove_dir_all(folder)).await {
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
        return Err(ApiError::new(503, "server is shutting down"));
    }
    let job = jobs::get(&state, &id)?;
    let waiting = {
        let data = job.data.lock().unwrap();
        if data.status != "running" {
            return Err(ApiError::new(409, "job is not running"));
        }
        data.items
            .iter()
            .filter(|item| item.status == "queued" && item.operation == "convert")
            .count()
    };
    if waiting == 0 {
        return Err(ApiError::new(409, "no queued items to stop"));
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
                    let (kind,payload)=match message {Ok(value)=>value,Err(tokio::sync::broadcast::error::RecvError::Lagged(_))=>("snapshot",job.data.lock().unwrap().snapshot()),Err(_)=>return None};
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
    tokio::task::spawn_blocking(move || store::rehydrate(&state.root, &state.jobs))
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::internal)
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
    entries.sort_by(|a, b| b["created_at"].as_str().cmp(&a["created_at"].as_str()));
    Ok(Json(json!(entries)))
}
pub(super) async fn delete(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    let job = jobs::get(&state, &id)?;
    let state = state.clone();
    tokio::task::spawn_blocking(move || remove_registered(job, &state.jobs, &id))
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
        return Err(ApiError::new(404, "job not found"));
    }
    if job.data.lock().unwrap().status == "running" {
        return Err(ApiError::new(409, "job is still running"));
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
