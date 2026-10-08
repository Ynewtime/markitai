use super::{
    State, files, http,
    jobs::{self, Job},
    security::Trusted,
    store, transaction,
    types::{
        ApiError, ApiResult, Item, JobOptions, RerunFailure, RerunOperation as Operation, now,
    },
};
use crate::diagnostics::AttemptDiagnostics;
use axum::{
    Json,
    extract::{Path, Request, State as ExtractState},
    http::StatusCode,
};
use markitai_core::output::create_private_dir;
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs,
    io::{self, Read},
    path::Path as FsPath,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

#[derive(Default)]
struct RetryBody {
    options: Option<JobOptions>,
    operation: Operation,
}
fn invalid_body(detail: impl Into<String>) -> ApiError {
    ApiError::new(422, "invalid_retry_body", detail)
}
impl RetryBody {
    /// Read the optional JSON body. Problems are named in the service's own words;
    /// a parser's position-in-text message is never passed on.
    fn parse(bytes: &[u8]) -> ApiResult<Self> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|_| invalid_body("retry body must be a JSON object"))?;
        let map = match value {
            Value::Null => return Ok(Self::default()),
            Value::Object(map) => map,
            _ => return Err(invalid_body("retry body must be a JSON object")),
        };
        if let Some(unknown) = map
            .keys()
            .find(|key| !["operation", "options"].contains(&key.as_str()))
        {
            let shown: String = unknown.chars().take(48).collect();
            return Err(invalid_body(format!(
                "unknown field '{shown}' in the retry body; supported fields: operation, options"
            )));
        }
        let operation = match map.get("operation") {
            None | Some(Value::Null) => Operation::Retry,
            Some(Value::String(name)) if name == "retry" => Operation::Retry,
            Some(Value::String(name)) if name == "enhance" => Operation::Enhance,
            Some(_) => return Err(invalid_body("operation must be 'retry' or 'enhance'")),
        };
        let options = match map.get("options") {
            None | Some(Value::Null) => None,
            Some(options) => Some(JobOptions::from_value(options.clone())?),
        };
        Ok(Self { options, operation })
    }
}
impl Operation {
    fn name(self) -> &'static str {
        match self {
            Self::Retry => "retry",
            Self::Enhance => "enhance",
        }
    }
    fn diagnostic_operation(self) -> crate::diagnostics::Operation {
        match self {
            Self::Retry => crate::diagnostics::Operation::Retry,
            Self::Enhance => crate::diagnostics::Operation::Enhance,
        }
    }
}
pub(super) struct Work {
    index: usize,
    prior: Item,
    prior_options: Value,
    cfg: Value,
    base: String,
    operation: Operation,
    runtime: Arc<markitai_core::LlmRuntime>,
    explicit: Option<String>,
}

pub(super) async fn retry(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
    request: Request,
) -> ApiResult<(StatusCode, Json<Value>)> {
    if state.closing.load(Ordering::SeqCst) {
        return Err(ApiError::new(
            503,
            "shutting_down",
            "server is shutting down",
        ));
    }
    let trusted = request.extensions().get::<Trusted>().is_some_and(|v| v.0);
    let bytes = axum::body::to_bytes(request.into_body(), 1024 * 1024)
        .await
        .map_err(|_| ApiError::new(413, "request_too_large", "retry body exceeds limit"))?;
    let body = RetryBody::parse(&bytes)?;
    let job = http::registered(&state, &id).await?;
    let admission_state = state.clone();
    let admission_job = job.clone();
    let (created,drain)=crate::task::blocking(move|| {
        let job=admission_job;let state=admission_state;
        let _access=job.access.lock().unwrap();
        if state.closing.load(Ordering::SeqCst){return Err(ApiError::new(503,"shutting_down","server is shutting down"));}
        if !state.jobs.lock().unwrap().get(&id).is_some_and(|known|Arc::ptr_eq(known,&job)){return Err(ApiError::new(404,"job_not_found","job not found"));}
        let mut data=job.data.lock().unwrap();
        if data.persistence_error.is_some(){return Err(ApiError::new(409,"persistence_failed","job persistence failed; restart to recover before retrying"));}
        let index=data.items.iter().position(|i|i.item_id==item_id).ok_or_else(||ApiError::new(404,"item_not_found","item not found"))?;
        let prior=data.items[index].clone();
        // Restoring a prior successful result must restore its repeat selections too.
        let mut prior_options=data.item_options.get(&item_id).unwrap_or(&data.options).clone();
        if let Some(fields)=prior_options.as_object_mut(){fields.remove("remote_processing");}
        if prior.skip_reason.as_deref()==Some("pending_batch"){return Err(ApiError::new(409,"batch_pending","provider batch enhancement is pending; collect that batch before retrying or enhancing this item"));}
        if !["done","error"].contains(&prior.status.as_str())||job.retry_pending.lock().unwrap().contains(&item_id){return Err(ApiError::new(409,"item_busy","item has not reached a terminal state yet; retry when done"));}
        if !prior.retryable{return Err(ApiError::new(409,"not_retryable","file items recorded from a CLI run cannot be retried or enhanced here; run the markitai CLI on the file again"));}
        if prior.kind=="file" {store::safe_file(&job.folder.join("uploads"),&prior.name).map_err(|_|ApiError::new(404,"upload_missing","original upload is no longer on disk"))?;}
        else if prior.kind=="url" {
            if !trusted{return Err(ApiError::new(403,"remote_url_forbidden","URL conversion requires loopback or token authentication; safe remote URL fetching is not yet available"));}
            let parsed=url::Url::parse(&prior.name).map_err(|_|ApiError::new(422,"invalid_url","invalid URL"))?;
            if !["http","https"].contains(&parsed.scheme())||parsed.host_str().is_none(){return Err(ApiError::new(422,"unsupported_url_scheme","URLs must use http or https"));}
        } else {return Err(ApiError::new(409,"no_source","item has no supported original source"));}
        let explicit_options=body.options.is_some();
        let mut opts=match body.options {Some(opts)=>opts,None=>{
            let saved=data.item_options.get(&item_id).unwrap_or(&data.options);
            let mut known=serde_json::to_value(JobOptions::default()).unwrap();
            if let Some(fields)=saved.as_object(){for (key,value) in fields{if let Some(target)=known.get_mut(key){*target=value.clone();}}}
            serde_json::from_value(known).map_err(|_|ApiError::new(422,"invalid_options","saved item options are invalid"))?
        }};
        // Older histories may contain a previous confirmation; never reuse it.
        if !explicit_options { opts.remote_processing=None; }
        let configuration=state.settings.snapshot();
        let mut cfg=opts.config_for_request(&configuration,trusted)?;
        // Options inherited from the item are the caller's earlier choice and keep the
        // core's failure policy; a request that names model processing now needs a model now.
        if explicit_options&&body.operation==Operation::Retry{opts.require_model(&configuration,&cfg)?;}
        if body.operation==Operation::Enhance&& (cfg["llm"]["enabled"]!=true||!markitai_core::llm_capabilities(&cfg).routable){return Err(ApiError::new(409,"llm_unavailable","LLM enhancement is unavailable; enable a routable LLM first"));}
        let base=files::exclusive_item_base(&data,&prior)?;
        let runtime=job.runtime.get_or_init(||Arc::new(markitai_core::LlmRuntime::new(cfg["llm"]["concurrency"].as_u64().unwrap_or(10).max(1) as usize).expect("validated LLM concurrency"))).clone();
        cfg["output"]["on_conflict"]=json!("overwrite");
        cfg["output"]["filename"]=json!(format!("{base}.md"));
        if body.operation==Operation::Retry{
            let inherited=data.options.clone();let ids=data.items.iter().map(|i|i.item_id.clone()).collect::<Vec<_>>();
            for id in ids{data.item_options.entry(id).or_insert_with(||inherited.clone());}
            let options=opts.saved();data.options=options.clone();data.item_options.insert(item_id.clone(),options);
        }
        data.bases.insert(item_id.clone(),base.clone());
        let job_id=data.id.clone();
        let item=&mut data.items[index];item.status="queued".into();item.error=None;item.error_code=None;item.output=None;item.duration_ms=None;item.finished_at=None;item.cost_usd=None;item.pricing=None;item.diagnostics=None;item.rerun_failure=None;item.llm_enhanced=false;item.operation=body.operation.name().into();item.skipped=false;item.skip_reason=None;item.warnings.clear();
        item.remote_processing=opts.remote_disclosure().or_else(||prior.remote_processing.clone());
        let created=json!({"job_id":job_id,"items":[item.created()]});
        let payload=json!(item);
        data.status="running".into();data.finished_at=None;data.persistence_error=None;
        job.active.fetch_add(1,Ordering::SeqCst);job.retry_pending.lock().unwrap().insert(item_id);
        let _=job.events.send(("item",payload));let _=job.events.send(("job",data.progress()));
        let mut queue=job.retry_queue.lock().unwrap();
        queue.push_back(Work{index,prior,prior_options,cfg,base,operation:body.operation,runtime,explicit:opts.strategy});
        let drain=!job.retry_draining.swap(true,Ordering::SeqCst);
        Ok((created,drain))
    }).await.map_err(ApiError::internal)??;
    if drain {
        let task_state = state.clone();
        let task = crate::task::spawn(async move {
            loop {
                let work = {
                    let mut queue = job.retry_queue.lock().unwrap();
                    let work = queue.pop_front();
                    if work.is_none() {
                        job.retry_draining.store(false, Ordering::SeqCst);
                    }
                    work
                };
                let Some(work) = work else { break };
                run(task_state.clone(), job.clone(), work).await;
                jobs::complete(&task_state, job.clone()).await;
            }
        });
        state.tasks.lock().unwrap().push(task);
    }
    Ok((StatusCode::ACCEPTED, Json(created)))
}

fn equal_files(left: &FsPath, right: &FsPath) -> io::Result<bool> {
    let mut a = fs::File::open(left)?;
    let mut b = fs::File::open(right)?;
    if a.metadata()?.len() != b.metadata()?.len() {
        return Ok(false);
    }
    let mut x = [0u8; 65536];
    let mut y = [0u8; 65536];
    loop {
        let n = a.read(&mut x)?;
        if n == 0 {
            return Ok(true);
        }
        b.read_exact(&mut y[..n])?;
        if x[..n] != y[..n] {
            return Ok(false);
        }
    }
}

async fn run(state: Arc<State>, job: Arc<Job>, work: Work) {
    let slots = if work.prior.kind == "url" {
        state.url_slots.clone()
    } else {
        state.file_slots.clone()
    };
    let mut closing = state.shutdown.subscribe();
    let permit = if state.closing.load(Ordering::SeqCst) {
        None
    } else {
        tokio::select! {biased;_=closing.changed()=>None,permit=slots.acquire_owned()=>permit.ok()}
    };
    if permit.is_none() || state.closing.load(Ordering::SeqCst) {
        failed(
            &job,
            work.index,
            &work.prior,
            &work.prior_options,
            RerunFailure::new(
                work.operation,
                "shutdown",
                "cancelled (server shutdown)".into(),
            ),
            0,
            None,
        );
        return;
    }
    {
        let mut data = job.data.lock().unwrap();
        data.items[work.index].status = "running".into();
        let _ = job.events.send(("item", json!(data.items[work.index])));
    }
    let worker = job.clone();
    let fallback = (
        work.index,
        work.prior.clone(),
        work.operation,
        work.prior_options.clone(),
    );
    let result = crate::task::blocking(move || {
        let _permit = permit;
        let started = Instant::now();
        let mut attempt_usage = markitai_core::ConversionUsage::default();
        let result = (|| -> ApiResult<()> {
            let stage = transaction::stage(&worker.folder).map_err(ApiError::internal)?;
            let out = stage.path().join("out");
            create_private_dir(&out).map_err(ApiError::internal)?;
            let source = if work.prior.kind == "url" {
                work.prior.name.clone()
            } else {
                store::safe_file(&worker.folder.join("uploads"), &work.prior.name)?
                    .to_string_lossy()
                    .into_owned()
            };
            let browser_runtime =
                markitai_core::BrowserRuntime::new(1).map_err(ApiError::internal)?;
            let explicit = work.explicit.as_deref();
            let converted = match markitai_core::convert_with_context_detailed(
                &source,
                markitai_core::ConvertOptions {
                    output_dir: Some(out.clone()),
                    config: Some(work.cfg.clone()),
                    ..Default::default()
                },
                markitai_core::ConvertContext {
                    explicit_fetch_strategy: explicit,
                    llm_runtime: Some(&work.runtime),
                    browser_runtime: Some(&browser_runtime),
                    // A rerun publishes into its job folder.
                    stdout_assets: None,
                },
            ) {
                Ok(result) => {
                    // Later checks and file publication can fail after the provider completed.
                    attempt_usage = result.usage.clone();
                    result
                }
                Err(failure)
                    if matches!(&failure.error, markitai_core::Error::ImageOnly(_))
                        && work.operation == Operation::Retry =>
                {
                    let mut data = worker.data.lock().unwrap();
                    let item = &mut data.items[work.index];
                    item.status = "done".into();
                    item.skipped = true;
                    item.skip_reason = Some("image_only".into());
                    item.error = Some("skipped (image_only)".into());
                    item.diagnostics = AttemptDiagnostics::completed(
                        work.operation.diagnostic_operation(),
                        failure.usage,
                    );
                    item.duration_ms =
                        Some(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
                    item.finished_at = Some(now());
                    worker
                        .retry_pending
                        .lock()
                        .unwrap()
                        .remove(&work.prior.item_id);
                    let _ = worker.events.send(("item", json!(item)));
                    let _ = worker.events.send(("job", data.progress()));
                    return Ok(());
                }
                Err(failure) => {
                    attempt_usage = failure.usage;
                    return Err(ApiError::new(
                        500,
                        failure.error.code(),
                        failure.error.to_string(),
                    ));
                }
            };
            if work.operation == Operation::Enhance
                && (converted.llm_output_path.is_none() || converted.skip_reason.is_some())
            {
                return Err(ApiError::new(
                    500,
                    "enhancement_failed",
                    "LLM enhancement did not produce an enhanced result",
                ));
            }
            let relative = |path: &FsPath| {
                path.strip_prefix(&out)
                    .ok()
                    .map(|p| p.to_string_lossy().into_owned())
            };
            let selected = converted
                .llm_output_path
                .as_deref()
                .or(converted.output_path.as_deref())
                .and_then(relative);
            if selected.is_none() {
                return Err(ApiError::new(
                    500,
                    "no_output",
                    "conversion produced no output",
                ));
            }
            let mut replacements = store::files(&out)
                .map_err(ApiError::internal)?
                .into_iter()
                .map(|(name, path)| (format!("out/{name}"), path))
                .collect::<Vec<_>>();
            // File work runs under the job's access lock only. Admission,
            // finalization and deletion take it too; conversions still running
            // change only their own rows, so the data lock is taken briefly.
            let _access = worker.access.lock().unwrap();
            let snapshot = worker.data.lock().unwrap().clone();
            let _sidecar_locks = super::sidecar::prepare(&out, &worker.folder.join("out"))
                .map_err(ApiError::internal)?;
            replacements.retain(|(name, _)| files::public_member(name));
            let mut shared = HashSet::new();
            for sibling in snapshot
                .items
                .iter()
                .filter(|i| i.item_id != work.prior.item_id)
            {
                shared.extend(files::owned_files(&worker.folder, &snapshot, sibling)?);
            }
            let mut retained = Vec::new();
            for (name, path) in replacements.drain(..) {
                let raw = name.strip_prefix("out/").unwrap();
                if shared.contains(raw) {
                    let existing = store::safe_file(&worker.folder.join("out"), raw)?;
                    if !equal_files(&existing, &path).map_err(ApiError::internal)? {
                        return Err(ApiError::new(
                            409,
                            "output_conflict",
                            "retry would replace another item's artifact",
                        ));
                    }
                } else {
                    retained.push((name, path));
                }
            }
            let stale = format!("{}.llm.md", work.base);
            let removals = if converted.llm_output_path.is_none()
                && !shared.contains(&stale)
                && store::safe_file(&worker.folder.join("out"), &stale).is_ok()
            {
                vec![format!("out/{stale}")]
            } else {
                Vec::new()
            };
            let published = transaction::publish(
                &worker.folder,
                stage,
                worker.sequence.fetch_add(1, Ordering::SeqCst),
                retained,
                removals,
            );
            let mut data = worker.data.lock().unwrap();
            let id = match published {
                Ok(id) => id,
                Err(error) => {
                    if transaction::requires_recovery(&error) {
                        data.persistence_error = Some(
                            "output rollback failed; restart to recover the previous result".into(),
                        );
                    }
                    return Err(ApiError::internal(error));
                }
            };
            data.transactions.push(id);
            let assets = converted
                .assets
                .iter()
                .chain(&converted.screenshots)
                .filter_map(|p| relative(p))
                .collect();
            data.assets.insert(work.prior.item_id.clone(), assets);
            let item = &mut data.items[work.index];
            item.status = "done".into();
            item.output = selected;
            item.llm_enhanced = converted.llm_output_path.is_some();
            item.duration_ms = Some(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
            item.finished_at = Some(now());
            item.cost_usd = Some(converted.usage.cost_usd);
            item.pricing = crate::pricing::Pricing::from_usage(&converted.usage);
            item.diagnostics = AttemptDiagnostics::completed(
                work.operation.diagnostic_operation(),
                converted.usage,
            );
            item.skipped = converted.skip_reason.is_some();
            item.skip_reason = converted.skip_reason;
            item.error = item.skip_reason.as_ref().map(|v| format!("skipped ({v})"));
            item.error_code = None;
            item.rerun_failure = None;
            item.warnings = converted.warnings;
            worker
                .retry_pending
                .lock()
                .unwrap()
                .remove(&work.prior.item_id);
            let _ = worker.events.send(("item", json!(item)));
            let _ = worker.events.send(("job", data.progress()));
            Ok(())
        })();
        if let Err(error) = result {
            let diagnostics = AttemptDiagnostics::failed(
                work.operation.diagnostic_operation(),
                error.detail.clone(),
                attempt_usage,
            );
            failed(
                &worker,
                work.index,
                &work.prior,
                &work.prior_options,
                RerunFailure::new(work.operation, error.reason, error.detail),
                started.elapsed().as_millis().min(u64::MAX as u128) as u64,
                diagnostics,
            );
        }
    })
    .await;
    if result.is_err() {
        failed(
            &job,
            fallback.0,
            &fallback.1,
            &fallback.3,
            RerunFailure::new(
                fallback.2,
                "internal_error",
                "internal retry worker failure".into(),
            ),
            0,
            None,
        );
    }
}
fn failed(
    job: &Job,
    index: usize,
    prior: &Item,
    prior_options: &Value,
    failure: RerunFailure,
    duration: u64,
    diagnostics: Option<AttemptDiagnostics>,
) {
    let mut data = job.data.lock().unwrap();
    let restore = data.persistence_error.is_none()
        && prior.status == "done"
        && prior.output.is_some()
        && !prior.skipped;
    if restore {
        // Do not revert job-wide options: another item's accepted retry may have
        // changed them since this work was queued. Each item's saved copy wins.
        data.item_options
            .insert(prior.item_id.clone(), prior_options.clone());
    }
    let item = &mut data.items[index];
    if restore {
        let remote_processing = item
            .remote_processing
            .clone()
            .or_else(|| prior.remote_processing.clone());
        *item = prior.clone();
        item.remote_processing = remote_processing;
        item.rerun_failure = Some(failure);
    } else {
        item.status = "error".into();
        item.error = Some(failure.error);
        item.error_code = Some(failure.error_code);
        item.rerun_failure = None;
        item.finished_at = Some(now());
        item.duration_ms = Some(duration);
    }
    // Output rollback must not resurrect a previous attempt's observation.
    item.diagnostics = diagnostics;
    job.retry_pending.lock().unwrap().remove(&prior.item_id);
    let _ = job.events.send(("item", json!(item)));
    let _ = job.events.send(("job", data.progress()));
}

pub(super) async fn delete(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
) -> ApiResult<StatusCode> {
    let job = http::registered(&state, &id).await?;
    crate::task::blocking(move || {
        let _access = job.access.lock().unwrap();
        if !state
            .jobs
            .lock()
            .unwrap()
            .get(&id)
            .is_some_and(|known| Arc::ptr_eq(known, &job))
        {
            return Err(ApiError::new(404, "job_not_found", "job not found"));
        }
        // An idle job changes only under its access lock, held here. Work on a
        // copy so readers of the data lock are not stalled by the file work.
        let mut data = job.data.lock().unwrap().clone();
        let index = data
            .items
            .iter()
            .position(|i| i.item_id == item_id)
            .ok_or_else(|| ApiError::new(404, "item_not_found", "item not found"))?;
        if data.status == "running"
            || job.active.load(Ordering::SeqCst) > 0
            || !["done", "error"].contains(&data.items[index].status.as_str())
        {
            return Err(ApiError::new(
                409,
                "job_running",
                "job is still running; retry when done",
            ));
        }
        files::exclusive_item_base(&data, &data.items[index])?;
        if data.items.len() == 1 {
            markitai_core::output::check_path(&job.folder, false).map_err(ApiError::internal)?;
            fs::remove_dir_all(&job.folder).map_err(ApiError::internal)?;
            state.jobs.lock().unwrap().remove(&id);
            return Ok(StatusCode::NO_CONTENT);
        }
        let selected = data.items[index].clone();
        let mut owned = files::owned_files(&job.folder, &data, &selected)?;
        for sibling in data.items.iter().filter(|i| i.item_id != item_id) {
            for name in files::owned_files(&job.folder, &data, sibling)? {
                owned.remove(&name);
            }
        }
        let mut removals = owned.iter().map(|n| format!("out/{n}")).collect::<Vec<_>>();
        if selected.kind == "file"
            && !data
                .items
                .iter()
                .any(|i| i.item_id != item_id && i.kind == "file" && i.name == selected.name)
            && store::safe_file(&job.folder.join("uploads"), &selected.name).is_ok()
        {
            removals.push(format!("uploads/{}", selected.name));
        }
        // The reference caches a job archive here; it would keep the deleted item.
        if store::safe_file(&job.folder, "archive.zip").is_ok() {
            removals.push("archive.zip".into());
        }
        let prior = data.clone();
        let stage = transaction::stage(&job.folder).map_err(ApiError::internal)?;
        let staged = stage.path().join("out");
        create_private_dir(&staged).map_err(ApiError::internal)?;
        let _sidecar_locks = super::sidecar::prune(&staged, &job.folder.join("out"), &owned)
            .map_err(ApiError::internal)?;
        let replacements = store::files(&staged)
            .map_err(ApiError::internal)?
            .into_iter()
            .map(|(name, path)| (format!("out/{name}"), path))
            .collect();
        let transaction = transaction::publish(
            &job.folder,
            stage,
            job.sequence.fetch_add(1, Ordering::SeqCst),
            replacements,
            removals,
        )
        .map_err(ApiError::internal)?;
        data.transactions.push(transaction);
        data.items.remove(index);
        data.bases.remove(&item_id);
        data.assets.remove(&item_id);
        data.item_options.remove(&item_id);
        let saved = store::measure(&job.folder).and_then(|size| {
            data.size = size;
            store::persist(&job.folder, &data)
        });
        if let Err(error) = saved {
            let transaction = data.transactions.last().unwrap().clone();
            if let Err(error) = transaction::abort(&job.folder, &transaction) {
                data.status = "error".into();
                data.persistence_error =
                    Some("item deletion rollback failed; restart to recover".into());
                *job.data.lock().unwrap() = data;
                state.persistence_failed.store(true, Ordering::SeqCst);
                return Err(ApiError::internal(error));
            }
            // persist may have renamed metadata before directory fsync failed.
            // Put the matching old row back along with its restored bytes.
            store::persist(&job.folder, &prior).map_err(ApiError::internal)?;
            return Err(ApiError::internal(error));
        }
        transaction::committed(&job.folder, &mut data);
        let progress = data.progress();
        *job.data.lock().unwrap() = data;
        let _ = job.events.send(("job", progress));
        Ok(StatusCode::NO_CONTENT)
    })
    .await
    .map_err(ApiError::internal)?
}
