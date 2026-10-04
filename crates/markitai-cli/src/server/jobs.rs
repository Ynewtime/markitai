use super::{
    State, store,
    types::{ApiError, ApiResult, Item, JobOptions, now},
};
use crate::diagnostics::{AttemptDiagnostics, Operation};
use futures_util::{StreamExt, stream::FuturesUnordered};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};
use tokio::sync::{broadcast, watch};

#[derive(Clone)]
pub(super) struct JobData {
    pub id: String,
    pub created_at: String,
    pub finished_at: Option<String>,
    pub status: String,
    pub persistence_error: Option<String>,
    pub options: Value,
    pub items: Vec<Item>,
    pub size: u64,
    pub bases: HashMap<String, String>,
    pub assets: HashMap<String, Vec<String>>,
    pub item_options: HashMap<String, Value>,
    pub transactions: Vec<String>,
}
impl JobData {
    pub fn progress(&self) -> Value {
        json!({"status":self.status,"done":self.items.iter().filter(|i|i.status=="done").count(),
            "failed":self.items.iter().filter(|i|i.status=="error").count(),"total":self.items.len()})
    }
    /// The persisted repeat selections, never permission for a new request.
    /// CLI/older histories may carry extra metadata; expose only option keys.
    pub fn with_item_options(&self, mut item: Value) -> Value {
        let saved = item["item_id"]
            .as_str()
            .and_then(|id| self.item_options.get(id))
            .unwrap_or(&self.options);
        let mut options =
            serde_json::to_value(JobOptions::default()).expect("job options serialize");
        let fields = options.as_object_mut().expect("job options are an object");
        fields.remove("remote_processing");
        if let Some(saved) = saved.as_object() {
            for (key, value) in saved {
                if let Some(target) = fields.get_mut(key) {
                    *target = value.clone();
                }
            }
        }
        item["options"] = options;
        item
    }

    pub fn snapshot(&self) -> Value {
        let mut value = self.progress();
        value["job_id"] = json!(self.id);
        value["created_at"] = json!(self.created_at);
        value["finished_at"] = json!(self.finished_at);
        value["options"] = self.options.clone();
        value["items"] = self
            .items
            .iter()
            .map(|item| self.with_item_options(json!(item)))
            .collect();
        if let Some(error) = &self.persistence_error {
            value["persistence_error"] = json!(error);
        }
        value
    }
    // Output totals must not silently omit old observations or price a failed
    // later attempt as if it belonged to the retained output.
    fn output_pricing(&self) -> Option<crate::pricing::Pricing> {
        let mut coverage = Vec::new();
        for item in &self.items {
            let Some(cost) = item.cost_usd else {
                if item.pricing.is_some()
                    || item.diagnostics.as_ref().is_some_and(|diagnostics| {
                        diagnostics.last_attempt.status == crate::diagnostics::Status::Done
                    })
                {
                    return None;
                }
                continue;
            };
            if let Some(pricing) = &item.pricing {
                coverage.push(pricing.clone());
                continue;
            }
            if let Some(diagnostics) = &item.diagnostics {
                let attempt = &diagnostics.last_attempt;
                if attempt.status == crate::diagnostics::Status::Done {
                    if item.status != "done" || cost != attempt.usage.cost_usd {
                        return None;
                    }
                    // Historical successful observations predate pricing counters.
                    // The helper retains their recorded request count as unknown.
                    coverage.push(crate::pricing::Pricing::from_usage(&attempt.usage)?);
                    continue;
                }
            }
            // A scalar cost provides no request count. Suppress the aggregate
            // completeness claim rather than invent an unpriced request.
            if cost > 0.0 {
                return None;
            }
        }
        crate::pricing::Pricing::aggregate(coverage.iter())
    }

    pub fn history(&self) -> Value {
        let max_duration = self.items.iter().filter_map(|i| i.duration_ms).max();
        let duration = if self.items.iter().any(|i| i.operation != "convert") {
            max_duration
        } else {
            chrono::DateTime::parse_from_rfc3339(&self.created_at)
                .ok()
                .zip(
                    self.finished_at
                        .as_deref()
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok()),
                )
                .map(|(start, end)| (end - start).num_milliseconds().max(0) as u64)
                .or(max_duration)
        };
        let cost = self
            .items
            .iter()
            .any(|i| i.cost_usd.is_some())
            .then(|| self.items.iter().filter_map(|i| i.cost_usd).sum::<f64>());
        let mut value = self.progress();
        value["job_id"] = json!(self.id);
        value["created_at"] = json!(self.created_at);
        value["finished_at"] = json!(self.finished_at);
        value["skipped"] = json!(self.items.iter().filter(|i| i.skipped).count());
        value["llm_enhanced"] = json!(self.items.iter().filter(|i| i.llm_enhanced).count());
        value["cost_usd"] = json!(cost);
        if let Some(disclosure) = self
            .items
            .iter()
            .find_map(|item| item.remote_processing.as_ref())
        {
            value["remote_processing"] = disclosure.clone();
        }
        if let Some(pricing) = self.output_pricing() {
            value["pricing"] = json!(pricing);
        }
        value["names_preview"] = json!(
            self.items
                .iter()
                .take(3)
                .map(|i| &i.name)
                .collect::<Vec<_>>()
        );
        value["kinds_preview"] = json!(
            self.items
                .iter()
                .take(3)
                .map(|i| &i.kind)
                .collect::<Vec<_>>()
        );
        value["duration_ms"] = json!(duration);
        value["size_bytes"] = json!(self.size);
        value["origin"] = json!(self.options["origin"].as_str().unwrap_or("web"));
        value["retryable"] = json!(self.items.iter().any(|i| i.retryable));
        value
    }
}

pub(super) struct Job {
    pub folder: PathBuf,
    pub data: Mutex<JobData>,
    pub access: Mutex<()>,
    pub active: AtomicUsize,
    pub sequence: AtomicU64,
    pub retry_queue: Mutex<VecDeque<super::rerun::Work>>,
    pub retry_draining: AtomicBool,
    pub retry_pending: Mutex<HashSet<String>>,
    pub runtime: OnceLock<Arc<markitai_core::LlmRuntime>>,
    pub events: broadcast::Sender<(&'static str, Value)>,
    /// Set by a stop request; original items still waiting for a slot are not dispatched.
    pub stop: watch::Sender<bool>,
}
impl Job {
    pub fn new(folder: PathBuf, data: JobData) -> Self {
        let (events, _) = broadcast::channel(128);
        let (stop, _) = watch::channel(false);
        Self {
            folder,
            data: Mutex::new(data),
            access: Mutex::new(()),
            active: AtomicUsize::new(0),
            sequence: AtomicU64::new(0),
            retry_queue: Mutex::new(VecDeque::new()),
            retry_draining: AtomicBool::new(false),
            retry_pending: Mutex::new(HashSet::new()),
            runtime: OnceLock::new(),
            events,
            stop,
        }
    }
}

pub(super) fn sanitize_name(name: &str) -> String {
    let leaf = name.rsplit(['/', '\\']).next().unwrap_or("");
    markitai_core::output_name::sanitize(leaf, "upload")
}
pub(super) fn unique_name(name: &str, taken: &mut HashSet<String>) -> String {
    let stem = std::path::Path::new(name)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let suffix = std::path::Path::new(name)
        .extension()
        .map(|s| format!(".{}", s.to_string_lossy()))
        .unwrap_or_default();
    let mut candidate = name.to_owned();
    let mut count = 2;
    while !taken.insert(caseless::default_case_fold_str(&candidate)) {
        candidate = format!("{stem} ({count}){suffix}");
        count += 1;
    }
    candidate
}
pub(super) fn reserve_outputs(items: &[Item]) -> HashMap<String, String> {
    let mut taken = HashSet::new();
    let mut result = HashMap::new();
    for item in items {
        let raw = if item.kind == "url" {
            markitai_core::output::url_name(&item.name, &Default::default())
        } else {
            item.name.clone()
        };
        let name = sanitize_name(&raw);
        let mut base = name.clone();
        let mut count = 2;
        loop {
            let keys = [format!("{base}.md"), format!("{base}.llm.md")]
                .map(|s| caseless::default_case_fold_str(&s));
            if keys.iter().all(|key| !taken.contains(key)) {
                taken.extend(keys);
                break;
            }
            base = format!("{name} ({count})");
            count += 1;
        }
        result.insert(item.item_id.clone(), base);
    }
    result
}

pub(super) fn get(state: &State, id: &str) -> ApiResult<Arc<Job>> {
    state
        .jobs
        .lock()
        .unwrap()
        .get(id)
        .cloned()
        .ok_or_else(|| ApiError::new(404, "job_not_found", "job not found"))
}

async fn convert_one(
    state: Arc<State>,
    job: Arc<Job>,
    index: usize,
    cfg: Value,
    runtime: Arc<markitai_core::LlmRuntime>,
    browser_runtime: Arc<markitai_core::BrowserRuntime>,
    explicit: Option<String>,
) {
    let item = job.data.lock().unwrap().items[index].clone();
    let semaphore = if item.kind == "url" {
        state.url_slots.clone()
    } else {
        state.file_slots.clone()
    };
    let mut closing = state.shutdown.subscribe();
    let mut stopped = job.stop.subscribe();
    let permit = if state.closing.load(Ordering::SeqCst) || *stopped.borrow() {
        None
    } else {
        tokio::select! { biased; _=closing.changed()=>None, _=stopped.wait_for(|stop| *stop)=>None, permit=semaphore.acquire_owned()=>permit.ok() }
    };
    let shutdown = state.closing.load(Ordering::SeqCst);
    // A stop that arrives together with the slot still wins: the item has not started.
    if permit.is_none() || shutdown || *job.stop.borrow() {
        let mut data = job.data.lock().unwrap();
        let item = &mut data.items[index];
        let (code, error) = if !shutdown && *job.stop.borrow() {
            ("cancelled", "cancelled (stopped by request)")
        } else {
            ("shutdown", "cancelled (server shutdown)")
        };
        item.status = "error".into();
        item.error = Some(error.into());
        item.error_code = Some(code.into());
        item.finished_at = Some(now());
        let _ = job.events.send(("item", json!(item)));
        return;
    }
    let base = job.data.lock().unwrap().bases[&item.item_id].clone();
    {
        let mut data = job.data.lock().unwrap();
        data.items[index].status = "running".into();
        let _ = job.events.send(("item", json!(data.items[index])));
    }
    let started = Instant::now();
    let out = job.folder.join("out");
    let upload = job.folder.join("uploads").join(&item.name);
    let target = if item.kind == "url" {
        item.name.clone()
    } else {
        upload.to_string_lossy().into_owned()
    };
    let mut cfg = cfg;
    cfg["output"]["filename"] = json!(format!("{base}.md"));
    let result = crate::task::blocking(move || {
        let _permit = permit;
        markitai_core::convert_with_context_detailed(
            &target,
            markitai_core::ConvertOptions {
                output_dir: Some(out),
                config: Some(cfg),
                ..Default::default()
            },
            markitai_core::ConvertContext {
                explicit_fetch_strategy: explicit.as_deref(),
                llm_runtime: Some(&runtime),
                browser_runtime: Some(&browser_runtime),
                // Jobs always publish into their own folder.
                stdout_assets: None,
            },
        )
    })
    .await;
    let mut data = job.data.lock().unwrap();
    let mut assets = None;
    let item = &mut data.items[index];
    item.duration_ms = Some(started.elapsed().as_millis().min(u64::MAX as u128) as u64);
    item.finished_at = Some(now());
    match result {
        Ok(Ok(result)) => {
            item.status = "done".into();
            item.skipped = result.skip_reason.is_some();
            item.skip_reason = result.skip_reason;
            item.error = item
                .skip_reason
                .as_ref()
                .map(|reason| format!("skipped ({reason})"));
            item.error_code = None;
            item.llm_enhanced = result.llm_output_path.is_some();
            item.cost_usd = Some(result.usage.cost_usd);
            item.pricing = crate::pricing::Pricing::from_usage(&result.usage);
            item.diagnostics = AttemptDiagnostics::completed(Operation::Convert, result.usage);
            item.warnings = result.warnings;
            item.output = result.llm_output_path.or(result.output_path).and_then(|p| {
                p.strip_prefix(job.folder.join("out"))
                    .ok()
                    .map(|p| p.to_string_lossy().into_owned())
            });
            assets = Some(
                result
                    .assets
                    .into_iter()
                    .chain(result.screenshots)
                    .filter_map(|p| {
                        p.strip_prefix(job.folder.join("out"))
                            .ok()
                            .map(|p| p.to_string_lossy().into_owned())
                    })
                    .collect(),
            );
        }
        Ok(Err(failure)) if matches!(&failure.error, markitai_core::Error::ImageOnly(_)) => {
            item.status = "done".into();
            item.skipped = true;
            item.skip_reason = Some("image_only".into());
            item.error = Some("skipped (image_only)".into());
        }
        Ok(Err(failure)) => {
            item.status = "error".into();
            let message = failure.error.to_string();
            item.error_code = Some(failure.code().into());
            item.diagnostics =
                AttemptDiagnostics::failed(Operation::Convert, message.clone(), failure.usage);
            item.error = Some(message);
        }
        Err(_) => {
            item.status = "error".into();
            item.error = Some("internal conversion error".into());
            item.error_code = Some("internal_error".into());
        }
    }
    let id = item.item_id.clone();
    let payload = json!(item);
    if let Some(assets) = assets {
        data.assets.insert(id, assets);
    }
    let _ = job.events.send(("item", payload));
    let _ = job.events.send(("job", data.progress()));
}

pub(super) fn start(state: Arc<State>, job: Arc<Job>, cfg: Value) -> ApiResult<()> {
    let concurrency = cfg["llm"]["concurrency"].as_u64().unwrap_or(10).max(1) as usize;
    let runtime =
        Arc::new(markitai_core::LlmRuntime::new(concurrency).map_err(ApiError::internal)?);
    let browser_runtime =
        Arc::new(markitai_core::BrowserRuntime::new(8).map_err(ApiError::internal)?);
    let _ = job.runtime.set(runtime.clone());
    let count = job.data.lock().unwrap().items.len();
    job.active.fetch_add(count, Ordering::SeqCst);
    let explicit = job.data.lock().unwrap().options["strategy"]
        .as_str()
        .map(str::to_owned);
    let task_state = state.clone();
    let task = crate::task::spawn(async move {
        let mut pending = FuturesUnordered::new();
        for index in 0..count {
            pending.push(convert_one(
                task_state.clone(),
                job.clone(),
                index,
                cfg.clone(),
                runtime.clone(),
                browser_runtime.clone(),
                explicit.clone(),
            ));
        }
        while pending.next().await.is_some() {
            complete(&task_state, job.clone()).await;
        }
        // History retains no Chromium processes or authenticated sessions.
        let _ = crate::task::blocking(move || browser_runtime.close()).await;
    });
    state.tasks.lock().unwrap().push(task);
    Ok(())
}

// Admission and finalization take access before data; no network work holds either.
pub(super) async fn complete(state: &Arc<State>, job: Arc<Job>) {
    let finalized = crate::task::blocking(move || {
        let _access = job.access.lock().unwrap();
        if job.active.fetch_sub(1, Ordering::SeqCst) == 1 {
            store::finish(&job)
        } else {
            Ok(())
        }
    })
    .await;
    if !matches!(finalized, Ok(Ok(()))) {
        state.persistence_failed.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod pricing_tests {
    use super::*;
    #[test]
    fn history_prices_retained_outputs_separately_from_a_failed_latest_attempt() {
        let known = json!({"m":{"requests":1,"input_tokens":4,"output_tokens":2,"cost_usd":0.5,"priced_requests":1,"unpriced_requests":0,"cost_status":"complete","pricing_snapshot":"catalog-v1"}});
        let usage = markitai_core::ConversionUsage {
            requests: 1,
            input_tokens: 4,
            output_tokens: 2,
            cost_usd: 0.5,
            by_model: known.as_object().unwrap().clone(),
        };
        let mut item = Item::new(
            1,
            "https://example.test/document".into(),
            "url",
            Some("base.md".into()),
        );
        item.status = "done".into();
        item.output = Some("base.md".into());
        item.cost_usd = Some(0.5);
        item.pricing = crate::pricing::Pricing::from_usage(&usage);
        let unknown = markitai_core::ConversionUsage { requests:1, by_model:json!({"unknown":{"requests":1,"input_tokens":0,"output_tokens":0,"cost_usd":0.0,"priced_requests":0,"unpriced_requests":1,"cost_status":"unknown"}}).as_object().unwrap().clone(), ..Default::default() };
        item.diagnostics =
            AttemptDiagnostics::failed(Operation::Enhance, "later provider failure", unknown);
        let data = JobData {
            id: "job".into(),
            created_at: now(),
            finished_at: Some(now()),
            status: "done".into(),
            persistence_error: None,
            options: json!({}),
            items: vec![item],
            size: 0,
            bases: HashMap::new(),
            assets: HashMap::new(),
            item_options: HashMap::new(),
            transactions: Vec::new(),
        };
        let history = data.history();
        assert_eq!(history["cost_usd"], 0.5);
        assert_eq!(history["pricing"]["cost_status"], "complete");
        assert_eq!(history["pricing"]["priced_requests"], 1);
        let snapshot = data.snapshot();
        assert_eq!(
            snapshot["items"][0]["diagnostics"]["last_attempt"]["usage"]["by_model"]["unknown"]["cost_status"],
            "unknown"
        );
        assert_eq!(snapshot["items"][0]["output"], "base.md");
    }

    fn fixture(items: Vec<Item>) -> JobData {
        JobData {
            id: "job".into(),
            created_at: now(),
            finished_at: Some(now()),
            status: "done".into(),
            persistence_error: None,
            options: json!({}),
            items,
            size: 0,
            bases: HashMap::new(),
            assets: HashMap::new(),
            item_options: HashMap::new(),
            transactions: Vec::new(),
        }
    }
    #[test]
    fn item_options_preserve_independent_routes_without_replaying_stored_consent() {
        let mut data = fixture(vec![
            Item::new(1, "a.txt".into(), "file", None),
            Item::new(2, "b.txt".into(), "file", None),
        ]);
        data.options = json!({"backend":"native","remote_processing":"cloudflare","origin":"cli","api_key":"not-public"});
        data.item_options.insert(
            "i1".into(),
            json!({"backend":"cloudflare","profile":"rag","remote_processing":"cloudflare"}),
        );
        let snapshot = data.snapshot();
        for (index, backend) in [(0, "cloudflare"), (1, "native")] {
            let options = &snapshot["items"][index]["options"];
            assert_eq!(options["backend"], backend);
            for hidden in ["remote_processing", "origin", "api_key"] {
                assert!(options.get(hidden).is_none());
            }
            // The SSE item adapter must match the snapshot's authoritative selections.
            let event=data.with_item_options(json!({"item_id":format!("i{}",index+1),"status":"done","options":{"remote_processing":"cloudflare"}}));
            assert_eq!(&event["options"], options);
        }
        assert_eq!(snapshot["items"][0]["options"]["profile"], "rag");
        assert!(snapshot["items"][1]["options"]["profile"].is_null());
        // Old histories with no per-item map still expose the job's saved selections.
        data.item_options.clear();
        assert_eq!(data.snapshot()["items"][0]["options"]["backend"], "native");
        // Presentation never mutates the stored metadata or converts disclosure into permission.
        assert_eq!(data.options["remote_processing"], "cloudflare");
    }

    fn priced_item() -> Item {
        let mut item = Item::new(1, "priced.md".into(), "file", None);
        item.status = "done".into();
        item.output = Some("priced.md".into());
        item.cost_usd = Some(0.5);
        item.pricing = crate::pricing::Pricing::from_usage(&markitai_core::ConversionUsage {
            requests:1, cost_usd:0.5, by_model:json!({"known":{"requests":1,"cost_usd":0.5,"priced_requests":1,"unpriced_requests":0,"cost_status":"complete","pricing_snapshot":"catalog-v1"}}).as_object().unwrap().clone(), ..Default::default()
        });
        item
    }
    #[test]
    fn history_mixed_legacy_observation_is_partial_without_inventing_request_counts() {
        let mut old = Item::new(2, "legacy.md".into(), "file", None);
        old.status = "done".into();
        old.output = Some("legacy.md".into());
        old.cost_usd = Some(0.25);
        old.diagnostics = AttemptDiagnostics::completed(Operation::Convert, markitai_core::ConversionUsage {
            requests:2, input_tokens:8, cost_usd:0.25,
            by_model:json!({"legacy":{"requests":2,"input_tokens":8,"output_tokens":0,"cost_usd":0.25}}).as_object().unwrap().clone(), ..Default::default()
        });
        let history = fixture(vec![priced_item(), old]).history();
        assert_eq!(history["cost_usd"], 0.75);
        assert_eq!(history["pricing"]["cost_status"], "partial");
        assert_eq!(history["pricing"]["priced_requests"], 1);
        assert_eq!(history["pricing"]["unpriced_requests"], 2);
        assert_eq!(
            history["pricing"]["pricing_snapshots"],
            json!(["catalog-v1"])
        );
        let mut missing_scalar = Item::new(3, "missing-scalar.md".into(), "file", None);
        missing_scalar.status = "done".into();
        missing_scalar.output = Some("missing-scalar.md".into());
        missing_scalar.diagnostics = AttemptDiagnostics::completed(
            Operation::Convert,
            markitai_core::ConversionUsage {
                requests: 2,
                ..Default::default()
            },
        );
        let history = fixture(vec![priced_item(), missing_scalar]).history();
        assert_eq!(history["cost_usd"], 0.5);
        assert!(history.get("pricing").is_none());
    }
    #[test]
    fn history_unattributed_legacy_cost_suppresses_completeness_not_the_subtotal() {
        let mut old = Item::new(2, "legacy.md".into(), "file", None);
        old.status = "done".into();
        old.output = Some("legacy.md".into());
        old.cost_usd = Some(0.25);
        assert!(old.diagnostics.is_none());
        let history = fixture(vec![priced_item(), old.clone()]).history();
        assert_eq!(history["cost_usd"], 0.75);
        assert!(history.get("pricing").is_none());
        old.diagnostics = AttemptDiagnostics::failed(
            Operation::Enhance,
            "later failure",
            markitai_core::ConversionUsage {
                requests: 7,
                ..Default::default()
            },
        );
        let history = fixture(vec![priced_item(), old]).history();
        assert_eq!(history["cost_usd"], 0.75);
        assert!(history.get("pricing").is_none());
    }
    #[test]
    fn history_does_not_invent_unknown_work_for_an_unmetered_zero_cost_output() {
        let mut no_model = Item::new(2, "plain.md".into(), "file", None);
        no_model.status = "done".into();
        no_model.output = Some("plain.md".into());
        no_model.cost_usd = Some(0.0);
        let history = fixture(vec![priced_item(), no_model]).history();
        assert_eq!(history["pricing"]["cost_status"], "complete");
        assert_eq!(history["pricing"]["priced_requests"], 1);
        assert_eq!(history["pricing"]["unpriced_requests"], 0);
    }
}
