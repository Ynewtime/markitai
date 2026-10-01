use super::{State, tools::Options};
use futures_util::{StreamExt, stream::FuturesUnordered};
use indexmap::IndexMap;
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{Arc, atomic::Ordering},
};

struct Job {
    directory: PathBuf,
    status: &'static str,
    slots: Vec<Option<Value>>,
    done: usize,
    failed: usize,
}

#[derive(Default)]
pub(super) struct Table {
    jobs: IndexMap<String, Job>,
    forgotten: VecDeque<String>,
}

impl Table {
    pub fn snapshot(&self, id: &str) -> Result<Value, String> {
        let Some(job) = self.jobs.get(id) else {
            return Err(if self.forgotten.iter().any(|old| old == id) {
                format!(
                    "Job {id:?} finished and has been forgotten — this server keeps 100 recent finished jobs. Its output files remain on disk."
                )
            } else {
                format!(
                    "Unknown job id {id:?}. Jobs live only in server memory; check the id or read the files the job wrote."
                )
            });
        };
        let mut value = json!({"job_id":id,"status":job.status,"total":job.slots.len(),"done":job.done,
            "failed":job.failed,"output_dir":job.directory,"results":job.slots.iter().flatten().collect::<Vec<_>>()});
        let summaries: Vec<crate::pricing::Pricing> = job
            .slots
            .iter()
            .flatten()
            .filter_map(|item| item.get("pricing"))
            .filter_map(|pricing| serde_json::from_value(pricing.clone()).ok())
            .collect();
        if let Some(pricing) = crate::pricing::Pricing::aggregate(summaries.iter()) {
            value["pricing"] = json!(pricing);
            let cost: f64 = job
                .slots
                .iter()
                .flatten()
                .filter_map(|item| {
                    item.get("cost_usd")
                        .or_else(|| item.pointer("/diagnostics/last_attempt/usage/cost_usd"))
                        .and_then(Value::as_f64)
                })
                .sum();
            // An empty sum is -0.0; a job without costs reports 0.
            if cost.is_finite() {
                value["cost_usd"] = json!(cost + 0.0);
            }
        }
        Ok(value)
    }

    fn finish(&mut self, id: &str, cancelled: bool) {
        if let Some(job) = self.jobs.get_mut(id) {
            job.status = if cancelled && job.done < job.slots.len() {
                "cancelled"
            } else {
                "completed"
            };
        }
        let finished: Vec<_> = self
            .jobs
            .iter()
            .filter(|(_, job)| job.status != "running")
            .map(|(id, _)| id.clone())
            .collect();
        for id in finished.iter().take(finished.len().saturating_sub(100)) {
            self.jobs.shift_remove(id);
            self.forgotten.push_back(id.clone());
            if self.forgotten.len() > 500 {
                self.forgotten.pop_front();
            }
        }
    }
}

pub(super) fn start(
    state: &Arc<State>,
    sources: Vec<String>,
    directory: PathBuf,
    options: Options,
    concurrency: usize,
) -> Result<Value, String> {
    let cfg = state.config()?;
    let runtime = Arc::new(
        markitai_core::LlmRuntime::new(cfg["llm"]["concurrency"].as_u64().unwrap_or(10) as usize)
            .map_err(|error| error.to_string())?,
    );
    let mut background = state.background.lock().unwrap();
    if state.closing.load(Ordering::SeqCst) {
        return Err("MCP server is shutting down".into());
    }
    background.retain(|task| !task.is_finished());
    let mut table = state.jobs.lock().unwrap();
    if table
        .jobs
        .values()
        .filter(|job| job.status == "running")
        .count()
        >= 100
    {
        return Err("MCP server already has 100 running jobs".into());
    }
    let id = loop {
        let candidate = uuid::Uuid::new_v4().simple().to_string()[..8].to_owned();
        if !table.jobs.contains_key(&candidate)
            && !table.forgotten.contains(&candidate)
            && !directory.join(format!("batch-{candidate}")).exists()
        {
            break candidate;
        }
    };
    let total = sources.len();
    table.jobs.insert(
        id.clone(),
        Job {
            directory: directory.clone(),
            status: "running",
            slots: vec![None; total],
            done: 0,
            failed: 0,
        },
    );
    drop(table);
    let ack = json!({"job_id":id,"status":"running","total":total,"output_dir":directory});
    let state = state.clone();
    background.push(crate::task::spawn(async move {
        let mut next = sources.into_iter().enumerate();
        let mut pending = FuturesUnordered::new();
        let mut exhausted = false;
        loop {
            while pending.len() < concurrency && !exhausted && !state.closing.load(Ordering::SeqCst) {
                let Some((index, source)) = next.next() else { exhausted = true; break; };
                let output = directory.join(format!("batch-{id}")).join(format!("{:04}", index + 1));
                let state = state.clone();
                let options = options.clone();
                let runtime = runtime.clone();
                pending.push(async move {
                    let result = state.convert(source.clone(), output, options, Some(runtime)).await;
                    (index, source, result)
                });
            }
            let Some((index, source, result)) = pending.next().await else { break; };
            let (value, failed) = match result {
                Ok(mut result) => {
                    let mut value = json!({"source":source,"status":"ok","markdown_file":result["markdown_file"],"cost_usd":result["cost_usd"],"warnings":result["warnings"]});
                    if let Some(pricing) = result.as_object_mut().unwrap().remove("pricing") {
                        value["pricing"] = pricing;
                    }
                    if let Some(diagnostics) = result.as_object_mut().unwrap().remove("diagnostics") {
                        value["diagnostics"] = diagnostics;
                    }
                    (value, false)
                },
                Err(error) => {
                    let mut value = json!({"source":source,"status":"error","error":error.message});
                    if let Some(diagnostics) = error.diagnostics {
                        if let Some(pricing) = crate::pricing::Pricing::from_usage(&diagnostics.last_attempt.usage) {
                            value["pricing"] = json!(pricing);
                        }
                        value["diagnostics"] = json!(diagnostics);
                    }
                    (value, true)
                },
            };
            if let Some(job) = state.jobs.lock().unwrap().jobs.get_mut(&id) {
                job.slots[index] = Some(value);
                job.done += 1;
                job.failed += usize::from(failed);
            }
        }
        state.jobs.lock().unwrap().finish(&id, state.closing.load(Ordering::SeqCst));
    }));
    Ok(ack)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_pricing_combines_success_and_failure_observations_once() {
        let mut table = Table::default();
        table.jobs.insert("priced".into(), Job { directory:PathBuf::from("/isolated"), status:"completed", done:2, failed:1, slots:vec![
            Some(json!({"source":"a","status":"ok","cost_usd":0.5,"pricing":{"priced_requests":1,"unpriced_requests":0,"cost_status":"complete","pricing_snapshots":["catalog-v1"]},"diagnostics":{"last_attempt":{"usage":{"cost_usd":0.5}}}})),
            Some(json!({"source":"b","status":"error","pricing":{"priced_requests":0,"unpriced_requests":1,"cost_status":"unknown","pricing_snapshots":[]},"diagnostics":{"last_attempt":{"usage":{"cost_usd":0.0}}}})),
        ] });
        let value = table.snapshot("priced").unwrap();
        assert_eq!(value["cost_usd"], 0.5);
        assert_eq!(
            value["pricing"],
            json!({"priced_requests":1,"unpriced_requests":1,"cost_status":"partial","pricing_snapshots":["catalog-v1"]})
        );
        assert_eq!(value["results"].as_array().unwrap().len(), 2);
    }
    #[test]
    fn retention_preserves_running_jobs_and_distinguishes_expired_ids() {
        let mut table = Table::default();
        table.jobs.insert(
            "running".into(),
            Job {
                directory: PathBuf::from("/isolated"),
                status: "running",
                slots: vec![None],
                done: 0,
                failed: 0,
            },
        );
        for index in 0..605 {
            let id = format!("job-{index}");
            table.jobs.insert(
                id.clone(),
                Job {
                    directory: PathBuf::from("/isolated"),
                    status: "running",
                    slots: vec![Some(
                        json!({"source":"x","status":"error","error":"missing"}),
                    )],
                    done: 1,
                    failed: 1,
                },
            );
            table.finish(&id, false);
        }
        assert_eq!(table.jobs.len(), 101);
        assert_eq!(table.forgotten.len(), 500);
        assert_eq!(table.snapshot("running").unwrap()["status"], "running");
        assert!(table.snapshot("job-0").unwrap_err().contains("Unknown"));
        assert!(table.snapshot("job-5").unwrap_err().contains("forgotten"));
        let status = table.snapshot("job-604").unwrap();
        assert_eq!(status["done"], 1);
        assert_eq!(status["failed"], 1);
        assert_eq!(status["status"], "completed");
    }
}
