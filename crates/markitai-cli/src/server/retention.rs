//! History retention. A saved job is deleted seven days after it finished,
//! as the reference service does (`JOB_TTL_HOURS = 168`, swept at startup and
//! hourly). An expired job is hidden from every route before a sweep removes
//! it; running jobs and jobs with admitted work never expire.
use super::{jobs::Job, store, types::instant};
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{Arc, Mutex, atomic::Ordering},
    time::{Duration, SystemTime},
};

/// Days a finished job is kept, reported as `limits.history_retention_days`.
pub(super) const DAYS: u64 = 7;
const PERIOD: Duration = Duration::from_secs(DAYS * 24 * 60 * 60);
/// How often a running server sweeps.
pub(super) const INTERVAL: Duration = Duration::from_secs(60 * 60);

/// When a job's retention started: its recorded `finished_at`, else (an older
/// or damaged record) its folder's modification time.
fn finished(recorded: Option<&str>, folder: &Path) -> Option<SystemTime> {
    match recorded.and_then(instant) {
        Some(time) => Some(time.into()),
        None => fs::metadata(folder).and_then(|meta| meta.modified()).ok(),
    }
}

fn past(start: Option<SystemTime>, now: SystemTime) -> bool {
    start.is_some_and(|start| now.duration_since(start).is_ok_and(|age| age > PERIOD))
}

/// Whether `job` has outlived its retention at `now`.
pub(super) fn expired(job: &Job, now: SystemTime) -> bool {
    if job.active.load(Ordering::SeqCst) > 0 {
        return false;
    }
    let recorded = {
        let data = job.data.lock().unwrap();
        if data.status == "running" {
            return false;
        }
        data.finished_at.clone()
    };
    past(finished(recorded.as_deref(), &job.folder), now)
}

/// Remove `job` as `DELETE /api/history/{id}` does, if it is still the
/// registered job of that id, idle and expired once its access lock is held.
fn remove(
    job: &Arc<Job>,
    registry: &Mutex<HashMap<String, Arc<Job>>>,
    id: &str,
    now: SystemTime,
) -> std::io::Result<bool> {
    let _access = job.access.lock().unwrap();
    let current = registry
        .lock()
        .unwrap()
        .get(id)
        .is_some_and(|known| Arc::ptr_eq(known, job));
    if !current || !expired(job, now) || !job.retry_pending.lock().unwrap().is_empty() {
        return Ok(false);
    }
    markitai_core::output::check_path(&job.folder, false).map_err(std::io::Error::other)?;
    fs::remove_dir_all(&job.folder)?;
    registry.lock().unwrap().remove(id);
    Ok(true)
}

/// A job folder no scan registered (unreadable or damaged): its own record's
/// `finished_at` decides, else its modification time.
fn unregistered_expired(folder: &Path, now: SystemTime) -> bool {
    let recorded = store::safe_file(folder, "meta.json")
        .ok()
        .filter(|path| fs::metadata(path).is_ok_and(|meta| meta.len() <= 16 * 1024 * 1024))
        .and_then(|path| fs::read(path).ok())
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| value["finished_at"].as_str().map(str::to_owned));
    past(finished(recorded.as_deref(), folder), now)
}

/// Delete every job whose retention ended before `now`; returns how many.
/// The caller has registered every readable job (`store::rehydrate`).
pub(super) fn sweep(
    root: &Path,
    registry: &Mutex<HashMap<String, Arc<Job>>>,
    now: SystemTime,
) -> std::io::Result<usize> {
    let mut removed = 0;
    let registered: Vec<(String, Arc<Job>)> = registry
        .lock()
        .unwrap()
        .iter()
        .map(|(id, job)| (id.clone(), job.clone()))
        .collect();
    for (id, job) in registered {
        if expired(&job, now) && remove(&job, registry, &id, now)? {
            removed += 1;
        }
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if !store::job_id(&id) || registry.lock().unwrap().contains_key(&id) {
            continue;
        }
        let folder = entry.path();
        if !entry.file_type()?.is_dir()
            || markitai_core::output::check_path(&folder, false).is_err()
            || !unregistered_expired(&folder, now)
        {
            continue;
        }
        // A scan may have registered it meanwhile; that job is swept next time.
        if registry.lock().unwrap().contains_key(&id) {
            continue;
        }
        fs::remove_dir_all(&folder)?;
        removed += 1;
    }
    Ok(removed)
}

/// Sweep on a blocking thread; a failure is logged, never fatal.
pub(super) async fn sweep_now(state: &Arc<super::State>) {
    let state = state.clone();
    let swept = crate::task::blocking(move || {
        store::rehydrate(&state.root, &state.jobs)?;
        sweep(&state.root, &state.jobs, SystemTime::now())
    })
    .await;
    match swept {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => eprintln!("Warning: Serve: history retention sweep failed: {error}"),
        Err(error) => eprintln!("Warning: Serve: history retention sweep failed: {error}"),
    }
}

/// Sweep every `INTERVAL` until shutdown.
pub(super) fn start(state: &Arc<super::State>) {
    let task_state = state.clone();
    let mut shutdown = state.shutdown.subscribe();
    let task = crate::task::spawn(async move {
        while !*shutdown.borrow() {
            tokio::select! {
                changed = shutdown.changed() => if changed.is_err() { break },
                _ = tokio::time::sleep(INTERVAL) => sweep_now(&task_state).await,
            }
        }
    });
    state.tasks.lock().unwrap().push(task);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::{jobs::JobData, types::Item};
    use markitai_core::output::create_private_dir;
    use serde_json::json;

    fn days_ago(now: SystemTime, days: u64) -> String {
        chrono::DateTime::<chrono::Utc>::from(now - Duration::from_secs(days * 24 * 60 * 60))
            .to_rfc3339()
    }
    fn saved(root: &Path, id: &str, status: &str, finished: Option<String>) -> Arc<Job> {
        let folder = root.join(id);
        create_private_dir(&folder.join("out")).unwrap();
        let mut item = Item::new(1, "a.txt".into(), "file", Some("a.md".into()));
        item.status = "done".into();
        let data = JobData {
            id: id.into(),
            created_at: finished.clone().unwrap_or_default(),
            finished_at: finished,
            status: status.into(),
            persistence_error: None,
            options: json!({}),
            items: vec![item],
            size: 0,
            bases: HashMap::new(),
            assets: HashMap::new(),
            item_options: HashMap::new(),
            transactions: Vec::new(),
            label: None,
            owner: None,
        };
        store::persist(&folder, &data).unwrap();
        Arc::new(Job::new(folder, data))
    }

    #[test]
    fn a_sweep_removes_jobs_finished_more_than_seven_days_ago_and_nothing_running() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let now = SystemTime::now();
        let registry = Mutex::new(HashMap::new());
        for (id, status, days) in [
            ("00000000000a", "done", 8),
            ("00000000000b", "done", 6),
            ("00000000000c", "running", 30),
            ("00000000000d", "error", 9),
        ] {
            let job = saved(root, id, status, Some(days_ago(now, days)));
            registry.lock().unwrap().insert(id.to_owned(), job);
        }
        // An old job with a queued retry is kept too.
        let busy = saved(root, "00000000000e", "done", Some(days_ago(now, 10)));
        busy.active.fetch_add(1, Ordering::SeqCst);
        registry
            .lock()
            .unwrap()
            .insert("00000000000e".into(), busy.clone());
        assert!(!expired(&busy, now));
        assert_eq!(sweep(root, &registry, now).unwrap(), 2);
        let mut kept: Vec<_> = registry.lock().unwrap().keys().cloned().collect();
        kept.sort();
        assert_eq!(kept, ["00000000000b", "00000000000c", "00000000000e"]);
        for id in ["00000000000a", "00000000000d"] {
            assert!(!root.join(id).exists(), "{id}");
        }
        // Two days later the six-day-old job has expired as well.
        let later = now + Duration::from_secs(2 * 24 * 60 * 60);
        let six = registry.lock().unwrap()["00000000000b"].clone();
        assert!(expired(&six, later));
        assert_eq!(sweep(root, &registry, later).unwrap(), 1);
        assert!(root.join("00000000000c").exists() && root.join("00000000000e").exists());
    }

    #[test]
    fn an_unregistered_folder_is_aged_by_its_record_or_else_its_modification_time() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let now = SystemTime::now();
        // A record no scan accepts, with a readable finished_at.
        let old = saved(root, "0000000000aa", "done", Some(days_ago(now, 8)));
        let recent = saved(root, "0000000000bb", "done", Some(days_ago(now, 1)));
        // No readable record: the folder's own time decides.
        let damaged = root.join("0000000000cc");
        create_private_dir(&damaged).unwrap();
        fs::write(damaged.join("meta.json"), "{").unwrap();
        // Not a job folder name: never touched.
        create_private_dir(&root.join("keep-me")).unwrap();
        create_private_dir(&root.join(".upload-x")).unwrap();
        let registry = Mutex::new(HashMap::new());
        assert_eq!(sweep(root, &registry, now).unwrap(), 1);
        assert!(!old.folder.exists());
        assert!(recent.folder.exists() && damaged.exists());
        let later = now + Duration::from_secs(8 * 24 * 60 * 60);
        assert_eq!(sweep(root, &registry, later).unwrap(), 2);
        assert!(!recent.folder.exists() && !damaged.exists());
        assert!(root.join("keep-me").exists() && root.join(".upload-x").exists());
    }
}
