//! The coordinator owns recovery mutations and admits work only after durable claims.
use super::*;
use crate::output_claims::{
    Claim, Error as ClaimError, MAX_NAMESPACE_PARENTS, MemberLeases, NamespaceBatch, Owner, Policy,
    PreparedDocument, PreparedNamespaces, PublicationGroup, RenderedMember, adopt_owner,
    reservation_members, reserve_keys,
};
use crate::run_state::{
    self, Entry, ItemKey, Limits, LoadOutcome, Mode, Scope, Snapshot, StateStore, Status, codec,
};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

pub(super) type Reservations = HashMap<(u64, u64), BTreeSet<ItemKey>>;

fn item_key(task: &Task) -> ItemKey {
    if is_url(&task.source) {
        ItemKey::Url(task.report_key.clone())
    } else {
        ItemKey::File(task.report_key.clone())
    }
}
fn entry<'a>(snapshot: &'a Snapshot, key: &ItemKey) -> Option<&'a Entry> {
    match key {
        ItemKey::File(key) => snapshot.documents.get(key),
        ItemKey::Url(key) => snapshot.urls.get(key),
    }
}
fn stem(task: &Task, cfg: &Value) -> String {
    cfg["output"]["filename"]
        .as_str()
        .map(|name| name.strip_suffix(".md").unwrap_or(name).to_owned())
        .or_else(|| task.reserved_stem.clone())
        .or_else(|| {
            task.filename
                .as_ref()
                .map(|name| name.strip_suffix(".md").unwrap_or(name).to_owned())
        })
        .unwrap_or_else(|| {
            if is_url(&task.source) {
                markitai_core::output::url_name(&task.source, &Default::default())
            } else {
                Path::new(&task.source)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            }
        })
}
fn members(stem: &str) -> [String; 2] {
    [format!("{stem}.md"), format!("{stem}.llm.md")]
}
fn present(path: &Path) -> Result<bool, ClaimError> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

/// A prior target grants no overwrite authority; the receipt must prove it.
pub(super) fn claim(
    task: &mut Task,
    cfg: &Value,
    owner: Option<Owner>,
    retry: Option<&Path>,
    reserved: &Reservations,
) -> Result<Option<Claim>, ClaimError> {
    let Some(directory) = task.output.as_deref() else {
        return Ok(None);
    };
    let mode = cfg["output"]["on_conflict"].as_str().unwrap_or("rename");
    let allow = config::enabled(cfg, "/output/allow_symlinks");
    let initial = if let Some(target) = retry {
        let parent = crate::report_store::resolve_path(
            target
                .parent()
                .ok_or_else(|| ClaimError::Invalid("retry target has no parent".into()))?,
        )?;
        if parent != crate::report_store::resolve_path(directory)? {
            return Err(ClaimError::Ownership(
                "retry target is outside its planned parent".into(),
            ));
        }
        target
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix(".md"))
            .ok_or_else(|| {
                ClaimError::Invalid("retry target is not a base Markdown filename".into())
            })?
            .to_owned()
    } else {
        stem(task, cfg)
    };
    // Bound conflict scans independently of the number of discovered work items.
    for version in 1..=10000usize {
        let candidate = if version == 1 {
            initial.clone()
        } else {
            format!("{initial}.v{version}")
        };
        let names = members(&candidate);
        let leases = match MemberLeases::acquire(directory, &names, allow) {
            Ok(leases) => leases,
            Err(ClaimError::Busy) if retry.is_none() && mode == "rename" => continue,
            Err(error) => return Err(error),
        };
        let identity = item_key(task);
        let blocked = leases.keys().iter().any(|key| {
            reserved
                .get(key)
                .is_some_and(|owners| owners.iter().any(|owner| owner != &identity))
        });
        if blocked {
            if retry.is_some() {
                return Err(ClaimError::Ownership(
                    "retry target is reserved by another completed or scheduled item".into(),
                ));
            }
            continue;
        }
        let occupied = names
            .iter()
            .map(|name| present(&leases.parent().join(name)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .any(|yes| yes);
        if retry.is_none() && mode == "rename" && occupied {
            continue;
        }
        let policy = if retry.is_some() {
            Policy::RetryOwned
        } else if mode == "overwrite" {
            Policy::Overwrite
        } else {
            Policy::NoClobber
        };
        let skip = retry.is_none() && mode == "skip" && occupied;
        task.reserved_stem = Some(candidate);
        return Claim::new(leases, owner, policy, skip).map(Some);
    }
    Err(ClaimError::Invalid(
        "output conflict candidate limit exceeded".into(),
    ))
}

fn options(cli: &Cli, cfg: &Value, scope: &Scope) -> Value {
    let mut options = json!({"llm":config::enabled(cfg,"/llm/enabled"),"ocr":config::enabled(cfg,"/ocr/enabled"),
        "screenshot":config::enabled(cfg,"/screenshot/enabled"),"alt":config::enabled(cfg,"/image/alt_enabled"),
        "desc":config::enabled(cfg,"/image/desc_enabled"),"input_dir":if scope.mode==Mode::Directory {scope.input.as_path()} else {scope.input.parent().unwrap()},
        "output_dir":scope.output});
    if scope.mode == Mode::Directory {
        options["scan_max_depth"] = json!(
            cli.max_depth
                .unwrap_or_else(|| cfg["batch"]["scan_max_depth"].as_u64().unwrap_or(5) as usize)
        );
        let globs: Vec<_> = cli
            .globs
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if !globs.is_empty() {
            options["glob_patterns"] = json!(globs);
        }
        options["concurrency"] = cfg["batch"]["concurrency"].clone();
    }
    options
}
fn discovered(tasks: &[Task], options: Value) -> Snapshot {
    let mut state = Snapshot {
        options: serde_json::from_value(options).expect("ordered state options"),
        ..Snapshot::default()
    };
    for task in tasks {
        if is_url(&task.source) {
            state.urls.insert(
                task.report_key.clone(),
                Entry {
                    source_file: task.source_file.clone(),
                    url: Some(task.source.clone()),
                    ..Entry::default()
                },
            );
        } else {
            state
                .documents
                .insert(task.report_key.clone(), Entry::default());
        }
    }
    state
}
fn owner(scope: &Scope, generation: &str, task: &Task) -> Owner {
    Owner {
        generation: generation.into(),
        mode: if scope.mode == Mode::Directory {
            "directory"
        } else {
            "url_list"
        }
        .into(),
        input: scope.input.clone(),
        output: scope.output.clone(),
        kind: if is_url(&task.source) { "url" } else { "file" }.into(),
        key: task.report_key.clone(),
    }
}
fn failure(task: &Task, index: usize, error: String) -> RunItem {
    recorded(
        task,
        index,
        Instant::now(),
        timestamp(),
        &Err(markitai_core::Error::Conversion(error).into()),
    )
}
fn terminal(store: &mut StateStore, task: &Task, record: &RunItem) -> Result<(), run_state::Error> {
    store.record(
        item_key(task),
        json!({"status":if record.status==ItemStatus::Failed {"failed"} else {"completed"},
        "output":record.output,"error":record.error,"diagnostics":record.diagnostics}),
    )?;
    Ok(())
}
struct Work {
    index: usize,
    task: Task,
    claim: Arc<Claim>,
    previous: Entry,
}

fn restore_unsent(work: Work, store: &mut StateStore, fatal: &mut Option<String>) {
    if let Err(error) = store.record(
        item_key(&work.task),
        json!({
            "status":work.previous.status, "target":work.previous.target,
            "output":work.previous.output, "error":work.previous.error,
            "diagnostics":work.previous.observations.get("diagnostics")
        }),
    ) {
        fatal.get_or_insert_with(|| error.to_string());
    }
    drop(work);
}

fn admit_window(
    admitted: Vec<Work>,
    queued: &mut VecDeque<Work>,
    store: &mut StateStore,
    fatal: &mut Option<String>,
) {
    if admitted.is_empty() {
        return;
    }
    if fatal.is_none()
        && let Err(error) = store.flush()
    {
        *fatal = Some(error.to_string());
    }
    for work in admitted {
        if fatal.is_some() || crate::signals::interrupted().is_some() {
            restore_unsent(work, store, fatal);
        } else {
            queued.push_back(work);
        }
    }
}

// Bound retained claims independently of configured concurrency. A conversion
// slot is released when preparation finishes, so serial batches can share a
// durability fence without keeping a worker blocked on publication.
const PUBLICATION_WINDOW: usize = 16;
const PUBLICATION_DELAY: Duration = Duration::from_millis(100);

struct Completion {
    work: Work,
    result: Result<PreparedItem, RunItem>,
}

struct CompletionSink<'a> {
    cfg: &'a Value,
    store: &'a mut StateStore,
    records: &'a mut Vec<RunItem>,
    fatal: &'a mut Option<String>,
}
impl CompletionSink<'_> {
    fn record(&mut self, work: Work, record: RunItem) {
        if let Err(error) = terminal(self.store, &work.task, &record) {
            self.fatal.get_or_insert_with(|| error.to_string());
        }
        self.records.push(record);
        // Keep this independent Arc even when group commit has returned an
        // error and already dropped its own copy of the claim.
        drop(work);
    }

    fn finish(
        &mut self,
        work: Work,
        item: PreparedItem,
        publish: impl FnOnce(
            markitai_core::PreparedConversion,
        ) -> markitai_core::DetailedResult<ConversionOutput>,
    ) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let PreparedItem {
                progress,
                conversion,
            } = item;
            let result = image_only_result(&work.task, publish(conversion));
            complete_item(&work.task, work.index, self.cfg, progress, result).0
        }));
        let record = result.unwrap_or_else(|_| {
            failure(
                &work.task,
                work.index,
                "Conversion publication panicked".into(),
            )
        });
        self.record(work, record);
    }
}

struct PendingPublications {
    group: PublicationGroup,
    items: Vec<(Work, PreparedItem)>,
    opened: Option<Instant>,
}
impl PendingPublications {
    fn new() -> Self {
        Self {
            group: PublicationGroup::new(),
            items: Vec::new(),
            opened: None,
        }
    }
    fn len(&self) -> usize {
        self.items.len()
    }
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    fn due(&self) -> bool {
        self.len() >= PUBLICATION_WINDOW
            || self
                .opened
                .is_some_and(|opened| opened.elapsed() >= PUBLICATION_DELAY)
    }
    fn wait(&self) -> Duration {
        self.opened
            .map(|opened| PUBLICATION_DELAY.saturating_sub(opened.elapsed()))
            .unwrap_or(Duration::from_millis(50))
            .min(Duration::from_millis(50))
    }

    fn commit(&mut self, sink: &mut CompletionSink<'_>) {
        if self.is_empty() {
            return;
        }
        let group = std::mem::replace(&mut self.group, PublicationGroup::new());
        let items = std::mem::take(&mut self.items);
        self.opened = None;
        let committed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| group.commit()));
        match committed {
            Ok(Ok(claims)) => {
                for (work, item) in items {
                    sink.finish(
                        work,
                        item,
                        markitai_core::PreparedConversion::finish_after_publication,
                    );
                }
                // Group-returned claims are additionally retained through every
                // terminal projection; each Work also owned its own Arc.
                drop(claims);
            }
            failure => {
                let message = match failure {
                    Ok(Err(error)) => error.to_string(),
                    Err(_) => "Output publication group panicked".into(),
                    Ok(Ok(_)) => unreachable!(),
                };
                for (work, item) in items {
                    sink.finish(work, item, |conversion| {
                        Err(conversion
                            .fail_publication(markitai_core::Error::Conversion(message.clone())))
                    });
                }
            }
        }
    }

    fn accept(&mut self, completion: Completion, sink: &mut CompletionSink<'_>) {
        let Completion { work, result } = completion;
        let mut item = match result {
            Ok(item) => item,
            Err(record) => {
                sink.record(work, record);
                return;
            }
        };
        let bytes = item
            .conversion
            .members()
            .iter()
            .try_fold(0_usize, |sum, member| sum.checked_add(member.bytes.len()));
        if item.conversion.members().is_empty()
            || !bytes.is_some_and(|bytes| PublicationGroup::new().can_fit(bytes))
        {
            self.commit(sink);
            let claim = Arc::clone(&work.claim);
            sink.finish(work, item, |conversion| {
                conversion.publish_immediately(claim.as_ref())
            });
            return;
        }
        let bytes = bytes.expect("bounded document size was checked");
        if !self.group.can_fit(bytes) {
            self.commit(sink);
        }
        let members = item
            .conversion
            .take_members()
            .into_iter()
            .map(|member| RenderedMember {
                path: member.path,
                bytes: member.bytes,
            })
            .collect();
        let prepared =
            PreparedDocument::prepare(Arc::clone(&work.claim), members).and_then(|document| {
                debug_assert_eq!(document.staged_bytes(), bytes);
                self.group.push(document)
            });
        match prepared {
            Ok(()) => {
                self.opened.get_or_insert_with(Instant::now);
                self.items.push((work, item));
            }
            Err(error) => sink.finish(work, item, |conversion| {
                Err(conversion
                    .fail_publication(markitai_core::Error::Conversion(error.to_string())))
            }),
        }
    }
}

// An older directory scan may have persisted work inside a package. Reject
// that scope before receipt adoption or checkpoint upgrade; replay is read-only
// and a stable state lock does not grant permission to rewrite old work.
fn reject_package_members(snapshot: &Snapshot, scope: &Scope) -> CliResult<()> {
    let mut checked = std::collections::HashMap::<PathBuf, bool>::new();
    let mut inspect = |source: PathBuf| -> CliResult<()> {
        let resolved = crate::report_store::resolve_path(&source).map_err(runtime)?;
        for path in [&source, &resolved] {
            for ancestor in path.ancestors().skip(1) {
                let package = *checked
                    .entry(ancestor.to_owned())
                    .or_insert_with(|| markitai_core::formats::is_numbers_package_path(ancestor));
                if package {
                    return Err(runtime(
                        "Recovery state contains an item inside a Numbers directory package; preserve this state and use a fresh output directory without --resume to convert the package as one document",
                    ));
                }
            }
        }
        Ok(())
    };
    for key in snapshot.documents.keys() {
        inspect(scope.input.join(key))?;
    }
    for entry in snapshot.urls.values() {
        if let Some(source) = entry
            .source_file
            .as_deref()
            .filter(|source| !source.is_empty())
        {
            inspect(PathBuf::from(source))?;
        }
    }
    Ok(())
}

struct NamespaceExecution<'a, Commit> {
    context: ConvertContext<'a>,
    commit: Commit,
}

#[derive(Clone, Copy)]
struct AdmissionSlots {
    files: usize,
    urls: usize,
    file_limit: usize,
    url_limit: usize,
}

fn select_admission_indices(
    files: &mut VecDeque<usize>,
    urls: &mut VecDeque<usize>,
    mut slots: AdmissionSlots,
    limit: usize,
) -> Vec<usize> {
    let mut selected = Vec::with_capacity(limit);
    while selected.len() < limit && crate::signals::interrupted().is_none() {
        let file = files.front().copied();
        let url = urls.front().copied();
        let file_available = file.is_some() && slots.files < slots.file_limit;
        let url_available = url.is_some() && slots.urls < slots.url_limit;
        // Fill available class capacity before buffering extra work from an
        // earlier-sorted class. Compare fractions without floating-point ties.
        let prefer_file = match (file_available, url_available) {
            (true, true) => {
                let file_load = slots.files as u128 * slots.url_limit as u128;
                let url_load = slots.urls as u128 * slots.file_limit as u128;
                file_load < url_load || (file_load == url_load && file < url)
            }
            (true, false) => true,
            (false, true) => false,
            (false, false) => match (file, url) {
                (Some(file), Some(url)) => file < url,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            },
        };
        let index = if prefer_file {
            slots.files += 1;
            files.pop_front().expect("selected pending file exists")
        } else {
            slots.urls += 1;
            urls.pop_front().expect("selected pending URL exists")
        };
        selected.push(index);
    }
    selected
}

struct NamespaceWindow {
    indices: Vec<usize>,
    failures: Vec<(usize, ClaimError)>,
    ready: Result<Option<PreparedNamespaces>, ClaimError>,
}

fn prepare_namespace_window(
    tasks: &[Task],
    indices: &[usize],
    cfg: &Value,
    commit: &mut impl FnMut(NamespaceBatch) -> Result<PreparedNamespaces, ClaimError>,
) -> NamespaceWindow {
    // An occupied skip historically does not need the receipt namespace at all.
    // Keep the entire policy, including resume retries, on the immediate path.
    if cfg["output"]["on_conflict"].as_str() == Some("skip") {
        return NamespaceWindow {
            indices: indices.to_vec(),
            failures: Vec::new(),
            ready: Ok(None),
        };
    }
    let mut batch = NamespaceBatch::new();
    let mut eligible = Vec::new();
    let mut failures = Vec::new();
    let allow = config::enabled(cfg, "/output/allow_symlinks");
    for &index in indices {
        let prepared = tasks[index]
            .output
            .as_deref()
            .ok_or_else(|| ClaimError::Invalid("batch task has no output parent".into()))
            .and_then(|parent| batch.prepare(parent, allow));
        match prepared {
            Ok(()) => eligible.push(index),
            Err(error) => failures.push((index, error)),
        }
    }
    let ready = if eligible.is_empty() {
        Ok(None)
    } else {
        commit(batch).map(Some)
    };
    NamespaceWindow {
        indices: eligible,
        failures,
        ready,
    }
}

pub(super) fn run(
    cli: &Cli,
    cfg: &Value,
    tasks: Vec<Task>,
    destination: BatchDestination<'_>,
    report_plan: Option<&report::ReportPlan>,
    clock: Instant,
    context: ConvertContext<'_>,
) -> CliResult<i32> {
    run_with_namespace(
        cli,
        cfg,
        tasks,
        destination,
        report_plan,
        clock,
        NamespaceExecution {
            context,
            commit: NamespaceBatch::commit,
        },
    )
}

fn run_with_namespace(
    cli: &Cli,
    cfg: &Value,
    mut tasks: Vec<Task>,
    destination: BatchDestination<'_>,
    report_plan: Option<&report::ReportPlan>,
    clock: Instant,
    execution: NamespaceExecution<
        '_,
        impl FnMut(NamespaceBatch) -> Result<PreparedNamespaces, ClaimError>,
    >,
) -> CliResult<i32> {
    let NamespaceExecution {
        context,
        mut commit,
    } = execution;
    let BatchDestination {
        mode,
        output,
        history,
    } = destination;
    let scope = Scope::new(
        if mode == RunMode::Directory {
            Mode::Directory
        } else {
            Mode::UrlList
        },
        Path::new(cli.input.as_deref().unwrap()),
        output,
    )
    .map_err(runtime)?;
    let state_options = options(cli, cfg, &scope);
    let hash = codec::task_hash(&scope, &state_options).map_err(runtime)?;
    if tasks.is_empty()
        && !scope
            .output
            .join(format!(".markitai/states/markitai.{hash}.state.json"))
            .exists()
    {
        if cli.json {
            emit_json(&[], None);
        }
        return Ok(0);
    }
    // Platform preflight precedes any request, worker or state mutation.
    #[cfg(not(unix))]
    return Err(runtime(
        "Durable output ownership is not implemented for this platform yet",
    ));
    let _signals = crate::signals::Guard::install().map_err(runtime)?;
    let allow = config::enabled(cfg, "/output/allow_symlinks");
    let mut store =
        StateStore::open(scope.clone(), &hash, allow, Limits::default()).map_err(runtime)?;
    let mut snapshot = if cli.resume {
        match store.load().map_err(runtime)? {
            LoadOutcome::Loaded { snapshot, warnings } => {
                for warning in warnings {
                    eprintln!("Warning: {warning}");
                }
                *snapshot
            }
            LoadOutcome::Missing => {
                if !cli.quiet {
                    say!(
                        "No recovery state matches these paths and options; starting a fresh batch.",
                        "没有与这些路径和选项匹配的恢复状态，将开始全新的批处理。"
                    );
                }
                Snapshot::default()
            }
            LoadOutcome::Corrupt { reason } => {
                say!(
                    "Warning: recovery state could not be loaded ({reason}); preserving it before a fresh checkpoint",
                    "Warning: 无法加载恢复状态（{reason}）；将先保留它，再建立新的检查点"
                );
                Snapshot::default()
            }
        }
    } else {
        Snapshot::default()
    };
    if cli.resume {
        reject_package_members(&snapshot, &scope)?;
    }
    let fresh = discovered(&tasks, state_options);
    let url_order: Vec<_> = fresh.urls.keys().cloned().collect();
    let native = snapshot.checkpoint.is_some();
    if let Some(checkpoint) = &snapshot.checkpoint {
        // Copy publication evidence before persisting an adopted URL identity. Keeping the old receipt makes a crash before begin safe.
        let mut adopted = BTreeSet::new();
        for key in &url_order {
            let bare = fresh.urls[key].url.as_deref().unwrap();
            if key == bare
                || snapshot.urls.contains_key(key)
                || fresh.urls.contains_key(bare)
                || !adopted.insert(bare.to_owned())
            {
                continue;
            }
            let Some(old) = snapshot.urls.get(bare) else {
                continue;
            };
            if old.status != Status::Failed || old.target.is_none() {
                continue;
            }
            let Some(path) = old.target.as_ref().or(old.output.as_ref()) else {
                continue;
            };
            let Some(parent) = path.parent() else {
                continue;
            };
            let previous = Owner {
                generation: checkpoint.generation.clone(),
                mode: if scope.mode == Mode::Directory {
                    "directory"
                } else {
                    "url_list"
                }
                .into(),
                input: scope.input.clone(),
                output: scope.output.clone(),
                kind: "url".into(),
                key: bare.into(),
            };
            let mut next = previous.clone();
            next.key = key.clone();
            let names = if old.target.is_some() {
                let base = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.strip_suffix(".md"))
                    .ok_or_else(|| runtime("Adopted URL target is not a Markdown filename"))?;
                members(base).to_vec()
            } else if let Some(names) =
                reservation_members(parent, &previous, path).map_err(runtime)?
            {
                names
            } else {
                continue;
            };
            let leases = MemberLeases::acquire(parent, &names, allow).map_err(runtime)?;
            adopt_owner(&leases, &previous, &next).map_err(runtime)?;
        }
    }
    if cli.resume {
        codec::merge(&mut snapshot, &fresh, &url_order, Limits::default()).map_err(runtime)?;
    } else {
        snapshot = fresh;
    }
    if snapshot.options.is_empty() {
        snapshot.options = discovered(&[], options(cli, cfg, &scope)).options;
    }
    for task in tasks.iter_mut().filter(|task| is_url(&task.source)) {
        let key = item_key(task);
        if let Some(saved) = entry(&snapshot, &key) {
            task.source_file = saved.source_file.clone();
            let parent = codec::entry_parent(&key, saved, &scope, allow).map_err(runtime)?;
            if task
                .output
                .as_ref()
                .map(|path| crate::report_store::resolve_path(path))
                .transpose()
                .map_err(runtime)?
                .as_ref()
                != Some(&parent)
            {
                task.output = Some(parent);
            }
        }
    }
    // Directory recovery retains pending files even when discovery no longer sees them.
    if mode == RunMode::Directory {
        let keys = crate::sort::set(
            tasks
                .iter()
                .filter(|t| !is_url(&t.source))
                .map(|t| t.report_key.clone()),
        );
        for (key, entry) in &snapshot.documents {
            if entry.status != Status::Completed && !keys.contains(key) {
                tasks.push(Task {
                    source: scope.input.join(key).to_string_lossy().into_owned(),
                    display: key.clone(),
                    report_key: key.clone(),
                    output: Some(
                        scope
                            .output
                            .join(Path::new(key).parent().unwrap_or(Path::new(""))),
                    ),
                    filename: None,
                    reserved_stem: None,
                    source_file: None,
                });
            }
        }
    }
    if !native {
        // Legacy targets are naming hints, not native write authority. Do not upgrade an unproved target into a retry claim if interrupted before admission.
        for saved in snapshot
            .documents
            .values_mut()
            .chain(snapshot.urls.values_mut())
        {
            if saved.status != Status::Completed {
                saved.target = None;
            }
        }
    }
    let retries: Vec<_> = tasks
        .iter()
        .map(|task| {
            entry(&snapshot, &item_key(task))
                .filter(|entry| native && entry.status == Status::Failed)
                .and_then(|entry| entry.target.clone())
        })
        .collect();
    // Reserve all current identities consistently before work, but acquire OS locks only for active work.
    let pending: Vec<_> = tasks
        .iter()
        .enumerate()
        .filter(|(_, task)| {
            entry(&snapshot, &item_key(task)).is_some_and(|entry| entry.status != Status::Completed)
        })
        .map(|(index, _)| index)
        .collect();
    let mut ordinary: Vec<_> = pending
        .iter()
        .filter(|&&index| retries[index].is_none())
        .map(|&index| tasks[index].clone())
        .collect();
    reserve_batch_names(&mut ordinary, cfg)?;
    for (index, planned) in pending
        .iter()
        .copied()
        .filter(|&index| retries[index].is_none())
        .zip(ordinary)
    {
        tasks[index] = planned;
    }
    store.begin(snapshot).map_err(runtime)?;
    if !cli.quiet
        && let Some(backup) = store.legacy_backup()
    {
        let backup = backup.display();
        say!(
            "Preserved original legacy recovery files at {backup}. This backup does not undo output or model work.",
            "已把原有的旧版恢复文件保留在 {backup}。该备份不会撤销已写出的输出或已完成的模型请求。"
        );
    }
    let generation = store
        .snapshot()
        .unwrap()
        .checkpoint
        .as_ref()
        .unwrap()
        .generation
        .clone();
    let mut reserved = Reservations::new();
    let snapshot = store.snapshot().unwrap();
    for (kind, key, entry) in snapshot
        .documents
        .iter()
        .map(|(key, entry)| ("file", key, entry))
        .chain(snapshot.urls.iter().map(|(key, entry)| ("url", key, entry)))
        .filter(|(_, _, entry)| entry.status == Status::Completed)
    {
        let Some(path) = &entry.output else { continue };
        let Some(parent) = path.parent() else {
            continue;
        };
        let identity = Owner {
            generation: generation.clone(),
            mode: if scope.mode == Mode::Directory {
                "directory"
            } else {
                "url_list"
            }
            .into(),
            input: scope.input.clone(),
            output: scope.output.clone(),
            kind: kind.into(),
            key: key.clone(),
        };
        let names = match reservation_members(parent, &identity, path).map_err(runtime)? {
            Some(names) => names,
            None => {
                // Legacy state lacks publication family evidence. Protect both interpretations of an .llm.md result.
                let Some(base) = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.strip_suffix(".md"))
                else {
                    continue;
                };
                let mut names = crate::sort::set(members(base));
                if let Some(base) = base.strip_suffix(".llm") {
                    names.extend(members(base));
                }
                names.into_iter().collect()
            }
        };
        let item = if kind == "file" {
            ItemKey::File(key.clone())
        } else {
            ItemKey::Url(key.clone())
        };
        for names in names.chunks(2) {
            for key in reserve_keys(parent, names, allow).map_err(runtime)? {
                reserved.entry(key).or_default().insert(item.clone());
            }
        }
    }
    let file_limit = cfg["batch"]["concurrency"].as_u64().unwrap_or(10) as usize;
    let url_limit = cfg["batch"]["url_concurrency"].as_u64().unwrap_or(5) as usize;
    // Separate queues avoid rescanning every blocked item whenever one worker finishes.
    let (mut pending_urls, mut pending_files): (VecDeque<_>, VecDeque<_>) = pending
        .into_iter()
        .partition(|&index| is_url(&tasks[index].source));
    // An absent task class cannot use workers reserved for its concurrency cap.
    let count = file_limit
        .min(pending_files.len())
        .saturating_add(url_limit.min(pending_urls.len()))
        .max(1);
    let admission_window = PUBLICATION_WINDOW.max(count);
    let seconds = cfg["batch"]["state_flush_interval_seconds"]
        .as_u64()
        .unwrap_or(0);
    let interval = Duration::from_secs(if seconds > 0 { seconds } else { 5 });
    let (jobs, job_receiver) = mpsc::channel::<Work>();
    let job_receiver = Arc::new(Mutex::new(job_receiver));
    let (finished, receiver) = mpsc::channel::<Completion>();
    let mut publications = PendingPublications::new();
    let mut queued = VecDeque::<Work>::new();
    let mut records = Vec::new();
    let mut fatal = None;
    let mut signal = None;
    let mut last_flush = Instant::now();
    let mut active_files = 0usize;
    let mut active_urls = 0usize;
    std::thread::scope(|threads| {
        for _ in 0..count {
            let jobs = Arc::clone(&job_receiver);
            let finished = finished.clone();
            threads.spawn(move || {
                loop {
                    let work = match jobs.lock() {
                        Ok(receiver) => receiver.recv(),
                        Err(_) => break,
                    };
                    let Ok(work) = work else { break };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        prepare_item(&work.task, cfg, context, work.claim.as_ref())
                    }))
                    .map_err(|_| {
                        failure(&work.task, work.index, "Conversion worker panicked".into())
                    });
                    if finished.send(Completion { work, result }).is_err() {
                        break;
                    }
                }
            });
        }
        drop(finished);
        loop {
            if signal.is_none()
                && let Some(received) = crate::signals::interrupted()
            {
                signal = Some(received);
                say!(
                    "Interrupted: stopping new work and waiting for active conversions.",
                    "Interrupted: 不再派发新任务，等待正在进行的转换完成。"
                );
                if let Err(error) = store.flush() {
                    fatal.get_or_insert_with(|| error.to_string());
                }
            }
            if signal.is_some() || fatal.is_some() {
                for work in queued.drain(..) {
                    restore_unsent(work, &mut store, &mut fatal);
                }
            }
            let mut admitted = Vec::new();
            if signal.is_none() && fatal.is_none() {
                // Already-reserved work occupies its class even before dispatch.
                // Refill the other class when completed publication frees space,
                // rather than waiting for a long queued class to empty first.
                let queued_urls = queued
                    .iter()
                    .filter(|work| is_url(&work.task.source))
                    .count();
                let mut slots = AdmissionSlots {
                    files: active_files + queued.len() - queued_urls,
                    urls: active_urls + queued_urls,
                    file_limit,
                    url_limit,
                };
                'admission: loop {
                    let occupied = active_files
                        + active_urls
                        + publications.len()
                        + queued.len()
                        + admitted.len();
                    if crate::signals::interrupted().is_some() || occupied >= admission_window {
                        break;
                    }
                    // This subwindow bounds namespace descriptors, not worker count.
                    // Several subwindows can share the one durable admission flush.
                    let selected = select_admission_indices(
                        &mut pending_files,
                        &mut pending_urls,
                        slots,
                        (admission_window - occupied).min(MAX_NAMESPACE_PARENTS),
                    );
                    if selected.is_empty() {
                        break;
                    }
                    let NamespaceWindow {
                        indices,
                        failures,
                        ready,
                    } = prepare_namespace_window(&tasks, &selected, cfg, &mut commit);
                    for (index, error) in failures {
                        let record = failure(&tasks[index], index, error.to_string());
                        if let Err(error) = terminal(&mut store, &tasks[index], &record) {
                            fatal = Some(error.to_string());
                        }
                        records.push(record);
                        if fatal.is_some() {
                            break 'admission;
                        }
                    }
                    let ready = match ready {
                        Ok(ready) => ready,
                        Err(error) => {
                            fatal = Some(error.to_string());
                            break 'admission;
                        }
                    };
                    for index in indices {
                        if crate::signals::interrupted().is_some() {
                            break 'admission;
                        }
                        let previous = entry(store.snapshot().unwrap(), &item_key(&tasks[index]))
                            .unwrap()
                            .clone();
                        let identity = owner(&scope, &generation, &tasks[index]);
                        let claimed: Result<Option<Claim>, ClaimError> = (|| {
                            let parent =
                                tasks[index].output.as_deref().expect("batch output parent");
                            if let Some(ready) = &ready {
                                ready.validate(parent, allow)?;
                            }
                            let claim = claim(
                                &mut tasks[index],
                                cfg,
                                Some(identity),
                                retries[index].as_deref(),
                                &reserved,
                            )?;
                            if let Some(ready) = &ready {
                                ready.validate(
                                    tasks[index].output.as_deref().expect("batch output parent"),
                                    allow,
                                )?;
                            }
                            Ok(claim)
                        })(
                        );
                        let claim = match claimed {
                            Ok(Some(claim)) => claim,
                            Ok(None) => unreachable!("batch always writes"),
                            Err(error) => {
                                let record = failure(&tasks[index], index, error.to_string());
                                if let Err(error) = terminal(&mut store, &tasks[index], &record) {
                                    fatal = Some(error.to_string());
                                }
                                records.push(record);
                                if fatal.is_some() {
                                    break 'admission;
                                }
                                continue;
                            }
                        };
                        let target = claim.parent().join(format!(
                            "{}.md",
                            tasks[index].reserved_stem.as_ref().unwrap()
                        ));
                        if let Err(error) = store.record(
                            item_key(&tasks[index]),
                            json!({"status":"in_progress","target":target,"output":null,"error":null,"diagnostics":null}),
                        ) {
                            fatal = Some(error.to_string());
                            break 'admission;
                        }
                        if claim.is_skip() {
                            // A skip performs no provider work and owns no output name after its short check.
                            if let Err(error) = store.flush() {
                                fatal = Some(error.to_string());
                                break 'admission;
                            }
                            last_flush = Instant::now();
                            let record =
                                convert_item(&tasks[index], index, cfg, context, Some(&claim)).0;
                            if let Err(error) = terminal(&mut store, &tasks[index], &record) {
                                fatal = Some(error.to_string());
                            }
                            records.push(record);
                            if fatal.is_some() {
                                break 'admission;
                            }
                            continue;
                        }
                        for key in claim.keys() {
                            reserved
                                .entry(key)
                                .or_default()
                                .insert(item_key(&tasks[index]));
                        }
                        if is_url(&tasks[index].source) {
                            slots.urls += 1;
                        } else {
                            slots.files += 1;
                        }
                        admitted.push(Work {
                            index,
                            task: tasks[index].clone(),
                            claim: Arc::new(claim),
                            previous,
                        });
                    }
                }
            }
            if !admitted.is_empty() {
                admit_window(admitted, &mut queued, &mut store, &mut fatal);
                last_flush = Instant::now();
            }
            // Claims and all in_progress reservations are durable before this
            // queue is dispatched. Selection still obeys both independent
            // conversion caps, rather than starting the entire reserved window.
            while signal.is_none() && fatal.is_none() && crate::signals::interrupted().is_none() {
                let Some(position) = queued.iter().position(|work| {
                    if is_url(&work.task.source) {
                        active_urls < url_limit
                    } else {
                        active_files < file_limit
                    }
                }) else {
                    break;
                };
                let work = queued
                    .remove(position)
                    .expect("selected reserved work exists");
                let url = is_url(&work.task.source);
                match jobs.send(work) {
                    Ok(()) => {
                        if url {
                            active_urls += 1;
                        } else {
                            active_files += 1;
                        }
                    }
                    Err(error) => {
                        fatal.get_or_insert("Conversion worker queue closed unexpectedly".into());
                        restore_unsent(error.0, &mut store, &mut fatal);
                    }
                }
            }
            if fatal.is_some() || signal.is_some() || crate::signals::interrupted().is_some() {
                for work in queued.drain(..) {
                    restore_unsent(work, &mut store, &mut fatal);
                }
            }
            let draining = signal.is_some() || fatal.is_some();
            let queues_empty =
                pending_files.is_empty() && pending_urls.is_empty() && queued.is_empty();
            if publications.due() || draining || (queues_empty && active_files + active_urls == 0) {
                publications.commit(&mut CompletionSink {
                    cfg,
                    store: &mut store,
                    records: &mut records,
                    fatal: &mut fatal,
                });
            }
            if active_files + active_urls == 0 && (queues_empty || draining) {
                break;
            }
            // A finished preparation opens a conversion slot even while its
            // staged output awaits the next group fence. Never wait for a
            // completion when the only remaining work has not been dispatched.
            if active_files + active_urls == 0 {
                continue;
            }
            match receiver.recv_timeout(publications.wait()) {
                Ok(first) => {
                    let mut ready = Some(first);
                    while let Some(completion) = ready {
                        if is_url(&completion.work.task.source) {
                            active_urls -= 1;
                        } else {
                            active_files -= 1;
                        }
                        publications.accept(
                            completion,
                            &mut CompletionSink {
                                cfg,
                                store: &mut store,
                                records: &mut records,
                                fatal: &mut fatal,
                            },
                        );
                        ready = receiver.try_recv().ok();
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => (),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    fatal.get_or_insert("Conversion workers stopped unexpectedly".into());
                    for work in queued.drain(..) {
                        restore_unsent(work, &mut store, &mut fatal);
                    }
                    publications.commit(&mut CompletionSink {
                        cfg,
                        store: &mut store,
                        records: &mut records,
                        fatal: &mut fatal,
                    });
                    break;
                }
            }
            if last_flush.elapsed() >= interval {
                if let Err(error) = store.flush() {
                    fatal.get_or_insert_with(|| error.to_string());
                }
                last_flush = Instant::now();
            }
        }
        drop(jobs);
    });
    if signal.is_none()
        && let Some(received) = crate::signals::interrupted()
    {
        signal = Some(received);
        say!(
            "Interrupted: stopping new work and waiting for active conversions.",
            "Interrupted: 不再派发新任务，等待正在进行的转换完成。"
        );
    }
    if let Err(error) = store.compact() {
        fatal.get_or_insert_with(|| error.to_string());
    }
    if let Some(error) = &fatal {
        eprintln!("Error: {error}");
    }
    let storage_failed = fatal.is_some();
    if signal.is_none()
        && fatal.is_none()
        && let Some(plan) = report_plan
    {
        let finished = RunFinished {
            updated_at: timestamp(),
            duration_s: clock.elapsed().as_secs_f64(),
        };
        let rendered = if cli.resume {
            report::render_resumed(
                plan,
                &records,
                store.snapshot().unwrap(),
                &url_order,
                &finished,
            )
        } else {
            report::render(plan, &records, &finished)
        };
        match rendered.and_then(|bytes| report::publish(plan, &bytes)) {
            Ok(publication) if cli.verbose && !cli.quiet => match publication {
                crate::report_store::Publication::Written(path) => {
                    let path = path.display();
                    say!("Report: {path}", "报告：{path}")
                }
                crate::report_store::Publication::SkippedExisting(path) => {
                    let path = path.display();
                    say!(
                        "Existing report preserved: {path}",
                        "已保留现有报告：{path}"
                    )
                }
            },
            Ok(_) => (),
            Err(error) => {
                eprintln!("Error: {error}");
                fatal.get_or_insert(error);
            }
        }
    }
    if signal.is_none()
        && !storage_failed
        && let Some(plan) = history
    {
        plan.record(&records);
    }
    if signal.is_none()
        && let Some(received) = crate::signals::interrupted()
    {
        signal = Some(received);
        say!(
            "Interrupted: stopping new work and waiting for active conversions.",
            "Interrupted: 不再派发新任务，等待正在进行的转换完成。"
        );
    }
    crate::sort::by_key(&mut records, |record| record.index);
    let items: Vec<_> = records.iter().map(outcome).collect();
    let failed = records
        .iter()
        .filter(|record| record.status == ItemStatus::Failed)
        .count();
    if cli.json {
        emit_json(
            &items,
            fatal.as_deref().or(if signal.is_some() {
                Some("Conversion interrupted")
            } else {
                None
            }),
        );
    } else {
        for record in &records {
            if !cli.quiet {
                for warning in &record.warnings {
                    eprintln!("Warning: {}: {warning}", record.display);
                }
            }
            if let Some(error) = &record.error {
                eprintln!("Error: {}: {error}", record.display);
            }
        }
        if !cli.quiet {
            print_batch_summary(&records, clock.elapsed(), output);
        }
    }
    Ok(if let Some(signal) = signal {
        128 + signal
    } else if storage_failed {
        1
    } else if failed > 0 {
        10
    } else if fatal.is_some() {
        1
    } else {
        0
    })
}

#[cfg(test)]
mod publication_tests {
    use super::*;
    use std::fs;

    struct Fixture {
        directory: tempfile::TempDir,
        cfg: Value,
        scope: Scope,
        hash: String,
        store: StateStore,
        tasks: Vec<Task>,
    }
    impl Fixture {
        fn new(names: &[&str]) -> Self {
            let directory = tempfile::tempdir().unwrap();
            let input = directory.path().join("input");
            fs::create_dir(&input).unwrap();
            let scope = Scope::new(Mode::Directory, &input, &directory.path().join("out")).unwrap();
            let cfg = config::normalize(&json!({
                "llm":{"enabled":false},"ocr":{"enabled":false},
                "screenshot":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false},
                "history":{"record":false},"log":{"dir":null},
                "prompts":{"dir":directory.path().join("prompts")},
                "cache":{"enabled":false,"global_dir":directory.path().join("cache")},
                "output":{"on_conflict":"rename"}
            }))
            .unwrap();
            let tasks: Vec<_> = names
                .iter()
                .map(|name| {
                    let source = scope.input.join(name);
                    fs::write(&source, format!("Complete authored {name} text 🚀.\n")).unwrap();
                    Task {
                        source: source.to_string_lossy().into_owned(),
                        display: (*name).into(),
                        report_key: (*name).into(),
                        output: Some(scope.output.clone()),
                        filename: None,
                        reserved_stem: None,
                        source_file: None,
                    }
                })
                .collect();
            let hash = codec::task_hash(&scope, &json!({})).unwrap();
            let mut store =
                StateStore::open(scope.clone(), &hash, false, Limits::default()).unwrap();
            store.begin(discovered(&tasks, json!({}))).unwrap();
            Self {
                directory,
                cfg,
                scope,
                hash,
                store,
                tasks,
            }
        }
        fn work(&mut self, index: usize) -> Work {
            let work = self.unflushed_work(index);
            self.store.flush().unwrap();
            work
        }
        fn unflushed_work(&mut self, index: usize) -> Work {
            let mut task = self.tasks[index].clone();
            let checkpoint = self.store.snapshot().unwrap().checkpoint.as_ref().unwrap();
            let identity = owner(&self.scope, &checkpoint.generation, &task);
            let claim = Arc::new(
                claim(
                    &mut task,
                    &self.cfg,
                    Some(identity),
                    None,
                    &Reservations::new(),
                )
                .unwrap()
                .unwrap(),
            );
            self.store.record(item_key(&task), json!({"status":"in_progress",
                "target":claim.parent().join(format!("{}.md", task.reserved_stem.as_ref().unwrap())),
                "output":null,"error":null,"diagnostics":null})).unwrap();
            Work {
                index,
                task,
                claim,
                previous: Entry::default(),
            }
        }
        fn sink<'a>(
            &'a mut self,
            records: &'a mut Vec<RunItem>,
            fatal: &'a mut Option<String>,
        ) -> CompletionSink<'a> {
            CompletionSink {
                cfg: &self.cfg,
                store: &mut self.store,
                records,
                fatal,
            }
        }
    }
    fn prepare(work: &Work, cfg: &Value) -> PreparedItem {
        prepare_item(
            &work.task,
            cfg,
            ConvertContext::default(),
            work.claim.as_ref(),
        )
    }

    #[test]
    fn serial_preparations_publish_exact_bytes_once_then_record_and_recover_completed() {
        let mut fixture = Fixture::new(&["a.txt", "b.txt"]);
        let mut pending = PendingPublications::new();
        let mut records = Vec::new();
        let mut fatal = None;
        let mut expected = Vec::new();
        for index in 0..2 {
            let work = fixture.work(index);
            let prepared = prepare(&work, &fixture.cfg);
            assert_eq!(prepared.conversion.members().len(), 1);
            let member = &prepared.conversion.members()[0];
            expected.push((member.path.clone(), member.bytes.clone()));
            fs::remove_file(&work.task.source).unwrap();
            pending.accept(
                Completion {
                    work,
                    result: Ok(prepared),
                },
                &mut fixture.sink(&mut records, &mut fatal),
            );
            assert!(records.is_empty());
            assert!(expected.iter().all(|(path, _)| !path.exists()));
            assert_eq!(
                fixture.store.snapshot().unwrap().documents
                    [fixture.tasks[index].report_key.as_str()]
                .status,
                Status::InProgress
            );
        }
        pending.commit(&mut fixture.sink(&mut records, &mut fatal));
        assert!(fatal.is_none());
        assert_eq!(records.len(), 2);
        for (index, (path, bytes)) in expected.iter().enumerate() {
            assert_eq!(fs::read(path).unwrap(), *bytes);
            assert_eq!(records[index].status, ItemStatus::Completed);
            assert_eq!(records[index].output.as_ref(), Some(path));
            assert_eq!(records[index].usage.requests, 0);
            assert!(records[index].diagnostics.is_none());
            MemberLeases::acquire(
                path.parent().unwrap(),
                &[path.file_name().unwrap().to_string_lossy().into_owned()],
                false,
            )
            .unwrap();
        }
        fixture.store.compact().unwrap();
        let Fixture {
            directory,
            scope,
            hash,
            store,
            ..
        } = fixture;
        drop(store);
        let mut reopened = StateStore::open(scope, &hash, false, Limits::default()).unwrap();
        let LoadOutcome::Loaded { snapshot, warnings } = reopened.load().unwrap() else {
            panic!("completed checkpoint missing")
        };
        assert!(warnings.is_empty());
        assert!(
            snapshot
                .documents
                .values()
                .all(|entry| entry.status == Status::Completed)
        );
        assert_eq!(
            snapshot.documents["a.txt"].output.as_ref(),
            Some(&expected[0].0)
        );
        drop(reopened);
        drop(directory);
    }

    #[test]
    fn group_failure_preserves_foreign_bytes_and_records_no_provisional_success() {
        let mut fixture = Fixture::new(&["a.txt", "b.txt"]);
        let mut pending = PendingPublications::new();
        let mut records = Vec::new();
        let mut fatal = None;
        let mut paths = Vec::new();
        for index in 0..2 {
            let work = fixture.work(index);
            let prepared = prepare(&work, &fixture.cfg);
            paths.push(prepared.conversion.members()[0].path.clone());
            pending.accept(
                Completion {
                    work,
                    result: Ok(prepared),
                },
                &mut fixture.sink(&mut records, &mut fatal),
            );
        }
        fs::write(&paths[0], b"foreign replacement").unwrap();
        pending.commit(&mut fixture.sink(&mut records, &mut fatal));
        assert!(fatal.is_none());
        assert_eq!(fs::read(&paths[0]).unwrap(), b"foreign replacement");
        assert!(!paths[1].exists());
        assert_eq!(records.len(), 2);
        assert!(
            records
                .iter()
                .all(|record| record.status == ItemStatus::Failed
                    && record.output.is_none()
                    && record.error.is_some())
        );
        assert!(
            fixture
                .store
                .snapshot()
                .unwrap()
                .documents
                .values()
                .all(|entry| entry.status == Status::Failed && entry.output.is_none())
        );
    }

    #[test]
    fn missing_source_error_flushes_prepared_neighbors_without_reconverting_them() {
        let mut fixture = Fixture::new(&["a.txt", "b.txt"]);
        let mut pending = PendingPublications::new();
        let mut records = Vec::new();
        let mut fatal = None;
        let a = fixture.work(0);
        let prepared = prepare(&a, &fixture.cfg);
        let path = prepared.conversion.members()[0].path.clone();
        let bytes = prepared.conversion.members()[0].bytes.clone();
        fs::remove_file(&a.task.source).unwrap();
        pending.accept(
            Completion {
                work: a,
                result: Ok(prepared),
            },
            &mut fixture.sink(&mut records, &mut fatal),
        );
        let b = fixture.work(1);
        fs::remove_file(&b.task.source).unwrap();
        let prepared = prepare(&b, &fixture.cfg);
        assert!(prepared.conversion.members().is_empty());
        pending.accept(
            Completion {
                work: b,
                result: Ok(prepared),
            },
            &mut fixture.sink(&mut records, &mut fatal),
        );
        assert!(pending.is_empty());
        assert!(fatal.is_none());
        assert_eq!(fs::read(path).unwrap(), bytes);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].status, ItemStatus::Completed);
        assert_eq!(records[1].status, ItemStatus::Failed);
        assert!(records[1].output.is_none());
        assert_eq!(
            fixture.store.snapshot().unwrap().documents["b.txt"].status,
            Status::Failed
        );
    }

    #[test]
    fn dropping_uncommitted_group_keeps_in_progress_state_and_releases_claims() {
        let mut fixture = Fixture::new(&["a.txt"]);
        let work = fixture.work(0);
        let claim = Arc::downgrade(&work.claim);
        let prepared = prepare(&work, &fixture.cfg);
        let path = prepared.conversion.members()[0].path.clone();
        let mut pending = PendingPublications::new();
        let mut records = Vec::new();
        let mut fatal = None;
        pending.accept(
            Completion {
                work,
                result: Ok(prepared),
            },
            &mut fixture.sink(&mut records, &mut fatal),
        );
        assert!(claim.upgrade().is_some());
        assert!(matches!(
            MemberLeases::acquire(path.parent().unwrap(), &["a.txt.md".into()], false),
            Err(ClaimError::Busy)
        ));
        drop(pending);
        assert!(claim.upgrade().is_none());
        assert!(!path.exists());
        assert!(records.is_empty());
        assert_eq!(
            fixture.store.snapshot().unwrap().documents["a.txt"].status,
            Status::InProgress
        );
        MemberLeases::acquire(path.parent().unwrap(), &["a.txt.md".into()], false).unwrap();
    }

    #[test]
    fn first_window_flush_failure_dispatches_no_work_or_document() {
        let mut fixture = Fixture::new(&["a.txt", "b.txt"]);
        let a = fixture.unflushed_work(0);
        let b = fixture.unflushed_work(1);
        let claims = [Arc::downgrade(&a.claim), Arc::downgrade(&b.claim)];
        let journal = fixture.scope.output.join(format!(
            ".markitai/states/markitai.{}.state.jsonl",
            fixture.hash
        ));
        assert!(!journal.exists());
        fs::create_dir(&journal).unwrap();
        let mut queued = VecDeque::new();
        let mut fatal = None;
        admit_window(vec![a, b], &mut queued, &mut fixture.store, &mut fatal);
        assert!(fatal.is_some());
        assert!(
            queued.is_empty(),
            "a failed durable admission must never reach a worker"
        );
        assert!(claims.iter().all(|claim| claim.upgrade().is_none()));
        assert!(!fixture.scope.output.join("a.txt.md").exists());
        assert!(!fixture.scope.output.join("b.txt.md").exists());
        assert!(
            journal.is_dir(),
            "the obstructing foreign object must be preserved"
        );
    }

    #[test]
    fn unsent_reservation_restores_every_previous_terminal_field() {
        let mut fixture = Fixture::new(&["a.txt"]);
        let mut work = fixture.work(0);
        let diagnostics = crate::diagnostics::AttemptDiagnostics::failed(
            crate::diagnostics::Operation::Convert,
            "earlier paid failure",
            ConversionUsage {
                requests: 1,
                input_tokens: 16,
                output_tokens: 7,
                ..Default::default()
            },
        )
        .unwrap();
        work.previous = Entry {
            status: Status::Failed,
            target: Some(fixture.scope.output.join("a.txt.md")),
            output: Some(fixture.scope.output.join("a.txt.llm.md")),
            error: Some("earlier paid failure".into()),
            observations: serde_json::from_value(json!({"diagnostics": diagnostics})).unwrap(),
            ..Entry::default()
        };
        let previous = work.previous.clone();
        let weak = Arc::downgrade(&work.claim);
        let mut fatal = None;
        restore_unsent(work, &mut fixture.store, &mut fatal);
        assert!(fatal.is_none());
        fixture.store.flush().unwrap();
        let restored = &fixture.store.snapshot().unwrap().documents["a.txt"];
        assert_eq!(restored.status, previous.status);
        assert_eq!(restored.target, previous.target);
        assert_eq!(restored.output, previous.output);
        assert_eq!(restored.error, previous.error);
        assert_eq!(
            restored.observations["diagnostics"],
            previous.observations["diagnostics"]
        );
        assert!(weak.upgrade().is_none());
        assert!(!fixture.scope.output.join("a.txt.md").exists());
    }
}

#[cfg(all(test, unix))]
mod namespace_coordination_tests {
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::os::unix::fs::PermissionsExt;
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    mod bounded_fixture_io {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/support/bounded_read.rs"
        ));
    }

    fn child_case(case: &str) {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("private-home")).unwrap();
        fs::create_dir(root.path().join("tmp")).unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "app::batch_run::namespace_coordination_tests::namespace_coordinator_helper",
                "--ignored",
                "--nocapture",
            ])
            .env_clear();
        for key in ["HOME", "PATH", "LANG", "LC_ALL", "TZ"] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        let mut child = command
            .env("MARKITAI_NAMESPACE_COORDINATOR_ROOT", root.path())
            .env("MARKITAI_NAMESPACE_COORDINATOR_CASE", case)
            .env("MARKITAI_HOME", root.path().join("private-home"))
            .env("TMPDIR", root.path().join("tmp"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "namespace coordinator subprocess failed: {case}"
                );
                assert_eq!(
                    fs::read_to_string(root.path().join("passed")).unwrap(),
                    case
                );
                return;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("namespace coordinator subprocess timed out: {case}");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn invalid_parent_fails_only_its_item_and_the_good_item_publishes() {
        child_case("partial");
    }

    #[test]
    fn skip_keeps_its_existing_shortcut_even_with_an_invalid_records_path() {
        child_case("skip");
    }

    #[test]
    fn failed_namespace_or_admission_flush_never_sends_a_provider_get() {
        child_case("zero-requests");
    }

    #[test]
    fn earlier_file_backlog_allows_continued_url_admission_before_release() {
        child_case("mixed-backlog");
    }

    fn cfg(root: &Path, policy: &str) -> Value {
        config::normalize(&json!({
            "llm":{"enabled":false},"ocr":{"enabled":false},"screenshot":{"enabled":false},
            "image":{"alt_enabled":false,"desc_enabled":false},
            "batch":{"concurrency":1,"url_concurrency":20,"state_flush_interval_seconds":3600},
            "fetch":{"strategy":"static","remote_consent":"never"},
            "cache":{"enabled":false,"global_dir":root.join("cache")},
            "prompts":{"dir":root.join("prompts")},"history":{"record":false},
            "output":{"on_conflict":policy,"report":false},"log":{"dir":null}
        }))
        .unwrap()
    }

    fn task(source: &Path, key: &str, output: &Path) -> Task {
        Task {
            source: source.to_string_lossy().into_owned(),
            display: key.into(),
            report_key: key.into(),
            output: Some(output.to_owned()),
            filename: None,
            reserved_stem: None,
            source_file: None,
        }
    }

    fn cli(input: &Path, output: &Path) -> Cli {
        Cli::try_parse_from([
            "markitai".to_owned(),
            input.to_string_lossy().into_owned(),
            "-o".into(),
            output.to_string_lossy().into_owned(),
            "--quiet".into(),
        ])
        .unwrap()
    }

    fn execute(
        cli: &Cli,
        cfg: &Value,
        tasks: Vec<Task>,
        output: &Path,
        mode: RunMode,
        commit: impl FnMut(NamespaceBatch) -> Result<PreparedNamespaces, ClaimError>,
    ) -> i32 {
        run_with_namespace(
            cli,
            cfg,
            tasks,
            BatchDestination {
                mode,
                output,
                history: None,
            },
            None,
            Instant::now(),
            NamespaceExecution {
                context: ConvertContext::default(),
                commit,
            },
        )
        .unwrap()
    }

    fn checkpoint(output: &Path) -> Value {
        let paths: Vec<_> = fs::read_dir(output.join(".markitai/states"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .ends_with(".state.json")
            })
            .collect();
        assert_eq!(paths.len(), 1);
        serde_json::from_slice(&fs::read(&paths[0]).unwrap()).unwrap()
    }

    fn bad_records(parent: &Path) -> PathBuf {
        let ownership = parent.join(".markitai/ownership");
        fs::create_dir_all(&ownership).unwrap();
        fs::set_permissions(&ownership, fs::Permissions::from_mode(0o700)).unwrap();
        let records = ownership.join("records");
        fs::write(&records, "foreign records file").unwrap();
        records
    }

    struct Observer {
        address: String,
        requests: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
        worker: Option<std::thread::JoinHandle<()>>,
    }
    impl Observer {
        fn new() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let address = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let count = Arc::clone(&requests);
            let stopping = Arc::clone(&stop);
            let worker = std::thread::spawn(move || {
                loop {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            count.fetch_add(1, Ordering::SeqCst);
                            let _ = stream.set_nonblocking(false);
                            let mut reader = bounded_fixture_io::Reader::new(
                                &stream,
                                Instant::now() + Duration::from_secs(1),
                            );
                            let mut head = [0_u8; 8192];
                            let _ = reader.read(&mut head);
                            let body = b"<article><h1>Unexpected request</h1><p>A complete authored response allows an incorrectly dispatched conversion to finish.</p></article>";
                            let _ = write!(
                                stream,
                                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                body.len()
                            );
                            let _ = stream.write_all(body);
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            if stopping.load(Ordering::SeqCst) {
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(error) => panic!("observer accept failed: {error}"),
                    }
                }
            });
            Self {
                address,
                requests,
                stop,
                worker: Some(worker),
            }
        }
        fn finish(&mut self) -> usize {
            self.stop.store(true, Ordering::SeqCst);
            if let Some(worker) = self.worker.take() {
                worker.join().unwrap();
            }
            self.requests.load(Ordering::SeqCst)
        }
    }
    impl Drop for Observer {
        fn drop(&mut self) {
            self.finish();
        }
    }

    #[derive(Default)]
    struct GateState {
        files_started: usize,
        urls_started: usize,
        released: bool,
        mixed_observed: bool,
        errors: Vec<String>,
    }

    struct MixedGate {
        address: String,
        state: Arc<(Mutex<GateState>, std::sync::Condvar)>,
        stop: Arc<AtomicBool>,
        worker: Option<std::thread::JoinHandle<()>>,
    }
    impl MixedGate {
        fn new() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let address = format!("http://{}", listener.local_addr().unwrap());
            let state = Arc::new((Mutex::new(GateState::default()), std::sync::Condvar::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let shared = Arc::clone(&state);
            let stopping = Arc::clone(&stop);
            let worker = std::thread::spawn(move || {
                let mut handlers = Vec::new();
                loop {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let shared = Arc::clone(&shared);
                            handlers.push(std::thread::spawn(move || {
                                if let Err(error) = Self::respond(stream, &shared) {
                                    let (lock, changed) = &*shared;
                                    let mut state = lock.lock().unwrap();
                                    state.errors.push(error.to_string());
                                    changed.notify_all();
                                }
                            }));
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            if stopping.load(Ordering::SeqCst) {
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(error) => panic!("mixed observer accept failed: {error}"),
                    }
                }
                for handler in handlers {
                    handler.join().unwrap();
                }
            });
            Self {
                address,
                state,
                stop,
                worker: Some(worker),
            }
        }

        fn respond(
            mut stream: std::net::TcpStream,
            shared: &Arc<(Mutex<GateState>, std::sync::Condvar)>,
        ) -> io::Result<()> {
            // Accepted sockets can inherit the listener's nonblocking mode.
            stream.set_nonblocking(false)?;
            let mut reader =
                bounded_fixture_io::Reader::new(&stream, Instant::now() + Duration::from_secs(10));
            let mut bytes = Vec::new();
            let header_end = loop {
                if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                    break end + 4;
                }
                if bytes.len() > 64 * 1024 {
                    return Err(io::Error::other("fixture request headers too large"));
                }
                let mut buffer = [0_u8; 4096];
                let count = reader.read(&mut buffer)?;
                if count == 0 {
                    return Err(io::Error::other("fixture request closed before headers"));
                }
                bytes.extend_from_slice(&buffer[..count]);
            };
            let head = String::from_utf8_lossy(&bytes[..header_end]);
            let get = head.starts_with("GET ");
            let length = head
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if length > 1024 * 1024 {
                return Err(io::Error::other("fixture request body too large"));
            }
            while bytes.len() < header_end + length {
                let mut buffer = [0_u8; 4096];
                let count = reader.read(&mut buffer)?;
                if count == 0 {
                    return Err(io::Error::other("fixture request body incomplete"));
                }
                bytes.extend_from_slice(&buffer[..count]);
            }
            let (mime, body) = if get {
                let (lock, changed) = &**shared;
                let mut state = lock.lock().unwrap();
                state.urls_started += 1;
                changed.notify_all();
                drop(state);
                ("text/html", b"<article><h1>URLDOC</h1><p>URLDOC complete readable text for mixed class admission.</p></article>".to_vec())
            } else {
                let request: Value =
                    serde_json::from_slice(&bytes[header_end..header_end + length])
                        .map_err(io::Error::other)?;
                let source = request["messages"]
                    .as_array()
                    .ok_or_else(|| io::Error::other("missing messages"))?
                    .iter()
                    .filter(|message| message["role"] == "user")
                    .filter_map(|message| message["content"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                if source.contains("FILEPREFIX") {
                    let (lock, changed) = &**shared;
                    let mut state = lock.lock().unwrap();
                    state.files_started += 1;
                    changed.notify_all();
                    let (mut state, _) = changed
                        .wait_timeout_while(state, Duration::from_secs(15), |state| !state.released)
                        .unwrap();
                    if !state.released {
                        state.errors.push("file model gate timed out".into());
                        state.released = true;
                        changed.notify_all();
                    }
                }
                let answer = json!({"cleaned_markdown":source,"frontmatter":{"description":"Authored gate fixture","tags":["fixture"]}});
                ("application/json", serde_json::to_vec(&json!({"choices":[{"message":{"content":answer.to_string()},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}})).unwrap())
            };
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )?;
            stream.write_all(&body)?;
            Ok(())
        }

        fn observe_then_release(&self) -> std::thread::JoinHandle<()> {
            let shared = Arc::clone(&self.state);
            std::thread::spawn(move || {
                let (lock, changed) = &*shared;
                let state = lock.lock().unwrap();
                let (mut state, _) = changed
                    .wait_timeout_while(state, Duration::from_secs(10), |state| {
                        state.errors.is_empty()
                            && (state.files_started < 1 || state.urls_started < 6)
                    })
                    .unwrap();
                state.mixed_observed = !state.released
                    && state.errors.is_empty()
                    && state.files_started == 1
                    && state.urls_started == 6;
                state.released = true;
                changed.notify_all();
            })
        }

        fn finish(&mut self) {
            let (lock, changed) = &*self.state;
            lock.lock().unwrap().released = true;
            changed.notify_all();
            self.stop.store(true, Ordering::SeqCst);
            if let Some(worker) = self.worker.take() {
                worker.join().unwrap();
            }
        }
    }
    impl Drop for MixedGate {
        fn drop(&mut self) {
            self.finish();
        }
    }

    #[test]
    #[ignore = "subprocess fixture with a private configuration/state root"]
    fn namespace_coordinator_helper() {
        let Some(root) = std::env::var_os("MARKITAI_NAMESPACE_COORDINATOR_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        let case = std::env::var("MARKITAI_NAMESPACE_COORDINATOR_CASE").unwrap();
        match case.as_str() {
            "partial" => {
                let input = root.join("input");
                let output = root.join("out");
                let mut tasks = Vec::new();
                for name in ["bad", "good"] {
                    fs::create_dir_all(input.join(name)).unwrap();
                    let source = input.join(name).join("a.txt");
                    fs::write(&source, format!("Complete {name} content.\n")).unwrap();
                    tasks.push(task(&source, &format!("{name}/a.txt"), &output.join(name)));
                }
                let foreign = bad_records(&output.join("bad"));
                let cfg = cfg(&root, "rename");
                assert_eq!(
                    execute(
                        &cli(&input, &output),
                        &cfg,
                        tasks,
                        &output,
                        RunMode::Directory,
                        NamespaceBatch::commit
                    ),
                    10
                );
                let saved = checkpoint(&output);
                assert_eq!(saved["documents"]["bad/a.txt"]["status"], "failed");
                assert_eq!(saved["documents"]["good/a.txt"]["status"], "completed");
                assert!(
                    fs::read_to_string(output.join("good/a.txt.md"))
                        .unwrap()
                        .contains("Complete good content.")
                );
                assert!(!output.join("bad/a.txt.md").exists());
                assert_eq!(fs::read_to_string(foreign).unwrap(), "foreign records file");
            }
            "skip" => {
                let input = root.join("input");
                let output = root.join("out");
                fs::create_dir(&input).unwrap();
                let source = input.join("a.txt");
                fs::write(&source, "original input\n").unwrap();
                let foreign = bad_records(&output);
                let saved_output = output.join("a.txt.md");
                fs::write(&saved_output, "preserve existing output\n").unwrap();
                let cfg = cfg(&root, "skip");
                let mut namespace_calls = 0;
                assert_eq!(
                    execute(
                        &cli(&input, &output),
                        &cfg,
                        vec![task(&source, "a.txt", &output)],
                        &output,
                        RunMode::Directory,
                        |batch| {
                            namespace_calls += 1;
                            batch.commit()
                        }
                    ),
                    0
                );
                assert_eq!(namespace_calls, 0);
                assert_eq!(
                    fs::read_to_string(saved_output).unwrap(),
                    "preserve existing output\n"
                );
                assert_eq!(fs::read_to_string(foreign).unwrap(), "foreign records file");
                assert_eq!(
                    checkpoint(&output)["documents"]["a.txt"]["status"],
                    "completed"
                );
            }
            "mixed-backlog" => {
                let input = root.join("input");
                let output = root.join("out");
                fs::create_dir(&input).unwrap();
                let mut server = MixedGate::new();
                let mut tasks = Vec::new();
                for index in 0..20 {
                    let name = format!("a{index:02}.txt");
                    let source = input.join(&name);
                    fs::write(
                        &source,
                        format!("FILEPREFIX {index} complete authored input.\n"),
                    )
                    .unwrap();
                    tasks.push(task(&source, &name, &output));
                }
                let list = input.join("z.urls");
                let mut urls = String::new();
                for index in 0..6 {
                    let source = format!("{}/url{index}", server.address);
                    urls.push_str(&format!("{source} page{index}\n"));
                    tasks.push(Task {
                        display: source.clone(),
                        report_key: format!("{source} page{index}"),
                        source,
                        output: Some(output.clone()),
                        filename: Some(format!("page{index}.md")),
                        reserved_stem: None,
                        source_file: Some(list.to_string_lossy().into_owned()),
                    });
                }
                fs::write(&list, urls).unwrap();
                let mut cfg = cfg(&root, "rename");
                cfg["batch"]["url_concurrency"] = json!(2);
                cfg["llm"]["enabled"] = json!(true);
                cfg["llm"]["router_settings"] = json!({"num_retries":0,"timeout":20});
                cfg["llm"]["model_list"] = json!([{ "model_name":"default", "litellm_params":{
                    "model":"openai/namespace-fixture", "api_key":"authored-fixture", "api_base":format!("{}/v1", server.address) }}]);
                let monitor = server.observe_then_release();
                let code = execute(
                    &cli(&input, &output),
                    &cfg,
                    tasks,
                    &output,
                    RunMode::Directory,
                    NamespaceBatch::commit,
                );
                monitor.join().unwrap();
                server.finish();
                assert_eq!(code, 0);
                let (observed, errors, files_started, urls_started) = {
                    let state = server.state.0.lock().unwrap();
                    (
                        state.mixed_observed,
                        state.errors.clone(),
                        state.files_started,
                        state.urls_started,
                    )
                };
                assert!(
                    observed,
                    "all six URLs must start while the first file request remains held, including refills beyond the two URL slots; files={files_started}, URLs={urls_started}, fixture errors={errors:?}"
                );
                assert!(errors.is_empty(), "{errors:?}");
                assert_eq!(files_started, 20);
                assert_eq!(urls_started, 6);
                let saved = checkpoint(&output);
                assert_eq!(saved["documents"].as_object().unwrap().len(), 20);
                assert_eq!(saved["urls"].as_object().unwrap().len(), 6);
                for collection in ["documents", "urls"] {
                    for item in saved[collection].as_object().unwrap().values() {
                        assert_eq!(item["status"], "completed");
                        let body = fs::read_to_string(item["output"].as_str().unwrap()).unwrap();
                        assert!(body.contains(if collection == "documents" {
                            "FILEPREFIX"
                        } else {
                            "URLDOC"
                        }));
                    }
                }
            }
            "zero-requests" => {
                for fault in ["namespace", "journal"] {
                    let directory = root.join(fault);
                    fs::create_dir(&directory).unwrap();
                    let input = directory.join("sources.urls");
                    let output = directory.join("out");
                    let mut observer = Observer::new();
                    let tasks: Vec<_> = (0..17)
                        .map(|index| {
                            let source = format!("{}/page{index}", observer.address);
                            Task {
                                display: source.clone(),
                                report_key: format!("{source} page{index}"),
                                source,
                                output: Some(output.clone()),
                                filename: Some(format!("page{index}.md")),
                                reserved_stem: None,
                                source_file: Some(input.to_string_lossy().into_owned()),
                            }
                        })
                        .collect();
                    fs::write(
                        &input,
                        tasks
                            .iter()
                            .map(|task| format!("{}\n", task.report_key))
                            .collect::<String>(),
                    )
                    .unwrap();
                    let cfg = cfg(&directory, "rename");
                    let cli = cli(&input, &output);
                    let scope = Scope::new(Mode::UrlList, &input, &output).unwrap();
                    let hash = codec::task_hash(&scope, &options(&cli, &cfg, &scope)).unwrap();
                    let journal =
                        output.join(format!(".markitai/states/markitai.{hash}.state.jsonl"));
                    let mut commits = 0;
                    let code = execute(&cli, &cfg, tasks, &output, RunMode::UrlList, |batch| {
                        commits += 1;
                        if fault == "namespace" && commits == 2 {
                            return Err(ClaimError::Io(io::Error::other(
                                "injected namespace media-fence failure",
                            )));
                        }
                        let ready = batch.commit()?;
                        if fault == "journal" && commits == 2 {
                            fs::create_dir(&journal)?;
                        }
                        Ok(ready)
                    });
                    assert_eq!(code, 1);
                    assert_eq!(
                        commits, 2,
                        "seventeen configured slots span both bounded namespace windows"
                    );
                    assert_eq!(
                        observer.finish(),
                        0,
                        "no URL request can precede both namespace and journal fences"
                    );
                    assert!(!output.join("page0.md").exists());
                    assert!(!output.join("page16.md").exists());
                    let saved = checkpoint(&output);
                    assert_eq!(saved["urls"].as_object().unwrap().len(), 17);
                    for entry in saved["urls"].as_object().unwrap().values() {
                        assert_eq!(entry["status"], "pending");
                        assert!(entry.get("target").is_none_or(Value::is_null));
                        assert!(entry.get("diagnostics").is_none_or(Value::is_null));
                    }
                    let _released = MemberLeases::acquire(
                        &output,
                        &["page0.md".into(), "page0.llm.md".into()],
                        false,
                    )
                    .unwrap();
                    if fault == "journal" {
                        assert!(journal.is_dir());
                    }
                }
            }
            _ => panic!("unknown namespace coordinator case"),
        }
        fs::write(root.join("passed"), case).unwrap();
    }
}
