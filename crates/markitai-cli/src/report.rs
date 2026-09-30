use crate::{
    diagnostics::AttemptDiagnostics,
    pricing::{self, Pricing},
    report_store,
};
use markitai_core::ConversionUsage;
use serde::ser::{Serialize, SerializeMap, Serializer};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunMode {
    SingleFile,
    SingleUrl,
    Directory,
    UrlList,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ItemKind {
    File,
    Url,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ItemStatus {
    Completed,
    Skipped,
    Failed,
    Pending,
}

#[derive(Debug)]
pub(crate) struct RunItem {
    pub(crate) index: usize,
    pub(crate) kind: ItemKind,
    pub(crate) display: String,
    pub(crate) report_key: String,
    pub(crate) source_file: Option<String>,
    pub(crate) status: ItemStatus,
    pub(crate) output: Option<PathBuf>,
    // Private archive projection; never serialized into stdout or reports.
    pub(crate) history_output: Option<PathBuf>,
    pub(crate) history_eligible: bool,
    pub(crate) error: Option<String>,
    pub(crate) warnings: Vec<String>,
    pub(crate) skip_reason: Option<String>,
    pub(crate) started_at: String,
    pub(crate) completed_at: String,
    pub(crate) elapsed_s: f64,
    pub(crate) conversion_duration_s: Option<f64>,
    pub(crate) images: usize,
    pub(crate) screenshots: usize,
    pub(crate) usage: ConversionUsage,
    pub(crate) diagnostics: Option<AttemptDiagnostics>,
    pub(crate) llm_cache_hit: bool,
    pub(crate) fetch_cache_hit: bool,
    pub(crate) fetch_strategy: Option<String>,
}

#[derive(Debug)]
pub(crate) struct ReportOptions {
    llm: bool,
    ocr: bool,
    screenshot: bool,
    alt: bool,
    desc: bool,
    cache: bool,
    concurrency: u64,
    scan_max_depth: usize,
    globs: Vec<String>,
    models: Vec<String>,
}

impl ReportOptions {
    pub(crate) fn from_config(cfg: &Value, max_depth: Option<usize>, globs: &[String]) -> Self {
        let flag = |path| cfg.pointer(path).and_then(Value::as_bool).unwrap_or(false);
        Self {
            llm: flag("/llm/enabled"),
            ocr: flag("/ocr/enabled"),
            screenshot: flag("/screenshot/enabled"),
            alt: flag("/image/alt_enabled"),
            desc: flag("/image/desc_enabled"),
            cache: cfg
                .pointer("/cache/enabled")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            concurrency: cfg
                .pointer("/batch/concurrency")
                .and_then(Value::as_u64)
                .unwrap_or(10),
            scan_max_depth: max_depth.unwrap_or_else(|| {
                cfg.pointer("/batch/scan_max_depth")
                    .and_then(Value::as_u64)
                    .unwrap_or(5) as usize
            }),
            globs: globs
                .iter()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect(),
            models: cfg
                .pointer("/llm/model_list")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|model| {
                    model
                        .pointer("/litellm_params/model")
                        .and_then(Value::as_str)
                })
                .map(String::from)
                .collect(),
        }
    }

    fn hash_options(&self, mode: RunMode) -> Value {
        match mode {
            RunMode::SingleUrl => json!({"llm": self.llm}),
            RunMode::UrlList => json!({"llm": self.llm, "alt": self.alt, "desc": self.desc}),
            RunMode::SingleFile | RunMode::Directory => {
                let mut options = json!({"llm": self.llm, "ocr": self.ocr,
                    "screenshot": self.screenshot, "alt": self.alt, "desc": self.desc});
                if mode == RunMode::Directory {
                    options["scan_max_depth"] = json!(self.scan_max_depth);
                    if !self.globs.is_empty() {
                        options["glob_patterns"] = json!(self.globs);
                    }
                }
                options
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct RunInfo {
    pub(crate) mode: RunMode,
    pub(crate) input: String,
    pub(crate) output_dir: PathBuf,
    pub(crate) started_at: String,
    pub(crate) log_file: Option<PathBuf>,
    pub(crate) options: ReportOptions,
}

#[derive(Debug)]
pub(crate) struct RunFinished {
    pub(crate) updated_at: String,
    pub(crate) duration_s: f64,
}

#[derive(Debug)]
pub(crate) struct ReportPlan {
    run: RunInfo,
    task_hash: String,
    on_conflict: String,
    allow_symlinks: bool,
    input_dir: Option<PathBuf>,
    output_dir: PathBuf,
}

pub(crate) fn plan(
    run: RunInfo,
    selection: Option<bool>,
    on_conflict: &str,
    allow_symlinks: bool,
) -> Result<Option<ReportPlan>, String> {
    let batch = matches!(run.mode, RunMode::Directory | RunMode::UrlList);
    if !selection.unwrap_or(batch) {
        return Ok(None);
    }
    if !matches!(on_conflict, "rename" | "overwrite" | "skip") {
        return Err("Invalid report conflict policy".into());
    }
    let output_dir = report_store::resolve_path(&run.output_dir).map_err(|e| e.to_string())?;
    let input_dir = if run.mode == RunMode::Directory {
        Some(report_store::resolve_path(Path::new(&run.input)).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let input = match run.mode {
        RunMode::SingleUrl | RunMode::UrlList => output_dir.as_path(),
        _ => Path::new(&run.input),
    };
    let task_hash =
        report_store::task_hash(input, &output_dir, &run.options.hash_options(run.mode))
            .map_err(|e| e.to_string())?;
    Ok(Some(ReportPlan {
        run,
        task_hash,
        on_conflict: on_conflict.into(),
        allow_symlinks,
        input_dir,
        output_dir,
    }))
}

pub(crate) fn publish(
    plan: &ReportPlan,
    bytes: &[u8],
) -> Result<report_store::Publication, String> {
    // Retain the caller's path spelling so the store can inspect symlink components.
    report_store::publish(
        &plan.run.output_dir,
        &plan.task_hash,
        &plan.on_conflict,
        plan.allow_symlinks,
        plan.run.mode == RunMode::Directory,
        bytes,
    )
    .map_err(|e| e.to_string())
}

// Order is local to reports; enabling serde_json's preserve_order globally would
// also change the established CLI and bindings JSON representation.
enum Ordered {
    Value(Value),
    Object(Vec<(String, Ordered)>),
}

impl Serialize for Ordered {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Value(value) => value.serialize(serializer),
            Self::Object(fields) => {
                let mut map = serializer.serialize_map(Some(fields.len()))?;
                for (key, value) in fields {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

fn value(value: impl Into<Value>) -> Ordered {
    Ordered::Value(value.into())
}

fn object(fields: impl IntoIterator<Item = (&'static str, Ordered)>) -> Ordered {
    Ordered::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
}

fn path_value(path: Option<&Path>) -> Ordered {
    value(
        path.map(|p| Value::String(p.to_string_lossy().into_owned()))
            .unwrap_or(Value::Null),
    )
}

fn option_string(text: Option<&str>) -> Ordered {
    value(text.map(Value::from).unwrap_or(Value::Null))
}

fn duration(seconds: f64) -> Ordered {
    if seconds < 60.0 {
        value(format!("{seconds:.1}s"))
    } else {
        let seconds = seconds as u64;
        if seconds >= 3600 {
            value(format!(
                "{:02}:{:02}:{:02}",
                seconds / 3600,
                seconds / 60 % 60,
                seconds % 60
            ))
        } else {
            value(format!("{:02}:{:02}", seconds / 60, seconds % 60))
        }
    }
}

fn ordered_model(record: &Value) -> Ordered {
    let Some(record) = record.as_object() else {
        return value(record.clone());
    };
    let preferred = ["requests", "input_tokens", "output_tokens", "cost_usd"];
    let mut fields = Vec::with_capacity(record.len());
    for key in preferred {
        if let Some(v) = record.get(key) {
            fields.push((key.to_owned(), value(v.clone())));
        }
    }
    for (key, v) in record {
        if !preferred.contains(&key.as_str()) {
            fields.push((key.clone(), value(v.clone())));
        }
    }
    Ordered::Object(fields)
}

fn models(records: &Map<String, Value>) -> Ordered {
    Ordered::Object(
        records
            .iter()
            .map(|(name, record)| (name.clone(), ordered_model(record)))
            .collect(),
    )
}

fn counter(record: &Value, key: &str) -> u64 {
    record.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn totals(records: &Map<String, Value>) -> Result<(u64, u64, u64), String> {
    let mut result = (0u64, 0u64, 0u64);
    for record in records.values() {
        for (target, key) in [
            (&mut result.0, "requests"),
            (&mut result.1, "input_tokens"),
            (&mut result.2, "output_tokens"),
        ] {
            *target = target
                .checked_add(counter(record, key))
                .ok_or("Report usage counter overflow")?;
        }
    }
    Ok(result)
}

fn usage_block(records: &Map<String, Value>, cost: f64) -> Result<Ordered, String> {
    let (requests, input, output) = totals(records)?;
    let mut fields = vec![
        ("models", models(records)),
        ("requests", value(requests)),
        ("input_tokens", value(input)),
        ("output_tokens", value(output)),
        ("cost_usd", value(cost)),
    ];
    if let Some(pricing) = Pricing::from_models(records) {
        fields.push(("pricing", value(json!(pricing))));
    }
    Ok(object(fields))
}

fn aggregate(items: &[&RunItem], mode: RunMode) -> Result<(Map<String, Value>, f64), String> {
    let mut combined = Map::<String, Value>::new();
    let mut cost = 0.0;
    for item in items {
        if mode == RunMode::UrlList && item.status != ItemStatus::Completed {
            continue;
        }
        cost += item.usage.cost_usd;
        pricing::merge_models(&mut combined, &item.usage.by_model)?;
    }
    if !cost.is_finite() {
        return Err("Report usage total is not finite".into());
    }
    Ok((combined, cost))
}

fn completed(item: &RunItem, mode: RunMode) -> bool {
    item.status == ItemStatus::Completed
        || (mode == RunMode::Directory && item.status == ItemStatus::Skipped)
}

fn status(item: &RunItem, mode: RunMode) -> &'static str {
    if completed(item, mode) {
        "completed"
    } else if item.status == ItemStatus::Pending {
        "pending"
    } else if item.status == ItemStatus::Skipped {
        "skipped"
    } else {
        "failed"
    }
}

fn directory_entry(item: &RunItem) -> Ordered {
    let mut fields = vec![
        ("status", value(status(item, RunMode::Directory))),
        ("cache_hit", value(item.llm_cache_hit)),
        ("output", path_value(item.output.as_deref())),
        ("error", option_string(item.error.as_deref())),
    ];
    if item.kind == ItemKind::Url {
        fields.push((
            "fetch_strategy",
            option_string(item.fetch_strategy.as_deref()),
        ));
    }
    fields.extend([
        ("started_at", value(item.started_at.clone())),
        ("completed_at", value(item.completed_at.clone())),
        ("duration", duration(item.elapsed_s)),
        ("images", value(item.images)),
        ("screenshots", value(item.screenshots)),
        ("cost_usd", value(item.usage.cost_usd)),
        ("llm_usage", models(&item.usage.by_model)),
    ]);
    if let Some(pricing) = Pricing::from_usage(&item.usage) {
        fields.push(("pricing", value(json!(pricing))));
    }
    object(fields)
}

fn single_file_entry(item: &RunItem) -> Result<Ordered, String> {
    let (_, input, output) = totals(&item.usage.by_model)?;
    let mut fields = vec![
        ("status", value(status(item, RunMode::SingleFile))),
        ("output", path_value(item.output.as_deref())),
        ("error", option_string(item.error.as_deref())),
        ("duration", duration(item.elapsed_s)),
        ("images", value(item.images)),
        ("screenshots", value(item.screenshots)),
        (
            "llm_usage",
            object([
                ("cost_usd", value(item.usage.cost_usd)),
                ("input_tokens", value(input)),
                ("output_tokens", value(output)),
            ]),
        ),
    ];
    if let Some(pricing) = Pricing::from_usage(&item.usage) {
        fields.push(("pricing", value(json!(pricing))));
    }
    Ok(object(fields))
}

fn single_url_entry(item: &RunItem) -> Ordered {
    let mut fields = vec![
        ("status", value(status(item, RunMode::SingleUrl))),
        (
            "cache_hit",
            value(item.fetch_cache_hit || item.llm_cache_hit),
        ),
        (
            "cache_details",
            object([
                ("fetch", value(item.fetch_cache_hit)),
                ("llm", value(item.llm_cache_hit)),
            ]),
        ),
        ("output", path_value(item.output.as_deref())),
        ("error", option_string(item.error.as_deref())),
        (
            "fetch_strategy",
            option_string(item.fetch_strategy.as_deref()),
        ),
        ("duration", duration(item.elapsed_s)),
        ("images", value(item.images)),
        ("screenshots", value(item.screenshots)),
        ("llm_usage", models(&item.usage.by_model)),
    ];
    if let Some(pricing) = Pricing::from_usage(&item.usage) {
        fields.push(("pricing", value(json!(pricing))));
    }
    object(fields)
}

fn list_entry(item: &RunItem) -> Ordered {
    if item.status == ItemStatus::Pending {
        return object([
            ("status", value("pending")),
            ("output", path_value(item.output.as_deref())),
            ("error", option_string(item.error.as_deref())),
            ("skip_reason", value("pending_batch")),
        ]);
    }
    if item.status != ItemStatus::Completed {
        return object([
            ("status", value(status(item, RunMode::UrlList))),
            (
                "error",
                if item.status == ItemStatus::Skipped {
                    value("Output exists")
                } else {
                    option_string(item.error.as_deref())
                },
            ),
        ]);
    }
    object([
        ("status", value("completed")),
        ("output", path_value(item.output.as_deref())),
        ("error", option_string(item.error.as_deref())),
        (
            "fetch_strategy",
            option_string(item.fetch_strategy.as_deref()),
        ),
        ("images", value(item.images)),
        ("screenshots", value(item.screenshots)),
    ])
}

fn url_groups(items: &[&RunItem], mode: RunMode) -> Ordered {
    let mut groups = BTreeMap::<&str, Vec<&RunItem>>::new();
    for item in items.iter().filter(|item| item.kind == ItemKind::Url) {
        let source = match mode {
            RunMode::SingleUrl => "cli",
            RunMode::UrlList => "unknown.urls",
            _ => item.source_file.as_deref().unwrap_or("unknown.urls"),
        };
        groups.entry(source).or_default().push(item);
    }
    Ordered::Object(
        groups
            .into_iter()
            .map(|(source, group)| {
                let entry = object([
                    ("total", value(group.len())),
                    (
                        "completed",
                        value(group.iter().filter(|item| completed(item, mode)).count()),
                    ),
                    (
                        "failed",
                        value(
                            group
                                .iter()
                                .filter(|item| item.status == ItemStatus::Failed)
                                .count(),
                        ),
                    ),
                    (
                        "urls",
                        Ordered::Object(
                            group
                                .iter()
                                .map(|item| {
                                    let data = match mode {
                                        RunMode::SingleUrl => single_url_entry(item),
                                        RunMode::UrlList => list_entry(item),
                                        _ => directory_entry(item),
                                    };
                                    (item.report_key.clone(), data)
                                })
                                .collect(),
                        ),
                    ),
                ]);
                (source.into(), entry)
            })
            .collect(),
    )
}

fn options(plan: &ReportPlan, items: &[&RunItem]) -> Option<Ordered> {
    let opts = &plan.run.options;
    match plan.run.mode {
        RunMode::SingleUrl => Some(object([
            ("llm", value(opts.llm)),
            ("cache", value(opts.cache)),
            ("alt", value(opts.alt)),
            ("desc", value(opts.desc)),
            (
                "fetch_strategy",
                option_string(items[0].fetch_strategy.as_deref()),
            ),
        ])),
        RunMode::Directory => {
            let mut fields = vec![
                ("concurrency", value(opts.concurrency)),
                ("llm", value(opts.llm)),
                ("ocr", value(opts.ocr)),
                ("screenshot", value(opts.screenshot)),
                ("alt", value(opts.alt)),
                ("desc", value(opts.desc)),
            ];
            if opts.llm && !opts.models.is_empty() {
                fields.push(("models", value(json!(opts.models))));
            }
            fields.extend([
                ("input_dir", path_value(plan.input_dir.as_deref())),
                ("output_dir", path_value(Some(&plan.output_dir))),
                ("scan_max_depth", value(opts.scan_max_depth)),
            ]);
            if !opts.globs.is_empty() {
                fields.push(("glob_patterns", value(json!(opts.globs))));
            }
            Some(object(fields))
        }
        _ => None,
    }
}

fn summary(mode: RunMode, items: &[&RunItem], finished: &RunFinished) -> Ordered {
    let count = |kind, predicate: fn(&RunItem) -> bool| {
        items
            .iter()
            .filter(|item| item.kind == kind && predicate(item))
            .count()
    };
    let files = count(ItemKind::File, |_| true);
    let file_failed = count(ItemKind::File, |item| item.status == ItemStatus::Failed);
    let file_pending = count(ItemKind::File, |item| item.status == ItemStatus::Pending);
    let url_pending = count(ItemKind::Url, |item| item.status == ItemStatus::Pending);
    let urls = count(ItemKind::Url, |_| true);
    let url_failed = count(ItemKind::Url, |item| item.status == ItemStatus::Failed);
    let mut fields = vec![
        ("total_documents", value(files)),
        (
            "completed_documents",
            value(files - file_failed - file_pending),
        ),
        ("failed_documents", value(file_failed)),
    ];
    if mode == RunMode::Directory {
        fields.push(("pending_documents", value(file_failed + file_pending)));
    } else if file_pending > 0 {
        fields.push(("pending_documents", value(file_pending)));
    }
    if mode != RunMode::SingleFile {
        fields.extend([
            ("total_urls", value(urls)),
            (
                "completed_urls",
                value(
                    items
                        .iter()
                        .filter(|item| item.kind == ItemKind::Url && completed(item, mode))
                        .count(),
                ),
            ),
            ("failed_urls", value(url_failed)),
        ]);
    }
    if mode == RunMode::Directory {
        fields.extend([
            ("pending_urls", value(url_failed + url_pending)),
            (
                "url_cache_hits",
                value(
                    items
                        .iter()
                        .filter(|item| {
                            item.kind == ItemKind::Url
                                && completed(item, mode)
                                && item.llm_cache_hit
                        })
                        .count(),
                ),
            ),
            (
                "url_sources",
                value(
                    items
                        .iter()
                        .filter(|item| item.kind == ItemKind::Url)
                        .map(|item| item.source_file.as_deref().unwrap_or("unknown.urls"))
                        .collect::<BTreeSet<_>>()
                        .len(),
                ),
            ),
        ]);
    }
    if mode != RunMode::Directory && url_pending > 0 {
        fields.push(("pending_urls", value(url_pending)));
    }
    let elapsed = if matches!(mode, RunMode::SingleFile | RunMode::SingleUrl) {
        items[0].elapsed_s
    } else {
        finished.duration_s
    };
    fields.push(("duration", duration(elapsed)));
    if mode == RunMode::Directory {
        fields.push((
            "processing_time",
            duration(items.iter().map(|item| item.elapsed_s).sum()),
        ));
    }
    object(fields)
}

// These observations describe each latest attempt. They never enter the
// reference-shaped usage totals or acquire authority over an output path.
fn terminal_diagnostics<'a>(
    entries: impl IntoIterator<Item = (ItemKind, &'a str, &'a AttemptDiagnostics)>,
) -> Result<Option<Ordered>, String> {
    let mut documents = Vec::new();
    let mut urls = Vec::new();
    for (kind, key, diagnostics) in entries {
        diagnostics.validate()?;
        let target = if kind == ItemKind::File {
            &mut documents
        } else {
            &mut urls
        };
        target.push((
            key.to_owned(),
            value(serde_json::to_value(diagnostics).map_err(|e| e.to_string())?),
        ));
    }
    if documents.is_empty() && urls.is_empty() {
        return Ok(None);
    }
    documents.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(Some(object([
        ("documents", Ordered::Object(documents)),
        ("urls", Ordered::Object(urls)),
    ])))
}

pub(crate) fn render(
    plan: &ReportPlan,
    items: &[RunItem],
    finished: &RunFinished,
) -> Result<Vec<u8>, String> {
    if items.is_empty() {
        return Err("Cannot render a report without processed items".into());
    }
    let mode = plan.run.mode;
    if matches!(mode, RunMode::SingleFile | RunMode::SingleUrl)
        && (items.len() != 1
            || !matches!(items[0].status, ItemStatus::Completed | ItemStatus::Pending))
    {
        return Err("A single-item report requires one completed or provider-pending item".into());
    }
    let mut keys = BTreeSet::new();
    for item in items {
        if (mode == RunMode::SingleFile && item.kind != ItemKind::File)
            || (matches!(mode, RunMode::SingleUrl | RunMode::UrlList) && item.kind != ItemKind::Url)
        {
            return Err("Report item kind does not match run mode".into());
        }
        if !keys.insert((item.kind == ItemKind::Url, &item.report_key)) {
            return Err("Duplicate report item identity".into());
        }
        if !item.elapsed_s.is_finite() || item.elapsed_s < 0.0 || !item.usage.cost_usd.is_finite() {
            return Err("Report item contains an invalid numeric measurement".into());
        }
    }
    if !finished.duration_s.is_finite()
        || finished.duration_s < 0.0
        || !items
            .iter()
            .map(|item| item.elapsed_s)
            .sum::<f64>()
            .is_finite()
    {
        return Err("Report run duration is invalid".into());
    }
    let mut items: Vec<_> = items.iter().collect();
    items.sort_by_key(|item| item.index);
    let mut fields = vec![
        ("version", value("1.0")),
        ("generated_at", value(finished.updated_at.clone())),
    ];
    if mode == RunMode::Directory {
        fields.extend([
            ("started_at", value(plan.run.started_at.clone())),
            ("updated_at", value(finished.updated_at.clone())),
        ]);
    }
    fields.push(("log_file", path_value(plan.run.log_file.as_deref())));
    if let Some(options) = options(plan, &items) {
        fields.push(("options", options));
    }
    fields.push(("summary", summary(mode, &items, finished)));
    let usage = if matches!(mode, RunMode::SingleFile | RunMode::SingleUrl) {
        usage_block(&items[0].usage.by_model, items[0].usage.cost_usd)?
    } else {
        let (records, cost) = aggregate(&items, mode)?;
        usage_block(&records, cost)?
    };
    fields.push(("llm_usage", usage));
    if matches!(mode, RunMode::SingleFile | RunMode::Directory) {
        let mut documents = Vec::new();
        for item in items.iter().filter(|item| item.kind == ItemKind::File) {
            documents.push((
                item.report_key.clone(),
                if mode == RunMode::Directory {
                    directory_entry(item)
                } else {
                    single_file_entry(item)?
                },
            ));
        }
        documents.sort_by(|a, b| a.0.cmp(&b.0));
        fields.push(("documents", Ordered::Object(documents)));
    }
    if mode != RunMode::SingleFile {
        fields.push(("url_sources", url_groups(&items, mode)));
    }
    if let Some(diagnostics) = terminal_diagnostics(items.iter().filter_map(|item| {
        item.diagnostics
            .as_ref()
            .map(|diagnostics| (item.kind, item.report_key.as_str(), diagnostics))
    }))? {
        fields.push(("terminal_diagnostics", diagnostics));
    }
    serde_json::to_vec_pretty(&object(fields)).map_err(|e| e.to_string())
}

// Recovered state is a report projection, not another completed conversion.
// Keeping the variants separate prevents absent measurements from becoming a
// fabricated RunItem with this run's timestamps or elapsed time.
enum ResumedEntry<'a> {
    Observed(&'a RunItem),
    Recovered(&'a crate::run_state::Entry),
}

struct ResumedView<'a> {
    key: &'a str,
    kind: ItemKind,
    entry: ResumedEntry<'a>,
}

impl ResumedView<'_> {
    fn status(&self, mode: RunMode) -> &'static str {
        match self.entry {
            ResumedEntry::Observed(item) => status(item, mode),
            ResumedEntry::Recovered(entry) => match entry.status {
                crate::run_state::Status::Pending => "pending",
                crate::run_state::Status::InProgress => "in_progress",
                crate::run_state::Status::Completed => "completed",
                crate::run_state::Status::Failed => "failed",
            },
        }
    }

    fn source(&self, mode: RunMode) -> &str {
        if mode == RunMode::UrlList {
            return "unknown.urls";
        }
        match self.entry {
            ResumedEntry::Observed(item) => item.source_file.as_deref(),
            ResumedEntry::Recovered(entry) => entry.source_file.as_deref(),
        }
        .unwrap_or("unknown.urls")
    }

    fn elapsed(&self) -> Result<Option<f64>, String> {
        match self.entry {
            ResumedEntry::Observed(item) => Ok(Some(item.elapsed_s)),
            ResumedEntry::Recovered(entry) => match entry.observations.get("duration") {
                None | Some(Value::Null) => Ok(None),
                Some(measurement) => {
                    let seconds = measurement
                        .as_f64()
                        .filter(|number| number.is_finite() && *number >= 0.0)
                        .ok_or("Recovered report duration is invalid")?;
                    Ok(Some(seconds))
                }
            },
        }
    }

    fn cache_hit(&self) -> bool {
        match self.entry {
            ResumedEntry::Observed(item) => item.llm_cache_hit,
            ResumedEntry::Recovered(entry) => entry
                .observations
                .get("cache_hit")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }
    }
}

fn resumed_views<'a>(
    mode: RunMode,
    items: &'a [RunItem],
    state: &'a crate::run_state::Snapshot,
    active_url_keys: &'a [String],
) -> Result<Vec<ResumedView<'a>>, String> {
    let mut observed = BTreeMap::new();
    for item in items {
        if mode == RunMode::UrlList && item.kind != ItemKind::Url {
            return Err("Report item kind does not match run mode".into());
        }
        if observed
            .insert((item.kind == ItemKind::Url, item.report_key.as_str()), item)
            .is_some()
        {
            return Err("Duplicate report item identity".into());
        }
        if !item.elapsed_s.is_finite() || item.elapsed_s < 0.0 || !item.usage.cost_usd.is_finite() {
            return Err("Report item contains an invalid numeric measurement".into());
        }
    }
    let mut views = Vec::new();
    if mode == RunMode::UrlList {
        let mut seen = BTreeSet::new();
        for key in active_url_keys {
            if !seen.insert(key.as_str()) {
                return Err("Duplicate active URL report identity".into());
            }
            let entry = if let Some(item) = observed.remove(&(true, key.as_str())) {
                ResumedEntry::Observed(item)
            } else {
                ResumedEntry::Recovered(
                    state
                        .urls
                        .get(key)
                        .ok_or("Active URL is absent from recovery state and observations")?,
                )
            };
            views.push(ResumedView {
                key,
                kind: ItemKind::Url,
                entry,
            });
        }
    } else {
        for (key, entry) in &state.documents {
            let entry = observed
                .remove(&(false, key.as_str()))
                .map(ResumedEntry::Observed)
                .unwrap_or(ResumedEntry::Recovered(entry));
            views.push(ResumedView {
                key,
                kind: ItemKind::File,
                entry,
            });
        }
        for (key, entry) in &state.urls {
            let entry = observed
                .remove(&(true, key.as_str()))
                .map(ResumedEntry::Observed)
                .unwrap_or(ResumedEntry::Recovered(entry));
            views.push(ResumedView {
                key,
                kind: ItemKind::Url,
                entry,
            });
        }
        let mut newly_observed: Vec<_> = observed.into_values().collect();
        newly_observed.sort_by_key(|item| item.index);
        views.extend(newly_observed.into_iter().map(|item| ResumedView {
            key: &item.report_key,
            kind: item.kind,
            entry: ResumedEntry::Observed(item),
        }));
    }
    Ok(views)
}

fn saved_value(entry: &crate::run_state::Entry, key: &str, default: Value) -> Ordered {
    value(entry.observations.get(key).cloned().unwrap_or(default))
}

type SavedUsage<'a> = (Option<&'a Map<String, Value>>, f64);

fn saved_usage(entry: &crate::run_state::Entry) -> Result<SavedUsage<'_>, String> {
    let records = entry
        .observations
        .get("llm_usage")
        .map(|value| {
            value
                .as_object()
                .ok_or("Recovered model usage must be an object")
        })
        .transpose()?;
    for usage in records.into_iter().flat_map(Map::values) {
        let usage = usage
            .as_object()
            .ok_or("Recovered model measurement must be an object")?;
        for key in [
            "requests",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
        ] {
            if usage.get(key).is_some_and(|value| value.as_u64().is_none()) {
                return Err("Recovered usage counter is invalid".into());
            }
        }
        if usage
            .get("cost_usd")
            .is_some_and(|value| value.as_f64().is_none_or(|number| !number.is_finite()))
        {
            return Err("Recovered model cost is invalid".into());
        }
    }
    let cost = entry
        .observations
        .get("cost_usd")
        .map(|value| {
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or("Recovered report cost is invalid")
        })
        .transpose()?
        .unwrap_or(0.0);
    Ok((records, cost))
}

fn resumed_entry(view: &ResumedView<'_>, mode: RunMode) -> Result<Ordered, String> {
    let entry = match view.entry {
        ResumedEntry::Observed(item) => {
            return Ok(if mode == RunMode::UrlList {
                list_entry(item)
            } else {
                directory_entry(item)
            });
        }
        ResumedEntry::Recovered(entry) => entry,
    };
    if mode == RunMode::UrlList {
        if view.status(mode) != "completed" {
            return Ok(object([
                ("status", value(view.status(mode))),
                ("error", option_string(entry.error.as_deref())),
            ]));
        }
        return Ok(object([
            ("status", value("completed")),
            ("output", path_value(entry.output.as_deref())),
            ("error", value(Value::Null)),
            (
                "fetch_strategy",
                saved_value(entry, "fetch_strategy", Value::Null),
            ),
            ("images", saved_value(entry, "images", json!(0))),
            ("screenshots", saved_value(entry, "screenshots", json!(0))),
        ]));
    }
    let mut fields = vec![
        ("status", value(view.status(mode))),
        ("cache_hit", saved_value(entry, "cache_hit", json!(false))),
        ("output", path_value(entry.output.as_deref())),
        ("error", option_string(entry.error.as_deref())),
    ];
    if view.kind == ItemKind::Url {
        fields.push((
            "fetch_strategy",
            saved_value(entry, "fetch_strategy", Value::Null),
        ));
    }
    let (usage, cost) = saved_usage(entry)?;
    fields.extend([
        ("started_at", saved_value(entry, "started_at", Value::Null)),
        (
            "completed_at",
            saved_value(entry, "completed_at", Value::Null),
        ),
        (
            "duration",
            view.elapsed()?
                .map(duration)
                .unwrap_or_else(|| value(Value::Null)),
        ),
        ("images", saved_value(entry, "images", json!(0))),
        ("screenshots", saved_value(entry, "screenshots", json!(0))),
        ("cost_usd", value(cost)),
        ("llm_usage", usage.map(models).unwrap_or_else(|| object([]))),
    ]);
    if let Some(pricing) = usage.and_then(Pricing::from_models) {
        fields.push(("pricing", value(json!(pricing))));
    }
    Ok(object(fields))
}

fn resumed_usage(views: &[ResumedView<'_>], mode: RunMode) -> Result<Ordered, String> {
    let observed: Vec<_> = views
        .iter()
        .filter_map(|view| match view.entry {
            ResumedEntry::Observed(item) => Some(item),
            ResumedEntry::Recovered(_) => None,
        })
        .collect();
    let (mut combined, mut cost) = aggregate(&observed, mode)?;
    for view in views {
        if mode == RunMode::UrlList && view.status(mode) != "completed" {
            continue;
        }
        let ResumedEntry::Recovered(entry) = view.entry else {
            continue;
        };
        let (records, saved_cost) = saved_usage(entry)?;
        cost += saved_cost;
        if let Some(records) = records {
            pricing::merge_models(&mut combined, records)?;
        }
    }
    if !cost.is_finite() {
        return Err("Report usage total is not finite".into());
    }
    usage_block(&combined, cost)
}

fn resumed_summary(
    mode: RunMode,
    views: &[ResumedView<'_>],
    finished: &RunFinished,
) -> Result<Ordered, String> {
    let count = |kind, wanted: Option<&str>| {
        views
            .iter()
            .filter(|view| {
                view.kind == kind && wanted.is_none_or(|wanted| view.status(mode) == wanted)
            })
            .count()
    };
    let mut fields = vec![
        ("total_documents", value(count(ItemKind::File, None))),
        (
            "completed_documents",
            value(count(ItemKind::File, Some("completed"))),
        ),
        (
            "failed_documents",
            value(count(ItemKind::File, Some("failed"))),
        ),
    ];
    if mode == RunMode::Directory {
        fields.push((
            "pending_documents",
            value(count(ItemKind::File, Some("pending")) + count(ItemKind::File, Some("failed"))),
        ));
    }
    fields.extend([
        ("total_urls", value(count(ItemKind::Url, None))),
        (
            "completed_urls",
            value(count(ItemKind::Url, Some("completed"))),
        ),
        ("failed_urls", value(count(ItemKind::Url, Some("failed")))),
    ]);
    if mode == RunMode::Directory {
        fields.extend([
            (
                "pending_urls",
                value(count(ItemKind::Url, Some("pending")) + count(ItemKind::Url, Some("failed"))),
            ),
            (
                "url_cache_hits",
                value(
                    views
                        .iter()
                        .filter(|view| {
                            view.kind == ItemKind::Url
                                && view.status(mode) == "completed"
                                && view.cache_hit()
                        })
                        .count(),
                ),
            ),
            (
                "url_sources",
                value(
                    views
                        .iter()
                        .filter(|view| view.kind == ItemKind::Url)
                        .map(|view| view.source(mode))
                        .collect::<BTreeSet<_>>()
                        .len(),
                ),
            ),
        ]);
    }
    if mode == RunMode::UrlList {
        let pending = count(ItemKind::Url, Some("pending"));
        if pending > 0 {
            fields.push(("pending_urls", value(pending)));
        }
    }
    fields.push(("duration", duration(finished.duration_s)));
    if mode == RunMode::Directory {
        let elapsed = views.iter().try_fold(0.0, |sum, view| {
            view.elapsed().map(|seconds| sum + seconds.unwrap_or(0.0))
        })?;
        if !elapsed.is_finite() {
            return Err("Report processing time is invalid".into());
        }
        fields.push(("processing_time", duration(elapsed)));
    }
    Ok(object(fields))
}

pub(crate) fn render_resumed(
    plan: &ReportPlan,
    items: &[RunItem],
    state: &crate::run_state::Snapshot,
    active_url_keys: &[String],
    finished: &RunFinished,
) -> Result<Vec<u8>, String> {
    let mode = plan.run.mode;
    if !matches!(mode, RunMode::Directory | RunMode::UrlList) {
        return Err("Recovered reports require a batch run".into());
    }
    if !finished.duration_s.is_finite() || finished.duration_s < 0.0 {
        return Err("Report run duration is invalid".into());
    }
    let views = resumed_views(mode, items, state, active_url_keys)?;
    let mut fields = vec![
        ("version", value("1.0")),
        ("generated_at", value(finished.updated_at.clone())),
    ];
    if mode == RunMode::Directory {
        fields.extend([
            ("started_at", value(plan.run.started_at.clone())),
            ("updated_at", value(finished.updated_at.clone())),
        ]);
    }
    fields.push(("log_file", path_value(plan.run.log_file.as_deref())));
    if let Some(options) = options(plan, &[]) {
        fields.push(("options", options));
    }
    fields.push(("summary", resumed_summary(mode, &views, finished)?));
    fields.push(("llm_usage", resumed_usage(&views, mode)?));
    if mode == RunMode::Directory {
        let mut documents = views
            .iter()
            .filter(|view| view.kind == ItemKind::File)
            .map(|view| Ok((view.key.to_owned(), resumed_entry(view, mode)?)))
            .collect::<Result<Vec<_>, String>>()?;
        documents.sort_by(|left, right| left.0.cmp(&right.0));
        fields.push(("documents", Ordered::Object(documents)));
    }
    let mut groups = BTreeMap::<&str, Vec<&ResumedView<'_>>>::new();
    for view in views.iter().filter(|view| view.kind == ItemKind::Url) {
        groups.entry(view.source(mode)).or_default().push(view);
    }
    let groups = groups
        .into_iter()
        .map(|(source, views)| {
            let urls = views
                .iter()
                .map(|view| Ok((view.key.to_owned(), resumed_entry(view, mode)?)))
                .collect::<Result<Vec<_>, String>>()?;
            Ok((
                source.to_owned(),
                object([
                    ("total", value(views.len())),
                    (
                        "completed",
                        value(
                            views
                                .iter()
                                .filter(|view| view.status(mode) == "completed")
                                .count(),
                        ),
                    ),
                    (
                        "failed",
                        value(
                            views
                                .iter()
                                .filter(|view| view.status(mode) == "failed")
                                .count(),
                        ),
                    ),
                    ("urls", Ordered::Object(urls)),
                ]),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    fields.push(("url_sources", Ordered::Object(groups)));
    let recovered = views
        .iter()
        .map(|view| match view.entry {
            ResumedEntry::Observed(_) => Ok(None),
            ResumedEntry::Recovered(entry) => entry
                .observations
                .get("diagnostics")
                .filter(|value| !value.is_null())
                .map(AttemptDiagnostics::from_value)
                .transpose(),
        })
        .collect::<Result<Vec<_>, String>>()?;
    if let Some(diagnostics) = terminal_diagnostics(views.iter().zip(&recovered).filter_map(
        |(view, recovered)| {
            let diagnostics = match view.entry {
                ResumedEntry::Observed(item) => item.diagnostics.as_ref(),
                ResumedEntry::Recovered(_) => recovered.as_ref(),
            }?;
            Some((view.kind, view.key, diagnostics))
        },
    ))? {
        fields.push(("terminal_diagnostics", diagnostics));
    }
    serde_json::to_vec_pretty(&object(fields)).map_err(|error| error.to_string())
}

/// A batch's closing lines on the terminal: what was converted, how long it
/// took and what it cost, then skipped items by reason with the next step,
/// then the failed and unfinished counts (each failure's error is printed
/// before).
pub(crate) fn batch_summary(records: &[RunItem], elapsed: std::time::Duration) -> Vec<String> {
    const EXAMPLES: usize = 2;
    let noun = |count: usize, one: &str, many: &str| {
        format!("{count} {}", if count == 1 { one } else { many })
    };
    let completed = |kind: ItemKind| {
        records
            .iter()
            .filter(|record| record.kind == kind && record.status == ItemStatus::Completed)
            .count()
    };
    let mut parts = Vec::new();
    if completed(ItemKind::File) > 0 {
        parts.push(noun(completed(ItemKind::File), "file", "files"));
    }
    if completed(ItemKind::Url) > 0 {
        parts.push(noun(completed(ItemKind::Url), "URL", "URLs"));
    }
    let seconds = elapsed.as_secs();
    let mut detail = format!("{}:{:02}", seconds / 60, seconds % 60);
    let cost: f64 = records.iter().map(|record| record.usage.cost_usd).sum();
    if cost > 0.0 {
        detail.push_str(&format!(", ${cost:.3}"));
    }
    if records
        .iter()
        .any(|record| record.usage.requests > 0 && !record.usage.cost_complete())
    {
        detail.push_str(", cost incomplete");
    }
    let converted = if parts.is_empty() {
        "nothing converted".to_owned()
    } else {
        parts.join(", ")
    };
    let mut lines = vec![format!("Done: {converted} ({detail})")];
    let mut skipped: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for record in records
        .iter()
        .filter(|record| record.status == ItemStatus::Skipped)
    {
        skipped
            .entry(record.skip_reason.as_deref().unwrap_or("skipped"))
            .or_default()
            .push(&record.display);
    }
    for (reason, names) in skipped {
        let mut examples = names[..names.len().min(EXAMPLES)].join(", ");
        if names.len() > EXAMPLES {
            examples.push_str(", ...");
        }
        let hint = match reason {
            "image_only" => " Use --llm or --ocr for content extraction.",
            "exists" => " Set output.on_conflict to overwrite or rename to convert them again.",
            "pending_batch" => " Collect it later with --llm-batch-collect or --resume.",
            _ => "",
        };
        lines.push(format!(
            "Skipped {} ({reason}): {examples}.{hint}",
            noun(names.len(), "item", "items")
        ));
    }
    for (status, label) in [
        (ItemStatus::Failed, "Failed"),
        (ItemStatus::Pending, "Not finished"),
    ] {
        let count = records
            .iter()
            .filter(|record| record.status == status)
            .count();
        if count > 0 {
            lines.push(format!("{label}: {}", noun(count, "item", "items")));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_summary_names_counts_time_cost_skips_and_failures() {
        let mut records: Vec<RunItem> = (0..6)
            .map(|index| {
                item(
                    index,
                    if index < 3 {
                        ItemKind::File
                    } else {
                        ItemKind::Url
                    },
                    &format!("k{index}"),
                )
            })
            .collect();
        records[0].usage.cost_usd = 0.0126;
        for (index, reason) in [(1, "image_only"), (2, "image_only"), (4, "image_only")] {
            records[index].status = ItemStatus::Skipped;
            records[index].skip_reason = Some(reason.into());
        }
        records[5].status = ItemStatus::Failed;
        // A request without a reviewed price leaves the cost incomplete.
        records[3].usage.requests = 1;
        assert_eq!(
            batch_summary(&records, std::time::Duration::from_secs(75)),
            [
                "Done: 1 file, 1 URL (1:15, $0.013, cost incomplete)",
                "Skipped 3 items (image_only): display-k1, display-k2, .... Use --llm or --ocr for content extraction.",
                "Failed: 1 item",
            ]
        );
        records.iter_mut().for_each(|record| {
            record.status = ItemStatus::Skipped;
            record.skip_reason = Some("exists".into());
            record.usage.cost_usd = 0.0;
        });
        records[1].status = ItemStatus::Pending;
        records[1].usage.requests = 0;
        let lines = batch_summary(&records[..2], std::time::Duration::from_secs(3));
        assert_eq!(lines[0], "Done: nothing converted (0:03)");
        assert_eq!(lines[2], "Not finished: 1 item");
        assert!(
            lines[1].starts_with("Skipped 1 item (exists): display-k0. Set output.on_conflict"),
            "{lines:?}"
        );
    }

    fn item(index: usize, kind: ItemKind, key: &str) -> RunItem {
        RunItem {
            index,
            kind,
            display: format!("display-{key}"),
            report_key: key.into(),
            source_file: None,
            status: ItemStatus::Completed,
            output: Some(PathBuf::from("out/final.md")),
            history_output: None,
            history_eligible: true,
            error: None,
            warnings: vec!["not in reports".into()],
            skip_reason: None,
            started_at: "2026-09-28T12:00:00+08:00".into(),
            completed_at: "2026-09-28T12:00:01+08:00".into(),
            elapsed_s: 1.23456,
            conversion_duration_s: Some(1.12345),
            images: 2,
            screenshots: 0,
            usage: ConversionUsage::default(),
            diagnostics: None,
            llm_cache_hit: false,
            fetch_cache_hit: false,
            fetch_strategy: Some("static".into()),
        }
    }

    fn run(root: &Path, mode: RunMode, cfg: Value) -> RunInfo {
        RunInfo {
            mode,
            input: root.join("input 世界").to_string_lossy().into_owned(),
            output_dir: root.join("output"),
            started_at: "2026-09-28T12:00:00+08:00".into(),
            log_file: None,
            options: ReportOptions::from_config(&cfg, None, &[]),
        }
    }

    fn finish() -> RunFinished {
        RunFinished {
            updated_at: "2026-09-28T12:01:01+08:00".into(),
            duration_s: 61.9,
        }
    }

    fn decode(plan: &ReportPlan, items: &[RunItem]) -> (String, Value) {
        let bytes = render(plan, items, &finish()).unwrap();
        (
            String::from_utf8(bytes.clone()).unwrap(),
            serde_json::from_slice(&bytes).unwrap(),
        )
    }

    #[test]
    fn defaults_and_mode_specific_hashes_do_not_create_output() {
        let root = tempfile::tempdir().unwrap();
        for mode in [RunMode::SingleFile, RunMode::SingleUrl] {
            assert!(
                plan(run(root.path(), mode, json!({})), None, "rename", false)
                    .unwrap()
                    .is_none()
            );
        }
        for mode in [RunMode::Directory, RunMode::UrlList] {
            assert!(
                plan(run(root.path(), mode, json!({})), None, "rename", false)
                    .unwrap()
                    .is_some()
            );
            assert!(
                plan(
                    run(root.path(), mode, json!({})),
                    Some(false),
                    "rename",
                    false
                )
                .unwrap()
                .is_none()
            );
        }
        let cfg =
            json!({"llm":{"enabled":true},"ocr":{"enabled":true},"batch":{"scan_max_depth":9}});
        let single = plan(
            run(root.path(), RunMode::SingleFile, cfg.clone()),
            Some(true),
            "rename",
            false,
        )
        .unwrap()
        .unwrap();
        let expected = report_store::task_hash(
            Path::new(&single.run.input),
            &single.output_dir,
            &json!({"llm":true,"ocr":true,"screenshot":false,"alt":false,"desc":false}),
        )
        .unwrap();
        assert_eq!(single.task_hash, expected);
        let mut first_url = run(root.path(), RunMode::SingleUrl, cfg.clone());
        first_url.input = "https://example.test/a?key=first".into();
        let mut next_url = run(root.path(), RunMode::SingleUrl, cfg);
        next_url.input = "https://example.test/other".into();
        let a = plan(first_url, Some(true), "rename", false)
            .unwrap()
            .unwrap();
        let b = plan(next_url, Some(true), "rename", false)
            .unwrap()
            .unwrap();
        assert_eq!(a.task_hash, b.task_hash);
        assert_eq!(
            a.task_hash,
            report_store::task_hash(&a.output_dir, &a.output_dir, &json!({"llm":true})).unwrap()
        );
        assert!(!root.path().join("output").exists());
    }

    #[test]
    fn single_file_preserves_flat_usage_final_path_and_field_order() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan(
            run(root.path(), RunMode::SingleFile, json!({})),
            Some(true),
            "rename",
            false,
        )
        .unwrap()
        .unwrap();
        let mut row = item(0, ItemKind::File, "notes.txt");
        row.output = Some(root.path().join("chosen.md"));
        row.usage.cost_usd = 0.00000049;
        row.usage.input_tokens = 999;
        row.usage.by_model =
            json!({"z-model":{"requests":2,"input_tokens":7,"output_tokens":3,"cost_usd":8.0}})
                .as_object()
                .unwrap()
                .clone();
        let (text, body) = decode(&plan, &[row]);
        assert_eq!(
            body["documents"]["notes.txt"]["output"],
            root.path().join("chosen.md").to_string_lossy().as_ref()
        );
        assert_eq!(
            body["documents"]["notes.txt"]["llm_usage"],
            json!({"input_tokens":7,"output_tokens":3,"cost_usd":0.00000049})
        );
        assert_eq!(body["llm_usage"]["cost_usd"], 0.00000049);
        assert_eq!(body["llm_usage"]["requests"], 2);
        assert_eq!(body["summary"]["duration"], "1.2s");
        assert!(body.get("options").is_none() && body.get("url_sources").is_none());
        assert!(body["documents"]["notes.txt"].get("warnings").is_none());
        let keys = [
            "\"version\"",
            "\"generated_at\"",
            "\"log_file\"",
            "\"summary\"",
            "\"llm_usage\"",
            "\"documents\"",
        ];
        assert!(
            keys.windows(2)
                .all(|pair| text.find(pair[0]).unwrap() < text.find(pair[1]).unwrap())
        );
    }

    #[test]
    fn single_url_uses_actual_strategy_and_independent_cache_details() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan(run(root.path(), RunMode::SingleUrl,
            json!({"fetch":{"strategy":"auto","headers":{"Authorization":"secret"}},"llm":{"enabled":true,"api_key":"secret"}})), Some(true), "rename", false).unwrap().unwrap();
        let mut row = item(0, ItemKind::Url, "https://example.test/a");
        row.fetch_cache_hit = true;
        row.source_file = Some("must-not-appear.urls".into());
        let (text, body) = decode(&plan, &[row]);
        let url = &body["url_sources"]["cli"]["urls"]["https://example.test/a"];
        assert_eq!(url["cache_hit"], true);
        assert_eq!(url["cache_details"], json!({"fetch":true,"llm":false}));
        assert_eq!(body["options"]["fetch_strategy"], "static");
        assert_eq!(body["options"].as_object().unwrap().len(), 5);
        assert!(url.get("source_file").is_none() && body.get("documents").is_none());
        assert!(!text.contains("secret") && !text.contains("must-not-appear"));
    }

    #[test]
    fn directory_keeps_failed_pending_skips_completed_and_source_order() {
        let root = tempfile::tempdir().unwrap();
        let mut info = run(
            root.path(),
            RunMode::Directory,
            json!({"llm":{"enabled":true,
            "model_list":[{"litellm_params":{"model":"m","api_key":"secret"}}]}}),
        );
        info.options = ReportOptions::from_config(
            &json!({"llm":{"enabled":true,
            "model_list":[{"litellm_params":{"model":"m","api_key":"secret"}}]}}),
            Some(3),
            &[" *.txt ".into(), " ".into()],
        );
        let plan = plan(info, None, "rename", false).unwrap().unwrap();
        let mut failure = item(1, ItemKind::File, "z/bad.txt");
        failure.status = ItemStatus::Failed;
        failure.error = Some("failure".into());
        failure.output = None;
        let mut skip = item(0, ItemKind::File, "a/skip.txt");
        skip.status = ItemStatus::Skipped;
        skip.skip_reason = Some("exists".into());
        let mut first = item(2, ItemKind::Url, "https://example.test/z name");
        first.source_file = Some("links.urls".into());
        first.fetch_cache_hit = true;
        let mut second = item(3, ItemKind::Url, "https://example.test/a name.md");
        second.source_file = Some("links.urls".into());
        second.llm_cache_hit = true;
        second.usage.by_model =
            json!({"m":{"requests":1,"input_tokens":9,"output_tokens":4,"cost_usd":0.1}})
                .as_object()
                .unwrap()
                .clone();
        let (text, body) = decode(&plan, &[second, failure, first, skip]);
        assert_eq!(body["summary"]["completed_documents"], 1);
        assert_eq!(body["summary"]["failed_documents"], 1);
        assert_eq!(body["summary"]["pending_documents"], 1);
        assert_eq!(body["summary"]["url_cache_hits"], 1);
        assert_eq!(body["documents"]["a/skip.txt"]["status"], "completed");
        assert_eq!(body["summary"]["duration"], "01:01");
        assert_eq!(body["summary"]["processing_time"], "4.9s");
        assert_eq!(body["options"]["scan_max_depth"], 3);
        assert_eq!(body["options"]["glob_patterns"], json!(["*.txt"]));
        assert_eq!(body["options"]["models"], json!(["m"]));
        assert_eq!(body["llm_usage"]["models"]["m"]["cached_input_tokens"], 0);
        assert!(
            body["url_sources"]["links.urls"]["urls"]["https://example.test/z name"]
                .get("cache_details")
                .is_none()
        );
        assert!(
            text.find("https://example.test/z name").unwrap()
                < text.find("https://example.test/a name.md").unwrap()
        );
        assert!(!text.contains("secret") && !text.contains("not in reports"));
    }

    #[test]
    fn url_list_omits_details_and_keeps_unknown_source_and_skip_shape() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan(
            run(root.path(), RunMode::UrlList, json!({})),
            None,
            "rename",
            false,
        )
        .unwrap()
        .unwrap();
        let mut good = item(0, ItemKind::Url, "https://example.test/a name");
        good.source_file = Some("actual.urls".into());
        good.llm_cache_hit = true;
        let mut skip = item(1, ItemKind::Url, "https://example.test/a name.md");
        skip.status = ItemStatus::Skipped;
        skip.skip_reason = Some("exists".into());
        let mut bad = item(2, ItemKind::Url, "https://example.test/b");
        bad.status = ItemStatus::Failed;
        bad.error = Some("failed fetch".into());
        let (_, body) = decode(&plan, &[good, skip, bad]);
        assert_eq!(body["summary"]["completed_urls"], 1);
        assert_eq!(body["summary"]["failed_urls"], 1);
        assert_eq!(body["summary"]["total_urls"], 3);
        assert!(body.get("options").is_none());
        let urls = &body["url_sources"]["unknown.urls"]["urls"];
        assert_eq!(
            urls["https://example.test/a name.md"],
            json!({"status":"skipped","error":"Output exists"})
        );
        assert_eq!(
            urls["https://example.test/b"],
            json!({"status":"failed","error":"failed fetch"})
        );
        for key in [
            "duration",
            "cost_usd",
            "llm_usage",
            "cache_hit",
            "source_file",
        ] {
            assert!(urls["https://example.test/a name"].get(key).is_none());
        }
    }

    #[test]
    fn duration_boundaries_and_invalid_measurements_are_explicit() {
        for (seconds, expected) in [
            (0.0, "0.0s"),
            (59.94, "59.9s"),
            (60.0, "01:00"),
            (3599.9, "59:59"),
            (3600.0, "01:00:00"),
            (360000.0, "100:00:00"),
        ] {
            assert_eq!(serde_json::to_value(duration(seconds)).unwrap(), expected);
        }
        let root = tempfile::tempdir().unwrap();
        let plan = plan(
            run(root.path(), RunMode::SingleFile, json!({})),
            Some(true),
            "rename",
            false,
        )
        .unwrap()
        .unwrap();
        let mut bad = item(0, ItemKind::File, "a.txt");
        bad.elapsed_s = f64::NAN;
        assert!(render(&plan, &[bad], &finish()).is_err());
        assert!(render(&plan, &[], &finish()).is_err());
    }

    #[test]
    fn batch_usage_merges_models_but_takes_total_cost_from_items() {
        let root = tempfile::tempdir().unwrap();
        let plan = plan(
            run(root.path(), RunMode::UrlList, json!({})),
            None,
            "rename",
            false,
        )
        .unwrap()
        .unwrap();
        let mut first = item(0, ItemKind::Url, "https://example.test/first");
        first.usage.cost_usd = 0.123456789;
        first.usage.by_model = json!({"m":{"requests":1,"input_tokens":8,"output_tokens":2,"cached_input_tokens":3,"cost_usd":1.5}}).as_object().unwrap().clone();
        let mut second = item(1, ItemKind::Url, "https://example.test/second");
        second.usage.by_model =
            json!({"m":{"requests":2,"input_tokens":4,"output_tokens":5,"cost_usd":2.0}})
                .as_object()
                .unwrap()
                .clone();
        let mut skipped = item(2, ItemKind::Url, "https://example.test/skipped");
        skipped.status = ItemStatus::Skipped;
        skipped.usage.cost_usd = 9.0;
        skipped.usage.by_model = json!({"must-not-be-merged":{"requests":99}})
            .as_object()
            .unwrap()
            .clone();
        let (_, body) = decode(&plan, &[first, second, skipped]);
        assert_eq!(
            body["llm_usage"],
            json!({
                "models":{"m":{"requests":3,"input_tokens":12,"output_tokens":7,"cached_input_tokens":3,"cost_usd":3.5,"priced_requests":0,"unpriced_requests":3,"cost_status":"unknown"}},
                "requests":3,"input_tokens":12,"output_tokens":7,"cost_usd":0.123456789,
                "pricing":{"priced_requests":0,"unpriced_requests":3,"cost_status":"unknown","pricing_snapshots":[]},
            })
        );
    }

    fn recovered(status: crate::run_state::Status, observations: Value) -> crate::run_state::Entry {
        crate::run_state::Entry {
            status,
            observations: observations.as_object().unwrap().clone(),
            ..Default::default()
        }
    }

    fn resumed_plan(root: &Path, mode: RunMode) -> ReportPlan {
        plan(run(root, mode, json!({})), None, "rename", false)
            .unwrap()
            .unwrap()
    }

    fn decode_resumed(
        plan: &ReportPlan,
        items: &[RunItem],
        state: &crate::run_state::Snapshot,
        active: &[String],
    ) -> (String, Value) {
        let bytes = render_resumed(plan, items, state, active, &finish()).unwrap();
        (
            String::from_utf8(bytes.clone()).unwrap(),
            serde_json::from_slice(&bytes).unwrap(),
        )
    }

    #[test]
    fn completed_only_resume_preserves_rows_without_inventing_measurements() {
        let root = tempfile::tempdir().unwrap();
        let plan = resumed_plan(root.path(), RunMode::Directory);
        let mut state = crate::run_state::Snapshot::default();
        let mut row = recovered(crate::run_state::Status::Completed, json!({}));
        row.output = Some(root.path().join("previously-written.md"));
        state.documents.insert("removed-from-input.txt".into(), row);
        let (_, body) = decode_resumed(&plan, &[], &state, &[]);
        let row = &body["documents"]["removed-from-input.txt"];
        assert_eq!(row["status"], "completed");
        assert_eq!(
            row["output"],
            root.path()
                .join("previously-written.md")
                .to_string_lossy()
                .as_ref()
        );
        for key in ["started_at", "completed_at", "duration"] {
            assert!(row.get(key).unwrap().is_null());
        }
        assert_eq!(row["images"], 0);
        assert_eq!(row["screenshots"], 0);
        assert_eq!(row["cache_hit"], false);
        assert_eq!(row["llm_usage"], json!({}));
        assert_eq!(body["summary"]["completed_documents"], 1);
        assert_eq!(body["summary"]["failed_documents"], 0);
        assert_eq!(body["summary"]["pending_documents"], 0);
        assert_eq!(body["llm_usage"]["requests"], 0);
        assert_eq!(body["llm_usage"]["cost_usd"], 0.0);
    }

    #[test]
    fn resumed_directory_counts_real_pending_failed_and_in_progress_states() {
        let root = tempfile::tempdir().unwrap();
        let plan = resumed_plan(root.path(), RunMode::Directory);
        let mut state = crate::run_state::Snapshot::default();
        for (key, status) in [
            ("pending.txt", crate::run_state::Status::Pending),
            ("failed.txt", crate::run_state::Status::Failed),
            ("running.txt", crate::run_state::Status::InProgress),
        ] {
            state
                .documents
                .insert(key.into(), recovered(status, json!({})));
        }
        let (_, body) = decode_resumed(&plan, &[], &state, &[]);
        assert_eq!(body["documents"]["pending.txt"]["status"], "pending");
        assert_eq!(body["documents"]["running.txt"]["status"], "in_progress");
        assert_eq!(body["summary"]["total_documents"], 3);
        assert_eq!(body["summary"]["completed_documents"], 0);
        assert_eq!(body["summary"]["failed_documents"], 1);
        assert_eq!(body["summary"]["pending_documents"], 2);
    }

    #[test]
    fn observed_identity_replaces_old_usage_instead_of_adding_it_twice() {
        let root = tempfile::tempdir().unwrap();
        let plan = resumed_plan(root.path(), RunMode::Directory);
        let mut state = crate::run_state::Snapshot::default();
        state.documents.insert("retry.txt".into(), recovered(crate::run_state::Status::Failed,
            json!({"duration":100.0,"cost_usd":99.0,"images":99,"llm_usage":{"old":{"requests":99}}})));
        state.documents.insert("retained.txt".into(), recovered(crate::run_state::Status::Completed,
            json!({"duration":2.0,"images":4,"cost_usd":0.25,"llm_usage":{"m":{"requests":1,"input_tokens":7,"output_tokens":3,"cost_usd":0.25}}})));
        let mut observed = item(0, ItemKind::File, "retry.txt");
        observed.usage.cost_usd = 0.5;
        observed.usage.by_model =
            json!({"m":{"requests":2,"input_tokens":11,"output_tokens":5,"cost_usd":0.5}})
                .as_object()
                .unwrap()
                .clone();
        let (_, body) = decode_resumed(&plan, &[observed], &state, &[]);
        assert_eq!(body["documents"]["retry.txt"]["status"], "completed");
        assert_eq!(body["documents"]["retry.txt"]["images"], 2);
        assert_eq!(body["documents"]["retry.txt"]["duration"], "1.2s");
        assert_eq!(body["documents"]["retained.txt"]["duration"], "2.0s");
        assert_eq!(body["summary"]["processing_time"], "3.2s");
        assert_eq!(body["llm_usage"]["cost_usd"], 0.75);
        assert_eq!(body["llm_usage"]["requests"], 3);
        assert_eq!(body["llm_usage"]["input_tokens"], 18);
        assert!(body["llm_usage"]["models"].get("old").is_none());
    }

    #[test]
    fn resumed_url_list_keeps_only_current_raw_keys_in_requested_order() {
        let root = tempfile::tempdir().unwrap();
        let plan = resumed_plan(root.path(), RunMode::UrlList);
        let first = "https://example.test/shared name";
        let second = "https://example.test/shared name.md";
        let removed = "https://example.test/removed";
        let mut state = crate::run_state::Snapshot::default();
        state.urls.insert(
            second.into(),
            recovered(crate::run_state::Status::Pending, json!({})),
        );
        state.urls.insert(first.into(), recovered(crate::run_state::Status::Completed,
            json!({"fetch_strategy":"static","cost_usd":0.125,"llm_usage":{"saved":{"requests":1}}})));
        state.urls.insert(
            removed.into(),
            recovered(
                crate::run_state::Status::Completed,
                json!({"cost_usd":100.0,"llm_usage":{"removed":{"requests":100}}}),
            ),
        );
        let active = vec![first.into(), second.into()];
        let mut observed = item(0, ItemKind::Url, second);
        observed.status = ItemStatus::Failed;
        observed.error = Some("current failure".into());
        let (text, body) = decode_resumed(&plan, &[observed], &state, &active);
        let urls = &body["url_sources"]["unknown.urls"]["urls"];
        assert_eq!(urls[first]["status"], "completed");
        assert_eq!(urls[first]["images"], 0);
        assert_eq!(urls[first]["screenshots"], 0);
        assert_eq!(
            urls[second],
            json!({"status":"failed","error":"current failure"})
        );
        assert!(urls.get(removed).is_none());
        assert!(
            text.find(&format!("\"{first}\"")).unwrap()
                < text.find(&format!("\"{second}\"")).unwrap()
        );
        assert_eq!(body["summary"]["total_urls"], 2);
        assert_eq!(body["summary"]["completed_urls"], 1);
        assert_eq!(body["summary"]["failed_urls"], 1);
        assert_eq!(body["llm_usage"]["cost_usd"], 0.125);
        assert_eq!(body["llm_usage"]["requests"], 1);
        assert!(body["llm_usage"]["models"].get("removed").is_none());
    }

    #[test]
    fn resumed_url_list_all_completed_still_renders_without_observed_items() {
        let root = tempfile::tempdir().unwrap();
        let plan = resumed_plan(root.path(), RunMode::UrlList);
        let key = "https://example.test/done";
        let mut state = crate::run_state::Snapshot::default();
        state.urls.insert(
            key.into(),
            recovered(crate::run_state::Status::Completed, json!({})),
        );
        let (_, body) = decode_resumed(&plan, &[], &state, &[key.into()]);
        assert_eq!(body["summary"]["completed_urls"], 1);
        assert_eq!(
            body["url_sources"]["unknown.urls"]["urls"][key],
            json!({
                "status":"completed", "output":null, "error":null,
                "fetch_strategy":null,"images":0,"screenshots":0,
            })
        );
        assert!(
            render_resumed(
                &plan,
                &[],
                &state,
                &["https://example.test/absent".into()],
                &finish()
            )
            .is_err()
        );
    }

    #[test]
    fn fully_observed_batch_projection_matches_existing_render_bytes() {
        let root = tempfile::tempdir().unwrap();
        for mode in [RunMode::Directory, RunMode::UrlList] {
            let plan = resumed_plan(root.path(), mode);
            let mut first = item(0, ItemKind::Url, "https://example.test/z");
            first.source_file = Some("links.urls".into());
            let mut second = item(1, ItemKind::Url, "https://example.test/a");
            second.source_file = Some("links.urls".into());
            second.status = ItemStatus::Skipped;
            let items = vec![first, second];
            let mut state = crate::run_state::Snapshot::default();
            let keys: Vec<_> = items.iter().map(|item| item.report_key.clone()).collect();
            for key in &keys {
                state.urls.insert(
                    key.clone(),
                    recovered(crate::run_state::Status::Pending, json!({})),
                );
            }
            assert_eq!(
                render(&plan, &items, &finish()).unwrap(),
                render_resumed(&plan, &items, &state, &keys, &finish()).unwrap()
            );
        }
    }

    #[test]
    fn provider_pending_reports_preserve_base_without_counting_remote_completion() {
        let root = tempfile::tempdir().unwrap();
        for (mode, kind, key) in [
            (RunMode::SingleFile, ItemKind::File, "notes.txt"),
            (
                RunMode::SingleUrl,
                ItemKind::Url,
                "https://example.test/notes",
            ),
            (RunMode::Directory, ItemKind::File, "notes.txt"),
            (
                RunMode::UrlList,
                ItemKind::Url,
                "https://example.test/notes",
            ),
        ] {
            let plan = plan(
                run(root.path(), mode, json!({})),
                Some(true),
                "rename",
                false,
            )
            .unwrap()
            .unwrap();
            let mut row = item(0, kind, key);
            row.status = ItemStatus::Pending;
            row.output = Some(root.path().join("base.md"));
            let (_, body) = decode(&plan, &[row]);
            let (total, completed, failed, pending) = if kind == ItemKind::File {
                (
                    "total_documents",
                    "completed_documents",
                    "failed_documents",
                    "pending_documents",
                )
            } else {
                (
                    "total_urls",
                    "completed_urls",
                    "failed_urls",
                    "pending_urls",
                )
            };
            assert_eq!(body["summary"][total], 1);
            assert_eq!(body["summary"][completed], 0);
            assert_eq!(body["summary"][failed], 0);
            assert_eq!(body["summary"][pending], 1);
            let entry = if kind == ItemKind::File {
                &body["documents"][key]
            } else {
                &body["url_sources"][if mode == RunMode::SingleUrl {
                    "cli"
                } else {
                    "unknown.urls"
                }]["urls"][key]
            };
            assert_eq!(entry["status"], "pending");
            assert_eq!(
                entry["output"],
                root.path().join("base.md").to_string_lossy().as_ref()
            );
            assert!(body["llm_usage"].get("pricing").is_none());
        }
    }

    #[test]
    fn resumed_report_combines_known_and_legacy_requests_without_repricing_history() {
        let root = tempfile::tempdir().unwrap();
        let plan = resumed_plan(root.path(), RunMode::Directory);
        let mut state = crate::run_state::Snapshot::default();
        state.documents.insert("legacy.txt".into(), recovered(crate::run_state::Status::Completed,
            json!({"cost_usd":0.25,"llm_usage":{"same":{"requests":1,"input_tokens":7,"output_tokens":3,"cost_usd":0.25}}})));
        let mut current = item(0, ItemKind::File, "current.txt");
        current.usage.cost_usd = 0.5;
        current.usage.by_model = json!({"same":{"requests":2,"input_tokens":8,"output_tokens":4,"cost_usd":0.5,"priced_requests":2,"unpriced_requests":0,"cost_status":"complete","pricing_snapshot":"catalog-v1"}}).as_object().unwrap().clone();
        let body: Value = serde_json::from_slice(
            &render_resumed(&plan, &[current], &state, &[], &finish()).unwrap(),
        )
        .unwrap();
        assert_eq!(body["llm_usage"]["cost_usd"], 0.75);
        assert_eq!(
            body["llm_usage"]["pricing"],
            json!({"priced_requests":2,"unpriced_requests":1,"cost_status":"partial","pricing_snapshots":["catalog-v1"]})
        );
        assert_eq!(
            body["documents"]["legacy.txt"]["pricing"]["cost_status"],
            "unknown"
        );
        assert_eq!(
            body["documents"]["current.txt"]["pricing"]["cost_status"],
            "complete"
        );
        assert!(
            body["documents"]["legacy.txt"]["llm_usage"]["same"]
                .get("pricing_snapshot")
                .is_none()
        );
    }
}
