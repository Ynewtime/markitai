//! A cloud job remains separate from the local directory conversion checkpoint.
mod python_import;
mod resume;
mod store;
pub(super) use resume::resume;
use resume::{advance_frozen, reconciliation_handoff};

use super::logging::Shape;
use super::*;
use crate::output_claims::{Claim, MemberLeases, Owner, Policy};
use markitai_core::output::Publication;
use markitai_core::provider_batch as provider;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::time::Duration;

const MEMBER_LIMIT: u64 = 100 * 1024 * 1024;

/// The lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    markitai_core::hex(Sha256::digest(bytes))
}

fn member(path: &Path) -> CliResult<Vec<u8>> {
    let file = markitai_core::platform::open_read(path, false).map_err(runtime)?;
    let metadata = file.metadata().map_err(runtime)?;
    if !metadata.is_file() || metadata.len() > MEMBER_LIMIT {
        return Err(runtime(
            "Provider Batch output member is not a bounded regular file",
        ));
    }
    markitai_core::platform::read_limited(file, MEMBER_LIMIT)
        .map_err(runtime)?
        .ok_or_else(|| runtime("Provider Batch output member exceeds its byte limit"))
}

pub(super) fn reject_pending(input: &Path, output: &Path, cfg: &Value) -> CliResult<()> {
    if !output.exists() {
        return Ok(());
    }
    let input = crate::report_store::resolve_path(input).map_err(runtime)?;
    let pending = store::Store::pending(
        output,
        config::enabled(cfg, "/output/allow_symlinks"),
        Default::default(),
    )
    .map_err(runtime)?;
    if let Some(job) = pending.iter().find(|job| job.input_root == input) {
        return Err(runtime(format!(
            "Unfinished provider batch {} retains this input scope ({:?}); collect or resolve it before starting another run",
            job.batch_id.as_deref().unwrap_or(&job.id),
            job.phase
        )));
    }
    Ok(())
}

fn report_and_history(
    cli: &Cli,
    cfg: &Value,
    input: &str,
    output: &Path,
    items: &[RunItem],
    started: &str,
    clock: Instant,
) -> CliResult<()> {
    let report = report::plan(
        RunInfo {
            mode: RunMode::Directory,
            input: input.into(),
            output_dir: output.into(),
            started_at: started.into(),
            log_file: logging::path(),
            options: ReportOptions::from_config(cfg, cli.max_depth, &cli.globs),
        },
        cfg["output"]["report"].as_bool(),
        cfg["output"]["on_conflict"].as_str().unwrap_or("rename"),
        config::enabled(cfg, "/output/allow_symlinks"),
    )
    .map_err(runtime)?;
    finish_report(report.as_ref(), items, clock, !cli.quiet && !cli.json).map_err(runtime)?;
    if let Some(history) = crate::history::Plan::new(
        cfg,
        Some(output),
        RunMode::Directory,
        cli.preset.as_deref(),
        started,
        !cli.quiet && !cli.json,
    ) {
        history.record(items);
    }
    Ok(())
}

fn collect_command(cli: &Cli, id: &str, output: &Path) -> String {
    fn quote(value: &str) -> String {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
    let mut command = format!(
        "markitai --llm-batch-collect {} -o {}",
        quote(id),
        quote(&output.to_string_lossy())
    );
    if let Some(config) = &cli.config {
        command.push_str(&format!(" -c {}", quote(&config.to_string_lossy())));
    }
    command
}

fn handoff(cli: &Cli, store: &store::Store) -> Value {
    let batch = store
        .state()
        .batch
        .as_ref()
        .expect("submitted job has batch identity");
    let command = collect_command(cli, &batch.id, &store.state().output_root);
    logging::shaped(
        Shape::item(Mark::Info),
        &format!(
            "Provider batch {} is retained. Collect it later with:",
            batch.id
        ),
        None,
    );
    logging::shaped(Shape::COMMAND, &command, None);
    if cli.config_json.is_some() {
        eprintln!(
            "Warning: Re-supply the original model configuration when collecting; inline configuration is not written into this command."
        );
    }
    json!({"id":batch.id,"status":batch.status,"collect_command":command})
}

fn pending_record(record: &mut RunItem) {
    record.status = ItemStatus::Pending;
    record.skip_reason = Some("pending_batch".into());
    record.error =
        Some("Provider Batch enhancement is pending; collect it before retrying.".into());
}

pub(super) fn submit(
    cli: &Cli,
    input: &str,
    cfg: Value,
    mut tasks: Vec<Task>,
    output: &Path,
) -> CliResult<i32> {
    let wait = Duration::from_secs(cli.llm_batch_timeout.unwrap_or(3600));
    if Instant::now().checked_add(wait).is_none() {
        return Err((
            2,
            "--llm-batch-timeout exceeds this platform's supported clock range".into(),
        ));
    }
    if !config::enabled(&cfg, "/llm/enabled") {
        return Err((2, "--llm-batch requires enabled LLM processing".into()));
    }
    let session = provider::Session::new(&cfg).map_err(runtime)?;
    // Reject unsupported plans before even the base conversion can publish files.
    session.prepare("Batch capability check", "preflight.md", &Default::default(), &json!({
        "llm":{"pure":cfg["llm"]["pure"]},"ocr":cfg["ocr"],"screenshot":cfg["screenshot"],
        "image":{"alt_enabled":cfg["image"]["alt_enabled"],"desc_enabled":cfg["image"]["desc_enabled"]},"cache":{"enabled":false}
    })).map_err(runtime)?;
    markitai_core::output::check_path(output, config::enabled(&cfg, "/output/allow_symlinks"))
        .map_err(runtime)?;
    fs::create_dir_all(output).map_err(runtime)?;
    let output = crate::report_store::resolve_path(output).map_err(runtime)?;
    for id in python_import::pending_states(&output, &cfg)? {
        logging::shaped(
            Shape::item(Mark::Warning),
            &format!(
                "Python markitai left batch {id} pending in this output directory; collect it with:"
            ),
            None,
        );
        logging::shaped(Shape::COMMAND, &collect_command(cli, &id, &output), None);
    }
    let input_root = crate::report_store::resolve_path(Path::new(input)).map_err(runtime)?;
    let preparation =
        store::Store::lock_preparation(&output, config::enabled(&cfg, "/output/allow_symlinks"))
            .map_err(runtime)?;
    if !store::Store::pending(
        &output,
        config::enabled(&cfg, "/output/allow_symlinks"),
        Default::default(),
    )
    .map_err(runtime)?
    .is_empty()
    {
        return Err(runtime(
            "An unfinished provider batch owns this output directory; collect or resolve it before preparing another submission",
        ));
    }
    reserve_batch_names(&mut tasks, &cfg)?;
    let started = timestamp();
    let clock = Instant::now();
    let generation = uuid::Uuid::new_v4().to_string();
    let mut base_cfg = cfg.clone();
    base_cfg["llm"]["enabled"] = json!(false);
    let browser = markitai_core::BrowserRuntime::new(8).map_err(runtime)?;
    let context = ConvertContext {
        browser_runtime: Some(&browser),
        ..Default::default()
    };
    let mut records = Vec::with_capacity(tasks.len());
    let mut pending = Vec::new();
    let mut request_bytes = Vec::new();
    for (index, mut task) in tasks.into_iter().enumerate() {
        let owner = Owner {
            generation: generation.clone(),
            mode: "directory".into(),
            input: input_root.clone(),
            output: output.clone(),
            kind: "file".into(),
            key: task.report_key.clone(),
        };
        let claim = batch_run::claim(
            &mut task,
            &base_cfg,
            Some(owner.clone()),
            None,
            &Default::default(),
        )
        .map_err(runtime)?;
        let (mut record, converted) = convert_item(
            &task,
            index,
            &base_cfg,
            context,
            claim.as_ref().map(|claim| claim as &dyn Publication),
        );
        if let Ok(converted) = converted
            && record.status == ItemStatus::Completed
        {
            match session.prepare(
                converted.enhancement_source(),
                &task.source,
                &converted.frontmatter,
                &cfg,
            ) {
                Ok(prepared) => {
                    let base = converted.output_path.as_ref().ok_or_else(|| {
                        runtime("Provider Batch base conversion did not publish Markdown")
                    })?;
                    let base = crate::report_store::resolve_path(base).map_err(runtime)?;
                    let name = base
                        .file_name()
                        .and_then(|name| name.to_str())
                        .and_then(|name| name.strip_suffix(".md"))
                        .ok_or_else(|| runtime("Provider Batch base output is not Markdown"))?;
                    let enhanced = base.with_file_name(format!("{name}.llm.md"));
                    match prepared {
                        provider::Prepared::Cached(decoded) => {
                            claim
                                .as_ref()
                                .ok_or_else(|| {
                                    runtime("Provider Batch cached output has no claim")
                                })?
                                .publish(&enhanced, decoded.content().map_err(runtime)?.as_bytes())
                                .map_err(runtime)?;
                            record.output = Some(enhanced);
                            record.llm_cache_hit = true;
                            record.warnings.extend(decoded.warnings);
                        }
                        provider::Prepared::Request(plan) => {
                            let custom_id = provider::request_id(session.api(), &generation, index);
                            serde_json::to_writer(
                                &mut request_bytes,
                                &plan.request(&custom_id).map_err(runtime)?,
                            )
                            .map_err(runtime)?;
                            request_bytes.push(b'\n');
                            if request_bytes.len() > 200_000_000 {
                                return Err(runtime(
                                    "Provider Batch requests exceed their combined byte limit",
                                ));
                            }
                            pending.push(store::NewItem {
                                custom_id,
                                source: task.source,
                                key: task.report_key,
                                base: base.strip_prefix(&output).map_err(runtime)?.into(),
                                enhanced: enhanced.strip_prefix(&output).map_err(runtime)?.into(),
                                base_sha256: sha256_hex(&member(&base)?),
                                owner,
                                plan: serde_json::to_value(&*plan).map_err(runtime)?,
                            });
                            pending_record(&mut record);
                        }
                    }
                }
                Err(error) => {
                    record.status = ItemStatus::Failed;
                    record.error = Some(error.to_string());
                }
            }
        }
        records.push(record);
    }
    drop(browser);
    if pending.is_empty() {
        report_and_history(cli, &cfg, input, &output, &records, &started, clock)?;
        if cli.json {
            emit_json(&records.iter().map(outcome).collect::<Vec<_>>(), None);
        }
        return Ok(
            if records.iter().any(|item| item.status == ItemStatus::Failed) {
                10
            } else {
                0
            },
        );
    }
    let endpoint = session.identity();
    preparation.validate().map_err(runtime)?;
    let mut store = store::Store::create(
        &output,
        store::NewJob {
            id: generation,
            input_root,
            endpoint: store::Endpoint {
                provider: endpoint.provider.clone(),
                api_base: endpoint.api_base.clone(),
                model: endpoint.model.clone(),
            },
            items: pending,
        },
        config::enabled(&cfg, "/output/allow_symlinks"),
        Default::default(),
    )
    .map_err(runtime)?;
    drop(preparation);
    store
        .save_requests(&mut request_bytes.as_slice())
        .map_err(runtime)?;
    if let Some(reason) = advance_frozen(&mut store, &session, &cfg)? {
        return Ok(reconciliation_handoff(cli, &store, reason));
    }
    wait_and_finish(cli, &cfg, &session, store, records, (started, clock), wait)
}

fn wait_and_finish(
    cli: &Cli,
    cfg: &Value,
    session: &provider::Session,
    mut store: store::Store,
    mut records: Vec<RunItem>,
    timing: (String, Instant),
    wait: Duration,
) -> CliResult<i32> {
    let (started, clock) = timing;
    let input = store.state().input_root.to_string_lossy().into_owned();
    let output = store.state().output_root.clone();
    let client = session.client().map_err(runtime)?;
    let _signals = crate::signals::Guard::install().map_err(runtime)?;
    // A clock boundary reached during preparation hands the saved job back for collection.
    let deadline = Instant::now()
        .checked_add(wait)
        .unwrap_or_else(Instant::now);
    let mut next_poll = Instant::now();
    let mut pending_job = false;
    while !store.state().batch.as_ref().unwrap().status.terminal() {
        if crate::signals::interrupted().is_some() || Instant::now() >= deadline {
            pending_job = true;
            break;
        }
        if Instant::now() < next_poll {
            std::thread::sleep(
                Duration::from_millis(250).min(deadline.saturating_duration_since(Instant::now())),
            );
            continue;
        }
        if crate::signals::interrupted().is_some() {
            pending_job = true;
            break;
        }
        let id = &store.state().batch.as_ref().unwrap().id;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            pending_job = true;
            break;
        }
        let poll = session.client_with_timeout(remaining).map_err(runtime)?;
        match poll.retrieve(id) {
            Ok(batch) => store.bind_batch(batch).map_err(runtime)?,
            Err(error) => {
                eprintln!("Warning: Provider status could not be read: {error}");
                pending_job = true;
                break;
            }
        }
        next_poll = Instant::now() + Duration::from_secs(5);
    }
    let mut batch_json = Value::Null;
    let code = if pending_job {
        batch_json = handoff(cli, &store);
        for record in &mut records {
            if record.status == ItemStatus::Pending {
                record.error = Some(format!(
                    "Provider Batch enhancement is pending; collect with: {}",
                    batch_json["collect_command"]
                        .as_str()
                        .unwrap_or("the saved batch ID")
                ));
            }
        }
        2
    } else {
        let (code, collected) = collect_ready(&mut store, &client, cfg)?;
        let by_key: HashMap<_, _> = collected
            .into_iter()
            .map(|item| (item.report_key.clone(), item))
            .collect();
        let mut by_key = by_key;
        for record in &mut records {
            if let Some(collected) = by_key.remove(&record.report_key) {
                *record = collected;
            }
        }
        if code == 0 && records.iter().any(|item| item.status == ItemStatus::Failed) {
            10
        } else {
            code
        }
    };
    report_and_history(cli, cfg, &input, &output, &records, &started, clock)?;
    if cli.json {
        let mut value = envelope(&records.iter().map(outcome).collect::<Vec<_>>(), None);
        value["batch"] = batch_json;
        if code != 0 {
            value["ok"] = json!(false);
        }
        if code == 1 {
            value["error"] = json!(
                "Provider Batch collection could not finish; saved outputs and observed usage are retained."
            );
        }
        println!("{}", serde_json::to_string_pretty(&value).unwrap());
    }
    Ok(code)
}

pub(super) fn collect(cli: &Cli, id: &str, cfg: Value) -> CliResult<i32> {
    let output = cli.output.as_deref().ok_or_else(|| {
        (
            1,
            "--llm-batch-collect requires the original output directory with -o".into(),
        )
    })?;
    let mut store = match store::Store::open_by_batch(
        output,
        id,
        config::enabled(&cfg, "/output/allow_symlinks"),
        Default::default(),
    ) {
        Ok(store) => store,
        Err(store::Error::NotFound) => match python_import::import(output, id, &cfg)? {
            python_import::Outcome::Imported(store) => *store,
            python_import::Outcome::Collected => {
                logging::shaped(
                    Shape::item(Mark::Info),
                    &format!("Provider batch {id} was already collected by Python markitai."),
                    None,
                );
                return Ok(0);
            }
            python_import::Outcome::Absent => resume::bind_manual(output, id, &cfg)?,
        },
        Err(error) => return Err(runtime(error)),
    };
    if store.state().phase == store::Phase::Collected {
        logging::shaped(
            Shape::item(Mark::Info),
            &format!("Provider batch {id} was already collected."),
            None,
        );
        return Ok(0);
    }
    let session = provider::Session::for_collection(&cfg).map_err(runtime)?;
    resume::match_session(&store, &session)?;
    let client = session.client().map_err(runtime)?;
    let batch = client.retrieve(id).map_err(runtime)?;
    store.bind_batch(batch).map_err(runtime)?;
    if !store.state().batch.as_ref().unwrap().status.terminal() {
        handoff(cli, &store);
        return Ok(2);
    }
    let clock = Instant::now();
    let started = timestamp();
    let (code, records) = collect_ready(&mut store, &client, &cfg)?;
    report_and_history(
        cli,
        &cfg,
        &store.state().input_root.to_string_lossy(),
        &store.state().output_root,
        &records,
        &started,
        clock,
    )?;
    let usage = store.usage().map_err(runtime)?;
    logging::shaped(
        Shape::item(Mark::Success),
        &format!(
            "Collected provider batch {id}: {} observed requests; known priced subtotal ${:.6}.",
            usage.requests, usage.cost_usd
        ),
        None,
    );
    Ok(code)
}

/// The sum of two usage records, as the store totals them.
fn merged(
    left: &markitai_core::ConversionUsage,
    right: &markitai_core::ConversionUsage,
) -> markitai_core::ConversionUsage {
    let mut total = left.clone();
    total.cost_usd += right.cost_usd;
    total.requests = total.requests.saturating_add(right.requests);
    total.input_tokens = total.input_tokens.saturating_add(right.input_tokens);
    total.output_tokens = total.output_tokens.saturating_add(right.output_tokens);
    if crate::pricing::merge_models(&mut total.by_model, &right.by_model).is_err() {
        // Records the store accepted always merge; keep the document's own.
        total.by_model = left.by_model.clone();
    }
    total
}

fn raw_item(item: &provider::ResultItem) -> Value {
    json!({"custom_id":item.custom_id,"http_status":item.http_status,"body":item.body,"error":item.error,"request_id":item.request_id})
}
fn saved_item(bytes: &[u8]) -> CliResult<provider::ResultItem> {
    let raw: Value = serde_json::from_slice(bytes).map_err(runtime)?;
    Ok(provider::ResultItem {
        custom_id: raw["custom_id"]
            .as_str()
            .ok_or_else(|| runtime("Saved Batch custom ID is invalid"))?
            .into(),
        http_status: raw["http_status"]
            .as_u64()
            .and_then(|n| u16::try_from(n).ok()),
        body: raw.get("body").filter(|v| !v.is_null()).cloned(),
        error: raw.get("error").filter(|v| !v.is_null()).cloned(),
        request_id: raw["request_id"].as_str().map(str::to_owned),
    })
}

fn retained_output(
    source: &str,
    base: PathBuf,
    enhanced: Option<PathBuf>,
    usage: markitai_core::ConversionUsage,
    warnings: Vec<String>,
) -> ConversionOutput {
    let mut result = ConversionOutput::default();
    result.source = source.into();
    result.output_path = Some(base);
    result.llm_output_path = enhanced;
    result.usage = usage;
    result.warnings = warnings;
    result
}

fn collection_failure_records(store: &store::Store, message: &str) -> Vec<RunItem> {
    store
        .state()
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let task = Task {
                source: item.source.clone(),
                display: item.source.clone(),
                report_key: item.key.clone(),
                output: Some(store.state().output_root.clone()),
                filename: None,
                reserved_stem: None,
                source_file: None,
            };
            let usage = item
                .result
                .as_ref()
                .map(|record| record.usage.clone())
                .unwrap_or_default();
            let result = if let Some(published) = &item.finalized {
                Ok(retained_output(
                    &item.source,
                    store.state().output_root.join(&item.base),
                    (published.path == item.enhanced)
                        .then(|| store.state().output_root.join(&published.path)),
                    usage,
                    Vec::new(),
                ))
            } else {
                Err(markitai_core::ConversionFailure {
                    error: markitai_core::Error::Conversion(message.into()),
                    usage,
                })
            };
            let mut item_record = recorded(&task, index, Instant::now(), timestamp(), &result);
            if item_record.output.is_none() {
                item_record.output = Some(store.state().output_root.join(&item.base));
            }
            item_record
        })
        .collect()
}

fn collect_ready(
    store: &mut store::Store,
    client: &provider::Client,
    cfg: &Value,
) -> CliResult<(i32, Vec<RunItem>)> {
    match collect_ready_inner(store, client, cfg) {
        Ok(result) => Ok(result),
        Err((_, message)) => {
            eprintln!(
                "Error: Provider Batch collection stopped; previously recorded observations are retained: {message}"
            );
            Ok((1, collection_failure_records(store, &message)))
        }
    }
}

fn collect_ready_inner(
    store: &mut store::Store,
    client: &provider::Client,
    cfg: &Value,
) -> CliResult<(i32, Vec<RunItem>)> {
    // Complete rows can precede a malformed tail. Their presence alone does not
    // prove that a previous download passed validation across both result files.
    if store
        .state()
        .items
        .iter()
        .any(|item| item.finalized.is_none())
    {
        let ids = store.state().expected_ids();
        let downloaded = client.download_results(store.state().batch.as_ref().unwrap(), &ids);
        let (results, error) = match downloaded {
            Ok(results) => (results, None),
            Err(failure) => (failure.partial, Some(failure.error)),
        };
        for item in results.items {
            // An attached request (a Python image analysis) is priced by the
            // plan of the item whose output it belongs to.
            let (index, _) = store
                .state()
                .owner_of(&item.custom_id)
                .ok_or_else(|| runtime("Provider Batch result names an unknown request"))?;
            let owner = store.state().items[index].custom_id.clone();
            let plan = provider::FrozenPlan::from_value(store.read_plan(&owner).map_err(runtime)?)
                .map_err(runtime)?;
            let observed = plan.observe(&item).map_err(runtime)?;
            let recorded = store
                .record_result(
                    &item.custom_id,
                    &serde_json::to_vec(&raw_item(&item)).map_err(runtime)?,
                    observed,
                )
                .map_err(runtime)?;
            // Ledger replay retains the first quote, even across tariff updates.
            let _ = (recorded.inserted, recorded.usage);
        }
        if let Some(error) = error {
            eprintln!(
                "Error: Provider Batch collection stopped; recorded usage and result evidence are retained: {error}"
            );
            return Ok((1, collection_failure_records(store, &error.to_string())));
        }
    }
    let items = store.state().items.clone();
    let output = store.state().output_root.clone();
    let mut records = Vec::with_capacity(items.len());
    let mut failed = false;
    for (index, item) in items.iter().enumerate() {
        let task = Task {
            source: item.source.clone(),
            display: item.source.clone(),
            report_key: item.key.clone(),
            output: Some(output.clone()),
            filename: None,
            reserved_stem: None,
            source_file: None,
        };
        let clock = Instant::now();
        let started = timestamp();
        let mut provider_failed = false;
        let mut result = {
            let mut usage = item
                .result
                .as_ref()
                .map(|result| result.usage.clone())
                .unwrap_or_default();
            for attached in item
                .attached
                .iter()
                .filter_map(|attached| attached.result.as_ref())
            {
                usage = merged(&usage, &attached.usage);
            }
            let process = (|| -> CliResult<ConversionOutput> {
                if let Some(published) = &item.finalized {
                    return Ok(retained_output(
                        &item.source,
                        output.join(&item.base),
                        (published.path == item.enhanced).then(|| output.join(&published.path)),
                        usage.clone(),
                        Vec::new(),
                    ));
                }
                let raw = store
                    .read_result(&item.custom_id)
                    .map_err(runtime)?
                    .ok_or_else(|| {
                        provider_failed = true;
                        runtime("Provider Batch completed without a result for this item")
                    })?;
                let response = saved_item(&raw)?;
                let plan = provider::FrozenPlan::from_value(
                    store.read_plan(&item.custom_id).map_err(runtime)?,
                )
                .map_err(runtime)?;
                let mut descriptions = Vec::new();
                let decoded = match plan {
                    provider::FrozenPlan::Native(plan) => {
                        plan.decode_recorded(&response, cfg, usage.clone())
                    }
                    provider::FrozenPlan::Python(plan) => {
                        let images = store
                            .read_attached(&item.custom_id)
                            .map_err(runtime)?
                            .into_iter()
                            .map(|raw| raw.as_deref().map(saved_item).transpose())
                            .collect::<CliResult<Vec<_>>>()?;
                        plan.decode_recorded(&response, &images, usage.clone()).map(
                            |(decoded, extras)| {
                                descriptions = extras.descriptions;
                                decoded
                            },
                        )
                    }
                }
                .map_err(|error| {
                    provider_failed = true;
                    runtime(error)
                })?;
                let bytes = decoded.content().map_err(runtime)?.into_bytes();
                let base = output.join(&item.base);
                let enhanced = output.join(&item.enhanced);
                if sha256_hex(&member(&base)?) != item.base_sha256 {
                    return Err(runtime(
                        "Base Markdown changed after Batch submission; outputs were preserved",
                    ));
                }
                let parent = base
                    .parent()
                    .ok_or_else(|| runtime("Provider Batch output parent is missing"))?;
                let names = [
                    base.file_name()
                        .and_then(|v| v.to_str())
                        .ok_or_else(|| runtime("Invalid base filename"))?
                        .to_owned(),
                    enhanced
                        .file_name()
                        .and_then(|v| v.to_str())
                        .ok_or_else(|| runtime("Invalid enhanced filename"))?
                        .to_owned(),
                ];
                let leases = MemberLeases::acquire(
                    parent,
                    &names,
                    config::enabled(cfg, "/output/allow_symlinks"),
                )
                .map_err(runtime)?;
                let claim = Claim::new(leases, Some(item.owner.clone()), Policy::RetryOwned, false)
                    .map_err(runtime)?;
                if !enhanced.try_exists().map_err(runtime)? || member(&enhanced)? != bytes {
                    claim.publish(&enhanced, &bytes).map_err(runtime)?;
                }
                let receipt = claim.evidence_digest().map_err(runtime)?;
                for (directory, mut entry) in descriptions {
                    entry["asset"] = json!(
                        output
                            .join(entry["asset"].as_str().unwrap_or_default())
                            .to_string_lossy()
                    );
                    markitai_core::output::publish_image_descriptions(
                        &output.join(directory),
                        &[entry],
                        &item.source,
                        config::enabled(cfg, "/output/allow_symlinks"),
                    )
                    .map_err(runtime)?;
                }
                store
                    .mark_finalized(
                        &item.custom_id,
                        store::Published {
                            path: item.enhanced.clone(),
                            bytes: bytes.len() as u64,
                            sha256: sha256_hex(&bytes),
                            receipt_sha256: receipt,
                        },
                    )
                    .map_err(runtime)?;
                Ok(retained_output(
                    &item.source,
                    base,
                    Some(enhanced),
                    usage.clone(),
                    decoded.warnings,
                ))
            })();
            process.map_err(|(_, message)| markitai_core::ConversionFailure {
                error: markitai_core::Error::Conversion(message),
                usage,
            })
        };
        if provider_failed && cfg["llm"]["on_failure"] != "fail" {
            let failure = result.as_ref().expect_err("provider failure is an error");
            let warning = format!(
                "Provider Batch enhancement failed; base Markdown retained without an automatic live request: {failure}"
            );
            let usage = failure.usage.clone();
            result = (|| -> CliResult<ConversionOutput> {
                let base = output.join(&item.base);
                let enhanced = output.join(&item.enhanced);
                let bytes = member(&base)?;
                if sha256_hex(&bytes) != item.base_sha256 {
                    return Err(runtime(
                        "Base Markdown changed after Batch submission; fallback cannot claim it",
                    ));
                }
                let names = [
                    base.file_name()
                        .and_then(|v| v.to_str())
                        .ok_or_else(|| runtime("Invalid base filename"))?
                        .to_owned(),
                    enhanced
                        .file_name()
                        .and_then(|v| v.to_str())
                        .ok_or_else(|| runtime("Invalid enhanced filename"))?
                        .to_owned(),
                ];
                let leases = MemberLeases::acquire(
                    base.parent().unwrap(),
                    &names,
                    config::enabled(cfg, "/output/allow_symlinks"),
                )
                .map_err(runtime)?;
                let claim = Claim::new(leases, Some(item.owner.clone()), Policy::RetryOwned, false)
                    .map_err(runtime)?;
                store
                    .mark_finalized(
                        &item.custom_id,
                        store::Published {
                            path: item.base.clone(),
                            bytes: bytes.len() as u64,
                            sha256: sha256_hex(&bytes),
                            receipt_sha256: claim.evidence_digest().map_err(runtime)?,
                        },
                    )
                    .map_err(runtime)?;
                Ok(retained_output(
                    &item.source,
                    base,
                    None,
                    usage.clone(),
                    vec![warning],
                ))
            })()
            .map_err(|(_, message)| markitai_core::ConversionFailure {
                error: markitai_core::Error::Conversion(message),
                usage,
            });
        }
        if let Err(error) = &result {
            failed = true;
            store
                .mark_failed(&item.custom_id, &error.to_string())
                .map_err(runtime)?;
        }
        let mut record = recorded(&task, index, clock, started, &result);
        if result.is_err() {
            record.output = Some(output.join(&item.base));
        }
        // Reports leave warnings out; collection is the moment to show them
        // (a base kept after a failed answer, an image left unanalyzed).
        for warning in &record.warnings {
            logging::shaped(
                Shape::item(Mark::Warning),
                &format!("{}: {warning}", item.source),
                None,
            );
        }
        records.push(record);
    }
    if !failed {
        store.finish().map_err(runtime)?;
    }
    Ok((if failed { 10 } else { 0 }, records))
}
