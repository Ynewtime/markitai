//! Continue only the durable operation selected by the frozen job's phase.
use super::*;

pub(super) fn match_session(store: &store::Store, session: &provider::Session) -> CliResult<()> {
    let saved = &store.state().endpoint;
    let configured = session.identity();
    if configured.provider != saved.provider
        || configured.model != saved.model
        || configured.api_base != saved.api_base
    {
        return Err(runtime(
            "Configuration differs from the frozen model and endpoint",
        ));
    }
    Ok(())
}

fn verify_bases(store: &store::Store) -> CliResult<()> {
    for item in &store.state().items {
        let base = store.state().output_root.join(&item.base);
        if sha256_hex(&member(&base)?) != item.base_sha256 {
            return Err(runtime(
                "Base Markdown changed; frozen Batch submission was not continued",
            ));
        }
    }
    Ok(())
}

fn paid_claims(store: &store::Store, cfg: &Value) -> CliResult<Vec<Claim>> {
    let mut claims = Vec::with_capacity(store.state().items.len());
    for item in &store.state().items {
        let base = store.state().output_root.join(&item.base);
        let enhanced = store.state().output_root.join(&item.enhanced);
        let names = [&base, &enhanced]
            .into_iter()
            .map(|path| {
                path.file_name()
                    .and_then(|v| v.to_str())
                    .map(str::to_owned)
                    .ok_or_else(|| runtime("Invalid frozen output filename"))
            })
            .collect::<CliResult<Vec<_>>>()?;
        let leases = MemberLeases::acquire(
            base.parent()
                .ok_or_else(|| runtime("Frozen output parent is missing"))?,
            &names,
            config::enabled(cfg, "/output/allow_symlinks"),
        )
        .map_err(runtime)?;
        claims.push(
            Claim::new(leases, Some(item.owner.clone()), Policy::RetryOwned, false)
                .map_err(runtime)?,
        );
    }
    verify_bases(store)?;
    Ok(claims)
}

/// Some(reason) means unresolved creation: no absence or partial list authorizes POST.
pub(super) fn advance_frozen(
    store: &mut store::Store,
    session: &provider::Session,
    cfg: &Value,
) -> CliResult<Option<&'static str>> {
    use store::ResumeStep;
    match_session(store, session)?;
    let step = store.resume_step().map_err(runtime)?;
    let claims = if matches!(
        step,
        ResumeStep::PrepareRequests | ResumeStep::Upload | ResumeStep::Create
    ) {
        Some(paid_claims(store, cfg)?)
    } else {
        None
    };
    let client = session.client().map_err(runtime)?;
    loop {
        match store.resume_step().map_err(runtime)? {
            ResumeStep::PrepareRequests => {
                let mut requests = Vec::new();
                for item in &store.state().items {
                    let plan: provider::Plan =
                        serde_json::from_value(store.read_plan(&item.custom_id).map_err(runtime)?)
                            .map_err(runtime)?;
                    serde_json::to_writer(
                        &mut requests,
                        &plan.request(&item.custom_id).map_err(runtime)?,
                    )
                    .map_err(runtime)?;
                    requests.push(b'\n');
                    if requests.len() > 200_000_000 {
                        return Err(runtime(
                            "Frozen Batch requests exceed their combined byte limit",
                        ));
                    }
                }
                store
                    .save_requests(&mut requests.as_slice())
                    .map_err(runtime)?;
            }
            ResumeStep::Upload => {
                let uploaded = client
                    .upload(&store.request_path().map_err(runtime)?)
                    .map_err(runtime)?;
                store.save_uploaded(uploaded).map_err(runtime)?;
            }
            ResumeStep::Create => {
                for claim in claims.as_deref().unwrap_or(&[]) {
                    claim.evidence_digest().map_err(runtime)?;
                }
                verify_bases(store)?;
                let uploaded = store
                    .state()
                    .uploaded
                    .as_ref()
                    .ok_or_else(|| runtime("Frozen upload identity is missing"))?
                    .clone();
                let nonce = store.mark_creating().map_err(runtime)?;
                match client.create(&uploaded, &nonce) {
                    Ok(batch) => store.bind_batch(batch).map_err(runtime)?,
                    Err(error) => {
                        if matches!(error, provider::Error::CreateUncertain) {
                            store.mark_uncertain().map_err(runtime)?;
                        } else {
                            store.mark_rejected().map_err(runtime)?;
                        }
                        eprintln!(
                            "Submission evidence retained at {}. Use --llm-batch --resume -o to reconcile an uncertain submission.",
                            store.directory().display()
                        );
                        return Err(runtime(error));
                    }
                }
            }
            ResumeStep::Reconcile => {
                let uploaded = store
                    .state()
                    .uploaded
                    .as_ref()
                    .ok_or_else(|| runtime("Frozen upload identity is missing"))?;
                match client
                    .reconcile(uploaded, &store.state().id, Default::default())
                    .map_err(runtime)?
                {
                    provider::Reconciliation::Found(identity) => {
                        store.bind_reconciled(*identity).map_err(runtime)?
                    }
                    provider::Reconciliation::NotFound => {
                        return Ok(Some(
                            "No matching remote batch was observed; this does not prove that creation failed",
                        ));
                    }
                    provider::Reconciliation::Ambiguous(_) => {
                        return Ok(Some(
                            "Multiple matching remote batches were observed; choose and verify an exact batch ID",
                        ));
                    }
                    provider::Reconciliation::Incomplete => {
                        return Ok(Some(
                            "Remote reconciliation was incomplete; the uncertain submission remains retained",
                        ));
                    }
                }
            }
            ResumeStep::Collect | ResumeStep::Done => return Ok(None),
        }
    }
}

pub(super) fn reconciliation_handoff(cli: &Cli, store: &store::Store, reason: &str) -> i32 {
    eprintln!(
        "{reason}. No new batch was submitted. Resume later, or use --llm-batch-collect ID -o with this output directory."
    );
    if cli.json {
        let mut value = envelope(&[], Some(reason));
        value["batch"] = json!({"local_id":store.state().id,"status":"creation_unresolved"});
        println!("{}", serde_json::to_string_pretty(&value).unwrap());
    }
    2
}

pub(crate) fn resume(cli: &Cli, cfg: Value) -> CliResult<i32> {
    let wait = Duration::from_secs(cli.llm_batch_timeout.unwrap_or(3600));
    if Instant::now().checked_add(wait).is_none() {
        return Err((
            2,
            "--llm-batch-timeout exceeds this platform's supported clock range".into(),
        ));
    }
    let output = cli
        .output
        .as_deref()
        .ok_or_else(|| (2, "Frozen Batch resume requires -o".into()))?;
    let input = cli
        .input
        .as_deref()
        .map(|path| crate::report_store::resolve_path(Path::new(path)))
        .transpose()
        .map_err(runtime)?;
    let preparation =
        store::Store::lock_preparation(output, config::enabled(&cfg, "/output/allow_symlinks"))
            .map_err(runtime)?;
    let mut store = preparation
        .open_frozen(input.as_deref(), Default::default())
        .map_err(runtime)?
        .ok_or_else(|| {
            runtime("No unfinished frozen provider batch was found in this output directory")
        })?;
    drop(preparation);
    let session = match store.resume_step().map_err(runtime)? {
        store::ResumeStep::PrepareRequests
        | store::ResumeStep::Upload
        | store::ResumeStep::Create => {
            if !config::enabled(&cfg, "/llm/enabled") {
                return Err((
                    2,
                    "Frozen Batch submission requires enabled LLM processing".into(),
                ));
            }
            provider::Session::new(&cfg)
        }
        _ => provider::Session::for_collection(&cfg),
    }
    .map_err(runtime)?;
    if let Some(reason) = advance_frozen(&mut store, &session, &cfg)? {
        return Ok(reconciliation_handoff(cli, &store, reason));
    }
    let mut records = collection_failure_records(&store, "Provider Batch result is pending");
    for (record, item) in records.iter_mut().zip(&store.state().items) {
        if item.finalized.is_none() {
            pending_record(record);
        }
    }
    wait_and_finish(
        cli,
        &cfg,
        &session,
        store,
        records,
        (timestamp(), Instant::now()),
        wait,
    )
}

pub(super) fn bind_manual(output: &Path, id: &str, cfg: &Value) -> CliResult<store::Store> {
    let preparation =
        store::Store::lock_preparation(output, config::enabled(cfg, "/output/allow_symlinks"))
            .map_err(runtime)?;
    let mut store = preparation
        .open_frozen(None, Default::default())
        .map_err(runtime)?
        .ok_or_else(|| runtime("No unfinished frozen submission can be bound to this batch ID"))?;
    drop(preparation);
    if store.resume_step().map_err(runtime)? != store::ResumeStep::Reconcile {
        return Err(runtime(
            "Manual batch binding requires a locally uncertain submission",
        ));
    }
    let session = provider::Session::for_collection(cfg).map_err(runtime)?;
    match_session(&store, &session)?;
    let uploaded = store
        .state()
        .uploaded
        .as_ref()
        .ok_or_else(|| runtime("Frozen upload identity is missing"))?;
    let identity = session
        .client()
        .map_err(runtime)?
        .verify_binding(id, uploaded, &store.state().id)
        .map_err(runtime)?;
    store.bind_reconciled(identity).map_err(runtime)?;
    Ok(store)
}
