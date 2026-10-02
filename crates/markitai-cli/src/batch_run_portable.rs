//! Ordinary batch conversion on platforms without native ownership evidence.
use super::*;
use crate::output_claims::{Claim, Error as ClaimError, Owner};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

pub(super) type Reservations = ();

pub(super) fn claim(
    _task: &mut Task,
    _cfg: &Value,
    _owner: Option<Owner>,
    _retry: Option<&Path>,
    _reserved: &Reservations,
) -> Result<Option<Claim>, ClaimError> {
    Ok(None)
}

pub(super) fn run(
    cli: &Cli,
    cfg: &Value,
    mut tasks: Vec<Task>,
    destination: BatchDestination<'_>,
    report_plan: Option<&report::ReportPlan>,
    clock: Instant,
    context: ConvertContext<'_>,
) -> CliResult<i32> {
    if cli.resume {
        return Err(runtime(
            "--resume requires durable output ownership, which is not implemented for this platform",
        ));
    }
    if tasks.is_empty() {
        if cli.json {
            emit_json(&[], None);
        }
        return Ok(0);
    }
    reserve_batch_names(&mut tasks, cfg)?;
    let mut file_indices = Vec::new();
    let mut url_indices = Vec::new();
    for (index, task) in tasks.iter().enumerate() {
        if is_url(&task.source) {
            url_indices.push(index);
        } else {
            file_indices.push(index);
        }
    }
    let groups = [
        (
            file_indices,
            AtomicUsize::new(0),
            cfg["batch"]["concurrency"].as_u64().unwrap_or(10).max(1),
        ),
        (
            url_indices,
            AtomicUsize::new(0),
            cfg["batch"]["url_concurrency"].as_u64().unwrap_or(5).max(1),
        ),
    ];
    let (sender, receiver) = mpsc::channel();
    let mut records = std::thread::scope(|workers| {
        for (indices, next, limit) in &groups {
            for _ in 0..(*limit).min(indices.len() as u64) {
                let sender = sender.clone();
                let tasks = &tasks;
                workers.spawn(move || {
                    loop {
                        let position = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&index) = indices.get(position) else {
                            break;
                        };
                        let task = &tasks[index];
                        let started = Instant::now();
                        let started_at = timestamp();
                        let record = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            convert_item(task, index, cfg, context, None).0
                        }))
                        .unwrap_or_else(|_| {
                            recorded(
                                task,
                                index,
                                started,
                                started_at,
                                &Err(markitai_core::Error::Conversion(
                                    "Conversion worker panicked".into(),
                                )
                                .into()),
                            )
                        });
                        if sender.send(record).is_err() {
                            break;
                        }
                    }
                });
            }
        }
        drop(sender);
        let mut status =
            progress::Progress::new(tasks.len(), progress::wanted(cli.quiet, cli.json));
        let mut records = Vec::new();
        while let Ok(record) = receiver.recv() {
            records.push(record);
            status.update(
                records.len(),
                &tasks[records[records.len() - 1].index].display,
            );
        }
        status.finish();
        records
    });
    let report_error = finish_report(report_plan, &records, clock, cli.verbose && !cli.quiet).err();
    if let Some(error) = &report_error {
        eprintln!("Error: {error}");
    }
    if let Some(plan) = destination.history {
        plan.record(&records);
    }
    crate::sort::by_key(&mut records, |record| record.index);
    let failed = records
        .iter()
        .filter(|record| record.status == ItemStatus::Failed)
        .count();
    if cli.json {
        let items: Vec<_> = records.iter().map(outcome).collect();
        emit_json(&items, report_error.as_deref());
    } else {
        print_item_diagnostics(&records, cli.quiet, cli.verbose);
        if !cli.quiet {
            print_batch_summary(
                &records,
                &[],
                report::Resumed::default(),
                cli.verbose,
                clock.elapsed(),
                destination.output,
            );
        }
    }
    Ok(if failed > 0 {
        10
    } else if report_error.is_some() {
        1
    } else {
        0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli() -> Cli {
        Cli::try_parse_from(["markitai", "input", "-o", "output", "--quiet"]).unwrap()
    }

    fn task(source: &Path, output: &Path) -> Task {
        let name = source.file_name().unwrap().to_string_lossy().into_owned();
        Task {
            source: source.to_string_lossy().into_owned(),
            display: name.clone(),
            report_key: name,
            output: Some(output.into()),
            filename: None,
            reserved_stem: None,
            source_file: None,
        }
    }

    #[test]
    fn resume_is_rejected_before_writing_and_ordinary_claim_is_absent() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("not-created");
        let mut task = task(&temp.path().join("missing.txt"), &output);
        let cfg = config::defaults();
        assert!(claim(&mut task, &cfg, None, None, &()).unwrap().is_none());
        let mut cli = cli();
        cli.resume = true;
        let error = run(
            &cli,
            &cfg,
            vec![task],
            BatchDestination {
                mode: RunMode::Directory,
                output: &output,
                history: None,
            },
            None,
            Instant::now(),
            ConvertContext::default(),
        )
        .unwrap_err();
        assert!(error.1.contains("--resume") && error.1.contains("not implemented"));
        assert!(!output.exists());
    }

    #[test]
    fn ordinary_partial_batch_writes_a_report_without_recovery_or_ownership_files() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input");
        let output = temp.path().join("output");
        std::fs::create_dir(&input).unwrap();
        let good = input.join("good.txt");
        let bad = input.join("broken.docx");
        std::fs::write(&good, "Portable batch content.\n").unwrap();
        std::fs::write(&bad, "This is not an Office package.").unwrap();
        let mut cfg = config::defaults();
        cfg["cache"]["enabled"] = json!(false);
        cfg["llm"]["enabled"] = json!(false);
        cfg["batch"]["concurrency"] = json!(2);
        let plan = report::plan(
            RunInfo {
                mode: RunMode::Directory,
                input: input.to_string_lossy().into_owned(),
                output_dir: output.clone(),
                started_at: timestamp(),
                log_file: None,
                options: ReportOptions::from_config(&cfg, None, &[]),
            },
            None,
            "rename",
            false,
        )
        .unwrap()
        .unwrap();
        let result = run(
            &cli(),
            &cfg,
            vec![task(&good, &output), task(&bad, &output)],
            BatchDestination {
                mode: RunMode::Directory,
                output: &output,
                history: None,
            },
            Some(&plan),
            Instant::now(),
            ConvertContext::default(),
        )
        .unwrap();
        assert_eq!(result, 10);
        assert!(output.join("good.txt.md").is_file());
        assert!(!output.join(".markitai/states").exists());
        assert!(!output.join(".markitai/ownership").exists());
        let reports: Vec<_> = std::fs::read_dir(output.join(".markitai/reports"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(reports.len(), 1);
        let report: Value = serde_json::from_slice(&std::fs::read(&reports[0]).unwrap()).unwrap();
        assert_eq!(report["summary"]["total_documents"], 2);
        assert_eq!(report["summary"]["completed_documents"], 1);
        assert_eq!(report["summary"]["failed_documents"], 1);
        assert_eq!(report["documents"]["broken.docx"]["status"], "failed");
        // A separate report failure must not erase an existing conversion-failure exit code.
        std::fs::remove_file(&reports[0]).unwrap();
        let report_directory = output.join(".markitai/reports");
        std::fs::remove_dir(&report_directory).unwrap();
        std::fs::write(&report_directory, b"preserved obstruction").unwrap();
        assert_eq!(
            run(
                &cli(),
                &cfg,
                vec![task(&good, &output), task(&bad, &output)],
                BatchDestination {
                    mode: RunMode::Directory,
                    output: &output,
                    history: None,
                },
                Some(&plan),
                Instant::now(),
                ConvertContext::default()
            )
            .unwrap(),
            10
        );
        assert_eq!(
            std::fs::read(&report_directory).unwrap(),
            b"preserved obstruction"
        );
    }
}
