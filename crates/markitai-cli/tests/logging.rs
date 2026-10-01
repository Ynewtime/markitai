use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

fn invoke(root: &Path, cfg: Value, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for key in ["HOME", "PATH", "SYSTEMROOT", "USERPROFILE", "TMPDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .envs(env.iter().copied())
        .arg("--config-json")
        .arg(cfg.to_string())
        .args(args)
        .output()
        .unwrap()
}
fn records(path: &Path) -> Vec<Value> {
    let mut rows = Vec::new();
    for entry in std::fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "log")
        {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(
                    entry.metadata().unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
            rows.extend(
                std::fs::read_to_string(entry.path())
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str::<Value>(line).unwrap()),
            );
        }
    }
    rows
}

#[test]
fn file_logging_does_not_change_stdout_and_level_only_filters_file() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "# Hello\n世界\n").unwrap();
    let cfg =
        json!({"log":{"dir":"logs", "format":"json", "level":"ERROR"},"cache":{"enabled":false}});
    let output = invoke(
        root.path(),
        cfg,
        &["note.txt", "--pure", "--log-level", "DEBUG", "-q"],
        &[],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, "# Hello\n世界\n".as_bytes());
    assert!(output.stderr.is_empty());
    let rows = records(&root.path().join("logs"));
    assert!(rows.iter().any(|row| row["lvl"] == "DEBUG"));
    assert!(rows.iter().any(|row| row["msg"] == "Completed note.txt"));
    assert!(
        rows.iter()
            .any(|row| row["msg"] == "Run finished with exit code 0")
    );
    assert!(rows.iter().all(|row| row.as_object().unwrap().len() == 4));
    let output = invoke(
        root.path(),
        json!({"log":{"dir":"errors","format":"json"}}),
        &["note.txt", "--pure", "--log-level", "ERROR"],
        &[],
    );
    assert!(output.status.success());
    assert!(records(&root.path().join("errors")).is_empty());
    let output = invoke(
        root.path(),
        json!({}),
        &["note.txt", "--pure", "--log-level", "DEBUG"],
        &[],
    );
    assert!(output.status.success());
    assert!(!root.path().join("home").exists());
}

#[test]
fn the_log_level_is_accepted_in_any_case_and_still_filters_by_that_level() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "# Hello\n").unwrap();
    let cfg =
        json!({"log":{"dir":"logs","format":"json","level":"ERROR"},"cache":{"enabled":false}});
    for (index, spelling) in ["debug", "Debug", "DEBUG"].into_iter().enumerate() {
        let output = invoke(
            root.path(),
            cfg.clone(),
            &["note.txt", "--pure", "--log-level", spelling, "-q"],
            &[],
        );
        assert!(
            output.status.success(),
            "--log-level {spelling}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        // Each run writes its own file; every one of them carries DEBUG records.
        let rows = records(&root.path().join("logs"));
        let debug = rows.iter().filter(|row| row["lvl"] == "DEBUG").count();
        assert!(debug > index, "--log-level {spelling}: {rows:?}");
    }
    // The other levels follow the same rule (WARNING hides the DEBUG records of
    // its run), and a value that is no level fails.
    let before = records(&root.path().join("logs")).len();
    let output = invoke(
        root.path(),
        cfg.clone(),
        &["note.txt", "--pure", "--log-level", "warning", "-q"],
        &[],
    );
    assert!(output.status.success());
    let rows = records(&root.path().join("logs"));
    assert_eq!(rows.len(), before, "a clean run logs nothing at WARNING");
    let output = invoke(
        root.path(),
        cfg,
        &["note.txt", "--pure", "--log-level", "verbose"],
        &[],
    );
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid value 'verbose'"), "{stderr}");
    assert!(
        stderr.contains("DEBUG, INFO, WARNING, ERROR, CRITICAL"),
        "{stderr}"
    );
}

#[test]
fn parallel_json_results_and_log_rotation_keep_all_complete_records() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("input")).unwrap();
    for i in 0..40 {
        std::fs::write(
            root.path().join(format!("input/{i}.txt")),
            format!("body {i}\n"),
        )
        .unwrap();
    }
    let cfg = json!({"log":{"dir":"logs","format":"json","rotation":"1 KB"},"cache":{"enabled":false},"output":{"report":true}});
    let output = invoke(
        root.path(),
        cfg,
        &["input", "-o", "out", "--json", "-q", "-j", "8"],
        &[],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["totals"]["completed"], 40);
    let rows = records(&root.path().join("logs"));
    let completed = rows
        .iter()
        .filter(|row| row["msg"].as_str().unwrap().starts_with("Completed "))
        .count();
    assert_eq!(completed, 40);
    assert!(std::fs::read_dir(root.path().join("logs")).unwrap().count() > 1);
    let reports: Vec<_> = std::fs::read_dir(root.path().join("out/.markitai/reports"))
        .unwrap()
        .map(|p| p.unwrap().path())
        .collect();
    let report: Value = serde_json::from_slice(&std::fs::read(&reports[0]).unwrap()).unwrap();
    assert!(
        root.path()
            .join(report["log_file"].as_str().unwrap())
            .is_file()
    );
}

#[test]
fn file_log_env_precedence_redaction_and_failures_are_observable() {
    let root = tempfile::tempdir().unwrap();
    let cfg = json!({"log":{"dir":"ignored","format":"text"},"cache":{"enabled":false},"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"openai/test","api_key":"credential-that-must-not-leak"}}]}});
    let url = "http://user:password@127.0.0.1:1/private?custom=hidden-query#hidden-fragment";
    let output = invoke(
        root.path(),
        cfg,
        &[
            url,
            "-s",
            "static",
            "-o",
            "out",
            "--json",
            "--log-level",
            "ERROR",
        ],
        &[
            ("MARKITAI_LOG_DIR", "actual"),
            ("MARKITAI_LOG_FORMAT", "json"),
        ],
    );
    assert_eq!(output.status.code(), Some(1));
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["ok"], false);
    assert!(!root.path().join("ignored").exists());
    let rows = records(&root.path().join("actual"));
    assert!(!rows.is_empty());
    let logs = serde_json::to_string(&rows).unwrap();
    for secret in [
        "user:password",
        "hidden-query",
        "hidden-fragment",
        "credential-that-must-not-leak",
    ] {
        assert!(!logs.contains(secret), "{logs}");
    }
    assert!(rows.iter().all(|row| row["lvl"] == "ERROR"));
    std::fs::write(root.path().join("not-a-directory"), "preserve").unwrap();
    let output = invoke(
        root.path(),
        json!({"log":{"dir":"not-a-directory"}}),
        &["missing.txt", "-o", "out", "--json"],
        &[],
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap()["ok"],
        false
    );
    assert_eq!(
        std::fs::read_to_string(root.path().join("not-a-directory")).unwrap(),
        "preserve"
    );
}

#[test]
fn subcommands_do_not_enable_conversion_logs_or_change_their_stdout() {
    let root = tempfile::tempdir().unwrap();
    let output = invoke(
        root.path(),
        json!({"log":{"dir":"logs","format":"json"}}),
        &["--log-level", "DEBUG", "config", "get", "log.level"],
        &[],
    );
    assert!(output.status.success());
    assert_eq!(output.stdout, b"INFO\n");
    assert!(!root.path().join("logs").exists());
}
