//! What a new user reads on the terminal: help, result lines, skips, empty
//! batches, unusable output locations and configuration hints.
use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

/// A valid 1x1 RGBA PNG (every chunk checksum correct).
const TINY_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 11, 73, 68, 65, 84, 120, 156, 99, 96, 0, 2, 0, 0, 5, 0, 1,
    122, 94, 171, 63, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

fn invoke(root: &Path, args: &[&str]) -> Output {
    invoke_env(root, args, &[])
}

/// A `/`-separated relative path as the CLI prints it: output locations are
/// built with `Path::join`, so Windows shows its own separator (`\`) there.
fn native(path: &str) -> String {
    path.replace('/', std::path::MAIN_SEPARATOR_STR)
}

fn invoke_env(root: &Path, args: &[&str], extra: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for name in ["HOME", "PATH", "SYSTEMROOT", "USERPROFILE", "TMPDIR"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        // Runtime fixtures are independent of commands installed on the host.
        .env("COPILOT_CLI_PATH", root.join("missing-copilot"))
        .env("CLAUDE_CLI_PATH", root.join("missing-claude"))
        .env("CODEX_CLI_PATH", root.join("missing-codex"))
        .envs(extra.iter().copied())
        .args(args)
        .output()
        .expect("CLI starts")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Permission checks are meaningless for a user that bypasses them.
#[cfg(unix)]
fn permissions_enforced(directory: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let probe = directory.join("permission-probe");
    std::fs::create_dir(&probe).unwrap();
    std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o555)).unwrap();
    let enforced = std::fs::write(probe.join("file"), b"x").is_err();
    std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::remove_dir_all(&probe).unwrap();
    enforced
}

#[test]
fn help_groups_options_and_explains_every_subcommand_argument() {
    let root = tempfile::tempdir().unwrap();
    let help = stdout(&invoke(root.path(), &["--help"]));
    for heading in [
        "Output:",
        "Configuration:",
        "LLM, OCR and screenshots:",
        "URL fetching and backends:",
        "Batch processing:",
        "Cache and images:",
        "Messages and logging:",
        "Presets (-p):",
        "Examples:",
        "Exit status:",
    ] {
        assert!(help.contains(heading), "missing {heading}: {help}");
    }
    // Off-switches are described with their flag, not listed twice, yet still parse.
    assert!(help.contains("(--no-llm disables)"), "{help}");
    assert!(!help.contains("      --no-llm\n"), "{help}");
    // Every remote strategy is implemented, and auto names its opt-in.
    let flat = help.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        flat.contains("defuddle, jina and cloudflare (your account) send the URL"),
        "{help}"
    );
    assert!(
        flat.contains("only after you opt in with fetch.remote_consent"),
        "{help}"
    );
    assert!(
        !flat.contains("not implemented") && !flat.contains("unsupported"),
        "{help}"
    );
    std::fs::write(root.path().join("note.txt"), "Text\n").unwrap();
    let parsed = invoke(root.path(), &["note.txt", "--llm", "--no-llm", "--pure"]);
    assert!(parsed.status.success(), "{}", stderr(&parsed));
    assert_eq!(stdout(&parsed), "Text\n");

    for (args, phrase) in [
        (vec!["config", "--help"], "Keys use dot notation"),
        (vec!["config", "set", "--help"], "Dot-notation key"),
        (
            vec!["config", "list", "--help"],
            "Show secret values instead",
        ),
        (
            vec!["config", "validate", "--help"],
            "defaults to the configuration",
        ),
        (vec!["cache", "--help"], "Entry counts and disk use"),
        (vec!["cache", "clear", "--help"], "Clear without asking"),
        (vec!["cache", "stats", "--help"], "Maximum entries listed"),
        (vec!["init", "--help"], "Do not prompt"),
        (
            vec!["doctor", "--help"],
            "Install the official Chrome headless shell",
        ),
        (vec!["serve", "--help"], "Port to listen on"),
    ] {
        let output = invoke(root.path(), &args);
        assert!(output.status.success(), "{args:?}");
        assert!(
            stdout(&output).contains(phrase),
            "{args:?}: {}",
            stdout(&output)
        );
    }
    let invalid = invoke(root.path(), &["note.txt", "-j", "0"]);
    assert_eq!(invalid.status.code(), Some(2));
    assert!(
        stderr(&invalid).contains("must be at least 1"),
        "{}",
        stderr(&invalid)
    );
    assert!(!root.path().join("home").exists());
}

#[test]
fn json_without_output_says_why_it_needs_one() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "Text\n").unwrap();
    for args in [vec!["note.txt", "--json"], vec!["--json"]] {
        let output = invoke(root.path(), &args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(output.stdout.is_empty());
        assert!(
            stderr(&output).contains("stdout carries the JSON result"),
            "{}",
            stderr(&output)
        );
    }
}

#[test]
fn single_inputs_name_the_written_file_and_explain_skips() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "Text\n").unwrap();
    let first = invoke(root.path(), &["note.txt", "-o", "out"]);
    assert!(first.status.success(), "{}", stderr(&first));
    assert_eq!(
        stderr(&first).trim(),
        format!("Wrote {}", native("out/note.txt.md"))
    );
    // The renamed result is the one the user has to know about.
    let second = invoke(root.path(), &["note.txt", "-o", "out"]);
    assert_eq!(
        stderr(&second).trim(),
        format!(
            "Wrote {} (note.txt.md already exists)",
            native("out/note.txt.v2.md")
        )
    );
    let quiet = invoke(root.path(), &["note.txt", "-o", "out", "-q"]);
    assert!(quiet.status.success());
    assert!(quiet.stderr.is_empty(), "{}", stderr(&quiet));
    let json = invoke(root.path(), &["note.txt", "-o", "out", "--json"]);
    assert!(json.stderr.is_empty(), "{}", stderr(&json));
    serde_json::from_slice::<Value>(&json.stdout).unwrap();
    let stdout_mode = invoke(root.path(), &["note.txt"]);
    assert!(stdout_mode.stderr.is_empty(), "{}", stderr(&stdout_mode));

    let skip = invoke(
        root.path(),
        &[
            "note.txt",
            "-o",
            "out",
            "--config-json",
            r#"{"output":{"on_conflict":"skip"}}"#,
        ],
    );
    assert!(skip.status.success());
    assert!(
        stderr(&skip).contains("Skipped note.txt: its output already exists"),
        "{}",
        stderr(&skip)
    );

    std::fs::write(root.path().join("scan.png"), TINY_PNG).unwrap();
    for args in [vec!["scan.png"], vec!["scan.png", "-o", "images"]] {
        let output = invoke(root.path(), &args);
        assert!(output.status.success(), "{args:?}");
        assert!(output.stdout.is_empty());
        assert!(
            stderr(&output).contains(
                "Skipped scan.png: an image has no text to extract without --ocr or --llm"
            ),
            "{}",
            stderr(&output)
        );
    }
    assert!(invoke(root.path(), &["scan.png", "-q"]).stderr.is_empty());
    assert!(!root.path().join("images").exists());
    // The name is the one the report lists, even when the input is spelled as
    // an absolute path.
    let absolute = root.path().join("scan.png");
    let named = invoke(root.path(), &[absolute.to_str().unwrap()]);
    assert!(
        stderr(&named).starts_with("Skipped scan.png: "),
        "{}",
        stderr(&named)
    );

    std::fs::write(root.path().join("doc.txt"), "Text\n").unwrap();
    let warned = invoke(root.path(), &["doc.txt", "--alt", "--desc"]);
    assert!(warned.status.success());
    assert!(
        stderr(&warned).contains("--alt and --desc have no effect without --llm"),
        "{}",
        stderr(&warned)
    );
}

#[test]
fn a_md_output_name_never_replaces_its_source_and_a_trailing_slash_names_a_directory() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("self.md"), "# Mine\n").unwrap();
    let overwrite = r#"{"output":{"on_conflict":"overwrite"}}"#;
    let refused = invoke(
        root.path(),
        &["self.md", "-o", "self.md", "--config-json", overwrite],
    );
    assert_eq!(refused.status.code(), Some(2));
    assert!(
        stderr(&refused).contains("Output self.md is the input file itself"),
        "{}",
        stderr(&refused)
    );
    assert_eq!(
        std::fs::read_to_string(root.path().join("self.md")).unwrap(),
        "# Mine\n"
    );
    // Another name is still overwritten as asked.
    std::fs::write(root.path().join("copy.md"), "old").unwrap();
    let other = invoke(
        root.path(),
        &["self.md", "-o", "copy.md", "--config-json", overwrite],
    );
    assert!(other.status.success(), "{}", stderr(&other));

    std::fs::write(root.path().join("note.txt"), "Text\n").unwrap();
    let slash = invoke(root.path(), &["note.txt", "-o", "dir.md/"]);
    assert!(slash.status.success(), "{}", stderr(&slash));
    assert!(root.path().join("dir.md").is_dir());
    assert!(root.path().join("dir.md/note.txt.md").is_file());
}

#[test]
fn unusable_output_locations_are_named_before_any_conversion_state() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "Text\n").unwrap();
    std::fs::write(root.path().join("taken"), "a regular file").unwrap();
    std::fs::create_dir(root.path().join("batch")).unwrap();
    std::fs::write(root.path().join("batch/a.txt"), "A\n").unwrap();
    for input in ["note.txt", "batch"] {
        let output = invoke(root.path(), &[input, "-o", "taken"]);
        assert_eq!(output.status.code(), Some(1), "{input}");
        let message = stderr(&output);
        assert!(
            message.contains("taken exists and is not a directory"),
            "{input}: {message}"
        );
        assert!(
            !message.contains("claim") && !message.contains("State I/O"),
            "{message}"
        );
        let nested = invoke(root.path(), &[input, "-o", "taken/below"]);
        assert!(
            stderr(&nested)
                .contains("Cannot create output directory taken/below: taken is not a directory"),
            "{}",
            stderr(&nested)
        );
    }
    let json = invoke(root.path(), &["note.txt", "-o", "taken", "--json"]);
    assert_eq!(json.status.code(), Some(1));
    let text = stdout(&json);
    assert!(!text.contains("-0.0"), "{text}");
    let body: Value = serde_json::from_str(&text).unwrap();
    assert!(body["error"].as_str().unwrap().contains("not a directory"));
    assert_eq!(body["totals"]["duration_s"], json!(0.0));
    // A preview reports the problem but still lists its targets.
    let preview = invoke(root.path(), &["batch", "-o", "taken", "--dry-run"]);
    assert!(preview.status.success());
    assert!(stderr(&preview).contains("Warning: Output path taken exists"));
    assert!(stdout(&preview).contains("a.txt -> taken"));
    assert_eq!(
        std::fs::read_to_string(root.path().join("taken")).unwrap(),
        "a regular file"
    );

    #[cfg(unix)]
    if permissions_enforced(root.path()) {
        use std::os::unix::fs::PermissionsExt;
        let locked = root.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();
        for input in ["note.txt", "batch"] {
            let output = invoke(root.path(), &[input, "-o", "locked/out"]);
            assert_eq!(output.status.code(), Some(1));
            assert!(
                stderr(&output)
                    .contains("Cannot create output directory locked/out: locked is not writable"),
                "{}",
                stderr(&output)
            );
        }
        let output = invoke(root.path(), &["note.txt", "-o", "locked"]);
        assert!(stderr(&output).contains("Output directory locked is not writable"));
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
}

#[test]
fn missing_or_unreadable_single_inputs_create_no_output_directory() {
    let root = tempfile::tempdir().unwrap();
    let missing = invoke(root.path(), &["missing.txt", "-o", "out"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(stderr(&missing).contains("Input file not found: missing.txt"));
    let json = invoke(root.path(), &["missing.txt", "-o", "out", "--json"]);
    let body: Value = serde_json::from_slice(&json.stdout).unwrap();
    assert_eq!(body["totals"]["failed"], 1);
    assert!(!root.path().join("out").exists());

    #[cfg(unix)]
    if permissions_enforced(root.path()) {
        use std::os::unix::fs::PermissionsExt;
        let locked = root.path().join("locked.txt");
        std::fs::write(&locked, "secret-free text\n").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let output = invoke(root.path(), &["locked.txt", "-o", "out"]);
        assert_eq!(output.status.code(), Some(1));
        assert!(
            stderr(&output).contains("Cannot read locked.txt: Permission denied"),
            "{}",
            stderr(&output)
        );
        let json = invoke(root.path(), &["locked.txt", "-o", "out", "--json"]);
        let body: Value = serde_json::from_slice(&json.stdout).unwrap();
        assert_eq!(body["items"][0]["status"], "failed");
        assert!(
            body["items"][0]["error"]
                .as_str()
                .unwrap()
                .starts_with("Cannot read locked.txt")
        );
        assert!(!root.path().join("out").exists());
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
}

#[test]
fn empty_batches_and_previews_say_what_would_happen() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("empty")).unwrap();
    std::fs::create_dir(root.path().join("other")).unwrap();
    std::fs::write(root.path().join("other/data.xyz"), "binary").unwrap();
    for input in ["empty", "other"] {
        let output = invoke(root.path(), &[input, "-o", "out"]);
        assert!(output.status.success(), "{input}");
        assert!(
            stderr(&output).contains(&format!(
                "No supported files or .urls lists found in {input}; nothing to convert."
            )),
            "{}",
            stderr(&output)
        );
        assert!(
            invoke(root.path(), &[input, "-o", "out", "-q"])
                .stderr
                .is_empty()
        );
    }
    std::fs::create_dir(root.path().join("docs")).unwrap();
    std::fs::write(root.path().join("docs/a.txt"), "A\n").unwrap();
    std::fs::write(root.path().join("docs/b.md"), "B\n").unwrap();
    let filtered = invoke(root.path(), &["docs", "-o", "out", "-g", "*.pdf"]);
    assert!(
        stderr(&filtered).contains("match --glob in docs"),
        "{}",
        stderr(&filtered)
    );
    let preview = invoke(root.path(), &["docs", "-o", "out", "--dry-run"]);
    assert!(preview.status.success());
    assert_eq!(
        stdout(&preview),
        format!(
            "a.txt -> {}\nb.md -> {}\n",
            native("out/a.txt.md"),
            native("out/b.md.md")
        )
    );
    assert!(
        stderr(&preview).contains("Dry run: 2 files would be converted; nothing was written."),
        "{}",
        stderr(&preview)
    );
    assert!(!root.path().join("out").exists());
}

#[test]
fn batch_summary_names_failures_and_where_results_went() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("docs")).unwrap();
    std::fs::write(root.path().join("docs/good.txt"), "Good\n").unwrap();
    std::fs::write(root.path().join("docs/broken.docx"), "not a zip archive").unwrap();
    let output = invoke(root.path(), &["docs", "-o", "out"]);
    assert_eq!(output.status.code(), Some(10));
    let message = stderr(&output);
    assert!(message.contains("Error: broken.docx: "), "{message}");
    assert!(message.contains("Done: 1/2 files ("), "{message}");
    assert!(
        message.contains("Failed 1 item: broken.docx. See the errors above."),
        "{message}"
    );
    assert!(message.trim_end().ends_with("Output: out"), "{message}");

    // Every item failing for want of a model gets one remedy line.
    let output = invoke(root.path(), &["docs", "-o", "llm-out", "--llm"]);
    assert_eq!(output.status.code(), Some(10));
    let message = stderr(&output);
    assert_eq!(
        message.matches("No LLM model is configured").count(),
        1,
        "{message}"
    );
    let single = invoke(root.path(), &["docs/good.txt", "--llm"]);
    assert_eq!(single.status.code(), Some(1));
    assert!(stderr(&single).contains("Hint: set a provider API key such as OPENAI_API_KEY"));
}

#[test]
fn url_list_warnings_locate_rejected_entries_without_echoing_them() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("links.urls"),
        "\u{feff}# comment\n\nftp://user:entry-secret@example.com/file\n",
    )
    .unwrap();
    let output = invoke(root.path(), &["links.urls", "-o", "out"]);
    assert_eq!(output.status.code(), Some(1));
    let message = stderr(&output);
    assert!(
        message.contains("skipping line 3 in links.urls: not an HTTP(S) URL"),
        "{message}"
    );
    assert!(!message.contains("entry-secret"), "{message}");
    assert!(message.contains("No valid URLs found in links.urls."));

    std::fs::write(
        root.path().join("list.urls"),
        r#"["mailto:entry-secret@example.com", {"name": "no url"}]"#,
    )
    .unwrap();
    let message = stderr(&invoke(root.path(), &["list.urls", "-o", "out"]));
    assert!(
        message.contains("skipping entry 1 in list.urls"),
        "{message}"
    );
    assert!(
        message.contains("skipping entry 2 in list.urls: expected a URL string"),
        "{message}"
    );
    assert!(!message.contains("entry-secret"), "{message}");

    std::fs::write(root.path().join("bad.urls"), "[not json").unwrap();
    let message = stderr(&invoke(root.path(), &["bad.urls", "-o", "out"]));
    assert!(
        message.contains("Cannot parse bad.urls as a JSON URL list"),
        "{message}"
    );
}

#[test]
fn configuration_mistakes_point_at_the_available_choices() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "Text\n").unwrap();
    std::fs::write(
        root.path().join("markitai.json"),
        json!({"presets":{"team":{"llm":false}}}).to_string(),
    )
    .unwrap();
    let preset = invoke(root.path(), &["note.txt", "-p", "bogus"]);
    assert_eq!(preset.status.code(), Some(1));
    assert!(
        stderr(&preset)
            .contains("Unknown preset 'bogus'. Available: minimal, rich, standard, team"),
        "{}",
        stderr(&preset)
    );
    let key = invoke(root.path(), &["config", "get", "llm.nope"]);
    assert_eq!(key.status.code(), Some(1));
    assert!(stderr(&key).contains("markitai config list -f table"));

    // Unset model fields are omitted from the echo, not shown as redacted secrets.
    let set = invoke(
        root.path(),
        &[
            "-c",
            "fresh.json",
            "config",
            "set",
            "llm.model_list",
            r#"[{"model_name":"m","litellm_params":{"model":"openai/x","api_key":"echo-secret"}}]"#,
        ],
    );
    assert!(set.status.success(), "{}", stderr(&set));
    let echo = stdout(&set);
    assert!(echo.contains(r#""api_key":"[REDACTED]""#), "{echo}");
    assert!(
        !echo.contains("api_base") && !echo.contains("echo-secret"),
        "{echo}"
    );

    std::fs::remove_file(root.path().join("markitai.json")).unwrap();
    let path = invoke(root.path(), &["config", "path"]);
    assert!(stdout(&path).contains("Create one with `markitai init`"));
    let init = invoke(root.path(), &["init", "-y", "--local"]);
    assert!(init.status.success());
    assert!(stdout(&init).starts_with("Configuration created: "));
    assert_eq!(stdout(&init).lines().count(), 1);
    assert!(stderr(&init).contains("Next steps:"), "{}", stderr(&init));
    assert!(
        stderr(&init).contains("No API model was detected"),
        "{}",
        stderr(&init)
    );
}

#[test]
fn cache_statistics_use_readable_counts_and_sizes() {
    let root = tempfile::tempdir().unwrap();
    let stats = invoke(root.path(), &["cache", "stats"]);
    assert!(stats.status.success());
    let text = stdout(&stats);
    assert!(text.contains("LLM cache: 0 entries (0 B)"), "{text}");
    assert!(text.contains("URL fetch cache: 0 entries (0 B)"), "{text}");
}
