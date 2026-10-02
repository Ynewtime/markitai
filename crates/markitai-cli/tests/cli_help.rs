use std::path::Path;
use std::process::{Command, Output};

fn invoke(root: &Path, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for name in [
        "HOME",
        "PATH",
        "SYSTEMROOT",
        "USERPROFILE",
        "TMPDIR",
        "TEMP",
        "TMP",
    ] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn removed_aliases_explain_each_migration_without_touching_inputs_or_json_stdout() {
    let root = tempfile::tempdir().unwrap();
    let aliases = [
        ("--playwright", "-s playwright"),
        ("--defuddle", "-s defuddle"),
        ("--static", "-s static"),
        ("--jina", "-s jina"),
        ("--cloudflare", "-s cloudflare"),
        ("--kreuzberg", "RTF converts natively now"),
    ];
    for (alias, replacement) in aliases {
        for args in [
            vec![alias.to_owned(), "missing.txt".into(), "--dry-run".into()],
            vec![
                "missing.txt".into(),
                "-o".into(),
                "out".into(),
                "--json".into(),
                alias.into(),
            ],
            vec!["missing.txt".into(), format!("{alias}=true")],
        ] {
            let words = args.iter().map(String::as_str).collect::<Vec<_>>();
            let output = invoke(root.path(), &words);
            assert_eq!(output.status.code(), Some(2));
            assert!(output.stdout.is_empty());
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(
                stderr.contains(&format!("{alias} has been removed")),
                "{stderr}"
            );
            assert!(stderr.contains(replacement), "{stderr}");
            assert!(!stderr.contains("Input does not exist"), "{stderr}");
            assert!(!root.path().join("out").exists());
            assert!(!root.path().join("home").exists());
        }
    }
}

#[test]
fn removed_spelling_as_a_literal_path_or_option_value_is_not_intercepted() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("--static"), b"literal name\n").unwrap();
    for args in [
        vec!["--dry-run", "--", "--static"],
        vec!["--dry-run", "--output=--jina", "--", "--static"],
        vec![
            "--config-json",
            "--cloudflare",
            "--dry-run",
            "--",
            "--static",
        ],
    ] {
        let output = invoke(root.path(), &args);
        assert!(!String::from_utf8_lossy(&output.stderr).contains("has been removed"));
        if args[0] == "--dry-run" {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        } else {
            assert_eq!(output.status.code(), Some(2));
        }
    }
    assert!(!root.path().join("--jina").exists());
}

#[test]
fn help_explains_output_privacy_cache_and_unsupported_workflows() {
    let root = tempfile::tempdir().unwrap();
    for args in [vec!["--help"], vec!["-h"], vec!["--quiet", "--no-llm"]] {
        let output = invoke(root.path(), &args);
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let help = String::from_utf8(output.stdout).unwrap();
        let normalized = help.split_whitespace().collect::<Vec<_>>().join(" ");
        for phrase in [
            "minimal, standard, rich",
            "requires -o",
            "Usage errors remain on stderr",
            "while still writing",
            "matching paths/options",
            "Vision locally",
            "With --llm",
            "without --llm",
            "implies --screenshot",
            "quiet by default",
            "log.dir",
            "OpenAI Batch API",
            "original -o directory",
            "does not cancel it",
            "with Workers AI in your Cloudflare account",
        ] {
            assert!(
                normalized.contains(phrase),
                "Missing {phrase}: {normalized}"
            );
        }
        for alias in [
            "--playwright",
            "--defuddle",
            "--static",
            "--jina",
            "--cloudflare",
            "--kreuzberg",
        ] {
            assert!(
                !help.contains(alias),
                "Removed spelling is advertised: {alias}"
            );
        }
    }
    assert!(!root.path().join("home").exists());
}

#[test]
fn unrepresentable_batch_wait_fails_before_output_or_provider_admission() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("input")).unwrap();
    std::fs::write(root.path().join("input/document.txt"), "Authored source").unwrap();
    let output = invoke(
        root.path(),
        &[
            "input",
            "-o",
            "out",
            "--llm",
            "--llm-batch",
            "--json",
            "--llm-batch-timeout",
            "18446744073709551615",
        ],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("supported clock range"));
    assert!(!root.path().join("out").exists());
}
