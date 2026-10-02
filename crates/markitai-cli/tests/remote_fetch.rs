//! The Cloudflare backend and remote strategies from the command line. Every
//! case ends before a request leaves the machine: credentials are removed from
//! the child's environment, `MARKITAI_HOME` and the working directory are
//! empty temporary directories (no `.env`), and URLs are loopback addresses.
use std::path::Path;
use std::process::{Command, Output};

fn invoke(root: &Path, args: &[&str], env: &[(&str, &str)]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_markitai"))
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .env("MARKITAI_LANG", "en")
        .env_remove("MARKITAI_CONFIG")
        .env_remove("MARKITAI_NO_REMOTE_FETCH")
        .env_remove("CLOUDFLARE_API_TOKEN")
        .env_remove("CLOUDFLARE_ACCOUNT_ID")
        .env_remove("JINA_API_KEY")
        .env_remove("MODEL")
        .env("NO_PROXY", "127.0.0.1,localhost")
        .envs(env.iter().copied())
        .args(args)
        .output()
        .expect("CLI starts")
}

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("table.csv"), "name,count\nalpha,1\n").unwrap();
    std::fs::write(root.path().join("note.txt"), "Plain note\n").unwrap();
    root
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn the_cloudflare_backend_needs_credentials_and_says_how_to_set_them() {
    let root = fixture();
    let output = invoke(root.path(), &["table.csv", "-b", "cloudflare"], &[]);
    assert!(!output.status.success());
    let message = stderr(&output);
    assert!(
        message.contains("CLOUDFLARE_API_TOKEN") && message.contains("fetch.cloudflare.api_token"),
        "{message}"
    );
    assert!(!message.contains("not implemented"), "{message}");
    assert!(output.stdout.is_empty());
    // A missing env: reference names the variable, not a value.
    let output = invoke(
        root.path(),
        &[
            "table.csv",
            "-b",
            "cloudflare",
            "--config-json",
            r#"{"fetch":{"cloudflare":{"api_token":"env:NOT_SET_TOKEN","account_id":"abc123"}}}"#,
        ],
        &[],
    );
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("Environment variable not found: NOT_SET_TOKEN"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn the_cloudflare_backend_respects_the_remote_opt_outs_and_leaves_other_formats_native() {
    let root = fixture();
    for (args, env) in [
        (
            vec!["table.csv", "-b", "cloudflare", "--no-remote-fetch"],
            vec![],
        ),
        (
            vec!["table.csv", "-b", "cloudflare"],
            vec![("MARKITAI_NO_REMOTE_FETCH", "1")],
        ),
    ] {
        let output = invoke(root.path(), &args, &env);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            stderr(&output).contains("--no-remote-fetch"),
            "{}",
            stderr(&output)
        );
    }
    // Workers AI does not read plain text: the native reader converts it.
    let output = invoke(root.path(), &["note.txt", "-b", "cloudflare"], &[]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.ends_with("\n\nPlain note\n"), "{text}");
    assert!(!text.contains("cloudflare"), "{text}");
    // `-b native` keeps the native reader for a format Cloudflare reads.
    let output = invoke(root.path(), &["table.csv", "-b", "native"], &[]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(String::from_utf8_lossy(&output.stdout).contains("alpha"));
}

#[test]
fn remote_strategies_refuse_local_urls_and_missing_credentials_before_any_request() {
    let root = fixture();
    let output = invoke(
        root.path(),
        &["http://127.0.0.1:9/page", "-s", "cloudflare"],
        &[],
    );
    assert!(!output.status.success());
    let message = stderr(&output);
    assert!(message.contains("CLOUDFLARE_ACCOUNT_ID"), "{message}");
    assert!(!message.contains("not implemented"), "{message}");
    for strategy in ["jina", "defuddle"] {
        let output = invoke(
            root.path(),
            &["http://127.0.0.1:9/page", "-s", strategy],
            &[],
        );
        assert!(!output.status.success());
        assert!(
            stderr(&output).contains("Local URLs cannot be sent to remote extraction services"),
            "{}",
            stderr(&output)
        );
    }
    let output = invoke(
        root.path(),
        &["http://127.0.0.1:9/page", "-s", "jina", "--no-remote-fetch"],
        &[],
    );
    assert!(
        stderr(&output).contains("Remote fetching is disabled by policy"),
        "{}",
        stderr(&output)
    );
}
