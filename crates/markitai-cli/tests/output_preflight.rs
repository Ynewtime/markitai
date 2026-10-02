//! Read-only decisions must not leave write-claim markers behind.
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

struct Fixture(tempfile::TempDir);

impl Fixture {
    fn new() -> Self {
        let fixture = Self(tempfile::tempdir().unwrap());
        fs::create_dir(fixture.path("home")).unwrap();
        fs::create_dir(fixture.path("state")).unwrap();
        fs::write(fixture.path("state/config.json"), br#"{"history":{"record":false},"log":{"dir":null},"cache":{"enabled":false},"llm":{"enabled":false},"ocr":{"enabled":false},"screenshot":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false}}"#).unwrap();
        fixture
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.path().join(name)
    }
    fn run(&self, args: &[&str]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for name in ["PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .current_dir(self.0.path())
            .env("HOME", self.path("home"))
            .env("MARKITAI_HOME", self.path("state"))
            .args(args)
            .output()
            .unwrap()
    }
    fn set(&self, key: &str, value: &str) {
        let output = self.run(&["config", "set", key, value]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn good(output: &Output) {
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn occupied_skip_observes_the_result_without_creating_member_files_or_receipts() {
    let fixture = Fixture::new();
    fixture.set("output.on_conflict", "skip");
    fs::write(fixture.path("a.txt"), b"new text").unwrap();
    fs::create_dir(fixture.path("out")).unwrap();
    fs::write(fixture.path("out/a.txt.md"), b"old text").unwrap();
    let output = fixture.run(&["a.txt", "-o", "out", "--json"]);
    good(&output);
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["items"][0]["status"], "skipped");
    assert_eq!(fs::read(fixture.path("out/a.txt.md")).unwrap(), b"old text");
    assert!(!fixture.path("out/.markitai/ownership").exists());
}

#[test]
fn rename_does_not_claim_any_already_occupied_candidate() {
    let fixture = Fixture::new();
    fs::write(fixture.path("a.txt"), b"complete text").unwrap();
    fs::create_dir(fixture.path("out")).unwrap();
    for name in ["a.txt.md", "a.txt.v2.md", "a.txt.v3.llm.md"] {
        fs::write(fixture.path("out").join(name), b"old text").unwrap();
    }
    good(&fixture.run(&["a.txt", "-o", "out", "--json"]));
    assert!(fixture.path("out/.markitai/ownership/members").is_file());
    {
        let name = "names-v2";
        assert_eq!(
            fs::read_dir(fixture.path(&format!("out/.markitai/ownership/{name}")))
                .unwrap()
                .count(),
            0
        );
    }
    assert!(fixture.path("out/a.txt.v4.md").is_file());
}

#[cfg(unix)]
#[test]
fn user_directory_aliases_work_but_document_and_private_metadata_links_still_fail() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new();
    fs::create_dir(fixture.path("input")).unwrap();
    fs::create_dir(fixture.path("physical")).unwrap();
    fs::write(fixture.path("input/a.txt"), b"complete user document").unwrap();
    symlink("input", fixture.path("in-alias")).unwrap();
    symlink("physical", fixture.path("out-alias")).unwrap();
    good(&fixture.run(&["in-alias/a.txt", "-o", "out-alias", "--json"]));
    assert!(fixture.path("physical/a.txt.md").is_file());
    symlink("input/a.txt", fixture.path("leaf.txt")).unwrap();
    let output = fixture.run(&["leaf.txt", "-o", "leaf-out", "--json"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("document leaf is a link"));
    fs::create_dir(fixture.path("blocked")).unwrap();
    fs::create_dir(fixture.path("external")).unwrap();
    symlink(fixture.path("external"), fixture.path("blocked/.markitai")).unwrap();
    let output = fixture.run(&["input/a.txt", "-o", "blocked", "--json"]);
    assert!(!output.status.success());
    assert_eq!(fs::read_dir(fixture.path("external")).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn batch_unreadable_input_is_rejected_before_any_member_claim() {
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new();
    fs::create_dir(fixture.path("in")).unwrap();
    let source = fixture.path("in/blocked.txt");
    fs::write(&source, b"private fixture text").unwrap();
    fs::set_permissions(&source, fs::Permissions::from_mode(0o000)).unwrap();
    if fs::File::open(&source).is_ok() {
        fs::set_permissions(&source, fs::Permissions::from_mode(0o600)).unwrap();
        return; // An elevated runner can read mode-000 input.
    }
    let output = fixture.run(&["in", "-o", "out", "--json"]);
    fs::set_permissions(&source, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Cannot read"));
    assert!(!fixture.path("out/.markitai/ownership").exists());
}
