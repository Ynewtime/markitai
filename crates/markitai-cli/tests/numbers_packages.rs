#![cfg(unix)]
#[path = "support/console.rs"]
mod console;

use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

// The independent upstream ZIP fixture is expanded without changing its entry
// bytes. This checks container adaptation, not an independently exported package.
const NUMBERS: &[u8] =
    include_bytes!("../../markitai-core/src/formats/numbers/fixtures/test-1.numbers");

fn unpack(path: &Path) {
    std::fs::create_dir_all(path).unwrap();
    let mut zip = zip::ZipArchive::new(Cursor::new(NUMBERS)).unwrap();
    for index in 0..zip.len() {
        let mut entry = zip.by_index(index).unwrap();
        let target = path.join(entry.enclosed_name().unwrap());
        if entry.is_dir() {
            std::fs::create_dir_all(target).unwrap();
            continue;
        }
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        std::fs::write(target, bytes).unwrap();
    }
}
fn setup(root: &Path) -> Value {
    std::fs::create_dir(root.join("tmp")).unwrap();
    let cfg = json!({
        "llm":{"enabled":false},"ocr":{"enabled":false},"screenshot":{"enabled":false},
        "image":{"alt_enabled":false,"desc_enabled":false},"history":{"record":true},
        "cache":{"enabled":false},"prompts":{"dir":root.join("prompts")},"log":{"dir":null},
        "fetch":{"strategy":"static","remote_consent":"never","timeout":2},
        "batch":{"concurrency":1,"scan_max_depth":5,"scan_max_files":20},
        "output":{"report":true,"on_conflict":"overwrite"}
    });
    save(root, &cfg);
    cfg
}
fn save(root: &Path, cfg: &Value) {
    std::fs::write(root.join("markitai.json"), serde_json::to_vec(cfg).unwrap()).unwrap();
}
fn invoke(root: &Path, args: &[&str]) -> Output {
    let stdout = tempfile::NamedTempFile::new_in(root).unwrap();
    let stderr = tempfile::NamedTempFile::new_in(root).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_markitai"));
    console::english(cmd.env_clear());
    for key in [
        "HOME",
        "PATH",
        "SYSTEMROOT",
        "USERPROFILE",
        "LANG",
        "LC_ALL",
        "TZ",
    ] {
        if let Some(value) = std::env::var_os(key) {
            cmd.env(key, value);
        }
    }
    cmd.current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .env("TMPDIR", root.join("tmp"))
        .env("NO_PROXY", "127.0.0.1,localhost")
        .args(["--config", "markitai.json"])
        .args(args)
        .stdin(Stdio::null())
        .stdout(stdout.reopen().unwrap())
        .stderr(stderr.reopen().unwrap());
    let mut child = cmd.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "CLI timeout: {}",
                std::fs::read_to_string(stderr.path()).unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    Output {
        status,
        stdout: std::fs::read(stdout.path()).unwrap(),
        stderr: std::fs::read(stderr.path()).unwrap(),
    }
}
fn status(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
fn envelope(output: Output, code: i32) -> Value {
    status(&output, code);
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["version"], "1.0");
    value
}
fn read(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn one(root: &Path, suffix: &str) -> PathBuf {
    let matches: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(suffix)
        })
        .collect();
    assert_eq!(matches.len(), 1, "{matches:?}");
    matches[0].clone()
}
fn report(root: &Path, out: &str) -> Value {
    read(&one(
        &root.join(out).join(".markitai/reports"),
        ".report.json",
    ))
}
fn state_path(root: &Path) -> PathBuf {
    one(&root.join("out/.markitai/states"), ".state.json")
}
fn inventory(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    if !root.exists() {
        return files;
    }
    for entry in walkdir::WalkDir::new(root) {
        let entry = entry.unwrap();
        if entry.file_type().is_file() {
            files.insert(
                entry.path().strip_prefix(root).unwrap().to_owned(),
                std::fs::read(entry.path()).unwrap(),
            );
        }
    }
    files
}
fn jobs(root: &Path) -> Vec<(PathBuf, Value)> {
    let root = root.join("home/serve/jobs");
    if !root.exists() {
        return vec![];
    }
    let mut result = Vec::new();
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.file_name().unwrap() == ".publish.lock" {
            assert!(path.is_file());
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(path.is_dir() && name.len() == 12 && name.bytes().all(|b| b.is_ascii_hexdigit()));
        result.push((path.clone(), read(&path.join("meta.json"))));
    }
    result
}
fn assert_tables(markdown: &str) {
    let headings = [
        "# ZZZ\\_Sheet\\_1",
        "## ZZZ\\_Table\\_1",
        "## ZZZ\\_Table\\_2",
        "# ZZZ\\_Sheet\\_2",
        "## XXX\\_Table\\_1",
    ];
    let positions: Vec<_> = headings
        .iter()
        .map(|text| markdown.find(text).unwrap())
        .collect();
    assert!(positions.windows(2).all(|v| v[0] < v[1]));
    for text in ["YYY\\_ROW\\_4", "YYY\\_4\\_2", "ZZZ\\_3\\_3", "XXX\\_3\\_5"] {
        assert!(markdown.contains(text), "missing {text}");
    }
}
fn body(markdown: &str) -> &str {
    markdown
        .strip_prefix("---\n")
        .and_then(|text| text.split_once("\n---\n").map(|(_, body)| body))
        .unwrap_or(markdown)
}

#[test]
fn single_package_is_one_document_with_zip_parity_report_and_independent_history() {
    let root = tempfile::tempdir().unwrap();
    setup(root.path());
    let package = root.path().join("收入 表.NuMbErS");
    unpack(&package);
    let original = inventory(&package);
    std::fs::write(root.path().join("file.numbers"), NUMBERS).unwrap();
    let zip = invoke(root.path(), &["file.numbers"]);
    status(&zip, 0);
    let directory = invoke(root.path(), &["收入 表.NuMbErS"]);
    status(&directory, 0);
    assert_eq!(
        body(std::str::from_utf8(&directory.stdout).unwrap()),
        body(std::str::from_utf8(&zip.stdout).unwrap())
    );
    assert_tables(std::str::from_utf8(&directory.stdout).unwrap());
    let result = envelope(
        invoke(root.path(), &["收入 表.NuMbErS", "-o", "out", "--json"]),
        0,
    );
    assert_eq!(result["items"].as_array().unwrap().len(), 1);
    assert_eq!(result["items"][0]["status"], "completed");
    let output = root.path().join("out/收入 表.NuMbErS.md");
    let markdown = std::fs::read_to_string(&output).unwrap();
    assert_tables(&markdown);
    let saved = report(root.path(), "out");
    assert_eq!(
        saved["documents"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["收入 表.NuMbErS"]
    );
    assert!(saved.get("options").is_none() && saved.get("url_sources").is_none());
    assert!(!root.path().join("out/.markitai/states").exists());
    assert_eq!(inventory(&package), original);
    let history = jobs(root.path());
    assert_eq!(history.len(), 1);
    let (job, meta) = &history[0];
    assert_eq!(meta["items"].as_array().unwrap().len(), 1);
    assert_eq!(meta["items"][0]["name"], "收入 表.NuMbErS");
    let archived = job
        .join("out")
        .join(meta["items"][0]["output"].as_str().unwrap());
    assert_eq!(std::fs::read(&archived).unwrap(), markdown.as_bytes());
    std::fs::remove_dir_all(package).unwrap();
    std::fs::remove_file(output).unwrap();
    assert_tables(&std::fs::read_to_string(archived).unwrap());
}

#[test]
fn explicit_output_skip_dry_run_and_single_resume_keep_single_input_rules() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = setup(root.path());
    unpack(&root.path().join("book.numbers"));
    let preview = invoke(root.path(), &["book.numbers", "--dry-run"]);
    status(&preview, 0);
    assert_eq!(
        String::from_utf8(preview.stdout).unwrap().lines().count(),
        1
    );
    assert!(!root.path().join("out").exists());
    let resumed = invoke(
        root.path(),
        &["book.numbers", "-o", "out", "--resume", "--json"],
    );
    status(&resumed, 1);
    assert!(String::from_utf8_lossy(&resumed.stderr).contains("--resume for a single file or URL"));
    assert!(!root.path().join("out").exists());
    let result = envelope(
        invoke(
            root.path(),
            &["book.numbers", "-o", "out/custom.md", "--json"],
        ),
        0,
    );
    assert!(
        result["items"][0]["output"]
            .as_str()
            .unwrap()
            .ends_with("out/custom.md")
    );
    let old = std::fs::read(root.path().join("out/custom.md")).unwrap();
    cfg["output"]["on_conflict"] = json!("skip");
    save(root.path(), &cfg);
    let skipped = envelope(
        invoke(
            root.path(),
            &["book.numbers", "-o", "out/custom.md", "--json"],
        ),
        0,
    );
    assert_eq!(skipped["items"][0]["status"], "skipped");
    assert_eq!(
        std::fs::read(root.path().join("out/custom.md")).unwrap(),
        old
    );
}

#[test]
fn malformed_and_old_packages_fail_once_without_discovering_internal_documents() {
    for old in [false, true] {
        let root = tempfile::tempdir().unwrap();
        setup(root.path());
        let package = root.path().join("broken.numbers");
        std::fs::create_dir(&package).unwrap();
        std::fs::write(
            package.join("tempting.txt"),
            "Must not become an independent conversion.",
        )
        .unwrap();
        if old {
            std::fs::write(package.join("index.xml"), "<document>Old XML</document>").unwrap();
        }
        let result = envelope(
            invoke(root.path(), &["broken.numbers", "-o", "out", "--json"]),
            1,
        );
        assert_eq!(result["items"].as_array().unwrap().len(), 1);
        assert_eq!(result["items"][0]["status"], "failed");
        assert!(result["items"][0]["error"].is_string());
        assert!(!root.path().join("out/tempting.txt.md").exists());
        assert!(!root.path().join("out/.markitai/states").exists());
        let jobs = jobs(root.path());
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].1["items"][0]["name"], "broken.numbers");
        assert_eq!(jobs[0].1["items"][0]["status"], "error");
    }
}

#[test]
fn mixed_discovery_prunes_packages_before_globs_and_counts_each_as_one_input() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = setup(root.path());
    unpack(&root.path().join("input/book.numbers"));
    unpack(&root.path().join("input/sub/deeper.NUMBERS"));
    for name in [
        "input/a.txt",
        "input/book.numbers/tempting.txt",
        "input/sub/deeper.NUMBERS/tempting.txt",
    ] {
        std::fs::write(root.path().join(name), "Authored content.").unwrap();
    }
    std::fs::write(
        root.path().join("input/book.numbers/internal.urls"),
        "http://127.0.0.1:1/never",
    )
    .unwrap();
    let result = envelope(invoke(root.path(), &["input", "-o", "out", "--json"]), 0);
    let sources: Vec<_> = result["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["source"].as_str().unwrap())
        .collect();
    assert_eq!(sources, ["a.txt", "book.numbers", "sub/deeper.NUMBERS"]);
    let saved = read(&state_path(root.path()));
    assert_eq!(saved["documents"].as_object().unwrap().len(), 3);
    assert_eq!(saved["urls"], json!({}));
    assert_tables(&std::fs::read_to_string(root.path().join("out/sub/deeper.NUMBERS.md")).unwrap());
    let filtered = envelope(
        invoke(
            root.path(),
            &["input", "-o", "filtered", "--json", "--glob", "**/*.txt"],
        ),
        0,
    );
    assert_eq!(filtered["items"].as_array().unwrap().len(), 1);
    assert_eq!(filtered["items"][0]["source"], "a.txt");
    let excluded = envelope(
        invoke(
            root.path(),
            &[
                "input",
                "-o",
                "excluded",
                "--json",
                "--glob",
                "!**/*.numbers",
                "--glob",
                "!**/*.NUMBERS",
            ],
        ),
        0,
    );
    assert_eq!(excluded["items"].as_array().unwrap().len(), 1);
    assert_eq!(excluded["items"][0]["source"], "a.txt");
    let shallow = envelope(
        invoke(
            root.path(),
            &["input", "-o", "shallow", "--json", "--max-depth", "0"],
        ),
        0,
    );
    assert_eq!(shallow["items"].as_array().unwrap().len(), 2);
    assert!(
        shallow["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["source"] == "book.numbers")
    );
    cfg["batch"]["scan_max_files"] = json!(2);
    save(root.path(), &cfg);
    envelope(
        invoke(
            root.path(),
            &["input", "-o", "limit-ok", "--json", "--max-depth", "0"],
        ),
        0,
    );
    cfg["batch"]["scan_max_files"] = json!(1);
    save(root.path(), &cfg);
    let limited = invoke(
        root.path(),
        &["input", "-o", "limit-fail", "--json", "--max-depth", "0"],
    );
    status(&limited, 1);
    assert!(String::from_utf8_lossy(&limited.stderr).contains("scan_max_files"));
    assert!(!root.path().join("limit-fail").exists());
}

#[test]
fn resume_keeps_completed_packages_atomic_and_retries_only_the_repaired_package() {
    let root = tempfile::tempdir().unwrap();
    setup(root.path());
    unpack(&root.path().join("input/done.numbers"));
    let repaired = root.path().join("input/retry.numbers");
    std::fs::create_dir(&repaired).unwrap();
    std::fs::write(repaired.join("note.txt"), "Not a separate document.").unwrap();
    let first = envelope(invoke(root.path(), &["input", "-o", "out", "--json"]), 10);
    assert_eq!(first["items"].as_array().unwrap().len(), 2);
    assert_eq!(first["items"][0]["status"], "completed");
    assert_eq!(first["items"][1]["status"], "failed");
    let old_output = std::fs::read(root.path().join("out/done.numbers.md")).unwrap();
    let old_history = inventory(&root.path().join("home/serve/jobs"));
    // Removing the completed input makes any accidental re-conversion observable.
    std::fs::remove_dir_all(root.path().join("input/done.numbers")).unwrap();
    unpack(&repaired);
    let resumed = envelope(
        invoke(root.path(), &["input", "-o", "out", "--resume", "--json"]),
        0,
    );
    assert_eq!(resumed["items"].as_array().unwrap().len(), 1);
    assert_eq!(resumed["items"][0]["source"], "retry.numbers");
    assert_eq!(resumed["items"][0]["status"], "completed");
    assert_eq!(
        std::fs::read(root.path().join("out/done.numbers.md")).unwrap(),
        old_output
    );
    assert_tables(&std::fs::read_to_string(root.path().join("out/retry.numbers.md")).unwrap());
    let state = read(&state_path(root.path()));
    assert_eq!(state["documents"].as_object().unwrap().len(), 2);
    assert!(
        state["documents"]
            .as_object()
            .unwrap()
            .values()
            .all(|entry| entry["status"] == "completed")
    );
    let latest = report(root.path(), "out");
    assert_eq!(latest["documents"].as_object().unwrap().len(), 2);
    let new_history = inventory(&root.path().join("home/serve/jobs"));
    for (path, bytes) in &old_history {
        assert_eq!(new_history.get(path), Some(bytes));
    }
    let no_work = envelope(
        invoke(root.path(), &["input", "-o", "out", "--resume", "--json"]),
        0,
    );
    assert_eq!(no_work["items"], json!([]));
    assert_eq!(inventory(&root.path().join("home/serve/jobs")), new_history);
}

#[test]
fn old_saved_package_members_are_rejected_before_checkpoint_receipt_or_output_changes() {
    for native in [false, true] {
        for url_source in [false, true] {
            let root = tempfile::tempdir().unwrap();
            setup(root.path());
            std::fs::create_dir(root.path().join("input")).unwrap();
            std::fs::write(
                root.path().join("input/seed.txt"),
                "Saved ordinary document.",
            )
            .unwrap();
            envelope(invoke(root.path(), &["input", "-o", "out", "--json"]), 0);
            let state_path = state_path(root.path());
            let mut saved = read(&state_path);
            if !native {
                saved.as_object_mut().unwrap().remove("_markitai");
            }
            unpack(&root.path().join("input/book.numbers"));
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            if url_source {
                let url = format!("http://{}/must-not-fetch", listener.local_addr().unwrap());
                let source = root.path().join("input/book.numbers/internal.urls");
                std::fs::write(&source, format!("{url}\n")).unwrap();
                // Legacy provenance may be cwd-relative; native checkpoints use
                // absolute spellings. Both must hit the same package boundary.
                let recorded_source = if native {
                    source
                } else {
                    PathBuf::from("input/book.numbers/internal.urls")
                };
                saved["urls"][&url] = json!({"status":"failed","source_file":recorded_source,"error":"legacy package member"});
            } else {
                std::fs::write(
                    root.path().join("input/book.numbers/internal.txt"),
                    "Must not be scheduled.",
                )
                .unwrap();
                saved["documents"]["book.numbers/internal.txt"] =
                    json!({"status":"failed","error":"legacy package member"});
            }
            std::fs::write(&state_path, serde_json::to_vec_pretty(&saved).unwrap()).unwrap();
            // Valid empty journal bytes must not be compacted away on rejection.
            std::fs::write(state_path.with_extension("jsonl"), b"\n").unwrap();
            let before = inventory(&root.path().join("out"));
            let old_history = inventory(&root.path().join("home/serve/jobs"));
            let failed = invoke(root.path(), &["input", "-o", "out", "--resume", "--json"]);
            status(&failed, 1);
            assert!(
                String::from_utf8_lossy(&failed.stderr)
                    .contains("inside a Numbers directory package")
            );
            assert_eq!(inventory(&root.path().join("out")), before);
            assert_eq!(inventory(&root.path().join("home/serve/jobs")), old_history);
            assert_eq!(
                listener.accept().unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock
            );
        }
    }
}
