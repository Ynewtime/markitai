use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

// These tests exercise the installed CLI boundary: saved reports and stdout are
// different public JSON contracts. Every subprocess receives a private home and
// explicit configuration, with no inherited provider or user configuration.
fn invoke(root: &Path, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for name in ["PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .current_dir(root)
        .env("MARKITAI_HOME", root.join("home"))
        .env("NO_PROXY", "127.0.0.1,localhost")
        .stdin(Stdio::null())
        .args(["--config", "markitai.json"])
        .args(args)
        .output()
        .unwrap()
}

fn envelope(output: Output, exit: i32) -> Value {
    assert_eq!(
        output.status.code(),
        Some(exit),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // from_slice rejects trailing report paths, progress text or a second JSON
    // object, including when report publication fails after conversion succeeds.
    let value: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "not one JSON envelope: {error}; stdout={}",
            String::from_utf8_lossy(&output.stdout)
        )
    });
    assert_eq!(value["version"], "1.0");
    assert!(value["items"].is_array());
    value
}

fn configure(root: &Path) -> Value {
    let cfg = json!({
        "llm":{"enabled":false}, "ocr":{"enabled":false},
        "screenshot":{"enabled":false},
        "image":{"alt_enabled":false,"desc_enabled":false},
        "cache":{"enabled":false}, "history":{"record":false},
        "log":{"dir":null},
        "batch":{"concurrency":2,"url_concurrency":2,"scan_max_depth":8},
        "output":{"on_conflict":"overwrite"}
    });
    save(root, &cfg);
    cfg
}
fn save(root: &Path, cfg: &Value) {
    std::fs::write(root.join("markitai.json"), cfg.to_string()).unwrap();
}
fn input(root: &Path, relative: &str, content: &str) {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, content).unwrap();
}
fn report_paths(output_dir: &Path) -> Vec<PathBuf> {
    let directory = output_dir.join(".markitai/reports");
    if !directory.exists() {
        return Vec::new();
    }
    let mut paths: Vec<_> = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.sort();
    for path in &paths {
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with("markitai.") && name.ends_with(".report.json"),
            "unexpected publication residue: {}",
            path.display()
        );
        let hash = name.split('.').nth(1).unwrap();
        assert_eq!(hash.len(), 6);
        assert!(hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
    paths
}
fn report(output_dir: &Path) -> (String, Value) {
    let paths = report_paths(output_dir);
    assert_eq!(paths.len(), 1, "reports={paths:?}");
    let raw = std::fs::read_to_string(&paths[0]).unwrap();
    let value: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(value["version"], "1.0");
    timestamp(&value["generated_at"]);
    assert!(value["log_file"].is_null());
    duration(&value["summary"]["duration"]);
    (raw, value)
}
fn keys(value: &Value, expected: &[&str]) {
    let actual: BTreeSet<_> = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let expected: BTreeSet<_> = expected.iter().copied().collect();
    assert_eq!(actual, expected, "unexpected report shape: {value}");
}
fn ordered(raw: &str, names: &[&str]) {
    let positions: Vec<_> = names
        .iter()
        .map(|name| {
            raw.find(&serde_json::to_string(name).unwrap())
                .unwrap_or_else(|| panic!("missing ordered field {name}"))
        })
        .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "field order differs: {names:?} at {positions:?}"
    );
}
fn ordered_top(raw: &str, names: &[&str]) {
    struct Fields(Vec<String>);
    impl<'de> serde::Deserialize<'de> for Fields {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = Fields;
                fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    formatter.write_str("a report object")
                }
                fn visit_map<M: serde::de::MapAccess<'de>>(
                    self,
                    mut map: M,
                ) -> Result<Fields, M::Error> {
                    let mut fields = Vec::new();
                    while let Some(key) = map.next_key::<String>()? {
                        fields.push(key);
                        map.next_value::<serde::de::IgnoredAny>()?;
                    }
                    Ok(Fields(fields))
                }
            }
            deserializer.deserialize_map(Visitor)
        }
    }
    let fields: Fields = serde_json::from_str(raw).unwrap();
    assert_eq!(
        fields.0,
        names
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>()
    );
}
fn timestamp(value: &Value) {
    chrono::DateTime::parse_from_rfc3339(value.as_str().expect("timestamp must be a string"))
        .unwrap();
}
fn duration(value: &Value) {
    let text = value.as_str().expect("saved duration must be a string");
    if let Some(seconds) = text.strip_suffix('s') {
        let value: f64 = seconds.parse().unwrap();
        assert!(value.is_finite() && value >= 0.0);
        assert_eq!(seconds.split('.').nth(1).unwrap().len(), 1);
    } else {
        let parts: Vec<_> = text.split(':').collect();
        assert!((2..=3).contains(&parts.len()));
        for part in &parts {
            assert!(part.len() >= 2 && part.bytes().all(|byte| byte.is_ascii_digit()));
        }
        assert!(parts.last().unwrap().parse::<u64>().unwrap() < 60);
    }
}
fn zero_usage(value: &Value) {
    keys(
        value,
        &[
            "models",
            "requests",
            "input_tokens",
            "output_tokens",
            "cost_usd",
        ],
    );
    assert_eq!(
        value,
        &json!({"models":{},"requests":0,"input_tokens":0,"output_tokens":0,"cost_usd":0.0})
    );
}
/// A `/`-separated relative path with the platform's separator (`\` on
/// Windows), as a path the CLI built with `Path::join` is spelled.
fn native(path: &str) -> String {
    path.replace('/', std::path::MAIN_SEPARATOR_STR)
}
fn recorded_output(root: &Path, entry: &Value) -> PathBuf {
    let path = PathBuf::from(
        entry["output"]
            .as_str()
            .expect("successful item needs output path"),
    );
    let path = if path.is_absolute() {
        path
    } else {
        root.join(path)
    };
    assert!(
        path.is_file(),
        "report points to missing output: {}",
        path.display()
    );
    path
}
fn mode_inputs(root: &Path, server: &Server) -> [String; 4] {
    input(root, "note.txt", "A report fixture.\n");
    input(root, "input/note.txt", "A directory report fixture.\n");
    let url = format!("{}/page", server.base);
    input(root, "sources.urls", &format!("{url} custom\n"));
    [
        "note.txt".into(),
        "input".into(),
        url,
        "sources.urls".into(),
    ]
}

struct Server {
    base: String,
    gets: Arc<AtomicUsize>,
    posts: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Server {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let gets = Arc::new(AtomicUsize::new(0));
        let posts = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (get_count, post_count, stopped) = (gets.clone(), posts.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !stopped.load(Ordering::SeqCst) {
                let (mut stream, _) = match listener.accept() {
                    Ok(value) => value,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("mock accept failed: {error}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut bytes = Vec::new();
                let deadline = Instant::now() + Duration::from_secs(5);
                loop {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    assert!(!remaining.is_zero(), "mock request deadline exceeded");
                    stream.set_read_timeout(Some(remaining)).unwrap();
                    let mut chunk = [0u8; 4096];
                    let count = match stream.read(&mut chunk) {
                        Ok(count) => count,
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(error) => panic!("mock read: {error}"),
                    };
                    assert!(count > 0, "request ended before headers/body completed");
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(split) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&bytes[..split]);
                        let length = headers
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= split + 4 + length {
                            break;
                        }
                    }
                    assert!(bytes.len() < 1024 * 1024);
                }
                let request = String::from_utf8(bytes).unwrap();
                let (status, mime, body) = if request.starts_with("POST ") {
                    post_count.fetch_add(1, Ordering::SeqCst);
                    ("200 OK", "application/json", json!({"choices":[{"message":{"content":model_content(&serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap(),"# Enhanced\n\nLocal fixture answer.")},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":5}}).to_string())
                } else {
                    get_count.fetch_add(1, Ordering::SeqCst);
                    if request.starts_with("GET /missing") {
                        (
                            "404 Not Found",
                            "text/plain",
                            "Fixture page is missing".into(),
                        )
                    } else {
                        ("200 OK", "text/html", "<html><title>Local fixture</title><article><h1>Local fixture</h1><p>Report content from localhost. 世界</p></article></html>".into())
                    }
                };
                write!(stream, "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        Self {
            base,
            gets,
            posts,
            stop,
            thread: Some(thread),
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let joined = self.thread.take().unwrap().join();
        if !std::thread::panicking() {
            joined.expect("mock server failed");
        }
    }
}

#[test]
fn all_four_modes_obey_omitted_null_false_and_true_report_selection() {
    let server = Server::start();
    for selection in [
        None,
        Some(Value::Null),
        Some(json!(false)),
        Some(json!(true)),
    ] {
        for mode in 0..4 {
            let root = tempfile::tempdir().unwrap();
            let mut cfg = configure(root.path());
            if let Some(value) = &selection {
                cfg["output"]["report"] = value.clone();
            }
            save(root.path(), &cfg);
            let sources = mode_inputs(root.path(), &server);
            let stdout = envelope(
                invoke(
                    root.path(),
                    &[&sources[mode], "-o", "out", "--json", "--quiet"],
                ),
                0,
            );
            assert_eq!(stdout["ok"], true);
            assert_eq!(stdout["totals"]["completed"], 1);
            recorded_output(root.path(), &stdout["items"][0]);
            let enabled = selection
                .as_ref()
                .and_then(Value::as_bool)
                .unwrap_or(mode == 1 || mode == 3);
            assert_eq!(
                report_paths(&root.path().join("out")).len(),
                usize::from(enabled),
                "mode={mode}, selection={selection:?}"
            );
        }
    }
}

#[test]
fn single_file_report_keeps_basename_exact_target_types_and_order() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["output"]["report"] = json!(true);
    save(root.path(), &cfg);
    input(
        root.path(),
        "输入 space/笔记.txt",
        "An explicitly named output.\n",
    );
    let stdout = envelope(
        invoke(
            root.path(),
            &["输入 space/笔记.txt", "-o", "out space/chosen.md", "--json"],
        ),
        0,
    );
    let (raw, saved) = report(&root.path().join("out space"));
    keys(
        &saved,
        &[
            "version",
            "generated_at",
            "log_file",
            "summary",
            "llm_usage",
            "documents",
        ],
    );
    ordered_top(
        &raw,
        &[
            "version",
            "generated_at",
            "log_file",
            "summary",
            "llm_usage",
            "documents",
        ],
    );
    keys(&saved["documents"], &["笔记.txt"]);
    let item = &saved["documents"]["笔记.txt"];
    keys(
        item,
        &[
            "status",
            "output",
            "error",
            "duration",
            "images",
            "screenshots",
            "llm_usage",
        ],
    );
    keys(
        &saved["summary"],
        &[
            "total_documents",
            "completed_documents",
            "failed_documents",
            "duration",
        ],
    );
    assert_eq!(saved["summary"]["total_documents"], 1);
    assert_eq!(saved["summary"]["completed_documents"], 1);
    assert_eq!(saved["summary"]["failed_documents"], 0);
    assert_eq!(item["status"], "completed");
    assert!(item["error"].is_null());
    assert_eq!(item["images"].as_u64(), Some(0));
    assert_eq!(item["screenshots"].as_u64(), Some(0));
    duration(&item["duration"]);
    assert_eq!(item["duration"], saved["summary"]["duration"]);
    assert_eq!(
        item["llm_usage"],
        json!({"cost_usd":0.0,"input_tokens":0,"output_tokens":0})
    );
    zero_usage(&saved["llm_usage"]);
    assert_eq!(item["output"], stdout["items"][0]["output"]);
    assert_eq!(
        recorded_output(root.path(), item),
        root.path().join("out space/chosen.md")
    );
    assert!(stdout["items"][0]["duration_s"].is_number());
    assert!(saved.get("items").is_none());
    let quiet = invoke(
        root.path(),
        &[
            "输入 space/笔记.txt",
            "-o",
            "out space/chosen.md",
            "--quiet",
        ],
    );
    assert!(quiet.status.success());
    assert!(quiet.stdout.is_empty());
    let (_, quiet_saved) = report(&root.path().join("out space"));
    keys(
        &quiet_saved,
        &[
            "version",
            "generated_at",
            "log_file",
            "summary",
            "llm_usage",
            "documents",
        ],
    );
}

#[test]
fn directory_partial_failure_preserves_relative_paths_timestamps_and_pending_counts() {
    let root = tempfile::tempdir().unwrap();
    configure(root.path());
    input(root.path(), "input/root.txt", "Root document.\n");
    input(root.path(), "input/sub/root.txt", "Nested document.\n");
    input(root.path(), "input/bad.ipynb", "{broken");
    let stdout = envelope(
        invoke(root.path(), &["input", "-o", "out", "--json", "-j", "2"]),
        10,
    );
    assert_eq!(stdout["totals"]["completed"], 2);
    assert_eq!(stdout["totals"]["failed"], 1);
    let (raw, saved) = report(&root.path().join("out"));
    keys(
        &saved,
        &[
            "version",
            "generated_at",
            "started_at",
            "updated_at",
            "log_file",
            "options",
            "summary",
            "llm_usage",
            "documents",
            "url_sources",
        ],
    );
    ordered_top(
        &raw,
        &[
            "version",
            "generated_at",
            "started_at",
            "updated_at",
            "log_file",
            "options",
            "summary",
            "llm_usage",
            "documents",
            "url_sources",
        ],
    );
    ordered(&raw, &["bad.ipynb", "root.txt", "sub/root.txt"]);
    keys(
        &saved["documents"],
        &["bad.ipynb", "root.txt", "sub/root.txt"],
    );
    let options = &saved["options"];
    assert_eq!(options["concurrency"], 2);
    assert_eq!(options["scan_max_depth"], 8);
    assert_eq!(
        Path::new(options["input_dir"].as_str().unwrap()),
        markitai_core::platform::canonicalize(&root.path().join("input")).unwrap()
    );
    assert_eq!(
        Path::new(options["output_dir"].as_str().unwrap()),
        markitai_core::platform::canonicalize(&root.path().join("out")).unwrap()
    );
    for flag in ["llm", "ocr", "screenshot", "alt", "desc"] {
        assert_eq!(options[flag], false);
    }
    assert_eq!(saved["summary"]["total_documents"], 3);
    assert_eq!(saved["summary"]["completed_documents"], 2);
    assert_eq!(saved["summary"]["failed_documents"], 1);
    assert_eq!(saved["summary"]["pending_documents"], 1);
    assert_eq!(saved["summary"]["total_urls"], 0);
    assert_eq!(saved["url_sources"], json!({}));
    duration(&saved["summary"]["processing_time"]);
    timestamp(&saved["started_at"]);
    timestamp(&saved["updated_at"]);
    for (name, entry) in saved["documents"].as_object().unwrap() {
        keys(
            entry,
            &[
                "status",
                "cache_hit",
                "output",
                "error",
                "started_at",
                "completed_at",
                "duration",
                "images",
                "screenshots",
                "cost_usd",
                "llm_usage",
            ],
        );
        timestamp(&entry["started_at"]);
        timestamp(&entry["completed_at"]);
        let start =
            chrono::DateTime::parse_from_rfc3339(entry["started_at"].as_str().unwrap()).unwrap();
        let finish =
            chrono::DateTime::parse_from_rfc3339(entry["completed_at"].as_str().unwrap()).unwrap();
        assert!(finish >= start);
        duration(&entry["duration"]);
        assert_eq!(entry["cache_hit"], false);
        if name == "bad.ipynb" {
            assert_eq!(entry["status"], "failed");
            assert!(entry["output"].is_null());
            assert!(!entry["error"].as_str().unwrap().is_empty());
        } else {
            assert_eq!(entry["status"], "completed");
            assert!(entry["error"].is_null());
            assert_eq!(
                recorded_output(root.path(), entry),
                root.path().join("out").join(format!("{name}.md"))
            );
        }
    }
    zero_usage(&saved["llm_usage"]);
}

#[test]
fn mixed_directory_groups_urls_by_source_without_flattening_local_documents() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    configure(root.path());
    input(root.path(), "input/local.txt", "Local.\n");
    let first = format!("{}/first", server.base);
    let second = format!("{}/second", server.base);
    input(root.path(), "input/z.urls", &format!("{first} first\n"));
    input(
        root.path(),
        "input/sub/a.urls",
        &format!("{second} second.md\n"),
    );
    let stdout = envelope(invoke(root.path(), &["input", "-o", "out", "--json"]), 0);
    assert_eq!(stdout["totals"]["completed"], 3);
    let (raw, saved) = report(&root.path().join("out"));
    keys(&saved["documents"], &["local.txt"]);
    // A list is named by its path as discovered, in the platform's spelling
    // (`input\sub\a.urls` on Windows, as the reference's `str(path)` gives).
    let (nested, top) = (native("input/sub/a.urls"), native("input/z.urls"));
    keys(&saved["url_sources"], &[nested.as_str(), top.as_str()]);
    ordered(&raw, &[nested.as_str(), top.as_str()]);
    assert_eq!(saved["summary"]["total_documents"], 1);
    assert_eq!(saved["summary"]["total_urls"], 2);
    assert_eq!(saved["summary"]["completed_urls"], 2);
    assert_eq!(saved["summary"]["url_sources"], 2);
    assert_eq!(saved["summary"]["url_cache_hits"], 0);
    for (source, key, suffix) in [
        ("input/z.urls", format!("{first} first"), "out/first.md"),
        (
            "input/sub/a.urls",
            format!("{second} second.md"),
            "out/sub/second.md",
        ),
    ] {
        let group = &saved["url_sources"][native(source)];
        assert_eq!(group["total"], 1);
        assert_eq!(group["completed"], 1);
        assert_eq!(group["failed"], 0);
        let item = &group["urls"][&key];
        assert_eq!(recorded_output(root.path(), item), root.path().join(suffix));
        assert_eq!(item["cache_hit"], false);
        assert_eq!(item["fetch_strategy"], "static");
        duration(&item["duration"]);
        assert!(item.get("source_file").is_none());
        assert!(item.get("cache_details").is_none());
    }
}

#[test]
fn single_url_report_combines_fetch_cache_but_stdout_retains_llm_only_cache() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let mut cfg = configure(root.path());
    cfg["cache"]["enabled"] = json!(true);
    cfg["output"]["report"] = json!(true);
    save(root.path(), &cfg);
    let url = format!("{}/page.urls", server.base);
    for cached in [false, true] {
        let stdout = envelope(
            invoke(root.path(), &[&url, "-o", "out/page.md", "--json"]),
            0,
        );
        let (raw, saved) = report(&root.path().join("out"));
        keys(
            &saved,
            &[
                "version",
                "generated_at",
                "log_file",
                "options",
                "summary",
                "llm_usage",
                "url_sources",
            ],
        );
        ordered_top(
            &raw,
            &[
                "version",
                "generated_at",
                "log_file",
                "options",
                "summary",
                "llm_usage",
                "url_sources",
            ],
        );
        assert_eq!(
            saved["options"],
            json!({"llm":false,"cache":true,"alt":false,"desc":false,"fetch_strategy":"static"})
        );
        keys(&saved["url_sources"], &["cli"]);
        let item = &saved["url_sources"]["cli"]["urls"][&url];
        keys(
            item,
            &[
                "status",
                "cache_hit",
                "cache_details",
                "output",
                "error",
                "fetch_strategy",
                "duration",
                "images",
                "screenshots",
                "llm_usage",
            ],
        );
        assert_eq!(item["cache_hit"], cached);
        assert_eq!(item["cache_details"], json!({"fetch":cached,"llm":false}));
        assert_eq!(stdout["items"][0]["cache_hit"], false);
        assert_eq!(stdout["items"][0]["fetch_cache_hit"], cached);
        assert_eq!(item["fetch_strategy"], "static");
        duration(&item["duration"]);
        assert_eq!(
            recorded_output(root.path(), item),
            root.path().join("out/page.md")
        );
        zero_usage(&saved["llm_usage"]);
    }
    assert_eq!(server.gets.load(Ordering::SeqCst), 1);
}

#[test]
fn url_list_preserves_raw_named_identity_deduplicates_exact_pairs_and_records_reserved_paths() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    configure(root.path());
    let url = format!("{}/page", server.base);
    let other = format!("{}/other", server.base);
    input(
        root.path(),
        "sources.urls",
        &format!("{url} custom\n{url} custom\n{url} custom.md\n{other} custom\n"),
    );
    let stdout = envelope(
        invoke(root.path(), &["sources.urls", "-o", "out", "--json"]),
        0,
    );
    assert_eq!(stdout["totals"]["total"], 3);
    let (_, saved) = report(&root.path().join("out"));
    keys(
        &saved,
        &[
            "version",
            "generated_at",
            "log_file",
            "summary",
            "llm_usage",
            "url_sources",
        ],
    );
    keys(&saved["url_sources"], &["unknown.urls"]);
    let group = &saved["url_sources"]["unknown.urls"];
    assert_eq!(group["total"], 3);
    assert_eq!(group["completed"], 3);
    assert_eq!(saved["summary"]["total_urls"], 3);
    let expected = [
        format!("{url} custom"),
        format!("{url} custom.md"),
        format!("{other} custom"),
    ];
    keys(
        &group["urls"],
        &expected.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    let mut outputs = BTreeSet::new();
    for entry in group["urls"].as_object().unwrap().values() {
        keys(
            entry,
            &[
                "status",
                "output",
                "error",
                "fetch_strategy",
                "images",
                "screenshots",
            ],
        );
        assert_eq!(entry["status"], "completed");
        assert!(entry["error"].is_null());
        outputs.insert(recorded_output(root.path(), entry));
    }
    assert_eq!(
        outputs,
        ["custom.md", "custom.v2.md", "custom.v3.md"]
            .map(|name| root.path().join("out").join(name))
            .into_iter()
            .collect()
    );
    zero_usage(&saved["llm_usage"]);
}

#[test]
fn url_list_partial_failure_keeps_sparse_error_and_single_json_envelope() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    configure(root.path());
    let good = format!("{}/page", server.base);
    let bad = format!("{}/missing", server.base);
    input(
        root.path(),
        "sources.urls",
        &format!("{good} good\n{bad} bad\n"),
    );
    let stdout = envelope(
        invoke(root.path(), &["sources.urls", "-o", "out", "--json"]),
        10,
    );
    assert_eq!(stdout["ok"], false);
    assert_eq!(stdout["totals"]["failed"], 1);
    let (_, saved) = report(&root.path().join("out"));
    let group = &saved["url_sources"]["unknown.urls"];
    assert_eq!(group["total"], 2);
    assert_eq!(group["completed"], 1);
    assert_eq!(group["failed"], 1);
    assert_eq!(saved["summary"]["failed_urls"], 1);
    let failed = &group["urls"][format!("{bad} bad")];
    keys(failed, &["status", "error"]);
    assert_eq!(failed["status"], "failed");
    assert!(!failed["error"].as_str().unwrap().is_empty());
    assert_eq!(
        recorded_output(root.path(), &group["urls"][format!("{good} good")]),
        root.path().join("out/good.md")
    );
    assert!(!root.path().join("out/bad.md").exists());
}

#[test]
fn stdout_and_dry_runs_do_not_publish_reports_even_when_enabled() {
    let server = Server::start();
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["output"]["report"] = json!(true);
    save(root.path(), &cfg);
    let sources = mode_inputs(root.path(), &server);
    for source in [&sources[0], &sources[2]] {
        let output = invoke(root.path(), &[source, "--pure", "--quiet"]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.stdout.is_empty());
        assert!(report_paths(root.path()).is_empty());
    }
    let fetched = server.gets.load(Ordering::SeqCst);
    for source in &sources {
        let output = invoke(root.path(), &[source, "-o", "preview", "--dry-run"]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(report_paths(&root.path().join("preview")).is_empty());
    }
    assert_eq!(
        server.gets.load(Ordering::SeqCst),
        fetched,
        "dry run fetched a URL"
    );
}

#[test]
fn empty_and_invalid_inputs_never_create_success_report_shells() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["output"]["report"] = json!(true);
    save(root.path(), &cfg);
    std::fs::create_dir(root.path().join("empty")).unwrap();
    input(root.path(), "empty.urls", "# no entries\n");
    let directory = invoke(root.path(), &["empty", "-o", "out"]);
    assert!(directory.status.success());
    let empty_json = envelope(invoke(root.path(), &["empty", "-o", "out", "--json"]), 0);
    assert_eq!(empty_json["ok"], true);
    assert_eq!(empty_json["items"], json!([]));
    assert_eq!(empty_json["totals"]["total"], 0);
    let list = invoke(root.path(), &["empty.urls", "-o", "out"]);
    assert_eq!(list.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&list.stderr).contains("No valid URLs"));
    assert!(report_paths(&root.path().join("out")).is_empty());
    input(root.path(), "note.txt", "Never converted.\n");
    cfg["output"]["report"] = json!({"invalid":true});
    save(root.path(), &cfg);
    let invalid = invoke(root.path(), &["note.txt", "-o", "out"]);
    assert!(!invalid.status.success());
    assert!(report_paths(&root.path().join("out")).is_empty());
}

#[test]
fn single_conversion_failures_keep_runtime_envelopes_without_reports() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let mut cfg = configure(root.path());
    cfg["output"]["report"] = json!(true);
    save(root.path(), &cfg);
    input(root.path(), "bad.ipynb", "{invalid");
    input(root.path(), "bad.unknown", "Unsupported format.");
    let missing_url = format!("{}/missing", server.base);
    for source in [
        "absent.txt",
        "bad.ipynb",
        "bad.unknown",
        missing_url.as_str(),
    ] {
        let stdout = envelope(invoke(root.path(), &[source, "-o", "out", "--json"]), 1);
        assert_eq!(stdout["ok"], false);
        assert_eq!(stdout["items"][0]["status"], "failed");
        assert!(stdout["items"][0]["duration_s"].is_null());
        assert!(!stdout["items"][0]["error"].as_str().unwrap().is_empty());
        assert!(report_paths(&root.path().join("out")).is_empty());
    }
}

#[test]
fn report_rename_versions_preserve_earlier_bytes_and_overwrite_replaces_only_current_report() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["output"] = json!({"report":true,"on_conflict":"rename"});
    save(root.path(), &cfg);
    input(root.path(), "note.txt", "First body.\n");
    let mut snapshots: Vec<(PathBuf, Vec<u8>)> = Vec::new();
    for _ in 0..3 {
        let stdout = envelope(invoke(root.path(), &["note.txt", "-o", "out", "--json"]), 0);
        let paths = report_paths(&root.path().join("out"));
        assert_eq!(paths.len(), snapshots.len() + 1);
        for (path, bytes) in &snapshots {
            assert_eq!(&std::fs::read(path).unwrap(), bytes);
        }
        let new = paths
            .iter()
            .find(|path| !snapshots.iter().any(|(old, _)| old == *path))
            .unwrap();
        let bytes = std::fs::read(new).unwrap();
        let saved: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            saved["documents"]["note.txt"]["output"],
            stdout["items"][0]["output"]
        );
        recorded_output(root.path(), &saved["documents"]["note.txt"]);
        snapshots.push((new.clone(), bytes));
    }
    assert!(
        snapshots[1]
            .0
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".v2.report.json")
    );
    assert!(
        snapshots[2]
            .0
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .ends_with(".v3.report.json")
    );
    cfg["output"]["on_conflict"] = json!("overwrite");
    save(root.path(), &cfg);
    input(root.path(), "note.txt", "Replacement body.\n");
    envelope(invoke(root.path(), &["note.txt", "-o", "out", "--json"]), 0);
    assert_eq!(report_paths(&root.path().join("out")).len(), 3);
    for (path, bytes) in snapshots.iter().skip(1) {
        assert_eq!(&std::fs::read(path).unwrap(), bytes);
    }
    let current: Value = serde_json::from_slice(&std::fs::read(&snapshots[0].0).unwrap()).unwrap();
    let path = recorded_output(root.path(), &current["documents"]["note.txt"]);
    assert!(
        std::fs::read_to_string(path)
            .unwrap()
            .contains("Replacement body.")
    );
}

#[test]
fn skipped_runs_preserve_existing_reports_and_outputs_in_every_mode() {
    let server = Server::start();
    for mode in 0..4 {
        let root = tempfile::tempdir().unwrap();
        let mut cfg = configure(root.path());
        cfg["output"]["report"] = json!(true);
        save(root.path(), &cfg);
        let sources = mode_inputs(root.path(), &server);
        let first = envelope(
            invoke(root.path(), &[&sources[mode], "-o", "out", "--json"]),
            0,
        );
        let output_path = recorded_output(root.path(), &first["items"][0]);
        let report_path = report_paths(&root.path().join("out")).pop().unwrap();
        let report_bytes = std::fs::read(&report_path).unwrap();
        cfg["output"]["report"] = json!(false);
        save(root.path(), &cfg);
        let disabled = envelope(
            invoke(root.path(), &[&sources[mode], "-o", "out", "--json"]),
            0,
        );
        assert_eq!(disabled["items"][0]["status"], "completed");
        assert_eq!(std::fs::read(&report_path).unwrap(), report_bytes);
        let output_bytes = std::fs::read(&output_path).unwrap();
        cfg["output"]["report"] = json!(true);
        cfg["output"]["on_conflict"] = json!("skip");
        save(root.path(), &cfg);
        let skipped = envelope(
            invoke(root.path(), &[&sources[mode], "-o", "out", "--json"]),
            0,
        );
        assert_eq!(skipped["items"][0]["status"], "skipped");
        assert_eq!(std::fs::read(&output_path).unwrap(), output_bytes);
        assert_eq!(std::fs::read(&report_path).unwrap(), report_bytes);
        assert_eq!(report_paths(&root.path().join("out")).len(), 1);
    }
}

#[test]
fn report_publication_failure_keeps_completed_output_and_one_error_envelope() {
    for batch in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut cfg = configure(root.path());
        cfg["output"]["report"] = json!(true);
        save(root.path(), &cfg);
        input(
            root.path(),
            "input/note.txt",
            "Conversion survives report failure.\n",
        );
        input(
            root.path(),
            "out/.markitai/reports",
            "Existing regular file must survive.",
        );
        let source = if batch { "input" } else { "input/note.txt" };
        let stdout = envelope(invoke(root.path(), &[source, "-o", "out", "--json"]), 1);
        assert_eq!(stdout["ok"], false);
        assert!(!stdout["error"].as_str().unwrap().is_empty());
        assert_eq!(stdout["totals"]["completed"], 1);
        assert_eq!(stdout["totals"]["failed"], 0);
        assert_eq!(stdout["items"][0]["status"], "completed");
        recorded_output(root.path(), &stdout["items"][0]);
        assert_eq!(
            std::fs::read_to_string(root.path().join("out/.markitai/reports")).unwrap(),
            "Existing regular file must survive."
        );
    }
}

#[test]
fn local_model_usage_flows_into_mode_specific_reports_and_final_output_paths() {
    let server = Server::start();
    for url_mode in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut cfg = configure(root.path());
        cfg["llm"] = json!({"enabled":true,"on_failure":"fail","router_settings":{"num_retries":0},"model_list":[{"model_name":"mock","litellm_params":{"model":"openai/report-test","api_base":format!("{}/v1",server.base),"api_key":"fixture-only"}}]});
        cfg["output"]["report"] = json!(true);
        save(root.path(), &cfg);
        input(root.path(), "note.txt", "Raw source to enhance.\n");
        let source = if url_mode {
            format!("{}/page", server.base)
        } else {
            "note.txt".into()
        };
        let stdout = envelope(
            invoke(root.path(), &[&source, "-o", "out/final.md", "--json"]),
            0,
        );
        let (_, saved) = report(&root.path().join("out"));
        assert_eq!(saved["llm_usage"]["requests"], 1);
        assert_eq!(saved["llm_usage"]["input_tokens"], 9);
        assert_eq!(saved["llm_usage"]["output_tokens"], 5);
        assert_eq!(
            saved["llm_usage"]["models"],
            stdout["items"][0]["llm_usage"]
        );
        assert_eq!(saved["llm_usage"]["models"].as_object().unwrap().len(), 1);
        let item = if url_mode {
            &saved["url_sources"]["cli"]["urls"][&source]
        } else {
            &saved["documents"]["note.txt"]
        };
        if url_mode {
            assert_eq!(item["llm_usage"], stdout["items"][0]["llm_usage"]);
        } else {
            keys(
                &item["llm_usage"],
                &["input_tokens", "output_tokens", "cost_usd"],
            );
            assert_eq!(item["llm_usage"]["input_tokens"], 9);
            assert_eq!(item["llm_usage"]["output_tokens"], 5);
        }
        let path = recorded_output(root.path(), item);
        assert_eq!(item["output"], stdout["items"][0]["output"]);
        assert_eq!(path, root.path().join("out/final.md"));
        assert!(
            std::fs::read_to_string(path)
                .unwrap()
                .contains("Local fixture answer.")
        );
        assert!(
            !serde_json::to_string(&saved)
                .unwrap()
                .contains("fixture-only")
        );
    }
    assert_eq!(server.posts.load(Ordering::SeqCst), 2);
}

#[test]
fn fresh_url_list_report_distinguishes_skipped_items_from_failed_and_completed() {
    let root = tempfile::tempdir().unwrap();
    let server = Server::start();
    let mut cfg = configure(root.path());
    cfg["output"]["on_conflict"] = json!("skip");
    save(root.path(), &cfg);
    let old = format!("{}/old", server.base);
    let fresh = format!("{}/fresh", server.base);
    input(
        root.path(),
        "sources.urls",
        &format!("{old} old\n{fresh} fresh\n"),
    );
    input(
        root.path(),
        "out/old.md",
        "Existing output remains untouched.\n",
    );
    let stdout = envelope(
        invoke(root.path(), &["sources.urls", "-o", "out", "--json"]),
        0,
    );
    assert_eq!(stdout["totals"]["completed"], 1);
    assert_eq!(stdout["totals"]["skipped"], 1);
    let (_, saved) = report(&root.path().join("out"));
    let group = &saved["url_sources"]["unknown.urls"];
    assert_eq!(group["total"], 2);
    assert_eq!(group["completed"], 1);
    assert_eq!(group["failed"], 0);
    assert_eq!(
        group["urls"][format!("{old} old")],
        json!({"status":"skipped","error":"Output exists"})
    );
    recorded_output(root.path(), &group["urls"][format!("{fresh} fresh")]);
    assert_eq!(
        std::fs::read_to_string(root.path().join("out/old.md")).unwrap(),
        "Existing output remains untouched.\n"
    );
}

#[test]
fn nonzero_png_attachment_report_matches_persisted_asset_and_markdown_reference() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = configure(root.path());
    cfg["output"]["report"] = json!(true);
    // A downloadable original must survive the default image-preview filter
    // even when its dimensions are only one pixel.
    save(root.path(), &cfg);
    // A complete 1x1 RGBA PNG with valid chunk CRCs, embedded as a MIME
    // attachment. This reaches the native asset writer without OCR or a model.
    let png: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f,
        0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8,
        0xcf, 0xc0, 0xf0, 0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99, 0x3d, 0x1d, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];
    input(
        root.path(),
        "attachment.eml",
        concat!(
            "MIME-Version: 1.0\r\n",
            "Subject: Native PNG attachment\r\n",
            "Content-Type: multipart/mixed; boundary=report-fixture\r\n\r\n",
            "--report-fixture\r\n",
            "Content-Type: text/plain; charset=utf-8\r\n\r\n",
            "A body with one image attachment.\r\n",
            "--report-fixture\r\n",
            "Content-Type: image/png; name=pixel.png\r\n",
            "Content-Disposition: attachment; filename=pixel.png\r\n",
            "Content-Transfer-Encoding: base64\r\n\r\n",
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==\r\n",
            "--report-fixture--\r\n",
        ),
    );
    let stdout = envelope(
        invoke(root.path(), &["attachment.eml", "-o", "out", "--json"]),
        0,
    );
    let (_, saved) = report(&root.path().join("out"));
    let entry = &saved["documents"]["attachment.eml"];
    let assets: Vec<_> = std::fs::read_dir(root.path().join("out/.markitai/assets"))
        .unwrap()
        .map(|item| item.unwrap().path())
        .collect();
    assert_eq!(assets.len(), 1);
    assert_eq!(assets[0].extension().unwrap(), "png");
    assert_eq!(std::fs::read(&assets[0]).unwrap(), png);
    let asset_count = assets.len() as u64;
    assert_eq!(entry["images"].as_u64(), Some(asset_count));
    assert_eq!(stdout["items"][0]["images"].as_u64(), Some(asset_count));
    assert_eq!(entry["screenshots"].as_u64(), Some(0));
    assert_eq!(stdout["items"][0]["screenshots"].as_u64(), Some(0));
    let markdown = std::fs::read_to_string(recorded_output(root.path(), entry)).unwrap();
    let target = format!(
        ".markitai/assets/{}",
        assets[0].file_name().unwrap().to_str().unwrap()
    );
    assert!(markdown.contains(&format!(
        "## Attachments\n\n- [pixel.png]({target}) ({} B)",
        png.len()
    )));
    assert!(!markdown.contains("![pixel.png]"));
    assert!(!markdown.contains(".markitai/assets/email-1-pixel.png"));
    zero_usage(&saved["llm_usage"]);
}

// Document cleanup is structured; pure vision and connection probes remain text.
fn model_content(request: &Value, markdown: &str) -> String {
    let messages = request["messages"].as_array().unwrap();
    if !messages.iter().any(|message| {
        message["role"] == "system"
            && message["content"]
                .as_str()
                .is_some_and(|text| text.contains("MARKITAI_DOCUMENT_JSON_V1"))
    }) {
        return markdown.to_owned();
    }
    // Echo protected source spans once and in order; test-specific output still
    // identifies the model invocation/generation used by the existing assertions.
    let source = messages
        .iter()
        .filter(|message| message["role"] == "user")
        .filter_map(|message| message["content"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    json!({"cleaned_markdown":format!("{markdown}\n\n{source}"),"frontmatter":{"description":"Local test document","tags":["fixture"]}}).to_string()
}
