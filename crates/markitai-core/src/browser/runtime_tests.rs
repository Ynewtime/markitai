//! Selected tests launch actual Chromium only against a private loopback fixture.
use super::*;
use crate::BrowserRuntime;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;

const PDF: &[u8] = include_bytes!("../pdf_raster/fixtures/mixed-native-scanned-blank.pdf");

fn isolated(name: &str) -> bool {
    if std::env::var("MARKITAI_BROWSER_RUNTIME_CASE").as_deref() == Ok(name) {
        return false;
    }
    let directory = tempfile::tempdir().unwrap();
    let stdout = directory.path().join("stdout");
    let stderr = directory.path().join("stderr");
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            &format!("browser::runtime_tests::{name}"),
            "--ignored",
            "--nocapture",
        ])
        .env_clear()
        .env("MARKITAI_BROWSER_RUNTIME_CASE", name)
        .env("MARKITAI_HOME", directory.path().join("home"))
        .current_dir(directory.path())
        .stdout(std::fs::File::create(&stdout).unwrap())
        .stderr(std::fs::File::create(&stderr).unwrap());
    for name in [
        "HOME",
        "PATH",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "TZ",
        "MARKITAI_BROWSER_EXECUTABLE",
        "PLAYWRIGHT_BROWSERS_PATH",
    ] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    let mut child = command.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(90);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("runtime fixture child timed out: {name}");
        }
        thread::sleep(Duration::from_millis(10));
    };
    assert!(
        status.success(),
        "{name}: {}\n{}",
        std::fs::read_to_string(&stdout).unwrap(),
        std::fs::read_to_string(stderr).unwrap()
    );
    assert!(
        std::fs::read_to_string(stdout)
            .unwrap()
            .contains("1 passed")
    );
    true
}

#[derive(Clone)]
struct Request {
    path: String,
    cookie: String,
    authorization: String,
    language: String,
    agent: String,
}
#[derive(Default)]
struct State {
    requests: Vec<Request>,
    release: bool,
}
struct Site {
    base: String,
    state: Arc<(Mutex<State>, Condvar)>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
fn request(stream: &mut TcpStream) -> Option<Request> {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut request_reader =
        bounded_fixture_io::Reader::new(stream, std::time::Instant::now() + Duration::from_secs(3));
    stream
        .set_write_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0; 4096];
    while !bytes.windows(4).any(|value| value == b"\r\n\r\n") {
        let count = request_reader.read(&mut buffer).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&buffer[..count]);
        assert!(bytes.len() <= 32768);
    }
    let text = String::from_utf8(bytes).unwrap();
    let header = |name: &str| {
        text.lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                key.eq_ignore_ascii_case(name)
                    .then(|| value.trim().to_owned())
            })
            .unwrap_or_default()
    };
    Some(Request {
        path: text.lines().next()?.split_whitespace().nth(1)?.into(),
        cookie: header("Cookie"),
        authorization: header("Authorization"),
        language: header("Accept-Language"),
        agent: header("User-Agent"),
    })
}
impl Site {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new((Mutex::new(State::default()), Condvar::new()));
        let captured = state.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let worker = thread::spawn(move || {
            let mut workers = Vec::new();
            while !stopped.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(value) => value,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("fixture accept {error}"),
                };
                let state = captured.clone();
                workers.push(thread::spawn(move||{
                    let Some(request)=request(&mut stream)else{return};
                    {let mut shared=state.0.lock().unwrap();shared.requests.push(request.clone());state.1.notify_all();}
                    if request.path=="/hold" {
                        let guard=state.0.lock().unwrap();
                        let (guard,timeout)=state.1.wait_timeout_while(guard,Duration::from_secs(15),|state|!state.release).unwrap();
                        assert!(!timeout.timed_out()||guard.release,"fixture hold exceeded its bound");
                    }
                    let mut status=200;let mut headers=String::new();let mut mime="text/html";
                    let body=if request.path=="/redirect" {
                        status=302;headers.push_str("Location: /pdf-auth\r\n");Vec::new()
                    } else if request.path=="/pdf-auth" {
                        if request.authorization!=format!("Basic {}",base64::engine::general_purpose::STANDARD.encode("reader:fixture")) {
                            status=401;headers.push_str("WWW-Authenticate: Basic realm=\"runtime\"\r\n");b"Private fixture".to_vec()
                        }else{mime="application/pdf";PDF.to_vec()}
                    }else if request.path=="/fail" {
                        status=503;b"Fixture failure".to_vec()
                    }else if request.path=="/hang" {
                        b"<html><script>while(true){}</script></html>".to_vec()
                    }else if request.path=="/language" {
                        b"<html><body><article><p id='value'></p></article><script>document.getElementById('value').textContent='Language: '+navigator.language+' Month: '+new Date(0).toLocaleDateString(undefined,{month:'long',timeZone:'UTC'})+' Agent: '+navigator.userAgent;</script></body></html>".to_vec()
                    }else{
                        let seed=if request.path.starts_with("/seed") {
                            let marker=if request.path.ends_with("b"){"MARKERB"}else{"MARKERA"};
                            headers.push_str(&format!("Set-Cookie: saved={marker}; Path=/\r\n"));
                            format!("localStorage.setItem('saved','{marker}');")
                        }else{String::new()};
                        format!("<html><body><article><h1>Runtime fixture</h1><p id='value'></p></article><script>{seed}document.getElementById('value').textContent='Stored: '+(localStorage.getItem('saved')||'NONE')+' Cookie: '+document.cookie;</script></body></html>").into_bytes()
                    };
                    let head=format!("HTTP/1.1 {status} Fixture\r\nContent-Type: {mime}\r\nContent-Length: {}\r\n{headers}Connection: close\r\n\r\n",body.len());
                    let _=stream.write_all(head.as_bytes());let _=stream.write_all(&body);
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        Self {
            base,
            state,
            stop,
            worker: Some(worker),
        }
    }
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }
    fn count(&self, path: &str) -> usize {
        self.state
            .0
            .lock()
            .unwrap()
            .requests
            .iter()
            .filter(|r| r.path == path)
            .count()
    }
    fn wait(&self, path: &str) {
        let guard = self.state.0.lock().unwrap();
        let (guard, timeout) = self
            .state
            .1
            .wait_timeout_while(guard, Duration::from_secs(12), |state| {
                !state.requests.iter().any(|r| r.path == path)
            })
            .unwrap();
        assert!(
            !timeout.timed_out() || guard.requests.iter().any(|r| r.path == path),
            "missing fixture request {path}"
        );
    }
    fn release(&self) {
        self.state
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .release = true;
        self.state.1.notify_all();
    }
}
impl Drop for Site {
    fn drop(&mut self) {
        self.release();
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let joined = worker.join();
            if !thread::panicking() {
                joined.unwrap();
            }
        }
    }
}
fn config(persistent: bool) -> Value {
    json!({"fetch":{"playwright":{"session_mode":if persistent{"domain_persistent"}else{"isolated"},"session_ttl_seconds":600,"timeout":8000,"extra_wait_ms":0,"skip_auto_scroll":true}},"screenshot":{"viewport_width":320,"viewport_height":240,"tile_height":0}})
}
fn html(site: &Site, path: &str, cfg: &Value, runtime: &BrowserRuntime) -> String {
    fetch_with_runtime(&site.url(path), cfg, false, Some(runtime))
        .unwrap()
        .page()
        .html
}
fn gone(process: &(u32, PathBuf)) {
    assert!(!process.1.exists(), "private profile must be removed");
    #[cfg(unix)]
    assert_eq!(
        unsafe { libc::kill(process.0 as i32, 0) },
        -1,
        "browser process must be reaped"
    );
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn persistent_state_is_owned_but_isolated_contexts_never_replay_cookies() {
    if isolated("persistent_state_is_owned_but_isolated_contexts_never_replay_cookies") {
        return;
    }
    assert!(available());
    let site = Site::new();
    for persistent in [false, true] {
        let cfg = config(persistent);
        let runtime = BrowserRuntime::new(1).unwrap();
        assert!(html(&site, "/seed-a", &cfg, &runtime).contains("Stored: MARKERA"));
        let first = runtime.pool.processes();
        assert_eq!(first.len(), 1);
        let second = html(&site, "/read", &cfg, &runtime);
        assert!(second.contains(if persistent {
            "Stored: MARKERA"
        } else {
            "Stored: NONE"
        }));
        assert_eq!(second.contains("Cookie: saved=MARKERA"), persistent);
        assert_eq!(
            runtime.pool.processes(),
            first,
            "process must actually be reused"
        );
        let separate = BrowserRuntime::new(1).unwrap();
        assert!(html(&site, "/independent", &cfg, &separate).contains("Stored: NONE"));
        runtime.close();
        assert!(runtime.pool.processes().is_empty());
        gone(&first[0]);
    }
    let cfg = config(true);
    fetch(&site.url("/seed-a"), &cfg, false).unwrap();
    assert!(
        fetch(&site.url("/no-runtime"), &cfg, false)
            .unwrap()
            .page()
            .html
            .contains("Stored: NONE")
    );
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn identity_ttl_and_idle_eviction_preserve_scope_and_cleanup() {
    if isolated("identity_ttl_and_idle_eviction_preserve_scope_and_cleanup") {
        return;
    }
    assert!(available());
    let site = Site::new();
    let runtime = BrowserRuntime::new(2).unwrap();
    let cfg = config(true);
    html(&site, "/seed-a", &cfg, &runtime);
    let first = runtime.pool.processes()[0].clone();
    let mut different = cfg.clone();
    different["fetch"]["playwright"]["extra_http_headers"] = json!({"X-Account":"B"});
    assert!(html(&site, "/other-account", &different, &runtime).contains("Stored: NONE"));
    html(&site, "/seed-b", &different, &runtime);
    assert!(html(&site, "/read", &cfg, &runtime).contains("Stored: MARKERA"));
    runtime.pool.expire_idle();
    assert!(html(&site, "/after-expiry", &cfg, &runtime).contains("Stored: NONE"));
    gone(&first);
    let limited = BrowserRuntime::new(1).unwrap();
    html(&site, "/seed-a", &cfg, &limited);
    let prior = limited.pool.processes()[0].clone();
    let other_origin = Site::new();
    assert!(html(&other_origin, "/different-origin", &cfg, &limited).contains("Stored: NONE"));
    gone(&prior);
    assert_eq!(limited.pool.processes().len(), 1);
    assert!(html(&site, "/after-eviction", &cfg, &limited).contains("Stored: NONE"));
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn failed_navigation_and_panics_discard_the_process_without_repeating_gets() {
    if isolated("failed_navigation_and_panics_discard_the_process_without_repeating_gets") {
        return;
    }
    assert!(available());
    let site = Site::new();
    let runtime = BrowserRuntime::new(1).unwrap();
    let cfg = config(true);
    html(&site, "/seed-a", &cfg, &runtime);
    let process = runtime.pool.processes()[0].clone();
    let error = fetch_with_runtime(&site.url("/fail"), &cfg, false, Some(&runtime))
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("503"));
    assert_eq!(site.count("/fail"), 1);
    gone(&process);
    assert!(runtime.pool.processes().is_empty());
    assert!(html(&site, "/after-failure", &cfg, &runtime).contains("Stored: NONE"));
    let process = runtime.pool.processes()[0].clone();
    let mut short = cfg.clone();
    short["fetch"]["playwright"]["timeout"] = json!(400);
    assert!(fetch_with_runtime(&site.url("/hang"), &short, false, Some(&runtime)).is_err());
    assert_eq!(site.count("/hang"), 1);
    gone(&process);
    let url = Url::parse(&site.url("/unwind")).unwrap();
    let options = options::Options::from_config(&cfg, &url, false).unwrap();
    let process = std::sync::Mutex::new(None);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut lease = runtime
            .pool
            .acquire(&discover().unwrap(), &url, &cfg, &options)
            .unwrap();
        lease.browser().begin_page(&options).unwrap();
        *process.lock().unwrap() = lease.browser().test_process();
        panic!("authored in-flight unwind");
    }));
    assert!(result.is_err());
    gone(process.lock().unwrap().as_ref().unwrap());
    assert!(html(&site, "/after-panic", &cfg, &runtime).contains("Stored: NONE"));
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn same_identity_waits_other_identity_runs_and_close_wakes_queued_calls() {
    if isolated("same_identity_waits_other_identity_runs_and_close_wakes_queued_calls") {
        return;
    }
    assert!(available());
    let site = Site::new();
    let runtime = BrowserRuntime::new(2).unwrap();
    let cfg = config(true);
    let mut other = cfg.clone();
    other["fetch"]["playwright"]["user_agent"] = json!("Other private fixture identity");
    thread::scope(|scope| {
        let held = scope.spawn(|| html(&site, "/hold", &cfg, &runtime));
        site.wait("/hold");
        let processes = runtime.pool.processes();
        assert_eq!(processes.len(), 1);
        let waiting =
            scope.spawn(|| fetch_with_runtime(&site.url("/queued"), &cfg, false, Some(&runtime)));
        let deadline = Instant::now() + Duration::from_secs(3);
        while runtime.pool.waiting() == 0 {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(site.count("/queued"), 0);
        assert!(html(&site, "/other-active", &other, &runtime).contains("Stored: NONE"));
        assert_eq!(runtime.pool.processes().len(), 2);
        runtime.close();
        assert!(
            waiting
                .join()
                .unwrap()
                .err()
                .unwrap()
                .to_string()
                .contains("closed")
        );
        site.release();
        assert!(held.join().unwrap().contains("Stored: NONE"));
        gone(&processes[0]);
    });
    assert!(runtime.pool.processes().is_empty());
    assert_eq!(site.count("/queued"), 0);
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn authenticated_pdf_handoff_closes_its_page_and_keeps_the_owned_session() {
    if isolated("authenticated_pdf_handoff_closes_its_page_and_keeps_the_owned_session") {
        return;
    }
    assert!(available());
    let site = Site::new();
    let runtime = BrowserRuntime::new(1).unwrap();
    let mut cfg = config(true);
    cfg["fetch"]["playwright"]["http_credentials"] =
        json!({"username":"reader","password":"fixture"});
    html(&site, "/seed-a", &cfg, &runtime);
    let first = runtime.pool.processes();
    match fetch_with_runtime(&site.url("/redirect"), &cfg, false, Some(&runtime)).unwrap() {
        BrowserResponse::Pdf(pdf) => {
            assert_eq!(pdf.bytes, PDF);
            assert!(pdf.final_url.ends_with("/pdf-auth"));
            assert!(pdf.warnings.is_empty());
        }
        BrowserResponse::Page(_) => panic!("PDF must not become viewer HTML"),
    }
    assert_eq!(site.count("/redirect"), 1);
    let requests = site.state.0.lock().unwrap().requests.clone();
    assert_eq!(
        requests
            .iter()
            .filter(|r| r.path == "/pdf-auth" && !r.authorization.is_empty())
            .count(),
        1
    );
    assert!(
        requests
            .iter()
            .any(|r| r.path == "/pdf-auth" && r.cookie.contains("saved=MARKERA"))
    );
    assert!(html(&site, "/after-pdf", &cfg, &runtime).contains("Stored: MARKERA"));
    assert_eq!(runtime.pool.processes(), first);
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn all_owned_pages_close_before_a_context_is_reused() {
    if isolated("all_owned_pages_close_before_a_context_is_reused") {
        return;
    }
    assert!(available());
    let site = Site::new();
    for persistent in [false, true] {
        let cfg = config(persistent);
        let runtime = BrowserRuntime::new(1).unwrap();
        html(&site, "/seed-a", &cfg, &runtime);
        let processes = runtime.pool.processes();
        let url = Url::parse(&site.url("/new-page")).unwrap();
        let options = options::Options::from_config(&cfg, &url, false).unwrap();
        let mut lease = runtime
            .pool
            .acquire(&discover().unwrap(), &url, &cfg, &options)
            .unwrap();
        lease.browser().begin_page(&options).unwrap();
        let context = lease.browser().test_context().unwrap().to_owned();
        lease
            .browser()
            .call(
                "Target.createTarget",
                json!({"url":"about:blank","browserContextId":context}),
            )
            .unwrap();
        let targets = lease
            .browser()
            .call("Target.getTargets", json!({}))
            .unwrap();
        assert_eq!(
            targets["targetInfos"]
                .as_array()
                .unwrap()
                .iter()
                .filter(
                    |target| target["browserContextId"].as_str() == Some(context.as_str())
                        && target["type"] == "page"
                )
                .count(),
            2
        );
        lease.finish().unwrap();
        let targets = lease
            .browser()
            .call("Target.getTargets", json!({}))
            .unwrap();
        assert!(
            targets["targetInfos"]
                .as_array()
                .unwrap()
                .iter()
                .all(|target| target["browserContextId"].as_str() != Some(context.as_str()))
        );
        assert_eq!(lease.browser().test_context().is_some(), persistent);
        drop(lease);
        assert!(
            html(&site, "/after-popup", &cfg, &runtime).contains(if persistent {
                "Stored: MARKERA"
            } else {
                "Stored: NONE"
            })
        );
        assert_eq!(runtime.pool.processes(), processes);
        runtime.close();
        gone(&processes[0]);
    }
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn x_pages_are_requested_in_english_unless_the_caller_chooses_a_language() {
    if isolated("x_pages_are_requested_in_english_unless_the_caller_chooses_a_language") {
        return;
    }
    assert!(available());
    let site = Site::new();
    let runtime = BrowserRuntime::new(1).unwrap();
    let cfg = config(false);
    let text = html(&site, "/language", &cfg, &runtime);
    let sent = |site: &Site| {
        site.state
            .0
            .lock()
            .unwrap()
            .requests
            .iter()
            .rev()
            .find(|request| request.path == "/language")
            .cloned()
            .unwrap()
    };
    let request = sent(&site);
    // An ordinary site keeps the browser's own language and identity.
    assert_ne!(request.language, "en-US,en;q=0.9");
    assert!(request.agent.contains("Chrome/"), "{}", request.agent);
    assert!(text.contains("Month: "), "{text}");
    // A header the caller sends decides the language.
    let mut french = cfg.clone();
    french["fetch"]["playwright"]["extra_http_headers"] = json!({"accept-language":"fr-FR"});
    html(&site, "/language", &french, &runtime);
    assert_eq!(sent(&site).language, "fr-FR");
    // X's profile (it blocks `HeadlessChrome` and its reader parses English)
    // sees the browser's own version under the name Chrome, in English, in
    // both the header and the page.
    let url = Url::parse(&site.url("/language")).unwrap();
    let mut options = options::Options::from_config(&cfg, &url, false).unwrap();
    options.regular_user_agent = true;
    options.english = true;
    let mut lease = runtime
        .pool
        .acquire(&discover().unwrap(), &url, &cfg, &options)
        .unwrap();
    lease.browser().begin_page(&options).unwrap();
    let page = match fetch_page(
        &site.url("/language"),
        &cfg,
        false,
        &url,
        &options,
        lease.browser(),
    )
    .unwrap()
    {
        BrowserResponse::Page(page) => page.html,
        BrowserResponse::Pdf(_) => panic!("expected a page"),
    };
    lease.finish().unwrap();
    let request = sent(&site);
    assert!(request.agent.contains("Chrome/"), "{}", request.agent);
    assert!(
        !request.agent.contains("HeadlessChrome"),
        "{}",
        request.agent
    );
    assert_eq!(request.language, "en-US,en;q=0.9");
    assert!(!page.contains("HeadlessChrome"), "{page}");
    assert!(page.contains("Language: en-US"), "{page}");
    assert!(page.contains("Month: January"), "{page}");
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn a_failed_navigation_says_why_without_naming_the_address() {
    if isolated("a_failed_navigation_says_why_without_naming_the_address") {
        return;
    }
    assert!(available());
    // A port nothing listens on: the connection is refused.
    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = closed.local_addr().unwrap().port();
    drop(closed);
    let runtime = BrowserRuntime::new(1).unwrap();
    let error = fetch_with_runtime(
        &format!("http://127.0.0.1:{port}/private-path?token=secret"),
        &config(false),
        false,
        Some(&runtime),
    )
    .err()
    .unwrap()
    .to_string();
    assert!(error.contains("net::ERR_CONNECTION_REFUSED"), "{error}");
    for private in [port.to_string(), "private-path".into(), "secret".into()] {
        assert!(!error.contains(&private), "{error}");
    }
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
