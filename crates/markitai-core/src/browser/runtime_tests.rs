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
    isolated_with(name, &[])
}
/// [`isolated`], the child process also given `env`.
fn isolated_with(name: &str, env: &[(&str, &str)]) -> bool {
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
    command.envs(env.iter().copied());
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
    account: String,
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
        account: header("X-Account"),
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
                    } else if let Some(target)=request.path.strip_prefix("/to?u=") {
                        status=302;headers.push_str(&format!("Location: {target}\r\n"));Vec::new()
                    } else if request.path=="/pdf-auth" {
                        if request.authorization!=format!("Basic {}",base64::engine::general_purpose::STANDARD.encode("reader:fixture")) {
                            status=401;headers.push_str("WWW-Authenticate: Basic realm=\"runtime\"\r\n");b"Private fixture".to_vec()
                        }else{mime="application/pdf";PDF.to_vec()}
                    }else if request.path=="/fail" {
                        status=503;b"Fixture failure".to_vec()
                    }else if request.path=="/interstitial" {
                        if request.cookie.contains("passed=1") {
                            b"<html><body><article><h1>Behind the interstitial</h1><p>Interstitial passed with the page's own cookie.</p></article></body></html>".to_vec()
                        }else{
                            status=403;b"<html><body><div>Checking</div><script>document.cookie='passed=1; path=/';location.reload();</script></body></html>".to_vec()
                        }
                    }else if request.path=="/refused" {
                        status=403;format!("<html><body><h1>Access refused</h1><p>{}</p><script>window.x=1;</script></body></html>","This page explains in many words why the request was refused. ".repeat(8)).into_bytes()
                    }else if request.path=="/tall" {
                        // 500 blocks of 100 CSS pixels: 50 000 pixels.
                        let blocks:String=(1..=500).map(|n|format!("<section style='height:100px;margin:0'><h2 style='margin:0'>Block {n}</h2></section>")).collect();
                        format!("<html><head><title> An invented   tall page </title></head><body style='margin:0'>{blocks}</body></html>").into_bytes()
                    }else if request.path.starts_with("/remote") {
                        let remote=request.path.split("to=").nth(1).unwrap_or("");
                        format!("<html><body><p>Remote image page.</p><img src='/same-asset'><img src='{remote}'></body></html>").into_bytes()
                    }else if request.path=="/same-asset"||request.path=="/asset" {
                        mime="image/gif";b"GIF87a".to_vec()
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
fn an_interstitial_that_reloads_itself_is_awaited_and_a_refusal_with_content_is_not() {
    if isolated("an_interstitial_that_reloads_itself_is_awaited_and_a_refusal_with_content_is_not")
    {
        return;
    }
    assert!(available());
    let site = Site::new();
    let runtime = BrowserRuntime::new(1).unwrap();
    let cfg = config(false);
    // A 403 page that sets its cookie and reloads, as a browser is expected to.
    let page = html(&site, "/interstitial", &cfg, &runtime);
    assert!(page.contains("Interstitial passed"), "{page}");
    assert_eq!(site.count("/interstitial"), 2);
    // A refusal that explains itself is reported at once.
    let started = Instant::now();
    let error = fetch_with_runtime(&site.url("/refused"), &cfg, false, Some(&runtime))
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("HTTP 403"), "{error}");
    assert!(started.elapsed() < Duration::from_secs(5));
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
        Shots::None,
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
fn a_runtime_that_keeps_compare_screenshots_cuts_them_at_twenty_tiles() {
    if isolated("a_runtime_that_keeps_compare_screenshots_cuts_them_at_twenty_tiles") {
        return;
    }
    assert!(available());
    let site = Site::new();
    let mut cfg = config(false);
    cfg["screenshot"]["tile_height"] = json!(2000);
    let page = |runtime: &BrowserRuntime, screenshot: bool| {
        fetch_with_runtime(&site.url("/tall"), &cfg, screenshot, Some(runtime))
            .unwrap()
            .page()
    };
    // A runtime that keeps none takes none.
    let plain = BrowserRuntime::new(1).unwrap();
    let loaded = page(&plain, false);
    assert!(loaded.view.is_none() && loaded.screenshots.is_empty());
    let runtime = BrowserRuntime::new(1)
        .unwrap()
        .keeping_compare_screenshots();
    let loaded = page(&runtime, false);
    assert!(loaded.screenshots.is_empty());
    let view = loaded.view.expect("compare screenshots");
    // Twenty tiles of 2000 pixels, or as many as 50 million pixels allow.
    let blocks: Value = serde_json::from_slice(&view.blocks.expect("blocks")).unwrap();
    let width = blocks["width"].as_u64().unwrap();
    let height = 40_000u64.min(50_000_000 / width);
    assert_eq!(blocks["height"], height);
    assert_eq!(view.compare.len(), 20);
    for (index, tile) in view.compare.iter().enumerate() {
        let image = image::load_from_memory(tile).unwrap();
        assert_eq!(u64::from(image.width()), width);
        let expected = if index == 19 {
            height - 19 * 2000
        } else {
            2000
        };
        assert_eq!(u64::from(image.height()), expected, "tile {index}");
    }
    let headings: Vec<&Value> = blocks["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["tag"] == "h2")
        .collect();
    // The blocks that start above the cut; the rest are not shown.
    let shown = height.div_ceil(100) as usize;
    assert_eq!(headings.len(), shown);
    assert_eq!(headings[shown - 1]["text"], format!("Block {shown}"));
    assert_eq!(headings[shown - 1]["bbox"][1], (shown as f64 - 1.0) * 100.0);
    assert_eq!(
        view.quality,
        crate::page_capture::CaptureQuality {
            state: crate::page_capture::CaptureState::Ok,
            // "Block 1" … "Block 500", one space or line apart.
            text_length: (1..=500)
                .map(|n: u32| 6 + n.to_string().len() as u64)
                .sum::<u64>()
                + 499,
            title: Some("An invented tall page".into()),
            reason: None,
        }
    );
    // Screenshots the conversion asked for are its own, all of the page.
    let loaded = page(&runtime, true);
    let view = loaded.view.expect("screenshots");
    assert!(view.compare.is_empty());
    assert_eq!(loaded.screenshots.len(), 25);
    let blocks: Value = serde_json::from_slice(&view.blocks.unwrap()).unwrap();
    assert_eq!(blocks["height"], 50_000);
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

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn the_callers_headers_reach_the_initial_origin_only() {
    if isolated("the_callers_headers_reach_the_initial_origin_only") {
        return;
    }
    assert!(available());
    let site = Site::new();
    let other = Site::new();
    let runtime = BrowserRuntime::new(1).unwrap();
    let mut cfg = config(false);
    cfg["fetch"]["playwright"]["extra_http_headers"] = json!({"X-Account":"private-token"});
    // The page loads a same-origin image and one from the other origin.
    let path = format!("/remote?to={}/asset", other.base);
    html(&site, &path, &cfg, &runtime);
    site.wait("/same-asset");
    other.wait("/asset");
    let account = |site: &Site, path: &str| {
        site.state
            .0
            .lock()
            .unwrap()
            .requests
            .iter()
            .find(|request| request.path == path)
            .map(|request| request.account.clone())
            .unwrap()
    };
    // The document and its same-origin subresource carry the header.
    assert_eq!(account(&site, &path), "private-token");
    assert_eq!(account(&site, "/same-asset"), "private-token");
    // The other origin never sees it.
    assert_eq!(account(&other, "/asset"), "");
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn the_callers_headers_do_not_follow_a_redirect_to_another_origin() {
    if isolated("the_callers_headers_do_not_follow_a_redirect_to_another_origin") {
        return;
    }
    assert!(available());
    let site = Site::new();
    let other = Site::new();
    let runtime = BrowserRuntime::new(1).unwrap();
    let mut cfg = config(false);
    cfg["fetch"]["playwright"]["extra_http_headers"] = json!({"X-Account":"private-token"});
    // The initial origin answers with a redirect to the other origin's page.
    let path = format!("/to?u={}/seed", other.base);
    assert!(html(&site, &path, &cfg, &runtime).contains("Runtime fixture"));
    let account = |site: &Site, path: &str| {
        site.state
            .0
            .lock()
            .unwrap()
            .requests
            .iter()
            .find(|request| request.path == path)
            .map(|request| request.account.clone())
            .unwrap()
    };
    assert_eq!(account(&site, &path), "private-token");
    assert_eq!(account(&other, "/seed"), "");
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn a_restricted_browser_fetch_refuses_a_redirect_to_a_host_on_this_machine() {
    // The fixture's own address stands in for a public host; the redirect
    // target, another loopback address, is on this machine.
    if isolated_with(
        "a_restricted_browser_fetch_refuses_a_redirect_to_a_host_on_this_machine",
        &[("MARKITAI_TEST_PUBLIC_NET_ALLOW", "127.0.0.1")],
    ) {
        return;
    }
    assert!(available());
    let site = Site::new();
    let runtime = BrowserRuntime::new(1).unwrap();
    let cfg = config(false);
    let _restriction = crate::public_net::restrict(crate::public_net::Level::PublicOnly);
    // The allowed start is fetched under the restriction.
    let page = fetch_with_runtime(&site.url("/seed"), &cfg, false, Some(&runtime))
        .unwrap()
        .page()
        .html;
    assert!(page.contains("Runtime fixture"));
    // Its redirect to a refused address is blocked before any connection,
    // not refused by the (absent) server there.
    let port = site.base.rsplit(':').next().unwrap();
    let path = format!("/to?u=http://127.0.0.2:{port}/seed");
    let error = fetch_with_runtime(&site.url(&path), &cfg, false, Some(&runtime))
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("ERR_BLOCKED_BY_CLIENT"), "{error}");
    assert_eq!(site.count(&path), 1);
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn a_restricted_browser_fetch_refuses_a_host_on_this_machine() {
    if isolated("a_restricted_browser_fetch_refuses_a_host_on_this_machine") {
        return;
    }
    assert!(available());
    let site = Site::new();
    let runtime = BrowserRuntime::new(1).unwrap();
    let cfg = config(false);
    // Without the restriction the loopback page renders.
    assert!(html(&site, "/seed", &cfg, &runtime).contains("Runtime fixture"));
    // A restricted fetch (a user token's job, or a public requested host)
    // refuses the navigation, because the host is on this machine.
    let _restriction = crate::public_net::restrict(crate::public_net::Level::PublicOnly);
    let error = fetch_with_runtime(&site.url("/seed"), &cfg, false, Some(&runtime))
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("BlockedByClient") || error.contains("navigation failed"),
        "{error}"
    );
}

/// A loopback site for the render wait: `/static` is complete at load,
/// `/late` adds its text from a request answered after 700 ms, `/stream`
/// keeps an event stream open, and `/slow` changes its DOM every 100 ms.
fn settle_site() -> (String, Arc<AtomicBool>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let stopped = stop.clone();
    let worker = thread::spawn(move || {
        while !stopped.load(Ordering::Relaxed) {
            let Ok((mut stream, _)) = listener.accept() else {
                thread::sleep(Duration::from_millis(5));
                continue;
            };
            let stopped = stopped.clone();
            thread::spawn(move || {
                let Some(request) = request(&mut stream) else {
                    return;
                };
                let page = |body: &str| {
                    format!(
                        "<!doctype html><html><head><title>Settle</title></head><body>{body}</body></html>"
                    )
                };
                let (mime, body) = match request.path.as_str() {
                    "/static" => (
                        "text/html",
                        page(
                            "<article><h1>Complete</h1><p>Every word is here at load.</p></article>",
                        ),
                    ),
                    "/late" => (
                        "text/html",
                        page(
                            "<article><h1>Late</h1><p id=t>Loading</p></article><script>setTimeout(()=>fetch('/data').then(r=>r.text()).then(x=>{document.getElementById('t').textContent=x}),200)</script>",
                        ),
                    ),
                    "/data" => {
                        thread::sleep(Duration::from_millis(700));
                        ("text/plain", "Arrived after the load.".into())
                    }
                    "/stream" => (
                        "text/html",
                        page(
                            "<article><h1>Stream</h1><p>An open event stream does not hold the page.</p></article><script>new EventSource('/events')</script>",
                        ),
                    ),
                    "/events" => {
                        let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\ndata: hello\n\n");
                        while !stopped.load(Ordering::Relaxed) {
                            thread::sleep(Duration::from_millis(20));
                        }
                        return;
                    }
                    "/css" => (
                        "text/html",
                        page(
                            "<article><h1 class=x>Styled</h1><p class=y>Styled by scripts through the CSSOM.</p></article><style id=s></style><script>document.getElementById('s').sheet.insertRule('.x { color: rgb(1, 2, 3); }');const sheet=new CSSStyleSheet();sheet.replaceSync('.y { color: rgb(4, 5, 6); }');document.adoptedStyleSheets=[sheet];</script>",
                        ),
                    ),
                    "/slow" => (
                        "text/html",
                        page(
                            "<article><h1>Busy</h1><p id=n>0</p></article><script>let n=0;setInterval(()=>{document.getElementById('n').textContent=++n},100)</script>",
                        ),
                    ),
                    _ => ("text/plain", String::new()),
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            });
        }
    });
    (base, stop, worker)
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn the_kept_page_carries_the_style_rules_scripts_inserted() {
    if isolated("the_kept_page_carries_the_style_rules_scripts_inserted") {
        return;
    }
    assert!(available());
    let (base, stop, worker) = settle_site();
    let cfg =
        json!({"fetch":{"playwright":{"timeout":15000,"extra_wait_ms":0,"skip_auto_scroll":true}}});
    let html = fetch(&format!("{base}/css"), &cfg, false)
        .unwrap()
        .page()
        .html;
    assert!(html.contains(".x { color: rgb(1, 2, 3); }"), "{html}");
    assert!(html.contains(".y { color: rgb(4, 5, 6); }"), "{html}");
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap();
}

#[test]
#[ignore = "requires installed Chromium; private process and loopback only"]
fn the_render_wait_ends_once_the_page_has_settled() {
    if isolated("the_render_wait_ends_once_the_page_has_settled") {
        return;
    }
    assert!(available());
    let (base, stop, worker) = settle_site();
    let cfg = json!({"fetch":{"playwright":{"timeout":15000,"extra_wait_ms":3000}}});
    let runtime = BrowserRuntime::new(1).unwrap();
    let timed = |path: &str| {
        let started = Instant::now();
        let html = fetch_with_runtime(&format!("{base}{path}"), &cfg, false, Some(&runtime))
            .unwrap()
            .page()
            .html;
        (html, started.elapsed())
    };
    // Start the process once, so the timings below are the page's own.
    timed("/static");
    let (html, elapsed) = timed("/static");
    assert!(html.contains("Every word is here at load."));
    assert!(elapsed < Duration::from_millis(2500), "{elapsed:?}");
    // Text a script fetches after the load is waited for.
    let (html, _) = timed("/late");
    assert!(html.contains("Arrived after the load."), "{html}");
    // An open event stream does not hold the wait.
    let (html, elapsed) = timed("/stream");
    assert!(html.contains("does not hold the page"));
    assert!(elapsed < Duration::from_millis(2500), "{elapsed:?}");
    // A page that never stops changing keeps extra_wait_ms as its bound.
    let (_, elapsed) = timed("/slow");
    assert!(elapsed >= Duration::from_millis(3000), "{elapsed:?}");
    runtime.close();
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap();
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}

#[cfg(unix)]
mod app_bundle_diagnostic {
    use super::super::{SLOW_TO_START, diagnostic_within, inside_app_bundle, version_probe};
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::time::{Duration, Instant};

    fn script(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// A browser that never names its debugging port is tried twice and
    /// called slow to start; one that exits at once is not retried.
    #[test]
    fn a_launch_that_stays_slow_is_tried_twice_and_called_slow() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("launches");
        let slow = script(
            dir.path(),
            "slow",
            &format!("echo x >> '{}'; exec sleep 30", log.display()),
        );
        let started = Instant::now();
        let error = diagnostic_within(&slow, &[Duration::from_millis(300); 2]).unwrap_err();
        assert_eq!(error.to_string(), SLOW_TO_START);
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
        assert_eq!(std::fs::read_to_string(&log).unwrap().lines().count(), 2);
        let broken = script(
            dir.path(),
            "broken",
            &format!(
                "echo x >> '{}'; exit 1",
                dir.path().join("broken-launches").display()
            ),
        );
        let error = diagnostic_within(&broken, &[Duration::from_secs(5); 2]).unwrap_err();
        assert!(!error.to_string().contains("slow to start"), "{error}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("broken-launches"))
                .unwrap()
                .lines()
                .count(),
            1
        );
    }

    #[test]
    fn only_a_path_inside_an_app_bundle_is_one() {
        assert!(inside_app_bundle(Path::new(
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        )));
        assert!(inside_app_bundle(Path::new(
            "/x/ms-playwright/chromium-1/chrome-mac/Chromium.APP/Contents/MacOS/Chromium"
        )));
        assert!(!inside_app_bundle(Path::new("/usr/bin/chromium")));
        assert!(!inside_app_bundle(Path::new(
            "/opt/chrome.application/chrome"
        )));
    }

    #[test]
    fn the_version_probe_runs_the_browser_with_only_version_and_writes_nothing_beside_it() {
        let dir = tempfile::tempdir().unwrap();
        // Fails unless it is asked for exactly `--version`, and records any
        // file it would create in its working directory.
        let program = script(
            dir.path(),
            "browser",
            "[ \"$#\" = 1 ] && [ \"$1\" = --version ] && [ -z \"$(ls -A)\" ] && exit 0; exit 3",
        );
        version_probe(&program).unwrap();
        assert!(std::fs::read_dir(dir.path()).unwrap().count() == 1);
        let failing = script(dir.path(), "broken", "exit 1");
        assert!(version_probe(&failing).is_err());
    }
}
