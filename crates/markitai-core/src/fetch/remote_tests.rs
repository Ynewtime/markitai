//! Remote services against loopback fixtures: the `auto` chain's fallback
//! with each consent setting (a stand-in terminal answers `ask`), the
//! Defuddle and Jina options and pacing, Cloudflare Browser Rendering and
//! Workers AI `toMarkdown`, and what their failures say. No test reads the
//! environment, a `.env` file or the network beyond 127.0.0.1.

use super::cache_tests::{Reply, Server as PageServer, settings};
use super::consent::{ConsentRequest, Gate, RemoteFallback, RemoteNotice};
use super::remote::{Fixture, Limits};
use super::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ---- a loopback service that reads request bodies -------------------------

#[derive(Clone)]
struct Answer {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Answer {
    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: value.to_string().into_bytes(),
        }
    }
    fn text(status: u16, body: &str) -> Self {
        Self {
            status,
            headers: vec![("Content-Type".into(), "text/markdown".into())],
            body: body.as_bytes().to_vec(),
        }
    }
    fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

#[derive(Clone, Debug)]
struct Request {
    method: String,
    target: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

struct Mock {
    origin: String,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

fn read_request(stream: &mut std::net::TcpStream) -> Request {
    let mut data = Vec::new();
    let mut chunk = [0; 4096];
    let head_end = loop {
        if let Some(end) = data.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            break end;
        }
        let size = stream.read(&mut chunk).unwrap();
        assert!(size > 0, "the request ended inside its head");
        data.extend_from_slice(&chunk[..size]);
        assert!(data.len() < 1024 * 1024);
    };
    let head = String::from_utf8(data[..head_end].to_vec()).unwrap();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap().split(' ');
    let (method, target) = (
        first.next().unwrap().to_owned(),
        first.next().unwrap().to_owned(),
    );
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
        .collect();
    let find = |name: &str| {
        headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    let mut rest = data[head_end + 4..].to_vec();
    let mut more = |rest: &mut Vec<u8>| {
        let size = stream.read(&mut chunk).unwrap();
        assert!(size > 0, "the request ended inside its body");
        rest.extend_from_slice(&chunk[..size]);
    };
    let body = if let Some(length) = find("content-length") {
        let length: usize = length.parse().unwrap();
        while rest.len() < length {
            more(&mut rest);
        }
        rest.truncate(length);
        rest
    } else if find("transfer-encoding").is_some_and(|value| value.contains("chunked")) {
        let mut body = Vec::new();
        loop {
            let line_end = loop {
                if let Some(end) = rest.windows(2).position(|bytes| bytes == b"\r\n") {
                    break end;
                }
                more(&mut rest);
            };
            let size =
                usize::from_str_radix(std::str::from_utf8(&rest[..line_end]).unwrap().trim(), 16)
                    .unwrap();
            rest.drain(..line_end + 2);
            while rest.len() < size + 2 {
                more(&mut rest);
            }
            if size == 0 {
                break body;
            }
            body.extend_from_slice(&rest[..size]);
            rest.drain(..size + 2);
        }
    } else {
        Vec::new()
    };
    Request {
        method,
        target,
        headers,
        body,
    }
}

impl Mock {
    fn new(answers: Vec<Answer>) -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (log, stopped) = (Arc::clone(&requests), Arc::clone(&stop));
        let thread = std::thread::spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("loopback service: {error}"),
                };
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let request = read_request(&mut stream);
                let index = {
                    let mut log = log.lock().unwrap();
                    log.push(request);
                    log.len() - 1
                };
                let answer = answers
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| Answer::text(500, "unexpected request"));
                write!(
                    stream,
                    "HTTP/1.1 {} Test\r\nContent-Length: {}\r\nConnection: close\r\n",
                    answer.status,
                    answer.body.len()
                )
                .unwrap();
                for (name, value) in &answer.headers {
                    write!(stream, "{name}: {value}\r\n").unwrap();
                }
                stream.write_all(b"\r\n").unwrap();
                stream.write_all(&answer.body).unwrap();
            }
        });
        Self {
            origin,
            requests,
            stop,
            thread: Some(thread),
        }
    }
    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let result = self.thread.take().unwrap().join();
        if !std::thread::panicking() {
            result.expect("loopback service failed");
        }
    }
}

// ---- a stand-in terminal ---------------------------------------------------

#[derive(Default)]
struct Terminal {
    asked: AtomicUsize,
    requests: Mutex<Vec<ConsentRequest>>,
    notices: Mutex<Vec<RemoteNotice>>,
}

/// A gate whose host wrote `always` itself (`explicitly_always`), and whose
/// terminal answers `answer` (`None`: nobody can be asked).
fn gate(
    terminal: &Arc<Terminal>,
    explicitly_always: bool,
    answer: Option<bool>,
    notices: Option<std::path::PathBuf>,
) -> Gate {
    let (asking, telling) = (Arc::clone(terminal), Arc::clone(terminal));
    Gate::new(
        Some(RemoteFallback {
            explicitly_always,
            explicit_fallback_patterns: false,
            ask: answer.map(|answer| {
                Box::new(move |request: &ConsentRequest| {
                    asking.asked.fetch_add(1, Ordering::SeqCst);
                    asking.requests.lock().unwrap().push(request.clone());
                    answer
                }) as Box<dyn Fn(&ConsentRequest) -> bool + Send + Sync>
            }),
            notify: Box::new(move |notice| {
                telling.notices.lock().unwrap().push(notice.clone());
                true
            }),
        }),
        notices,
    )
}

/// A gate whose host says only whether the user wrote `fetch.fallback_patterns`.
fn patterns_gate(written: bool) -> Gate {
    Gate::new(
        Some(RemoteFallback {
            explicitly_always: false,
            explicit_fallback_patterns: written,
            ask: None,
            notify: Box::new(|_| true),
        }),
        None,
    )
}

#[test]
fn only_a_written_fallback_patterns_list_makes_a_domain_browser_first() {
    let post = Url::parse("https://x.com/user/status/1").unwrap();
    // The default configuration lists x.com, but nobody wrote it: static
    // first, with or without a host.
    let defaults = config::defaults();
    for gate in [Gate::new(None, None), patterns_gate(false)] {
        let fixture = Fixture::new(gate);
        let services = fixture.services("http://127.0.0.1:9");
        let steps = auto_order(&post, &defaults, false, &services);
        assert_eq!(steps[..2], [policy::Step::Static, policy::Step::Browser]);
    }
    // A list the user wrote applies to the domains it names.
    let mut cfg = config::defaults();
    cfg["fetch"]["fallback_patterns"] = json!(["x.com"]);
    let fixture = Fixture::new(patterns_gate(true));
    let services = fixture.services("http://127.0.0.1:9");
    let steps = auto_order(&post, &cfg, false, &services);
    assert_eq!(steps[..2], [policy::Step::Browser, policy::Step::Static]);
    let other = Url::parse("https://www.linkedin.com/in/someone").unwrap();
    assert_eq!(
        auto_order(&other, &cfg, false, &services)[0],
        policy::Step::Static
    );
}

fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

fn auto_settings() -> (tempfile::TempDir, Value) {
    let (directory, mut cfg) = settings();
    cfg["fetch"]["strategy"] = json!("auto");
    (directory, cfg)
}

fn markdown(outcome: &FetchOutcome) -> &str {
    &outcome.document().markdown
}

fn jina_page(content: &str) -> Answer {
    Answer::json(
        200,
        json!({"code": 200, "status": 20000, "data": {"title": " Remote title ", "content": content, "url": "https://example.com/"}}),
    )
}

const REFUSED: &str = "HTTP 403 for ";

// ---- the auto chain --------------------------------------------------------

#[test]
fn an_explicit_always_falls_back_in_order_with_the_configured_options_and_discloses_once() {
    let (directory, mut cfg) = auto_settings();
    cfg["fetch"]["remote_consent"] = json!("always");
    cfg["fetch"]["jina"] = json!({
        "api_key": "env:TEST_JINA_KEY", "timeout": 5, "rpm": 50, "no_cache": true,
        "target_selector": "article.post", "wait_for_selector": "#main"
    });
    let page = PageServer::new(vec![
        Reply::text("no").status(403),
        Reply::text("no").status(403),
    ]);
    let service = Mock::new(vec![
        Answer::text(503, "busy"),
        jina_page("# Remote text\n\nRead through Jina."),
        Answer::text(503, "busy"),
        jina_page("# Second\n\nAgain."),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(
        &terminal,
        true,
        None,
        Some(directory.path().join("notices")),
    ));
    let mut services = fixture.services(&service.origin);
    services.vars = vars(&[("TEST_JINA_KEY", "jina-secret-0123456789")]);
    let source = page.url("/article?id=7");

    let outcome = fetch_with_services(&source, &cfg, None, true, None, &services).unwrap();
    assert_eq!(markdown(&outcome), "# Remote text\n\nRead through Jina.");
    assert_eq!(outcome.document().metadata["fetch_strategy"], "jina");
    assert_eq!(outcome.document().metadata["title"], "Remote title");
    assert!(!outcome.cache_hit);

    let requests = service.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "GET");
    assert_eq!(
        requests[0].target,
        format!(
            "/defuddle/{}",
            url::form_urlencoded::byte_serialize(source.as_bytes()).collect::<String>()
        )
    );
    let jina = &requests[1];
    assert_eq!(jina.target, format!("/jina/{source}"));
    assert_eq!(jina.header("Accept"), Some("application/json"));
    assert_eq!(
        jina.header("Authorization"),
        Some("Bearer jina-secret-0123456789")
    );
    assert_eq!(jina.header("X-No-Cache"), Some("true"));
    assert_eq!(jina.header("X-Target-Selector"), Some("article.post"));
    assert_eq!(jina.header("X-Wait-For-Selector"), Some("#main"));
    // Cloudflare has no credentials, so it is neither tried nor named.
    assert_eq!(
        *terminal.notices.lock().unwrap(),
        [RemoteNotice::Disclosure {
            services: vec!["defuddle", "jina"]
        }]
    );
    assert!(directory.path().join("notices/remote-fetch").is_file());
    assert_eq!(terminal.asked.load(Ordering::SeqCst), 0);

    // A second page in the same process: no second notice.
    let second = fetch_with_services(&page.url("/other"), &cfg, None, true, None, &services);
    assert_eq!(markdown(&second.unwrap()), "# Second\n\nAgain.");
    assert_eq!(terminal.notices.lock().unwrap().len(), 1);
    assert_eq!(page.requests().len(), 2);
    // Remote readings are not stored in the page cache.
    assert!(!directory.path().join("fetch_cache.db").exists());
}

#[test]
fn the_default_always_and_never_keep_auto_local_with_the_same_failure() {
    for consent in [None, Some("never"), Some("always")] {
        let (_directory, mut cfg) = auto_settings();
        if let Some(consent) = consent {
            cfg["fetch"]["remote_consent"] = json!(consent);
        }
        let page = PageServer::new(vec![Reply::text("no").status(403)]);
        let service = Mock::new(Vec::new());
        let terminal = Arc::new(Terminal::default());
        // The configuration's `always` is the default unless the host says
        // the user wrote it.
        let fixture = Fixture::new(gate(&terminal, consent == Some("never"), Some(true), None));
        let services = fixture.services(&service.origin);
        let Err(Error::Fetch(message)) =
            fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services)
        else {
            panic!("a refusal stays a failure");
        };
        assert!(message.starts_with(REFUSED), "{message}");
        assert!(!message.contains("remote"), "{message}");
        assert!(service.requests().is_empty());
        assert_eq!(terminal.asked.load(Ordering::SeqCst), 0);
        assert!(terminal.notices.lock().unwrap().is_empty());
    }
}

#[test]
fn ask_asks_once_per_process_and_a_no_keeps_every_later_page_local() {
    let (_directory, mut cfg) = auto_settings();
    cfg["fetch"]["remote_consent"] = json!("ask");
    let page = PageServer::new(vec![
        Reply::text("no").status(403),
        Reply::text("no").status(429),
    ]);
    let service = Mock::new(Vec::new());
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, Some(false), None));
    let services = fixture.services(&service.origin);
    for path in ["/one?token=x", "/two"] {
        let error = fetch_with_services(&page.url(path), &cfg, None, true, None, &services);
        assert!(error.is_err());
    }
    // The first page carries a token, so it never reaches the question.
    assert_eq!(terminal.asked.load(Ordering::SeqCst), 1);
    let request = terminal.requests.lock().unwrap()[0].clone();
    assert_eq!(request.url, page.url("/two"));
    assert_eq!(request.services, ["defuddle", "jina"]);
    assert!(service.requests().is_empty());

    // Yes: the services are tried for this and every later page.
    let page = PageServer::new(vec![
        Reply::text("no").status(403),
        Reply::text("no").status(403),
    ]);
    let service = Mock::new(vec![
        Answer::text(
            200,
            "---\ntitle: From defuddle\n---\n\n# Body\n\nDefuddle text.",
        ),
        Answer::text(200, "# Again\n\nMore defuddle text."),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, Some(true), None));
    let services = fixture.services(&service.origin);
    let first = fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services).unwrap();
    assert_eq!(markdown(&first), "# Body\n\nDefuddle text.");
    assert_eq!(first.document().metadata["title"], "From defuddle");
    assert_eq!(first.document().metadata["fetch_strategy"], "defuddle");
    let second = fetch_with_services(&page.url("/b"), &cfg, None, true, None, &services).unwrap();
    assert_eq!(markdown(&second), "# Again\n\nMore defuddle text.");
    assert_eq!(terminal.asked.load(Ordering::SeqCst), 1);
    assert!(terminal.notices.lock().unwrap().is_empty());
}

#[test]
fn ask_without_a_terminal_skips_remote_services_with_one_hint() {
    let (_directory, mut cfg) = auto_settings();
    cfg["fetch"]["remote_consent"] = json!("ask");
    let page = PageServer::new(vec![
        Reply::text("no").status(403),
        Reply::text("no").status(403),
    ]);
    let service = Mock::new(Vec::new());
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let services = fixture.services(&service.origin);
    for path in ["/a", "/b"] {
        let Err(Error::Fetch(message)) =
            fetch_with_services(&page.url(path), &cfg, None, true, None, &services)
        else {
            panic!("no consent, no remote reading");
        };
        assert!(message.starts_with(REFUSED), "{message}");
    }
    assert_eq!(*terminal.notices.lock().unwrap(), [RemoteNotice::NotAsked]);
    assert!(service.requests().is_empty());
}

#[test]
fn a_missing_page_local_only_patterns_and_private_names_never_reach_a_service() {
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, true, None, None));
    let service = Mock::new(Vec::new());

    let (_directory, mut cfg) = auto_settings();
    cfg["fetch"]["remote_consent"] = json!("always");
    let page = PageServer::new(vec![Reply::text("gone").status(404)]);
    let services = fixture.services(&service.origin);
    let Err(Error::Fetch(message)) =
        fetch_with_services(&page.url("/gone"), &cfg, None, true, None, &services)
    else {
        panic!("a missing page fails");
    };
    assert!(message.starts_with("HTTP 404 for "), "{message}");

    cfg["fetch"]["policy"]["local_only_patterns"] = json!(["127.0.0.1"]);
    let page = PageServer::new(vec![Reply::text("no").status(403)]);
    assert!(fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services).is_err());

    cfg["fetch"]["policy"]["local_only_patterns"] = json!([]);
    let services = Services {
        vars: vars(&[("NO_PROXY", "localhost, 127.0.0.1")]),
        ..fixture.services(&service.origin)
    };
    let page = PageServer::new(vec![Reply::text("no").status(403)]);
    assert!(fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services).is_err());

    let services = Services {
        private_name: policy::private_name,
        ..fixture.services(&service.origin)
    };
    let page = PageServer::new(vec![Reply::text("no").status(403)]);
    assert!(fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services).is_err());

    // A host that resolves to a private address is refused at the first
    // remote step, and the failure says so.
    let services = Services {
        public_addresses: |_| {
            Err(Error::Fetch(
                "Private URLs cannot be sent to remote extraction services".into(),
            ))
        },
        ..fixture.services(&service.origin)
    };
    let page = PageServer::new(vec![Reply::text("no").status(403)]);
    let Err(Error::Fetch(message)) =
        fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services)
    else {
        panic!("a private target fails");
    };
    assert!(message.starts_with(REFUSED), "{message}");
    assert!(message.ends_with("remote services failed as well (Private URLs cannot be sent to remote extraction services)"), "{message}");
    assert!(service.requests().is_empty());
    assert!(terminal.notices.lock().unwrap().is_empty());
}

#[test]
fn priorities_and_hops_decide_which_strategies_run() {
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, true, None, None));
    let (_directory, mut cfg) = auto_settings();
    cfg["fetch"]["remote_consent"] = json!("always");
    cfg["fetch"]["policy"]["strategy_priority"] = json!(["jina", "static"]);
    let page = PageServer::new(vec![Reply::text("static page")]);
    let service = Mock::new(vec![
        jina_page("# First\n\nJina first."),
        Answer::text(503, "down"),
    ]);
    let services = fixture.services(&service.origin);
    let outcome = fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services).unwrap();
    assert_eq!(markdown(&outcome), "# First\n\nJina first.");
    assert!(page.requests().is_empty());
    // The service fails: static is next.
    let outcome = fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services).unwrap();
    assert_eq!(markdown(&outcome), "static page");
    assert_eq!(page.requests().len(), 1);
    // One hop: only the first strategy runs.
    cfg["fetch"]["policy"]["max_strategy_hops"] = json!(1);
    let error = fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services);
    let Err(Error::Fetch(message)) = error else {
        panic!("the one strategy failed");
    };
    assert!(
        message.starts_with("No strategy could read the page: HTTP 500 from the jina service"),
        "{message}"
    );
    assert_eq!(page.requests().len(), 1);
}

#[test]
fn a_script_rendered_shell_without_a_browser_is_read_remotely_or_kept() {
    let shell = format!(
        "<!DOCTYPE html><html><head><title>App to Read</title></head><body><h1>App to Read</h1><div id=\"root\"></div><script>{}</script></body></html>",
        "var data = 1;".repeat(400)
    );
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, true, None, None));
    let (_directory, mut cfg) = auto_settings();
    cfg["fetch"]["remote_consent"] = json!("always");
    cfg["cache"]["enabled"] = json!(false);
    let page = PageServer::new(vec![Reply::html(&shell), Reply::html(&shell)]);
    let service = Mock::new(vec![
        Answer::text(200, "# The app\n\nRendered elsewhere."),
        Answer::text(503, "down"),
        Answer::text(503, "down"),
    ]);
    let services = fixture.services(&service.origin);
    let outcome =
        fetch_with_services(&page.url("/app"), &cfg, None, true, None, &services).unwrap();
    assert_eq!(markdown(&outcome), "# The app\n\nRendered elsewhere.");
    let kept = fetch_with_services(&page.url("/app"), &cfg, None, true, None, &services).unwrap();
    assert!(markdown(&kept).contains("App to Read"));
    let warnings = &kept.document().warnings;
    assert_eq!(warnings[0], JS_SHELL);
    assert!(
        warnings[1].starts_with("Remote extraction failed (HTTP 503 from the defuddle service"),
        "{warnings:?}"
    );
}

#[test]
fn remote_failures_follow_the_local_one_without_tokens_or_account_ids() {
    let token = "cf-token-SECRET-0123456789";
    let account = "0123456789abcdef0123456789abcdef";
    let (_directory, mut cfg) = auto_settings();
    cfg["fetch"]["remote_consent"] = json!("always");
    cfg["fetch"]["cloudflare"]["account_id"] = json!("env:TEST_CF_ACCOUNT");
    let page = PageServer::new(vec![Reply::text("no").status(403)]);
    let service = Mock::new(vec![
        Answer::text(429, "slow down"),
        Answer::json(
            451,
            json!({"code": 451, "name": "SecurityCompromiseError", "message": "Anonymous access to domain example.com blocked until later"}),
        ),
        Answer::json(
            403,
            json!({"success": false, "errors": [{"code": 10000, "message": format!("Authentication error for account {account} with {token}")}]}),
        ),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, true, None, None));
    let mut services = fixture.services(&service.origin);
    services.vars = vars(&[
        ("CLOUDFLARE_API_TOKEN", token),
        ("TEST_CF_ACCOUNT", account),
    ]);
    let Err(Error::Fetch(message)) =
        fetch_with_services(&page.url("/a"), &cfg, None, true, None, &services)
    else {
        panic!("every strategy failed");
    };
    assert!(message.starts_with(REFUSED), "{message}");
    assert!(message.contains("; remote services failed as well (HTTP 429 from the defuddle service: rate limited; try again later; HTTP 451 from the jina service: Anonymous access to domain example.com blocked until later (code 451); set fetch.jina.api_key or JINA_API_KEY"), "{message}");
    assert!(
        message.contains("HTTP 403 from the cloudflare service: Authentication error for account REDACTED with REDACTED (code 10000); check fetch.cloudflare.api_token"),
        "{message}"
    );
    assert!(
        !message.contains(token) && !message.contains(account),
        "{message}"
    );
    assert!(!message.contains(&service.origin), "{message}");
    let requests = service.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[2].target,
        format!("/cloudflare/accounts/{account}/browser-rendering/content")
    );
    assert_eq!(
        terminal.notices.lock().unwrap()[0],
        RemoteNotice::Disclosure {
            services: vec!["defuddle", "jina", "cloudflare"]
        }
    );
}

// ---- selected remote strategies -------------------------------------------

#[test]
fn a_selected_strategy_follows_consent_and_local_only_patterns() {
    let (_directory, mut cfg) = settings();
    cfg["fetch"]["strategy"] = json!("jina");
    cfg["fetch"]["remote_consent"] = json!("ask");
    let service = Mock::new(vec![
        jina_page("# Chosen\n\nJina."),
        jina_page("# Asked\n\nJina."),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let services = fixture.services(&service.origin);
    let source = "https://example.com/post";
    // `-s jina` for this run is the answer to `ask`.
    let chosen = fetch_with_services(source, &cfg, Some("jina"), true, None, &services).unwrap();
    assert_eq!(markdown(&chosen), "# Chosen\n\nJina.");
    // From the configuration, with nobody to ask: refused.
    let Err(Error::Fetch(message)) = fetch_with_services(source, &cfg, None, true, None, &services)
    else {
        panic!("ask without an answer refuses");
    };
    assert!(message.starts_with(consent::DISABLED), "{message}");
    // With a terminal that says yes.
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, Some(true), None));
    let services = fixture.services(&service.origin);
    let asked = fetch_with_services(source, &cfg, None, true, None, &services).unwrap();
    assert_eq!(markdown(&asked), "# Asked\n\nJina.");
    assert_eq!(terminal.requests.lock().unwrap()[0].services, ["jina"]);
    // A local-only pattern keeps a configured strategy away; `-s` overrides it.
    cfg["fetch"]["remote_consent"] = json!("always");
    cfg["fetch"]["policy"]["local_only_patterns"] = json!(["example.com"]);
    let Err(Error::Fetch(message)) = fetch_with_services(source, &cfg, None, true, None, &services)
    else {
        panic!("local-only refuses a configured remote strategy");
    };
    assert!(message.contains("local_only_patterns"), "{message}");
    assert_eq!(service.requests().len(), 2);
    // Credential material never leaves, whatever was chosen.
    let Err(Error::Fetch(message)) = fetch_with_services(
        "https://example.com/reset?token=abc",
        &cfg,
        Some("jina"),
        true,
        None,
        &services,
    ) else {
        panic!("credentials never leave");
    };
    assert_eq!(
        message,
        "Credentialed URLs cannot be sent to remote extraction services"
    );
    let off = Services {
        vars: vars(&[("MARKITAI_NO_REMOTE_FETCH", "1")]),
        ..fixture.services(&service.origin)
    };
    let Err(Error::Fetch(message)) =
        fetch_with_services(source, &cfg, Some("jina"), true, None, &off)
    else {
        panic!("the hard opt-out refuses");
    };
    assert!(message.starts_with(consent::DISABLED), "{message}");
    assert_eq!(service.requests().len(), 2);
}

#[test]
fn defuddle_requests_are_paced_by_their_rpm() {
    let (_directory, mut cfg) = settings();
    cfg["fetch"]["strategy"] = json!("defuddle");
    cfg["fetch"]["defuddle"] = json!({"rpm": 1, "timeout": 5});
    let service = Mock::new(vec![
        Answer::text(200, "# One\n\nFirst."),
        Answer::text(200, "# Two\n\nSecond."),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture {
        limits: Limits::new(Duration::from_millis(400)),
        ..Fixture::new(gate(&terminal, false, None, None))
    };
    let services = fixture.services(&service.origin);
    let started = std::time::Instant::now();
    for expected in ["# One\n\nFirst.", "# Two\n\nSecond."] {
        let outcome =
            fetch_with_services("https://example.com/p", &cfg, None, true, None, &services)
                .unwrap();
        assert_eq!(markdown(&outcome), expected);
    }
    assert!(
        started.elapsed() >= Duration::from_millis(350),
        "{:?}",
        started.elapsed()
    );
}

#[test]
fn the_limiter_admits_a_window_of_requests_and_then_waits() {
    let limits = Limits::new(Duration::from_millis(300));
    let started = std::time::Instant::now();
    limits.acquire("jina", 2);
    limits.acquire("jina", 2);
    limits.acquire("defuddle", 2);
    assert!(started.elapsed() < Duration::from_millis(200));
    limits.acquire("jina", 2);
    assert!(started.elapsed() >= Duration::from_millis(250));
}

// ---- Cloudflare -----------------------------------------------------------

fn cloudflare_cfg() -> Value {
    let (_directory, mut cfg) = settings();
    cfg["fetch"]["strategy"] = json!("cloudflare");
    cfg["fetch"]["cloudflare"] = json!({
        "api_token": "env:TEST_CF_TOKEN", "account_id": "acc0123", "timeout": 45000,
        "wait_until": "load", "cache_ttl": 5, "reject_resource_patterns": ["/\\.png$/"],
        "user_agent": "Test Agent", "cookies": [{"name": "a", "value": "b", "domain": "example.com"}],
        "wait_for_selector": "main", "http_credentials": {"username": "reader", "password": "env:TEST_SITE_PASSWORD"}
    });
    cfg
}

#[test]
fn cloudflare_renders_with_its_options_and_repeats_a_429() {
    let cfg = cloudflare_cfg();
    let html = "<html><head><title>Rendered page</title></head><body><article><h1>Rendered page</h1><p>Text that Cloudflare's browser rendered for the native extraction.</p></article></body></html>";
    let service = Mock::new(vec![
        Answer::json(429, json!({"success": false})),
        Answer::json(200, json!({"success": true, "result": html}))
            .header("X-Browser-Ms-Used", "1234"),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let mut services = fixture.services(&service.origin);
    services.vars = vars(&[
        ("TEST_CF_TOKEN", "cf-test-token-0123"),
        ("TEST_SITE_PASSWORD", "site-password"),
    ]);
    let source = "https://example.com/article";
    let outcome =
        fetch_with_services(source, &cfg, Some("cloudflare"), true, None, &services).unwrap();
    assert!(markdown(&outcome).contains("Text that Cloudflare's browser rendered"));
    let metadata = &outcome.document().metadata;
    assert_eq!(metadata["fetch_strategy"], "cloudflare");
    assert_eq!(metadata["renderer"], "cloudflare");
    assert_eq!(metadata["browser_ms_used"], "1234");
    assert_eq!(metadata["title"], "Rendered page");
    let requests = service.requests();
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert_eq!(request.method, "POST");
        assert_eq!(
            request.target,
            "/cloudflare/accounts/acc0123/browser-rendering/content?cacheTTL=5"
        );
        assert_eq!(
            request.header("Authorization"),
            Some("Bearer cf-test-token-0123")
        );
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(
            body,
            json!({
                "url": source,
                "gotoOptions": {"timeout": 45000, "waitUntil": "load"},
                "rejectRequestPattern": ["/\\.png$/"],
                "userAgent": "Test Agent",
                "cookies": [{"name": "a", "value": "b", "domain": "example.com"}],
                "waitForSelector": {"selector": "main"},
                "authenticate": {"username": "reader", "password": "site-password"}
            })
        );
    }
}

#[test]
fn cloudflare_failures_say_what_to_do_and_never_show_the_account() {
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let source = "https://example.com/article";
    // No credentials: the setup is described, nothing is sent.
    let service = Mock::new(Vec::new());
    let mut cfg = cloudflare_cfg();
    cfg["fetch"]["cloudflare"]["api_token"] = Value::Null;
    let services = fixture.services(&service.origin);
    let Err(Error::Config(message)) =
        fetch_with_services(source, &cfg, Some("cloudflare"), true, None, &services)
    else {
        panic!("missing credentials are a configuration error");
    };
    assert!(
        message.contains("CLOUDFLARE_API_TOKEN") && message.contains("Browser Rendering / Edit"),
        "{message}"
    );
    // An env: reference to a variable that is not set names the variable.
    let cfg = cloudflare_cfg();
    let Err(Error::Config(message)) =
        fetch_with_services(source, &cfg, Some("cloudflare"), true, None, &services)
    else {
        panic!("a missing variable is a configuration error");
    };
    assert_eq!(message, "Environment variable not found: TEST_CF_TOKEN");
    assert!(service.requests().is_empty());

    // Rate limited three times, then an unsuccessful envelope.
    let service = Mock::new(vec![
        Answer::json(429, json!({})),
        Answer::json(429, json!({})),
        Answer::json(429, json!({})),
        Answer::json(
            200,
            json!({"success": false, "errors": [{"code": 7003, "message": "No route for account acc0123"}]}),
        ),
    ]);
    let mut services = fixture.services(&service.origin);
    services.vars = vars(&[
        ("TEST_CF_TOKEN", "cf-test-token-0123"),
        ("TEST_SITE_PASSWORD", "pw"),
    ]);
    let Err(Error::Fetch(message)) =
        fetch_with_services(source, &cfg, Some("cloudflare"), true, None, &services)
    else {
        panic!("rate limited");
    };
    assert_eq!(
        message,
        "HTTP 429 from the cloudflare service: rate limited; try again later"
    );
    let Err(Error::Fetch(message)) =
        fetch_with_services(source, &cfg, Some("cloudflare"), true, None, &services)
    else {
        panic!("an unsuccessful envelope fails");
    };
    assert_eq!(
        message,
        "The cloudflare service could not render the page: No route for account REDACTED (code 7003)"
    );
    assert_eq!(service.requests().len(), 4);

    // An account id that is not one is refused before any request.
    let mut cfg = cloudflare_cfg();
    cfg["fetch"]["cloudflare"]["account_id"] = json!("acc/../../other?x=1");
    let Err(Error::Config(message)) =
        fetch_with_services(source, &cfg, Some("cloudflare"), true, None, &services)
    else {
        panic!("a malformed account id is refused");
    };
    assert!(!message.contains("other"), "{message}");
    assert_eq!(service.requests().len(), 4);
}

#[test]
fn a_verification_page_cloudflare_rendered_is_a_failure() {
    let cfg = cloudflare_cfg();
    let challenge = "<html><head><title>Just a moment...</title></head><body><div id=\"cf-browser-verification\"></div><p>Checking your browser before accessing example.com.</p></body></html>";
    let service = Mock::new(vec![Answer::json(
        200,
        json!({"success": true, "result": challenge}),
    )]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let mut services = fixture.services(&service.origin);
    services.vars = vars(&[("TEST_CF_TOKEN", "t0123"), ("TEST_SITE_PASSWORD", "pw")]);
    let Err(Error::Fetch(message)) = fetch_with_services(
        "https://example.com/article",
        &cfg,
        Some("cloudflare"),
        true,
        None,
        &services,
    ) else {
        panic!("a challenge is not content");
    };
    assert!(message.contains("challenge"), "{message}");
}

#[test]
fn workers_ai_converts_a_local_file_and_refuses_without_consent_or_credentials() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("report.pdf");
    std::fs::write(&file, b"%PDF-1.4 fixture bytes").unwrap();
    let image = directory.path().join("photo.png");
    std::fs::write(&image, b"\x89PNG fixture").unwrap();
    let mut cfg = config::defaults();
    cfg["fetch"]["cloudflare"] = json!({"convert_enabled": true, "account_id": "acc0123"});
    assert!(cloudflare::converts(&cfg, &file, "pdf"));
    assert!(cloudflare::converts(&cfg, &image, "PNG"));
    assert!(!cloudflare::converts(&cfg, &file, "pptx"));
    assert!(!cloudflare::converts(&cfg, directory.path(), "numbers"));
    let service = Mock::new(vec![
        Answer::json(
            200,
            json!({"success": true, "result": [{"name": "report.pdf", "mimeType": "application/pdf", "format": "markdown", "tokens": 42, "data": "# Report\n\nConverted by Workers AI."}]}),
        ),
        Answer::json(
            200,
            json!({"success": true, "result": [{"name": "photo.png", "format": "markdown", "data": "A photo."}]}),
        ),
        Answer::json(
            200,
            json!({"success": true, "result": [{"name": "report.pdf", "format": "error", "error": "Conversion failed for acc0123"}]}),
        ),
        Answer::json(
            401,
            json!({"success": false, "errors": [{"code": 10000, "message": "Authentication error"}]}),
        ),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let mut services = fixture.services(&service.origin);
    services.vars = vars(&[("CLOUDFLARE_API_TOKEN", "workers-token-0123")]);

    let document = cloudflare::convert_file_with(&file, "pdf", &cfg, &services).unwrap();
    assert_eq!(document.markdown, "# Report\n\nConverted by Workers AI.");
    assert_eq!(document.metadata["converter"], "cloudflare-tomarkdown");
    assert_eq!(document.metadata["tokens"], 42);
    assert!(document.warnings.is_empty());
    let photo = cloudflare::convert_file_with(&image, "png", &cfg, &services).unwrap();
    assert!(photo.warnings[0].contains("Neurons"));
    let Err(Error::Conversion(message)) =
        cloudflare::convert_file_with(&file, "pdf", &cfg, &services)
    else {
        panic!("an error result fails");
    };
    assert_eq!(
        message,
        "The cloudflare service could not convert the file: Conversion failed for REDACTED"
    );
    let Err(Error::Conversion(message)) =
        cloudflare::convert_file_with(&file, "pdf", &cfg, &services)
    else {
        panic!("a refused token fails");
    };
    assert!(message.starts_with("HTTP 401 from the cloudflare service: Authentication error (code 10000); check fetch.cloudflare.api_token"), "{message}");

    let requests = service.requests();
    assert_eq!(requests.len(), 4);
    let upload = &requests[0];
    assert_eq!(upload.method, "POST");
    assert_eq!(upload.target, "/cloudflare/accounts/acc0123/ai/tomarkdown");
    assert_eq!(
        upload.header("Authorization"),
        Some("Bearer workers-token-0123")
    );
    assert!(
        upload
            .header("Content-Type")
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
    );
    let body = String::from_utf8_lossy(&upload.body);
    assert!(
        body.contains("name=\"files\"; filename=\"report.pdf\""),
        "{body}"
    );
    assert!(body.contains("Content-Type: application/pdf"), "{body}");
    assert!(body.contains("%PDF-1.4 fixture bytes"), "{body}");
    let photo_upload = String::from_utf8_lossy(&requests[1].body);
    assert!(
        photo_upload.contains("Content-Type: image/png"),
        "{photo_upload}"
    );

    // Never, the hard opt-out and missing credentials send nothing.
    cfg["fetch"]["remote_consent"] = json!("never");
    assert!(matches!(
        cloudflare::convert_file_with(&file, "pdf", &cfg, &services),
        Err(Error::Config(message)) if message.contains("--no-remote-fetch")
    ));
    cfg["fetch"]["remote_consent"] = json!("always");
    let off = Services {
        vars: vars(&[
            ("CLOUDFLARE_API_TOKEN", "workers-token-0123"),
            ("MARKITAI_NO_REMOTE_FETCH", "true"),
        ]),
        ..fixture.services(&service.origin)
    };
    assert!(matches!(
        cloudflare::convert_file_with(&file, "pdf", &cfg, &off),
        Err(Error::Config(_))
    ));
    let bare = fixture.services(&service.origin);
    assert!(matches!(
        cloudflare::convert_file_with(&file, "pdf", &cfg, &bare),
        Err(Error::Config(message)) if message.contains("CLOUDFLARE_ACCOUNT_ID")
    ));
    assert_eq!(service.requests().len(), 4);
}
