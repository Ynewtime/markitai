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
fn a_selected_strategy_announces_its_service_once_and_only_when_a_url_leaves() {
    let (_directory, mut cfg) = settings();
    cfg["fetch"]["strategy"] = json!("jina");
    let service = Mock::new(vec![
        jina_page("# One\n\nJina."),
        jina_page("# Two\n\nJina."),
    ]);
    let home = tempfile::tempdir().unwrap();
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, Some(home.path().into())));
    let services = fixture.services(&service.origin);
    // Refused before anything is sent: no notice.
    assert!(
        fetch_with_services(
            "https://example.com/reset?token=abc",
            &cfg,
            Some("jina"),
            true,
            None,
            &services,
        )
        .is_err()
    );
    assert!(terminal.notices.lock().unwrap().is_empty());
    assert!(!home.path().join("remote-strategy-jina").exists());
    // Two pages through the strategy: one notice, recorded for the home.
    for page in ["https://example.com/a", "https://example.com/b"] {
        fetch_with_services(page, &cfg, None, true, None, &services).unwrap();
    }
    assert_eq!(
        *terminal.notices.lock().unwrap(),
        [RemoteNotice::Strategy { service: "jina" }]
    );
    assert!(home.path().join("remote-strategy-jina").is_file());
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
fn services_receive_the_url_that_was_checked_not_the_typed_text() {
    let source = "https://Example.COM/a b?q=1";
    let checked = Url::parse(source).unwrap();
    assert_ne!(checked.as_str(), source);
    let html = "<html><head><title>Page</title></head><body><article><h1>Page</h1><p>Text that the remote service rendered for the extraction.</p></article></body></html>";
    for strategy in ["defuddle", "jina", "cloudflare"] {
        let (answer, mut cfg) = match strategy {
            "defuddle" => (Answer::text(200, "# Page\n\nText."), settings().1),
            "jina" => (jina_page("# Page\n\nText."), settings().1),
            _ => (
                Answer::json(200, json!({"success": true, "result": html})),
                cloudflare_cfg(),
            ),
        };
        cfg["fetch"]["strategy"] = json!(strategy);
        let service = Mock::new(vec![answer]);
        let terminal = Arc::new(Terminal::default());
        let fixture = Fixture::new(gate(&terminal, false, None, None));
        let mut services = fixture.services(&service.origin);
        services.vars = vars(&[
            ("TEST_CF_TOKEN", "cf-test-token-0123"),
            ("TEST_SITE_PASSWORD", "site-password"),
        ]);
        fetch_with_services(source, &cfg, Some(strategy), true, None, &services).unwrap();
        let request = &service.requests()[0];
        match strategy {
            "defuddle" => assert_eq!(
                request.target,
                format!(
                    "/defuddle/{}",
                    url::form_urlencoded::byte_serialize(checked.as_str().as_bytes())
                        .collect::<String>()
                )
            ),
            "jina" => assert_eq!(request.target, format!("/jina/{checked}")),
            _ => {
                let body: Value = serde_json::from_slice(&request.body).unwrap();
                assert_eq!(body["url"], checked.as_str());
            }
        }
    }
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
            .header("X-Browser-Ms-Used", "2378.702880859375"),
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
    // A whole number of milliseconds, written as a number like duration_ms.
    assert_eq!(metadata["browser_ms_used"], json!(2379));
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
    assert!(
        message.starts_with(
            "The cloudflare service was shown a challenge page instead of the content; the site turns automated readers away; open the page in your browser and save it"
        ),
        "{message}"
    );
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

// ---- remote readings that are the site's refusal --------------------------

const ZHIHU_QUESTION: &str = "https://www.zhihu.com/question/19550225";
const ZHIHU_LOGIN: &str = "请您登录后查看更多专业优质内容。";

/// Jina's answer for a Zhihu question in the real-service check of 2d01ccb
/// (2026-10-02, `fetch.remote_consent: always`): Zhihu's security check,
/// which was written as the page. `title: None` leaves only the login request.
fn zhihu_login_wall(title: Option<&str>) -> Answer {
    let mut data = json!({
        "description": "",
        "url": ZHIHU_QUESTION,
        "content": format!("![Image 1](https://www.zhihu.com/question/19550225)\n\n![Image 2: ZhiHu logo](https://static.zhihu.com/heifetz/assets/wechat-share-logo.39ea9ecd.png)\n\n{ZHIHU_LOGIN}"),
        "httpStatus": 200,
        "httpStatusText": "OK",
    });
    if let Some(title) = title {
        data["title"] = json!(title);
    }
    Answer::json(200, json!({"code": 200, "status": 20000, "data": data}))
}

/// The local refusal the real check recorded for the same question.
fn zhihu_refusal(url: &Url) -> Error {
    http_failure(
        reqwest::StatusCode::FORBIDDEN,
        None,
        url,
        &Evidence::default(),
    )
}

#[test]
fn a_zhihu_login_wall_from_jina_fails_with_the_site_aware_refusal_and_every_service_tried() {
    let (_directory, mut cfg) = auto_settings();
    cfg["fetch"]["remote_consent"] = json!("always");
    let service = Mock::new(vec![
        Answer::text(502, "bad gateway"),
        zhihu_login_wall(Some("安全验证 - 知乎")),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, true, None, None));
    let services = fixture.services(&service.origin);
    let url = Url::parse(ZHIHU_QUESTION).unwrap();
    // The default order for the page, with the static request answered as
    // Zhihu answered it (no request leaves the machine).
    let steps = auto_order(&url, &cfg, false, &services);
    let mut local = Some(zhihu_refusal(&url));
    let result = chain::run(
        &steps,
        chain::Attempts {
            static_fetch: &mut || Err(local.take().expect("one static request")),
            browser_ready: &|| false,
            render: &mut |_| unreachable!("the static refusal ends the local steps"),
            remote_ready: &mut |service| match service {
                Service::Cloudflare => chain::Readiness::SkipService,
                _ => chain::Readiness::Ready,
            },
            remote: &mut |service| {
                remote::fetch(service, ZHIHU_QUESTION, &url, &cfg, &services).map(remote::outcome)
            },
            what_works: sites::what_works(&url),
        },
    );
    let Err(Error::Fetch(message)) = result else {
        panic!("Zhihu's security check is not the page");
    };
    assert_eq!(
        message,
        format!(
            "{}; remote services failed as well (HTTP 502 from the defuddle service: the service had a server error; The jina service was shown Zhihu's verification page instead of the content)",
            zhihu_refusal(&url)
        )
    );
    assert!(
        message.starts_with("HTTP 403 for https://www.zhihu.com/question/19550225: Zhihu refuses automated clients; open the page in your browser and save it"),
        "{message}"
    );
    assert_eq!(
        message.matches("Webpage, HTML Only").count(),
        1,
        "{message}"
    );
    assert_eq!(service.requests().len(), 2);
}

#[test]
fn a_remote_reading_that_is_a_refusal_page_goes_on_to_the_next_service() {
    let (_directory, mut cfg) = auto_settings();
    cfg["fetch"]["remote_consent"] = json!("always");
    cfg["fetch"]["policy"]["strategy_priority"] = json!(["defuddle", "jina"]);
    let answer = "这个问题有很多回答。".repeat(30);
    let service = Mock::new(vec![
        // defuddle's Markdown of the security check, then Jina's page.
        Answer::text(
            200,
            &format!("---\ntitle: \"安全验证 - 知乎\"\n---\n\n{ZHIHU_LOGIN}"),
        ),
        jina_page(&format!("# 问题\n\n{answer}\n\n{ZHIHU_LOGIN}")),
        // Both are shown the check.
        Answer::text(
            200,
            &format!("---\ntitle: \"安全验证 - 知乎\"\n---\n\n{ZHIHU_LOGIN}"),
        ),
        zhihu_login_wall(None),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, true, None, None));
    let services = fixture.services(&service.origin);
    // A page with the same words among its own content is read.
    let outcome = fetch_with_services(ZHIHU_QUESTION, &cfg, None, true, None, &services).unwrap();
    assert!(markdown(&outcome).starts_with("# 问题\n\n这个问题有很多回答。"));
    assert_eq!(outcome.document().metadata["fetch_strategy"], "jina");
    // Every service turned away: the site's refusal leads, then each service.
    let Err(Error::Fetch(message)) =
        fetch_with_services(ZHIHU_QUESTION, &cfg, None, true, None, &services)
    else {
        panic!("no service read the page");
    };
    assert!(
        message.starts_with("Zhihu refuses automated clients; open the page in your browser and save it (File > Save Page As…, 'Webpage, HTML Only'), then convert the saved file; or give the local browser your own logged-in cookies for zhihu.com"),
        "{message}"
    );
    assert!(
        message.ends_with("; the remote services tried failed (The defuddle service was shown Zhihu's verification page instead of the content; The jina service was shown Zhihu's login page instead of the content)"),
        "{message}"
    );
    assert_eq!(service.requests().len(), 4);
}

#[test]
fn a_selected_service_shown_a_refusal_says_so_and_what_works() {
    let (_directory, mut cfg) = settings();
    cfg["fetch"]["strategy"] = json!("jina");
    let service = Mock::new(vec![
        zhihu_login_wall(None),
        Answer::json(
            200,
            json!({"code": 200, "status": 20000, "data": {"title": "Just a moment...", "url": "https://example.com/a", "content": "## example.com\n\nVerifying you are human. This may take a few seconds.\n\nexample.com needs to review the security of your connection before proceeding.\n\nPerformance & security by Cloudflare"}}),
        ),
        Answer::json(
            200,
            json!({"code": 200, "status": 20000, "data": {"title": "Just a moment: notes on waiting", "url": "https://example.com/b", "content": format!("# Just a moment\n\n{}", "An essay about waiting, long enough to be a page of its own. ".repeat(12))}}),
        ),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let services = fixture.services(&service.origin);
    let Err(Error::Fetch(message)) =
        fetch_with_services(ZHIHU_QUESTION, &cfg, Some("jina"), true, None, &services)
    else {
        panic!("a login page is not the page");
    };
    assert!(
        message.starts_with("The jina service was shown Zhihu's login page instead of the content; Zhihu refuses automated clients; open the page in your browser"),
        "{message}"
    );
    // A challenge on a site that is not known says what works in general.
    let Err(Error::Fetch(message)) = fetch_with_services(
        "https://example.com/a",
        &cfg,
        Some("jina"),
        true,
        None,
        &services,
    ) else {
        panic!("a challenge is not the page");
    };
    assert_eq!(
        message,
        "The jina service was shown a challenge page instead of the content; the site turns automated readers away; open the page in your browser and save it (File > Save Page As…, 'Webpage, HTML Only'), then convert the saved file"
    );
    // An article that merely has such a title is read.
    let essay = fetch_with_services(
        "https://example.com/b",
        &cfg,
        Some("jina"),
        true,
        None,
        &services,
    )
    .unwrap();
    assert!(markdown(&essay).contains("An essay about waiting"));
}

/// Zhihu's JSON refusal as Cloudflare Browser Rendering returned it for the
/// same question in the rerun on 6f76411 (2026-10-02, `remote_consent:
/// always`): Chromium's page for a JSON answer, which was written as one
/// fenced code block titled `19550225`.
const ZHIHU_JSON: &str = r#"{"error":{"message":"您当前请求存在异常，暂时限制本次访问。如有疑问，您可以通过手机摇一摇或登录后私信知乎小管家反馈。c887aece583d10ea97647fb0e4aeeb5d","code":40362}}"#;

fn zhihu_json_page() -> String {
    format!(
        r#"<html><head><meta name="color-scheme" content="light dark"><meta charset="utf-8"></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">{ZHIHU_JSON}</pre><div class="json-formatter-container"></div></body></html>"#
    )
}

/// Browser Rendering's `/content` answer; `meta` is optional, as each field.
fn cloudflare_rendered(html: &str, meta: Option<Value>) -> Answer {
    let mut envelope = json!({"success": true, "errors": [], "messages": [], "result": html});
    if let Some(meta) = meta {
        envelope["meta"] = meta;
    }
    Answer::json(200, envelope)
}

fn cloudflare_auto(cfg: &mut Value) {
    cfg["fetch"]["remote_consent"] = json!("always");
    cfg["fetch"]["cloudflare"] = json!({"api_token": "env:TEST_CF_TOKEN", "account_id": "acc0123"});
}

const DEFUDDLE_FAILED: &str = "HTTP 502 from the defuddle service: the service had a server error";
const JINA_LOGIN: &str = "The jina service was shown Zhihu's login page instead of the content";
const CLOUDFLARE_JSON: &str = "The cloudflare service was shown Zhihu's JSON refusal instead of the content, which said: 您当前请求存在异常，暂时限制本次访问。如有疑问，您可以通过手机摇一摇或登录后私信知乎小管家反馈。c887aece583d10ea97647fb0e4aeeb5d (code 40362)";
const CLOUDFLARE_403: &str =
    "The cloudflare service received HTTP 403 from the site instead of the content";

#[test]
fn a_zhihu_json_refusal_cloudflare_rendered_fails_with_every_service_and_why() {
    // The fixture is what was written: the extraction makes it one fenced
    // block (the output later titled it by the address).
    let written = formats::extract_html(&zhihu_json_page(), Some(ZHIHU_QUESTION)).unwrap();
    assert_eq!(written.markdown.trim(), format!("```\n{ZHIHU_JSON}\n```"));

    let (_directory, mut cfg) = auto_settings();
    cloudflare_auto(&mut cfg);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, true, None, None));
    let url = Url::parse(ZHIHU_QUESTION).unwrap();
    // Without `meta`, the markup; with the origin's `meta.status`, the status.
    for (meta, cloudflare) in [
        (None, CLOUDFLARE_JSON),
        (
            Some(json!({"status": 403, "finalUrl": ZHIHU_QUESTION, "title": ""})),
            CLOUDFLARE_403,
        ),
    ] {
        let service = Mock::new(vec![
            Answer::text(502, "bad gateway"),
            zhihu_login_wall(None),
            cloudflare_rendered(&zhihu_json_page(), meta),
        ]);
        let mut services = fixture.services(&service.origin);
        services.vars = vars(&[("TEST_CF_TOKEN", "cf-test-token-0123")]);
        // The default order, with the static request answered as Zhihu
        // answered it (no request leaves the machine).
        let steps = auto_order(&url, &cfg, false, &services);
        let mut local = Some(zhihu_refusal(&url));
        let result = chain::run(
            &steps,
            chain::Attempts {
                static_fetch: &mut || Err(local.take().expect("one static request")),
                browser_ready: &|| false,
                render: &mut |_| unreachable!("the static refusal ends the local steps"),
                remote_ready: &mut |_| chain::Readiness::Ready,
                remote: &mut |service| {
                    remote::fetch(service, ZHIHU_QUESTION, &url, &cfg, &services)
                        .map(remote::outcome)
                },
                what_works: sites::what_works(&url),
            },
        );
        let Err(Error::Fetch(message)) = result else {
            panic!("Zhihu's JSON refusal is not the page");
        };
        assert_eq!(
            message,
            format!(
                "{}; remote services failed as well ({DEFUDDLE_FAILED}; {JINA_LOGIN}; {cloudflare})",
                zhihu_refusal(&url)
            )
        );
        assert!(
            message.starts_with("HTTP 403 for https://www.zhihu.com/question/19550225: Zhihu refuses automated clients; open the page in your browser and save it"),
            "{message}"
        );
        assert_eq!(
            message.matches("Webpage, HTML Only").count(),
            1,
            "{message}"
        );
        let requests = service.requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            requests[2].target,
            "/cloudflare/accounts/acc0123/browser-rendering/content"
        );
        assert!(!message.contains("acc0123") && !message.contains("cf-test-token"));
    }
}

#[test]
fn remote_services_alone_turned_away_by_zhihu_lead_with_the_site_and_write_nothing() {
    let (directory, mut cfg) = auto_settings();
    cloudflare_auto(&mut cfg);
    cfg["fetch"]["policy"]["strategy_priority"] = json!(["defuddle", "jina", "cloudflare"]);
    let service = Mock::new(vec![
        Answer::text(502, "bad gateway"),
        zhihu_login_wall(None),
        cloudflare_rendered(&zhihu_json_page(), Some(json!({"title": ""}))),
        Answer::text(502, "bad gateway"),
        zhihu_login_wall(None),
        cloudflare_rendered(&zhihu_json_page(), Some(json!({"status": 403}))),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, true, None, None));
    let mut services = fixture.services(&service.origin);
    services.vars = vars(&[("TEST_CF_TOKEN", "cf-test-token-0123")]);
    let url = Url::parse(ZHIHU_QUESTION).unwrap();
    for cloudflare in [CLOUDFLARE_JSON, CLOUDFLARE_403] {
        let Err(Error::Fetch(message)) =
            fetch_with_services(ZHIHU_QUESTION, &cfg, None, true, None, &services)
        else {
            panic!("no service read the page");
        };
        assert_eq!(
            message,
            format!(
                "{}; the remote services tried failed ({DEFUDDLE_FAILED}; {JINA_LOGIN}; {cloudflare})",
                sites::what_works(&url)
            )
        );
    }
    assert_eq!(service.requests().len(), 6);
    assert!(!directory.path().join("fetch_cache.db").exists());
}

#[test]
fn a_page_with_a_json_error_example_and_the_origins_other_statuses_read_as_before() {
    let mut cfg = cloudflare_cfg();
    cfg["fetch"]["cloudflare"] = json!({"api_token": "env:TEST_CF_TOKEN", "account_id": "acc0123"});
    let article = format!(
        "<html><head><title>Handling refusals</title></head><body><article><h1>Handling refusals</h1><p>When the API turns a client away it answers with a JSON error object such as this one, and the client should wait before it tries again:</p><pre><code>{ZHIHU_JSON}</code></pre><p>The code names the reason; the message is meant for a person.</p></article></body></html>"
    );
    let service = Mock::new(vec![
        // An article that shows a JSON error, with and without the origin's status.
        cloudflare_rendered(&article, None),
        cloudflare_rendered(
            &article,
            Some(json!({"status": 200, "finalUrl": "https://example.com/errors"})),
        ),
        // The origin's other failures are the page's.
        cloudflare_rendered(&article, Some(json!({"status": 404}))),
        cloudflare_rendered(&article, Some(json!({"status": 503}))),
        // The same article read through Jina.
        jina_page(&format!(
            "# Handling refusals\n\nWhen the API turns a client away it answers with:\n\n```json\n{ZHIHU_JSON}\n```\n\nWait before trying again."
        )),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let mut services = fixture.services(&service.origin);
    services.vars = vars(&[("TEST_CF_TOKEN", "cf-test-token-0123")]);
    let source = "https://example.com/errors";
    for _ in 0..2 {
        let outcome =
            fetch_with_services(source, &cfg, Some("cloudflare"), true, None, &services).unwrap();
        assert!(
            markdown(&outcome).contains("40362")
                && markdown(&outcome).contains("Handling refusals"),
            "{}",
            markdown(&outcome)
        );
        assert_eq!(outcome.document().metadata["fetch_strategy"], "cloudflare");
    }
    for expected in [
        "The cloudflare service received HTTP 404 from the site: the page may have been removed or is not public",
        "The cloudflare service received HTTP 503 from the site: the site had a server error",
    ] {
        let Err(Error::Fetch(message)) =
            fetch_with_services(source, &cfg, Some("cloudflare"), true, None, &services)
        else {
            panic!("the origin's failure is the page's");
        };
        assert_eq!(message, expected);
    }
    let mut jina = cfg.clone();
    jina["fetch"]["strategy"] = json!("jina");
    let outcome = fetch_with_services(source, &jina, Some("jina"), true, None, &services).unwrap();
    assert!(markdown(&outcome).contains("```json\n{\"error\""));
    assert_eq!(service.requests().len(), 5);
}

#[test]
fn jina_warnings_are_kept_and_a_bypassed_cache_asks_jina_for_a_fresh_reading() {
    let (_directory, mut cfg) = settings();
    cfg["fetch"]["strategy"] = json!("jina");
    let cached =
        "This is a cached snapshot of the original page, consider retry with caching opt-out.";
    let reading = |content: &str| {
        Answer::json(
            200,
            json!({"code": 200, "status": 20000, "data": {"title": "Example Domain", "url": "https://example.com/", "content": content, "warning": cached, "httpStatus": 200}}),
        )
    };
    let service = Mock::new(vec![
        reading("# One\n\nFirst reading."),
        reading("# Two\n\nSecond reading."),
        reading("# Three\n\nThird reading."),
        reading("# Four\n\nFourth reading."),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let services = fixture.services(&service.origin);
    let source = "https://example.com/";
    let read = |cfg: &Value| {
        fetch_with_services(source, cfg, Some("jina"), true, None, &services)
            .unwrap()
            .document()
            .warnings
            .clone()
    };
    // The cache is in use: no opt-out, and the warning says how to get one.
    assert_eq!(
        read(&cfg),
        [format!(
            "The jina service said: {cached} Run with --no-cache (or set fetch.jina.no_cache) to ask Jina for a fresh reading."
        )]
    );
    // --no-cache, and a --no-cache-for pattern that matches, ask for a fresh
    // reading; the warning is kept as Jina said it.
    cfg["cache"]["no_cache"] = json!(true);
    assert_eq!(read(&cfg), [format!("The jina service said: {cached}")]);
    cfg["cache"]["no_cache"] = json!(false);
    cfg["cache"]["no_cache_patterns"] = json!(["example.com"]);
    assert_eq!(read(&cfg), [format!("The jina service said: {cached}")]);
    // A pattern for another site does not.
    cfg["cache"]["no_cache_patterns"] = json!(["other.test"]);
    read(&cfg);
    let sent: Vec<Option<&str>> = service
        .requests()
        .iter()
        .map(|request| request.header("X-No-Cache").map(|_| "sent"))
        .collect();
    assert_eq!(sent, [None, Some("sent"), Some("sent"), None]);
    for request in service.requests() {
        if let Some(value) = request.header("X-No-Cache") {
            assert_eq!(value, "true");
        }
    }
}

#[test]
fn jina_text_answers_header_lines_and_page_statuses_are_read() {
    let (_directory, mut cfg) = settings();
    cfg["fetch"]["strategy"] = json!("jina");
    let plain = |body: &str| Answer {
        status: 200,
        headers: vec![("Content-Type".into(), "text/plain; charset=utf-8".into())],
        body: body.as_bytes().to_vec(),
    };
    let data = |data: Value| Answer::json(200, json!({"code": 200, "status": 20000, "data": data}));
    let service = Mock::new(vec![
        // The text form, as `r.jina.ai` answers without `Accept: application/json`.
        plain(
            "Title: Example Domain\r\n\r\nURL Source: https://example.com/\r\n\r\nWarning: This is a cached snapshot of the original page, consider retry with caching opt-out.\r\n\r\nMarkdown Content:\r\n# Example\r\n\r\nBody text of the page.",
        ),
        // Header lines inside the JSON content are Jina's.
        data(
            json!({"url": "https://example.com/", "content": "Title: Inner\nURL Source: https://example.com/\nPublished Time: 2026-01-01\n\nMarkdown Content:\n# Inner\n\nText of the page."}),
        ),
        // A page whose own first line looks like a header is kept whole.
        data(
            json!({"title": "Notes", "content": "Title: a working note\n\nThe paragraph after it."}),
        ),
        // The page's own status.
        data(json!({"title": "Forbidden", "content": "403 Forbidden", "httpStatus": 403})),
        data(
            json!({"title": "Not found", "content": "This page does not exist.", "httpStatus": 404}),
        ),
        data(
            json!({"title": "Blocked", "content": "Blocked.", "warning": "Target URL returned error 429: Too Many Requests"}),
        ),
        // Neither JSON nor the text form.
        plain("<html>an interstitial</html>"),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let services = fixture.services(&service.origin);
    let run = || {
        fetch_with_services(
            "https://example.com/",
            &cfg,
            Some("jina"),
            true,
            None,
            &services,
        )
    };

    let text = run().unwrap();
    assert_eq!(markdown(&text), "# Example\n\nBody text of the page.");
    assert_eq!(text.document().metadata["title"], "Example Domain");
    assert!(
        text.document().warnings[0].starts_with("The jina service said: This is a cached snapshot"),
        "{:?}",
        text.document().warnings
    );
    let inner = run().unwrap();
    assert_eq!(markdown(&inner), "# Inner\n\nText of the page.");
    assert_eq!(inner.document().metadata["title"], "Inner");
    let note = run().unwrap();
    assert_eq!(
        markdown(&note),
        "Title: a working note\n\nThe paragraph after it."
    );
    let failure = |result: Result<FetchOutcome>| match result {
        Err(Error::Fetch(message)) => message,
        other => panic!(
            "not a fetch failure: {:?}",
            other.map(|outcome| outcome.document().markdown.clone())
        ),
    };
    assert_eq!(
        failure(run()),
        "The jina service received HTTP 403 from the site instead of the content; the site turns automated readers away; open the page in your browser and save it (File > Save Page As…, 'Webpage, HTML Only'), then convert the saved file"
    );
    assert_eq!(
        failure(run()),
        "The jina service received HTTP 404 from the site: the page may have been removed or is not public"
    );
    assert!(
        failure(run()).starts_with(
            "The jina service received HTTP 429 from the site instead of the content;"
        )
    );
    assert_eq!(
        failure(run()),
        "The jina service returned an answer that is not JSON"
    );
}

// ---- Workers AI toMarkdown's frame ------------------------------------------

/// Workers AI `toMarkdown` answers recorded in the real-service check of
/// 2d01ccb (2026-10-02) for the reference repository's public fixtures
/// `tests/fixtures/sample.pdf` and `sample.docx`.
const TOMARKDOWN_PDF: &str =
    include_str!("../../tests/fixtures/cloudflare-tomarkdown/sample.pdf.md");
const TOMARKDOWN_DOCX: &str =
    include_str!("../../tests/fixtures/cloudflare-tomarkdown/sample.docx.md");

#[test]
fn workers_ai_output_loses_its_frame_and_pages_read_like_the_native_pdf_reader() {
    let directory = tempfile::tempdir().unwrap();
    let pdf = directory.path().join("sample.pdf");
    std::fs::write(&pdf, b"%PDF-1.4 fixture bytes").unwrap();
    let docx = directory.path().join("sample.docx");
    std::fs::write(&docx, b"PK fixture bytes").unwrap();
    let report = directory.path().join("report.pdf");
    std::fs::write(&report, b"%PDF-1.4 fixture bytes").unwrap();
    let mut cfg = config::defaults();
    cfg["fetch"]["cloudflare"] = json!({"convert_enabled": true, "account_id": "acc0123"});
    let converted = |name: &str, data: &str| {
        Answer::json(
            200,
            json!({"success": true, "result": [{"name": name, "format": "markdown", "tokens": 1200, "data": data}]}),
        )
    };
    let service = Mock::new(vec![
        converted("sample.pdf", TOMARKDOWN_PDF),
        converted("sample.docx", TOMARKDOWN_DOCX),
        converted(
            "report.pdf",
            "# report.pdf\n\n## Metadata\n\n- PDFFormatVersion=1.7\n- Title=Quarterly Report\n- Author=Ada Lovelace\n- CreationDate=D:20240102\n\n## Contents\n\n### Page 1\n\nFirst page.\n\n### Page 2\n\n### Page 3\n\n### Page 7\n\nA heading of the document's own.",
        ),
        converted(
            "report.pdf",
            "# report.pdf\n\n## Metadata\n\n- Title=Microsoft Word - report.docx\n- CreationDate=not a date\n\n## Contents\n\n### Page 1\n\nText.",
        ),
    ]);
    let terminal = Arc::new(Terminal::default());
    let fixture = Fixture::new(gate(&terminal, false, None, None));
    let mut services = fixture.services(&service.origin);
    services.vars = vars(&[("CLOUDFLARE_API_TOKEN", "workers-token-0123")]);

    let document = cloudflare::convert_file_with(&pdf, "pdf", &cfg, &services).unwrap();
    let text = &document.markdown;
    assert!(
        text.starts_with("<!-- Page number: 1 -->\n\nLorem ipsumLorem ipsum dolor sit amet"),
        "{text}"
    );
    for gone in [
        "# sample.pdf",
        "## Metadata",
        "PDFFormatVersion",
        "Producer=",
        "## Contents",
        "### Page",
    ] {
        assert!(!text.contains(gone), "{gone}: {text}");
    }
    let markers: Vec<&str> = text
        .lines()
        .filter(|line| line.starts_with("<!-- Page number: "))
        .collect();
    assert_eq!(
        markers,
        (1..=5)
            .map(|page| format!("<!-- Page number: {page} -->"))
            .collect::<Vec<_>>()
    );
    assert!(
        text.contains("\n\n<!-- Page number: 2 -->\n\nIn non mauris justo."),
        "{text}"
    );
    assert!(
        text.ends_with("sed turpis imperdiet eleifend sit amet id sapien."),
        "{text}"
    );
    let metadata = &document.metadata;
    assert_eq!(metadata["pages"], 5);
    assert_eq!(metadata["date"], "2017-08-16T14:42:28+02:00");
    assert_eq!(metadata["converter"], "cloudflare-tomarkdown");
    assert_eq!(metadata["tokens"], 1200);
    for absent in ["title", "author", "Creator", "Producer", "PDFFormatVersion"] {
        assert!(!metadata.contains_key(absent), "{absent}: {metadata:?}");
    }

    let document = cloudflare::convert_file_with(&docx, "docx", &cfg, &services).unwrap();
    assert!(
        document
            .markdown
            .starts_with("# Markitai Snapshot Fixture\n\nThis is a synthetic paragraph"),
        "{}",
        document.markdown
    );
    assert!(!document.metadata.contains_key("pages"));

    let document = cloudflare::convert_file_with(&report, "pdf", &cfg, &services).unwrap();
    assert_eq!(
        document.markdown,
        "<!-- Page number: 1 -->\n\nFirst page.\n\n<!-- Page number: 2 -->\n\n<!-- Page number: 3 -->\n\n### Page 7\n\nA heading of the document's own."
    );
    assert_eq!(document.metadata["title"], "Quarterly Report");
    assert_eq!(document.metadata["author"], "Ada Lovelace");
    assert_eq!(document.metadata["date"], "2024-01-02");
    assert_eq!(document.metadata["pages"], 3);
    // A title that only names a file, and a date that is none, are dropped.
    let document = cloudflare::convert_file_with(&report, "pdf", &cfg, &services).unwrap();
    assert_eq!(document.markdown, "<!-- Page number: 1 -->\n\nText.");
    assert!(!document.metadata.contains_key("title"));
    assert!(!document.metadata.contains_key("date"));
}
