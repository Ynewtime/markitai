//! Failure messages, retries, character encodings, script-rendered pages and
//! `<meta>` refreshes of fetched pages, against loopback servers.

use super::cache_tests::{Reply, Server, settings};
use super::*;
use std::cell::Cell;

const CHINESE: &str = "这是一段用于测试字符编码的中文文本，包含常用汉字和标点符号。第二段：风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。";

fn run(server: &Server, path: &str, cfg: &Value) -> Result<FetchOutcome> {
    fetch_with_context(&server.url(path), cfg, None, true)
}

fn failure(result: Result<FetchOutcome>) -> String {
    match result {
        Err(Error::Fetch(message)) => message,
        Err(other) => panic!("not a fetch failure: {other:?}"),
        Ok(outcome) => panic!("expected a failure, read {:?}", outcome.document().markdown),
    }
}

fn gbk_page(meta: &str) -> Vec<u8> {
    let source = format!(
        "<html><head>{meta}<title>编码测试</title></head><body><article><h1>编码测试</h1><p>{CHINESE}</p></article></body></html>"
    );
    let (bytes, _, unmappable) = encoding_rs::GBK.encode(&source);
    assert!(!unmappable);
    bytes.into_owned()
}

fn stub(markdown: &str) -> FetchOutcome {
    FetchOutcome {
        content: FetchContent::Document(Document {
            markdown: markdown.into(),
            ..Default::default()
        }),
        cache_hit: false,
        screenshots: Vec::new(),
    }
}

fn shell_outcome() -> FetchOutcome {
    let mut outcome = stub("# Shell");
    outcome.content.warnings_mut().push(JS_SHELL.into());
    outcome
}

// ---- failure messages -----------------------------------------------------

#[test]
fn http_failures_name_the_page_and_say_what_to_try() {
    for (status, hint) in [
        (404, "the page may have been removed or is not public"),
        (410, "the page may have been removed or is not public"),
        (
            401,
            "the site refused access; it may block automated clients or need a login",
        ),
        (
            403,
            "the site refused access; it may block automated clients or need a login",
        ),
        (429, "rate limited; try again later"),
        (500, "the site had a server error"),
        (503, "the site had a server error"),
    ] {
        let (_directory, cfg) = settings();
        let server = Server::new(vec![Reply::text("no").status(status)]);
        let message = failure(run(&server, "/gone/page?id=5&view=full", &cfg));
        // The web interface recognizes the message by its leading status.
        assert!(
            message.starts_with(&format!(
                "HTTP {status} for {}",
                server.url("/gone/page?id=5&view=full")
            )),
            "{message}"
        );
        assert!(message.ends_with(&format!(": {hint}")), "{message}");
    }
    // Other statuses name the page and add no guess.
    let (_directory, cfg) = settings();
    let server = Server::new(vec![Reply::text("no").status(418)]);
    let message = failure(run(&server, "/tea", &cfg));
    assert_eq!(message, format!("HTTP 418 for {}", server.url("/tea")));
}

#[test]
fn failure_messages_leave_out_credentials_and_secret_looking_queries() {
    let (_directory, cfg) = settings();
    let server = Server::new(vec![Reply::text("no").status(404)]);
    let source = server
        .url("/private/page?token=abc&api_key=k&session=s1&sig=9&auth=u&code=7&id=42&x=aB3dE5gH7jK9mN1pQ3sT5vW7yZ#frag")
        .replacen("http://", "http://user:hunter2@", 1);
    let message = failure(fetch_with_context(&source, &cfg, None, true));
    for secret in [
        "hunter2",
        "user:",
        "abc",
        "api_key=k",
        "s1",
        "9&",
        "u&",
        "7&",
        "aB3dE5",
        "frag",
    ] {
        assert!(!message.contains(secret), "{secret}: {message}");
    }
    assert!(message.contains("id=42"), "{message}");
    assert!(message.contains("token=REDACTED"), "{message}");
    assert!(message.contains("x=REDACTED"), "{message}");
    assert!(message.contains("/private/page?"), "{message}");

    let long = Url::parse(&format!("https://example.com/p?q={}", "a".repeat(400))).unwrap();
    let shown = shown_url(&long);
    assert_eq!(shown.chars().count(), 200, "{shown}");
    assert!(shown.ends_with('…'));
    assert_eq!(
        shown_url(&Url::parse("https://u:p@example.com/a?author=bob&tag=x").unwrap()),
        "https://example.com/a?author=bob&tag=x"
    );
}

#[test]
fn the_error_kind_stays_a_fetch_error() {
    let (_directory, cfg) = settings();
    let server = Server::new(vec![Reply::text("no").status(404)]);
    let error = run(&server, "/x", &cfg).err().unwrap();
    assert_eq!(error.code(), "fetch_error");
}

// ---- retries --------------------------------------------------------------

#[test]
fn a_tls_handshake_that_ends_early_is_retried() {
    // The failure of a site whose TLS endpoint closes the connection while
    // the handshake is under way: a plain listener that accepts and closes.
    // Nothing was sent, so the connection is attempted again, twice at most,
    // after pauses of 250 ms and 750 ms.
    let (_directory, cfg) = settings();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("https://{}/page", listener.local_addr().unwrap());
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (stopped, count) = (stop.clone(), accepted.clone());
    let thread = std::thread::spawn(move || {
        use std::sync::atomic::Ordering::SeqCst;
        while !stopped.load(SeqCst) {
            match listener.accept() {
                Ok(_) => {
                    count.fetch_add(1, SeqCst);
                }
                Err(_) => std::thread::sleep(Duration::from_millis(2)),
            }
        }
    });
    let started = std::time::Instant::now();
    let message = failure(fetch_with_context(&url, &cfg, None, true));
    let elapsed = started.elapsed();
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    thread.join().unwrap();
    // An end of stream or, depending on the platform and timing, a reset.
    let lower = message.to_lowercase();
    assert!(
        lower.contains("eof") || lower.contains("reset"),
        "{message}"
    );
    assert_eq!(
        accepted.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "{message}"
    );
    // The pauses before the second and third attempt: 250 ms and 750 ms.
    assert!(elapsed >= Duration::from_millis(900), "{elapsed:?}");
}

#[test]
fn statuses_refusals_clean_closes_resets_after_the_request_and_timeouts_are_never_retried() {
    let (_directory, cfg) = settings();
    for status in [400, 403, 404, 410, 429, 500, 502, 503] {
        let server = Server::new(vec![Reply::text("no").status(status), Reply::text("yes")]);
        failure(run(&server, "/once", &cfg));
        assert_eq!(server.requests().len(), 1, "{status}");
    }
    // A server that reads the request and closes the connection cleanly
    // without answering is reported at once.
    let server = Server::new(vec![Reply::dropped(), Reply::text("yes")]);
    failure(run(&server, "/closed", &cfg));
    assert_eq!(server.requests().len(), 1);
    // A connection reset after the request went out is not repeated either.
    #[cfg(unix)]
    {
        let server = Server::new(vec![Reply::reset(), Reply::text("yes")]);
        let message = failure(run(&server, "/reset", &cfg));
        assert!(message.to_lowercase().contains("reset"), "{message}");
        assert_eq!(server.requests().len(), 1);
    }
    // What the classifier sees: a refused connection and a timeout are not
    // transient either.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let refused = client(5)
        .unwrap()
        .get(format!("http://127.0.0.1:{port}/"))
        .send()
        .unwrap_err();
    assert!(!connection_dropped(&refused), "{refused:?}");
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let slow = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(150))
        .build()
        .unwrap()
        .get(format!("http://{}/", listener.local_addr().unwrap()))
        .send()
        .unwrap_err();
    assert!(slow.is_timeout() && !connection_dropped(&slow), "{slow:?}");
    #[cfg(unix)]
    {
        let server = Server::new(vec![Reply::reset()]);
        let reset = client(5).unwrap().get(server.url("/")).send().unwrap_err();
        assert!(!connection_dropped(&reset), "{reset:?}");
    }
}

// ---- character encodings --------------------------------------------------

#[test]
fn legacy_pages_read_correctly_when_the_header_is_wrong_or_missing() {
    for (header, meta) in [
        // The header says UTF-8 and the page itself says GBK.
        ("text/html; charset=utf-8", "<meta charset=gbk>"),
        // The header says UTF-8 and nothing else says anything.
        ("text/html; charset=utf-8", ""),
        // Nothing declares an encoding.
        ("text/html", ""),
        // A correct declaration wins.
        ("text/html; charset=gbk", ""),
        ("text/html; charset=GB2312", "<meta charset=big5>"),
    ] {
        let (_directory, cfg) = settings();
        let server = Server::new(vec![Reply::bytes(header, &gbk_page(meta))]);
        let outcome = run(&server, "/page", &cfg).unwrap();
        let document = outcome.document();
        assert!(document.markdown.contains(CHINESE), "{header} {meta}");
        assert_eq!(document.metadata["title"], "编码测试");
        assert!(document.warnings.is_empty(), "{:?}", document.warnings);
    }
}

#[test]
fn undeclared_western_and_nearly_utf8_pages_keep_a_reading_and_say_so_when_unsure() {
    let (_directory, cfg) = settings();
    let latin = b"<title>Menu</title><article><p>Caf\xe9 au lait et cr\xe8me br\xfbl\xe9e, servis chaque matin dans notre petit salon.</p></article>";
    let mut mostly_utf8 = format!("<article><p>{}</p></article>", CHINESE.repeat(3)).into_bytes();
    // Just after `<article><p>`: one bad byte, between two characters.
    mostly_utf8.insert(12, 0xff);
    let server = Server::new(vec![
        Reply::bytes("text/html", latin),
        Reply::bytes("text/html; charset=utf-8", &mostly_utf8),
    ]);
    let first = run(&server, "/latin", &cfg).unwrap();
    assert!(
        first
            .document()
            .markdown
            .contains("Café au lait et crème brûlée")
    );
    let second = run(&server, "/utf8", &cfg).unwrap();
    assert!(second.document().markdown.contains(CHINESE));
    assert_eq!(second.document().warnings.len(), 1);
    assert!(second.document().warnings[0].contains("1 invalid byte sequence;"));
}

#[test]
fn a_page_read_with_a_warning_is_not_cached() {
    let (_directory, cfg) = settings();
    let mut page = format!("<article><p>{}</p></article>", CHINESE.repeat(3)).into_bytes();
    page.insert(12, 0xff);
    let reply = Reply::bytes("text/html; charset=utf-8", &page);
    let server = Server::new(vec![reply.clone(), reply]);
    assert!(!run(&server, "/p", &cfg).unwrap().cache_hit);
    assert!(!run(&server, "/p", &cfg).unwrap().cache_hit);
    assert_eq!(server.requests().len(), 2);
}

// ---- script-rendered pages ------------------------------------------------

fn quotes_shell() -> String {
    let data = (0..16)
        .map(|n| {
            format!(
                "{{\"text\": \"The world as we have created it is a process of our thinking {n}.\", \"author\": {{\"name\": \"Albert Einstein\", \"goodreads_link\": \"/author/show/9810.Albert_Einstein\", \"slug\": \"Albert-Einstein\"}}, \"tags\": [\"change\", \"deep-thoughts\", \"thinking\", \"world\"]}}"
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><title>Quotes to Scrape</title></head><body><div class=\"container\"><h1><a href=\"/\">Quotes to Scrape</a></h1><p><a href=\"/login\">Login</a></p><div id=\"quotes\"></div></div><script src=\"/static/jquery.js\"></script><script>var data = [{data}]; for (var i in data) {{ document.getElementById('quotes').innerHTML += data[i].text; }}</script></body></html>"
    )
}

#[test]
fn explicit_static_keeps_a_script_rendered_shell_with_a_warning() {
    let (directory, cfg) = settings();
    let reply = Reply::html(&quotes_shell());
    let server = Server::new(vec![reply.clone(), reply]);
    let outcome = run(&server, "/js/", &cfg).unwrap();
    assert!(outcome.document().markdown.contains("Quotes to Scrape"));
    assert_eq!(outcome.document().warnings, [JS_SHELL]);
    assert!(JS_SHELL.contains("-s playwright") && JS_SHELL.contains("auto strategy"));
    // The shell is not stored as a good page.
    assert!(!run(&server, "/js/", &cfg).unwrap().cache_hit);
    assert!(!directory.path().join("fetch_cache.db").exists());
}

#[test]
fn explicit_static_says_what_a_page_that_needs_javascript_requires() {
    for html in [
        // Only a mount point and a script.
        "<title>App</title><div id=\"root\"></div><script src=\"/assets/main.js\"></script>",
        "<title>App</title><body><div id=\"__next\"> </div><script>window.__NEXT_DATA__ = {};</script>",
        // The page asks for JavaScript in its own text.
        "<body><div id=\"root\"></div><noscript>You need to enable JavaScript to run this app.</noscript><script src=\"/a.js\"></script></body>",
        "<p>Please enable JavaScript to continue.</p>",
    ] {
        let (_directory, cfg) = settings();
        let server = Server::new(vec![Reply::html(html)]);
        let message = failure(run(&server, "/", &cfg));
        assert_eq!(
            message, "The page needs JavaScript; use -s playwright or the default auto strategy",
            "{html}"
        );
        assert!(!message.contains("not implemented"));
    }
}

#[test]
fn short_pages_without_script_evidence_and_long_pages_are_ordinary() {
    let article = format!(
        "<title>Long</title><article><h1>Long</h1><p>{}</p></article><script>{}</script>",
        "A real article with many words in it. ".repeat(10),
        "x".repeat(5000)
    );
    let small_script = format!(
        "<title>Soon</title><p>Coming soon.</p><script>{}</script>",
        "analytics();".repeat(40)
    );
    let structured = format!(
        "<title>Soon</title><p>Coming soon.</p><script type=\"application/ld+json\">{}</script>",
        "{\"a\": \"b\"}".repeat(600)
    );
    for html in [
        "<title>Hi</title><p>A short page without scripts.</p>".to_string(),
        article,
        small_script,
        structured,
        // A mount point that has content is not a shell.
        "<title>App</title><div id=\"root\"><p>Server rendered text.</p></div><script src=\"/a.js\"></script>"
            .to_string(),
    ] {
        let (_directory, cfg) = settings();
        let server = Server::new(vec![Reply::html(&html)]);
        let outcome = run(&server, "/", &cfg).unwrap();
        assert!(outcome.document().warnings.is_empty(), "{html}");
    }
}

#[test]
fn script_evidence_needs_a_script_and_one_of_three_signals() {
    assert!(script_rendered(&quotes_shell()));
    assert!(script_rendered(
        "<div id=\"app\"></div><script src=\"a.js\"></script>"
    ));
    assert!(script_rendered(
        "<app-root></app-root><script src=\"a.js\"></script>"
    ));
    assert!(script_rendered(
        "<noscript>This site works best with JavaScript.</noscript><script src=\"a.js\"></script>"
    ));
    // A mount point holding only script source is still empty.
    assert!(script_rendered(
        "<div id=\"root\"><script>var a = 1;</script></div><script src=\"a.js\"></script>"
    ));
    assert!(!script_rendered("<div id=\"root\"></div>"));
    assert!(!script_rendered(
        "<p>Text</p><script src=\"a.js\"></script>"
    ));
    assert!(!script_rendered(
        "<div id=\"root\">Text</div><script src=\"a.js\"></script>"
    ));
}

#[test]
fn auto_sends_a_page_that_needs_javascript_to_the_browser_and_keeps_static_text_otherwise() {
    let render_with = |expected: Option<bool>| {
        let asked = Cell::new(None);
        let outcome = auto_result(
            Err(Error::Fetch(
                match expected {
                    Some(true) => JS_REQUIRED,
                    _ => JS_EMPTY,
                }
                .into(),
            )),
            || true,
            |learn| {
                asked.set(Some(learn));
                Ok(stub("rendered"))
            },
        )
        .unwrap();
        assert_eq!(outcome.document().markdown, "rendered");
        asked.get().unwrap()
    };
    // Only a page that says so in its own text teaches the routing.
    assert!(render_with(Some(true)));
    assert!(!render_with(Some(false)));

    // No browser: say what is missing.
    let error = auto_result(
        Err(Error::Fetch(JS_REQUIRED.into())),
        || false,
        |_| panic!("no browser to render with"),
    )
    .err()
    .unwrap();
    let message = error.to_string();
    assert!(
        message.contains("no local browser") && message.contains("MARKITAI_BROWSER_EXECUTABLE")
    );
    assert!(!message.contains("not implemented"));

    // Other failures are untouched and never look for a browser.
    let error = auto_result(
        Err(Error::Fetch("HTTP 404 for https://example.com/".into())),
        || panic!("not needed"),
        |_| panic!("not needed"),
    )
    .err()
    .unwrap();
    assert_eq!(error.to_string(), "HTTP 404 for https://example.com/");
    let ordinary = auto_result(
        Ok(stub("fine")),
        || panic!("not needed"),
        |_| panic!("not needed"),
    );
    assert_eq!(ordinary.unwrap().document().markdown, "fine");
}

#[test]
fn a_shell_goes_to_the_browser_and_falls_back_to_its_static_text_with_a_warning() {
    // The browser renders it.
    let asked = Cell::new(None);
    let rendered = auto_result(
        Ok(shell_outcome()),
        || true,
        |learn| {
            asked.set(Some(learn));
            Ok(stub("rendered quotes"))
        },
    )
    .unwrap();
    assert_eq!(rendered.document().markdown, "rendered quotes");
    // A heuristic shell never teaches the routing.
    assert_eq!(asked.get(), Some(false));

    // No browser: the static text stays, with the page's warning.
    let kept = auto_result(Ok(shell_outcome()), || false, |_| panic!("no browser")).unwrap();
    assert_eq!(kept.document().markdown, "# Shell");
    assert_eq!(kept.document().warnings, [JS_SHELL]);

    // The browser fails: static text, the page's warning and the reason.
    let kept = auto_result(
        Ok(shell_outcome()),
        || true,
        |_| Err(Error::Fetch("Browser navigation timed out".into())),
    )
    .unwrap();
    assert_eq!(kept.document().markdown, "# Shell");
    assert_eq!(kept.document().warnings.len(), 2);
    assert!(kept.document().warnings[1].contains("Browser navigation timed out"));

    // Configuration and input errors are not swallowed.
    let error = auto_result(
        Ok(shell_outcome()),
        || true,
        |_| Err(Error::Config("fetch.playwright.session_ttl_seconds".into())),
    )
    .err()
    .unwrap();
    assert!(matches!(error, Error::Config(_)));
}

// ---- <meta> refresh -------------------------------------------------------

#[test]
fn a_page_that_only_redirects_with_a_meta_refresh_is_followed() {
    for content in [
        "0;url=/target",
        "0; URL=/target",
        "1;url='/target'",
        "2, '/target'",
        "0.5; url = /target",
        " 0 ; url=target",
    ] {
        let (_directory, cfg) = settings();
        let server = Server::new(vec![
            Reply::html(&format!(
                "<html><head><meta http-equiv=\"Refresh\" content=\"{content}\"><title>Redirecting…</title></head><body>Redirecting…</body></html>"
            )),
            Reply::html(
                "<title>Target</title><article><h1>Target</h1><p>The real page.</p></article>",
            ),
        ]);
        let outcome = run(&server, "/start", &cfg).unwrap();
        assert!(
            outcome.document().markdown.contains("The real page."),
            "{content}"
        );
        assert_eq!(outcome.document().metadata["title"], "Target");
        let requests = server.requests();
        assert_eq!(requests.len(), 2, "{content}");
        assert!(
            requests[1].starts_with("GET /target "),
            "{content}: {}",
            requests[1]
        );
    }
}

#[test]
fn a_bodyless_refresh_page_and_cached_results_use_the_final_page() {
    let (directory, cfg) = settings();
    let server = Server::new(vec![
        Reply::html("<meta http-equiv=refresh content=\"0;url=/final\">")
            .header("ETag", "\"stub\""),
        Reply::html("<article><p>Final page text.</p></article>")
            .header("ETag", "\"final\"")
            .header("Last-Modified", "Mon, 28 Sep 2026 09:00:00 GMT"),
    ]);
    let first = run(&server, "/start", &cfg).unwrap();
    assert!(first.document().markdown.contains("Final page text."));
    // The first page's validators describe the stub and the second's the
    // final page; neither can revalidate the pair, so the entry is a plain
    // TTL entry.
    let second = run(&server, "/start", &cfg).unwrap();
    assert!(second.cache_hit);
    assert_eq!(server.requests().len(), 2);
    let db = rusqlite::Connection::open(directory.path().join("fetch_cache.db")).unwrap();
    let (final_url, etag): (String, Option<String>) = db
        .query_row("SELECT final_url, etag FROM fetch_cache", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(final_url, server.url("/final"));
    assert!(etag.is_none_or(|etag| etag.is_empty()));
}

#[test]
fn slow_self_pointing_or_unsafe_refreshes_are_not_followed() {
    for refresh in [
        "<meta http-equiv=refresh content=\"5;url=/elsewhere\">",
        "<meta http-equiv=refresh content=\"30\">",
        "<meta http-equiv=refresh content=\"0;url=/start\">",
        "<meta http-equiv=refresh content=\"0;url=javascript:alert(1)\">",
        "<meta http-equiv=refresh content=\"0;url=file:///etc/passwd\">",
        "<meta http-equiv=refresh content=\"0;url=ftp://example.com/x\">",
        "<meta http-equiv=refresh content=\"0;url=data:text/html,hi\">",
        "<meta http-equiv=refresh content=\"0;\">",
        "<meta http-equiv=refresh content=\"soon;url=/elsewhere\">",
        "<noscript><meta http-equiv=refresh content=\"0;url=/elsewhere\"></noscript>",
        "<meta name=refresh content=\"0;url=/elsewhere\">",
    ] {
        let (_directory, cfg) = settings();
        let server = Server::new(vec![
            Reply::html(&format!(
                "<title>Stub</title>{refresh}<p>Stub page text.</p>"
            )),
            Reply::html("<p>Wrong page</p>"),
        ]);
        let outcome = run(&server, "/start", &cfg).unwrap();
        assert!(
            outcome.document().markdown.contains("Stub page text."),
            "{refresh}"
        );
        assert_eq!(server.requests().len(), 1, "{refresh}");
    }
}

#[test]
fn a_page_with_real_content_keeps_its_own_text_despite_a_refresh() {
    let (_directory, cfg) = settings();
    let article = format!(
        "<title>News</title><meta http-equiv=refresh content=\"1;url=/other\"><article><h1>News</h1><p>{}</p></article>",
        "Plenty of words in a real article that readers came for. ".repeat(6)
    );
    let server = Server::new(vec![Reply::html(&article), Reply::html("<p>Other</p>")]);
    let outcome = run(&server, "/news", &cfg).unwrap();
    assert!(outcome.document().markdown.contains("Plenty of words"));
    assert_eq!(server.requests().len(), 1);
}

#[test]
fn refresh_chains_are_bounded() {
    let (_directory, cfg) = settings();
    let hop = |to: &str| {
        Reply::html(&format!(
            "<meta http-equiv=refresh content=\"0;url={to}\"><p>Redirecting…</p>"
        ))
    };
    let server = Server::new(vec![
        hop("/b"),
        hop("/a"),
        hop("/b"),
        hop("/a"),
        hop("/b"),
        hop("/a"),
        hop("/b"),
        hop("/a"),
    ]);
    let message = failure(run(&server, "/a", &cfg));
    assert!(
        message.starts_with("Too many <meta> refresh redirects"),
        "{message}"
    );
    // The request itself and five followed refreshes.
    assert_eq!(server.requests().len(), 1 + MAX_REFRESHES);

    // A short chain of two refreshes arrives.
    let server = Server::new(vec![
        hop("/second"),
        hop("/third"),
        Reply::html("<article><p>Third page text.</p></article>"),
    ]);
    let outcome = run(&server, "/first", &cfg).unwrap();
    assert!(outcome.document().markdown.contains("Third page text."));
}

#[test]
fn a_refresh_forwards_no_credentials_to_another_origin() {
    let (_directory, cfg) = settings();
    let authorization = |request: &str| {
        request
            .lines()
            .find(|line| line.to_ascii_lowercase().starts_with("authorization:"))
            .map(str::to_owned)
    };
    let other = Server::new(vec![Reply::html(
        "<article><p>Other origin text.</p></article>",
    )]);
    let first = Server::new(vec![Reply::html(&format!(
        "<meta http-equiv=refresh content=\"0;url={}\"><p>Redirecting…</p>",
        other.url("/landing")
    ))]);
    let source = first
        .url("/start")
        .replacen("http://", "http://user:hunter2@", 1);
    let outcome = fetch_with_context(&source, &cfg, None, true).unwrap();
    assert!(outcome.document().markdown.contains("Other origin text."));
    assert!(authorization(&first.requests()[0]).is_some());
    assert!(
        authorization(&other.requests()[0]).is_none(),
        "{}",
        other.requests()[0]
    );

    // Not even within the same origin: a refresh is page content, and the
    // credentials the reader typed into the URL are not the page's to reuse.
    let (_directory, cfg) = settings();
    let same = Server::new(vec![
        Reply::html("<meta http-equiv=refresh content=\"0;url=/same\"><p>Redirecting…</p>"),
        Reply::html("<article><p>Same origin text.</p></article>"),
    ]);
    let source = same
        .url("/start")
        .replacen("http://", "http://user:hunter2@", 1);
    let outcome = fetch_with_context(&source, &cfg, None, true).unwrap();
    assert!(outcome.document().markdown.contains("Same origin text."));
    assert!(authorization(&same.requests()[0]).is_some());
    assert!(authorization(&same.requests()[1]).is_none());
}

#[test]
fn refresh_content_is_read_as_the_html_standard_reads_it() {
    let base = Url::parse("https://example.com/dir/page").unwrap();
    let target = |content: &str| refresh_target(content, &base).map(String::from);
    assert_eq!(target("0;url=/x").as_deref(), Some("https://example.com/x"));
    assert_eq!(
        target("0; url=x").as_deref(),
        Some("https://example.com/dir/x")
    );
    assert_eq!(
        target("2,url=https://other.org/y").as_deref(),
        Some("https://other.org/y")
    );
    assert_eq!(target("0 /x").as_deref(), Some("https://example.com/x"));
    assert_eq!(
        target("0;url=https://u:p@other.org/x").as_deref(),
        Some("https://other.org/x")
    );
    assert_eq!(
        target("0;URL = 'x y'").as_deref(),
        Some("https://example.com/dir/x%20y")
    );
    assert_eq!(
        target("0;urls.html").as_deref(),
        Some("https://example.com/dir/urls.html")
    );
    assert_eq!(target("2.0;/x").as_deref(), Some("https://example.com/x"));
    for refused in [
        "3;url=/x",
        "2.5;url=/x",
        "",
        ";url=/x",
        "0",
        "0;url=",
        "0;url=''",
        "0;url=page",
    ] {
        assert_eq!(target(refused), None, "{refused:?}");
    }
}
