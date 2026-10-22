//! A URL fetch held to public addresses: a user token's job (`public_only`),
//! and any fetch whose requested host is public, reaches public addresses
//! only, so a page cannot redirect or meta-refresh into this machine or its
//! network. The loopback fixture is treated as public through the debug-only
//! `MARKITAI_TEST_PUBLIC_NET_ALLOW`; a target outside it is refused.
//!
//! Its own test binary: the allowlist is set once here and read by the fetch
//! on its own threads, not shared with the rest of the suite.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::LazyLock;

use markitai_core::{ConvertContext, ConvertOptions};

/// Loopback counts as public for this binary only.
static ALLOW: LazyLock<()> = LazyLock::new(|| {
    // SAFETY: set once, before any fetch thread reads the variable.
    unsafe { std::env::set_var("MARKITAI_TEST_PUBLIC_NET_ALLOW", "127.0.0.1") };
});

/// A loopback server answering by path until it is dropped.
struct Fixture {
    base: String,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Fixture {
    fn new() -> Self {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let reply_base = base.clone();
        let thread = std::thread::spawn(move || {
            while !stopped.load(Ordering::Relaxed) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("loopback: {error}"),
                };
                stream
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut buffer = [0u8; 2048];
                let read = stream.read(&mut buffer).unwrap_or(0);
                let request = String::from_utf8_lossy(&buffer[..read]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                let article = "<html><body><article><h1>Fixture</h1><p>\
                    Enough words on this page for it to be extracted as the \
                    article, several times over so the reader keeps it. </p></article></body></html>";
                let (status, extra, body): (u16, String, String) = match path {
                    "/redirect-private" => (
                        302,
                        "Location: http://10.0.0.1/secret\r\n".into(),
                        String::new(),
                    ),
                    "/redirect-ok" => (
                        302,
                        format!("Location: {reply_base}/landing\r\n"),
                        String::new(),
                    ),
                    "/meta-private" => (
                        200,
                        "Content-Type: text/html\r\n".into(),
                        "<html><head><meta http-equiv=\"refresh\" \
                         content=\"0;url=http://10.0.0.1/secret\"></head><body>x</body></html>"
                            .into(),
                    ),
                    _ => (200, "Content-Type: text/html\r\n".into(), article.into()),
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        Self {
            base,
            stop,
            thread: Some(thread),
        }
    }
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn convert(url: &str, public_only: bool) -> markitai_core::Result<String> {
    convert_by(url, public_only, "static")
}

fn convert_by(url: &str, public_only: bool, strategy: &str) -> markitai_core::Result<String> {
    LazyLock::force(&ALLOW);
    let output = tempfile::tempdir().unwrap();
    let context = ConvertContext {
        public_only,
        ..ConvertContext::default()
    };
    let options = ConvertOptions {
        output_dir: Some(output.path().to_path_buf()),
        config: Some(serde_json::json!({"fetch": {"strategy": strategy}})),
        ..Default::default()
    };
    markitai_core::convert_with_context(url, options, context).map(|out| out.markdown)
}

/// A refused page fetch says so as a page's refusal, not a model's.
const REFUSED: &str =
    "Fetch target refused: a private or local address (this fetch may reach public addresses only)";

#[test]
fn a_public_only_fetch_refuses_a_redirect_into_a_private_address() {
    let site = Fixture::new();
    let error = convert(&site.url("/redirect-private"), true).unwrap_err();
    assert!(error.to_string().contains(REFUSED), "{error}");
}

#[test]
fn a_public_only_fetch_refuses_a_meta_refresh_into_a_private_address() {
    let site = Fixture::new();
    let error = convert(&site.url("/meta-private"), true).unwrap_err();
    assert!(error.to_string().contains(REFUSED), "{error}");
}

#[test]
fn a_public_only_fetch_still_follows_a_public_redirect_and_reads_the_page() {
    let site = Fixture::new();
    let markdown = convert(&site.url("/redirect-ok"), true).unwrap();
    assert!(markdown.contains("Fixture"), "{markdown}");
}

#[test]
fn a_public_requested_host_cannot_redirect_inward_without_the_user_flag() {
    // Not a user job (public_only false), but the requested host is public, so
    // a redirect into a private address is still refused (CLI and admin too).
    let site = Fixture::new();
    let error = convert(&site.url("/redirect-private"), false).unwrap_err();
    assert!(error.to_string().contains(REFUSED), "{error}");
}

#[test]
fn a_users_private_address_is_refused_as_a_page_before_any_request() {
    let error = convert("http://10.255.255.1/page", true).unwrap_err();
    assert!(error.to_string().contains(REFUSED), "{error}");
}

/// The browser route words the refusal as the static route does, not as
/// Chromium's `net::ERR_BLOCKED_BY_CLIENT`.
#[test]
#[ignore = "requires installed Chromium; loopback fixture only"]
fn the_browser_names_a_refused_redirect_as_the_static_route_does() {
    let site = Fixture::new();
    let error = convert_by(&site.url("/redirect-private"), true, "playwright").unwrap_err();
    assert!(error.to_string().contains(REFUSED), "{error}");
    assert!(!error.to_string().contains("BLOCKED_BY_CLIENT"), "{error}");
}
