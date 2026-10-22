//! A CLI or admin fetch of a public host keeps the proxy: the target is
//! checked here first, then the request goes to the proxy, and a redirect
//! into a private address is refused before the proxy is asked for it.
//!
//! Its own test binary: `HTTPS_PROXY`, the debug-only
//! `MARKITAI_TEST_PUBLIC_NET_ALLOW` (which makes a TEST-NET-2 address count as
//! public) and `MARKITAI_HOME` are set once here. Loopback is always direct,
//! so the "public" target is that unroutable address, answered by the proxy.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, LazyLock, Mutex};

use markitai_core::{ConvertContext, ConvertOptions};

/// An address outside every network, standing for a public host.
const PUBLIC: &str = "198.51.100.7";

/// A forward proxy on loopback that answers absolute-URI requests itself and
/// records each request line.
struct Proxy {
    port: u16,
    seen: Arc<Mutex<Vec<String>>>,
    _home: tempfile::TempDir,
}

static PROXY: LazyLock<Proxy> = LazyLock::new(|| {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&seen);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut buffer = [0u8; 4096];
            let read = stream.read(&mut buffer).unwrap_or(0);
            let request = String::from_utf8_lossy(&buffer[..read]).into_owned();
            let line = request.lines().next().unwrap_or_default().to_owned();
            recorded.lock().unwrap().push(line.clone());
            let target = line.split_whitespace().nth(1).unwrap_or("/");
            let path = target
                .strip_prefix(&format!("http://{PUBLIC}"))
                .unwrap_or(target);
            let article = "<html><body><article><h1>Through the proxy</h1><p>\
                Enough words on this page for it to be extracted as the \
                article, several times over so the reader keeps it. </p></article></body></html>";
            let (status, extra, body) = match path {
                "/redirect-private" => (302, "Location: http://10.0.0.1/secret\r\n", String::new()),
                "/meta-private" => (
                    200,
                    "Content-Type: text/html\r\n",
                    "<html><head><meta http-equiv=\"refresh\" \
                     content=\"0;url=http://10.0.0.1/secret\"></head><body>x</body></html>"
                        .into(),
                ),
                _ => (200, "Content-Type: text/html\r\n", article.to_owned()),
            };
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    let home = tempfile::tempdir().unwrap();
    // SAFETY: set once, before any fetch thread reads the environment.
    unsafe {
        std::env::set_var("MARKITAI_TEST_PUBLIC_NET_ALLOW", PUBLIC);
        // Named, so the public-only resolver must let the proxy's own
        // loopback name through.
        std::env::set_var("HTTPS_PROXY", format!("http://localhost:{port}"));
        std::env::set_var("MARKITAI_HOME", home.path());
        for name in ["NO_PROXY", "no_proxy", "HTTP_PROXY", "ALL_PROXY"] {
            std::env::remove_var(name);
        }
    }
    Proxy {
        port,
        seen,
        _home: home,
    }
});

fn convert(path: &str) -> markitai_core::Result<String> {
    LazyLock::force(&PROXY);
    let output = tempfile::tempdir().unwrap();
    let options = ConvertOptions {
        output_dir: Some(output.path().to_path_buf()),
        config: Some(serde_json::json!({"fetch": {"strategy": "static", "timeout": 20}})),
        ..Default::default()
    };
    markitai_core::convert_with_context(
        &format!("http://{PUBLIC}{path}"),
        options,
        ConvertContext::default(),
    )
    .map(|out| out.markdown)
}

fn seen() -> Vec<String> {
    PROXY.seen.lock().unwrap().clone()
}

#[test]
fn a_public_host_is_fetched_through_the_proxy() {
    let markdown = convert("/article").unwrap();
    assert!(markdown.contains("Through the proxy"), "{markdown}");
    assert!(
        seen()
            .iter()
            .any(|line| line.starts_with(&format!("GET http://{PUBLIC}/article "))),
        "proxy on {} saw {:?}",
        PROXY.port,
        seen()
    );
}

#[test]
fn a_redirect_into_a_private_address_is_refused_before_the_proxy() {
    let error = convert("/redirect-private").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Fetch target refused: a private or local address"),
        "{error}"
    );
    assert!(
        !seen().iter().any(|line| line.contains("10.0.0.1")),
        "{:?}",
        seen()
    );
}

#[test]
fn a_meta_refresh_into_a_private_address_is_refused_before_the_proxy() {
    let error = convert("/meta-private").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Fetch target refused: a private or local address"),
        "{error}"
    );
    assert!(
        !seen().iter().any(|line| line.contains("10.0.0.1")),
        "{:?}",
        seen()
    );
}
