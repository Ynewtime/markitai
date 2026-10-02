use super::cache_tests::{Reply, Server, settings};
use super::*;

const PDF: &[u8] = include_bytes!("../pdf_raster/fixtures/mixed-native-scanned-blank.pdf");

fn received(outcome: FetchOutcome) -> DownloadedPdf {
    assert!(!outcome.cache_hit);
    assert!(outcome.screenshots.is_empty());
    match outcome.content {
        FetchContent::Pdf(pdf) => pdf,
        FetchContent::Document(_) => panic!("expected owned PDF response bytes"),
    }
}

#[test]
fn deferred_static_and_auto_capture_each_download_extensionless_pdf_once() {
    for (strategy, ocr, capture, only) in [
        ("static", true, false, false),
        ("static", false, true, false),
        ("auto", true, false, false),
        ("auto", false, true, false),
        ("auto", false, false, true),
    ] {
        let (_directory, mut cfg) = settings();
        cfg["cache"]["enabled"] = json!(false);
        cfg["fetch"]["strategy"] = json!(strategy);
        cfg["ocr"]["enabled"] = json!(ocr);
        cfg["screenshot"]["enabled"] = json!(capture);
        cfg["screenshot"]["screenshot_only"] = json!(only);
        let server = Server::new(vec![Reply::bytes("application/octet-stream", PDF)]);
        let source = server.url("/download?id=opaque");
        // PDF classification also permits a config-only `only` flag without
        // implying capture or requiring an output directory at fetch time.
        let pdf = received(fetch_with_context(&source, &cfg, None, false).unwrap());
        assert_eq!(pdf.bytes, PDF);
        assert_eq!(pdf.final_url, source);
        assert!(pdf.warnings.is_empty());
        assert_eq!(
            server.requests().len(),
            1,
            "{strategy}/{ocr}/{capture}/{only}"
        );
        assert!(server.requests()[0].contains(STATIC_ACCEPT));
        assert_eq!(cfg["screenshot"]["enabled"], capture);
    }
}

#[test]
fn redirect_handoff_keeps_final_url_and_original_request_identity() {
    let (_directory, mut cfg) = settings();
    cfg["ocr"]["enabled"] = json!(true);
    let server = Server::new(vec![
        Reply::text("")
            .status(302)
            .header("Location", "/opaque?token=private&view=full"),
        Reply::bytes("application/x-pdf; charset=binary", PDF),
    ]);
    let source = server.url("/original.pdf");
    let pdf = received(fetch_with_context(&source, &cfg, Some("static"), true).unwrap());
    assert_eq!(pdf.bytes, PDF);
    assert_eq!(pdf.final_url, server.url("/opaque?token=private&view=full"));
    let requests = server.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with("GET /original.pdf "));
    assert!(requests[1].starts_with("GET /opaque?token=private&view=full "));
    assert!(
        fetch_cache::Cache::from_config(&cfg)
            .unwrap()
            .get(&source, Some("static"))
            .unwrap()
            .is_none()
    );
}

#[test]
fn text_and_html_remain_documents_even_with_pdf_labels_or_paths() {
    for (mime, body, expected) in [
        (
            "text/plain",
            "%PDF-1.7\nThis is a literal file-format example.",
            "%PDF-1.7",
        ),
        ("text/markdown", "%PDF-1.7\nA Markdown example.", "%PDF-1.7"),
        (
            "text/html",
            "<article><p>Authoritative HTML response.</p></article>",
            "Authoritative HTML",
        ),
        (
            "application/pdf",
            "<!doctype html><html><article><p>HTML download error page.</p></article></html>",
            "HTML download error",
        ),
    ] {
        let (_directory, mut cfg) = settings();
        cfg["ocr"]["enabled"] = json!(true);
        let server = Server::new(vec![Reply::bytes(mime, body.as_bytes())]);
        let outcome = fetch_with_context(&server.url("/looks-like.pdf"), &cfg, None, true).unwrap();
        assert!(outcome.document().markdown.contains(expected));
        assert_eq!(server.requests().len(), 1);
    }
}

fn response(mime: &str, path: &str, bytes: Vec<u8>) -> StaticResponse {
    StaticResponse {
        effective_url: Url::parse(&format!("https://example.test/{path}")).unwrap(),
        content_type: mime.into(),
        mime: mime.into(),
        etag: None,
        last_modified: None,
        bytes,
    }
}

#[test]
fn binary_hints_require_header_evidence_but_pdf_mime_can_report_corruption_later() {
    assert!(
        response(
            "application/pdf",
            "opaque",
            b"broken representation".to_vec()
        )
        .is_pdf()
    );
    assert!(response("application/x-pdf", "opaque", Vec::new()).is_pdf());
    assert!(response("", "opaque", b"%PDF-2.0\n".to_vec()).is_pdf());
    assert!(
        response(
            "application/octet-stream",
            "opaque",
            b"leading bytes\n%PDF-1.7\n".to_vec()
        )
        .is_pdf()
    );
    assert!(response("application/custom", "file.PDF", b"%PDF-1.7\n".to_vec()).is_pdf());
    assert!(!response("application/custom", "opaque", b"%PDF-1.7\n".to_vec()).is_pdf());
    assert!(
        !response(
            "application/octet-stream",
            "file.pdf",
            b"unrelated bytes".to_vec()
        )
        .is_pdf()
    );
    assert!(!response("text/xml", "file.pdf", b"%PDF-1.7\n".to_vec()).is_pdf());
    let mut boundary = vec![b' '; 1016];
    boundary.extend_from_slice(b"%PDF-1.7");
    assert!(response("", "opaque", boundary.clone()).is_pdf());
    boundary.insert(0, b' ');
    assert!(!response("", "opaque", boundary).is_pdf());
}

#[test]
fn accepted_conditional_pdf_removes_old_html_before_a_downstream_parse_failure() {
    let (_directory, mut cfg) = settings();
    let server = Server::new(vec![
        Reply::html("<article><p>Old cached HTML.</p></article>").header("ETag", "old-html"),
        Reply::bytes("application/pdf", b"corrupt PDF representation"),
        Reply::bytes("application/pdf", PDF),
    ]);
    let source = server.url("/download");
    assert!(
        fetch_with_context(&source, &cfg, None, true)
            .unwrap()
            .document()
            .markdown
            .contains("Old cached")
    );
    cfg["ocr"]["enabled"] = json!(true);
    let pdf = received(fetch_with_context(&source, &cfg, None, true).unwrap());
    assert_eq!(pdf.bytes, b"corrupt PDF representation");
    assert!(formats::extract_pdf_pages(&pdf.bytes).is_err());
    assert_eq!(
        server.requests().len(),
        2,
        "reader failure must not trigger another GET"
    );
    let cache = fetch_cache::Cache::from_config(&cfg).unwrap();
    assert!(cache.get(&source, None).unwrap().is_none());
    assert!(
        server.requests()[1]
            .to_ascii_lowercase()
            .contains("if-none-match: old-html")
    );
    assert_eq!(
        received(fetch_with_context(&source, &cfg, None, true).unwrap()).bytes,
        PDF
    );
    assert_eq!(server.requests().len(), 3);
    assert!(
        !server.requests()[2]
            .to_ascii_lowercase()
            .contains("if-none-match:")
    );
    assert!(cache.get(&source, None).unwrap().is_none());
}

#[test]
fn auto_capture_probe_bypasses_fresh_html_cache_and_invalidates_it_for_pdf() {
    let (_directory, mut cfg) = settings();
    let server = Server::new(vec![
        Reply::html("<article><p>Previous unvalidated HTML.</p></article>"),
        Reply::bytes("application/octet-stream", PDF),
    ]);
    let source = server.url("/opaque");
    fetch_with_context(&source, &cfg, None, true).unwrap();
    assert!(
        fetch_cache::Cache::from_config(&cfg)
            .unwrap()
            .get(&source, None)
            .unwrap()
            .is_some()
    );
    cfg["fetch"]["strategy"] = json!("auto");
    cfg["screenshot"]["enabled"] = json!(true);
    assert_eq!(
        received(fetch_with_context(&source, &cfg, None, true).unwrap()).bytes,
        PDF
    );
    assert_eq!(server.requests().len(), 2);
    assert!(
        !server.requests()[1]
            .to_ascii_lowercase()
            .contains("if-none-match:")
    );
    assert!(
        fetch_cache::Cache::from_config(&cfg)
            .unwrap()
            .get(&source, None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn default_pdf_extraction_stays_document_only_when_no_media_is_requested() {
    let (_directory, cfg) = settings();
    let server = Server::new(vec![Reply::bytes("application/pdf", PDF)]);
    let outcome = fetch_with_context(&server.url("/document"), &cfg, None, true).unwrap();
    assert_eq!(outcome.document().metadata["format"], "PDF");
    assert!(
        outcome
            .document()
            .markdown
            .contains("Native page retains its structured text.")
    );
    assert!(!outcome.cache_hit);
    assert_eq!(server.requests().len(), 1);
}

#[test]
fn classification_rejects_invisible_html_capture_without_starting_a_browser() {
    for strategy in ["auto", "static"] {
        let (_directory, mut cfg) = settings();
        cfg["cache"]["enabled"] = json!(false);
        cfg["fetch"]["strategy"] = json!(strategy);
        cfg["screenshot"]["screenshot_only"] = json!(true);
        let server = Server::new(vec![Reply::html("<article><p>Actual HTML.</p></article>")]);
        let error = fetch_with_context(&server.url("/opaque"), &cfg, None, false)
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidInput(_)), "{error}");
        assert!(error.to_string().contains("output_dir"));
        assert_eq!(server.requests().len(), 1);
    }
    for strategy in ["jina", "defuddle"] {
        let (_directory, mut cfg) = settings();
        cfg["fetch"]["strategy"] = json!(strategy);
        cfg["screenshot"]["screenshot_only"] = json!(true);
        let server = Server::new(Vec::new());
        assert!(matches!(
            fetch_with_context(&server.url("/opaque"), &cfg, None, false),
            Err(Error::InvalidInput(_))
        ));
        assert!(server.requests().is_empty());
    }
}

#[test]
fn authenticated_response_classification_keeps_pdf_memory_and_html_output_contracts() {
    let (_directory, mut cfg) = settings();
    cfg["screenshot"]["screenshot_only"] = json!(true);
    let pdf = browser::BrowserResponse::Pdf(browser::BrowserPdf {
        bytes: PDF.to_vec(),
        final_url: "https://example.test/private-download".into(),
        warnings: Vec::new(),
    });
    let result = received(browser_response_outcome(pdf, &cfg, false).unwrap());
    assert_eq!(result.bytes, PDF);
    assert_eq!(result.strategy, "playwright");
    let html = browser::BrowserResponse::Page(browser::BrowserPage {
        html: "<article><p>Private HTML content.</p></article>".into(),
        final_url: "https://example.test/private-page".into(),
        title: "Private page".into(),
        screenshots: Vec::new(),
        warnings: Vec::new(),
    });
    assert!(matches!(
        browser_response_outcome(html, &cfg, false),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn auto_probe_does_not_hide_status_or_body_limits_behind_browser_fallback() {
    let (_directory, mut cfg) = settings();
    cfg["fetch"]["strategy"] = json!("auto");
    cfg["screenshot"]["enabled"] = json!(true);
    let server = Server::new(vec![Reply::text("temporarily unavailable").status(503)]);
    let error = fetch_with_context(&server.url("/opaque"), &cfg, None, true)
        .err()
        .unwrap();
    assert!(error.to_string().contains("HTTP 503"), "{error}");
    assert_eq!(server.requests().len(), 1);

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/download", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let mut request_reader = bounded_fixture_io::Reader::new(
            &stream,
            std::time::Instant::now() + Duration::from_secs(3),
        );
        let mut request = [0u8; 4096];
        assert!(request_reader.read(&mut request).unwrap() > 0);
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", MAX_RESPONSE + 1).unwrap();
    });
    let result = fetch_with_context(&url, &cfg, None, true);
    handle.join().unwrap();
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("Response exceeds 100 MiB")
    );
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}

#[test]
fn hidden_text_policy_defers_off_and_remove_without_media_or_a_second_download() {
    for mode in ["off", "remove"] {
        let (_directory, mut cfg) = settings();
        cfg["security"]["pdf_sanitize"] = json!(mode);
        cfg["ocr"]["enabled"] = json!(false);
        cfg["screenshot"]["enabled"] = json!(false);
        cfg["screenshot"]["screenshot_only"] = json!(false);
        let server = Server::new(vec![Reply::bytes("application/pdf", PDF)]);
        let source = server.url("/opaque");
        let pdf = received(fetch_with_context(&source, &cfg, Some("static"), true).unwrap());
        assert_eq!(pdf.bytes, PDF);
        assert_eq!(pdf.final_url, source);
        assert_eq!(server.requests().len(), 1);
        assert!(
            fetch_cache::Cache::from_config(&cfg)
                .unwrap()
                .get(&source, Some("static"))
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn policy_deferred_pdf_invalidates_html_cache_during_conditional_revalidation() {
    let (_directory, mut cfg) = settings();
    cfg["ocr"]["enabled"] = json!(false);
    cfg["screenshot"]["enabled"] = json!(false);
    cfg["screenshot"]["screenshot_only"] = json!(false);
    let server = Server::new(vec![
        Reply::html("<article><p>Old cached HTML report.</p></article>").header("ETag", "old-html"),
        Reply::bytes("application/pdf", PDF),
        Reply::bytes("application/pdf", PDF),
    ]);
    let source = server.url("/document");
    assert!(
        fetch_with_context(&source, &cfg, Some("static"), true)
            .unwrap()
            .document()
            .markdown
            .contains("Old cached")
    );
    for mode in ["off", "remove"] {
        cfg["security"]["pdf_sanitize"] = json!(mode);
        let pdf = received(fetch_with_context(&source, &cfg, Some("static"), true).unwrap());
        assert_eq!(pdf.bytes, PDF);
        assert!(
            fetch_cache::Cache::from_config(&cfg)
                .unwrap()
                .get(&source, Some("static"))
                .unwrap()
                .is_none()
        );
    }
    let requests = server.requests();
    assert_eq!(requests.len(), 3);
    assert!(
        requests[1]
            .to_ascii_lowercase()
            .contains("if-none-match: old-html")
    );
    assert!(!requests[2].to_ascii_lowercase().contains("if-none-match:"));
}
