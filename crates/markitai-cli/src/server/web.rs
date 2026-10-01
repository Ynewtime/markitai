//! Fixed embedded resources; no filesystem fallback or user-controlled asset paths.
use super::State;
use axum::{
    Router,
    body::Body,
    http::{HeaderValue, header},
    response::Response,
    routing::get,
};
use std::sync::Arc;

fn asset(bytes: &'static [u8], content_type: &'static str) -> Response {
    let mut response = Response::new(Body::from(bytes));
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    headers.insert("content-security-policy", HeaderValue::from_static("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' blob:; font-src 'self'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"));
    response
}

pub(super) fn routes() -> Router<Arc<State>> {
    Router::new()
        .route(
            "/",
            get(|| async { asset(include_bytes!("web/index.html"), "text/html; charset=utf-8") }),
        )
        .route(
            "/ui/app.js",
            get(|| async {
                asset(
                    include_bytes!("web/app.js"),
                    "text/javascript; charset=utf-8",
                )
            }),
        )
        .route(
            "/ui/api.js",
            get(|| async {
                asset(
                    include_bytes!("web/api.js"),
                    "text/javascript; charset=utf-8",
                )
            }),
        )
        .route(
            "/ui/preview.js",
            get(|| async {
                asset(
                    include_bytes!("web/preview.js"),
                    "text/javascript; charset=utf-8",
                )
            }),
        )
        .route(
            "/ui/result-tools.js",
            get(|| async {
                asset(
                    include_bytes!("web/result-tools.js"),
                    "text/javascript; charset=utf-8",
                )
            }),
        )
        .route(
            "/ui/settings.js",
            get(|| async {
                asset(
                    include_bytes!("web/settings.js"),
                    "text/javascript; charset=utf-8",
                )
            }),
        )
        .route(
            "/ui/i18n.js",
            get(|| async {
                asset(
                    include_bytes!("web/i18n.js"),
                    "text/javascript; charset=utf-8",
                )
            }),
        )
        .route(
            "/ui/boot.js",
            get(|| async {
                asset(
                    include_bytes!("web/boot.js"),
                    "text/javascript; charset=utf-8",
                )
            }),
        )
        .route(
            "/ui/icon.svg",
            get(|| async { asset(include_bytes!("web/icon.svg"), "image/svg+xml") }),
        )
        .route(
            "/ui/style.css",
            get(|| async { asset(include_bytes!("web/style.css"), "text/css; charset=utf-8") }),
        )
        .route(
            "/ui/marked.js",
            get(|| async {
                asset(
                    include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../vendor/web/marked.js"
                    )),
                    "text/javascript; charset=utf-8",
                )
            }),
        )
        .route(
            "/ui/purify.js",
            get(|| async {
                asset(
                    include_bytes!(concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/../../vendor/web/purify.js"
                    )),
                    "text/javascript; charset=utf-8",
                )
            }),
        )
}
