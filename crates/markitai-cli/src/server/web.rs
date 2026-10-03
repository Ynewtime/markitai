//! The embedded browser workbench: fixed routes to files compiled into the
//! binary, never a filesystem lookup or a user-chosen path. The built bundle and
//! the vendored libraries are stored gzip-compressed; they are sent as they are
//! to clients that accept gzip and decompressed once for the others. Every
//! response revalidates (`no-cache`) against a content ETag, so a reload of an
//! unchanged page costs only 304 answers.
use super::State;
use axum::{
    Router,
    body::Body,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::Response,
    routing::get,
};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    sync::{Arc, OnceLock},
};

const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' blob:; font-src 'self'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

/// One embedded file. `gzip` marks bytes stored compressed.
struct Asset {
    bytes: &'static [u8],
    gzip: bool,
    content_type: &'static str,
    etag: OnceLock<HeaderValue>,
    plain: OnceLock<Vec<u8>>,
}

impl Asset {
    const fn new(bytes: &'static [u8], gzip: bool, content_type: &'static str) -> Self {
        Self {
            bytes,
            gzip,
            content_type,
            etag: OnceLock::new(),
            plain: OnceLock::new(),
        }
    }
    const fn raw(bytes: &'static [u8], content_type: &'static str) -> Self {
        Self::new(bytes, false, content_type)
    }
    const fn gzip(bytes: &'static [u8], content_type: &'static str) -> Self {
        Self::new(bytes, true, content_type)
    }
    /// A weak validator: both encodings of one file share it.
    fn etag(&self) -> &HeaderValue {
        self.etag.get_or_init(|| {
            let digest = Sha256::digest(self.bytes);
            let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
            HeaderValue::from_str(&format!("W/\"{hex}\"")).expect("hex is a valid header value")
        })
    }
    /// The decompressed file, for clients that do not accept gzip.
    fn plain(&self) -> &[u8] {
        if !self.gzip {
            return self.bytes;
        }
        self.plain.get_or_init(|| {
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(self.bytes)
                .read_to_end(&mut out)
                .expect("embedded gzip assets are produced by build.mjs and verified by tests");
            out
        })
    }
}

macro_rules! dist {
    ($name:literal) => {
        include_bytes!(concat!("web/dist/", $name))
    };
}
macro_rules! vendor {
    ($name:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../vendor/web/",
            $name
        ))
    };
}

const JS: &str = "text/javascript; charset=utf-8";
static INDEX: Asset = Asset::raw(dist!("index.html"), "text/html; charset=utf-8");
static APP_JS: Asset = Asset::gzip(dist!("app.js.gz"), JS);
static APP_CSS: Asset = Asset::gzip(dist!("app.css.gz"), "text/css; charset=utf-8");
static BOOT_JS: Asset = Asset::raw(dist!("boot.js"), JS);
static LOGO: Asset = Asset::raw(dist!("logo.svg"), "image/svg+xml");
static MARKED: Asset = Asset::gzip(dist!("marked.js.gz"), JS);
static PURIFY: Asset = Asset::gzip(dist!("purify.js.gz"), JS);
static FONT: Asset = Asset::raw(vendor!("inter-latin-wght-normal.woff2"), "font/woff2");

/// Whether the request accepts gzip (`gzip` or `*` with a nonzero weight).
fn accepts_gzip(headers: &HeaderMap) -> bool {
    headers
        .get_all(header::ACCEPT_ENCODING)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|part| {
            let mut fields = part.split(';');
            let coding = fields.next().unwrap_or("").trim();
            let weight = fields
                .filter_map(|field| field.trim().strip_prefix("q="))
                .find_map(|q| q.trim().parse::<f32>().ok())
                .unwrap_or(1.0);
            (coding.eq_ignore_ascii_case("gzip") || coding == "*") && weight > 0.0
        })
}

/// `If-None-Match` holds this validator (weak comparison) or `*`.
fn not_modified(headers: &HeaderMap, etag: &HeaderValue) -> bool {
    let ours = etag.to_str().unwrap_or("").trim_start_matches("W/");
    headers
        .get_all(header::IF_NONE_MATCH)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(|tag| tag.trim())
        .any(|tag| tag == "*" || tag.trim_start_matches("W/") == ours)
}

fn serve(asset: &'static Asset, request: &HeaderMap) -> Response {
    let gzip = asset.gzip && accepts_gzip(request);
    let mut response = if not_modified(request, asset.etag()) {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = StatusCode::NOT_MODIFIED;
        response
    } else {
        let body: &'static [u8] = if gzip { asset.bytes } else { asset.plain() };
        let mut response = Response::new(Body::from(body));
        if gzip {
            response
                .headers_mut()
                .insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        }
        response
    };
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(asset.content_type),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(header::ETAG, asset.etag().clone());
    if asset.gzip {
        headers.insert(header::VARY, HeaderValue::from_static("accept-encoding"));
    }
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    headers.insert("content-security-policy", HeaderValue::from_static(CSP));
    response
}

fn route(asset: &'static Asset) -> axum::routing::MethodRouter<Arc<State>> {
    get(move |headers: HeaderMap| async move { serve(asset, &headers) })
}

pub(super) fn routes() -> Router<Arc<State>> {
    Router::new()
        // The workspace view is a real address so reloads and Back keep it.
        .route("/", route(&INDEX))
        .route("/jobs", route(&INDEX))
        .route("/ui/app.js", route(&APP_JS))
        .route("/ui/app.css", route(&APP_CSS))
        .route("/ui/boot.js", route(&BOOT_JS))
        .route("/ui/logo.svg", route(&LOGO))
        .route("/ui/inter-latin-wght.woff2", route(&FONT))
        .route("/ui/marked.js", route(&MARKED))
        .route("/ui/purify.js", route(&PURIFY))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(*name, HeaderValue::from_static(value));
        }
        map
    }

    #[test]
    fn gzip_is_chosen_by_coding_and_weight() {
        assert!(accepts_gzip(&headers(&[("accept-encoding", "gzip, br")])));
        assert!(accepts_gzip(&headers(&[(
            "accept-encoding",
            "br;q=1, GZIP;q=0.5"
        )])));
        assert!(accepts_gzip(&headers(&[("accept-encoding", "*")])));
        assert!(!accepts_gzip(&headers(&[("accept-encoding", "gzip;q=0")])));
        assert!(!accepts_gzip(&headers(&[(
            "accept-encoding",
            "br, deflate"
        )])));
        assert!(!accepts_gzip(&headers(&[])));
    }

    #[test]
    fn validators_compare_weakly_and_every_embedded_file_decodes() {
        let tag = APP_JS.etag().clone();
        let text = tag.to_str().unwrap().to_owned();
        assert!(text.starts_with("W/\"") && text.len() == 20, "{text}");
        let strong = text.trim_start_matches("W/").to_owned();
        let other = HeaderValue::from_static("\"0000000000000000\"");
        assert!(not_modified(
            &headers(&[("if-none-match", "\"x\", *")]),
            &tag
        ));
        let mut list = HeaderMap::new();
        list.insert(
            header::IF_NONE_MATCH,
            HeaderValue::from_str(&format!("\"abc\", {strong}")).unwrap(),
        );
        assert!(not_modified(&list, &tag));
        assert!(!not_modified(&list, &other));
        for asset in [&APP_JS, &APP_CSS, &MARKED, &PURIFY] {
            assert!(!asset.plain().is_empty() && asset.plain().len() > asset.bytes.len());
        }
        assert_eq!(MARKED.plain(), vendor!("marked.js"));
        assert_eq!(PURIFY.plain(), vendor!("purify.js"));
    }

    /// Field names of one `export interface` in the workbench's API types,
    /// with `?` marking an optional field (inherited fields included).
    fn interface(name: &str) -> Vec<(String, bool)> {
        let source = include_str!("web/src/api/types.ts");
        let start = source
            .find(&format!("export interface {name} "))
            .unwrap_or_else(|| panic!("interface {name}"));
        let head_end = start + source[start..].find('{').unwrap();
        let body_end = head_end + source[head_end..].find("\n}").unwrap();
        let mut fields: Vec<(String, bool)> = source[head_end + 1..body_end]
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                let (field, _) = line.split_once(':')?;
                let optional = field.ends_with('?');
                let field = field.trim_end_matches('?');
                (!line.starts_with("//")
                    && field.chars().all(|c| c == '_' || c.is_ascii_alphanumeric()))
                .then(|| (field.to_owned(), optional))
            })
            .collect();
        if let Some(parent) = source[start..head_end]
            .split_once(" extends ")
            .map(|(_, parent)| parent.trim())
        {
            fields.extend(interface(parent));
        }
        fields
    }

    fn keys(value: &serde_json::Value) -> std::collections::BTreeSet<String> {
        value.as_object().unwrap().keys().cloned().collect()
    }

    /// The page reads exactly what the service writes: every serialized key is
    /// declared, and every field the page treats as always present is sent.
    fn agree(name: &str, full: &serde_json::Value, minimal: &serde_json::Value) {
        let declared = interface(name);
        let all: std::collections::BTreeSet<String> =
            declared.iter().map(|(field, _)| field.clone()).collect();
        assert_eq!(
            keys(full),
            all,
            "{name}: serialized keys and declared fields differ"
        );
        for (field, optional) in &declared {
            if !optional {
                assert!(
                    minimal.get(field).is_some(),
                    "{name}.{field} is declared required but omitted"
                );
            }
        }
    }

    #[test]
    fn the_workbench_types_match_the_service_payloads() {
        use super::super::{
            jobs::JobData,
            types::{Item, RerunFailure, RerunOperation, now},
        };
        use crate::diagnostics::{AttemptDiagnostics, Operation};
        use serde_json::json;
        use std::collections::HashMap;
        let minimal = Item::new(1, "a.pdf".into(), "file", None);
        let mut full = minimal.clone();
        full.error = Some("failed".into());
        full.error_code = Some("conversion_error".into());
        let usage = markitai_core::ConversionUsage {
            requests: 1,
            cost_usd: 0.5,
            by_model: json!({"m":{"requests":1,"cost_usd":0.5,"priced_requests":1,"unpriced_requests":0,"cost_status":"complete"}})
                .as_object()
                .unwrap()
                .clone(),
            ..Default::default()
        };
        full.cost_usd = Some(0.5);
        full.pricing = crate::pricing::Pricing::from_usage(&usage);
        full.diagnostics = AttemptDiagnostics::failed(Operation::Retry, "later failure", usage);
        let failure = RerunFailure::new(
            RerunOperation::Retry,
            "conversion_error",
            "later failure".into(),
        );
        agree("RerunFailure", &json!(failure), &json!(failure));
        full.rerun_failure = Some(failure);
        assert!(full.pricing.is_some() && full.diagnostics.is_some());
        agree("ItemPayload", &json!(full), &json!(minimal));
        let job = |items: Vec<Item>, persistence: Option<String>| JobData {
            id: "job".into(),
            created_at: now(),
            finished_at: Some(now()),
            status: "done".into(),
            persistence_error: persistence,
            options: json!({}),
            items,
            size: 0,
            bases: HashMap::new(),
            assets: HashMap::new(),
            item_options: HashMap::new(),
            transactions: Vec::new(),
        };
        let mut priced = full.clone();
        priced.status = "done".into();
        let complete = job(vec![priced], Some("history could not be persisted".into()));
        let plain = job(vec![minimal], None);
        agree("JobSnapshot", &complete.snapshot(), &plain.snapshot());
        agree("HistoryEntry", &complete.history(), &plain.history());
    }
}
