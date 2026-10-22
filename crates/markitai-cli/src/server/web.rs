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

const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' blob: https:; media-src https:; font-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

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
static HIGHLIGHT_JS: Asset = Asset::gzip(dist!("highlight.js.gz"), JS);
static BRANDS_JS: Asset = Asset::gzip(dist!("brands.js.gz"), JS);
static I18N_ZH: Asset = Asset::gzip(dist!("i18n-zh.js.gz"), JS);
static I18N_JA: Asset = Asset::gzip(dist!("i18n-ja.js.gz"), JS);
static MATH_JS: Asset = Asset::gzip(dist!("math.js.gz"), JS);
static MATH_CSS: Asset = Asset::gzip(dist!("math.css.gz"), "text/css; charset=utf-8");
static MERMAID_JS: Asset = Asset::gzip(dist!("mermaid.js.gz"), JS);
static LOGO: Asset = Asset::raw(dist!("logo.svg"), "image/svg+xml");
static MARKED: Asset = Asset::gzip(dist!("marked.js.gz"), JS);
static PURIFY: Asset = Asset::gzip(dist!("purify.js.gz"), JS);
static FONT: Asset = Asset::raw(vendor!("inter-latin-wght-normal.woff2"), "font/woff2");
// The other Inter subsets: a browser fetches one only for text that needs it.
static FONT_LATIN_EXT: Asset =
    Asset::raw(vendor!("inter-latin-ext-wght-normal.woff2"), "font/woff2");
static FONT_CYRILLIC: Asset = Asset::raw(vendor!("inter-cyrillic-wght-normal.woff2"), "font/woff2");
static FONT_CYRILLIC_EXT: Asset = Asset::raw(
    vendor!("inter-cyrillic-ext-wght-normal.woff2"),
    "font/woff2",
);
static FONT_GREEK: Asset = Asset::raw(vendor!("inter-greek-wght-normal.woff2"), "font/woff2");
static FONT_GREEK_EXT: Asset =
    Asset::raw(vendor!("inter-greek-ext-wght-normal.woff2"), "font/woff2");
static FONT_VIETNAMESE: Asset =
    Asset::raw(vendor!("inter-vietnamese-wght-normal.woff2"), "font/woff2");

/// The KaTeX faces `dist/math.css` names, each served at `/ui/<file name>`; a browser fetches only the
/// faces a formula uses.
macro_rules! katex_fonts {
    ($($name:literal),* $(,)?) => {
        &[$(($name, {
            static FACE: Asset = Asset::raw(vendor!($name), "font/woff2");
            &FACE
        })),*]
    };
}
static KATEX_FONTS: &[(&str, &Asset)] = katex_fonts![
    "KaTeX_AMS-Regular.woff2",
    "KaTeX_Caligraphic-Bold.woff2",
    "KaTeX_Caligraphic-Regular.woff2",
    "KaTeX_Fraktur-Bold.woff2",
    "KaTeX_Fraktur-Regular.woff2",
    "KaTeX_Main-Bold.woff2",
    "KaTeX_Main-BoldItalic.woff2",
    "KaTeX_Main-Italic.woff2",
    "KaTeX_Main-Regular.woff2",
    "KaTeX_Math-BoldItalic.woff2",
    "KaTeX_Math-Italic.woff2",
    "KaTeX_SansSerif-Bold.woff2",
    "KaTeX_SansSerif-Italic.woff2",
    "KaTeX_SansSerif-Regular.woff2",
    "KaTeX_Script-Regular.woff2",
    "KaTeX_Size1-Regular.woff2",
    "KaTeX_Size2-Regular.woff2",
    "KaTeX_Size3-Regular.woff2",
    "KaTeX_Size4-Regular.woff2",
    "KaTeX_Typewriter-Regular.woff2",
];

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

/// A browser asking for a page that does not exist gets the workbench with
/// status 404, so it can say so and offer the way back; anything else (an API
/// client, an asset) gets the JSON error.
pub(super) fn missing_page(
    method: &axum::http::Method,
    path: &str,
    headers: &HeaderMap,
) -> Option<Response> {
    let page = (method == axum::http::Method::GET || method == axum::http::Method::HEAD)
        && path != "/api"
        && !path.starts_with("/api/")
        && !path.starts_with("/ui/")
        && headers
            .get_all(header::ACCEPT)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .any(|value| value.contains("text/html"));
    page.then(|| {
        let mut headers = headers.clone();
        // Always the page itself, never "not modified".
        headers.remove(header::IF_NONE_MATCH);
        let mut response = serve(&INDEX, &headers);
        *response.status_mut() = StatusCode::NOT_FOUND;
        response
    })
}

pub(super) fn routes() -> Router<Arc<State>> {
    let mut router = Router::new()
        // Every view is a real address so reloads and Back keep it.
        .route("/", route(&INDEX))
        .route("/jobs", route(&INDEX))
        .route("/settings", route(&INDEX))
        .route("/settings/models", route(&INDEX))
        .route("/settings/appearance", route(&INDEX))
        .route("/settings/access", route(&INDEX))
        .route("/settings/config", route(&INDEX))
        .route("/ui/app.js", route(&APP_JS))
        .route("/ui/app.css", route(&APP_CSS))
        .route("/ui/boot.js", route(&BOOT_JS))
        .route("/ui/highlight.js", route(&HIGHLIGHT_JS))
        .route("/ui/brands.js", route(&BRANDS_JS))
        .route("/ui/math.js", route(&MATH_JS))
        .route("/ui/math.css", route(&MATH_CSS))
        .route("/ui/mermaid.js", route(&MERMAID_JS))
        .route("/ui/i18n-zh.js", route(&I18N_ZH))
        .route("/ui/i18n-ja.js", route(&I18N_JA))
        .route("/ui/logo.svg", route(&LOGO))
        .route("/ui/inter-latin-wght.woff2", route(&FONT))
        .route("/ui/inter-latin-ext-wght.woff2", route(&FONT_LATIN_EXT))
        .route("/ui/inter-cyrillic-wght.woff2", route(&FONT_CYRILLIC))
        .route(
            "/ui/inter-cyrillic-ext-wght.woff2",
            route(&FONT_CYRILLIC_EXT),
        )
        .route("/ui/inter-greek-wght.woff2", route(&FONT_GREEK))
        .route("/ui/inter-greek-ext-wght.woff2", route(&FONT_GREEK_EXT))
        .route("/ui/inter-vietnamese-wght.woff2", route(&FONT_VIETNAMESE))
        .route("/ui/marked.js", route(&MARKED))
        .route("/ui/purify.js", route(&PURIFY));
    for (name, face) in KATEX_FONTS {
        router = router.route(&format!("/ui/{name}"), route(face));
    }
    router
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
        for asset in [
            &APP_JS,
            &APP_CSS,
            &HIGHLIGHT_JS,
            &BRANDS_JS,
            &MATH_JS,
            &MATH_CSS,
            &MERMAID_JS,
            &I18N_ZH,
            &I18N_JA,
            &MARKED,
            &PURIFY,
        ] {
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
        // Inline object members belong to that nested shape, not its parent.
        let mut nested = 0usize;
        let mut fields: Vec<(String, bool)> = source[head_end + 1..body_end]
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                if line.starts_with("//") {
                    return None;
                }
                let top_level = nested == 0;
                for character in line.chars() {
                    match character {
                        '{' => nested += 1,
                        '}' => nested = nested.checked_sub(1).expect("balanced interface objects"),
                        _ => {}
                    }
                }
                if !top_level {
                    return None;
                }
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
            types::{Item, JobOptions, RerunFailure, RerunOperation, now},
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
        let selected = JobOptions::from_value(json!({
            "preset":"minimal", "llm":false, "ocr":false, "profile":"rag",
            "alt":false, "desc":false, "screenshot":false, "screenshot_only":false,
            "pure":false, "no_cache":true, "no_compress":true,
            "strategy":"static", "backend":"cloudflare", "remote_processing":"cloudflare"
        }))
        .unwrap();
        agree(
            "JobOptions",
            &json!(selected),
            &json!(JobOptions::default()),
        );
        let disclosure = selected
            .remote_disclosure()
            .expect("selected request has disclosure");
        agree("RemoteProcessing", &disclosure, &disclosure);
        full.remote_processing = Some(disclosure);
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
            label: None,
            owner: None,
        };
        full.status = "done".into();
        full.models = vec!["m".into()];
        full.source_size = Some(5);
        let priced = full.clone();
        let mut complete = job(vec![priced], Some("history could not be persisted".into()));
        complete
            .item_options
            .insert(full.item_id.clone(), selected.saved());
        let plain = job(vec![minimal.clone()], None);
        // Item.options is serialized by the response adapter, not the bare Item.
        let full_payload = complete.with_item_options(json!(full));
        let minimal_payload = plain.with_item_options(json!(minimal));
        agree("ItemPayload", &full_payload, &minimal_payload);
        agree("ItemPayload", &full_payload, &json!(minimal)); // older server shape
        assert_eq!(full_payload["options"]["backend"], "cloudflare");
        assert!(full_payload["options"].get("remote_processing").is_none());
        assert!(minimal_payload.get("remote_processing").is_none());
        assert_eq!(complete.snapshot()["items"][0], full_payload);
        assert_eq!(plain.snapshot()["items"][0], minimal_payload);
        agree("JobSnapshot", &complete.snapshot(), &plain.snapshot());
        agree(
            "HistoryEntry",
            &complete.history_with_items(),
            &plain.history(),
        );
        assert_eq!(
            complete.history_with_items()["items"][0],
            complete.snapshot()["items"][0]
        );
    }

    #[test]
    fn the_workbench_capability_and_option_fields_match_the_documented_contract() {
        let schema: serde_json::Value = serde_json::from_str(include_str!("openapi.json")).unwrap();
        for (interface_name, schema_name) in [
            ("Capabilities", "Capabilities"),
            ("CloudflareCapability", "CloudflareCapabilities"),
            ("RemoteProcessing", "RemoteProcessingDisclosure"),
            ("JobOptions", "JobOptions"),
            ("ProviderCard", "ProviderCard"),
        ] {
            let declared: std::collections::BTreeSet<_> = interface(interface_name)
                .into_iter()
                .map(|(name, _)| name)
                .collect();
            assert_eq!(
                declared,
                keys(&schema["components"]["schemas"][schema_name]["properties"]),
                "{interface_name}: every documented field must be declared exactly once"
            );
        }
        // Nested llm fields must not be mistaken for top-level capabilities.
        assert_eq!(interface("Capabilities").len(), 13);
        assert_eq!(interface("CloudflareCapability").len(), 6);
    }
}
