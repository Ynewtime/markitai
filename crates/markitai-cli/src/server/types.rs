use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub(super) const MAX_ITEMS: usize = 1000;
pub(super) const MAX_UPLOAD: usize = 100 * 1024 * 1024;
pub(super) const MAX_REQUEST: usize = 5 * 1024 * 1024 * 1024 + 64 * 1024 * 1024;

pub(super) fn now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// The instant a stored RFC 3339 timestamp names. CLI histories record local
/// time with its offset and the server records UTC, so their text does not sort.
pub(super) fn instant(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|time| time.with_timezone(&Utc))
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct JobOptions {
    pub preset: Option<String>,
    pub llm: Option<bool>,
    pub ocr: Option<bool>,
    pub profile: Option<String>,
    pub alt: Option<bool>,
    pub desc: Option<bool>,
    pub screenshot: Option<bool>,
    pub screenshot_only: Option<bool>,
    pub pure: Option<bool>,
    pub no_cache: Option<bool>,
    pub no_compress: Option<bool>,
    pub strategy: Option<String>,
    pub backend: Option<String>,
    pub remote_processing: Option<String>,
}

/// Option names the service accepts, by value type. Anything else is rejected
/// by name, so a caller never reads a JSON parser's position-in-text message.
const BOOLEAN_OPTIONS: [&str; 9] = [
    "llm",
    "ocr",
    "alt",
    "desc",
    "screenshot",
    "screenshot_only",
    "pure",
    "no_cache",
    "no_compress",
];
const TEXT_OPTIONS: [&str; 5] = [
    "preset",
    "profile",
    "strategy",
    "backend",
    "remote_processing",
];
const PROFILES: [&str; 3] = ["rag", "obsidian", "okf"];
const STRATEGIES: [&str; 6] = [
    "auto",
    "static",
    "playwright",
    "defuddle",
    "jina",
    "cloudflare",
];

fn invalid_options(detail: impl Into<String>) -> ApiError {
    ApiError::new(422, "invalid_options", detail)
}

/// A caller-supplied name, shortened so an oversized key cannot bloat a response.
fn quoted(name: &str) -> String {
    let mut shown: String = name.chars().take(48).collect();
    if shown.len() < name.len() {
        shown.push('…');
    }
    format!("'{shown}'")
}

/// A preset from the configuration, or one of the three built-in definitions.
fn preset_definition(base: &Value, name: &str) -> Option<Value> {
    let name = name.to_ascii_lowercase();
    base["presets"]
        .get(&name)
        .cloned()
        .or_else(|| match name.as_str() {
            "minimal" => {
                Some(json!({"llm":false,"ocr":false,"alt":false,"desc":false,"screenshot":false}))
            }
            "standard" => {
                Some(json!({"llm":true,"ocr":false,"alt":true,"desc":true,"screenshot":false}))
            }
            "rich" => {
                Some(json!({"llm":true,"ocr":false,"alt":true,"desc":true,"screenshot":true}))
            }
            _ => None,
        })
}

impl JobOptions {
    /// Read the `options` form field. Every failure names the offending option
    /// instead of passing a JSON parser's message through to the caller.
    pub fn parse(bytes: &[u8]) -> ApiResult<Self> {
        let value = serde_json::from_slice(bytes)
            .map_err(|_| invalid_options("options must be a JSON object"))?;
        Self::from_value(value)
    }

    pub fn from_value(value: Value) -> ApiResult<Self> {
        let map = match &value {
            Value::Null => return Ok(Self::default()),
            Value::Object(map) => map,
            _ => return Err(invalid_options("options must be a JSON object")),
        };
        for (key, entry) in map {
            if BOOLEAN_OPTIONS.contains(&key.as_str()) {
                if !entry.is_null() && !entry.is_boolean() {
                    return Err(invalid_options(format!(
                        "option {} must be true or false",
                        quoted(key)
                    )));
                }
            } else if TEXT_OPTIONS.contains(&key.as_str()) {
                if !entry.is_null() && !entry.is_string() {
                    return Err(invalid_options(format!(
                        "option {} must be text",
                        quoted(key)
                    )));
                }
            } else {
                return Err(invalid_options(format!(
                    "unknown option {}; supported options: {}, {}",
                    quoted(key),
                    TEXT_OPTIONS.join(", "),
                    BOOLEAN_OPTIONS.join(", ")
                )));
            }
        }
        for (key, allowed) in [
            ("profile", &PROFILES[..]),
            ("strategy", &STRATEGIES[..]),
            ("backend", &["native", "cloudflare"][..]),
            ("remote_processing", &["cloudflare"][..]),
        ] {
            if let Some(chosen) = map.get(key).and_then(Value::as_str)
                && !allowed.contains(&chosen)
            {
                return Err(invalid_options(format!(
                    "option '{key}' must be one of: {}",
                    allowed.join(", ")
                )));
            }
        }
        serde_json::from_value(value).map_err(|_| invalid_options("options could not be read"))
    }

    /// Whether the request itself asks for model processing, as opposed to a
    /// server configuration that happens to enable it.
    pub fn requests_llm(&self, base: &Value) -> bool {
        match (self.llm, &self.preset) {
            (Some(explicit), _) => explicit,
            (None, Some(name)) => {
                preset_definition(base, name).is_some_and(|preset| preset["llm"] == true)
            }
            (None, None) => false,
        }
    }

    /// Refuse a request for model processing when no model can serve it, before
    /// any job exists: the alternative is one identical failure per item.
    pub fn require_model(&self, base: &Value, cfg: &Value) -> ApiResult<()> {
        if self.requests_llm(base) && !markitai_core::llm_capabilities(cfg).routable {
            return Err(ApiError::new(
                422,
                "llm_unavailable",
                "this request needs a model, but none is available; add a connection, or set MODEL and a provider API key",
            ));
        }
        Ok(())
    }

    fn requests_cloudflare(&self) -> bool {
        self.strategy.as_deref() == Some("cloudflare")
            || self.backend.as_deref() == Some("cloudflare")
    }

    /// History remembers selections, never permission for a later request.
    pub fn saved(&self) -> Value {
        let mut saved = self.clone();
        saved.remote_processing = None;
        serde_json::to_value(saved).expect("job options serialize")
    }

    pub fn remote_disclosure(&self) -> Option<Value> {
        (self.requests_cloudflare() && self.remote_processing.as_deref() == Some("cloudflare"))
            .then(|| {
                json!({"provider":"cloudflare","requested":true,"execution":"unknown",
                "external_charges":"not_included",
                "notice":"Cloudflare requested; external charges not included"})
            })
    }

    pub fn config_for_request(&self, base: &Value, trusted: bool) -> ApiResult<Value> {
        self.config_for_request_with(base, trusted, || {
            markitai_core::cloudflare_capabilities(base)
        })
    }

    fn config_for_request_with(
        &self,
        base: &Value,
        trusted: bool,
        capability: impl FnOnce() -> Value,
    ) -> ApiResult<Value> {
        let mut cfg = self.config(base)?;
        let selected = self.requests_cloudflare();
        if !selected {
            if self.remote_processing.is_some() {
                return Err(invalid_options(
                    "remote_processing requires a Cloudflare strategy or backend in this request",
                ));
            }
            return Ok(cfg);
        }
        if !trusted {
            return Err(ApiError::new(
                403,
                "remote_processing_forbidden",
                "Cloudflare processing requires a trusted authenticated request",
            ));
        }
        if self.remote_processing.as_deref() != Some("cloudflare") {
            return Err(ApiError::new(
                422,
                "remote_processing_confirmation_required",
                "Confirm Cloudflare processing for this request",
            ));
        }
        // This consent cannot authorize another provider, including a strategy
        // inherited from the server's configuration.
        if matches!(cfg["fetch"]["strategy"].as_str(), Some("defuddle" | "jina")) {
            return Err(invalid_options(
                "Cloudflare confirmation cannot authorize another remote strategy; choose auto, static, playwright or cloudflare",
            ));
        }
        let capability = capability();
        if capability["available"] != true {
            let (reason, detail) = match capability["reason"].as_str() {
                Some("disabled_by_policy") => (
                    "remote_processing_disabled",
                    "Cloudflare processing is disabled by server policy",
                ),
                _ => (
                    "cloudflare_unavailable",
                    "Cloudflare is not configured and locally ready on this server",
                ),
            };
            return Err(ApiError::new(422, reason, detail));
        }
        // Never escalate to always: core's explicit strategy answers ask for
        // Cloudflare only; auto still has no host consent and remains local.
        cfg["fetch"]["remote_consent"] = json!("ask");
        Ok(cfg)
    }

    pub fn config(&self, base: &Value) -> ApiResult<Value> {
        let mut cfg = base.clone();
        // Cloudflare defaults are not a user's confirmation for this request.
        if cfg["fetch"]["strategy"] == "cloudflare" {
            cfg["fetch"]["strategy"] = json!("auto");
        }
        cfg["fetch"]["cloudflare"]["convert_enabled"] = json!(false);
        if let Some(name) = &self.preset {
            let name = name.to_ascii_lowercase();
            let preset = preset_definition(base, &name).ok_or_else(|| {
                ApiError::new(422, "unknown_preset", format!("unknown preset '{name}'"))
            })?;
            for (key, path) in [
                ("llm", "/llm/enabled"),
                ("ocr", "/ocr/enabled"),
                ("alt", "/image/alt_enabled"),
                ("desc", "/image/desc_enabled"),
                ("screenshot", "/screenshot/enabled"),
            ] {
                if let Some(value) = preset.get(key) {
                    *cfg.pointer_mut(path).unwrap() = value.clone();
                }
            }
        }
        for (path, value) in [
            ("/llm/enabled", self.llm),
            ("/ocr/enabled", self.ocr),
            ("/image/alt_enabled", self.alt),
            ("/image/desc_enabled", self.desc),
            ("/screenshot/enabled", self.screenshot),
            ("/screenshot/screenshot_only", self.screenshot_only),
            ("/llm/pure", self.pure),
            ("/cache/no_cache", self.no_cache),
        ] {
            if let Some(value) = value {
                *cfg.pointer_mut(path).unwrap() = json!(value);
            }
        }
        if self.screenshot_only == Some(true) {
            cfg["screenshot"]["enabled"] = json!(true);
        }
        if let Some(value) = self.no_compress {
            cfg["image"]["compress"] = json!(!value);
        }
        if let Some(value) = &self.profile {
            cfg["output"]["profile"] = json!(value);
        }
        if let Some(value) = &self.strategy {
            cfg["fetch"]["strategy"] = json!(value);
        }
        if let Some(value) = &self.backend {
            if !["native", "cloudflare"].contains(&value.as_str()) {
                return Err(ApiError::new(422, "invalid_options", "invalid backend"));
            }
            cfg["fetch"]["cloudflare"]["convert_enabled"] = json!(value == "cloudflare");
        }
        if cfg["llm"]["enabled"] == true {
            cfg["llm"]["keep_base"] = json!(true);
        }
        // A service never waits for terminal input or writes user-selected output paths.
        if cfg["fetch"]["remote_consent"] == "ask" {
            cfg["fetch"]["remote_consent"] = json!("never");
        }
        cfg["history"]["record"] = json!(false);
        if let Some(output) = cfg["output"].as_object_mut() {
            output.remove("filename");
            output.remove("reserved_stem");
        }
        cfg["output"]["allow_symlinks"] = json!(false);
        markitai_core::config::normalize(&cfg)
            .map_err(|error| ApiError::new(422, "invalid_options", error.to_string()))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum RerunOperation {
    #[default]
    Retry,
    Enhance,
}

/// The latest unsuccessful rerun whose previous successful result was retained.
/// This records an operation outcome independently of observed model usage.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RerunFailure {
    pub operation: RerunOperation,
    pub error_code: String,
    pub error: String,
    pub failed_at: String,
}
impl RerunFailure {
    pub fn new(operation: RerunOperation, code: &str, error: String) -> Self {
        Self {
            operation,
            error_code: code.into(),
            // Service details already follow the conversion error contract;
            // keep the persisted, rendered notice bounded as well.
            error: error.chars().take(1024).collect(),
            failed_at: now(),
        }
    }
    pub fn from_value(value: &Value) -> Result<Self, ()> {
        let failure: Self = serde_json::from_value(value.clone()).map_err(|_| ())?;
        if failure.error_code.is_empty()
            || failure.error_code.len() > 64
            || !failure
                .error_code
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b == b'_')
            || failure.error.is_empty()
            || failure.error.len() > 4096
            || failure.failed_at.len() > 64
            || chrono::DateTime::parse_from_rfc3339(&failure.failed_at).is_err()
        {
            return Err(());
        }
        Ok(failure)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Item {
    pub item_id: String,
    pub name: String,
    pub kind: String,
    pub status: String,
    pub error: Option<String>,
    /// Stable category of a failed item's `error`: the core's conversion error
    /// code or a service cause. Older and CLI-recorded histories have none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    pub output: Option<String>,
    pub output_name: Option<String>,
    pub duration_ms: Option<u64>,
    pub finished_at: Option<String>,
    pub cost_usd: Option<f64>,
    /// At least one accepted attempt for this item requested Cloudflare. This
    /// does not establish execution, request counts, or actual external fees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_processing: Option<Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::pricing::deserialize_optional"
    )]
    pub pricing: Option<crate::pricing::Pricing>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<crate::diagnostics::AttemptDiagnostics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerun_failure: Option<RerunFailure>,
    #[serde(default)]
    pub llm_enhanced: bool,
    #[serde(default = "convert_operation")]
    pub operation: String,
    #[serde(default)]
    pub skipped: bool,
    #[serde(default)]
    pub skip_reason: Option<String>,
    #[serde(default = "yes")]
    pub retryable: bool,
    #[serde(default)]
    pub warnings: Vec<String>,
}
fn convert_operation() -> String {
    "convert".into()
}
fn yes() -> bool {
    true
}
impl Item {
    pub fn new(index: usize, name: String, kind: &str, output_name: Option<String>) -> Self {
        Self {
            item_id: format!("i{index}"),
            name,
            kind: kind.into(),
            status: "queued".into(),
            error: None,
            error_code: None,
            output: None,
            output_name,
            duration_ms: None,
            finished_at: None,
            cost_usd: None,
            remote_processing: None,
            pricing: None,
            diagnostics: None,
            rerun_failure: None,
            llm_enhanced: false,
            operation: "convert".into(),
            skipped: false,
            skip_reason: None,
            retryable: true,
            warnings: Vec::new(),
        }
    }
    pub fn created(&self) -> Value {
        json!({"item_id":self.item_id,"name":self.name,"kind":self.kind})
    }
}

pub(super) type ApiResult<T> = Result<T, ApiError>;
#[derive(Debug)]
pub(super) struct ApiError {
    pub status: StatusCode,
    /// Stable machine-readable cause, finer than the status-derived `code`.
    /// Clients localize by it; `detail` keeps the service's English wording.
    pub reason: &'static str,
    pub detail: String,
    structured_detail: Option<Value>,
}
impl ApiError {
    pub fn new(status: u16, reason: &'static str, detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            reason,
            detail: detail.into(),
            structured_detail: None,
        }
    }
    pub fn structured(status: u16, reason: &'static str, detail: Value) -> Self {
        Self {
            status: StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            reason,
            detail: "structured API error".into(),
            structured_detail: Some(detail),
        }
    }
    pub fn internal(error: impl std::fmt::Display) -> Self {
        eprintln!("Serve: {error}");
        Self::new(500, "internal_error", "internal server error")
    }
    /// A request body the multipart reader rejected; an exceeded body limit stays
    /// distinct. The reader's own wording (boundary and parser internals) is never
    /// passed on: `detail` is one fixed sentence per reason.
    pub fn multipart(status: StatusCode) -> Self {
        if status == StatusCode::PAYLOAD_TOO_LARGE {
            Self::new(
                status.as_u16(),
                "request_too_large",
                "request exceeds the size limit",
            )
        } else {
            Self::new(
                status.as_u16(),
                "invalid_multipart",
                "send the files, urls and options as multipart/form-data, or urls and options as application/x-www-form-urlencoded",
            )
        }
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = match self.status.as_u16() {
            400 => "bad_request",
            401 => "unauthorized",
            403 => "forbidden",
            404 => "not_found",
            405 => "method_not_allowed",
            409 => "conflict",
            413 => "payload_too_large",
            422 => "invalid_request",
            429 => "rate_limited",
            503 => "unavailable",
            _ => "server_error",
        };
        let detail = self.structured_detail.unwrap_or(Value::String(self.detail));
        let mut response = (
            self.status,
            Json(json!({"detail":detail,"code":code,"reason":self.reason})),
        )
            .into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response
                .headers_mut()
                .insert("www-authenticate", "Bearer".parse().unwrap());
        }
        response
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    async fn body(error: ApiError) -> (u16, Value) {
        let response = error.into_response();
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn errors_add_a_stable_reason_beside_the_existing_detail_and_status_code() {
        let (status, value) = body(ApiError::new(
            413,
            "file_too_large",
            "file exceeds upload limit",
        ))
        .await;
        assert_eq!(status, 413);
        assert_eq!(
            value,
            json!({"detail":"file exceeds upload limit","code":"payload_too_large","reason":"file_too_large"})
        );
        let (status, value) = body(ApiError::structured(
            409,
            "stale_revision",
            json!({"code":"stale_revision","current_revision":"r2"}),
        ))
        .await;
        assert_eq!(status, 409);
        assert_eq!(value["detail"]["current_revision"], "r2");
        assert_eq!(value["code"], "conflict");
        assert_eq!(value["reason"], "stale_revision");
        let (_, value) = body(ApiError::internal("disk detail stays in the log")).await;
        assert_eq!(
            value,
            json!({"detail":"internal server error","code":"server_error","reason":"internal_error"})
        );
    }

    #[test]
    fn a_multipart_body_over_the_limit_is_distinct_from_a_malformed_one() {
        assert_eq!(
            ApiError::multipart(StatusCode::PAYLOAD_TOO_LARGE).reason,
            "request_too_large"
        );
        let malformed = ApiError::multipart(StatusCode::BAD_REQUEST);
        assert_eq!(malformed.reason, "invalid_multipart");
        assert_eq!(malformed.status, StatusCode::BAD_REQUEST);
        // The reader's boundary and parser wording never reaches the caller.
        assert!(!malformed.detail.contains("boundary"));
    }

    #[test]
    fn option_errors_name_the_option_and_never_quote_a_parser() {
        let detail = |text: &str| JobOptions::parse(text.as_bytes()).unwrap_err().detail;
        assert_eq!(
            detail(r#"{"bogus":true}"#),
            "unknown option 'bogus'; supported options: preset, profile, strategy, backend, remote_processing, llm, ocr, alt, desc, screenshot, screenshot_only, pure, no_cache, no_compress"
        );
        assert_eq!(
            detail(r#"{"llm":"yes"}"#),
            "option 'llm' must be true or false"
        );
        assert_eq!(detail(r#"{"preset":3}"#), "option 'preset' must be text");
        assert_eq!(
            detail(r#"{"profile":"markdown"}"#),
            "option 'profile' must be one of: rag, obsidian, okf"
        );
        assert_eq!(
            detail(r#"{"strategy":"fast"}"#),
            "option 'strategy' must be one of: auto, static, playwright, defuddle, jina, cloudflare"
        );
        for broken in ["{not json", "[1]", "\"text\"", ""] {
            assert_eq!(detail(broken), "options must be a JSON object", "{broken}");
        }
        for text in [r#"{"bogus":1}"#, r#"{"llm":1}"#, "{not json"] {
            let error = JobOptions::parse(text.as_bytes()).unwrap_err();
            assert_eq!(
                (error.status.as_u16(), error.reason),
                (422, "invalid_options")
            );
            assert!(
                !error.detail.contains("line 1 column") && !error.detail.contains("expected"),
                "{}",
                error.detail
            );
        }
        let long = format!(r#"{{"{}":1}}"#, "k".repeat(500));
        assert!(detail(&long).len() < 300);
        let parsed = JobOptions::parse(br#"{"llm":true,"profile":"rag","ocr":null}"#).unwrap();
        assert_eq!(
            (parsed.llm, parsed.profile.as_deref(), parsed.ocr),
            (Some(true), Some("rag"), None)
        );
        assert!(JobOptions::parse(b"null").unwrap().preset.is_none());
    }

    #[test]
    fn only_a_request_for_a_model_needs_a_routable_one() {
        let base = markitai_core::config::normalize(
            &json!({"llm":{"enabled":false},"presets":{"offline":{"llm":false,"ocr":true,"alt":false,"desc":false,"screenshot":false}}}),
        )
        .unwrap();
        let options = |text: &str| JobOptions::parse(text.as_bytes()).unwrap();
        for (text, wanted) in [
            ("{}", false),
            (r#"{"llm":true}"#, true),
            (r#"{"llm":false}"#, false),
            (r#"{"preset":"standard"}"#, true),
            (r#"{"preset":"RICH"}"#, true),
            (r#"{"preset":"minimal"}"#, false),
            (r#"{"preset":"offline"}"#, false),
            (r#"{"preset":"standard","llm":false}"#, false),
            (r#"{"preset":"minimal","llm":true}"#, true),
            (r#"{"alt":true,"desc":true}"#, false),
        ] {
            assert_eq!(options(text).requests_llm(&base), wanted, "{text}");
        }
        // A model-free request is accepted whatever the model state is.
        let cfg = options("{}").config(&base).unwrap();
        assert!(options("{}").require_model(&base, &cfg).is_ok());
    }

    #[test]
    fn item_error_codes_are_additive_and_older_items_still_load() {
        let mut item = Item::new(1, "a.xyz".into(), "file", None);
        let fresh = serde_json::to_value(&item).unwrap();
        assert!(fresh.get("error_code").is_none(), "{fresh}");
        let restored: Item = serde_json::from_value(fresh).unwrap();
        assert!(restored.error_code.is_none());
        item.status = "error".into();
        item.error = Some("Unsupported file format: '.xyz'.".into());
        item.error_code = Some("unsupported".into());
        let failed = serde_json::to_value(&item).unwrap();
        assert_eq!(failed["error_code"], "unsupported");
        assert_eq!(failed["error"], "Unsupported file format: '.xyz'.");
    }
}

#[cfg(test)]
mod pricing_tests {
    use super::*;
    #[test]
    fn historical_items_keep_their_output_when_optional_pricing_is_missing_or_invalid() {
        let mut item = Item::new(1, "legacy.md".into(), "file", Some("legacy.md".into()));
        item.status = "done".into();
        item.output = Some("legacy.md".into());
        item.cost_usd = Some(0.75);
        let original = serde_json::to_value(item).unwrap();
        assert!(original.get("pricing").is_none());
        for bad in [
            json!(null),
            json!({"cost_status":"complete"}),
            json!({"priced_requests":0,"unpriced_requests":2,"cost_status":"complete","pricing_snapshots":[]}),
        ] {
            let mut value = original.clone();
            value["pricing"] = bad;
            let restored: Item = serde_json::from_value(value).unwrap();
            assert_eq!(restored.output.as_deref(), Some("legacy.md"));
            assert_eq!(restored.cost_usd, Some(0.75));
            assert!(restored.pricing.is_none());
        }
        let mut value = original;
        value["pricing"] = json!({"priced_requests":1,"unpriced_requests":2,"cost_status":"partial","pricing_snapshots":["catalog-v1"]});
        let restored: Item = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(restored).unwrap(), value);
    }
}

#[cfg(test)]
mod cloudflare_request_tests {
    use super::*;
    fn base(policy: &str) -> Value {
        markitai_core::config::normalize(&json!({"fetch":{"remote_consent":policy,
            "strategy":"cloudflare","cloudflare":{"convert_enabled":true}}}))
        .unwrap()
    }
    fn ready() -> Value {
        json!({"available":true,"reason":null})
    }
    fn options(value: Value) -> JobOptions {
        JobOptions::from_value(value).unwrap()
    }

    #[test]
    fn defaults_do_not_send_files_or_urls_to_cloudflare() {
        for policy in ["ask", "always", "never"] {
            let cfg = JobOptions::default()
                .config_for_request_with(&base(policy), false, || {
                    panic!("must not read credentials")
                })
                .unwrap();
            assert_eq!(cfg["fetch"]["strategy"], "auto");
            assert_eq!(cfg["fetch"]["cloudflare"]["convert_enabled"], false);
        }
    }
    #[test]
    fn cloudflare_requires_current_confirmation_and_trust_before_readiness() {
        for selection in [
            json!({"strategy":"cloudflare"}),
            json!({"backend":"cloudflare"}),
        ] {
            let opt = options(selection.clone());
            let denied = opt
                .config_for_request_with(&base("ask"), false, || panic!("no credential lookup"))
                .unwrap_err();
            assert_eq!(
                (denied.status.as_u16(), denied.reason),
                (403, "remote_processing_forbidden")
            );
            assert_eq!(
                opt.config_for_request_with(&base("ask"), true, || panic!("no credential lookup"))
                    .unwrap_err()
                    .reason,
                "remote_processing_confirmation_required"
            );
            let mut confirmed = selection;
            confirmed["remote_processing"] = json!("cloudflare");
            let confirmed = options(confirmed);
            assert_eq!(
                confirmed
                    .config_for_request_with(&base("always"), true, ready)
                    .unwrap()["fetch"]["remote_consent"],
                "ask"
            );
            assert!(confirmed.saved()["remote_processing"].is_null());
            let inherited = options(confirmed.saved());
            assert_eq!(
                inherited
                    .config_for_request_with(&base("ask"), true, ready)
                    .unwrap_err()
                    .reason,
                "remote_processing_confirmation_required"
            );
        }
    }
    #[test]
    fn policy_and_readiness_refusals_cannot_be_overridden() {
        let opt = options(json!({"backend":"cloudflare","remote_processing":"cloudflare"}));
        for reason in [
            "disabled_by_policy",
            "not_configured",
            "invalid_configuration",
        ] {
            let result = opt
                .config_for_request_with(
                    &base("ask"),
                    true,
                    || json!({"available":false,"reason":reason}),
                )
                .unwrap_err();
            assert_eq!(result.status.as_u16(), 422);
            assert_eq!(
                result.reason,
                if reason == "disabled_by_policy" {
                    "remote_processing_disabled"
                } else {
                    "cloudflare_unavailable"
                }
            );
        }
    }
    #[test]
    fn confirmation_is_strict_and_cannot_authorize_other_services() {
        for value in [
            json!("always"),
            json!("Cloudflare"),
            json!(true),
            json!(1),
            json!([]),
        ] {
            assert!(JobOptions::from_value(json!({"remote_processing":value})).is_err());
        }
        assert!(
            options(json!({"remote_processing":"cloudflare"}))
                .config_for_request_with(&base("ask"), true, ready)
                .is_err()
        );
        for strategy in ["defuddle", "jina"] {
            let opt = options(
                json!({"backend":"cloudflare","strategy":strategy,"remote_processing":"cloudflare"}),
            );
            assert!(
                opt.config_for_request_with(&base("ask"), true, ready)
                    .is_err()
            );
        }
        for forbidden in ["api_key", "api_token", "account_id", "endpoint", "api_base"] {
            let mut value = json!({"backend":"cloudflare","remote_processing":"cloudflare"});
            value[forbidden] = json!("synthetic");
            assert!(JobOptions::from_value(value).is_err());
        }
    }
    #[test]
    fn request_disclosure_is_separate_from_the_llm_ledger_and_survives_history() {
        let mut item = Item::new(1, "local.txt".into(), "file", None);
        assert!(
            serde_json::to_value(&item)
                .unwrap()
                .get("remote_processing")
                .is_none()
        );
        item.remote_processing =
            options(json!({"backend":"cloudflare","remote_processing":"cloudflare"}))
                .remote_disclosure();
        item.cost_usd = Some(0.125);
        let encoded = serde_json::to_value(&item).unwrap();
        assert_eq!(encoded["remote_processing"]["execution"], "unknown");
        assert_eq!(
            encoded["remote_processing"]["external_charges"],
            "not_included"
        );
        assert_eq!(encoded["cost_usd"], 0.125);
        let restored: Item = serde_json::from_value(encoded).unwrap();
        assert_eq!(restored.remote_processing, item.remote_processing);
        assert_eq!(restored.cost_usd, item.cost_usd);
    }
}
