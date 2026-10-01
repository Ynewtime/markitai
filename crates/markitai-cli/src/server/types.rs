use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub(super) const MAX_ITEMS: usize = 1000;
pub(super) const MAX_UPLOAD: usize = 100 * 1024 * 1024;
pub(super) const MAX_REQUEST: usize = 5 * 1024 * 1024 * 1024 + 64 * 1024 * 1024;

pub(super) fn now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
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
}

impl JobOptions {
    pub fn config(&self, base: &Value) -> ApiResult<Value> {
        let mut cfg = base.clone();
        if let Some(name) = &self.preset {
            let name = name.to_ascii_lowercase();
            let preset = cfg["presets"].get(&name).cloned().or_else(|| match name.as_str() {
                "minimal" => Some(json!({"llm":false,"ocr":false,"alt":false,"desc":false,"screenshot":false})),
                "standard" => Some(json!({"llm":true,"ocr":false,"alt":true,"desc":true,"screenshot":false})),
                "rich" => Some(json!({"llm":true,"ocr":false,"alt":true,"desc":true,"screenshot":true})),
                _ => None,
            }).ok_or_else(|| ApiError::new(422, "unknown_preset", format!("unknown preset '{name}'")))?;
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
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::pricing::deserialize_optional"
    )]
    pub pricing: Option<crate::pricing::Pricing>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<crate::diagnostics::AttemptDiagnostics>,
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
            pricing: None,
            diagnostics: None,
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
    /// A request body the multipart reader rejected; an exceeded body limit stays distinct.
    pub fn multipart(status: StatusCode, detail: impl Into<String>) -> Self {
        let reason = if status == StatusCode::PAYLOAD_TOO_LARGE {
            "request_too_large"
        } else {
            "invalid_multipart"
        };
        Self::new(status.as_u16(), reason, detail)
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
            ApiError::multipart(StatusCode::PAYLOAD_TOO_LARGE, "length limit").reason,
            "request_too_large"
        );
        let malformed = ApiError::multipart(StatusCode::BAD_REQUEST, "invalid multipart body");
        assert_eq!(malformed.reason, "invalid_multipart");
        assert_eq!(malformed.status, StatusCode::BAD_REQUEST);
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
