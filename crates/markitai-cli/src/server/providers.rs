//! Trusted provider actions are transient and never persist request credentials.
//! `security::guard` admits only trusted connections to these routes.
use super::{
    State,
    types::{ApiError, ApiResult},
};
use axum::{
    Json, Router,
    extract::{Request, State as ExtractState},
    response::{IntoResponse, Response},
};
use serde_json::Value;
use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};
use tokio::sync::Semaphore;

api_routes! {
    get "/api/settings/llm/detected" => detected;
    post "/api/settings/llm/model-discovery" => discover;
    post "/api/settings/llm/test" => probe;
}

pub(super) fn routes() -> Router<Arc<State>> {
    api_routes()
}
fn response(value: ApiResult<Value>) -> Response {
    let mut response = match value {
        Ok(value) => Json(value).into_response(),
        Err(error) => error.into_response(),
    };
    response
        .headers_mut()
        .insert("cache-control", "no-store".parse().unwrap());
    response
}
async fn body(request: Request) -> ApiResult<Value> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if !content_type
        .split(';')
        .next()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
    {
        return Err(ApiError::new(
            422,
            "json_required",
            "Provider request must use application/json",
        ));
    }
    let bytes = axum::body::to_bytes(request.into_body(), 64 * 1024)
        .await
        .map_err(|_| ApiError::new(413, "request_too_large", "Provider request exceeds 64 KiB"))?;
    serde_json::from_slice(&bytes)
        .map_err(|_| ApiError::new(422, "invalid_json", "Provider request is not valid JSON"))
}
async fn detected() -> Response {
    response(Ok(serde_json::json!(
        markitai_core::provider_management::detected()
    )))
}
async fn discover(ExtractState(state): ExtractState<Arc<State>>, request: Request) -> Response {
    let request = match body(request)
        .await
        .and_then(|value| state.settings.resolve_discovery(&value))
    {
        Ok(value) => value,
        Err(error) => return response(Err(error)),
    };
    response(network(request, false).await)
}
async fn probe(ExtractState(state): ExtractState<Arc<State>>, request: Request) -> Response {
    let request = match body(request)
        .await
        .and_then(|value| state.settings.resolve_probe(&value))
    {
        Ok(value) => value,
        Err(error) => return response(Err(error)),
    };
    response(network(request, true).await)
}
async fn network(mut request: Value, is_probe: bool) -> ApiResult<Value> {
    static SLOTS: OnceLock<Arc<Semaphore>> = OnceLock::new();
    let permit = SLOTS
        .get_or_init(|| Arc::new(Semaphore::new(8)))
        .clone()
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::new(
                429,
                "provider_busy",
                "Too many provider requests are active",
            )
        })?;
    // This marker is produced by the settings resolver, never accepted from JSON clients.
    let use_environment = request
        .as_object_mut()
        .and_then(|request| request.remove("use_environment_credentials"))
        .and_then(|value| value.as_bool())
        == Some(true);
    let task = crate::task::blocking(move || {
        let _permit = permit;
        if is_probe {
            if use_environment {
                markitai_core::provider_management::probe(&request)
            } else {
                markitai_core::provider_management::probe_explicit(&request)
            }
        } else if use_environment {
            markitai_core::provider_management::discover(&request)
        } else {
            markitai_core::provider_management::discover_explicit(&request)
        }
    });
    match tokio::time::timeout(Duration::from_secs(if is_probe { 30 } else { 20 }), task).await {
        Ok(Ok(Ok(value))) => Ok(value),
        Ok(Ok(Err(markitai_core::Error::InvalidInput(_) | markitai_core::Error::Config(_)))) => {
            Err(ApiError::new(
                422,
                "invalid_provider_request",
                "Provider request or environment reference is invalid",
            ))
        }
        Ok(Ok(Err(_))) | Ok(Err(_)) => Err(ApiError::new(
            503,
            "provider_unavailable",
            "Provider operation could not complete",
        )),
        Err(_) => {
            if is_probe {
                Ok(serde_json::json!({"ok":false,"detail":"Model connection test timed out"}))
            } else {
                Err(ApiError::new(
                    503,
                    "discovery_timeout",
                    "Model discovery timed out",
                ))
            }
        }
    }
}
