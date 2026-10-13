use super::{ApiResult, invalid, model, store::Mutation};
use crate::server::State;
use axum::{
    Json, Router,
    extract::{Path, Query, State as ExtractState, rejection::JsonRejection},
    http::StatusCode,
};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

api_routes! {
    get "/api/settings/llm" => view;
    get "/api/settings/llm/providers" => providers;
    get "/api/settings/llm/providers/{id}/credentials" => credentials;
    post "/api/settings/llm/models" => add;
    put "/api/settings/llm/models/{id}" => update_legacy;
    delete "/api/settings/llm/models/{id}" => delete_legacy;
    post "/api/settings/llm/deployments/batch" => batch;
    patch "/api/settings/llm/deployments/{id}" => update;
    delete "/api/settings/llm/deployments/{id}" => delete;
    patch "/api/settings/llm/providers/{id}" => update_provider;
    delete "/api/settings/llm/providers/{id}" => delete_provider;
    post "/api/settings/llm/config/open" => open;
    get "/api/settings/config" => config_view;
    put "/api/settings/config" => config_replace;
}

pub(in crate::server) fn routes() -> Router<Arc<State>> {
    api_routes().layer(axum::extract::DefaultBodyLimit::max(1024 * 1024))
}
fn body(value: Result<Json<Value>, JsonRejection>) -> ApiResult<Value> {
    value
        .map(|Json(v)| v)
        .map_err(|_| invalid("invalid settings JSON body"))
}
async fn view(ExtractState(state): ExtractState<Arc<State>>) -> Json<Value> {
    Json(state.settings.view())
}
#[derive(Deserialize, Default)]
struct Refresh {
    #[serde(default)]
    refresh: bool,
}
async fn providers(
    ExtractState(state): ExtractState<Arc<State>>,
    Query(query): Query<Refresh>,
) -> Json<Value> {
    // `refresh` is still validated for compatibility; the list is computed on
    // every request, so its value is not used.
    let _ = query.refresh;
    Json(state.settings.providers())
}
async fn credentials(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(id): Path<String>,
) -> ApiResult<Json<Value>> {
    state.settings.credentials(&id).map(Json)
}
async fn mutate(state: Arc<State>, command: Mutation) -> ApiResult<Json<Value>> {
    crate::task::blocking(move || state.settings.mutate(command))
        .await
        .map_err(|_| super::failure())?
        .map(Json)
}
async fn add(
    ExtractState(state): ExtractState<Arc<State>>,
    value: Result<Json<Value>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    mutate(state, Mutation::Add(body(value)?)).await
}
async fn batch(
    ExtractState(state): ExtractState<Arc<State>>,
    value: Result<Json<Value>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    mutate(state, Mutation::Batch(body(value)?)).await
}
async fn update_legacy(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(key): Path<String>,
    value: Result<Json<Value>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    mutate(
        state,
        Mutation::Update {
            key,
            body: body(value)?,
            legacy: true,
        },
    )
    .await
}
async fn update(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(key): Path<String>,
    value: Result<Json<Value>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    mutate(
        state,
        Mutation::Update {
            key,
            body: body(value)?,
            legacy: false,
        },
    )
    .await
}
async fn delete_legacy(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(key): Path<String>,
) -> ApiResult<Json<Value>> {
    mutate(
        state,
        Mutation::Delete {
            key,
            revision: None,
            legacy: true,
        },
    )
    .await
}
#[derive(Deserialize)]
struct Revision {
    expected_revision: String,
}
async fn delete(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(key): Path<String>,
    query: Result<Query<Revision>, axum::extract::rejection::QueryRejection>,
) -> ApiResult<Json<Value>> {
    let revision = query
        .map_err(|_| invalid("expected_revision is required"))?
        .0
        .expected_revision;
    mutate(
        state,
        Mutation::Delete {
            key,
            revision: Some(revision),
            legacy: false,
        },
    )
    .await
}
async fn update_provider(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(key): Path<String>,
    value: Result<Json<Value>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    let body = body(value)?;
    let revision = model::string(&body, "expected_revision", true, true)?.unwrap();
    mutate(
        state,
        Mutation::Provider {
            key,
            body: Some(body),
            revision,
        },
    )
    .await
}
async fn delete_provider(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(key): Path<String>,
    query: Result<Query<Revision>, axum::extract::rejection::QueryRejection>,
) -> ApiResult<Json<Value>> {
    let revision = query
        .map_err(|_| invalid("expected_revision is required"))?
        .0
        .expected_revision;
    mutate(
        state,
        Mutation::Provider {
            key,
            body: None,
            revision,
        },
    )
    .await
}
async fn open(ExtractState(state): ExtractState<Arc<State>>) -> ApiResult<StatusCode> {
    crate::task::blocking(move || {
        let path = state.settings.config_path()?;
        crate::server::open_config(&path).map_err(|_| {
            super::ApiError::new(
                500,
                "settings_io_failed",
                "Could not open the configuration file",
            )
        })?;
        Ok(StatusCode::NO_CONTENT)
    })
    .await
    .map_err(|_| super::failure())?
}

/// `text` at most.
const MAX_CONFIG_TEXT: usize = 1024 * 1024;
/// A `PUT /api/settings/config` body at most.
const MAX_CONFIG_BODY: usize = 3 * 512 * 1024;

fn too_large() -> super::ApiError {
    super::ApiError::new(
        413,
        "config_too_large",
        "the configuration text exceeds 1 MiB",
    )
}

/// `GET /api/settings/config`: the configuration file, secrets masked.
async fn config_view(ExtractState(state): ExtractState<Arc<State>>) -> ApiResult<Json<Value>> {
    crate::task::blocking(move || state.settings.config_view())
        .await
        .map_err(|_| super::failure())?
        .map(Json)
}

/// `PUT /api/settings/config` with `{expected_revision, text}`.
async fn config_replace(
    ExtractState(state): ExtractState<Arc<State>>,
    request: axum::extract::Request,
) -> ApiResult<Json<Value>> {
    let json = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"));
    let bytes = axum::body::to_bytes(request.into_body(), MAX_CONFIG_BODY)
        .await
        .map_err(|_| too_large())?;
    let shape = || invalid("send a JSON body {\"expected_revision\": string, \"text\": string}");
    if !json {
        return Err(shape());
    }
    let body: Value = serde_json::from_slice(&bytes).map_err(|_| shape())?;
    let fields = body
        .as_object()
        .filter(|fields| {
            fields.len() == 2
                && fields
                    .keys()
                    .all(|key| key == "expected_revision" || key == "text")
        })
        .ok_or_else(shape)?;
    let (Some(expected), Some(text)) = (
        fields["expected_revision"].as_str(),
        fields["text"].as_str(),
    ) else {
        return Err(shape());
    };
    if text.len() > MAX_CONFIG_TEXT {
        return Err(too_large());
    }
    let (expected, text) = (expected.to_owned(), text.to_owned());
    crate::task::blocking(move || state.settings.replace_config(&expected, &text))
        .await
        .map_err(|_| super::failure())?
        .map(Json)
}
