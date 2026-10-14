//! `/api/me/models*`: a user token's own models, with the settings routes'
//! suffixes and shapes. Only a user token may use them (anyone else:
//! `user_token_required`); `security::guard` marks every answer `no-store`.
use super::{ApiResult, failure, invalid, model, store::Mutation, users};
use crate::server::{State, providers, security::Principal};
use axum::{
    Json, Router,
    extract::{Path, Query, Request, State as ExtractState, rejection::JsonRejection},
    response::Response,
};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

api_routes! {
    get "/api/me/models" => view;
    get "/api/me/models/providers" => list_providers;
    get "/api/me/models/providers/{id}/credentials" => credentials;
    post "/api/me/models/model-discovery" => discover;
    post "/api/me/models/test" => probe;
    post "/api/me/models/deployments/batch" => batch;
    patch "/api/me/models/deployments/{id}" => update;
    delete "/api/me/models/deployments/{id}" => delete;
    patch "/api/me/models/providers/{id}" => update_provider;
    delete "/api/me/models/providers/{id}" => delete_provider;
}

pub(in crate::server) fn routes() -> Router<Arc<State>> {
    api_routes().layer(axum::extract::DefaultBodyLimit::max(1024 * 1024))
}

fn body(value: Result<Json<Value>, JsonRejection>) -> ApiResult<Value> {
    value
        .map(|Json(v)| v)
        .map_err(|_| invalid("invalid settings JSON body"))
}

async fn view(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    crate::task::blocking(move || users::view(&state.settings, &state.users, &id))
        .await
        .map_err(|_| failure())
        .map(Json)
}

#[derive(Deserialize, Default)]
struct Refresh {
    #[serde(default)]
    refresh: bool,
}

async fn list_providers(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    Query(query): Query<Refresh>,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    // Computed on every request, as the admin's list is.
    let _ = query.refresh;
    crate::task::blocking(move || users::providers(&state.users, &id))
        .await
        .map_err(|_| failure())
        .map(Json)
}

async fn credentials(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    Path(connection): Path<String>,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    crate::task::blocking(move || users::credentials(&state.users, &id, &connection))
        .await
        .map_err(|_| failure())?
        .map(Json)
}

async fn network(
    state: Arc<State>,
    principal: Principal,
    request: Request,
    is_probe: bool,
) -> Response {
    let id = match users::caller(&principal) {
        Ok(id) => id,
        Err(error) => return providers::response(Err(error)),
    };
    let resolved = match providers::body(request).await {
        Ok(body) => {
            if is_probe {
                users::resolve_probe(&state.users, &id, &body)
            } else {
                users::resolve_discovery(&state.users, &id, &body)
            }
        }
        Err(error) => Err(error),
    };
    let request = match resolved {
        Ok(request) => request,
        Err(error) => return providers::response(Err(error)),
    };
    providers::response(
        providers::network(request, is_probe)
            .await
            .map(users::answer),
    )
}

async fn discover(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    request: Request,
) -> Response {
    network(state, principal, request, false).await
}

async fn probe(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    request: Request,
) -> Response {
    network(state, principal, request, true).await
}

async fn mutate(state: Arc<State>, id: String, command: Mutation) -> ApiResult<Json<Value>> {
    crate::task::blocking(move || {
        let alive = || state.access.exists(&id);
        users::mutate(&state.settings, &state.users, &id, &alive, command)
    })
    .await
    .map_err(|_| failure())?
    .map(Json)
}

async fn batch(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    value: Result<Json<Value>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    mutate(state, id, Mutation::Batch(body(value)?)).await
}

async fn update(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    Path(key): Path<String>,
    value: Result<Json<Value>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    let body = body(value)?;
    // As the settings' PATCH, with `expected_revision` required; `share` is
    // the admin's and refused as an unknown field.
    model::string(&body, "expected_revision", true, true)?;
    mutate(
        state,
        id,
        Mutation::Update {
            key,
            body,
            legacy: false,
            share: None,
        },
    )
    .await
}

#[derive(Deserialize)]
struct Revision {
    expected_revision: String,
}

fn revision(
    query: Result<Query<Revision>, axum::extract::rejection::QueryRejection>,
) -> ApiResult<String> {
    Ok(query
        .map_err(|_| invalid("expected_revision is required"))?
        .0
        .expected_revision)
}

async fn delete(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    Path(key): Path<String>,
    query: Result<Query<Revision>, axum::extract::rejection::QueryRejection>,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    let revision = revision(query)?;
    mutate(
        state,
        id,
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
    principal: Principal,
    Path(key): Path<String>,
    value: Result<Json<Value>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    let body = body(value)?;
    let revision = model::string(&body, "expected_revision", true, true)?.unwrap();
    mutate(
        state,
        id,
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
    principal: Principal,
    Path(key): Path<String>,
    query: Result<Query<Revision>, axum::extract::rejection::QueryRejection>,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    let revision = revision(query)?;
    mutate(
        state,
        id,
        Mutation::Provider {
            key,
            body: None,
            revision,
        },
    )
    .await
}
