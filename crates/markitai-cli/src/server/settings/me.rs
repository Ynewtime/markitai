//! `/api/me/models*`: a user token's own models, with the settings routes'
//! suffixes and shapes, its weights for and tests of the models the admin
//! shares, and `/api/me/config`, its Edit config. Only a user token may use
//! them (anyone else: `user_token_required`); `security::guard` marks every
//! answer `no-store`.
use super::{ApiError, ApiResult, failure, invalid, model, store::Mutation, user_config, users};
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
    patch "/api/me/models/shared/{id}" => update_shared;
    get "/api/me/config" => config_view;
    put "/api/me/config" => config_replace;
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
        Ok(body) if is_probe && body.get("shared_id").is_some() => {
            return shared_probe(state, id, &body).await;
        }
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

/// Ends a user's test of a shared model when it is dropped.
struct Testing {
    state: Arc<State>,
    id: String,
}
impl Drop for Testing {
    fn drop(&mut self) {
        self.state.users.end_test(&self.id);
    }
}

/// `POST /api/me/models/test` with `{shared_id}`: the admin's entry of a
/// shared model tested on the server, one at a time and a few per minute
/// for each user; the answer says only whether it passed and why not.
async fn shared_probe(state: Arc<State>, id: String, body: &Value) -> Response {
    let shared = match body.as_object().filter(|fields| fields.len() == 1) {
        Some(fields) => match fields["shared_id"].as_str() {
            Some(shared) => shared.to_owned(),
            None => return providers::response(Err(invalid("shared_id must be a string"))),
        },
        None => {
            return providers::response(Err(invalid(
                "shared_id cannot be combined with other probe fields",
            )));
        }
    };
    let request = match users::resolve_shared_probe(&state.settings, &shared) {
        Ok(request) => request,
        Err(error) => return providers::response(Err(error)),
    };
    if let Err(seconds) = state.users.begin_test(&id) {
        let mut response = providers::response(Err(ApiError::new(
            429,
            "provider_busy",
            format!(
                "you can test a shared model once at a time and {} times a minute; try again shortly",
                users::SHARED_TESTS_PER_MINUTE
            ),
        )));
        response
            .headers_mut()
            .insert("retry-after", seconds.to_string().parse().unwrap());
        return response;
    }
    let _testing = Testing {
        state: state.clone(),
        id,
    };
    providers::response(users::shared_answer(
        providers::network(request, true).await,
    ))
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

/// `PATCH /api/me/models/shared/{id}` with exactly
/// `{expected_revision, weight}`: the user's weight for a shared model.
async fn update_shared(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    Path(shared): Path<String>,
    request: Request,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    let body = providers::body(request).await?;
    let shape =
        || invalid("send exactly {\"expected_revision\": string, \"weight\": integer or null}");
    let fields = body
        .as_object()
        .filter(|fields| {
            fields.len() == 2
                && fields.contains_key("expected_revision")
                && fields.contains_key("weight")
        })
        .ok_or_else(shape)?;
    let expected = fields["expected_revision"]
        .as_str()
        .ok_or_else(shape)?
        .to_owned();
    let weight = match &fields["weight"] {
        Value::Null => None,
        value => Some(model::weight(value)?),
    };
    crate::task::blocking(move || {
        let alive = || state.access.exists(&id);
        users::set_shared_weight(
            &state.settings,
            &state.users,
            &id,
            &alive,
            &shared,
            &expected,
            weight,
        )
    })
    .await
    .map_err(|_| failure())?
    .map(Json)
}

/// `GET /api/me/config`: the user's models file as the server writes it,
/// every secret masked.
async fn config_view(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    crate::task::blocking(move || user_config::config_view(&state.settings, &state.users, &id))
        .await
        .map_err(|_| failure())
        .map(Json)
}

/// `PUT /api/me/config` with `{expected_revision, text}`.
async fn config_replace(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    request: Request,
) -> ApiResult<Json<Value>> {
    let id = users::caller(&principal)?;
    let (expected, text) = super::handlers::config_body(request).await?;
    crate::task::blocking(move || {
        let alive = || state.access.exists(&id);
        user_config::replace_config(&state.settings, &state.users, &id, &alive, &expected, &text)
    })
    .await
    .map_err(|_| failure())?
    .map(Json)
}
