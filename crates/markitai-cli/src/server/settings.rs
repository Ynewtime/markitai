//! Persistent service settings. Network operations consume a detached configuration snapshot.
mod handlers;
mod identity;
mod model;
mod store;
mod views;

pub(super) use handlers::routes;
use serde_json::Value;
use std::path::PathBuf;
pub(super) use store::{Store, load_base};

pub(crate) struct SettingsSource {
    pub path: PathBuf,
    pub origin: String,
    pub overrides: Option<Value>,
}

use super::types::{ApiError, ApiResult};
fn invalid(message: &str) -> ApiError {
    ApiError::new(422, "invalid_settings_request", message)
}
fn missing() -> ApiError {
    ApiError::new(
        404,
        "settings_entry_missing",
        "This saved configuration no longer exists. Refresh settings and try again.",
    )
}
fn failure() -> ApiError {
    ApiError::new(
        500,
        "settings_io_failed",
        "Could not read or save the configuration file",
    )
}
