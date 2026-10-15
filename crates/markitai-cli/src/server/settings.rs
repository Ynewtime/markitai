//! Persistent service settings. Network operations consume a detached configuration snapshot.
mod editor;
mod handlers;
mod identity;
mod me;
mod model;
mod pool;
mod store;
mod user_config;
mod users;
mod views;

#[cfg(test)]
pub(super) use handlers::ROUTES;
pub(super) use handlers::routes;
#[cfg(test)]
pub(super) use me::ROUTES as USER_ROUTES;
pub(super) use me::routes as user_routes;
use serde_json::Value;
use std::path::PathBuf;
pub(super) use store::{Store, load_base, load_written};
pub(super) use users::{Users, capabilities as user_capabilities, conversion_config};

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
/// The settings' "could not read or save" answer, for routes beside them.
pub(super) fn io_failure() -> ApiError {
    failure()
}
fn failure() -> ApiError {
    ApiError::new(
        500,
        "settings_io_failed",
        "Could not read or save the configuration file",
    )
}
