//! Single-use download tickets. A browser saves a large file by navigating to
//! it, and a navigation cannot carry the `Authorization` header, so an
//! authenticated client first asks for a ticket: a random value that admits
//! exactly one GET of exactly one download path within a minute. The ticket
//! rides in the URL instead of the service token; once used or expired it is
//! worthless, so a URL kept in the browser's download list or history grants
//! nothing. Only the SHA-256 of a ticket is held.
use super::{
    State,
    types::{ApiError, ApiResult},
};
use axum::{Json, extract::State as ExtractState, http::StatusCode};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// How long a ticket stays valid after it is issued.
pub(super) const LIFETIME: Duration = Duration::from_secs(60);
/// Outstanding tickets at most; expired ones are dropped before counting.
const CAPACITY: usize = 64;
const MAX_PATH: usize = 4096;

#[derive(Default)]
pub(super) struct Tickets(Mutex<HashMap<[u8; 32], (String, Instant)>>);

fn digest(ticket: &str) -> [u8; 32] {
    Sha256::digest(ticket.as_bytes()).into()
}

/// Whether `path` is one of the download routes a ticket may name: a job's
/// file, a job's archive or the history archive. Paths are compared as sent,
/// so they must be the exact encoded form the browser will request.
pub(super) fn downloadable(path: &str) -> bool {
    if path.len() > MAX_PATH
        || !path.is_ascii()
        || path.contains(['?', '#', '\\'])
        || path
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b' ')
    {
        return false;
    }
    if path == "/api/history/archive" {
        return true;
    }
    let Some(rest) = path.strip_prefix("/api/jobs/") else {
        return false;
    };
    let Some((job, rest)) = rest.split_once('/') else {
        return false;
    };
    if job.is_empty() || job == "." || job == ".." {
        return false;
    }
    if rest == "archive" {
        return true;
    }
    let Some(relpath) = rest.strip_prefix("files/") else {
        return false;
    };
    !relpath.is_empty()
        && relpath
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

impl Tickets {
    /// Issue a ticket for `path`, or refuse when too many are outstanding.
    pub(super) fn issue(&self, path: &str, now: Instant) -> ApiResult<String> {
        let mut tickets = self.0.lock().unwrap();
        tickets.retain(|_, (_, expires)| *expires > now);
        if tickets.len() >= CAPACITY {
            return Err(ApiError::new(
                429,
                "too_many_tickets",
                "too many download tickets are outstanding; use or let them expire first",
            ));
        }
        let ticket = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        tickets.insert(digest(&ticket), (path.to_owned(), now + LIFETIME));
        Ok(ticket)
    }

    /// Spend a ticket on a GET of `path`. A ticket presented for another path
    /// is spent as well: it was disclosed, so it must not stay usable.
    pub(super) fn redeem(&self, ticket: &str, path: &str, now: Instant) -> bool {
        let Some((issued, expires)) = self.0.lock().unwrap().remove(&digest(ticket)) else {
            return false;
        };
        expires > now && issued == path
    }
}

/// `POST /api/download-tickets` with `{"path": "/api/jobs/…/archive"}`.
pub(super) async fn issue(
    ExtractState(state): ExtractState<Arc<State>>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let invalid = || {
        ApiError::new(
            422,
            "invalid_ticket_path",
            "a download ticket needs a JSON body {\"path\": …} naming a job file, a job archive or the history archive",
        )
    };
    let Json(body) = body.map_err(|_| invalid())?;
    let object = body.as_object().ok_or_else(invalid)?;
    if object.keys().any(|key| key != "path") {
        return Err(invalid());
    }
    let path = object
        .get("path")
        .and_then(Value::as_str)
        .filter(|path| downloadable(path))
        .ok_or_else(invalid)?;
    let ticket = state.tickets.issue(path, Instant::now())?;
    Ok((
        StatusCode::CREATED,
        Json(
            json!({"ticket":ticket,"url":format!("{path}?ticket={ticket}"),"expires_in":LIFETIME.as_secs()}),
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_download_routes_can_be_named() {
        for path in [
            "/api/history/archive",
            "/api/jobs/0123456789ab/archive",
            "/api/jobs/0123456789ab/files/notes.md",
            "/api/jobs/0123456789ab/files/assets/a%20b.png",
        ] {
            assert!(downloadable(path), "{path}");
        }
        for path in [
            "/api/settings/llm",
            "/api/settings/llm/providers/x/credentials",
            "/api/jobs/0123456789ab",
            "/api/jobs/0123456789ab/events",
            "/api/jobs/0123456789ab/files/",
            "/api/jobs/0123456789ab/files/../meta.json",
            "/api/jobs/0123456789ab/files/a//b",
            "/api/jobs/0123456789ab/files/a?token=x",
            "/api/jobs/0123456789ab/archive#x",
            "/api/jobs//archive",
            "/api/jobs/../archive",
            "/api/history",
            "/api/history/archive/",
            "/jobs",
            "/api/jobs/x/files/a b",
            "/api/jobs/x/files/naïve.md",
        ] {
            assert!(!downloadable(path), "{path}");
        }
        assert!(!downloadable(&format!(
            "/api/jobs/x/files/{}",
            "a".repeat(MAX_PATH)
        )));
    }

    #[test]
    fn a_ticket_admits_one_request_for_its_path_until_it_expires() {
        let tickets = Tickets::default();
        let now = Instant::now();
        let path = "/api/jobs/0123456789ab/archive";
        let ticket = tickets.issue(path, now).unwrap();
        assert_eq!(ticket.len(), 64);
        assert!(ticket.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert!(tickets.redeem(&ticket, path, now));
        assert!(!tickets.redeem(&ticket, path, now), "single use");

        // Presented for another path, the ticket is spent and refused.
        let ticket = tickets.issue(path, now).unwrap();
        assert!(!tickets.redeem(&ticket, "/api/history/archive", now));
        assert!(!tickets.redeem(&ticket, path, now));

        let ticket = tickets.issue(path, now).unwrap();
        assert!(!tickets.redeem(&ticket, path, now + LIFETIME));
        assert!(!tickets.redeem("not-issued", path, now));
        // Only the digest is held.
        assert!(
            !tickets
                .0
                .lock()
                .unwrap()
                .keys()
                .any(|key| key.as_slice() == ticket.as_bytes())
        );
    }

    #[test]
    fn outstanding_tickets_are_bounded_and_expired_ones_free_room() {
        let tickets = Tickets::default();
        let now = Instant::now();
        for _ in 0..CAPACITY {
            tickets.issue("/api/history/archive", now).unwrap();
        }
        let refused = tickets.issue("/api/history/archive", now).unwrap_err();
        assert_eq!(refused.status.as_u16(), 429);
        assert!(
            tickets
                .issue(
                    "/api/history/archive",
                    now + LIFETIME + Duration::from_secs(1)
                )
                .is_ok()
        );
    }
}
