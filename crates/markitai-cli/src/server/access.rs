//! User access tokens. The server token is the admin's; an admin creates
//! user tokens in the workbench, each with a label. A user token admits every
//! API route except the settings and the token routes, and its holder sees
//! only the jobs it created. Only each token's SHA-256 is kept, in
//! `MARKITAI_HOME/serve/access.json` (owner-only, replaced atomically), so the
//! tokens survive restarts and admin-token regeneration; the token itself is
//! shown once, when it is created.
use super::{
    State,
    security::equal_secret,
    types::{ApiError, ApiResult, now},
};
use axum::{
    Json,
    extract::{Path, Request, State as ExtractState},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::{Path as FilePath, PathBuf},
    sync::{Arc, Mutex, RwLock},
};

/// User tokens at most.
pub(super) const MAX_TOKENS: usize = 50;
/// A label's characters at most.
const MAX_LABEL: usize = 60;
const PREFIX: &str = "mku_";
/// The file's format number (`version`).
const FORMAT: u64 = 1;

#[derive(Clone, Serialize, Deserialize)]
struct Entry {
    id: String,
    label: String,
    sha256: String,
    hint: String,
    created_at: String,
}

#[derive(Serialize, Deserialize)]
struct Saved {
    version: u32,
    tokens: Vec<Entry>,
}

pub(super) struct Access {
    path: PathBuf,
    tokens: RwLock<Vec<Entry>>,
    /// When each token was last presented; memory only.
    used: Mutex<HashMap<String, String>>,
    /// Serializes file replacements; readers keep the token list lock.
    write: Mutex<()>,
}

/// SHA-256 of `value`, lowercase hexadecimal.
fn digest(value: &str) -> String {
    markitai_core::hex(Sha256::digest(value.as_bytes()))
}

fn hex_of(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Whether `id` has the form of a user-token id (`u_` and 12 hex digits).
pub(super) fn valid_id(id: &str) -> bool {
    id.strip_prefix("u_").is_some_and(|rest| hex_of(rest, 12))
}

fn valid(entry: &Entry) -> bool {
    valid_id(&entry.id)
        && hex_of(&entry.sha256, 64)
        && label(&entry.label).is_some_and(|label| label == entry.label)
        && entry.hint.chars().count() == 4
        && hex_of(&entry.hint, 4)
        && super::types::instant(&entry.created_at).is_some()
}

/// A trimmed label of 1 to 60 characters without control characters.
fn label(text: &str) -> Option<String> {
    let text = text.trim();
    let count = text.chars().count();
    ((1..=MAX_LABEL).contains(&count) && !text.chars().any(char::is_control))
        .then(|| text.to_owned())
}

/// A fresh user token: `mku_` and 64 hex digits, 32 bytes derived by SHA-256
/// from three random UUIDs (the source of the server token).
fn new_token() -> String {
    let mut hasher = Sha256::new();
    for _ in 0..3 {
        hasher.update(uuid::Uuid::new_v4().as_bytes());
    }
    format!("{PREFIX}{}", markitai_core::hex(hasher.finalize()))
}

fn new_id() -> String {
    // A v4 UUID's first 12 hex digits are all random.
    format!("u_{}", &uuid::Uuid::new_v4().simple().to_string()[..12])
}

fn io_failed() -> ApiError {
    ApiError::new(
        500,
        "settings_io_failed",
        "Could not save the access tokens",
    )
}

impl Access {
    /// The tokens saved at `path`. A missing file holds none; an unreadable or
    /// invalid one is left as it is (until an admin creates or revokes a
    /// token) and the service starts without user tokens. A file a newer
    /// markitai wrote stops the start instead: replacing it would lose its
    /// tokens.
    pub(super) fn load(path: PathBuf) -> Result<Self, String> {
        let tokens = match read(&path) {
            Ok(tokens) => tokens,
            Err(Unreadable::Newer(version)) => {
                return Err(format!(
                    "Access tokens {} were written by a newer markitai (format {version}, this markitai reads {FORMAT}) and are left unchanged; upgrade markitai to serve with them, or start it with another MARKITAI_HOME",
                    path.display()
                ));
            }
            Err(Unreadable::Invalid(error)) => {
                eprintln!(
                    "Warning: Serve: ignoring the saved access tokens ({error}); no user token works until an admin creates or revokes one."
                );
                Vec::new()
            }
        };
        Ok(Self {
            path,
            tokens: RwLock::new(tokens),
            used: Mutex::new(HashMap::new()),
            write: Mutex::new(()),
        })
    }

    /// The user token `presented` is, as (id, label). The value is hashed and
    /// compared with every stored digest in constant time.
    pub(super) fn authenticate(&self, presented: &str) -> Option<(String, String)> {
        let presented = digest(presented);
        let tokens = self.tokens.read().unwrap_or_else(|e| e.into_inner());
        let mut found = None;
        for entry in tokens.iter() {
            if equal_secret(presented.as_bytes(), entry.sha256.as_bytes()) && found.is_none() {
                found = Some((entry.id.clone(), entry.label.clone()));
            }
        }
        if let Some((id, _)) = &found {
            self.used
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(id.clone(), now());
        }
        found
    }

    /// Whether the user token `id` still exists.
    pub(super) fn exists(&self, id: &str) -> bool {
        self.tokens
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|entry| entry.id == id)
    }

    fn public(&self, entry: &Entry) -> Value {
        let used = self
            .used
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&entry.id)
            .cloned();
        json!({"id":entry.id,"label":entry.label,"hint":entry.hint,
            "created_at":entry.created_at,"last_used_at":used})
    }

    /// Every user token, oldest first, without the token itself.
    pub(super) fn list(&self) -> Value {
        let tokens = self.tokens.read().unwrap_or_else(|e| e.into_inner());
        json!({"tokens":tokens.iter().map(|entry| self.public(entry)).collect::<Vec<_>>()})
    }

    /// Save a new token named `label` and return its public form with the
    /// token. Blocking: it writes the file.
    fn create(&self, label: String) -> ApiResult<Value> {
        let _write = self.write.lock().unwrap_or_else(|e| e.into_inner());
        let mut next = self
            .tokens
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if next.len() >= MAX_TOKENS {
            return Err(ApiError::new(
                409,
                "token_limit",
                format!("there are already {MAX_TOKENS} access tokens; revoke one first"),
            ));
        }
        let token = new_token();
        let mut id = new_id();
        while next.iter().any(|entry| entry.id == id) {
            id = new_id();
        }
        let entry = Entry {
            id,
            label,
            sha256: digest(&token),
            hint: token[token.len() - 4..].to_owned(),
            created_at: now(),
        };
        next.push(entry.clone());
        save(&self.path, &next).map_err(|error| {
            eprintln!("Error: Serve: could not save the access tokens: {error}");
            io_failed()
        })?;
        *self.tokens.write().unwrap_or_else(|e| e.into_inner()) = next;
        let mut created = self.public(&entry);
        created["token"] = json!(token);
        Ok(created)
    }

    /// Remove the token `id` and return its label. Blocking: it writes the file.
    fn revoke(&self, id: &str) -> ApiResult<String> {
        let _write = self.write.lock().unwrap_or_else(|e| e.into_inner());
        let mut next = self
            .tokens
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let index = next
            .iter()
            .position(|entry| entry.id == id)
            .ok_or_else(|| ApiError::new(404, "token_not_found", "access token not found"))?;
        let removed = next.remove(index);
        save(&self.path, &next).map_err(|error| {
            eprintln!("Error: Serve: could not save the access tokens: {error}");
            io_failed()
        })?;
        *self.tokens.write().unwrap_or_else(|e| e.into_inner()) = next;
        self.used
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id);
        Ok(removed.label)
    }
}

/// Why the saved tokens are not used.
enum Unreadable {
    /// A newer markitai's format number.
    Newer(u64),
    Invalid(String),
}

impl From<String> for Unreadable {
    fn from(error: String) -> Self {
        Self::Invalid(error)
    }
}

fn read(path: &FilePath) -> Result<Vec<Entry>, Unreadable> {
    let bytes = match markitai_core::config::read_file(path) {
        Ok(None) => return Ok(Vec::new()),
        Ok(Some(bytes)) => bytes,
        Err(error) => return Err(format!("{}: {error}", path.display()).into()),
    };
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    // Before the shape: a newer format may have changed it.
    if let Some(version) = value["version"]
        .as_u64()
        .filter(|version| *version > FORMAT)
    {
        return Err(Unreadable::Newer(version));
    }
    let saved: Saved =
        serde_json::from_value(value).map_err(|error| format!("{}: {error}", path.display()))?;
    if saved.version != FORMAT as u32 {
        return Err(format!("{}: unsupported version {}", path.display(), saved.version).into());
    }
    if saved.tokens.len() > MAX_TOKENS || !saved.tokens.iter().all(valid) {
        return Err(format!("{}: invalid token entry", path.display()).into());
    }
    let mut ids = std::collections::HashSet::new();
    if !saved.tokens.iter().all(|entry| ids.insert(&entry.id)) {
        return Err(format!("{}: duplicate token id", path.display()).into());
    }
    Ok(saved.tokens)
}

/// Replace the file atomically (staged beside it, owner-only, then renamed).
fn save(path: &FilePath, tokens: &[Entry]) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(&Saved {
        version: FORMAT as u32,
        tokens: tokens.to_vec(),
    })
    .map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    let staged = markitai_core::config::stage_file(path, &bytes)?;
    let directory = staged.directory().to_owned();
    // The staged file is created owner-only (0600 on Unix) and renamed whole.
    staged.publish()?;
    markitai_core::platform::sync_directory(&directory)
}

fn token_disabled() -> ApiError {
    ApiError::new(
        409,
        "token_disabled",
        "the server runs without an access token (--no-auth); user tokens are unavailable",
    )
}

fn no_store(mut response: Response) -> Response {
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}

fn announce(text: String) {
    crate::app::logging::shaped(
        crate::app::logging::Shape::item(crate::term::Mark::Action),
        &text,
        None,
    );
}

/// `GET /api/access-tokens`: the user tokens, oldest first.
pub(super) async fn list(ExtractState(state): ExtractState<Arc<State>>) -> ApiResult<Response> {
    if !state.auth_enabled() {
        return Err(token_disabled());
    }
    let worker = state.clone();
    // With how many models of its own each user has saved (never which).
    let list = crate::task::blocking(move || {
        let mut list = worker.access.list();
        for token in list["tokens"].as_array_mut().into_iter().flatten() {
            let count = worker.users.count(token["id"].as_str().unwrap_or_default());
            token["model_count"] = json!(count);
        }
        list
    })
    .await
    .map_err(ApiError::internal)?;
    Ok(no_store(Json(list).into_response()))
}

/// `POST /api/access-tokens` with `{"label": …}`: a new user token, shown
/// this once.
pub(super) async fn create(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
) -> ApiResult<Response> {
    let bytes = axum::body::to_bytes(request.into_body(), 1024)
        .await
        .map_err(|_| ApiError::new(413, "request_too_large", "request exceeds 1 KiB"))?;
    if !state.auth_enabled() {
        return Err(token_disabled());
    }
    let shape = || ApiError::new(422, "invalid_json", "send a JSON body {\"label\": \"…\"}");
    let body: Value = serde_json::from_slice(&bytes).map_err(|_| shape())?;
    let text = body
        .as_object()
        .filter(|object| object.len() == 1)
        .and_then(|object| object.get("label"))
        .and_then(Value::as_str)
        .ok_or_else(shape)?;
    let label = label(text).ok_or_else(|| {
        ApiError::new(
            422,
            "invalid_label",
            format!("label must be plain text of 1 to {MAX_LABEL} characters"),
        )
    })?;
    let worker = state.clone();
    let created = crate::task::blocking(move || worker.access.create(label))
        .await
        .map_err(ApiError::internal)??;
    announce(format!(
        "Access token \"{}\" created from the web workbench.",
        created["label"].as_str().unwrap_or_default()
    ));
    Ok(no_store(
        (StatusCode::CREATED, Json(created)).into_response(),
    ))
}

/// `DELETE /api/access-tokens/{id}`: the token stops working at once, with
/// its download tickets and open event streams.
pub(super) async fn revoke(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    if !state.auth_enabled() {
        return Err(token_disabled());
    }
    let worker = state.clone();
    let revoked = id.clone();
    // The token goes first; then its models, which no request of it can
    // write any more. A deletion that fails is retried at the next start.
    let (label, deleted) = crate::task::blocking(move || {
        let label = worker.access.revoke(&revoked)?;
        let deleted = worker.users.delete(&revoked).map_err(|error| {
            eprintln!(
                "Warning: Serve: could not delete the models of the revoked access token \"{label}\" ({error}); the next start deletes them."
            );
        });
        Ok::<_, ApiError>((label, deleted.unwrap_or(0)))
    })
    .await
    .map_err(ApiError::internal)??;
    state.tickets.revoke_user(&id);
    // Every open stream reconnects with its token; only this one is refused.
    state
        .token_generation
        .send_modify(|generation| *generation += 1);
    announce(if deleted == 1 {
        format!("Access token \"{label}\" revoked from the web workbench; its 1 model was deleted.")
    } else if deleted > 1 {
        format!(
            "Access token \"{label}\" revoked from the web workbench; its {deleted} models were deleted."
        )
    } else {
        format!("Access token \"{label}\" revoked from the web workbench.")
    });
    Ok(no_store(StatusCode::NO_CONTENT.into_response()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_ids_and_labels_have_their_forms() {
        let token = new_token();
        assert!(token.starts_with(PREFIX));
        assert!(hex_of(&token[PREFIX.len()..], 64), "{token}");
        assert_ne!(token, new_token());
        let id = new_id();
        assert!(valid_id(&id), "{id}");
        assert_eq!(label("  Team laptop "), Some("Team laptop".into()));
        assert_eq!(label(&"界".repeat(60)), Some("界".repeat(60)));
        for refused in ["", "   ", "a\nb", "a\u{7}b"] {
            assert_eq!(label(refused), None, "{refused:?}");
        }
        assert_eq!(label(&"a".repeat(61)), None);
    }

    #[test]
    fn only_digests_are_saved_and_a_restart_reads_them_back() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("serve/access.json");
        let access = Access::load(path.clone()).unwrap();
        let created = access.create("Laptop".into()).unwrap();
        let token = created["token"].as_str().unwrap().to_owned();
        assert_eq!(created["hint"], token[token.len() - 4..]);
        assert_eq!(created["last_used_at"], Value::Null);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains(&token));
        assert!(!text.contains(&token[PREFIX.len()..]));
        assert!(text.contains(&digest(&token)));
        let saved: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(saved["version"], 1);
        let keys: Vec<_> = saved["tokens"][0]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(keys, ["created_at", "hint", "id", "label", "sha256"]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let restarted = Access::load(path.clone()).unwrap();
        let (id, label) = restarted.authenticate(&token).unwrap();
        assert_eq!(
            (id.as_str(), label.as_str()),
            (created["id"].as_str().unwrap(), "Laptop")
        );
        assert!(restarted.authenticate(&format!("{token}x")).is_none());
        assert!(restarted.authenticate(&token[..token.len() - 1]).is_none());
        assert!(restarted.list()["tokens"][0]["last_used_at"].is_string());
        assert!(restarted.list()["tokens"][0].get("token").is_none());
        assert_eq!(restarted.revoke(&id).unwrap(), "Laptop");
        assert!(restarted.authenticate(&token).is_none());
        assert_eq!(restarted.revoke(&id).unwrap_err().reason, "token_not_found");
        assert!(
            Access::load(path).unwrap().list()["tokens"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn an_invalid_file_is_kept_until_a_token_changes_and_the_limit_holds() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("access.json");
        std::fs::write(&path, b"{not json").unwrap();
        let access = Access::load(path.clone()).unwrap();
        assert!(access.list()["tokens"].as_array().unwrap().is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), b"{not json");
        for index in 0..MAX_TOKENS {
            access.create(format!("Token {index}")).unwrap();
        }
        assert_eq!(
            access.create("One more".into()).unwrap_err().reason,
            "token_limit"
        );
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["tokens"].as_array().unwrap().len(), MAX_TOKENS);
        // A file that cannot be replaced changes nothing in memory.
        let blocked = Access::load(temp.path().join("missing-dir/file/access.json")).unwrap();
        std::fs::write(temp.path().join("missing-dir"), b"a file, not a folder").unwrap();
        assert_eq!(
            blocked.create("Blocked".into()).unwrap_err().reason,
            "settings_io_failed"
        );
        assert!(blocked.list()["tokens"].as_array().unwrap().is_empty());
    }

    #[test]
    fn a_newer_format_is_refused_and_left_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("access.json");
        // A newer format may change the rest of the shape too.
        let newer = br#"{"version":2,"tokens":{"renamed":true}}"#;
        std::fs::write(&path, newer).unwrap();
        let Err(error) = Access::load(path.clone()) else {
            panic!("a newer file was used");
        };
        assert!(error.starts_with(&format!("Access tokens {}", path.display())));
        assert!(error.contains(
            "were written by a newer markitai (format 2, this markitai reads 1) and are left unchanged"
        ));
        assert_eq!(std::fs::read(&path).unwrap(), newer);
        // An older or unknown version is a damaged file, kept until a change.
        for version in ["0", "\"2\""] {
            std::fs::write(&path, format!(r#"{{"version":{version},"tokens":[]}}"#)).unwrap();
            assert!(Access::load(path.clone()).is_ok());
        }
    }
}
