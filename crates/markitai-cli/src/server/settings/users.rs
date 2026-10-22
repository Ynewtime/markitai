//! A user token's own models. Each token keeps its models and connections in
//! `MARKITAI_HOME/serve/user-models/<token id>.json` (owner-only, replaced
//! atomically), never in the configuration file. They are changed with the
//! settings' own operations (`store::apply`), then held to rules that keep
//! the server's credentials and machine out of reach: no provider that signs
//! in with the server's own account, a key or an address of the user's own,
//! no address on the server's machine, no environment credentials at all.
use super::{
    ApiError, ApiResult, failure, identity, invalid, missing, model, pool,
    store::{self, Data, Mutation, Store},
    views,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// A user token's own models at most.
pub(in crate::server) const LIMIT: usize = 50;

/// One user's models and connections, as `llm.model_list` and
/// `llm.providers` entries, and its own weight for shared models (by the
/// admin's deployment id) where it differs from the admin's.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(in crate::server) struct Lists {
    pub models: Vec<Value>,
    pub providers: Vec<Value>,
    #[serde(default)]
    pub shared_weights: BTreeMap<String, u64>,
}

#[derive(Serialize, Deserialize)]
struct Saved {
    version: u32,
    models: Vec<Value>,
    providers: Vec<Value>,
    /// Absent when empty, so a file without overrides keeps its old shape.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    shared_weights: BTreeMap<String, u64>,
    /// The user's own conversion defaults (`/api/convert-defaults`); absent
    /// when it follows the server's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    convert_defaults: Option<serde_json::Map<String, Value>>,
}

/// A user's tests of shared models at most: one at a time, and this many per
/// rolling minute (the admin's key pays for them).
pub(in crate::server) const SHARED_TESTS_PER_MINUTE: usize = 6;
const SHARED_TEST_WINDOW: Duration = Duration::from_secs(60);

/// One user's recent tests of shared models.
#[derive(Default)]
struct Tests {
    running: bool,
    started: VecDeque<Instant>,
}

pub(in crate::server) struct Users {
    directory: PathBuf,
    /// Every user's lists read so far; the lock also serializes writes.
    lists: Mutex<HashMap<String, Lists>>,
    /// Every user's tests of shared models, in memory only.
    tests: Mutex<HashMap<String, Tests>>,
}

fn user_file(directory: &Path, id: &str) -> PathBuf {
    directory.join(format!("{id}.json"))
}

fn read(path: &Path) -> Result<Lists, String> {
    let bytes = match markitai_core::config::read_file(path) {
        Ok(None) => return Ok(Lists::default()),
        Ok(Some(bytes)) => bytes,
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    let saved: Saved =
        serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    if saved.version != 1 {
        return Err(format!(
            "{}: unsupported version {}",
            path.display(),
            saved.version
        ));
    }
    let models_valid = saved
        .models
        .iter()
        .all(|entry| entry.is_object() && entry["litellm_params"]["model"].is_string());
    let providers_valid = saved
        .providers
        .iter()
        .all(|entry| entry.is_object() && entry["id"].is_string() && entry["provider"].is_string());
    if !models_valid
        || !providers_valid
        || saved.models.len() > LIMIT
        || saved.shared_weights.values().any(|w| *w > MAX_WEIGHT)
    {
        return Err(format!("{}: invalid model entry", path.display()));
    }
    Ok(Lists {
        models: saved.models,
        providers: saved.providers,
        shared_weights: saved.shared_weights,
    })
}

/// Replace the file atomically (staged beside it, owner-only, then renamed);
/// whether the directory is known to be durable.
fn save(
    directory: &Path,
    id: &str,
    lists: &Lists,
    convert_defaults: Option<&serde_json::Map<String, Value>>,
) -> std::io::Result<bool> {
    markitai_core::output::create_private_dir(directory)?;
    let mut bytes = serde_json::to_vec_pretty(&Saved {
        version: 1,
        models: lists.models.clone(),
        providers: lists.providers.clone(),
        shared_weights: lists.shared_weights.clone(),
        convert_defaults: convert_defaults.cloned(),
    })
    .map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    let staged = markitai_core::config::stage_file(&user_file(directory, id), &bytes)?;
    let folder = staged.directory().to_owned();
    staged.publish()?;
    Ok(markitai_core::platform::sync_directory(&folder).is_ok())
}

impl Users {
    pub(in crate::server) fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            lists: Mutex::new(HashMap::new()),
            tests: Mutex::new(HashMap::new()),
        }
    }

    /// Whether the user `id` has a file of its own.
    pub(in crate::server) fn exists(&self, id: &str) -> bool {
        user_file(&self.directory, id).is_file()
    }

    /// Start a test of a shared model for the user `id`, within the bound;
    /// else how many seconds until one may start.
    pub(in crate::server) fn begin_test(&self, id: &str) -> Result<(), u64> {
        let mut tests = self.tests.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let entry = tests.entry(id.to_owned()).or_default();
        while entry
            .started
            .front()
            .is_some_and(|at| now.duration_since(*at) >= SHARED_TEST_WINDOW)
        {
            entry.started.pop_front();
        }
        if entry.running {
            return Err(1);
        }
        if entry.started.len() >= SHARED_TESTS_PER_MINUTE {
            let oldest = *entry.started.front().unwrap();
            let left = SHARED_TEST_WINDOW.saturating_sub(now.duration_since(oldest));
            return Err(left.as_secs_f64().ceil().max(1.0) as u64);
        }
        entry.running = true;
        entry.started.push_back(now);
        Ok(())
    }

    /// The user `id`'s test of a shared model has ended.
    pub(in crate::server) fn end_test(&self, id: &str) {
        let mut tests = self.tests.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = tests.get_mut(id) {
            entry.running = false;
        }
    }

    fn cached<'a>(&self, lists: &'a mut HashMap<String, Lists>, id: &str) -> &'a mut Lists {
        lists.entry(id.to_owned()).or_insert_with(|| {
            read(&user_file(&self.directory, id)).unwrap_or_else(|error| {
                eprintln!(
                    "Warning: Serve: ignoring a user token's saved models ({error}); it has none until it saves one."
                );
                Lists::default()
            })
        })
    }

    /// The user `id`'s models and connections. A missing file holds none; an
    /// unreadable or invalid one is left as it is until the user's next
    /// change replaces it, and holds none meanwhile.
    pub(in crate::server) fn lists(&self, id: &str) -> Lists {
        let mut lists = self.lists.lock().unwrap_or_else(|e| e.into_inner());
        self.cached(&mut lists, id).clone()
    }

    /// How many models of its own the user `id` has.
    pub(in crate::server) fn count(&self, id: &str) -> usize {
        self.lists(id).models.len()
    }

    /// Change the user `id`'s lists with `change` and save them, if the
    /// token still exists (`alive`, asked under the lock a revocation's
    /// deletion takes too). A change that changes nothing writes nothing.
    pub(super) fn write(
        &self,
        id: &str,
        alive: &dyn Fn() -> bool,
        change: impl FnOnce(&mut Lists) -> ApiResult<()>,
    ) -> ApiResult<Lists> {
        let mut lists = self.lists.lock().unwrap_or_else(|e| e.into_inner());
        if !alive() {
            return Err(ApiError::new(
                401,
                "token_required",
                "this access token was revoked; sign in again",
            ));
        }
        let current = self.cached(&mut lists, id).clone();
        let mut next = current.clone();
        change(&mut next)?;
        if next == current {
            return Ok(next);
        }
        let defaults = read_defaults(&user_file(&self.directory, id));
        let durable = save(&self.directory, id, &next, defaults.as_ref()).map_err(|error| {
            eprintln!("Error: Serve: could not save a user token's models: {error}");
            failure()
        })?;
        lists.insert(id.to_owned(), next.clone());
        if !durable {
            return Err(ApiError::new(
                500,
                "settings_durability_unknown",
                "Your models were saved, but directory durability could not be confirmed; reload them before another change",
            ));
        }
        Ok(next)
    }

    /// Delete the user `id`'s models (its token was revoked); how many there
    /// were.
    pub(in crate::server) fn delete(&self, id: &str) -> std::io::Result<usize> {
        let mut lists = self.lists.lock().unwrap_or_else(|e| e.into_inner());
        let count = self.cached(&mut lists, id).models.len();
        lists.remove(id);
        match std::fs::remove_file(user_file(&self.directory, id)) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
        let _ = markitai_core::platform::sync_directory(&self.directory);
        Ok(count)
    }

    /// Delete the files of tokens that no longer exist (`current` says which
    /// do), and staged files a stopped write left behind.
    pub(in crate::server) fn sweep(&self, current: impl Fn(&str) -> bool) {
        let Ok(entries) = std::fs::read_dir(&self.directory) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let orphan = match name.strip_suffix(".json") {
                Some(id) => !current(id),
                None => name.starts_with(".tmp"),
            };
            let regular = entry
                .path()
                .symlink_metadata()
                .is_ok_and(|metadata| metadata.is_file());
            if orphan
                && regular
                && let Err(error) = std::fs::remove_file(entry.path())
            {
                eprintln!(
                    "Warning: Serve: could not delete the saved models of a revoked access token ({}): {error}",
                    entry.path().display()
                );
            }
        }
    }
}

/// The conversion defaults a user's file keeps; none when it has none, or
/// when they are unreadable (the next save of defaults replaces them).
fn read_defaults(path: &Path) -> Option<serde_json::Map<String, Value>> {
    let bytes = markitai_core::config::read_file(path).ok()??;
    let saved: Value = serde_json::from_slice(&bytes).ok()?;
    crate::server::convert_defaults::checked(saved.get("convert_defaults")?).ok()
}

impl Users {
    /// The user `id`'s own conversion defaults, if it saved any.
    pub(in crate::server) fn convert_defaults(
        &self,
        id: &str,
    ) -> Option<serde_json::Map<String, Value>> {
        let _lists = self.lists.lock().unwrap_or_else(|e| e.into_inner());
        read_defaults(&user_file(&self.directory, id))
    }

    /// Replace the user `id`'s conversion defaults with what `change` makes of
    /// the current ones (`None`: follow the server's), if its token still
    /// exists; its models stay as they are.
    pub(in crate::server) fn set_convert_defaults(
        &self,
        id: &str,
        alive: &dyn Fn() -> bool,
        change: impl FnOnce(
            Option<&serde_json::Map<String, Value>>,
        ) -> ApiResult<Option<serde_json::Map<String, Value>>>,
    ) -> ApiResult<Option<serde_json::Map<String, Value>>> {
        let mut lists = self.lists.lock().unwrap_or_else(|e| e.into_inner());
        if !alive() {
            return Err(ApiError::new(
                401,
                "token_required",
                "this access token was revoked; sign in again",
            ));
        }
        let path = user_file(&self.directory, id);
        let current = read_defaults(&path);
        let next = change(current.as_ref())?;
        if next == current {
            return Ok(next);
        }
        let models = self.cached(&mut lists, id).clone();
        let durable = save(&self.directory, id, &models, next.as_ref()).map_err(|error| {
            eprintln!("Error: Serve: could not save a user token's conversion defaults: {error}");
            failure()
        })?;
        if !durable {
            return Err(ApiError::new(
                500,
                "settings_durability_unknown",
                "Your defaults were saved, but directory durability could not be confirmed; reload them before another change",
            ));
        }
        Ok(next)
    }
}

fn server_credentials() -> ApiError {
    ApiError::new(
        422,
        "server_credentials",
        "this provider signs in with the server's own account; add a model with your own API key",
    )
}
fn key_or_address_required() -> ApiError {
    ApiError::new(
        422,
        "key_or_address_required",
        "a model needs your own API key, or the address of a server that takes none",
    )
}
fn server_address() -> ApiError {
    ApiError::new(
        422,
        "server_address",
        "that address is the server's own machine, which belongs to the admin; use another address",
    )
}
fn environment_credentials() -> ApiError {
    invalid("environment credentials are the server's")
}
fn user_token_required() -> ApiError {
    ApiError::new(
        403,
        "user_token_required",
        "these routes manage a user token's own models; the admin's are under /api/settings/llm",
    )
}

/// The user token a request acts for, else `user_token_required`.
pub(in crate::server) fn caller(
    principal: &crate::server::security::Principal,
) -> ApiResult<String> {
    principal
        .user()
        .map(str::to_owned)
        .ok_or_else(user_token_required)
}

fn real(value: &Value) -> bool {
    model::real_key(value.as_str())
}
fn text(value: &Value) -> Option<&str> {
    value.as_str().filter(|value| !value.trim().is_empty())
}

/// Refuse one connection of a user: `provider` with `key` at `base`.
fn check_connection(provider: &str, key: &Value, base: &Value) -> ApiResult<()> {
    let provider = provider.trim().to_ascii_lowercase();
    if model::server_login(&provider) || provider == "bedrock" && !real(key) {
        return Err(server_credentials());
    }
    if [key, base].iter().any(|value| {
        value
            .as_str()
            .is_some_and(|v| v.trim_start().starts_with("env:"))
    }) {
        return Err(invalid(
            "environment variables are the server's own; enter the key itself",
        ));
    }
    let base = text(base);
    check_address(&provider, base)?;
    if !real(key) && base.is_none() {
        return Err(key_or_address_required());
    }
    Ok(())
}

/// Refuse an address that is not HTTP(S), or one where requests of
/// `provider` would go to the server's own machine (its documented address
/// when `base` is none).
fn check_address(provider: &str, base: Option<&str>) -> ApiResult<()> {
    if let Some(base) = base
        && url::Url::parse(base.trim())
            .ok()
            .filter(|url| matches!(url.scheme(), "http" | "https") && url.host().is_some())
            .is_none()
    {
        return Err(invalid("api_base must be an http or https address"));
    }
    if base
        .or_else(|| model::default_base(provider))
        .is_some_and(model::server_machine)
    {
        return Err(server_address());
    }
    Ok(())
}

/// Refuse a user's model entry the rules do not allow, with its linked
/// connection's key and address.
pub(super) fn check_entry(entry: &Value, providers: &[Value]) -> ApiResult<()> {
    if let Some(id) = model::linked(entry)
        && !providers.iter().any(|p| p["id"] == id)
    {
        return Err(missing());
    }
    let params = model::params(entry, providers);
    check_connection(
        &model::provider(entry),
        &params["api_key"],
        &params["api_base"],
    )
}

/// Hold a user's lists after a change to the rules, then seal them: no entry
/// or connection may ever use the environment's credentials.
pub(super) fn check(lists: &mut Lists) -> ApiResult<()> {
    for connection in &lists.providers {
        check_connection(
            connection["provider"].as_str().unwrap_or_default(),
            &connection["api_key"],
            &connection["api_base"],
        )?;
    }
    for entry in &lists.models {
        check_entry(entry, &lists.providers)?;
    }
    if lists.models.len() > LIMIT {
        return Err(ApiError::new(
            409,
            "model_limit",
            format!("you have {LIMIT} models already; delete one first"),
        ));
    }
    for entry in &mut lists.models {
        entry["litellm_params"]["use_environment_credentials"] = json!(false);
    }
    for connection in &mut lists.providers {
        connection["use_environment_credentials"] = json!(false);
    }
    Ok(())
}

/// Refuse what a request names before anything changes: a provider that
/// signs in with the server's own account, or the environment's credentials.
pub(super) fn precheck(entry: &Value) -> ApiResult<()> {
    let prefix = model::provider(entry);
    let named = entry["provider"].as_str().map(str::to_ascii_lowercase);
    if model::server_login(&prefix) || named.as_deref().is_some_and(model::server_login) {
        return Err(server_credentials());
    }
    if entry["credential_provider_id"]
        .as_str()
        .is_some_and(|id| id.starts_with("env:"))
    {
        return Err(environment_credentials());
    }
    Ok(())
}

/// The user's lists as settings data, for the settings' views and resolvers.
pub(super) fn data(lists: Lists) -> Data {
    Data {
        cfg: Arc::new(Value::Null),
        revision: revision(&lists),
        models: lists.models,
        providers: lists.providers,
        detected: Vec::new(),
    }
}

/// The revision of a user's lists: its models, connections and weights of
/// shared models.
pub(super) fn revision(lists: &Lists) -> String {
    identity::user_revision(&lists.models, &lists.providers, &lists.shared_weights)
}

/// A weight at most, as the admin's `weight` field takes it.
pub(super) const MAX_WEIGHT: u64 = i64::MAX as u64;

/// One admin model users currently see: shared, admin weight above 0.
pub(super) struct Shared {
    pub id: String,
    pub model: Value,
    pub provider: String,
    /// The admin's weight.
    pub weight: u64,
}

/// The admin's models users currently see, in the saved order.
pub(super) fn shared_models(store: &Store) -> Vec<Shared> {
    store.with_data(|admin| {
        pool::shared(admin)
            .into_iter()
            .map(|(index, entry)| Shared {
                id: identity::id(entry, index),
                model: entry["litellm_params"]["model"].clone(),
                provider: model::connection_provider(entry, &admin.providers),
                weight: pool::weight(entry),
            })
            .collect()
    })
}

/// A user's weight for the shared model `shared`: its own, else the admin's.
pub(super) fn effective_weight(lists: &Lists, shared: &Shared) -> u64 {
    lists
        .shared_weights
        .get(&shared.id)
        .copied()
        .unwrap_or(shared.weight)
}

/// Set the user's weight for `shared` to `weight`: an override only when it
/// differs from the admin's weight (`None` or the admin's: none).
pub(super) fn set_weight(lists: &mut Lists, shared: &Shared, weight: Option<u64>) {
    match weight.filter(|weight| *weight != shared.weight) {
        Some(weight) => {
            lists.shared_weights.insert(shared.id.clone(), weight);
        }
        None => {
            lists.shared_weights.remove(&shared.id);
        }
    }
}

/// The configuration a conversion of the user `owner` runs with: the admin's
/// runtime configuration routing to the user's pool only.
pub(in crate::server) fn conversion_config(store: &Store, users: &Users, owner: &str) -> Value {
    let own = users.lists(owner);
    store.with_data(|admin| {
        pool::build(&admin.cfg, &own, admin).unwrap_or_else(|error| {
            eprintln!(
                "Warning: Serve: a user token's models could not be combined ({error}); its conversions run without models."
            );
            pool::empty(&admin.cfg)
        })
    })
}

/// What `GET /api/capabilities` says of a user's models: the pool's
/// routability, whether it has any model, and their ids.
pub(in crate::server) fn capabilities(pool: &Value) -> Value {
    let mut llm = markitai_core::llm_capabilities(pool);
    llm.models = pool::models(pool);
    llm.configured = !llm.models.is_empty();
    json!(llm)
}

/// `GET /api/me/models` for the user `id`.
pub(in crate::server) fn view(store: &Store, users: &Users, id: &str) -> Value {
    shaped(store, users, id, users.lists(id))
}

fn shaped(store: &Store, users: &Users, id: &str, lists: Lists) -> Value {
    let pool = conversion_config(store, users, id);
    let routable = markitai_core::llm_capabilities(&pool).routable;
    // What a shared model is, never its key, address or connection: the
    // user's weight for it and the admin's.
    let shared = shared_models(store)
        .iter()
        .map(|shared| {
            json!({"id":shared.id,"model":shared.model,"provider":shared.provider,
                "weight":effective_weight(&lists,shared),"default_weight":shared.weight})
        })
        .collect::<Vec<_>>();
    views::user_payload(&data(lists), shared, routable)
}

/// One of the `/api/me/models` changes for the user `id`; the view after it.
pub(in crate::server) fn mutate(
    store: &Store,
    users: &Users,
    id: &str,
    alive: &dyn Fn() -> bool,
    command: Mutation,
) -> ApiResult<Value> {
    if let Mutation::Batch(body) = &command
        && let Some(entries) = body["deployments"].as_array()
    {
        for entry in entries {
            precheck(entry)?;
        }
    }
    let lists = users.write(id, alive, |lists| {
        let current = revision(lists);
        store::apply_at(&mut lists.models, &mut lists.providers, command, current)?;
        check(lists)
    })?;
    Ok(shaped(store, users, id, lists))
}

fn stale(current: String) -> ApiError {
    ApiError::structured(
        409,
        "stale_revision",
        json!({"code":"stale_revision","current_revision":current}),
    )
}

/// `PATCH /api/me/models/shared/{shared}` for the user `id`: its own weight
/// for a shared model (`None`: the admin's).
pub(in crate::server) fn set_shared_weight(
    store: &Store,
    users: &Users,
    id: &str,
    alive: &dyn Fn() -> bool,
    shared: &str,
    expected: &str,
    weight: Option<u64>,
) -> ApiResult<Value> {
    let listed = shared_models(store);
    let target = listed
        .iter()
        .find(|model| model.id == shared)
        .ok_or_else(missing)?;
    let lists = users.write(id, alive, |lists| {
        let current = revision(lists);
        if expected != current {
            return Err(stale(current));
        }
        set_weight(lists, target, weight);
        Ok(())
    })?;
    Ok(shaped(store, users, id, lists))
}

/// `POST /api/me/models/test` with `{shared_id}`: the admin's entry of that
/// shared model, resolved as the admin's own test of it would be.
pub(in crate::server) fn resolve_shared_probe(store: &Store, shared: &str) -> ApiResult<Value> {
    store.with_data(|admin| {
        pool::shared(admin)
            .into_iter()
            .find(|(index, entry)| identity::id(entry, *index) == shared)
            .ok_or_else(|| invalid("deployment not found"))?;
        store::resolve_probe(admin, &json!({"deployment_id":shared}))
    })
}

/// The one sentence a failed test of a shared model says, whatever failed.
pub(in crate::server) const SHARED_TEST_FAILED: &str =
    "The shared model did not pass its test; ask the admin to check it";

/// A test answer of a shared model for a user: whether it passed and the
/// test's reason, never the provider's words, an address or a variable.
pub(in crate::server) fn shared_answer(value: ApiResult<Value>) -> ApiResult<Value> {
    let reason = match value {
        Ok(value) if value["ok"] == true => {
            return Ok(json!({"ok":true,"detail":"Model test passed"}));
        }
        Ok(value) => value["reason"]
            .as_str()
            .filter(|reason| {
                markitai_core::provider_management::probe_reasons().any(|known| known == *reason)
            })
            .unwrap_or("provider_error")
            .to_owned(),
        // Every server's tests at once: busy, as for any test.
        Err(error) if error.reason == "provider_busy" => return Err(error),
        Err(_) => "provider_error".to_owned(),
    };
    Ok(json!({"ok":false,"reason":reason,"detail":SHARED_TEST_FAILED}))
}

/// `GET /api/me/models/providers`.
pub(in crate::server) fn providers(users: &Users, id: &str) -> Value {
    views::user_providers(&data(users.lists(id)))
}

/// `GET /api/me/models/providers/{id}/credentials`: the user's connection
/// `connection`, the key only masked, never the environment's.
pub(in crate::server) fn credentials(
    users: &Users,
    id: &str,
    connection: &str,
) -> ApiResult<Value> {
    if connection.starts_with("env:") {
        return Err(missing());
    }
    let mut value = store::credentials(&data(users.lists(id)), connection)?;
    if value["api_key"]
        .as_str()
        .is_some_and(|key| key.starts_with("env:"))
    {
        value["api_key"] = Value::Null;
        value["api_key_masked"] = Value::Null;
    }
    value["api_key_source"] = Value::Null;
    if value["api_base_placeholder"]
        .as_str()
        .is_some_and(model::server_machine)
    {
        value["api_base_placeholder"] = Value::Null;
    }
    Ok(value)
}

/// The keyless placeholder for a request without a key of its own at an
/// address of its own: the server's key is never sent there.
fn keyless(request: &mut Value) {
    if !real(&request["api_key"]) && text(&request["api_base"]).is_some() {
        request["api_key"] = json!(model::NO_KEY);
    }
}

/// `POST /api/me/models/model-discovery`, resolved over the user's own
/// connections, with literal credentials only.
pub(in crate::server) fn resolve_discovery(
    users: &Users,
    id: &str,
    body: &Value,
) -> ApiResult<Value> {
    if body["provider_id"]
        .as_str()
        .is_some_and(|id| id.starts_with("env:"))
    {
        return Err(environment_credentials());
    }
    let provider = body["provider"]
        .as_str()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if model::server_login(&provider) {
        return Err(server_credentials());
    }
    let mut request = store::resolve_discovery(&data(users.lists(id)), body)?;
    request["use_environment_credentials"] = json!(false);
    request["public_only"] = json!(true);
    let provider = request["provider"].as_str().unwrap_or_default().to_owned();
    if provider == "bedrock" && !real(&request["api_key"]) {
        return Err(server_credentials());
    }
    check_address(&provider, text(&request["api_base"]))?;
    keyless(&mut request);
    Ok(request)
}

/// `POST /api/me/models/test`: one of the user's own models, or a draft.
pub(in crate::server) fn resolve_probe(users: &Users, id: &str, body: &Value) -> ApiResult<Value> {
    model::object(body, &["deployment_id", "model", "api_key", "api_base"])?;
    if let Some(name) = body["model"].as_str() {
        precheck(&json!({"litellm_params":{"model":name}}))?;
    }
    let mut request = store::resolve_probe(&data(users.lists(id)), body)?;
    request["use_environment_credentials"] = json!(false);
    request["public_only"] = json!(true);
    let name = request["model"].as_str().unwrap_or_default().to_owned();
    check_connection(
        &model::provider(&json!({"litellm_params":{"model":name}})),
        &request["api_key"],
        &request["api_base"],
    )?;
    keyless(&mut request);
    Ok(request)
}

/// A test or discovery answer for a user: no environment variable named.
pub(in crate::server) fn answer(mut value: Value) -> Value {
    if let Some(fields) = value.as_object_mut()
        && fields.remove("key_variable").is_some()
    {
        fields.insert(
            "detail".into(),
            json!("This model needs an API key, and none is set"),
        );
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_servers_own_machine_is_found_by_its_address() {
        for own in [
            "http://localhost:11434",
            "http://LOCALHOST./v1",
            "http://models.localhost:8000/v1",
            "http://127.0.0.1:1234/v1",
            "http://127.8.9.10/v1",
            "http://127.1/v1",
            "http://[::1]:8000/v1",
            "http://[::ffff:127.0.0.1]/v1",
            "http://0.0.0.0:8000",
            "http://[::]:8000",
        ] {
            assert!(model::server_machine(own), "{own}");
        }
        for other in [
            "https://api.openai.com/v1",
            "http://192.0.2.10:8000/v1",
            "http://[2001:db8::1]/v1",
            "http://localhost.example/v1",
            "not an address",
        ] {
            assert!(!model::server_machine(other), "{other}");
        }
    }

    #[test]
    fn the_rules_refuse_the_servers_credentials_and_machine() {
        let reason = |provider: &str, key: Value, base: Value| {
            check_connection(provider, &key, &base)
                .err()
                .map(|error| error.reason)
        };
        let none = Value::Null;
        assert_eq!(
            reason("openai", json!("user-a-fixture-key"), none.clone()),
            None
        );
        assert_eq!(
            reason("openai", none.clone(), json!("https://models.example/v1")),
            None
        );
        assert_eq!(
            reason(
                "openai",
                json!(model::NO_KEY),
                json!("https://models.example/v1")
            ),
            None
        );
        for login in [
            "claude-agent",
            "copilot",
            "chatgpt",
            "vertex_ai",
            "watsonx",
            "gigachat",
        ] {
            assert_eq!(
                reason(login, json!("user-a-fixture-key"), none.clone()),
                Some("server_credentials"),
                "{login}"
            );
        }
        assert_eq!(
            reason("bedrock", none.clone(), none.clone()),
            Some("server_credentials")
        );
        assert_eq!(
            reason("bedrock", json!("user-a-fixture-key"), none.clone()),
            None
        );
        assert_eq!(
            reason("openai", none.clone(), none.clone()),
            Some("key_or_address_required")
        );
        assert_eq!(
            reason("openai", json!(model::NO_KEY), none.clone()),
            Some("key_or_address_required")
        );
        assert_eq!(
            reason("ollama", none.clone(), none.clone()),
            Some("server_address")
        );
        assert_eq!(
            reason("ollama", json!("user-a-fixture-key"), none.clone()),
            Some("server_address")
        );
        assert_eq!(
            reason("ollama", none.clone(), json!("http://127.0.0.1:11434")),
            Some("server_address")
        );
        assert_eq!(
            reason("ollama", none.clone(), json!("http://192.0.2.4:11434")),
            None
        );
        assert_eq!(
            reason("openai", json!("env:OPENAI_API_KEY"), none.clone()),
            Some("invalid_settings_request")
        );
        assert_eq!(
            reason(
                "openai",
                json!("user-a-fixture-key"),
                json!("ftp://models.example")
            ),
            Some("invalid_settings_request")
        );
    }

    #[test]
    fn shared_model_tests_are_one_at_a_time_and_six_a_minute_per_user() {
        let temp = tempfile::tempdir().unwrap();
        let users = Users::new(temp.path().join("serve/user-models"));
        users.begin_test("u_000000000001").unwrap();
        // One at a time, for that user only.
        assert_eq!(users.begin_test("u_000000000001"), Err(1));
        users.begin_test("u_000000000002").unwrap();
        users.end_test("u_000000000001");
        for _ in 1..SHARED_TESTS_PER_MINUTE {
            users.begin_test("u_000000000001").unwrap();
            users.end_test("u_000000000001");
        }
        // The seventh within the minute waits for the first to leave it.
        let wait = users.begin_test("u_000000000001").unwrap_err();
        assert!((1..=60).contains(&wait), "{wait}");
        users.end_test("u_000000000002");
        users.begin_test("u_000000000002").unwrap();
    }

    #[test]
    fn a_files_shared_weights_are_kept_and_count_in_its_revision() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("serve/user-models");
        let users = Users::new(directory.clone());
        let alive = || true;
        let entry = json!({"model_name":"default","litellm_params":{"model":"openai/m","api_key":"user-a-fixture-key"}});
        let plain = users
            .write("u_000000000001", &alive, |lists| {
                lists.models.push(entry.clone());
                Ok(())
            })
            .unwrap();
        // No overrides: the revision of v1.15, and no key in the file.
        assert_eq!(
            revision(&plain),
            identity::revision(&plain.models, &plain.providers)
        );
        let path = directory.join("u_000000000001.json");
        assert!(
            !std::fs::read_to_string(&path)
                .unwrap()
                .contains("shared_weights")
        );
        let weighted = users
            .write("u_000000000001", &alive, |lists| {
                lists.shared_weights.insert("s1".into(), 0);
                Ok(())
            })
            .unwrap();
        assert_ne!(revision(&weighted), revision(&plain));
        let saved: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["shared_weights"], json!({"s1":0}));
        assert_eq!(saved["version"], 1);
        let restarted = Users::new(directory.clone());
        assert_eq!(restarted.lists("u_000000000001"), weighted);
        // A change that changes nothing writes nothing.
        std::fs::write(&path, b"{not json").unwrap();
        users.write("u_000000000001", &alive, |_| Ok(())).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"{not json");
    }

    #[test]
    fn files_are_owner_only_and_revoked_or_orphaned_ones_are_deleted() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("serve/user-models");
        let users = Users::new(directory.clone());
        let alive = || true;
        let saved = users
            .write("u_000000000001", &alive, |lists| {
                lists.models.push(json!({"model_name":"default","litellm_params":{"model":"openai/m","api_key":"user-a-fixture-key"}}));
                Ok(())
            })
            .unwrap();
        assert_eq!(saved.models.len(), 1);
        let path = directory.join("u_000000000001.json");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&path), 0o600);
            assert_eq!(mode(&directory), 0o700);
        }
        let restarted = Users::new(directory.clone());
        assert_eq!(restarted.count("u_000000000001"), 1);
        assert_eq!(restarted.count("u_000000000002"), 0);
        // A revoked token cannot write; an invalid file holds nothing and is
        // kept until a write replaces it.
        let gone = || false;
        assert_eq!(
            restarted
                .write("u_000000000001", &gone, |_| Ok(()))
                .unwrap_err()
                .reason,
            "token_required"
        );
        std::fs::write(directory.join("u_000000000003.json"), b"{not json").unwrap();
        assert_eq!(restarted.count("u_000000000003"), 0);
        assert_eq!(
            std::fs::read(directory.join("u_000000000003.json")).unwrap(),
            b"{not json"
        );
        std::fs::write(directory.join("u_000000000004.json"), b"{}").unwrap();
        restarted.sweep(|id| id != "u_000000000004");
        assert!(!directory.join("u_000000000004.json").exists());
        assert!(path.exists());
        assert_eq!(restarted.delete("u_000000000001").unwrap(), 1);
        assert!(!path.exists());
        assert_eq!(restarted.count("u_000000000001"), 0);
    }
}
