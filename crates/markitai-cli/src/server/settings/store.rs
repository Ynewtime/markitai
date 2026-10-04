use super::{
    ApiError, ApiResult, SettingsSource, failure, identity, invalid, missing, model, views,
};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    path::Path,
    sync::{Arc, Mutex},
};
const LIMIT: usize = 8 * 1024 * 1024;

pub(super) struct Data {
    pub cfg: Arc<Value>,
    pub models: Vec<Value>,
    pub providers: Vec<Value>,
    pub detected: Vec<Value>,
    pub revision: String,
}
pub(in crate::server) struct Store {
    pub source: SettingsSource,
    inner: Mutex<Data>,
    read_only: bool,
}
pub(super) enum Mutation {
    Add(Value),
    Batch(Value),
    Update {
        key: String,
        body: Value,
        legacy: bool,
    },
    Delete {
        key: String,
        revision: Option<String>,
        legacy: bool,
    },
    Provider {
        key: String,
        body: Option<Value>,
        revision: String,
    },
}
impl Store {
    pub fn new(base: Value, mut source: SettingsSource) -> ApiResult<Self> {
        if !source.path.is_absolute() {
            source.path = std::env::current_dir()
                .map_err(|_| failure())?
                .join(&source.path);
        }
        let raw = read(&source.path)?;
        let raw = parse(raw.as_deref())?;
        let models = list(&raw, "model_list")
            .unwrap_or_else(|| list(&base, "model_list").unwrap_or_default());
        let providers =
            list(&raw, "providers").unwrap_or_else(|| list(&base, "providers").unwrap_or_default());
        let read_only = source.overrides.as_ref().is_some_and(|v| {
            v.pointer("/llm/model_list").is_some() || v.pointer("/llm/providers").is_some()
        });
        let detected = if models.is_empty() {
            if let Ok(model) = std::env::var("MODEL")
                && !model.is_empty()
            {
                vec![json!({"model_name":"default","litellm_params":{"model":model}})]
            } else {
                markitai_core::provider_management::detected()
                    .into_iter()
                    .filter_map(|entry| {
                        entry["model"].as_str().map(
                            |name| json!({"model_name":"default","litellm_params":{"model":name}}),
                        )
                    })
                    .collect()
            }
        } else {
            Vec::new()
        };
        let cfg = if read_only {
            base
        } else {
            runtime(&base, &models, &providers, &detected)?
        };
        let revision = identity::revision(&models, &providers);
        Ok(Self {
            source,
            inner: Mutex::new(Data {
                cfg: Arc::new(cfg),
                models,
                providers,
                detected,
                revision,
            }),
            read_only,
        })
    }
    pub fn snapshot(&self) -> Arc<Value> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cfg
            .clone()
    }
    pub fn view(&self) -> Value {
        views::payload(
            &self.inner.lock().unwrap_or_else(|e| e.into_inner()),
            &self.source,
        )
    }
    pub fn providers(&self) -> Value {
        views::providers(&self.inner.lock().unwrap_or_else(|e| e.into_inner()))
    }
    pub fn credentials(&self, id: &str) -> ApiResult<Value> {
        let data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let connection = connection(&data, id, None)?;
        Ok(
            json!({"api_key":connection["api_key"],"api_base":connection["api_base"],"api_base_placeholder":markitai_core::provider_management::provider_default_base(connection["provider"].as_str().unwrap_or_default())}),
        )
    }
    pub fn resolve_discovery(&self, body: &Value) -> ApiResult<Value> {
        model::object(
            body,
            &[
                "provider",
                "provider_id",
                "deployment_id",
                "api_key",
                "api_base",
                "refresh",
            ],
        )?;
        let provider = model::string(body, "provider", true, true)?.unwrap();
        let provider_id = model::string(body, "provider_id", false, false)?;
        let deployment_id = model::string(body, "deployment_id", false, false)?;
        let key = model::client_field(body, "api_key")?;
        let base = model::client_field(body, "api_base")?;
        let refresh = match body.get("refresh") {
            None => false,
            Some(v) => v
                .as_bool()
                .ok_or_else(|| invalid("refresh must be boolean"))?,
        };
        let data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if provider_id.is_some() && deployment_id.is_some() {
            return Err(invalid("send one credential reference"));
        }
        let fallback = if provider_id.is_some() || deployment_id.is_some() {
            connection(
                &data,
                provider_id.as_deref().unwrap_or(""),
                deployment_id.as_deref(),
            )?
        } else {
            json!({})
        };
        let provider = provider.to_ascii_lowercase();
        let supplied_key = body.get("api_key").is_some();
        let resolved_base = if body.get("api_base").is_some() {
            base.map(Value::String).unwrap_or(Value::Null)
        } else {
            fallback["api_base"].clone()
        };
        if let Some(old_provider) = fallback["provider"].as_str() {
            model::check_reuse(
                old_provider,
                &provider,
                fallback["api_base"].as_str(),
                resolved_base.as_str(),
                supplied_key,
            )?;
        }
        let allow_environment = !supplied_key
            && if fallback["provider"].is_string() {
                fallback["use_environment_credentials"].as_bool() != Some(false)
            } else {
                body.get("api_base").is_none()
            };
        let resolved_key = if supplied_key {
            key.map(Value::String).unwrap_or(Value::Null)
        } else {
            fallback["api_key"].clone()
        };
        Ok(
            json!({"provider":provider,"api_key":resolved_key,"api_base":resolved_base,"refresh":refresh,"use_environment_credentials":allow_environment || model::local(&provider)}),
        )
    }

    pub fn resolve_probe(&self, body: &Value) -> ApiResult<Value> {
        model::object(
            body,
            &[
                "deployment_id",
                "model_name",
                "model",
                "api_key",
                "api_base",
            ],
        )?;
        let id = model::string(body, "deployment_id", false, true)?;
        let group = model::string(body, "model_name", false, true)?;
        let name = model::string(body, "model", false, true)?;
        let key = model::client_field(body, "api_key")?;
        let base = model::client_field(body, "api_base")?;
        if [&id, &group, &name]
            .iter()
            .any(|v| v.as_deref() == Some(""))
        {
            return Err(invalid("model references cannot be blank"));
        }
        let data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if id.is_some() || group.is_some() {
            if id.is_some() && group.is_some() || name.is_some() || key.is_some() || base.is_some()
            {
                return Err(invalid(
                    "stored reference cannot be combined with other probe fields",
                ));
            }
            let entry = if let Some(id) = id {
                data.models
                    .iter()
                    .enumerate()
                    .chain(data.detected.iter().enumerate())
                    .find(|(index, entry)| identity::id(entry, *index) == id)
                    .map(|(_, entry)| entry)
                    .ok_or_else(|| invalid("deployment not found"))?
            } else {
                let name = group.unwrap();
                let index = model::legacy(&data.models, &name).map_err(|error| {
                    if error.status.as_u16() == 404 {
                        invalid("model_name not found")
                    } else {
                        error
                    }
                })?;
                &data.models[index]
            };
            return Ok(probe_params(entry, &data.providers));
        }
        let name = name.ok_or_else(|| invalid("deployment_id, model_name or model is required"))?;
        let provider = name
            .split_once('/')
            .map_or("openai", |(provider, _)| provider);
        let supplied_key = body.get("api_key").is_some();
        let mut result = json!({"model":name,"api_key":key,"api_base":base,"use_environment_credentials":!supplied_key && body.get("api_base").is_none() || model::local(provider)});
        if !supplied_key && !model::local(provider) {
            let candidates: Vec<Value> = data
                .models
                .iter()
                .chain(&data.detected)
                .filter(|entry| entry["litellm_params"]["model"] == name)
                .map(|entry| probe_params(entry, &data.providers))
                .collect();
            let selected = if body.get("api_base").is_some() {
                let selected = candidates.iter().find(|params| {
                    model::same_base(provider, params["api_base"].as_str(), base.as_deref())
                });
                if selected.is_none() && !candidates.is_empty() {
                    return Err(invalid(
                        "saved model credentials cannot be used with another endpoint",
                    ));
                }
                selected
            } else {
                candidates.first()
            };
            if let Some(params) = selected {
                result["api_key"] = params["api_key"].clone();
                result["use_environment_credentials"] =
                    params["use_environment_credentials"].clone();
                // Send the original configured endpoint, never a request-controlled alias.
                result["api_base"] = params["api_base"].clone();
            }
        }
        Ok(result)
    }
    pub fn config_path(&self) -> ApiResult<std::path::PathBuf> {
        if read(&self.source.path)?.is_none() {
            return Err(ApiError::new(
                404,
                "config_missing",
                "config file does not exist yet; save a model to create it",
            ));
        }
        Ok(self.source.path.clone())
    }
    pub(super) fn mutate(&self, command: Mutation) -> ApiResult<Value> {
        if self.read_only {
            return Err(ApiError::new(
                409,
                "settings_read_only",
                "Session model/provider overrides are active; edit the selected configuration file and restart without those overrides",
            ));
        }
        let mut data = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let original = read(&self.source.path)?;
        let mut raw = parse(original.as_deref())?;
        let mut models = list(&raw, "model_list").unwrap_or_else(|| data.models.clone());
        let mut providers = list(&raw, "providers").unwrap_or_else(|| data.providers.clone());
        let current = identity::revision(&models, &providers);
        let (expected, backfill) = match &command {
            Mutation::Add(v) => (model::string(v, "expected_revision", false, false)?, false),
            Mutation::Batch(v) => (
                Some(model::string(v, "expected_revision", true, true)?.unwrap()),
                true,
            ),
            Mutation::Update { body, legacy, .. } => (
                model::string(body, "expected_revision", !*legacy, !*legacy)?,
                !*legacy,
            ),
            Mutation::Delete {
                revision, legacy, ..
            } => (revision.clone(), !*legacy),
            Mutation::Provider { revision, .. } => (Some(revision.clone()), true),
        };
        if let Some(expected) = expected
            && expected != current
        {
            return Err(ApiError::structured(
                409,
                "stale_revision",
                json!({"code":"stale_revision","current_revision":current}),
            ));
        }
        let credentials = models.clone();
        let mapping = if backfill {
            identity::backfill(&mut models)
        } else {
            HashMap::new()
        };
        match command {
            Mutation::Add(body) => model::create(&body, &mut models, &mut providers, &credentials)?,
            Mutation::Batch(body) => {
                model::object(&body, &["expected_revision", "deployments"])?;
                let entries = body["deployments"]
                    .as_array()
                    .filter(|v| !v.is_empty() && v.len() <= 50)
                    .ok_or_else(|| invalid("deployments must contain 1 to 50 entries"))?;
                for entry in entries {
                    model::create(entry, &mut models, &mut providers, &credentials)?;
                }
            }
            Mutation::Update { key, body, legacy } => {
                let index = if legacy {
                    model::legacy(&models, &key)?
                } else {
                    model::find(&models, &key, &mapping)?
                };
                model::update(&mut models[index], &body, &providers)?;
            }
            Mutation::Delete { key, legacy, .. } => {
                let index = if legacy {
                    model::legacy(&models, &key)?
                } else {
                    model::find(&models, &key, &mapping)?
                };
                model::remove(&mut models, &mut providers, index);
            }
            Mutation::Provider { key, body, .. } => {
                model::mutate_provider(&mut models, &mut providers, &key, &mapping, body.as_ref())?
            }
        }
        let next = runtime(&data.cfg, &models, &providers, &data.detected)?;
        if !raw["llm"].is_object() {
            raw["llm"] = json!({});
        }
        let include_providers = !providers.is_empty()
            || raw["llm"].get("providers").is_some()
            || !data.providers.is_empty();
        raw["llm"]["model_list"] = json!(models);
        if include_providers {
            raw["llm"]["providers"] = json!(providers);
        }
        let durable = save(&self.source.path, &raw, original.as_deref())?;
        data.cfg = Arc::new(next);
        data.revision = identity::revision(&models, &providers);
        data.models = models;
        data.providers = providers;
        if !durable {
            return Err(ApiError::new(
                500,
                "settings_durability_unknown",
                "Configuration was written and activated, but directory durability could not be confirmed; reload settings before another change",
            ));
        }
        Ok(views::payload(&data, &self.source))
    }
}
pub(in crate::server) fn load_base(source: &SettingsSource) -> ApiResult<Value> {
    let bytes = read(&source.path)?;
    let mut value = parse(bytes.as_deref())?;
    if let Some(overrides) = source.overrides.clone() {
        markitai_core::config::merge(&mut value, overrides);
    }
    markitai_core::config::normalize(&value).map_err(|_| invalid("invalid service configuration"))
}
fn list(value: &Value, key: &str) -> Option<Vec<Value>> {
    value.get("llm")?.get(key)?.as_array().cloned()
}
fn runtime(
    base: &Value,
    models: &[Value],
    providers: &[Value],
    detected: &[Value],
) -> ApiResult<Value> {
    let mut cfg = base.clone();
    let mut seen = std::collections::HashSet::new();
    let mut effective = Vec::new();
    for entry in models.iter().chain(detected) {
        let key = (
            entry["litellm_params"]["model"].clone().to_string(),
            entry["litellm_params"]["api_base"].clone().to_string(),
            entry["model_name"].clone().to_string(),
        );
        if seen.insert(key) {
            effective.push(entry.clone());
        }
    }
    cfg["llm"]["model_list"] = json!(effective);
    cfg["llm"]["providers"] = json!(providers);
    markitai_core::config::normalize(&cfg)
        .map_err(|_| invalid("invalid model/provider configuration"))
}
fn connection(data: &Data, id: &str, deployment: Option<&str>) -> ApiResult<Value> {
    if id.starts_with("env:") {
        return model::environment_connection(id);
    }
    if let Some(deployment) = deployment {
        model::find(&data.models, deployment, &HashMap::new())?;
    }
    if let Some(saved) = data.providers.iter().find(|p| p["id"] == id) {
        return Ok(saved.clone());
    }
    let id = if let Some(id) = id.strip_prefix("legacy:") {
        id
    } else if let Some(id) = deployment {
        id
    } else {
        return Err(missing());
    };
    let entry = &data.models[model::find(&data.models, id, &HashMap::new())?];
    let params = probe_params(entry, &data.providers);
    Ok(
        json!({"provider":model::provider(entry),"api_key":params["api_key"],"api_base":params["api_base"],"use_environment_credentials":params["use_environment_credentials"]}),
    )
}
fn probe_params(entry: &Value, providers: &[Value]) -> Value {
    let params = model::params(entry, providers);
    json!({"model":params["model"],"api_key":params["api_key"],"api_base":params["api_base"],"use_environment_credentials":params["use_environment_credentials"].as_bool() != Some(false)})
}

fn parse(bytes: Option<&[u8]>) -> ApiResult<Value> {
    let value: Value = if let Some(bytes) = bytes {
        serde_json::from_slice(bytes).map_err(|_| failure())?
    } else {
        json!({})
    };
    if !value.is_object() {
        return Err(failure());
    }
    Ok(value)
}
fn read(path: &Path) -> ApiResult<Option<Vec<u8>>> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Ok(m) if m.is_file() && !m.file_type().is_symlink() && m.len() <= LIMIT as u64 => {}
        _ => return Err(failure()),
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(|_| failure())?;
    if !file.metadata().map_err(|_| failure())?.is_file() {
        return Err(failure());
    }
    let mut bytes = Vec::new();
    file.take(LIMIT as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| failure())?;
    if bytes.len() > LIMIT {
        return Err(failure());
    }
    Ok(Some(bytes))
}
fn save(path: &Path, value: &Value, original: Option<&[u8]>) -> ApiResult<bool> {
    save_with_sync(path, value, original, |_parent| {
        #[cfg(unix)]
        fs::File::open(_parent)?.sync_all()?;
        Ok(())
    })
}
fn save_with_sync(
    path: &Path,
    value: &Value,
    original: Option<&[u8]>,
    sync: impl FnOnce(&Path) -> std::io::Result<()>,
) -> ApiResult<bool> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|_| failure())?;
    bytes.push(b'\n');
    if bytes.len() > LIMIT {
        return Err(ApiError::new(
            422,
            "config_too_large",
            "configuration exceeds 8 MiB limit",
        ));
    }
    markitai_core::output::check_path(path, false).map_err(|_| failure())?;
    let parent = path.parent().ok_or_else(failure)?;
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(parent).map_err(|_| failure())?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|_| failure())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|_| failure())?;
    }
    file.write_all(&bytes)
        .and_then(|()| file.as_file().sync_all())
        .map_err(|_| failure())?;
    if read(path)?.as_deref() != original {
        return Err(ApiError::new(
            409,
            "config_changed",
            "Configuration changed during save; reload settings",
        ));
    }
    file.persist(path).map_err(|_| failure())?;
    // The rename has committed. A later sync failure cannot truthfully roll back memory.
    Ok(sync(parent).is_ok())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
