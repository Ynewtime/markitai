//! Configuration normalization is independent of converter availability.
//! Embedded metadata contains types, bounds and defaults, without documentation prose.

use crate::{Error, Result};
use serde_json::{Map, Number, Value, json};
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// An empty variable counts as unset: as a path it would name the current
/// directory, and every path starts with it.
fn variable(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

fn user_home() -> Option<std::ffi::OsString> {
    variable("HOME").or_else(|| variable("USERPROFILE"))
}

/// A relative `MARKITAI_HOME` is resolved against the working directory the
/// process had when the home was first needed, so every later use names the
/// same absolute folder.
pub fn home() -> PathBuf {
    static START: OnceLock<Option<PathBuf>> = OnceLock::new();
    let Some(path) = variable("MARKITAI_HOME").map(|path| expand_home(Path::new(&path))) else {
        return user_home()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".markitai");
    };
    match START.get_or_init(|| std::env::current_dir().ok()) {
        Some(start) if path.is_relative() => start.join(path),
        _ => path,
    }
}

pub fn expand_home(path: &Path) -> PathBuf {
    if let Ok(rest) = path.strip_prefix("~")
        && let Some(user) = user_home()
    {
        PathBuf::from(user).join(rest)
    } else {
        path.to_owned()
    }
}

/// Resolve legacy default state paths into the explicitly isolated Rust home.
/// Custom paths retain their meaning; the serialized default remains compatible.
pub fn state_path(path: &Path) -> PathBuf {
    if variable("MARKITAI_HOME").is_some()
        && let Ok(rest) = path.strip_prefix("~/.markitai")
    {
        home().join(rest)
    } else {
        expand_home(path)
    }
}

pub fn schema() -> &'static Value {
    static SCHEMA: OnceLock<Value> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        serde_json::from_str(include_str!("config_contract.json"))
            .expect("embedded configuration contract is valid JSON")
    })
}

pub fn defaults() -> Value {
    static DEFAULTS: OnceLock<Value> = OnceLock::new();
    DEFAULTS
        .get_or_init(|| normalize(&json!({})).expect("embedded defaults satisfy their contract"))
        .clone()
}

/// Recursively merge objects; arrays and scalar values replace the previous value.
pub fn merge(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Object(base), Value::Object(overlay)) => {
            for (key, value) in overlay {
                merge(base.entry(key).or_insert(Value::Null), value);
            }
        }
        (base, value) => *base = value,
    }
}

fn choose_path(
    explicit: Option<&Path>,
    env_path: Option<&Path>,
    cwd: &Path,
    user_home: &Path,
) -> Option<PathBuf> {
    explicit
        .map(PathBuf::from)
        .or_else(|| env_path.map(PathBuf::from))
        .or_else(|| {
            let local = cwd.join("markitai.json");
            local.exists().then_some(local)
        })
        .or_else(|| {
            let global = user_home.join("config.json");
            global.exists().then_some(global)
        })
}

/// The selected file, including an explicitly requested path that is missing.
pub fn selected_path(explicit: Option<&Path>) -> Option<PathBuf> {
    selected_source(explicit).map(|(path, _)| path)
}

/// The selected file, and whether it is `./markitai.json` found in the
/// working directory rather than named (`-c`, `MARKITAI_CONFIG`) or the
/// user's own `config.json`: a folder may come with one its user did not write.
pub fn selected_source(explicit: Option<&Path>) -> Option<(PathBuf, bool)> {
    let environment = environment();
    let env_path = environment
        .get("MARKITAI_CONFIG")
        .filter(|s| !s.is_empty())
        .map(Path::new);
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let project = explicit.is_none() && env_path.is_none();
    choose_path(explicit, env_path, &cwd, &home()).map(|path| {
        let found_here = project && path == cwd.join("markitai.json");
        (expand_home(&path), found_here)
    })
}

/// An opt-in environment switch such as `MARKITAI_PURE`: `1`, `true`, `yes`
/// or `on`, trimmed and in any case, turn it on; any other value turns it off.
/// `None` when the value is blank.
pub fn env_opt_in(value: &str) -> Option<bool> {
    let value = value.trim().to_ascii_lowercase();
    (!value.is_empty()).then_some(matches!(value.as_str(), "1" | "true" | "yes" | "on"))
}

/// An opt-out environment switch such as `MARKITAI_NO_REMOTE_FETCH`: set by
/// any value except a blank one, `0`, `false` or `no` (trimmed, any case), as
/// the reference reads it, so an unrecognized value fails safe.
pub fn env_opt_out(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no"
    )
}

/// The built-in presets, in the order they are offered.
pub const BUILTIN_PRESETS: [&str; 3] = ["minimal", "standard", "rich"];

/// The feature switches of preset `name` (case-insensitive): an entry of the
/// configuration's `presets`, which may redefine a built-in name, else the
/// built-in `minimal`, `standard` or `rich`. None of them enables OCR.
pub fn preset(cfg: &Value, name: &str) -> Option<Value> {
    let name = name.to_lowercase();
    cfg["presets"].get(&name).cloned().or_else(|| {
        let (llm, screenshot) = match name.as_str() {
            "minimal" => (false, false),
            "standard" => (true, false),
            "rich" => (true, true),
            _ => return None,
        };
        Some(json!({"llm":llm,"ocr":false,"alt":llm,"desc":llm,"screenshot":screenshot}))
    })
}

/// Largest configuration file that is read or written.
pub const FILE_LIMIT: usize = 8 * 1024 * 1024;

/// Read the configuration file at `path`; `None` when nothing is there. A
/// final symbolic link (a dotfile-managed configuration) is followed. The file
/// must be regular (`InvalidInput` otherwise) and at most [`FILE_LIMIT`] bytes
/// (`FileTooLarge`); opening never blocks on a FIFO.
pub fn read_file(path: &Path) -> std::io::Result<Option<Vec<u8>>> {
    use std::io::ErrorKind;
    let regular = |metadata: std::fs::Metadata| {
        if metadata.is_file() {
            Ok(())
        } else {
            Err(std::io::Error::new(
                ErrorKind::InvalidInput,
                "configuration must be a regular file",
            ))
        }
    };
    match std::fs::metadata(path) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        metadata => regular(metadata?)?,
    }
    let file = match crate::platform::open_read(path, true) {
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        file => file?,
    };
    // The name may have changed since the check above.
    regular(file.metadata()?)?;
    match crate::platform::read_limited(file, FILE_LIMIT as u64)? {
        Some(bytes) => Ok(Some(bytes)),
        None => Err(std::io::Error::new(
            ErrorKind::FileTooLarge,
            "configuration exceeds 8 MiB",
        )),
    }
}

/// A replacement configuration, staged beside the file it replaces.
pub struct StagedFile {
    temporary: tempfile::NamedTempFile,
    destination: PathBuf,
}

/// Stage `bytes` to replace the configuration at `path`. A final symbolic link
/// is kept and the file it names is replaced, as [`read_file`] reads through
/// it. An existing destination must be a regular file and `bytes` at most
/// [`FILE_LIMIT`]. A missing parent directory is created private; the staged
/// file is owner-only.
pub fn stage_file(path: &Path, bytes: &[u8]) -> std::io::Result<StagedFile> {
    use std::io::{ErrorKind, Write};
    if bytes.len() > FILE_LIMIT {
        return Err(std::io::Error::new(
            ErrorKind::FileTooLarge,
            "configuration exceeds 8 MiB",
        ));
    }
    let destination = match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_symlink() => crate::platform::canonicalize(path)?,
        Err(error) if error.kind() != ErrorKind::NotFound => return Err(error),
        _ => path.to_owned(),
    };
    match std::fs::symlink_metadata(&destination) {
        Ok(metadata) if !metadata.is_file() => {
            return Err(std::io::Error::new(
                ErrorKind::InvalidInput,
                "configuration must be a regular file",
            ));
        }
        Err(error) if error.kind() != ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    crate::platform::private_directory()
        .recursive(true)
        .create(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file().sync_all()?;
    Ok(StagedFile {
        temporary,
        destination,
    })
}

impl StagedFile {
    /// The directory whose entry [`StagedFile::publish`] replaces.
    pub fn directory(&self) -> &Path {
        self.temporary.path().parent().unwrap_or(Path::new("."))
    }

    /// Atomically replace the destination. Durability needs a later
    /// [`crate::platform::sync_directory`] of [`StagedFile::directory`].
    pub fn publish(self) -> std::io::Result<()> {
        crate::platform::persist(self.temporary, &self.destination)
            .map(drop)
            .map_err(|error| error.error)
    }
}

/// Durably replace the configuration at `path` with `bytes` ([`stage_file`]).
pub fn write_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let staged = stage_file(path, bytes)?;
    let directory = staged.directory().to_owned();
    staged.publish()?;
    crate::platform::sync_directory(&directory)
}

pub fn load(explicit: Option<&Path>, overrides: Option<Value>) -> Result<Value> {
    normalize(&raw(explicit, overrides)?)
}

/// The selected file with `overrides` merged over it, before defaults are
/// filled in. A caller that must tell a value the user wrote from the same
/// value as a default (such as `fetch.remote_consent`) reads it here and
/// normalizes the result itself; [`load`] is `normalize(&raw(..))`.
pub fn raw(explicit: Option<&Path>, overrides: Option<Value>) -> Result<Value> {
    let selected = selected_source(explicit);
    let mut value = json!({});
    if let Some((path, project)) = selected {
        match read_file(&path) {
            Ok(Some(bytes)) => {
                value = serde_json::from_slice(&bytes).map_err(|error| {
                    Error::Config(format!("Invalid JSON in {}: {error}", path.display()))
                })?;
                if !value.is_object() {
                    return Err(invalid("", "must be a JSON object"));
                }
                if project {
                    for note in seal_project_models(&mut value) {
                        warn_once(note);
                    }
                }
            }
            Ok(None) => {
                // The CLI separately rejects a missing explicit -c. The library
                // loader historically warns and falls back without selecting another file.
                eprintln!(
                    "Warning: Config file not found: {} (using defaults)",
                    path.display()
                );
            }
            Err(error) => {
                return Err(Error::Config(format!(
                    "Cannot read config {}: {error}",
                    path.display()
                )));
            }
        }
    }
    if let Some(overrides) = overrides {
        if !overrides.is_object() {
            return Err(invalid("", "overrides must be a JSON object"));
        }
        merge(&mut value, overrides);
    }
    Ok(value)
}

/// Return the effective configuration, including defaults for nested model entries.
/// Unknown model fields are ignored, matching the reference model's extra-field rule.
/// Valid configuration is not a promise that its requested runtime feature is implemented.
pub fn normalize(value: &Value) -> Result<Value> {
    let value = normalize_node(value, schema(), "")?;
    validate_fetch(&value["fetch"])?;
    Ok(value)
}

/// Normalize one public configuration model for native language adapters.
pub fn normalize_model(model: &str, value: &Value) -> Result<Value> {
    if model == "MarkitaiConfig" {
        return normalize(value);
    }
    let node = schema()["$defs"]
        .get(model)
        .filter(|node| node.get("properties").is_some())
        .ok_or_else(|| invalid(model, "is not a configuration model"))?;
    let value = normalize_node(value, node, model)?;
    match model {
        "FetchConfig" => validate_fetch(&value)?,
        "FetchPolicyConfig" => validate_policy(&value)?,
        "DomainProfileConfig" => {
            validate_strategy_list(value.get("strategy_priority"), "strategy_priority")?
        }
        _ => {}
    }
    Ok(value)
}

fn validate_fetch(value: &Value) -> Result<()> {
    validate_policy(&value["policy"])?;
    if let Some(profiles) = value.get("domain_profiles").and_then(Value::as_object) {
        for (domain, profile) in profiles {
            validate_strategy_list(
                profile.get("strategy_priority"),
                &format!("fetch.domain_profiles.{domain}.strategy_priority"),
            )?;
        }
    }
    Ok(())
}

fn validate_policy(value: &Value) -> Result<()> {
    validate_strategy_list(
        value.get("strategy_priority"),
        "fetch.policy.strategy_priority",
    )?;
    if let Some(patterns) = value.get("local_only_patterns").and_then(Value::as_array) {
        for (index, pattern) in patterns.iter().enumerate() {
            let pattern = pattern.as_str().expect("normalized string array").trim();
            if pattern.is_empty() {
                return Err(invalid(
                    &format!("fetch.policy.local_only_patterns[{index}]"),
                    "must not be empty",
                ));
            }
            if pattern.contains('/') && !valid_cidr(pattern) {
                return Err(invalid(
                    &format!("fetch.policy.local_only_patterns[{index}]"),
                    "must contain a valid IPv4 or IPv6 CIDR",
                ));
            }
        }
    }
    Ok(())
}

pub fn validate(value: &Value) -> Result<()> {
    normalize(value).map(|_| ())
}

/// An accepted configuration key that currently has no runtime effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigWarning {
    pub key: &'static str,
    pub reason: &'static str,
}

/// Inspect raw user configuration, before defaults are filled in. This is
/// advisory; callers validate separately and decide where to display warnings.
pub fn warnings(value: &Value) -> Vec<ConfigWarning> {
    [
        (
            "/batch/heavy_task_limit",
            "batch.heavy_task_limit",
            "has no effect; use batch.concurrency to limit file conversions",
        ),
        (
            "/office/macos_fallback",
            "office.macos_fallback",
            "has no effect; Office conversion does not automate macOS applications",
        ),
        (
            "/image/stdout_fetch_external",
            "image.stdout_fetch_external",
            "has no effect; terminal inline-image display is unavailable",
        ),
    ]
    .into_iter()
    .filter(|(pointer, _, _)| value.pointer(pointer).is_some())
    .map(|(_, key, reason)| ConfigWarning { key, reason })
    .collect()
}

const REDACTED: &str = "[REDACTED]";

fn normalized_key(key: &str) -> String {
    let chars: Vec<_> = key.chars().collect();
    let mut result = String::new();
    for (index, ch) in chars.iter().copied().enumerate() {
        if !ch.is_ascii_alphanumeric() {
            if !result.is_empty() && !result.ends_with('_') {
                result.push('_');
            }
            continue;
        }
        if ch.is_ascii_uppercase() && index > 0 {
            let previous = chars[index - 1];
            let boundary = previous.is_ascii_lowercase()
                || previous.is_ascii_digit()
                || (previous.is_ascii_uppercase()
                    && chars.get(index + 1).is_some_and(char::is_ascii_lowercase));
            if boundary && !result.ends_with('_') {
                result.push('_');
            }
        }
        result.push(ch.to_ascii_lowercase());
    }
    result.trim_end_matches('_').to_owned()
}

fn sensitive_key(key: &str) -> bool {
    let normalized = normalized_key(key);
    let parts: Vec<_> = normalized.split('_').collect();
    normalized == "apikey"
        || normalized.ends_with("_apikey")
        || (parts.contains(&"api") && parts.contains(&"key"))
        || parts.iter().any(|part| {
            matches!(
                *part,
                "token"
                    | "authorization"
                    | "cookie"
                    | "cookies"
                    | "credential"
                    | "credentials"
                    | "passwd"
                    | "password"
                    | "secret"
            )
        })
}

/// Whether a configuration key names a secret: the rule [`redact`] (and
/// `markitai config show`) hides values by.
pub fn secret_key(key: &str) -> bool {
    sensitive_key(key)
}

/// Whether a configuration key names an `extra_http_headers` object, whose
/// values [`redact`] hides whatever their header names.
pub fn header_map_key(key: &str) -> bool {
    let key = normalized_key(key);
    key == "extra_http_headers" || key.ends_with("_extra_http_headers")
}

/// Whether a configuration key names an endpoint address (`api_base`,
/// `*_api_base`), which [`redact`] shows without its credentials.
pub fn api_base_key(key: &str) -> bool {
    let key = normalized_key(key);
    key == "api_base" || key.ends_with("_api_base")
}

fn redact_named(key: &str, value: &Value) -> Option<Value> {
    if header_map_key(key) {
        return Some(match value.as_object() {
            Some(headers) => object(headers.keys().map(|key| (key.clone(), json!(REDACTED)))),
            None => json!(REDACTED),
        });
    }
    let environment_reference = value
        .as_str()
        .is_some_and(|value| value.starts_with("env:"));
    if api_base_key(key) {
        if environment_reference {
            return Some(value.clone());
        }
        let origin = value.as_str().and_then(|raw| {
            let url = url::Url::parse(raw).ok()?;
            let host = url.host_str()?;
            // URL normalization drops an explicit default port; retain it in the
            // displayed origin without retaining any user information or path.
            let authority = raw.split_once("://")?.1.split(['/', '?', '#']).next()?;
            let authority = authority.rsplit('@').next()?;
            let explicit_port = if authority.starts_with('[') {
                authority.split_once(']')?.1.strip_prefix(':')
            } else {
                authority.rsplit_once(':').map(|(_, port)| port)
            }
            .and_then(|port| port.parse::<u16>().ok());
            let port = url.port().or(explicit_port);
            Some(match port {
                Some(port) => format!("{}://{host}:{port}", url.scheme()),
                None => format!("{}://{host}", url.scheme()),
            })
        });
        return Some(json!(origin.as_deref().unwrap_or(REDACTED)));
    }
    sensitive_key(key).then(|| {
        if environment_reference {
            value.clone()
        } else {
            json!(REDACTED)
        }
    })
}

/// Prepare configuration for display without resolving environment references.
/// Header values are always inline, while API base URLs expose only their origin.
pub fn redact(value: &Value) -> Value {
    match value {
        Value::Object(values) => object(values.iter().map(|(key, value)| {
            (
                key.clone(),
                redact_named(key, value).unwrap_or_else(|| redact(value)),
            )
        })),
        Value::Array(values) => Value::Array(values.iter().map(redact).collect()),
        value => value.clone(),
    }
}

/// Select a value before redacting it so children of sensitive containers remain
/// addressable without accidentally exposing them.
pub fn redact_for_key(key: &str, value: &Value) -> Value {
    let parts: Vec<_> = key.split('.').collect();
    if parts
        .iter()
        .take(parts.len().saturating_sub(1))
        .any(|part| normalized_key(part) == "extra_http_headers" || sensitive_key(part))
    {
        return json!(REDACTED);
    }
    redact_named(key, value).unwrap_or_else(|| redact(value))
}

/// Display models with unset nullable fields omitted, while retaining explicit
/// null values in arbitrary dictionaries and direct reads of nullable fields.
pub fn display_value(config: &Value, key: Option<&str>) -> Result<Value> {
    let (value, node) = if let Some(key) = key {
        let pointer = key_pointer(key)?;
        let value = config
            .pointer(&pointer)
            .ok_or_else(|| invalid(key, "does not exist"))?;
        let parts = pointer_parts(&pointer);
        (value, schema_at(schema(), config, &parts, key)?)
    } else {
        (config, schema())
    };
    Ok(omit_model_nulls(value, node))
}

fn omit_model_nulls(value: &Value, node: &Value) -> Value {
    let node = dereference(node);
    if let Some(alternatives) = node.get("anyOf").and_then(Value::as_array)
        && let Some(node) = alternatives
            .iter()
            .find(|node| node.get("type").and_then(Value::as_str) != Some("null"))
    {
        return omit_model_nulls(value, node);
    }
    match value {
        Value::Object(values) => {
            let properties = node.get("properties").and_then(Value::as_object);
            object(values.iter().filter_map(|(key, value)| {
                if properties.is_some() && value.is_null() {
                    return None;
                }
                let child = properties
                    .and_then(|properties| properties.get(key))
                    .or_else(|| node.get("additionalProperties"))
                    .unwrap_or(&Value::Null);
                Some((key.clone(), omit_model_nulls(value, child)))
            }))
        }
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| omit_model_nulls(value, node.get("items").unwrap_or(&Value::Null)))
                .collect(),
        ),
        value => value.clone(),
    }
}

/// `Value::Object(entries.collect())` without the sort that collecting runs
/// first, compiled again for every iterator type. The keys come from one map,
/// so they are distinct and inserting them in turn builds the same map.
fn object(entries: impl Iterator<Item = (String, Value)>) -> Value {
    let mut map = Map::new();
    map.extend(entries);
    Value::Object(map)
}

fn invalid(path: &str, reason: &str) -> Error {
    Error::Config(format!(
        "{} {reason}",
        if path.is_empty() {
            "Configuration"
        } else {
            path
        }
    ))
}
fn child_path(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.into()
    } else {
        format!("{path}.{key}")
    }
}
fn dereference(node: &Value) -> &Value {
    if let Some(reference) = node.get("$ref").and_then(Value::as_str) {
        schema()
            .pointer(reference.strip_prefix('#').expect("local schema reference"))
            .expect("defined schema reference")
    } else {
        node
    }
}

fn normalize_node(value: &Value, node: &Value, path: &str) -> Result<Value> {
    let node = dereference(node);
    if let Some(alternatives) = node.get("anyOf").and_then(Value::as_array) {
        let mut first_error = None;
        for alternative in alternatives {
            match normalize_node(value, alternative, path) {
                Ok(value) => return Ok(value),
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }
        return Err(first_error.unwrap_or_else(|| invalid(path, "does not match an allowed type")));
    }
    if let Some(choices) = node.get("enum").and_then(Value::as_array)
        && !choices.contains(value)
    {
        let options = choices
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string())
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(invalid(path, &format!("must be one of: {options}")));
    }
    let normalized = match node.get("type").and_then(Value::as_str) {
        Some("null") if value.is_null() => Value::Null,
        Some("null") => return Err(invalid(path, "must be null")),
        Some("string") if value.is_string() => value.clone(),
        Some("string") => return Err(invalid(path, "must be a string")),
        Some("boolean") => {
            Value::Bool(parse_bool(value).ok_or_else(|| invalid(path, "must be a boolean"))?)
        }
        Some("integer") => Value::Number(
            parse_integer(value)
                .ok_or_else(|| invalid(path, "must be an integer representable in 64 bits"))?,
        ),
        Some("number") => {
            let number = parse_number(value)
                .and_then(Number::from_f64)
                .ok_or_else(|| invalid(path, "must be a finite number"))?;
            Value::Number(number)
        }
        Some("array") => {
            let values = value
                .as_array()
                .ok_or_else(|| invalid(path, "must be an array"))?;
            let items = node.get("items").unwrap_or(&Value::Null);
            Value::Array(
                values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| normalize_node(value, items, &format!("{path}[{index}]")))
                    .collect::<Result<_>>()?,
            )
        }
        Some("object") => {
            let values = value
                .as_object()
                .ok_or_else(|| invalid(path, "must be an object"))?;
            let mut result = Map::new();
            if let Some(properties) = node.get("properties").and_then(Value::as_object) {
                for (key, field) in properties {
                    let path = child_path(path, key);
                    let value = values
                        .get(key)
                        .or_else(|| field.get("default"))
                        .ok_or_else(|| invalid(&path, "is required"))?;
                    result.insert(key.clone(), normalize_node(value, field, &path)?);
                }
                // Internal: a service hides where this model is served from
                // the failures it reports (see `llm::Deployment`). Like any
                // other unknown model field, anything but a boolean is dropped.
                if path.ends_with(".model_info")
                    && let Some(value) = values.get("hidden_endpoint").filter(|v| v.is_boolean())
                {
                    result.insert("hidden_endpoint".into(), value.clone());
                }
                // Internal: a service's user model reaches public addresses
                // only (see `public_net`); dropped unless a boolean.
                if path.ends_with(".model_info")
                    && let Some(value) = values.get("public_only").filter(|v| v.is_boolean())
                {
                    result.insert("public_only".into(), value.clone());
                }
                if path == "output" {
                    // Internal write targets are never emitted as public defaults.
                    for key in ["filename", "reserved_stem"] {
                        if let Some(value) = values.get(key) {
                            if !value.is_string() {
                                return Err(invalid(&child_path(path, key), "must be a string"));
                            }
                            result.insert(key.into(), value.clone());
                        }
                    }
                }
            } else if let Some(additional) =
                node.get("additionalProperties").filter(|n| n.is_object())
            {
                for (key, value) in values {
                    result.insert(
                        key.clone(),
                        normalize_node(value, additional, &child_path(path, key))?,
                    );
                }
            } else {
                result = values.clone();
            }
            Value::Object(result)
        }
        None => value.clone(),
        Some(_) => return Err(invalid(path, "uses an unsupported embedded schema type")),
    };
    if let Some(number) = normalized.as_f64() {
        if let Some(minimum) = node.get("minimum").and_then(Value::as_f64)
            && number < minimum
        {
            return Err(invalid(path, &format!("must be at least {minimum}")));
        }
        if let Some(maximum) = node.get("maximum").and_then(Value::as_f64)
            && number > maximum
        {
            return Err(invalid(path, &format!("must be at most {maximum}")));
        }
    }
    Ok(normalized)
}

fn parse_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::Number(number) => match number.as_f64()? {
            0.0 => Some(false),
            1.0 => Some(true),
            _ => None,
        },
        Value::String(value) => match value.to_ascii_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
            _ => None,
        },
        _ => None,
    }
}
fn remove_numeric_underscores(value: &str) -> Option<String> {
    let chars = value.as_bytes();
    for (index, byte) in chars.iter().enumerate() {
        if *byte == b'_'
            && (index == 0
                || index + 1 == chars.len()
                || !chars[index - 1].is_ascii_digit()
                || !chars[index + 1].is_ascii_digit())
        {
            return None;
        }
    }
    Some(value.replace('_', ""))
}
fn parse_integer(value: &Value) -> Option<Number> {
    match value {
        Value::Bool(value) => Some(Number::from(u8::from(*value))),
        Value::Number(number) if number.is_i64() || number.is_u64() => Some(number.clone()),
        Value::Number(number) => integer_from_float(number.as_f64()?),
        Value::String(value) => {
            let value = value.trim();
            let (whole, fraction) = value
                .split_once('.')
                .map(|(a, b)| (a, Some(b)))
                .unwrap_or((value, None));
            if fraction.is_some_and(|fraction| {
                fraction.is_empty() || !fraction.bytes().all(|byte| byte == b'0')
            }) {
                return None;
            }
            let whole = remove_numeric_underscores(whole)?;
            let digits = whole.strip_prefix(['+', '-']).unwrap_or(&whole);
            if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
            whole
                .parse::<i64>()
                .map(Number::from)
                .or_else(|_| whole.parse::<u64>().map(Number::from))
                .ok()
        }
        _ => None,
    }
}
fn integer_from_float(number: f64) -> Option<Number> {
    if !number.is_finite() || number.fract() != 0.0 {
        return None;
    }
    if number >= i64::MIN as f64 && number < 0.0 {
        Some(Number::from(number as i64))
    } else if number >= 0.0 && number < u64::MAX as f64 {
        Some(Number::from(number as u64))
    } else {
        None
    }
}
fn parse_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64().filter(|v| v.is_finite()),
        Value::Bool(value) => Some(u8::from(*value) as f64),
        Value::String(value) => remove_numeric_underscores(value.trim())?
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite()),
        _ => None,
    }
}

fn validate_strategy_list(value: Option<&Value>, path: &str) -> Result<()> {
    let Some(values) = value.and_then(Value::as_array) else {
        return Ok(());
    };
    if values.is_empty() {
        return Err(invalid(path, "must not be empty if set"));
    }
    let mut seen = HashSet::new();
    for value in values {
        let strategy = value.as_str().expect("normalized strategy string");
        if !["static", "playwright", "defuddle", "jina", "cloudflare"].contains(&strategy) {
            return Err(invalid(path, "contains an invalid fetch strategy"));
        }
        if !seen.insert(strategy) {
            return Err(invalid(path, "contains duplicate strategies"));
        }
    }
    Ok(())
}
fn valid_cidr(value: &str) -> bool {
    let Some((address, prefix)) = value.split_once('/') else {
        return false;
    };
    let address = if let Some((address, scope)) = address.split_once('%') {
        if scope.is_empty() || scope.contains('%') || !address.contains(':') {
            return false;
        }
        address
    } else {
        address
    };
    let numeric_prefix = !prefix.is_empty() && prefix.bytes().all(|byte| byte.is_ascii_digit());
    match address.parse::<IpAddr>() {
        Ok(IpAddr::V4(_)) => {
            if numeric_prefix && let Ok(prefix) = prefix.parse::<u8>() {
                return prefix <= 32;
            }
            let Ok(mask) = prefix.parse::<std::net::Ipv4Addr>() else {
                return false;
            };
            let mask = u32::from(mask);
            mask.leading_ones() + mask.trailing_zeros() == 32
                || mask.leading_zeros() + mask.trailing_ones() == 32
        }
        Ok(IpAddr::V6(_)) => {
            numeric_prefix && prefix.parse::<u8>().is_ok_and(|prefix| prefix <= 128)
        }
        Err(_) => false,
    }
}

/// Resolve an explicit env: reference without mutating the process environment.
pub fn resolve_env_value(
    value: &str,
    vars: &HashMap<String, String>,
    strict: bool,
) -> Result<Option<String>> {
    if let Some(name) = value.strip_prefix("env:") {
        match vars.get(name) {
            Some(value) => Ok(Some(value.clone())),
            None if !strict => Ok(None),
            None => Err(Error::Config(format!(
                "Environment variable not found: {name}"
            ))),
        }
    } else {
        Ok(Some(value.to_owned()))
    }
}

/// A fallback variable is used only when no nonempty explicit value was supplied.
/// A missing explicit env: reference in non-strict mode does not fall back again.
pub fn resolve_optional(
    value: Option<&str>,
    fallback: Option<&str>,
    vars: &HashMap<String, String>,
    strict: bool,
) -> Result<Option<String>> {
    match value.filter(|value| !value.is_empty()) {
        Some(value) => resolve_env_value(value, vars, strict),
        None => Ok(fallback.and_then(|key| vars.get(key)).cloned()),
    }
}

/// Read dotenv as data without changing the host process environment: the
/// working directory's `.env`, then `~/.markitai/.env`; the process
/// environment wins over both. A working directory may come with a `.env`
/// its user did not write, so names that choose programs, paths, endpoints,
/// proxies or credential files are not taken from it (one warning lists them).
pub fn environment() -> HashMap<String, String> {
    let project = PathBuf::from(".env");
    let own = home().join(".env");
    let trusted_here = matches!(
        (project.canonicalize(), own.canonicalize()),
        (Ok(here), Ok(home)) if here == home
    );
    let (vars, ignored) = environment_from(
        std::env::vars(),
        dotenvy::from_path_iter(&project)
            .ok()
            .map(|iter| iter.flatten().collect())
            .unwrap_or_default(),
        trusted_here,
        dotenvy::from_path_iter(&own)
            .ok()
            .map(|iter| iter.flatten().collect())
            .unwrap_or_default(),
    );
    if !ignored.is_empty() {
        warn_once(format!(
            "Warning: Ignored {} from ./.env: a folder's .env may not set programs, paths, endpoints, proxies, home folders or credential files; set them in your environment or in {}",
            ignored.join(", "),
            own.display()
        ));
    }
    vars
}

/// `process` over the working directory's `project` pairs (without the
/// names [`project_may_not_set`] refuses, unless `trusted_here`) over the
/// user's `own` pairs; and the refused names, sorted.
fn environment_from(
    process: impl Iterator<Item = (String, String)>,
    project: Vec<(String, String)>,
    trusted_here: bool,
    own: Vec<(String, String)>,
) -> (HashMap<String, String>, Vec<String>) {
    let mut vars: HashMap<String, String> = process.collect();
    let mut ignored = std::collections::BTreeSet::new();
    for (key, value) in project {
        if !trusted_here && project_may_not_set(&key) {
            ignored.insert(key);
            continue;
        }
        vars.entry(key).or_insert(value);
    }
    for (key, value) in own {
        vars.entry(key).or_insert(value);
    }
    (vars, ignored.into_iter().collect())
}

/// A name a working directory's `.env` may not set: programs and search
/// paths (`*_PATH`, `*_EXECUTABLE`, `PATH`, `LD_*`, `DYLD_*`, interpreter and
/// shell hooks), endpoints (`*_API_BASE`, `*_BASE_URL`, `*_ENDPOINT`),
/// proxies, home, configuration and temporary folders, credential files
/// (`AWS_*_FILE`, `GOOGLE_APPLICATION_CREDENTIALS`) and certificate bundles.
pub fn project_may_not_set(name: &str) -> bool {
    let name = name.trim().to_ascii_uppercase();
    const EXACT: &[&str] = &[
        "PATH",
        "PATHEXT",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "TMPDIR",
        "TMP",
        "TEMP",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "CLOUDSDK_CONFIG",
        "MARKITAI_CONFIG",
        "MARKITAI_HOME",
        "MARKITAI_LOG_DIR",
        "NODE_OPTIONS",
        "BASH_ENV",
        "ENV",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "AWS_CA_BUNDLE",
    ];
    EXACT.contains(&name.as_str())
        || ["LD_", "DYLD_", "XDG_", "PYTHON"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
        || [
            "_PATH",
            "_EXECUTABLE",
            "_API_BASE",
            "_BASE_URL",
            "_ENDPOINT",
            "_HOME",
            "_CONFIG_DIR",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
        || (name.starts_with("AWS_") && name.ends_with("_FILE"))
}

/// Print `message` on stderr once per process.
fn warn_once(message: String) {
    static SHOWN: OnceLock<std::sync::Mutex<HashSet<String>>> = OnceLock::new();
    let shown = SHOWN.get_or_init(Default::default);
    if shown
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(message.clone())
    {
        eprintln!("{message}");
    }
}

/// A configuration found in the working directory may come with a folder its
/// user did not write: its models and connections whose address is not their
/// provider's documented one never use a key from the environment (neither
/// the provider's variable nor an `env:` reference, which becomes the
/// `markitai-no-key` placeholder). The returned warnings name each change.
pub fn seal_project_models(value: &mut Value) -> Vec<String> {
    let mut notes = Vec::new();
    let connections: Vec<Value> = value
        .pointer("/llm/providers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if let Some(list) = value
        .pointer_mut("/llm/providers")
        .and_then(Value::as_array_mut)
    {
        for connection in list.iter_mut().filter(|c| c.is_object()) {
            let provider = connection["provider"].as_str().unwrap_or("").to_owned();
            if custom_address(&provider, connection.get("api_base")) {
                let label = format!(
                    "connection {}",
                    connection["id"].as_str().unwrap_or("(unnamed)")
                );
                seal(connection, &label, &mut notes);
            }
        }
    }
    if let Some(list) = value
        .pointer_mut("/llm/model_list")
        .and_then(Value::as_array_mut)
    {
        for entry in list.iter_mut() {
            let linked = entry
                .pointer("/model_info/provider_id")
                .and_then(Value::as_str)
                .and_then(|id| connections.iter().find(|c| c["id"].as_str() == Some(id)));
            let Some(params) = entry.get_mut("litellm_params").filter(|p| p.is_object()) else {
                continue;
            };
            let model = params["model"].as_str().unwrap_or("").to_owned();
            let provider = linked
                .and_then(|c| c["provider"].as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| {
                    model
                        .split_once('/')
                        .map_or("openai", |(prefix, _)| prefix)
                        .to_owned()
                });
            let base = params
                .get("api_base")
                .filter(|base| base.as_str().is_some_and(|b| !b.trim().is_empty()))
                .or_else(|| linked.and_then(|c| c.get("api_base")))
                .cloned();
            if custom_address(&provider, base.as_ref()) {
                seal(params, &format!("model {model}"), &mut notes);
            }
        }
    }
    notes
}

/// Whether `base` names an address that is not `provider`'s documented one
/// (an `env:` reference is not a documented address).
fn custom_address(provider: &str, base: Option<&Value>) -> bool {
    let Some(base) = base
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|base| !base.is_empty())
    else {
        return false;
    };
    let origin = |text: &str| {
        url::Url::parse(text).ok().and_then(|url| {
            Some((
                url.scheme().to_owned(),
                url.host_str()?.to_ascii_lowercase(),
                url.port_or_known_default()?,
            ))
        })
    };
    let documented = crate::provider_management::provider_default_base(provider).and_then(origin);
    documented.is_none() || documented != origin(base)
}

/// Keep `params` (a model's or a connection's) off the environment's keys.
fn seal(params: &mut Value, label: &str, notes: &mut Vec<String>) {
    params["use_environment_credentials"] = json!(false);
    if let Some(reference) = params["api_key"]
        .as_str()
        .filter(|key| key.trim_start().starts_with("env:"))
        .map(str::to_owned)
    {
        params["api_key"] = json!(crate::provider_management::NO_KEY);
        notes.push(format!(
            "Warning: ./markitai.json in this folder: {label} names its own address, so its key reference {reference} is not read from the environment; write the key in the entry, or keep this configuration in your own config file or pass it with -c"
        ));
    }
}

/// Whether this build can route an `llm.model_list` model whose
/// `litellm_params.model` starts with `prefix/` (a model without a prefix is
/// `openai`). Other prefixes fail with an error naming the `openai/<model>`
/// plus `api_base` route for OpenAI-compatible endpoints.
pub fn llm_provider_supported(prefix: &str) -> bool {
    crate::llm::providers::supported(prefix)
}

pub fn enabled(config: &Value, path: &str) -> bool {
    config
        .pointer(path)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Translate the public dotted/indexed key syntax into an escaped JSON pointer.
pub fn key_pointer(key: &str) -> Result<String> {
    let mut tokens = Vec::new();
    for part in key.split('.') {
        if part.is_empty() {
            return Err(invalid(key, "has an empty path component"));
        }
        if let Some((name, suffix)) = part.split_once('[') {
            let index = suffix
                .strip_suffix(']')
                .ok_or_else(|| invalid(key, "has an invalid array index"))?;
            if name.is_empty()
                || index.is_empty()
                || !index.bytes().all(|byte| byte.is_ascii_digit())
            {
                return Err(invalid(key, "has an invalid array index"));
            }
            tokens.push(name.to_owned());
            tokens.push(
                index
                    .parse::<usize>()
                    .map_err(|_| invalid(key, "has an invalid array index"))?
                    .to_string(),
            );
        } else {
            if part.contains(']') {
                return Err(invalid(key, "has an invalid array index"));
            }
            tokens.push(part.to_owned());
        }
    }
    Ok(tokens
        .iter()
        .map(|part| format!("/{}", part.replace('~', "~0").replace('/', "~1")))
        .collect())
}

fn pointer_parts(pointer: &str) -> Vec<String> {
    pointer
        .split('/')
        .skip(1)
        .map(|part| part.replace("~1", "/").replace("~0", "~"))
        .collect()
}

fn schema_at<'a>(
    node: &'a Value,
    current: &Value,
    parts: &[String],
    path: &str,
) -> Result<&'a Value> {
    let node = dereference(node);
    if parts.is_empty() {
        return Ok(node);
    }
    if let Some(alternatives) = node.get("anyOf").and_then(Value::as_array) {
        let alternative = alternatives
            .iter()
            .find(|value| value.get("type").and_then(Value::as_str) != Some("null"))
            .ok_or_else(|| invalid(path, "cannot be traversed"))?;
        return schema_at(alternative, current, parts, path);
    }
    let part = &parts[0];
    if let Some(properties) = node.get("properties").and_then(Value::as_object) {
        let child = properties
            .get(part)
            .ok_or_else(|| invalid(path, "is not a known configuration key"))?;
        return schema_at(
            child,
            current.get(part).unwrap_or(&Value::Null),
            &parts[1..],
            path,
        );
    }
    if node.get("type").and_then(Value::as_str) == Some("array") {
        let index: usize = part
            .parse()
            .map_err(|_| invalid(path, "requires an array index"))?;
        let value = current
            .as_array()
            .and_then(|array| array.get(index))
            .ok_or_else(|| invalid(path, "has an out-of-range array index"))?;
        return schema_at(
            node.get("items").unwrap_or(&Value::Null),
            value,
            &parts[1..],
            path,
        );
    }
    if node.get("type").and_then(Value::as_str) == Some("object") {
        if current.get(part).is_none() && parts.len() > 1 {
            return Err(invalid(
                path,
                "has a missing map entry; set that entry first",
            ));
        }
        if let Some(additional) = node
            .get("additionalProperties")
            .filter(|value| value.is_object())
        {
            return schema_at(
                additional,
                current.get(part).unwrap_or(&Value::Null),
                &parts[1..],
                path,
            );
        }
        if parts.len() == 1 || current.get(part).is_some() {
            return Ok(&Value::Null);
        }
    }
    Err(invalid(path, "cannot be traversed"))
}

/// Parse a CLI value using its declared field type so string credentials and
/// paths such as "012345" or "true" are never inferred as numbers or booleans.
pub fn parse_cli_value(raw: &Value, key: &str, text: &str) -> Result<Value> {
    let pointer = key_pointer(key)?;
    let parts = pointer_parts(&pointer);
    let effective = normalize(raw)?;
    let mut field = schema_at(schema(), &effective, &parts, key)?;
    if let Some(alternatives) = field.get("anyOf").and_then(Value::as_array)
        && let Some(non_null) = alternatives
            .iter()
            .find(|node| node.get("type").and_then(Value::as_str) != Some("null"))
    {
        field = dereference(non_null);
    }
    match field.get("type").and_then(Value::as_str) {
        Some("string" | "boolean" | "integer" | "number") => Ok(json!(text)),
        _ => Ok(serde_json::from_str(text).unwrap_or_else(|_| json!(text))),
    }
}

/// Validate and apply one explicit edit, preserving the caller's sparse raw file.
/// Errors leave `raw` unchanged; unknown model fields cannot silently turn into no-ops.
pub fn set_value(raw: &mut Value, key: &str, value: Value) -> Result<Value> {
    let pointer = key_pointer(key)?;
    let parts = pointer_parts(&pointer);
    let mut effective = normalize(raw)?;
    schema_at(schema(), &effective, &parts, key)?;
    assign(&mut effective, &parts, value.clone(), key)?;
    let normalized = normalize(&effective)?;
    let normalized_value = normalized
        .pointer(&pointer)
        .cloned()
        .ok_or_else(|| invalid(key, "does not exist after validation"))?;
    let mut updated = raw.clone();
    assign(&mut updated, &parts, normalized_value.clone(), key)?;
    *raw = updated;
    Ok(normalized_value)
}

fn assign(target: &mut Value, parts: &[String], value: Value, path: &str) -> Result<()> {
    let Some((part, tail)) = parts.split_first() else {
        *target = value;
        return Ok(());
    };
    if let Some(array) = target.as_array_mut() {
        let index: usize = part
            .parse()
            .map_err(|_| invalid(path, "requires an array index"))?;
        let child = array
            .get_mut(index)
            .ok_or_else(|| invalid(path, "has an out-of-range array index"))?;
        return assign(child, tail, value, path);
    }
    let object = target
        .as_object_mut()
        .ok_or_else(|| invalid(path, "cannot be traversed"))?;
    if tail.is_empty() {
        object.insert(part.clone(), value);
        return Ok(());
    }
    assign(
        object.entry(part.clone()).or_insert_with(|| json!({})),
        tail,
        value,
        path,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_switches_trim_ignore_case_and_opt_outs_fail_safe() {
        // MARKITAI_NO_REMOTE_FETCH used to need an exact lowercase match, so
        // " 1" or "Y" left remote services enabled; it reads like
        // MARKITAI_NO_VLM_OCR now.
        for value in ["1", " 1 ", "true", "enabled", "Yes", "off", "arbitrary"] {
            assert!(env_opt_out(value), "{value}");
        }
        for value in ["", " ", "0", " false ", "No"] {
            assert!(!env_opt_out(value), "{value}");
        }
        // MARKITAI_PURE was case-sensitive and ignored `on`.
        for value in ["1", " TRUE ", "Yes", "on"] {
            assert_eq!(env_opt_in(value), Some(true), "{value}");
        }
        for value in ["0", "off", "no", "anything"] {
            assert_eq!(env_opt_in(value), Some(false), "{value}");
        }
        assert_eq!(env_opt_in(" "), None);
    }

    #[test]
    fn configured_presets_win_over_built_in_names_in_any_case() {
        let cfg =
            normalize(&json!({"presets":{"standard":{"desc":true},"team":{"ocr":true}}})).unwrap();
        // The reference looks a name up in the configuration first; the CLI
        // used to apply the built-in `standard` regardless.
        assert_eq!(
            preset(&cfg, "Standard").unwrap(),
            json!({"llm":false,"ocr":false,"alt":false,"desc":true,"screenshot":false})
        );
        assert_eq!(preset(&cfg, "TEAM").unwrap()["ocr"], true);
        assert_eq!(
            preset(&cfg, "rich").unwrap(),
            json!({"llm":true,"ocr":false,"alt":true,"desc":true,"screenshot":true})
        );
        assert_eq!(preset(&cfg, "minimal").unwrap()["llm"], false);
        assert_eq!(preset(&cfg, "bogus"), None);
    }

    #[test]
    fn config_files_are_bounded_regular_files_written_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new/config.json");
        assert_eq!(read_file(&path).unwrap(), None);
        write_file(&path, b"{}\n").unwrap();
        assert_eq!(read_file(&path).unwrap().as_deref(), Some(&b"{}\n"[..]));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&path), 0o600);
            assert_eq!(mode(path.parent().unwrap()), 0o700);
        }
        let large = vec![b' '; FILE_LIMIT + 1];
        let error = write_file(&path, &large).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::FileTooLarge);
        std::fs::write(&path, &large).unwrap();
        let error = read_file(&path).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::FileTooLarge);
        let error = read_file(dir.path()).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        let error = write_file(dir.path().join("new").as_path(), b"{}").unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[cfg(unix)]
    #[test]
    fn a_config_symlink_is_kept_and_a_fifo_never_blocks() {
        use std::os::unix::fs::FileTypeExt;
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("dotfiles/markitai.json");
        std::fs::create_dir(real.parent().unwrap()).unwrap();
        std::fs::write(&real, b"{\"old\":true}").unwrap();
        let link = dir.path().join("config.json");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert_eq!(
            read_file(&link).unwrap().as_deref(),
            Some(&b"{\"old\":true}"[..])
        );
        write_file(&link, b"{}").unwrap();
        assert!(std::fs::symlink_metadata(&link).unwrap().is_symlink());
        assert_eq!(std::fs::read(&real).unwrap(), b"{}");
        let fifo = dir.path().join("fifo.json");
        let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        let error = read_file(&fifo).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        let error = write_file(&fifo, b"{}").unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            std::fs::symlink_metadata(&fifo)
                .unwrap()
                .file_type()
                .is_fifo()
        );
    }

    #[test]
    fn inactive_key_warnings_only_inspect_explicit_raw_fields() {
        assert!(warnings(&json!({})).is_empty());
        let raw = json!({
            "batch":{"heavy_task_limit":0},
            "office":{"macos_fallback":false},
            "image":{"stdout_fetch_external":false},
            "llm":{"model_list":[{"model_name":"small","litellm_params":{"model":"openai/model"},"model_info":{"max_input_tokens":1024}}]},
            "unknown":{"heavy_task_limit":1}
        });
        assert!(validate(&raw).is_ok());
        assert_eq!(
            warnings(&raw)
                .iter()
                .map(|warning| warning.key)
                .collect::<Vec<_>>(),
            [
                "batch.heavy_task_limit",
                "office.macos_fallback",
                "image.stdout_fetch_external"
            ]
        );
        assert!(warnings(&json!({"batch":{},"office":{},"image":{}})).is_empty());
    }

    #[test]
    fn cidr_prefixes_and_ipv6_scopes_match_reference_network_validation() {
        for valid in [
            "192.0.2.1/032",
            "fe80::1%eth0/64",
            "fe80::1%a b/64",
            "::1/064",
        ] {
            assert!(valid_cidr(valid), "{valid}");
            assert!(
                normalize_model("FetchPolicyConfig", &json!({"local_only_patterns":[valid]}))
                    .is_ok()
            );
        }
        for invalid in [
            "192.0.2.1/+8",
            "::1/+8",
            "fe80::1%/64",
            "fe80::1%eth%0/64",
            "192.0.2.1%eth0/24",
        ] {
            assert!(!valid_cidr(invalid), "{invalid}");
            assert!(
                normalize_model(
                    "FetchPolicyConfig",
                    &json!({"local_only_patterns":[invalid]})
                )
                .is_err()
            );
        }
    }

    #[test]
    fn configuration_display_protects_credentials_without_hiding_safe_diagnostics() {
        let raw = json!({"apiKey":"secret-a","API-TOKEN":"secret-b","max_tokens":42,
            "nested":{"custom_api_base":"https://user:pass@[::1]:443/private?key=secret#fragment",
                "api_key":"env:SAFE_REFERENCE","extra_http_headers":{"X-Trace":"env:INLINE"}},
            "http_credentials":{"username":"hidden"}});
        let visible = redact(&raw);
        assert_eq!(visible["apiKey"], REDACTED);
        assert_eq!(visible["API-TOKEN"], REDACTED);
        assert_eq!(visible["max_tokens"], 42);
        assert_eq!(visible["nested"]["custom_api_base"], "https://[::1]:443");
        assert_eq!(visible["nested"]["api_key"], "env:SAFE_REFERENCE");
        assert_eq!(visible["nested"]["extra_http_headers"]["X-Trace"], REDACTED);
        assert_eq!(
            redact_for_key(
                "fetch.playwright.http_credentials.username",
                &json!("hidden")
            ),
            REDACTED
        );
        assert_eq!(
            redact_for_key(
                "fetch.playwright.extra_http_headers.X-Trace",
                &json!("env:INLINE")
            ),
            REDACTED
        );
        assert_eq!(
            redact_for_key("llm.api_base", &json!("not a URL")),
            REDACTED
        );
        assert_eq!(raw["apiKey"], "secret-a");
    }

    #[test]
    fn standalone_model_normalization_retains_required_and_custom_rules() {
        assert_eq!(
            normalize_model("LiteLLMParams", &json!({"model":"test"})).unwrap()["weight"],
            1
        );
        assert!(normalize_model("LiteLLMParams", &json!({})).is_err());
        assert!(normalize_model("FetchPolicyConfig", &json!({"strategy_priority":[]})).is_err());
        assert!(
            normalize_model(
                "FetchPolicyConfig",
                &json!({"local_only_patterns":["192.0.2.0/99"]})
            )
            .is_err()
        );
        assert!(
            normalize_model(
                "DomainProfileConfig",
                &json!({"strategy_priority":["static","static"]})
            )
            .is_err()
        );
        assert!(normalize_model("MissingConfig", &json!({})).is_err());
    }

    #[test]
    fn defaults_match_pinned_contract_fixture() {
        let expected: Value =
            serde_json::from_str(include_str!("../tests/fixtures/config_defaults.json")).unwrap();
        assert_eq!(defaults(), expected);
        assert_eq!(normalize(&defaults()).unwrap(), defaults());
    }

    #[test]
    fn a_folders_env_may_not_choose_programs_paths_endpoints_or_credential_files() {
        for name in [
            "COPILOT_CLI_PATH",
            "claude_cli_path",
            "MARKITAI_BROWSER_EXECUTABLE",
            "PLAYWRIGHT_BROWSERS_PATH",
            "PATH",
            "LD_PRELOAD",
            "DYLD_INSERT_LIBRARIES",
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_BEDROCK_RUNTIME_ENDPOINT",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "https_proxy",
            "NO_PROXY",
            "HOME",
            "CODEX_HOME",
            "CLAUDE_CONFIG_DIR",
            "XDG_CONFIG_HOME",
            "MARKITAI_CONFIG",
            "NODE_OPTIONS",
            "PYTHONPATH",
            "SSL_CERT_FILE",
        ] {
            assert!(project_may_not_set(name), "{name}");
        }
        for name in [
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "MODEL",
            "MARKITAI_LANG",
            "AWS_REGION",
            "AWS_PROFILE",
        ] {
            assert!(!project_may_not_set(name), "{name}");
        }
    }

    #[test]
    fn the_process_wins_and_a_folders_env_loses_the_refused_names() {
        let pairs = |list: &[(&str, &str)]| -> Vec<(String, String)> {
            list.iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect()
        };
        let project = pairs(&[
            ("OPENAI_API_KEY", "from-folder"),
            ("COPILOT_CLI_PATH", "/folder/fake-cli"),
            ("OPENAI_API_BASE", "https://folder.example/v1"),
            ("MODEL", "folder-model"),
        ]);
        let own = pairs(&[
            ("COPILOT_CLI_PATH", "/own/copilot"),
            ("DEEPSEEK_API_KEY", "own-key"),
            ("MODEL", "own-model"),
        ]);
        let (vars, ignored) = environment_from(
            pairs(&[("MODEL", "process-model")]).into_iter(),
            project.clone(),
            false,
            own.clone(),
        );
        assert_eq!(vars["MODEL"], "process-model");
        assert_eq!(vars["OPENAI_API_KEY"], "from-folder");
        assert_eq!(vars["COPILOT_CLI_PATH"], "/own/copilot");
        assert_eq!(vars["DEEPSEEK_API_KEY"], "own-key");
        assert!(!vars.contains_key("OPENAI_API_BASE"));
        assert_eq!(ignored, ["COPILOT_CLI_PATH", "OPENAI_API_BASE"]);
        // Run from ~/.markitai itself, the folder's .env is the user's own.
        let (vars, ignored) = environment_from(std::iter::empty(), project, true, Vec::new());
        assert_eq!(vars["COPILOT_CLI_PATH"], "/folder/fake-cli");
        assert!(ignored.is_empty());
    }

    #[test]
    fn a_project_config_keeps_environment_keys_at_documented_endpoints_only() {
        let mut value = json!({"llm":{
        "providers":[
            {"id":"lan","provider":"custom","api_base":"http://model.example:8080/v1"},
            {"id":"official","provider":"deepseek"}
        ],
        "model_list":[
            {"model_name":"a","litellm_params":{"model":"openai/gpt-x","api_base":"https://capture.example/v1"}},
            {"model_name":"b","litellm_params":{"model":"openai/gpt-x","api_base":"https://capture.example/v1","api_key":"env:OPENAI_API_KEY"}},
            {"model_name":"c","litellm_params":{"model":"deepseek/deepseek-chat","api_base":"https://api.deepseek.com","api_key":"env:DEEPSEEK_API_KEY"}},
            {"model_name":"d","litellm_params":{"model":"openai/gpt-x"}},
            {"model_name":"e","litellm_params":{"model":"openai/local"},"model_info":{"provider_id":"lan"}},
            {"model_name":"f","litellm_params":{"model":"openai/literal","api_base":"https://capture.example/v1","api_key":"sk-written-here"}},
            {"model_name":"g","litellm_params":{"model":"openai/gpt-x","api_base":"env:SOME_BASE"}}
        ]}});
        let notes = seal_project_models(&mut value);
        let params = |index: usize| value["llm"]["model_list"][index]["litellm_params"].clone();
        assert_eq!(params(0)["use_environment_credentials"], false);
        assert!(params(0).get("api_key").is_none());
        assert_eq!(params(1)["api_key"], crate::provider_management::NO_KEY);
        assert_eq!(params(1)["use_environment_credentials"], false);
        // The provider's own endpoint, and no address at all, keep the environment.
        assert_eq!(params(2)["api_key"], "env:DEEPSEEK_API_KEY");
        assert!(params(2).get("use_environment_credentials").is_none());
        assert!(params(3).get("use_environment_credentials").is_none());
        // A model at a connection's custom address, and the connection itself.
        assert_eq!(params(4)["use_environment_credentials"], false);
        assert_eq!(
            value["llm"]["providers"][0]["use_environment_credentials"],
            false
        );
        assert!(
            value["llm"]["providers"][1]
                .get("use_environment_credentials")
                .is_none()
        );
        // A key written in the entry stays; an env: address counts as custom.
        assert_eq!(params(5)["api_key"], "sk-written-here");
        assert_eq!(params(6)["use_environment_credentials"], false);
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("env:OPENAI_API_KEY"), "{}", notes[0]);
        assert!(normalize(&value).is_ok());
    }

    #[test]
    fn file_selection_has_one_winner() {
        let dir = tempfile::tempdir().unwrap();
        let cwd = dir.path().join("project");
        let home = dir.path().join("home");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(cwd.join("markitai.json"), "{}").unwrap();
        std::fs::write(home.join("config.json"), "{}").unwrap();
        let explicit = dir.path().join("explicit.json");
        let environment = dir.path().join("environment.json");
        assert_eq!(
            choose_path(Some(&explicit), Some(&environment), &cwd, &home),
            Some(explicit)
        );
        assert_eq!(
            choose_path(None, Some(&environment), &cwd, &home),
            Some(environment)
        );
        assert_eq!(
            choose_path(None, None, &cwd, &home),
            Some(cwd.join("markitai.json"))
        );
        std::fs::remove_file(cwd.join("markitai.json")).unwrap();
        assert_eq!(
            choose_path(None, None, &cwd, &home),
            Some(home.join("config.json"))
        );
    }

    #[test]
    fn missing_selected_file_defaults_but_malformed_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        assert_eq!(load(Some(&path), None).unwrap(), defaults());
        std::fs::write(&path, "{bad").unwrap();
        assert!(load(Some(&path), None).is_err());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "{bad");
    }

    #[test]
    fn overlay_preserves_siblings_and_replaces_model_arrays() {
        let mut value = json!({"llm":{"keep_base":true,"model_list":[{"model_name":"old","litellm_params":{"model":"old"}}]}});
        merge(
            &mut value,
            json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"new"}}]}}),
        );
        let cfg = normalize(&value).unwrap();
        assert_eq!(cfg["llm"]["keep_base"], true);
        assert_eq!(cfg["llm"]["model_list"].as_array().unwrap().len(), 1);
        assert_eq!(cfg["llm"]["model_list"][0]["litellm_params"]["weight"], 1);
        assert_eq!(cfg["llm"]["model_list"][0]["model_info"], Value::Null);
        // The one internal model field is kept; other unknown ones are not.
        let cfg = normalize(
            &json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"m"},
            "model_info":{"hidden_endpoint":true,"other":1}}]}}),
        )
        .unwrap();
        assert_eq!(
            cfg["llm"]["model_list"][0]["model_info"]["hidden_endpoint"],
            true
        );
        assert!(
            cfg["llm"]["model_list"][0]["model_info"]
                .get("other")
                .is_none()
        );
        let cfg = normalize(
            &json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"m"},
            "model_info":{"hidden_endpoint":"yes"}}]}}),
        )
        .unwrap();
        assert!(
            cfg["llm"]["model_list"][0]["model_info"]
                .get("hidden_endpoint")
                .is_none()
        );
    }

    #[test]
    fn schema_acceptance_is_independent_of_runtime_availability() {
        let cfg=normalize(&json!({"ocr":{"enabled":true,"lang":"zh"},"fetch":{"strategy":"playwright","domain_profiles":{"example.com":{"extra_wait_ms":150}}},"presets":{"custom":{"llm":true}}})).unwrap();
        assert_eq!(cfg["ocr"]["per_page_routing"], true);
        assert_eq!(
            cfg["fetch"]["domain_profiles"]["example.com"]["skip_auto_scroll"],
            false
        );
        assert_eq!(cfg["presets"]["custom"]["ocr"], false);
    }

    #[test]
    fn recognized_coercions_and_unknown_fields_match_reference_behavior() {
        let cfg=normalize(&json!({"future":3,"llm":{"enabled":"YES","concurrency":"1_024","future":true},"image":{"quality":75.0},"fetch":{"playwright":{"timeout":false}}})).unwrap();
        assert!(cfg.get("future").is_none());
        assert!(cfg["llm"].get("future").is_none());
        assert_eq!(cfg["llm"]["enabled"], true);
        assert_eq!(cfg["llm"]["concurrency"], 1024);
        assert_eq!(cfg["image"]["quality"], 75);
        assert_eq!(cfg["fetch"]["playwright"]["timeout"], 0);
        assert!(normalize(&json!({"llm":{"concurrency":4096}})).is_ok());
        for input in [
            json!({"llm":{"enabled":" true "}}),
            json!({"llm":{"concurrency":"1e2"}}),
            json!({"image":{"quality":75.2}}),
            json!({"output":{"dir":42}}),
        ] {
            assert!(normalize(&input).is_err(), "{input}");
        }
    }

    #[test]
    fn custom_fetch_validation_rejects_invalid_orders_and_networks() {
        for input in [
            json!({"fetch":{"policy":{"strategy_priority":[]}}}),
            json!({"fetch":{"policy":{"strategy_priority":["static","static"]}}}),
            json!({"fetch":{"domain_profiles":{"example.com":{"strategy_priority":["auto"]}}}}),
            json!({"fetch":{"policy":{"local_only_patterns":[" "]}}}),
            json!({"fetch":{"policy":{"local_only_patterns":["192.168.1.1/33"]}}}),
            json!({"fetch":{"policy":{"local_only_patterns":["host.example/path"]}}}),
        ] {
            assert!(normalize(&input).is_err(), "{input}");
        }
        assert!(normalize(&json!({"fetch":{"policy":{"local_only_patterns":["localhost",".corp.example","10.5.1.4/8","192.168.1.0/255.255.255.0","192.168.1.0/0.0.0.255","fe80::/64"]}}})).is_ok());
    }

    #[test]
    fn required_deployment_fields_and_nested_types_are_checked() {
        for input in [
            json!({"llm":{"model_list":[{"litellm_params":{"model":"a"}}]}}),
            json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{}}]}}),
            json!({"llm":{"providers":[{"id":"key"}]}}),
            json!({"fetch":{"playwright":{"extra_http_headers":{"header":1}}}}),
            json!({"fetch":{"playwright":{"cookies":[{"name":"a","value":null}]}}}),
            json!({"llm":{"router_settings":{"fallbacks":["not a map"]}}}),
        ] {
            assert!(normalize(&input).is_err(), "{input}");
        }
    }

    #[test]
    fn internal_targets_survive_but_are_not_default_or_editable_keys() {
        let cfg = normalize(&json!({"output":{"filename":"exact.md","reserved_stem":"note.v2"}}))
            .unwrap();
        assert_eq!(cfg["output"]["filename"], "exact.md");
        assert_eq!(cfg["output"]["reserved_stem"], "note.v2");
        assert!(defaults()["output"].get("filename").is_none());
        assert!(set_value(&mut json!({}), "output.reserved_stem", json!("x")).is_err());
    }

    #[test]
    fn explicit_edits_reject_unknown_leaves_and_rollback_invalid_values() {
        let mut raw = json!({"unknown_original":{"kept":true},"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"test"}}]}});
        let original = raw.clone();
        assert!(set_value(&mut raw, "llm.concurreny", json!(3)).is_err());
        assert!(set_value(&mut raw, "image.quality", json!(101)).is_err());
        assert!(set_value(&mut raw, "llm.model_list[1].model_name", json!("missing")).is_err());
        assert_eq!(raw, original);
        assert_eq!(
            set_value(
                &mut raw,
                "llm.model_list[0].litellm_params.weight",
                json!("2")
            )
            .unwrap(),
            json!(2)
        );
        assert_eq!(raw["llm"]["model_list"][0]["litellm_params"]["weight"], 2);
        assert_eq!(raw["unknown_original"], original["unknown_original"]);
        assert!(raw.get("image").is_none());
        set_value(&mut raw, "presets.custom", json!({"llm":true})).unwrap();
        assert_eq!(raw["presets"]["custom"]["ocr"], false);
    }

    #[test]
    fn env_resolution_is_lazy_and_missing_explicit_reference_never_uses_fallback() {
        let vars = HashMap::from([
            ("SECRET".into(), "value".into()),
            ("JINA_API_KEY".into(), "fallback".into()),
            ("EMPTY".into(), String::new()),
        ]);
        assert_eq!(
            resolve_env_value("env:SECRET", &vars, true)
                .unwrap()
                .as_deref(),
            Some("value")
        );
        assert_eq!(
            resolve_env_value("env:EMPTY", &vars, true)
                .unwrap()
                .as_deref(),
            Some("")
        );
        assert!(resolve_env_value("env:MISSING", &vars, true).is_err());
        assert_eq!(
            resolve_optional(Some("env:MISSING"), Some("JINA_API_KEY"), &vars, false).unwrap(),
            None
        );
        assert_eq!(
            resolve_optional(None, Some("JINA_API_KEY"), &vars, false)
                .unwrap()
                .as_deref(),
            Some("fallback")
        );
        assert!(normalize(&json!({"fetch":{"jina":{"api_key":"env:MISSING"}}})).is_ok());
    }
    #[test]
    fn numeric_bounds_cover_independent_subsystems() {
        for (path, invalid_value) in [
            ("/image/quality", json!(0)),
            ("/image/quality", json!(101)),
            ("/screenshot/quality", json!(101)),
            ("/batch/concurrency", json!(0)),
            ("/batch/scan_max_depth", json!(-1)),
            ("/batch/scan_max_files", json!(0)),
            ("/batch/heavy_task_limit", json!(-1)),
            ("/llm/router_settings/num_retries", json!(-1)),
            ("/llm/router_settings/timeout", json!(0)),
            ("/llm/max_requests_per_document", json!(-1)),
            ("/llm/max_cost_per_document_usd", json!(-0.1)),
            ("/cache/fetch_ttl_seconds", json!(-1)),
            ("/fetch/policy/max_strategy_hops", json!(7)),
            ("/fetch/playwright/session_ttl_seconds", json!(59)),
            ("/fetch/playwright/session_ttl_seconds", json!(7201)),
            ("/fetch/jina/rpm", json!(0)),
            ("/fetch/defuddle/rpm", json!(0)),
        ] {
            let mut cfg = defaults();
            *cfg.pointer_mut(path).unwrap() = invalid_value;
            assert!(validate(&cfg).is_err(), "{path}");
        }
        let cfg = normalize(
            &json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"test","weight":0}}]},"fetch":{"domain_profiles":{"example.com":{"extra_wait_ms":30001}}}}),
        );
        assert!(cfg.is_err());
    }

    #[test]
    fn cli_string_fields_keep_literal_values() {
        let mut raw = json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"test"}}]}});
        for (key, text) in [
            ("output.dir", "true"),
            ("llm.model_list[0].litellm_params.api_key", "0123456789"),
            ("prompts.cleaner_system", "null"),
        ] {
            let parsed = parse_cli_value(&raw, key, text).unwrap();
            assert_eq!(set_value(&mut raw, key, parsed).unwrap(), json!(text));
        }
        let parsed = parse_cli_value(&raw, "image.quality", "76").unwrap();
        assert_eq!(
            set_value(&mut raw, "image.quality", parsed).unwrap(),
            json!(76)
        );
        let parsed = parse_cli_value(&raw, "output.report", "null").unwrap();
        assert!(set_value(&mut raw, "output.report", parsed).is_err());
    }
}
