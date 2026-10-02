//! Configuration normalization is independent of converter availability.
//! Embedded metadata contains types, bounds and defaults, without documentation prose.

use crate::{Error, Result};
use serde_json::{Map, Number, Value, json};
use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub fn home() -> PathBuf {
    std::env::var_os("MARKITAI_HOME")
        .map(|path| expand_home(Path::new(&path)))
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".markitai")
        })
}

pub fn expand_home(path: &Path) -> PathBuf {
    if let Ok(rest) = path.strip_prefix("~")
        && let Some(user) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))
    {
        PathBuf::from(user).join(rest)
    } else {
        path.to_owned()
    }
}

/// Resolve legacy default state paths into the explicitly isolated Rust home.
/// Custom paths retain their meaning; the serialized default remains compatible.
pub fn state_path(path: &Path) -> PathBuf {
    if std::env::var_os("MARKITAI_HOME").is_some()
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
    let environment = environment();
    let env_path = environment
        .get("MARKITAI_CONFIG")
        .filter(|s| !s.is_empty())
        .map(Path::new);
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    choose_path(explicit, env_path, &cwd, &home()).map(|path| expand_home(&path))
}

pub fn load(explicit: Option<&Path>, overrides: Option<Value>) -> Result<Value> {
    normalize(&raw(explicit, overrides)?)
}

/// The selected file with `overrides` merged over it, before defaults are
/// filled in. A caller that must tell a value the user wrote from the same
/// value as a default (such as `fetch.remote_consent`) reads it here and
/// normalizes the result itself; [`load`] is `normalize(&raw(..))`.
pub fn raw(explicit: Option<&Path>, overrides: Option<Value>) -> Result<Value> {
    let selected = selected_path(explicit);
    let mut value = json!({});
    if let Some(path) = selected {
        match std::fs::read(&path) {
            Ok(bytes) => {
                value = serde_json::from_slice(&bytes).map_err(|error| {
                    Error::Config(format!("Invalid JSON in {}: {error}", path.display()))
                })?;
                if !value.is_object() {
                    return Err(invalid("", "must be a JSON object"));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
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

fn redact_named(key: &str, value: &Value) -> Option<Value> {
    let key_normalized = normalized_key(key);
    if key_normalized == "extra_http_headers" || key_normalized.ends_with("_extra_http_headers") {
        return Some(match value.as_object() {
            Some(headers) => object(headers.keys().map(|key| (key.clone(), json!(REDACTED)))),
            None => json!(REDACTED),
        });
    }
    let environment_reference = value
        .as_str()
        .is_some_and(|value| value.starts_with("env:"));
    if key_normalized == "api_base" || key_normalized.ends_with("_api_base") {
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

/// Read dotenv as data without changing the host process environment.
pub fn environment() -> HashMap<String, String> {
    let mut vars: HashMap<String, String> = std::env::vars().collect();
    for path in [PathBuf::from(".env"), home().join(".env")] {
        if let Ok(iter) = dotenvy::from_path_iter(path) {
            for (key, value) in iter.flatten() {
                vars.entry(key).or_insert(value);
            }
        }
    }
    vars
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
    fn defaults_match_verified_reference_fixture() {
        let expected: Value =
            serde_json::from_str(include_str!("../tests/fixtures/config_defaults.json")).unwrap();
        assert_eq!(defaults(), expected);
        assert_eq!(normalize(&defaults()).unwrap(), defaults());
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
