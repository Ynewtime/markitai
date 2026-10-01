//! The installed official Claude runtime owns subscription authentication.
mod process;
#[cfg(test)]
mod tests;

use super::{AuthStatus, FailureKind, Model, ObservedCall, Request};
use base64::Engine;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

pub const CLI_VERSION: &str = "2.1.284";
/// `CLI_VERSION` as numbers: the oldest release of the supported line.
const PINNED: (u64, u64, u64) = (2, 1, 284);
pub const SDK_VERSION: &str = "0.3.284";
const INPUT_LIMIT: usize = 10 * 1024 * 1024;
const TEXT_LIMIT: usize = 8 * 1024 * 1024;

/// This configuration contains paths, never extracted subscription tokens.
#[derive(Clone)]
pub struct Config {
    executable: PathBuf,
    environment: HashMap<String, String>,
}
impl Config {
    pub fn from_env(env: &HashMap<String, String>) -> crate::Result<Self> {
        let path = env
            .get("CLAUDE_CLI_PATH")
            .map(PathBuf::from)
            .or_else(|| {
                env.get("PATH").and_then(|value| {
                    std::env::split_paths(value)
                        .map(|dir| {
                            dir.join(if cfg!(windows) {
                                "claude.exe"
                            } else {
                                "claude"
                            })
                        })
                        .find(|p| p.is_file())
                })
            })
            .ok_or_else(|| {
                crate::Error::Unsupported(
                    "The supported official Claude CLI is not installed".into(),
                )
            })?;
        let executable = path
            .canonicalize()
            .map_err(|_| crate::Error::InvalidInput("Claude executable is unavailable".into()))?;
        if !executable.is_file() {
            return Err(crate::Error::InvalidInput(
                "Claude executable must be a regular file".into(),
            ));
        }
        let mut environment = HashMap::new();
        // Preserve the user's official login location. Provider overrides and OAuth
        // values are neither inspected nor passed through this subscription adapter.
        for name in [
            "HOME",
            "PATH",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "SystemRoot",
            "SYSTEMROOT",
            "WINDIR",
            "TMPDIR",
            "TMP",
            "TEMP",
            "LANG",
            "LC_ALL",
            "CLAUDE_CONFIG_DIR",
        ] {
            if let Some(value) = env.get(name).cloned().or_else(|| std::env::var(name).ok()) {
                environment.insert(name.into(), value);
            }
        }
        Ok(Self {
            executable,
            environment,
        })
    }
    pub fn executable(&self) -> &Path {
        &self.executable
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TokenCounts {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct AggregateUsage {
    pub totals: TokenCounts,
    pub by_model: BTreeMap<String, TokenCounts>,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct UsageEvidence {
    /// Actual distinct assistant API message IDs. Aggregate totals are not extra calls.
    pub calls: Vec<ObservedCall>,
    /// Runtime-reported totals, potentially available without a known request count.
    pub aggregate: Option<AggregateUsage>,
}
#[derive(Debug)]
pub struct Completion {
    pub text: String,
    pub usage: UsageEvidence,
    pub warnings: Vec<String>,
}
#[derive(Debug)]
pub struct Failure {
    pub kind: FailureKind,
    pub error: crate::Error,
    pub usage: Box<UsageEvidence>,
}
impl Failure {
    fn new(kind: FailureKind, message: &'static str) -> Self {
        Self {
            kind,
            error: crate::Error::Conversion(message.into()),
            usage: Box::default(),
        }
    }
}

fn protocol() -> Failure {
    Failure::new(
        FailureKind::Protocol,
        "Claude returned an invalid or conflicting protocol response",
    )
}
fn limit() -> Failure {
    Failure::new(
        FailureKind::ResourceLimit,
        "Claude request or response exceeds its resource limits",
    )
}
fn text(value: &Value, max: usize) -> Result<&str, Failure> {
    let s = value.as_str().ok_or_else(protocol)?;
    if s.len() > max || s.contains('\0') {
        return Err(limit());
    }
    Ok(s)
}
fn name(value: &Value) -> Result<&str, Failure> {
    let s = text(value, 512)?;
    if s.is_empty() || s.chars().any(char::is_control) {
        return Err(protocol());
    }
    Ok(s)
}
fn json_line(value: &Value) -> Result<Vec<u8>, Failure> {
    let mut bytes = serde_json::to_vec(value).map_err(|_| protocol())?;
    bytes.push(b'\n');
    if bytes.len() > INPUT_LIMIT {
        return Err(limit());
    }
    Ok(bytes)
}

fn small_command(
    config: &Config,
    args: &[&str],
    deadline: Instant,
    cancel: Option<&AtomicBool>,
) -> Result<(Vec<u8>, std::process::ExitStatus), Failure> {
    let args: Vec<OsString> = args.iter().map(OsString::from).collect();
    let mut process =
        process::Process::spawn(config, &args, process::workspace()?, deadline, cancel)?;
    process.send(None, cancel)?;
    let mut output = Vec::new();
    while let Some(line) = process.next(cancel)? {
        if output.len().saturating_add(line.len()) > 64 * 1024 {
            return Err(limit());
        }
        output.extend_from_slice(&line);
    }
    let status = process.finish(cancel)?;
    Ok((output, status))
}
/// Whether `version` (`2.1.285` or `2.1.285 (Claude Code)`) is a release this
/// adapter speaks: the pinned 2.1.284 or a later patch of the 2.1 line. The
/// official runtime updates itself between patch releases, which keep the
/// stream-json protocol of their line; a new minor or major line or an older
/// patch fails explicitly before any request, and the stream itself is still
/// validated message by message.
fn supported_version(version: &str) -> bool {
    let version = version.strip_suffix(" (Claude Code)").unwrap_or(version);
    let mut parts = version.split('.').map(|part| {
        (!part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
            .then(|| part.parse::<u64>().ok())
            .flatten()
    });
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(Some(major)), Some(Some(minor)), Some(Some(patch)), None) => {
            (major, minor) == (PINNED.0, PINNED.1) && patch >= PINNED.2
        }
        _ => false,
    }
}
/// The installed runtime's release, when it is one this adapter speaks.
fn version(
    config: &Config,
    deadline: Instant,
    cancel: Option<&AtomicBool>,
) -> Result<String, Failure> {
    let (bytes, status) = small_command(config, &["--version"], deadline, cancel)?;
    let value = std::str::from_utf8(&bytes).map_err(|_| protocol())?.trim();
    if !status.success() || !supported_version(value) {
        return Err(Failure::new(
            FailureKind::Unsupported,
            "Installed Claude CLI does not match the supported protocol version",
        ));
    }
    Ok(value
        .strip_suffix(" (Claude Code)")
        .unwrap_or(value)
        .to_owned())
}
pub fn status(config: &Config, timeout: Duration) -> Result<AuthStatus, Failure> {
    let deadline = process::deadline(timeout)?;
    let installed = version(config, deadline, None)?;
    let (bytes, exit) = small_command(config, &["auth", "status"], deadline, None)?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|_| protocol())?;
    let logged = value["loggedIn"].as_bool().ok_or_else(protocol)?;
    if (logged && !exit.success()) || (!logged && !matches!(exit.code(), Some(0 | 1))) {
        return Err(protocol());
    }
    let method = value.get("authMethod").and_then(Value::as_str);
    let provider = value.get("apiProvider").and_then(Value::as_str);
    let subscribed =
        logged && method == Some("claude.ai") && provider.is_none_or(|p| p == "firstParty");
    let user = value
        .get("email")
        .filter(|v| !v.is_null())
        .map(|v| name(v).map(str::to_owned))
        .transpose()?;
    let subscription = value
        .get("subscriptionType")
        .filter(|v| !v.is_null())
        .map(|v| name(v).map(str::to_owned))
        .transpose()?;
    Ok(AuthStatus {
        provider: "claude-agent",
        authenticated: subscribed,
        user,
        expires_at: None,
        error: (!subscribed).then(|| {
            if logged {
                "Claude CLI is authenticated through a non-subscription or unrecognized method"
            } else {
                "Claude CLI is not signed in"
            }
            .into()
        }),
        details: json!({"source":"official_cli","cli_version":installed,"supported_from":CLI_VERSION,"native_adapter":true,"auth_method":method.filter(|m| matches!(*m, "claude.ai" | "apiKey")),"subscription":subscription}),
    })
}
fn private_file(workspace: &Path, file: &str, bytes: &[u8]) -> Result<PathBuf, Failure> {
    use std::io::Write;
    let path = workspace.join(file);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path).map_err(|_| protocol())?;
    file.write_all(bytes).map_err(|_| protocol())?;
    Ok(path)
}
fn start(
    config: &Config,
    model: Option<&str>,
    system: &str,
    deadline: Instant,
    cancel: Option<&AtomicBool>,
) -> Result<process::Process, Failure> {
    let workspace = process::workspace()?;
    let prompt = private_file(workspace.path(), "system.txt", system.as_bytes())?;
    let settings = private_file(
        workspace.path(),
        "settings.json",
        br#"{"disableAllHooks":true,"autoMemoryEnabled":false}"#,
    )?;
    let mcp = private_file(workspace.path(), "mcp.json", br#"{"mcpServers":{}}"#)?;
    let mut args: Vec<OsString> = [
        "--print",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--safe-mode",
        "--restricted",
        "--tools",
        "",
        "--disallowedTools",
        "*",
        "--permission-prompts",
        "none",
        "--permission-mode",
        "dontAsk",
        "--strict-mcp-config",
        "--setting-sources=",
        "--no-session-persistence",
        "--no-chrome",
        "--disable-slash-commands",
        "--max-turns",
        "1",
        "--system-prompt-file",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    args.push(prompt.into_os_string());
    args.push("--settings".into());
    args.push(settings.into_os_string());
    args.push("--mcp-config".into());
    args.push(mcp.into_os_string());
    if let Some(model) = model {
        args.push("--model".into());
        args.push(model.into());
    }
    process::Process::spawn(config, &args, workspace, deadline, cancel)
}
fn initialize(
    process: &mut process::Process,
    cancel: Option<&AtomicBool>,
) -> Result<Value, Failure> {
    process.send(Some(json_line(&json!({"type":"control_request","request_id":"markitai-initialize","request":{"subtype":"initialize","hooks":null,"agents":{},"skills":[]}}))?), cancel)?;
    loop {
        let bytes = process.next(cancel)?.ok_or_else(protocol)?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| protocol())?;
        match value["type"].as_str() {
            Some("control_response") => {
                let response = &value["response"];
                if response["request_id"] != "markitai-initialize"
                    || response["subtype"] != "success"
                {
                    return Err(protocol());
                }
                let result = response
                    .get("response")
                    .filter(|v| v.is_object())
                    .ok_or_else(protocol)?
                    .clone();
                let account = &result["account"];
                if account["apiProvider"] != "firstParty"
                    || account["subscriptionType"]
                        .as_str()
                        .is_none_or(str::is_empty)
                    || account
                        .get("apiKeySource")
                        .filter(|v| !v.is_null())
                        .and_then(Value::as_str)
                        .is_some_and(|s| !matches!(s, "" | "none"))
                {
                    return Err(Failure::new(
                        FailureKind::Authentication,
                        "Claude CLI did not confirm a subscription account for this session",
                    ));
                }
                return Ok(result);
            }
            Some("system") if matches!(value["subtype"].as_str(), Some("init" | "status")) => {
                validate_system(&value)?
            }
            Some("control_request") => {
                return Err(Failure::new(
                    FailureKind::Permission,
                    "Claude requested a tool, hook or permission callback",
                ));
            }
            _ => return Err(protocol()),
        }
    }
}
fn catalog(value: &Value) -> Result<Vec<(Model, Option<String>)>, Failure> {
    let entries = value["models"].as_array().ok_or_else(protocol)?;
    if entries.len() > 4096 {
        return Err(limit());
    }
    let mut models = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for entry in entries {
        let id = name(&entry["value"])?;
        if !seen.insert(id.to_owned()) {
            return Err(protocol());
        }
        let label = name(&entry["displayName"])?;
        let resolved = entry
            .get("resolvedModel")
            .filter(|v| !v.is_null())
            .map(|v| name(v).map(str::to_owned))
            .transpose()?;
        // The official catalog does not expose vision support. False means not
        // advertised here; callers may explicitly choose the supported image path.
        models.push((
            Model {
                model: format!("claude-agent/{id}"),
                label: label.into(),
                supports_vision: false,
            },
            resolved,
        ));
    }
    Ok(models)
}
pub fn models(config: &Config, timeout: Duration) -> Result<Vec<Model>, Failure> {
    let deadline = process::deadline(timeout)?;
    version(config, deadline, None)?;
    let mut process = start(config, None, "", deadline, None)?;
    let result = catalog(&initialize(&mut process, None)?)?;
    process.send(None, None)?;
    while let Some(bytes) = process.next(None)? {
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| protocol())?;
        if value["type"] != "system" {
            return Err(protocol());
        }
        validate_system(&value)?;
    }
    if !process.finish(None)?.success() {
        return Err(protocol());
    }
    Ok(result.into_iter().map(|v| v.0).collect())
}
fn validate_system(value: &Value) -> Result<(), Failure> {
    match value["subtype"].as_str() {
        Some("init") => {
            if value
                .get("claude_code_version")
                .is_some_and(|v| !v.as_str().is_some_and(supported_version))
            {
                return Err(protocol());
            }
            if value.get("apiKeySource").is_some_and(|v| v != "none") {
                return Err(Failure::new(
                    FailureKind::Authentication,
                    "Claude changed to a non-subscription credential source",
                ));
            }
            // Agent names are catalog metadata, not granted tools. This pinned
            // runtime advertises its four built-ins even with an empty tool set.
            // Unknown/custom names remain excluded; actual Agent tool availability
            // and task/subagent events are independently rejected below.
            if let Some(agents) = value.get("agents")
                && !agents.as_array().is_some_and(|names| {
                    names.len() <= 4
                        && names.iter().all(|name| {
                            matches!(
                                name.as_str(),
                                Some("claude" | "Explore" | "general-purpose" | "Plan")
                            )
                        })
                })
            {
                return Err(Failure::new(
                    FailureKind::Permission,
                    "Claude advertised an unrequested agent extension",
                ));
            }
            for key in ["tools", "mcp_servers", "plugins", "skills"] {
                if let Some(v) = value.get(key)
                    && !v.as_array().is_some_and(Vec::is_empty)
                {
                    return Err(Failure::new(
                        FailureKind::Permission,
                        "Claude enabled an unrequested tool or extension",
                    ));
                }
            }
            if value
                .get("permissionMode")
                .and_then(Value::as_str)
                .is_some_and(|v| v != "dontAsk")
            {
                return Err(Failure::new(
                    FailureKind::Permission,
                    "Claude changed the configured permission mode",
                ));
            }
            Ok(())
        }
        Some("status") if value["status"].is_null() => Ok(()),
        Some("session_state_changed")
            if matches!(value["state"].as_str(), Some("running" | "idle")) =>
        {
            Ok(())
        }
        Some("api_retry") => Ok(()),
        Some(
            "permission_denied" | "hook_started" | "hook_progress" | "hook_response"
            | "task_started" | "task_progress" | "task_notification",
        ) => Err(Failure::new(
            FailureKind::Permission,
            "Claude attempted an unrequested tool, hook or background task",
        )),
        _ => Err(protocol()),
    }
}
fn counts(value: &Value, camel: bool) -> Result<TokenCounts, Failure> {
    if !value.is_object() {
        return Err(protocol());
    }
    let keys = if camel {
        [
            "inputTokens",
            "outputTokens",
            "cacheReadInputTokens",
            "cacheCreationInputTokens",
        ]
    } else {
        [
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ]
    };
    let mut output = [None; 4];
    for (i, key) in keys.iter().enumerate() {
        if let Some(v) = value.get(*key).filter(|v| !v.is_null()) {
            output[i] = Some(
                v.as_u64()
                    .filter(|n| *n <= 9_007_199_254_740_991)
                    .ok_or_else(protocol)?,
            );
        }
    }
    Ok(TokenCounts {
        input_tokens: output[0],
        output_tokens: output[1],
        cache_read_tokens: output[2],
        cache_write_tokens: output[3],
    })
}
fn aggregate(value: &Value) -> Result<Option<AggregateUsage>, Failure> {
    let mut result = AggregateUsage::default();
    let mut observed = false;
    if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
        result.totals = counts(usage, false)?;
        observed = true;
    }
    if let Some(models) = value.get("modelUsage").filter(|v| !v.is_null()) {
        let models = models.as_object().ok_or_else(protocol)?;
        if models.len() > 64 {
            return Err(limit());
        }
        for (model, usage) in models {
            name(&Value::String(model.clone()))?;
            result.by_model.insert(model.clone(), counts(usage, true)?);
            observed = true;
        }
    }
    Ok(observed.then_some(result))
}
#[derive(Default)]
struct Evidence {
    usage: UsageEvidence,
    messages: HashMap<String, usize>,
    session: Option<String>,
}
impl Evidence {
    fn event(&mut self, value: &Value, expected: &str) -> Result<(), Failure> {
        if let Some(id) = value.get("session_id").filter(|v| !v.is_null()) {
            let id = name(id)?;
            if self.session.as_ref().is_some_and(|old| old != id) {
                return Err(protocol());
            }
            self.session.get_or_insert_with(|| id.to_owned());
        }
        if value["type"] == "result" {
            self.usage.aggregate = aggregate(value)?;
        }
        if value["type"] != "assistant" {
            return Ok(());
        }
        let message = &value["message"];
        let id = name(&message["id"])?;
        let model = name(&message["model"])?;
        if let Some(usage) = message.get("usage").filter(|v| !v.is_null()) {
            let c = counts(usage, false)?;
            let call = ObservedCall {
                model: format!("claude-agent/{model}"),
                input_tokens: c.input_tokens,
                output_tokens: c.output_tokens,
                cache_read_tokens: c.cache_read_tokens,
                cache_write_tokens: c.cache_write_tokens,
            };
            if let Some(index) = self.messages.get(id).copied() {
                let mut previous = self.usage.calls[index].clone();
                if previous.model != call.model {
                    return Err(protocol());
                }
                for (old, new) in [
                    (&mut previous.input_tokens, call.input_tokens),
                    (&mut previous.output_tokens, call.output_tokens),
                    (&mut previous.cache_read_tokens, call.cache_read_tokens),
                    (&mut previous.cache_write_tokens, call.cache_write_tokens),
                ] {
                    if let Some(n) = new {
                        if old.is_some_and(|o| n < o) {
                            return Err(protocol());
                        }
                        *old = Some(n);
                    }
                }
                self.usage.calls[index] = previous;
            } else {
                if self.usage.calls.len() >= 64 {
                    return Err(limit());
                }
                self.messages.insert(id.into(), self.usage.calls.len());
                self.usage.calls.push(call);
            }
        }
        if model != expected {
            return Err(Failure::new(
                FailureKind::Unsupported,
                "Claude served a different model than requested",
            ));
        }
        if value
            .get("parent_tool_use_id")
            .is_some_and(|v| !v.is_null())
        {
            return Err(Failure::new(
                FailureKind::Permission,
                "Claude returned an unrequested subagent response",
            ));
        }
        if let Some(error) = value.get("error").filter(|v| !v.is_null()) {
            return Err(Failure::new(
                match error.as_str() {
                    Some("authentication_failed") => FailureKind::Authentication,
                    Some("rate_limit") => FailureKind::RateLimit,
                    _ => FailureKind::Transport,
                },
                "Claude runtime reported an unsuccessful model response",
            ));
        }
        let blocks = message["content"].as_array().ok_or_else(protocol)?;
        if blocks.iter().any(|b| {
            !matches!(
                b["type"].as_str(),
                Some("text" | "thinking" | "redacted_thinking")
            )
        }) {
            return Err(Failure::new(
                FailureKind::Permission,
                "Claude returned an unrequested tool operation",
            ));
        }
        if message["stop_reason"] == "max_tokens" {
            return Err(Failure::new(
                FailureKind::Truncated,
                "Claude response was truncated",
            ));
        }
        Ok(())
    }
}
fn input(request: &Request<'_>) -> Result<Vec<u8>, Failure> {
    if request.system.len() > TEXT_LIMIT
        || request.user.len() > TEXT_LIMIT
        || request.images.len() > 100
    {
        return Err(limit());
    }
    let raw = request
        .images
        .iter()
        .try_fold(0usize, |n, (_, b)| n.checked_add(b.len()))
        .ok_or_else(limit)?;
    if raw > INPUT_LIMIT / 4 * 3 {
        return Err(limit());
    }
    let mut content = vec![json!({"type":"text","text":request.user})];
    for (mime, bytes) in request.images {
        if bytes.is_empty()
            || !matches!(
                *mime,
                "image/png" | "image/jpeg" | "image/gif" | "image/webp"
            )
        {
            return Err(Failure::new(
                FailureKind::InvalidRequest,
                "Claude requires PNG, JPEG, GIF or WebP image content",
            ));
        }
        content.push(json!({"type":"image","source":{"type":"base64","media_type":mime,"data":base64::engine::general_purpose::STANDARD.encode(bytes)}}));
    }
    json_line(
        &json!({"type":"user","message":{"role":"user","content":content},"parent_tool_use_id":null,"session_id":""}),
    )
}
pub fn complete(config: &Config, request: Request<'_>) -> Result<Completion, Failure> {
    let bytes = input(&request)?;
    let model = request
        .model
        .strip_prefix("claude-agent/")
        .unwrap_or(request.model);
    if model.is_empty()
        || model.len() > 512
        || model.starts_with('-')
        || model.chars().any(char::is_control)
    {
        return Err(Failure::new(
            FailureKind::InvalidRequest,
            "Claude model identifier is invalid",
        ));
    }
    let deadline = process::deadline(request.timeout)?;
    version(config, deadline, request.cancel)?;
    let mut process = start(
        config,
        Some(model),
        request.system,
        deadline,
        request.cancel,
    )?;
    let init = initialize(&mut process, request.cancel)?;
    let catalog = catalog(&init)?;
    let expected = catalog
        .iter()
        .find(|(entry, _)| entry.model.strip_prefix("claude-agent/") == Some(model))
        .and_then(|(_, canonical)| canonical.as_deref())
        .unwrap_or(model)
        .to_owned();
    process.send(Some(bytes), request.cancel)?;
    process.send(None, request.cancel)?;
    let mut evidence = Evidence::default();
    let result = (|| {
        let mut terminal = None;
        while let Some(bytes) = process.next(request.cancel)? {
            let value: Value = serde_json::from_slice(&bytes).map_err(|_| protocol())?;
            if terminal.is_some() {
                if value["type"] == "system"
                    && value["subtype"] == "session_state_changed"
                    && value["state"] == "idle"
                {
                    evidence.event(&value, &expected)?;
                    continue;
                }
                return Err(protocol());
            }
            evidence.event(&value, &expected)?;
            match value["type"].as_str() {
                Some("system") => validate_system(&value)?,
                Some("assistant") => {}
                Some("result") => {
                    if evidence
                        .usage
                        .aggregate
                        .as_ref()
                        .is_some_and(|u| u.by_model.keys().any(|m| m != &expected))
                    {
                        return Err(Failure::new(
                            FailureKind::Unsupported,
                            "Claude reported an unrequested model in aggregate usage",
                        ));
                    }
                    if value["queued_turn_count"].as_u64().is_some_and(|n| n != 0)
                        || value["result_index"].as_u64().is_some_and(|n| n != 0)
                    {
                        return Err(protocol());
                    }
                    if value["permission_denials"]
                        .as_array()
                        .is_some_and(|a| !a.is_empty())
                        || !value["deferred_tool_use"].is_null()
                    {
                        return Err(Failure::new(
                            FailureKind::Permission,
                            "Claude required an unrequested tool or permission",
                        ));
                    }
                    if value["subtype"] != "success" || value["is_error"] != false {
                        let kind = match value["api_error_status"].as_u64() {
                            Some(401 | 403) => FailureKind::Authentication,
                            Some(429) => FailureKind::RateLimit,
                            _ => match value["subtype"].as_str() {
                                Some("error_max_turns" | "error_max_budget_usd") => {
                                    FailureKind::Truncated
                                }
                                _ => FailureKind::Transport,
                            },
                        };
                        return Err(Failure::new(
                            kind,
                            "Claude did not complete the requested response",
                        ));
                    }
                    if value
                        .get("terminal_reason")
                        .filter(|v| !v.is_null())
                        .and_then(Value::as_str)
                        .is_some_and(|s| s != "completed")
                    {
                        return Err(Failure::new(
                            FailureKind::Cancelled,
                            "Claude response did not reach normal completion",
                        ));
                    }
                    if value["stop_reason"] == "max_tokens" {
                        return Err(Failure::new(
                            FailureKind::Truncated,
                            "Claude response was truncated",
                        ));
                    }
                    if value
                        .get("origin")
                        .filter(|v| !v.is_null())
                        .is_some_and(|v| v["kind"] != "human")
                    {
                        return Err(protocol());
                    }
                    let output = text(&value["result"], TEXT_LIMIT)?;
                    if output.trim().is_empty() {
                        return Err(protocol());
                    }
                    terminal = Some(output.to_owned());
                }
                Some("control_request") => {
                    return Err(Failure::new(
                        FailureKind::Permission,
                        "Claude requested a tool, hook or permission callback",
                    ));
                }
                _ => return Err(protocol()),
            }
        }
        if !process.finish(request.cancel)?.success() {
            return Err(Failure::new(
                FailureKind::Transport,
                "Claude exited unsuccessfully after producing output",
            ));
        }
        terminal.ok_or_else(protocol)
    })();
    match result {
        Ok(text) => {
            let mut warnings = vec!["Claude subscription dollar cost is unknown; runtime dollar estimates are not treated as paid subscription charges".into()];
            if evidence.usage.calls.is_empty() {
                warnings.push(
                    if evidence.usage.aggregate.is_some() {
                        "Claude reported aggregate tokens without an observed API request count"
                    } else {
                        "Claude did not report token usage; missing usage is unknown, not zero"
                    }
                    .into(),
                );
            }
            Ok(Completion {
                text,
                usage: evidence.usage,
                warnings,
            })
        }
        Err(mut error) => {
            error.usage = Box::new(evidence.usage);
            Err(error)
        }
    }
}
