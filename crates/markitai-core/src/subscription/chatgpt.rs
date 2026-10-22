//! Bounded document requests through a pinned, installed official Codex runtime.
mod events;
#[cfg(test)]
mod tests;

use super::supervisor::{Admission, Messages, ProcessFailure, Runtime};
use super::{
    AuthStatus, FailureKind, Model, Release, Request, compare_release, note_newer_release,
};
use crate::process_groups::Workspace;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

pub const CLI_VERSION: &str = "0.159.0";
/// The model of the catalog compiled into this build, used when the
/// installed Codex cannot list its models.
pub const BUNDLED_MODEL: &str = "gpt-5.5";
/// The model offered when the installed Codex cannot be asked (Python
/// markitai's default); otherwise the first model Codex lists is the default.
pub const FALLBACK_MODEL: &str = "gpt-6-luna";
const CATALOG_LIMIT: usize = 8 * 1024 * 1024;
const INPUT_LIMIT: usize = 10 * 1024 * 1024;
const IMAGE_LIMIT: usize = 32 * 1024 * 1024;
const TEXT_LIMIT: usize = 8 * 1024 * 1024;
const CATALOG: &[u8] = include_bytes!("chatgpt/models.json");

#[derive(Clone)]
pub struct Config {
    executable: PathBuf,
    environment: HashMap<String, String>,
}
impl Config {
    pub fn from_env(env: &HashMap<String, String>) -> crate::Result<Self> {
        let executable = super::locate(
            env,
            "CODEX_CLI_PATH",
            "codex",
            "Codex",
            "The supported official Codex CLI is not installed",
        )?;
        // Preserve official login locations without reading, copying or replacing
        // credentials. API keys, endpoints, proxies and remote overrides are excluded.
        let environment = super::retain(env, &["CODEX_HOME"]);
        Ok(Self {
            executable,
            environment,
        })
    }
    pub fn executable(&self) -> &Path {
        &self.executable
    }
}

/// Official exec reports turn totals, not a reliable underlying request count.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TokenTotals {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub reasoning_output_tokens: u64,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct UsageEvidence {
    pub aggregate: Option<TokenTotals>,
}
#[derive(Debug)]
pub struct Completion {
    pub text: String,
    pub usage: UsageEvidence,
}
#[derive(Debug)]
pub struct Failure {
    pub kind: FailureKind,
    pub error: crate::Error,
    pub usage: UsageEvidence,
}
impl Failure {
    fn new(kind: FailureKind, message: &'static str) -> Self {
        Self {
            kind,
            error: crate::Error::Conversion(message.into()),
            usage: UsageEvidence::default(),
        }
    }
}
impl From<ProcessFailure> for Failure {
    fn from(failure: ProcessFailure) -> Self {
        Self::new(failure.kind, failure.message)
    }
}

/// Login status is read from stderr, so its first 64 KiB are kept.
static RUNTIME: Runtime = Runtime {
    name: "chatgpt",
    environment: &[],
    stderr_kept: 64 * 1024,
    messages: Messages {
        unsupported: "Bounded ChatGPT process-tree cleanup is unavailable on this platform",
        cancelled: "ChatGPT request was cancelled",
        timeout: "ChatGPT runtime deadline exceeded; completion is unknown",
        malformed: "ChatGPT output is malformed or exceeds its limits",
        limit: "ChatGPT runtime resource limit exceeded",
        transport: "ChatGPT runtime process communication failed",
    },
    admission: Admission::new(),
};
fn protocol() -> Failure {
    Failure::new(
        FailureKind::Protocol,
        "Codex returned an invalid or conflicting protocol response",
    )
}
fn limit() -> Failure {
    Failure::new(
        FailureKind::ResourceLimit,
        "Codex request or response exceeds its resource limits",
    )
}

fn private_file(workspace: &Workspace, name: &str, bytes: &[u8]) -> Result<PathBuf, Failure> {
    workspace.file(name, bytes).map_err(|_| protocol())
}
fn config_arg(args: &mut Vec<OsString>, key: &str, value: &Value) {
    args.push("-c".into());
    args.push(format!("{key}={value}").into());
}
fn small_command(
    config: &Config,
    args: &[OsString],
    deadline: Instant,
    cancel: Option<&AtomicBool>,
) -> Result<(Vec<u8>, Vec<u8>, std::process::ExitStatus), Failure> {
    bounded_command(config, args, deadline, cancel, 64 * 1024)
}
fn bounded_command(
    config: &Config,
    args: &[OsString],
    deadline: Instant,
    cancel: Option<&AtomicBool>,
    bound: usize,
) -> Result<(Vec<u8>, Vec<u8>, std::process::ExitStatus), Failure> {
    let mut child = RUNTIME.spawn(
        &config.executable,
        &config.environment,
        args,
        RUNTIME.workspace()?,
        deadline,
        cancel,
    )?;
    child.send(None, cancel)?;
    let mut output = Vec::new();
    while let Some(line) = child.next(cancel)? {
        if output.len().saturating_add(line.len()) > bound {
            return Err(limit());
        }
        output.extend_from_slice(&line);
    }
    let exit = child.finish(cancel)?;
    Ok((output, child.stderr(), exit))
}
/// The installed Codex release: the verified 0.159.0, or a newer one, which
/// is used with a one-time warning (its events are still validated one by
/// one, so an incompatible change fails as a protocol error). An older or
/// unreadable release is refused.
fn version(
    config: &Config,
    deadline: Instant,
    cancel: Option<&AtomicBool>,
) -> Result<(String, Release), Failure> {
    let (out, _, exit) = small_command(config, &["--version".into()], deadline, cancel)?;
    let installed = std::str::from_utf8(&out)
        .ok()
        .map(str::trim)
        .and_then(|text| text.strip_prefix("codex-cli "))
        .filter(|_| exit.success())
        .map(|installed| (installed, compare_release(installed, CLI_VERSION, 0)))
        .filter(|(_, release)| *release != Release::Older);
    let Some((installed, release)) = installed else {
        return Err(Failure::new(
            FailureKind::Unsupported,
            "Installed Codex CLI is older than the supported release 0.159.0",
        ));
    };
    Ok((installed.to_owned(), release))
}
fn signed_in(
    config: &Config,
    deadline: Instant,
    cancel: Option<&AtomicBool>,
) -> Result<bool, Failure> {
    // Status has no ignore-user-config flag. It only checks official login state;
    // never print its raw stderr, which can contain a partially displayed API key.
    let args = ["login".into(), "status".into()];
    let (out, err, exit) = small_command(config, &args, deadline, cancel)?;
    let err = std::str::from_utf8(&err).map_err(|_| protocol())?.trim();
    if !out.is_empty() {
        return Err(protocol());
    }
    if exit.success() && err == "Logged in using ChatGPT" {
        return Ok(true);
    }
    if (exit.code() == Some(1) && err == "Not logged in")
        || (exit.success() && err.starts_with("Logged in using "))
    {
        return Ok(false);
    }
    Err(Failure::new(
        FailureKind::Authentication,
        "Official Codex login status could not be verified",
    ))
}
pub fn status(config: &Config, timeout: Duration) -> Result<AuthStatus, Failure> {
    let deadline = RUNTIME.deadline(timeout)?;
    let (installed, release) = version(config, deadline, None)?;
    let authenticated = signed_in(config, deadline, None)?;
    Ok(AuthStatus {
        provider: "chatgpt",
        authenticated,
        user: None,
        expires_at: None,
        error: (!authenticated)
            .then(|| "Codex CLI is not signed in with a ChatGPT subscription".into()),
        details: json!({"source":"official_cli","cli_version":installed,"supported_from":CLI_VERSION,
            "newer_than_verified":release == Release::Newer,"native_adapter":true,"account_identity_available":false}),
    })
}
/// Whether `name` can be given to Codex as a model: letters, digits and
/// `.`, `-`, `_`, `:`, at most 128 bytes.
pub fn valid_model(name: &str) -> bool {
    (1..=128).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b':'))
}

/// The models the installed Codex offers, its default first (Codex's own
/// `priority` order, which puts the newest current model first), as `codex
/// debug models` lists them: with `refresh`, Codex updates its list for the
/// signed-in account; without, it reads the catalog shipped with the binary.
/// Hidden entries (internal review models) are left out. `None` when this
/// Codex cannot tell: the command fails or prints no catalog.
fn offered(
    config: &Config,
    deadline: Instant,
    cancel: Option<&AtomicBool>,
    refresh: bool,
) -> Result<Option<Vec<Value>>, Failure> {
    let mut args: Vec<OsString> = vec!["debug".into(), "models".into()];
    if !refresh {
        args.push("--bundled".into());
    }
    let (out, _, exit) = bounded_command(config, &args, deadline, cancel, CATALOG_LIMIT)?;
    if !exit.success() {
        return Ok(None);
    }
    let Some(models) = serde_json::from_slice::<Value>(&out)
        .ok()
        .and_then(|value| value.get("models").and_then(Value::as_array).cloned())
    else {
        return Ok(None);
    };
    let mut listed: Vec<Value> = models
        .into_iter()
        .filter(|entry| {
            entry["visibility"] == "list" && entry["slug"].as_str().is_some_and(valid_model)
        })
        .collect();
    listed.sort_by_key(|entry| entry["priority"].as_i64().unwrap_or(i64::MAX));
    Ok(Some(listed))
}

/// The bundled catalog of the installed Codex, asked once per executable,
/// release and Codex home in a process.
fn bundled(
    config: &Config,
    installed: &str,
    deadline: Instant,
    cancel: Option<&AtomicBool>,
) -> Result<Option<Vec<Value>>, Failure> {
    type Known = HashMap<(PathBuf, String, Option<String>), Option<Vec<Value>>>;
    static KNOWN: std::sync::OnceLock<std::sync::Mutex<Known>> = std::sync::OnceLock::new();
    let key = (
        config.executable.clone(),
        installed.to_owned(),
        config.environment.get("CODEX_HOME").cloned(),
    );
    let known = KNOWN.get_or_init(Default::default);
    if let Some(list) = known.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(list.clone());
    }
    let list = offered(config, deadline, cancel, false)?;
    known
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, list.clone());
    Ok(list)
}

/// `entry` as the one model of the catalog an exec reads, with the tool
/// surfaces this adapter keeps off switched off in the entry too (the
/// configuration already disables them): no patch tool, no code mode, no
/// experimental tools, no sub-agents, no search tool.
fn restricted(entry: &Value) -> Value {
    let mut entry = entry.clone();
    for (key, value) in [
        ("apply_patch_tool_type", Value::Null),
        ("tool_mode", Value::Null),
        ("experimental_supported_tools", json!([])),
        ("multi_agent_version", Value::Null),
        ("multi_agent_reasoning_effort", Value::Null),
        ("supports_search_tool", json!(false)),
    ] {
        if let Some(object) = entry.as_object_mut() {
            object.insert(key.into(), value);
        }
    }
    json!({"models":[entry]})
}

/// The catalog an exec of `model` reads: the installed Codex's own entry,
/// restricted; the compiled catalog for its model when Codex cannot list
/// its models; `None` (Codex's own catalog, with a one-time warning) for
/// another model then. A model Codex can list and does not offer is refused.
fn catalog_for(
    config: &Config,
    installed: &str,
    model: &str,
    deadline: Instant,
    cancel: Option<&AtomicBool>,
) -> Result<Option<Vec<u8>>, Failure> {
    let find = |list: &[Value]| {
        list.iter()
            .find(|entry| entry["slug"] == model)
            .map(restricted)
    };
    let shipped = bundled(config, installed, deadline, cancel)?;
    if let Some(entry) = shipped.as_deref().and_then(find) {
        return Ok(Some(serde_json::to_vec(&entry).map_err(|_| protocol())?));
    }
    // A model newer than the binary is known only to the account's list.
    match offered(config, deadline, cancel, true)? {
        Some(account) => match find(&account) {
            Some(entry) => Ok(Some(serde_json::to_vec(&entry).map_err(|_| protocol())?)),
            None => {
                let names: Vec<String> = account
                    .iter()
                    .filter_map(|entry| entry["slug"].as_str())
                    .take(12)
                    .map(|slug| format!("chatgpt/{slug}"))
                    .collect();
                Err(Failure {
                    kind: FailureKind::Unsupported,
                    error: crate::Error::Unsupported(format!(
                        "The installed Codex does not offer chatgpt/{model}; it offers {}",
                        names.join(", ")
                    )),
                    usage: UsageEvidence::default(),
                })
            }
        },
        None if model == BUNDLED_MODEL => Ok(Some(CATALOG.to_vec())),
        None => {
            note_unvalidated_model(model);
            Ok(None)
        }
    }
}

/// The one-time notice that a model was sent without Codex confirming it
/// offers the model.
fn note_unvalidated_model(model: &str) {
    static NOTED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    if !crate::llm::document_scoped() {
        return;
    }
    let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
    if noted.iter().any(|seen| seen == model) {
        return;
    }
    noted.push(model.to_owned());
    crate::llm::note_document_warning(format!(
        "The installed Codex could not list its models, so chatgpt/{model} is sent as named, unvalidated"
    ));
}

/// The model `chatgpt/` defaults to: the first one the installed Codex
/// lists (its default, the newest current model), else [`FALLBACK_MODEL`].
pub fn default_model(config: &Config, timeout: Duration) -> String {
    RUNTIME
        .deadline(timeout)
        .ok()
        .and_then(|deadline| {
            let (installed, _) = version(config, deadline, None).ok()?;
            bundled(config, &installed, deadline, None).ok().flatten()
        })
        .and_then(|list| {
            list.first()
                .and_then(|entry| entry["slug"].as_str())
                .map(str::to_owned)
        })
        .unwrap_or_else(|| FALLBACK_MODEL.into())
}

/// The models the signed-in account can use through the installed Codex,
/// its default first; the compiled catalog's model when Codex cannot list
/// them.
pub fn models(config: &Config, timeout: Duration) -> Result<Vec<Model>, Failure> {
    if !status(config, timeout)?.authenticated {
        return Err(Failure::new(
            FailureKind::Authentication,
            "Codex CLI is not signed in with a ChatGPT subscription",
        ));
    }
    let deadline = RUNTIME.deadline(timeout)?;
    let (installed, _) = version(config, deadline, None)?;
    let listed = match offered(config, deadline, None, true)? {
        Some(list) => Some(list),
        None => bundled(config, &installed, deadline, None)?,
    };
    let Some(listed) = listed else {
        return Ok(vec![Model {
            model: format!("chatgpt/{BUNDLED_MODEL}"),
            label: "GPT-5.5".into(),
            supports_vision: true,
        }]);
    };
    Ok(listed
        .iter()
        .filter_map(|entry| {
            let slug = entry["slug"].as_str()?;
            Some(Model {
                model: format!("chatgpt/{slug}"),
                label: entry["display_name"]
                    .as_str()
                    .filter(|label| label.len() <= 128)
                    .unwrap_or(slug)
                    .into(),
                supports_vision: entry["input_modalities"]
                    .as_array()
                    .is_some_and(|kinds| kinds.iter().any(|kind| kind == "image")),
            })
        })
        .collect())
}
fn exec_args(
    workspace: &Workspace,
    system: &str,
    images: &[(&str, &[u8])],
    model: &str,
    catalog: Option<&[u8]>,
) -> Result<Vec<OsString>, Failure> {
    let prompt = private_file(workspace, "system.txt", system.as_bytes())?;
    let mut args = Vec::new();
    if let Some(catalog) = catalog {
        let catalog = private_file(workspace, "models.json", catalog)?;
        config_arg(&mut args, "model_catalog_json", &json!(catalog));
    }
    for (name, value) in [
        ("model_instructions_file", json!(prompt)),
        ("model_provider", json!("openai")),
        ("sqlite_home", json!(workspace.directory("state"))),
        ("log_dir", json!(workspace.directory("log"))),
    ] {
        config_arg(&mut args, name, &value);
    }
    // No forced_login_method: the official implementation may log out an
    // incompatible stored login. The separate status check has a documented race
    // if another process replaces the account before exec loads its credentials.
    for name in [
        "analytics.enabled",
        "feedback.enabled",
        "features.enable_request_compression",
        "features.shell_snapshot",
        "features.shell_snapshot_v2",
        "tools.experimental_request_user_input.enabled",
        "skills.bundled.enabled",
        "skills.include_instructions",
        "tools.update_plan.enabled",
    ] {
        config_arg(&mut args, name, &json!(false));
    }
    for feature in [
        "shell_tool",
        "apply_patch_freeform",
        "multi_agent",
        "multi_agent_v2",
        "code_mode",
        "code_mode_host",
        "apps",
        "plugins",
        "remote_plugin",
        "recommended_plugins",
        "codex_hooks",
        "hooks",
        "plugin_hooks",
        "tool_search",
        "search_tool",
        "memory_tool",
        "view_image",
        "request_permissions_tool",
        "skill_search",
        "skill_mcp_dependency_install",
        "sleep_tool",
        "current_time_reminder",
        "token_budget",
        "goals",
    ] {
        config_arg(&mut args, &format!("features.{feature}"), &json!(false));
    }
    config_arg(
        &mut args,
        "features.skip_host_skill_discovery",
        &json!(true),
    );
    config_arg(&mut args, "project_doc_max_bytes", &json!(0));
    config_arg(&mut args, "web_search", &json!("disabled"));
    config_arg(&mut args, "mcp_servers", &json!({}));
    args.extend(
        [
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--json",
            "--color",
            "never",
            "--model",
            model,
        ]
        .into_iter()
        .map(OsString::from),
    );
    for (index, (mime, bytes)) in images.iter().enumerate() {
        let extension = match (*mime, image::guess_format(bytes).ok()) {
            ("image/png", Some(image::ImageFormat::Png)) => "png",
            ("image/jpeg", Some(image::ImageFormat::Jpeg)) => "jpg",
            ("image/webp", Some(image::ImageFormat::WebP)) => "webp",
            _ => {
                return Err(Failure::new(
                    FailureKind::InvalidRequest,
                    "Codex image MIME does not match a supported image format",
                ));
            }
        };
        let path = private_file(workspace, &format!("image-{index:04}.{extension}"), bytes)?;
        args.push("--image".into());
        args.push(path.into_os_string());
    }
    args.push("-".into());
    Ok(args)
}
pub fn complete(config: &Config, request: Request<'_>) -> Result<Completion, Failure> {
    let deadline = RUNTIME.deadline(request.timeout)?;
    let model = request
        .model
        .strip_prefix("chatgpt/")
        .unwrap_or(request.model);
    if !valid_model(model) {
        return Err(Failure::new(
            FailureKind::InvalidRequest,
            "ChatGPT model name is invalid",
        ));
    }
    if request.system.len().saturating_add(request.user.len()) > INPUT_LIMIT
        || request.system.contains('\0')
        || request.user.contains('\0')
        || request.images.len() > 100
    {
        return Err(limit());
    }
    let image_bytes = request
        .images
        .iter()
        .try_fold(0usize, |n, (_, bytes)| n.checked_add(bytes.len()))
        .ok_or_else(limit)?;
    if image_bytes > IMAGE_LIMIT {
        return Err(limit());
    }
    let (installed, release) = version(config, deadline, request.cancel)?;
    if release == Release::Newer {
        note_newer_release("Codex", &installed, CLI_VERSION);
    }
    if !signed_in(config, deadline, request.cancel)? {
        return Err(Failure::new(
            FailureKind::Authentication,
            "Codex CLI is not signed in with a ChatGPT subscription",
        ));
    }
    let catalog = catalog_for(config, &installed, model, deadline, request.cancel)?;
    let workspace = RUNTIME.workspace()?;
    let args = exec_args(
        &workspace,
        request.system,
        request.images,
        model,
        catalog.as_deref(),
    )?;
    let mut child = RUNTIME.spawn(
        &config.executable,
        &config.environment,
        &args,
        workspace,
        deadline,
        request.cancel,
    )?;
    let mut events = events::Events::default();
    let outcome = (|| {
        child.send(Some(request.user.as_bytes().to_vec()), request.cancel)?;
        child.send(None, request.cancel)?;
        while let Some(line) = child.next(request.cancel)? {
            let value: Value = serde_json::from_slice(&line).map_err(|_| protocol())?;
            events.accept(&value)?;
        }
        let exit = child.finish(request.cancel)?;
        events.finish(exit.success())
    })();
    let usage = events.usage;
    match outcome {
        Ok(text) => Ok(Completion { text, usage }),
        Err(mut error) => {
            error.usage = usage;
            Err(error)
        }
    }
}
