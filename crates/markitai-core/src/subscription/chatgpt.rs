//! Bounded document requests through a pinned, installed official Codex runtime.
mod events;
mod process;
#[cfg(test)]
mod tests;

use super::{AuthStatus, FailureKind, Model, Request};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

pub const CLI_VERSION: &str = "0.159.0";
pub const MODEL: &str = "gpt-5.5";
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

fn private_file(workspace: &Path, name: &str, bytes: &[u8]) -> Result<PathBuf, Failure> {
    use std::io::Write;
    let path = workspace.join(name);
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
    let mut child = process::Process::spawn(config, args, process::workspace()?, deadline, cancel)?;
    child.send(None, cancel)?;
    let mut output = Vec::new();
    while let Some(line) = child.next(cancel)? {
        if output.len().saturating_add(line.len()) > 64 * 1024 {
            return Err(limit());
        }
        output.extend_from_slice(&line);
    }
    let exit = child.finish(cancel)?;
    Ok((output, child.stderr(), exit))
}
fn version(config: &Config, deadline: Instant, cancel: Option<&AtomicBool>) -> Result<(), Failure> {
    let (out, _, exit) = small_command(config, &["--version".into()], deadline, cancel)?;
    if !exit.success() || std::str::from_utf8(&out).ok().map(str::trim) != Some("codex-cli 0.159.0")
    {
        return Err(Failure::new(
            FailureKind::Unsupported,
            "Installed Codex CLI does not match the supported protocol version",
        ));
    }
    Ok(())
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
    let deadline = process::deadline(timeout)?;
    version(config, deadline, None)?;
    let authenticated = signed_in(config, deadline, None)?;
    Ok(AuthStatus {
        provider: "chatgpt",
        authenticated,
        user: None,
        expires_at: None,
        error: (!authenticated)
            .then(|| "Codex CLI is not signed in with a ChatGPT subscription".into()),
        details: json!({"source":"official_cli","cli_version":CLI_VERSION,"native_adapter":true,"account_identity_available":false}),
    })
}
pub fn models(config: &Config, timeout: Duration) -> Result<Vec<Model>, Failure> {
    if !status(config, timeout)?.authenticated {
        return Err(Failure::new(
            FailureKind::Authentication,
            "Codex CLI is not signed in with a ChatGPT subscription",
        ));
    }
    // This is the tested adapter allowlist, not account entitlement discovery.
    Ok(vec![Model {
        model: format!("chatgpt/{MODEL}"),
        label: "GPT-5.5 (pinned Codex adapter)".into(),
        supports_vision: true,
    }])
}
fn exec_args(
    workspace: &Path,
    system: &str,
    images: &[(&str, &[u8])],
) -> Result<Vec<OsString>, Failure> {
    let catalog = private_file(workspace, "models.json", CATALOG)?;
    let prompt = private_file(workspace, "system.txt", system.as_bytes())?;
    let mut args = Vec::new();
    for (name, value) in [
        ("model_catalog_json", json!(catalog)),
        ("model_instructions_file", json!(prompt)),
        ("model_provider", json!("openai")),
        ("sqlite_home", json!(workspace.join("state"))),
        ("log_dir", json!(workspace.join("log"))),
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
            MODEL,
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
    let deadline = process::deadline(request.timeout)?;
    if request
        .model
        .strip_prefix("chatgpt/")
        .unwrap_or(request.model)
        != MODEL
    {
        return Err(Failure::new(
            FailureKind::Unsupported,
            "The Codex adapter has not validated this model's tool capabilities",
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
    version(config, deadline, request.cancel)?;
    if !signed_in(config, deadline, request.cancel)? {
        return Err(Failure::new(
            FailureKind::Authentication,
            "Codex CLI is not signed in with a ChatGPT subscription",
        ));
    }
    let workspace = process::workspace()?;
    let args = exec_args(workspace.path(), request.system, request.images)?;
    let mut child = process::Process::spawn(config, &args, workspace, deadline, request.cancel)?;
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
