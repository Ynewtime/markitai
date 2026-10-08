//! Optional subscription providers communicate with an installed official runtime.
mod copilot;
#[cfg(test)]
pub(crate) mod fake_runtime;
mod line;
mod process;
#[cfg(test)]
mod tests;

pub use copilot::{complete, models, status};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

pub const COPILOT_PROTOCOL: u64 = 3;
pub const COPILOT_CLI_VERSION: &str = "1.0.90-2";

/// Credentials are deliberately excluded from Debug and serialization.
#[derive(Clone)]
pub struct CopilotConfig {
    executable: PathBuf,
    environment: HashMap<String, String>,
    token: Option<String>,
}
impl CopilotConfig {
    pub fn from_env(env: &HashMap<String, String>) -> crate::Result<Self> {
        let executable = locate(
            env,
            "COPILOT_CLI_PATH",
            "copilot",
            "Copilot",
            "Copilot CLI is unavailable; install the supported official runtime",
        )?;
        let token = ["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]
            .iter()
            .find_map(|key| variable(env, key).filter(|value| !value.is_empty()))
            .map(str::to_owned);
        if token
            .as_ref()
            .is_some_and(|value| value.len() > 16 * 1024 || value.contains(['\r', '\n', '\0']))
        {
            return Err(crate::Error::InvalidInput(
                "Copilot credential is invalid or oversized".into(),
            ));
        }
        // Retain OS process prerequisites and the explicitly selected official auth
        // home. Other model-provider variables must not silently turn this into BYOK.
        let environment = retain(env, &["COPILOT_HOME", "COPILOT_CACHE_HOME"]);
        Ok(Self {
            executable,
            environment,
            token,
        })
    }
    pub fn executable(&self) -> &std::path::Path {
        &self.executable
    }
}

/// The value of `name` in `env`, matched as the platform matches variable
/// names: Windows ignores case, so `Path` answers for `PATH`.
pub(crate) fn variable<'a>(env: &'a HashMap<String, String>, name: &str) -> Option<&'a str> {
    env.get(name)
        .or_else(|| {
            cfg!(windows)
                .then(|| {
                    env.iter()
                        .find(|(key, _)| key.eq_ignore_ascii_case(name))
                        .map(|(_, value)| value)
                })
                .flatten()
        })
        .map(String::as_str)
}

/// Variables every official runtime needs to start and to find its own
/// configuration: the search path, home and temporary directories, locale.
const PROCESS_PREREQUISITES: &[&str] = &[
    "HOME",
    "PATH",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "SystemRoot",
    "SYSTEMROOT",
    "WINDIR",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
];

/// What Windows command shims and Node.js runtimes also read: the command
/// processor, the extensions it searches, and the standard system folders.
/// None of these holds a credential.
const WINDOWS_PREREQUISITES: &[&str] = &[
    "PATHEXT",
    "ComSpec",
    "SystemDrive",
    "ProgramData",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramW6432",
    "CommonProgramFiles",
    "HOMEDRIVE",
    "HOMEPATH",
    "USERNAME",
    "COMPUTERNAME",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "OS",
];

/// The runtime environment: the process prerequisites and the runtime's own
/// `names`, each taken from `env` or else from this process.
pub(crate) fn retain(env: &HashMap<String, String>, names: &[&str]) -> HashMap<String, String> {
    let windows: &[&str] = if cfg!(windows) {
        WINDOWS_PREREQUISITES
    } else {
        &[]
    };
    let mut environment = HashMap::new();
    for &name in PROCESS_PREREQUISITES.iter().chain(windows).chain(names) {
        if let Some(value) = variable(env, name)
            .map(str::to_owned)
            .or_else(|| std::env::var(name).ok())
        {
            environment.insert(name.into(), value);
        }
    }
    environment
}

/// An official runtime: the path in `explicit`, else `program` on PATH. On
/// Windows both honour PATHEXT, so an npm `.cmd` shim is found and started
/// through the command processor. `label` names the runtime in errors and
/// `missing` says how to install it.
pub(crate) fn locate(
    env: &HashMap<String, String>,
    explicit: &str,
    program: &str,
    label: &str,
    missing: &'static str,
) -> crate::Result<PathBuf> {
    use std::ffi::OsStr;
    let pathext = variable(env, "PATHEXT").map(OsStr::new);
    let path = match variable(env, explicit) {
        Some(path) => crate::process_groups::configured_program(Path::new(path), pathext),
        None => crate::process_groups::find_program(
            &[program],
            variable(env, "PATH").map(OsStr::new),
            pathext,
        )
        .ok_or_else(|| crate::Error::Unsupported(missing.into()))?,
    };
    let executable = path
        .canonicalize()
        .map_err(|_| crate::Error::InvalidInput(format!("{label} executable is unavailable")))?;
    if !executable.is_file() {
        return Err(crate::Error::InvalidInput(format!(
            "{label} executable must be a regular file"
        )));
    }
    if cfg!(windows) && !crate::process_groups::launchable(&executable) {
        return Err(crate::Error::InvalidInput(format!(
            "{label} executable must be a .exe, .com, .bat or .cmd file"
        )));
    }
    Ok(crate::process_groups::plain(executable))
}

/// An official-runtime command found on PATH or at its configured location.
/// Presence is not verification of its version, protocol or login state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstalledRuntime {
    pub provider: &'static str,
    pub label: &'static str,
}

/// Look only for executable files. No command is started and no authentication
/// configuration, token or runtime state is read.
pub fn installed_runtimes(env: &HashMap<String, String>) -> Vec<InstalledRuntime> {
    [
        ("copilot", "Copilot CLI", "COPILOT_CLI_PATH", "copilot"),
        ("claude", "Claude Code", "CLAUDE_CLI_PATH", "claude"),
        ("chatgpt", "Codex CLI", "CODEX_CLI_PATH", "codex"),
    ]
    .into_iter()
    .filter_map(|(provider, label, explicit, program)| {
        let path = locate(env, explicit, program, label, "Runtime is not installed").ok()?;
        crate::process_groups::launchable(&path).then_some(InstalledRuntime { provider, label })
    })
    .collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct AuthStatus {
    pub provider: &'static str,
    pub authenticated: bool,
    pub user: Option<String>,
    pub expires_at: Option<String>,
    pub error: Option<String>,
    pub details: serde_json::Value,
}
#[derive(Clone, Debug, Serialize)]
pub struct Model {
    pub model: String,
    pub label: String,
    pub supports_vision: bool,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct UsageEvidence {
    /// One entry for each distinct runtime-reported API call, not an estimate.
    pub calls: Vec<ObservedCall>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ObservedCall {
    pub model: String,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    Authentication,
    Unsupported,
    Transport,
    Timeout,
    Cancelled,
    Protocol,
    Permission,
    ResourceLimit,
    Refusal,
    Truncated,
    RateLimit,
    InvalidRequest,
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
#[derive(Debug)]
pub struct Completion {
    pub text: String,
    pub usage: UsageEvidence,
}
pub struct Request<'a> {
    pub model: &'a str,
    pub system: &'a str,
    pub user: &'a str,
    pub images: &'a [(&'a str, &'a [u8])],
    pub timeout: Duration,
    pub cancel: Option<&'a AtomicBool>,
}

pub mod claude;

pub mod chatgpt;
