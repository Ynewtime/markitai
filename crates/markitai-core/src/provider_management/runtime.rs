//! Why a subscription provider's official runtime lists no models, said as a
//! cause with its next step instead of one sentence for every failure: the
//! runtime is not installed, is not a version the adapter speaks, is not
//! signed in, or failed in some other way. Only the adapters' own fixed
//! messages are repeated; nothing the runtime printed is.
use crate::Error;
use crate::subscription::{self, FailureKind};
use serde_json::{Value, json};
use std::time::Duration;

/// One official runtime and what a person does about it.
pub(super) struct Runtime {
    pub provider: &'static str,
    pub name: &'static str,
    /// The variable that selects the executable.
    pub path_variable: &'static str,
    /// The releases the adapter speaks, as a sentence part: the verified one
    /// or a newer one (used with a one-time warning).
    pub required: &'static str,
    /// Installs the verified release (the runtimes update themselves, so the release is named).
    pub install: &'static str,
    pub login: &'static str,
}

pub(super) const COPILOT: Runtime = Runtime {
    provider: "copilot",
    name: "GitHub Copilot CLI",
    path_variable: "COPILOT_CLI_PATH",
    required: "1.0.90-2 or later",
    install: "npm install -g @github/copilot@1.0.90-2",
    login: "markitai auth copilot login",
};
pub(super) const CLAUDE: Runtime = Runtime {
    provider: "claude-agent",
    name: "Claude Code CLI",
    path_variable: "CLAUDE_CLI_PATH",
    required: "2.1.284 or later",
    install: "npm install -g @anthropic-ai/claude-code@2.1.284",
    login: "markitai auth claude login",
};
pub(super) const CODEX: Runtime = Runtime {
    provider: "chatgpt",
    name: "Codex CLI",
    path_variable: "CODEX_CLI_PATH",
    required: "0.159.0 or later",
    install: "npm install -g @openai/codex@0.159.0",
    login: "markitai auth chatgpt login",
};

/// What went wrong, as the adapters report it.
pub(super) struct Cause {
    /// `None`: the runtime could not even be located or prepared.
    pub kind: Option<FailureKind>,
    pub error: Error,
    /// The sign-in state, when the runtime was asked after a failure.
    pub signed_out: Option<String>,
    /// The release the installed runtime reports, when it could be asked.
    pub installed: Option<String>,
}
impl Cause {
    pub fn setup(error: Error) -> Self {
        Self {
            kind: None,
            error,
            signed_out: None,
            installed: None,
        }
    }
    pub fn failure(kind: FailureKind, error: Error) -> Self {
        Self {
            kind: Some(kind),
            error,
            signed_out: None,
            installed: None,
        }
    }
}

fn message(error: &Error) -> String {
    match error {
        Error::InvalidInput(text)
        | Error::Unsupported(text)
        | Error::Conversion(text)
        | Error::Config(text) => text.clone(),
        other => other.to_string(),
    }
}

/// The discovery result of a runtime that gave no list.
pub(super) fn unavailable(runtime: &Runtime, cause: &Cause) -> Value {
    let name = runtime.name;
    let text = message(&cause.error);
    // (reason, detail, next step kind and command)
    let (reason, detail, step): (&str, String, Option<(&str, &str)>) = match cause.kind {
        None => match &cause.error {
            Error::Unsupported(_) => (
                "runtime_missing",
                format!("{name} was not found on the server's PATH"),
                Some(("install", runtime.install)),
            ),
            Error::InvalidInput(text) if text.ends_with("executable is unavailable") => (
                "runtime_missing",
                format!(
                    "{} points to a file that does not exist",
                    runtime.path_variable
                ),
                Some(("install", runtime.install)),
            ),
            _ => ("runtime_failed", text, None),
        },
        Some(FailureKind::Unsupported) if !text.contains("BYOK") => (
            "runtime_unsupported",
            match &cause.installed {
                Some(version) => format!(
                    "{name} {version} is installed, but the adapter needs {}",
                    runtime.required
                ),
                None => format!(
                    "The installed {name} is not a release the adapter speaks; it needs {}",
                    runtime.required
                ),
            },
            Some(("install", runtime.install)),
        ),
        Some(FailureKind::Authentication) => {
            ("not_signed_in", text, Some(("sign_in", runtime.login)))
        }
        Some(FailureKind::Timeout) => ("timeout", format!("{name} did not answer in time"), None),
        Some(_) => match &cause.signed_out {
            Some(said) => (
                "not_signed_in",
                said.clone(),
                Some(("sign_in", runtime.login)),
            ),
            None => ("runtime_failed", text, None),
        },
    };
    let mut result = json!({
        "provider": runtime.provider, "status": "unavailable", "source": "official_cli",
        "authoritative": false, "cached": false, "stale": false, "models": [],
        "detail": detail, "reason": reason,
        "runtime": {"name": name, "required": runtime.required},
    });
    if let Some(version) = &cause.installed {
        result["runtime"]["installed"] = json!(version);
    }
    if let Some((kind, command)) = step {
        result["next_step"] = json!({"kind": kind, "command": command});
    }
    result
}

/// Models of the Claude runtime, with the cause of a failure explained: its
/// release when it does not match, and its sign-in state when the failure is
/// not one of those.
pub(super) fn claude_models(
    env: &std::collections::HashMap<String, String>,
) -> Result<Vec<subscription::Model>, Cause> {
    use subscription::claude;
    let config = claude::Config::from_env(env).map_err(Cause::setup)?;
    let signed_out = |config: &claude::Config| {
        claude::status(config, Duration::from_secs(4))
            .ok()
            .filter(|status| !status.authenticated)
            .map(|status| {
                status
                    .error
                    .unwrap_or_else(|| "Claude Code CLI is not signed in".into())
            })
    };
    match claude::models(&config, Duration::from_secs(15)) {
        // The catalog needs no login, but a conversion does: a list nothing can use is not offered.
        Ok(models) => match signed_out(&config) {
            Some(said) => Err(Cause {
                kind: Some(FailureKind::Authentication),
                error: Error::Conversion(said),
                signed_out: None,
                installed: None,
            }),
            None => Ok(models),
        },
        Err(failure) => {
            let mut cause = Cause::failure(failure.kind, failure.error);
            match failure.kind {
                FailureKind::Unsupported => {
                    cause.installed = claude::reported_version(&config, Duration::from_secs(4));
                }
                FailureKind::Authentication | FailureKind::Timeout => {}
                _ => cause.signed_out = signed_out(&config),
            }
            Err(cause)
        }
    }
}

/// Discovery for a subscription provider, from the given environment: a stored
/// login can change within one HOME and an account independently of this
/// process, so the runtime is asked every time and nothing is cached.
pub(super) fn discover(provider: &str, env: &std::collections::HashMap<String, String>) -> Value {
    match provider {
        "copilot" => {
            let result = subscription::CopilotConfig::from_env(env)
                .map_err(Cause::setup)
                .and_then(|config| {
                    subscription::models(&config, Duration::from_secs(15))
                        .map_err(|failure| Cause::failure(failure.kind, failure.error))
                });
            match result {
                Ok(models) => {
                    json!({"provider":"copilot","status":"ok","source":"official_cli","authoritative":true,"cached":false,"stale":false,"models":models,"detail":"Models reported by the authenticated official Copilot runtime"})
                }
                Err(cause) => unavailable(&COPILOT, &cause),
            }
        }
        "claude-agent" => match claude_models(env) {
            Ok(models) => {
                json!({"provider":"claude-agent","status":"ok","source":"official_cli","authoritative":true,"cached":false,"stale":false,"models":models,"detail":"Models reported by the authenticated official Claude runtime"})
            }
            Err(cause) => unavailable(&CLAUDE, &cause),
        },
        _ => {
            let result = subscription::chatgpt::Config::from_env(env)
                .map_err(Cause::setup)
                .and_then(|config| {
                    subscription::chatgpt::models(&config, Duration::from_secs(15))
                        .map_err(|failure| Cause::failure(failure.kind, failure.error))
                });
            match result {
                Ok(models) => {
                    json!({"provider":"chatgpt","status":"ok","source":"official_cli","authoritative":true,"cached":false,"stale":false,"models":models,"detail":"Models the installed Codex offers for the signed-in account, its default first"})
                }
                Err(cause) => unavailable(&CODEX, &cause),
            }
        }
    }
}
