//! Authentication stays with the official runtime; no token store is inspected here.
use super::{CliResult, runtime};
use clap::Subcommand;
use markitai_core::{config, subscription};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Subcommand, Debug, Clone)]
pub(super) enum Command {
    /// GitHub Copilot through the installed official CLI.
    #[command(
        after_help = "Examples:\n  markitai auth copilot status           Is the runtime signed in?\n  markitai auth copilot status --json    Machine-readable status\n  markitai auth copilot login            Sign in through the official runtime"
    )]
    Copilot {
        #[command(subcommand)]
        command: Option<Action>,
    },
    /// Claude subscription adapter status.
    #[command(
        after_help = "Examples:\n  markitai auth claude status           Is the runtime signed in?\n  markitai auth claude status --json    Machine-readable status\n  markitai auth claude login            Sign in through the official runtime"
    )]
    Claude {
        #[command(subcommand)]
        command: Option<Action>,
    },
    /// ChatGPT subscription adapter status.
    #[command(
        after_help = "Examples:\n  markitai auth chatgpt status           Is the runtime signed in?\n  markitai auth chatgpt status --json    Machine-readable status\n  markitai auth chatgpt login            Sign in through the official runtime"
    )]
    Chatgpt {
        #[command(subcommand)]
        command: Option<Action>,
    },
}
#[derive(Subcommand, Debug, Clone)]
pub(super) enum Action {
    /// Inspect existing authentication without initiating login.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Open the official runtime's interactive login flow.
    Login,
}

pub(super) fn copilot_status(env: &HashMap<String, String>) -> subscription::AuthStatus {
    let result = subscription::CopilotConfig::from_env(env).and_then(|cfg| {
        subscription::status(&cfg, Duration::from_secs(15)).map_err(|failure| failure.error)
    });
    result.unwrap_or_else(|error| subscription::AuthStatus {
        provider: "copilot",
        authenticated: false,
        user: None,
        expires_at: None,
        error: Some(error.to_string()),
        details: serde_json::json!({"source":"official_cli","verification":"unavailable"}),
    })
}
pub(super) fn claude_status(env: &HashMap<String, String>) -> subscription::AuthStatus {
    subscription::claude::Config::from_env(env)
        .and_then(|cfg| subscription::claude::status(&cfg, Duration::from_secs(15)).map_err(|failure| failure.error))
        .unwrap_or_else(|error| subscription::AuthStatus {
            provider: "claude-agent", authenticated: false, user: None, expires_at: None,
            error: Some(error.to_string()), details: serde_json::json!({"source":"official_cli","verification":"unavailable","native_adapter":true}),
        })
}
pub(super) fn chatgpt_status(env: &HashMap<String, String>) -> subscription::AuthStatus {
    subscription::chatgpt::Config::from_env(env)
        .and_then(|cfg| subscription::chatgpt::status(&cfg, Duration::from_secs(15)).map_err(|failure| failure.error))
        .unwrap_or_else(|error| subscription::AuthStatus {
            provider: "chatgpt", authenticated: false, user: None, expires_at: None,
            error: Some(error.to_string()), details: serde_json::json!({"source":"official_cli","verification":"unavailable","native_adapter":true}),
        })
}

fn display(status: &subscription::AuthStatus) {
    println!(
        "{}: {}",
        status.provider,
        if status.authenticated {
            "authenticated"
        } else {
            "unavailable"
        }
    );
    if let Some(user) = &status.user {
        println!("  Account: {}", terminal_text(user));
    }
    if let Some(error) = &status.error {
        println!("  {}", terminal_text(error));
        // A signed-out runtime is one command away from working.
        if error.contains("not signed in") {
            let name = status
                .provider
                .strip_suffix("-agent")
                .unwrap_or(status.provider);
            println!("  Next: markitai auth {name} login");
        }
    }
}
fn terminal_text(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !ch.is_control())
        .take(2048)
        .collect()
}

pub(super) fn run(command: Option<&Command>) -> CliResult<i32> {
    let env = config::environment();
    let Some(command) = command else {
        display(&claude_status(&env));
        display(&chatgpt_status(&env));
        display(&copilot_status(&env));
        return Ok(0);
    };
    let (provider, action) = match command {
        Command::Copilot { command } => ("copilot", command),
        Command::Claude { command } => ("claude-agent", command),
        Command::Chatgpt { command } => ("chatgpt", command),
    };
    if matches!(action, Some(Action::Login)) {
        return login(&env, provider);
    }
    let status = if provider == "copilot" {
        copilot_status(&env)
    } else if provider == "claude-agent" {
        claude_status(&env)
    } else {
        chatgpt_status(&env)
    };
    if matches!(action, Some(Action::Status { json: true })) {
        let mut value = serde_json::to_value(status).map_err(runtime)?;
        if provider == "claude-agent" {
            value["cli_path"] = subscription::claude::Config::from_env(&env)
                .ok()
                .map(|cfg| serde_json::json!(cfg.executable()))
                .unwrap_or(serde_json::Value::Null);
            value["sdk_installed"] = serde_json::json!(false);
        }
        println!("{}", serde_json::to_string_pretty(&value).map_err(runtime)?);
        // The reference's JSON status exits zero even for unavailable authentication.
        Ok(0)
    } else {
        display(&status);
        Ok(if status.authenticated { 0 } else { 1 })
    }
}

/// The official runtime's interactive login command: its own arguments, the
/// variables a terminal program needs, and the runtime's own configuration.
/// Nothing else from this environment is passed on.
fn login_command(
    env: &HashMap<String, String>,
    provider: &str,
) -> CliResult<std::process::Command> {
    let executable = if provider == "claude-agent" {
        subscription::claude::Config::from_env(env)
            .map_err(runtime)?
            .executable()
            .to_owned()
    } else if provider == "chatgpt" {
        subscription::chatgpt::Config::from_env(env)
            .map_err(runtime)?
            .executable()
            .to_owned()
    } else {
        subscription::CopilotConfig::from_env(env)
            .map_err(runtime)?
            .executable()
            .to_owned()
    };
    let mut command = std::process::Command::new(executable);
    command.env_clear();
    if provider == "claude-agent" {
        command.args(["auth", "login"]);
    } else {
        command.arg("login");
    }
    let terminal: &[&str] = &[
        "HOME",
        "PATH",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TERM",
        "COLORTERM",
    ];
    // Windows programs also find their profile, temporary and system folders
    // and the command processor through the environment.
    let windows: &[&str] = if cfg!(windows) {
        &[
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "SystemRoot",
            "WINDIR",
            "TEMP",
            "TMP",
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
        ]
    } else {
        &[]
    };
    let provider_keys: &[&str] = if provider == "claude-agent" {
        if cfg!(windows) {
            &["CLAUDE_CONFIG_DIR", "CLAUDE_CODE_GIT_BASH_PATH"]
        } else {
            &["CLAUDE_CONFIG_DIR"]
        }
    } else if provider == "chatgpt" {
        &["CODEX_HOME"]
    } else {
        &[
            "COPILOT_HOME",
            "COPILOT_CACHE_HOME",
            "COPILOT_GITHUB_TOKEN",
            "GH_TOKEN",
            "GITHUB_TOKEN",
        ]
    };
    for &key in terminal.iter().chain(windows).chain(provider_keys) {
        if let Some(value) = variable(env, key)
            .cloned()
            .or_else(|| std::env::var(key).ok())
        {
            command.env(key, value);
        }
    }
    Ok(command)
}

/// The value of `name` in `env`; Windows variable names ignore case.
fn variable<'a>(env: &'a HashMap<String, String>, name: &str) -> Option<&'a String> {
    env.get(name).or_else(|| {
        cfg!(windows)
            .then(|| {
                env.iter()
                    .find(|(key, _)| key.eq_ignore_ascii_case(name))
                    .map(|(_, value)| value)
            })
            .flatten()
    })
}

#[cfg(unix)]
fn login(env: &HashMap<String, String>, provider: &str) -> CliResult<i32> {
    use std::os::unix::process::CommandExt;
    let mut command = login_command(env, provider)?;
    // Replacing this process preserves terminal ownership, signals, and the exact
    // official login exit status without adding an unbounded intermediary waiter.
    let _error = command.exec();
    Err(runtime(
        "The official subscription login process could not be started",
    ))
}
/// Windows cannot replace a process, so the login runs as a child sharing
/// this console. It receives Ctrl-C and Ctrl-Break itself while this process
/// ignores them, waits for it and exits with its exit status.
#[cfg(windows)]
fn login(env: &HashMap<String, String>, provider: &str) -> CliResult<i32> {
    let mut command = login_command(env, provider)?;
    let _delegated = crate::signals::Delegated::install().map_err(runtime)?;
    let mut child = command
        .spawn()
        .map_err(|_| runtime("The official subscription login process could not be started"))?;
    let status = child.wait().map_err(runtime)?;
    Ok(status.code().unwrap_or(1))
}
#[cfg(not(any(unix, windows)))]
fn login(_env: &HashMap<String, String>, _provider: &str) -> CliResult<i32> {
    Err(runtime(
        "Subscription interactive login delegation is not supported on this platform",
    ))
}
