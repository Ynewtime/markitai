//! The `markitai-mcp` launcher name is a command of its own: it answers
//! `--version` and shows its own name in help and usage errors, while no
//! argument at all still starts the MCP service.
#![cfg(any(unix, windows))]

use std::path::Path;
use std::process::{Command, Output, Stdio};

struct Launcher {
    root: tempfile::TempDir,
}

impl Launcher {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            env!("CARGO_BIN_EXE_markitai"),
            root.path().join("markitai-mcp"),
        )
        .unwrap();
        #[cfg(windows)]
        std::fs::copy(
            env!("CARGO_BIN_EXE_markitai"),
            root.path().join("markitai-mcp.exe"),
        )
        .unwrap();
        Self { root }
    }

    fn run(&self, language: &str, args: &[&str]) -> Output {
        self.run_as("markitai-mcp", language, args)
    }

    fn run_as(&self, name: &str, language: &str, args: &[&str]) -> Output {
        let path = if name == "markitai" {
            Path::new(env!("CARGO_BIN_EXE_markitai")).to_owned()
        } else {
            self.root.path().join(if cfg!(windows) {
                format!("{name}.exe")
            } else {
                name.to_owned()
            })
        };
        let mut command = Command::new(path);
        command.env_clear();
        for name in [
            "PATH",
            "TMPDIR",
            "TEMP",
            "TMP",
            "SYSTEMROOT",
            "WINDIR",
            "PATHEXT",
        ] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .current_dir(self.root.path())
            .env("HOME", self.root.path().join("home"))
            .env("USERPROFILE", self.root.path().join("home"))
            .env("MARKITAI_HOME", self.root.path().join("home"))
            .env("MARKITAI_LANG", language)
            .args(args)
            .stdin(Stdio::null())
            .output()
            .unwrap()
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn version_of(output: &Output) -> String {
    text(&output.stdout)
        .split_whitespace()
        .last()
        .unwrap()
        .to_owned()
}

#[test]
fn the_launcher_reports_its_version_under_its_own_name() {
    let launcher = Launcher::new();
    let main = launcher.run_as("markitai", "en", &["--version"]);
    assert_eq!(main.status.code(), Some(0));
    for flag in ["--version", "-V"] {
        let output = launcher.run("en", &[flag]);
        assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
        assert_eq!(
            text(&output.stdout),
            format!("markitai-mcp {}\n", version_of(&main)),
            "{flag}"
        );
        assert!(output.stderr.is_empty());
    }
    // Repeating an option is accepted, as it is for the main command: the last one wins.
    let repeated = launcher.run(
        "en",
        &[
            "-c",
            "a.json",
            "-c",
            "b.json",
            "--config-json",
            "{}",
            "--config-json",
            "{}",
            "-V",
        ],
    );
    assert_eq!(
        repeated.status.code(),
        Some(0),
        "{}",
        text(&repeated.stderr)
    );
    // The same flag stays refused on the subcommand, whose help is unchanged.
    let subcommand = launcher.run_as("markitai", "en", &["mcp", "--version"]);
    assert_eq!(subcommand.status.code(), Some(2));
}

#[test]
fn the_launcher_help_and_usage_errors_say_markitai_mcp() {
    let launcher = Launcher::new();
    let output = launcher.run("en", &["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let help = text(&output.stdout);
    assert!(
        help.starts_with("Run the native MCP service over standard input/output\n"),
        "{help}"
    );
    assert!(help.contains("Usage: markitai-mcp [OPTIONS]\n"), "{help}");
    assert!(!help.contains("markitai-mcp mcp"), "{help}");
    assert!(!help.contains("Commands:"), "{help}");
    for option in [
        "-h, --help",
        "-V, --version",
        "-c, --config <PATH>",
        "--config-json <JSON>",
    ] {
        assert!(help.contains(option), "{option}\n{help}");
    }
    let output = launcher.run("en", &["--no-such-option"]);
    assert_eq!(output.status.code(), Some(2));
    let error = text(&output.stderr);
    assert!(
        error.contains("unexpected argument '--no-such-option'"),
        "{error}"
    );
    assert!(error.contains("Usage: markitai-mcp [OPTIONS]"), "{error}");
    assert!(!error.contains("markitai-mcp mcp"), "{error}");
    // Conversion options belong to the main command.
    let output = launcher.run("en", &["note.txt"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("unexpected argument 'note.txt'"));
}

#[test]
fn the_launcher_help_follows_the_terminal_language() {
    let launcher = Launcher::new();
    let output = launcher.run("zh", &["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let help = text(&output.stdout);
    assert!(help.contains("用法: markitai-mcp"), "{help}");
    assert!(help.contains("-V, --version"), "{help}");
    assert!(help.contains("显示版本"), "{help}");
    for english in ["Usage:", "Print help", "Print version", "Options:"] {
        assert!(!help.contains(english), "{english}\n{help}");
    }
    let output = launcher.run("zh", &["-V"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(text(&output.stdout).starts_with("markitai-mcp "));
}

#[test]
fn the_main_command_and_its_mcp_subcommand_keep_their_names() {
    let launcher = Launcher::new();
    let output = launcher.run_as("markitai", "en", &["mcp", "--help"]);
    assert_eq!(output.status.code(), Some(0));
    let help = text(&output.stdout);
    assert!(help.contains("Usage: markitai mcp [OPTIONS]"), "{help}");
    assert!(!help.contains("--version"), "{help}");
}

#[cfg(windows)]
#[test]
fn the_launcher_accepts_windows_case_insensitive_executable_names() {
    let launcher = Launcher::new();
    let output = launcher.run_as("MARKITAI-MCP", "en", &["--version"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(text(&output.stdout).starts_with("markitai-mcp "));
    let output = launcher.run_as("MARKITAI-MCP", "en", &["--help"]);
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    assert!(text(&output.stdout).contains("Usage: markitai-mcp [OPTIONS]"));
}
