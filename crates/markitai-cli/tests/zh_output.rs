//! With `MARKITAI_LANG=zh` no line a person reads is left in English: the main
//! commands run in a home of their own and every line of their terminal output
//! is checked (see `support/prose.rs` for what counts as English prose).
//! Standard output that is data (a converted document, JSON, `config list`,
//! the dry-run listing) is not read; the `serve` startup block is checked in
//! `serve_startup.rs`. No command reaches the network: the one URL is a
//! refused local port.
use std::path::Path;
use std::process::{Command, Stdio};

#[path = "support/console.rs"]
mod console;
#[path = "support/prose.rs"]
mod prose;

/// A valid 1x1 RGBA PNG: an image without text, skipped without OCR.
const TINY_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 11, 73, 68, 65, 84, 120, 156, 99, 96, 0, 2, 0, 0, 5, 0, 1,
    122, 94, 171, 63, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

struct Run<'a> {
    root: &'a Path,
    /// Lines found, with the command that printed them.
    english: Vec<String>,
}

impl Run<'_> {
    /// Run `args` in `cwd` (the root by default); `human_stdout` says whether
    /// standard output is meant for a person.
    fn check(&mut self, cwd: Option<&Path>, human_stdout: bool, args: &[&str]) {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        console::detached(&mut command);
        command.env_clear();
        for name in ["PATH", "SYSTEMROOT", "USERPROFILE", "TMPDIR"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        let output = command
            .current_dir(cwd.unwrap_or(self.root))
            .env("HOME", self.root)
            .env("MARKITAI_HOME", self.root.join("home"))
            .env("MARKITAI_LANG", "zh")
            .env("COPILOT_CLI_PATH", self.root.join("missing-copilot"))
            .env("CLAUDE_CLI_PATH", self.root.join("missing-claude"))
            .env("CODEX_CLI_PATH", self.root.join("missing-codex"))
            .env("NO_PROXY", "*")
            .args(args)
            .stdin(Stdio::null())
            .output()
            .expect("CLI starts");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let streams = [
            ("stderr", Some(&stderr)),
            ("stdout", human_stdout.then_some(&stdout)),
        ];
        for (name, text) in streams {
            for line in text.into_iter().flat_map(|text| text.lines()) {
                if prose::english_prose(line) {
                    self.english
                        .push(format!("markitai {} ({name}): {line}", args.join(" ")));
                }
            }
        }
    }
}

#[test]
fn the_main_commands_print_no_english_prose_in_chinese() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let docs = root.join("docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::create_dir_all(root.join("home")).unwrap();
    std::fs::write(docs.join("a.md"), "# Title\n\nHello.\n").unwrap();
    std::fs::write(
        docs.join("b.html"),
        "<html><body><h1>H</h1><p>Para</p></body></html>",
    )
    .unwrap();
    std::fs::write(docs.join("c.png"), TINY_PNG).unwrap();
    std::fs::write(docs.join("broken.pdf"), b"%PDF-1.4\nnot really").unwrap();
    std::fs::write(root.join("weird.xyz"), "x").unwrap();
    std::fs::write(root.join("bad.json"), r#"{"llm": {"enabled": "x"}}"#).unwrap();
    std::fs::write(root.join("broken.json"), r#"{"llm": "#).unwrap();
    std::fs::write(root.join("afile"), "x").unwrap();
    // A folder whose .env and markitai.json the CLI does not fully trust.
    let folder = root.join("folder");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join(".env"), "OPENAI_API_BASE=http://127.0.0.1:9\n").unwrap();
    std::fs::write(
        folder.join("markitai.json"),
        r#"{"llm":{"model_list":[{"model_name":"default","litellm_params":{"model":"openai/gpt-x","api_base":"http://127.0.0.1:9","api_key":"env:OPENAI_API_KEY"}}]}}"#,
    )
    .unwrap();
    std::fs::write(folder.join("a.md"), "# A\n").unwrap();

    let path = |path: &Path| path.to_str().unwrap().to_owned();
    let (docs_s, out) = (path(&docs), path(&root.join("out")));
    let a = path(&docs.join("a.md"));
    let mut run = Run {
        root,
        english: Vec::new(),
    };
    // Help and usage errors.
    for args in [
        &["--help"][..],
        &["config", "--help"],
        &["serve", "--help"],
        &["--nope"],
        &["config", "nope"],
        &["config", "get"],
        &["config", "list", "-f", "nope"],
        &["-j", "0", &a],
        &["--playwright", &a],
    ] {
        run.check(None, true, args);
    }
    // Conversions: their stdout is the document or JSON.
    for args in [
        &[a.as_str()][..],
        &[&a, "-o", &out],
        &[&a, "-o", &out],
        &[&docs_s, "-o", &out],
        &[&docs_s, "-o", &out, "-v"],
        &[&docs_s, "-o", &out, "--dry-run"],
        &[&docs_s, "-o", &out, "--glob", "*.nope"],
        &[&docs_s],
        &[&a, "--json"],
        &[&docs_s, "-o", &path(&root.join("afile"))],
        &[&path(&root.join("missing.md"))],
        &[&path(&root.join("weird.xyz"))],
        &[&a, "--preset", "nope"],
        &[&a, "--llm"],
        &["http://127.0.0.1:9/page", "-o", &out],
    ] {
        run.check(None, false, args);
    }
    run.check(Some(&folder), false, &["a.md", "-o", &out]);
    // Configuration.
    for args in [
        &["config", "path"][..],
        &["config", "validate"],
        &["config", "validate", &path(&root.join("bad.json"))],
        &["config", "validate", &path(&root.join("broken.json"))],
        &["config", "get", "nope.key"],
        &["config", "set", "llm.enabled", "maybe"],
        &["config", "set", "nope.key", "1"],
        &["config", "edit"],
        &["init"],
        &["init", "-y"],
        &["config", "set", "output.on_conflict", "skip"],
        &["-I"],
    ] {
        run.check(None, true, args);
    }
    // Diagnostics, caches, sign-in and the MCP service.
    for args in [
        &["doctor"][..],
        &["cache", "stats"],
        &["cache", "stats", "-v"],
        &["cache", "clear"],
        &["cache", "clear", "-y"],
        &["cache", "spa-domains"],
        &["cache", "spa-domains", "--clear"],
        &["auth"],
        &["auth", "claude", "status"],
    ] {
        run.check(None, true, args);
    }
    run.check(Some(&folder), true, &["doctor"]);
    run.check(None, false, &["mcp"]);
    assert!(
        run.english.is_empty(),
        "English on a Chinese terminal:\n{}",
        run.english.join("\n")
    );
}
