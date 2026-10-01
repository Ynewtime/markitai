//! `--help` in Chinese and English.
//!
//! English help is pinned byte for byte by a snapshot of the root command and
//! every subcommand (`-h` and `--help`) taken before the Chinese wording was
//! added; Chinese help must offer exactly the same commands and options, fit
//! an 80 column terminal and leave no English frame behind. Regenerate the
//! snapshot after an intended English change with
//! `MARKITAI_UPDATE_HELP_SNAPSHOT=1 cargo test -p markitai-cli --test help_language`.
#![cfg(unix)]
use std::collections::BTreeSet;
use std::path::Path;
use std::process::{Command, Output, Stdio};

const SNAPSHOT: &str = "tests/fixtures/help_en.txt";

fn run(root: &Path, envs: &[(&str, &str)], args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for name in ["PATH", "TMPDIR"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .current_dir(root)
        .env("HOME", root)
        .env("MARKITAI_HOME", root.join("home"))
        .envs(envs.iter().copied())
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("CLI starts")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).unwrap()
}

/// Names listed under the commands heading of a help text.
fn children(help: &str, heading: &str) -> Vec<String> {
    let mut lines = help.lines().skip_while(|line| line.trim() != heading);
    lines.next();
    lines
        .take_while(|line| line.starts_with("  ") && !line.trim().is_empty())
        .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
        .collect()
}

/// The command paths of the tree, found through the help texts themselves.
fn command_paths(root: &Path, envs: &[(&str, &str)], heading: &str) -> Vec<Vec<String>> {
    let mut paths = vec![Vec::new()];
    let mut index = 0;
    while index < paths.len() {
        let mut args: Vec<&str> = paths[index].iter().map(String::as_str).collect();
        args.push("-h");
        let help = stdout(&run(root, envs, &args));
        let known = paths[index].clone();
        for name in children(&help, heading) {
            let mut path = known.clone();
            path.push(name);
            paths.push(path);
        }
        index += 1;
    }
    paths
}

/// Every help text the program prints, in one string.
fn snapshot(root: &Path, envs: &[(&str, &str)], heading: &str) -> String {
    let mut text = String::new();
    for path in command_paths(root, envs, heading) {
        for flag in ["-h", "--help"] {
            let mut args: Vec<&str> = path.iter().map(String::as_str).collect();
            args.push(flag);
            let output = run(root, envs, &args);
            let title = ["markitai"]
                .into_iter()
                .chain(args.iter().copied())
                .collect::<Vec<_>>()
                .join(" ");
            text.push_str(&format!(
                "=== {title} (exit {}, stderr {} bytes) ===\n{}",
                output.status.code().unwrap(),
                output.stderr.len(),
                stdout(&output)
            ));
        }
    }
    // The other ways to reach the same texts.
    for args in [
        vec![],
        vec!["--quiet", "--no-llm"],
        vec!["--version"],
        vec!["-V"],
        vec!["config"],
        vec!["cache"],
        vec!["auth", "copilot", "--bogus"],
    ] {
        let output = run(root, envs, &args);
        let title = ["markitai"]
            .into_iter()
            .chain(args.iter().copied())
            .collect::<Vec<_>>()
            .join(" ");
        text.push_str(&format!(
            "=== {title} (exit {}, stderr {} bytes) ===\n{}--- stderr ---\n{}",
            output.status.code().unwrap(),
            output.stderr.len(),
            stdout(&output),
            String::from_utf8(output.stderr).unwrap()
        ));
    }
    text
}

fn snapshot_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(SNAPSHOT)
}

#[test]
fn english_help_is_unchanged_in_every_way_the_language_can_be_chosen() {
    let root = tempfile::tempdir().unwrap();
    let current = snapshot(root.path(), &[], "Commands:");
    if std::env::var_os("MARKITAI_UPDATE_HELP_SNAPSHOT").is_some() {
        std::fs::write(snapshot_path(), &current).unwrap();
    }
    let expected = std::fs::read_to_string(snapshot_path()).unwrap();
    // 59 help texts: the root and 25 commands twice, and seven other entries.
    assert_eq!(expected.matches("\n=== ").count() + 1, 59);
    assert!(
        current == expected,
        "English help differs from {SNAPSHOT}; if the change is intended, regenerate the snapshot"
    );
    // Choosing English explicitly, or a language that is not Chinese, changes nothing.
    for envs in [
        vec![("MARKITAI_LANG", "en")],
        vec![("LANG", "en_US.UTF-8")],
        vec![("LC_ALL", "fr_FR.UTF-8")],
        // A set MARKITAI_LANG decides on its own, even against LANG.
        vec![("MARKITAI_LANG", "en"), ("LANG", "zh_CN.UTF-8")],
    ] {
        assert!(
            snapshot(root.path(), &envs, "Commands:") == expected,
            "English help changed under {envs:?}"
        );
    }
    assert!(!root.path().join("home").exists());
}

/// The option names a help text defines, from its option lines (two spaces
/// before `-x, --name`, six before a long-only `--name`); wrapped help text
/// sits deeper.
fn option_names(help: &str) -> BTreeSet<String> {
    help.lines()
        .filter(|line| {
            let indent = line.len() - line.trim_start().len();
            line.trim_start().starts_with('-') && (indent == 2 || indent == 6)
        })
        .flat_map(|line| line.split_whitespace().take(2).map(str::to_owned))
        .map(|word| word.trim_end_matches(',').to_owned())
        .filter(|word| word.starts_with('-'))
        .collect()
}

fn width(line: &str) -> usize {
    line.chars()
        .map(|c| match c as u32 {
            0x2E80..=0xA4CF
            | 0xAC00..=0xD7A3
            | 0xF900..=0xFAFF
            | 0xFE30..=0xFE6F
            | 0xFF00..=0xFF60 => 2,
            _ => 1,
        })
        .sum()
}

#[test]
fn chinese_help_offers_the_same_commands_and_options_inside_80_columns() {
    let root = tempfile::tempdir().unwrap();
    let zh = [("MARKITAI_LANG", "zh")];
    let english_paths = command_paths(root.path(), &[], "Commands:");
    let chinese_paths = command_paths(root.path(), &zh, "命令:");
    assert_eq!(english_paths, chinese_paths);
    assert_eq!(chinese_paths.len(), 26);
    for path in &chinese_paths {
        for flag in ["-h", "--help"] {
            let mut args: Vec<&str> = path.iter().map(String::as_str).collect();
            args.push(flag);
            let english = run(root.path(), &[], &args);
            let chinese = run(root.path(), &zh, &args);
            assert_eq!(chinese.status.code(), Some(0), "{args:?}");
            assert!(chinese.stderr.is_empty(), "{args:?}");
            let (english, chinese) = (stdout(&english), stdout(&chinese));
            assert_eq!(
                option_names(&english),
                option_names(&chinese),
                "{args:?} offers different options"
            );
            assert!(chinese.contains("用法: markitai"), "{args:?}\n{chinese}");
            for frame in [
                "Usage:",
                "Options:",
                "Arguments:",
                "Commands:",
                "Print help",
                "Print version",
                "possible values",
                "[default",
            ] {
                assert!(!chinese.contains(frame), "{args:?}: {frame}\n{chinese}");
            }
            for line in chinese.lines() {
                assert!(width(line) <= 80, "{args:?} line is too wide: {line}");
            }
        }
    }
    // The root's sections keep the order English help shows.
    let help = stdout(&run(root.path(), &zh, &["--help"]));
    let mut last = 0;
    for heading in [
        "\n命令:\n",
        "\n参数:\n",
        "\n选项:\n",
        "\n输出:\n",
        "\n配置:\n",
        "\nLLM、OCR 与截图:\n",
        "\nURL 抓取与后端:\n",
        "\n批量处理:\n",
        "\n缓存与图片:\n",
        "\n消息与日志:\n",
        "\n-p 预设:\n",
        "\n示例:\n",
        "\n退出状态：",
    ] {
        let at = help
            .find(heading)
            .unwrap_or_else(|| panic!("{heading:?} missing\n{help}"));
        assert!(at > last, "{heading:?} is out of order\n{help}");
        last = at;
    }
}

#[test]
fn chinese_help_is_chosen_like_the_other_localized_commands() {
    let root = tempfile::tempdir().unwrap();
    let in_chinese =
        |envs: &[(&str, &str)]| stdout(&run(root.path(), envs, &["--help"])).contains("用法:");
    assert!(in_chinese(&[("MARKITAI_LANG", "zh")]));
    assert!(in_chinese(&[("MARKITAI_LANG", "zh_CN.UTF-8")]));
    assert!(in_chinese(&[("LANG", "zh_CN.UTF-8")]));
    assert!(in_chinese(&[("LC_ALL", "zh_TW.UTF-8")]));
    assert!(in_chinese(&[
        ("MARKITAI_LANG", "zh"),
        ("LANG", "en_US.UTF-8")
    ]));
    assert!(!in_chinese(&[
        ("MARKITAI_LANG", "en"),
        ("LANG", "zh_CN.UTF-8")
    ]));
    assert!(!in_chinese(&[]));

    let zh = [("MARKITAI_LANG", "zh")];
    // The same help is shown without input; the version line has no wording.
    let shown = stdout(&run(root.path(), &zh, &["-h"]));
    assert_eq!(stdout(&run(root.path(), &zh, &[])), format!("{shown}\n"));
    assert_eq!(
        stdout(&run(root.path(), &zh, &["--quiet", "--no-llm"])),
        format!("{shown}\n")
    );
    for flag in ["--version", "-V"] {
        assert_eq!(
            stdout(&run(root.path(), &zh, &[flag])),
            stdout(&run(root.path(), &[], &[flag]))
        );
    }
    // Both help forms exist in both languages; the short one is shorter only
    // where a command has a long description.
    for args in [["init", "-h"], ["init", "--help"]] {
        let help = stdout(&run(root.path(), &zh, &args));
        assert!(help.contains("创建或更新配置文件"), "{help}");
    }
    let short = stdout(&run(root.path(), &zh, &["doctor", "-h"]));
    let long = stdout(&run(root.path(), &zh, &["doctor", "--help"]));
    assert!(short.contains("显示帮助（--help 查看详细说明）"), "{short}");
    assert!(long.contains("显示帮助（-h 查看摘要）"), "{long}");
    assert!(long.contains("不会发送模型请求"), "{long}");
    assert!(!short.contains("不会发送模型请求"), "{short}");
}

#[test]
fn chinese_help_does_not_change_how_arguments_are_parsed_or_reported() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "Text\n").unwrap();
    std::fs::create_dir(root.path().join("docs")).unwrap();
    std::fs::write(root.path().join("docs/a.txt"), "A\n").unwrap();
    let zh = [("MARKITAI_LANG", "zh")];
    let en = [("MARKITAI_LANG", "en")];
    for args in [
        vec!["note.txt", "--llm", "--no-llm", "--pure"],
        vec!["note.txt", "-o", "out/chosen.md", "-q"],
        vec!["docs", "-o", "out", "--dry-run", "-g", "*.txt"],
        vec!["config", "get", "output.on_conflict"],
        vec!["config", "list", "-f", "table"],
        vec!["-c", "missing.json", "config", "path"],
        vec!["note.txt", "-j", "0"],
        vec!["--bogus"],
        vec!["config", "set"],
        vec!["note.txt", "--json"],
        vec!["cache", "stats", "--limit", "x"],
        vec!["-p", "nope", "note.txt"],
    ] {
        let english = run(root.path(), &en, &args);
        let chinese = run(root.path(), &zh, &args);
        assert_eq!(english.status.code(), chinese.status.code(), "{args:?}");
        assert_eq!(english.stdout, chinese.stdout, "{args:?}");
        // Usage errors from the parser stay English in both languages.
        if english.stderr.starts_with(b"error:") {
            assert_eq!(english.stderr, chinese.stderr, "{args:?}");
        }
    }
    let usage = run(root.path(), &zh, &["--bogus"]);
    let message = String::from_utf8(usage.stderr).unwrap();
    assert!(
        message.contains("unexpected argument '--bogus'"),
        "{message}"
    );
    // A short option cluster, a repeated switch and the hidden off switches work.
    let output = run(
        root.path(),
        &zh,
        &[
            "note.txt",
            "--pure",
            "-qq",
            "--no-ocr",
            "--no-record-history",
        ],
    );
    assert!(output.status.success());
    assert_eq!(stdout(&output), "Text\n");
}
