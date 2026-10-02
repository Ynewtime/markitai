//! Terminal language: `MARKITAI_LANG`, then `LC_ALL`, `LC_MESSAGES` and `LANG` choose
//! Chinese or English for the commands the reference localizes, while JSON,
//! help and exit codes stay the same in both languages.
use serde_json::json;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

/// A clean environment: nothing from the developer's locale leaks in.
fn command(root: &Path, envs: &[(&str, &str)]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for name in ["SYSTEMROOT", "USERPROFILE", "TMPDIR"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = bin.join("soffice");
        std::fs::write(&path, "#!/bin/sh\nprintf 'LibreOffice 26.2.0.1\\n'\n").unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    command
        .current_dir(root)
        .env("HOME", root)
        .env("PATH", bin)
        .env("MARKITAI_HOME", root.join("home"))
        .env("MARKITAI_BROWSER_EXECUTABLE", root.join("absent-browser"))
        .env("PLAYWRIGHT_BROWSERS_PATH", root.join("browser-cache"))
        .envs(envs.iter().copied())
        .stdin(Stdio::null());
    command
}

fn run(root: &Path, envs: &[(&str, &str)], args: &[&str]) -> Output {
    command(root, envs).args(args).output().expect("CLI starts")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).unwrap()
}

const ZH: &[(&str, &str)] = &[("MARKITAI_LANG", "zh")];
const EN: &[(&str, &str)] = &[("MARKITAI_LANG", "en")];

#[test]
fn doctor_frame_is_chinese_while_check_values_and_exit_status_are_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let output = run(dir.path(), ZH, &["doctor"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let text = stdout(&output);
    let lines: Vec<_> = text.lines().collect();
    assert!(lines[0].ends_with("— 原生诊断"), "{text}");
    assert_eq!(
        lines[1],
        "配置文件：内建默认值（未找到配置文件；可运行 `markitai init` 创建）"
    );
    // The status word is translated; the name and message are the JSON values.
    assert!(
        text.contains("LLM API：缺失 — No models configured in llm.model_list"),
        "{text}"
    );
    assert!(text.contains("Serve：正常 — "), "{text}");
    assert!(text.contains("Vision Model：警告 — "), "{text}");
    // How many optional backends are ready depends on the platform.
    let summary = lines.last().copied().unwrap_or_default();
    assert!(
        summary.starts_with("总结：配置要求的检查均已就绪；")
            && summary.ends_with(" 项可选检查未就绪（见上方提示）。"),
        "{text}"
    );

    let saved = dir.path().join("markitai.json");
    std::fs::write(&saved, r#"{"fetch":{"strategy":"playwright"}}"#).unwrap();
    let output = run(dir.path(), ZH, &["doctor"]);
    assert_eq!(output.status.code(), Some(1));
    let text = stdout(&output);
    assert!(
        text.lines().nth(1).unwrap().starts_with("配置文件："),
        "{text}"
    );
    assert!(text.contains("markitai.json"), "{text}");
    assert!(
        text.contains("Chromium (native CDP)：缺失（配置要求） — "),
        "{text}"
    );
    assert!(
        text.contains(
            "总结：1 项配置要求的检查未就绪：Chromium (native CDP)。请按上方提示处理后重新运行 `markitai doctor`。"
        ),
        "{text}"
    );
    let english = run(dir.path(), EN, &["doctor"]);
    assert_eq!(english.status.code(), Some(1));
    assert!(
        stdout(&english).contains("Chromium (native CDP): missing (required by configuration) — ")
    );
}

#[test]
fn doctor_summary_joins_several_blocked_checks() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = json!({"fetch":{"strategy":"playwright"},"llm":{"model_list":[{"model_name":"m","litellm_params":{"model":"openai/x","api_key":"env:I18N_ABSENT_KEY"}}]}});
    let output = run(
        dir.path(),
        ZH,
        &["--config-json", &cfg.to_string(), "doctor"],
    );
    assert_eq!(output.status.code(), Some(1));
    let text = stdout(&output);
    assert!(
        text.contains("总结：2 项配置要求的检查未就绪：Chromium (native CDP)、LLM API。"),
        "{text}"
    );
    assert!(text.contains("LLM API：错误（配置要求） — "), "{text}");
    let english = stdout(&run(
        dir.path(),
        EN,
        &["--config-json", &cfg.to_string(), "doctor"],
    ));
    assert!(
        english.contains(
            "Summary: 2 checks required by the configuration are not ready: Chromium (native CDP), LLM API."
        ),
        "{english}"
    );
}

#[test]
fn doctor_repair_progress_follows_the_language_without_starting_an_installer() {
    let dir = tempfile::tempdir().unwrap();
    let output = run(dir.path(), ZH, &["doctor", "--fix"]);
    assert_eq!(output.status.code(), Some(1));
    let errors = stderr(&output);
    assert!(
        errors.contains("正在 Markitai 私有目录中安装官方 Chrome headless shell……"),
        "{errors}"
    );
    // The reason comes from the installer and stays as reported.
    assert!(errors.contains("浏览器修复失败："), "{errors}");
    assert!(errors.contains("no installer was started"), "{errors}");
}

#[test]
fn json_and_exit_codes_do_not_depend_on_the_language() {
    let dir = tempfile::tempdir().unwrap();
    for args in [
        &["doctor", "--json"][..],
        &["cache", "stats", "--json"],
        &["cache", "spa-domains", "--json"],
        &["config", "list"],
    ] {
        let chinese = run(dir.path(), ZH, args);
        let english = run(dir.path(), EN, args);
        assert_eq!(chinese.status.code(), english.status.code(), "{args:?}");
        assert_eq!(chinese.stdout, english.stdout, "{args:?}");
        assert!(!chinese.stdout.is_empty(), "{args:?}");
    }
    // Help and the conversion lines have Chinese wording too; their tests are
    // in `help_language.rs` and `cli_zh.rs`. Their exit codes are the same.
    for args in [&["--help"][..], &["cache", "--help"]] {
        let chinese = run(dir.path(), ZH, args);
        let english = run(dir.path(), EN, args);
        assert_eq!(chinese.status.code(), english.status.code(), "{args:?}");
        assert_ne!(chinese.stdout, english.stdout, "{args:?}");
    }
    std::fs::write(dir.path().join("note.txt"), "hello\n").unwrap();
    let output = run(dir.path(), ZH, &["note.txt", "-o", "out"]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(stderr(&output), "已写入 out/note.txt.md\n");
    let output = run(dir.path(), EN, &["note.txt", "-o", "out"]);
    assert!(stderr(&output).starts_with("Wrote "), "{}", stderr(&output));
}

#[test]
fn cache_statistics_and_clearing_speak_chinese() {
    let dir = tempfile::tempdir().unwrap();
    let stats = run(dir.path(), ZH, &["cache", "stats"]);
    assert!(stats.status.success());
    assert_eq!(
        stdout(&stats),
        "缓存：已启用\nLLM 缓存：0 条（0 B）\nURL 抓取缓存：0 条（0 B）\n"
    );
    let disabled = run(
        dir.path(),
        ZH,
        &[
            "--config-json",
            r#"{"cache":{"enabled":false}}"#,
            "cache",
            "stats",
        ],
    );
    assert!(stdout(&disabled).starts_with("缓存：已禁用\n"));
    let english = run(dir.path(), EN, &["cache", "stats"]);
    assert_eq!(
        stdout(&english),
        "Cache enabled: true\nLLM cache: 0 entries (0 B)\nURL fetch cache: 0 entries (0 B)\n"
    );

    let clear = run(
        dir.path(),
        ZH,
        &["cache", "clear", "-y", "--include-spa-domains"],
    );
    assert!(clear.status.success(), "{}", stderr(&clear));
    assert_eq!(
        stdout(&clear),
        "已清理 0 条缓存\n已清理 0 个已学习的 SPA 域名\n"
    );

    let mut child = command(dir.path(), ZH)
        .args(["cache", "clear"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"n\n").unwrap();
    let declined = child.wait_with_output().unwrap();
    assert!(declined.status.success());
    let text = stdout(&declined);
    assert!(text.starts_with("清理 LLM 与 URL 抓取缓存（"), "{text}");
    assert!(text.ends_with("）？[y/N]：已取消\n"), "{text}");

    let mut child = command(dir.path(), ZH)
        .args(["cache", "clear", "--include-spa-domains"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"y\n").unwrap();
    let accepted = child.wait_with_output().unwrap();
    assert!(accepted.status.success());
    let text = stdout(&accepted);
    assert!(
        text.starts_with("清理 LLM 与 URL 抓取缓存及已学习的浏览器域名（"),
        "{text}"
    );
    assert!(text.contains("？[y/N]：已清理 0 条缓存\n"), "{text}");
}

#[test]
fn learned_browser_domains_list_and_clear_in_chinese() {
    let dir = tempfile::tempdir().unwrap();
    let list = run(dir.path(), ZH, &["cache", "spa-domains"]);
    assert!(list.status.success());
    assert_eq!(stdout(&list), "没有已学习的 SPA 域名。\n");
    let clear = run(dir.path(), ZH, &["cache", "spa-domains", "--clear"]);
    assert!(clear.status.success());
    assert_eq!(stdout(&clear), "已清理 0 个已学习的 SPA 域名\n");
    let english = run(dir.path(), EN, &["cache", "spa-domains", "--clear"]);
    assert_eq!(stdout(&english), "Cleared 0 learned SPA domains\n");
}

#[test]
fn config_path_and_validate_speak_chinese_but_keep_paths_and_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = run(dir.path(), ZH, &["config", "path"]);
    assert!(path.status.success());
    let message = stdout(&path);
    assert!(
        message.starts_with("未找到配置文件，正在使用内建默认值。可运行 `markitai init` 创建。\n"),
        "{message}"
    );
    // Where a file is looked for, in the order it is used, and which one
    // `config set` would create.
    assert!(
        message.contains("查找顺序：-c FILE、环境变量 MARKITAI_CONFIG、./markitai.json、")
            && message.contains("config.json。`markitai config set KEY VALUE` 会创建 "),
        "{message}"
    );
    let valid = run(dir.path(), ZH, &["config", "validate"]);
    assert!(valid.status.success());
    assert_eq!(stdout(&valid), "配置有效\n");

    std::fs::write(dir.path().join("markitai.json"), "{}").unwrap();
    let path = run(dir.path(), ZH, &["config", "path"]);
    // A found file is printed as a bare path for scripts in every language.
    let english = run(dir.path(), EN, &["config", "path"]);
    assert_eq!(path.stdout, english.stdout);
    assert!(
        Path::new(stdout(&path).trim_end()).ends_with("markitai.json"),
        "{}",
        stdout(&path)
    );
    std::fs::write(
        dir.path().join("markitai.json"),
        r#"{"batch":{"concurrency":0}}"#,
    )
    .unwrap();
    let invalid = run(dir.path(), ZH, &["config", "validate"]);
    let english = run(dir.path(), EN, &["config", "validate"]);
    assert_eq!(invalid.status.code(), Some(1));
    assert_eq!(invalid.status.code(), english.status.code());
    assert_eq!(invalid.stderr, english.stderr);
    assert!(invalid.stdout.is_empty());
}

#[test]
fn lc_all_precedes_lc_messages_and_lang_and_markitai_lang_wins() {
    let dir = tempfile::tempdir().unwrap();
    let valid = |envs: &[(&str, &str)]| stdout(&run(dir.path(), envs, &["config", "validate"]));
    assert_eq!(valid(&[]), "Configuration is valid\n");
    assert_eq!(valid(&[("LANG", "zh_CN.UTF-8")]), "配置有效\n");
    assert_eq!(valid(&[("LC_ALL", "zh_TW.UTF-8")]), "配置有效\n");
    assert_eq!(valid(&[("MARKITAI_LANG", "ZH")]), "配置有效\n");
    assert_eq!(
        valid(&[("MARKITAI_LANG", "en_US"), ("LANG", "zh_CN.UTF-8")]),
        "Configuration is valid\n"
    );
    assert_eq!(
        valid(&[("MARKITAI_LANG", "fr"), ("LANG", "zh_CN.UTF-8")]),
        "Configuration is valid\n"
    );
    assert_eq!(
        valid(&[("MARKITAI_LANG", ""), ("LANG", "zh_CN.UTF-8")]),
        "配置有效\n"
    );
    // As in POSIX, LC_ALL overrides LC_MESSAGES, which overrides LANG; the
    // reference read LANG first, so a user who exported LC_ALL got English.
    assert_eq!(
        valid(&[("LANG", "en_US.UTF-8"), ("LC_ALL", "zh_CN.UTF-8")]),
        "配置有效\n"
    );
    assert_eq!(
        valid(&[("LANG", "en_US.UTF-8"), ("LC_MESSAGES", "zh_CN.UTF-8")]),
        "配置有效\n"
    );
    assert_eq!(
        valid(&[("LC_ALL", "en_US.UTF-8"), ("LANG", "zh_CN.UTF-8")]),
        "Configuration is valid\n"
    );
    // The C and POSIX locales name no language, so they never hide a later one.
    assert_eq!(
        valid(&[("LANG", "C"), ("LC_ALL", "zh_CN.UTF-8")]),
        "配置有效\n"
    );
    assert_eq!(
        valid(&[("LC_ALL", "POSIX"), ("LANG", "zh_CN.UTF-8")]),
        "配置有效\n"
    );
    assert_eq!(valid(&[("LANG", "C")]), "Configuration is valid\n");
}

#[test]
fn dotenv_files_supply_the_language_after_the_process_environment() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("home")).unwrap();
    std::fs::write(dir.path().join("home/.env"), "MARKITAI_LANG=zh\n").unwrap();
    let valid = |envs: &[(&str, &str)]| stdout(&run(dir.path(), envs, &["config", "validate"]));
    assert_eq!(valid(&[]), "配置有效\n");
    assert_eq!(valid(EN), "Configuration is valid\n");
    // The current directory's .env is read before the one in MARKITAI_HOME.
    std::fs::write(dir.path().join(".env"), "MARKITAI_LANG=en\n").unwrap();
    assert_eq!(valid(&[]), "Configuration is valid\n");
}
