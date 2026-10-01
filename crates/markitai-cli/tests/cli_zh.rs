//! Chinese wording of `init` and of the lines a conversion prints on stderr
//! (results, skips, batch summary, previews, reports). Reasons that come from
//! the converter, paths, JSON, the file log and exit codes are the same in both
//! languages.
use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

const ZH: &[(&str, &str)] = &[("MARKITAI_LANG", "zh")];
const EN: &[(&str, &str)] = &[("MARKITAI_LANG", "en")];

fn command(root: &Path, envs: &[(&str, &str)]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for name in ["PATH", "SYSTEMROOT", "USERPROFILE", "TMPDIR"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .current_dir(root)
        .env("HOME", root)
        .env("MARKITAI_HOME", root.join("home"))
        .envs(envs.iter().copied());
    command
}

fn run(root: &Path, envs: &[(&str, &str)], args: &[&str]) -> Output {
    command(root, envs)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("CLI starts")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8(output.stderr.clone()).unwrap()
}

fn has_chinese(text: &str) -> bool {
    text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

/// A directory with two documents, an image without text and a broken file.
fn batch_input(root: &Path) {
    std::fs::create_dir(root.join("docs")).unwrap();
    std::fs::write(root.join("docs/good.txt"), "Good\n").unwrap();
    std::fs::write(root.join("docs/second.txt"), "Second\n").unwrap();
    std::fs::write(root.join("docs/broken.docx"), "not a zip archive").unwrap();
    std::fs::write(root.join("docs/pic.png"), "x").unwrap();
}

#[test]
fn batch_summary_and_item_lines_are_chinese_with_the_converters_reasons_kept() {
    let root = tempfile::tempdir().unwrap();
    batch_input(root.path());
    let output = run(root.path(), ZH, &["docs", "-o", "out"]);
    assert_eq!(output.status.code(), Some(10));
    assert!(output.stdout.is_empty());
    let lines: Vec<String> = stderr(&output).lines().map(str::to_owned).collect();
    // The failure keeps the converter's English reason after a Chinese frame.
    assert!(
        lines[0].starts_with("Error: broken.docx: Native document conversion failed: "),
        "{lines:?}"
    );
    assert!(lines[1].starts_with("完成：2 个文件（0:0"), "{lines:?}");
    assert_eq!(
        &lines[2..],
        [
            "已跳过 1 项（image_only）：pic.png。请使用 --llm 或 --ocr 提取内容。",
            "失败 1 项：broken.docx。详见上方的错误信息。",
            "输出目录：out",
        ]
    );

    // Warnings and errors still show with -q; the summary does not.
    let quiet = run(root.path(), ZH, &["docs", "-o", "quiet", "-q"]);
    assert_eq!(quiet.status.code(), Some(10));
    let message = stderr(&quiet);
    assert!(message.starts_with("Error: broken.docx: "), "{message}");
    assert_eq!(message.lines().count(), 1, "{message}");

    // Every item failing for want of a model gets one remedy line.
    let llm = run(root.path(), ZH, &["docs", "-o", "llm-out", "--llm"]);
    assert_eq!(llm.status.code(), Some(10));
    let message = stderr(&llm);
    assert_eq!(message.matches("未配置 LLM 模型：").count(), 1, "{message}");
    assert!(
        message.contains("失败 4 项：broken.docx、good.txt 等。"),
        "{message}"
    );

    // The same run in English prints the English lines it always did.
    let english = run(root.path(), EN, &["docs", "-o", "english"]);
    let message = stderr(&english);
    assert!(message.contains("Done: 2 files ("), "{message}");
    assert!(
        message.contains(
            "Skipped 1 item (image_only): pic.png. Use --llm or --ocr for content extraction."
        ),
        "{message}"
    );
    assert!(message.trim_end().ends_with("Output: english"), "{message}");
    assert!(!has_chinese(&message));
}

#[test]
fn json_output_and_the_file_log_do_not_depend_on_the_language() {
    let envelope = |envs: &[(&str, &str)]| {
        let root = tempfile::tempdir().unwrap();
        batch_input(root.path());
        let log = root.path().join("logs");
        let config = json!({"log": {"dir": log}}).to_string();
        let output = run(
            root.path(),
            envs,
            &["docs", "-o", "out", "--json", "--config-json", &config],
        );
        assert_eq!(output.status.code(), Some(10));
        let mut body: Value = serde_json::from_slice(&output.stdout).unwrap();
        // Timings differ from run to run.
        body["totals"]["duration_s"] = Value::Null;
        for item in body["items"].as_array_mut().unwrap() {
            item["duration_s"] = Value::Null;
        }
        // Human lines on stderr are Chinese or English; JSON runs print the
        // same item lines, and the log keeps English either way.
        let mut entries: Vec<String> = std::fs::read_dir(&log)
            .unwrap()
            .flat_map(|file| {
                std::fs::read_to_string(file.unwrap().path())
                    .unwrap()
                    .lines()
                    .map(|line| line.split_once(" | ").unwrap().1.to_owned())
                    .collect::<Vec<_>>()
            })
            .collect();
        entries.sort();
        (body, entries, stderr(&output))
    };
    let (zh_json, zh_log, zh_text) = envelope(ZH);
    let (en_json, en_log, en_text) = envelope(EN);
    assert_eq!(zh_json, en_json);
    assert_eq!(zh_log, en_log);
    assert!(
        zh_log
            .iter()
            .any(|line| line.contains("Converting good.txt"))
    );
    assert!(!zh_log.iter().any(|line| has_chinese(line)), "{zh_log:?}");
    // A --json run prints no human lines at all, in either language.
    assert_eq!(zh_text, en_text);
    assert_eq!(zh_text, "");
}

#[test]
fn the_file_log_of_a_chinese_run_holds_the_english_lines_of_the_console() {
    let root = tempfile::tempdir().unwrap();
    batch_input(root.path());
    let log = root.path().join("logs");
    let config = json!({"log": {"dir": log}}).to_string();
    let output = run(
        root.path(),
        ZH,
        &["docs", "-o", "out", "--config-json", &config],
    );
    assert_eq!(output.status.code(), Some(10));
    let console = stderr(&output);
    assert!(console.contains("完成：2 个文件"), "{console}");
    let file = std::fs::read_dir(&log)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let text = std::fs::read_to_string(file).unwrap();
    for line in [
        "| ERROR | cli | Error: broken.docx: Native document conversion failed: ",
        "| INFO  | cli | Done: 2 files (0:0",
        "| INFO  | cli | Skipped 1 item (image_only): pic.png. Use --llm or --ocr for content extraction.",
        "| INFO  | cli | Failed 1 item: broken.docx. See the errors above.",
        "| INFO  | cli | Output: out",
    ] {
        assert!(text.contains(line), "{line}\n{text}");
    }
    assert!(!has_chinese(&text), "{text}");
}

#[test]
fn previews_empty_directories_and_url_lists_report_in_chinese() {
    let root = tempfile::tempdir().unwrap();
    batch_input(root.path());
    std::fs::create_dir(root.path().join("empty")).unwrap();
    let preview = run(root.path(), ZH, &["docs", "-o", "out", "--dry-run"]);
    assert!(preview.status.success());
    assert_eq!(
        stdout(&preview),
        "broken.docx -> out/\ngood.txt -> out/\npic.png -> out/\nsecond.txt -> out/\n"
    );
    assert_eq!(
        stderr(&preview),
        "预览：将转换 4 个文件；未写出任何内容。\n"
    );
    assert!(!root.path().join("out").exists());
    let english = run(root.path(), EN, &["docs", "-o", "out", "--dry-run"]);
    assert_eq!(
        stderr(&english),
        "Dry run: 4 files would be converted; nothing was written.\n"
    );
    assert_eq!(english.stdout, preview.stdout);

    let empty = run(root.path(), ZH, &["empty", "-o", "out"]);
    assert!(empty.status.success());
    assert_eq!(
        stderr(&empty),
        "在 empty 中没有找到受支持的文件或 .urls 列表，无需转换。\n"
    );
    let filtered = run(root.path(), ZH, &["docs", "-o", "out", "-g", "*.pdf"]);
    assert_eq!(
        stderr(&filtered),
        "在 docs 中没有与 --glob 匹配的受支持文件或 .urls 列表，无需转换。\n"
    );
    assert!(
        run(root.path(), ZH, &["empty", "-o", "out", "-q"])
            .stderr
            .is_empty()
    );

    // Rejected entries are located without echoing them.
    std::fs::write(
        root.path().join("links.urls"),
        "# comment\nftp://user:entry-secret@example.com/file\nhttps://example.com/page\n",
    )
    .unwrap();
    std::fs::write(
        root.path().join("list.urls"),
        "[1, \"https://example.com/a\"]",
    )
    .unwrap();
    let urls = run(root.path(), ZH, &["links.urls", "-o", "out", "--dry-run"]);
    assert!(urls.status.success(), "{}", stderr(&urls));
    let message = stderr(&urls);
    assert!(
        message.starts_with("Warning: 跳过 links.urls 的第 2 行：不是 HTTP(S) URL\n"),
        "{message}"
    );
    assert!(
        message.contains("预览：将转换 1 个 URL；未写出任何内容。"),
        "{message}"
    );
    assert!(!message.contains("entry-secret"), "{message}");
    let array = run(root.path(), ZH, &["list.urls", "-o", "out", "--dry-run"]);
    assert!(
        stderr(&array).starts_with(
            "Warning: 跳过 list.urls 的第 1 项：应为 URL 字符串，或带有 \"url\" 字段的对象\n"
        ),
        "{}",
        stderr(&array)
    );
    // No valid URL: the frame is Chinese, the reason is the one JSON would carry.
    std::fs::write(root.path().join("bad.urls"), "ftp://x/y\n").unwrap();
    let bad = run(root.path(), ZH, &["bad.urls", "-o", "out"]);
    assert_eq!(bad.status.code(), Some(1));
    assert!(
        stderr(&bad).ends_with("Error: No valid URLs found in bad.urls.\n"),
        "{}",
        stderr(&bad)
    );
}

#[test]
fn single_inputs_name_the_written_file_skips_and_missing_models_in_chinese() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.txt"), "Text\n").unwrap();
    std::fs::write(root.path().join("pic.png"), "x").unwrap();
    let first = run(root.path(), ZH, &["note.txt", "-o", "out"]);
    assert_eq!(stderr(&first), "已写入 out/note.txt.md\n");
    assert_eq!(
        stderr(&run(root.path(), EN, &["note.txt", "-o", "out2"])),
        "Wrote out2/note.txt.md\n"
    );
    // The renamed file is the one named.
    let again = run(root.path(), ZH, &["note.txt", "-o", "out"]);
    assert_eq!(stderr(&again), "已写入 out/note.txt.v2.md\n");

    let image = run(root.path(), ZH, &["pic.png", "-o", "out"]);
    assert_eq!(
        stderr(&image),
        "已跳过 pic.png：图片没有文字可提取，需要 --ocr 或 --llm。用 --ocr 做本地文字识别，或用 --llm 调用视觉模型。\n"
    );
    assert_eq!(
        stderr(&run(root.path(), EN, &["pic.png", "-o", "out"])),
        "Skipped pic.png: an image has no text to extract without --ocr or --llm. Use --ocr for local text recognition or --llm for a vision model.\n"
    );
    let skip = run(
        root.path(),
        ZH,
        &[
            "note.txt",
            "-o",
            "out",
            "--config-json",
            r#"{"output":{"on_conflict":"skip"}}"#,
        ],
    );
    assert_eq!(
        stderr(&skip),
        "已跳过 note.txt：输出已存在，且 output.on_conflict 为 skip。设为 rename 或 overwrite 可重新转换。\n"
    );

    let model = run(root.path(), ZH, &["note.txt", "--llm"]);
    assert_eq!(model.status.code(), Some(1));
    let message = stderr(&model);
    assert!(
        message.starts_with("Error: No model configured; "),
        "{message}"
    );
    assert!(
        message.contains("\nHint: 请设置供应商的 API key，例如 OPENAI_API_KEY"),
        "{message}"
    );

    let alt = run(
        root.path(),
        ZH,
        &["note.txt", "--alt", "--desc", "-o", "alt"],
    );
    assert_eq!(
        stderr(&alt),
        "Warning: 未启用 --llm（或 standard/rich 预设）时，--alt 和 --desc 不起作用\n已写入 alt/note.txt.md\n"
    );

    // A report path appears with -v; the report itself is unchanged.
    let verbose = run(
        root.path(),
        ZH,
        &[
            "note.txt",
            "-o",
            "rep",
            "-v",
            "--config-json",
            r#"{"output":{"report":true}}"#,
        ],
    );
    assert!(verbose.status.success(), "{}", stderr(&verbose));
    let message = stderr(&verbose);
    assert!(
        message.starts_with("报告：") && message.contains(".markitai/reports/"),
        "{message}"
    );
    assert!(message.ends_with("已写入 rep/note.txt.md\n"), "{message}");
    // Without the file the output location is as before.
    assert!(
        run(root.path(), ZH, &["missing.txt", "-o", "x"])
            .status
            .code()
            == Some(1)
    );
}

#[test]
fn resuming_without_state_and_the_report_of_a_batch_speak_chinese() {
    let root = tempfile::tempdir().unwrap();
    batch_input(root.path());
    let fresh = run(root.path(), ZH, &["docs", "-o", "out", "--resume", "-v"]);
    assert_eq!(fresh.status.code(), Some(10));
    let message = stderr(&fresh);
    assert!(
        message.starts_with("没有与这些路径和选项匹配的恢复状态，将开始全新的批处理。\n"),
        "{message}"
    );
    assert!(
        message.contains("\n报告：") && message.ends_with("输出目录：out\n"),
        "{message}"
    );
    // A second report is kept as it is when the policy says skip.
    let again = run(
        root.path(),
        ZH,
        &[
            "docs",
            "-o",
            "out",
            "-v",
            "--config-json",
            r#"{"output":{"on_conflict":"skip"}}"#,
        ],
    );
    assert!(
        stderr(&again).starts_with("已保留现有报告："),
        "{}",
        stderr(&again)
    );
    let english = run(root.path(), EN, &["docs", "-o", "out", "--resume", "-v"]);
    assert!(
        stderr(&english).contains("Report: "),
        "{}",
        stderr(&english)
    );
}

#[test]
fn init_without_prompts_says_what_it_did_in_chinese() {
    let root = tempfile::tempdir().unwrap();
    let created = run(root.path(), ZH, &["init", "-y", "--local"]);
    assert!(created.status.success(), "{}", stderr(&created));
    assert_eq!(stdout(&created), "配置已创建：markitai.json\n");
    let message = stderr(&created);
    assert_eq!(
        message,
        "后续步骤：\n  markitai FILE -o DIR    转换文档\n  markitai doctor         检查模型和可选后端\n未检测到 API 模型：请设置供应商的 API key（如 OPENAI_API_KEY），然后再次运行 `markitai init -y` 添加。\n"
    );
    let file = std::fs::read(root.path().join("markitai.json")).unwrap();

    let same = run(root.path(), ZH, &["init", "-y", "--local"]);
    assert_eq!(stdout(&same), "配置已是最新：markitai.json\n");
    assert!(same.stderr.is_empty());

    let models = [
        ("MODEL", "openai/chosen"),
        ("OPENAI_API_KEY", "not-a-real-key"),
    ];
    let updated = run(
        root.path(),
        &[ZH, &models].concat(),
        &["init", "-y", "--local"],
    );
    assert_eq!(stdout(&updated), "配置已更新：markitai.json\n");
    assert!(updated.stderr.is_empty());

    // A new file with a detected model explains the next step in Chinese.
    let fresh = run(
        root.path(),
        &[ZH, &models].concat(),
        &["init", "-y", "-o", "other.json"],
    );
    assert_eq!(stdout(&fresh), "配置已创建：other.json\n");
    assert!(
        stderr(&fresh).ends_with(
            "已保存检测到的模型，但 LLM 处理仍处于关闭状态：转换时加上 --llm，或运行 `markitai config set llm.enabled true`。\n"
        ),
        "{}",
        stderr(&fresh)
    );

    // The files are the same whichever language wrote them.
    let english_root = tempfile::tempdir().unwrap();
    let english = run(english_root.path(), EN, &["init", "-y", "--local"]);
    assert_eq!(stdout(&english), "Configuration created: markitai.json\n");
    assert_eq!(
        std::fs::read(english_root.path().join("markitai.json")).unwrap(),
        file
    );
    let english = run(
        english_root.path(),
        &[EN, &models].concat(),
        &["init", "-y", "-o", "other.json"],
    );
    assert!(
        stderr(&english).ends_with(
            "add --llm to a conversion or run `markitai config set llm.enabled true`.\n"
        )
    );
    assert_eq!(
        std::fs::read(english_root.path().join("other.json")).unwrap(),
        std::fs::read(root.path().join("other.json")).unwrap()
    );

    // Without a terminal the wizard refuses, in Chinese, with the same status.
    let refused = run(root.path(), ZH, &["init"]);
    assert_eq!(refused.status.code(), Some(2));
    assert_eq!(
        stderr(&refused),
        "Error: 交互式配置需要终端；自动化场景请使用 config set 或 init --yes\n"
    );
    assert!(refused.stdout.is_empty());
    let edit = run(root.path(), ZH, &["config", "edit"]);
    assert_eq!(edit.status.code(), Some(2));
}

#[cfg(unix)]
mod wizard {
    use super::*;
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::process::Stdio;

    /// Runs the CLI on a pseudo terminal, feeds it `script` and returns its
    /// exit status, what it showed on the terminal and what it printed on
    /// standard output.
    pub fn terminal(
        root: &Path,
        envs: &[(&str, &str)],
        args: &[&str],
        script: &[u8],
    ) -> (std::process::ExitStatus, String, String) {
        let (mut master, mut slave) = (-1, -1);
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            0
        );
        let mut master = unsafe { std::fs::File::from_raw_fd(master) };
        let slave = unsafe { std::fs::File::from_raw_fd(slave) };
        let mut reader = master.try_clone().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        let read = std::thread::spawn(move || {
            let mut all = Vec::new();
            let mut buf = [0; 4096];
            while std::time::Instant::now() < deadline {
                let mut descriptor = libc::pollfd {
                    fd: reader.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                let ready = unsafe { libc::poll(&mut descriptor, 1, 100) };
                if ready < 0 {
                    if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                        continue;
                    }
                    break;
                }
                if ready == 0 {
                    continue;
                }
                match reader.read(&mut buf) {
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Ok(0) | Err(_) => break,
                    Ok(n) => all.extend_from_slice(&buf[..n]),
                }
            }
            all
        });
        let mut child = command(root, envs)
            .args(args)
            .stdin(slave.try_clone().unwrap())
            .stderr(slave)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        master.write_all(script).unwrap();
        let mut standard = child.stdout.take().unwrap();
        let output = std::thread::spawn(move || {
            let mut text = String::new();
            standard.read_to_string(&mut text).unwrap();
            text
        });
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("Interactive CLI did not finish");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        drop(master);
        let shown = String::from_utf8_lossy(&read.join().unwrap()).into_owned();
        (status, shown, output.join().unwrap())
    }
}

#[cfg(unix)]
#[test]
fn the_init_wizard_asks_and_answers_in_chinese() {
    use wizard::terminal;
    let root = tempfile::tempdir().unwrap();
    let models = [
        ("MODEL", "openai/chosen"),
        ("OPENAI_API_KEY", "not-a-real-key"),
    ];
    let envs = [ZH, &models].concat();

    // Where to save, then a project file.
    let (status, shown, printed) = terminal(root.path(), &envs, &["init"], b"x\n2\n");
    assert!(status.success(), "{shown}");
    for line in [
        "原生配置初始化；不会向任何供应商发出请求。",
        "检测到模型：openai/chosen",
        "保存到：1 用户配置，2 ./markitai.json，q 取消 [1]：",
        "请选择其中之一：1, 2",
        "后续步骤：",
    ] {
        assert!(shown.contains(line), "{line}\n{shown}");
    }
    assert_eq!(printed, "配置已创建：markitai.json\n");
    assert!(root.path().join("markitai.json").is_file());
    assert!(!root.path().join("home").exists());

    // The file exists now: ask what to do with it; keep it.
    let (status, shown, printed) = terminal(root.path(), &envs, &["init", "--local"], b"9\n3\n");
    assert!(status.success(), "{shown}");
    for line in [
        "配置文件已存在：markitai.json",
        "1 追加检测到的模型，2 覆盖，3 保留 [3]：",
        "请选择其中之一：1, 2, 3",
        "已保留 markitai.json",
    ] {
        assert!(shown.contains(line), "{line}\n{shown}");
    }
    assert_eq!(printed, "");

    // Cancelling and overwriting.
    let (status, shown, printed) = terminal(root.path(), &envs, &["init"], b"q\n");
    assert!(status.success(), "{shown}");
    assert_eq!(printed, "");
    let (status, shown, printed) =
        terminal(root.path(), ZH, &["init", "-o", "markitai.json"], b"2\n");
    assert!(status.success(), "{shown}");
    assert!(
        shown.contains("未检测到 API 模型，LLM 保持关闭。"),
        "{shown}"
    );
    assert_eq!(printed, "配置已创建：markitai.json\n");

    // English runs keep their prompts.
    let (status, shown, _) = terminal(
        root.path(),
        &[EN, &models].concat(),
        &["init", "--local"],
        b"3\n",
    );
    assert!(status.success(), "{shown}");
    assert!(
        shown.contains("1 update detected models, 2 overwrite, 3 keep [3]: "),
        "{shown}"
    );
    assert!(!has_chinese(&shown));
}

#[cfg(unix)]
#[test]
fn an_interrupted_batch_says_so_in_chinese_while_the_log_keeps_english() {
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    let root = tempfile::tempdir().unwrap();
    // A server that accepts the request and never answers keeps the item active.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/slow", listener.local_addr().unwrap());
    std::fs::write(root.path().join("list.urls"), format!("{url} slow\n")).unwrap();
    let log = root.path().join("logs");
    let config = json!({"log": {"dir": log}, "cache": {"enabled": false}}).to_string();
    let mut child = command(root.path(), ZH)
        .env("NO_PROXY", "127.0.0.1,localhost")
        .args(["list.urls", "-o", "out", "--config-json", &config])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let held = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                assert!(Instant::now() < deadline, "the CLI never asked for the URL");
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("accept failed: {error}"),
        }
    };
    let stderr = child.stderr.take().unwrap();
    let lines = std::thread::spawn(move || {
        BufReader::new(stderr)
            .lines()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
    });
    assert_eq!(
        unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) },
        0
    );
    // The acknowledgement comes first; the held request is then released.
    std::thread::sleep(Duration::from_millis(500));
    drop(held);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("the interrupted CLI did not exit");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(status.code(), Some(130));
    let lines = lines.join().unwrap();
    assert!(
        lines.contains(&"Interrupted: 不再派发新任务，等待正在进行的转换完成。".to_owned()),
        "{lines:?}"
    );
    let file = std::fs::read_dir(&log)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let text = std::fs::read_to_string(file).unwrap();
    assert!(
        text.contains(
            "| WARNING | cli | Interrupted: stopping new work and waiting for active conversions."
        ),
        "{text}"
    );
    assert!(!has_chinese(&text), "{text}");
}
