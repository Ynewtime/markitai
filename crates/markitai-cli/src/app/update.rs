//! `markitai update`: check for a newer release and, for a setup-script
//! install, replace this installation with it; any other install gets the
//! command that updates it.
//!
//! Exit status: 0 up to date or updated, 10 a newer release exists but this
//! install is not updated in place, 1 a failure (nothing changed, or put
//! back), 2 a downgrade that needs `--yes` without a terminal, 130 Ctrl-C.
//! The latest release is installed without a question; a downgrade asks
//! first, and so does restarting a running `markitai serve`.
use super::i18n::{self, Lang};
use super::logging::{self, Shape};
use super::{CliResult, progress};
use crate::term::{Mark, Style, Tone};
use crate::update::check::{self, Base, Failure};
use crate::update::control;
use crate::update::install::{self, Error};
use crate::update::method::{self, Method, SetupInstall};
use crate::update::version::Version;
use crate::update::{self as updates};
use serde_json::json;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub(super) const NEWER: i32 = 10;
const FAILED: i32 = 1;
const CANCELLED: i32 = 130;
const SETUP_SH: &str = "curl -fsSL https://markitai.dev/setup.sh | sh";
const SETUP_PS1: &str =
    "powershell -ExecutionPolicy ByPass -c \"irm https://markitai.dev/setup.ps1 | iex\"";
const PYTHON_1X: &str = "uv tool install \"git+https://github.com/Ynewtime/markitai@v1.2.0#subdirectory=packages/markitai\" --with \"pymupdf4llm<2\"";

pub(super) struct Options {
    pub(super) check: bool,
    pub(super) json: bool,
    pub(super) version: Option<String>,
    pub(super) yes: bool,
}

/// The lines, in the terminal language; nothing on stderr with `--json`
/// (stdout carries the object).
struct Out {
    lang: Lang,
    quiet: bool,
}

impl Out {
    fn say(&self, shape: Shape, english: String, chinese: String) {
        if self.quiet {
            return;
        }
        let console = (self.lang == Lang::Zh).then_some(chinese);
        logging::shaped(shape, &english, console.as_deref());
    }

    fn title(&self, english: String, chinese: String) {
        if self.quiet {
            return;
        }
        let text = if self.lang == Lang::Zh {
            chinese
        } else {
            english
        };
        logging::console(
            Shape {
                mark: Some(Mark::Title),
                indent: 0,
                tone: Some(Tone::Structure),
                wrap: true,
            },
            &text,
        );
    }

    /// A gray line under the title (the description, the link).
    fn under_title(&self, text: &str, wrap: bool) {
        if !self.quiet {
            logging::console(
                Shape {
                    mark: None,
                    indent: 2,
                    tone: Some(Tone::Secondary),
                    wrap,
                },
                text,
            );
        }
    }

    fn command(&self, command: &str) {
        if !self.quiet {
            logging::console(Shape::COMMAND, command);
        }
    }

    fn blank(&self) {
        if !self.quiet {
            logging::blank();
        }
    }

    /// `→ label` then the command alone on the next line.
    fn run(&self, english: String, chinese: String, command: &str) {
        self.say(Shape::item(Mark::Action), english, chinese);
        self.command(command);
    }
}

/// A path as the person reads it: `~/…` under the home folder.
fn shown(path: &Path) -> String {
    if !cfg!(windows)
        && let Some(home) = std::env::var_os("HOME").filter(|home| !home.is_empty())
        && let Ok(rest) = path.strip_prefix(&home)
    {
        return format!("~/{}", rest.display());
    }
    path.display().to_string()
}

fn elapsed(started: Instant) -> String {
    progress::clock(started.elapsed().as_secs())
}

fn interactive() -> bool {
    // Debug builds: the tests answer the prompts through a pipe.
    if cfg!(debug_assertions) && std::env::var_os("MARKITAI_TEST_UPDATE_INTERACTIVE").is_some() {
        return true;
    }
    io::stdin().is_terminal() && io::stderr().is_terminal()
}

/// `? question [y/N] `, answered from standard input.
fn ask(lang: Lang, english: &str, chinese: &str, default_yes: bool) -> bool {
    let question = if lang == Lang::Zh { chinese } else { english };
    progress::clear();
    let mut stderr = io::stderr().lock();
    let _ = write!(stderr, "{}", Style::stderr().prompt(2, question));
    let _ = stderr.flush();
    drop(stderr);
    let mut answer = String::new();
    let read = io::stdin().lock().read_line(&mut answer).unwrap_or(0);
    // Ctrl-C at the prompt is "no", whatever follows it (Enter alone included).
    if read == 0 || crate::signals::interrupted().is_some() {
        let _ = writeln!(io::stderr());
        return false;
    }
    match answer.trim().to_lowercase().as_str() {
        "" => default_yes,
        "y" | "yes" | "是" | "好" => true,
        _ => false,
    }
}

fn offline_words(reason: &str, lang: Lang) -> &'static str {
    match (reason, lang) {
        ("timeout", Lang::En) => ": connection timed out",
        ("timeout", Lang::Zh) => "：连接超时",
        ("refused", Lang::En) => ": connection refused",
        ("refused", Lang::Zh) => "：连接被拒绝",
        _ => "",
    }
}

/// A failed request in both languages.
fn failure_text(failure: &Failure) -> (String, String) {
    match failure {
        Failure::Offline { host, reason } => (
            format!("Can't reach {host}{}", offline_words(reason, Lang::En)),
            format!("无法连接 {host}{}", offline_words(reason, Lang::Zh)),
        ),
        Failure::RateLimited { host, status, .. } => (
            format!("{host} answered HTTP {status}"),
            format!("{host} 返回 HTTP {status}"),
        ),
        Failure::Status { host, code } => (
            format!("{host} answered HTTP {code}"),
            format!("{host} 返回 HTTP {code}"),
        ),
        Failure::Missing { what } => (format!("No release {what}"), format!("没有发行版 {what}")),
        Failure::Malformed { what } => (
            format!("Unreadable answer: {what}"),
            format!("无法读取的应答：{what}"),
        ),
        Failure::Setup(reason) => (reason.clone(), reason.clone()),
    }
}

struct Report {
    current: String,
    latest: Option<String>,
    newer: bool,
    method: &'static str,
    command: Option<String>,
    notes_url: Option<String>,
    action: &'static str,
    error: Option<(String, String)>,
}

impl Report {
    fn print(&self) {
        let mut value = json!({
            "current": self.current,
            "latest": self.latest,
            "newer": self.newer,
            "method": self.method,
            "command": self.command,
            "notes_url": self.notes_url,
            "action": self.action,
        });
        if let Some((code, reason)) = &self.error {
            value["error"] = json!({"code": code, "reason": reason});
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".into())
        );
    }
}

pub(super) fn run(options: &Options) -> CliResult<i32> {
    let lang = i18n::lang();
    let out = Out {
        lang,
        quiet: options.json,
    };
    let (current_text, _) = updates::current_identity();
    let detected = method::detect(&method::Facts::current());
    let mut report = Report {
        current: current_text.clone(),
        latest: None,
        newer: false,
        method: detected.key(),
        command: detected.command(),
        notes_url: None,
        action: "none",
        error: None,
    };
    let finish = |report: &Report, code: i32| -> CliResult<i32> {
        if options.json {
            report.print();
        }
        Ok(code)
    };
    if detected == Method::Source {
        out.say(
            Shape::verdict(Mark::Skip),
            format!("Source build ({current_text}); no update check"),
            format!("源码构建（{current_text}），不检查更新"),
        );
        return finish(&report, 0);
    }
    let Some(current) = Version::parse(&current_text) else {
        return finish(&report, 0);
    };
    let wanted = match &options.version {
        Some(text) => match Version::parse(text) {
            Some(version) if version.major < 2 => {
                out.say(
                    Shape::verdict(Mark::Error),
                    "markitai 1.x is the Python package and no longer on PyPI; install 1.2.0 from its Git tag:".into(),
                    "markitai 1.x 是 Python 包，已不在 PyPI 上；请从 Git tag 安装 1.2.0：".into(),
                );
                out.command(PYTHON_1X);
                report.error = Some(("unsupported_version".into(), "1.x".into()));
                return finish(&report, FAILED);
            }
            Some(version) => Some(version),
            None => {
                return Err((2, format!("--version needs X.Y.Z, not {text}")));
            }
        },
        None => None,
    };
    let base = match updates::base_for(&detected) {
        Ok(base) => base,
        Err(reason) => {
            out.say(Shape::verdict(Mark::Error), reason.clone(), reason.clone());
            report.error = Some(("offline".into(), reason));
            return finish(&report, FAILED);
        }
    };
    let home = markitai_core::config::home();
    // `sudo markitai update` would leave root's files in another user's install and home.
    let install = match &detected {
        Method::Setup(install) => Some(install),
        _ => None,
    };
    if let Some(owned) = not_roots(install, &home) {
        let owned = shown(&owned);
        out.say(
            Shape::verdict(Mark::Error),
            format!("{owned} isn't root's; run markitai update as its owner"),
            format!("{owned} 不属于 root；请用它的所有者运行 markitai update"),
        );
        report.error = Some(("not_writable".into(), format!("{owned} isn't root's")));
        return finish(&report, FAILED);
    }
    // The latest release, asked now; recorded for the notice too. The attempt
    // is recorded first (as a failure until it answers), so a check that
    // never finishes still backs off.
    let before = updates::attempting(&home);
    let latest = match check::latest(&base, check::COMMAND) {
        // A mirror's `latest` may name a pre-release: never installed unasked.
        Ok(latest) if latest.is_prerelease() && wanted.is_none() => {
            out.say(
                Shape::verdict(Mark::Success),
                format!("markitai {current} is up to date"),
                format!("markitai {current} 已是最新"),
            );
            return finish(&report, 0);
        }
        Ok(latest) => latest,
        Err(failure) => {
            if let Failure::RateLimited { retry_at, .. } = &failure {
                updates::rate_limited(&home, *retry_at);
            }
            let (english, chinese) = failure_text(&failure);
            report.error = Some((failure.code().into(), english.clone()));
            if options.check {
                out.say(Shape::verdict(Mark::Error), english, chinese);
            } else {
                out.title(
                    format!("Update markitai {current}"),
                    format!("更新 markitai {current}"),
                );
                out.blank();
                out.say(Shape::item(Mark::Error), english, chinese);
                nothing_changed(&out);
                out.blank();
                failed_verdict(&out);
            }
            return finish(&report, FAILED);
        }
    };
    report.latest = Some(latest.to_string());
    report.newer = latest > current;
    report.notes_url = Some(base.notes_url(&latest));
    // Saved before the description is asked for.
    updates::learned(&home, &latest);
    let cached = before
        .as_ref()
        .filter(|cache| {
            cache.latest.as_deref() == Some(&latest.to_string()) && cache.lead.is_some()
        })
        .map(|cache| (cache.lead.clone(), cache.lead_zh.clone()));
    let target = wanted.clone().unwrap_or_else(|| latest.clone());
    let zh = out.lang == Lang::Zh;
    let (lead, lead_zh) = if target == latest {
        cached.unwrap_or_else(|| {
            let leads = check::leads(&base, &latest, check::COMMAND);
            updates::described(&home, &latest, leads.clone());
            leads
        })
    } else if zh {
        check::leads(&base, &target, check::COMMAND)
    } else {
        (check::lead(&base, &target, check::COMMAND, false), None)
    };
    // In Chinese the Chinese description, else the English one.
    let lead = if zh { lead_zh.or(lead) } else { lead };
    if target == current || (wanted.is_none() && latest < current) {
        out.say(
            Shape::verdict(Mark::Success),
            format!("markitai {current} is up to date"),
            format!("markitai {current} 已是最新"),
        );
        return finish(&report, 0);
    }
    let command = detected
        .command()
        .unwrap_or_else(|| "markitai update".into());
    if options.check {
        if report.newer && !options.json {
            logging::console(
                Shape::PLAIN,
                &updates::notice_line(lang, Style::stderr(), 0, &latest, &current, &command),
            );
        } else if !report.newer {
            out.say(
                Shape::verdict(Mark::Success),
                format!("markitai {current} is up to date"),
                format!("markitai {current} 已是最新"),
            );
        }
        return finish(&report, if report.newer { NEWER } else { 0 });
    }
    let header = |out: &Out, title: (String, String)| {
        out.title(title.0, title.1);
        if let Some(lead) = &lead {
            out.under_title(lead, true);
        }
        out.under_title(&base.notes_url(&target), false);
        out.blank();
    };
    let Method::Setup(install) = &detected else {
        header(
            &out,
            (
                format!("markitai {current} → {target}"),
                format!("markitai {current} → {target}"),
            ),
        );
        not_in_place(&out, &detected);
        return finish(&report, NEWER);
    };
    let downgrade = target < current;
    header(
        &out,
        if downgrade {
            (
                format!("Downgrade markitai {current} → {target}"),
                format!("降级 markitai {current} → {target}"),
            )
        } else {
            (
                format!("Update markitai {current} → {target}"),
                format!("更新 markitai {current} → {target}"),
            )
        },
    );
    if downgrade && !options.yes {
        if options.json || !interactive() {
            out.say(
                Shape::item(Mark::Error),
                "Needs confirmation: run it in a terminal, or add --yes".into(),
                "需要确认：在终端里运行，或加 --yes".into(),
            );
            out.blank();
            out.say(
                Shape::verdict(Mark::Skip),
                "Not updated".into(),
                "未更新".into(),
            );
            report.error = Some(("confirmation_needed".into(), "downgrade needs --yes".into()));
            return finish(&report, 2);
        }
        if !ask(
            lang,
            &format!("{target} is older than this version; continue? [y/N]"),
            &format!("{target} 比当前旧，继续？[y/N]"),
            false,
        ) {
            out.blank();
            out.say(
                Shape::verdict(Mark::Skip),
                "Not updated".into(),
                "未更新".into(),
            );
            return finish(&report, 0);
        }
    }
    let started = Instant::now();
    match apply(&out, install, &base, &target, options) {
        Ok(restarted) => {
            report.action = if restarted { "restarted" } else { "updated" };
            // Interrupted at the restart prompt: installed, nothing restarted.
            let interrupted = crate::signals::interrupted().is_some();
            out.blank();
            let time = elapsed(started);
            if downgrade {
                out.say(
                    Shape::verdict(Mark::Success),
                    format!("Downgraded to {target} ({time})"),
                    format!("已降级到 {target}（{time}）"),
                );
            } else {
                out.say(
                    Shape::verdict(Mark::Success),
                    format!("Updated to {target} ({time})"),
                    format!("已更新到 {target}（{time}）"),
                );
            }
            finish(&report, if interrupted { CANCELLED } else { 0 })
        }
        Err(error) => {
            report.error = Some((error.code().into(), error.reason()));
            explain(&out, &error, &current, install);
            out.blank();
            if matches!(error, Error::Cancelled) {
                out.say(
                    Shape::verdict(Mark::Skip),
                    "Nothing changed".into(),
                    "未做改动".into(),
                );
                finish(&report, CANCELLED)
            } else {
                failed_verdict(&out);
                finish(&report, FAILED)
            }
        }
    }
}

/// Run as root: the first of the folders an update writes (the install's,
/// `MARKITAI_HOME`) that belongs to another user. A folder not made yet is
/// judged by the folder it would be made in.
fn not_roots(install: Option<&SetupInstall>, home: &Path) -> Option<PathBuf> {
    #[cfg(unix)]
    {
        // SAFETY: geteuid only reads this process's effective user.
        if unsafe { libc::geteuid() } != 0 {
            return None;
        }
        not_owned_by(0, install, home)
    }
    #[cfg(not(unix))]
    {
        let _ = (install, home);
        None
    }
}

#[cfg(unix)]
fn not_owned_by(user: u32, install: Option<&SetupInstall>, home: &Path) -> Option<PathBuf> {
    use std::os::unix::fs::MetadataExt;
    let mut folders = vec![home.to_owned()];
    if let Some(install) = install {
        folders.push(install.dir.clone());
        folders.extend(install.receipt_dir.parent().map(Path::to_owned));
    }
    folders.into_iter().find(|folder| {
        folder
            .ancestors()
            .find_map(|path| std::fs::metadata(path).ok())
            .is_some_and(|metadata| metadata.uid() != user)
    })
}

fn nothing_changed(out: &Out) {
    out.say(
        Shape::item(Mark::Info),
        "Nothing changed".into(),
        "未做改动".into(),
    );
}

fn failed_verdict(out: &Out) {
    out.say(
        Shape::verdict(Mark::Error),
        "Update failed".into(),
        "更新失败".into(),
    );
}

/// The steps for an install markitai does not replace itself.
fn not_in_place(out: &Out, detected: &Method) {
    match detected {
        Method::Unknown { path } => {
            let path = path
                .as_deref()
                .map(shown)
                .unwrap_or_else(|| "markitai".into());
            out.say(
                Shape::item(Mark::Warning),
                format!("Can't tell how this copy was installed: {path}"),
                format!("无法确定安装方式：{path}"),
            );
            out.run(
                "Download it from the release page, or reinstall with the setup script:".into(),
                "从发行页下载，或用安装脚本重装：".into(),
                if cfg!(windows) { SETUP_PS1 } else { SETUP_SH },
            );
        }
        other => {
            let label = other.label();
            out.say(
                Shape::item(Mark::Info),
                format!("Installed with: {label}"),
                format!("安装方式：{label}"),
            );
            match other.command() {
                Some(command) => out.run("Run:".into(), "运行：".into(), &command),
                None => out.say(
                    Shape::item(Mark::Action),
                    format!("Update it with {label}"),
                    format!("用 {label} 更新"),
                ),
            }
        }
    }
}

/// [`Error::reason`] in Chinese, for the serve terminal.
pub(crate) fn reason_zh(error: &Error) -> String {
    match error {
        Error::Fetch(failure) => failure_text(failure).1,
        Error::Checksum { expected, actual } => format!(
            "SHA-256 不匹配，已删除下载的文件（{} ≠ {}）",
            install::short_digest(actual),
            install::short_digest(expected)
        ),
        Error::DiskFull { dir, needed } => format!(
            "磁盘空间不足：{} 需要 {}",
            dir.display(),
            install::megabytes(*needed)
        ),
        Error::NotWritable { dir, .. } => format!("无法写入 {}：权限不足", dir.display()),
        Error::FileBusy { name } => format!("无法替换 {name}：另一个程序正在使用"),
        Error::StartFailed { reason, .. } => format!("新版本无法启动：{reason}"),
        Error::Foreign { dir, names } => format!(
            "{} 里有不是这次安装创建的文件：{}",
            dir.display(),
            names.join("、")
        ),
        Error::Archive(reason) => format!("发行归档无法使用：{reason}"),
        Error::Io { error, .. } => format!("无法安装：{error}"),
        Error::Cancelled => "已取消".into(),
        Error::Busy => "另一个更新正在进行".into(),
    }
}

/// What a failure means and what to do, as step lines.
fn explain(out: &Out, error: &Error, current: &Version, install: &SetupInstall) {
    let step =
        |english: String, chinese: String| out.say(Shape::item(Mark::Error), english, chinese);
    match error {
        Error::Fetch(failure) => {
            let (english, chinese) = failure_text(failure);
            step(english, chinese);
        }
        Error::Checksum { expected, actual } => {
            step(
                "SHA-256 mismatch; the download was deleted".into(),
                "SHA-256 不匹配，已删除下载的文件".into(),
            );
            let detail = format!(
                "{} ≠ {}",
                install::short_digest(actual),
                install::short_digest(expected)
            );
            if !out.quiet {
                logging::console(
                    Shape {
                        mark: None,
                        indent: 4,
                        tone: Some(Tone::Secondary),
                        wrap: false,
                    },
                    &detail,
                );
            }
        }
        Error::DiskFull { dir, needed } => {
            let (dir, size) = (shown(dir), install::megabytes(*needed));
            step(
                format!("Not enough disk space: {dir} needs {size}"),
                format!("磁盘空间不足：{dir} 需要 {size}"),
            );
        }
        Error::NotWritable { dir, .. } => {
            let dir = shown(dir);
            step(
                format!("Can't write to {dir}: permission denied"),
                format!("无法写入 {dir}：权限不足"),
            );
        }
        Error::FileBusy { name } => step(
            format!("Can't replace {name}: another program is using it"),
            format!("无法替换 {name}：另一个程序正在使用"),
        ),
        Error::StartFailed { reason, .. } => step(
            format!("The new version didn't start: {reason}"),
            format!("新版本无法启动：{reason}"),
        ),
        Error::Foreign { dir, names } => {
            let dir = shown(dir);
            step(
                format!(
                    "{dir} holds files this install didn't create: {}",
                    names.join(", ")
                ),
                format!("{dir} 里有不是这次安装创建的文件：{}", names.join("、")),
            );
        }
        Error::Archive(reason) => step(
            format!("The release archive is unusable: {reason}"),
            format!("发行归档无法使用：{reason}"),
        ),
        Error::Io { error, .. } => step(
            format!("Can't install: {error}"),
            format!("无法安装：{error}"),
        ),
        Error::Busy => step(
            "Another update is running; try again when it has finished".into(),
            "另一个更新正在进行；它结束后再试".into(),
        ),
        Error::Cancelled => {
            out.say(
                Shape::item(Mark::Skip),
                "Cancelled; the partial download was deleted".into(),
                "已取消，删除了下载到一半的文件".into(),
            );
            return;
        }
    }
    if error.rolled_back() {
        out.say(
            Shape::item(Mark::Info),
            format!("Rolled back; still {current}"),
            format!("已恢复，仍是 {current}"),
        );
    } else {
        nothing_changed(out);
    }
    match error {
        Error::FileBusy { .. } => out.run(
            "Close the program using it (such as an MCP client) and try again:".into(),
            "关闭使用它的程序（如 MCP 客户端）后重试：".into(),
            "markitai update",
        ),
        Error::NotWritable {
            root_owned: true, ..
        } if !cfg!(windows) => out.run(
            "Run it as the user who installed it:".into(),
            "用安装它的用户运行：".into(),
            "sudo markitai update",
        ),
        Error::Foreign { .. } => out.run(
            "Move them away, or reinstall with the setup script:".into(),
            "移走它们，或用安装脚本重装：".into(),
            if cfg!(windows) { SETUP_PS1 } else { SETUP_SH },
        ),
        _ => {}
    }
    let _ = install;
}

/// Download, check, dry-start and install `target`; then offer to restart a
/// running server. Returns whether a server was restarted.
fn apply(
    out: &Out,
    install: &SetupInstall,
    base: &Base,
    target: &Version,
    options: &Options,
) -> Result<bool, Error> {
    let home = markitai_core::config::home();
    // Ctrl-C is recorded and acted on between steps until the swap begins.
    let _guard = crate::signals::Guard::install().ok();
    let cancelled = || crate::signals::interrupted().is_some();
    let _exclusive = install::exclusive(&home)?;
    let pending = control::Pending::load(&home);
    let keep: Vec<_> = pending
        .iter()
        .flat_map(|pending| pending.swapped.aside.iter().map(|(_, aside)| aside.clone()))
        .collect();
    install::remove_leftovers(install, &keep);
    let work = install::prepare(install)?;
    let release = install::resolve(base, target, &install.target(), check::COMMAND)?;
    let downloaded = {
        let started = Instant::now();
        let mut status = progress::Status::new(!out.quiet && progress::wanted(false, false));
        let lang = out.lang;
        install::download(
            &work,
            &release,
            check::COMMAND,
            &mut |done, total| {
                let amount = match total {
                    Some(total) => format!(
                        "{:.1} / {:.1} MB",
                        done as f64 / 1_000_000.0,
                        total as f64 / 1_000_000.0
                    ),
                    None => install::megabytes(done),
                };
                let clock = elapsed(started);
                match lang {
                    Lang::En => status.show(
                        &format!("Downloading {amount}"),
                        &format!("  {clock} elapsed"),
                    ),
                    Lang::Zh => {
                        status.show(&format!("正在下载 {amount}"), &format!("  已用时 {clock}"))
                    }
                }
            },
            &cancelled,
        )?
    };
    let size = install::megabytes(downloaded.size);
    let archive = &release.archive;
    out.say(
        Shape::item(Mark::Success),
        format!("Downloaded {archive} ({size})"),
        format!("已下载 {archive}（{size}）"),
    );
    out.say(
        Shape::item(Mark::Success),
        "SHA-256 verified".into(),
        "已校验 SHA-256".into(),
    );
    let unpacked = install::unpack(&work, &downloaded)?;
    let program = install::program(&unpacked, install.script)?;
    install::dry_start(&program, &["--version".into()], Some(&format!(" {target}")))?;
    if cancelled() {
        return Err(Error::Cancelled);
    }
    let swapped = install::swap(install, &release, &unpacked)?;
    let installed = shown(&install.exe);
    out.say(
        Shape::item(Mark::Success),
        format!("Installed: {installed}"),
        format!("已安装：{installed}"),
    );
    let restarted = offer_restart(out, install, target, options, &home, &swapped)?;
    if !restarted.kept {
        swapped.commit();
    }
    drop(work);
    Ok(restarted.done)
}

struct Restarted {
    done: bool,
    /// The old program stays aside: the restarted server commits it.
    kept: bool,
}

/// R2: a running server of this home is offered a restart into `target`;
/// without a terminal, with `--yes` or `--json` only the hint.
fn offer_restart(
    out: &Out,
    install: &SetupInstall,
    target: &Version,
    options: &Options,
    home: &Path,
    swapped: &install::Swapped,
) -> Result<Restarted, Error> {
    let none = Ok(Restarted {
        done: false,
        kept: false,
    });
    let Some((control, status)) = control::running(home) else {
        return none;
    };
    let hint = |out: &Out| {
        out.run(
            format!("Restart serve to use {target}:"),
            format!("重启 serve 以使用 {target}："),
            "markitai serve",
        );
    };
    let address = control.address.clone();
    let version = status.version.clone();
    // Only a server running this install's program restarts into the new version.
    let restartable = control.runs(&install.exe)
        && control::supports_handover(target)
        && Version::parse(&status.version)
            .is_some_and(|running| control::supports_handover(&running));
    if options.yes || options.json || !interactive() || !restartable {
        hint(out);
        return none;
    }
    out.say(
        Shape::item(Mark::Warning),
        format!("markitai serve is running: {address} ({version})"),
        format!("markitai serve 正在运行：{address}（{version}）"),
    );
    let count = status.waiting_jobs;
    let (english, chinese) = if count == 0 {
        (
            "Restart it now? [Y/n]".to_owned(),
            "现在重启它？[Y/n]".to_owned(),
        )
    } else {
        (
            format!(
                "Restart it now ({count} conversion{} finish{} first)? [Y/n]",
                if count == 1 { "" } else { "s" },
                if count == 1 { "es" } else { "" }
            ),
            format!("现在重启它（{count} 个转换会先完成）？[Y/n]"),
        )
    };
    if !ask(out.lang, &english, &chinese, true) {
        hint(out);
        return none;
    }
    let pending = control::Pending::new(
        &status.version,
        &target.to_string(),
        install.exe.clone(),
        swapped.clone(),
        control.pid,
    );
    if pending.save(home).is_err() {
        hint(out);
        return none;
    }
    if let Err(reason) = control::request_restart(&control) {
        control::Pending::clear(home);
        out.say(
            Shape::item(Mark::Error),
            format!("markitai serve didn't restart: {reason}"),
            format!("markitai serve 没有重启：{reason}"),
        );
        hint(out);
        return none;
    }
    let mut status_line = progress::Status::new(progress::wanted(false, false));
    let started = Instant::now();
    let lang = out.lang;
    let waited = control::wait_for_restart(
        home,
        &control,
        &target.to_string(),
        Duration::from_secs(30 * 60),
        &mut |count| {
            let clock = elapsed(started);
            match lang {
                Lang::En => status_line.show(
                    &format!(
                        "Waiting for {count} conversion{} to finish",
                        if count == 1 { "" } else { "s" }
                    ),
                    &format!("  {clock} elapsed"),
                ),
                Lang::Zh => status_line.show(
                    &format!("等待 {count} 个转换完成"),
                    &format!("  已用时 {clock}"),
                ),
            }
        },
        &|| crate::signals::interrupted().is_some(),
    );
    drop(status_line);
    match waited {
        control::Restart::Done(status) => {
            let version = status.version;
            out.say(
                Shape::item(Mark::Success),
                format!("markitai serve restarted: {version}"),
                format!("markitai serve 已重启：{version}"),
            );
            Ok(Restarted {
                done: true,
                kept: true,
            })
        }
        // The server put the old version back.
        control::Restart::RolledBack(reason) => Err(Error::StartFailed {
            reason,
            rolled_back: true,
        }),
        control::Restart::Other(version) => {
            out.say(
                Shape::item(Mark::Warning),
                format!("markitai serve came back as {version}"),
                format!("markitai serve 重启后仍是 {version}"),
            );
            control::Pending::clear(home);
            Ok(Restarted {
                done: false,
                kept: false,
            })
        }
        control::Restart::Pending => {
            out.say(
                Shape::item(Mark::Info),
                "markitai serve restarts by itself once its conversions finish".into(),
                "markitai serve 会在转换完成后自行重启".into(),
            );
            // The server owns the pending marker now and commits it.
            Ok(Restarted {
                done: false,
                kept: true,
            })
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn root_is_refused_over_another_users_install_or_home() {
        use std::os::unix::fs::MetadataExt;
        let root = tempfile::tempdir().unwrap();
        let me = std::fs::metadata(root.path()).unwrap().uid();
        let bin = root.path().join("bin");
        std::fs::create_dir(&bin).unwrap();
        let install = SetupInstall {
            script: method::Script::Shell,
            exe: bin.join("markitai"),
            dir: bin,
            receipt_dir: root.path().join("share/markitai/current"),
            receipt: method::Receipt::parse("version=2.1.0\n"),
        };
        // A home not made yet is judged by the folder it would be made in.
        let home = root.path().join("home/.markitai");
        assert_eq!(not_owned_by(me, Some(&install), &home), None);
        assert_eq!(not_owned_by(me + 1, None, &home), Some(home.clone()));
        assert_eq!(
            not_owned_by(me + 1, Some(&install), &home),
            Some(home.clone())
        );
    }
}
