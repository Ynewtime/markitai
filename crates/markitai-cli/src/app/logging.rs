//! The CLI owns its sinks; core and language bindings do not install global loggers.
use crate::term::{Mark, Style, Tone};
use chrono::{DateTime, Local, NaiveDateTime};
use markitai_core::config;
use markitai_core::config::log_policy::{self, Rotation};
use serde_json::{Value, json};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

static ACTIVE: Mutex<Option<Session>> = Mutex::new(None);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Level {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}
impl Level {
    fn parse(value: &str) -> io::Result<Self> {
        match value {
            "DEBUG" => Ok(Self::Debug),
            "INFO" => Ok(Self::Info),
            "WARNING" => Ok(Self::Warning),
            "ERROR" => Ok(Self::Error),
            "CRITICAL" => Ok(Self::Critical),
            _ => Err(invalid("Invalid file log level")),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }
}

struct Sink {
    file: File,
    _session: SessionLock,
    first_path: PathBuf,
    stem: String,
    sequence: u64,
    bytes: u64,
    rotation: Rotation,
    /// When the current file is rotated by time: the first local time after
    /// it opened that `rotation` names.
    limit: Option<NaiveDateTime>,
    json: bool,
    minimum: Level,
    failed: bool,
}
struct Session {
    sink: Option<Sink>,
    secrets: Vec<String>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

pub(super) fn start(cfg: &Value, level: Option<&str>) -> io::Result<()> {
    let env = config::environment();
    let directory = env
        .get("MARKITAI_LOG_DIR")
        .filter(|v| !v.is_empty())
        .map(String::as_str)
        .or_else(|| cfg.pointer("/log/dir").and_then(Value::as_str))
        .filter(|v| !v.is_empty());
    let mut secrets = Vec::new();
    collect_secrets(cfg, &config::redact(cfg), &mut secrets);
    for (key, value) in &env {
        let key = key.to_ascii_uppercase();
        if ["API_KEY", "TOKEN", "PASSWORD", "SECRET", "AUTHORIZATION"]
            .iter()
            .any(|part| key.contains(part))
            && !value.is_empty()
        {
            secrets.push(value.clone());
        }
    }
    crate::sort::by_key(&mut secrets, |value| std::cmp::Reverse(value.len()));
    secrets.dedup();
    let sink = directory
        .map(|directory| {
            let directory = config::state_path(Path::new(directory));
            // Configuration validation runs the same parsers, so a value
            // that validates is never refused here.
            let rotation = log_policy::rotation(
                cfg.pointer("/log/rotation")
                    .and_then(Value::as_str)
                    .unwrap_or("10 MB"),
            )
            .map_err(|error| invalid(&error.to_string()))?;
            let retention = log_policy::retention(
                cfg.pointer("/log/retention")
                    .and_then(Value::as_str)
                    .unwrap_or("7 days"),
            )
            .map_err(|error| invalid(&error.to_string()))?;
            let minimum = Level::parse(
                level
                    .or_else(|| cfg.pointer("/log/level").and_then(Value::as_str))
                    .unwrap_or("INFO"),
            )?;
            let format = env
                .get("MARKITAI_LOG_FORMAT")
                .filter(|v| matches!(v.as_str(), "text" | "json"))
                .map(String::as_str)
                .or_else(|| cfg.pointer("/log/format").and_then(Value::as_str))
                .unwrap_or("text");
            private_directory(&directory)?;
            prune(&directory, retention)?;
            let opened = Local::now();
            let stem = format!(
                "markitai_{}_{}",
                opened.format("%Y%m%d_%H%M%S_%6f"),
                std::process::id()
            );
            let first_path = directory.join(format!("{stem}.log"));
            let session = SessionLock::acquire(&directory.join(format!("{stem}.lock")))?;
            let file = private_file(&first_path)?;
            Ok::<_, io::Error>(Sink {
                file,
                _session: session,
                first_path,
                stem,
                sequence: 0,
                bytes: 0,
                rotation,
                limit: rotation.next_after(opened.naive_local()),
                json: format == "json",
                minimum,
                failed: false,
            })
        })
        .transpose()?;
    let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
    *active = Some(Session { sink, secrets });
    Ok(())
}

fn collect_secrets(original: &Value, redacted: &Value, out: &mut Vec<String>) {
    if original != redacted && redacted.as_str() == Some("[REDACTED]") {
        match original {
            Value::String(value) if !value.is_empty() && !value.starts_with("env:") => {
                out.push(value.clone())
            }
            Value::Object(values) => {
                for value in values.values() {
                    collect_secrets(value, &json!("[REDACTED]"), out);
                }
            }
            Value::Array(values) => {
                for value in values {
                    collect_secrets(value, &json!("[REDACTED]"), out);
                }
            }
            _ => {}
        }
    } else {
        match original {
            Value::Object(values) => {
                for (key, value) in values {
                    collect_secrets(value, &redacted[key], out);
                }
            }
            Value::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    collect_secrets(value, &redacted[index], out);
                }
            }
            _ => {}
        }
    }
}

pub(super) fn path() -> Option<PathBuf> {
    ACTIVE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|session| session.sink.as_ref().map(|sink| sink.first_path.clone()))
}

pub(super) fn event(level: Level, message: impl AsRef<str>) {
    let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(session) = active.as_mut() else {
        return;
    };
    let Some(sink) = session.sink.as_mut() else {
        return;
    };
    if level < sink.minimum || sink.failed {
        return;
    }
    let message = redact(message.as_ref(), &session.secrets);
    if sink.write(level, &message).is_err() {
        sink.failed = true;
    }
}

/// How a diagnostic sits on the terminal: its mark, the columns before it
/// and the tone of its text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    pub(crate) mark: Option<Mark>,
    pub(crate) indent: usize,
    pub(crate) tone: Option<Tone>,
    /// Whether a long line wraps on the terminal; a command to copy or a line
    /// scripts read does not.
    pub(crate) wrap: bool,
}

impl Shape {
    /// A run's item line: `  <mark> text`.
    pub(crate) const fn item(mark: Mark) -> Self {
        Self {
            mark: Some(mark),
            indent: 2,
            tone: None,
            wrap: true,
        }
    }
    /// A command's closing line: the mark at column 0.
    pub(crate) const fn verdict(mark: Mark) -> Self {
        Self {
            mark: Some(mark),
            indent: 0,
            tone: None,
            wrap: true,
        }
    }
    /// Text under an item's text, without a mark.
    pub(crate) const fn under(indent: usize) -> Self {
        Self {
            mark: None,
            indent,
            tone: None,
            wrap: true,
        }
    }
    /// A section header: bold at column 0.
    pub(crate) const SECTION: Self = Self {
        mark: None,
        indent: 0,
        tone: Some(Tone::Structure),
        wrap: true,
    };
    /// A command to copy under an item: its own line, the accent colour,
    /// never wrapped.
    pub(crate) const COMMAND: Self = Self {
        mark: None,
        indent: 4,
        tone: Some(Tone::Accent),
        wrap: false,
    };
    /// A line printed as it is.
    pub(crate) const PLAIN: Self = Self {
        mark: None,
        indent: 0,
        tone: None,
        wrap: true,
    };
}

/// The labels a diagnostic's English text starts with, the file log level
/// each stands for and the mark the terminal shows instead. The file log
/// keeps the label so it can be searched; `Interrupted:` and `Note:` stay in
/// the text on the terminal too.
fn labelled(english: &str) -> (Level, Shape, usize) {
    for (label, level, mark, keep) in [
        ("Error: ", Level::Error, Mark::Error, false),
        ("Warning: ", Level::Warning, Mark::Warning, false),
        ("Hint: ", Level::Info, Mark::Action, false),
        ("Interrupted: ", Level::Warning, Mark::Warning, true),
        ("Note: ", Level::Info, Mark::Info, true),
        // An explanatory notice shown with -v: the info mark, no label.
        ("Info: ", Level::Info, Mark::Info, false),
    ] {
        if english.starts_with(label) {
            return (level, Shape::item(mark), if keep { 0 } else { label.len() });
        }
    }
    (Level::Info, Shape::PLAIN, 0)
}

/// All existing CLI diagnostics keep their console policy and gain redaction.
/// A leading `Error:`, `Warning:` or `Hint:` label becomes the matching mark
/// on the terminal (see [`labelled`]).
pub(crate) fn diagnostic(args: std::fmt::Arguments<'_>) {
    emit(args.to_string(), None, None);
}

/// A diagnostic whose console text is in the terminal language while the file
/// log keeps `english`: log lines read the same in every language.
pub(super) fn diagnostic_as(english: std::fmt::Arguments<'_>, console: std::fmt::Arguments<'_>) {
    emit(english.to_string(), Some(console.to_string()), None);
}

/// A diagnostic in an explicit shape. The file log gets `english` as it is,
/// at the level its label stands for (see [`labelled`]); a label is left off
/// the terminal, where the mark says it.
pub(crate) fn shaped(shape: Shape, english: &str, console: Option<&str>) {
    emit(english.to_owned(), console.map(str::to_owned), Some(shape));
}

/// The error that ends a command, at column 0: `✗ message`. The file log
/// keeps `Error: message`.
pub(super) fn fatal(message: &str) {
    emit(
        format!("Error: {message}"),
        None,
        Some(Shape::verdict(Mark::Error)),
    );
}

fn emit(original: String, console: Option<String>, shape: Option<Shape>) {
    let message = {
        let active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        redact(
            console.as_deref().unwrap_or(&original),
            active.as_ref().map(|s| s.secrets.as_slice()).unwrap_or(&[]),
        )
    };
    // The file log's level and text come from the English line alone; the
    // shape only places the line on the terminal.
    let (level, label_shape, label) = labelled(&original);
    let shape = shape.unwrap_or(label_shape);
    // The console text carries the same label as the English one.
    let body = message.strip_prefix(&original[..label]).unwrap_or(&message);
    // A line built without the terminal language (a run-level error, a
    // converter's reason) is translated here when it is a known message, so a
    // Chinese run is Chinese throughout; the file log keeps the English.
    let body = if console.is_none() && super::i18n::lang() == super::i18n::Lang::Zh {
        super::reasons::localize(body)
    } else {
        body.to_owned()
    };
    event(level, &original);
    // A status line on the terminal gives way to the line that follows it.
    super::progress::clear();
    let line = render(shape, &body);
    write_line(&line);
}

/// The library's own stderr lines (see `markitai_core::console`) that came
/// before the terminal language was known; `None` once it is.
static EARLY: Mutex<Option<Vec<String>>> = Mutex::new(Some(Vec::new()));

/// The library's sink: its lines are diagnostics like the CLI's own, with
/// marks, the file log and the terminal language. Choosing the language reads
/// the environment, which may itself warn (a folder's `.env`), so a line that
/// comes before the language is known waits for [`language_known`].
pub(super) fn library_line(text: &str) {
    if let Some(early) = EARLY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        early.push(text.to_owned());
        return;
    }
    emit(text.to_owned(), None, None);
}

/// The terminal language is known: print the library lines that waited for it.
pub(super) fn language_known() {
    let early = EARLY.lock().unwrap_or_else(|e| e.into_inner()).take();
    for line in early.unwrap_or_default() {
        emit(line, None, None);
    }
}

/// A line for the terminal alone, as it is: neither redacted nor logged.
/// For the service's startup lines, whose addresses carry the access token
/// the person who started it needs to copy.
pub(crate) fn console(shape: Shape, text: &str) {
    super::progress::clear();
    write_line(&render(shape, text));
}

/// `text` in `shape` as stderr shows it.
fn render(shape: Shape, text: &str) -> String {
    let style = Style::stderr();
    if shape.wrap {
        return style.line(shape.indent, shape.mark, text, shape.tone);
    }
    let mut line = " ".repeat(shape.indent);
    if let Some(mark) = shape.mark {
        line.push_str(&style.mark(mark));
        line.push(' ');
    }
    match shape.tone {
        Some(tone) => line.push_str(&style.paint(tone, text)),
        None => line.push_str(text),
    }
    line
}

/// An empty line on stderr between blocks; the file log does not record it.
pub(crate) fn blank() {
    super::progress::clear();
    write_line("");
}

/// One line on stderr. A failed write is ignored: the run's outcome does not
/// depend on its diagnostics. Unit tests print through the test harness,
/// which captures the lines of passing tests.
fn write_line(line: &str) {
    #[cfg(test)]
    std::eprintln!("{line}");
    #[cfg(not(test))]
    let _ = writeln!(io::stderr().lock(), "{line}");
}

pub(super) fn finish(code: i32) -> io::Result<()> {
    event(
        if code == 0 {
            Level::Info
        } else {
            Level::Warning
        },
        format!("Run finished with exit code {code}"),
    );
    let session = ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(mut sink) = session.and_then(|s| s.sink) {
        sink.file.flush()?;
        sink.file.sync_data()?;
        if sink.failed {
            return Err(io::Error::other(
                "File logging failed; conversion output has been preserved",
            ));
        }
    }
    Ok(())
}

impl Sink {
    fn write(&mut self, level: Level, message: &str) -> io::Result<()> {
        self.write_at(level, message, Local::now())
    }

    /// Write one record at `now`, first rotating to the session's next file
    /// when the record would take the current one past the size limit or the
    /// time limit has been reached. An empty file is not rotated.
    fn write_at(&mut self, level: Level, message: &str, now: DateTime<Local>) -> io::Result<()> {
        let line = if self.json {
            format!(
                "{}\n",
                json!({"ts": now.format("%Y-%m-%dT%H:%M:%S").to_string(), "lvl":level.name(), "src":"cli", "msg":message})
            )
        } else {
            format!(
                "{} | {:<5} | cli | {}\n",
                now.format("%Y-%m-%d %H:%M:%S"),
                level.name(),
                message.replace('\n', "\\n").replace('\r', "\\r")
            )
        };
        let local = now.naive_local();
        let due = match self.rotation {
            Rotation::Size(limit) => self.bytes.saturating_add(line.len() as u64) > limit,
            _ => self.limit.is_some_and(|limit| local >= limit),
        };
        if due && self.bytes == 0 {
            self.limit = self.rotation.next_after(local);
        } else if due {
            self.file.flush()?;
            self.file.sync_data()?;
            self.sequence = self
                .sequence
                .checked_add(1)
                .ok_or_else(|| io::Error::other("Log rotation exhausted"))?;
            self.file = private_file(
                &self
                    .first_path
                    .with_file_name(format!("{}_{:06}.log", self.stem, self.sequence)),
            )?;
            self.bytes = 0;
            self.limit = self.rotation.next_after(local);
        }
        self.file.write_all(line.as_bytes())?;
        self.bytes += line.len() as u64;
        Ok(())
    }
}

fn private_directory(path: &Path) -> io::Result<()> {
    if fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink() || !m.is_dir()) {
        return Err(invalid("Log directory must be a directory, not a symlink"));
    }
    markitai_core::platform::private_directory()
        .recursive(true)
        .create(path)
}
fn private_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    markitai_core::platform::private_file(&mut options).open(path)
}

/// A session's liveness: `<stem>.lock`, held while its logs may still grow.
/// The logs themselves stay unlocked because a Windows lock is mandatory and
/// would refuse anyone reading a live log. The session removes the lock when
/// it ends; an abandoned one is pruned with the logs.
struct SessionLock {
    file: File,
    path: PathBuf,
}

impl SessionLock {
    fn acquire(path: &Path) -> io::Result<Self> {
        let file = private_file(path)?;
        file.lock()?;
        Ok(Self {
            file,
            path: path.to_owned(),
        })
    }
}

impl Drop for SessionLock {
    fn drop(&mut self) {
        // Removed while still held, so no pruner can take it in between.
        let _ = fs::remove_file(&self.path);
        let _ = self.file.unlock();
    }
}

/// Lock `path` exclusively when no live session holds it.
fn released(path: &Path) -> Option<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    let file = markitai_core::platform::open_no_follow(&options, path).ok()?;
    file.try_lock().is_ok().then_some(file)
}

/// Remove expired logs whose session has ended, then expired session locks
/// nobody holds. A log's session is alive while its `<stem>.lock` is held; a
/// log without one (written by an earlier version, which locked the log
/// itself) is tested by locking the log.
fn prune(directory: &Path, retention: Duration) -> io::Result<()> {
    let now = SystemTime::now();
    let mut locks = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(owned) = owned_name(name) else {
            continue;
        };
        let metadata = fs::symlink_metadata(entry.path())?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || !now
                .duration_since(metadata.modified()?)
                .is_ok_and(|age| age > retention)
        {
            continue;
        }
        match owned {
            Owned::Lock => locks.push(entry.path()),
            Owned::Log(stem) => {
                let owner = session_lock(directory, stem).unwrap_or_else(|| entry.path());
                if let Some(_held) = released(&owner) {
                    fs::remove_file(entry.path())?;
                }
            }
        }
    }
    for lock in locks {
        if let Some(_held) = released(&lock) {
            let _ = fs::remove_file(&lock);
        }
    }
    Ok(())
}

enum Owned<'a> {
    /// A log, with its name before `.log`.
    Log(&'a str),
    Lock,
}

/// `markitai_<date>_<time>_<micros>[_<pid>[_<sequence>]].log`, or a session
/// lock `markitai_<date>_<time>_<micros>_<pid>.lock`.
fn owned_name(name: &str) -> Option<Owned<'_>> {
    let rest = name.strip_prefix("markitai_")?;
    let (stem, owned) = match rest.strip_suffix(".log") {
        Some(stem) => (stem, Owned::Log(&name[..name.len() - ".log".len()])),
        None => (rest.strip_suffix(".lock")?, Owned::Lock),
    };
    let parts: Vec<_> = stem.split('_').collect();
    let counts = if matches!(owned, Owned::Lock) {
        4..=4
    } else {
        3..=5
    };
    (counts.contains(&parts.len())
        && parts[0].len() == 8
        && parts[1].len() == 6
        && parts[2].len() == 6
        && parts
            .iter()
            .all(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit())))
    .then_some(owned)
}

/// The session lock of a log named `stem.log`: the log's own stem, or the
/// stem before a rotation sequence.
fn session_lock(directory: &Path, stem: &str) -> Option<PathBuf> {
    let session = stem.rsplit_once('_').map(|(session, _)| session);
    [Some(stem), session.filter(|_| stem.split('_').count() == 6)]
        .into_iter()
        .flatten()
        .map(|stem| directory.join(format!("{stem}.lock")))
        .find(|path| {
            fs::symlink_metadata(path)
                .is_ok_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
        })
}

/// Chinese text runs on without spaces, so its punctuation ends a URL; otherwise
/// the sentence after the URL would be taken for a path and hidden.
const CHINESE_PUNCTUATION: &str = "，。、；：！？（）【】「」『』《》“”";

fn redact(message: &str, secrets: &[String]) -> String {
    let mut safe = String::with_capacity(message.len().min(65536));
    let mut rest = message;
    while let Some(start) = rest
        .find("http://")
        .into_iter()
        .chain(rest.find("https://"))
        .min()
    {
        safe.push_str(&rest[..start]);
        let url = &rest[start..];
        let end = url
            .find(|c: char| {
                c.is_whitespace()
                    || matches!(c, '<' | '>' | '\'' | '"')
                    || CHINESE_PUNCTUATION.contains(c)
            })
            .unwrap_or(url.len());
        // The core's redaction: the path segments output metadata hides,
        // and no query or fragment.
        safe.push_str(&markitai_core::output::log_url(&url[..end]));
        rest = &url[end..];
    }
    safe.push_str(rest);
    // A short configured secret (a one-letter user name) is hidden only where
    // it stands alone, as the core hides it everywhere it shows provider words.
    for secret in secrets {
        safe = markitai_core::output::hide_secret(&safe, secret, "[REDACTED]");
    }
    safe = safe
        .chars()
        .map(|c| {
            if c.is_control() && !matches!(c, '\n' | '\r' | '\t') {
                ' '
            } else {
                c
            }
        })
        .collect();
    if safe.len() > 65536 {
        let mut end = 65536;
        while !safe.is_char_boundary(end) {
            end -= 1;
        }
        safe.truncate(end);
        safe.push_str(" [truncated]");
    }
    safe
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn redaction_removes_url_credentials_and_configured_secrets_before_encoding() {
        let result = redact(
            "Failed https://alice:pass@example.com/path?key=secret#fragment secret-value\u{1b}[31m",
            &["secret-value".into()],
        );
        assert_eq!(result, "Failed https://example.com/path [REDACTED] [31m");
        let cfg = json!({"llm":{"model_list":[{"litellm_params":{"api_key":"key-secret"}}]},"fetch":{"playwright":{"extra_http_headers":{"Authorization":"Bearer header-secret"}}}});
        let mut secrets = vec![];
        collect_secrets(&cfg, &config::redact(&cfg), &mut secrets);
        assert!(secrets.contains(&"key-secret".into()));
        assert!(secrets.contains(&"Bearer header-secret".into()));
    }
    #[test]
    fn a_short_secret_is_hidden_only_where_it_stands_alone() {
        let secrets = ["u".to_owned(), "c".to_owned(), "k3y".to_owned()];
        assert_eq!(
            redact(
                "Saved /tmp/scratch/out.md as user u with cookie c=k3y",
                &secrets
            ),
            "Saved /tmp/scratch/out.md as user [REDACTED] with cookie [REDACTED]=[REDACTED]"
        );
        // A long secret is still hidden inside other text.
        assert_eq!(
            redact("token=abcd1234efgh", &["abcd1234efgh".into()]),
            "token=[REDACTED]"
        );
        assert_eq!(
            redact("prefixabcd1234", &["abcd1234".into()]),
            "prefix[REDACTED]"
        );
    }
    #[test]
    fn secret_path_segments_are_hidden_as_in_output_metadata() {
        assert_eq!(
            redact(
                "Converting http://127.0.0.1:8080/reset/Ab3dEf9GhIjK1mN2oP4qR7sT and https://example.test/token=Zx9Yw8Vu7Ts6Rq5Po4Nm/page?pw=p1",
                &[]
            ),
            "Converting http://127.0.0.1:8080/reset/[REDACTED] and https://example.test/[REDACTED]/page"
        );
        // A short code after a token route, as the log always hid it.
        assert_eq!(
            redact("Fetched https://example.test/verify/123456", &[]),
            "Fetched https://example.test/verify/[REDACTED]"
        );
        // Ordinary segments, percent-escaped words included, stay readable.
        assert_eq!(
            redact(
                "Fetched https://example.test/wiki/%E4%B8%AD%E6%96%87 and https://example.test/author/jane-doe",
                &[]
            ),
            "Fetched https://example.test/wiki/%E4%B8%AD%E6%96%87 and https://example.test/author/jane-doe"
        );
    }
    #[test]
    fn chinese_punctuation_ends_a_url_instead_of_hiding_the_text_after_it() {
        assert_eq!(
            redact(
                "未处理 2 项：https://example.com/a、https://example.com/b。使用相同命令并加上 --resume 可继续。",
                &[]
            ),
            "未处理 2 项：https://example.com/a、https://example.com/b。使用相同命令并加上 --resume 可继续。"
        );
        // The same cuts still hide what a URL carries.
        assert_eq!(
            redact("见（https://u:p@example.com/x?token=1）。", &[]),
            "见（https://example.com/x）。"
        );
        // Without Chinese punctuation nothing changes: a trailing comma stays.
        assert_eq!(
            redact("https://example.com/a, https://example.com/b.", &[]),
            "https://example.com/a, https://example.com/b."
        );
    }
    #[test]
    fn rotation_retains_valid_complete_json_and_retention_ignores_unrelated_files() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("markitai_20260101_120000_123456_999.log");
        let mut sink = Sink {
            file: private_file(&path).unwrap(),
            _session: SessionLock::acquire(
                &temp.path().join("markitai_20260101_120000_123456_999.lock"),
            )
            .unwrap(),
            first_path: path.clone(),
            stem: path.file_stem().unwrap().to_str().unwrap().into(),
            sequence: 0,
            bytes: 0,
            rotation: Rotation::Size(100),
            limit: None,
            json: true,
            minimum: Level::Debug,
            failed: false,
        };
        for i in 0..20 {
            sink.write(Level::Info, &format!("{i}: \"hello\"\n世界"))
                .unwrap();
        }
        let logs = |directory: &Path| {
            fs::read_dir(directory)
                .unwrap()
                .filter(|entry| {
                    entry
                        .as_ref()
                        .unwrap()
                        .path()
                        .extension()
                        .is_some_and(|extension| extension == "log")
                })
                .count()
        };
        let rotated = logs(temp.path());
        assert!(rotated > 1);
        // An abandoned session's lock is nobody's, even though it is a lock.
        let abandoned = temp.path().join("markitai_20250101_120000_000001_7.lock");
        fs::write(&abandoned, b"").unwrap();
        // The live session's logs, rotated ones included, survive even an
        // immediate retention.
        prune(temp.path(), Duration::ZERO).unwrap();
        assert!(path.exists());
        assert_eq!(logs(temp.path()), rotated);
        assert!(!abandoned.exists());
        // Its logs stay readable while it runs: only the sidecar is locked.
        assert!(!fs::read_to_string(&path).unwrap().is_empty());
        drop(sink);
        assert!(
            !temp
                .path()
                .join("markitai_20260101_120000_123456_999.lock")
                .exists()
        );
        let mut count = 0;
        for entry in fs::read_dir(temp.path()).unwrap() {
            let entry = entry.unwrap();
            for line in fs::read_to_string(entry.path()).unwrap().lines() {
                let row: Value = serde_json::from_str(line).unwrap();
                assert_eq!(row["lvl"], "INFO");
                count += 1;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                assert_eq!(
                    entry.metadata().unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
        }
        assert_eq!(count, 20);
        fs::write(temp.path().join("markitai_notes.log"), "keep").unwrap();
        fs::write(temp.path().join("other.log"), "keep").unwrap();
        prune(temp.path(), Duration::ZERO).unwrap();
        assert!(!path.exists());
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 2);
    }
    #[test]
    fn time_rotation_moves_to_the_next_file_once_its_limit_passes() {
        let temp = tempfile::tempdir().unwrap();
        let stem = "markitai_20260101_120000_123456_999";
        let path = temp.path().join(format!("{stem}.log"));
        let opened = Local::now();
        let rotation = log_policy::rotation("1 s").unwrap();
        let mut sink = Sink {
            file: private_file(&path).unwrap(),
            _session: SessionLock::acquire(&temp.path().join(format!("{stem}.lock"))).unwrap(),
            first_path: path.clone(),
            stem: stem.into(),
            sequence: 0,
            bytes: 0,
            rotation,
            limit: rotation.next_after(opened.naive_local()),
            json: false,
            minimum: Level::Debug,
            failed: false,
        };
        let later = |millis| opened + chrono::TimeDelta::milliseconds(millis);
        let file = |sequence: u64| temp.path().join(format!("{stem}_{sequence:06}.log"));
        sink.write_at(Level::Info, "first", opened).unwrap();
        sink.write_at(Level::Info, "still first", later(999))
            .unwrap();
        assert!(!file(1).exists());
        sink.write_at(Level::Info, "second", later(1_000)).unwrap();
        sink.write_at(Level::Info, "still second", later(1_500))
            .unwrap();
        // The second file opened at 1 s, so it lasts until 2 s.
        sink.write_at(Level::Info, "third", later(2_000)).unwrap();
        // Hours later, one record still makes one file.
        sink.write_at(Level::Info, "fourth", later(7_200_000))
            .unwrap();
        drop(sink);
        let read = |path: &Path| fs::read_to_string(path).unwrap();
        let first = read(&path);
        assert!(first.contains("| first\n") && first.contains("| still first\n"));
        assert!(read(&file(1)).contains("| second\n"));
        assert!(read(&file(1)).contains("| still second\n"));
        assert!(read(&file(2)).ends_with("| third\n"));
        assert!(read(&file(3)).ends_with("| fourth\n"));
        assert!(!file(4).exists());
    }

    #[test]
    fn a_time_limit_reached_before_any_record_does_not_leave_an_empty_file() {
        let temp = tempfile::tempdir().unwrap();
        let stem = "markitai_20260101_120000_123456_998";
        let path = temp.path().join(format!("{stem}.log"));
        let opened = Local::now();
        let rotation = log_policy::rotation("daily").unwrap();
        let mut sink = Sink {
            file: private_file(&path).unwrap(),
            _session: SessionLock::acquire(&temp.path().join(format!("{stem}.lock"))).unwrap(),
            first_path: path.clone(),
            stem: stem.into(),
            sequence: 0,
            bytes: 0,
            rotation,
            limit: rotation.next_after(opened.naive_local()),
            json: true,
            minimum: Level::Debug,
            failed: false,
        };
        let tomorrow = opened + chrono::TimeDelta::days(1);
        sink.write_at(Level::Info, "late start", tomorrow).unwrap();
        sink.write_at(Level::Info, "same day", tomorrow).unwrap();
        drop(sink);
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 2);
        assert!(!temp.path().join(format!("{stem}_000001.log")).exists());
    }
}
