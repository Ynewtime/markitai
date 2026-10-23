//! One terminal style for every line the CLI prints, shared with Python
//! markitai 1.3.0 (`markitai.utils.term`) so both look and behave the same.
//!
//! Colour means one thing: bold for structure (titles, section headers), cyan
//! for the accent (the title mark, prompts, an action in progress, commands to
//! copy), green only for success, yellow only for warnings, red only for errors
//! and gray for secondary text. Only the normal eight colours plus gray 90 are
//! used: terminal themes tune those for light and dark backgrounds alike, which
//! they do not do for the bright ones, magenta or blue. Marks carry the meaning
//! and colour only reinforces it, so the marks stay when colour is off.
//!
//! Each stream is judged on its own: colour on a terminal unless `NO_COLOR` or
//! `TERM=dumb` (`FORCE_COLOR` turns it on off a terminal); the ASCII marks where
//! the output cannot draw the Unicode ones, or where its reader would decode
//! them with another code page (a Windows pipe); wrapping with a hanging indent
//! only on a terminal, so piped output keeps one line per message. Glyphs drawn
//! around the marks (the spinner, the status line's ellipsis) follow the same
//! choice.

use std::io::IsTerminal;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stream {
    Stdout,
    Stderr,
}

/// What a line is about. Each mark has one meaning and one colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mark {
    /// `◆` the title, once per command.
    Title,
    /// `✓` a check or step that worked.
    Success,
    /// `!` something optional is missing or degraded; the run continues.
    Warning,
    /// `✗` something failed or blocks the requested work.
    Error,
    /// `→` an action in progress, or the next thing to do.
    Action,
    /// `•` a neutral fact or list item.
    Info,
    /// `○` skipped on purpose.
    Skip,
    /// `?` a prompt waiting for an answer.
    Prompt,
}

impl Mark {
    /// The Unicode mark and its ASCII stand-in.
    fn glyphs(self) -> (&'static str, &'static str) {
        match self {
            Self::Title => ("◆", "*"),
            Self::Success => ("✓", "+"),
            Self::Warning => ("!", "!"),
            Self::Error => ("✗", "x"),
            Self::Action => ("→", ">"),
            Self::Info => ("•", "-"),
            Self::Skip => ("○", "o"),
            Self::Prompt => ("?", "?"),
        }
    }

    fn tone(self) -> Tone {
        match self {
            Self::Title | Self::Action | Self::Prompt => Tone::Accent,
            Self::Success => Tone::Success,
            Self::Warning => Tone::Warning,
            Self::Error => Tone::Error,
            Self::Info | Self::Skip => Tone::Secondary,
        }
    }
}

/// The colour roles of the style guide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tone {
    /// Bold, never coloured: a coloured header reads as a status.
    Structure,
    Accent,
    Success,
    Warning,
    Error,
    /// Details, durations, defaults and skipped steps.
    Secondary,
}

/// How one output stream renders styled lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Style {
    pub(crate) color: bool,
    pub(crate) unicode: bool,
    /// Columns to wrap at; `None` off a terminal, where a message stays one line.
    pub(crate) width: Option<usize>,
    /// The classic blue Windows PowerShell console repaints dark yellow as its
    /// near-white text colour, so there warnings, errors, success and the
    /// accent use the bright console slots instead.
    pub(crate) bright: bool,
}

/// A console line that fills the window by itself makes the Windows console
/// wrap once more, leaving a blank row: wrapped text stops one column short.
pub(crate) const EDGE: usize = if cfg!(windows) { 1 } else { 0 };
/// Wrapped text keeps at least this many columns, however narrow the terminal.
const MIN_ROOM: usize = 20;

impl Style {
    /// What a pipe receives on macOS and Linux: Unicode marks, no colour, one
    /// line per message.
    #[cfg(test)]
    pub(crate) const PLAIN: Self = Self {
        color: false,
        unicode: true,
        width: None,
        bright: false,
    };

    /// The style of standard output, judged once per process.
    pub(crate) fn stdout() -> Self {
        static STYLE: OnceLock<Style> = OnceLock::new();
        *STYLE.get_or_init(|| detect(Stream::Stdout))
    }

    /// The style of standard error, judged once per process.
    pub(crate) fn stderr() -> Self {
        static STYLE: OnceLock<Style> = OnceLock::new();
        *STYLE.get_or_init(|| detect(Stream::Stderr))
    }

    pub(crate) fn of(stream: Stream) -> Self {
        match stream {
            Stream::Stdout => Self::stdout(),
            Stream::Stderr => Self::stderr(),
        }
    }

    /// The mark itself, Unicode or ASCII, without colour.
    pub(crate) fn glyph(self, mark: Mark) -> &'static str {
        let (unicode, ascii) = mark.glyphs();
        if self.unicode { unicode } else { ascii }
    }

    /// `text` in the colour of `tone`; unchanged without colour.
    pub(crate) fn paint(self, tone: Tone, text: &str) -> String {
        if !self.color || text.is_empty() {
            return text.to_owned();
        }
        let code = match (tone, self.bright) {
            (Tone::Structure, _) => "1",
            (Tone::Accent, false) => "36",
            (Tone::Accent, true) => "96",
            (Tone::Success, false) => "32",
            (Tone::Success, true) => "92",
            (Tone::Warning, false) => "33",
            (Tone::Warning, true) => "93",
            (Tone::Error, false) => "31",
            (Tone::Error, true) => "91",
            (Tone::Secondary, _) => "90",
        };
        format!("\x1b[{code}m{text}\x1b[0m")
    }

    /// The mark in its own colour.
    pub(crate) fn mark(self, mark: Mark) -> String {
        self.paint(mark.tone(), self.glyph(mark))
    }

    /// `◆ Title`: accent mark, bold text. Callers follow it with one blank line.
    pub(crate) fn title(self, text: &str) -> String {
        format!(
            "{} {}",
            self.mark(Mark::Title),
            self.paint(Tone::Structure, text)
        )
    }

    /// A section header: bold, column 0, no mark.
    pub(crate) fn section(self, text: &str) -> String {
        self.paint(Tone::Structure, text)
    }

    /// An item line: two columns in, then the mark.
    pub(crate) fn item(self, mark: Mark, text: &str) -> String {
        self.line(2, Some(mark), text, None)
    }

    /// A closing verdict: the mark at column 0. Callers put one blank line
    /// before it when other blocks precede it.
    pub(crate) fn summary(self, mark: Mark, text: &str) -> String {
        self.line(0, Some(mark), text, None)
    }

    /// A detail under an item: gray, two columns further in than its text.
    pub(crate) fn detail(self, text: &str) -> String {
        self.line(4, None, text, Some(Tone::Secondary))
    }

    /// The general form: `<indent><mark> <text>`, the mark optional and the
    /// text in an optional tone. On a terminal a long text wraps at word
    /// boundaries with a hanging indent under its first character, and a word
    /// longer than the line (a path, a URL) stays whole; off a terminal every
    /// line of the text stays one line.
    pub(crate) fn line(
        self,
        indent: usize,
        mark: Option<Mark>,
        text: &str,
        tone: Option<Tone>,
    ) -> String {
        let mut out = " ".repeat(indent);
        let mut hang = indent;
        if let Some(mark) = mark {
            out.push_str(&self.mark(mark));
            out.push(' ');
            hang += width(self.glyph(mark)) + 1;
        }
        let room = self
            .width
            .map(|columns| columns.saturating_sub(hang + EDGE).max(MIN_ROOM));
        let mut first = true;
        for paragraph in text.split('\n') {
            let rows = match room {
                Some(room) => wrap(paragraph, room),
                None => vec![paragraph.trim_end().to_owned()],
            };
            for row in rows {
                if !first {
                    out.push('\n');
                    if !row.is_empty() {
                        out.push_str(&" ".repeat(hang));
                    }
                }
                first = false;
                match tone {
                    Some(tone) => out.push_str(&self.paint(tone, &row)),
                    None => out.push_str(&row),
                }
            }
        }
        out
    }

    /// A command to copy: its own line, accent colour, never wrapped, so a
    /// terminal narrower than the command wraps it visually and copying it
    /// back still gives one line.
    pub(crate) fn command(self, indent: usize, command: &str) -> String {
        format!(
            "{}{}",
            " ".repeat(indent),
            self.paint(Tone::Accent, command)
        )
    }

    /// A prompt label: `? question [default]: ` with the accent mark and the
    /// trailing bracketed default (if any) in gray. Never wrapped; the answer
    /// follows on the same line.
    pub(crate) fn prompt(self, indent: usize, label: &str) -> String {
        let label = label.trim_end();
        let (body, end) = if let Some(body) = label.strip_suffix(':') {
            (body, ": ")
        } else if let Some(body) = label.strip_suffix('：') {
            (body, "：")
        } else {
            (label, " ")
        };
        let body = body.trim_end();
        let shown = match body.rfind(['[', '［']) {
            Some(at) if body.ends_with([']', '］']) => {
                let (question, hint) = body.split_at(at);
                format!("{question}{}", self.paint(Tone::Secondary, hint))
            }
            _ => body.to_owned(),
        };
        format!(
            "{}{} {shown}{end}",
            " ".repeat(indent),
            self.mark(Mark::Prompt)
        )
    }

    /// Clap's help and usage-error colours in this scheme: bold headers and
    /// usage, option and command names in the accent (they are typed) and red
    /// only for the error label. Defaults and possible values stay plain:
    /// clap wraps a styled one with a trailing space before it.
    pub(crate) fn clap(self) -> clap::builder::Styles {
        use clap::builder::styling::{AnsiColor, Effects, Style as Clap, Styles};
        let (accent, error) = if self.bright {
            (AnsiColor::BrightCyan, AnsiColor::BrightRed)
        } else {
            (AnsiColor::Cyan, AnsiColor::Red)
        };
        Styles::styled()
            .header(Effects::BOLD.into())
            .usage(Effects::BOLD.into())
            .literal(accent.on_default())
            .placeholder(Clap::new())
            .error(error.on_default() | Effects::BOLD)
            .valid(accent.on_default())
            .invalid(Effects::BOLD.into())
            .context(Clap::new())
            .context_value(Clap::new())
    }

    /// Clap's colour choice for this stream.
    pub(crate) fn clap_color(self) -> clap::ColorChoice {
        if self.color {
            clap::ColorChoice::Always
        } else {
            clap::ColorChoice::Never
        }
    }
}

/// Columns a character occupies: wide East Asian characters take two.
pub(crate) fn char_width(character: char) -> usize {
    if u32::from(character) >= 0x1100
        && (matches!(u32::from(character), 0x1100..=0x115f | 0x2e80..=0xa4cf | 0xac00..=0xd7a3
            | 0xf900..=0xfaff | 0xfe30..=0xfe6f | 0xff00..=0xff60 | 0xffe0..=0xffe6
            | 0x1f300..=0x1faff | 0x20000..=0x3fffd))
    {
        2
    } else {
        1
    }
}

/// Columns `text` occupies on a terminal.
pub(crate) fn width(text: &str) -> usize {
    text.chars().map(char_width).sum()
}

/// Punctuation that must not start a line.
fn closing(character: char) -> bool {
    "，。、；：！？）」』】》〉…".contains(character)
}

/// Punctuation that must not end a line.
fn opening(character: char) -> bool {
    "（「『【《〈".contains(character)
}

/// `text` cut into the pieces a line may break between, each with the spaces
/// in front of it. Words break only at spaces; a run of wide characters may
/// also break between them, since such text has no spaces to break at.
fn pieces(text: &str) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    let mut spaces = 0;
    let mut piece = String::new();
    let mut previous: Option<char> = None;
    for character in text.chars() {
        if character == ' ' {
            if !piece.is_empty() {
                out.push((spaces, std::mem::take(&mut piece)));
                spaces = 0;
            }
            spaces += 1;
            previous = None;
            continue;
        }
        let breaks = previous.is_some_and(|previous| {
            (char_width(previous) == 2 || char_width(character) == 2)
                && !closing(character)
                && !opening(previous)
        });
        if breaks && !piece.is_empty() {
            out.push((spaces, std::mem::take(&mut piece)));
            spaces = 0;
        }
        piece.push(character);
        previous = Some(character);
    }
    if !piece.is_empty() {
        out.push((spaces, piece));
    }
    out
}

/// Rows of at most `room` columns, broken between pieces; a piece wider than
/// the row stays whole. The spaces at a break are dropped.
fn wrap(text: &str, room: usize) -> Vec<String> {
    let mut rows = Vec::new();
    let mut row = String::new();
    let mut used = 0;
    for (spaces, piece) in pieces(text) {
        let columns = width(&piece);
        if row.is_empty() {
            row = piece;
            used = columns;
        } else if used + spaces + columns > room {
            rows.push(std::mem::replace(&mut row, piece));
            used = columns;
        } else {
            row.push_str(&" ".repeat(spaces));
            row.push_str(&piece);
            used += spaces + columns;
        }
    }
    rows.push(row);
    rows
}

/// Whether a locale name (`zh_CN.UTF-8`, `en_US.utf8`, macOS `UTF-8`) names
/// the UTF-8 character set.
#[cfg(any(unix, test))]
fn names_utf8(locale: &str) -> bool {
    let locale = locale.to_ascii_lowercase();
    locale.contains("utf-8") || locale.contains("utf8")
}

/// A nonempty environment variable.
fn variable(name: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

fn detect(stream: Stream) -> Style {
    let terminal = match stream {
        Stream::Stdout => std::io::stdout().is_terminal(),
        Stream::Stderr => std::io::stderr().is_terminal(),
    };
    let dumb = variable("TERM").is_some_and(|term| term == "dumb");
    let forced = variable("FORCE_COLOR").is_some_and(|value| value != "0" && value != "false");
    let color = variable("NO_COLOR").is_none()
        && !dumb
        && if terminal {
            platform::ansi(stream)
        } else {
            forced
        };
    let width = (terminal && !dumb).then(|| {
        variable("COLUMNS")
            .and_then(|value| value.to_str()?.parse::<usize>().ok())
            .filter(|columns| *columns > 0)
            .unwrap_or_else(|| platform::columns(stream).unwrap_or(80))
    });
    Style {
        color,
        unicode: !dumb && platform::unicode(stream, terminal),
        width,
        bright: color && terminal && platform::classic_powershell(stream),
    }
}

/// What a Windows output stream is, as far as the marks are concerned.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WindowsOutput {
    /// A console; `vt` says whether it reads escape sequences.
    Console { vt: bool },
    /// A terminal that is not a console: mintty or another msys or Cygwin
    /// pseudo-terminal, which decodes the UTF-8 bytes itself.
    Pty,
    /// A pipe or a file, with the output code page of the console the
    /// process is attached to; `None` without a console (a service, a
    /// detached process).
    Redirected { code_page: Option<u32> },
}

/// Whether a Windows stream gets the Unicode marks. A console draws them when
/// it reads escape sequences (Windows 10 version 1511 and later, Windows
/// Terminal, VS Code); one that cannot (older Windows, the console host's "Use
/// legacy console") gets the ASCII set. A pipe or a file follows the console's
/// output code page, because Windows PowerShell decodes a program's output with
/// it (`[Console]::OutputEncoding`): read at code page 437, the UTF-8 bytes of
/// `✓` show as `Γ£ô`. So redirected output carries the Unicode marks only at
/// 65001 (UTF-8). Without a console there is no code page to follow, and the
/// marks stay Unicode, as on macOS and Linux.
#[cfg(any(windows, test))]
fn windows_unicode(output: WindowsOutput) -> bool {
    const UTF8: u32 = 65001;
    match output {
        WindowsOutput::Console { vt } => vt,
        WindowsOutput::Pty => true,
        WindowsOutput::Redirected { code_page } => code_page.is_none_or(|page| page == UTF8),
    }
}

/// Whether the output can draw Chinese: a UTF-8 locale on Unix; on Windows a
/// modern terminal, redirected output, or a console whose code page is CJK or
/// UTF-8 (the legacy console's font follows its code page).
pub(crate) fn cjk_capable() -> bool {
    platform::cjk_capable()
}

/// Whether the terminal behind `stream` reads escape sequences, for cursor
/// control as well as colour (enabled on first use on Windows).
pub(crate) fn vt(stream: Stream) -> bool {
    platform::ansi(stream)
}

/// The visible width of the terminal behind `stream`, 80 when it cannot be read.
pub(crate) fn columns(stream: Stream) -> usize {
    platform::columns(stream).unwrap_or(80)
}

#[cfg(unix)]
mod platform {
    use super::Stream;

    pub(super) fn ansi(_: Stream) -> bool {
        true
    }

    /// The marks need output a terminal decodes as UTF-8. A locale naming
    /// another character set (`en_US.ISO-8859-1`, `zh_CN.GBK`, a bare
    /// `zh_CN`) gets the ASCII set; the C locale or none counts as UTF-8, as
    /// Python's runtime takes it (PEP 538/540), since scripts and containers
    /// set it on UTF-8 terminals.
    pub(super) fn unicode(_: Stream, _terminal: bool) -> bool {
        utf8_output()
    }

    /// Chinese prints as mojibake on the same terminals the marks do.
    pub(super) fn cjk_capable() -> bool {
        utf8_output()
    }

    /// Whether the locale that sets the character set (`LC_ALL`, then
    /// `LC_CTYPE`, then `LANG`) is UTF-8, C/POSIX or unset.
    fn utf8_output() -> bool {
        ["LC_ALL", "LC_CTYPE", "LANG"]
            .into_iter()
            .find_map(super::variable)
            .is_none_or(|locale| {
                let locale = locale.to_string_lossy();
                super::names_utf8(&locale)
                    || ["C", "POSIX"].contains(&locale.as_ref())
                    || locale.starts_with("C.")
                    || locale.starts_with("POSIX.")
            })
    }

    pub(super) fn classic_powershell(_: Stream) -> bool {
        false
    }

    pub(super) fn columns(stream: Stream) -> Option<usize> {
        let fd = match stream {
            Stream::Stdout => libc::STDOUT_FILENO,
            Stream::Stderr => libc::STDERR_FILENO,
        };
        let mut size: libc::winsize = unsafe { std::mem::zeroed() };
        // SAFETY: TIOCGWINSZ writes one winsize into initialized local storage.
        (unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut size) } == 0 && size.ws_col > 0)
            .then_some(usize::from(size.ws_col))
    }
}

#[cfg(windows)]
mod platform {
    use super::Stream;
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Console::{
        CONSOLE_SCREEN_BUFFER_INFO, ENABLE_VIRTUAL_TERMINAL_PROCESSING, GetConsoleMode,
        GetConsoleOutputCP, GetConsoleScreenBufferInfo, GetStdHandle, STD_ERROR_HANDLE,
        STD_OUTPUT_HANDLE, SetConsoleMode,
    };

    fn handle(stream: Stream) -> Option<HANDLE> {
        // SAFETY: GetStdHandle only reads the process's standard handle table.
        let handle = unsafe {
            GetStdHandle(match stream {
                Stream::Stdout => STD_OUTPUT_HANDLE,
                Stream::Stderr => STD_ERROR_HANDLE,
            })
        };
        (!handle.is_null() && handle != INVALID_HANDLE_VALUE).then_some(handle)
    }

    fn enable_vt(stream: Stream) -> bool {
        let Some(handle) = handle(stream) else {
            return false;
        };
        let mut mode = 0;
        // SAFETY: the console APIs inspect a process-owned handle; the pointer
        // refers to initialized local storage and no ownership is transferred.
        unsafe {
            GetConsoleMode(handle, &mut mode) != 0
                && (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING != 0
                    || SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0)
        }
    }

    /// Whether the console behind `stream` reads escape sequences, enabling
    /// them on first use. A console that cannot gets no colour.
    pub(super) fn ansi(stream: Stream) -> bool {
        static STDOUT: OnceLock<bool> = OnceLock::new();
        static STDERR: OnceLock<bool> = OnceLock::new();
        let cell = match stream {
            Stream::Stdout => &STDOUT,
            Stream::Stderr => &STDERR,
        };
        *cell.get_or_init(|| enable_vt(stream))
    }

    /// A terminal other than the legacy console host: Windows Terminal, VS
    /// Code, ConEmu, mintty or an SSH client. They draw Chinese whatever the
    /// code page and keep their own colour schemes.
    fn modern_terminal() -> bool {
        ["WT_SESSION", "TERM_PROGRAM", "SSH_CONNECTION", "SSH_CLIENT"]
            .into_iter()
            .any(|name| super::variable(name).is_some())
            || super::variable("ConEmuANSI").is_some_and(|value| value == "ON")
            || super::variable("TERM").is_some_and(|term| term != "dumb")
    }

    /// Whether `stream` is a Windows console, rather than a pipe, a file or
    /// an msys terminal (a pipe that mintty reads, which the standard
    /// library counts as a terminal).
    fn console(stream: Stream) -> bool {
        let Some(handle) = handle(stream) else {
            return false;
        };
        let mut mode = 0;
        // SAFETY: GetConsoleMode only fills initialized local storage for a
        // process-owned handle.
        unsafe { GetConsoleMode(handle, &mut mode) != 0 }
    }

    /// The marks rule of [`super::windows_unicode`] for `stream`; `terminal`
    /// is the standard library's answer, a console or an msys terminal.
    pub(super) fn unicode(stream: Stream, terminal: bool) -> bool {
        use super::WindowsOutput;
        super::windows_unicode(if console(stream) {
            WindowsOutput::Console { vt: ansi(stream) }
        } else if terminal {
            WindowsOutput::Pty
        } else {
            // SAFETY: GetConsoleOutputCP takes no arguments; 0 means no console.
            let page = unsafe { GetConsoleOutputCP() };
            WindowsOutput::Redirected {
                code_page: (page != 0).then_some(page),
            }
        })
    }

    /// Chinese on a console needs a font that has it: a modern terminal, or
    /// a CJK or UTF-8 code page (the legacy console's font follows it).
    /// Redirected output is UTF-8 and carries it, as it carries the marks.
    pub(super) fn cjk_capable() -> bool {
        use std::io::IsTerminal;
        // SAFETY: GetConsoleOutputCP takes no arguments; 0 means no console.
        let console = modern_terminal()
            || matches!(
                unsafe { GetConsoleOutputCP() },
                932 | 936 | 949 | 950 | 54936 | 65001
            );
        [
            std::io::stdout().is_terminal(),
            std::io::stderr().is_terminal(),
        ]
        .into_iter()
        .all(|terminal| !terminal || console)
    }

    fn buffer(stream: Stream) -> Option<CONSOLE_SCREEN_BUFFER_INFO> {
        let handle = handle(stream)?;
        // SAFETY: the API fills initialized local storage for a process handle.
        let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
        (unsafe { GetConsoleScreenBufferInfo(handle, &mut info) } != 0).then_some(info)
    }

    /// The "Windows PowerShell" shortcut paints a navy DarkMagenta background
    /// and makes DarkYellow its near-white text colour.
    pub(super) fn classic_powershell(stream: Stream) -> bool {
        !modern_terminal()
            && buffer(stream).is_some_and(|info| {
                let attributes = info.wAttributes;
                attributes & 0x0f == 6 && (attributes >> 4) & 0x0f == 5
            })
    }

    pub(super) fn columns(stream: Stream) -> Option<usize> {
        // The viewport, not the scrollback buffer, is the width users see.
        let info = buffer(stream)?;
        super::viewport_width(info.srWindow.Left, info.srWindow.Right)
    }
}

#[cfg(not(any(unix, windows)))]
mod platform {
    use super::Stream;
    pub(super) fn ansi(_: Stream) -> bool {
        true
    }
    pub(super) fn unicode(_: Stream, _terminal: bool) -> bool {
        true
    }
    pub(super) fn cjk_capable() -> bool {
        true
    }
    pub(super) fn classic_powershell(_: Stream) -> bool {
        false
    }
    pub(super) fn columns(_: Stream) -> Option<usize> {
        None
    }
}

#[cfg(any(windows, test))]
fn viewport_width(left: i16, right: i16) -> Option<usize> {
    usize::try_from(i32::from(right) - i32::from(left) + 1)
        .ok()
        .filter(|n| *n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const COLOR: Style = Style {
        color: true,
        unicode: true,
        width: None,
        bright: false,
    };
    const ASCII: Style = Style {
        unicode: false,
        ..Style::PLAIN
    };

    fn at(width: usize) -> Style {
        Style {
            width: Some(width + EDGE),
            ..Style::PLAIN
        }
    }

    #[test]
    fn every_mark_has_one_colour_and_an_ascii_stand_in() {
        let marks = [
            (Mark::Title, "\x1b[36m◆\x1b[0m", "*"),
            (Mark::Success, "\x1b[32m✓\x1b[0m", "+"),
            (Mark::Warning, "\x1b[33m!\x1b[0m", "!"),
            (Mark::Error, "\x1b[31m✗\x1b[0m", "x"),
            (Mark::Action, "\x1b[36m→\x1b[0m", ">"),
            (Mark::Info, "\x1b[90m•\x1b[0m", "-"),
            (Mark::Skip, "\x1b[90m○\x1b[0m", "o"),
            (Mark::Prompt, "\x1b[36m?\x1b[0m", "?"),
        ];
        for (mark, colored, ascii) in marks {
            assert_eq!(COLOR.mark(mark), colored, "{mark:?}");
            assert_eq!(ASCII.mark(mark), ascii, "{mark:?}");
            assert_eq!(Style::PLAIN.mark(mark), mark.glyphs().0, "{mark:?}");
        }
        // The classic PowerShell palette moves only the status colours and the
        // accent to the bright slots.
        let bright = Style {
            bright: true,
            ..COLOR
        };
        assert_eq!(bright.mark(Mark::Warning), "\x1b[93m!\x1b[0m");
        assert_eq!(bright.mark(Mark::Success), "\x1b[92m✓\x1b[0m");
        assert_eq!(bright.mark(Mark::Error), "\x1b[91m✗\x1b[0m");
        assert_eq!(bright.mark(Mark::Title), "\x1b[96m◆\x1b[0m");
        assert_eq!(bright.mark(Mark::Info), "\x1b[90m•\x1b[0m");
        assert_eq!(bright.section("LLM"), "\x1b[1mLLM\x1b[0m");
    }

    #[test]
    fn titles_sections_items_and_commands_render_with_and_without_colour() {
        let render = |style: Style| {
            [
                style.title("System Check"),
                style.section("Optional Capabilities"),
                style.item(Mark::Success, "Serve: ok"),
                style.item(Mark::Info, "Config: defaults"),
                style.summary(
                    Mark::Success,
                    "Ready. Not available (optional): LibreOffice",
                ),
                style.command(4, "markitai doctor --fix"),
                style.detail("log excerpt"),
            ]
            .join("\n")
        };
        assert_eq!(
            render(COLOR),
            "\x1b[36m◆\x1b[0m \x1b[1mSystem Check\x1b[0m\n\
             \x1b[1mOptional Capabilities\x1b[0m\n  \
             \x1b[32m✓\x1b[0m Serve: ok\n  \
             \x1b[90m•\x1b[0m Config: defaults\n\
             \x1b[32m✓\x1b[0m Ready. Not available (optional): LibreOffice\n    \
             \x1b[36mmarkitai doctor --fix\x1b[0m\n    \
             \x1b[90mlog excerpt\x1b[0m"
        );
        assert_eq!(
            render(Style::PLAIN),
            "◆ System Check\nOptional Capabilities\n  ✓ Serve: ok\n  • Config: defaults\n\
             ✓ Ready. Not available (optional): LibreOffice\n    markitai doctor --fix\n    log excerpt"
        );
        assert_eq!(
            render(ASCII),
            "* System Check\nOptional Capabilities\n  + Serve: ok\n  - Config: defaults\n\
             + Ready. Not available (optional): LibreOffice\n    markitai doctor --fix\n    log excerpt"
        );
    }

    #[test]
    fn a_terminal_wraps_under_the_text_and_a_pipe_keeps_one_line() {
        let text = "VLM OCR: --ocr --llm sends page images to 4 vision models (MARKITAI_NO_VLM_OCR=1 keeps local OCR)";
        assert_eq!(
            at(40).item(Mark::Success, text),
            "  ✓ VLM OCR: --ocr --llm sends page\n    images to 4 vision models\n    (MARKITAI_NO_VLM_OCR=1 keeps local\n    OCR)"
        );
        assert_eq!(
            Style::PLAIN.item(Mark::Success, text),
            format!("  ✓ {text}")
        );
        // A summary hangs under its text at column 2; every row fits.
        let summary = at(40).summary(
            Mark::Success,
            "Ready. Not available (optional): Chromium, LibreOffice, LLM API",
        );
        assert_eq!(
            summary,
            "✓ Ready. Not available (optional):\n  Chromium, LibreOffice, LLM API"
        );
        assert!(summary.lines().all(|row| width(row) <= 40), "{summary}");
        // Colour wraps the mark only; the rows stay as wide.
        let colored = Style {
            width: Some(40 + EDGE),
            ..COLOR
        };
        assert_eq!(
            colored
                .item(Mark::Error, text)
                .replace("\x1b[31m✗\x1b[0m", "✓"),
            at(40).item(Mark::Success, text)
        );
    }

    #[test]
    fn long_words_stay_whole_and_spacing_inside_a_row_is_kept() {
        let path = "/home/user/.local/share/markitai/models/ocr/model-0123abcd/model.onnx";
        assert_eq!(
            at(30).item(Mark::Info, &format!("Config: {path}")),
            format!("  • Config:\n    {path}")
        );
        assert_eq!(
            at(60).item(Mark::Info, "Convert a file:   markitai FILE -o DIR"),
            "  • Convert a file:   markitai FILE -o DIR"
        );
        // A text narrower than the minimum still gets 20 columns.
        assert_eq!(
            at(10).item(Mark::Warning, "one two three four five six"),
            "  ! one two three four\n    five six"
        );
        // Commands are never wrapped.
        let command = "markitai doctor --fix --and --a --long --tail --of --flags";
        assert_eq!(at(20).command(4, command), format!("    {command}"));
    }

    #[test]
    fn chinese_wraps_between_characters_but_not_before_closing_punctuation() {
        let text = "已就绪。以下可选功能不可用：浏览器、LibreOffice、LLM API";
        let wrapped = at(24).summary(Mark::Warning, text);
        assert_eq!(
            wrapped,
            "! 已就绪。以下可选功能不\n  可用：浏览器、\n  LibreOffice、LLM API"
        );
        assert!(wrapped.lines().all(|row| width(row) <= 24), "{wrapped}");
        assert!(
            wrapped
                .lines()
                .all(|row| !row.trim_start().starts_with(['，', '。', '、', '：']))
        );
    }

    #[test]
    fn line_breaks_in_the_text_continue_under_it_on_a_terminal_and_in_a_pipe() {
        for style in [Style::PLAIN, at(80)] {
            assert_eq!(
                style.item(Mark::Warning, "first\n\nsecond"),
                "  ! first\n\n    second"
            );
        }
    }

    #[test]
    fn prompts_mark_the_question_and_gray_the_default() {
        assert_eq!(
            COLOR.prompt(2, "Input: 1 file, 2 directory, 3 URL [1]: "),
            "  \x1b[36m?\x1b[0m Input: 1 file, 2 directory, 3 URL \x1b[90m[1]\x1b[0m: "
        );
        assert_eq!(
            Style::PLAIN.prompt(0, "保存到：1 用户配置，2 ./markitai.json，q 取消 [1]："),
            "? 保存到：1 用户配置，2 ./markitai.json，q 取消 [1]："
        );
        assert_eq!(
            Style::PLAIN.prompt(2, "Input path or URL: "),
            "  ? Input path or URL: "
        );
        assert_eq!(
            ASCII.prompt(0, "Clear caches? [y/N]: "),
            "? Clear caches? [y/N]: "
        );
    }

    #[test]
    fn windows_consoles_follow_escape_sequences_and_pipes_the_code_page() {
        use WindowsOutput::{Console, Pty, Redirected};
        // A console: Unicode where it reads escape sequences, ASCII on a
        // legacy one.
        assert!(windows_unicode(Console { vt: true }));
        assert!(!windows_unicode(Console { vt: false }));
        // mintty and other msys terminals decode UTF-8 themselves.
        assert!(windows_unicode(Pty));
        // A pipe or a file: only a UTF-8 console code page reads the marks.
        for page in [437, 850, 936, 1252] {
            assert!(!windows_unicode(Redirected {
                code_page: Some(page)
            }));
        }
        assert!(windows_unicode(Redirected {
            code_page: Some(65001)
        }));
        // No console (a service, a detached process): no code page to follow.
        assert!(windows_unicode(Redirected { code_page: None }));
    }

    #[test]
    fn only_utf8_locales_count_as_utf8() {
        for locale in ["zh_CN.UTF-8", "en_US.utf8", "UTF-8", "C.UTF-8"] {
            assert!(names_utf8(locale), "{locale}");
        }
        for locale in ["C", "POSIX", "zh_CN", "zh_CN.GBK", "en_US.ISO-8859-1"] {
            assert!(!names_utf8(locale), "{locale}");
        }
    }

    #[test]
    fn viewport_width_counts_both_edges() {
        assert_eq!(viewport_width(30, 109), Some(80));
        assert_eq!(viewport_width(10, 9), None);
        assert_eq!(viewport_width(10, 8), None);
    }
}
