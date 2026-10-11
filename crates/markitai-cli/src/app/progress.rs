//! One overwritten status line for work that takes a while, on a terminal.
//!
//! It exists only while stderr is a terminal and the run is neither `--quiet`
//! nor `--json`, so a pipe, a log file or a script sees exactly the lines it
//! saw before. Every other stderr line goes through `logging`, which calls
//! [`clear`] first: warnings and the closing summary always start on a clean
//! line instead of being glued to the status.

use super::i18n::{self, Lang};
use crate::term::{Stream, Style, Tone, char_width as width};
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Whether a status line is currently on the screen (shared with the signal
/// handlers, which erase it before ending the process).
use crate::signals::STATUS_LINE as SHOWN;

/// Nothing is drawn before this much time has passed, so a run that finishes
/// at once leaves no flicker behind it.
const QUIET_PERIOD: Duration = Duration::from_millis(500);
/// Redrawing faster than this only costs terminal traffic.
const REDRAW: Duration = Duration::from_millis(100);
/// A single conversion shows its spinner only after it has taken this long.
const SPINNER_AFTER: Duration = Duration::from_secs(2);

/// Columns occupied by the previous line; consoles without VT erase them
/// with spaces instead of printing escape sequences as ordinary text.
static LAST_WIDTH: AtomicUsize = AtomicUsize::new(0);

/// Whether stderr reads escape sequences (enabled on first use on Windows).
fn ansi() -> bool {
    crate::term::vt(Stream::Stderr)
}

/// Erase the line without touching the shown flag. The Windows termination
/// handler shares this operation after it takes the flag itself.
pub(super) fn erase() {
    let previous = LAST_WIDTH.swap(0, Ordering::SeqCst);
    let text = erase_text(ansi(), previous.min(columns().saturating_sub(1)));
    let mut stderr = io::stderr().lock();
    let _ = stderr.write_all(text.as_bytes());
    let _ = stderr.flush();
}

fn erase_text(ansi: bool, previous: usize) -> String {
    if ansi {
        "\r\x1b[K".into()
    } else {
        format!("\r{}\r", " ".repeat(previous))
    }
}

/// Erase the status line, if there is one, and leave the cursor at its start.
pub(super) fn clear() {
    if SHOWN.swap(false, Ordering::SeqCst) {
        erase();
    }
}

/// An interactive stderr, unless silence, JSON or a dumb terminal was asked
/// for. Windows consoles without VT still get a space-cleared status line.
pub(super) fn wanted(quiet: bool, json: bool) -> bool {
    !quiet
        && !json
        && io::stderr().is_terminal()
        && std::env::var_os("TERM").is_none_or(|term| term != "dumb")
}

/// Terminal's visible width in columns, 80 when it cannot be read.
pub(super) fn columns() -> usize {
    crate::term::columns(Stream::Stderr)
}

/// `text` cut to `limit` columns, keeping its end (the file name) and marking
/// the cut with an ellipsis at the start.
fn fit(text: &str, limit: usize) -> String {
    let total: usize = text.chars().map(width).sum();
    if total <= limit {
        return text.to_owned();
    }
    let mut kept = Vec::new();
    let mut used = 1; // the ellipsis
    for character in text.chars().rev() {
        let columns = width(character);
        if used + columns > limit {
            break;
        }
        used += columns;
        kept.push(character);
    }
    std::iter::once('…').chain(kept.into_iter().rev()).collect()
}

fn clock(seconds: u64) -> String {
    if seconds >= 3600 {
        format!(
            "{}:{:02}:{:02}",
            seconds / 3600,
            seconds % 3600 / 60,
            seconds % 60
        )
    } else {
        format!("{}:{:02}", seconds / 60, seconds % 60)
    }
}

/// `styled` drawn over the previous line; `line` is the same text without
/// colour, which decides how many columns it covers.
fn redraw_text(line: &str, styled: &str, ansi: bool, previous: usize) -> String {
    if ansi {
        format!("\r{styled}\x1b[K")
    } else {
        let current: usize = line.chars().map(width).sum();
        let padding = " ".repeat(previous.saturating_sub(current));
        if padding.is_empty() {
            format!("\r{styled}")
        } else {
            format!("\r{styled}{padding}\r{styled}")
        }
    }
}

fn draw(line: &str, styled: &str) {
    let current: usize = line.chars().map(width).sum();
    let previous = LAST_WIDTH
        .swap(current, Ordering::SeqCst)
        .min(columns().saturating_sub(1));
    let text = redraw_text(line, styled, ansi(), previous);
    let mut stderr = io::stderr().lock();
    let _ = stderr.write_all(text.as_bytes());
    let _ = stderr.flush();
    SHOWN.store(true, Ordering::SeqCst);
}

/// `[12/340] name  elapsed 0:41`, cut to the terminal width. `elapsed` is
/// the time the batch has taken so far: a remaining-time estimate swings with
/// each item's size and with model requests, so it is not shown.
#[cfg(test)]
fn batch_line(
    done: usize,
    total: usize,
    name: &str,
    elapsed: Duration,
    lang: Lang,
    columns: usize,
) -> String {
    let (head, tail) = batch_parts(done, total, name, elapsed, lang, columns);
    head + &tail
}

/// [`batch_line`] as the counter and name, and the clock, which is shown in gray.
fn batch_parts(
    done: usize,
    total: usize,
    name: &str,
    elapsed: Duration,
    lang: Lang,
    columns: usize,
) -> (String, String) {
    let counter = format!("[{done}/{total}]");
    let elapsed = clock(elapsed.as_secs());
    let tail = match lang {
        Lang::En => format!("  elapsed {elapsed}"),
        Lang::Zh => format!("  已用时 {elapsed}"),
    };
    let tail_width: usize = tail.chars().map(width).sum();
    let room = columns
        .saturating_sub(1)
        .saturating_sub(counter.len() + 1 + tail_width);
    if room < 8 {
        // Too narrow for a name: the counter (with the clock when it fits)
        // still says how far along.
        let limit = columns.saturating_sub(1);
        if counter.len() + tail_width <= limit {
            return (counter, tail);
        }
        return (fit(&counter, limit), String::new());
    }
    (format!("{counter} {}", fit(name, room)), tail)
}

/// The status line of a batch. Dropping it erases the line.
pub(super) struct Progress {
    enabled: bool,
    total: usize,
    started: Instant,
    drawn: Option<Instant>,
    line: String,
}

impl Progress {
    pub(super) fn new(total: usize, enabled: bool) -> Self {
        Self {
            enabled: enabled && total > 0,
            total,
            started: Instant::now(),
            drawn: None,
            line: String::new(),
        }
    }

    /// Show `done` items finished and `current` as the latest one started.
    pub(super) fn update(&mut self, done: usize, current: &str) {
        if !self.enabled || self.started.elapsed() < QUIET_PERIOD {
            return;
        }
        if self.drawn.is_some_and(|at| at.elapsed() < REDRAW) {
            return;
        }
        self.drawn = Some(Instant::now());
        let done = done.min(self.total);
        let elapsed = self.started.elapsed();
        let (head, tail) = batch_parts(done, self.total, current, elapsed, i18n::lang(), columns());
        let line = format!("{head}{tail}");
        // The clock changes once a second; an identical line is not worth
        // terminal traffic (unless something erased it since).
        if line != self.line || !SHOWN.load(Ordering::SeqCst) {
            let styled = format!("{head}{}", Style::stderr().paint(Tone::Secondary, &tail));
            draw(&line, &styled);
            self.line = line;
        }
    }

    /// Erase the line now, before the caller prints its own lines.
    pub(super) fn finish(&mut self) {
        self.enabled = false;
        clear();
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        clear();
    }
}

/// A line for one long conversion: `⠋ Converting name … 7s`, the spinner in
/// the accent colour (`| / - \\` where the output takes no Unicode marks) and
/// the seconds in gray. It appears only
/// after the conversion has taken [`SPINNER_AFTER`], runs on its own thread
/// and erases itself when dropped.
pub(super) struct Spinner {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Spinner {
    pub(super) fn start(enabled: bool, name: &str) -> Self {
        if !enabled {
            return Self {
                stop: None,
                thread: None,
            };
        }
        let (stop, stopped) = mpsc::channel::<()>();
        let name = name.to_owned();
        let thread = std::thread::spawn(move || {
            let started = Instant::now();
            if stopped.recv_timeout(SPINNER_AFTER) != Err(mpsc::RecvTimeoutError::Timeout) {
                return;
            }
            const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            const ASCII_FRAMES: [char; 4] = ['|', '/', '-', '\\'];
            let style = Style::stderr();
            let frames: &[char] = if style.unicode {
                &FRAMES
            } else {
                &ASCII_FRAMES
            };
            let mut frame = 0;
            loop {
                let seconds = started.elapsed().as_secs();
                let label = match i18n::lang() {
                    Lang::En => "Converting",
                    Lang::Zh => "正在转换",
                };
                let room = columns().saturating_sub(1);
                let tail = format!(" {seconds}s");
                let spinner = frames[frame % frames.len()].to_string();
                let head = format!("{spinner} {label} ");
                let name = fit(
                    &name,
                    room.saturating_sub(head.chars().map(width).sum::<usize>() + tail.len()),
                );
                draw(
                    &format!("{head}{name}{tail}"),
                    &format!(
                        "{} {label} {name}{}",
                        style.paint(Tone::Accent, &spinner),
                        style.paint(Tone::Secondary, &tail)
                    ),
                );
                frame += 1;
                if stopped.recv_timeout(Duration::from_millis(120))
                    != Err(mpsc::RecvTimeoutError::Timeout)
                {
                    return;
                }
            }
        });
        Self {
            stop: Some(stop),
            thread: Some(thread),
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
            clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consoles_without_vt_clear_previous_columns_without_escape_sequences() {
        assert_eq!(redraw_text("报告", "报告", true, 9), "\r报告\x1b[K");
        assert_eq!(redraw_text("报告", "报告", false, 9), "\r报告     \r报告");
        assert_eq!(redraw_text("longer", "longer", false, 3), "\rlonger");
        assert_eq!(erase_text(false, 4), "\r    \r");
        // Colour does not count as columns to clear.
        assert_eq!(
            redraw_text("a 1s", "a \x1b[90m1s\x1b[0m", false, 6),
            "\ra \x1b[90m1s\x1b[0m  \ra \x1b[90m1s\x1b[0m"
        );
    }

    #[test]
    fn the_line_names_progress_and_the_time_taken_so_far() {
        let line = |done, total, seconds| {
            batch_line(
                done,
                total,
                "docs/report.pdf",
                Duration::from_secs(seconds),
                Lang::En,
                80,
            )
        };
        assert_eq!(line(0, 340, 0), "[0/340] docs/report.pdf  elapsed 0:00");
        assert_eq!(line(12, 340, 164), "[12/340] docs/report.pdf  elapsed 2:44");
        // Partial seconds do not round up ahead of the clock.
        assert_eq!(
            batch_line(1, 4, "a.docx", Duration::from_millis(9900), Lang::Zh, 80),
            "[1/4] a.docx  已用时 0:09"
        );
        assert_eq!(clock(3725), "1:02:05");
    }

    #[test]
    fn a_long_name_keeps_its_end_and_the_line_never_fills_the_terminal() {
        let line = batch_line(
            3,
            9,
            "a/very/long/directory/structure/with/many/parts/final-report-2026.pdf",
            Duration::from_secs(2),
            Lang::En,
            48,
        );
        assert!(line.starts_with("[3/9] …"), "{line}");
        assert!(line.contains("final-report-2026.pdf"), "{line}");
        assert!(line.chars().map(width).sum::<usize>() < 48, "{line}");
        // Wide characters count double.
        let wide = batch_line(1, 2, &"报告".repeat(30), Duration::ZERO, Lang::En, 40);
        assert!(wide.chars().map(width).sum::<usize>() < 40, "{wide}");
        // A terminal too narrow for a name still gets the counter.
        assert_eq!(
            batch_line(1, 2, "name.pdf", Duration::ZERO, Lang::En, 10),
            "[1/2]"
        );
        assert_eq!(
            batch_line(1, 2, "name.pdf", Duration::ZERO, Lang::En, 24),
            "[1/2]  elapsed 0:00"
        );
    }
}
