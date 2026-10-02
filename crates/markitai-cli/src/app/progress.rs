//! One overwritten status line for work that takes a while, on a terminal.
//!
//! It exists only while stderr is a terminal and the run is neither `--quiet`
//! nor `--json`, so a pipe, a log file or a script sees exactly the lines it
//! saw before. Every other stderr line goes through `logging`, which calls
//! [`clear`] first: warnings and the closing summary always start on a clean
//! line instead of being glued to the status.

use super::i18n::{self, Lang};
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::Ordering;
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

/// Erase the status line, if there is one, and leave the cursor at its start.
pub(super) fn clear() {
    if SHOWN.swap(false, Ordering::SeqCst) {
        let mut stderr = io::stderr().lock();
        let _ = stderr.write_all(b"\r\x1b[K");
        let _ = stderr.flush();
    }
}

/// Whether the status line is wanted: an interactive stderr that understands
/// cursor control (Unix terminals; Windows consoles are not asked to interpret
/// escape sequences), and a run that has not asked for silence or JSON.
pub(super) fn wanted(quiet: bool, json: bool) -> bool {
    cfg!(unix)
        && !quiet
        && !json
        && io::stderr().is_terminal()
        && std::env::var_os("TERM").is_none_or(|term| term != "dumb")
}

/// Terminal width in columns, 80 when it cannot be read.
fn columns() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: TIOCGWINSZ fills the `winsize` the pointer refers to.
        let mut size: libc::winsize = unsafe { std::mem::zeroed() };
        if unsafe { libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size) } == 0
            && size.ws_col > 0
        {
            return usize::from(size.ws_col);
        }
    }
    80
}

/// Columns a character occupies: wide East Asian characters take two.
fn width(character: char) -> usize {
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

fn draw(line: &str) {
    let mut stderr = io::stderr().lock();
    let _ = write!(stderr, "\r{line}\x1b[K");
    let _ = stderr.flush();
    SHOWN.store(true, Ordering::SeqCst);
}

/// `[12/340] name  ETA 0:41`, cut to the terminal width. `eta` is the time
/// still to go, when there is an estimate.
pub(super) fn batch_line(
    done: usize,
    total: usize,
    name: &str,
    eta: Option<Duration>,
    lang: Lang,
    columns: usize,
) -> String {
    let counter = format!("[{done}/{total}]");
    let estimate = eta.map(|eta| {
        let eta = clock(eta.as_secs_f64().round() as u64);
        match lang {
            Lang::En => format!("ETA {eta}"),
            Lang::Zh => format!("预计剩余 {eta}"),
        }
    });
    let tail = estimate.map(|text| format!("  {text}")).unwrap_or_default();
    let tail_width: usize = tail.chars().map(width).sum();
    let room = columns
        .saturating_sub(1)
        .saturating_sub(counter.len() + 1 + tail_width);
    if room < 8 {
        // Too narrow for a name: the counter alone still says how far along.
        return fit(&format!("{counter}{tail}"), columns.saturating_sub(1));
    }
    format!("{counter} {}{tail}", fit(name, room))
}

/// The status line of a batch. Dropping it erases the line.
pub(super) struct Progress {
    enabled: bool,
    total: usize,
    started: Instant,
    drawn: Option<Instant>,
    line: String,
    /// The estimate made when the latest item finished, and when.
    anchor: Option<(Duration, Instant)>,
    finished: usize,
}

impl Progress {
    pub(super) fn new(total: usize, enabled: bool) -> Self {
        Self {
            enabled: enabled && total > 0,
            total,
            started: Instant::now(),
            drawn: None,
            line: String::new(),
            anchor: None,
            finished: 0,
        }
    }

    /// The time still to go. Each finished item re-estimates it from the pace
    /// so far; between completions it counts down, so a slow item does not
    /// make the estimate climb. An estimate needs a finished item and a little
    /// history to be of any use.
    fn eta(&mut self, done: usize) -> Option<Duration> {
        if done != self.finished {
            self.finished = done;
            let elapsed = self.started.elapsed();
            self.anchor = (done > 0 && done < self.total && elapsed >= Duration::from_secs(1))
                .then(|| {
                    let remaining =
                        elapsed.as_secs_f64() / done as f64 * (self.total - done) as f64;
                    (Duration::from_secs_f64(remaining), Instant::now())
                });
        }
        self.anchor
            .map(|(estimate, at)| estimate.saturating_sub(at.elapsed()))
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
        let eta = self.eta(done);
        let line = batch_line(done, self.total, current, eta, i18n::lang(), columns());
        // The estimate changes about once a second; an identical line is not
        // worth terminal traffic (unless something erased it since).
        if line != self.line || !SHOWN.load(Ordering::SeqCst) {
            draw(&line);
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

/// A line for one long conversion: `⠋ Converting name … 7s`. It appears only
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
            let mut frame = 0;
            loop {
                let seconds = started.elapsed().as_secs();
                let label = match i18n::lang() {
                    Lang::En => "Converting",
                    Lang::Zh => "正在转换",
                };
                let room = columns().saturating_sub(1);
                let tail = format!(" {seconds}s");
                let head = format!("{} {label} ", FRAMES[frame % FRAMES.len()]);
                let name = fit(
                    &name,
                    room.saturating_sub(head.chars().map(width).sum::<usize>() + tail.len()),
                );
                draw(&format!("{head}{name}{tail}"));
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
    fn the_line_names_progress_and_an_estimate_once_there_is_history() {
        let line = |done, total, eta: Option<u64>| {
            batch_line(
                done,
                total,
                "docs/report.pdf",
                eta.map(Duration::from_secs),
                Lang::En,
                80,
            )
        };
        assert_eq!(line(0, 340, None), "[0/340] docs/report.pdf");
        assert_eq!(
            line(12, 340, Some(164)),
            "[12/340] docs/report.pdf  ETA 2:44"
        );
        assert_eq!(
            batch_line(1, 4, "a.docx", Some(Duration::from_secs(9)), Lang::Zh, 80),
            "[1/4] a.docx  预计剩余 0:09"
        );
        assert_eq!(clock(3725), "1:02:05");
    }

    #[test]
    fn the_estimate_is_made_when_an_item_finishes_and_counts_down_after_that() {
        let mut progress = Progress::new(10, true);
        // No finished item, or no history yet: no estimate.
        assert_eq!(progress.eta(0), None);
        assert_eq!(progress.eta(3), None);
        // Pretend the run began 6 s ago. 4 items in 6 s leave 6 at 1.5 s each.
        progress.started = Instant::now() - Duration::from_secs(6);
        let first = progress.eta(4).unwrap();
        assert!(
            (Duration::from_millis(8800)..=Duration::from_millis(9200)).contains(&first),
            "{first:?}"
        );
        // The same count again does not re-estimate; time passing only lowers it.
        std::thread::sleep(Duration::from_millis(30));
        let later = progress.eta(4).unwrap();
        assert!(later < first, "{later:?} {first:?}");
        // Nothing left to estimate at the end.
        assert_eq!(progress.eta(10), None);
    }

    #[test]
    fn a_long_name_keeps_its_end_and_the_line_never_fills_the_terminal() {
        let line = batch_line(
            3,
            9,
            "a/very/long/directory/structure/with/many/parts/final-report-2026.pdf",
            Some(Duration::from_secs(2)),
            Lang::En,
            40,
        );
        assert!(line.starts_with("[3/9] …"), "{line}");
        assert!(line.contains("final-report-2026.pdf"), "{line}");
        assert!(line.chars().map(width).sum::<usize>() < 40, "{line}");
        // Wide characters count double.
        let wide = batch_line(1, 2, &"报告".repeat(30), None, Lang::En, 40);
        assert!(wide.chars().map(width).sum::<usize>() < 40, "{wide}");
        // A terminal too narrow for a name still gets the counter.
        assert_eq!(batch_line(1, 2, "name.pdf", None, Lang::En, 10), "[1/2]");
    }
}
