//! `MARKITAI_OCR_TRACE`: one line on stderr per scanned page and stage, for
//! finding where local OCR spends its time on a given machine. Off unless
//! the variable is a nonempty value other than `0`, `false` or `no`.
//!
//! A page's stages run alongside other pages' (pages in flight share the OCR
//! threads), so for clean per-stage numbers run with one page in flight:
//! `MARKITAI_OCR_THREADS=1`, or a short page range.

use std::cell::Cell;
use std::fmt::Arguments;
use std::time::Instant;

/// The environment variable that turns the trace on.
const VARIABLE: &str = "MARKITAI_OCR_TRACE";

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var(VARIABLE).is_ok_and(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "no")
        })
    })
}

thread_local! {
    static PAGE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Name the page this thread works on in its trace lines.
pub(crate) fn set_page(number: usize) {
    PAGE.with(|page| page.set(Some(number)));
}

/// Write one trace line (when tracing), tagged with the thread's page.
pub(crate) fn line(text: Arguments<'_>) {
    if enabled() {
        let page = PAGE
            .with(Cell::get)
            .map_or(String::new(), |n| format!(" page={n}"));
        eprintln!("ocr-trace{page} {text}");
    }
}

/// Milliseconds between calls to [`Lap::ms`]; inert (zero) when not tracing.
pub(crate) struct Lap(Option<Instant>);

impl Lap {
    pub(crate) fn start() -> Lap {
        Lap(enabled().then(Instant::now))
    }

    /// Milliseconds since the start or the previous call.
    pub(crate) fn ms(&mut self) -> f64 {
        let Some(last) = self.0.as_mut() else {
            return 0.0;
        };
        let now = Instant::now();
        let ms = now.duration_since(*last).as_secs_f64() * 1e3;
        *last = now;
        ms
    }
}

// Only the portable engine (tract) reports operators.
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
/// Time spent in each operator of each model, summed over the run so far.
static OPERATORS: std::sync::Mutex<Vec<(String, std::time::Duration, u64)>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
/// Add one evaluation of operator `op` of `model` to the totals.
pub(crate) fn operator(model: &str, op: &str, took: std::time::Duration) {
    let key = format!("{model}/{op}");
    let mut totals = OPERATORS.lock().unwrap_or_else(|e| e.into_inner());
    match totals.iter_mut().find(|(name, ..)| *name == key) {
        Some((_, total, count)) => {
            *total += took;
            *count += 1;
        }
        None => totals.push((key, took, 1)),
    }
}

#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
/// The operators that took longest so far (summed over threads, so with
/// several running at once the sum can exceed the wall time).
pub(crate) fn report(top: usize) {
    if !enabled() {
        return;
    }
    let mut totals = OPERATORS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    totals.sort_by_key(|entry| std::cmp::Reverse(entry.1));
    let text: Vec<String> = totals
        .iter()
        .take(top)
        .map(|(name, total, count)| format!("{name}={:.0}ms/{count}", total.as_secs_f64() * 1e3))
        .collect();
    line(format_args!("operators so far: {}", text.join(" ")));
}
