//! Where the library's own stderr lines go: the notes it prints while it
//! works (a local OCR model download, a configuration it ignored), as opposed
//! to the warnings it returns with a result.
//!
//! A program with a terminal style of its own (the CLI: marks, the terminal
//! language, its file log) installs a sink once; without one the lines go to
//! standard error as they are.
use std::sync::OnceLock;

static SINK: OnceLock<fn(&str)> = OnceLock::new();

/// Send every later line to `sink`. Only the first call has an effect.
pub fn set_sink(sink: fn(&str)) {
    let _ = SINK.set(sink);
}

/// One line, in English; a `Warning: ` label marks a warning.
pub(crate) fn line(text: &str) {
    match SINK.get() {
        Some(sink) => sink(text),
        None => eprintln!("{text}"),
    }
}
