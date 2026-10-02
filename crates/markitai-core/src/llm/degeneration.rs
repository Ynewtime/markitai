//! A model answer that ends by repeating one passage over and over has
//! stopped transcribing: the copies after the first carry nothing, and the
//! loop usually took the place of the rest of the page. This module finds
//! such a tail, keeps one copy and reports how much it removed, so the caller
//! can warn and keep the salvaged answer out of the persistent cache (the
//! next run asks again instead of replaying the damage).
//!
//! Two linear scans look at the end of the answer only:
//!
//! * a run of identical lines (blank lines between them are skipped), and
//! * a periodic tail: one passage of at least [`MIN_PERIOD_BYTES`] repeated
//!   back to back, found with the prefix function of the reversed tail.
//!
//! Repetition that documents really contain is left alone in three ways.
//! Rows that differ (tables, lists, logs with timestamps) are not
//! repetition. A passage the source text holds as often is the source's
//! own. And a passage made mostly of punctuation or table cells (blank form
//! rows, rules, fill-in lines, closing braces) must repeat far more often
//! than a page holds before it counts as a loop.

/// Copies, the kept one included, before textual repetition counts.
const MIN_COPIES: usize = 6;
/// Copies before a structural unit (table row, rule, fill-in line) counts.
const MIN_STRUCTURAL_COPIES: usize = 64;
/// Visible weight the copies must add up to (a character outside ASCII,
/// such as a Han character, weighs two).
const MIN_REPEATED_WEIGHT: usize = 120;
const MIN_STRUCTURAL_REPEATED_WEIGHT: usize = 1024;
/// Shorter periods are dot leaders, digit runs or rules, not loops.
const MIN_PERIOD_BYTES: usize = 16;
/// The periodic scan reads at most this much of the end of the answer; the
/// run it finds is then followed backwards without a limit.
const SCAN_BYTES: usize = 256 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Salvage {
    /// The answer with the repeated tail removed.
    pub text: String,
    /// Copies the answer had, and copies kept.
    pub copies: usize,
    pub kept: usize,
    /// Unicode scalar values removed.
    pub removed: usize,
}

impl Salvage {
    pub fn warning(&self) -> String {
        let kept = if self.kept == 1 {
            "one copy was kept".to_owned()
        } else {
            format!("the {} copies the source has were kept", self.kept)
        };
        format!(
            "The model's answer ended by repeating one passage {} times: {} repeated characters were removed and {kept}. The answer is not cached, so a later run asks again.",
            self.copies, self.removed
        )
    }
}

/// Passes over one answer: the copy a pass keeps can end in a loop of its
/// own (a repeated line that itself repeats one sentence).
const PASSES: usize = 3;

/// The answer without its degenerate tail, or `None` when it has none.
/// `source` is the text the model was given (empty for pictures alone).
pub(super) fn salvage(answer: &str, source: &str) -> Option<Salvage> {
    let original = answer.trim_end();
    let mut text = original;
    let mut first = None;
    for _ in 0..PASSES {
        let Some(found) = repeated_lines(text).or_else(|| periodic_tail(text)) else {
            break;
        };
        let in_source = found.occurrences_in(source);
        if in_source >= found.copies {
            break;
        }
        let kept = in_source.max(1);
        first.get_or_insert((found.copies, kept));
        text = text[..found.ends[kept - 1]].trim_end();
    }
    let (copies, kept) = first?;
    Some(Salvage {
        removed: original[text.len()..].chars().count(),
        text: text.to_owned(),
        copies,
        kept,
    })
}

/// A repeated tail: `copies` copies of `unit`, the last one possibly cut
/// short, running to the end of the trimmed answer.
struct Repeat<'a> {
    unit: &'a str,
    copies: usize,
    /// The byte offset after each full copy, first copy first.
    ends: Vec<usize>,
    lines: bool,
}

impl Repeat<'_> {
    /// How often the source holds the unit, comparing words without regard
    /// to how whitespace was laid out.
    fn occurrences_in(&self, source: &str) -> usize {
        let unit = words(self.unit);
        if unit.is_empty() || source.is_empty() {
            return 0;
        }
        if self.lines {
            source.lines().filter(|line| words(line) == unit).count()
        } else {
            words(source).matches(unit.as_str()).count()
        }
    }
}

/// The visible words of a passage, joined by single spaces.
fn words(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether enough repeated material is there to call it a loop.
fn qualifies(unit: &str, copies: usize) -> bool {
    let (mut visible, mut letters, mut weight) = (0usize, 0usize, 0usize);
    for c in unit.chars().filter(|c| !c.is_whitespace()) {
        visible += 1;
        letters += usize::from(c.is_alphanumeric());
        weight += if c.is_ascii() { 1 } else { 2 };
    }
    if visible == 0 {
        return false;
    }
    let table = unit
        .lines()
        .filter(|line| !line.trim().is_empty())
        .all(|line| line.trim_start().starts_with('|'));
    let (min_copies, min_weight) = if table || letters * 2 < visible {
        (MIN_STRUCTURAL_COPIES, MIN_STRUCTURAL_REPEATED_WEIGHT)
    } else {
        (MIN_COPIES, MIN_REPEATED_WEIGHT)
    };
    copies >= min_copies && copies.saturating_mul(weight) >= min_weight
}

/// The same line repeated at the end; blank lines between copies are
/// skipped. Leading whitespace counts, so differently indented code differs.
fn repeated_lines(text: &str) -> Option<Repeat<'_>> {
    let mut unit: Option<&str> = None;
    // Line ends of the run, last copy first.
    let mut ends = Vec::new();
    let mut end = text.len();
    loop {
        let start = text[..end].rfind('\n').map_or(0, |at| at + 1);
        let line = text[start..end].trim_end();
        if !line.trim_start().is_empty() {
            match unit {
                None => unit = Some(line),
                Some(unit) if unit == line => (),
                Some(_) => break,
            }
            ends.push(start + line.len());
        }
        if start == 0 {
            break;
        }
        end = start - 1;
    }
    let unit = unit?;
    ends.reverse();
    let copies = ends.len();
    qualifies(unit, copies).then_some(Repeat {
        unit,
        copies,
        ends,
        lines: true,
    })
}

/// The longest periodic suffix with a period of at least
/// [`MIN_PERIOD_BYTES`] that holds at least [`MIN_COPIES`] periods.
fn periodic_tail(text: &str) -> Option<Repeat<'_>> {
    let bytes = text.as_bytes();
    let mut window = text.len().saturating_sub(SCAN_BYTES);
    while !text.is_char_boundary(window) {
        window += 1;
    }
    let tail = &bytes[window..];
    let n = tail.len();
    let shortest = MIN_PERIOD_BYTES * MIN_COPIES;
    if n < shortest {
        return None;
    }
    // Prefix function of the reversed tail: border[i] is the longest proper
    // border of the last i + 1 bytes, whose smallest period is therefore
    // i + 1 - border[i].
    let at = |i: usize| tail[n - 1 - i];
    let mut border = vec![0u32; n];
    let mut k = 0usize;
    for i in 1..n {
        while k > 0 && at(i) != at(k) {
            k = border[k - 1] as usize;
        }
        if at(i) == at(k) {
            k += 1;
        }
        border[i] = k as u32;
    }
    let period = (shortest..=n).rev().find_map(|length| {
        let period = length - border[length - 1] as usize;
        (period >= MIN_PERIOD_BYTES && length >= MIN_COPIES * period).then_some((length, period))
    });
    let (length, period) = period?;
    // Follow the run backwards past the scanned window.
    let mut start = text.len() - length;
    while start > 0 && bytes[start - 1] == bytes[start - 1 + period] {
        start -= 1;
    }
    // The run may begin inside a sentence ("…text. Sentence. Sentence."
    // repeats from the full stop on); start it where a copy reads as a whole:
    // after a line break, or for a passage within one line, after a space or
    // a sentence end.
    let multiline = bytes[start..start + period].contains(&b'\n');
    let natural = |at: usize| {
        at == 0
            || match bytes[at - 1] {
                b'\n' => true,
                _ if multiline => false,
                byte => {
                    byte.is_ascii_whitespace()
                        || text[..at].ends_with(['。', '！', '？', '.', '!', '?'])
                }
            }
    };
    let start = (start..start + period)
        .find(|&at| text.is_char_boundary(at) && natural(at))
        .or_else(|| (start..start + period).find(|&at| text.is_char_boundary(at)))?;
    // A last copy cut short (trailing whitespace trimmed, or the answer
    // stopped mid-passage) still counts as one.
    let full = (text.len() - start) / period;
    let copies = full + usize::from(!(text.len() - start).is_multiple_of(period));
    let unit = &text[start..start + period];
    qualifies(unit, copies).then(|| Repeat {
        unit,
        copies,
        ends: (1..=full).map(|copy| start + copy * period).collect(),
        lines: false,
    })
}

#[cfg(test)]
mod tests;
