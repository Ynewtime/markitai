//! The Markdown of one run of inline content (a paragraph, a heading, a
//! table cell): the renderer writes markup and document text into a [`Line`],
//! and the text is escaped once the whole run is known, only where CommonMark
//! (with GFM's strikethrough) would read a character as syntax in that place.
//! Escaping each character wherever it might matter (every `*`, `_`, `[`,
//! `` ` ``, `\`) made `snake_case`, `[!tip]`, `C:\Users` and a form's
//! `________` hard to read in the Markdown itself, for no difference in what
//! a renderer shows.

use super::is_punctuation;

/// Where inline content is written, which decides what a line break becomes
/// and which characters can be read as Markdown there.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Place {
    /// A paragraph's lines, in a list item, quotation or note too: a line
    /// break is a hard break, two spaces at the line's end (normal output's
    /// cleanup of line ends keeps them where a line continues the paragraph).
    Text,
    /// A table cell, which holds inline content only: a line break is a line
    /// of the cell, joined with `<br>` (see `table_cell`).
    Cell,
    /// A heading or a link's text, which Markdown keeps on one line: a line
    /// break is a space.
    OneLine,
}

/// What a stretch of document text sits inside, as far as escaping it is
/// concerned.
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Literal {
    /// A link's text or an image's description, which a bracket would end.
    pub label: bool,
    /// Between the `*` or `**` of emphasis the renderer writes.
    pub emphasis: bool,
    /// Between the `~~` of a strikethrough the renderer writes.
    pub strike: bool,
}

/// One run of inline content as it is written: markup (`None`) as it stands,
/// document text (`Some`) escaped by [`Line::finish`].
#[derive(Default)]
pub(super) struct Line {
    pieces: Vec<(String, Option<Literal>)>,
    /// A line break was just written or left out: the whitespace after it
    /// starts the next line, where it would only indent it.
    after_break: bool,
}

impl Line {
    pub(super) fn markup(&mut self, text: &str) {
        self.push(text, None);
    }

    pub(super) fn text(&mut self, text: &str, kind: Literal) {
        self.push(text, Some(kind));
    }

    fn push(&mut self, text: &str, kind: Option<Literal>) {
        let text = if self.after_break {
            text.trim_start_matches([' ', '\t'])
        } else {
            text
        };
        if text.is_empty() {
            return;
        }
        self.after_break = false;
        match self.pieces.last_mut() {
            Some((last, last_kind)) if same_kind(*last_kind, kind) => last.push_str(text),
            _ => self.pieces.push((text.to_owned(), kind)),
        }
    }

    /// The last character written, before escaping (which only ever adds a
    /// backslash before a character).
    pub(super) fn last_char(&self) -> Option<char> {
        self.pieces
            .last()
            .and_then(|(text, _)| text.chars().next_back())
    }

    /// A line break written as `written` (empty where it shows nothing),
    /// without the spaces before it.
    pub(super) fn line_break(&mut self, written: &str) {
        while let Some((last, _)) = self.pieces.last_mut() {
            let kept = last.trim_end_matches([' ', '\t']).len();
            last.truncate(kept);
            if !last.is_empty() {
                break;
            }
            self.pieces.pop();
        }
        if !written.is_empty() {
            self.pieces.push((written.to_owned(), None));
        }
        self.after_break = true;
    }

    /// The Markdown of the line: markup as written, and each character of
    /// document text escaped where it would otherwise be syntax.
    pub(super) fn finish(self, place: Place) -> String {
        let mut chars = Vec::new();
        let mut kinds = Vec::new();
        for (text, kind) in &self.pieces {
            for c in text.chars() {
                chars.push(c);
                kinds.push(*kind);
            }
        }
        let escaped = escapes(&chars, &kinds, place);
        let mut output = String::with_capacity(chars.len() + 8);
        for (c, escaped) in chars.into_iter().zip(escaped) {
            match escaped {
                Escape::None => output.push(c),
                Escape::Backslash => {
                    output.push('\\');
                    output.push(c);
                }
                // A backtick beside the renderer's code span, which a
                // backslash would not part from it (`` \`` `` is still a
                // backtick string of two when a code span looks for its
                // end), or a `!` before its link.
                Escape::Reference => output.push_str(&format!("&#{};", u32::from(c))),
            }
        }
        output
    }
}

fn same_kind(a: Option<Literal>, b: Option<Literal>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.label == b.label && a.emphasis == b.emphasis && a.strike == b.strike
        }
        _ => false,
    }
}

/// How a character of document text is written.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Escape {
    None,
    Backslash,
    /// As a numeric character reference (`&#96;`, `&#33;`), where a
    /// backslash would not keep it from joining the markup beside it.
    Reference,
}

/// A maximal run of one emphasis delimiter (`*`, `_`, `~`), across text and
/// markup.
struct Run {
    start: usize,
    end: usize,
    mark: char,
    opens: bool,
    closes: bool,
    /// Holds document text, markup, or both (a marker the renderer writes
    /// beside a literal one, which would join it: `***x**`).
    text: bool,
    markup: bool,
    /// Its text stands between markers of the same character.
    inside: bool,
}

fn runs(chars: &[char], kinds: &[Option<Literal>]) -> Vec<Run> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mark = chars[start];
        if !matches!(mark, '*' | '_' | '~') {
            start += 1;
            continue;
        }
        let end = start + chars[start..].iter().take_while(|&&c| c == mark).count();
        let before = start.checked_sub(1).map(|at| chars[at]);
        let after = chars.get(end).copied();
        // The start and end of the line count as whitespace.
        let space = |c: Option<char>| c.is_none_or(char::is_whitespace);
        let punct = |c: Option<char>| c.is_some_and(is_punctuation);
        let left = !space(after) && (!punct(after) || space(before) || punct(before));
        let right = !space(before) && (!punct(before) || space(after) || punct(after));
        let (opens, closes) = if mark == '_' {
            // An underscore between two letters or digits neither opens nor
            // closes (`snake_case`).
            (
                left && (!right || punct(before)),
                right && (!left || punct(after)),
            )
        } else {
            (left, right)
        };
        // micromark (remark, MDX) also lets a run open before another
        // delimiter character and close after one (`*_*1` is emphasis there,
        // though not in CommonMark): text is escaped for both readings.
        let marker = |c: Option<char>| matches!(c, Some('*' | '_' | '~'));
        let kinds = &kinds[start..end];
        runs.push(Run {
            start,
            end,
            mark,
            opens: opens || marker(after),
            closes: closes || marker(before),
            text: kinds.iter().any(Option::is_some),
            markup: kinds.iter().any(Option::is_none),
            inside: kinds.iter().flatten().any(|kind| match mark {
                '*' => kind.emphasis,
                '~' => kind.strike,
                _ => false,
            }),
        });
        start = end;
    }
    runs
}

/// How each character of the line is written: only document text is ever
/// escaped, and of it only what would be read as Markdown where it stands.
///
/// - `*`, `_` and `~` runs that can open or close emphasis (CommonMark's
///   flanking rules) and have a partner that could pair with them (another
///   such run of the line's text, or one of the renderer's markers that can
///   both open and close), or stand inside the renderer's own emphasis, or
///   touch its markers. A marker that only opens or only closes pairs with
///   its own partner around its text (the delimiter algorithm matches the
///   nearest opener first), so text outside cannot take it.
/// - a backtick run when a later run of the same length (text or markup)
///   could close a code span; one that touches the renderer's code span is
///   written as a character reference.
/// - `\` before punctuation or at the end of a line (blanks after it are
///   trimmed), `<` that opens a tag
///   or an autolink (a `>` follows, or it starts an HTML block at the start
///   of a paragraph's line), `&` that starts an entity, `!` before a link the
///   renderer writes (as `&#33;`) and `[` after a `!`, `]` before `(` after a
///   `[`, `[` that would open a footnote mark (`[^1]`), and either bracket
///   inside a link's text.
///
/// Block syntax at the start of a line (`#`, `>`, list markers, rules,
/// fences, setext underlines) is `literal_heading_marks`' part, and `|` in
/// a table cell `table_cell`'s.
fn escapes(chars: &[char], kinds: &[Option<Literal>], place: Place) -> Vec<Escape> {
    let mut escape = vec![Escape::None; chars.len()];
    let runs = runs(chars, kinds);
    // The runs a text run could pair with: other text runs, and markers the
    // renderer writes that can both open and close (between two punctuation
    // marks, `.**|.**>`): processed as a closer, such a marker would take a
    // literal opener before it instead of its own partner. A marker beside
    // literal delimiters it would join (`**` + `**` of text, which get a
    // backslash) stands beside punctuation once they are escaped, so it may
    // then do either.
    let can_pair_open = |run: &Run| match (run.markup, run.text) {
        (true, true) => true,
        (true, false) => run.opens && run.closes,
        _ => run.opens,
    };
    let can_pair_close = |run: &Run| match (run.markup, run.text) {
        (true, true) => true,
        (true, false) => run.opens && run.closes,
        _ => run.closes,
    };
    let slot = |mark: char| match mark {
        '*' => 0,
        '_' => 1,
        _ => 2,
    };
    // Whether an earlier run of the same mark can open, and a later one can
    // close.
    let mut opened = Vec::with_capacity(runs.len());
    let mut seen = [false; 3];
    for run in &runs {
        opened.push(seen[slot(run.mark)]);
        if can_pair_open(run) {
            seen[slot(run.mark)] = true;
        }
    }
    let mut closed = vec![false; runs.len()];
    let mut seen = [false; 3];
    for (index, run) in runs.iter().enumerate().rev() {
        closed[index] = seen[slot(run.mark)];
        if can_pair_close(run) {
            seen[slot(run.mark)] = true;
        }
    }
    for (index, run) in runs.iter().enumerate() {
        if !run.text {
            continue;
        }
        let active = run.markup
            || ((run.opens || run.closes)
                && (run.inside || (run.opens && closed[index]) || (run.closes && opened[index])));
        if active {
            for at in run.start..run.end {
                if kinds[at].is_some() {
                    escape[at] = Escape::Backslash;
                }
            }
        }
    }
    backticks(chars, kinds, &mut escape);
    let last_close = chars.iter().rposition(|&c| c == '>');
    let first_open = chars.iter().position(|&c| c == '[');
    for (at, &c) in chars.iter().enumerate() {
        let Some(kind) = kinds[at] else {
            continue;
        };
        let next = chars.get(at + 1).copied();
        let needed = match c {
            // Before punctuation, or with nothing but blanks after it on its
            // line: line ends are trimmed, and a backslash ending a line is
            // a hard break.
            '\\' => {
                next.is_some_and(|next| next.is_ascii_punctuation())
                    || chars[at + 1..]
                        .iter()
                        .take_while(|&&c| c != '\n')
                        .all(|&c| c == ' ' || c == '\t')
            }
            '<' => {
                next.is_some_and(|next| {
                    next.is_ascii_alphabetic() || matches!(next, '/' | '!' | '?')
                }) && (last_close.is_some_and(|close| close > at)
                    || (place == Place::Text && starts_line(chars, at)))
            }
            '&' => entity_follows(&chars[at + 1..]),
            // `!` before a link the renderer writes would make it an image.
            // Normal output's image repairs read `![` whatever stands before
            // it (`\![a](b))` would lose its last `)`), so the `!` is a
            // reference there, and a literal `![` has its bracket escaped.
            '!' if next == Some('[') && kinds[at + 1].is_none() => {
                escape[at] = Escape::Reference;
                false
            }
            '[' if at > 0 && chars[at - 1] == '!' => true,
            '[' => kind.label || (next == Some('^') && footnote_label(&chars[at + 2..])),
            ']' => kind.label || (next == Some('(') && first_open.is_some_and(|open| open < at)),
            _ => false,
        };
        if needed && escape[at] == Escape::None {
            escape[at] = Escape::Backslash;
        }
    }
    escape
}

/// Backticks of document text: a run that touches a backtick of the
/// renderer's code span is written as references, and another run is
/// escaped when a later run of the same length (text or markup, escaped or
/// not: a code span's end ignores backslashes) could close a code span it
/// opens.
fn backticks(chars: &[char], kinds: &[Option<Literal>], escape: &mut [Escape]) {
    // Runs of one kind (text or markup), and whether a text run touches
    // markup backticks.
    let mut runs: Vec<(usize, usize, bool, bool)> = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        if chars[start] != '`' {
            start += 1;
            continue;
        }
        let whole = start + chars[start..].iter().take_while(|&&c| c == '`').count();
        let mixed = kinds[start..whole].iter().any(Option::is_some)
            && kinds[start..whole].iter().any(Option::is_none);
        let mut at = start;
        while at < whole {
            let text = kinds[at].is_some();
            let end = at
                + kinds[at..whole]
                    .iter()
                    .take_while(|kind| kind.is_some() == text)
                    .count();
            runs.push((at, end, text, text && mixed));
            at = end;
        }
        start = whole;
    }
    let mut later = std::collections::HashSet::new();
    for &(start, end, text, touching) in runs.iter().rev() {
        if touching {
            escape[start..end].fill(Escape::Reference);
            continue;
        }
        if text && later.contains(&(end - start)) {
            escape[start..end].fill(Escape::Backslash);
        }
        later.insert(end - start);
    }
}

/// Whether only spaces or tabs stand between the line's start and `at`.
fn starts_line(chars: &[char], at: usize) -> bool {
    chars[..at]
        .iter()
        .rev()
        .take_while(|&&c| c != '\n')
        .all(|&c| c == ' ' || c == '\t')
}

/// Whether the characters after an `&` make it a character reference:
/// `&name;`, `&#123;` or `&#x1F;`.
fn entity_follows(rest: &[char]) -> bool {
    let Some(end) = rest.iter().take(33).position(|&c| c == ';') else {
        return false;
    };
    match &rest[..end] {
        ['#', 'x' | 'X', hex @ ..] => {
            (1..=6).contains(&hex.len()) && hex.iter().all(char::is_ascii_hexdigit)
        }
        ['#', digits @ ..] => {
            (1..=7).contains(&digits.len()) && digits.iter().all(char::is_ascii_digit)
        }
        [first, name @ ..] => {
            first.is_ascii_alphabetic() && name.iter().all(char::is_ascii_alphanumeric)
        }
        [] => false,
    }
}

/// Whether the characters after `[^` make a footnote mark's label: one or
/// more characters, none of them whitespace or a bracket, up to a `]`.
fn footnote_label(rest: &[char]) -> bool {
    rest.iter()
        .position(|&c| c == ']')
        .is_some_and(|end| end > 0 && rest[..end].iter().all(|&c| !c.is_whitespace() && c != '['))
}
