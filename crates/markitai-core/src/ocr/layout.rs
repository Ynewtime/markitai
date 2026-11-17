//! Reading aids that need only the recognized lines: turning a page whose
//! text runs sideways or upside down upright, finding table cells that a
//! reading missed or garbled so that `vision` can read them again, mending
//! zeros read as look-alike letters, keeping the indentation of code,
//! closing up letter-spaced text and splitting table rows into cells.
// Off macOS the portable engine uses the turned pages, zeros, code and
// joining steps; the readings again of table cells serve Vision.
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use super::{Line, cjk};

/// Which way a page's text runs, from the corners the recognizer gives
/// each line (its top-left to top-right edge).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Turn {
    /// Left to right.
    Upright,
    /// Top to bottom: the page was turned a quarter clockwise.
    Clockwise,
    /// Right to left: the page is upside down.
    Half,
    /// Bottom to top: the page was turned a quarter counter-clockwise.
    Counter,
}

/// Lines with fewer letters than this do not vote on the page's direction.
const MIN_VOTE: usize = 3;
/// The share of the voting letters that must run one way to turn the page.
const TURNED: usize = 3; // in four

fn turn_of([dx, dy]: [f32; 2]) -> Option<Turn> {
    if !dx.is_finite() || !dy.is_finite() || dx == 0.0 && dy == 0.0 {
        return None;
    }
    Some(if dx.abs() >= dy.abs() {
        if dx > 0.0 { Turn::Upright } else { Turn::Half }
    } else if dy > 0.0 {
        Turn::Clockwise
    } else {
        Turn::Counter
    })
}

/// The way most of the text runs, by letters. A page is turned only when
/// three quarters of its letters run the same other way: a sideways label
/// on an upright page stays where it is.
pub(super) fn orientation(lines: &[Line]) -> Turn {
    let mut votes = [0usize; 4];
    for line in lines {
        let letters = line.text.chars().filter(|c| !c.is_whitespace()).count();
        if letters >= MIN_VOTE
            && let Some(turn) = turn_of(line.direction)
        {
            votes[turn as usize] += letters;
        }
    }
    let total: usize = votes.iter().sum();
    [Turn::Clockwise, Turn::Half, Turn::Counter]
        .into_iter()
        .find(|turn| total > 0 && votes[*turn as usize] * 4 >= total * TURNED)
        .unwrap_or(Turn::Upright)
}

/// Turn the lines of a `width` by `height` page upright, rectangles
/// included, and return the upright page's size.
pub(super) fn upright(lines: &mut [Line], turn: Turn, width: u32, height: u32) -> (u32, u32) {
    if turn == Turn::Upright {
        return (width, height);
    }
    let (w, h) = (width as f32, height as f32);
    for line in lines.iter_mut() {
        let [l, t, r, b] = line.bounds;
        line.bounds = match turn {
            Turn::Upright => line.bounds,
            // Turned back a quarter counter-clockwise: (x, y) -> (y, W - x).
            Turn::Clockwise => [t, w - r, b, w - l],
            Turn::Half => [w - r, h - b, w - l, h - t],
            // Turned back a quarter clockwise: (x, y) -> (H - y, x).
            Turn::Counter => [h - b, l, h - t, r],
        }
        .map(|n| n.max(0.0));
        line.direction = [1.0, 0.0];
    }
    if turn == Turn::Half {
        (width, height)
    } else {
        (height, width)
    }
}

/// Letters the recognizer reads in place of a zero.
fn zero_like(character: char) -> bool {
    matches!(character, 'O' | 'o' | 'ø' | 'Ø')
}

/// Characters a number may hold besides digits.
fn numeric_mark(character: char) -> bool {
    matches!(
        character,
        '.' | ','
            | ':'
            | ';'
            | '/'
            | '-'
            | '+'
            | '%'
            | '$'
            | '€'
            | '£'
            | '¥'
            | '#'
            | '('
            | ')'
            | '['
            | ']'
            | '\''
            | '"'
            | '−'
            | '–'
    )
}

/// Characters that end a token: spaces, and Chinese, Japanese and Korean
/// letters and punctuation, which are not written with spaces.
fn token_end(character: char) -> bool {
    character.is_whitespace()
        || cjk::han(character)
        || cjk::kana(character)
        || matches!(character, '\u{3000}'..='\u{303f}' | '\u{ac00}'..='\u{d7a3}' | '\u{ff00}'..='\u{ffef}')
}

/// `text` with zeros mended: in a token of digits and number marks only,
/// with at least two digits, a run of letters that look like a zero
/// (`O`, `o`, `ø`) between two digits is read as zeros (`2ø26` is 2026).
/// Letters in words, and at the ends of numbers, are kept.
pub(super) fn digits(text: &str) -> Option<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut mended = characters.clone();
    let mut changed = false;
    let mut start = 0;
    while start < characters.len() {
        let end = (start..characters.len())
            .find(|&i| token_end(characters[i]))
            .unwrap_or(characters.len());
        let token = &characters[start..end];
        let numeric = token
            .iter()
            .all(|&c| c.is_ascii_digit() || zero_like(c) || numeric_mark(c))
            && token.iter().filter(|c| c.is_ascii_digit()).count() >= 2;
        if numeric {
            let mut i = 0;
            while i < token.len() {
                if !zero_like(token[i]) {
                    i += 1;
                    continue;
                }
                let run = (i..token.len())
                    .find(|&j| !zero_like(token[j]))
                    .unwrap_or(token.len());
                let bounded = i > 0
                    && token[i - 1].is_ascii_digit()
                    && run < token.len()
                    && token[run].is_ascii_digit();
                if bounded {
                    mended[start + i..start + run].fill('0');
                    changed = true;
                }
                i = run;
            }
        }
        start = end + 1;
    }
    changed.then(|| mended.into_iter().collect())
}

/// A region of the page to read again, enlarged, and what to do with it.
#[derive(Debug, PartialEq)]
pub(super) struct Reread {
    /// Pixel rectangle `[left, top, right, bottom]` of the original page.
    pub region: [f32; 4],
    /// The region's left edge with the row's previous cell included, whose
    /// text shows the recognizer which way up a lone digit stands.
    pub context: f32,
    /// The height of the region's row, in pixels.
    pub height: f32,
    pub target: Target,
}

impl Reread {
    /// The readings to try, in order, as a rectangle, an enlargement factor
    /// and a white margin in original pixels. Measured on 62 cells missing
    /// from 50 rendered tables, a cell read with the previous one, twice and
    /// then three times enlarged, and the cell alone with a margin of one
    /// row height, twice enlarged, recover 48 (one wrongly: `QI` for `Q1`),
    /// while the cell alone twice and three times enlarged recovers 27.
    pub fn attempts(&self) -> Vec<([f32; 4], f32, f32)> {
        let [_, top, right, bottom] = self.region;
        let wide = [self.context, top, right, bottom];
        match self.target {
            Target::Cell { .. } => vec![
                (wide, 2.0, 0.0),
                (wide, 3.0, 0.0),
                (self.region, 2.0, self.height),
            ],
            Target::Mend { .. } => vec![
                (self.region, 2.0, 0.0),
                (self.region, 3.0, 0.0),
                (self.region, 2.0, self.height),
            ],
        }
    }
}

#[derive(Debug, PartialEq)]
pub(super) enum Target {
    /// A table cell the reading missed: what the region reads inside this
    /// column span and row band becomes a new line.
    Cell {
        span: [f32; 2],
        band: [f32; 2],
        max_chars: usize,
        /// Whether the column holds numbers.
        numeric: bool,
    },
    /// A line whose digits hold garbage letters: the region's reading
    /// replaces its text when it mends exactly those letters.
    Mend { index: usize },
}

/// Regions read again per image, at most; each is read at up to two sizes.
pub(super) const MAX_REREADS: usize = 12;
/// Cells of a table row are this many of their heights apart, at least.
const GUTTER: f32 = 1.0;
/// A column whose cells hold more characters than this (by median) is text,
/// not numbers or short labels, and is not searched for missing cells.
const SHORT_CELL: usize = 12;
/// Padding of a region around its column, and above and below its row, in
/// row heights. Tight regions were read less reliably than these.
const PAD_X: f32 = 0.5;
const PAD_Y: f32 = 0.35;

/// Lines, as indices, grouped into rows (lines sharing most of a vertical
/// band, as in assembly), top to bottom, each row left to right.
fn rows_of(lines: &[Line]) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lines.len())
        .filter(|&i| !lines[i].text.trim().is_empty())
        .collect();
    crate::sort::by(&mut order, |&a, &b| {
        super::top_then_left(&lines[a], &lines[b])
    });
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for index in order {
        let next = lines[index].bounds;
        let joins = rows.last().is_some_and(|row| {
            let first = lines[row[0]].bounds;
            let overlap = first[3].min(next[3]) - first[1].max(next[1]);
            overlap > 0.0 && overlap >= 0.6 * (first[3] - first[1]).min(next[3] - next[1])
        });
        if joins {
            rows.last_mut().unwrap().push(index);
        } else {
            rows.push(vec![index]);
        }
    }
    for row in &mut rows {
        crate::sort::by(row, |&a, &b| {
            lines[a].bounds[0].total_cmp(&lines[b].bounds[0])
        });
    }
    rows
}

fn median(mut values: Vec<f32>) -> f32 {
    crate::sort::by(&mut values, f32::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

/// What a reading's tables and numbers need.
#[derive(Debug, Default)]
pub(super) struct Found {
    /// Regions to read again.
    pub rereads: Vec<Reread>,
    /// Lines that are a cell of a column of numbers holding only a Cyrillic
    /// letter drawn like a digit (`З`), with that digit.
    pub digits: Vec<(usize, char)>,
}

/// Regions to read again: when `latin` (a reading of Latin script), lines
/// whose numbers hold letters of other alphabets (`1,200,00łł`), then cells
/// missing from the rows of a table; at most [`MAX_REREADS`]. In a Latin
/// reading, cells of numbers read as a Cyrillic look-alike are digits.
pub(super) fn rereads(lines: &[Line], width: u32, height: u32, latin: bool) -> Found {
    let mut mends = Vec::new();
    if latin {
        for (index, line) in lines.iter().enumerate() {
            if garbled(&line.text) {
                let h = line.bounds[3] - line.bounds[1];
                let region = clip(
                    [
                        line.bounds[0] - PAD_X * h,
                        line.bounds[1] - PAD_Y * h,
                        line.bounds[2] + PAD_X * h,
                        line.bounds[3] + PAD_Y * h,
                    ],
                    width,
                    height,
                );
                mends.push(Reread {
                    region,
                    context: region[0],
                    height: h,
                    target: Target::Mend { index },
                });
            }
        }
    }
    let mut found = holes(lines, width, height);
    if !latin {
        found.digits.clear();
    }
    mends.append(&mut found.rereads);
    found.rereads = mends;
    found.rereads.truncate(MAX_REREADS);
    found
}

fn clip([l, t, r, b]: [f32; 4], width: u32, height: u32) -> [f32; 4] {
    let (w, h) = (width as f32, height as f32);
    [
        l.clamp(0.0, w),
        t.clamp(0.0, h),
        r.clamp(0.0, w),
        b.clamp(0.0, h),
    ]
}

/// Cells missing from table rows. A table is three or more rows of cells at
/// least a row height apart; its columns are those of the rows with the
/// most cells (two or more such rows), and a row with fewer cells, each in
/// one column, misses the cells of the other short columns.
fn holes(lines: &[Line], width: u32, height: u32) -> Found {
    let rows = rows_of(lines);
    let tall = |row: &[usize]| {
        median(
            row.iter()
                .map(|&i| lines[i].bounds[3] - lines[i].bounds[1])
                .collect(),
        )
    };
    let table: Vec<usize> = (0..rows.len())
        .filter(|&r| {
            let h = tall(&rows[r]);
            rows[r].len() >= 2
                && rows[r]
                    .windows(2)
                    .all(|pair| lines[pair[1]].bounds[0] - lines[pair[0]].bounds[2] >= GUTTER * h)
        })
        .collect();
    let mut found = Found::default();
    let Some(most) = table.iter().map(|&r| rows[r].len()).max() else {
        return found;
    };
    let full: Vec<usize> = table
        .iter()
        .copied()
        .filter(|&r| rows[r].len() == most)
        .collect();
    if table.len() < 3 || full.len() < 2 {
        return found;
    }
    let spans: Vec<[f32; 2]> = (0..most)
        .map(|column| {
            full.iter()
                .fold([f32::INFINITY, f32::NEG_INFINITY], |[l, r], &row| {
                    let b = lines[rows[row][column]].bounds;
                    [l.min(b[0]), r.max(b[2])]
                })
        })
        .collect();
    if spans.windows(2).any(|pair| pair[0][1] >= pair[1][0]) {
        return found;
    }
    // The column of each cell of a table row: the one span it overlaps.
    let column_of = |index: usize| -> Option<usize> {
        let b = lines[index].bounds;
        let mut overlapped = (0..most).filter(|&c| b[0] < spans[c][1] && b[2] > spans[c][0]);
        let column = overlapped.next()?;
        overlapped.next().is_none().then_some(column)
    };
    // The lines of each column, from the rows whose cells each lie in one.
    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); most];
    let mut placed: Vec<Option<Vec<Option<usize>>>> = Vec::new();
    for &r in &table {
        let columns: Vec<Option<usize>> = rows[r].iter().map(|&i| column_of(i)).collect();
        let mut seen = vec![false; most];
        let aligned = columns.iter().all(|column| match column {
            Some(c) if !seen[*c] => {
                seen[*c] = true;
                true
            }
            _ => false,
        });
        if aligned {
            for (&i, column) in rows[r].iter().zip(&columns) {
                cells[column.unwrap()].push(i);
            }
        }
        placed.push(aligned.then_some(columns));
    }
    let count = |i: &usize| lines[*i].text.chars().count();
    let max_chars: Vec<usize> = cells
        .iter()
        .map(|c| c.iter().map(count).max().unwrap_or(0))
        .collect();
    let short: Vec<bool> = cells
        .iter()
        .map(|c| {
            let mut counts: Vec<usize> = c.iter().map(count).collect();
            counts.sort_unstable();
            !counts.is_empty() && counts[counts.len() / 2] <= SHORT_CELL
        })
        .collect();
    // Columns of numbers: at least half of their cells are.
    let numeric: Vec<bool> = cells
        .iter()
        .map(|c| c.iter().filter(|&&i| number(&lines[i].text)).count() * 2 >= c.len())
        .collect();
    for (column, c) in cells.iter().enumerate() {
        for &i in c.iter().filter(|_| numeric[column]) {
            if let Some(digit) = digit_like(&lines[i].text) {
                found.digits.push((i, digit));
            }
        }
    }
    for (position, &r) in table.iter().enumerate() {
        let Some(columns) = &placed[position] else {
            continue;
        };
        if columns.len() == most {
            continue;
        }
        let row = &rows[r];
        let h = tall(row);
        let top = row
            .iter()
            .map(|&i| lines[i].bounds[1])
            .fold(f32::INFINITY, f32::min);
        let bottom = row
            .iter()
            .map(|&i| lines[i].bounds[3])
            .fold(f32::NEG_INFINITY, f32::max);
        // Keep clear of the rows above and below.
        let above = r.checked_sub(1).map(|p| {
            rows[p]
                .iter()
                .map(|&i| lines[i].bounds[3])
                .fold(f32::NEG_INFINITY, f32::max)
        });
        let below = rows.get(r + 1).map(|n| {
            n.iter()
                .map(|&i| lines[i].bounds[1])
                .fold(f32::INFINITY, f32::min)
        });
        let y0 = (top - PAD_Y * h).max(above.unwrap_or(f32::NEG_INFINITY));
        let y1 = (bottom + PAD_Y * h).min(below.unwrap_or(f32::INFINITY));
        for column in (0..most).filter(|c| short[*c] && !columns.contains(&Some(*c))) {
            let [s0, s1] = spans[column];
            let mut x0 = s0 - PAD_X * h;
            let mut x1 = s1 + PAD_X * h;
            let mut previous = None;
            // Keep clear of the row's own cells on either side.
            for (&i, placed) in row.iter().zip(columns) {
                let b = lines[i].bounds;
                if placed.is_some_and(|c| c < column) {
                    x0 = x0.max(b[2] + 0.25 * h);
                    previous = Some(b[0] - 0.3 * h);
                } else {
                    x1 = x1.min(b[0] - 0.25 * h);
                }
            }
            let region = clip([x0, y0, x1, y1], width, height);
            if region[2] - region[0] >= 0.5 * h && region[3] - region[1] >= 0.5 * h {
                found.rereads.push(Reread {
                    region,
                    context: previous.map_or(region[0], |left| left.max(0.0)),
                    height: h,
                    target: Target::Cell {
                        span: [x0, x1],
                        band: [top, bottom],
                        max_chars: max_chars[column].max(1) * 2,
                        numeric: numeric[column],
                    },
                });
            }
        }
    }
    found
}

/// A letter of another alphabet, a bullet or a noncharacter in a number:
/// what the English recognizer makes of some digits and points
/// (`1,200,00łł`, `202łąłą`, `183•33`, `15\u{fffe}41`). Zeros read as `ø`
/// are mended without a second reading ([`digits`]).
fn garbage(character: char) -> bool {
    character.is_alphabetic() && !character.is_ascii() && !zero_like(character)
        || matches!(character, '•' | '\u{fffd}'..='\u{ffff}')
}

/// Whether a line holds a token with a digit and such a character.
fn garbled(text: &str) -> bool {
    text.split_whitespace()
        .any(|token| token.contains(|c: char| c.is_ascii_digit()) && token.contains(garbage))
}

/// Whether a cell is a number: digits and number marks, one digit at least.
fn number(text: &str) -> bool {
    let text = text.trim();
    text.contains(|c: char| c.is_ascii_digit())
        && text.chars().all(|c| c.is_ascii_digit() || numeric_mark(c))
}

/// The digit a cell holding only a Cyrillic letter drawn like one stands for
/// in a column of numbers (`З` for 3, which the English recognizer reads).
fn digit_like(text: &str) -> Option<char> {
    match text.trim() {
        "З" | "з" => Some('3'),
        "О" | "о" => Some('0'),
        "б" => Some('6'),
        _ => None,
    }
}

/// What a missing cell's region reads: the lines whose centres lie in the
/// cell's column span and row band, left to right, unless they are too
/// long for the column, hold no letter or digit (a rule read as `|`), or,
/// in a reading of Latin script (`latin`), hold letters of other alphabets;
/// a Cyrillic look-alike of a digit in a `numeric` column is that digit.
pub(super) fn cell(
    found: Vec<Line>,
    span: [f32; 2],
    band: [f32; 2],
    max_chars: usize,
    numeric: bool,
    latin: bool,
) -> Option<Line> {
    let mut inside: Vec<Line> = found
        .into_iter()
        .filter(|line| {
            let x = (line.bounds[0] + line.bounds[2]) / 2.0;
            let y = (line.bounds[1] + line.bounds[3]) / 2.0;
            !line.text.trim().is_empty()
                && x >= span[0]
                && x <= span[1]
                && y >= band[0]
                && y <= band[1]
        })
        .collect();
    crate::sort::by(&mut inside, |a, b| a.bounds[0].total_cmp(&b.bounds[0]));
    let mut text = inside
        .iter()
        .map(|l| l.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    if let Some(digit) = digit_like(&text).filter(|_| numeric && latin) {
        text = digit.to_string();
    }
    let letters = text.chars().filter(|c| c.is_alphanumeric()).count();
    let rule = text
        .chars()
        .all(|c| !c.is_alphanumeric() || matches!(c, 'I' | 'l'));
    if inside.is_empty()
        || letters == 0
        || rule
        || text.chars().count() > max_chars
        || latin && text.chars().any(garbage)
    {
        return None;
    }
    let bounds = inside
        .iter()
        .fold([f32::INFINITY, f32::INFINITY, 0.0f32, 0.0f32], |a, l| {
            [
                a[0].min(l.bounds[0]),
                a[1].min(l.bounds[1]),
                a[2].max(l.bounds[2]),
                a[3].max(l.bounds[3]),
            ]
        });
    let confidence = inside.iter().map(|l| l.confidence).sum::<f32>() / inside.len() as f32;
    Some(Line {
        text,
        confidence,
        bounds,
        direction: [1.0, 0.0],
        marks: Default::default(),
    })
}

/// The text of a garbled line as a second `reading` mends it: the same
/// tokens, each garbled one replaced by digits and number marks that keep
/// its clean beginning and end and are no longer than it.
pub(super) fn mend(original: &str, reading: &str) -> Option<String> {
    let before: Vec<&str> = original.split_whitespace().collect();
    let after: Vec<&str> = reading.split_whitespace().collect();
    if before.len() != after.len() || before == after {
        return None;
    }
    for (old, new) in before.iter().zip(&after) {
        if old == new {
            continue;
        }
        let characters: Vec<char> = old.chars().collect();
        // The garbage, with Latin letters next to it (`72.7zął`).
        let mut first = characters.iter().position(|c| garbage(*c))?;
        let mut last = characters.iter().rposition(|c| garbage(*c))?;
        while first > 0 && characters[first - 1].is_ascii_alphabetic() {
            first -= 1;
        }
        while last + 1 < characters.len() && characters[last + 1].is_ascii_alphabetic() {
            last += 1;
        }
        let head: String = characters[..first].iter().collect();
        let tail: String = characters[last + 1..].iter().collect();
        let middle = new
            .strip_prefix(head.as_str())?
            .strip_suffix(tail.as_str())?;
        let clean = middle
            .chars()
            .all(|c| c.is_ascii_digit() || numeric_mark(c));
        if !clean || new.chars().count() > characters.len() || new.chars().any(garbage) {
            return None;
        }
    }
    Some(after.join(" "))
}

/// Code drawn in a fixed-pitch font: its rows, their pitch and left edge.
#[derive(Debug, PartialEq)]
pub(super) struct Code {
    /// The block's rows (as `rows` passes them) that are code: `start..end`.
    pub rows: std::ops::Range<usize>,
    /// Pixels per character.
    pub pitch: f32,
    /// The left edge of the least indented row.
    pub left: f32,
}

/// Rows of code need at least this many rows.
const MIN_CODE_ROWS: usize = 3;
/// A row's pitch (width per character) may differ by this share from the
/// others' in a fixed-pitch font.
const PITCH_SPREAD: f32 = 0.15;
/// Characters a row needs before its pitch is measured.
const MIN_PITCH_CHARS: usize = 3;

fn codelike(text: &str) -> bool {
    let text = text.trim();
    text.ends_with(':')
        || text.starts_with('#')
        || text.starts_with("//")
        || text.contains(['(', ')', '{', '}', '[', ']', '=', ';', '<', '>'])
}

/// The run of rows that is code, if any: at least three consecutive rows
/// of one line each in a fixed pitch (every measured row within 15% of the
/// median width per character, rows of a few characters excepted), with a
/// row indented by at least one and a half characters and a third of the
/// rows holding code punctuation (brackets, `=`, `;`, or ending in `:`).
/// Text in proportional fonts, receipts and tables, whose rows all start at
/// the same edge, are not code.
pub(super) fn code(lines: &[Line], rows: &[std::ops::Range<usize>]) -> Option<Code> {
    let pitch_of = |row: &std::ops::Range<usize>| -> Option<f32> {
        if row.len() != 1 {
            return None;
        }
        let line = &lines[row.start];
        let count = line.text.chars().count();
        (line.text.chars().filter(|c| !c.is_whitespace()).count() >= MIN_PITCH_CHARS)
            .then(|| (line.bounds[2] - line.bounds[0]) / count as f32)
    };
    let pitches: Vec<Option<f32>> = rows.iter().map(pitch_of).collect();
    let measured: Vec<f32> = pitches.iter().flatten().copied().collect();
    if measured.len() < MIN_CODE_ROWS {
        return None;
    }
    let pitch = median(measured);
    if pitch <= 0.0 {
        return None;
    }
    // Rows of one line in the pitch, or too short to measure, may be code.
    let fits: Vec<bool> = rows
        .iter()
        .zip(&pitches)
        .map(|(row, p)| row.len() == 1 && p.is_none_or(|p| (p / pitch - 1.0).abs() <= PITCH_SPREAD))
        .collect();
    let mut best: Option<std::ops::Range<usize>> = None;
    let mut start = 0;
    while start < rows.len() {
        if !fits[start] {
            start += 1;
            continue;
        }
        let end = (start..rows.len())
            .find(|&i| !fits[i])
            .unwrap_or(rows.len());
        if best.as_ref().is_none_or(|b| end - start > b.len()) {
            best = Some(start..end);
        }
        start = end;
    }
    let run = best?;
    let members = &rows[run.clone()];
    let measured = members.iter().filter(|row| pitch_of(row).is_some()).count();
    let left = members
        .iter()
        .map(|row| lines[row.start].bounds[0])
        .fold(f32::INFINITY, f32::min);
    let indented = members
        .iter()
        .any(|row| lines[row.start].bounds[0] - left >= 1.5 * pitch);
    let punctuated = members
        .iter()
        .filter(|row| codelike(&lines[row.start].text))
        .count();
    (run.len() >= MIN_CODE_ROWS
        && measured >= MIN_CODE_ROWS
        && indented
        && punctuated * 3 >= run.len())
    .then_some(Code {
        rows: run,
        pitch,
        left,
    })
}

/// The indentation, in characters, of a code line starting at `left`.
pub(super) fn indent(code: &Code, left: f32) -> usize {
    const MAX_INDENT: f32 = 80.0;
    ((left - code.left) / code.pitch)
        .round()
        .clamp(0.0, MAX_INDENT) as usize
}

/// An ink-free gap at least this many line heights wide between a Chinese or
/// Japanese letter and a Latin one is a space. Measured on 201 rendered
/// Chinese images: such gaps are 0.02 to 0.12 line heights where the text has
/// no space and 0.09 to 0.51 (a twentieth below 0.2) where it has one.
pub(super) const SPACE: f32 = 0.22;

/// Positions `n` where `characters[n]` and `characters[n + 1]` are a Chinese
/// or Japanese letter and a Latin letter or digit, in either order, with no
/// space between them.
pub(super) fn junctions(characters: &[char]) -> Vec<usize> {
    let wide = |c: char| cjk::han(c) || cjk::kana(c);
    let narrow = |c: char| c.is_ascii_alphanumeric();
    (0..characters.len().saturating_sub(1))
        .filter(|&n| {
            let (a, b) = (characters[n], characters[n + 1]);
            (wide(a) && narrow(b)) || (narrow(a) && wide(b))
        })
        .collect()
}

/// Whether a line of `characters` may want a space the recognizer left out:
/// a Chinese or Japanese letter meets a Latin letter or digit, or the line
/// has two pairs of digits, or of Chinese letters, read together.
pub(super) fn may_space(characters: &[char]) -> bool {
    !junctions(characters).is_empty()
        || digit_pairs(characters).len() >= 2
        || han_pairs(characters).len() >= 2
}

/// Positions `n` where `characters[n]` and `characters[n + 1]` are digits
/// with no space between them.
pub(super) fn digit_pairs(characters: &[char]) -> Vec<usize> {
    (0..characters.len().saturating_sub(1))
        .filter(|&n| characters[n].is_ascii_digit() && characters[n + 1].is_ascii_digit())
        .collect()
}

/// Positions `n` where `characters[n]` and `characters[n + 1]` are Chinese
/// letters with no space between them.
pub(super) fn han_pairs(characters: &[char]) -> Vec<usize> {
    (0..characters.len().saturating_sub(1))
        .filter(|&n| cjk::han(characters[n]) && cjk::han(characters[n + 1]))
        .collect()
}

fn gray(image: &image::RgbImage, x: u32, y: u32) -> i16 {
    luma(image.get_pixel(x, y))
}

fn luma(pixel: &image::Rgb<u8>) -> i16 {
    let [r, g, b] = pixel.0.map(u32::from);
    ((r * 299 + g * 587 + b * 114) / 1000) as i16
}

/// The pixel rows `[top, bottom)` and columns `[left, right)` of `area`
/// within the image, if nonempty.
fn pixel_area(image: &image::RgbImage, area: [f32; 4]) -> Option<[u32; 4]> {
    let (width, height) = image.dimensions();
    let [l, t, r, b] = area.map(|n| n.max(0.0));
    let area = [
        (l as u32).min(width),
        (t as u32).min(height),
        (r.ceil() as u32).min(width),
        (b.ceil() as u32).min(height),
    ];
    (area[2] > area[0] && area[3] > area[1]).then_some(area)
}

/// The background gray of a line's pixel rectangle `area`: its median gray,
/// which on a page of text is the paper (or, in dark mode, the dark ground).
pub(super) fn background(image: &image::RgbImage, area: [f32; 4]) -> Option<i16> {
    let [left, top, right, bottom] = pixel_area(image, area)?;
    let mut counts = [0usize; 256];
    for x in (left..right).step_by(3) {
        for y in (top..bottom).step_by(3) {
            counts[gray(image, x, y) as usize] += 1;
        }
    }
    let half = counts.iter().sum::<usize>() / 2;
    let mut seen = 0;
    (0..256)
        .find(|&g| {
            seen += counts[g];
            seen > half
        })
        .map(|g| g as i16)
}

/// Whether the image shows a space between two characters centred at `x[0]`
/// and `x[1]` on a line spanning the pixel rows `band`, on a `background`
/// gray: the widest run of columns between them without ink (a pixel whose
/// gray is this far from the background) is at least [`SPACE`] line heights.
pub(super) fn spaced(
    image: &image::RgbImage,
    band: [f32; 2],
    x: [f32; 2],
    background: i16,
) -> bool {
    const INK: i16 = 80;
    let Some([left, top, right, bottom]) = pixel_area(image, [x[0], band[0], x[1], band[1]]) else {
        return false;
    };
    let tall = bottom - top;
    if tall < 4 {
        return false;
    }
    let (mut run, mut widest) = (0u32, 0u32);
    for x in left..right {
        let ink = (top..bottom).any(|y| (gray(image, x, y) - background).abs() > INK);
        run = if ink { 0 } else { run + 1 };
        widest = widest.max(run);
    }
    widest as f32 >= SPACE * tall as f32
}

/// A pixel this far from the background gray is a digit's ink: low enough
/// that the hairlines of a thin serif `0` close it (at the 80 of [`spaced`]
/// its two sides would be two glyphs).
const DIGIT_INK: i16 = 40;

/// Two digits read with no space between them stand a space apart when the
/// centres of their ink are at least this many times as far apart as the
/// line's digits usually stand, and the gap between them is at least
/// [`SPACE`] line heights. Figures share one width, so a narrow `1` has a
/// wide gap on each side yet stands no farther from its neighbours, while a
/// space adds a third to a half of a figure. Measured on 732 rendered images
/// of number rows (28 fonts, 14 to 32 pixels, clean and blurred JPEG): the
/// digits of a number stand at most 1.22 times the usual distance apart,
/// digits a space apart mostly 1.35 times or more.
const DIGIT_SPACE: f32 = 1.35;

/// Positions `n` (from [`digit_pairs`]) where the recognizer read the digits
/// `characters[n]` and `characters[n + 1]` together but `image`, a cut-out
/// line on a `background` gray, shows a space between them: two numbers a
/// word space or two apart, as the cells of a table row often stand. `x(n)`
/// is where the recognizer places character `n`, in the image's columns.
///
/// The image's columns of ink (any pixel [`DIGIT_INK`] from the background)
/// run in glyphs, and each character takes the run nearest its place. A run
/// that digits in a row take together (the figures of one number, run
/// together by blur) is cut between each two at the column with the least
/// ink between their places. A pair is measured when its digits have a run
/// or part each, side by side, and spaced as [`DIGIT_SPACE`] says, against
/// the median distance (the lower middle one) between the centres of the
/// line's measured pairs of whole runs; a centre weighs each column by its
/// ink, so the flag of a `1` does not pull it aside. Digits on either side
/// of ink that runs on (a space blurred over), or with ink between them (a
/// point the recognizer dropped), and lines with fewer than two measured
/// pairs of whole runs, stay as read.
pub(super) fn digit_spaces(
    image: &image::RgbImage,
    background: i16,
    characters: &[char],
    x: impl Fn(usize) -> f32,
) -> Vec<usize> {
    let pairs = digit_pairs(characters);
    let (width, height) = image.dimensions();
    if pairs.len() < 2 || height < 4 {
        return Vec::new();
    }
    let width = width as usize;
    let (strongest, weight) = column_ink(image, background);
    // Runs of ink columns: first column and end.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for column in 0..=width {
        if strongest.get(column).is_some_and(|&ink| ink > DIGIT_INK) {
            start.get_or_insert(column);
        } else if let Some(first) = start.take() {
            runs.push((first, column));
        }
    }
    let distance = |run: &(usize, usize), at: f32| {
        let (first, end) = (run.0 as f32, run.1 as f32);
        if (first..end).contains(&at) {
            0.0
        } else {
            (at - first).abs().min((at - (end - 1.0)).abs())
        }
    };
    let nearest = |at: f32| {
        let after = runs.partition_point(|run| run.0 as f32 <= at);
        [after.checked_sub(1), (after < runs.len()).then_some(after)]
            .into_iter()
            .flatten()
            .min_by(|&a, &b| distance(&runs[a], at).total_cmp(&distance(&runs[b], at)))
    };
    // The characters (not spaces) whose place each run is nearest to.
    let mut claimants: Vec<Vec<usize>> = vec![Vec::new(); runs.len()];
    for (n, character) in characters.iter().enumerate() {
        if !character.is_whitespace()
            && let Some(run) = nearest(x(n))
        {
            claimants[run].push(n);
        }
    }
    // Each character's glyph: its run, or its part of a run that digits in a
    // row share (blurred figures of one number run together), cut between
    // each two at the column with the least ink between their places. A
    // glyph: first column, end, the centre of its ink, and whether it was cut.
    let column = |at: f32| (at.max(0.0) as usize).min(width);
    let cut = |(first, end): (usize, usize), members: &[usize]| {
        let mut cuts = vec![first];
        for pair in members.windows(2) {
            let low = (cuts[cuts.len() - 1] + 1).max(column(x(pair[0])));
            let high = (end - 1).min(column(x(pair[1])) + 1);
            cuts.push((low..high).min_by_key(|&at| (weight[at], at))?);
        }
        cuts.push(end);
        Some(cuts)
    };
    let centre = |first: usize, end: usize| {
        // An ink column weighs more than nothing: its ink passes DIGIT_INK.
        let (sum, moment) = (first..end).fold((0.0f64, 0.0f64), |(sum, moment), at| {
            let weight = f64::from(weight[at]);
            (sum + weight, moment + weight * (at as f64 + 0.5))
        });
        (moment / sum) as f32
    };
    let mut glyphs: Vec<(usize, usize, f32, bool)> = Vec::new();
    let mut glyph_of = vec![None; characters.len()];
    let mut claims: Vec<usize> = Vec::new();
    for (&run, members) in runs.iter().zip(&claimants) {
        let shared = members.len() > 1
            && members.windows(2).all(|pair| pair[1] == pair[0] + 1)
            && members.iter().all(|&n| characters[n].is_ascii_digit());
        match shared.then(|| cut(run, members)).flatten() {
            Some(cuts) => {
                for (part, &n) in members.iter().enumerate() {
                    let (first, end) = (cuts[part], cuts[part + 1]);
                    glyph_of[n] = Some(glyphs.len());
                    glyphs.push((first, end, centre(first, end), true));
                    claims.push(1);
                }
            }
            None => {
                for &n in members {
                    glyph_of[n] = Some(glyphs.len());
                }
                glyphs.push((run.0, run.1, centre(run.0, run.1), false));
                claims.push(members.len());
            }
        }
    }
    // Each measured pair: its position, the distance between the centres,
    // the gap between the glyphs, and whether a glyph was cut from a run.
    let measured: Vec<(usize, f32, usize, bool)> = pairs
        .into_iter()
        .filter_map(|n| {
            let (left, right) = (glyph_of[n]?, glyph_of[n + 1]?);
            (right == left + 1 && claims[left] == 1 && claims[right] == 1).then(|| {
                let ((_, end, from, left_cut), (next, _, to, right_cut)) =
                    (glyphs[left], glyphs[right]);
                (n, to - from, next - end, left_cut || right_cut)
            })
        })
        .collect();
    // The usual distance from glyphs that are runs of their own: a cut
    // glyph's centre is less sure.
    let mut distances: Vec<f32> = measured
        .iter()
        .filter(|pair| !pair.3)
        .map(|pair| pair.1)
        .collect();
    if distances.len() < 2 {
        return Vec::new();
    }
    crate::sort::by(&mut distances, f32::total_cmp);
    let usual = distances[(distances.len() - 1) / 2];
    measured
        .into_iter()
        .filter(|&(_, apart, gap, _)| {
            gap as f32 >= SPACE * height as f32 && apart >= DIGIT_SPACE * usual
        })
        .map(|(n, ..)| n)
        .collect()
}

/// Each column of a cut-out line on a `background` gray: its strongest ink
/// (how far its farthest pixel is from the background) and its weight, the
/// ink beyond half of [`DIGIT_INK`] summed down the column.
fn column_ink(image: &image::RgbImage, background: i16) -> (Vec<i16>, Vec<u32>) {
    let width = image.width() as usize;
    let mut strongest = vec![0i16; width];
    let mut weight = vec![0u32; width];
    for (column, _, pixel) in image.enumerate_pixels() {
        let ink = (luma(pixel) - background).abs();
        let column = column as usize;
        strongest[column] = strongest[column].max(ink);
        weight[column] += (ink - DIGIT_INK / 2).max(0) as u32;
    }
    (strongest, weight)
}

/// Two Chinese letters read with no space between them stand a space apart
/// when the centres of their ink stand at least this many times as far
/// apart as the line's letters usually do, and the gap between them is at
/// least [`SPACE`] line heights. Chinese letters share one width, a space
/// adds a quarter to a third of it. Measured on 1,680 rendered images of
/// table headers, label and number rows and prose (21 Chinese fonts, 13 to
/// 36 pixels, clean and blurred JPEG): on lines [`HAN_MIN_HEIGHT`] pixels
/// tall or more, the 44,600 pairs of letters set solid the rule measures
/// stood at most 1.17 times the usual distance apart, and four of five
/// pairs a space apart 1.25 times or more.
const HAN_SPACE: f32 = 1.25;

/// Lines shorter than this many pixels are left as read: in small blurred
/// print a letter's ink centre wanders by as much as a space adds (letters
/// set solid stood up to 1.27 times the usual distance apart).
const HAN_MIN_HEIGHT: u32 = 20;

/// A letter cut out wider than this many of the line's usual distances
/// between letters holds part of its neighbour (the cut went through a
/// letter whose parts stand apart, as in `川` or `计`), and its pairs are
/// left as read.
const HAN_WIDE: f32 = 1.05;

/// Positions `n` (from [`han_pairs`]) where the recognizer read the Chinese
/// letters `characters[n]` and `characters[n + 1]` together but `image`, a
/// cut-out line on a `background` gray, shows a space between them: the
/// words of a table header a space or two apart. `x(n)` is where the
/// recognizer places character `n`, in the image's columns.
///
/// Chinese letters are often drawn in parts with ink-free columns between
/// them, so the line is cut between each two characters rather than at its
/// ink-free runs: in the middle of the widest run of columns without ink
/// (any pixel [`DIGIT_INK`] from the background) between their places, or
/// at the column with the least ink when there is none. Each letter's ink is
/// the ink columns between its cuts; a pair is measured when both letters
/// have some, neither is wider than [`HAN_WIDE`] (an ill cut), and is spaced
/// as [`HAN_SPACE`] says against the median distance (the lower middle one)
/// between the centres of the line's measured pairs' ink. Lines with fewer
/// than two measured pairs, as a header of one-letter words, and lines under
/// [`HAN_MIN_HEIGHT`], stay as read.
pub(super) fn han_spaces(
    image: &image::RgbImage,
    background: i16,
    characters: &[char],
    x: impl Fn(usize) -> f32,
) -> Vec<usize> {
    let pairs = han_pairs(characters);
    let height = image.height();
    if pairs.len() < 2 || height < HAN_MIN_HEIGHT {
        return Vec::new();
    }
    let letters = cut_letters(image, background, characters, x);
    // Each measured pair: its position, the distance between the centres,
    // the gap between the letters' ink and the wider letter's width.
    let measured: Vec<(usize, f32, f32, f32)> = pairs
        .into_iter()
        .filter_map(|n| {
            let ((first, end), (next, next_end)) = (letters[n]?, letters[n + 1]?);
            Some((
                n,
                (next + next_end) as f32 / 2.0 - (first + end) as f32 / 2.0,
                next as f32 - end as f32,
                (end - first).max(next_end - next) as f32,
            ))
        })
        .collect();
    if measured.len() < 2 {
        return Vec::new();
    }
    let usual = lower_median(measured.iter().map(|pair| pair.1));
    if usual <= 0.0 {
        return Vec::new();
    }
    measured
        .into_iter()
        .filter(|&(_, apart, gap, wide)| {
            gap >= SPACE * height as f32 && apart >= HAN_SPACE * usual && wide <= HAN_WIDE * usual
        })
        .map(|(n, ..)| n)
        .collect()
}

/// The lower middle of `values` (the lower of the two middle ones of an
/// even count); 0 for none.
fn lower_median(values: impl Iterator<Item = f32>) -> f32 {
    let mut values: Vec<f32> = values.collect();
    crate::sort::by(&mut values, f32::total_cmp);
    values
        .get(values.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(0.0)
}

/// Each character's ink on a cut-out line on a `background` gray, its first
/// column and end, where `x(n)` places character `n`: the line is cut
/// between each two characters shown (not spaces) in the middle of the
/// widest run of columns without ink (any pixel [`DIGIT_INK`] from the
/// background) between their places, or at the column with the least ink
/// when there is none, and a character's ink is the ink columns between its
/// cuts. None for a space, or a character with no ink between its cuts.
fn cut_letters(
    image: &image::RgbImage,
    background: i16,
    characters: &[char],
    x: impl Fn(usize) -> f32,
) -> Vec<Option<(usize, usize)>> {
    let width = image.width() as usize;
    let (strongest, weight) = column_ink(image, background);
    let ink = |column: usize| strongest[column] > DIGIT_INK;
    let column = |at: f32| (at.max(0.0) as usize).min(width);
    // The characters shown (not spaces), and the cut after each but the last.
    let shown: Vec<usize> = (0..characters.len())
        .filter(|&n| !characters[n].is_whitespace())
        .collect();
    let cuts: Vec<f32> = shown
        .windows(2)
        .map(|pair| {
            let (low, high) = (column(x(pair[0])), (column(x(pair[1])) + 1).min(width));
            // The widest ink-free run, its first column and length.
            let (mut widest, mut start) = (None::<(usize, usize)>, None);
            for at in low..high {
                if ink(at) {
                    start = None;
                    continue;
                }
                let first = *start.get_or_insert(at);
                if widest.is_none_or(|(_, length)| at + 1 - first > length) {
                    widest = Some((first, at + 1 - first));
                }
            }
            match widest {
                Some((first, length)) => first as f32 + length as f32 / 2.0,
                None => (low..high)
                    .min_by_key(|&at| (weight[at], at))
                    .map_or((low + high) as f32 / 2.0, |at| at as f32 + 0.5),
            }
        })
        .collect();
    let mut letters = vec![None; characters.len()];
    for (k, &n) in shown.iter().enumerate() {
        let left = k.checked_sub(1).map_or(0, |k| cuts[k] as usize);
        let right = cuts
            .get(k)
            .map_or(width, |cut| (cut.ceil() as usize).min(width));
        letters[n] = (left..right)
            .find(|&at| ink(at))
            .zip((left..right).rfind(|&at| ink(at)))
            .map(|(first, last)| (first, last + 1));
    }
    letters
}

/// A Chinese or Japanese letter and a Latin letter stand a space apart when
/// the gap between their ink is wider than the line's usual gap between two
/// Chinese letters by at least this many of its usual distances between
/// Chinese letters' centres. Measured on 288 rendered images of Chinese
/// prose with Latin words set with and without spaces (eight fonts, two of
/// them a Latin font beside a Chinese one, 16 to 32 pixels, clean and
/// blurred JPEG): letters set solid passed the usual gap by at most 0.25 (a
/// hundredth of them by 0.13), letters a space apart by 0.06 or more (a
/// twentieth below 0.13).
const LETTER_JUNCTION: f32 = 0.1;

/// A Chinese or Japanese letter and a digit stand a space apart when the gap
/// between their ink passes the usual gap by this many usual distances, and
/// the distance between their centres passes half the usual distance
/// between Chinese letters' centres and half the digits' own by
/// [`DIGIT_JUNCTION`] of the former: a narrow figure, a `1`, has a wide gap
/// on each side yet stands no farther from its neighbours.
const DIGIT_GAP: f32 = 0.06;

/// See [`DIGIT_GAP`]. Read with the CLI on 3,180 rendered images of table
/// headers, label and number rows and Chinese prose with and without Latin
/// words (28 Chinese and Japanese fonts, five Latin ones beside them, 13 to
/// 40 pixels, clean and blurred JPEG; tuned on 528 of them), 26 digits set
/// solid beside a Chinese letter were split off where the ink-free gap alone
/// split 361, and 26 a space apart were kept with none where it kept 20.
const DIGIT_JUNCTION: f32 = 0.08;

/// A lone digit's distance from its neighbouring figures, in usual distances
/// between Chinese letters' centres: figures are a little over half as wide
/// as Chinese letters.
const LONE_DIGIT: f32 = 0.6;

/// Positions `n` (from [`junctions`]) where the recognizer read a Chinese or
/// Japanese letter and a Latin letter or digit together but `image`, a
/// cut-out line on a `background` gray, shows a space between them. `x(n)`
/// is where the recognizer places character `n`, in the image's columns.
///
/// Characters are cut apart as [`han_spaces`] cuts them. A Chinese letter's
/// and a Latin letter's own sides leave a gap between them as wide as two
/// Chinese letters leave, so a space is a gap wider than the line's usual one
/// between Chinese letters by [`LETTER_JUNCTION`] ([`DIGIT_GAP`] beside a
/// digit); and a digit is measured by its centre too ([`DIGIT_JUNCTION`]),
/// against the distance between the centres of its run's neighbouring
/// figures, or [`LONE_DIGIT`] when it has none. The usual gap and distance
/// are the medians (the lower middle ones)
/// over the line's neighbouring Chinese letters. A line with no two Chinese
/// letters side by side to measure, and a pair without ink of its own, have
/// a space where the widest ink-free run between them is [`SPACE`] line
/// heights ([`spaced`]).
pub(super) fn junction_spaces(
    image: &image::RgbImage,
    background: i16,
    characters: &[char],
    x: impl Fn(usize) -> f32,
) -> Vec<usize> {
    let junctions = junctions(characters);
    if junctions.is_empty() {
        return Vec::new();
    }
    let height = image.height() as f32;
    let letters = cut_letters(image, background, characters, &x);
    let centre = |(first, end): (usize, usize)| (first + end) as f32 / 2.0;
    // The line's neighbouring Chinese letters: the distances between their
    // centres and the gaps between their ink.
    let measured: Vec<(f32, f32)> = han_pairs(characters)
        .into_iter()
        .filter_map(|n| {
            let (left, right) = (letters[n]?, letters[n + 1]?);
            Some((centre(right) - centre(left), right.0 as f32 - left.1 as f32))
        })
        .collect();
    let pitch = lower_median(measured.iter().map(|pair| pair.0));
    let usual_gap = lower_median(measured.iter().map(|pair| pair.1)).max(0.0);
    let ink_free = |n: usize| spaced(image, [0.0, height], [x(n), x(n + 1)], background);
    let run = |c: char| c.is_ascii_alphanumeric() || ".,%-:/".contains(c);
    junctions
        .into_iter()
        .filter(|&n| {
            let (Some(left), Some(right)) = (letters[n], letters[n + 1]) else {
                return ink_free(n);
            };
            if measured.is_empty() || pitch <= 0.0 {
                return ink_free(n);
            }
            let excess = right.0 as f32 - left.1 as f32 - usual_gap;
            let (latin, letter, figure) = if characters[n].is_ascii() {
                (n, right, left)
            } else {
                (n + 1, left, right)
            };
            if !characters[latin].is_ascii_digit() {
                return excess >= LETTER_JUNCTION * pitch;
            }
            if excess < DIGIT_GAP * pitch {
                return false;
            }
            // The run of figures and their signs the digit stands in.
            let first = (0..latin)
                .rev()
                .find(|&k| !run(characters[k]))
                .map_or(0, |k| k + 1);
            let end = (latin..characters.len())
                .find(|&k| !run(characters[k]))
                .unwrap_or(characters.len());
            let figures = lower_median((first..end.saturating_sub(1)).filter_map(|k| {
                let both = characters[k].is_ascii_digit() && characters[k + 1].is_ascii_digit();
                both.then_some((letters[k]?, letters[k + 1]?))
                    .map(|(a, b)| centre(b) - centre(a))
            }));
            let figures = if figures > 0.0 {
                figures
            } else {
                LONE_DIGIT * pitch
            };
            let apart = (centre(figure) - centre(letter)).abs();
            apart - pitch / 2.0 - figures / 2.0 >= DIGIT_JUNCTION * pitch
        })
        .collect()
}

/// Whether two lines side by side on a row are written without a space: a
/// Chinese or Japanese letter or full-width punctuation meets the other line
/// with less than a sixth of a line height between their rectangles.
pub(super) fn touching(left: &Line, right: &Line) -> bool {
    const TOUCH: f32 = 0.15;
    let wide = |c: Option<char>| {
        c.is_some_and(|c| {
            cjk::han(c)
                || cjk::kana(c)
                || matches!(c, '\u{3000}'..='\u{303f}' | '\u{ff00}'..='\u{ffef}')
        })
    };
    let h = (left.bounds[3] - left.bounds[1]).min(right.bounds[3] - right.bounds[1]);
    right.bounds[0] - left.bounds[2] < TOUCH * h
        && (wide(left.text.trim_end().chars().last())
            || wide(right.text.trim_start().chars().next()))
}

/// Letter-spaced runs this many characters long or longer are closed up.
const MIN_SPACED: usize = 4;

/// `text` with its letter-spaced runs closed up: letters set one by one
/// with a space between each (`M a r k i t a i`, `3 . 5 0`), as a
/// letter-spaced line or Latin set in full-width cells reads. A run is at
/// least [`MIN_SPACED`] single letters or digits (the first may be two: the
/// recognizer sometimes keeps the first pair together), each followed by a
/// single space, between characters that are not Latin letters or digits;
/// its single units hold a letter, or it is a decimal number (`3 . 5 0`).
/// A row of digits alone (`1 2 3 4 5`, a scale) is kept.
pub(super) fn closed_up(text: &str) -> Option<String> {
    let characters: Vec<char> = text.chars().collect();
    let alnum = |i: usize| characters.get(i).is_some_and(char::is_ascii_alphanumeric);
    let mut out = String::with_capacity(text.len());
    let mut changed = false;
    let mut i = 0;
    while i < characters.len() {
        if alnum(i) && (i == 0 || !alnum(i - 1)) {
            // The first unit, one or two letters, then single letters, digits
            // or a decimal point, each after one space.
            let mut end = i + 1 + usize::from(alnum(i + 1));
            let mut units = 1;
            // Whether the single units hold a letter or a decimal point.
            let mut letters = false;
            let mut point = false;
            while characters.get(end) == Some(&' ') {
                let next = end + 1;
                let unit = match characters.get(next) {
                    Some(c) if c.is_ascii_alphanumeric() => true,
                    // A decimal point between two digits.
                    Some('.' | ',') => {
                        characters[end - 1].is_ascii_digit()
                            && characters.get(next + 1) == Some(&' ')
                            && characters.get(next + 2).is_some_and(char::is_ascii_digit)
                    }
                    _ => false,
                };
                if !unit || alnum(next + 1) {
                    break;
                }
                letters |= characters[next].is_ascii_alphabetic();
                point |= !characters[next].is_ascii_alphanumeric();
                units += 1;
                end = next + 1;
            }
            if units >= MIN_SPACED && (letters || point) && !alnum(end) {
                out.extend(characters[i..end].iter().filter(|c| **c != ' '));
                changed = true;
                i = end;
                continue;
            }
        }
        out.push(characters[i]);
        i += 1;
    }
    changed.then_some(out)
}

/// The rules a table drawn with vertical lines reads as.
fn rule(character: char) -> bool {
    matches!(character, '|' | '｜' | '丨' | '│')
}

/// A row's text as table cells: split at the vertical rules the recognizer
/// read (`|`) and at wide gaps (two spaces or more), without the empty
/// cells of rules at either end. Also whether a rule divided it.
pub(super) fn table_cells(text: &str) -> (Vec<String>, bool) {
    let ruled = text.contains(rule);
    let mut cells: Vec<String> = text
        .split(rule)
        .flat_map(|part| part.split("  "))
        .map(str::trim)
        .map(str::to_owned)
        .collect();
    // Two spaces or more leave empty pieces between the words.
    if !ruled {
        cells.retain(|cell| !cell.is_empty());
    }
    while cells.first().is_some_and(String::is_empty) {
        cells.remove(0);
    }
    while cells.last().is_some_and(String::is_empty) {
        cells.pop();
    }
    (cells, ruled)
}

/// A table cell that holds a number with spaces in it (`1 2`, `1. 20`,
/// `3 . 5 0`) without them. Numbers a space apart, each of two characters
/// or more from its first figure to its last (`6.20 6,273.11`, the cells of
/// two columns read as one line), keep their spaces.
pub(super) fn number_cell(cell: &str) -> Option<String> {
    let squeezed: String = cell.split_whitespace().collect();
    let whole = |piece: &str| {
        let figures = piece.trim_start_matches(['-', '+', '$', '€', '£', '¥', '(']);
        piece.chars().count() >= 2
            && figures.starts_with(|c: char| c.is_ascii_digit())
            && piece.ends_with(|c: char| c.is_ascii_digit() || c == '%' || c == ')')
    };
    (squeezed.len() != cell.len()
        && squeezed.chars().any(|c| c.is_ascii_digit())
        && squeezed
            .chars()
            .all(|c| c.is_ascii_digit() || numeric_mark(c))
        && !cell.split_whitespace().all(whole))
    .then_some(squeezed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letter_spaced_runs_close_up_and_words_and_digit_rows_stay() {
        for (spaced, closed) in [
            (
                "M a r k i t a i 将各种文档格式转换为 Ma r k d o w n，并可选地",
                "Markitai 将各种文档格式转换为 Markdown，并可选地",
            ),
            ("单价 3 . 5 0 元", "单价 3.50 元"),
            ("T I T L E", "TITLE"),
        ] {
            assert_eq!(closed_up(spaced).as_deref(), Some(closed), "{spaced}");
        }
        for kept in [
            "I am a test of words",
            "Rate it 1 2 3 4 5",
            "x y z",
            "a b c dog",
            "1 2",
            "한국어 문장도 포함합니다",
        ] {
            assert_eq!(closed_up(kept), None, "{kept}");
        }
    }

    #[test]
    fn rows_split_into_cells_at_rules_and_wide_gaps() {
        assert_eq!(
            table_cells("苹果 | 1 2 | 3.50"),
            (vec!["苹果".into(), "1 2".into(), "3.50".into()], true)
        );
        assert_eq!(
            table_cells("项目  数量   单价"),
            (vec!["项目".into(), "数量".into(), "单价".into()], false)
        );
        assert_eq!(
            table_cells("| a | | c |"),
            (vec!["a".into(), String::new(), "c".into()], true)
        );
        assert_eq!(table_cells("one plain sentence").0.len(), 1);
        assert_eq!(number_cell("1 2").as_deref(), Some("12"));
        assert_eq!(number_cell("1. 20").as_deref(), Some("1.20"));
        assert_eq!(number_cell("3 . 5 0").as_deref(), Some("3.50"));
        assert_eq!(number_cell("12 3").as_deref(), Some("123"));
        assert_eq!(number_cell("3.50"), None);
        assert_eq!(number_cell("香 蕉"), None);
        // Two numbers of two columns read as one line keep their space.
        for numbers in ["6.20 6,273.11", "-0 741.81", "12 3.50", "5% (1,200)"] {
            assert_eq!(number_cell(numbers), None, "{numbers}");
        }
    }

    fn line(text: &str, bounds: [f32; 4]) -> Line {
        Line {
            text: text.into(),
            confidence: 1.0,
            bounds,
            direction: [1.0, 0.0],
            marks: Default::default(),
        }
    }

    #[test]
    fn the_direction_most_letters_run_turns_the_page() {
        let turned = |direction: [f32; 2]| {
            let mut l = line("The quick brown fox", [0.0, 0.0, 10.0, 100.0]);
            l.direction = direction;
            l
        };
        assert_eq!(orientation(&[turned([1.0, 0.02])]), Turn::Upright);
        assert_eq!(orientation(&[turned([0.0, 800.0])]), Turn::Clockwise);
        assert_eq!(orientation(&[turned([0.0, -800.0])]), Turn::Counter);
        assert_eq!(orientation(&[turned([-800.0, 3.0])]), Turn::Half);
        // A sideways label on an upright page does not turn it.
        let page = [
            line("An upright line of text here", [0.0; 4]),
            turned([0.0, 9.0]),
        ];
        assert_eq!(orientation(&page), Turn::Upright);
        // Two letters do not vote, and no corners means upright.
        assert_eq!(
            orientation(&[{
                let mut l = turned([0.0, 9.0]);
                l.text = "ab".into();
                l
            }]),
            Turn::Upright
        );
        assert_eq!(orientation(&[turned([0.0, 0.0])]), Turn::Upright);
        assert_eq!(orientation(&[]), Turn::Upright);
    }

    #[test]
    fn turned_lines_take_upright_rectangles_and_order() {
        // A 500 by 1200 page turned a quarter clockwise: the first line is
        // the rightmost column, running down.
        let mut lines = vec![
            line("first", [420., 40., 458., 840.]),
            line("second", [358., 40., 398., 1064.]),
        ];
        assert_eq!(upright(&mut lines, Turn::Clockwise, 500, 1200), (1200, 500));
        assert_eq!(lines[0].bounds, [40., 42., 840., 80.]);
        assert!(lines[0].bounds[1] < lines[1].bounds[1]);
        let mut lines = vec![
            line("first", [42., 360., 80., 1160.]),
            line("second", [102., 136., 142., 1160.]),
        ];
        assert_eq!(upright(&mut lines, Turn::Counter, 500, 1200), (1200, 500));
        assert!(lines[0].bounds[1] < lines[1].bounds[1]);
        assert_eq!(lines[0].bounds, [40., 42., 840., 80.]);
        let mut lines = vec![line("first", [360., 420., 1160., 458.])];
        assert_eq!(upright(&mut lines, Turn::Half, 1200, 500), (1200, 500));
        assert_eq!(lines[0].bounds, [40., 42., 840., 80.]);
        assert_eq!(lines[0].direction, [1.0, 0.0]);
    }

    #[test]
    fn zeros_read_as_letters_are_mended_only_between_digits() {
        assert_eq!(digits("2ø26").as_deref(), Some("2026"));
        assert_eq!(
            digits("Invoice 2O26-0042 on 1o.5").as_deref(),
            Some("Invoice 2026-0042 on 1o.5")
        );
        assert_eq!(
            digits("1,2O0,000 and 2øø6 but 1,2OO,000").as_deref(),
            Some("1,200,000 and 2006 but 1,2OO,000")
        );
        assert_eq!(digits("于2O26年发布").as_deref(), Some("于2026年发布"));
        // Words, formulas, the ends of numbers and lone digits are kept.
        for kept in [
            "Fe2O3", "CO2", "O2", "10O", "O10", "2O", "No. 2O", "Room 1O", "Zoo 2000", "2oo4x",
            "x2O26", "ø",
        ] {
            assert_eq!(digits(kept), None, "{kept}");
        }
        assert_eq!(digits("Total 12.50"), None);
    }

    fn cells(rows: &[&[(&str, f32)]]) -> Vec<Line> {
        let mut lines = Vec::new();
        for (r, row) in rows.iter().enumerate() {
            let top = 30.0 + 90.0 * r as f32;
            for (text, left) in row.iter() {
                lines.push(line(
                    text,
                    [
                        *left,
                        top,
                        left + 20.0 * text.chars().count() as f32,
                        top + 36.0,
                    ],
                ));
            }
        }
        lines
    }

    #[test]
    fn a_cell_missing_from_a_table_row_is_a_region_to_read_again() {
        let table = cells(&[
            &[("Item", 60.), ("Qty", 360.), ("Price", 660.)],
            &[("Apple", 60.), ("1.20", 660.)],
            &[("Banana", 60.), ("12", 360.), ("0.50", 660.)],
            &[("Cherry", 60.), ("7", 360.), ("12.00", 660.)],
        ]);
        let found = rereads(&table, 1000, 420, true).rereads;
        assert_eq!(found.len(), 1, "{found:?}");
        let Target::Cell {
            span,
            band,
            max_chars,
            numeric,
        } = found[0].target
        else {
            panic!()
        };
        // The Qty column, padded by half a row height, in Apple's row.
        assert_eq!(span, [360.0 - 18.0, 420.0 + 18.0]);
        assert_eq!(band, [120.0, 156.0]);
        assert_eq!(max_chars, 6);
        assert!(numeric);
        let [l, t, r, b] = found[0].region;
        assert!(l > 220.0 && l < 360.0 && r > 420.0 && r < 660.0, "{l} {r}");
        assert!(
            (66.0..120.0).contains(&t) && b > 156.0 && b <= 210.0,
            "{t} {b}"
        );
        // It is read with Apple, the previous cell, for context.
        assert_eq!(found[0].context, 60.0 - 0.3 * 36.0);
        assert_eq!(found[0].attempts()[0].0, [found[0].context, t, r, b]);
        assert_eq!(found[0].attempts()[2], (found[0].region, 2.0, 36.0));
        // A cell of numbers read as a Cyrillic look-alike is its digit.
        let mut cyrillic = table;
        cyrillic[7].text = "З".into();
        let found = rereads(&cyrillic, 1000, 420, true);
        assert_eq!(found.digits, [(7, '3')]);
        assert!(rereads(&cyrillic, 1000, 420, false).digits.is_empty());
        // A complete table, prose, two columns of prose and a two-row table
        // have nothing to read again.
        let complete = cells(&[
            &[("Item", 60.), ("Qty", 360.)],
            &[("Apple", 60.), ("3", 360.)],
            &[("Banana", 60.), ("12", 360.)],
        ]);
        assert!(rereads(&complete, 1000, 420, true).rereads.is_empty());
        let prose = cells(&[
            &[("The quick brown fox jumps over", 40.)],
            &[("the lazy dog.", 40.)],
            &[("Again.", 40.)],
        ]);
        assert!(rereads(&prose, 1000, 420, true).rereads.is_empty());
        let columns = cells(&[
            &[
                ("Left column first line of text", 40.),
                ("Right column first line of text", 740.),
            ],
            &[
                ("Left column second line of text", 40.),
                ("Right column second line", 740.),
            ],
            &[("Left column third line of text", 40.)],
        ]);
        assert!(rereads(&columns, 1400, 420, true).rereads.is_empty());
        let two = cells(&[&[("Item", 60.), ("Qty", 360.)], &[("Apple", 60.)]]);
        assert!(rereads(&two, 1000, 420, true).rereads.is_empty());
        // A cell spanning two columns is not a row with a missing cell.
        let spanning = cells(&[
            &[("Item", 60.), ("Qty", 360.), ("Price", 660.)],
            &[("Banana", 60.), ("12", 360.), ("0.50", 660.)],
            &[("Cherry", 60.), ("7", 360.), ("12.00", 660.)],
            &[("A note across the two columns", 60.), ("1.20", 660.)],
        ]);
        assert!(rereads(&spanning, 1000, 420, true).rereads.is_empty());
    }

    #[test]
    fn numbers_holding_letters_of_other_alphabets_are_read_again_and_mended() {
        let lines = [
            line("Q2 202łąłą", [58., 237., 180., 275.]),
            line("1,200,00łł", [412., 144., 560., 188.]),
            line("Zürich 2026 and 3ème", [0., 300., 400., 340.]),
        ];
        let found = rereads(&lines, 1160, 420, true).rereads;
        assert_eq!(
            found.iter().map(|r| &r.target).collect::<Vec<_>>(),
            [
                &Target::Mend { index: 0 },
                &Target::Mend { index: 1 },
                &Target::Mend { index: 2 }
            ]
        );
        // Only for Latin readings.
        assert!(rereads(&lines, 1160, 420, false).rereads.is_empty());
        assert_eq!(mend("Q2 202łąłą", "Q2 2026").as_deref(), Some("Q2 2026"));
        assert_eq!(
            mend("1,200,00łł", "1,200,000").as_deref(),
            Some("1,200,000")
        );
        // Anything but digits in place of the garbage, a changed clean part,
        // another token count, or garbage again: no change.
        for reading in [
            "Q2 202x",
            "Q3 2026",
            "Q2 2026 extra",
            "Q22026",
            "Q2 202ąą",
            "Q2 2026000000",
        ] {
            assert_eq!(mend("Q2 202łąłą", reading), None, "{reading}");
        }
        assert_eq!(mend("3ème", "3ème"), None);
    }

    #[test]
    fn a_missing_cell_keeps_what_its_region_reads_inside_the_cell() {
        let found = vec![
            line("3", [360., 149., 380., 177.]),
            line("1.20", [660., 148., 732., 180.]),
        ];
        let cell = cell(found, [342., 438.], [140., 186.], 6, true, true).unwrap();
        assert_eq!(
            (cell.text.as_str(), cell.bounds),
            ("3", [360., 149., 380., 177.])
        );
        let read = |text: &str, latin: bool| {
            super::cell(
                vec![line(text, [360., 149., 380., 177.])],
                [342., 438.],
                [140., 186.],
                6,
                true,
                latin,
            )
        };
        // A rule read as a bar or a letter, text too long for the column,
        // and letters of other alphabets in a Latin reading are not cells.
        for junk in ["|", "l", "too long for it", "IÒ"] {
            assert!(read(junk, true).is_none(), "{junk}");
        }
        assert_eq!(read("IÒ", false).unwrap().text, "IÒ");
        assert!(super::cell(Vec::new(), [342., 438.], [140., 186.], 6, true, true).is_none());
        // In a column of numbers, a Cyrillic look-alike is its digit.
        assert_eq!(read("З", true).unwrap().text, "3");
        assert_eq!(read("З", false).unwrap().text, "З");
        // A lone Cyrillic letter in text (Ukrainian `з`) is not read again.
        assert!(!garbled("з") && !garbled("Зима") && !garbled("Ü") && garbled("15\u{fffe}41"));
        // Latin letters beside the garbage, and a bullet for a point, are mended too.
        assert!(garbled("72.7zął") && garbled("183•33") && !garbled("72.7z"));
        // A zero read as ø is mended without reading it again.
        assert!(!garbled("2ø26"));
        assert_eq!(mend("72.7zął", "72.72").as_deref(), Some("72.72"));
        assert_eq!(mend("183•33", "183.33").as_deref(), Some("183.33"));
        assert_eq!(mend("72.7zął", "72.7z2"), None);
    }

    #[test]
    fn indented_rows_in_a_fixed_pitch_with_code_punctuation_are_code() {
        // `img_code.png`, as Vision boxes it: Menlo at 34 pixels.
        let rows = [
            ("def fib(n):", [40., 44., 258., 76.]),
            ("a, b = 0, 1", [120., 104., 342., 144.]),
            ("for _ in range(n):", [122., 159., 484., 203.]),
            ("yield a", [197., 216., 343., 255.]),
            ("a, b = b, a + b", [198., 275., 502., 322.]),
        ];
        let lines: Vec<Line> = rows.iter().map(|(t, b)| line(t, *b)).collect();
        let ranges: Vec<_> = (0..lines.len()).map(|i| i..i + 1).collect();
        let found = code(&lines, &ranges).unwrap();
        assert_eq!(found.rows, 0..5);
        let indents: Vec<usize> = lines.iter().map(|l| indent(&found, l.bounds[0])).collect();
        assert_eq!(indents, [0, 4, 4, 8, 8]);
        // Without indentation (a receipt) or code punctuation, or in a
        // proportional font, rows are text.
        let flat: Vec<Line> = rows
            .iter()
            .map(|(t, b)| line(t, [40., b[1], 40. + b[2] - b[0], b[3]]))
            .collect();
        assert!(code(&flat, &ranges).is_none());
        let words: Vec<Line> = ["Some text here", "more of it", "and yet more", "words"]
            .iter()
            .enumerate()
            .map(|(i, t)| {
                line(
                    t,
                    [
                        40. + 80. * (i % 2) as f32,
                        40. * i as f32,
                        40. + 80. * (i % 2) as f32 + 20. * t.len() as f32,
                        40. * i as f32 + 30.,
                    ],
                )
            })
            .collect();
        assert!(code(&words, &ranges[..4]).is_none());
        let proportional: Vec<Line> = rows
            .iter()
            .enumerate()
            .map(|(i, (t, b))| {
                line(
                    t,
                    [
                        b[0],
                        b[1],
                        b[0] + (b[2] - b[0]) * (1.0 + 0.3 * (i % 2) as f32),
                        b[3],
                    ],
                )
            })
            .collect();
        assert!(code(&proportional, &ranges).is_none());
        // A title above the code in another font is left out of the run.
        let mut titled = vec![line("Fibonacci numbers", [40., 0., 300., 30.])];
        titled.extend(lines.iter().map(|l| {
            line(
                &l.text,
                [
                    l.bounds[0],
                    l.bounds[1] + 40.,
                    l.bounds[2],
                    l.bounds[3] + 40.,
                ],
            )
        }));
        let ranges: Vec<_> = (0..titled.len()).map(|i| i..i + 1).collect();
        assert_eq!(code(&titled, &ranges).unwrap().rows, 1..6);
    }

    #[test]
    fn an_ink_free_gap_of_a_fifth_of_a_line_between_chinese_and_latin_is_a_space() {
        // Two glyph outlines, 30 pixels wide, on a 40-pixel line, `gap`
        // pixels apart, in `ink` on `paper`.
        let page = |gap: u32, ink: u8, paper: u8| {
            image::RgbImage::from_fn(140, 40, |x, y| {
                let outline = |left: u32| {
                    (left..left + 30).contains(&x)
                        && (x == left || x == left + 29 || y == 5 || y == 34 || y == 20)
                };
                image::Rgb(
                    [if outline(10) || outline(40 + gap) {
                        ink
                    } else {
                        paper
                    }; 3],
                )
            })
        };
        for (ink, paper) in [(0, 255), (230, 20)] {
            let measure = |gap: u32| {
                let image = page(gap, ink, paper);
                let background = background(&image, [0., 0., 140., 40.]).unwrap();
                assert_eq!(background, i16::from(paper));
                spaced(&image, [0., 40.], [25., 55. + gap as f32], background)
            };
            // 0.22 line heights is 8.8 pixels.
            assert!(!measure(2) && !measure(8));
            assert!(measure(9) && measure(16));
        }
        let characters: Vec<char> = "在CLI与 PDF中，1.3版".chars().collect();
        assert_eq!(junctions(&characters), [0, 3, 8, 13]);
        assert!(junctions(&"Plain English".chars().collect::<Vec<_>>()).is_empty());
    }

    #[test]
    fn chinese_meeting_a_line_with_no_gap_is_written_without_a_space() {
        let a = line("在", [100., 0., 150., 50.]);
        let b = line("CLI", [153., 0., 230., 50.]);
        assert!(touching(&a, &b));
        assert!(!touching(
            &line("在", [100., 0., 150., 50.]),
            &line("CLI", [170., 0., 230., 50.])
        ));
        assert!(!touching(
            &line("Hello", [100., 0., 150., 50.]),
            &line("World", [152., 0., 230., 50.])
        ));
    }

    /// A 40-pixel line on white with a block of ink from row 8 to row 32 at
    /// each `(left, width)`: digits on a pitch of 16 pixels take 12, a `1`
    /// takes 2 in the middle of its 16.
    fn glyphs(blocks: &[(u32, u32)], width: u32) -> image::RgbImage {
        image::RgbImage::from_fn(width, 40, |x, y| {
            let ink = (8..32).contains(&y)
                && blocks
                    .iter()
                    .any(|&(left, wide)| (left..left + wide).contains(&x));
            image::Rgb([if ink { 0 } else { 255 }; 3])
        })
    }

    /// `digit_spaces` for `text`, each character placed at `places` (the
    /// recognizer's places run a little right of the glyphs).
    fn digit_spaces_of(image: &image::RgbImage, text: &str, places: &[f32]) -> Vec<usize> {
        let characters: Vec<char> = text.chars().collect();
        digit_spaces(image, 255, &characters, |n| places[n])
    }

    #[test]
    fn digits_a_space_apart_are_told_from_a_narrow_one() {
        // `30 100`, read as `30100`: the 0 and 1 stand 24 pixels apart, one
        // and a half figures; the 1 and 0 after it have a gap of 9 pixels,
        // over 0.22 line heights, but stand one figure apart.
        let line = glyphs(&[(2, 12), (18, 12), (47, 2), (58, 12), (74, 12)], 100);
        assert_eq!(
            digit_spaces_of(&line, "30100", &[12., 28., 52., 68., 84.]),
            [1]
        );
        // `1001`: gaps of 9 pixels beside each 1, every figure one apart.
        let line = glyphs(&[(7, 2), (18, 12), (34, 12), (55, 2)], 70);
        assert!(digit_spaces_of(&line, "1001", &[12., 28., 44., 60.]).is_empty());
        // `1 2`, read as `12`: one pair tells nothing of the line's figures.
        let line = glyphs(&[(7, 2), (34, 12)], 60);
        assert!(digit_spaces_of(&line, "12", &[12., 44.]).is_empty());
        assert_eq!(
            digit_pairs(&"x 12a345".chars().collect::<Vec<_>>()),
            [2, 5, 6]
        );
    }

    #[test]
    fn digits_whose_ink_touches_or_holds_a_dropped_point_stay_as_read() {
        // `45 6`, the 4 and 5 touching: one run for two digits.
        let line = glyphs(&[(2, 14), (16, 12), (42, 12)], 60);
        assert!(digit_spaces_of(&line, "456", &[12., 26., 52.]).is_empty());
        // `3.50 12`, read as `35012`: the point between 3 and 5 is ink of its
        // own, so only the space after 50 is put back.
        let line = glyphs(
            &[(2, 12), (21, 3), (26, 12), (42, 12), (71, 2), (82, 12)],
            100,
        );
        assert_eq!(
            digit_spaces_of(&line, "35012", &[12., 36., 52., 76., 92.]),
            [2]
        );
    }

    #[test]
    fn figures_of_one_number_that_run_together_are_cut_apart_and_measured() {
        // `506 78`, read as `50678`, the 7 and 8 joined by a faint bridge
        // (rows 18 to 22 of columns 70 and 71): the run is cut at the bridge,
        // and the 7 stands 24 pixels from the 6, one and a half figures.
        let mut line = glyphs(&[(2, 12), (18, 12), (34, 12), (58, 12), (72, 12)], 100);
        for x in 70..72 {
            for y in 18..22 {
                line.put_pixel(x, y, image::Rgb([0; 3]));
            }
        }
        let places = [8., 24., 40., 64., 78.];
        assert_eq!(digit_spaces_of(&line, "50678", &places), [2]);
        // The same run claimed by a point and a digit is not cut: `50.78`.
        assert!(digit_spaces_of(&line, "506.8", &places).is_empty());
        // Only pairs of whole runs set the usual distance: in `5 0678` the
        // one whole pair cannot.
        let line = glyphs(&[(2, 12), (26, 12), (40, 14), (54, 12), (66, 12)], 90);
        assert!(digit_spaces_of(&line, "50678", &[8., 32., 47., 60., 72.]).is_empty());
    }

    /// A line `height` pixels tall on white with a letter's ink from a sixth
    /// to five sixths of it at each `(left, width)`.
    fn letters(blocks: &[(u32, u32)], width: u32, height: u32) -> image::RgbImage {
        image::RgbImage::from_fn(width, height, |x, y| {
            let ink = (height / 6..height * 5 / 6).contains(&y)
                && blocks
                    .iter()
                    .any(|&(left, wide)| (left..left + wide).contains(&x));
            image::Rgb([if ink { 0 } else { 255 }; 3])
        })
    }

    fn han_spaces_of(image: &image::RgbImage, text: &str, places: &[f32]) -> Vec<usize> {
        let characters: Vec<char> = text.chars().collect();
        han_spaces(image, 255, &characters, |n| places[n])
    }

    #[test]
    fn chinese_words_a_space_apart_are_told_from_letters_set_solid() {
        // `项目 数量 单价` read as one word: letters 30 pixels wide on a
        // pitch of 32, a space adding 10.
        let lefts = [2, 34, 76, 108, 150, 182];
        let blocks: Vec<(u32, u32)> = lefts.iter().map(|&left| (left, 30)).collect();
        let places: Vec<f32> = lefts.iter().map(|&left| left as f32 + 15.0).collect();
        let header = letters(&blocks, 220, 40);
        assert_eq!(han_spaces_of(&header, "项目数量单价", &places), [1, 3]);
        // Small print is left as read.
        let small = letters(&blocks, 220, HAN_MIN_HEIGHT - 1);
        assert!(han_spaces_of(&small, "项目数量单价", &places).is_empty());
        // Set solid, with letters drawn in three strokes (`川`) whose gaps
        // are wider than those between letters: the cuts go through the
        // strokes, yet no pair is spaced.
        let mut blocks = Vec::new();
        for left in [2, 66, 130] {
            blocks.extend([(left, 4), (left + 13, 4), (left + 26, 4)]);
        }
        blocks.extend([(34, 30), (98, 30), (162, 30)]);
        let places = [17., 49., 81., 113., 145., 177.];
        let prose = letters(&blocks, 200, 40);
        assert!(han_spaces_of(&prose, "川的川字川有", &places).is_empty());
        assert_eq!(
            han_pairs(&"共有 12个项目".chars().collect::<Vec<_>>()),
            [0, 5, 6]
        );
    }

    fn junction_spaces_of(image: &image::RgbImage, text: &str, places: &[f32]) -> Vec<usize> {
        let characters: Vec<char> = text.chars().collect();
        junction_spaces(image, 255, &characters, |n| places[n])
    }

    #[test]
    fn a_narrow_figure_in_prose_set_solid_keeps_no_space_and_a_spaced_number_gets_one() {
        // Chinese letters 30 pixels wide on a pitch of 32 (a gap of 2) on a
        // 40-pixel line; figures on a pitch of 18, a `1` 4 pixels wide and a
        // `2` 12, each in the middle of its place.
        let han = |left: u32| (left, 30);
        let figure = |centre: u32, wide: u32| (centre - wide / 2, wide);
        // `共有12个项目`, set solid: the `1` leaves 9 pixels on its left, more
        // than SPACE line heights (8.8) of ink-free columns, yet its centre
        // stands where a figure's does.
        let blocks = [
            han(2),
            han(34),
            figure(75, 4),
            figure(93, 12),
            han(104),
            han(136),
            han(168),
        ];
        let places = [17., 49., 75., 93., 119., 151., 183.];
        let prose = letters(&blocks, 210, 40);
        assert!(junction_spaces_of(&prose, "共有12个项目", &places).is_empty());
        // `收入 12 万元`: a space (12 pixels) on each side of the number.
        let blocks = [
            han(2),
            han(34),
            figure(87, 4),
            figure(105, 12),
            han(128),
            han(160),
        ];
        let places = [17., 49., 87., 105., 143., 175.];
        let spaced = letters(&blocks, 200, 40);
        assert_eq!(junction_spaces_of(&spaced, "收入12万元", &places), [1, 3]);
        // A Latin letter is measured by the gap alone: `用 OCR` and `用OCR`.
        let word = |gap: u32| {
            let latin = |left: u32| (left + gap, 14);
            let blocks = [han(2), han(34), latin(66), latin(82), latin(98)];
            let at = |centre: f32| centre + gap as f32;
            let places = [17., 49., at(73.), at(89.), at(105.)];
            junction_spaces_of(&letters(&blocks, 140, 40), "使用OCR", &places)
        };
        assert_eq!(word(12), [1]);
        assert!(word(0).is_empty());
        // With no two Chinese letters to measure, the ink-free run decides.
        let blocks = [han(2), figure(48, 4)];
        let lone = letters(&blocks, 60, 40);
        assert_eq!(junction_spaces_of(&lone, "共1", &[17., 48.]), [0]);
    }
}
