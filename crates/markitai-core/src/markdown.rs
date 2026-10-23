//! Deterministic cleanup applied to normal output; pure output bypasses it.

use regex::Regex;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

fn pattern(source: &str) -> Regex {
    Regex::new(source).expect("static Markdown pattern")
}

/// A link whose text a line break splits, with the `!` of an image when there
/// is one (the first group): image syntax is never repaired, since the repair
/// keeps only the first line of the text. The text holds no bracket, so the
/// match opens at the link's own `[`, not at a stray one before it.
static BROKEN_LINK: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"(!?)\[([^\[\]]*?)\n+([^\[\]]*?)\]\(([^)]+)\)"));
static PLACEHOLDER_LINE: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"(?m)^__MARKITAI_[A-Z_]+_?\d*__\s*$"));
static PLACEHOLDER_IMAGE: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"!\[[^\]]*\]\(__MARKITAI_[A-Z_]+_?\d*__\)\s*\n?"));
static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| pattern(r"__MARKITAI_[A-Z_]+_?\d*__"));
static DOUBLE_ALT: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"!\[[^\]]*\](!\[[^\]]*\]\([^)]+\))"));
static EMPTY_IMAGE: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"!\[[^\]]*\]\((?:\.markitai/assets/)?\)\s*\n?"));
static EXTRA_PAREN: LazyLock<Regex> = LazyLock::new(|| pattern(r"(!\[[^\]]*\]\([^)]+\))\)+"));
static PAGE: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"<!-- (?:Page|Slide) (?:number: ?)?\d+ -->"));
static SLIDE: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"^<!--\s*Slide\s+(number:\s*)?\d+\s*-->"));
static BLANKS: LazyLock<Regex> = LazyLock::new(|| pattern(r"\n{3,}"));
static DELIMITER_ROW: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$"));

/// Prose without the delimiter rows (`| --- | --- |`) a table already has. A
/// pipe table has one, right after its header; a model rewriting a table
/// sometimes repeats it, and the repeat would show as a row of dashes.
fn without_stray_delimiters(text: &str) -> String {
    if !text.contains("---") || !text.contains('|') {
        return text.to_owned();
    }
    // The current line's place in its run of table lines.
    let mut row = 0usize;
    let mut output = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let content = line.trim_end_matches(['\n', '\r']);
        if !content.trim_start().starts_with('|') {
            row = 0;
        } else {
            row += 1;
            if row > 2 && DELIMITER_ROW.is_match(content) {
                continue;
            }
        }
        output.push_str(line);
    }
    output
}

/// Prose lines without the hard breaks content removed after rendering
/// leaves behind (a filtered image between two breaks, or before one): a line
/// that is only a hard break's `\`, and a hard break that ends a paragraph,
/// which Markdown would show as a backslash. A line indented four spaces or
/// more may be code and is left as it is; fenced code is not part of prose.
fn without_dangling_breaks(mut lines: Vec<&str>) -> Vec<&str> {
    let prose = |line: &str| line.len() - line.trim_start_matches(' ').len() < 4;
    lines.retain(|line| !(prose(line) && line.trim_start() == "\\"));
    for index in 0..lines.len() {
        let line = lines[index];
        if prose(line)
            && ends_escaped(line)
            && lines.get(index + 1).is_none_or(|next| next.is_empty())
        {
            lines[index] = line[..line.len() - 1].trim_end();
        }
    }
    lines
}

/// Lines without the blanks at their end, except a hard break written as two
/// spaces (or more, written as two) on a line the next one continues: not the
/// last line of a paragraph, of the document, or before a heading or a fenced
/// block. A line of blanks only is empty, as Markdown reads it, and a line
/// ending in a backslash hard break or an escape gains no spaces.
fn trimmed_lines(text: &str) -> Vec<&str> {
    let lines: Vec<&str> = text.split('\n').collect();
    let continues = |next: &str| {
        let next = next.trim_start();
        !next.is_empty()
            && !next.starts_with(|ch| ('\u{fdd0}'..='\u{fdef}').contains(&ch))
            && !next.starts_with('#')
            && !SLIDE.is_match(next)
    };
    lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let content = line.trim_end();
            let line = line.strip_suffix('\r').unwrap_or(line);
            // What stands before the spaces (a no-break space is text) stays.
            let before = line.trim_end_matches(' ');
            let kept = !content.is_empty()
                && line.len() - before.len() >= 2
                && !ends_escaped(before)
                && lines.get(index + 1).is_some_and(|next| continues(next));
            if kept {
                &line[..before.len() + 2]
            } else {
                content
            }
        })
        .collect()
}

/// Whether the character at byte `at` follows an odd number of backslashes,
/// which make it text.
fn escaped_at(text: &str, at: usize) -> bool {
    ends_escaped(&text[..at])
}

/// Whether `text` ends in an odd number of backslashes, which make the
/// character after it text.
fn ends_escaped(text: &str) -> bool {
    text.bytes().rev().take_while(|&byte| byte == b'\\').count() % 2 == 1
}

/// Pages under this many are left alone: too few to call a line running.
const RUNNING_MIN_PAGES: usize = 4;
/// A running line is on at least this share of the pages, and on at least
/// [`RUNNING_MIN_COUNT`] of them.
const RUNNING_MIN_SHARE: f64 = 0.6;
const RUNNING_MIN_COUNT: usize = 3;
/// Non-empty lines at each edge of a page that may be running lines.
const EDGE_LINES: usize = 2;
/// Longer lines are body text.
const RUNNING_MAX_CHARS: usize = 100;
/// A line that differs from page to page in its numbers alone (`Page 3 of
/// 9`) is running only with at most this many words of letters besides
/// them: a templated body sentence (`Point 3 is made on slide 3.`) is not.
const NUMBERED_RUNNING_MAX_WORDS: usize = 6;

/// The key a page's line is matched by across pages: white space
/// collapsed, lower case, each run of digits `#`. `None` when the line is
/// no candidate: empty, longer than [`RUNNING_MAX_CHARS`], or Markdown
/// structure that repeats across pages without being a header or footer:
/// a heading, a table row, a picture (each page's scan or figure, its
/// placeholder numbered), HTML or a comment, a quotation, fenced code or a
/// protected literal block.
fn running_key(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty()
        || line.starts_with(['#', '|', '<', '>', '`', '~'])
        || line.starts_with("![")
        || line.starts_with(|ch| ('\u{fdd0}'..='\u{fdef}').contains(&ch))
        || line.chars().count() > RUNNING_MAX_CHARS
    {
        return None;
    }
    let mut key = String::with_capacity(line.len());
    let mut digits = false;
    for word in line.split_whitespace() {
        if !key.is_empty() {
            key.push(' ');
            digits = false;
        }
        for ch in word.chars().flat_map(char::to_lowercase) {
            if ch.is_numeric() {
                if !digits {
                    key.push('#');
                }
                digits = true;
            } else {
                key.push(ch);
                digits = false;
            }
        }
    }
    Some(key)
}

/// The indices of a page's first and last [`EDGE_LINES`] non-empty lines.
fn edge_indices(lines: &[&str]) -> Vec<usize> {
    let filled: Vec<usize> = (0..lines.len())
        .filter(|&index| !lines[index].trim().is_empty())
        .collect();
    let mut edges: Vec<usize> = filled.iter().take(EDGE_LINES).copied().collect();
    edges.extend(filled.iter().rev().take(EDGE_LINES));
    edges.sort_unstable();
    edges.dedup();
    edges
}

/// Running headers and footers of the pages of marker-delimited output
/// (`<!-- Page number: N -->`, `<!-- Slide number: N -->`), as Python
/// markitai 1.3.0 removes them: a page's first and last two non-empty lines
/// are candidates when they are at most [`RUNNING_MAX_CHARS`] long and no
/// heading or table row; one whose key ([`running_key`]) is at that edge on
/// [`RUNNING_MIN_SHARE`] of the pages, and on [`RUNNING_MIN_COUNT`] or more,
/// is removed from those edges, unless most of the pages showing it show it
/// more than once (the shape of numbered body lines) or it differs in its
/// numbers on longer lines ([`NUMBERED_RUNNING_MAX_WORDS`]). The third line
/// inside two removed ones is examined in the same pass (a slide footer of
/// date, text and number), and nothing is looked at again after a removal.
/// Text before the first marker, and documents of fewer than
/// [`RUNNING_MIN_PAGES`] pages, are left alone; nothing elsewhere in a page
/// is removed.
fn clean_running_lines(source: String) -> String {
    let markers: Vec<_> = PAGE.find_iter(&source).collect();
    if markers.len() < RUNNING_MIN_PAGES {
        return source;
    }
    let pages: Vec<&str> = markers
        .iter()
        .enumerate()
        .map(|(index, marker)| {
            let end = markers
                .get(index + 1)
                .map_or(source.len(), |next| next.start());
            &source[marker.end()..end]
        })
        .collect();
    let lines: Vec<Vec<&str>> = pages.iter().map(|page| page.lines().collect()).collect();
    let keys: Vec<Vec<Option<String>>> = lines
        .iter()
        .map(|lines| lines.iter().map(|line| running_key(line)).collect())
        .collect();
    let threshold = RUNNING_MIN_COUNT.max((pages.len() as f64 * RUNNING_MIN_SHARE).ceil() as usize);
    // The keys at `positions` (per page) that pass the counts.
    let running = |positions: &[Vec<usize>]| -> HashSet<String> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        let mut repeats: HashMap<&str, usize> = HashMap::new();
        let mut spellings: HashMap<&str, HashSet<String>> = HashMap::new();
        for (page, indices) in positions.iter().enumerate() {
            let mut seen = HashSet::new();
            for &index in indices {
                let Some(key) = keys[page][index].as_deref() else {
                    continue;
                };
                spellings.entry(key).or_default().insert(
                    lines[page][index]
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .to_lowercase(),
                );
                if !seen.insert(key) {
                    continue;
                }
                *counts.entry(key).or_default() += 1;
                if keys[page]
                    .iter()
                    .flatten()
                    .filter(|other| *other == key)
                    .count()
                    > 1
                {
                    *repeats.entry(key).or_default() += 1;
                }
            }
        }
        counts
            .into_iter()
            .filter(|&(key, count)| {
                let numbered = spellings[key].len() > 1;
                let words = key
                    .split_whitespace()
                    .filter(|word| word.chars().any(char::is_alphabetic))
                    .count();
                count >= threshold
                    && repeats.get(key).copied().unwrap_or(0) * 2 < count
                    && !(numbered && words > NUMBERED_RUNNING_MAX_WORDS)
            })
            .map(|(key, _)| key.to_owned())
            .collect()
    };
    let edges: Vec<Vec<usize>> = lines.iter().map(|lines| edge_indices(lines)).collect();
    let repeated = running(&edges);
    if repeated.is_empty() {
        return source;
    }
    // The third line of an edge whose two outer lines both go.
    let inner: Vec<Vec<usize>> = lines
        .iter()
        .zip(&keys)
        .map(|(lines, keys)| {
            let filled: Vec<usize> = (0..lines.len())
                .filter(|&index| !lines[index].trim().is_empty())
                .collect();
            let tail: Vec<usize> = filled.iter().rev().copied().collect();
            [filled.as_slice(), tail.as_slice()]
                .into_iter()
                .filter(|run| {
                    run.len() > EDGE_LINES
                        && run[..EDGE_LINES].iter().all(|&index| {
                            keys[index]
                                .as_ref()
                                .is_some_and(|key| repeated.contains(key))
                        })
                })
                .map(|run| run[EDGE_LINES])
                .collect()
        })
        .collect();
    let repeated_inner = running(&inner);
    let mut output = String::with_capacity(source.len());
    output.push_str(&source[..markers[0].end()]);
    for (page, text) in pages.iter().enumerate() {
        let gone = |index: &usize, set: &HashSet<String>| {
            keys[page][*index]
                .as_ref()
                .is_some_and(|key| set.contains(key))
        };
        let drop: HashSet<usize> = edges[page]
            .iter()
            .filter(|index| gone(index, &repeated))
            .chain(
                inner[page]
                    .iter()
                    .filter(|index| gone(index, &repeated_inner)),
            )
            .copied()
            .collect();
        if drop.is_empty() {
            output.push_str(text);
        } else {
            let kept: Vec<&str> = lines[page]
                .iter()
                .enumerate()
                .filter(|(index, _)| !drop.contains(index))
                .map(|(_, line)| *line)
                .collect();
            output.push_str(&kept.join("\n"));
            // The line ends that part the page from the next marker.
            output.push_str(&text[text.trim_end_matches(['\n', '\r']).len()..]);
        }
        if let Some(next) = markers.get(page + 1) {
            output.push_str(next.as_str());
        }
    }
    output
}

// A fence's source range is retained separately from all prose transformations.
// Delimiters use more copies of a noncharacter than occur in the entire source;
// none of the repairs can synthesize or collide with such a delimiter.
struct Literals<'a> {
    blocks: Vec<(String, &'a str)>,
}

#[derive(Clone, Copy)]
struct Fence {
    marker: u8,
    width: usize,
    quotes: usize,
    closing_column: usize,
}

fn fence_prefix(mut line: &str, opening: bool) -> (&str, usize) {
    let mut quotes = 0;
    loop {
        line = line.trim_start_matches([' ', '\t']);
        if let Some(rest) = line.strip_prefix('>') {
            quotes += 1;
            line = rest;
            continue;
        }
        if opening {
            let bytes = line.as_bytes();
            let list_width = if matches!(bytes.first(), Some(b'-' | b'+' | b'*')) {
                1
            } else {
                let digits = bytes
                    .iter()
                    .take_while(|byte| byte.is_ascii_digit())
                    .count();
                if (1..=9).contains(&digits) && matches!(bytes.get(digits), Some(b'.' | b')')) {
                    digits + 1
                } else {
                    0
                }
            };
            if list_width > 0 && matches!(bytes.get(list_width), Some(b' ' | b'\t')) {
                line = &line[list_width..];
                continue;
            }
            if line.starts_with("[^")
                && let Some(end) = line.find("]:")
                && matches!(bytes.get(end + 2), Some(b' ' | b'\t'))
            {
                line = &line[end + 2..];
                continue;
            }
        }
        return (line, quotes);
    }
}

fn prefix_column(prefix: &str) -> usize {
    prefix.chars().fold(0, |column, ch| {
        if ch == '\t' {
            (column / 4 + 1) * 4
        } else {
            column + 1
        }
    })
}

fn opening_fence(line: &str) -> Option<Fence> {
    let (content, quotes) = fence_prefix(line, true);
    let bytes = content.as_bytes();
    let &marker @ (b'`' | b'~') = bytes.first()? else {
        return None;
    };
    let width = bytes.iter().take_while(|&&byte| byte == marker).count();
    (width >= 3 && (marker != b'`' || !content[width..].contains('`'))).then_some(Fence {
        marker,
        width,
        quotes,
        closing_column: prefix_column(&line[..line.len() - content.len()]) + 3,
    })
}

fn closes_fence(line: &str, fence: Fence) -> bool {
    let (content, quotes) = fence_prefix(line, false);
    let width = content
        .as_bytes()
        .iter()
        .take_while(|&&byte| byte == fence.marker)
        .count();
    quotes == fence.quotes
        && prefix_column(&line[..line.len() - content.len()]) <= fence.closing_column
        && width >= fence.width
        && content[width..]
            .chars()
            .all(|ch| matches!(ch, ' ' | '\t' | '\r'))
}

impl<'a> Literals<'a> {
    fn protect(source: &'a str) -> (Self, String) {
        let mut counts = [0usize; 32];
        for ch in source.chars() {
            if (0xfdd0..=0xfdef).contains(&(ch as u32)) {
                counts[ch as usize - 0xfdd0] += 1;
            }
        }
        let (index, count) = counts.iter().enumerate().min_by_key(|(_, n)| **n).unwrap();
        let delimiter = char::from_u32(0xfdd0 + index as u32)
            .unwrap()
            .to_string()
            .repeat(count + 1);
        let mut ranges = Vec::new();
        let mut opened: Option<(usize, Fence)> = None;
        let mut offset = 0;
        for line in source.split_inclusive('\n') {
            let content = line.strip_suffix('\n').unwrap_or(line);
            if let Some((start, fence)) = opened {
                if closes_fence(content, fence) {
                    ranges.push(start..offset + content.len());
                    opened = None;
                }
            } else if let Some(fence) = opening_fence(content) {
                opened = Some((offset, fence));
            }
            offset += line.len();
        }
        if let Some((start, _)) = opened {
            // Leave one final LF to the normal document-ending convention. Any
            // preceding empty lines and all trailing spaces remain in the block.
            let end = source.len() - usize::from(source.ends_with('\n'));
            ranges.push(start..end);
        }
        let mut blocks = Vec::with_capacity(ranges.len());
        let mut masked = String::with_capacity(source.len());
        let mut offset = 0;
        for (index, range) in ranges.into_iter().enumerate() {
            // Longer than a footer candidate and free of Markdown syntax.
            let marker =
                format!("{delimiter}markitai-protected-fenced-literal-block-{index}{delimiter}");
            masked.push_str(&source[offset..range.start]);
            masked.push_str(&marker);
            blocks.push((marker, &source[range.clone()]));
            offset = range.end;
        }
        masked.push_str(&source[offset..]);
        (Self { blocks }, masked)
    }

    fn prose(&self, text: &str, transform: impl Fn(&str) -> String) -> String {
        let mut remaining = text;
        let mut output = String::with_capacity(text.len());
        for (marker, _) in &self.blocks {
            let (before, after) = remaining
                .split_once(marker)
                .expect("retained literal block");
            let prose = transform(before);
            output.push_str(&prose);
            // Image/placeholder repairs may consume trailing whitespace, but a
            // protected block must still start on its own source line.
            if before.ends_with('\n') && !prose.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(marker);
            remaining = after;
        }
        output.push_str(&transform(remaining));
        output
    }

    fn restore(&self, text: &str) -> String {
        let mut remaining = text;
        let mut output = String::with_capacity(text.len());
        for (marker, literal) in &self.blocks {
            let (before, after) = remaining
                .split_once(marker)
                .expect("retained literal block");
            output.push_str(before);
            output.push_str(literal);
            remaining = after;
        }
        output.push_str(remaining);
        output
    }
}

pub(crate) fn normalize(source: &str) -> String {
    let (literals, masked) = Literals::protect(source);
    let mut text = literals.prose(&masked, |part| {
        let mut text = part.to_owned();
        loop {
            let repaired = Cell::new(false);
            let next = BROKEN_LINK
                .replace_all(&text, |captures: &regex::Captures<'_>| {
                    if !captures[1].is_empty() {
                        return captures[0].to_owned();
                    }
                    // A bracket written as text (`\[`, `\]`) is no link: the
                    // repair would join a hard break's lines into one and drop
                    // the second.
                    let start = captures.get(0).map_or(0, |found| found.start());
                    if escaped_at(&text, start) || ends_escaped(&captures[3]) {
                        return captures[0].to_owned();
                    }
                    repaired.set(true);
                    format!("[{}]({})", captures[2].trim(), &captures[4])
                })
                .into_owned();
            if !repaired.get() {
                break;
            }
            text = next;
        }
        text
    });
    text = clean_running_lines(text);
    text = literals.prose(&text, |part| {
        let mut text = part.to_owned();
        if text.contains("__MARKITAI_") {
            text = PLACEHOLDER_LINE.replace_all(&text, "").into_owned();
            text = PLACEHOLDER_IMAGE.replace_all(&text, "").into_owned();
            text = PLACEHOLDER.replace_all(&text, "").into_owned();
        }
        if text.contains("![") {
            text = DOUBLE_ALT.replace_all(&text, "$1").into_owned();
            text = EMPTY_IMAGE.replace_all(&text, "").into_owned();
            text = EXTRA_PAREN.replace_all(&text, "$1").into_owned();
        }
        without_stray_delimiters(&text)
    });
    let lines = without_dangling_breaks(trimmed_lines(&text));
    let mut output: Vec<&str> = Vec::with_capacity(lines.len());
    for (index, line) in lines.iter().enumerate() {
        let hashes = line.as_bytes().iter().take_while(|&&ch| ch == b'#').count();
        let heading = (1..=6).contains(&hashes)
            && line[hashes..]
                .chars()
                .next()
                .is_none_or(char::is_whitespace);
        if heading || SLIDE.is_match(line) {
            if output.last().is_some_and(|line| !line.is_empty()) {
                output.push("");
            }
            output.push(*line);
            if lines.get(index + 1).is_some_and(|line| !line.is_empty()) {
                output.push("");
            }
        } else {
            output.push(*line);
        }
    }
    // A hard break left at a paragraph's end (before a heading's blank line,
    // or a dropped dangling break) shows nothing.
    for index in 0..output.len() {
        if output.get(index + 1).is_none_or(|next| next.is_empty()) {
            output[index] = output[index].trim_end();
        }
    }
    let joined = output.join("\n");
    literals.restore(&format!("{}\n", BLANKS.replace_all(joined.trim(), "\n\n")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_fences_keep_heading_content_and_normalize_only_outer_spacing() {
        let input = "  \n# Heading  \nText\n````md\n```rust\n# code\n```\n~~~~\n## still code\n````\n## Next\nBody\n\n\n";
        assert_eq!(
            normalize(input),
            "# Heading\n\nText\n````md\n```rust\n# code\n```\n~~~~\n## still code\n````\n\n## Next\n\nBody\n"
        );
    }

    #[test]
    fn repairs_links_placeholders_and_repeated_page_footers() {
        assert_eq!(
            normalize(
                "[Title\n\nDescription](/url)\n\n![a]![b](pic))\n![](\u{5f}_MARKITAI_IMAGE_1__)"
            ),
            "[Title](/url)\n\n![b](pic)\n"
        );
        let input = (1..=4).map(|n| format!("<!-- Page number: {n} -->\n# Page {n}\nLong content for page {n} that must remain.\nFooter\n{n}\n")).collect::<String>();
        let output = normalize(&input);
        assert!(!output.contains("Footer"));
        assert!(output.contains("Long content for page 2 that must remain."));
    }

    fn marker_pages(bodies: &[String], marker: &str) -> String {
        bodies
            .iter()
            .enumerate()
            .map(|(index, body)| format!("<!-- {marker}: {} -->\n\n{body}", index + 1))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    const BODIES: [&str; 5] = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"];

    #[test]
    fn running_headers_and_footers_leave_every_page() {
        let bodies: Vec<String> = BODIES
            .iter()
            .enumerate()
            .map(|(index, body)| {
                format!(
                    "ACME Confidential\n\n{body} opens.\n\n{body} goes on.\n\n{body} ends.\n\nAcme Corp\n{}",
                    index + 1
                )
            })
            .collect();
        let output = normalize(&marker_pages(&bodies, "Page number"));
        assert!(!output.contains("ACME Confidential") && !output.contains("Acme Corp"));
        for (index, body) in BODIES.iter().enumerate() {
            assert!(
                output.contains(&format!("{body} opens.\n\n{body} goes on.\n\n{body} ends.")),
                "{output}"
            );
            assert!(output.contains(&format!("<!-- Page number: {} -->", index + 1)));
        }
    }

    #[test]
    fn a_three_line_slide_footer_goes_and_a_body_line_like_it_stays() {
        // Date, text and number at the foot of each slide; slide 3 also
        // names the company in its body.
        let bodies: Vec<String> = (1..=5)
            .map(|n| {
                let extra = if n == 3 { "\n\nAcme Corp" } else { "" };
                format!("# Topic {n}\n\nBody point {n} discusses the subject at length for slide {n}.{extra}\n\n2026-10-01\n\nAcme Corp\n\n{n}")
            })
            .collect();
        let output = normalize(&marker_pages(&bodies, "Slide number"));
        assert!(!output.contains("2026-10-01"), "{output}");
        assert_eq!(output.matches("Acme Corp").count(), 1, "{output}");
        assert!(output.contains("slide 3.\n\nAcme Corp"), "{output}");
        // A long body line that differs only in its numbers is no footer.
        assert_eq!(
            output.matches("discusses the subject").count(),
            5,
            "{output}"
        );
        for n in 1..=5 {
            assert!(output.contains(&format!("# Topic {n}")));
        }
    }

    #[test]
    fn lines_on_half_the_slides_headings_tables_and_short_documents_stay() {
        let half: Vec<String> = ["Plan", "Risks", "Budget", "Team"]
            .iter()
            .enumerate()
            .map(|(index, topic)| {
                let ask = if index < 2 { "\n\nQuestions?" } else { "" };
                format!("# {topic}\n\nAbout {topic}.{ask}")
            })
            .collect();
        let output = normalize(&marker_pages(&half, "Slide number"));
        assert_eq!(output.matches("Questions?").count(), 2);
        let tables: Vec<String> = (1..=4)
            .map(|n| format!("## Totals\n\n| Item | Cost |\n| --- | --- |\n| Item {n} | {n} |\n| Total | n/a |"))
            .collect();
        let output = normalize(&marker_pages(&tables, "Page number"));
        assert_eq!(output.matches("| Total | n/a |").count(), 4);
        assert_eq!(output.matches("## Totals").count(), 4);
        let short: Vec<String> = (1..=3)
            .map(|n| format!("Body {n}.\n\nAcme Corp\n{n}"))
            .collect();
        let output = normalize(&marker_pages(&short, "Page number"));
        assert_eq!(output.matches("Acme Corp").count(), 3);
    }

    #[test]
    fn numbered_and_long_body_lines_are_never_running_lines() {
        // A four-page scan read by OCR: twelve numbered lines a page.
        let scan: Vec<String> = (1..=4)
            .map(|page| {
                (1..=12)
                    .map(|line| format!("Page {page} line {line}: The quick brown fox jumps over the lazy dog {page}{line:02}."))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect();
        let output = normalize(&marker_pages(&scan, "Page number"));
        assert_eq!(output.matches("The quick brown fox").count(), 48);
        // A 60-page report of numbered paragraphs.
        let report: Vec<String> = (1..=60)
            .map(|page| {
                let paragraphs = (1..=9)
                    .map(|k| format!("The committee reviewed section {page}.{k}. The annual plan describes the library, its reading rooms and the budget lines for staff."))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                format!("# Chapter {page}\n\n{paragraphs}")
            })
            .collect();
        let output = normalize(&marker_pages(&report, "Page number"));
        assert_eq!(output.matches("The committee reviewed").count(), 540);
        // The same notice of more than 100 characters on every page.
        let notice = "This report was prepared for the members of the board and is shared with them under the terms of the annual agreement.";
        let long: Vec<String> = BODIES
            .iter()
            .map(|body| format!("{notice}\n\n{body} opens.\n\n{body} ends."))
            .collect();
        let output = normalize(&marker_pages(&long, "Page number"));
        assert_eq!(output.matches(notice).count(), 5);
    }

    #[test]
    fn pictures_and_html_at_a_page_edge_are_no_running_lines() {
        let bodies: Vec<String> = BODIES
            .iter()
            .enumerate()
            .map(|(index, body)| {
                let n = index + 1;
                format!("{body} opens.\n\n{body} ends.\n\n![Image on page {n}](\u{5f}_MARKITAI_IMAGE_{n}__)\n\n<!-- note -->")
            })
            .collect();
        let output = marker_pages(&bodies, "Page number");
        assert_eq!(clean_running_lines(output.clone()), output);
    }

    #[test]
    fn running_line_removal_does_not_cascade_into_the_body() {
        let bodies: Vec<String> = BODIES
            .iter()
            .map(|body| format!("Acme Corp\n\nAnnual report\n\nDraft\n\nSee below.\n\n{body} one.\n\n{body} two.\n\n{body} three."))
            .collect();
        let output = normalize(&marker_pages(&bodies, "Page number"));
        for line in ["Acme Corp", "Annual report", "Draft"] {
            assert!(!output.contains(line), "{output}");
        }
        assert_eq!(output.matches("See below.").count(), 5);
        // A line found anywhere but at a page's edge stays, and a page
        // number line goes with its footer.
        let bodies: Vec<String> = (1..=4)
            .map(|n| {
                let lead = if n == 1 {
                    "Intro.\n\nHistory.\n\nAcme Corp\n\nwas founded in 1999.\n\n"
                } else {
                    ""
                };
                let text = [
                    "More text.",
                    "Revenue grew.",
                    "Costs stayed flat.",
                    "Outlook is good.",
                ];
                format!("{lead}{}\n\nAcme Corp\n{n}", text[n - 1])
            })
            .collect();
        let output = normalize(&marker_pages(&bodies, "Page number"));
        assert_eq!(output.matches("Acme Corp").count(), 1, "{output}");
        assert!(output.contains("Acme Corp\n\nwas founded in 1999."));
    }

    #[test]
    fn a_table_keeps_one_delimiter_row() {
        // A model repeated the delimiter row of a page's table.
        assert_eq!(
            normalize(
                "| Tool | What it does |\n| --- | --- |\n| --- | --- |\n| [lamp](https://lamp.example) | Shows the lamp |\n| :--- | ---: |\n"
            ),
            "| Tool | What it does |\n| --- | --- |\n| [lamp](https://lamp.example) | Shows the lamp |\n"
        );
        // Each table keeps its own; fenced code and a dash row of text stay.
        let kept =
            "| A |\n| --- |\n\n| B |\n| --- |\n| -- |\n\n```\n| x |\n| --- |\n| --- |\n```\n";
        assert_eq!(normalize(kept), kept);
    }

    #[test]
    fn hard_breaks_and_brackets_written_as_text_survive_the_link_repair() {
        // A backslash hard break and a two-space one are both kept.
        assert_eq!(normalize("one\\\ntwo  \nthree"), "one\\\ntwo  \nthree\n");
        // An escaped bracket is text: nothing between it and a later `](`
        // is a link split by a line break, and no line is dropped.
        for source in [
            "a \\[note\\\nsecond line\\](x)\n",
            "a [note\\\nsecond line\\](x)\n",
        ] {
            assert_eq!(normalize(source), source);
        }
        // An escaped backslash before a real bracket leaves the repair on.
        assert_eq!(normalize("\\\\[Title\nmore](/u)"), "\\\\[Title](/u)\n");
    }

    #[test]
    fn hard_breaks_left_without_a_line_to_break_are_dropped() {
        // A filtered image removed from between, before or after hard breaks.
        assert_eq!(
            normalize("Before.\n\n\\\nCaption\n\nText\\\n\\\nmore\\\n\nEnd\\"),
            "Before.\n\nCaption\n\nText\\\nmore\n\nEnd\n"
        );
        // An escaped backslash is text, and code keeps its line ends.
        let kept = "C:\\\\\n\n    echo \\\n\n```sh\nmake \\\n\\\n```\n";
        assert_eq!(normalize(kept), kept);
    }

    #[test]
    fn two_space_hard_breaks_are_kept_only_where_a_line_continues() {
        // More spaces are written as two; a tab is no hard break; a line of
        // blanks only is empty.
        assert_eq!(
            normalize("one   \ntwo\t\nthree  \n   \nfour\u{a0}  \r\nfive"),
            "one  \ntwo\nthree\n\nfour\u{a0}  \nfive\n"
        );
        // Not at a paragraph's or the document's end, before a heading or a
        // fenced block, or after a backslash (a hard break or an escape).
        assert_eq!(
            normalize("a  \n\nb  \n# H  \nc  \n```\nx  \ny\n```\nd\\  \ne  "),
            "a\n\nb\n\n# H\n\nc\n```\nx  \ny\n```\nd\\\ne\n"
        );
        // A break whose next line a removed image or placeholder took.
        assert_eq!(
            normalize("Text  \n\u{5f}_MARKITAI_IMAGE_1__\n\nEnd  \n![a]()"),
            "Text\n\nEnd\n"
        );
        assert_eq!(
            normalize("Text  \n![](.markitai/assets/)  \nmore"),
            "Text  \nmore\n"
        );
        // Normalizing again, as enhanced output is, keeps them.
        let once = normalize("Poem:  \nRoses are red,  \nViolets are blue.  \n");
        assert_eq!(once, "Poem:  \nRoses are red,  \nViolets are blue.\n");
        assert_eq!(normalize(&once), once);
    }

    #[test]
    fn a_stray_bracket_before_a_link_keeps_the_text_between_them() {
        // An unmatched `[` is text: the repair starts at the bracket that
        // opens the link, so nothing between the two is dropped.
        let source = "Index a[i is out of range\nand this whole sentence matters.\n\nSecond paragraph keeps going.\n\nSee [docs](https://x.test).\n";
        assert_eq!(normalize(source), source);
        assert_eq!(
            normalize("Use a[0 here.\n\n[Title\n\nDescription](/url)\n"),
            "Use a[0 here.\n\n[Title](/url)\n"
        );
    }

    #[test]
    fn image_syntax_is_not_repaired_as_a_link_split_by_a_line_break() {
        // The repair keeps a link's first line only; an image's text must
        // survive whole, and a link beside it is still repaired.
        assert_eq!(
            normalize("![first line\nsecond line](pic.png)\n\n[Title\n\nDescription](/url)\n"),
            "![first line\nsecond line](pic.png)\n\n[Title](/url)\n"
        );
        assert_eq!(
            normalize("Wow! [A\nB](/a) ![C\n\nD](/c) [E\nF](/e)"),
            "Wow! [A](/a) ![C\n\nD](/c) [E](/e)\n"
        );
    }

    #[test]
    fn fenced_examples_bypass_every_prose_repair_and_keep_exact_whitespace() {
        let block = "````markdown  \r\n__MARKITAI_IMAGE_1__\r\n![a]![b](pic))  \n![]()\n[Title\n\nDescription](/url)\n<!-- Slide 1 -->\n# literal  \n\n\n\nend\t \n```` \t";
        assert_eq!(
            normalize(&format!("Before  \n{block}\nAfter  \n\n\n")),
            format!("Before\n{block}\nAfter\n")
        );
    }

    #[test]
    fn tables_that_end_several_pages_keep_their_rows() {
        let input = (1..=3)
            .map(|n| format!("<!-- Slide number: {n} -->\n# Slide {n}\n\n| A | B |\n| --- | --- |\n| {n} | x |\n"))
            .collect::<String>();
        let output = normalize(&input);
        assert_eq!(output.matches("| --- | --- |").count(), 3, "{output}");
        assert_eq!(output.matches("| A | B |").count(), 3, "{output}");
    }

    #[test]
    fn code_page_markers_and_footer_examples_are_not_document_pages() {
        let pages = (1..=4)
            .map(|n| format!("<!-- Page number: {n} -->\n# Page {n}\nLong content for page {n} that must remain.\nFooter\n{n}\n"))
            .collect::<String>();
        let block = format!("```markdown\n{pages}```");
        assert_eq!(normalize(&block), format!("{block}\n"));
        let output = normalize(&format!("{block}\n\n{pages}"));
        assert!(output.starts_with(&format!("{block}\n")));
        assert_eq!(output.matches("Footer").count(), 4);
        assert!(output.contains("Long content for page 2 that must remain."));
    }

    #[test]
    fn closing_fences_require_matching_marker_width_and_empty_tail() {
        let block = "~~~~lang\n~~~\n```\n~~~~ trailing text\n# still literal  \n\n\n~~~~~\t";
        assert_eq!(
            normalize(&format!("{block}\n# Prose  \nText")),
            format!("{block}\n\n# Prose\n\nText\n")
        );
        let block = "````rust\n```\n````not a close\n# literal  \n`````";
        assert_eq!(normalize(block), format!("{block}\n"));
    }

    #[test]
    fn list_footnote_and_quote_containers_preserve_fenced_literals() {
        for block in [
            "- ```rust\n  fn main() {  \n\n\n  }\n  ```",
            "1. > ~~~text\n   > [Title\n   > Description](/url)  \n   > ~~~",
            "[^note]:\n    ```markdown\n    __MARKITAI_IMAGE_1__  \n\n\n    ```",
            "[^note]: - ```text\n      # literal  \n      ```",
        ] {
            assert_eq!(normalize(block), format!("{block}\n"));
        }
        let block = "```text\n> ```\n# still literal  \n```";
        assert_eq!(normalize(block), format!("{block}\n"));
    }

    #[test]
    fn unclosed_fence_keeps_remainder_and_document_newline_convention() {
        for block in [
            "```rust\nlet token = \"__MARKITAI_IMAGE_1__\";  ",
            "~~~\nlast  \n\n\n",
            "    ```\n\tvalue\t \n\n",
        ] {
            let expected = if block.ends_with('\n') {
                block.to_owned()
            } else {
                format!("{block}\n")
            };
            assert_eq!(normalize(block), expected);
        }
    }

    #[test]
    fn malformed_prose_links_cannot_consume_an_intervening_code_block() {
        let source = "[start\n```text\nprotected()  \n\n\n```\nend](/example)\n";
        assert_eq!(normalize(source), source);
        assert_eq!(
            normalize("```not ` an opener\n[Title\nDescription](/url)"),
            "```not ` an opener\n[Title](/url)\n"
        );
    }

    #[test]
    fn literal_markers_cannot_collide_with_source_text() {
        let noncharacters = (0xfdd0..=0xfdef)
            .map(|value| char::from_u32(value).unwrap())
            .collect::<String>();
        let prose =
            format!("{noncharacters}\u{fdd0}markitai-protected-fenced-literal-block-0\u{fdd0}");
        let source = format!("{prose}\n```text\n{prose}  \n\n\n```\n");
        assert_eq!(normalize(&source), source);
    }

    #[test]
    fn removed_images_cannot_join_prose_to_an_opening_fence() {
        for image in ["![](__MARKITAI_IMAGE_1__)", "![]()"] {
            assert_eq!(
                normalize(&format!("Before {image}\n```text\nliteral  \n```\nAfter")),
                "Before\n```text\nliteral  \n```\nAfter\n"
            );
        }
    }

    #[test]
    fn deeply_indented_delimiters_inside_fences_remain_literal() {
        for delimiter in ["    ```", "\t```", "- ```"] {
            let block = format!("```text\n{delimiter}\n__MARKITAI_IMAGE_1__  \n# literal  \n```");
            assert_eq!(normalize(&block), format!("{block}\n"));
        }
        let block = "[^note]:\n    ~~~text\n        ~~~\n    __MARKITAI_IMAGE_1__  \n    ~~~";
        assert_eq!(normalize(block), format!("{block}\n"));
    }
}
