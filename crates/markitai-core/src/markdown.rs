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

fn clean_footers(source: String) -> String {
    let markers: Vec<_> = PAGE.find_iter(&source).collect();
    if markers.len() < 3 {
        return source;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut pages = 0;
    for (index, marker) in markers.iter().enumerate() {
        let end = markers
            .get(index + 1)
            .map_or(source.len(), |next| next.start());
        let lines: Vec<_> = source[marker.end()..end]
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        if lines.len() < 2 {
            continue;
        }
        let endings: Vec<_> = lines
            .iter()
            .rev()
            .take(4)
            .copied()
            // Markdown structure (a table row or its separator, a quote, a
            // fence, a comment) repeats across pages without being a footer.
            .filter(|line| {
                line.chars().count() < 30
                    && !line.starts_with(['#', '!', '[', '-', '*', '|', '>', '`', '~', '<'])
            })
            .collect();
        if !endings.is_empty() {
            pages += 1;
            for line in endings {
                *counts.entry(line).or_default() += 1;
            }
        }
    }
    if pages < 3 {
        return source;
    }
    let numeric = |line: &str| !line.is_empty() && line.chars().all(|ch| ch.is_numeric());
    let common: HashSet<_> = counts
        .into_iter()
        .filter(|(line, count)| count * 2 >= pages && !numeric(line))
        .map(|(line, _)| line)
        .collect();
    if common.is_empty() {
        return source;
    }
    let lines: Vec<_> = source.split('\n').collect();
    let mut output = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if common.contains(line.trim())
            || (numeric(line.trim())
                && (1..=2).any(|offset| {
                    index
                        .checked_sub(offset)
                        .is_some_and(|before| common.contains(lines[before].trim()))
                }))
        {
            continue;
        }
        output.push(*line);
    }
    output.join("\n")
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
    text = clean_footers(text);
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
        text
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
        let input = (1..=3).map(|n| format!("<!-- Page number: {n} -->\n# Page {n}\nLong content for page {n} that must remain.\nFooter\n{n}\n")).collect::<String>();
        let output = normalize(&input);
        assert!(!output.contains("Footer"));
        assert!(output.contains("Long content for page 2 that must remain."));
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
        let pages = (1..=3)
            .map(|n| format!("<!-- Page number: {n} -->\n# Page {n}\nLong content for page {n} that must remain.\nFooter\n{n}\n"))
            .collect::<String>();
        let block = format!("```markdown\n{pages}```");
        assert_eq!(normalize(&block), format!("{block}\n"));
        let output = normalize(&format!("{pages}\n{block}\n"));
        assert!(output.ends_with(&format!("{block}\n")));
        assert_eq!(output.matches("Footer").count(), 3);
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
