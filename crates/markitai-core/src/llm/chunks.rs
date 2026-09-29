//! Source-owned literals are opaque to the model and restored only after validation.
use crate::{Error, Result};
use sha2::{Digest, Sha256};

pub(super) const LIMIT: usize = 32_000;

pub(super) struct Protected {
    pub text: String,
    prefix: String,
    literals: Vec<(String, String)>,
    pub page_starts: Vec<(usize, usize)>,
    ranges: Vec<std::ops::Range<usize>>,
}

impl Protected {
    pub fn new(source: &str) -> Self {
        let digest = crate::hex(Sha256::digest(source.as_bytes()));
        let mut prefix = format!("⟦MKTI:{}:", &digest[..16]);
        while source.contains(&prefix) {
            prefix.push('_');
        }
        let mut ranges = Vec::new();
        let mut fence: Option<(usize, u8, usize, usize)> = None;
        let mut offset = 0;
        for line in source.split_inclusive('\n') {
            let (body, quotes) = line_prefix(line);
            let body = if fence.is_some() {
                body.trim_start_matches(' ')
            } else {
                body
            };
            let marker = body.as_bytes().first().copied().unwrap_or_default();
            let width = body.bytes().take_while(|&b| b == marker).count();
            if let Some((start, ch, count, depth)) = fence {
                if marker == ch
                    && width >= count
                    && quotes == depth
                    && body[width..].trim().is_empty()
                {
                    ranges.push(start..offset + line.len());
                    fence = None;
                }
            } else if matches!(marker, b'`' | b'~')
                && width >= 3
                && (marker != b'`' || !body[width..].contains('`'))
            {
                fence = Some((offset, marker, width, quotes));
            } else if body.starts_with("    ") || line.starts_with('\t') {
                // Indented code is retained conservatively, including nested list code.
                ranges.push(offset..offset + line.len());
            } else if body.starts_with('[') && body.contains("]:") {
                ranges.push(offset..offset + line.len());
            }
            offset += line.len();
        }
        if let Some((start, ..)) = fence {
            ranges.push(start..source.len());
        }
        // Scan only gaps; code and reference definitions cannot introduce markup.
        let blocks = ranges.clone();
        let mut cursor = 0;
        for range in blocks
            .into_iter()
            .chain(std::iter::once(source.len()..source.len()))
        {
            inline_ranges(source, cursor, range.start, &mut ranges);
            cursor = range.end;
        }
        ranges.sort_by_key(|r| r.start);
        // Record semantic boundaries before adjacent literal ranges coalesce.
        // A fence ending immediately before page 1 must not hide that page.
        let mut page_starts = Vec::new();
        for range in &ranges {
            let literal = source[range.clone()].trim();
            if let Some(inner) = literal
                .strip_prefix("<!--")
                .and_then(|s| s.strip_suffix("-->"))
            {
                let inner = inner.trim();
                if let Some(number) = inner
                    .strip_prefix("Page number:")
                    .or_else(|| inner.strip_prefix("Slide number:"))
                    .or_else(|| inner.strip_prefix("Page "))
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .filter(|n| *n > 0)
                {
                    page_starts.push((range.start, number));
                }
            }
        }
        page_starts.dedup();
        let mut merged: Vec<std::ops::Range<usize>> = Vec::new();
        for range in ranges {
            if let Some(last) = merged.last_mut()
                && range.start <= last.end
            {
                last.end = last.end.max(range.end);
            } else {
                merged.push(range);
            }
        }
        let mut text = String::new();
        let mut literals = Vec::new();
        cursor = 0;
        let source_ranges = merged.clone();
        for range in merged {
            text.push_str(&source[cursor..range.start]);
            let token = format!("{prefix}{:08}⟧", literals.len());
            text.push_str(&token);
            // Keep block boundaries around an opaque multi-line literal.
            if source[range.clone()].ends_with('\n') {
                text.push('\n');
            }
            literals.push((token, source[range.clone()].to_owned()));
            cursor = range.end;
        }
        text.push_str(&source[cursor..]);
        Self {
            text,
            prefix,
            literals,
            page_starts,
            ranges: source_ranges,
        }
    }

    pub fn without_markers(&self, text: &str) -> String {
        let mut output = String::new();
        let mut rest = text;
        while let Some(start) = rest.find(&self.prefix) {
            output.push_str(&rest[..start]);
            let Some(end) = rest[start..].find('⟧') else {
                return output;
            };
            rest = &rest[start + end + '⟧'.len_utf8()..];
        }
        output.push_str(rest);
        output
    }

    pub fn boundary_after(&self, source: &str, mut offset: usize) -> usize {
        while offset < source.len() && !source.is_char_boundary(offset) {
            offset += 1;
        }
        loop {
            let index = self.ranges.partition_point(|range| range.end <= offset);
            if let Some(range) = self.ranges.get(index)
                && range.start < offset
                && offset < range.end
            {
                offset = range.end;
            } else {
                break;
            }
        }
        // A line boundary cannot split prose words or a protected source literal.
        if offset == 0 || source.as_bytes().get(offset.wrapping_sub(1)) == Some(&b'\n') {
            return offset;
        }
        let end = source[offset..]
            .find('\n')
            .map_or(source.len(), |n| offset + n + 1);
        let index = self.ranges.partition_point(|range| range.end <= end);
        self.ranges
            .get(index)
            .filter(|range| range.start < end && end < range.end)
            .map_or(end, |range| range.end)
    }

    pub fn split(&self) -> Vec<String> {
        split(&self.text, Some(&self.prefix))
    }

    pub fn validate(&self, original: &str, answer: &str) -> Result<()> {
        let tokens = |text: &str| -> Vec<String> {
            text.match_indices(&self.prefix)
                .map(|(start, _)| {
                    let end = text[start..]
                        .find('⟧')
                        .map(|end| start + end + '⟧'.len_utf8())
                        .unwrap_or(text.len());
                    text[start..end].to_owned()
                })
                .collect()
        };
        let expected = tokens(original);
        let actual = tokens(answer);
        if expected != actual {
            return Err(Error::Conversion(
                "LLM changed, removed, duplicated or reordered a protected document marker".into(),
            ));
        }
        Ok(())
    }

    pub fn restore(&self, answer: &str) -> Result<String> {
        self.validate(&self.text, answer)?;
        let mut rest = answer;
        let mut output = String::with_capacity(answer.len());
        for (token, literal) in &self.literals {
            let Some(at) = rest.find(token) else {
                unreachable!("validated marker")
            };
            output.push_str(&rest[..at]);
            output.push_str(literal);
            let end = at + token.len();
            let skip = usize::from(literal.ends_with('\n') && rest[end..].starts_with('\n'));
            rest = &rest[end + skip..];
        }
        output.push_str(rest);
        Ok(output)
    }
}

fn line_prefix(mut line: &str) -> (&str, usize) {
    let mut quotes = 0;
    loop {
        let spaces = line.bytes().take_while(|&b| b == b' ').count();
        if spaces <= 3 {
            line = &line[spaces..];
        }
        if let Some(rest) = line.strip_prefix('>') {
            quotes += 1;
            line = rest.strip_prefix(' ').unwrap_or(rest);
        } else {
            break;
        }
    }
    // A fence can begin a list item/footnote; closers need no list marker.
    for marker in ["- ", "+ ", "* "] {
        if let Some(rest) = line.strip_prefix(marker) {
            return (rest.trim_start_matches(' '), quotes);
        }
    }
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0
        && matches!(line.as_bytes().get(digits), Some(b'.' | b')'))
        && line.as_bytes().get(digits + 1) == Some(&b' ')
    {
        line = line[digits + 2..].trim_start_matches(' ');
    }
    if line.starts_with("[^")
        && let Some(end) = line.find("]: ")
    {
        line = line[end + 3..].trim_start_matches(' ');
    }
    (line, quotes)
}

struct InlineIndex {
    paired: std::collections::HashMap<usize, usize>,
    delimiters: std::collections::HashMap<usize, usize>,
    math_square: Vec<usize>,
    math_round: Vec<usize>,
}
impl InlineIndex {
    fn new(text: &str) -> Self {
        let mut index = Self {
            paired: Default::default(),
            delimiters: Default::default(),
            math_square: Vec::new(),
            math_round: Vec::new(),
        };
        let mut square = Vec::new();
        let mut round = Vec::new();
        let mut last = std::collections::HashMap::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                match bytes[i + 1] {
                    b']' => index.math_square.push(i),
                    b')' => index.math_round.push(i),
                    _ => (),
                }
                i += 1 + text[i + 1..].chars().next().unwrap().len_utf8();
                continue;
            }
            match bytes[i] {
                b'[' => square.push(i),
                b'(' => round.push(i),
                b']' => {
                    if let Some(open) = square.pop() {
                        index.paired.insert(open, i + 1);
                    }
                }
                b')' => {
                    if let Some(open) = round.pop() {
                        index.paired.insert(open, i + 1);
                    }
                }
                marker @ (b'`' | b'$') => {
                    let width = bytes[i..]
                        .iter()
                        .take_while(|&&byte| byte == marker)
                        .count();
                    if (marker == b'`' || width <= 2)
                        && let Some(open) = last.insert((marker, width), i)
                    {
                        index.delimiters.insert(open, i + width);
                    }
                    i += width;
                    continue;
                }
                _ => (),
            }
            i += text[i..].chars().next().unwrap().len_utf8();
        }
        index
    }
}

fn inline_ranges(source: &str, start: usize, end: usize, ranges: &mut Vec<std::ops::Range<usize>>) {
    let source = &source[start..end];
    let index = InlineIndex::new(source);
    let bytes = source.as_bytes();
    let mut i = 0;
    let mut html_failed_until = 0;
    let mut comment_failed = false;
    while i < source.len() {
        let rest = &source[i..];
        if rest.starts_with("\\[") || rest.starts_with("\\(") {
            let ends = if bytes[i + 1] == b'[' {
                &index.math_square
            } else {
                &index.math_round
            };
            if let Some(&at) = ends.get(ends.partition_point(|&at| at < i + 2)) {
                ranges.push(start + i..start + at + 2);
                i = at + 2;
                continue;
            }
        }
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 1 + source[i + 1..].chars().next().unwrap().len_utf8();
            continue;
        }
        let mut next = None;
        if bytes[i] == b'`' {
            next = index.delimiters.get(&i).copied();
            if next.is_none() {
                i += rest.bytes().take_while(|&b| b == b'`').count();
                continue;
            }
        } else if rest.starts_with("<!--") {
            if !comment_failed {
                next = rest.find("-->").map(|at| i + at + 3);
                comment_failed = next.is_none();
            }
        } else if bytes[i] == b'<' && i >= html_failed_until {
            let mut quote = None;
            let mut scanned = i + 1;
            // Unclosed attribute strings cannot make each following '<' rescan
            // the same tail. Scan a bounded window, then skip that failed window.
            for (at, ch) in rest.char_indices().skip(1).take(16_384) {
                scanned = i + at + ch.len_utf8();
                if let Some(q) = quote {
                    if ch == q {
                        quote = None;
                    }
                } else if ch == '\'' || ch == '"' {
                    quote = Some(ch);
                } else if ch == '>' {
                    next = Some(i + at + 1);
                    break;
                } else if ch == '\n' {
                    break;
                }
            }
            if next.is_none() {
                html_failed_until = scanned;
            }
        } else if bytes[i] == b'$' {
            let width = rest.bytes().take_while(|&b| b == b'$').count();
            if width <= 2 && !rest[width..].starts_with(char::is_whitespace) {
                next = index
                    .delimiters
                    .get(&i)
                    .copied()
                    .filter(|&end| end > i + width * 2);
            }
            if next.is_none() {
                i += width;
                continue;
            }
        } else if bytes[i] == b'[' || rest.starts_with("![") {
            let open = i + usize::from(rest.starts_with('!'));
            if let Some(&label) = index.paired.get(&open) {
                if matches!(bytes.get(label), Some(b'(' | b'[')) {
                    next = index.paired.get(&label).copied();
                } else if source[open..].starts_with("[[") {
                    next = Some(label);
                }
            }
        }
        if let Some(end) = next {
            ranges.push(start + i..start + end);
            i = end;
        } else {
            i += rest.chars().next().unwrap().len_utf8();
        }
    }
}

/// Every scalar is included once; oversized prose blocks split at newline then
/// scalar boundaries. An opaque protected marker is never bisected.
fn split(text: &str, protected_prefix: Option<&str>) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut rest = text;
    while rest.chars().take(LIMIT + 1).count() > LIMIT {
        let mut end = rest
            .char_indices()
            .nth(LIMIT)
            .map_or(rest.len(), |(at, _)| at);
        if let Some(prefix) = protected_prefix {
            // Include a bounded lookahead when the scalar boundary falls inside
            // the prefix itself, including its opening multibyte bracket.
            let mut lookahead = end.saturating_add(prefix.len()).min(rest.len());
            while !rest.is_char_boundary(lookahead) {
                lookahead += 1;
            }
            if let Some((start, _)) = rest[..lookahead]
                .rmatch_indices(prefix)
                .find(|(start, _)| *start < end)
                && let Some(close) = rest[start..].find('⟧')
            {
                let token_end = start + close + '⟧'.len_utf8();
                if end < token_end {
                    // This prefix is absent from source text: only generated tokens
                    // can match it. A token at the start must still advance the loop.
                    end = if start > 0 { start } else { token_end };
                }
            }
        }
        let prefix = &rest[..end];
        if let Some(at) = prefix.rfind("\n\n").filter(|&at| at > 0) {
            end = at + 2;
        } else if let Some(at) = prefix.rfind('\n').filter(|&at| at > 0) {
            end = at + 1;
        }
        chunks.push(rest[..end].to_owned());
        rest = &rest[end..];
    }
    if !rest.is_empty() || chunks.is_empty() {
        chunks.push(rest.to_owned());
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unmatched_wide_markup_keeps_bytes_and_later_valid_links() {
        let source = format!(
            "{}\n{}\n{}\n[end](target)\n",
            "[".repeat(100_000),
            "<".repeat(100_000),
            "$ ".repeat(50_000)
        );
        let protected = Protected::new(&source);
        assert_eq!(protected.restore(&protected.text).unwrap(), source);
        assert!(!protected.text.contains("[end](target)"));
        assert_eq!(protected.split().concat(), protected.text);
    }

    #[test]
    fn literals_and_nested_fences_keep_all_original_bytes() {
        let source = "Intro\n\n- ````rust\n  let x = `foo`;  \n\n\n  ```\n  ````\n\n![x](a(b).png) [link](https://example.test/?a=1) $x^2$\n<!-- Page 2 -->\n__MARKITAI_IMAGE_0__";
        let protected = Protected::new(source);
        assert!(!protected.text.contains("let x"));
        assert_eq!(protected.restore(&protected.text).unwrap(), source);
        let token = &protected.literals[0].0;
        assert!(
            protected
                .restore(&protected.text.replace(token, ""))
                .is_err()
        );
        assert!(
            protected
                .restore(&format!("{}{token}", protected.text))
                .is_err()
        );
    }
    #[test]
    fn unicode_chunks_keep_tail_and_exact_concatenation() {
        let text = format!("{}\n\n{}TAIL", "甲".repeat(33_000), "乙".repeat(35_000));
        let chunks = split(&text, None);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|s| s.chars().count() <= LIMIT));
        assert_eq!(chunks.concat(), text);
        assert!(chunks.last().unwrap().ends_with("TAIL"));
    }
    #[test]
    fn source_marker_lookalikes_are_plain_text_and_chunks_always_advance() {
        for source in [
            format!("⟦MKTI:{}TAIL", "界".repeat(LIMIT * 2)),
            format!(
                "{}⟦MKTI:0123456789abcdef:00000000⟧{}TAIL",
                "x".repeat(LIMIT - 4),
                "y".repeat(LIMIT)
            ),
        ] {
            let protected = Protected::new(&source);
            let chunks = protected.split();
            assert!(chunks.len() <= 3);
            assert!(
                chunks
                    .iter()
                    .all(|chunk| !chunk.is_empty() && chunk.chars().count() <= LIMIT)
            );
            assert_eq!(chunks[0].chars().count(), LIMIT);
            assert_eq!(chunks.concat(), source);
            assert_eq!(protected.restore(&chunks.concat()).unwrap(), source);
        }
    }

    #[test]
    fn generated_literal_token_crossing_chunk_boundary_remains_whole() {
        // Exercise boundaries in the opening glyph, hash, index and closing glyph.
        for distance in 1..=48 {
            let source = format!(
                "{}[source](assets/original.png){}TAIL",
                "界".repeat(LIMIT - distance),
                "y".repeat(LIMIT)
            );
            let protected = Protected::new(&source);
            let chunks = protected.split();
            assert!(
                chunks
                    .iter()
                    .all(|chunk| !chunk.is_empty() && chunk.chars().count() <= LIMIT)
            );
            assert_eq!(chunks.concat(), protected.text);
            for (token, _) in &protected.literals {
                assert_eq!(
                    chunks.iter().filter(|chunk| chunk.contains(token)).count(),
                    1
                );
            }
            assert_eq!(protected.restore(&chunks.concat()).unwrap(), source);
        }
    }

    #[test]
    fn very_long_literal_is_one_restorable_token_not_a_truncated_chunk() {
        let source = format!("```text\n{}\n```\nAfter", "x  \n".repeat(50_000));
        let protected = Protected::new(&source);
        assert!(protected.text.chars().count() < LIMIT);
        assert_eq!(protected.restore(&protected.text).unwrap(), source);
    }
}
