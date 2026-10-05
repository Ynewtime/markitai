//! Source-owned literals are opaque to the model and restored only after validation.
use crate::{Error, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Unicode scalar values per chunk when no model declares a smaller window.
pub(super) const LIMIT: usize = 32_000;
/// Share of a declared input window a request may fill, in percent: the
/// token estimate is conservative, but no tokenizer is shipped to prove it.
const WINDOW_SHARE: usize = 90;
/// Chunks smaller than this would spend most of each request on the prompt.
const MIN_CHUNK_TOKENS: usize = 256;

/// How large one chunk may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Limit {
    /// Unicode scalar values.
    pub scalars: usize,
    /// Estimated tokens, in fifths of a token (see [`weight`]), when a
    /// configured model declares its input window.
    pub fifths: Option<usize>,
}

impl Limit {
    pub const DEFAULT: Self = Self {
        scalars: LIMIT,
        fifths: None,
    };
}

/// Fifths of a token one character may cost: two and a half ASCII
/// characters, or one other character, per token. Real tokenizers do better
/// on prose (about four ASCII characters) and on most Han characters, so the
/// estimate errs towards smaller chunks; Markdown tables and digits come
/// closest to it.
fn weight(ch: char) -> usize {
    if ch.is_ascii() { 2 } else { 5 }
}

/// A conservative token count for text sent to a model.
pub(super) fn estimate_tokens(text: &str) -> usize {
    text.chars().map(weight).sum::<usize>().div_ceil(5)
}

/// The chunk size for a configuration. Without `model_info.max_input_tokens`
/// the fixed 32,000-character chunks apply. With it, a chunk must also fit
/// the smallest declared window among the enabled models, after the prompt
/// (`overhead`, estimated only when needed) and a tenth of headroom. A
/// larger window never makes chunks larger: the answer repeats the chunk,
/// and output caps and request timeouts stay where they were.
pub(super) fn limit(cfg: &Value, overhead: impl FnOnce() -> Result<usize>) -> Result<Limit> {
    let Some(window) = declared_window(cfg)? else {
        return Ok(Limit::DEFAULT);
    };
    let overhead = overhead()?;
    let available = (window.saturating_mul(WINDOW_SHARE) / 100).saturating_sub(overhead);
    if available < MIN_CHUNK_TOKENS {
        return Err(Error::Config(format!(
            "llm.model_list model_info.max_input_tokens is {window}, which leaves no room for document text after the prompt (about {overhead} tokens); raise it or shorten the prompt"
        )));
    }
    Ok(Limit {
        scalars: LIMIT,
        fifths: Some(available.saturating_mul(5)),
    })
}

/// The smallest `model_info.max_input_tokens` of the enabled deployments
/// that declare one. A deployment without it takes the default chunks.
fn declared_window(cfg: &Value) -> Result<Option<usize>> {
    let Some(models) = cfg.pointer("/llm/model_list").and_then(Value::as_array) else {
        return Ok(None);
    };
    let mut smallest: Option<usize> = None;
    for model in models {
        let enabled = model
            .pointer("/litellm_params/weight")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            > 0
            && model
                .pointer("/litellm_params/model")
                .and_then(Value::as_str)
                .is_some_and(|name| !name.is_empty());
        match model.pointer("/model_info/max_input_tokens") {
            None | Some(Value::Null) => (),
            Some(value) => {
                let window = value
                    .as_u64()
                    .filter(|tokens| *tokens > 0)
                    .and_then(|tokens| usize::try_from(tokens).ok())
                    .ok_or_else(|| {
                        Error::Config(
                            "LLM model_info.max_input_tokens must be a positive integer when configured"
                                .into(),
                        )
                    })?;
                if enabled {
                    smallest = Some(smallest.map_or(window, |current| current.min(window)));
                }
            }
        }
    }
    Ok(smallest)
}

/// Whitespace-insensitive comparison form for duplicate detection.
fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A protected literal without its enclosing fence delimiters.
fn literal_core(literal: &str) -> String {
    let is_fence = |line: &str| {
        let body = line.trim_start_matches(['>', ' ']);
        body.starts_with("```") || body.starts_with("~~~")
    };
    literal
        .lines()
        .filter(|line| !is_fence(line))
        .collect::<Vec<_>>()
        .join("\n")
}

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
        crate::sort::by_key(&mut ranges, |r| r.start);
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
        self.split_within(Limit::DEFAULT)
    }

    pub fn split_within(&self, limit: Limit) -> Vec<String> {
        split(&self.text, Some(&self.prefix), limit)
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

    /// Reject an answer that re-creates the content a marker already stands
    /// for (for example transcribing a fenced block visible in an image):
    /// restore puts the literal back, so the second copy would duplicate it.
    pub fn no_literal_copies(&self, answer: &str) -> Result<()> {
        let haystack = squash(&self.without_markers(answer));
        for (_, literal) in &self.literals {
            let core = squash(&literal_core(literal));
            if core.chars().count() >= 32 && haystack.contains(&core) {
                return Err(Error::Conversion(
                    "LLM answer re-creates content a protected marker already stands for".into(),
                ));
            }
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

/// The byte offset of the first character that does not fit the limit, or
/// `None` when all of `text` fits. At least one character always fits.
fn overflow(text: &str, limit: Limit) -> Option<usize> {
    let mut fifths = 0usize;
    for (count, (at, ch)) in text.char_indices().enumerate() {
        if count == limit.scalars {
            return Some(at);
        }
        if let Some(most) = limit.fifths {
            fifths += weight(ch);
            if fifths > most && count > 0 {
                return Some(at);
            }
        }
    }
    None
}

/// Every scalar is included once; oversized prose blocks split at newline then
/// scalar boundaries. An opaque protected marker is never bisected.
fn split(text: &str, protected_prefix: Option<&str>, limit: Limit) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut rest = text;
    while let Some(overflow) = overflow(rest, limit) {
        let mut end = overflow;
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
    fn answers_that_transcribe_a_protected_literal_are_rejected() {
        let source = "Setup:\n\n```\ncd app\nbun install\nbunx vite --port 5173\n```\n\nThen open the preview.\n";
        let protected = Protected::new(source);
        let token = &protected.literals[0].0;
        // A faithful answer keeps the token and nothing of its content.
        let faithful = format!("Setup:\n\n{token}\n\nThen open the preview.\n");
        assert!(protected.no_literal_copies(&faithful).is_ok());
        assert_eq!(protected.restore(&faithful).unwrap(), source);
        // A transcription of the token's content (even re-fenced with a
        // language tag) duplicates what restore puts back.
        let copied = format!(
            "Setup:\n\n{token}\n\n```sh\ncd app\nbun install\nbunx vite --port 5173\n```\n\nThen open the preview.\n"
        );
        assert!(protected.no_literal_copies(&copied).is_err());
    }
    #[test]
    fn unicode_chunks_keep_tail_and_exact_concatenation() {
        let text = format!("{}\n\n{}TAIL", "甲".repeat(33_000), "乙".repeat(35_000));
        let chunks = split(&text, None, Limit::DEFAULT);
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

    fn models(entries: serde_json::Value) -> Value {
        serde_json::json!({"llm":{"model_list":entries}})
    }

    #[test]
    fn a_declared_window_shrinks_chunks_and_never_grows_them() {
        let none = |_: usize| -> Result<usize> { panic!("no window, no prompt estimate") };
        // Nothing declared: the fixed chunks, without building a prompt.
        for cfg in [
            serde_json::json!({}),
            models(
                serde_json::json!([{"model_name":"default","litellm_params":{"model":"openai/a"}}]),
            ),
            models(
                serde_json::json!([{"model_name":"default","litellm_params":{"model":"openai/a"},"model_info":{"max_input_tokens":null}}]),
            ),
        ] {
            assert_eq!(limit(&cfg, || none(0)).unwrap(), Limit::DEFAULT);
        }
        // The smallest enabled window wins; a disabled one does not count.
        let cfg = models(serde_json::json!([
            {"model_name":"default","litellm_params":{"model":"ollama/small"},"model_info":{"max_input_tokens":8192}},
            {"model_name":"default","litellm_params":{"model":"openai/large"},"model_info":{"max_input_tokens":1000000}},
            {"model_name":"default","litellm_params":{"model":"openai/off","weight":0},"model_info":{"max_input_tokens":1000}},
            {"model_name":"default","litellm_params":{"model":"openai/plain"}}
        ]));
        let sized = limit(&cfg, || Ok(700)).unwrap();
        assert_eq!(sized.scalars, LIMIT);
        assert_eq!(sized.fifths, Some((8192 * 90 / 100 - 700) * 5));
        // A huge window keeps the default size.
        let huge = models(
            serde_json::json!([{"model_name":"default","litellm_params":{"model":"openai/a"},"model_info":{"max_input_tokens":2000000}}]),
        );
        let text = "word ".repeat(20_000);
        let chunks = split(&text, None, limit(&huge, || Ok(500)).unwrap());
        assert_eq!(chunks, split(&text, None, Limit::DEFAULT));
        // Invalid or too small windows are explicit errors.
        for bad in [
            serde_json::json!(0),
            serde_json::json!(-5),
            serde_json::json!("8k"),
            serde_json::json!(1.5),
        ] {
            let cfg = models(
                serde_json::json!([{"model_name":"default","litellm_params":{"model":"openai/a"},"model_info":{"max_input_tokens":bad}}]),
            );
            let Err(Error::Config(message)) = limit(&cfg, || Ok(10)) else {
                panic!("{bad} must be refused");
            };
            assert!(message.contains("positive integer"), "{message}");
        }
        let tiny = models(
            serde_json::json!([{"model_name":"default","litellm_params":{"model":"openai/a"},"model_info":{"max_input_tokens":600}}]),
        );
        let Err(Error::Config(message)) = limit(&tiny, || Ok(400)) else {
            panic!("a window the prompt fills must be refused");
        };
        assert!(
            message.contains("600") && message.contains("400"),
            "{message}"
        );
    }

    #[test]
    fn window_sized_chunks_fit_the_estimate_for_ascii_and_wide_text() {
        let limit = Limit {
            scalars: LIMIT,
            fifths: Some(2_000 * 5),
        };
        for text in [
            format!(
                "{}\n\n{}",
                "Plain English prose. ".repeat(2_000),
                "| a | 1 |\n".repeat(900)
            ),
            format!(
                "{}\n{}",
                "中文段落内容。".repeat(1_500),
                "混合 mixed 文本 text\n".repeat(400)
            ),
            "x".repeat(30_000),
        ] {
            let chunks = split(&text, None, limit);
            assert!(chunks.len() > 1);
            assert_eq!(chunks.concat(), text);
            for chunk in &chunks {
                assert!(
                    estimate_tokens(chunk) <= 2_000,
                    "{}",
                    estimate_tokens(chunk)
                );
                assert!(!chunk.is_empty());
            }
        }
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("中文"), 2);
        assert_eq!(estimate_tokens(""), 0);
        // Protected markers stay whole under a token limit too.
        let source = format!("{}`code`{}", "y".repeat(4_990), "z".repeat(6_000));
        let protected = Protected::new(&source);
        let chunks = protected.split_within(Limit {
            scalars: LIMIT,
            fifths: Some(2_000 * 5),
        });
        assert_eq!(chunks.concat(), protected.text);
        assert_eq!(protected.restore(&chunks.concat()).unwrap(), source);
        assert_eq!(
            chunks
                .iter()
                .filter(|chunk| chunk.contains(&protected.literals[0].0))
                .count(),
            1
        );
    }
}
