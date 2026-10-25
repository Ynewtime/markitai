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

/// Whether the link whose address marker starts at `at` (right after `](`)
/// is written as a picture: a `!` just before the `[` that opens its words.
fn pictured_link(text: &str, at: usize) -> bool {
    let bytes = text.as_bytes();
    let Some(mut index) = at
        .checked_sub(2)
        .filter(|&close| bytes.get(close) == Some(&b']'))
    else {
        return false;
    };
    let mut depth = 0usize;
    loop {
        match bytes[index] {
            b']' => depth += 1,
            b'[' => {
                depth -= 1;
                if depth == 0 {
                    return index > 0 && bytes[index - 1] == b'!';
                }
            }
            _ => {}
        }
        if index == 0 {
            return false;
        }
        index -= 1;
    }
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

/// Where a dropped marker goes back.
enum Place {
    Start,
    End,
    /// Before the answer's only line equal to this source line.
    Before(String),
}

pub(super) struct Protected {
    pub text: String,
    prefix: String,
    literals: Vec<(String, String)>,
    /// Whether the literal at the same index stood alone on its line: a
    /// comment or picture that is a block of its own.
    blocks: Vec<bool>,
    /// Whether the literal at the same index is a link's address, written
    /// between the `](` and `)` of a link whose text the model sees.
    addresses: Vec<bool>,
    pub page_starts: Vec<(usize, usize)>,
    ranges: Vec<std::ops::Range<usize>>,
}

/// How a document becomes the model's text. A Provider Batch plan records
/// the rules its request was written with, so an answer collected after an
/// upgrade is checked and restored against the same markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rules {
    /// markitai 2.0.0: a link and every HTML tag are one marker each.
    Whole,
    /// From 2.0.1: a link's words and attribute-less styling tags stay
    /// readable; only the link's address is a marker.
    Readable,
}

impl Protected {
    pub fn new(source: &str) -> Self {
        Self::with_rules(source, Rules::Readable)
    }

    pub(super) fn with_rules(source: &str, rules: Rules) -> Self {
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
        // One styling tag's opening and closing stay one pair across the code
        // and other blocks between them.
        let mut styled = Styled::default();
        for range in blocks
            .into_iter()
            .chain(std::iter::once(source.len()..source.len()))
        {
            inline_ranges(
                source,
                cursor,
                range.start,
                &mut ranges,
                0,
                rules,
                &mut styled,
            );
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
        let mut blocks = Vec::new();
        let mut addresses = Vec::new();
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
            blocks.push(stands_alone(source, &range));
            addresses.push(
                source[..range.start].ends_with("](") && source[range.end..].starts_with(')'),
            );
            cursor = range.end;
        }
        text.push_str(&source[cursor..]);
        Self {
            text,
            prefix,
            literals,
            blocks,
            addresses,
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

    fn tokens(&self, text: &str) -> Vec<String> {
        text.match_indices(&self.prefix)
            .map(|(start, _)| {
                let end = text[start..]
                    .find('⟧')
                    .map(|end| start + end + '⟧'.len_utf8())
                    .unwrap_or(text.len());
                text[start..end].to_owned()
            })
            .collect()
    }

    /// Put back markers the answer dropped where only one place fits, as
    /// Python markitai's `restore_dropped_placeholders` does: cleanup models
    /// (Gemini Flash in tool-call mode, Claude Haiku 4.5) drop the page marker
    /// that opens a one-page PDF or the screenshot comment that closes it as
    /// noise, and rejecting the whole answer for that pays for a second
    /// request. Only a marker that stood alone on its line goes back: at the
    /// start or the end of the chunk, or between neighbours that survived (the
    /// next line of the source occurs once in the answer, with the source's
    /// previous line right above it). A moved, altered or repeated marker, a
    /// lost inline one (code, a link, an inline picture) or one whose
    /// neighbours were rewritten stays missing, so `validate` rejects the
    /// answer; the content guards still judge the text. `None` when nothing
    /// was put back.
    pub fn repair(&self, original: &str, answer: &str) -> Option<String> {
        let expected = self.tokens(original);
        let actual = self.tokens(answer);
        if expected == actual {
            return None;
        }
        // The answer's markers must be the source's in order, some left out.
        let mut missing = Vec::new();
        let mut next = 0;
        for token in &expected {
            if actual.get(next) == Some(token) {
                next += 1;
            } else {
                missing.push(token.clone());
            }
        }
        if next != actual.len() || missing.is_empty() {
            return None;
        }
        let lines = |text: &str| -> Vec<String> {
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_owned)
                .collect()
        };
        let source = lines(original);
        let mut repaired = answer.to_owned();
        for token in &missing {
            let block = self
                .literals
                .iter()
                .position(|(literal, _)| literal == token)
                .is_some_and(|index| self.blocks[index]);
            let at = source.iter().position(|line| line == token)?;
            if !block {
                return None;
            }
            // Neighbours that are themselves still missing do not anchor.
            let present =
                |line: &&String| !missing.contains(line) || repaired.contains(line.as_str());
            let above = source[..at].iter().rev().find(present).cloned();
            let below = source[at + 1..].iter().find(present).cloned();
            let current = lines(&repaired);
            // At the start or the end of the chunk only one place fits.
            let place = match (&above, &below) {
                (None, _) => Some(Place::Start),
                (_, None) => Some(Place::End),
                (Some(_), Some(below)) => {
                    let found: Vec<usize> = current
                        .iter()
                        .enumerate()
                        .filter(|(_, line)| *line == below)
                        .map(|(index, _)| index)
                        .collect();
                    match found.as_slice() {
                        [index]
                            if index.checked_sub(1).map(|up| &current[up]) == above.as_ref() =>
                        {
                            Some(Place::Before(below.clone()))
                        }
                        _ => None,
                    }
                }
            }?;
            repaired = match place {
                Place::Start => format!("{token}\n\n{}", repaired.trim_start()),
                Place::End => format!("{}\n\n{token}\n", repaired.trim_end()),
                Place::Before(below) => {
                    // The line of `below` in the answer, by its byte offset.
                    let mut offset = 0;
                    let mut start = None;
                    for line in repaired.split_inclusive('\n') {
                        if line.trim() == below {
                            start = Some(offset);
                            break;
                        }
                        offset += line.len();
                    }
                    let start = start?;
                    format!("{}{token}\n\n{}", &repaired[..start], &repaired[start..])
                }
            };
        }
        (self.tokens(&repaired) == expected).then_some(repaired)
    }

    pub fn validate(&self, original: &str, answer: &str) -> Result<()> {
        let expected = self.tokens(original);
        let actual = self.tokens(answer);
        if expected == actual {
            return self.links_kept(answer);
        }
        // Name what went wrong and which markers, as Python markitai does:
        // the user sees what was at stake, and a resend tells the model.
        let mut left: Vec<&String> = actual.iter().collect();
        let mut dropped = Vec::new();
        for token in &expected {
            match left.iter().position(|seen| *seen == token) {
                Some(index) => {
                    left.remove(index);
                }
                None => dropped.push(token),
            }
        }
        let (what, tokens): (&str, Vec<&String>) = if !dropped.is_empty() {
            ("dropped", dropped)
        } else if !left.is_empty() {
            ("repeated or changed", left)
        } else {
            let moved = expected
                .iter()
                .zip(&actual)
                .filter(|(want, got)| want != got)
                .map(|(want, _)| want)
                .collect();
            ("moved", moved)
        };
        let mut names: Vec<String> = tokens
            .iter()
            .take(5)
            .map(|token| self.describe(token))
            .collect();
        if tokens.len() > 5 {
            names.push(format!("{} more", tokens.len() - 5));
        }
        Err(Error::Conversion(format!(
            "the answer {what} {} of the {} markers that keep the document's pages, pictures, links and code exactly as they are ({})",
            tokens.len(),
            expected.len(),
            names.join("; ")
        )))
    }

    /// A link's address marker must stay inside its link (`](⟦…⟧)`): out of
    /// it, the address would come back as bare text after the words.
    fn links_kept(&self, answer: &str) -> Result<()> {
        let loose: Vec<String> = self
            .literals
            .iter()
            .zip(&self.addresses)
            .filter(|(_, address)| **address)
            .filter(|((token, _), _)| {
                answer.match_indices(token.as_str()).any(|(at, _)| {
                    !answer[..at].ends_with("](") || !answer[at + token.len()..].starts_with(')')
                })
            })
            .map(|((token, _), _)| self.describe(token))
            .collect();
        if !loose.is_empty() {
            return Err(Error::Conversion(format!(
                "the answer took {} of the document's link addresses out of their links ({}); keep each as [text](marker)",
                loose.len(),
                loose.join("; ")
            )));
        }
        // `![words](marker)` keeps the marker in place but would restore the
        // link as a picture of its address.
        let pictured: Vec<String> = self
            .literals
            .iter()
            .zip(&self.addresses)
            .filter(|(_, address)| **address)
            .filter(|((token, _), _)| {
                let was = |text: &str| {
                    text.match_indices(token.as_str())
                        .any(|(at, _)| pictured_link(text, at))
                };
                was(answer) && !was(&self.text)
            })
            .map(|((token, _), _)| self.describe(token))
            .collect();
        if pictured.is_empty() {
            return Ok(());
        }
        Err(Error::Conversion(format!(
            "the answer turned {} of the document's links into pictures ({}); keep each as [text](marker)",
            pictured.len(),
            pictured.join("; ")
        )))
    }

    /// A marker as a person and a model can tell it: what it stands for,
    /// then the marker itself.
    fn describe(&self, token: &str) -> String {
        let literal = self
            .literals
            .iter()
            .find(|(marker, _)| marker == token)
            .map_or("", |(_, literal)| literal.trim());
        let short = |text: &str| {
            let text: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
            if text.chars().count() > 40 {
                format!("{}…", text.chars().take(40).collect::<String>())
            } else {
                text
            }
        };
        let address = self
            .literals
            .iter()
            .zip(&self.addresses)
            .any(|((marker, _), address)| *address && marker == token);
        let what = if address {
            format!("the address of a link ({})", short(literal))
        } else if let Some(page) = literal
            .strip_prefix("<!-- Page number: ")
            .and_then(|rest| rest.strip_suffix("-->"))
        {
            format!("the start of page {}", page.trim())
        } else if let Some(page) = literal
            .strip_prefix("<!-- ![Page ")
            .and_then(|rest| rest.split_once(']'))
            .map(|(page, _)| page)
        {
            format!("the screenshot of page {page}")
        } else if let Some(alt) = literal
            .strip_prefix("![")
            .and_then(|rest| rest.split_once("]("))
            .map(|(alt, _)| alt)
        {
            format!("the picture “{}”", short(alt))
        } else if let Some(text) = literal
            .strip_prefix('[')
            .and_then(|rest| rest.split_once("]("))
            .map(|(text, _)| text)
        {
            format!("the link “{}”", short(text))
        } else if literal.starts_with("```") || literal.starts_with("~~~") {
            "a code block".into()
        } else if literal.starts_with('`') {
            format!("the code {}", short(literal))
        } else if literal.starts_with('$') {
            "a formula".into()
        } else if literal.is_empty() {
            "a marker".into()
        } else {
            format!("“{}”", short(literal))
        };
        format!("{what} {token}")
    }

    /// Reject an answer that re-creates the content a marker already stands
    /// for (for example transcribing a fenced block visible in an image):
    /// restore puts the literal back, so the second copy would duplicate it.
    /// Text the source itself shows (a link whose words are its address) is
    /// no copy: only more of it than the source shows is.
    pub fn no_literal_copies(&self, answer: &str) -> Result<()> {
        let haystack = squash(&self.without_markers(answer));
        let shown = squash(&self.without_markers(&self.text));
        for (_, literal) in &self.literals {
            let core = squash(&literal_core(literal));
            if core.chars().count() >= 32
                && haystack.matches(&core).count() > shown.matches(&core).count()
            {
                return Err(Error::Conversion(
                    "the model's answer wrote out again a part of the document that is kept exactly as it is (code, a link, an image), which would have doubled it, so it was not used".into(),
                ));
            }
        }
        Ok(())
    }

    /// The pictures (by literal index) that are the scans of the `unread`
    /// pages: the picture blocks before the screenshot comment
    /// (`<!-- ![Page N](…) -->`) that closes page N.
    pub fn scan_pictures(&self, unread: &[usize]) -> Vec<usize> {
        let mut found = Vec::new();
        let mut pictures = Vec::new();
        for (index, ((_, literal), &block)) in self.literals.iter().zip(&self.blocks).enumerate() {
            let literal = literal.trim();
            if block && literal.starts_with("![") {
                pictures.push(index);
            } else if let Some(page) = literal
                .strip_prefix("<!-- ![Page ")
                .and_then(|rest| rest.split_once(']'))
                .and_then(|(number, _)| number.parse::<usize>().ok())
            {
                if unread.contains(&page) {
                    found.append(&mut pictures);
                }
                pictures.clear();
            }
        }
        found
    }

    /// The pages (by number) whose screenshot comment the `answer` follows
    /// with text of its own before the next protected token: the pages the
    /// model wrote something for, as opposed to one it skipped.
    pub fn pages_with_text(&self, answer: &str) -> Vec<usize> {
        let mut pages = Vec::new();
        for (token, literal) in &self.literals {
            let Some(page) = literal
                .trim()
                .strip_prefix("<!-- ![Page ")
                .and_then(|rest| rest.split_once(']'))
                .and_then(|(number, _)| number.parse::<usize>().ok())
            else {
                continue;
            };
            let Some(at) = answer.find(token.as_str()) else {
                continue;
            };
            let after = &answer[at + token.len()..];
            let span = after.find(&self.prefix).map_or(after, |end| &after[..end]);
            if span.chars().any(char::is_alphanumeric) {
                pages.push(page);
            }
        }
        pages
    }

    pub fn restore(&self, answer: &str) -> Result<String> {
        self.restore_hiding(answer, &[])
    }

    /// [`Protected::restore`], writing the literals in `hidden` (indices) as
    /// HTML comments, so a picture stays in the file without being shown.
    pub fn restore_hiding(&self, answer: &str, hidden: &[usize]) -> Result<String> {
        self.validate(&self.text, answer)?;
        let mut rest = answer;
        let mut output = String::with_capacity(answer.len());
        for (index, ((token, literal), &block)) in
            self.literals.iter().zip(&self.blocks).enumerate()
        {
            let Some(at) = rest.find(token) else {
                unreachable!("validated marker")
            };
            let hidden_literal;
            let literal = if hidden.contains(&index) {
                hidden_literal = format!("<!-- {} -->", literal.trim());
                &hidden_literal
            } else {
                literal
            };
            output.push_str(&rest[..at]);
            if block && !output.is_empty() && !output.ends_with('\n') {
                // The model glued the block to the text before it: it needs
                // a blank line to stay a block of its own.
                let kept = output.trim_end_matches([' ', '\t']).len();
                output.truncate(kept);
                if !output.is_empty() && !output.ends_with('\n') {
                    output.push_str("\n\n");
                }
            }
            output.push_str(literal);
            let end = at + token.len();
            let skip = usize::from(literal.ends_with('\n') && rest[end..].starts_with('\n'));
            rest = &rest[end + skip..];
            if block && !literal.ends_with('\n') {
                let text = rest.trim_start_matches([' ', '\t']);
                if !text.is_empty() && !text.starts_with('\n') {
                    rest = text;
                    output.push_str("\n\n");
                }
            }
        }
        output.push_str(rest);
        Ok(output)
    }
}

/// Whether the range is one whole line of `source`, and an HTML comment or a
/// picture: a block (a page marker, a screenshot reference) rather than a
/// part of a sentence.
fn stands_alone(source: &str, range: &std::ops::Range<usize>) -> bool {
    let literal = source[range.clone()].trim_end_matches('\n');
    let single = !literal.contains('\n');
    let begins = range.start == 0 || source[..range.start].ends_with('\n');
    let ends = range.end >= source.len()
        || source[range.end..].starts_with('\n')
        || literal.len() < range.len();
    single
        && begins
        && ends
        && ((literal.starts_with("<!--") && literal.ends_with("-->"))
            || (literal.starts_with("![") && literal.ends_with(')')))
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

/// HTML tags that only style the text they wrap. Without attributes they
/// stay visible: as opaque markers around a phrase (`⟦…⟧text⟦…⟧`) cleanup
/// models drop them, which costs a second request, and losing one changes no
/// word of the text (Python markitai never hid them). An opening tag with
/// attributes is protected, and so is the closing tag that ends it: a pair is
/// either visible or hidden as a whole. Returns the lower-case name, whether
/// the tag closes, and whether it carries anything after its name.
fn styling_tag(tag: &str) -> Option<(String, bool, bool)> {
    let inner = tag.strip_prefix('<')?.strip_suffix('>')?;
    let (closing, inner) = match inner.strip_prefix('/') {
        Some(rest) => (true, rest),
        None => (false, inner),
    };
    let width = inner
        .find(|ch: char| ch.is_whitespace() || ch == '/')
        .unwrap_or(inner.len());
    let name = inner[..width].to_ascii_lowercase();
    ["u", "ins", "mark", "small", "b", "strong", "i", "em"]
        .contains(&name.as_str())
        .then_some((name, closing, width < inner.len()))
}

/// How deeply link texts inside link texts are scanned before a link is
/// kept whole again.
const LINK_DEPTH: u8 = 4;

/// Per styling tag name, whether each tag still open carried attributes.
type Styled = std::collections::HashMap<String, Vec<bool>>;

fn inline_ranges(
    full: &str,
    start: usize,
    end: usize,
    ranges: &mut Vec<std::ops::Range<usize>>,
    depth: u8,
    rules: Rules,
    styled: &mut Styled,
) {
    let source = &full[start..end];
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
            } else if let Some(end) = next
                && rules == Rules::Readable
                && let Some((name, closing, attributed)) = styling_tag(&source[i..end])
            {
                let open = styled.entry(name).or_default();
                let hidden = if closing {
                    open.pop().unwrap_or(false) || attributed
                } else {
                    open.push(attributed);
                    attributed
                };
                if !hidden {
                    i = end;
                    continue;
                }
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
                    // A link keeps its text readable and only its address
                    // (`[text](⟦…⟧)`): hidden whole, its words vanish from
                    // the sentence and a model reading the page image writes
                    // them back without the marker.
                    if let Some(after) = next
                        && open == i
                        && bytes[label] == b'('
                        && after > label + 2
                        && depth < LINK_DEPTH
                        && rules == Rules::Readable
                    {
                        inline_ranges(
                            full,
                            start + open + 1,
                            start + label - 1,
                            ranges,
                            depth + 1,
                            rules,
                            styled,
                        );
                        ranges.push(start + label + 1..start + after - 1);
                        i = after;
                        continue;
                    }
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
    #[test]
    fn markers_dropped_at_the_edges_of_a_one_page_pdf_go_back() {
        let source = "<!-- Page number: 1 -->\n\n# مقدمة\n\nRust text.\n\n<!-- ![Page 1](.markitai/screenshots/a.pdf.page0001.jpg) -->\n";
        let protected = Protected::new(source);
        let text = &protected.text;
        let tokens = protected.tokens(text);
        assert_eq!(tokens.len(), 2);
        let body = "# مقدمة\n\nRust text, cleaned.\n";
        // Both dropped, then each one alone.
        for answer in [
            body.to_owned(),
            format!("{}\n\n{body}", tokens[0]),
            format!("{body}\n{}\n", tokens[1]),
        ] {
            let repaired = protected
                .repair(text, &answer)
                .unwrap_or_else(|| panic!("not repaired: {text:?} / {answer:?}"));
            protected.validate(text, &repaired).unwrap();
            assert!(repaired.starts_with(&tokens[0]), "{repaired}");
            assert!(repaired.trim_end().ends_with(&tokens[1]), "{repaired}");
            assert!(repaired.contains("Rust text, cleaned."));
        }
        // A complete answer needs nothing.
        assert!(protected.repair(text, text).is_none());
    }

    #[test]
    fn a_dropped_marker_between_surviving_neighbours_goes_back_in_its_place() {
        let source = "Intro line.\n\n<!-- Page number: 2 -->\n\nSecond page.\n";
        let protected = Protected::new(source);
        let text = &protected.text;
        let token = protected.tokens(text).remove(0);
        let repaired = protected
            .repair(text, "Intro line.\n\nSecond page.\n")
            .unwrap();
        assert_eq!(
            repaired,
            format!("Intro line.\n\n{token}\n\nSecond page.\n")
        );
        // Its neighbours were rewritten: no single place fits.
        assert!(protected.repair(text, "Intro.\n\nSecond page.\n").is_none());
        assert!(
            protected
                .repair(text, "Second page.\n\nIntro line.\n\nSecond page.\n")
                .is_none()
        );
    }

    #[test]
    fn moved_altered_or_inline_markers_are_not_repaired() {
        let source =
            "<!-- Page number: 1 -->\n\nRun `cargo test` now.\n\n<!-- Page number: 2 -->\n\nEnd.\n";
        let protected = Protected::new(source);
        let text = &protected.text;
        let tokens = protected.tokens(text);
        assert_eq!(tokens.len(), 3);
        // The inline code marker is gone: not a block, so the answer stays rejected.
        let lost_code = format!("{}\n\nRun now.\n\n{}\n\nEnd.\n", tokens[0], tokens[2]);
        assert!(protected.repair(text, &lost_code).is_none());
        // Swapped markers.
        let moved = format!(
            "{}\n\nRun {} now.\n\n{}\n\nEnd.\n",
            tokens[2], tokens[1], tokens[0]
        );
        assert!(protected.repair(text, &moved).is_none());
        assert!(protected.validate(text, &moved).is_err());
        // The error names what went wrong and the markers concerned.
        let error = protected
            .validate(text, &lost_code)
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            format!(
                "the answer dropped 1 of the 3 markers that keep the document's pages, pictures, links and code exactly as they are (the code `cargo test` {})",
                tokens[1]
            )
        );
        let error = protected.validate(text, &moved).unwrap_err().to_string();
        assert!(
            error.starts_with("the answer moved 2 of the 3 markers")
                && error.contains(&format!("the start of page 1 {}", tokens[0]))
                && error.contains(&format!("the start of page 2 {}", tokens[2])),
            "{error}"
        );
        let repeated = format!("{}\n\n{}\n\n{text}", tokens[0], tokens[0]);
        let error = protected.validate(text, &repeated).unwrap_err().to_string();
        assert!(
            error.starts_with("the answer repeated or changed 2 of the 3"),
            "{error}"
        );
    }

    use super::*;

    #[test]
    fn the_scan_of_an_unread_page_is_hidden_as_a_comment_and_other_pictures_stay() {
        let source = "![Image on page 1](.markitai/assets/a.jpg)\n\n<!-- ![Page 1](.markitai/screenshots/p1.jpg) -->\n\nNative text.\n\n![A chart](.markitai/assets/b.jpg)\n\n<!-- ![Page 2](.markitai/screenshots/p2.jpg) -->\n";
        let protected = Protected::new(source);
        // Page 1 is unread: its picture is the first literal; page 2 is not.
        assert_eq!(protected.scan_pictures(&[1]), [0]);
        assert_eq!(protected.scan_pictures(&[2]), [2]);
        assert!(protected.scan_pictures(&[]).is_empty());
        let answer = protected.text.replace("Native text.", "Transcribed.");
        let shown = protected.restore_hiding(&answer, &[]).unwrap();
        let hidden = protected.restore_hiding(&answer, &[0]).unwrap();
        assert!(
            shown.contains("\n![Image on page 1](") || shown.starts_with("![Image on page 1](")
        );
        assert!(hidden.starts_with("<!-- ![Image on page 1](.markitai/assets/a.jpg) -->"));
        assert!(hidden.contains("\n![A chart](.markitai/assets/b.jpg)"));
    }

    #[test]
    fn only_the_scans_of_pages_the_model_wrote_text_for_are_hidden() {
        let source = "![Image on page 1](.markitai/assets/a.jpg)\n\n<!-- ![Page 1](.markitai/screenshots/p1.jpg) -->\n\n![Image on page 2](.markitai/assets/b.jpg)\n\n<!-- ![Page 2](.markitai/screenshots/p2.jpg) -->\n";
        let protected = Protected::new(source);
        let token = |n: usize| protected.literals[n].0.clone();
        // Text after page 1's comment, none after page 2's.
        let answer = format!(
            "{}\n\n{}\n\nWords of page one.\n\n{}\n\n{}\n",
            token(0),
            token(1),
            token(2),
            token(3)
        );
        assert_eq!(protected.pages_with_text(&answer), [1]);
        let unread = [1, 2];
        let read: Vec<usize> = unread
            .iter()
            .copied()
            .filter(|page| protected.pages_with_text(&answer).contains(page))
            .collect();
        let hidden = protected.scan_pictures(&read);
        assert_eq!(hidden, [0]);
        let restored = protected.restore_hiding(&answer, &hidden).unwrap();
        assert!(restored.contains("<!-- ![Image on page 1](.markitai/assets/a.jpg) -->"));
        assert!(restored.contains("\n![Image on page 2](.markitai/assets/b.jpg)"));
    }

    #[test]
    fn a_block_the_model_glued_to_text_gets_its_own_lines_back() {
        let source = "<!-- Page number: 1 -->\n\n# Language comparison\n\nEach language has trade-offs.\n\n<!-- ![Page 1](.markitai/screenshots/a.jpg) -->\n\n![Image](.markitai/assets/x.jpg)\n";
        let protected = Protected::new(source);
        assert_eq!(protected.blocks, [true, true, true]);
        let token = |n: usize| protected.literals[n].0.clone();
        // The model's answer with every token run into its neighbour.
        let glued = format!(
            "{}# Language comparison\n\nEach language has trade-offs.{}{} After",
            token(0),
            token(1),
            token(2)
        );
        assert_eq!(
            protected.restore(&glued).unwrap(),
            "<!-- Page number: 1 -->\n\n# Language comparison\n\nEach language has trade-offs.\n\n<!-- ![Page 1](.markitai/screenshots/a.jpg) -->\n\n![Image](.markitai/assets/x.jpg)\n\nAfter"
        );
        // Tokens the model kept on their own lines are not changed.
        assert_eq!(protected.restore(&protected.text).unwrap(), source);
        let single = format!("{}\n# T\n\nText\n{}\n{}\n", token(0), token(1), token(2));
        assert_eq!(
            protected.restore(&single).unwrap(),
            format!(
                "{}\n# T\n\nText\n{}\n{}\n",
                protected.literals[0].1, protected.literals[1].1, protected.literals[2].1
            )
        );
    }

    #[test]
    fn a_link_keeps_its_words_visible_and_only_its_address_hidden() {
        let source = "Highlights:\n\n- <u>Readings from the ferry ramp are provisional.</u>\n- [North pier tables are online.](https://example.org/tides?pier=north \"Tables\") They list every reading.\n- See [`gauge.csv`](./data/gauge.csv) and ![chart](./chart.png).\n";
        let protected = Protected::new(source);
        let text = &protected.text;
        // Styling tags and link words reach the model as they are.
        assert!(
            text.contains("- <u>Readings from the ferry ramp are provisional.</u>\n"),
            "{text}"
        );
        assert!(
            text.contains("- [North pier tables are online.](⟦MKTI:"),
            "{text}"
        );
        assert!(!text.contains("example.org"), "{text}");
        // Code inside a link's text and a picture stay whole literals.
        assert!(!text.contains("gauge.csv"), "{text}");
        assert!(!text.contains("chart.png"), "{text}");
        assert_eq!(protected.restore(text).unwrap(), source);
        // An answer that rewords the link and drops the underline keeps the address.
        let answer = text.replace("<u>", "").replace("</u>", "").replace(
            "North pier tables are online.",
            "The north pier tables are online.",
        );
        assert_eq!(
            protected.restore(&answer).unwrap(),
            source
                .replace("<u>", "")
                .replace("</u>", "")
                .replace("[North pier", "[The north pier")
        );
        // An address taken out of its link is refused, naming it.
        let token = protected.tokens(text).remove(0);
        let loose = text.replace(&format!("]({token})"), &format!("] {token}"));
        let error = protected.validate(text, &loose).unwrap_err().to_string();
        assert!(
            error.starts_with("the answer took 1 of the document's link addresses out of their links (the address of a link (https://example.org/tides?pier=north"),
            "{error}"
        );
        assert!(protected.restore(&loose).is_err());
    }

    #[test]
    fn styling_tags_with_attributes_and_meaningful_tags_stay_protected() {
        let source = "x<sup>2</sup> and H<sub>2</sub>O, <s>old</s>, a<br>b, <u class=\"k\">kept</u>, <B>bold</B>.\n";
        let protected = Protected::new(source);
        let text = &protected.text;
        for hidden in ["<sup>", "<sub>", "<s>", "<br>", "<u class"] {
            assert!(!text.contains(hidden), "{hidden}: {text}");
        }
        assert!(text.contains("<B>bold</B>"), "{text}");
        assert_eq!(protected.restore(text).unwrap(), source);
    }

    #[test]
    fn an_attributed_styling_tag_is_hidden_with_its_closing_tag() {
        let source = "a <B class=x>c</B> d <u>plain</u> <b title=\"t\">x <b>y</b> z</b> </i> e.\n";
        let protected = Protected::new(source);
        let text = &protected.text;
        // the attributed pairs are hidden whole; the inner plain pair and a
        // stray closer stay visible
        assert!(
            !text.contains("<B class") && !text.contains("</B>"),
            "{text}"
        );
        assert!(text.contains("<u>plain</u>"), "{text}");
        assert!(text.contains("<b>y</b>"), "{text}");
        assert!(
            !text.contains("title=") && !text.contains("z</b>"),
            "{text}"
        );
        assert!(text.contains(" </i> e."), "{text}");
        assert_eq!(protected.tokens(text).len(), 4, "{text}");
        assert_eq!(protected.restore(text).unwrap(), source);
        // A model that drops the visible pairs still keeps both ends of the hidden ones.
        let answer = text.replace("<u>", "").replace("</u>", "");
        assert_eq!(
            protected.restore(&answer).unwrap(),
            source.replace("<u>", "").replace("</u>", "")
        );
    }

    #[test]
    fn a_comment_inside_a_sentence_is_not_a_block() {
        let source = "A sentence <!-- note --> goes on.\n\n- ![x](y)\n";
        let protected = Protected::new(source);
        assert_eq!(protected.blocks, [false, false]);
        assert_eq!(protected.restore(&protected.text).unwrap(), source);
    }
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
    fn a_link_whose_words_are_its_address_is_no_copy_of_it() {
        // Readable links hide only the address; here the words are the same
        // address, so an unchanged answer shows it once, as the source does.
        let url = "https://example.test/a/very/long/path/that/exceeds/thirty-two-characters";
        let source = format!("See [{url}]({url}) for details.\n");
        let protected = Protected::new(&source);
        assert!(protected.no_literal_copies(&protected.text).is_ok());
        assert_eq!(protected.restore(&protected.text).unwrap(), source);
        // Writing the address out once more, beside the link, is a copy.
        let copied = protected
            .text
            .replace("for details", &format!("at {url} for details"));
        assert!(protected.no_literal_copies(&copied).is_err());
    }
    #[test]
    fn an_answer_that_makes_a_link_a_picture_is_refused() {
        let source =
            "Read [the release](https://e.test/a) now, or [the [old] notes](https://e.test/b).\n";
        let protected = Protected::new(source);
        assert!(protected.validate(&protected.text, &protected.text).is_ok());
        for words in ["[the release]", "[the [old] notes]"] {
            let answer = protected.text.replace(words, &format!("!{words}"));
            let refused = protected
                .validate(&protected.text, &answer)
                .unwrap_err()
                .to_string();
            assert!(refused.contains("into pictures"), "{refused}");
        }
        // A link whose words are a picture stays a link.
        let badge = "[![build](badge.png)](https://e.test/ci) ok\n";
        let protected = Protected::new(badge);
        assert!(protected.validate(&protected.text, &protected.text).is_ok());
        assert_eq!(protected.restore(&protected.text).unwrap(), badge);
    }
    #[test]
    fn an_attributed_styling_pair_split_by_a_code_block_stays_one_pair() {
        let source = "Text <u class=\"x\">before\n\n```\ncode line\n```\n\nafter</u> end.\n";
        let protected = Protected::new(source);
        // both ends hidden, as for a pair on one line
        assert!(
            !protected.text.contains("<u class") && !protected.text.contains("</u>"),
            "{}",
            protected.text
        );
        assert_eq!(protected.restore(&protected.text).unwrap(), source);
        // a plain pair split the same way stays visible at both ends
        let plain = "Text <u>before\n\n```\ncode\n```\n\nafter</u> end.\n";
        let protected = Protected::new(plain);
        assert!(
            protected.text.contains("<u>before") && protected.text.contains("after</u>"),
            "{}",
            protected.text
        );
        // 2.0.0's rules are unchanged: every tag is a marker
        let whole = Protected::with_rules(source, Rules::Whole);
        assert!(!whole.text.contains("</u>"));
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
