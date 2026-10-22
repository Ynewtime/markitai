//! The configuration file as text for the workbench's Edit config page. The
//! page receives the file's own text with each secret string literal replaced
//! by its masked form; what it sends back is saved exactly as written, except
//! that each masked value it kept in place is put back as the stored literal.
//! serde_json validates and parses; a small scanner locates the literals.
use super::model::masked_key;
use markitai_core::config::{api_base_key, header_map_key, secret_key};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, ops::Range};

const MASK: char = '\u{2026}';

/// What a string at some position of the file is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Plain,
    /// Under a key the redaction rule names (or anywhere beneath one), or in
    /// an `extra_http_headers` object.
    Secret,
    /// An endpoint address (`api_base`, `*_api_base`).
    Address,
}

/// One step of a JSON path.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Step {
    Key(String),
    Index(usize),
}

/// A string value of a JSON text: where it is and its literal's byte span
/// (quotes included).
#[derive(Debug)]
struct Literal {
    path: Vec<Step>,
    span: Range<usize>,
}

impl Literal {
    /// Its JSON pointer (RFC 6901).
    fn pointer(&self) -> String {
        let mut pointer = String::new();
        for step in &self.path {
            pointer.push('/');
            match step {
                Step::Key(key) => pointer.push_str(&key.replace('~', "~0").replace('/', "~1")),
                Step::Index(index) => pointer.push_str(&index.to_string()),
            }
        }
        pointer
    }
    /// What the value is: a secret anywhere beneath a key the redaction rule
    /// names or an `extra_http_headers` object; an address under `api_base`.
    fn kind(&self) -> Kind {
        let keys = || {
            self.path.iter().filter_map(|step| match step {
                Step::Key(key) => Some(key.as_str()),
                Step::Index(_) => None,
            })
        };
        if keys().any(|key| secret_key(key) || header_map_key(key)) {
            Kind::Secret
        } else if matches!(self.path.last(), Some(Step::Key(key)) if api_base_key(key)) {
            Kind::Address
        } else {
            Kind::Plain
        }
    }
}

/// A position-aware walk over a JSON text that serde_json has accepted.
struct Scanner<'a> {
    text: &'a str,
    at: usize,
    path: Vec<Step>,
    found: Vec<Literal>,
    /// The first key written twice in one object: where its second literal
    /// starts, and the key.
    duplicate: Option<(usize, String)>,
}

impl Scanner<'_> {
    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.at).copied()
    }
    fn space(&mut self) {
        while self
            .peek()
            .is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
        {
            self.at += 1;
        }
    }
    fn expect(&mut self, byte: u8) -> Option<()> {
        self.space();
        (self.peek()? == byte).then(|| self.at += 1)
    }
    /// The span of the string literal at the cursor, quotes included.
    fn string(&mut self) -> Option<Range<usize>> {
        let start = self.at;
        (self.peek()? == b'"').then_some(())?;
        self.at += 1;
        loop {
            match self.peek()? {
                b'\\' => self.at += 2,
                b'"' => {
                    self.at += 1;
                    return Some(start..self.at);
                }
                _ => self.at += 1,
            }
        }
    }
    fn value(&mut self) -> Option<()> {
        self.space();
        match self.peek()? {
            b'{' => {
                self.at += 1;
                self.space();
                if self.peek()? == b'}' {
                    self.at += 1;
                    return Some(());
                }
                let mut keys = std::collections::HashSet::new();
                loop {
                    self.space();
                    let span = self.string()?;
                    let start = span.start;
                    let key: String = serde_json::from_str(&self.text[span]).ok()?;
                    if !keys.insert(key.clone()) && self.duplicate.is_none() {
                        self.duplicate = Some((start, key.clone()));
                    }
                    self.expect(b':')?;
                    self.path.push(Step::Key(key));
                    self.value()?;
                    self.path.pop();
                    self.space();
                    match self.peek()? {
                        b',' => self.at += 1,
                        b'}' => {
                            self.at += 1;
                            return Some(());
                        }
                        _ => return None,
                    }
                }
            }
            b'[' => {
                self.at += 1;
                self.space();
                if self.peek()? == b']' {
                    self.at += 1;
                    return Some(());
                }
                let mut index = 0;
                loop {
                    self.path.push(Step::Index(index));
                    self.value()?;
                    self.path.pop();
                    index += 1;
                    self.space();
                    match self.peek()? {
                        b',' => self.at += 1,
                        b']' => {
                            self.at += 1;
                            return Some(());
                        }
                        _ => return None,
                    }
                }
            }
            b'"' => {
                let span = self.string()?;
                self.found.push(Literal {
                    path: self.path.clone(),
                    span,
                });
                Some(())
            }
            // A number, true, false or null.
            _ => {
                let start = self.at;
                while self.peek().is_some_and(|byte| {
                    !matches!(byte, b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
                }) {
                    self.at += 1;
                }
                (self.at > start).then_some(())
            }
        }
    }
}

/// Every string value of `text` (not object keys) in text order; `None` for a
/// text that is not one JSON value.
fn literals(text: &str) -> Option<Vec<Literal>> {
    scan(text).map(|scanner| scanner.found)
}

fn scan(text: &str) -> Option<Scanner<'_>> {
    let mut scanner = Scanner {
        text,
        at: 0,
        path: Vec::new(),
        found: Vec::new(),
        duplicate: None,
    };
    scanner.value()?;
    scanner.space();
    (scanner.at == text.len()).then_some(scanner)
}

/// `text` parsed as the Edit config page's JSON, or why not, as
/// `line L, column C: message`. A key written twice in one object is refused,
/// as Python markitai refuses it: JSON keeps the last value, so a secret
/// written first would be neither used nor masked.
pub(super) fn parse(text: &str) -> Result<Value, String> {
    let value = serde_json::from_str::<Value>(text).map_err(|error| position(&error))?;
    if let Some((at, key)) = scan(text).and_then(|scanner| scanner.duplicate) {
        let before = &text[..at];
        let line = before.matches('\n').count() + 1;
        let column = before
            .rsplit('\n')
            .next()
            .map_or(0, |row| row.chars().count())
            + 1;
        return Err(format!(
            "line {line}, column {column}: the key {} appears twice in one object",
            Value::String(key)
        ));
    }
    Ok(value)
}

/// The file's revision: `sha256:` and the hex SHA-256 of its bytes, or
/// `absent`.
pub(super) fn revision(bytes: Option<&[u8]>) -> String {
    match bytes {
        Some(bytes) => format!("sha256:{}", markitai_core::hex(Sha256::digest(bytes))),
        None => "absent".into(),
    }
}

/// A JSON parse error as `line L, column C: message`.
pub(super) fn position(error: &serde_json::Error) -> String {
    let message = error.to_string();
    let suffix = format!(" at line {} column {}", error.line(), error.column());
    let message = message.strip_suffix(&suffix).unwrap_or(&message);
    format!(
        "line {}, column {}: {message}",
        error.line(),
        error.column()
    )
}

/// The masked form of `text`, when it is masked.
fn masked(text: &str, kind: Kind) -> Option<String> {
    match kind {
        Kind::Plain => None,
        Kind::Secret => (!text.is_empty() && !text.starts_with("env:")).then(|| masked_key(text)),
        Kind::Address => {
            let shown = markitai_core::output::masked_url(text);
            (shown != text).then_some(shown)
        }
    }
}

/// The decoded value of a string literal.
fn decoded(text: &str, span: &Range<usize>) -> Option<String> {
    serde_json::from_str(&text[span.clone()]).ok()
}

/// Each masked literal of a JSON object text, in text order: its JSON
/// pointer, span, stored literal and the masked value GET sends. Every
/// occurrence is listed, a repeated key's too.
fn secrets(text: &str) -> Vec<(String, Range<usize>, String, String)> {
    let mut found = Vec::new();
    for literal in literals(text).unwrap_or_default() {
        let Some(value) = decoded(text, &literal.span) else {
            continue;
        };
        if let Some(shown) = masked(&value, literal.kind()) {
            found.push((
                literal.pointer(),
                literal.span.clone(),
                text[literal.span].to_owned(),
                shown,
            ));
        }
    }
    found
}

/// `text` with every span in `replacements` (ascending, disjoint) replaced.
fn replaced(text: &str, mut replacements: Vec<(Range<usize>, String)>) -> String {
    replacements.sort_by_key(|(span, _)| span.start);
    let mut result = String::with_capacity(text.len());
    let mut from = 0;
    for (span, replacement) in replacements {
        result.push_str(&text[from..span.start]);
        result.push_str(&replacement);
        from = span.end;
    }
    result.push_str(&text[from..]);
    result
}

/// The file's text with each secret literal masked, and how many.
pub(super) fn mask(text: &str) -> (String, usize) {
    let replacements: Vec<_> = secrets(text)
        .into_iter()
        .map(|(_, span, _, shown)| (span, Value::String(shown).to_string()))
        .collect();
    let count = replacements.len();
    (replaced(text, replacements), count)
}

/// The Edit config view of the file `bytes` (`None`: no file).
pub(super) fn view(path: &str, origin: &str, bytes: Option<&[u8]>, read_only: bool) -> Value {
    let (text, masked, problem) = match bytes {
        None => (Some("{}\n".to_owned()), 0, None),
        Some(bytes) => match std::str::from_utf8(bytes) {
            Err(_) => (
                None,
                0,
                Some("the configuration is not UTF-8 text".to_owned()),
            ),
            Ok(text) => match parse(text) {
                Err(problem) => (None, 0, Some(problem)),
                Ok(value) if !value.is_object() => (
                    None,
                    0,
                    Some("the configuration must be a JSON object".to_owned()),
                ),
                Ok(_) => {
                    let (text, count) = mask(text);
                    (Some(text), count, None)
                }
            },
        },
    };
    json!({"path":path,"origin":origin,"exists":bytes.is_some(),"revision":revision(bytes),
        "text":text,"masked":masked,"problem":problem,"read_only":read_only})
}

/// The client's `text` (already accepted by serde_json) with each masked
/// value it kept at the position GET sent it replaced by the stored literal
/// of `current` (the file's text); a masked value anywhere else is refused.
pub(super) fn unmask(text: &str, current: Option<&str>) -> Result<String, String> {
    // A text with a repeated key is refused before it gets here; the first
    // occurrence of a pointer is kept.
    let mut stored = HashMap::new();
    for (pointer, span, original, shown) in current.map(secrets).unwrap_or_default() {
        stored.entry(pointer).or_insert((span, original, shown));
    }
    let found = literals(text).ok_or("the configuration text could not be read")?;
    let mut replacements = Vec::new();
    for literal in found {
        let value =
            decoded(text, &literal.span).ok_or("the configuration text could not be read")?;
        let pointer = literal.pointer();
        if let Some((_, original, shown)) = stored.get(&pointer)
            && *shown == value
        {
            replacements.push((literal.span, original.clone()));
            continue;
        }
        if literal.kind() != Kind::Plain && value.contains(MASK) {
            return Err(format!(
                "{pointer}: a masked value can only be kept where it was; type the full value or an env: reference"
            ));
        }
    }
    Ok(replaced(text, replacements))
}

/// Whether anything but `llm.model_list` and `llm.providers` differs.
pub(super) fn restart_required(old: Option<&Value>, new: &Value) -> bool {
    fn rest(value: &Value) -> Value {
        let mut value = value.clone();
        if let Some(object) = value.as_object_mut()
            && let Some(llm) = object.get_mut("llm").and_then(Value::as_object_mut)
        {
            llm.remove("model_list");
            llm.remove("providers");
            if llm.is_empty() {
                object.remove("llm");
            }
        }
        value
    }
    old.is_none_or(|old| rest(old) != rest(new))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unusual layout, keys out of order, escapes in keys and values.
    const FILE: &str = "{\n\t\"notes\" : { \"password\":\"abc\",\"empty_secret\":\"\", \"title\":\"a \\\"quoted\\\" \\u00e9 value\" },\n   \"llm\":{\"providers\":[ {\"id\":\"p1\", \"provider\":\"openai\",\"api_key\":\"env:OPENAI_API_KEY\",\"api_base\":\"https://api.example/v1\"} ],\n  \"model_list\" :[{\"model_name\":\"default\",\"litellm_params\":{\n      \"model\":\"openai/gpt-4o-mini\",  \"api_key\":\"sk-test-abcdef1234567890\",\n      \"api_base\":\"https://user:pass@gateway.example/v1?key=s3cret&x=1\" }}]},\n\"fetch\":{\"playwright\":{\"extra_http_headers\":{\"X-Custom\":\"header-value-123\",\"Accept\":\"text/html\"}},\n \"cloudflare\":{\"api_token\":\"cf-token-abcdefghijkl\",\"account_id\":\"acct-1\", \"cookies\":[{\"a/b~c\":\"cookie-value-1\"}]}}, \"n\": [1, -2.5e3, true, false, null, {}, []]\n}";

    #[test]
    fn a_repeated_key_is_refused_and_every_occurrence_masked() {
        let twice = "{\"a\": {\"api_key\": \"sk-first-abcdef123456\",\n \"api_key\": \"sk-second-abcdef123456\"}}";
        assert_eq!(
            parse(twice).unwrap_err(),
            "line 2, column 2: the key \"api_key\" appears twice in one object"
        );
        // The same key in two objects is no repeat.
        assert!(parse("{\"a\": {\"k\": 1}, \"b\": {\"k\": 1}}").is_ok());
        assert!(
            parse("{\"a\": 1,}")
                .unwrap_err()
                .starts_with("line 1, column 9: ")
        );
        let (masked, count) = mask(twice);
        assert_eq!(count, 2);
        assert!(
            !masked.contains("first") && !masked.contains("second"),
            "{masked}"
        );
    }

    #[test]
    fn the_scanner_finds_every_string_value_with_its_pointer_and_span() {
        let found = literals(FILE).unwrap();
        let listed: Vec<_> = found
            .iter()
            .map(|literal| (literal.pointer(), &FILE[literal.span.clone()]))
            .collect();
        assert_eq!(listed[0], ("/notes/password".into(), "\"abc\""));
        assert_eq!(
            listed[2],
            ("/notes/title".into(), "\"a \\\"quoted\\\" \\u00e9 value\"")
        );
        assert!(listed.contains(&("/llm/providers/0/id".into(), "\"p1\"")));
        assert!(listed.contains(&(
            "/fetch/cloudflare/cookies/0/a~1b~0c".into(),
            "\"cookie-value-1\""
        )));
        // Keys are not values; numbers, booleans and null are not strings.
        assert!(!listed.iter().any(|(_, literal)| *literal == "\"notes\""));
        assert_eq!(listed.len(), 16);
        let value: Value = serde_json::from_str(FILE).unwrap();
        for literal in &found {
            assert_eq!(
                value.pointer(&literal.pointer()).unwrap(),
                &serde_json::from_str::<Value>(&FILE[literal.span.clone()]).unwrap()
            );
        }
        assert!(literals("{\"a\":1} x").is_none());
        assert!(literals("[\"\\\\\", \"x\"]").unwrap().len() == 2);
    }

    #[test]
    fn secrets_headers_and_addresses_are_masked_in_the_files_own_text() {
        let view = view("~/config.json", "user", Some(FILE.as_bytes()), false);
        let text = view["text"].as_str().unwrap().to_owned();
        // Seven masked: password, both headers, the key, the address, the
        // Cloudflare token and the cookie; not the empty value or env: references.
        assert_eq!(view["masked"], 7);
        for secret in [
            "abcdef1234567890",
            "pass@",
            "s3cret",
            "header-value-123",
            "text/html",
            "cf-token",
            "cookie-value-1",
        ] {
            assert!(!text.contains(secret), "{secret} in {text}");
        }
        // Only the masked literals differ: layout and order are the file's.
        let expected = FILE
            .replace("\"abc\"", "\"\u{2026}\"")
            .replace("\"sk-test-abcdef1234567890\"", "\"sk\u{2026}7890\"")
            .replace("\"header-value-123\"", "\"he\u{2026}-123\"")
            .replace("\"text/html\"", "\"\u{2026}ml\"")
            .replace("\"cf-token-abcdefghijkl\"", "\"cf\u{2026}ijkl\"")
            .replace("\"cookie-value-1\"", "\"co\u{2026}ue-1\"");
        let address = FILE
            .split("\"api_base\":\"https://user")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        let shown = markitai_core::output::masked_url(&format!("https://user{address}"));
        let expected = expected.replace(
            &format!("\"https://user{address}\""),
            &Value::String(shown).to_string(),
        );
        assert_eq!(text, expected);
        assert!(text.contains("env:OPENAI_API_KEY") && text.contains("\"acct-1\""));
    }

    #[test]
    fn kept_masks_come_back_as_the_stored_literals_and_moved_ones_are_refused() {
        let (masked, _) = mask(FILE);
        // Edited in place: every kept mask is restored, the rest is as sent.
        let sent = masked
            .replace("\"acct-1\"", "\"acct-2\"")
            .replace("\"\u{2026}\"", "\"typed-new-password\"")
            .replace("\n", "\r\n");
        let saved = unmask(&sent, Some(FILE)).unwrap();
        let expected = FILE
            .replace("\"acct-1\"", "\"acct-2\"")
            .replace("\"abc\"", "\"typed-new-password\"")
            .replace("\n", "\r\n");
        assert_eq!(saved, expected);
        // Unchanged text comes back byte for byte.
        assert_eq!(unmask(&masked, Some(FILE)).unwrap(), FILE);
        // A masked value moved elsewhere is refused by its pointer.
        let moved = masked.replace(
            "\"account_id\":\"acct-1\"",
            "\"secret_copy\":\"cf\u{2026}ijkl\"",
        );
        assert_eq!(
            unmask(&moved, Some(FILE)).unwrap_err(),
            "/fetch/cloudflare/secret_copy: a masked value can only be kept where it was; type the full value or an env: reference"
        );
        let swapped = masked.replace("\"X-Custom\":", "\"X-Other\":");
        assert!(
            unmask(&swapped, Some(FILE))
                .unwrap_err()
                .starts_with("/fetch/playwright/extra_http_headers/X-Other: ")
        );
        // A plain value may contain the character; without a file nothing is stored.
        let plain = masked.replace("a \\\"quoted\\\" \\u00e9 value", "To be continued\u{2026}");
        unmask(&plain, Some(FILE)).unwrap();
        assert!(unmask(&masked, None).is_err());
        assert_eq!(unmask("{\"a\": \"b\"}", None).unwrap(), "{\"a\": \"b\"}");
    }

    #[test]
    fn revisions_positions_and_restart_needs() {
        assert_eq!(revision(None), "absent");
        assert_eq!(
            revision(Some(b"{}")),
            "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        );
        let error = serde_json::from_str::<Value>("{\n  \"a\": 1\n  \"b\": 2\n}").unwrap_err();
        assert_eq!(position(&error), "line 3, column 3: expected `,` or `}`");
        let broken = view("p", "explicit", Some(b"{\"a\":"), false);
        assert_eq!(broken["text"], Value::Null);
        assert!(
            broken["problem"]
                .as_str()
                .unwrap()
                .starts_with("line 1, column 5: ")
        );
        let binary = view("p", "explicit", Some(b"{\"a\":\"\xff\"}"), false);
        assert_eq!(binary["problem"], "the configuration is not UTF-8 text");
        let absent = view("p", "default", None, true);
        assert_eq!(
            absent,
            json!({"path":"p","origin":"default","exists":false,"revision":"absent",
                "text":"{}\n","masked":0,"problem":null,"read_only":true})
        );
        let old = json!({"llm":{"enabled":true,"model_list":[]},"batch":{"concurrency":2}});
        let models = json!({"llm":{"enabled":true,"model_list":[{"model_name":"m"}],"providers":[]},"batch":{"concurrency":2}});
        assert!(!restart_required(Some(&old), &models));
        assert!(!restart_required(
            Some(&json!({})),
            &json!({"llm":{"model_list":[]}})
        ));
        let other = json!({"llm":{"enabled":true,"model_list":[]},"batch":{"concurrency":3}});
        assert!(restart_required(Some(&old), &other));
        assert!(restart_required(None, &old));
    }
}
