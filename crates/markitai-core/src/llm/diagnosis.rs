//! What a failed LLM request tells the user: the kind of transport failure,
//! the provider's own error message and the deployment that failed.
//!
//! Provider wording is public only after it is cleaned: credentials and
//! credential-like tokens are replaced, URLs lose userinfo and secret
//! parameters, long quotations are dropped, and a message that repeats any
//! stretch of the request (document text or prompts) is withheld entirely.

use super::{Deployment, Prompts};
use serde_json::Value;

/// Characters of provider wording kept in an error.
const SAID_CHARS: usize = 200;
/// A quotation longer than this is replaced: it may be request content.
const QUOTE_CHARS: usize = 40;
/// Letters and digits in a stretch that, also found in the request, marks
/// the message as an echo of it. Windows overlap by `ECHO_WINDOW - ECHO_STEP`,
/// so any shared stretch of `ECHO_WINDOW + ECHO_STEP - 1` is found.
const ECHO_WINDOW: usize = 16;
const ECHO_STEP: usize = 4;

/// `[deployment] at [host]`, for the end of a failure message: the
/// configured model and the endpoint's host, never its path or query.
pub(super) fn deployment(entry: &Deployment) -> String {
    let host = url::Url::parse(&entry.endpoint).ok().and_then(|url| {
        let host = url.host_str()?.to_owned();
        Some(match url.port() {
            Some(port) => format!("{host}:{port}"),
            None => host,
        })
    });
    match host {
        Some(host) => format!("deployment {} at {host}", entry.id),
        None => format!("deployment {}", entry.id),
    }
}

/// The failure of a request that received no HTTP response, named by kind:
/// a timeout, a refused connection, an unresolved host name, a failed TLS
/// handshake or a connection closed before the response.
pub(super) fn transport(error: reqwest::Error, timeout: u64) -> String {
    let error = error.without_url();
    if error.is_timeout() {
        return if error.is_connect() {
            format!(
                "LLM request timed out: no connection within {} s",
                timeout.min(15)
            )
        } else {
            format!("LLM request timed out: no response within {timeout} s")
        };
    }
    let chain = Chain::of(&error);
    if chain.refused() {
        return "LLM request failed: connection refused".into();
    }
    let kind = if chain.unresolved() {
        Some("host name not resolved")
    } else if chain.tls() {
        Some("TLS handshake failed")
    } else if chain.dropped() {
        Some("connection closed without a response")
    } else if error.is_connect() {
        Some("cannot connect")
    } else {
        None
    };
    match (kind, chain.innermost()) {
        (Some(kind), Some(detail)) => format!("LLM request failed: {kind} ({detail})"),
        (Some(kind), None) => format!("LLM request failed: {kind}"),
        (None, Some(detail)) => format!("LLM request failed: {detail}"),
        (None, None) => "LLM request failed".into(),
    }
}

/// The failure of reading a response body after its status arrived.
pub(super) fn unreadable(error: &std::io::Error, timed_out: bool, timeout: u64) -> String {
    if timed_out {
        return format!("Cannot read LLM response: no complete response within {timeout} s");
    }
    let chain = Chain::of(error);
    let kind = if chain.dropped() {
        "connection closed"
    } else {
        "read failed"
    };
    match chain.innermost() {
        Some(detail) => format!("Cannot read LLM response: {kind} ({detail})"),
        None => format!("Cannot read LLM response: {kind}"),
    }
}

/// A successful status whose body is not JSON: the content type and size,
/// and an HTML page's title (a gateway's sign-in or error page). Other body
/// text is never shown: it may be the model's answer.
pub(super) fn not_json(
    status: u16,
    content_type: Option<&str>,
    body: &[u8],
    entry: &Deployment,
    prompts: &Prompts,
) -> String {
    let kind = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| identifier(value, 80))
        .unwrap_or("no content type");
    let mut message = format!(
        "LLM response is not valid JSON (HTTP {status}, {kind}, {} bytes)",
        body.len()
    );
    if let Some(title) = html_title(body).and_then(|title| shown(&title, entry, prompts)) {
        message.push_str(": ");
        message.push_str(&title);
    }
    message
}

/// `HTTP {status}` with the provider's error type or code and its cleaned
/// message, when its body carries them.
pub(super) fn http(status: u16, body: &[u8], entry: &Deployment, prompts: &Prompts) -> String {
    let (code, said) = match serde_json::from_slice::<Value>(body) {
        Ok(value) => {
            // Gemini's OpenAI-compatible endpoint wraps its error in a list.
            let value = match &value {
                Value::Array(items) => items.first().cloned().unwrap_or(Value::Null),
                _ => value,
            };
            (error_code(&value, status), error_message(&value))
        }
        Err(_) => (None, html_title(body).or_else(|| plain_line(body))),
    };
    let said = said.and_then(|said| shown(&said, entry, prompts));
    match (code, said) {
        (Some(code), Some(said)) => format!("LLM returned HTTP {status} ({code}): {said}"),
        (Some(code), None) => format!("LLM returned HTTP {status} ({code})"),
        (None, Some(said)) => format!("LLM returned HTTP {status}: {said}"),
        (None, None) => format!("LLM returned HTTP {status}"),
    }
}

/// The error's type, code and status names (`invalid_request_error`,
/// `rate_limit_exceeded`, `RESOURCE_EXHAUSTED`), joined when they differ.
fn error_code(value: &Value, status: u16) -> Option<String> {
    let mut names: Vec<String> = Vec::new();
    for pointer in ["/error/type", "/error/code", "/error/status", "/code"] {
        let name = match value.pointer(pointer) {
            Some(Value::String(text)) => text.clone(),
            // A numeric code that repeats the HTTP status adds nothing.
            Some(Value::Number(number)) if number.as_u64() != Some(u64::from(status)) => {
                number.to_string()
            }
            _ => continue,
        };
        if identifier(&name, 60) && !names.contains(&name) {
            names.push(name);
        }
    }
    (!names.is_empty()).then(|| names.join("/"))
}

fn error_message(value: &Value) -> Option<String> {
    [
        "/error/message",
        "/message",
        "/detail",
        "/error",
        "/errors/0/message",
    ]
    .iter()
    .find_map(|pointer| value.pointer(pointer).and_then(Value::as_str))
    .map(str::to_owned)
}

/// A short identifier: letters, digits and `_ . - /`, nothing else.
fn identifier(text: &str, limit: usize) -> bool {
    !text.is_empty()
        && text.len() <= limit
        && text
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-' | '/' | '+'))
}

/// The text of an HTML page's `<title>` or first `<h1>`.
fn html_title(body: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(&body[..body.len().min(16 * 1024)]);
    let lower = text.to_ascii_lowercase();
    if !lower.trim_start().starts_with('<') {
        return None;
    }
    ["title", "h1"].iter().find_map(|tag| {
        let open = lower.find(&format!("<{tag}"))?;
        let start = open + lower[open..].find('>')? + 1;
        let end = start + lower[start..].find(&format!("</{tag}"))?;
        let inner = &text[start..end];
        // Strip nested tags.
        let mut plain = String::new();
        let mut in_tag = false;
        for ch in inner.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => plain.push(ch),
                _ => {}
            }
        }
        let plain = plain.trim().to_owned();
        (!plain.is_empty()).then_some(plain)
    })
}

/// A short plain-text error body's first line, such as a proxy's
/// `upstream connect error or disconnect/reset before headers`.
fn plain_line(body: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(body).ok()?.trim();
    let line = text.lines().next()?.trim();
    (!line.is_empty() && line.len() <= 300 && !line.starts_with(['<', '{', '[']))
        .then(|| line.to_owned())
}

/// Provider wording as it may be shown, or `None` when nothing safe is left.
fn shown(text: &str, entry: &Deployment, prompts: &Prompts) -> Option<String> {
    let text: String = text
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect();
    let mut text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(key) = entry.key.as_deref().filter(|key| key.len() >= 4) {
        text = text.replace(key, "[REDACTED]");
    }
    let text = without_quotations(&without_credentials(&text));
    if text.trim().is_empty() {
        return None;
    }
    if echoes(&text, prompts) {
        return Some("[message withheld: it repeats the request]".into());
    }
    let mut said: String = text.chars().take(SAID_CHARS).collect();
    if said.chars().count() < text.chars().count() {
        said.push('…');
    }
    Some(said)
}

/// Each word that is or carries a credential replaced: a known API key or
/// token prefix, the word after `Bearer`, a JWT, or a long opaque token. A
/// URL loses its userinfo, query and fragment.
fn without_credentials(text: &str) -> String {
    const PREFIXES: [&str; 14] = [
        "sk-",
        "sk_",
        "rk_",
        "pk_",
        "aiza",
        "xai-",
        "gsk_",
        "hf_",
        "ghp_",
        "gho_",
        "ghu_",
        "github_pat_",
        "eyj",
        "api_key=",
    ];
    let mut out: Vec<String> = Vec::new();
    let mut after_bearer = false;
    for word in text.split(' ') {
        let core = word.trim_matches(|ch: char| !ch.is_alphanumeric() && !matches!(ch, '_' | '-'));
        let lower = core.to_ascii_lowercase();
        let replaced = if word.contains("://") {
            let start = word.find(|ch: char| ch.is_ascii_alphabetic()).unwrap_or(0);
            let end = word
                .rfind(|ch: char| ch.is_ascii_alphanumeric() || ch == '/')
                .map_or(word.len(), |end| end + 1);
            let url = &word[start..end.max(start)];
            let shown = url::Url::parse(url).map_or_else(
                |_| "[URL]".into(),
                |mut parsed| {
                    let _ = parsed.set_username("");
                    let _ = parsed.set_password(None);
                    parsed.set_query(None);
                    parsed.set_fragment(None);
                    crate::output::redact_url(parsed.as_str())
                },
            );
            Some(format!(
                "{}{shown}{}",
                &word[..start],
                &word[end.max(start)..]
            ))
        } else if after_bearer && !core.is_empty()
            || PREFIXES.iter().any(|prefix| lower.starts_with(prefix)) && core.len() >= 8
            || opaque(core)
        {
            Some(word.replace(core, "[REDACTED]"))
        } else {
            None
        };
        after_bearer = lower == "bearer";
        out.push(replaced.unwrap_or_else(|| word.to_owned()));
    }
    out.join(" ")
}

/// An unbroken run of 24 or more letters and digits that mixes both, as keys,
/// tokens and hashes have; a model name's short dash-separated parts do not.
fn opaque(word: &str) -> bool {
    word.split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|run| {
            run.len() >= 24
                && run.bytes().any(|byte| byte.is_ascii_digit())
                && run.bytes().any(|byte| byte.is_ascii_alphabetic())
        })
}

/// Quotations longer than [`QUOTE_CHARS`] replaced by `'…'`: a provider may
/// quote the request it refused.
fn without_quotations(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(['\'', '"', '`', '“', '‘']) {
        let quote = rest[open..].chars().next().expect("a quote character");
        let close_quote = match quote {
            '“' => '”',
            '‘' => '’',
            other => other,
        };
        let inner_start = open + quote.len_utf8();
        // An apostrophe inside a word (`can't`) opens no quotation.
        let in_word = quote == '\''
            && rest[..open]
                .chars()
                .next_back()
                .is_some_and(char::is_alphanumeric);
        match rest[inner_start..].find(close_quote) {
            Some(length) if !in_word => {
                let inner = &rest[inner_start..inner_start + length];
                out.push_str(&rest[..inner_start]);
                if inner.chars().count() > QUOTE_CHARS {
                    out.push('…');
                } else {
                    out.push_str(inner);
                }
                out.push(close_quote);
                rest = &rest[inner_start + length + close_quote.len_utf8()..];
            }
            _ => {
                out.push_str(&rest[..inner_start]);
                rest = &rest[inner_start..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Whether the message repeats a stretch of the request's prompts, compared
/// by letters and digits only so escaping and spacing do not hide it.
fn echoes(message: &str, prompts: &Prompts) -> bool {
    let fold = |text: &str| -> Vec<char> {
        text.chars()
            .filter(|ch| ch.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    };
    let said = fold(&unescaped(message));
    if said.len() < ECHO_WINDOW {
        return false;
    }
    let request: String = fold(&prompts.system)
        .into_iter()
        .chain(['\u{0}'])
        .chain(fold(&prompts.user))
        .collect();
    (0..=said.len() - ECHO_WINDOW)
        .step_by(ECHO_STEP)
        .chain([said.len() - ECHO_WINDOW])
        .any(|start| {
            let window: String = said[start..start + ECHO_WINDOW].iter().collect();
            request.contains(&window)
        })
}

/// `\n`, `\t` and `\uXXXX` escapes a provider may have applied to a quote.
fn unescaped(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n' | 'r' | 't') => out.push(' '),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    Some(decoded) => out.push(decoded),
                    None => out.push_str(&hex),
                }
            }
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

/// The source chain of a transport error, read for its kind.
struct Chain<'a>(Vec<&'a (dyn std::error::Error + 'static)>);

impl<'a> Chain<'a> {
    fn of(error: &'a (dyn std::error::Error + 'static)) -> Self {
        let mut links = Vec::new();
        let mut current = Some(error);
        while let Some(link) = current {
            links.push(link);
            // io::Error::source skips its wrapped error; read the wrapper.
            current = match link
                .downcast_ref::<std::io::Error>()
                .and_then(|io| io.get_ref())
            {
                Some(inner) => Some(inner as &(dyn std::error::Error + 'static)),
                None => link.source(),
            };
            if links.len() > 16 {
                break;
            }
        }
        Self(links)
    }

    fn io_kind(&self, kinds: &[std::io::ErrorKind]) -> bool {
        self.0.iter().any(|link| {
            link.downcast_ref::<std::io::Error>()
                .is_some_and(|io| kinds.contains(&io.kind()))
        })
    }

    fn says(&self, words: &[&str]) -> bool {
        self.0.iter().any(|link| {
            let text = link.to_string().to_ascii_lowercase();
            words.iter().any(|word| text.contains(word))
        })
    }

    fn refused(&self) -> bool {
        self.io_kind(&[std::io::ErrorKind::ConnectionRefused])
            // WSAECONNREFUSED, when the I/O error is not in the chain.
            || self.says(&["connection refused", "os error 10061"])
    }

    fn unresolved(&self) -> bool {
        self.says(&[
            "dns error",
            "failed to lookup address",
            "name or service not known",
            "no such host",
            "nodename nor servname",
        ])
    }

    fn tls(&self) -> bool {
        self.says(&[
            "certificate",
            "tls",
            "ssl",
            "handshake",
            "alert",
            "corrupt message",
            "peer is incompatible",
            "peer misbehaved",
        ])
    }

    fn dropped(&self) -> bool {
        use std::io::ErrorKind::{BrokenPipe, ConnectionAborted, ConnectionReset, UnexpectedEof};
        self.io_kind(&[
            BrokenPipe,
            ConnectionAborted,
            ConnectionReset,
            UnexpectedEof,
        ]) || self.says(&[
            "connection closed",
            "connection reset",
            "connection aborted",
            "broken pipe",
            "os error 10053",
            "os error 10054",
        ])
    }

    /// The innermost cause's own words: an OS or TLS message, without URLs.
    fn innermost(&self) -> Option<String> {
        let text = self.0.last()?.to_string();
        let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let text = without_credentials(&text);
        let text: String = text.chars().take(SAID_CHARS).collect();
        (!text.is_empty()).then_some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompts(user: &str) -> Prompts {
        Prompts {
            system: "Clean up the document.".into(),
            user: user.into(),
            image: None,
            cache_scope: String::new(),
        }
    }

    fn entry(key: &str) -> Deployment {
        Deployment {
            id: "openai/gpt-test".into(),
            explicit_id: None,
            group: "default".into(),
            model: "gpt-test".into(),
            provider: "openai".into(),
            weight: 1,
            key: Some(key.into()),
            endpoint: "https://user:pw@llm.example.test:8443/v1/chat/completions?key=secret".into(),
            protocol: super::super::Protocol::Chat,
            max_tokens: None,
            reasoning_effort: None,
            supports_vision: None,
        }
    }

    #[test]
    fn the_deployment_is_named_with_its_host_only() {
        assert_eq!(
            deployment(&entry("k")),
            "deployment openai/gpt-test at llm.example.test:8443"
        );
    }

    #[test]
    fn provider_errors_keep_their_type_and_message() {
        let body = br#"{"error":{"message":"Invalid 'max_tokens': integer above maximum value.","type":"invalid_request_error","param":"max_tokens","code":"integer_above_max_value"}}"#;
        assert_eq!(
            http(400, body, &entry("k"), &prompts("text")),
            "LLM returned HTTP 400 (invalid_request_error/integer_above_max_value): Invalid 'max_tokens': integer above maximum value."
        );
        // Anthropic and Gemini shapes.
        let anthropic =
            br#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        assert_eq!(
            http(529, anthropic, &entry("k"), &prompts("text")),
            "LLM returned HTTP 529 (overloaded_error): Overloaded"
        );
        let gemini = br#"[{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT"}}]"#;
        assert_eq!(
            http(400, gemini, &entry("k"), &prompts("text")),
            "LLM returned HTTP 400 (INVALID_ARGUMENT): API key not valid. Please pass a valid API key."
        );
        let html = b"<html><head><title>502 Bad Gateway</title></head><body>nginx</body></html>";
        assert_eq!(
            http(502, html, &entry("k"), &prompts("text")),
            "LLM returned HTTP 502: 502 Bad Gateway"
        );
        assert_eq!(
            http(
                503,
                b"upstream connect error or disconnect/reset before headers",
                &entry("k"),
                &prompts("text")
            ),
            "LLM returned HTTP 503: upstream connect error or disconnect/reset before headers"
        );
        assert_eq!(
            http(500, b"", &entry("k"), &prompts("text")),
            "LLM returned HTTP 500"
        );
    }

    #[test]
    fn provider_words_never_carry_credentials_or_the_request() {
        let key = "sk-live-0123456789abcdefABCDEF";
        let body = format!(
            r#"{{"error":{{"message":"Incorrect API key provided: {key}. Also sk-proj-****wxyz, Bearer abc.def and https://u:p@api.example.test/keys?api_key=zzz#frag; trace 9f8e7d6c5b4a39281706f5e4d3c2b1a0ffeeddcc","type":"invalid_request_error"}}}}"#
        );
        let said = http(401, body.as_bytes(), &entry(key), &prompts("text"));
        assert!(!said.contains(key), "{said}");
        assert!(
            !said.contains("wxyz") && !said.contains("abc.def"),
            "{said}"
        );
        assert!(
            !said.contains("u:p") && !said.contains("zzz") && !said.contains("frag"),
            "{said}"
        );
        assert!(!said.contains("9f8e7d6c"), "{said}");
        assert!(said.contains("https://api.example.test/keys"), "{said}");
        // Model names and request ids stay readable.
        let missing = br#"{"error":{"message":"The model gemini-2.5-flash-preview-09-2025-thinking does not exist (request 123e4567-e89b-12d3-a456-426614174000)"}}"#;
        assert_eq!(
            http(404, missing, &entry(key), &prompts("text")),
            "LLM returned HTTP 404: The model gemini-2.5-flash-preview-09-2025-thinking does not exist (request 123e4567-e89b-12d3-a456-426614174000)"
        );
        assert!(said.starts_with("LLM returned HTTP 401 (invalid_request_error): Incorrect API key provided: [REDACTED]."), "{said}");

        // A quotation of the document is dropped, and an unquoted echo
        // withholds the whole message, also through JSON escapes.
        let document =
            "Confidential: the quarterly widget count rose from 12 to 15 in the north region.";
        let quoted = br#"{"error":{"message":"Cannot parse content 'Confidential: the quarterly widget count rose from 12' near byte 7"}}"#;
        let said = http(400, quoted, &entry("k"), &prompts(document));
        assert_eq!(
            said,
            "LLM returned HTTP 400: Cannot parse content '…' near byte 7"
        );
        let echoed = br#"{"error":{"message":"Bad input near: the quarterly\nwidget count rose from twelve"}}"#;
        let said = http(400, echoed, &entry("k"), &prompts(document));
        assert_eq!(
            said,
            "LLM returned HTTP 400: [message withheld: it repeats the request]"
        );
        let escaped = "{\"error\":{\"message\":\"Bad input near: \\\\u5b63\\\\u5ea6\\\\u9500\\\\u552e\\\\u989d\\\\u4ece\\\\u5341\\\\u4e8c\\\\u589e\\\\u957f\\\\u5230\\\\u5341\\\\u4e94\\\\u4e07\\\\u5143\\\\u4ee5\\\\u4e0a\"}}";
        let said = http(
            400,
            escaped.as_bytes(),
            &entry("k"),
            &prompts("全年季度销售额从十二增长到十五万元以上。"),
        );
        assert!(
            said.ends_with("[message withheld: it repeats the request]"),
            "{said}"
        );
        // Messages longer than the cap are cut.
        let long = format!(r#"{{"error":{{"message":"{}"}}}}"#, "word ".repeat(100));
        assert!(http(400, long.as_bytes(), &entry("k"), &prompts("x")).ends_with('…'));
    }

    #[test]
    fn a_body_that_is_not_json_is_described_without_its_text() {
        assert_eq!(
            not_json(
                200,
                Some("text/html; charset=utf-8"),
                b"<html><head><title>Sign in</title></head></html>",
                &entry("k"),
                &prompts("x")
            ),
            "LLM response is not valid JSON (HTTP 200, text/html, 48 bytes): Sign in"
        );
        assert_eq!(
            not_json(
                200,
                None,
                b"Here is the cleaned document",
                &entry("k"),
                &prompts("x")
            ),
            "LLM response is not valid JSON (HTTP 200, no content type, 28 bytes)"
        );
    }

    /// The error chain hyper builds for a failed name lookup.
    #[derive(Debug)]
    struct Link(
        &'static str,
        Option<Box<dyn std::error::Error + Send + Sync>>,
    );
    impl std::fmt::Display for Link {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.0)
        }
    }
    impl std::error::Error for Link {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.1
                .as_deref()
                .map(|error| error as &(dyn std::error::Error + 'static))
        }
    }

    #[test]
    fn a_failed_name_lookup_is_recognized_in_the_chain() {
        let lookup = std::io::Error::other(
            "failed to lookup address information: Name or service not known",
        );
        let error = Link(
            "client error (Connect)",
            Some(Box::new(Link("dns error", Some(Box::new(lookup))))),
        );
        let chain = Chain::of(&error);
        assert!(chain.unresolved() && !chain.refused() && !chain.tls());
        assert_eq!(
            chain.innermost().as_deref(),
            Some("failed to lookup address information: Name or service not known")
        );
    }
}
