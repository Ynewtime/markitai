//! Provider envelopes carry data only. Capability selection never probes a model.
use super::*;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Mode {
    Tools,
    JsonSchema,
    JsonText,
}
#[derive(Clone, Copy)]
pub(super) enum Schema {
    Document,
    ImageAnalysis,
}
#[derive(Clone, Copy)]
pub(super) struct Wire {
    pub mode: Mode,
    pub schema: Schema,
}

impl Schema {
    fn name(self) -> &'static str {
        match self {
            Self::Document => "MarkitaiDocument",
            Self::ImageAnalysis => "MarkitaiImageAnalysis",
        }
    }
    fn value(self) -> Value {
        match self {
            Self::Document => {
                json!({"type":"object","additionalProperties":false,"required":["cleaned_markdown","frontmatter"],"properties":{
                    "cleaned_markdown":{"type":"string"},"frontmatter":{"type":"object","additionalProperties":false,"required":["description","tags"],"properties":{
                        "description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}}
                    }}
                }})
            }
            Self::ImageAnalysis => {
                json!({"type":"object","additionalProperties":false,"required":["caption","description","extracted_text"],"properties":{
                    "caption":{"type":"string"},"description":{"type":"string"},"extracted_text":{"type":"string"}
                }})
            }
        }
    }
}

// Exact identities verified against official provider contracts on 2026-09-29.
// Sources and exclusions are recorded in the structured-transport planning doc.
// An OpenAI-compatible endpoint or a vision flag alone establishes neither bit.
pub(super) fn capabilities(entry: &Deployment) -> (bool, bool) {
    match (
        entry.provider.as_str(),
        entry.model.as_str(),
        entry.protocol,
    ) {
        ("openai", "gpt-4.1" | "gpt-4.1-2025-04-14", Protocol::Chat) => (true, true),
        ("anthropic", "claude-haiku-4-5" | "claude-haiku-4-5-20251001", Protocol::Anthropic) => {
            (true, true)
        }
        (
            "anthropic",
            "claude-opus-5-5" | "claude-sonnet-5-5" | "claude-fable-5-1" | "claude-mythos-5-1",
            Protocol::Anthropic,
        ) => (false, true),
        ("gemini", "gemini-3.8-flash", Protocol::Chat) => (false, true),
        _ => (false, false),
    }
}
fn modes(prompts: &Prompts, cfg: &Value, env: &HashMap<String, String>) -> Result<Vec<Mode>> {
    let entries = deployments(cfg, env)?;
    let groups = fallback_groups(cfg, &entries)?;
    // This is the same reachable request pool used by routing, including
    // credential-free local endpoints. Disabled entries were already removed.
    let mut flags = (true, true);
    let mut count = 0;
    for entry in entries.iter().filter(|entry| {
        groups.contains(&entry.group)
            && (prompts.image.is_none() || entry.supports_vision != Some(false))
    }) {
        let current = capabilities(entry);
        flags.0 &= current.0;
        flags.1 &= current.1;
        count += 1;
    }
    let mut result = Vec::with_capacity(3);
    if count > 0 && flags.0 {
        result.push(Mode::Tools);
    }
    if count > 0 && flags.1 {
        result.push(Mode::JsonSchema);
    }
    result.push(Mode::JsonText);
    Ok(result)
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Self::Tools => "tool-call",
            Self::JsonSchema => "JSON-schema",
            Self::JsonText => "JSON-text",
        }
    }
}

impl Wire {
    pub(super) fn payload(self, entry: &Deployment, prompts: &Prompts) -> Value {
        let mut body = super::payload(entry, prompts);
        let schema = self.schema.value();
        let name = self.schema.name();
        match (self.mode, entry.protocol) {
            (Mode::Tools, Protocol::Anthropic) => {
                body["tools"] = json!([{"name":name,"description":"Return the complete validated document data.","input_schema":schema,"strict":true}]);
                body["tool_choice"] =
                    json!({"type":"tool","name":name,"disable_parallel_tool_use":true});
            }
            (Mode::Tools, _) => {
                body["tools"] = json!([{"type":"function","function":{"name":name,"description":"Return the complete validated document data.","parameters":schema,"strict":true}}]);
                body["tool_choice"] = json!({"type":"function","function":{"name":name}});
                body["parallel_tool_calls"] = json!(false);
            }
            (Mode::JsonSchema, Protocol::Anthropic) => {
                body["output_config"] = json!({"format":{"type":"json_schema","schema":schema}});
            }
            (Mode::JsonSchema, _) => {
                body["response_format"] = json!({"type":"json_schema","json_schema":{"name":name,"strict":true,"schema":schema}});
            }
            (Mode::JsonText, _) => {}
        }
        body
    }

    pub(super) fn rejected(self, status: u16, bytes: &[u8]) -> bool {
        if !matches!(status, 400 | 422) || self.mode == Mode::JsonText {
            return false;
        }
        let Ok(body) = serde_json::from_slice::<Value>(bytes) else {
            return false;
        };
        let Some(error) = body.get("error").filter(|value| value.is_object()) else {
            return false;
        };
        // A bounded, explicit parameter rejection is different from malformed
        // images, context overflow or unrelated request errors. Never echo it.
        let parameter = error.get("param").and_then(Value::as_str).unwrap_or("");
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let fields: &[&str] = match self.mode {
            Mode::Tools => &["tools", "tool_choice", "parallel_tool_calls"],
            Mode::JsonSchema => &[
                "response_format",
                "json_schema",
                "output_config",
                "output_config.format",
            ],
            Mode::JsonText => &[],
        };
        let named = fields.iter().any(|field| {
            parameter == *field
                || parameter.starts_with(&format!("{field}."))
                || parameter.starts_with(&format!("{field}["))
        });
        let unsupported = [
            "not supported",
            "unsupported",
            "not permitted",
            "not allowed",
            "unknown parameter",
            "unrecognized",
            "extra inputs are not permitted",
        ]
        .iter()
        .any(|word| message.contains(word));
        named || unsupported && fields.iter().any(|field| message.contains(field))
    }

    pub(super) fn decode(
        self,
        protocol: Protocol,
        data: &Value,
    ) -> std::result::Result<String, Failure> {
        if data
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            == Some("length")
            || data.get("stop_reason").and_then(Value::as_str) == Some("max_tokens")
        {
            return Err(blocked(
                FailureKind::Truncated,
                "LLM output was truncated by its token limit",
            ));
        }
        if data
            .pointer("/choices/0/message/refusal")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
            || data.get("stop_reason").and_then(Value::as_str) == Some("refusal")
            || data
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str)
                == Some("content_filter")
        {
            return Err(blocked(
                FailureKind::Refusal,
                "LLM declined the structured request",
            ));
        }
        let invalid = || Failure::terminal("LLM returned an invalid structured response envelope");
        if self.mode == Mode::Tools {
            if protocol == Protocol::Anthropic {
                let blocks = data
                    .get("content")
                    .and_then(Value::as_array)
                    .ok_or_else(invalid)?;
                let mut calls = blocks
                    .iter()
                    .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"));
                let call = calls.next().ok_or_else(invalid)?;
                if calls.next().is_some()
                    || call.get("name").and_then(Value::as_str) != Some(self.schema.name())
                {
                    return Err(invalid());
                }
                return call
                    .get("input")
                    .filter(|value| value.is_object())
                    .map(Value::to_string)
                    .ok_or_else(invalid);
            }
            let calls = data
                .pointer("/choices/0/message/tool_calls")
                .and_then(Value::as_array)
                .ok_or_else(invalid)?;
            if calls.len() != 1
                || calls[0].get("type").and_then(Value::as_str) != Some("function")
                || calls[0].pointer("/function/name").and_then(Value::as_str)
                    != Some(self.schema.name())
            {
                return Err(invalid());
            }
            return calls[0]
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(invalid);
        }
        if protocol == Protocol::Anthropic {
            let blocks = data
                .get("content")
                .and_then(Value::as_array)
                .ok_or_else(invalid)?;
            if blocks
                .iter()
                .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
            {
                return Err(invalid());
            }
            let text = blocks
                .iter()
                .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            if text.trim().is_empty() {
                Err(Failure::terminal(&empty_answer(data)))
            } else {
                Ok(text)
            }
        } else {
            if data
                .pointer("/choices/0/message/tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| !calls.is_empty())
            {
                return Err(invalid());
            }
            let Some(content) = data
                .pointer("/choices/0/message/content")
                .filter(|content| !content.is_null())
            else {
                return Err(
                    if data
                        .pointer("/choices/0/message")
                        .is_some_and(Value::is_object)
                    {
                        Failure::terminal(&empty_answer(data))
                    } else {
                        invalid()
                    },
                );
            };
            let text = content
                .as_str()
                .map(str::to_owned)
                .or_else(|| {
                    content.as_array().map(|blocks| {
                        blocks
                            .iter()
                            .filter_map(|block| block.get("text").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                })
                .ok_or_else(invalid)?;
            if text.trim().is_empty() {
                Err(Failure::terminal(&empty_answer(data)))
            } else {
                Ok(text)
            }
        }
    }
}
fn blocked(kind: FailureKind, message: &str) -> Failure {
    Failure {
        kind,
        error: Error::Conversion(message.into()),
        retryable: false,
        fatal: false,
        document_fatal: true,
        retry_after: None,
    }
}

pub(super) struct Request<'a> {
    pub prompts: &'a Prompts,
    pub schema: Schema,
    pub stop: Option<&'a AtomicBool>,
}

pub(super) fn run<T>(
    request: Request<'_>,
    cfg: &Value,
    env: &HashMap<String, String>,
    runtime: Option<&LlmRuntime>,
    validate: impl Fn(&Value) -> Result<T>,
) -> std::result::Result<(T, ConversionUsage), VisionFailure> {
    let _own_scope = DocumentScope::shared()
        .is_none()
        .then(|| DocumentScope::new(cfg));
    let before = document_usage().expect("structured request scope installed");
    let ladder = modes(request.prompts, cfg, env)?;
    let mut last = Error::Conversion("LLM returned no valid structured result".into());
    let mut rejected: Option<(Mode, String)> = None;
    for mode in ladder {
        let attempts = if mode == Mode::JsonText { 3 } else { 1 };
        // A rejected answer is never answered by the same request again: a
        // later attempt names the rejection, so it is a different request.
        let mut corrected = None;
        for attempt in 0..attempts {
            // A discarded answer is paid and its follow-up request is another
            // one: both are visible as a warning, and the budget counts both.
            if let Some((previous, reason)) = rejected.take()
                && !document_exhausted()
            {
                note_document_warning(format!(
                    "LLM {} answer was rejected ({reason}); the request was sent again in {} mode",
                    previous.label(),
                    mode.label()
                ));
            }
            let text = match run_mode(
                corrected.as_ref().unwrap_or(request.prompts),
                cfg,
                env,
                &mut std::thread::sleep,
                runtime,
                request.stop,
                Some(Wire {
                    mode,
                    schema: request.schema,
                }),
            ) {
                Ok((text, _)) => text,
                Err(failure) => {
                    // Transport already exhausted its configured routing policy.
                    // Preserve its disposition for callers (a web page may still
                    // use text fallback), but never restart it in another mode.
                    if !failure.allow_text_fallback
                        || !matches!(
                            failure.kind,
                            FailureKind::Validation | FailureKind::ModeRejected
                        )
                    {
                        return Err(failure);
                    }
                    // An empty or malformed envelope is the provider's answer to
                    // this exact request; sending it unchanged gets it again.
                    if mode == Mode::JsonText {
                        return Err(failure);
                    }
                    rejected = Some((mode, failure.error.to_string()));
                    last = failure.error;
                    break;
                }
            };
            match parse(&text).and_then(|value| validate(&value)) {
                Ok(answer) => {
                    return Ok((
                        answer,
                        usage_difference(&document_usage().expect("scope installed"), &before),
                    ));
                }
                Err(error) => {
                    rejected = Some((mode, error.to_string()));
                    last = error;
                }
            }
            if mode == Mode::JsonText
                && attempt + 1 == attempts
                && let Some(value) = repair(&text)
                && let Ok(answer) = validate(&value)
            {
                return Ok((
                    answer,
                    usage_difference(&document_usage().expect("scope installed"), &before),
                ));
            }
            if document_exhausted() {
                return Err(VisionFailure::blocked(Error::Conversion(format!(
                    "{last}; LLM per-document request budget exhausted during structured validation"
                ))));
            }
            if mode == Mode::JsonText {
                corrected = Some(super::corrected(request.prompts, &last));
            }
        }
    }
    Err(last.into())
}

fn json_text(text: &str) -> &str {
    let text = text.trim();
    text.strip_prefix("```json\n")
        .or_else(|| text.strip_prefix("```\n"))
        .and_then(|value| value.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(text)
}
pub(super) fn parse(text: &str) -> Result<Value> {
    serde_json::from_str(json_text(text))
        .map_err(|_| Error::Conversion("LLM response is not valid structured JSON".into()))
}
// Only delete commas immediately before an existing object/array closer, outside
// strings. No missing quote, closer, key, text or metadata is synthesized.
fn repair(text: &str) -> Option<Value> {
    let text = json_text(text);
    if text.len() > 1024 * 1024 {
        return None;
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut quoted = false;
    let mut escape = false;
    let mut changed = false;
    for (index, &byte) in bytes.iter().enumerate() {
        if quoted {
            out.push(byte);
            if escape {
                escape = false;
            } else if byte == b'\\' {
                escape = true;
            } else if byte == b'"' {
                quoted = false;
            }
        } else if byte == b'"' {
            quoted = true;
            out.push(byte);
        } else if byte == b','
            && bytes[index + 1..]
                .iter()
                .find(|byte| !byte.is_ascii_whitespace())
                .is_some_and(|byte| matches!(byte, b'}' | b']'))
        {
            changed = true;
        } else {
            out.push(byte);
        }
    }
    if !changed || quoted {
        return None;
    }
    serde_json::from_slice(&out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repair_only_removes_complete_trailing_commas_and_keeps_literals() {
        assert_eq!(
            repair(r#"{"text":"Literal ,} and \\\"", "nested":[1,2,],}"#).unwrap()["nested"],
            json!([1, 2])
        );
        for value in [
            r#"{"text":"cut off"#,
            r#"{"text":"valid", "other":[1,2,"#,
            "{'text':'python',}",
            "prefix {\"text\":1,}",
        ] {
            assert!(repair(value).is_none(), "{value}");
        }
        assert!(repair("{\"text\":\"unchanged\"}").is_none());
    }
    #[test]
    fn refusal_and_truncation_do_not_admit_complete_looking_tool_json() {
        let wire = Wire {
            mode: Mode::Tools,
            schema: Schema::Document,
        };
        let data = json!({"choices":[{"message":{"tool_calls":[{"type":"function","function":{"name":"MarkitaiDocument","arguments":"{}"}}]},"finish_reason":"length"}]});
        assert!(matches!(
            wire.decode(Protocol::Chat, &data),
            Err(Failure {
                kind: FailureKind::Truncated,
                ..
            })
        ));
        let data = json!({"content":[{"type":"tool_use","name":"MarkitaiDocument","input":{}}],"stop_reason":"refusal"});
        assert!(matches!(
            wire.decode(Protocol::Anthropic, &data),
            Err(Failure {
                kind: FailureKind::Refusal,
                ..
            })
        ));
    }
    #[test]
    fn parameter_rejection_does_not_reclassify_bad_images() {
        let wire = Wire {
            mode: Mode::Tools,
            schema: Schema::Document,
        };
        assert!(wire.rejected(
            400,
            br#"{"error":{"param":"tool_choice","message":"not supported"}}"#
        ));
        assert!(wire.rejected(
            422,
            br#"{"error":{"message":"tools: extra inputs are not permitted"}}"#
        ));
        assert!(!wire.rejected(
            400,
            br#"{"error":{"param":"messages","message":"invalid image; tools not relevant"}}"#
        ));
        assert!(!wire.rejected(
            401,
            br#"{"error":{"param":"tools","message":"not supported"}}"#
        ));
    }
}
