use super::process::{Process, limit, protocol};
use super::*;
use base64::Engine;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};

const MAX_TEXT: usize = 8 * 1024 * 1024;
const MAX_IMAGE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CALLS: usize = 64;

fn no_events(message: &Value) -> Result<(), Failure> {
    if message.get("method").and_then(Value::as_str) == Some("session.event") {
        return Err(protocol());
    }
    Ok(())
}
fn start(
    config: &CopilotConfig,
    timeout: Duration,
    cancel: Option<&AtomicBool>,
) -> Result<Process, Failure> {
    let mut process = Process::spawn(config, timeout, cancel)?;
    let connected = process.call("connect", json!({"clientInfo":{"editorName":"markitai","editorVersion":env!("CARGO_PKG_VERSION")},"supportedTaskKinds":[]}), cancel, &mut no_events)?;
    if connected.get("protocolVersion").and_then(Value::as_u64) != Some(COPILOT_PROTOCOL) {
        return Err(Failure::new(
            FailureKind::Unsupported,
            "Copilot protocol version is unsupported; the adapter requires protocol 3",
        ));
    }
    let version = process.call("status.get", json!({}), cancel, &mut no_events)?;
    if version.get("protocolVersion").and_then(Value::as_u64) != Some(COPILOT_PROTOCOL)
        || version.get("version").and_then(Value::as_str) != Some(COPILOT_CLI_VERSION)
    {
        return Err(Failure::new(
            FailureKind::Unsupported,
            "Copilot CLI version is outside the verified adapter contract",
        ));
    }
    Ok(process)
}
fn authenticated(
    process: &mut Process,
    cancel: Option<&AtomicBool>,
) -> Result<AuthStatus, Failure> {
    let value = process.call("auth.getStatus", json!({}), cancel, &mut no_events)?;
    let authenticated = value
        .get("isAuthenticated")
        .and_then(Value::as_bool)
        .ok_or_else(protocol)?;
    let user = optional_string(&value, "login", 256)?;
    let source = optional_string(&value, "authType", 32)?;
    let host = optional_string(&value, "host", 1024)?;
    Ok(AuthStatus {
        provider: "copilot",
        authenticated,
        user,
        expires_at: None,
        error: (!authenticated)
            .then(|| "Copilot is not authenticated; run markitai auth copilot login".into()),
        details: json!({"source":"official_cli","verification":"runtime-auth-status","auth_type":source,"host":host,"cli_version":COPILOT_CLI_VERSION,"protocol_version":COPILOT_PROTOCOL}),
    })
}
pub fn status(config: &CopilotConfig, timeout: Duration) -> Result<AuthStatus, Failure> {
    authenticated(&mut start(config, timeout, None)?, None)
}
fn require_auth(process: &mut Process, cancel: Option<&AtomicBool>) -> Result<(), Failure> {
    let auth = authenticated(process, cancel)?;
    if auth.details.get("auth_type").and_then(Value::as_str) == Some("api-key") {
        return Err(Failure::new(
            FailureKind::Unsupported,
            "Copilot is configured for BYOK instead of subscription authentication",
        ));
    }
    if !auth.authenticated {
        return Err(Failure::new(
            FailureKind::Authentication,
            "Copilot is not authenticated; run markitai auth copilot login",
        ));
    }
    Ok(())
}
pub fn models(config: &CopilotConfig, timeout: Duration) -> Result<Vec<Model>, Failure> {
    let mut process = start(config, timeout, None)?;
    require_auth(&mut process, None)?;
    let value = process.call("models.list", json!({}), None, &mut no_events)?;
    let rows = value
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(protocol)?;
    if rows.len() > 4096 {
        return Err(limit());
    }
    let mut models = BTreeMap::new();
    for row in rows {
        if matches!(
            row.pointer("/policy/state").and_then(Value::as_str),
            Some("disabled")
        ) {
            continue;
        }
        let id = required_string(row, "id", 512)?;
        if id.contains(['\r', '\n', '\0']) {
            return Err(protocol());
        }
        let label = optional_string(row, "name", 512)?.unwrap_or_else(|| id.clone());
        let supports_vision = row
            .pointer("/capabilities/supports/vision")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        models.insert(
            id.clone(),
            Model {
                model: format!("copilot/{id}"),
                label,
                supports_vision,
            },
        );
    }
    Ok(models.into_values().collect())
}
fn session_config(model: &str, system: &str, cwd: &std::path::Path) -> Value {
    json!({"model":model,"clientName":"markitai","systemMessage":{"mode":"replace","content":system},
        "availableTools":[],"excludedTools":[],"toolFilterPrecedence":"excluded","tools":[],
        "requestPermission":true,"requestUserInput":false,"requestElicitation":false,"hooks":false,
        "workingDirectory":cwd,"configDir":cwd,"enableConfigDiscovery":false,
        "mcpServers":{},"customAgents":[],"customAgentsLocalOnly":true,"isExperimentalMode":false,
        "enableSessionTelemetry":false,"enableGitHubTelemetryForwarding":false,
        "enableOnDemandInstructionDiscovery":false,"enableFileHooks":false,"enableHostGitOperations":false,
        "enableSessionStore":false,"enableSkills":false,"skillDirectories":[],"pluginDirectories":[],
        "instructionDirectories":[],"organizationCustomInstructions":"","memory":{"enabled":false},
        "skipEmbeddingRetrieval":true,"embeddingCacheStorage":"in-memory","mcpOAuthTokenStorage":"in-memory",
        "streaming":false,"includeSubAgentStreamingEvents":false,"infiniteSessions":{"enabled":false}})
}

pub fn complete(config: &CopilotConfig, request: Request<'_>) -> Result<Completion, Failure> {
    let model = request
        .model
        .strip_prefix("copilot/")
        .unwrap_or(request.model);
    if model.is_empty()
        || model.len() > 512
        || model.contains(['\r', '\n', '\0'])
        || request.user.is_empty()
        || request.user.len() > MAX_TEXT
        || request.system.len() > MAX_TEXT
        || request.images.len() > 100
    {
        return Err(limit());
    }
    let mut images_size = 0usize;
    for (mime, bytes) in request.images {
        images_size = images_size.checked_add(bytes.len()).ok_or_else(limit)?;
        if images_size > MAX_IMAGE_BYTES
            || bytes.is_empty()
            || !matches!(
                *mime,
                "image/png" | "image/jpeg" | "image/webp" | "image/gif"
            )
        {
            return Err(limit());
        }
    }
    let mut state = Events {
        expected_model: model.to_owned(),
        ..Events::default()
    };
    let result = (|| {
        let mut process = start(config, request.timeout, request.cancel)?;
        require_auth(&mut process, request.cancel)?;
        let settings = session_config(model, request.system, process.workspace.path());
        let created =
            process.call(
                "session.create",
                settings,
                request.cancel,
                &mut |message| match message
                    .pointer("/params/event/type")
                    .and_then(Value::as_str)
                {
                    Some("session.error") => Err(Failure::new(
                        FailureKind::Transport,
                        "Copilot could not create the requested session",
                    )),
                    Some(
                        "permission.requested" | "external_tool.requested" | "tool.execution_start",
                    ) => Err(Failure::new(
                        FailureKind::Permission,
                        "Copilot requested a tool during session creation",
                    )),
                    _ => Ok(()),
                },
            )?;
        state.session = required_string(&created, "sessionId", 512)?;
        let attachments: Vec<Value> = request.images.iter().enumerate().map(|(index,(mime,bytes))| {
            json!({"type":"blob","data":base64::engine::general_purpose::STANDARD.encode(bytes),"mimeType":mime,"displayName":format!("image-{}",index+1)})
        }).collect();
        let reply = process.call("session.send", json!({"sessionId":state.session,"prompt":request.user,"attachments":attachments,"mode":"enqueue"}), request.cancel, &mut |event| state.accept(event))?;
        let message_id = required_string(&reply, "messageId", 512)?;
        if state
            .origin
            .as_deref()
            .is_some_and(|origin| origin != message_id)
        {
            return Err(protocol());
        }
        state.expected_origin = Some(message_id);
        while !state.idle {
            let event = process.next(request.cancel)?;
            if event.get("method").is_none() {
                return Err(protocol());
            }
            state.accept(&event)?;
        }
        state.text()
    })();
    match result {
        Ok(text) => Ok(Completion {
            text,
            usage: state.usage,
        }),
        Err(mut error) => {
            error.usage = state.usage;
            Err(error)
        }
    }
}
#[derive(Default)]
struct Events {
    session: String,
    expected_model: String,
    expected_origin: Option<String>,
    origin: Option<String>,
    idle: bool,
    usage: UsageEvidence,
    calls: HashMap<String, ObservedCall>,
    event_ids: HashMap<String, ObservedCall>,
    message_key: Option<String>,
    chunks: BTreeMap<usize, String>,
    chunk_count: usize,
}
impl Events {
    fn accept(&mut self, message: &Value) -> Result<(), Failure> {
        if message.get("method").and_then(Value::as_str) != Some("session.event") {
            return Ok(());
        }
        let params = message.get("params").ok_or_else(protocol)?;
        if params.get("sessionId").and_then(Value::as_str) != Some(self.session.as_str()) {
            return Ok(());
        }
        let event = params.get("event").ok_or_else(protocol)?;
        let kind = required_string(event, "type", 128)?;
        let data = event
            .get("data")
            .filter(|value| value.is_object())
            .ok_or_else(protocol)?;
        // Subagents are not enabled. Their content cannot stand in for the requested
        // result; their usage, if nevertheless observed, must not be silently lost.
        let subagent = event.get("agentId").is_some_and(|value| !value.is_null())
            || data
                .get("parentToolCallId")
                .is_some_and(|value| !value.is_null());
        if kind == "assistant.usage" {
            self.record_usage(event, data)?;
            if subagent {
                return Err(Failure::new(
                    FailureKind::Permission,
                    "Copilot unexpectedly invoked another agent",
                ));
            }
            if data.get("isByok").and_then(Value::as_bool) == Some(true) {
                return Err(Failure::new(
                    FailureKind::Unsupported,
                    "Copilot selected BYOK instead of the requested subscription",
                ));
            }
            if data.get("contentFilterTriggered").and_then(Value::as_bool) == Some(true)
                || data.get("finishReason").and_then(Value::as_str) == Some("content_filter")
            {
                return Err(Failure::new(
                    FailureKind::Refusal,
                    "Copilot refused or filtered the response",
                ));
            }
            if data.get("finishReason").and_then(Value::as_str) == Some("length") {
                return Err(Failure::new(
                    FailureKind::Truncated,
                    "Copilot response was truncated",
                ));
            }
            if data
                .get("availableToolCount")
                .and_then(Value::as_u64)
                .is_some_and(|count| count > 0)
                || data
                    .get("numToolCalls")
                    .and_then(Value::as_u64)
                    .is_some_and(|count| count > 0)
            {
                return Err(Failure::new(
                    FailureKind::Permission,
                    "Copilot unexpectedly requested tools",
                ));
            }
            return Ok(());
        }
        if subagent {
            return Err(Failure::new(
                FailureKind::Permission,
                "Copilot unexpectedly invoked another agent",
            ));
        }
        match kind.as_str() {
            "permission.requested"
            | "external_tool.requested"
            | "tool.execution_start"
            | "user_input.requested"
            | "auto_mode_switch.requested" => Err(Failure::new(
                FailureKind::Permission,
                "Copilot requested an unsupported tool or permission",
            )),
            "session.error" => Err(match data.get("errorType").and_then(Value::as_str) {
                Some("authentication" | "authorization") => Failure::new(
                    FailureKind::Authentication,
                    "Copilot authentication or authorization failed",
                ),
                Some("rate_limit" | "quota") => Failure::new(
                    FailureKind::RateLimit,
                    "Copilot subscription quota or rate limit was reached",
                ),
                Some("query" | "model_not_supported") => Failure::new(
                    FailureKind::InvalidRequest,
                    "Copilot rejected the requested model or input",
                ),
                Some("context_limit") => Failure::new(
                    FailureKind::Truncated,
                    "Copilot could not fit the requested document",
                ),
                _ => Failure::new(
                    FailureKind::Transport,
                    "Copilot reported a terminal model error",
                ),
            }),
            "session.idle" => {
                if data.get("aborted").and_then(Value::as_bool) == Some(true) {
                    return Err(Failure::new(
                        FailureKind::Cancelled,
                        "Copilot request was interrupted",
                    ));
                }
                if data.get("mode").and_then(Value::as_str) == Some("autopilot") {
                    return Err(protocol());
                }
                self.idle = true;
                Ok(())
            }
            "assistant.message" => self.record_message(data),
            _ => Ok(()),
        }
    }
    fn record_usage(&mut self, event: &Value, data: &Value) -> Result<(), Failure> {
        let id = required_string(event, "id", 512)?;
        let call = ObservedCall {
            model: required_string(data, "model", 512)?,
            input_tokens: counter(data, "inputTokens")?,
            output_tokens: counter(data, "outputTokens")?,
            cache_read_tokens: counter(data, "cacheReadTokens")?,
            cache_write_tokens: counter(data, "cacheWriteTokens")?,
        };
        if let Some(previous) = self.event_ids.get(&id) {
            return if previous == &call {
                Ok(())
            } else {
                Err(protocol())
            };
        }
        if self.event_ids.len() == MAX_CALLS {
            return Err(limit());
        }
        let key = optional_string(data, "apiCallId", 512)?.unwrap_or_else(|| id.clone());
        if let Some(previous) = self.calls.get(&key) {
            if previous != &call {
                return Err(protocol());
            }
        } else {
            self.calls.insert(key, call.clone());
            self.usage.calls.push(call.clone());
        }
        // Keep only observed counters, not potentially large quota/debug payloads.
        self.event_ids.insert(id, call.clone());
        if call.model != self.expected_model {
            return Err(Failure::new(
                FailureKind::Unsupported,
                "Copilot switched away from the requested model",
            ));
        }
        Ok(())
    }
    fn record_message(&mut self, data: &Value) -> Result<(), Failure> {
        if matches!(
            data.get("phase").and_then(Value::as_str),
            Some("analysis" | "reasoning")
        ) {
            return Ok(());
        }
        let origin = optional_string(data, "originatingMessageId", 512)?;
        if origin
            .as_ref()
            .zip(self.expected_origin.as_ref())
            .is_some_and(|(left, right)| left != right)
        {
            return Err(protocol());
        }
        if origin.is_some() {
            self.origin = origin;
        }
        let content = required_string(data, "content", MAX_TEXT)?;
        let count = counter(data, "chunkCount")?.unwrap_or(1);
        let index = counter(data, "chunkIndex")?.unwrap_or(0);
        if count == 0 || count > 256 || index >= count {
            return Err(protocol());
        }
        let key = if count == 1 {
            required_string(data, "messageId", 512)?
        } else {
            required_string(data, "apiCallId", 512)?
        };
        if self.message_key.as_ref() != Some(&key) {
            self.message_key = Some(key);
            self.chunks.clear();
            self.chunk_count = count as usize;
        }
        if self.chunk_count != count as usize {
            return Err(protocol());
        }
        if let Some(previous) = self.chunks.get(&(index as usize)) {
            if previous != &content {
                return Err(protocol());
            }
        } else {
            self.chunks.insert(index as usize, content);
        }
        if self.chunks.values().map(String::len).sum::<usize>() > MAX_TEXT {
            return Err(limit());
        }
        Ok(())
    }
    fn text(&self) -> Result<String, Failure> {
        if !self.idle || self.chunk_count == 0 || self.chunks.len() != self.chunk_count {
            return Err(Failure::new(
                FailureKind::Truncated,
                "Copilot completed without a complete final message",
            ));
        }
        let result: String = self.chunks.values().map(String::as_str).collect();
        if result.trim().is_empty() {
            return Err(Failure::new(
                FailureKind::Protocol,
                "Copilot completed with an empty response",
            ));
        }
        Ok(result)
    }
}
fn counter(value: &Value, key: &str) -> Result<Option<u64>, Failure> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(protocol),
    }
}
fn optional_string(value: &Value, key: &str, limit: usize) -> Result<Option<String>, Failure> {
    match value.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .filter(|value| value.len() <= limit)
            .map(|value| Some(value.to_owned()))
            .ok_or_else(protocol),
    }
}
fn required_string(value: &Value, key: &str, limit: usize) -> Result<String, Failure> {
    optional_string(value, key, limit)?
        .filter(|value| !value.is_empty())
        .ok_or_else(protocol)
}
