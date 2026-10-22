use super::{Failure, FailureKind, TEXT_LIMIT, TokenTotals, UsageEvidence, limit, protocol};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Default)]
pub(super) struct Events {
    thread: bool,
    turn: bool,
    completed: bool,
    completed_ids: HashSet<String>,
    text: Option<String>,
    text_bytes: usize,
    pub(super) usage: UsageEvidence,
}
/// What a failed turn's (lowercased) words say, as a fixed error: the
/// plan's usage limit, which lasts hours; a rate limit or another failed
/// turn, which may pass.
fn turn_failure(said: &str) -> Failure {
    if said.contains("usage limit") || said.contains("usage_limit") {
        Failure::new(
            FailureKind::ResourceLimit,
            "The ChatGPT plan's usage limit is reached; Codex can be used again once it resets",
        )
    } else if said.contains("429")
        || said.contains("too many requests")
        || said.contains("rate limit")
    {
        Failure::new(
            FailureKind::RateLimit,
            "Codex was rate limited (HTTP 429) and did not complete the document request",
        )
    } else {
        Failure::new(
            FailureKind::Transport,
            "Codex did not complete the document request",
        )
    }
}

fn string(value: &Value, max: usize) -> Result<&str, Failure> {
    value
        .as_str()
        .filter(|s| s.len() <= max && !s.contains('\0'))
        .ok_or_else(protocol)
}
fn tokens(value: &Value) -> Result<u64, Failure> {
    value
        .as_u64()
        .filter(|n| *n <= 9_007_199_254_740_991)
        .ok_or_else(protocol)
}
fn totals(value: &Value) -> Result<TokenTotals, Failure> {
    let result = TokenTotals {
        input_tokens: tokens(&value["input_tokens"])?,
        output_tokens: tokens(&value["output_tokens"])?,
        cached_input_tokens: tokens(&value["cached_input_tokens"])?,
        cache_creation_input_tokens: value
            .get("cache_write_input_tokens")
            .map(tokens)
            .transpose()?
            .unwrap_or(0),
        reasoning_output_tokens: tokens(&value["reasoning_output_tokens"])?,
    };
    if result.cached_input_tokens > result.input_tokens
        || result.cache_creation_input_tokens > result.input_tokens
        || result.reasoning_output_tokens > result.output_tokens
    {
        return Err(protocol());
    }
    Ok(result)
}
fn startup_warning(message: &str) -> bool {
    message.starts_with("`[features].codex_hooks` is deprecated.")
        || message.starts_with("`[features].memory_tool` is deprecated.")
        || message.starts_with("Under-development features enabled: skip_host_skill_discovery.")
}
impl Events {
    pub(super) fn accept(&mut self, value: &Value) -> Result<(), Failure> {
        if self.completed {
            return Err(protocol());
        }
        match value["type"].as_str() {
            Some("thread.started") if !self.thread && !self.turn => {
                if string(&value["thread_id"], 256)?.is_empty() {
                    return Err(protocol());
                }
                self.thread = true;
            }
            Some("turn.started") if self.thread && !self.turn => self.turn = true,
            Some("item.started" | "item.updated" | "item.completed") if self.thread => {
                let item = &value["item"];
                let id = string(&item["id"], 256)?;
                if id.is_empty() {
                    return Err(protocol());
                }
                let done = value["type"] == "item.completed";
                if done && !self.completed_ids.insert(id.into()) {
                    return Err(protocol());
                }
                match item["type"].as_str() {
                    Some("agent_message") if self.turn => {
                        let text = string(&item["text"], TEXT_LIMIT)?;
                        if done {
                            self.text_bytes =
                                self.text_bytes.checked_add(text.len()).ok_or_else(limit)?;
                            if self.text_bytes > TEXT_LIMIT {
                                return Err(limit());
                            }
                            // Official exec also selects the last completed agent message.
                            self.text = Some(text.into());
                        }
                    }
                    Some("reasoning") if self.turn => {
                        string(&item["text"], TEXT_LIMIT)?;
                    }
                    Some("error") if !self.turn && done => {
                        if !startup_warning(string(&item["message"], 16 * 1024)?) {
                            return Err(Failure::new(
                                FailureKind::Protocol,
                                "Codex reported an unexpected startup warning",
                            ));
                        }
                    }
                    Some(
                        "command_execution" | "file_change" | "mcp_tool_call" | "collab_tool_call"
                        | "web_search" | "todo_list",
                    ) => {
                        return Err(Failure::new(
                            FailureKind::Permission,
                            "Codex attempted an unrequested tool or background operation",
                        ));
                    }
                    _ => return Err(protocol()),
                }
            }
            Some("turn.completed") if self.thread && self.turn => {
                self.usage.aggregate = Some(totals(&value["usage"])?);
                self.completed = true;
            }
            Some("turn.failed" | "error") => {
                // Raw diagnostics may include credentials or model text. The
                // public error is fixed; retain any previously observed totals.
                let said = value
                    .pointer("/error/message")
                    .or_else(|| value.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                return Err(turn_failure(&said));
            }
            _ => return Err(protocol()),
        }
        Ok(())
    }
    pub(super) fn finish(&self, success: bool) -> Result<String, Failure> {
        if !success {
            return Err(Failure::new(
                FailureKind::Transport,
                "Codex exited before successful completion",
            ));
        }
        if !self.completed {
            return Err(Failure::new(
                FailureKind::Truncated,
                "Codex ended without a completed turn",
            ));
        }
        self.text
            .as_ref()
            .filter(|text| !text.trim().is_empty())
            .cloned()
            .ok_or_else(protocol)
    }
}
