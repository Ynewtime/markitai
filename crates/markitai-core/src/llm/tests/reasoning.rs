//! `litellm_params.reasoning_effort`: DeepSeek's thinking switch and effort,
//! the OpenAI-compatible field elsewhere, explicit refusals where it cannot
//! be honoured, and a truncation error that names spent reasoning.
use super::*;

fn entry(params: Value) -> Result<Deployment> {
    let cfg = config::normalize(
        &json!({"llm":{"model_list":[{"model_name":"default","litellm_params":params}]}}),
    )?;
    deployments(&cfg, &HashMap::new()).map(|mut entries| entries.remove(0))
}

fn sent(params: Value) -> Value {
    payload(&entry(params).unwrap(), &plain())
}

#[test]
fn deepseek_cleans_without_thinking_unless_an_effort_is_configured() {
    let base = json!({"model":"deepseek/deepseek-flash","api_key":"fake-test-key"});
    let request = sent(base.clone());
    assert_eq!(request["thinking"], json!({"type":"disabled"}));
    assert!(request.get("reasoning_effort").is_none());

    let mut low = base.clone();
    low["reasoning_effort"] = json!("low");
    let request = sent(low);
    assert_eq!(request["thinking"], json!({"type":"enabled"}));
    assert_eq!(request["reasoning_effort"], "low");

    let mut none = base;
    none["reasoning_effort"] = json!("none");
    assert_eq!(sent(none)["thinking"], json!({"type":"disabled"}));
}

#[test]
fn other_chat_providers_send_the_effort_only_when_configured() {
    let request = sent(json!({"model":"openai/gpt-test","api_key":"k"}));
    assert!(request.get("reasoning_effort").is_none() && request.get("thinking").is_none());
    for (model, effort) in [("openai/gpt-test", "high"), ("gemini/gemini-test", "none")] {
        let request = sent(json!({"model":model,"api_key":"k","reasoning_effort":effort}));
        assert_eq!(request["reasoning_effort"], effort, "{model}");
        assert!(request.get("thinking").is_none(), "{model}");
    }
}

#[test]
fn unsupported_reasoning_settings_are_explicit_errors() {
    let anthropic =
        sent(json!({"model":"anthropic/claude-haiku-4-5","api_key":"k","reasoning_effort":"none"}));
    assert!(anthropic.get("reasoning_effort").is_none() && anthropic.get("thinking").is_none());
    let error = entry(
        json!({"model":"anthropic/claude-haiku-4-5","api_key":"k","reasoning_effort":"high"}),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error}");
    for model in ["claude-agent/sonnet", "copilot/fixture", "chatgpt/gpt-5.5"] {
        let error = entry(json!({"model":model,"reasoning_effort":"low"})).unwrap_err();
        assert!(
            error.to_string().contains("reasoning_effort"),
            "{model}: {error}"
        );
    }
    let error =
        entry(json!({"model":"deepseek/deepseek-flash","reasoning_effort":"extreme"})).unwrap_err();
    assert!(
        error.to_string().contains(
            "reasoning_effort must be one of: none, minimal, low, medium, high, xhigh, max"
        ),
        "{error}"
    );
}

fn exhausted() -> Value {
    json!({"model":"deepseek-flash","choices":[{"message":{"content":""},"finish_reason":"length"}],"usage":{"prompt_tokens":1799,"completion_tokens":8192,"completion_tokens_details":{"reasoning_tokens":8192}}})
}

#[test]
fn a_budget_spent_on_reasoning_is_named_in_the_truncation_error() {
    let message = "LLM output was truncated by its token limit: reasoning used all 8192 output tokens; lower litellm_params.reasoning_effort or raise max_tokens";
    let server = Mock::new(vec![(200, exhausted())]);
    let mut cfg = cfg("deepseek/deepseek-flash", &server.base);
    cfg["llm"]["model_list"][0]["litellm_params"]["reasoning_effort"] = json!("max");
    let error = run(&plain(), &cfg, &HashMap::new(), &mut |_| {}).unwrap_err();
    assert_eq!(error.to_string(), message);
    let requests = server.finish();
    assert_eq!(requests[0].1["reasoning_effort"], "max");

    // A structured clean-up fails the same way, after one request.
    let server = Mock::new(vec![(200, exhausted())]);
    let source = "# Essay\n\nA paragraph.";
    let error = process_document_with_runtime(
        source,
        "essay.html",
        "essay.html",
        false,
        &super::cfg("deepseek/deepseek-flash", &server.base),
        None,
    )
    .unwrap_err();
    assert_eq!(error.to_string(), message);
    let requests = server.finish();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].1["thinking"], json!({"type":"disabled"}));

    // Without reasoning counts it names the count and the setting to raise.
    let server = Mock::new(vec![(
        200,
        json!({"choices":[{"message":{"content":"cut"},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":9}}),
    )]);
    let error = run(
        &plain(),
        &super::cfg("openai/test", &server.base),
        &HashMap::new(),
        &mut |_| {},
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "LLM output was truncated by its token limit after 9 output tokens; raise litellm_params.max_tokens for this model"
    );
    server.finish();
}

#[test]
fn models_that_reason_by_default_are_asked_not_to() {
    // OpenAI models that accept `none`, with or without a dated suffix.
    for model in [
        "openai/gpt-6-luna",
        "openai/gpt-5.5-2026-04-23",
        "openai/gpt-5.1",
    ] {
        let request = sent(json!({"model":model,"api_key":"k"}));
        assert_eq!(request["reasoning_effort"], "none", "{model}");
    }
    // Not those that refuse it, nor a name that only starts like one.
    for model in [
        "openai/gpt-6.1-sol",
        "openai/gpt-5",
        "openai/gpt-5.10",
        "azure/gpt-6-luna",
    ] {
        let mut params = json!({"model":model,"api_key":"k"});
        if model.starts_with("azure/") {
            params["api_base"] = json!("https://example.openai.azure.com");
            params["api_version"] = json!("2024-10-21");
        }
        assert!(sent(params).get("reasoning_effort").is_none(), "{model}");
    }
    // A configured effort wins.
    let request = sent(json!({"model":"openai/gpt-6-luna","api_key":"k","reasoning_effort":"low"}));
    assert_eq!(request["reasoning_effort"], "low");
    // Claude Haiku 5.5 thinks unless told not to; the other current Claude
    // models refuse the switch, and older ones do not think by default.
    let haiku = sent(json!({"model":"anthropic/claude-haiku-5-5","api_key":"k"}));
    assert_eq!(haiku["thinking"], json!({"type":"disabled"}));
    for model in [
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-opus-5-5",
        "anthropic/claude-fable-5-1",
        "anthropic/claude-haiku-4-5",
    ] {
        assert!(
            sent(json!({"model":model,"api_key":"k"}))
                .get("thinking")
                .is_none(),
            "{model}"
        );
    }
}

#[test]
fn openai_caps_use_max_completion_tokens_where_required() {
    let field = |model: &str, base: Option<&str>| {
        let mut params = json!({"model":model,"api_key":"k","max_tokens":512});
        if let Some(base) = base {
            params["api_base"] = json!(base);
        }
        let request = sent(params);
        if request.get("max_completion_tokens").is_some() {
            "max_completion_tokens"
        } else {
            assert_eq!(request["max_tokens"], 512, "{model}");
            "max_tokens"
        }
    };
    // OpenAI itself: always.
    assert_eq!(field("openai/gpt-4.1", None), "max_completion_tokens");
    // Elsewhere: GPT-5 and later, and the o-series.
    let gateway = Some("http://127.0.0.1:9/v1");
    for model in [
        "openai/gpt-6-luna",
        "openai/gpt-5.5",
        "openai/o4-mini",
        "openai/org/gpt-7",
    ] {
        assert_eq!(field(model, gateway), "max_completion_tokens", "{model}");
    }
    for model in [
        "openai/llama-3",
        "openai/gpt-4o",
        "openai/omni-x",
        "groq/gpt-6-luna",
    ] {
        assert_eq!(field(model, gateway), "max_tokens", "{model}");
    }
}

#[test]
fn a_configured_cap_above_the_model_limit_is_lowered_to_it() {
    let request =
        sent(json!({"model":"gemini/gemini-3.8-flash","api_key":"k","max_tokens":1_000_000}));
    assert_eq!(request["max_tokens"], 65_536);
    let request = sent(json!({"model":"gemini/gemini-3.8-flash","api_key":"k","max_tokens":2_000}));
    assert_eq!(request["max_tokens"], 2_000);
    // A model not reviewed keeps the configured cap.
    let request =
        sent(json!({"model":"gemini/gemini-unreviewed","api_key":"k","max_tokens":1_000_000}));
    assert_eq!(request["max_tokens"], 1_000_000);
}
