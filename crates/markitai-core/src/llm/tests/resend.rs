//! A request is never sent again unchanged after the provider answered it
//! with nothing or with an answer the application rejected; every request
//! counts against the per-document budget.
use super::*;

/// Gemini's OpenAI-compatible endpoint, answering a recitation-blocked
/// request: a paid response without content.
fn empty_gemini_answer() -> Value {
    json!({"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant"}}],"model":"gemini-flash-lite-latest","object":"chat.completion","usage":{"completion_tokens":0,"prompt_tokens":3903,"total_tokens":3903}})
}

fn document_answer(markdown: &str) -> Value {
    success(
        &json!({"cleaned_markdown":markdown,"frontmatter":{"description":"A scan","tags":["scan"]}})
            .to_string(),
    )
}

#[test]
fn an_empty_visual_answer_is_sent_once_not_until_the_cap() {
    let server = Mock::new(vec![(200, empty_gemini_answer())]);
    let mut cfg = cfg("gemini/gemini-flash-lite-latest", &server.base);
    cfg["llm"]["max_requests_per_document"] = json!(3);
    cfg["llm"]["router_settings"]["num_retries"] = json!(1);
    let scope = DocumentScope::new(&cfg);
    let frames: Vec<_> = (1..=3)
        .map(|number| VisionFrame {
            number,
            mime: "image/png",
            bytes: b"scanned page",
        })
        .collect();
    let failure = process_vision_with_runtime(
        VisionRequest {
            markdown: "",
            source_label: "scan.pdf",
            cache_context: "scan.pdf",
            kind: VisionKind::PagedDocument,
            frames: &frames,
        },
        &cfg,
        None,
    )
    .unwrap_err();
    assert_eq!(
        failure.error.to_string(),
        "LLM returned no text (finish reason: stop)"
    );
    assert_eq!(server.finish().len(), 1);
    let usage = scope.usage();
    assert_eq!((usage.requests, usage.input_tokens), (1, 3903));
}

#[test]
fn a_rejected_structured_answer_is_retried_with_a_different_request() {
    let source = "# Scan\n\nThe recovered page text.";
    let server = Mock::new(vec![
        (200, success("not structured JSON")),
        (200, document_answer(source)),
    ]);
    let cfg = cfg("openai/test", &server.base);
    let enhanced =
        process_document_with_runtime(source, "scan.md", "scan.md", false, &cfg, None).unwrap();
    assert_eq!(enhanced.markdown, source);
    let requests = server.finish();
    assert_eq!(requests.len(), 2);
    let system = |index: usize| {
        requests[index].1["messages"][0]["content"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    assert!(!system(0).contains("was rejected"));
    assert!(
        system(1).ends_with("The previous answer to this request was rejected: LLM response is not valid structured JSON. Return a complete answer that follows these instructions exactly."),
        "{}",
        system(1)
    );
    assert_eq!(requests[0].1["messages"][1], requests[1].1["messages"][1]);
}

#[test]
fn an_empty_paid_answer_is_not_sent_again_and_its_usage_is_kept() {
    let server = Mock::new(vec![(200, success(" "))]);
    let cfg = cfg("openai/test", &server.base);
    let scope = DocumentScope::new(&cfg);
    let error = run(&plain(), &cfg, &HashMap::new(), &mut |_| {
        panic!("an empty answer is not retried after a backoff")
    })
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "LLM returned no text (finish reason: stop)"
    );
    assert_eq!(server.finish().len(), 1);
    let usage = scope.usage();
    assert_eq!(
        (usage.requests, usage.input_tokens, usage.output_tokens),
        (1, 11, 7)
    );
}

#[test]
fn an_empty_answer_moves_to_a_sibling_deployment_once() {
    let server = Mock::new(vec![(200, success(" ")), (200, success("real"))]);
    let cfg = config::normalize(&json!({"llm":{"enabled":true,"model_list":[
        {"model_name":"default","litellm_params":{"model":"openai/first","api_key":"fake-test-key","api_base":server.base}},
        {"model_name":"default","litellm_params":{"model":"openai/second","api_key":"fake-test-key","api_base":server.base}}
    ],"router_settings":{"timeout":3,"num_retries":2}},"cache":{"enabled":false}}))
    .unwrap();
    let (text, usage) = run(&plain(), &cfg, &HashMap::new(), &mut |_| {
        panic!("a sibling answers without a backoff")
    })
    .unwrap();
    assert_eq!((text.as_str(), usage.requests), ("real", 2));
    let requests = server.finish();
    assert_ne!(requests[0].1["model"], requests[1].1["model"]);
}

fn haiku_tool_answer(input: Value) -> Value {
    json!({"model":"claude-haiku-4-5","content":[{"type":"tool_use","id":"toolu_1","name":"MarkitaiDocument","input":input}],"stop_reason":"tool_use","usage":{"input_tokens":2224,"output_tokens":1338}})
}

fn haiku_text_answer(text: &str) -> Value {
    json!({"model":"claude-haiku-4-5","content":[{"type":"text","text":text}],"stop_reason":"end_turn","usage":{"input_tokens":1754,"output_tokens":1374}})
}

#[test]
fn a_discarded_structured_answer_and_its_second_request_are_warned_and_counted() {
    let source = "# Slides\n\nThe deck's first slide.";
    let rejected = json!({"cleaned_markdown":source,"frontmatter":{"description":"","tags":[]}});
    let accepted =
        json!({"cleaned_markdown":source,"frontmatter":{"description":"A deck","tags":["deck"]}});
    let server = Mock::new(vec![
        (200, haiku_tool_answer(rejected.clone())),
        (200, haiku_text_answer(&accepted.to_string())),
    ]);
    let cfg = cfg("anthropic/claude-haiku-4-5", &server.base);
    let enhanced =
        process_document_with_runtime(source, "deck.pptx", "deck.pptx", false, &cfg, None).unwrap();
    assert_eq!(enhanced.usage.requests, 2);
    assert_eq!(
        enhanced.warnings,
        [
            "LLM tool-call answer was rejected (LLM document response requires cleaned_markdown plus nonempty description and string tags); the request was sent again in JSON-schema mode"
        ]
    );
    let requests = server.finish();
    assert!(requests[0].1.get("tools").is_some());
    assert!(requests[1].1.pointer("/output_config/format").is_some());

    // With a budget of one request the discarded answer is the only one.
    let server = Mock::new(vec![(200, haiku_tool_answer(rejected))]);
    let mut cfg = super::cfg("anthropic/claude-haiku-4-5", &server.base);
    cfg["llm"]["max_requests_per_document"] = json!(1);
    let scope = DocumentScope::new(&cfg);
    let error = process_document_with_runtime(source, "deck.pptx", "deck.pptx", false, &cfg, None)
        .unwrap_err();
    assert!(error.to_string().contains("budget exhausted"), "{error}");
    assert_eq!(server.finish().len(), 1);
    assert_eq!(scope.usage().requests, 1);
    assert!(scope.take_warnings().is_empty());
}

#[test]
fn transport_retries_and_image_analysis_fallbacks_are_warned() {
    let server = Mock::new(vec![
        (503, json!({"error":{"message":"overloaded"}})),
        (200, success("real")),
    ]);
    let cfg = cfg("openai/test", &server.base);
    let scope = DocumentScope::new(&cfg);
    run(&plain(), &cfg, &HashMap::new(), &mut |_| {}).unwrap();
    assert_eq!(
        scope.take_warnings(),
        ["LLM request to openai/test failed (LLM returned HTTP 503) and was sent again"]
    );
    assert_eq!(server.finish().len(), 2);
    drop(scope);

    let server = Mock::new(vec![
        (200, success("not structured JSON")),
        (200, success("still not structured JSON")),
        (200, success("never structured JSON")),
        (200, success("Caption")),
        (200, success("Description")),
    ]);
    let cfg = super::cfg("openai/test", &server.base);
    let scope = DocumentScope::new(&cfg);
    analyze_images_with_runtime(
        "context",
        "source",
        &[("image/png", b"bytes")],
        &cfg,
        None,
        None,
    )
    .unwrap();
    let warnings = scope.take_warnings();
    assert_eq!(
        warnings,
        [
            "LLM JSON-text answer was rejected (LLM response is not valid structured JSON); the request was sent again in JSON-text mode",
            "Structured image analysis failed (LLM response is not valid structured JSON); separate caption and description requests were sent",
        ],
        "{warnings:?}"
    );
    assert_eq!(server.finish().len(), 5);
}
