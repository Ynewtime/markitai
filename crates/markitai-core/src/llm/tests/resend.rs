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
