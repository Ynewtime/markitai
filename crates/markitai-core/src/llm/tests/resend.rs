//! A request is never sent again unchanged after the provider answered it
//! with nothing or with an answer the application rejected; every request
//! counts against the per-document budget.
use super::*;

/// A paid response without content, in the shape of Gemini's
/// OpenAI-compatible endpoint.
fn empty_gemini_answer() -> Value {
    json!({"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant"}}],"model":"gemini-flash-lite-latest","object":"chat.completion","usage":{"completion_tokens":0,"prompt_tokens":3903,"total_tokens":3903}})
}

/// The answer Gemini gave the live benchmark's 3-page scan of a widely
/// copied sample PDF (captured 2026-10-09; the native API reported
/// `finishReason: RECITATION` with citations of published copies).
fn recitation_answer() -> Value {
    json!({"choices":[{"finish_reason":"content_filter: RECITATION","index":0,"message":{"role":"assistant"}}],"created":1791512310,"id":"8U7IavrWEKfY9tMPoNKT0Ac","model":"gemini-flash-lite-latest","object":"chat.completion","usage":{"completion_tokens":0,"prompt_tokens":3903,"total_tokens":3903}})
}

const RECITATION: &str = "LLM provider withheld the answer (finish reason: content_filter: RECITATION): it resembles existing published text, which the provider does not reproduce";

fn scan(frames: &[VisionFrame<'_>], cfg: &Value) -> VisionFailure {
    process_vision_with_runtime(
        VisionRequest {
            markdown: "",
            source_label: "scan.pdf",
            cache_context: "scan.pdf",
            kind: VisionKind::PagedDocument,
            frames,
            unread_pages: &[],
            screenshot_only: false,
        },
        cfg,
        None,
    )
    .unwrap_err()
}

#[test]
fn a_recitation_block_is_named_and_never_sent_again() {
    let server = Mock::new(vec![(200, recitation_answer())]);
    let mut cfg = cfg("gemini/gemini-flash-lite-latest", &server.base);
    cfg["llm"]["max_requests_per_document"] = json!(3);
    let scope = DocumentScope::new(&cfg);
    let frames: Vec<_> = (1..=3)
        .map(|number| VisionFrame {
            number,
            mime: "image/png",
            bytes: b"scanned page",
        })
        .collect();
    let failure = scan(&frames, &cfg);
    assert_eq!(failure.error.to_string(), RECITATION);
    // A provider block is final for the document, like a refusal.
    assert!(!failure.allow_text_fallback);
    assert_eq!(server.finish().len(), 1);
    assert_eq!(scope.usage().requests, 1);
    drop(scope);

    // Plain requests name it too and do not resend it to the same model.
    let server = Mock::new(vec![(200, recitation_answer())]);
    let error = run(
        &plain(),
        &super::cfg("gemini/gemini-flash-lite-latest", &server.base),
        &HashMap::new(),
        &mut |_| panic!("a filtered answer is not retried after a backoff"),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), RECITATION);
    assert_eq!(server.finish().len(), 1);
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
            unread_pages: &[],
            screenshot_only: false,
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

fn tool_answer(input: &Value) -> Value {
    json!({"model":"gpt-4.1","choices":[{"message":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"MarkitaiDocument","arguments":input.to_string()}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2224,"completion_tokens":1338}})
}

#[test]
fn a_discarded_structured_answer_and_its_second_request_are_warned_and_counted() {
    let source = "# Slides\n\nThe deck's first slide.";
    let rejected = json!({"cleaned_markdown":source,"frontmatter":{"description":"","tags":[]}});
    let accepted =
        json!({"cleaned_markdown":source,"frontmatter":{"description":"A deck","tags":["deck"]}});
    let server = Mock::new(vec![
        (200, tool_answer(&rejected)),
        (200, success(&accepted.to_string())),
    ]);
    let cfg = cfg("openai/gpt-4.1", &server.base);
    let enhanced =
        process_document_with_runtime(source, "deck.pptx", "deck.pptx", false, &cfg, None).unwrap();
    assert_eq!(enhanced.usage.requests, 2);
    assert_eq!(
        enhanced.warnings,
        [
            "A model answer could not be used, so the request was sent again; a later answer was used"
        ]
    );
    let requests = server.finish();
    assert!(requests[0].1.get("tools").is_some());
    assert!(
        requests[1]
            .1
            .pointer("/response_format/json_schema")
            .is_some()
    );

    // With a budget of one request the discarded answer is the only one.
    let server = Mock::new(vec![(200, tool_answer(&rejected))]);
    let mut cfg = super::cfg("openai/gpt-4.1", &server.base);
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
        [
            "LLM request to openai/test failed (LLM returned HTTP 503: overloaded) and was sent again"
        ]
    );
    assert_eq!(server.finish().len(), 2);
    drop(scope);

    // The second rejection repeats the first: a third request would be the
    // second one again, so the separate requests follow at once.
    let server = Mock::new(vec![
        (200, success("not structured JSON")),
        (200, success("still not structured JSON")),
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
    let requests = server.finish();
    assert_eq!(requests.len(), 4);
    assert_ne!(requests[0].1, requests[1].1);
}

#[test]
fn a_repeated_rejection_is_never_answered_by_the_same_request() {
    // The same rejection twice ends the structured attempts after two
    // requests; a different one earns a third, which names it. The separate
    // caption and description requests follow either way.
    let wrong_type = json!({"caption": 1, "description": "d", "extracted_text": ""}).to_string();
    for (structured, total) in [
        (vec!["no JSON", "no JSON either"], 4),
        (vec!["no JSON", wrong_type.as_str(), "no JSON again"], 5),
    ] {
        let answers = structured
            .iter()
            .chain(&["Caption", "Description"])
            .map(|text| (200, success(text)))
            .collect();
        let server = Mock::new(answers);
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
        drop(scope);
        let requests = server.finish();
        assert_eq!(requests.len(), total);
        let sent: Vec<&Value> = requests[..structured.len()]
            .iter()
            .map(|(_, body)| body)
            .collect();
        for (index, request) in sent.iter().enumerate() {
            for earlier in &sent[..index] {
                assert_ne!(
                    request, earlier,
                    "a structured request was sent twice unchanged"
                );
            }
        }
    }
}

fn doc_tool_answer(markdown: &str) -> Value {
    tool_answer(
        &json!({"cleaned_markdown":markdown,"frontmatter":{"description":"A scan","tags":["scan"]}}),
    )
}

fn scanned_frames() -> Vec<VisionFrame<'static>> {
    (1..=3)
        .map(|number| VisionFrame {
            number,
            mime: "image/png",
            bytes: b"scanned page",
        })
        .collect()
}

fn read_scans(
    frames: &[VisionFrame<'_>],
    cfg: &Value,
    unread: &[usize],
) -> std::result::Result<Enhancement, VisionFailure> {
    process_vision_with_runtime(
        VisionRequest {
            markdown: "",
            source_label: "scan.pdf",
            cache_context: "scan.pdf",
            kind: VisionKind::PagedDocument,
            frames,
            unread_pages: unread,
            screenshot_only: false,
        },
        cfg,
        None,
    )
}

#[test]
fn gemini_flash_names_and_aliases_answer_in_a_forced_tool_call_first() {
    for model in [
        "gemini/gemini-3.8-flash",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-flash-latest",
    ] {
        let server = Mock::new(vec![(
            200,
            doc_tool_answer("# Scan\n\nThe recovered page text."),
        )]);
        let cfg = cfg(model, &server.base);
        let scope = DocumentScope::new(&cfg);
        let answer = read_scans(&scanned_frames(), &cfg, &[]).unwrap();
        assert!(
            answer.markdown.contains("The recovered page text."),
            "{model}"
        );
        drop(scope);
        let requests = server.finish();
        assert_eq!(requests.len(), 1, "{model}");
        let body = &requests[0].1;
        assert_eq!(
            body["tools"][0]["function"]["name"], "MarkitaiDocument",
            "{model}"
        );
        assert_eq!(
            body["tool_choice"]["function"]["name"], "MarkitaiDocument",
            "{model}"
        );
        assert!(body.get("response_format").is_none(), "{model}");
    }
}

#[test]
fn a_recitation_stop_on_a_scan_is_requested_once_more_in_the_next_mode() {
    let server = Mock::new(vec![
        (200, recitation_answer()),
        (200, document_answer("# Scan\n\nThe recovered page text.")),
    ]);
    let mut cfg = cfg("gemini/gemini-flash-latest", &server.base);
    cfg["llm"]["max_requests_per_document"] = json!(5);
    let scope = DocumentScope::new(&cfg);
    let answer = read_scans(&scanned_frames(), &cfg, &[1, 2, 3]).unwrap();
    assert!(answer.markdown.contains("The recovered page text."));
    let requests = server.finish();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].1.get("tools").is_some() && requests[0].1.get("response_format").is_none());
    assert!(requests[1].1.get("tools").is_none());
    assert_eq!(requests[1].1["response_format"]["type"], "json_schema");
    assert_eq!(scope.usage().requests, 2);
    let mut warnings = answer.warnings.clone();
    warnings.extend(scope.take_warnings());
    assert!(
        warnings.iter().any(|w| w
            == "The provider withheld a model answer as a recitation of published text, so the request was sent again; a later answer was used"),
        "{warnings:?}"
    );
}

#[test]
fn a_second_recitation_stop_fails_after_exactly_one_extra_request() {
    let server = Mock::new(vec![(200, recitation_answer()), (200, recitation_answer())]);
    let mut cfg = cfg("gemini/gemini-flash-latest", &server.base);
    cfg["llm"]["max_requests_per_document"] = json!(5);
    let scope = DocumentScope::new(&cfg);
    let failure = read_scans(&scanned_frames(), &cfg, &[1, 2, 3]).unwrap_err();
    assert_eq!(failure.error.to_string(), RECITATION);
    assert_eq!(server.finish().len(), 2);
    assert_eq!(scope.usage().requests, 2);
}

#[test]
fn a_recitation_stop_without_unread_pages_or_another_mode_is_not_sent_again() {
    // Pages with text: the stop is final, as before.
    let server = Mock::new(vec![(200, recitation_answer())]);
    let cfg_text = cfg("gemini/gemini-flash-latest", &server.base);
    let scope = DocumentScope::new(&cfg_text);
    assert!(read_scans(&scanned_frames(), &cfg_text, &[]).is_err());
    assert_eq!(server.finish().len(), 1);
    drop(scope);
    // A model with no other answer mode to try.
    let server = Mock::new(vec![(200, recitation_answer())]);
    let cfg_other = cfg("gemini/gemini-2.5-flash", &server.base);
    let scope = DocumentScope::new(&cfg_other);
    assert!(read_scans(&scanned_frames(), &cfg_other, &[1, 2, 3]).is_err());
    assert_eq!(server.finish().len(), 1);
    drop(scope);
}

#[test]
fn scanned_pages_are_asked_for_in_the_cleanup_framing_with_the_rules_once() {
    let server = Mock::new(vec![(
        200,
        doc_tool_answer("# Scan\n\nThe recovered page text."),
    )]);
    let cfg = cfg("gemini/gemini-flash-latest", &server.base);
    let scope = DocumentScope::new(&cfg);
    read_scans(&scanned_frames(), &cfg, &[1, 2, 3]).unwrap();
    drop(scope);
    let requests = server.finish();
    let system = requests[0].1["messages"][0]["content"].as_str().unwrap();
    assert!(
        system.contains("Frames 1, 2, 3 are scanned pages"),
        "{system}"
    );
    assert!(
        system.contains("Supplement the missing content from the page image"),
        "{system}"
    );
    assert!(system.contains("original language and wording"), "{system}");
    // The content of page N follows page N's own tokens, not the page before it.
    assert!(
        system.contains("directly after that page's own tokens"),
        "{system}"
    );
    assert!(
        system.contains("the content after page marker N is page N's"),
        "{system}"
    );
    assert!(!system.contains("transcribe all of its text"), "{system}");
    assert_eq!(
        system.matches("Read every attached image in order").count(),
        1,
        "{system}"
    );
    // Each page image comes right after its own label, as Python labels them.
    let content = requests[0].1["messages"][1]["content"].as_array().unwrap();
    let parts: Vec<String> = content[1..]
        .iter()
        .map(|part| match part["type"].as_str().unwrap() {
            "text" => part["text"].as_str().unwrap().to_owned(),
            other => other.to_owned(),
        })
        .collect();
    assert_eq!(
        parts,
        [
            "Page 1:",
            "image_url",
            "Page 2:",
            "image_url",
            "Page 3:",
            "image_url"
        ]
    );
}

#[test]
fn a_later_batch_of_scanned_pages_is_resent_once_after_a_recitation_stop() {
    // Twelve pages are two batches; page 11 is a scan in the second. One
    // request at a time, so the answers arrive in the order given.
    let server = Mock::new(vec![
        (
            200,
            doc_tool_answer("# First ten pages\n\nText of the first batch."),
        ),
        (200, recitation_answer()),
        (200, document_answer("Text of the scanned page eleven.")),
    ]);
    let mut cfg = cfg("gemini/gemini-flash-latest", &server.base);
    cfg["llm"]["max_requests_per_document"] = json!(6);
    cfg["llm"]["concurrency"] = json!(1);
    let frames: Vec<_> = (1..=12)
        .map(|number| VisionFrame {
            number,
            mime: "image/png",
            bytes: b"page",
        })
        .collect();
    let scope = DocumentScope::new(&cfg);
    let answer = read_scans(&frames, &cfg, &[11]).unwrap();
    assert!(answer.markdown.contains("Text of the first batch."));
    assert!(answer.markdown.contains("Text of the scanned page eleven."));
    let requests = server.finish();
    assert_eq!(requests.len(), 3);
    assert!(requests[1].1.get("tools").is_some());
    assert_eq!(requests[2].1["response_format"]["type"], "json_schema");
    assert_eq!(scope.usage().requests, 3);
}
