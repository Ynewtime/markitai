use super::*;
use crate::llm::{
    Deployment, DocumentScope, Prompts, Protocol, admit_document_attempt_for, price_identity,
    record_usage_class, run_with_runtime,
};
use crate::pricing::UnknownPrice;
use std::collections::HashMap;
use std::sync::{Arc, Barrier};

fn entry() -> Deployment {
    Deployment {
        id: "openai/gpt-4.1".into(),
        explicit_id: None,
        group: "default".into(),
        model: "gpt-4.1".into(),
        provider: "openai".into(),
        weight: 1,
        key: Some("test-not-a-provider-key".into()),
        endpoint: "https://api.openai.com/v1/chat/completions".into(),
        protocol: Protocol::Chat,
        max_tokens: None,
        supports_vision: None,
    }
}
fn cfg(limit: f64) -> Value {
    json!({"llm":{"max_cost_per_document_usd":limit,"max_requests_per_document":50}})
}
fn paid(input: u64, output: u64) -> Value {
    json!({"usage":{"prompt_tokens":input,"completion_tokens":output}})
}
fn known() -> Quote {
    pricing::quote(
        &price_identity(&entry()),
        &paid(1000, 100),
        BillingClass::Standard,
    )
}

#[test]
fn merge_retains_known_subtotal_and_marks_legacy_and_unknown_attempts() {
    let mut usage = response("m", 1000, 100, known());
    let first = usage.clone();
    merge(
        &mut usage,
        &response("m", 2, 3, Quote::Unknown(UnknownPrice::Model)),
    );
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.cost_usd, 0.0028);
    assert_eq!(usage.by_model["m"]["priced_requests"], 1);
    assert_eq!(usage.by_model["m"]["unpriced_requests"], 1);
    assert_eq!(usage.by_model["m"]["cost_status"], "partial");
    let delta = difference(&usage, &first);
    assert_eq!(delta.requests, 1);
    assert_eq!(delta.cost_usd, 0.0);
    assert_eq!(delta.by_model["m"]["cost_status"], "unknown");
    assert!(delta.by_model["m"].get("pricing_snapshot").is_none());
    let mut legacy = ConversionUsage {
        requests: 1,
        cost_usd: 0.2,
        ..Default::default()
    };
    legacy.by_model.insert(
        "m".into(),
        json!({"requests":1,"input_tokens":0,"output_tokens":0,"cost_usd":0.2}),
    );
    merge(&mut legacy, &first);
    assert_eq!(legacy.by_model["m"]["cost_status"], "partial");
    assert_eq!(legacy.by_model["m"]["unpriced_requests"], 1);
    assert!(difference(&usage, &usage).by_model.is_empty());
}

#[test]
fn mixed_snapshot_provenance_is_sorted_and_not_silently_overwritten() {
    let Quote::Known { amount, tariff, .. } = known() else {
        panic!("known tariff");
    };
    let mut usage = response(
        "m",
        1,
        1,
        Quote::Known {
            amount,
            tariff,
            snapshot: "z",
        },
    );
    merge(
        &mut usage,
        &response(
            "m",
            1,
            1,
            Quote::Known {
                amount,
                tariff,
                snapshot: "a",
            },
        ),
    );
    assert!(usage.by_model["m"].get("pricing_snapshot").is_none());
    assert_eq!(usage.by_model["m"]["pricing_snapshots"], json!(["a", "z"]));
    assert_eq!(usage.by_model["m"]["cost_status"], "complete");
}

#[test]
fn equality_allows_next_attempt_and_crossing_preserves_paid_result() {
    let scope = DocumentScope::new(&cfg(0.125));
    let entry = entry();
    let mut usage = ConversionUsage::default();
    admit_document_attempt_for(Some(&entry)).unwrap();
    record_usage_class(&mut usage, &entry, &paid(62_500, 0), BillingClass::Standard);
    admit_document_attempt_for(Some(&entry)).unwrap();
    record_usage_class(&mut usage, &entry, &paid(1, 0), BillingClass::Standard);
    let error = admit_document_attempt_for(Some(&entry)).unwrap_err();
    assert!(error.to_string().contains("dollar budget exhausted"));
    assert_eq!(scope.current.lock().unwrap().attempts, 2);
    assert_eq!(scope.usage().requests, 2);
    assert_eq!(scope.usage().input_tokens, 62_501);
    assert!((scope.usage().cost_usd - 0.125002).abs() < 1e-12);
}

#[test]
fn missing_price_after_response_stops_only_budgeted_continuations() {
    let entry = entry();
    for limit in [0.0, 0.125] {
        let scope = DocumentScope::new(&cfg(limit));
        let mut usage = ConversionUsage::default();
        admit_document_attempt_for(Some(&entry)).unwrap();
        record_usage_class(
            &mut usage,
            &entry,
            &json!({"usage":{"prompt_tokens":1}}),
            BillingClass::Standard,
        );
        assert_eq!(scope.usage().by_model[&entry.id]["cost_status"], "unknown");
        assert_eq!(scope.usage().requests, 1);
        assert_eq!(
            admit_document_attempt_for(Some(&entry)).is_err(),
            limit > 0.0
        );
    }
}

#[test]
fn invalid_or_unknown_budget_admission_never_contacts_loopback_or_spends_attempt() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut config = cfg(0.125);
    config["llm"]["model_list"] = json!([{"model_name":"default","litellm_params":{
        "model":"openai/gpt-4.1","api_key":"test-only","api_base":format!("http://{}/v1", listener.local_addr().unwrap())}}]);
    config["llm"]["router_settings"] =
        json!({"num_retries":0,"timeout":1,"routing_strategy":"least-busy"});
    let scope = DocumentScope::new(&config);
    let runtime = crate::LlmRuntime::new(1).unwrap();
    let prompt = Prompts {
        system: "s".into(),
        user: "u".into(),
        image: None,
        cache_scope: String::new(),
    };
    let error = run_with_runtime(
        &prompt,
        &config,
        &HashMap::new(),
        &mut |_| {},
        Some(&runtime),
    )
    .unwrap_err();
    assert!(error.to_string().contains("verified tariff"));
    assert_eq!(scope.current.lock().unwrap().attempts, 0);
    assert_eq!(scope.usage().requests, 0);
    assert!(
        matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
    );
    drop(scope);
    let invalid = DocumentScope::new(&json!({"llm":{"max_cost_per_document_usd":"not numeric"}}));
    assert!(matches!(
        admit_document_attempt_for(Some(&entry())),
        Err(Error::Config(_))
    ));
    assert_eq!(invalid.current.lock().unwrap().attempts, 0);
}

#[test]
fn simultaneous_admitted_work_drains_then_shared_budget_refuses_more() {
    let scope = DocumentScope::new(&cfg(0.125));
    let shared = DocumentScope::shared().unwrap();
    let gate = Arc::new(Barrier::new(2));
    std::thread::scope(|threads| {
        for _ in 0..2 {
            let shared = shared.clone();
            let gate = gate.clone();
            threads.spawn(move || {
                let _scope = DocumentScope::enter(shared);
                let entry = entry();
                admit_document_attempt_for(Some(&entry)).unwrap();
                gate.wait();
                record_usage_class(
                    &mut ConversionUsage::default(),
                    &entry,
                    &paid(62_500, 0),
                    BillingClass::Standard,
                );
            });
        }
    });
    assert_eq!(scope.usage().requests, 2);
    assert_eq!(scope.usage().cost_usd, 0.25);
    assert!(admit_document_attempt_for(Some(&entry())).is_err());
    assert_eq!(scope.current.lock().unwrap().attempts, 2);
}

#[test]
fn batch_and_standard_paid_invalid_answers_keep_distinct_rates() {
    let entry = entry();
    let mut usage = ConversionUsage::default();
    let body = json!({"error":{"message":"invalid semantic answer"},"usage":{"prompt_tokens":1000,"completion_tokens":100}});
    record_usage_class(&mut usage, &entry, &body, BillingClass::Batch);
    assert!((usage.cost_usd - 0.0014).abs() < 1e-12);
    record_usage_class(&mut usage, &entry, &body, BillingClass::Standard);
    assert!((usage.cost_usd - 0.0042).abs() < 1e-12);
    assert_eq!(usage.requests, 2);
    assert_eq!(usage.by_model[&entry.id]["priced_requests"], 2);
    assert_eq!(usage.by_model[&entry.id]["cost_status"], "complete");
}

#[test]
fn zero_token_observation_is_priced_but_absent_work_has_no_synthetic_row() {
    let mut usage = ConversionUsage::default();
    merge(&mut usage, &ConversionUsage::default());
    assert!(usage.by_model.is_empty());
    record_usage_class(&mut usage, &entry(), &paid(0, 0), BillingClass::Standard);
    assert_eq!(usage.requests, 1);
    assert_eq!(usage.cost_usd, 0.0);
    assert_eq!(usage.by_model["openai/gpt-4.1"]["cost_status"], "complete");
}

#[test]
fn the_incomplete_cost_warning_names_each_model_and_its_reason() {
    let mut usage = response("m", 1000, 100, known());
    assert!(
        usage.unpriced_warning().is_none(),
        "a complete quote says nothing"
    );
    merge(
        &mut usage,
        &response(
            "openai/gpt-4.1-mini",
            10,
            5,
            Quote::Unknown(UnknownPrice::Model),
        ),
    );
    assert_eq!(
        usage.unpriced_warning().as_deref(),
        Some(
            "Cost is incomplete: 1 request to openai/gpt-4.1-mini has no reviewed price. cost_usd is the known priced subtotal; the complete cost is unknown."
        )
    );
    // Several unpriced requests to one model read as a population, not as "some".
    let mut twice = response(
        "openai/gpt-4.1-mini",
        10,
        5,
        Quote::Unknown(UnknownPrice::Model),
    );
    merge(
        &mut twice,
        &response(
            "openai/gpt-4.1-mini",
            10,
            5,
            Quote::Unknown(UnknownPrice::Model),
        ),
    );
    assert_eq!(
        twice.unpriced_warning().as_deref(),
        Some(
            "Cost is incomplete: 2 requests to openai/gpt-4.1-mini have no reviewed price. cost_usd is the known priced subtotal; the complete cost is unknown."
        )
    );
    // A subscription row keeps its own notice and never appears here.
    let subscription = crate::ConversionUsage {
        requests: 1,
        by_model: [(
            "chatgpt/gpt-5.5".to_owned(),
            json!({"requests":1,"priced_requests":0,"unpriced_requests":0,"cost_status":"unknown"}),
        )]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    assert!(subscription.unpriced_warning().is_none());
    assert!(!subscription.has_unpriced_non_subscription());
    // Provider-reported counts are missing, which is a different root cause.
    let uncounted: crate::ConversionUsage = serde_json::from_value(json!({
        "cost_usd": 0.0,
        "requests": 3,
        "input_tokens": 0,
        "output_tokens": 0,
        "by_model": {"local/vllm-x": {"requests":3,"priced_requests":0,"unpriced_requests":0,"incomplete_request_observations":3,"cost_status":"unknown"}}
    }))
    .expect("usage");
    assert_eq!(
        uncounted.unpriced_warning().as_deref(),
        Some(
            "Cost is incomplete: 3 requests to local/vllm-x reported no usage counts. cost_usd is the known priced subtotal; the complete cost is unknown."
        )
    );
    // More models than the warning names are counted instead of listed.
    let mut many = crate::ConversionUsage::default();
    for index in 0..5 {
        merge(
            &mut many,
            &response(
                &format!("local/model-{index}"),
                1,
                1,
                Quote::Unknown(UnknownPrice::Model),
            ),
        );
    }
    let warning = many.unpriced_warning().expect("a breakdown");
    assert!(warning.contains("local/model-0"), "{warning}");
    assert!(warning.contains("2 more models"), "{warning}");
}
