//! Which variable a provider's key is read from, and what a request without
//! a usable key does.
use super::*;

fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

fn single(params: Value) -> Value {
    config::normalize(
        &json!({"llm":{"model_list":[{"model_name":"default","litellm_params":params}]}}),
    )
    .unwrap()
}

fn resolved(model: &str, env: &[(&str, &str)]) -> Deployment {
    deployments(&single(json!({"model":model})), &vars(env))
        .unwrap()
        .remove(0)
}

#[test]
fn key_aliases_are_read_as_litellm_reads_them() {
    let gemini = resolved("gemini/flash-test", &[("GOOGLE_API_KEY", "google-key")]);
    assert_eq!(gemini.key.as_deref(), Some("google-key"));
    assert_eq!(gemini.key_variable.as_deref(), Some("GOOGLE_API_KEY"));
    // Google's SDK reads GOOGLE_API_KEY first when both are set.
    let both = resolved(
        "gemini/flash-test",
        &[
            ("GEMINI_API_KEY", "gemini-key"),
            ("GOOGLE_API_KEY", "google-key"),
        ],
    );
    assert_eq!(both.key.as_deref(), Some("google-key"));
    let gemini = resolved(
        "gemini/flash-test",
        &[("GEMINI_API_KEY", "gemini-key"), ("GOOGLE_API_KEY", " ")],
    );
    assert_eq!(gemini.key.as_deref(), Some("gemini-key"));
    let router = resolved("openrouter/a/b", &[("OR_API_KEY", "or-key")]);
    assert_eq!(router.key.as_deref(), Some("or-key"));
    let router = resolved(
        "openrouter/a/b",
        &[
            ("OPENROUTER_API_KEY", "openrouter-key"),
            ("OR_API_KEY", "or-key"),
        ],
    );
    assert_eq!(router.key.as_deref(), Some("openrouter-key"));
    let claude = resolved(
        "anthropic/claude-test",
        &[("ANTHROPIC_AUTH_TOKEN", "gateway-token")],
    );
    assert_eq!(claude.key.as_deref(), Some("gateway-token"));
    assert_eq!(claude.key_variable.as_deref(), Some("ANTHROPIC_AUTH_TOKEN"));
    // A key written in the configuration names no variable; a reference does.
    let written = deployments(
        &single(json!({"model":"openai/x","api_key":"sk-written"})),
        &HashMap::new(),
    )
    .unwrap()
    .remove(0);
    assert_eq!(written.key_variable, None);
    let referenced = deployments(
        &single(json!({"model":"openai/x","api_key":"env:TEAM_KEY"})),
        &vars(&[("TEAM_KEY", "team")]),
    )
    .unwrap()
    .remove(0);
    assert_eq!(referenced.key_variable.as_deref(), Some("TEAM_KEY"));
}

#[test]
fn key_aliases_join_the_automatic_pool() {
    for (variable, provider) in [
        ("GOOGLE_API_KEY", "gemini"),
        ("ANTHROPIC_AUTH_TOKEN", "anthropic"),
        ("OR_API_KEY", "openrouter"),
    ] {
        let env = vars(&[(variable, "alias-key")]);
        let joined: Vec<_> = keyed_defaults(&env).map(|(name, _)| name).collect();
        assert_eq!(joined, [provider], "{variable}");
    }
}

#[test]
fn an_anthropic_auth_token_is_sent_as_a_bearer_token() {
    let server = Mock::new(vec![
        (
            200,
            json!({"model":"claude-test","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}),
        ),
        (
            200,
            json!({"model":"claude-test","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}),
        ),
    ]);
    let mut cfg = cfg("anthropic/claude-test", &server.base);
    cfg["llm"]["model_list"][0]["litellm_params"]
        .as_object_mut()
        .unwrap()
        .remove("api_key");
    let token = vars(&[("ANTHROPIC_AUTH_TOKEN", "gateway-token")]);
    run(&plain(), &cfg, &token, &mut |_| {}).unwrap();
    let key = vars(&[
        ("ANTHROPIC_API_KEY", "api-key"),
        ("ANTHROPIC_AUTH_TOKEN", "gateway-token"),
    ]);
    run(&plain(), &cfg, &key, &mut |_| {}).unwrap();
    let requests = server.finish();
    let headers: Vec<String> = requests
        .iter()
        .map(|(headers, _)| headers.to_ascii_lowercase())
        .collect();
    assert!(
        headers[0].contains("authorization: bearer gateway-token"),
        "{}",
        headers[0]
    );
    assert!(!headers[0].contains("x-api-key"), "{}", headers[0]);
    assert!(headers[1].contains("x-api-key: api-key"), "{}", headers[1]);
    assert!(!headers[1].contains("authorization"), "{}", headers[1]);
}

fn pool_of(models: Value) -> Value {
    config::normalize(&json!({"llm":{"model_list":models},"cache":{"enabled":false},
        "prompts":{"dir":"/nonexistent/markitai-test-prompts"},"router_settings":{"num_retries":0}}))
    .unwrap()
}

#[test]
fn a_model_without_its_key_is_left_out_before_anything_is_sent() {
    let server = Mock::new(vec![(200, success("answer"))]);
    let cfg = pool_of(json!([
        {"model_name":"default","litellm_params":{"model":"deepseek/keys-left-out"}},
        {"model_name":"default","litellm_params":{"model":"openai/keys-kept","api_key":"fake-test-key","api_base":server.base}}
    ]));
    let (missing, usable) = key_check(&cfg, &HashMap::new());
    assert_eq!(
        missing,
        [crate::llm::keys::MissingKey {
            model: "deepseek/keys-left-out".into(),
            variable: Some("DEEPSEEK_API_KEY".into()),
        }]
    );
    assert!(usable);
    let scope = DocumentScope::new(&cfg);
    // Every request goes to the model that has its key.
    let (text, _) = run(&plain(), &cfg, &HashMap::new(), &mut |_| {}).unwrap();
    assert_eq!(text, "answer");
    let warnings = scope.take_warnings();
    let left_out: Vec<_> = warnings
        .iter()
        .filter(|warning| warning.contains("left out of the model pool"))
        .collect();
    assert_eq!(left_out.len(), 1, "{warnings:?}");
    assert!(
        left_out[0].starts_with(
            "deepseek/keys-left-out is left out of the model pool: DEEPSEEK_API_KEY is not set; set it with: "
        ),
        "{}",
        left_out[0]
    );
    assert_eq!(server.finish().len(), 1);
}

#[test]
fn a_pool_without_any_key_fails_naming_the_variables_and_sends_nothing() {
    let cfg = pool_of(json!([
        {"model_name":"default","litellm_params":{"model":"deepseek/keys-none"}}
    ]));
    let error = run(&plain(), &cfg, &HashMap::new(), &mut |_| {}).unwrap_err();
    let Error::MissingApiKey(message) = &error else {
        panic!("{error:?}")
    };
    assert_eq!(error.code(), "no_model_configured");
    assert!(
        message.starts_with(
            "DEEPSEEK_API_KEY is not set, and deepseek/keys-none needs it; set it with: "
        ),
        "{message}"
    );
    assert!(message.contains("DEEPSEEK_API_KEY=your_key"), "{message}");
    // Two variables: each is named with its models.
    let cfg = pool_of(json!([
        {"model_name":"default","litellm_params":{"model":"deepseek/a"}},
        {"model_name":"default","litellm_params":{"model":"gemini/b"}},
        {"model_name":"default","litellm_params":{"model":"openai/c","api_key":"env:TEAM_KEY_ABSENT"}}
    ]));
    let message = run(&plain(), &cfg, &HashMap::new(), &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(
        message.starts_with(
            "No LLM model can be used: DEEPSEEK_API_KEY is not set (deepseek/a); GEMINI_API_KEY is not set (gemini/b); TEAM_KEY_ABSENT is not set (openai/c). Set one of them, for example "
        ),
        "{message}"
    );
    // An empty or blank variable counts as unset.
    let blank = vars(&[("DEEPSEEK_API_KEY", " ")]);
    let (missing, usable) = key_check(
        &pool_of(json!([{"model_name":"default","litellm_params":{"model":"deepseek/a"}}])),
        &blank,
    );
    assert_eq!((missing.len(), usable), (1, false));
}

#[test]
fn a_model_with_its_own_address_is_tried_and_local_servers_need_no_key() {
    for (params, env) in [
        // A gateway or a local server may check no key.
        (
            json!({"model":"openai/x","api_base":"http://127.0.0.1:9/v1"}),
            vec![],
        ),
        (
            json!({"model":"deepseek/x"}),
            vec![("DEEPSEEK_API_BASE", "http://127.0.0.1:9/v1")],
        ),
        (json!({"model":"ollama/x"}), vec![]),
        (json!({"model":"lm_studio/x"}), vec![]),
    ] {
        let (missing, usable) = key_check(&single(params.clone()), &vars(&env));
        assert!(missing.is_empty() && usable, "{params}");
    }
    // Azure has no address of its own to be exempted by.
    let (missing, _) = key_check(
        &single(
            json!({"model":"azure/d","api_base":"https://example.openai.azure.com","api_version":"2024-10-21"}),
        ),
        &HashMap::new(),
    );
    assert_eq!(missing[0].variable.as_deref(), Some("AZURE_API_KEY"));
    // A connection that reads no environment names no variable.
    let (missing, _) = key_check(
        &single(json!({"model":"openai/x","use_environment_credentials":false})),
        &vars(&[("OPENAI_API_KEY", "server-key")]),
    );
    assert_eq!(missing[0].variable, None);
    assert!(
        crate::llm::keys::no_usable_model_message(&missing).contains("set litellm_params.api_key"),
        "{missing:?}"
    );
}

#[test]
fn a_refused_key_names_the_variable_that_held_it() {
    let refusal = json!({"error":{"message":"Authentication Fails","type":"authentication_error"}});
    let gemini = json!([{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT","details":[{"reason":"API_KEY_INVALID"}]}}]);
    let server = Mock::new(vec![(401, refusal.clone()), (400, gemini), (401, refusal)]);
    let one = |model: &str, key: Option<&str>| {
        let mut params = json!({"model":model,"api_base":server.base});
        if let Some(key) = key {
            params["api_key"] = json!(key);
        }
        pool_of(json!([{"model_name":"default","litellm_params":params}]))
    };
    let env = vars(&[
        ("DEEPSEEK_API_KEY", "fake-test-key"),
        ("GOOGLE_API_KEY", "fake-test-key"),
    ]);
    let deepseek = run(&plain(), &one("deepseek/x", None), &env, &mut |_| {})
        .unwrap_err()
        .to_string();
    let set = crate::llm::keys::set_command("DEEPSEEK_API_KEY");
    assert!(
        deepseek.starts_with(&format!(
            "LLM returned HTTP 401 (authentication_error): the API key in DEEPSEEK_API_KEY was refused; set a valid one with: {set}. Provider message: Authentication Fails"
        )),
        "{deepseek}"
    );
    let gemini = run(&plain(), &one("gemini/x", None), &env, &mut |_| {})
        .unwrap_err()
        .to_string();
    assert!(
        gemini.contains("the API key in GOOGLE_API_KEY was refused"),
        "{gemini}"
    );
    // A keyless request to a server that wants a key.
    let keyless = run(
        &plain(),
        &one("openai/x", None),
        &HashMap::new(),
        &mut |_| {},
    )
    .unwrap_err()
    .to_string();
    assert!(
        keyless.contains("the server asks for an API key and none was sent; set one with: ")
            && keyless.contains("OPENAI_API_KEY"),
        "{keyless}"
    );
    let requests = server.finish();
    assert_eq!(requests.len(), 3);
    assert!(!requests[2].0.to_ascii_lowercase().contains("authorization"));
}
