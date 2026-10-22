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
