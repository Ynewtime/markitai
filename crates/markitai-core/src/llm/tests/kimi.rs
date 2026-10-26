//! Moonshot's open platform and Kimi Code: their regional endpoints, Kimi
//! Code's Messages requests, and what a key refused at one region names.
use super::*;

fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

fn resolved(params: Value, env: &[(&str, &str)]) -> Deployment {
    let cfg = config::normalize(
        &json!({"llm":{"model_list":[{"model_name":"default","litellm_params":params}]}}),
    )
    .unwrap();
    deployments(&cfg, &vars(env)).unwrap().remove(0)
}

#[test]
fn kimi_code_is_reached_at_its_messages_endpoint_with_its_own_variables() {
    let kimi = resolved(
        json!({"model":"kimi_code/k3"}),
        &[("KIMI_API_KEY", "fake-kimi-key")],
    );
    assert_eq!(kimi.protocol, Protocol::Anthropic);
    assert_eq!(kimi.endpoint, "https://api.kimi.com/coding/v1/messages");
    assert_eq!(kimi.key.as_deref(), Some("fake-kimi-key"));
    assert_eq!(kimi.key_variable.as_deref(), Some("KIMI_API_KEY"));
    // Kimi Code CLI's base variable moves it to the kimi.ai endpoint.
    let global = resolved(
        json!({"model":"kimi_code/k3","api_key":"fake-kimi-key"}),
        &[("KIMI_CODE_BASE_URL", "https://api.kimi.ai/coding/v1")],
    );
    assert_eq!(global.endpoint, "https://api.kimi.ai/coding/v1/messages");
    // A key from the China platform is used at its own endpoint.
    let china = resolved(
        json!({"model":"moonshot/kimi-k3","api_key":"fake-moonshot-key","api_base":"https://api.moonshot.cn/v1"}),
        &[],
    );
    assert_eq!(
        china.endpoint,
        "https://api.moonshot.cn/v1/chat/completions"
    );
    let china = resolved(
        json!({"model":"moonshot/kimi-k3"}),
        &[
            ("MOONSHOT_API_KEY", "fake-moonshot-key"),
            ("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1"),
        ],
    );
    assert_eq!(
        china.endpoint,
        "https://api.moonshot.cn/v1/chat/completions"
    );
}

#[test]
fn a_kimi_code_request_is_a_messages_request_with_room_to_think() {
    let answer = json!({"id":"msg_1","type":"message","role":"assistant","model":"k3",
        "content":[{"type":"thinking","thinking":"…"},{"type":"text","text":"cleaned"}],
        "stop_reason":"end_turn","usage":{"input_tokens":11,"output_tokens":7}});
    let server = Mock::new(vec![(200, answer)]);
    let cfg = config::normalize(
        &json!({"llm":{"model_list":[{"model_name":"default","litellm_params":{
            "model":"kimi_code/k3","api_key":"fake-kimi-key","api_base":server.base}}]},
        "cache":{"enabled":false},"prompts":{"dir":"/nonexistent/markitai-test-prompts"}}),
    )
    .unwrap();
    let (text, _) = run(&plain(), &cfg, &HashMap::new(), &mut |_| {}).unwrap();
    assert_eq!(text, "cleaned");
    let requests = server.finish();
    let (headers, body) = &requests[0];
    let headers = headers.to_ascii_lowercase();
    assert!(headers.starts_with("post /v1/messages "), "{headers}");
    assert!(headers.contains("x-api-key: fake-kimi-key"), "{headers}");
    assert!(headers.contains("anthropic-version: "), "{headers}");
    assert!(!headers.contains("authorization:"), "{headers}");
    // No changed client identity: markitai's own User-Agent.
    assert!(headers.contains("user-agent: markitai/"), "{headers}");
    assert_eq!(body["model"], "k3");
    assert_eq!(body["max_tokens"], 32_768);
    assert!(body.get("temperature").is_none() && body.get("thinking").is_none());
}

#[test]
fn a_key_refused_at_one_region_names_the_other_regions_and_the_sibling_service() {
    let hint = |params: Value, env: &[(&str, &str)]| {
        crate::llm::keys::refused_hint(&resolved(params, env))
    };
    let international = hint(
        json!({"model":"moonshot/kimi-k3","api_key":"fake-moonshot-key"}),
        &[],
    );
    assert_eq!(
        international,
        "the api_key configured for this model was refused; check that key. api.moonshot.ai accepts keys from platform.kimi.ai only: a key from platform.kimi.com needs api_base https://api.moonshot.cn/v1, and a Kimi Code membership key is used as kimi_code/<model>"
    );
    let china = hint(
        json!({"model":"moonshot/kimi-k3"}),
        &[
            ("MOONSHOT_API_KEY", "fake-moonshot-key"),
            ("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1"),
        ],
    );
    assert!(
        china.starts_with("the API key in MOONSHOT_API_KEY was refused; set a valid one with: ")
            && china.ends_with(". api.moonshot.cn accepts keys from platform.kimi.com only: a key from platform.kimi.ai needs api_base https://api.moonshot.ai/v1, and a Kimi Code membership key is used as kimi_code/<model>"),
        "{china}"
    );
    let kimi = hint(
        json!({"model":"kimi_code/k3","api_key":"fake-kimi-key"}),
        &[],
    );
    assert!(
        kimi.ends_with(". api.kimi.com accepts keys from kimi.com only: a key from kimi.ai needs api_base https://api.kimi.ai/coding/v1, and a Kimi platform key is used as moonshot/<model>"),
        "{kimi}"
    );
    // Another address (a gateway) and a keyless request say nothing of regions.
    let gateway = hint(
        json!({"model":"moonshot/kimi-k3","api_key":"fake-moonshot-key","api_base":"https://gateway.example/v1"}),
        &[],
    );
    assert!(!gateway.contains("platform.kimi"), "{gateway}");
    let mut keyless = resolved(
        json!({"model":"moonshot/kimi-k3","api_key":"fake-moonshot-key"}),
        &[],
    );
    keyless.key = None;
    assert!(!crate::llm::keys::refused_hint(&keyless).contains("platform.kimi"));
}

#[test]
fn a_key_refused_at_a_minimax_zai_or_model_studio_region_names_the_others() {
    let hint = |params: Value, env: &[(&str, &str)]| {
        crate::llm::keys::refused_hint(&resolved(params, env))
    };
    let china = hint(
        json!({"model":"minimax/MiniMax-M3","api_key":"fake-minimax-key","api_base":"https://api.minimax.cn/v1"}),
        &[],
    );
    assert!(
        china.ends_with(". api.minimax.cn accepts keys from platform.minimax.cn only: a key from platform.minimax.io needs api_base https://api.minimax.io/v1"),
        "{china}"
    );
    let beijing = hint(
        json!({"model":"dashscope/qwen-plus"}),
        &[("DASHSCOPE_API_KEY", "fake-dashscope-key")],
    );
    assert!(
        beijing.ends_with(". dashscope.aliyuncs.com accepts keys from Model Studio's Beijing region only: \
            a key from Model Studio's Singapore region needs api_base https://dashscope-intl.aliyuncs.com/compatible-mode/v1, \
            a key from Model Studio's US (Virginia) region needs api_base https://dashscope-us.aliyuncs.com/compatible-mode/v1, \
            a key from Model Studio's Hong Kong region needs api_base https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1, \
            and a Coding Plan key (sk-sp-…) works only at the Coding Plan's endpoints, in coding tools"),
        "{beijing}"
    );
    // A request URL names its region as its base does.
    assert!(
        crate::llm::providers::region_hint(
            "dashscope",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        )
        .unwrap()
        .starts_with("dashscope-intl.aliyuncs.com accepts keys from Model Studio's Singapore region only: a key from Model Studio's Beijing region needs api_base https://dashscope.aliyuncs.com/compatible-mode/v1, ")
    );
    // Providers whose documentation names no per-region keys say nothing of regions.
    let zhipu = hint(
        json!({"model":"zai/glm-5","api_key":"fake-zai-key","api_base":"https://open.bigmodel.cn/api/paas/v4"}),
        &[],
    );
    assert!(
        zhipu.ends_with(". open.bigmodel.cn accepts keys from bigmodel.cn only: a key from z.ai needs api_base https://api.z.ai/api/paas/v4"),
        "{zhipu}"
    );
    // A Coding Plan endpoint is no region: it takes either kind of key.
    assert!(
        crate::llm::providers::region_hint(
            "zai",
            "https://api.z.ai/api/coding/paas/v4/chat/completions"
        )
        .unwrap()
        .starts_with("api.z.ai accepts keys from z.ai only: ")
    );
    for model in ["qwencloud/qwen-plus", "openai/gpt-5"] {
        let other = hint(json!({"model":model,"api_key":"fake-key"}), &[]);
        assert!(!other.contains("accepts keys"), "{other}");
    }
}
