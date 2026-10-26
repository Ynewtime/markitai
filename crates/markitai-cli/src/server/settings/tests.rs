use super::*;
use std::fs;
fn setup(value: Value) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    fs::write(&path, value.to_string()).unwrap();
    let cfg = markitai_core::config::normalize(&value).unwrap();
    let store = Store::new(
        cfg,
        SettingsSource {
            path,
            origin: "explicit".into(),
            overrides: None,
        },
    )
    .unwrap_or_else(|e| panic!("{}", e.detail));
    (dir, store)
}
fn fixture() -> Value {
    json!({"unknown":{"nested":[1,"keep"]},"llm":{"enabled":false,"model_list":[{"model_name":"same","litellm_params":{"model":"openai/old","api_key":"fixture-secret","api_base":"https://user:pass@example.test/custom?token=hidden"},"custom":{"keep":true}}]}})
}
fn ok(result: ApiResult<Value>) -> Value {
    result.unwrap_or_else(|e| panic!("{}", e.detail))
}
fn current(store: &Store) -> String {
    store.view()["revision"].as_str().unwrap().to_owned()
}
fn saved(store: &Store) -> Value {
    serde_json::from_slice(&fs::read(&store.source.path).unwrap()).unwrap()
}
#[test]
fn python_canonical_revision_and_legacy_unicode_identity() {
    let models = vec![
        json!({"model_name":"组😀","litellm_params":{"model":"openai/a","api_key":"env:MOCK","weight":0},"extra":{"z":1,"a":2}}),
    ];
    assert_eq!(
        identity::revision(&models, &[]),
        "ce3361f64bb48a272bcc171d751d613483a30d9d99bd7dc96b5ad1e4bf0d0291"
    );
    assert_eq!(identity::id(&models[0], 0), "legacy-697cae68cedcc432a143");
    let mut changed = models[0].clone();
    changed["litellm_params"]["api_key"] = json!("other");
    assert_eq!(identity::id(&changed, 0), identity::id(&models[0], 0));
    assert_ne!(
        identity::revision(&[changed], &[]),
        identity::revision(&models, &[])
    );
}
#[test]
fn identity_backfill_preserves_unknown_and_omission_null_are_distinct() {
    let (_dir, store) = setup(fixture());
    let view = store.view();
    let id = view["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let revision = current(&store);
    let first = ok(store.mutate(Mutation::Update {
        key: id,
        legacy: false, share: None,
        body: json!({"expected_revision":revision,"model":"openai/new","api_key":"fixture-secret","api_base":null}),
    }));
    let raw = saved(&store);
    assert_eq!(raw["unknown"], fixture()["unknown"]);
    assert_eq!(raw["llm"]["model_list"][0]["custom"], json!({"keep":true}));
    assert_eq!(
        raw["llm"]["model_list"][0]["litellm_params"]["api_key"],
        "fixture-secret"
    );
    assert!(
        raw["llm"]["model_list"][0]["litellm_params"]
            .get("api_base")
            .is_none()
    );
    let id = first["deployments"][0]["deployment_id"].as_str().unwrap();
    assert!(uuid::Uuid::parse_str(id).is_ok());
    assert_eq!(store.snapshot()["llm"]["enabled"], false);
    assert_eq!(
        store.snapshot()["llm"]["model_list"][0]["litellm_params"]["model"],
        "openai/new"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&store.source.path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}
#[test]
fn batch_cas_and_invalid_second_item_never_write_or_replace_runtime() {
    let (_dir, store) = setup(fixture());
    let before = fs::read(&store.source.path).unwrap();
    let snapshot = store.snapshot();
    let revision = current(&store);
    let error=store.mutate(Mutation::Batch(json!({"expected_revision":revision,"deployments":[{"model_name":"x","model":"openai/x"},{"model_name":"bad","model":"openai/bad","weight":-1}]}))).err().unwrap();
    assert_eq!(error.status.as_u16(), 422);
    assert_eq!(fs::read(&store.source.path).unwrap(), before);
    assert!(Arc::ptr_eq(&snapshot, &store.snapshot()));
    ok(store.mutate(Mutation::Add(
        json!({"model_name":"same","model":"openai/second","expected_revision":revision}),
    )));
    let after = fs::read(&store.source.path).unwrap();
    assert_eq!(
        store
            .mutate(Mutation::Add(
                json!({"model_name":"third","model":"openai/third","expected_revision":revision})
            ))
            .err()
            .unwrap()
            .status
            .as_u16(),
        409
    );
    assert_eq!(fs::read(&store.source.path).unwrap(), after);
    assert_eq!(
        store
            .mutate(Mutation::Update {
                key: "same".into(),
                body: json!({"model":"openai/not-picked"}),
                legacy: true,
                share: None,
            })
            .err()
            .unwrap()
            .status
            .as_u16(),
        409
    );
}
#[test]
fn connection_migration_updates_all_matching_rows_then_delete_keeps_other_connection() {
    let mut raw = fixture();
    let first = raw["llm"]["model_list"][0].clone();
    let mut second = first.clone();
    second["model_name"] = json!("second");
    let mut other = first.clone();
    other["model_name"] = json!("other");
    other["litellm_params"]["api_key"] = json!("different");
    raw["llm"]["model_list"] = json!([first, second, other]);
    let (_dir, store) = setup(raw);
    let id = format!(
        "legacy:{}",
        store.view()["deployments"][0]["deployment_id"]
            .as_str()
            .unwrap()
    );
    ok(store.mutate(Mutation::Provider {
        key: id,
        revision: current(&store),
        body: Some(
            json!({"expected_revision":current(&store),"api_key":"fixture-next-key","api_base":null}),
        ),
    }));
    let raw = saved(&store);
    let provider = raw["llm"]["providers"][0]["id"]
        .as_str()
        .unwrap()
        .to_owned();
    for i in 0..2 {
        assert_eq!(
            raw["llm"]["model_list"][i]["litellm_params"]["api_key"],
            "fixture-next-key"
        );
        assert_eq!(
            raw["llm"]["model_list"][i]["model_info"]["provider_id"],
            provider
        );
    }
    let credentials = ok(store.credentials(&provider));
    assert_eq!(credentials["api_key_masked"], "fi\u{2026}-key");
    assert_eq!(credentials["api_key"], credentials["api_key_masked"]);
    assert!(!credentials.to_string().contains("fixture-next-key"));
    assert!(credentials["api_base"].is_null());
    ok(store.mutate(Mutation::Provider {
        key: provider,
        revision: current(&store),
        body: None,
    }));
    assert_eq!(
        saved(&store)["llm"]["model_list"].as_array().unwrap().len(),
        1
    );
    assert_eq!(saved(&store)["llm"]["model_list"][0]["model_name"], "other");
}
#[test]
fn deleting_last_model_preserves_connection_and_reuses_without_secret_echo() {
    let (_dir, store) = setup(fixture());
    ok(store.mutate(Mutation::Delete {
        key: "same".into(),
        revision: None,
        legacy: true,
    }));
    let raw = saved(&store);
    assert!(raw["llm"]["model_list"].as_array().unwrap().is_empty());
    let id = raw["llm"]["providers"][0]["id"].as_str().unwrap();
    let response=ok(store.mutate(Mutation::Batch(json!({"expected_revision":current(&store),"deployments":[{"model_name":"restored","model":"openai/new","credential_provider_id":id}]}))));
    assert!(!response.to_string().contains("fixture-secret"));
    assert_eq!(
        saved(&store)["llm"]["model_list"][0]["litellm_params"]["api_key"],
        "fixture-secret"
    );
    let cards = store.providers().to_string();
    for secret in ["fixture-secret", "hidden", "user:pass", "/custom"] {
        assert!(!cards.contains(secret), "{secret}");
    }
}
#[test]
fn probe_and_discovery_resolve_detached_private_values_without_writes() {
    let (_dir, store) = setup(fixture());
    let before = fs::read(&store.source.path).unwrap();
    let id = store.view()["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let request = ok(store.resolve_probe(&json!({"deployment_id":id})));
    assert_eq!(request["api_key"], "fixture-secret");
    assert_eq!(
        store
            .resolve_probe(&json!({"deployment_id":id,"model":"openai/x"}))
            .err()
            .unwrap()
            .status
            .as_u16(),
        422
    );
    assert_eq!(
        ok(store.resolve_probe(&json!({"model":"openai/old","api_key":"override"})))["api_key"],
        "override"
    );
    assert!(store.resolve_discovery(&json!({"provider":"openai","provider_id":format!("legacy:{id}"),"api_base":"http://127.0.0.1/v1","refresh":true})).is_err());
    let discovery = ok(store.resolve_discovery(
        &json!({"provider":"openai","provider_id":format!("legacy:{id}"),"refresh":true}),
    ));
    assert_eq!(discovery["api_key"], "fixture-secret");
    assert_eq!(
        discovery["api_base"],
        fixture()["llm"]["model_list"][0]["litellm_params"]["api_base"]
    );
    assert_eq!(fs::read(&store.source.path).unwrap(), before);
}
#[test]
fn external_edits_are_preserved_and_stale_revision_is_rejected() {
    let (_dir, store) = setup(fixture());
    let revision = current(&store);
    let mut raw = saved(&store);
    raw["llm"]["model_list"][0]["litellm_params"]["model"] = json!("openai/external");
    fs::write(&store.source.path, raw.to_string()).unwrap();
    let before = fs::read(&store.source.path).unwrap();
    assert_eq!(
        store
            .mutate(Mutation::Add(
                json!({"expected_revision":revision,"model_name":"x","model":"openai/x"})
            ))
            .err()
            .unwrap()
            .status
            .as_u16(),
        409
    );
    assert_eq!(fs::read(&store.source.path).unwrap(), before);
    raw["external"] = json!(42);
    fs::write(&store.source.path, raw.to_string()).unwrap();
    ok(store.mutate(Mutation::Add(json!({"model_name":"x","model":"openai/x"}))));
    assert_eq!(saved(&store)["external"], 42);
    assert_eq!(
        store.snapshot()["llm"]["model_list"][0]["litellm_params"]["model"],
        "openai/external"
    );
}
#[test]
fn session_overrides_and_invalid_storage_cannot_replace_saved_state() {
    let (dir, store) = setup(fixture());
    let mut cfg = store.snapshot().as_ref().clone();
    cfg["llm"]["model_list"] = json!([]);
    let original = fs::read(&store.source.path).unwrap();
    let blocked = Store::new(
        cfg,
        SettingsSource {
            path: store.source.path.clone(),
            origin: "explicit".into(),
            overrides: Some(json!({"llm":{"model_list":[]}})),
        },
    )
    .unwrap_or_else(|e| panic!("{}", e.detail));
    assert!(
        blocked.snapshot()["llm"]["model_list"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        blocked
            .mutate(Mutation::Add(json!({"model_name":"x","model":"openai/x"})))
            .err()
            .unwrap()
            .status
            .as_u16(),
        409
    );
    assert_eq!(fs::read(&store.source.path).unwrap(), original);
    fs::remove_file(&store.source.path).unwrap();
    fs::create_dir(&store.source.path).unwrap();
    assert_eq!(
        store
            .mutate(Mutation::Add(json!({"model_name":"x","model":"openai/x"})))
            .err()
            .unwrap()
            .status
            .as_u16(),
        500
    );
    assert_eq!(
        store.snapshot()["llm"]["model_list"][0]["litellm_params"]["model"],
        "openai/old"
    );
    drop(dir);
}
#[cfg(unix)]
#[test]
fn symlinked_config_saves_through_the_link_and_fifo_config_fails() {
    use std::os::unix::fs::symlink;
    let (dir, store) = setup(fixture());
    // A dotfile-managed configuration: the link stays, its target changes.
    let real = dir.path().join("real.json");
    fs::rename(&store.source.path, &real).unwrap();
    symlink(&real, &store.source.path).unwrap();
    assert_eq!(store.config_path().unwrap(), store.source.path);
    ok(store.mutate(Mutation::Delete {
        key: "same".into(),
        revision: None,
        legacy: true,
    }));
    assert!(
        fs::symlink_metadata(&store.source.path)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    let raw: Value = serde_json::from_slice(&fs::read(&real).unwrap()).unwrap();
    assert!(raw["llm"]["model_list"].as_array().unwrap().is_empty());
    assert_eq!(raw["unknown"], fixture()["unknown"]);
    fs::remove_file(&store.source.path).unwrap();
    let path = std::ffi::CString::new(store.source.path.as_os_str().as_encoded_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    assert!(store.config_path().is_err());
}
#[test]
fn detected_models_remain_session_only_after_saving_one_deployment() {
    let (_dir, store) = setup(fixture());
    {
        let mut data = store.inner.lock().unwrap();
        data.detected = vec![
            json!({"model_name":"default","litellm_params":{"model":"openai/detected-one"}}),
            json!({"model_name":"default","litellm_params":{"model":"anthropic/detected-two"}}),
        ];
    }
    let view=ok(store.mutate(Mutation::Batch(json!({"expected_revision":current(&store),"deployments":[{"model_name":"default","model":"openai/detected-one"}]}))));
    assert_eq!(view["detected"].as_array().unwrap().len(), 2);
    assert_eq!(
        saved(&store)["llm"]["model_list"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        store.snapshot()["llm"]["model_list"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn postpublication_sync_failure_is_distinct_from_failure_before_rename() {
    let (_dir, store) = setup(fixture());
    let old = fs::read(&store.source.path).unwrap();
    let mut next = fixture();
    next["unknown"]["committed"] = json!(true);
    let durable = save_with_sync(&store.source.path, &next, Some(&old), |_| {
        Err(std::io::Error::other("authored sync failure"))
    })
    .unwrap_or_else(|e| panic!("{}", e.detail));
    assert!(!durable);
    assert_eq!(saved(&store), next);
}

#[test]
fn clearing_provider_only_deployment_detaches_without_changing_siblings() {
    let raw = json!({"llm":{"model_list":[{"model_name":"one","litellm_params":{"model":"openai/one"},"model_info":{"id":"one-id","provider_id":"connection"}},{"model_name":"two","litellm_params":{"model":"openai/two"},"model_info":{"id":"two-id","provider_id":"connection"}}],"providers":[{"id":"connection","provider":"openai","api_key":"fixture-key","api_base":"http://127.0.0.1:8123/v1"}]}});
    let (_dir, store) = setup(raw.clone());
    assert_eq!(
        ok(store.resolve_discovery(&json!({"provider":"openai","deployment_id":"one-id"})))["api_key"],
        "fixture-key"
    );
    ok(store.mutate(Mutation::Update {
        key: "one-id".into(),
        legacy: false,
        share: None,
        body: json!({"expected_revision":current(&store),"api_key":null}),
    }));
    let saved = saved(&store);
    let entry = &saved["llm"]["model_list"][0];
    assert!(entry["model_info"].get("provider_id").is_none());
    // Keyless at its address: the placeholder, never the environment's key.
    assert_eq!(entry["litellm_params"]["api_key"], model::NO_KEY);
    assert_eq!(
        entry["litellm_params"]["use_environment_credentials"],
        false
    );
    assert_eq!(
        entry["litellm_params"]["api_base"],
        "http://127.0.0.1:8123/v1"
    );
    assert_eq!(saved["llm"]["providers"], raw["llm"]["providers"]);
    assert_eq!(saved["llm"]["model_list"][1], raw["llm"]["model_list"][1]);
    assert_eq!(
        ok(store.resolve_probe(&json!({"deployment_id":"one-id"})))["api_key"],
        model::NO_KEY
    );
    assert_eq!(
        ok(store.resolve_probe(&json!({"deployment_id":"two-id"})))["api_key"],
        "fixture-key"
    );
}

#[test]
fn every_routed_provider_is_offered_with_its_group_mark_endpoint_key_and_discovery() {
    let (_dir, store) = setup(json!({"llm":{"enabled":false,"model_list":[]}}));
    let cards = store.providers()["providers"].as_array().unwrap().clone();
    let card = |name: &str| {
        cards
            .iter()
            .find(|card| card["provider"] == name && card["kind"] != "environment")
            .unwrap_or_else(|| panic!("no {name} card"))
            .clone()
    };
    // One card per catalog provider, from the environment or built in.
    let catalog = markitai_core::provider_management::catalog();
    assert_eq!(cards.len(), catalog.len());
    for known in &catalog {
        let offered = cards
            .iter()
            .find(|card| card["provider"] == known.provider)
            .unwrap();
        assert_eq!(offered["group"], known.group, "{}", known.provider);
        assert_eq!(offered["icon"], json!(known.icon), "{}", known.provider);
        assert_eq!(
            offered["aliases"],
            json!(known.aliases),
            "{}",
            known.provider
        );
    }
    // Every mark the catalog names is one the workbench vendors.
    let marks = include_str!("../web/src/components/brand-marks.ts");
    for known in &catalog {
        if let Some(icon) = known.icon {
            assert!(marks.contains(&format!("\n  {icon}: {{ viewBox")), "{icon}");
        }
    }
    // The reference's providers and those that are not OpenAI-compatible
    // HTTP APIs are "common"; the added OpenAI-compatible ones "compatible".
    let common: Vec<_> = cards
        .iter()
        .filter(|card| card["kind"] == "common")
        .map(|card| card["provider"].as_str().unwrap())
        .collect();
    assert_eq!(
        common,
        [
            "openai",
            "anthropic",
            "gemini",
            "deepseek",
            "openrouter",
            "azure",
            "ollama",
            "chatgpt",
            "claude-agent",
            "copilot",
            "kimi_code",
            "bedrock",
            "vertex_ai",
            "custom"
        ]
    );
    // A provider whose keys belong to a region lists its regions, the default first.
    assert_eq!(
        card("minimax")["regions"],
        json!(["https://api.minimax.io/v1", "https://api.minimax.cn/v1"])
    );
    assert_eq!(
        card("dashscope")["regions"][0],
        card("dashscope")["default_base"]
    );
    assert!(card("groq").get("regions").is_none());
    // Kimi Code takes a key of its own and lists its models.
    let kimi = card("kimi_code");
    assert_eq!(
        (
            kimi["label"].clone(),
            kimi["group"].clone(),
            kimi["key_variable"].clone(),
            kimi["default_base"].clone(),
            kimi["supports_discovery"].clone()
        ),
        (
            json!("Kimi Code"),
            json!("makers"),
            json!("KIMI_API_KEY"),
            json!("https://api.kimi.com/coding/v1"),
            json!(true)
        )
    );
    assert_eq!(card("mistral")["kind"], "compatible");
    assert_eq!(card("mistral")["group"], "common");
    assert_eq!(card("azure")["label"], "Azure OpenAI");
    assert_eq!(card("custom")["label"], "OpenAI-compatible endpoint");
    assert_eq!(card("databricks")["icon"], Value::Null);
    assert!(
        card("ollama")["aliases"]
            .as_array()
            .unwrap()
            .contains(&json!("ollama_chat"))
    );
    // A subscription runtime signs in by itself and lists its own models;
    // the clouds read their own credentials and take model IDs by hand.
    let copilot = card("copilot");
    assert_eq!(
        (
            copilot["status"].clone(),
            copilot["key_optional"].clone(),
            copilot["supports_discovery"].clone(),
            copilot["key_variable"].clone()
        ),
        (json!("unknown"), json!(true), json!(true), Value::Null)
    );
    let bedrock = card("bedrock");
    assert_eq!(
        (
            bedrock["supports_discovery"].clone(),
            bedrock["key_optional"].clone(),
            bedrock["default_base"].clone()
        ),
        (json!(false), json!(true), Value::Null)
    );
    let groq = card("groq");
    assert_eq!(groq["label"], "Groq");
    assert_eq!(groq["default_base"], "https://api.groq.com/openai/v1");
    assert_eq!(groq["key_variable"], "GROQ_API_KEY");
    assert_eq!(groq["status"], "needs_credentials");
    assert_eq!(
        (
            groq["supports_discovery"].clone(),
            groq["key_optional"].clone()
        ),
        (json!(true), json!(false))
    );
    for name in ["perplexity", "zai", "fireworks_ai"] {
        assert_eq!(card(name)["supports_discovery"], true, "{name}");
    }
    assert_eq!(
        card("zai")["regions"],
        json!([
            "https://api.z.ai/api/paas/v4",
            "https://open.bigmodel.cn/api/paas/v4"
        ])
    );
    let studio = card("lm_studio");
    assert_eq!(
        (studio["status"].clone(), studio["key_optional"].clone()),
        (json!("unknown"), json!(true))
    );
    assert_eq!(studio["default_base"], "http://localhost:1234/v1");
    let vllm = card("hosted_vllm");
    assert_eq!(
        (vllm["status"].clone(), vllm["default_base"].clone()),
        (json!("needs_credentials"), Value::Null)
    );
    assert_eq!(card("custom")["key_variable"], Value::Null);
    assert_eq!(card("openai")["key_variable"], "OPENAI_API_KEY");
}

#[test]
fn a_subscription_with_models_stays_offered_but_a_saved_provider_does_not() {
    let (_dir, store) = setup(json!({"llm":{"enabled":false,"model_list":[
        {"model_name":"default","litellm_params":{"model":"copilot/gpt-5"}},
        {"model_name":"default","litellm_params":{"model":"groq/llama","api_key":"literal-test-key"}}
    ]}}));
    let cards = store.providers()["providers"].as_array().unwrap().clone();
    let kinds = |name: &str| {
        cards
            .iter()
            .filter(|card| card["provider"] == name)
            .map(|card| card["kind"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(kinds("copilot"), ["common"]);
    assert_eq!(kinds("groq"), ["configured"]);
}

#[test]
fn added_prefixes_count_as_routable_by_key_or_local_address() {
    for (model, extra, routable) in [
        ("groq/llama", json!({"api_key":"literal-test-key"}), true),
        (
            "groq/llama",
            json!({"api_key":"env:MARKITAI_TEST_MISSING_KEY_VARIABLE"}),
            false,
        ),
        ("lm_studio/local", json!({}), true),
        ("hosted_vllm/local", json!({}), false),
        (
            "hosted_vllm/local",
            json!({"api_base":"http://127.0.0.1:8000/v1"}),
            true,
        ),
        ("ollama_chat/llama", json!({}), true),
        // A Bedrock API key routes Bedrock; an unimplemented prefix never routes.
        ("bedrock/x", json!({"api_key":"literal-test-key"}), true),
        ("watsonx/x", json!({"api_key":"literal-test-key"}), false),
    ] {
        let mut params = json!({"model":model});
        for (key, value) in extra.as_object().unwrap() {
            params[key] = value.clone();
        }
        let (_dir, store) = setup(
            json!({"llm":{"enabled":false,"model_list":[{"model_name":"default","litellm_params":params}]}}),
        );
        assert_eq!(store.view()["routable"], routable, "{model} {extra}");
    }
}

#[test]
fn revision_uses_python_float_notation_at_decimal_and_exponent_boundaries() {
    let models = vec![json!({"extra":[1e-7,1e-6,0.0001,1e15,1e16,-0.0,1.2345678901234567e20]})];
    assert_eq!(
        identity::revision(&models, &[]),
        "a1dcbb94d85b3ece5a60547341a09feebe93d1f79aa1fdcf49461a2bf4bab2b8"
    );
}

#[test]
fn http_credentials_reject_environment_references_before_resolution() {
    let (_dir, store) = setup(fixture());
    for value in [
        "env:AUTHORED_CANARY",
        "env:AUTHORED_MISSING",
        " env:AUTHORED_CANARY",
    ] {
        for field in ["api_key", "api_base"] {
            let mut discovery = json!({"provider":"openai"});
            discovery[field] = json!(value);
            assert_eq!(
                store
                    .resolve_discovery(&discovery)
                    .err()
                    .unwrap()
                    .status
                    .as_u16(),
                422
            );
            assert_eq!(
                store.resolve_discovery(&discovery).err().unwrap().detail,
                model::ENV_REFERENCE
            );
            let mut probe = json!({"model":"openai/old"});
            probe[field] = json!(value);
            assert_eq!(
                store.resolve_probe(&probe).err().unwrap().status.as_u16(),
                422
            );
            let mut add = json!({"model":"openai/new","model_name":"new"});
            add[field] = json!(value);
            assert_eq!(
                store
                    .mutate(Mutation::Add(add))
                    .err()
                    .unwrap()
                    .status
                    .as_u16(),
                422
            );
        }
    }
}

#[test]
fn stored_credentials_bind_provider_and_complete_endpoint() {
    let (_dir, store) = setup(fixture());
    let id = store.view()["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    for body in [
        json!({"provider":"anthropic","deployment_id":id}),
        json!({"provider":"openai","deployment_id":id,"api_base":"http://127.0.0.1:9/v1"}),
        json!({"provider":"openai","deployment_id":id,"api_base":null}),
    ] {
        let refused = store.resolve_discovery(&body).err().unwrap();
        assert_eq!(refused.status.as_u16(), 422);
        // Both ways out are named, as Python markitai names them.
        assert_eq!(refused.detail, model::KEY_REUSE);
    }
    assert!(
        store
            .resolve_probe(&json!({"model":"openai/old","api_base":"http://127.0.0.1:9/v1"}))
            .is_err()
    );
    assert!(model::same_base(
        Some("https://EXAMPLE.test:443/v1/"),
        Some("https://example.test/v1")
    ));
    assert!(!model::same_base(
        Some("https://example.test/v1"),
        Some("https://example.test/steal")
    ));
    assert!(!model::same_base(
        Some("https://example.test/v1?tenant=a"),
        Some("https://example.test/v1?tenant=b")
    ));
}

#[test]
fn explicit_empty_connections_never_select_server_environment() {
    let (_dir, store) = setup(fixture());
    for key in [Value::Null, json!(""), json!("own-key")] {
        let result = ok(store.resolve_discovery(
            &json!({"provider":"ollama","api_base":"http://127.0.0.1:9","api_key":key}),
        ));
        assert_eq!(result["use_environment_credentials"], false);
        assert_eq!(result["api_key"], key);
        let result = ok(store.resolve_probe(
            &json!({"model":"openai/old","api_base":"http://127.0.0.1:9","api_key":key}),
        ));
        assert_eq!(result["use_environment_credentials"], false);
        assert_ne!(result["api_key"], "fixture-secret");
    }
    assert_eq!(
        ok(store.resolve_discovery(&json!({"provider":"ollama","api_base":"http://127.0.0.1:9"})))
            ["use_environment_credentials"],
        false
    );
}

#[test]
fn settings_mutations_cannot_retarget_retained_server_keys() {
    let (_dir, store) = setup(fixture());
    let id = store.view()["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let before = fs::read(&store.source.path).unwrap();
    for body in [
        json!({"api_base":"http://127.0.0.1:9"}),
        json!({"model":"anthropic/x"}),
        json!({"api_key":"env:AUTHORED_CANARY"}),
    ] {
        let mut body = body;
        body["expected_revision"] = json!(current(&store));
        assert!(
            store
                .mutate(Mutation::Update {
                    key: id.clone(),
                    body,
                    legacy: false,
                    share: None,
                })
                .is_err()
        );
    }
    for body in [
        json!({"api_base":"http://127.0.0.1:9"}),
        json!({"api_key":"env:AUTHORED_CANARY"}),
    ] {
        assert!(
            store
                .mutate(Mutation::Provider {
                    key: format!("legacy:{id}"),
                    body: Some(body),
                    revision: current(&store)
                })
                .is_err()
        );
    }
    assert!(store.mutate(Mutation::Add(json!({"model_name":"leak","model":"openai/old","credential_deployment_id":id,"api_base":"http://127.0.0.1:9"}))).is_err());
    assert_eq!(fs::read(&store.source.path).unwrap(), before);
}

#[test]
fn literal_mode_survives_save_clear_delete_restore_and_reload() {
    let (_dir, store) = setup(json!({"llm":{"model_list":[]}}));
    ok(store.mutate(Mutation::Add(json!({"model_name":"local","model":"ollama/test","api_base":"http://127.0.0.1:9911","api_key":null}))));
    assert_eq!(
        saved(&store)["llm"]["model_list"][0]["litellm_params"]["use_environment_credentials"],
        false
    );
    ok(store.mutate(Mutation::Delete {
        key: "local".into(),
        legacy: true,
        revision: None,
    }));
    let raw = saved(&store);
    let provider = raw["llm"]["providers"][0]["id"].as_str().unwrap();
    assert_eq!(
        raw["llm"]["providers"][0]["use_environment_credentials"],
        false
    );
    ok(store.mutate(Mutation::Add(
        json!({"model_name":"restored","model":"ollama/test","credential_provider_id":provider}),
    )));
    let (_reload_dir, reload) = setup(saved(&store));
    assert_eq!(
        ok(reload.resolve_probe(&json!({"model_name":"restored"})))["use_environment_credentials"],
        false
    );
}

#[test]
fn environment_provider_references_are_named_not_arbitrary_variables() {
    let (_dir, store) = setup(fixture());
    let result =
        ok(store.resolve_discovery(&json!({"provider":"openai","provider_id":"env:openai"})));
    assert_eq!(result["use_environment_credentials"], true);
    assert!(result["api_key"].is_null());
    for id in ["env:AWS_SECRET_ACCESS_KEY", "env:AUTHORED_MISSING", "env:"] {
        assert_eq!(
            store
                .resolve_discovery(&json!({"provider":"openai","provider_id":id}))
                .err()
                .unwrap()
                .status
                .as_u16(),
            422
        );
    }
    assert!(
        store
            .resolve_discovery(&json!({"provider":"anthropic","provider_id":"env:openai"}))
            .is_err()
    );
    assert!(store.resolve_discovery(&json!({"provider":"openai","provider_id":"env:openai","api_base":"http://127.0.0.1:9"})).is_err());
    ok(store.mutate(Mutation::Add(json!({"model_name":"known-env","model":"openai/test","credential_provider_id":"env:openai"}))));
    assert_eq!(
        saved(&store)["llm"]["model_list"][1]["litellm_params"]["use_environment_credentials"],
        true
    );
}

#[test]
fn an_environment_key_is_saved_at_its_providers_other_region_only() {
    let (_dir, store) = setup(json!({"llm":{"model_list":[]}}));
    ok(store.mutate(Mutation::Add(json!({
        "model_name":"default",
        "model":"moonshot/kimi-k3",
        "credential_provider_id":"env:moonshot",
        "api_base":"https://api.moonshot.cn/v1"
    }))));
    let params = &saved(&store)["llm"]["model_list"][0]["litellm_params"];
    assert_eq!(params["api_base"], "https://api.moonshot.cn/v1");
    assert_eq!(params["use_environment_credentials"], true);
    assert!(params["api_key"].is_null());
    // Each regional provider's other regions, likewise.
    for (model, credential, base) in [
        (
            "minimax/MiniMax-M3",
            "env:minimax",
            "https://api.minimax.cn/v1",
        ),
        (
            "dashscope/qwen-plus",
            "env:dashscope",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ),
    ] {
        ok(store.mutate(Mutation::Add(json!({
            "model_name":"default",
            "model":model,
            "credential_provider_id":credential,
            "api_base":base
        }))));
        let list = saved(&store)["llm"]["model_list"].clone();
        let params = &list
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["litellm_params"]["model"] == model)
            .unwrap()["litellm_params"];
        assert_eq!(params["api_base"], base, "{model}");
        assert_eq!(params["use_environment_credentials"], true, "{model}");
    }
    // Another address, another provider's region, or another provider's key: as before.
    for (model, base) in [
        ("moonshot/kimi-k3", "https://api.example.test/v1"),
        ("moonshot/kimi-k3", "https://api.kimi.com/coding/v1"),
        ("kimi_code/k3", "https://api.kimi.com/coding/v1"),
    ] {
        let refused = store
            .mutate(Mutation::Add(json!({
                "model_name":"default",
                "model":model,
                "credential_provider_id":"env:moonshot",
                "api_base":base
            })))
            .err()
            .unwrap();
        assert_eq!(refused.status.as_u16(), 422, "{model} {base}");
        assert_eq!(refused.detail, model::KEY_REUSE, "{model} {base}");
    }
}

#[test]
fn unprefixed_model_keeps_openai_identity_through_save_discovery_and_probe() {
    let (_dir, store) = setup(json!({"llm":{"model_list":[]}}));
    let view = ok(store.mutate(Mutation::Add(json!({
        "model_name":"unprefixed",
        "model":"fixture-unprefixed",
        "api_key":"authored-literal-key",
        "api_base":"http://127.0.0.1:9911/v1"
    }))));
    let id = view["deployments"][0]["deployment_id"].as_str().unwrap();
    let raw = saved(&store);
    assert_eq!(
        raw["llm"]["model_list"][0]["litellm_params"]["model"],
        "fixture-unprefixed"
    );
    assert_eq!(raw["llm"]["providers"][0]["provider"], "openai");
    let discovery = ok(store.resolve_discovery(&json!({"provider":"openai","deployment_id":id})));
    assert_eq!(discovery["api_key"], "authored-literal-key");
    assert_eq!(discovery["api_base"], "http://127.0.0.1:9911/v1");
    assert_eq!(discovery["use_environment_credentials"], false);
    let probe = ok(store.resolve_probe(&json!({"model":"fixture-unprefixed"})));
    assert_eq!(probe["api_key"], "authored-literal-key");
    assert_eq!(probe["api_base"], "http://127.0.0.1:9911/v1");
    assert_eq!(probe["use_environment_credentials"], false);
    assert!(
        store
            .resolve_discovery(&json!({"provider":"anthropic","deployment_id":id}))
            .is_err()
    );
    assert!(
        store
            .resolve_probe(
                &json!({"model":"fixture-unprefixed","api_base":"http://127.0.0.1:9912/v1"})
            )
            .is_err()
    );
}

#[test]
fn provider_base_updates_cannot_retarget_linked_deployment_overrides() {
    let original = json!({"llm":{
        "providers":[{"id":"provider-a","provider":"openai","api_key":"authored-provider-key-a","api_base":"https://endpoint-a.example.test/v1"}],
        "model_list":[
            {"model_name":"inherited","litellm_params":{"model":"openai/a"},"model_info":{"id":"inherited-deployment","provider_id":"provider-a"}},
            {"model_name":"overridden","litellm_params":{"model":"openai/b","api_key":"authored-deployment-key-b","api_base":"https://endpoint-b.example.test/v1"},"model_info":{"id":"overridden-deployment","provider_id":"provider-a"}}
        ]
    }});
    let (_dir, store) = setup(original);
    let before = fs::read(&store.source.path).unwrap();
    let revision = current(&store);
    for endpoint in [
        "https://endpoint-a.example.test/v1",
        "https://ENDPOINT-A.example.test:443/v1/",
    ] {
        let error = store
            .mutate(Mutation::Provider {
                key: "provider-a".into(),
                revision: revision.clone(),
                body: Some(json!({"api_base":endpoint})),
            })
            .err()
            .unwrap();
        assert_eq!(error.status.as_u16(), 422);
        assert_eq!(current(&store), revision);
        assert_eq!(fs::read(&store.source.path).unwrap(), before);
        let probe = ok(store.resolve_probe(&json!({"deployment_id":"overridden-deployment"})));
        assert_eq!(probe["api_key"], "authored-deployment-key-b");
        assert_eq!(probe["api_base"], "https://endpoint-b.example.test/v1");
    }
    ok(store.mutate(Mutation::Provider {
        key:"provider-a".into(),
        revision,
        body:Some(json!({"api_base":"https://endpoint-a.example.test/v1","api_key":"authored-explicit-replacement"})),
    }));
    for id in ["inherited-deployment", "overridden-deployment"] {
        let probe = ok(store.resolve_probe(&json!({"deployment_id":id})));
        assert_eq!(probe["api_key"], "authored-explicit-replacement");
        assert_eq!(probe["api_base"], "https://endpoint-a.example.test/v1");
        assert_eq!(probe["use_environment_credentials"], false);
    }
}

#[test]
fn an_openai_compatible_endpoint_is_saved_as_a_custom_connection() {
    let (_dir, store) = setup(json!({"llm":{"enabled":false}}));
    let base = "http://127.0.0.1:9133/v1";
    let view = ok(store.mutate(Mutation::Batch(json!({"expected_revision":current(&store),
        "deployments":[{"provider":"custom","model":"openai/fake-model","api_key":"k","api_base":base,"model_name":"default","weight":1}]}))));
    assert_eq!(view["deployments"][0]["model"], "openai/fake-model");
    assert_eq!(view["deployments"][0]["api_base"], "http://127.0.0.1:9133");
    assert_eq!(view["routable"], true);
    let raw = saved(&store);
    let connection = raw["llm"]["providers"][0].clone();
    assert_eq!(connection["provider"], "custom");
    assert_eq!(connection["api_base"], base);
    let id = connection["id"].as_str().unwrap().to_owned();
    assert_eq!(raw["llm"]["model_list"][0]["model_info"]["provider_id"], id);
    let deployment = view["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    // The saved card is the endpoint; its credentials and discovery use it.
    let cards = store.providers()["providers"].as_array().unwrap().clone();
    let card = cards.iter().find(|card| card["provider_id"] == id).unwrap();
    assert_eq!(
        (card["provider"].clone(), card["model_count"].clone()),
        (json!("custom"), json!(1))
    );
    assert_eq!(card["api_base"], "http://127.0.0.1:9133");
    assert_eq!(ok(store.credentials(&id))["api_base"], base);
    let discovery = ok(store.resolve_discovery(&json!({"provider":"custom","provider_id":id})));
    assert_eq!(
        (discovery["api_key"].clone(), discovery["api_base"].clone()),
        (json!("k"), json!(base))
    );
    let discovery =
        ok(store.resolve_discovery(&json!({"provider":"custom","deployment_id":deployment})));
    assert_eq!(discovery["api_base"], base);
    // More models of the endpoint, with or without naming the provider.
    for reference in [
        json!({"model_name":"default","model":"openai/second","credential_provider_id":id}),
        json!({"model_name":"default","model":"openai/third","provider":"custom","credential_deployment_id":deployment}),
        json!({"model_name":"default","model":"openai/fourth","credential_deployment_id":deployment}),
    ] {
        ok(store.mutate(Mutation::Batch(
            json!({"expected_revision":current(&store),"deployments":[reference]}),
        )));
    }
    let raw = saved(&store);
    assert_eq!(raw["llm"]["providers"].as_array().unwrap().len(), 1);
    for entry in raw["llm"]["model_list"].as_array().unwrap() {
        assert_eq!(entry["model_info"]["provider_id"], id, "{entry}");
        assert_eq!(entry["litellm_params"]["api_base"], base, "{entry}");
    }
    // Renaming a model keeps the endpoint and needs no new key.
    ok(store.mutate(Mutation::Update {
        key: deployment.clone(),
        legacy: false,
        share: None,
        body: json!({"expected_revision":current(&store),"model":"openai/renamed"}),
    }));
    let probe = ok(store.resolve_probe(&json!({"deployment_id":deployment})));
    assert_eq!(
        (
            probe["model"].clone(),
            probe["api_key"].clone(),
            probe["api_base"].clone()
        ),
        (json!("openai/renamed"), json!("k"), json!(base))
    );
    // Conversions route to the endpoint.
    let cfg = store.snapshot();
    assert!(markitai_core::llm_capabilities(&cfg).routable);
    let routed = cfg["llm"]["model_list"].as_array().unwrap();
    assert_eq!(routed.len(), 4);
    assert!(
        routed
            .iter()
            .all(|entry| entry["litellm_params"]["api_base"] == base)
    );
    // Deleting the endpoint removes its models.
    ok(store.mutate(Mutation::Provider {
        key: id,
        revision: current(&store),
        body: None,
    }));
    assert!(
        saved(&store)["llm"]["model_list"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn only_openai_models_belong_to_an_openai_compatible_endpoint() {
    let (_dir, store) = setup(json!({"llm":{"enabled":false}}));
    for deployment in [
        json!({"provider":"custom","model":"anthropic/x","api_key":"k","api_base":"http://127.0.0.1:9/v1"}),
        json!({"provider":"custom","model":"openai/x","api_key":"k"}),
        json!({"provider":"custom","model":"openai/x","api_key":"k","api_base":""}),
        json!({"provider":"anthropic","model":"openai/x","api_key":"k"}),
        json!({"provider":"openai","model":"anthropic/x","api_key":"k"}),
        json!({"provider":"groq","model":"openai/x","api_key":"k"}),
    ] {
        let mut entry = deployment.clone();
        entry["model_name"] = json!("default");
        let error = store
            .mutate(Mutation::Batch(
                json!({"expected_revision":current(&store),"deployments":[entry]}),
            ))
            .err()
            .unwrap_or_else(|| panic!("accepted {deployment}"));
        assert_eq!(
            (error.status.as_u16(), error.reason),
            (422, "invalid_settings_request"),
            "{deployment}"
        );
    }
    assert!(!store.source.path.exists() || saved(&store)["llm"].get("model_list").is_none());
}

#[test]
fn saved_endpoints_keep_the_openai_and_compatible_cards_on_offer() {
    let (_dir, store) = setup(json!({"llm":{"enabled":false}}));
    ok(store.mutate(Mutation::Batch(json!({"expected_revision":current(&store),
        "deployments":[
            {"provider":"custom","model":"openai/fake-model","api_key":"k","api_base":"http://127.0.0.1:9133/v1","model_name":"default"},
            {"model":"openai/gpt-test","api_key":"real-openai-test-key","model_name":"default"},
            {"model":"anthropic/claude-test","api_key":"anthropic-test-key","model_name":"default"}
        ]}))));
    let cards = store.providers()["providers"].as_array().unwrap().clone();
    let common = |name: &str| {
        cards
            .iter()
            .any(|card| card["kind"] == "common" && card["provider"] == name)
    };
    assert!(common("custom") && common("openai"));
    // A saved connection still hides the card of any other provider.
    assert!(!common("anthropic"));
    let configured: Vec<_> = cards
        .iter()
        .filter(|card| card["kind"] == "configured")
        .map(|card| card["provider"].as_str().unwrap())
        .collect();
    assert_eq!(configured, ["custom", "openai", "anthropic"]);
}

#[test]
fn stored_keys_are_shown_masked_as_the_reference_shows_them() {
    for (key, shown) in [
        ("sk-fixture-1234567890", "sk\u{2026}7890"),
        ("twelve-chars", "tw\u{2026}hars"),
        ("eleven-char", "\u{2026}ar"),
        ("four", "\u{2026}ur"),
        ("abc", "\u{2026}"),
        ("", "\u{2026}"),
        ("密钥密钥密钥密钥密钥密钥", "密钥\u{2026}密钥密钥"),
        ("env:OPENAI_API_KEY", "env:OPENAI_API_KEY"),
    ] {
        assert_eq!(model::masked_key(key), shown, "{key}");
    }
}
#[test]
fn a_masked_key_sent_back_is_refused_and_a_missing_one_keeps_the_saved_key() {
    let (_dir, store) = setup(fixture());
    let id = format!(
        "legacy:{}",
        store.view()["deployments"][0]["deployment_id"]
            .as_str()
            .unwrap()
    );
    let shown = ok(store.credentials(&id))["api_key_masked"].clone();
    assert_eq!(shown, "fi\u{2026}cret");
    let before = fs::read(&store.source.path).unwrap();
    let error = store
        .mutate(Mutation::Provider {
            key: id.clone(),
            revision: current(&store),
            body: Some(json!({"expected_revision":current(&store),"api_key":shown})),
        })
        .err()
        .unwrap();
    assert_eq!(
        (error.status.as_u16(), error.reason),
        (422, "invalid_settings_request")
    );
    assert_eq!(fs::read(&store.source.path).unwrap(), before);
    let add = json!({"expected_revision":current(&store),"model":"openai/new","api_key":"sk\u{2026}7890","model_name":"masked"});
    assert_eq!(
        store
            .mutate(Mutation::Add(add))
            .err()
            .unwrap()
            .status
            .as_u16(),
        422
    );
    assert!(
        store
            .resolve_discovery(&json!({"provider":"openai","api_key":"sk\u{2026}7890"}))
            .is_err()
    );
    // Saving only the address it already has keeps the stored key.
    let base = saved(&store)["llm"]["model_list"][0]["litellm_params"]["api_base"].clone();
    ok(store.mutate(Mutation::Provider {
        key: id,
        revision: current(&store),
        body: Some(json!({"expected_revision":current(&store),"api_base":base})),
    }));
    let raw = saved(&store);
    assert_eq!(
        raw["llm"]["model_list"][0]["litellm_params"]["api_key"],
        "fixture-secret"
    );
    let provider = raw["llm"]["providers"][0]["id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        ok(store.credentials(&provider))["api_key_masked"],
        "fi\u{2026}cret"
    );
}

#[test]
fn a_saved_address_is_shown_masked_and_kept_when_sent_back_as_shown() {
    // The fixture's address carries userinfo and a token parameter.
    let (_dir, store) = setup(fixture());
    let id = format!(
        "legacy:{}",
        store.view()["deployments"][0]["deployment_id"]
            .as_str()
            .unwrap()
    );
    let shown = ok(store.credentials(&id))["api_base"].clone();
    assert_eq!(shown, "https://\u{2026}@example.test/custom?token=\u{2026}");
    let stored = "https://user:pass@example.test/custom?token=hidden";
    // Sent back as shown, alone: the address does not change.
    ok(store.mutate(Mutation::Provider {
        key: id.clone(),
        revision: current(&store),
        body: Some(json!({"expected_revision":current(&store),"api_base":shown})),
    }));
    assert_eq!(
        saved(&store)["llm"]["model_list"][0]["litellm_params"]["api_base"],
        stored
    );
    // Saving gives the legacy deployment a lasting id: address it again.
    let id = format!(
        "legacy:{}",
        store.view()["deployments"][0]["deployment_id"]
            .as_str()
            .unwrap()
    );
    // Sent back as shown with a new key: the key changes, the address stays.
    ok(store.mutate(Mutation::Provider {
        key: id,
        revision: current(&store),
        body: Some(
            json!({"expected_revision":current(&store),"api_base":shown,"api_key":"fixture-next-key"}),
        ),
    }));
    let raw = saved(&store);
    assert_eq!(
        raw["llm"]["model_list"][0]["litellm_params"]["api_base"],
        stored
    );
    assert_eq!(
        raw["llm"]["model_list"][0]["litellm_params"]["api_key"],
        "fixture-next-key"
    );
    let provider = raw["llm"]["providers"][0]["id"]
        .as_str()
        .unwrap()
        .to_owned();
    // Discovery with the shown address uses the stored one.
    let discovery = ok(store
        .resolve_discovery(&json!({"provider":"openai","provider_id":provider,"api_base":shown})));
    assert_eq!(discovery["api_base"], stored);
    // An edited address that still holds a mask is refused, and so is one
    // for a new deployment or a probe.
    let edited = "https://\u{2026}@example.test/other?token=\u{2026}";
    let error = store
        .mutate(Mutation::Provider {
            key: provider.clone(),
            revision: current(&store),
            body: Some(
                json!({"expected_revision":current(&store),"api_base":edited,"api_key":"k2"}),
            ),
        })
        .err()
        .unwrap();
    assert_eq!(
        (error.status.as_u16(), error.reason),
        (422, "invalid_settings_request")
    );
    let add = json!({"expected_revision":current(&store),"model":"openai/new","api_base":edited,"api_key":"k","model_name":"masked"});
    assert_eq!(
        store
            .mutate(Mutation::Add(add))
            .err()
            .unwrap()
            .status
            .as_u16(),
        422
    );
    assert!(
        store
            .resolve_probe(&json!({"model":"openai/x","api_base":edited,"api_key":"k"}))
            .is_err()
    );
    assert_eq!(
        saved(&store)["llm"]["model_list"][0]["litellm_params"]["api_base"],
        stored
    );
    // No settings response carries the stored address's secrets.
    for response in [
        store.view(),
        store.providers(),
        ok(store.credentials(&provider)),
    ] {
        let text = response.to_string();
        assert!(
            !text.contains("pass@") && !text.contains("hidden"),
            "{text}"
        );
    }
}

#[test]
fn keyless_settings_writes_store_the_placeholder_and_never_the_environment() {
    let (_dir, store) = setup(json!({"llm":{"enabled":false,"model_list":[]}}));
    let base = "http://127.0.0.1:9/v1";
    ok(store.mutate(Mutation::Batch(json!({"expected_revision":current(&store),"deployments":[
        {"model_name":"default","model":"openai/a","provider":"openai","api_base":base},
        {"model_name":"default","model":"openai/b","provider":"custom","api_base":base,"api_key":""},
        {"model_name":"default","model":"openai/c","credential_provider_id":"env:openai"},
        {"model_name":"default","model":"claude-agent/sonnet"}
    ]}))));
    let raw = saved(&store);
    let models = raw["llm"]["model_list"].as_array().unwrap();
    for entry in &models[..2] {
        assert_eq!(entry["litellm_params"]["api_key"], model::NO_KEY, "{entry}");
        assert_eq!(
            entry["litellm_params"]["use_environment_credentials"],
            false
        );
    }
    // The environment card keeps its own reference at its own endpoint, and a
    // subscription runtime takes no key.
    assert!(models[2]["litellm_params"].get("api_key").is_none());
    assert_eq!(
        models[2]["litellm_params"]["use_environment_credentials"],
        true
    );
    assert!(models[3]["litellm_params"].get("api_key").is_none());
    for saved in raw["llm"]["providers"].as_array().unwrap() {
        assert_eq!(saved["api_key"], model::NO_KEY, "{saved}");
    }
    // Shown as no key anywhere.
    let view = store.view();
    for entry in view["deployments"].as_array().unwrap() {
        assert_eq!(entry["api_key_configured"], false, "{entry}");
    }
    let cards = store.providers()["providers"].as_array().unwrap().clone();
    // The two saved connections (the environment card's model is listed on
    // its own, as a deployment's connection).
    let configured: Vec<_> = cards
        .iter()
        .filter(|card| card["kind"] == "configured" && card["deployment_id"].is_null())
        .collect();
    assert_eq!(configured.len(), 2);
    for card in configured {
        assert_eq!(card["api_key_configured"], false, "{card}");
        let credentials = ok(store.credentials(card["provider_id"].as_str().unwrap()));
        assert_eq!(credentials["api_key"], Value::Null, "{credentials}");
        assert_eq!(credentials["api_key_masked"], Value::Null);
        assert_eq!(credentials["api_key_source"], Value::Null);
        // Discovery and tests of the saved connection never use the environment.
        let id = card["provider_id"].as_str().unwrap();
        let discovery =
            ok(store.resolve_discovery(&json!({"provider":card["provider"],"provider_id":id})));
        assert_eq!(discovery["use_environment_credentials"], false);
    }
    // A draft test at an address the caller names uses the placeholder too.
    for key in [None, Some(json!(null)), Some(json!(""))] {
        let mut body = json!({"model":"groq/x","api_base":base});
        if let Some(key) = key {
            body["api_key"] = key;
        }
        let probe = ok(store.resolve_probe(&body));
        assert_eq!(probe["api_key"], model::NO_KEY, "{body}");
        assert_eq!(probe["use_environment_credentials"], false);
    }
    // Without an address a draft is the environment's own provider endpoint.
    let probe = ok(store.resolve_probe(&json!({"model":"groq/x"})));
    assert_eq!(probe["api_key"], Value::Null);
    assert_eq!(probe["use_environment_credentials"], true);
}

#[test]
fn keyless_cloud_entries_use_the_server_credentials_only_at_the_cloud_endpoint() {
    let (_dir, store) = setup(json!({"llm":{"enabled":false,"model_list":[]}}));
    ok(store.mutate(Mutation::Batch(json!({"expected_revision":current(&store),"deployments":[
        {"model_name":"default","model":"bedrock/anthropic.claude-x","provider":"bedrock"},
        {"model_name":"default","model":"vertex_ai/gemini-2.5-flash","provider":"vertex_ai"},
        {"model_name":"default","model":"bedrock/anthropic.claude-y","provider":"bedrock","api_key":"bedrock-test-key"},
        {"model_name":"default","model":"bedrock/anthropic.claude-z","provider":"bedrock","api_base":"http://127.0.0.1:9"}
    ]}))));
    let raw = saved(&store);
    let models = raw["llm"]["model_list"].as_array().unwrap();
    // Neither key nor address: the cloud's own endpoint with the server's
    // credentials, as Python markitai's LiteLLM uses them.
    for entry in &models[..2] {
        assert!(entry["litellm_params"].get("api_key").is_none(), "{entry}");
        assert!(entry["litellm_params"].get("api_base").is_none(), "{entry}");
        assert_eq!(
            entry["litellm_params"]["use_environment_credentials"], true,
            "{entry}"
        );
    }
    // A Bedrock API key is used alone; an address never gets them: Bedrock
    // keeps its address with the placeholder, which it sends as its bearer
    // key and never signs with the server's AWS chain (core cloud_auth tests).
    assert_eq!(
        models[2]["litellm_params"]["use_environment_credentials"],
        false
    );
    assert_eq!(models[3]["litellm_params"]["api_key"], model::NO_KEY);
    assert_eq!(
        models[3]["litellm_params"]["api_base"],
        "http://127.0.0.1:9"
    );
    assert_eq!(
        models[3]["litellm_params"]["use_environment_credentials"],
        false
    );
    // A draft test is the same entry as it would be saved.
    let probe = ok(store.resolve_probe(&json!({"model":"vertex_ai/gemini-2.5-flash"})));
    assert_eq!(probe["use_environment_credentials"], true);
    // Clearing the Bedrock key goes back to the server's credentials.
    let id = store.view()["deployments"][2]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    ok(store.mutate(Mutation::Update {
        key: id,
        legacy: false,
        share: None,
        body: json!({"expected_revision":current(&store),"api_key":null}),
    }));
    let raw = saved(&store);
    let entry = &raw["llm"]["model_list"][2];
    assert_eq!(
        entry["litellm_params"]["model"],
        "bedrock/anthropic.claude-y"
    );
    assert!(entry["litellm_params"].get("api_key").is_none(), "{entry}");
    assert_eq!(
        entry["litellm_params"]["use_environment_credentials"], true,
        "{entry}"
    );
}

#[test]
fn a_cloud_login_model_is_refused_an_address_at_save_and_test_time() {
    let vertex = "api_base: Vertex AI models send the server's own Google credentials to any address, so here they can only use Vertex AI's own endpoint; set another address in the config file";
    let refused = |result: ApiResult<Value>, detail: &str| {
        let error = result.expect_err("refused");
        assert_eq!(
            (error.status.as_u16(), error.reason, error.detail.as_str()),
            (422, "invalid_settings_request", detail)
        );
    };
    let original = json!({"llm":{"enabled":false,
    "providers":[{"id":"gcp","provider":"vertex_ai","use_environment_credentials":true}],
    "model_list":[
        {"model_name":"default","litellm_params":{"model":"vertex_ai/gemini-2.5-flash","use_environment_credentials":true},"model_info":{"id":"v","provider_id":"gcp"}},
        {"model_name":"default","litellm_params":{"model":"openai/gpt-x","api_key":"sk-test-0000000000","api_base":"https://gateway.example/v1"},"model_info":{"id":"o"}}
    ]}});
    let (_dir, store) = setup(original);
    let before = fs::read(&store.source.path).unwrap();
    // Saving, whatever key the entry has, and through every Python route
    // name; watsonx and GigaChat name their token.
    for (model, key, detail) in [
        ("vertex_ai/gemini-2.5-pro", None, vertex.to_owned()),
        (
            "vertex_ai/gemini-2.5-pro",
            Some("typed-key"),
            vertex.to_owned(),
        ),
        ("Vertex_AI_Beta/claude-x", None, vertex.to_owned()),
        (
            "watsonx/ibm/granite",
            None,
            vertex
                .replace("Vertex AI", "watsonx")
                .replace("Google credentials", "WATSONX_TOKEN"),
        ),
        (
            "gigachat/GigaChat-Max",
            None,
            vertex
                .replace("Vertex AI", "GigaChat")
                .replace("Google credentials", "GIGACHAT_ACCESS_TOKEN"),
        ),
    ] {
        let mut deployment =
            json!({"model_name":"default","model":model,"api_base":"https://capture.example/v1"});
        if let Some(key) = key {
            deployment["api_key"] = json!(key);
        }
        refused(
            store.mutate(Mutation::Batch(
                json!({"expected_revision":current(&store),"deployments":[deployment]}),
            )),
            &detail,
        );
        // A draft test at the caller's address is refused before any request.
        let mut probe = json!({"model":model,"api_base":"https://capture.example/v1"});
        if let Some(key) = key {
            probe["api_key"] = json!(key);
        }
        refused(store.resolve_probe(&probe), &detail);
    }
    // A model added to the saved Vertex AI connection with an address, an
    // address set on that connection or one of its models, and a model
    // moved to Vertex AI while it keeps another address.
    refused(
        store.mutate(Mutation::Batch(json!({"expected_revision":current(&store),"deployments":[
            {"model_name":"default","model":"vertex_ai/gemini-2.5-pro","credential_provider_id":"gcp","api_base":"https://capture.example","api_key":null}
        ]}))),
        vertex,
    );
    refused(
        store.mutate(Mutation::Provider {
            key: "gcp".into(),
            revision: current(&store),
            body: Some(
                json!({"expected_revision":current(&store),"api_base":"https://capture.example"}),
            ),
        }),
        vertex,
    );
    let ids: Vec<String> = store.view()["deployments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["deployment_id"].as_str().unwrap().to_owned())
        .collect();
    refused(
        store.mutate(Mutation::Update {
            key: ids[0].clone(),
            legacy: false, share: None,
            body: json!({"expected_revision":current(&store),"api_base":"https://capture.example","api_key":null}),
        }),
        vertex,
    );
    refused(
        store.mutate(Mutation::Update {
            key: ids[1].clone(),
            legacy: false,
            share: None,
            body: json!({"expected_revision":current(&store),"model":"vertex_ai/gemini-2.5-pro"}),
        }),
        vertex,
    );
    assert_eq!(fs::read(&store.source.path).unwrap(), before);
    // Without an address they keep the cloud's own endpoint; a saved model's
    // own test is the owner's configuration, and Bedrock keeps its address.
    ok(store.mutate(Mutation::Batch(json!({"expected_revision":current(&store),"deployments":[
        {"model_name":"default","model":"vertex_ai/gemini-2.5-pro","provider":"vertex_ai"},
        {"model_name":"default","model":"bedrock/anthropic.claude-x","provider":"bedrock","api_base":"https://capture.example"}
    ]}))));
    ok(store.resolve_probe(&json!({"model":"vertex_ai/gemini-2.5-pro"})));
    ok(store.resolve_probe(&json!({"deployment_id":ids[0]})));
    let probe = ok(store.resolve_probe(
        &json!({"model":"bedrock/anthropic.claude-y","api_base":"https://capture.example"}),
    ));
    assert_eq!(probe["api_key"], model::NO_KEY);
    assert_eq!(probe["use_environment_credentials"], false);
    // Clearing an address is allowed.
    ok(store.mutate(Mutation::Update {
        key: ids[0].clone(),
        legacy: false,
        share: None,
        body: json!({"expected_revision":current(&store),"api_base":null}),
    }));
}

#[test]
fn a_keyed_connection_cannot_become_keyless_at_another_host() {
    let original = json!({"llm":{
        "providers":[{"id":"gateway","provider":"openai","api_key":"authored-gateway-key","api_base":"https://gateway.example.test/v1","use_environment_credentials":false}],
        "model_list":[{"model_name":"default","litellm_params":{"model":"openai/a","api_key":"authored-gateway-key","api_base":"https://gateway.example.test/v1","use_environment_credentials":false},"model_info":{"id":"a","provider_id":"gateway"}}]}});
    let (_dir, store) = setup(original);
    let before = fs::read(&store.source.path).unwrap();
    let refuse = |error: ApiError| {
        assert_eq!(error.status.as_u16(), 422);
        assert_eq!(error.reason, "invalid_settings_request");
        assert_eq!(error.detail, model::KEYLESS_MOVE);
    };
    for other in [
        "http://127.0.0.1:9/v1",
        "https://gateway.example.test:8443/v1",
    ] {
        let change = json!({"api_key":null,"api_base":other});
        let mut body = change.clone();
        body["expected_revision"] = json!(current(&store));
        refuse(
            store
                .mutate(Mutation::Provider {
                    key: "gateway".into(),
                    body: Some(body.clone()),
                    revision: current(&store),
                })
                .err()
                .unwrap(),
        );
        refuse(
            store
                .mutate(Mutation::Update {
                    key: "a".into(),
                    body,
                    legacy: false,
                    share: None,
                })
                .err()
                .unwrap(),
        );
        assert_eq!(fs::read(&store.source.path).unwrap(), before);
    }
    // In place, the key may go: the placeholder takes its place.
    let same = json!({"expected_revision":current(&store),"api_key":null,"api_base":"https://GATEWAY.example.test:443/other"});
    ok(store.mutate(Mutation::Update {
        key: "a".into(),
        body: same,
        legacy: false,
        share: None,
    }));
    let entry = &saved(&store)["llm"]["model_list"][0]["litellm_params"];
    assert_eq!(entry["api_key"], model::NO_KEY);
    assert_eq!(entry["use_environment_credentials"], false);
    // Clearing a connection's key and address together returns it to the
    // provider's own endpoint, still without the environment's key.
    ok(store.mutate(Mutation::Provider {
        key: "gateway".into(),
        body: Some(json!({"api_key":null,"api_base":null})),
        revision: current(&store),
    }));
    let raw = saved(&store);
    assert!(raw["llm"]["providers"][0].get("api_key").is_none());
    assert_eq!(
        raw["llm"]["providers"][0]["use_environment_credentials"],
        false
    );
    // A keyless connection may move, keeping the placeholder.
    let (_dir, store) = setup(json!({"llm":{"model_list":[]}}));
    ok(store.mutate(Mutation::Add(
        json!({"model_name":"default","model":"lm_studio/x","api_base":"http://127.0.0.1:1234/v1"}),
    )));
    let id = saved(&store)["llm"]["providers"][0]["id"]
        .as_str()
        .unwrap()
        .to_owned();
    ok(store.mutate(Mutation::Provider {
        key: id,
        body: Some(json!({"api_key":null,"api_base":"http://192.0.2.7:1234/v1"})),
        revision: current(&store),
    }));
    let raw = saved(&store);
    assert_eq!(raw["llm"]["providers"][0]["api_key"], model::NO_KEY);
    assert_eq!(
        raw["llm"]["model_list"][0]["litellm_params"]["api_key"],
        model::NO_KEY
    );
    assert_eq!(
        raw["llm"]["model_list"][0]["litellm_params"]["api_base"],
        "http://192.0.2.7:1234/v1"
    );
}

#[test]
fn a_python_keyless_placeholder_reads_as_no_key() {
    let (_dir, store) = setup(json!({"llm":{"model_list":[
        {"model_name":"default","litellm_params":{"model":"openai/x","api_key":"markitai-no-key","api_base":"http://127.0.0.1:9/v1"},"model_info":{"id":"python"}}
    ]}}));
    assert_eq!(store.view()["deployments"][0]["api_key_configured"], false);
    let card = store.providers()["providers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|card| card["deployment_id"] == "python")
        .unwrap()
        .clone();
    assert_eq!(card["api_key_configured"], false);
    let credentials = ok(store.credentials(card["provider_id"].as_str().unwrap()));
    assert_eq!(credentials["api_key_masked"], Value::Null);
    assert_eq!(credentials["api_key_source"], Value::Null);
}

fn edit_fixture() -> Value {
    let mut value = fixture();
    value["fetch"] = json!({"playwright":{"extra_http_headers":{"X-Trace":"trace-header-value"}},
        "cloudflare":{"api_token":"env:FIXTURE_TOKEN"}});
    value["batch"] = json!({"concurrency":2});
    value
}
fn refused(result: ApiResult<Value>) -> (u16, &'static str, String) {
    let error = result.err().unwrap();
    (error.status.as_u16(), error.reason, error.detail)
}
#[test]
fn config_edit_masks_secrets_and_a_round_trip_keeps_them() {
    let (_dir, store) = setup(edit_fixture());
    let view = ok(store.config_view());
    let text = view["text"].as_str().unwrap().to_owned();
    for secret in [
        "fixture-secret",
        "user:pass",
        "hidden",
        "trace-header-value",
    ] {
        assert!(!text.contains(secret), "{secret} in {text}");
    }
    // The key, the address and the header; the env: reference stays.
    assert_eq!(view["masked"], 3);
    assert!(text.contains("env:FIXTURE_TOKEN"));
    assert!(text.ends_with('}'));
    assert_eq!(
        (
            view["exists"].clone(),
            view["origin"].clone(),
            view["read_only"].clone()
        ),
        (json!(true), json!("explicit"), json!(false))
    );
    let revision = view["revision"].as_str().unwrap().to_owned();
    assert!(revision.starts_with("sha256:") && revision.len() == 71);

    // Unchanged masked values come back; a changed batch needs a restart.
    let mut edited: Value = serde_json::from_str(&text).unwrap();
    edited["batch"]["concurrency"] = json!(3);
    let answer = ok(store.replace_config(&revision, &edited.to_string()));
    assert_eq!(answer["restart_required"], true);
    let mut expected = edit_fixture();
    expected["batch"]["concurrency"] = json!(3);
    assert_eq!(saved(&store), expected);
    assert_ne!(answer["revision"], json!(revision));
    assert_eq!(answer["masked"], 3);

    // A model change alone applies at once and needs no restart.
    let mut edited: Value = serde_json::from_str(answer["text"].as_str().unwrap()).unwrap();
    edited["llm"]["model_list"][0]["litellm_params"]["model"] = json!("openai/edited");
    edited["llm"]["model_list"]
        .as_array_mut()
        .unwrap()
        .push(json!({"model_name":"second","litellm_params":{"model":"openai/second","api_key":"second-secret-value"}}));
    let answer =
        ok(store.replace_config(answer["revision"].as_str().unwrap(), &edited.to_string()));
    assert_eq!(answer["restart_required"], false);
    assert_eq!(answer["masked"], 4);
    let models: Vec<_> = store.view()["deployments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["model"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(models, ["openai/edited", "openai/second"]);
    assert_eq!(
        store.snapshot()["llm"]["model_list"][1]["litellm_params"]["api_key"],
        "second-secret-value"
    );
    let raw = saved(&store);
    assert_eq!(
        raw["llm"]["model_list"][0]["litellm_params"]["api_key"],
        "fixture-secret"
    );
    assert_eq!(
        raw["llm"]["model_list"][0]["litellm_params"]["api_base"],
        "https://user:pass@example.test/custom?token=hidden"
    );
    assert_eq!(
        store.view()["revision"],
        json!(identity::revision(
            raw["llm"]["model_list"].as_array().unwrap(),
            &[]
        ))
    );
}
#[test]
fn config_edit_refuses_a_repeated_key_and_shows_neither_value() {
    let (_dir, store) = setup(edit_fixture());
    // JSON keeps the last of two values: the first secret was shown in clear.
    let twice = r#"{"llm":{"model_list":[{"model_name":"d","litellm_params":{"model":"openai/x",
"api_key":"sk-FIRST1111222233334444","api_key":"sk-SECOND555566667777"}}]}}"#;
    fs::write(&store.source.path, twice).unwrap();
    let view = ok(store.config_view());
    assert_eq!(view["text"], Value::Null);
    assert_eq!(
        view["problem"],
        "line 2, column 38: the key \"api_key\" appears twice in one object"
    );
    let shown = view.to_string();
    assert!(
        !shown.contains("FIRST") && !shown.contains("SECOND"),
        "{shown}"
    );
    let revision = view["revision"].as_str().unwrap().to_owned();
    let (status, reason, detail) = refused(store.replace_config(&revision, twice));
    assert_eq!((status, reason), (422, "invalid_config"));
    assert!(
        detail.ends_with("the key \"api_key\" appears twice in one object"),
        "{detail}"
    );
    assert_eq!(fs::read_to_string(&store.source.path).unwrap(), twice);
}
#[test]
fn config_edit_refuses_stale_invalid_and_moved_masked_text() {
    let (_dir, store) = setup(edit_fixture());
    let view = ok(store.config_view());
    let revision = view["revision"].as_str().unwrap().to_owned();
    let text = view["text"].as_str().unwrap().to_owned();
    let original = fs::read(&store.source.path).unwrap();
    assert_eq!(
        refused(store.replace_config("sha256:0", &text)).1,
        "stale_revision"
    );
    let (status, reason, detail) = refused(store.replace_config(&revision, "{\n  \"a\": 1,\n}"));
    assert_eq!((status, reason), (422, "invalid_config"));
    assert!(detail.starts_with("line 3, column 1: "), "{detail}");
    assert_eq!(
        refused(store.replace_config(&revision, "[]")).2,
        "the configuration must be a JSON object"
    );
    let (status, reason, detail) =
        refused(store.replace_config(&revision, r#"{"batch":{"concurrency":0}}"#));
    assert_eq!((status, reason), (422, "invalid_config"));
    assert!(detail.contains("concurrency"), "{detail}");
    let (status, reason, detail) =
        refused(store.replace_config(&revision, r#"{"llm":{"model_list":[{"model_name":"x"}]}}"#));
    assert_eq!((status, reason), (422, "invalid_config"), "{detail}");
    let mut moved: Value = serde_json::from_str(&text).unwrap();
    moved["fetch"]["cloudflare"]["api_token"] =
        moved["llm"]["model_list"][0]["litellm_params"]["api_key"].clone();
    assert_eq!(
        refused(store.replace_config(&revision, &moved.to_string())).2,
        "/fetch/cloudflare/api_token: a masked value can only be kept where it was; type the full value or an env: reference"
    );
    assert_eq!(fs::read(&store.source.path).unwrap(), original);
    assert_eq!(current(&store), store.view()["revision"].as_str().unwrap());
}
#[test]
fn config_edit_is_refused_under_session_overrides_and_handles_missing_or_broken_files() {
    let (dir, store) = setup(edit_fixture());
    let blocked = Store::new(
        store.snapshot().as_ref().clone(),
        SettingsSource {
            path: store.source.path.clone(),
            origin: "explicit".into(),
            overrides: Some(json!({"llm":{"model_list":[]}})),
        },
    )
    .unwrap_or_else(|e| panic!("{}", e.detail));
    let view = ok(blocked.config_view());
    assert_eq!(view["read_only"], true);
    assert_eq!(
        refused(blocked.replace_config(view["revision"].as_str().unwrap(), "{}")).1,
        "settings_read_only"
    );
    // No file: an empty document, created by the first save.
    let absent = Store::new(
        markitai_core::config::normalize(&json!({})).unwrap(),
        SettingsSource {
            path: dir.path().join("new/config.json"),
            origin: "default".into(),
            overrides: None,
        },
    )
    .unwrap_or_else(|e| panic!("{}", e.detail));
    let view = ok(absent.config_view());
    assert_eq!(
        (
            view["exists"].clone(),
            view["revision"].clone(),
            view["text"].clone()
        ),
        (json!(false), json!("absent"), json!("{}\n"))
    );
    let answer = ok(absent.replace_config("absent", r#"{"batch":{"concurrency":4}}"#));
    assert_eq!(answer["exists"], true);
    assert_eq!(answer["restart_required"], true);
    assert_eq!(
        fs::read_to_string(dir.path().join("new/config.json")).unwrap(),
        r#"{"batch":{"concurrency":4}}"#
    );
    // A broken file is shown as a problem and can be replaced.
    fs::write(&store.source.path, "{\"llm\": ").unwrap();
    let view = ok(store.config_view());
    assert_eq!(view["text"], Value::Null);
    assert!(
        view["problem"]
            .as_str()
            .unwrap()
            .starts_with("line 1, column 8: ")
    );
    let answer = ok(store.replace_config(view["revision"].as_str().unwrap(), "{}"));
    assert_eq!(answer["restart_required"], true);
    assert_eq!(answer["text"], "{}");
}
#[test]
fn config_edit_keeps_the_files_own_layout_and_saves_exactly_what_was_sent() {
    // Keys out of order, tabs, odd spacing and no trailing newline.
    let file = "{\n\t\"llm\" : {\"model_list\":[ {\"model_name\":\"default\",\n\t\t\"litellm_params\":{\"model\":\"openai/first\",  \"api_key\":\"layout-secret-0001\"}} ],\n\t\"enabled\":false},\n  \"batch\":{ \"concurrency\" : 2 },\n\"fetch\":{\"playwright\":{\"extra_http_headers\":{\"X-Trace\":\"layout-header-value\"}}}}";
    let (_dir, store) = setup(json!({}));
    fs::write(&store.source.path, file).unwrap();
    let view = ok(store.config_view());
    let text = view["text"].as_str().unwrap().to_owned();
    assert_eq!(
        text,
        file.replace("\"layout-secret-0001\"", "\"la\u{2026}0001\"")
            .replace("\"layout-header-value\"", "\"la\u{2026}alue\"")
    );
    assert_eq!(view["masked"], 2);
    // The user changes the batch and adds a model in their own layout.
    let sent = text.replace("\"concurrency\" : 2", "\"concurrency\" : 3").replace(
        "} ],",
        "},\n\t\t{\"model_name\":\"default\",\"litellm_params\":{\"model\":\"openai/second\"}} ],",
    );
    let answer = ok(store.replace_config(view["revision"].as_str().unwrap(), &sent));
    let expected = sent
        .replace("\"la\u{2026}0001\"", "\"layout-secret-0001\"")
        .replace("\"la\u{2026}alue\"", "\"layout-header-value\"");
    assert_eq!(fs::read_to_string(&store.source.path).unwrap(), expected);
    assert_eq!(answer["restart_required"], true);
    assert_eq!(answer["text"], json!(sent));
    assert_eq!(
        answer["revision"],
        json!(editor::revision(Some(expected.as_bytes())))
    );
    let models: Vec<_> = store.view()["deployments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["model"].clone())
        .collect();
    assert_eq!(models, [json!("openai/first"), json!("openai/second")]);
}
#[test]
fn sharing_is_saved_in_the_model_entry_and_shown_on_saved_rows_only() {
    let (_dir, store) = setup(fixture());
    let view = store.view();
    assert_eq!(view["deployments"][0]["share"], "private");
    let id = view["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let shared = ok(store.mutate(Mutation::Update {
        key: id.clone(),
        body: json!({"expected_revision":current(&store)}),
        legacy: false,
        share: Some(true),
    }));
    assert_eq!(shared["deployments"][0]["share"], "shared");
    let file = saved(&store);
    let entry = &file["llm"]["model_list"][0];
    assert_eq!(entry["model_info"]["share"], "shared");
    // Nothing else of the entry changed.
    assert_eq!(entry["litellm_params"]["api_key"], "fixture-secret");
    assert_eq!(entry["custom"], json!({"keep":true}));
    let id = shared["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let private = ok(store.mutate(Mutation::Update {
        key: id,
        body: json!({"expected_revision":current(&store),"weight":2}),
        legacy: false,
        share: Some(false),
    }));
    assert_eq!(private["deployments"][0]["share"], "private");
    assert_eq!(private["deployments"][0]["weight"], 2);
    let file = saved(&store);
    assert!(
        file["llm"]["model_list"][0]["model_info"]
            .get("share")
            .is_none()
    );
}
/// Several models of one connection written by hand (no ids), as the
/// round-18 report had them: the provider card stands for the first.
fn garden() -> Value {
    let entry = |name: &str| {
        json!({"model_name":"default","litellm_params":{"model":format!("openai/{name}"),
            "api_key":"sk-fixture-garden-0000","api_base":"https://garden.example/v1"}})
    };
    json!({"llm":{"model_list":[entry("garden-lite"),entry("garden-flash"),entry("garden-pro")]}})
}
fn models_of(store: &Store) -> Vec<String> {
    store.view()["deployments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["model"].as_str().unwrap().to_owned())
        .collect()
}
#[test]
fn one_batch_adds_and_removes_through_the_card_of_a_removed_model() {
    for removed in ["openai/garden-lite", "openai/garden-flash"] {
        let (_dir, store) = setup(garden());
        let view = store.view();
        assert_eq!(view["batch_remove"], true);
        let card = store.providers()["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|card| card["kind"] == "configured")
            .cloned()
            .unwrap();
        let anchor = card["deployment_id"].as_str().unwrap().to_owned();
        assert_eq!(anchor, view["deployments"][0]["deployment_id"]);
        let target = view["deployments"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["model"] == removed)
            .unwrap()["deployment_id"]
            .clone();
        let after = ok(store.mutate(Mutation::Batch(json!({
            "expected_revision": current(&store),
            "remove": [target],
            "deployments": [{"model_name":"default","model":"openai/garden-new","weight":1,
                "provider":"openai","credential_deployment_id":anchor}],
        }))));
        let mut expected = vec![
            "openai/garden-lite",
            "openai/garden-flash",
            "openai/garden-pro",
        ];
        expected.retain(|model| *model != removed);
        expected.push("openai/garden-new");
        assert_eq!(models_of(&store), expected);
        assert_eq!(after["revision"], json!(current(&store)));
        // The new model uses the removed model's connection, saved once.
        let raw = saved(&store);
        let providers = raw["llm"]["providers"].as_array().unwrap();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0]["api_key"], "sk-fixture-garden-0000");
        assert!(
            raw["llm"]["model_list"]
                .as_array()
                .unwrap()
                .iter()
                .all(|entry| entry["model_info"]["provider_id"] == providers[0]["id"])
        );
    }
}
#[test]
fn a_batch_is_all_or_nothing_and_checks_every_id_against_its_revision() {
    let (_dir, store) = setup(garden());
    let view = store.view();
    let ids: Vec<Value> = view["deployments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["deployment_id"].clone())
        .collect();
    let revision = current(&store);
    let before = fs::read(&store.source.path).unwrap();
    let add = json!({"model_name":"default","model":"openai/garden-new","credential_deployment_id":ids[0]});
    for (body, status, reason) in [
        // An unknown id, one id twice, an invalid addition, nothing at all.
        (
            json!({"expected_revision":revision,"remove":[ids[1],"no-such-id"],"deployments":[add]}),
            404,
            "settings_entry_missing",
        ),
        (
            json!({"expected_revision":revision,"remove":[ids[1],ids[1]]}),
            422,
            "invalid_settings_request",
        ),
        (
            json!({"expected_revision":revision,"remove":[ids[1]],"deployments":[{"model_name":"x","model":"openai/x","weight":101}]}),
            422,
            "invalid_weight",
        ),
        (
            json!({"expected_revision":revision,"remove":[],"deployments":[]}),
            422,
            "invalid_settings_request",
        ),
        (
            json!({"expected_revision":revision,"remove":[7]}),
            422,
            "invalid_settings_request",
        ),
        (
            json!({"expected_revision":"stale","remove":[ids[1]]}),
            409,
            "stale_revision",
        ),
    ] {
        let (code, why, _) = refused(store.mutate(Mutation::Batch(body.clone())));
        assert_eq!((code, why), (status, reason), "{body}");
        assert_eq!(fs::read(&store.source.path).unwrap(), before, "{body}");
    }
    // Removals alone; ids from before the write, in any order.
    ok(store.mutate(Mutation::Batch(
        json!({"expected_revision":revision,"remove":[ids[2],ids[0]]}),
    )));
    assert_eq!(models_of(&store), ["openai/garden-flash"]);
}
#[test]
fn every_weight_write_is_a_whole_number_from_0_to_100() {
    let (_dir, store) = setup(garden());
    let id = store.view()["deployments"][1]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let before = fs::read(&store.source.path).unwrap();
    for weight in [json!(101), json!(-1), json!(1.5), json!("3"), json!(1e3)] {
        let revision = current(&store);
        for result in [
            store.mutate(Mutation::Batch(json!({"expected_revision":revision,
                "deployments":[{"model_name":"x","model":"openai/x","weight":weight}]}))),
            store.mutate(Mutation::Add(
                json!({"model_name":"x","model":"openai/x","weight":weight}),
            )),
            store.mutate(Mutation::Update {
                key: id.clone(),
                legacy: false,
                share: None,
                body: json!({"expected_revision":revision,"weight":weight}),
            }),
        ] {
            assert_eq!(
                refused(result),
                (422, "invalid_weight", model::WEIGHT_RULE.to_owned()),
                "{weight}"
            );
        }
    }
    assert_eq!(fs::read(&store.source.path).unwrap(), before);
    for weight in [0, 100] {
        // The first write gives the hand-written entries their ids.
        let id = store.view()["deployments"][1]["deployment_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let view = ok(store.mutate(Mutation::Update {
            key: id,
            legacy: false,
            share: None,
            body: json!({"expected_revision":current(&store),"weight":weight}),
        }));
        assert_eq!(view["deployments"][1]["weight"], weight);
    }
}
#[test]
fn edit_config_refuses_a_new_weight_over_100_and_keeps_an_old_one() {
    let mut value = garden();
    value["llm"]["model_list"][2]["litellm_params"]["weight"] = json!(500);
    let (_dir, store) = setup(value);
    // A file written before the limit still loads and shows its weight.
    assert_eq!(store.view()["deployments"][2]["weight"], 500);
    let view = ok(store.config_view());
    let text = view["text"].as_str().unwrap().to_owned();
    let revision = view["revision"].as_str().unwrap().to_owned();
    let before = fs::read(&store.source.path).unwrap();
    let mut edited: Value = serde_json::from_str(&text).unwrap();
    edited["llm"]["model_list"][0]["litellm_params"]["weight"] = json!(101);
    assert_eq!(
        refused(store.replace_config(&revision, &edited.to_string())),
        (
            422,
            "invalid_weight",
            "/llm/model_list/0 (openai/garden-lite in default): Routing weight must be a whole number from 0 to 100".into()
        )
    );
    edited["llm"]["model_list"][0]["litellm_params"]["weight"] = json!(2.5);
    assert_eq!(
        refused(store.replace_config(&revision, &edited.to_string())).1,
        "invalid_weight"
    );
    // Another entry's old weight changed to another big one is refused too.
    edited["llm"]["model_list"][0]["litellm_params"]["weight"] = json!(100);
    edited["llm"]["model_list"][2]["litellm_params"]["weight"] = json!(600);
    assert!(
        refused(store.replace_config(&revision, &edited.to_string()))
            .2
            .starts_with("/llm/model_list/2 ")
    );
    assert_eq!(fs::read(&store.source.path).unwrap(), before);
    // The old weight left as it was saves with the new one.
    edited["llm"]["model_list"][2]["litellm_params"]["weight"] = json!(500);
    ok(store.replace_config(&revision, &edited.to_string()));
    let weights: Vec<_> = store.view()["deployments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["weight"].clone())
        .collect();
    assert_eq!(weights, [json!(100), json!(1), json!(500)]);
}

#[test]
fn a_project_config_keeps_models_at_their_own_address_off_environment_keys() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("markitai.json");
    fs::write(
        &path,
        r#"{"llm":{"model_list":[
            {"model_name":"a","litellm_params":{"model":"openai/gpt-x","api_base":"https://capture.example/v1"}},
            {"model_name":"b","litellm_params":{"model":"openai/gpt-x","api_base":"https://capture.example/v1","api_key":"env:OPENAI_API_KEY"}}
        ]}}"#,
    )
    .unwrap();
    let sealed = |origin: &str| {
        let source = SettingsSource {
            path: path.clone(),
            origin: origin.into(),
            overrides: None,
        };
        let base = load_base(&source).unwrap_or_else(|e| panic!("{}", e.detail));
        let store = Store::new(base, source).unwrap_or_else(|e| panic!("{}", e.detail));
        let snapshot = store.snapshot();
        let list = snapshot["llm"]["model_list"].as_array().unwrap().clone();
        list.iter()
            .map(|entry| {
                (
                    entry["litellm_params"]["use_environment_credentials"].clone(),
                    entry["litellm_params"]["api_key"].clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        sealed("project"),
        [
            (json!(false), Value::Null),
            (
                json!(false),
                json!(markitai_core::provider_management::NO_KEY)
            )
        ]
    );
    // Named with -c (or the user's own file), the same text keeps its meaning.
    assert_eq!(
        sealed("explicit"),
        [
            (Value::Null, Value::Null),
            (Value::Null, json!("env:OPENAI_API_KEY"))
        ]
    );
    // The file itself is never rewritten.
    assert!(
        !fs::read_to_string(&path)
            .unwrap()
            .contains("use_environment_credentials")
    );
}
