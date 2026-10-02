use super::*;
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
        legacy: false,
        body: json!({"expected_revision":revision,"model":"openai/new","api_base":null}),
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
                legacy: true
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
            json!({"expected_revision":current(&store),"api_key":"env:NEXT","api_base":null}),
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
            "env:NEXT"
        );
        assert_eq!(
            raw["llm"]["model_list"][i]["model_info"]["provider_id"],
            provider
        );
    }
    assert_eq!(ok(store.credentials(&provider))["api_key"], "env:NEXT");
    assert!(ok(store.credentials(&provider))["api_base"].is_null());
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
    let discovery=ok(store.resolve_discovery(&json!({"provider":"openai","provider_id":format!("legacy:{id}"),"api_base":"http://127.0.0.1/v1","refresh":true})));
    assert_eq!(discovery["api_key"], "fixture-secret");
    assert_eq!(discovery["api_base"], "http://127.0.0.1/v1");
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
fn symlink_and_fifo_config_fail_before_read_or_write() {
    use std::os::unix::fs::symlink;
    let (dir, store) = setup(fixture());
    let real = dir.path().join("real.json");
    fs::rename(&store.source.path, &real).unwrap();
    symlink(&real, &store.source.path).unwrap();
    assert!(store.config_path().is_err());
    assert!(
        store
            .mutate(Mutation::Delete {
                key: "same".into(),
                revision: None,
                legacy: true
            })
            .is_err()
    );
    assert_eq!(fs::read(&real).unwrap(), fixture().to_string().as_bytes());
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
        body: json!({"expected_revision":current(&store),"api_key":null}),
    }));
    let saved = saved(&store);
    let entry = &saved["llm"]["model_list"][0];
    assert!(entry["model_info"].get("provider_id").is_none());
    assert!(entry["litellm_params"].get("api_key").is_none());
    assert_eq!(
        entry["litellm_params"]["api_base"],
        "http://127.0.0.1:8123/v1"
    );
    assert_eq!(saved["llm"]["providers"], raw["llm"]["providers"]);
    assert_eq!(saved["llm"]["model_list"][1], raw["llm"]["model_list"][1]);
    assert!(ok(store.resolve_probe(&json!({"deployment_id":"one-id"})))["api_key"].is_null());
    assert_eq!(
        ok(store.resolve_probe(&json!({"deployment_id":"two-id"})))["api_key"],
        "fixture-key"
    );
}

#[test]
fn every_openai_compatible_prefix_is_offered_with_its_endpoint_key_and_discovery() {
    let (_dir, store) = setup(json!({"llm":{"enabled":false,"model_list":[]}}));
    let cards = store.providers()["providers"].as_array().unwrap().clone();
    let card = |name: &str| {
        cards
            .iter()
            .find(|card| card["provider"] == name && card["kind"] != "environment")
            .unwrap_or_else(|| panic!("no {name} card"))
            .clone()
    };
    // The reference's eight stay first and "common"; the added ones follow.
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
            "ollama",
            "deepseek",
            "openrouter",
            "azure",
            "custom"
        ]
    );
    let compatible = cards
        .iter()
        .filter(|card| card["kind"] == "compatible")
        .count();
    assert_eq!(compatible, 16);
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
        assert_eq!(card(name)["supports_discovery"], false, "{name}");
    }
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
        ("bedrock/x", json!({"api_key":"literal-test-key"}), false),
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
