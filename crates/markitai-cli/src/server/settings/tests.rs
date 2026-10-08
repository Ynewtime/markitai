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
        legacy: false,
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
    assert_eq!(
        ok(store.credentials(&provider))["api_key"],
        "fixture-next-key"
    );
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
        assert_eq!(
            store
                .resolve_discovery(&body)
                .err()
                .unwrap()
                .status
                .as_u16(),
            422
        );
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
                    legacy: false
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
