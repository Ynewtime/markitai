//! A user token's own models (`/api/me/models*`) and the models an admin
//! shares: kept apart per user, held to rules that keep the server's
//! credentials and machine out of reach, routed in one pool for that user's
//! conversions, and deleted with the token. The server's environment holds a
//! sentinel key and endpoint that no user conversion may ever reach.
use super::workbench::Endpoint;
use super::*;

const ADMIN: &str = "admin-test-token-0001";
const USER_A_KEY: &str = "user-a-fixture-key";
const USER_B_KEY: &str = "user-b-fixture-key";
const ADMIN_KEY: &str = "admin-fixture-key";
const SENTINEL_KEY: &str = "env-sentinel-key";
const NOTES: &[u8] = b"# Notes\n\nA paragraph the endpoint returns unchanged.\n";

/// The server with the admin token and the environment a user must never
/// reach: a key, a model and an endpoint (`sentinel`) for automatic routing.
fn start(root: &Path, sentinel: &Endpoint) -> Server {
    let base = format!("http://127.0.0.1:{}/v1", sentinel.port);
    Server::start_with(
        root,
        &[
            ("MARKITAI_SERVE_TOKEN", std::ffi::OsStr::new(ADMIN)),
            ("OPENAI_API_KEY", std::ffi::OsStr::new(SENTINEL_KEY)),
            ("OPENAI_API_BASE", std::ffi::OsStr::new(&base)),
            ("MODEL", std::ffi::OsStr::new("openai/env-model")),
        ],
    )
}
fn call(server: &Server, method: &str, path: &str, token: &str, body: Option<Value>) -> Reply {
    let authorization = format!("Bearer {token}");
    let mut headers = vec![("Authorization", authorization.as_str())];
    let body = body.map(|value| value.to_string()).unwrap_or_default();
    if !body.is_empty() {
        headers.push(("Content-Type", "application/json"));
    }
    server.request(method, path, &headers, body.as_bytes())
}
fn ok(server: &Server, method: &str, path: &str, token: &str, body: Option<Value>) -> Value {
    let reply = call(server, method, path, token, body);
    assert_eq!(reply.status, 200, "{method} {path}: {}", reply.text());
    reply.json()
}
fn reason(reply: &Reply) -> (u16, Value) {
    (reply.status, reply.json()["reason"].clone())
}
/// A new user token: (id, token).
fn user(server: &Server, label: &str) -> (String, String) {
    let reply = call(
        server,
        "POST",
        "/api/access-tokens",
        ADMIN,
        Some(json!({"label":label})),
    );
    assert_eq!(reply.status, 201, "{}", reply.text());
    let created = reply.json();
    (
        created["id"].as_str().unwrap().to_owned(),
        created["token"].as_str().unwrap().to_owned(),
    )
}
fn revision(server: &Server, path: &str, token: &str) -> Value {
    ok(server, "GET", path, token, None)["revision"].clone()
}
fn add(server: &Server, token: &str, deployments: Value) -> Reply {
    let revision = revision(server, "/api/me/models", token);
    call(
        server,
        "POST",
        "/api/me/models/deployments/batch",
        token,
        Some(json!({"expected_revision":revision,"deployments":deployments})),
    )
}
/// Submit `files` as `token` with `options`; the reply.
fn submit(server: &Server, token: &str, files: &[(&str, &[u8])], options: Value) -> Reply {
    let (content_type, body) = multipart(files, json!([]), options);
    let authorization = format!("Bearer {token}");
    server.request(
        "POST",
        "/api/jobs",
        &[
            ("Content-Type", content_type.as_str()),
            ("Authorization", authorization.as_str()),
        ],
        &body,
    )
}
fn done(server: &Server, token: &str, id: &str) -> Value {
    let mut value = Value::Null;
    until("job completion", || {
        value = ok(server, "GET", &format!("/api/jobs/{id}"), token, None);
        value["status"] != "running"
    });
    value
}
/// An LLM job of `token` that converts; its finished snapshot.
fn converted(server: &Server, token: &str) -> Value {
    let reply = submit(server, token, &[("notes.md", NOTES)], json!({"llm":true}));
    assert_eq!(reply.status, 201, "{}", reply.text());
    let id = reply.json()["job_id"].as_str().unwrap().to_owned();
    let snapshot = done(server, token, &id);
    let item = &snapshot["items"][0];
    assert_eq!(item["status"], "done", "{item}");
    assert_eq!(item["llm_enhanced"], true, "{item}");
    snapshot
}
fn keys(value: &Value) -> Vec<String> {
    value.as_object().unwrap().keys().cloned().collect()
}
/// What Authorization headers an endpoint received, emptying its record.
fn received(endpoint: &Endpoint) -> Vec<String> {
    std::mem::take(&mut *endpoint.requests.lock().unwrap())
        .into_iter()
        .map(|(_, authorization, _)| authorization)
        .collect()
}
/// An address of this machine that is not loopback, where a user's own
/// model server can listen; `None` without one.
fn lan_address() -> Option<std::net::IpAddr> {
    let socket = std::net::UdpSocket::bind(("0.0.0.0", 0)).ok()?;
    socket.connect(("192.0.2.1", 80)).ok()?;
    let ip = socket.local_addr().ok()?.ip();
    (!ip.is_loopback() && !ip.is_unspecified()).then_some(ip)
}

#[test]
fn users_keep_their_own_models_apart_masked_and_within_the_rules() {
    let sentinel = Endpoint::start();
    let temp = tempfile::tempdir().unwrap();
    let server = start(temp.path(), &sentinel);
    let (a_id, a) = user(&server, "Laptop A");
    let (_, b) = user(&server, "Laptop B");
    assert_eq!(
        ok(&server, "GET", "/api/capabilities", &a, None)["features"]["user_models"],
        true
    );

    // Only a user token: the admin is told where its own models are.
    let reply = call(&server, "GET", "/api/me/models", ADMIN, None);
    assert_eq!(reason(&reply), (403, json!("user_token_required")));
    assert_eq!(reply.headers["cache-control"], "no-store");
    let reply = call(&server, "GET", "/api/settings/llm", &a, None);
    assert_eq!(reason(&reply), (403, json!("settings_forbidden")));

    let empty = ok(&server, "GET", "/api/me/models", &a, None);
    assert_eq!(
        empty,
        json!({"configured":false,"routable":false,"source":"user","config_path":null,
            "config_origin":"user","revision":empty["revision"],"deployments":[],"detected":[],
            "shared":[],"limit":50,"batch_remove":true})
    );
    let reply = add(
        &server,
        &a,
        json!([{"model_name":"default","model":"openai/own-model","api_key":USER_A_KEY}]),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.headers["cache-control"], "no-store");
    let view = reply.json();
    assert_eq!(view["configured"], true);
    assert_eq!(view["routable"], true);
    let row = &view["deployments"][0];
    assert_eq!(row["model"], "openai/own-model");
    assert_eq!(row["api_key_configured"], true);
    assert!(row.get("share").is_none(), "{row}");
    let deployment = row["deployment_id"].as_str().unwrap().to_owned();
    let cards = ok(&server, "GET", "/api/me/models/providers", &a, None);
    let card = cards["providers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|card| card["kind"] == "configured")
        .unwrap()
        .clone();
    let connection = card["provider_id"].as_str().unwrap().to_owned();

    // A's own: its connection shows the key masked only.
    let credentials = ok(
        &server,
        "GET",
        &format!("/api/me/models/providers/{connection}/credentials"),
        &a,
        None,
    );
    assert_eq!(credentials["api_key_masked"], "us…-key");
    assert_eq!(credentials["api_key_source"], Value::Null);

    // B sees nothing of A's and can change or test none of it.
    let other = ok(&server, "GET", "/api/me/models", &b, None);
    assert_eq!(other["deployments"], json!([]));
    let b_revision = other["revision"].clone();
    for (method, path, body, expected) in [
        (
            "PATCH",
            format!("/api/me/models/deployments/{deployment}"),
            Some(json!({"expected_revision":b_revision,"weight":3})),
            (404, "settings_entry_missing"),
        ),
        (
            "DELETE",
            format!(
                "/api/me/models/deployments/{deployment}?expected_revision={}",
                b_revision.as_str().unwrap()
            ),
            None,
            (404, "settings_entry_missing"),
        ),
        (
            "POST",
            "/api/me/models/test".to_owned(),
            Some(json!({"deployment_id":deployment})),
            (422, "invalid_settings_request"),
        ),
        (
            "GET",
            format!("/api/me/models/providers/{connection}/credentials"),
            None,
            (404, "settings_entry_missing"),
        ),
        (
            "PATCH",
            format!("/api/me/models/providers/{connection}"),
            Some(json!({"expected_revision":b_revision,"api_key":"replaced-key-value"})),
            (404, "settings_entry_missing"),
        ),
        (
            "POST",
            "/api/me/models/deployments/batch".to_owned(),
            Some(
                json!({"expected_revision":b_revision,"deployments":[{"model_name":"default",
                "model":"openai/x","credential_deployment_id":deployment}]}),
            ),
            (404, "settings_entry_missing"),
        ),
    ] {
        let reply = call(&server, method, &path, &b, body);
        assert_eq!(
            reason(&reply),
            (expected.0, json!(expected.1)),
            "{method} {path}: {}",
            reply.text()
        );
    }
    // The admin's settings show no user's model.
    let settings = ok(&server, "GET", "/api/settings/llm", ADMIN, None);
    assert_eq!(settings["deployments"], json!([]));

    // The rules: no server account or login, a key or an address of one's
    // own, never the server's machine or environment, at most 50 models.
    let revision_a = revision(&server, "/api/me/models", &a);
    for (deployment, expected) in [
        (
            json!({"model":"claude-agent/sonnet"}),
            (422, "server_credentials"),
        ),
        (
            json!({"model":"vertex_ai/gemini-pro","api_key":USER_A_KEY}),
            (422, "server_credentials"),
        ),
        (
            json!({"model":"bedrock/anthropic.claude"}),
            (422, "server_credentials"),
        ),
        (
            json!({"model":"openai/no-key"}),
            (422, "key_or_address_required"),
        ),
        (json!({"model":"ollama/llama3"}), (422, "server_address")),
        (
            json!({"model":"ollama/llama3","api_base":"http://127.0.0.1:11434"}),
            (422, "server_address"),
        ),
        (
            json!({"model":"openai/local","api_key":USER_A_KEY,"api_base":"http://localhost:9/v1"}),
            (422, "server_address"),
        ),
        (
            json!({"model":"openai/x","credential_provider_id":"env:openai"}),
            (422, "invalid_settings_request"),
        ),
        (
            json!({"model":"openai/x","api_key":"env:OPENAI_API_KEY"}),
            (422, "invalid_settings_request"),
        ),
    ] {
        let mut deployment = deployment;
        deployment["model_name"] = json!("default");
        let reply = call(
            &server,
            "POST",
            "/api/me/models/deployments/batch",
            &a,
            Some(json!({"expected_revision":revision_a,"deployments":[deployment.clone()]})),
        );
        assert_eq!(
            reason(&reply),
            (expected.0, json!(expected.1)),
            "{deployment}: {}",
            reply.text()
        );
    }
    let reply = call(
        &server,
        "POST",
        "/api/me/models/deployments/batch",
        &a,
        Some(
            json!({"expected_revision":"stale","deployments":[{"model_name":"default","model":"openai/y","api_key":USER_A_KEY}]}),
        ),
    );
    assert_eq!(reply.status, 409);
    assert_eq!(
        reply.json()["detail"],
        json!({"code":"stale_revision","current_revision":revision_a})
    );
    // `share` is the admin's.
    let reply = call(
        &server,
        "PATCH",
        &format!("/api/me/models/deployments/{deployment}"),
        &a,
        Some(json!({"expected_revision":revision_a,"share":"shared"})),
    );
    assert_eq!(reason(&reply), (422, json!("invalid_settings_request")));
    // Tests and discoveries follow the rules before anything is sent.
    for (path, body, expected) in [
        (
            "/api/me/models/test",
            json!({"model":"openai/draft"}),
            "key_or_address_required",
        ),
        (
            "/api/me/models/test",
            json!({"model":"copilot/gpt-5"}),
            "server_credentials",
        ),
        (
            "/api/me/models/test",
            json!({"model":"openai/draft","api_key":USER_A_KEY,"api_base":"http://[::1]:9/v1"}),
            "server_address",
        ),
        (
            "/api/me/models/model-discovery",
            json!({"provider":"ollama"}),
            "server_address",
        ),
        (
            "/api/me/models/model-discovery",
            json!({"provider":"claude-agent"}),
            "server_credentials",
        ),
        (
            "/api/me/models/model-discovery",
            json!({"provider":"openai","provider_id":"env:openai"}),
            "invalid_settings_request",
        ),
    ] {
        let reply = call(&server, "POST", path, &a, Some(body.clone()));
        assert_eq!(
            reason(&reply),
            (422, json!(expected)),
            "{body}: {}",
            reply.text()
        );
    }
    // 50 models at most.
    let many: Vec<Value> = (0..49)
        .map(|index| json!({"model_name":"default","model":format!("openai/m{index}"),"api_key":USER_A_KEY,"weight":0}))
        .collect();
    let reply = add(&server, &a, json!(many));
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.json()["deployments"].as_array().unwrap().len(), 50);
    let reply = add(
        &server,
        &a,
        json!([{"model_name":"default","model":"openai/one-more","api_key":USER_A_KEY}]),
    );
    assert_eq!(reason(&reply), (409, json!("model_limit")));

    // A non-LLM job, its history and every answer: no key anywhere.
    let reply = submit(&server, &a, &[("notes.md", NOTES)], json!({}));
    assert_eq!(reply.status, 201, "{}", reply.text());
    let job = reply.json()["job_id"].as_str().unwrap().to_owned();
    done(&server, &a, &job);
    let mut answers = Vec::new();
    for token in [ADMIN, a.as_str(), b.as_str()] {
        for path in [
            "/api/capabilities",
            "/api/history?include=items",
            "/api/me/models",
            "/api/me/models/providers",
            "/api/settings/llm",
            "/api/settings/llm/providers",
            "/api/settings/config",
            "/api/access-tokens",
        ] {
            answers.push(call(&server, "GET", path, token, None).text());
        }
        answers.push(call(&server, "GET", &format!("/api/jobs/{job}"), token, None).text());
    }
    for answer in &answers {
        for secret in [USER_A_KEY, ADMIN_KEY, SENTINEL_KEY] {
            assert!(!answer.contains(secret), "{secret} in {answer}");
        }
    }
    let tokens = ok(&server, "GET", "/api/access-tokens", ADMIN, None);
    let counts: HashMap<String, Value> = tokens["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                t["label"].as_str().unwrap().to_owned(),
                t["model_count"].clone(),
            )
        })
        .collect();
    assert_eq!(counts["Laptop A"], 50);
    assert_eq!(counts["Laptop B"], 0);

    // Saved owner-only beside the history, never in the configuration file.
    let folder = temp.path().join("home/serve/user-models");
    let file = folder.join(format!("{a_id}.json"));
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&file), 0o600);
        assert_eq!(mode(&folder), 0o700);
    }
    let saved: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!(saved["version"], 1);
    assert!(saved.to_string().contains(USER_A_KEY));
    for entry in saved["models"].as_array().unwrap() {
        assert_eq!(
            entry["litellm_params"]["use_environment_credentials"],
            false
        );
    }
    let config = std::fs::read_to_string(temp.path().join("config.json")).unwrap();
    assert!(!config.contains(USER_A_KEY));
    assert!(received(&sentinel).is_empty());
    server.stop();
}

#[test]
fn shared_models_serve_users_at_once_and_revoked_users_lose_their_own() {
    let sentinel = Endpoint::start();
    let admin_model = Endpoint::start();
    let temp = tempfile::tempdir().unwrap();
    // The admin's model, saved before the start (so no model is detected
    // from the environment), private at first.
    let base = format!("http://127.0.0.1:{}/v1", admin_model.port);
    configure(temp.path());
    let path = temp.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["llm"]["model_list"] = json!([{"model_name":"default","litellm_params":{"model":"openai/fake-model",
        "api_key":ADMIN_KEY,"api_base":base,"use_environment_credentials":false},
        "model_info":{"id":"shared-model-1","provider_id":"admin-connection-1"}}]);
    config["llm"]["providers"] = json!([{"id":"admin-connection-1","provider":"custom","api_key":ADMIN_KEY,
        "api_base":base,"use_environment_credentials":false}]);
    std::fs::write(&path, config.to_string()).unwrap();
    let server = start(temp.path(), &sentinel);
    let (a_id, a) = user(&server, "Laptop A");
    let (b_id, b) = user(&server, "Laptop B");
    let view = ok(&server, "GET", "/api/settings/llm", ADMIN, None);
    assert_eq!(view["deployments"][0]["share"], "private");
    assert_eq!(view["detected"], json!([]));
    let shared_id = view["deployments"][0]["deployment_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let admin_job = converted(&server, ADMIN);
    assert_eq!(received(&admin_model), [format!("Bearer {ADMIN_KEY}")]);

    // B has nothing: no LLM job, while a plain conversion works and sends
    // no model request at all.
    let capabilities = ok(&server, "GET", "/api/capabilities", &b, None);
    assert_eq!(
        capabilities["llm"],
        json!({"configured":false,"routable":false,"effective":false,"models":[]})
    );
    let reply = submit(&server, &b, &[("notes.md", NOTES)], json!({"llm":true}));
    assert_eq!(reason(&reply), (422, json!("llm_unavailable")));
    let reply = submit(&server, &b, &[("notes.md", NOTES)], json!({}));
    assert_eq!(reply.status, 201, "{}", reply.text());
    let plain = done(&server, &b, reply.json()["job_id"].as_str().unwrap());
    assert_eq!(plain["items"][0]["status"], "done");
    assert_eq!(plain["items"][0]["llm_enhanced"], false);

    // Shared: B's next conversion uses it, and sees only what it is.
    let share = |value: &str| {
        let revision_admin = revision(&server, "/api/settings/llm", ADMIN);
        ok(
            &server,
            "PATCH",
            &format!("/api/settings/llm/deployments/{shared_id}"),
            ADMIN,
            Some(json!({"expected_revision":revision_admin,"share":value})),
        )
    };
    let revision_admin = revision(&server, "/api/settings/llm", ADMIN);
    let reply = call(
        &server,
        "PATCH",
        &format!("/api/settings/llm/deployments/{shared_id}"),
        ADMIN,
        Some(json!({"expected_revision":revision_admin,"share":"public"})),
    );
    assert_eq!(reason(&reply), (422, json!("invalid_settings_request")));
    assert_eq!(share("shared")["deployments"][0]["share"], "shared");
    let config: Value =
        serde_json::from_slice(&std::fs::read(temp.path().join("config.json")).unwrap()).unwrap();
    assert_eq!(
        config["llm"]["model_list"][0]["model_info"]["share"],
        "shared"
    );
    let view = ok(&server, "GET", "/api/me/models", &b, None);
    assert_eq!(
        view["shared"],
        json!([{"id":shared_id,"model":"openai/fake-model","provider":"custom","weight":1,"default_weight":1}])
    );
    let mut fields = keys(&view["shared"][0]);
    fields.sort();
    assert_eq!(
        fields,
        ["default_weight", "id", "model", "provider", "weight"]
    );
    assert_eq!(view["routable"], true);
    assert_eq!(view["configured"], false);
    let capabilities = ok(&server, "GET", "/api/capabilities", &b, None);
    assert_eq!(capabilities["llm"]["routable"], true);
    assert_eq!(capabilities["llm"]["models"], json!(["openai/fake-model"]));
    let user_job = converted(&server, &b);
    assert_eq!(received(&admin_model), [format!("Bearer {ADMIN_KEY}")]);
    let (mine, theirs) = (&user_job["items"][0], &admin_job["items"][0]);
    assert_eq!(mine["models"], json!(["fake-model"]));
    assert_eq!(mine["pricing"], theirs["pricing"]);
    assert_eq!(mine["cost_usd"], theirs["cost_usd"]);
    for answer in [
        view.to_string(),
        capabilities.to_string(),
        user_job.to_string(),
    ] {
        for hidden in [ADMIN_KEY, &base, "api_key_configured", "routing_group"] {
            assert!(!answer.contains(hidden), "{hidden} in {answer}");
        }
    }

    // Private again: at once nothing for B.
    assert_eq!(share("private")["deployments"][0]["share"], "private");
    let view = ok(&server, "GET", "/api/me/models", &b, None);
    assert_eq!(view["shared"], json!([]));
    assert_eq!(view["routable"], false);
    assert_eq!(
        ok(&server, "GET", "/api/capabilities", &b, None)["llm"]["routable"],
        false
    );
    let reply = submit(&server, &b, &[("notes.md", NOTES)], json!({"llm":true}));
    assert_eq!(reason(&reply), (422, json!("llm_unavailable")));

    // A's own model, at an address of this machine that is not loopback,
    // with the shared one: A's conversions reach only those two.
    let lan = lan_address();
    let own_model = lan.map(Endpoint::start_at);
    if let Some(own_model) = &own_model {
        let own_base = format!("http://{}:{}/v1", lan.unwrap(), own_model.port);
        let reply = add(
            &server,
            &a,
            json!([{"provider":"custom","model":"openai/fake-model","api_key":USER_A_KEY,
                "api_base":own_base,"model_name":"default"}]),
        );
        assert_eq!(reply.status, 200, "{}", reply.text());
        let id = reply.json()["deployments"][0]["deployment_id"]
            .as_str()
            .unwrap()
            .to_owned();
        // Only its own while nothing is shared.
        converted(&server, &a);
        assert_eq!(received(own_model), [format!("Bearer {USER_A_KEY}")]);
        assert!(received(&admin_model).is_empty());
        // Own model off, shared on: the shared one only.
        share("shared");
        let revision_a = revision(&server, "/api/me/models", &a);
        ok(
            &server,
            "PATCH",
            &format!("/api/me/models/deployments/{id}"),
            &a,
            Some(json!({"expected_revision":revision_a,"weight":0})),
        );
        converted(&server, &a);
        assert!(received(own_model).is_empty());
        assert_eq!(received(&admin_model), [format!("Bearer {ADMIN_KEY}")]);
    } else {
        eprintln!("no address beside loopback: the own-model routing check is skipped");
        share("shared");
        let reply = add(
            &server,
            &a,
            json!([{"model_name":"default","model":"openai/own-model","api_key":USER_A_KEY,"weight":0}]),
        );
        assert_eq!(reply.status, 200, "{}", reply.text());
    }
    let a_job = converted(&server, &a);
    received(&admin_model);

    // B's own model, kept byte for byte through an admin-token regeneration.
    let reply = add(
        &server,
        &b,
        json!([{"model_name":"default","model":"openai/b-model","api_key":USER_B_KEY,"weight":0}]),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    let folder = temp.path().join("home/serve/user-models");
    let b_file = folder.join(format!("{b_id}.json"));
    let b_bytes = std::fs::read(&b_file).unwrap();
    let counts = ok(&server, "GET", "/api/access-tokens", ADMIN, None);
    let counted: Vec<_> = counts["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["model_count"].clone())
        .collect();
    assert_eq!(counted, [json!(1), json!(1)]);
    let regenerated = call(
        &server,
        "POST",
        "/api/access-token/regenerate",
        ADMIN,
        Some(json!({})),
    );
    assert_eq!(regenerated.status, 200, "{}", regenerated.text());
    let admin = regenerated.json()["token"].as_str().unwrap().to_owned();
    assert_eq!(std::fs::read(&b_file).unwrap(), b_bytes);

    // Revoking A deletes its models; its job is the shared models' only.
    let a_file = folder.join(format!("{a_id}.json"));
    let orphan = folder.join("u_0000000000ff.json");
    std::fs::copy(&a_file, &orphan).unwrap();
    let reply = call(
        &server,
        "DELETE",
        &format!("/api/access-tokens/{a_id}"),
        &admin,
        None,
    );
    assert_eq!(reply.status, 204, "{}", reply.text());
    assert!(!a_file.exists());
    until("revocation line", || {
        server.captured.lock().unwrap().contains(
            "Access token \"Laptop A\" revoked from the web workbench; its 1 model was deleted.",
        )
    });
    let job = a_job["job_id"].as_str().unwrap();
    let item = a_job["items"][0]["item_id"].as_str().unwrap();
    let reply = call(
        &server,
        "POST",
        &format!("/api/jobs/{job}/items/{item}/retry"),
        &admin,
        Some(json!({"options":{"llm":true}})),
    );
    assert_eq!(reply.status, 202, "{}", reply.text());
    let retried = done(&server, &admin, job);
    assert_eq!(retried["items"][0]["status"], "done", "{retried}");
    assert_eq!(received(&admin_model), [format!("Bearer {ADMIN_KEY}")]);
    if let Some(own_model) = &own_model {
        assert!(received(own_model).is_empty());
    }
    assert!(received(&sentinel).is_empty());
    server.stop();

    // A restart deletes the file of a token that no longer exists.
    assert!(orphan.exists());
    let restarted = start(temp.path(), &sentinel);
    assert!(!orphan.exists());
    assert_eq!(std::fs::read(&b_file).unwrap(), b_bytes);
    assert_eq!(
        ok(&restarted, "GET", "/api/me/models", &b, None)["deployments"][0]["model"],
        "openai/b-model"
    );
    // A shared model that fails names itself to the user, never its host.
    let host = format!("127.0.0.1:{}", admin_model.port);
    drop(admin_model);
    let reply = submit(&restarted, &b, &[("notes.md", NOTES)], json!({"llm":true}));
    assert_eq!(reply.status, 201, "{}", reply.text());
    let failed = done(&restarted, &b, reply.json()["job_id"].as_str().unwrap());
    let text = failed.to_string();
    assert!(text.contains("deployment openai/fake-model"), "{text}");
    assert!(!text.contains(&host), "{text}");
    restarted.stop();
    assert!(received(&sentinel).is_empty());
}

/// A model server that answers every request with `status` and an error
/// whose words name where it is; records each Authorization header.
struct Failing {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl Failing {
    fn start(status: u16) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen, stopping) = (requests.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::SeqCst) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(_) => break,
                };
                stream.set_nonblocking(false).unwrap();
                stream.set_read_timeout(Some(WAIT)).unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut start = String::new();
                reader.read_line(&mut start).unwrap();
                let (mut length, mut authorization) = (0, String::new());
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap();
                        } else if name.eq_ignore_ascii_case("authorization") {
                            authorization = value.trim().to_owned();
                        }
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                seen.lock().unwrap().push(authorization);
                let answer = json!({"error":{"message":format!("provider-words: no such thing at 127.0.0.1:{port}"),
                    "type":"fixture_error","code":"fixture_code"}})
                .to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}",
                    answer.len()
                );
            }
        });
        Self {
            port,
            requests,
            stop,
            thread: Some(thread),
        }
    }
    fn received(&self) -> Vec<String> {
        std::mem::take(&mut *self.requests.lock().unwrap())
    }
}
impl Drop for Failing {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// An admin model entry with its own connection, shared or private.
fn admin_model(id: &str, model: &str, base: &str, weight: u64, shared: bool) -> (Value, Value) {
    let connection = format!("{id}-connection");
    let mut info = json!({"id":id,"provider_id":connection});
    if shared {
        info["share"] = json!("shared");
    }
    (
        json!({"model_name":"default","litellm_params":{"model":model,"api_key":ADMIN_KEY,
            "api_base":base,"weight":weight,"use_environment_credentials":false},"model_info":info}),
        json!({"id":connection,"provider":"custom","api_key":ADMIN_KEY,"api_base":base,
            "use_environment_credentials":false}),
    )
}

/// The admin's configuration file with `models` (entry, connection) saved.
fn save_admin_models(root: &Path, models: Vec<(Value, Value)>) {
    configure(root);
    let path = root.join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let (entries, connections): (Vec<_>, Vec<_>) = models.into_iter().unzip();
    config["llm"]["model_list"] = json!(entries);
    config["llm"]["providers"] = json!(connections);
    std::fs::write(&path, config.to_string()).unwrap();
}
fn base(port: u16) -> String {
    format!("http://127.0.0.1:{port}/v1")
}
fn patch_shared(server: &Server, token: &str, id: &str, body: Value) -> Reply {
    call(
        server,
        "PATCH",
        &format!("/api/me/models/shared/{id}"),
        token,
        Some(body),
    )
}

#[test]
fn users_weigh_shared_models_for_themselves_only() {
    let sentinel = Endpoint::start();
    let first = Endpoint::start();
    let second = Endpoint::start();
    let temp = tempfile::tempdir().unwrap();
    save_admin_models(
        temp.path(),
        vec![
            admin_model("s-one", "openai/fake-model", &base(first.port), 1, true),
            admin_model("s-two", "openai/second-model", &base(second.port), 2, true),
        ],
    );
    let server = start(temp.path(), &sentinel);
    let (a_id, a) = user(&server, "Laptop A");
    let (_, b) = user(&server, "Laptop B");
    let view = ok(&server, "GET", "/api/me/models", &a, None);
    assert_eq!(
        view["shared"],
        json!([
            {"id":"s-one","model":"openai/fake-model","provider":"custom","weight":1,"default_weight":1},
            {"id":"s-two","model":"openai/second-model","provider":"custom","weight":2,"default_weight":2}
        ])
    );
    let original = view["revision"].clone();

    // Refusals change nothing.
    let reply = patch_shared(
        &server,
        &a,
        "no-such-model",
        json!({"expected_revision":original,"weight":3}),
    );
    assert_eq!(reason(&reply), (404, json!("settings_entry_missing")));
    let reply = patch_shared(
        &server,
        &a,
        "s-two",
        json!({"expected_revision":"stale","weight":3}),
    );
    assert_eq!(reply.status, 409);
    assert_eq!(
        reply.json()["detail"],
        json!({"code":"stale_revision","current_revision":original})
    );
    // A weight is a whole number from 0 to 100.
    for weight in [json!("3"), json!(-1), json!(1.5), json!(101)] {
        let reply = patch_shared(
            &server,
            &a,
            "s-two",
            json!({"expected_revision":original,"weight":weight}),
        );
        assert_eq!(reason(&reply), (422, json!("invalid_weight")), "{weight}");
        assert_eq!(
            reply.json()["detail"],
            "Routing weight must be a whole number from 0 to 100"
        );
    }
    for body in [
        json!({"expected_revision":original}),
        json!({"weight":3}),
        json!({"expected_revision":original,"weight":3,"share":"shared"}),
        json!({"expected_revision":7,"weight":3}),
        json!([]),
    ] {
        let reply = patch_shared(&server, &a, "s-two", body.clone());
        assert_eq!(
            reason(&reply),
            (422, json!("invalid_settings_request")),
            "{body}: {}",
            reply.text()
        );
    }
    let authorization = format!("Bearer {a}");
    let reply = server.request(
        "PATCH",
        "/api/me/models/shared/s-two",
        &[("Authorization", authorization.as_str())],
        b"{}",
    );
    assert_eq!(reason(&reply), (422, json!("json_required")));
    let reply = server.request(
        "PATCH",
        "/api/me/models/shared/s-two",
        &[
            ("Authorization", authorization.as_str()),
            ("Content-Type", "application/json"),
        ],
        b"{not json",
    );
    assert_eq!(reason(&reply), (422, json!("invalid_json")));
    let reply = patch_shared(
        &server,
        ADMIN,
        "s-two",
        json!({"expected_revision":original,"weight":0}),
    );
    assert_eq!(reason(&reply), (403, json!("user_token_required")));
    assert_eq!(reply.headers["cache-control"], "no-store");
    assert_eq!(revision(&server, "/api/me/models", &a), original);
    let file = temp
        .path()
        .join(format!("home/serve/user-models/{a_id}.json"));
    assert!(!file.exists());

    // 0: off for A only, still listed; A's conversions reach the first only.
    let reply = patch_shared(
        &server,
        &a,
        "s-two",
        json!({"expected_revision":original,"weight":0}),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.headers["cache-control"], "no-store");
    let view = reply.json();
    assert_eq!(view["shared"][1]["weight"], 0);
    assert_eq!(view["shared"][1]["default_weight"], 2);
    assert_ne!(view["revision"], original);
    assert_eq!(view["routable"], true);
    let saved: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!(saved["shared_weights"], json!({"s-two":0}));
    assert_eq!(saved["version"], 1);
    assert_eq!(
        ok(&server, "GET", "/api/capabilities", &a, None)["llm"]["models"],
        json!(["openai/fake-model"])
    );
    assert_eq!(
        ok(&server, "GET", "/api/capabilities", &b, None)["llm"]["models"],
        json!(["openai/fake-model", "openai/second-model"])
    );
    let b_view = ok(&server, "GET", "/api/me/models", &b, None);
    assert_eq!(b_view["shared"][1]["weight"], 2);
    for _ in 0..3 {
        converted(&server, &a);
    }
    assert_eq!(received(&first).len(), 3);
    assert!(received(&second).is_empty());
    // Both off: A's pool is empty, B's is not.
    let revision_a = view["revision"].clone();
    let view = ok(
        &server,
        "PATCH",
        "/api/me/models/shared/s-one",
        &a,
        Some(json!({"expected_revision":revision_a,"weight":0})),
    );
    assert_eq!(view["routable"], false);
    let reply = submit(&server, &a, &[("notes.md", NOTES)], json!({"llm":true}));
    assert_eq!(reason(&reply), (422, json!("llm_unavailable")));
    assert_eq!(
        ok(&server, "GET", "/api/capabilities", &b, None)["llm"]["routable"],
        true
    );
    // The admin's view and weights are its own.
    let settings = ok(&server, "GET", "/api/settings/llm", ADMIN, None);
    let weights: Vec<_> = settings["deployments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["weight"].clone())
        .collect();
    assert_eq!(weights, [json!(1), json!(2)]);

    // An override kept while the model is private comes back with it.
    let share = |id: &str, value: &str| {
        let revision_admin = revision(&server, "/api/settings/llm", ADMIN);
        ok(
            &server,
            "PATCH",
            &format!("/api/settings/llm/deployments/{id}"),
            ADMIN,
            Some(json!({"expected_revision":revision_admin,"share":value})),
        );
    };
    share("s-one", "private");
    let view = ok(&server, "GET", "/api/me/models", &a, None);
    assert_eq!(view["shared"].as_array().unwrap().len(), 1);
    let reply = patch_shared(
        &server,
        &a,
        "s-one",
        json!({"expected_revision":view["revision"],"weight":4}),
    );
    assert_eq!(reason(&reply), (404, json!("settings_entry_missing")));
    share("s-one", "shared");
    let view = ok(&server, "GET", "/api/me/models", &a, None);
    assert_eq!(view["shared"][0]["weight"], 0);

    // The admin's weight, or null, removes the override; with none left the
    // revision is the one before any.
    let view = ok(
        &server,
        "PATCH",
        "/api/me/models/shared/s-one",
        &a,
        Some(json!({"expected_revision":view["revision"],"weight":1})),
    );
    assert_eq!(view["shared"][0]["weight"], 1);
    let view = ok(
        &server,
        "PATCH",
        "/api/me/models/shared/s-two",
        &a,
        Some(json!({"expected_revision":view["revision"],"weight":null})),
    );
    assert_eq!(view["shared"][1]["weight"], 2);
    assert_eq!(view["revision"], original);
    let saved: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert!(saved.get("shared_weights").is_none(), "{saved}");
    // Unchanged: no write, same revision.
    let bytes = std::fs::read(&file).unwrap();
    let view = ok(
        &server,
        "PATCH",
        "/api/me/models/shared/s-two",
        &a,
        Some(json!({"expected_revision":original,"weight":2})),
    );
    assert_eq!(view["revision"], original);
    assert_eq!(std::fs::read(&file).unwrap(), bytes);
    // A heavier weight for one is its own.
    let view = ok(
        &server,
        "PATCH",
        "/api/me/models/shared/s-one",
        &a,
        Some(json!({"expected_revision":original,"weight":9})),
    );
    assert_eq!(view["shared"][0]["weight"], 9);
    assert_eq!(view["shared"][0]["default_weight"], 1);
    assert_eq!(
        ok(&server, "GET", "/api/me/models", &b, None)["shared"][0]["weight"],
        1
    );
    server.stop();
    assert!(received(&sentinel).is_empty());
}

#[test]
fn a_user_tests_a_shared_model_without_learning_where_it_is() {
    let sentinel = Endpoint::start();
    let working = Endpoint::start();
    let (rejected, unknown, broken) = (
        Failing::start(401),
        Failing::start(404),
        Failing::start(500),
    );
    let refused = {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        listener.local_addr().unwrap().port()
    };
    let temp = tempfile::tempdir().unwrap();
    // A shared model without a key, whose provider's key variable is unset.
    let (keyless, _) = admin_model("s-keyless", "deepseek/fixture-chat", "", 1, true);
    let mut keyless = keyless;
    let params = keyless["litellm_params"].as_object_mut().unwrap();
    params.remove("api_key");
    params.remove("api_base");
    params.insert("use_environment_credentials".into(), json!(true));
    keyless["model_info"]
        .as_object_mut()
        .unwrap()
        .remove("provider_id");
    let mut models = vec![
        admin_model("s-ok", "openai/fake-model", &base(working.port), 1, true),
        admin_model("s-401", "openai/fake-model", &base(rejected.port), 1, true),
        admin_model("s-404", "openai/fake-model", &base(unknown.port), 1, true),
        admin_model("s-500", "openai/fake-model", &base(broken.port), 1, true),
        admin_model("s-refused", "openai/fake-model", &base(refused), 1, true),
        admin_model(
            "private",
            "openai/fake-model",
            &base(working.port),
            1,
            false,
        ),
        admin_model("s-off", "openai/fake-model", &base(working.port), 0, true),
    ];
    save_admin_models(temp.path(), std::mem::take(&mut models));
    {
        let path = temp.path().join("config.json");
        let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["llm"]["model_list"]
            .as_array_mut()
            .unwrap()
            .push(keyless);
        std::fs::write(&path, config.to_string()).unwrap();
    }
    let server = start(temp.path(), &sentinel);
    let (_, a) = user(&server, "Laptop A");
    let (_, b) = user(&server, "Laptop B");
    let test =
        |token: &str, body: Value| call(&server, "POST", "/api/me/models/test", token, Some(body));

    // Refused before anything is sent, and not counted.
    for body in [
        json!({"shared_id":"s-ok","model":"openai/fake-model"}),
        json!({"shared_id":"s-ok","deployment_id":"s-ok"}),
        json!({"shared_id":5}),
        json!({"shared_id":"no-such-model"}),
        json!({"shared_id":"private"}),
        json!({"shared_id":"s-off"}),
        json!({"deployment_id":"s-ok"}),
    ] {
        let reply = test(&a, body.clone());
        assert_eq!(
            reason(&reply),
            (422, json!("invalid_settings_request")),
            "{body}: {}",
            reply.text()
        );
    }
    assert_eq!(
        test(&a, json!({"shared_id":"no-such-model"})).json()["detail"],
        "deployment not found"
    );
    let reply = test(ADMIN, json!({"shared_id":"s-ok"}));
    assert_eq!(reason(&reply), (403, json!("user_token_required")));
    assert!(received(&working).is_empty());

    // Six tests: one passes, five fail, each saying only why.
    let failed = "The shared model did not pass its test; ask the admin to check it";
    let mut answers = Vec::new();
    for (id, expected) in [
        ("s-ok", json!({"ok":true,"detail":"Model test passed"})),
        (
            "s-401",
            json!({"ok":false,"reason":"key_rejected","detail":failed}),
        ),
        (
            "s-404",
            json!({"ok":false,"reason":"model_not_found","detail":failed}),
        ),
        (
            "s-500",
            json!({"ok":false,"reason":"provider_error","detail":failed}),
        ),
        (
            "s-refused",
            json!({"ok":false,"reason":"network","detail":failed}),
        ),
        (
            "s-keyless",
            json!({"ok":false,"reason":"no_key","detail":failed}),
        ),
    ] {
        let reply = test(&a, json!({"shared_id":id}));
        assert_eq!(reply.status, 200, "{id}: {}", reply.text());
        assert_eq!(reply.headers["cache-control"], "no-store");
        assert_eq!(reply.json(), expected, "{id}");
        answers.push(reply.text());
    }
    // The admin's key paid, sent where the admin serves each model.
    assert_eq!(received(&working), [format!("Bearer {ADMIN_KEY}")]);
    for endpoint in [&rejected, &unknown, &broken] {
        assert_eq!(endpoint.received(), [format!("Bearer {ADMIN_KEY}")]);
    }
    for answer in &answers {
        for hidden in [
            ADMIN_KEY.to_owned(),
            "127.0.0.1".to_owned(),
            "localhost".to_owned(),
            "provider-words".to_owned(),
            "fixture_code".to_owned(),
            "DEEPSEEK_API_KEY".to_owned(),
            "key_variable".to_owned(),
            working.port.to_string(),
            rejected.port.to_string(),
            unknown.port.to_string(),
            broken.port.to_string(),
            refused.to_string(),
        ] {
            assert!(!answer.contains(&hidden), "{hidden} in {answer}");
        }
    }

    // A seventh within the minute waits, and nothing is sent; B is not held
    // to A's bound.
    let reply = test(&a, json!({"shared_id":"s-ok"}));
    assert_eq!(reason(&reply), (429, json!("provider_busy")));
    let wait: u64 = reply.headers["retry-after"].parse().unwrap();
    assert!((1..=60).contains(&wait), "{wait}");
    assert!(received(&working).is_empty());
    let reply = test(&b, json!({"shared_id":"s-ok"}));
    assert_eq!(
        reply.json(),
        json!({"ok":true,"detail":"Model test passed"})
    );
    assert_eq!(received(&working), [format!("Bearer {ADMIN_KEY}")]);
    server.stop();
    assert!(received(&sentinel).is_empty());
}

fn config_put(server: &Server, token: &str, expected: &Value, text: &str) -> Reply {
    call(
        server,
        "PUT",
        "/api/me/config",
        token,
        Some(json!({"expected_revision":expected,"text":text})),
    )
}
/// `text` (a document) with `change` applied, pretty-printed as the page
/// would send it back; masked values stay where they were.
fn edited(text: &str, change: impl FnOnce(&mut Value)) -> String {
    let mut document: Value = serde_json::from_str(text).unwrap();
    change(&mut document);
    serde_json::to_string_pretty(&document).unwrap() + "\n"
}

#[test]
fn a_users_edit_config_is_its_own_models_file() {
    let sentinel = Endpoint::start();
    let shared = Endpoint::start();
    let temp = tempfile::tempdir().unwrap();
    save_admin_models(
        temp.path(),
        vec![admin_model(
            "s-one",
            "openai/fake-model",
            &base(shared.port),
            1,
            true,
        )],
    );
    let server = start(temp.path(), &sentinel);
    let (a_id, a) = user(&server, "Laptop A");
    let (_, b) = user(&server, "Laptop B");
    let features = &ok(&server, "GET", "/api/capabilities", &a, None)["features"];
    assert_eq!(features["user_config"], true);
    for (method, body) in [
        ("GET", None),
        (
            "PUT",
            Some(json!({"expected_revision":"absent","text":"{}"})),
        ),
    ] {
        let reply = call(&server, method, "/api/me/config", ADMIN, body);
        assert_eq!(reason(&reply), (403, json!("user_token_required")));
        assert_eq!(reply.headers["cache-control"], "no-store");
    }

    // Nothing saved yet: the shared model only, in the fixed layout.
    let reply = call(&server, "GET", "/api/me/config", &a, None);
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.headers["cache-control"], "no-store");
    let document = reply.json();
    assert_eq!(
        document,
        json!({"path":null,"origin":"user","exists":false,
            "revision":revision(&server, "/api/me/models", &a),
            "text":"{\n  \"models\": [],\n  \"providers\": [],\n  \"shared\": [\n    {\n      \"id\": \"s-one\",\n      \"model\": \"openai/fake-model\",\n      \"weight\": 1\n    }\n  ]\n}\n",
            "masked":0,"problem":null,"read_only":false})
    );

    // A's model: its key masked in the model and its connection.
    let reply = add(
        &server,
        &a,
        json!([{"model_name":"default","model":"openai/own-model","api_key":USER_A_KEY}]),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    let document = ok(&server, "GET", "/api/me/config", &a, None);
    assert_eq!(document["exists"], true);
    assert_eq!(document["masked"], 2);
    assert_eq!(
        document["revision"],
        revision(&server, "/api/me/models", &a)
    );
    let text = document["text"].as_str().unwrap().to_owned();
    assert!(
        !text.contains(USER_A_KEY) && text.contains("us…-key"),
        "{text}"
    );
    let file = temp
        .path()
        .join(format!("home/serve/user-models/{a_id}.json"));
    let bytes = std::fs::read(&file).unwrap();

    // Unchanged: nothing written, same revision.
    let reply = config_put(&server, &a, &document["revision"], &text);
    assert_eq!(reply.status, 200, "{}", reply.text());
    let saved = reply.json();
    assert_eq!(saved["restart_required"], false);
    assert_eq!(saved["revision"], document["revision"]);
    assert_eq!(saved["text"], document["text"]);
    assert_eq!(std::fs::read(&file).unwrap(), bytes);

    // Refusals, each before anything is saved.
    let current = document["revision"].clone();
    let refusals: Vec<(String, (u16, &str), Option<&str>)> = vec![
        (
            "{\n  \"models\": [\n}".into(),
            (422, "invalid_config"),
            Some("line 3, column 1: "),
        ),
        ("[]".into(), (422, "invalid_config"), None),
        (
            edited(&text, |d| d["llm"] = json!({})),
            (422, "invalid_config"),
            Some("llm is not part of your models file"),
        ),
        (
            edited(&text, |d| d["models"] = json!({})),
            (422, "invalid_config"),
            Some("models must be a list"),
        ),
        (
            edited(&text, |d| d["shared"][0]["id"] = json!("s-unknown")),
            (422, "invalid_config"),
            Some("s-unknown is not a model the admin shares"),
        ),
        (
            edited(&text, |d| d["shared"][0]["weight"] = json!("heavy")),
            (422, "invalid_weight"),
            Some("shared/0 (s-one): Routing weight must be a whole number from 0 to 100"),
        ),
        (
            edited(&text, |d| d["shared"][0]["weight"] = json!(101)),
            (422, "invalid_weight"),
            Some("shared/0 (s-one): Routing weight"),
        ),
        (
            edited(&text, |d| {
                d["models"][0]["litellm_params"]["weight"] = json!(250)
            }),
            (422, "invalid_weight"),
            Some(
                "models/0 (openai/own-model): Routing weight must be a whole number from 0 to 100",
            ),
        ),
        (
            edited(&text, |d| {
                let mut copy = d["models"][0].clone();
                copy["litellm_params"]["api_key"] = json!(USER_B_KEY);
                d["models"].as_array_mut().unwrap().push(copy);
            }),
            (422, "invalid_config"),
            Some("is used twice"),
        ),
        (
            // A kept mask moved to another model.
            edited(&text, |d| {
                let mut copy = d["models"][0].clone();
                copy["model_info"] = json!({});
                d["models"].as_array_mut().unwrap().push(copy);
            }),
            (422, "invalid_config"),
            Some("/models/1/litellm_params/api_key: a masked value"),
        ),
        (
            edited(&text, |d| {
                d["models"][0]["litellm_params"]["model"] = json!("claude-agent/sonnet")
            }),
            (422, "server_credentials"),
            None,
        ),
        (
            edited(&text, |d| {
                d["models"] =
                    json!([{"model_name":"default","litellm_params":{"model":"openai/no-key"}}]);
                d["providers"] = json!([]);
            }),
            (422, "key_or_address_required"),
            None,
        ),
        (
            edited(&text, |d| {
                d["models"] = json!([{"model_name":"default","litellm_params":{"model":"openai/x",
                    "api_key":USER_A_KEY,"api_base":"http://localhost:9/v1"}}]);
            }),
            (422, "server_address"),
            None,
        ),
        (
            edited(&text, |d| {
                d["models"] = json!([{"model_name":"default","litellm_params":{"model":"openai/x",
                    "api_key":"env:OPENAI_API_KEY"}}]);
            }),
            (422, "invalid_settings_request"),
            None,
        ),
        (
            edited(&text, |d| {
                d["models"] = json!(
                    (0..51)
                        .map(|index| json!({"model_name":"default",
                    "litellm_params":{"model":format!("openai/m{index}"),"api_key":USER_A_KEY}}))
                        .collect::<Vec<_>>()
                );
            }),
            (409, "model_limit"),
            None,
        ),
    ];
    for (sent, expected, detail) in refusals {
        let reply = config_put(&server, &a, &current, &sent);
        assert_eq!(
            reason(&reply),
            (expected.0, json!(expected.1)),
            "{sent}: {}",
            reply.text()
        );
        if let Some(detail) = detail {
            let shown = reply.json()["detail"].as_str().unwrap().to_owned();
            assert!(shown.contains(detail), "{shown}");
        }
    }
    let reply = config_put(&server, &a, &json!("stale"), &text);
    assert_eq!(reply.status, 409);
    assert_eq!(
        reply.json()["detail"],
        json!({"code":"stale_revision","current_revision":current})
    );
    let reply = call(
        &server,
        "PUT",
        "/api/me/config",
        &a,
        Some(json!({"expected_revision":current,"text":text,"extra":1})),
    );
    assert_eq!(reason(&reply), (422, json!("invalid_settings_request")));
    let large = format!("{{\"models\":[],\"pad\":\"{}\"}}", "x".repeat(1024 * 1024));
    let reply = config_put(&server, &a, &current, &large);
    assert_eq!(reason(&reply), (413, json!("config_too_large")));
    assert_eq!(std::fs::read(&file).unwrap(), bytes);

    // Edited in place: the kept key comes back, a weight of its own, a new
    // model gets an id.
    let sent = edited(&text, |d| {
        d["models"][0]["model_name"] = json!("renamed");
        d["shared"][0]["weight"] = json!(3);
        d["shared"][0]["default_weight"] = json!(99);
        d["models"]
            .as_array_mut()
            .unwrap()
            .push(json!({"model_name":"default",
            "litellm_params":{"model":"openai/second-own","api_key":USER_B_KEY,"weight":0}}));
    });
    let reply = config_put(&server, &a, &current, &sent);
    assert_eq!(reply.status, 200, "{}", reply.text());
    let saved = reply.json();
    assert_eq!(saved["restart_required"], false);
    assert_eq!(saved["masked"], 3);
    let view = ok(&server, "GET", "/api/me/models", &a, None);
    assert_eq!(saved["revision"], view["revision"]);
    assert_eq!(view["shared"][0]["weight"], 3);
    assert_eq!(view["shared"][0]["default_weight"], 1);
    assert_eq!(view["deployments"][0]["routing_group"], "renamed");
    let second = view["deployments"][1]["deployment_id"].as_str().unwrap();
    assert!(!second.starts_with("legacy-"), "{second}");
    let stored: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!(stored["models"][0]["litellm_params"]["api_key"], USER_A_KEY);
    assert_eq!(stored["shared_weights"], json!({"s-one":3}));
    for entry in stored["models"].as_array().unwrap() {
        assert_eq!(
            entry["litellm_params"]["use_environment_credentials"],
            false
        );
    }
    // The shared entry left out: the admin's weight again.
    let text = saved["text"].as_str().unwrap().to_owned();
    let sent = edited(&text, |d| d["shared"] = json!([]));
    let saved = config_put(&server, &a, &saved["revision"], &sent).json();
    assert_eq!(
        ok(&server, "GET", "/api/me/models", &a, None)["shared"][0]["weight"],
        1
    );
    assert!(saved["text"].as_str().unwrap().contains("\"weight\": 1"));
    let stored: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert!(stored.get("shared_weights").is_none(), "{stored}");

    // B's document is B's own; the admin's file holds no user's key.
    let other = ok(&server, "GET", "/api/me/config", &b, None);
    assert_eq!(other["exists"], false);
    assert!(
        other["text"]
            .as_str()
            .unwrap()
            .starts_with("{\n  \"models\": [],")
    );
    let config = std::fs::read_to_string(temp.path().join("config.json")).unwrap();
    assert!(!config.contains(USER_A_KEY) && !config.contains(USER_B_KEY));
    server.stop();
    assert!(received(&sentinel).is_empty());
}

#[test]
fn capabilities_name_the_build_for_every_caller() {
    let sentinel = Endpoint::start();
    let temp = tempfile::tempdir().unwrap();
    let server = start(temp.path(), &sentinel);
    let (_, a) = user(&server, "Laptop A");
    for token in [ADMIN, a.as_str()] {
        let capabilities = ok(&server, "GET", "/api/capabilities", token, None);
        let build = &capabilities["build"];
        let mut fields = keys(build);
        fields.sort();
        assert_eq!(fields, ["built_at", "commit", "dirty", "version"]);
        assert_eq!(build["version"], capabilities["version"]);
        assert!(build["dirty"].is_boolean());
        match &build["commit"] {
            Value::Null => {}
            Value::String(commit) => assert!(
                commit.len() >= 4 && commit.bytes().all(|b| b.is_ascii_hexdigit()),
                "{commit}"
            ),
            other => panic!("commit {other}"),
        }
        match &build["built_at"] {
            Value::Null => {}
            Value::String(at) => {
                assert!(at.ends_with('Z'), "{at}");
                chrono::DateTime::parse_from_rfc3339(at).unwrap();
            }
            other => panic!("built_at {other}"),
        }
        assert_eq!(capabilities["features"]["user_config"], true);
    }
    server.stop();
    // Without tokens there is no user and no user's Edit config.
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let capabilities = server.json("/api/capabilities");
    assert_eq!(capabilities["features"]["user_config"], false);
    assert!(capabilities["build"].is_object());
    server.stop();
}

/// A user's provider page saves additions and removals as one change, by
/// the ids of the list its revision names (round 18).
#[test]
fn a_users_batch_adds_and_removes_in_one_change() {
    let sentinel = Endpoint::start();
    let temp = tempfile::tempdir().unwrap();
    let server = start(temp.path(), &sentinel);
    let (_, a) = user(&server, "Laptop A");
    let model = |name: &str| {
        json!({"model_name":"default","model":format!("openai/{name}"),"api_key":USER_A_KEY,
            "api_base":"https://garden.example/v1","weight":2})
    };
    let reply = add(
        &server,
        &a,
        json!([model("garden-lite"), model("garden-flash")]),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    let view = reply.json();
    assert_eq!(view["batch_remove"], true);
    let card = ok(&server, "GET", "/api/me/models/providers", &a, None)["providers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|card| card["kind"] == "configured")
        .cloned()
        .unwrap();
    let lite = view["deployments"][0]["deployment_id"].clone();
    let batch = |body: Value| {
        call(
            &server,
            "POST",
            "/api/me/models/deployments/batch",
            &a,
            Some(body),
        )
    };
    // Refused as a whole: an unknown id, or a weight above 100.
    for (body, expected) in [
        (
            json!({"expected_revision":view["revision"],"remove":[lite,"no-such-id"]}),
            (404, json!("settings_entry_missing")),
        ),
        (
            json!({"expected_revision":view["revision"],"remove":[lite],
                "deployments":[{"model_name":"default","model":"openai/garden-new",
                "credential_provider_id":card["provider_id"],"weight":150}]}),
            (422, json!("invalid_weight")),
        ),
    ] {
        assert_eq!(reason(&batch(body.clone())), expected, "{body}");
        assert_eq!(revision(&server, "/api/me/models", &a), view["revision"]);
    }
    let reply = batch(json!({"expected_revision":view["revision"],"remove":[lite],
        "deployments":[{"model_name":"default","model":"openai/garden-new",
        "credential_provider_id":card["provider_id"],"weight":3}]}));
    assert_eq!(reply.status, 200, "{}", reply.text());
    let rows: Vec<_> = reply.json()["deployments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| (row["model"].clone(), row["weight"].clone()))
        .collect();
    assert_eq!(
        rows,
        [
            (json!("openai/garden-flash"), json!(2)),
            (json!("openai/garden-new"), json!(3))
        ]
    );
}
