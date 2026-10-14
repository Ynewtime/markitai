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
            "shared":[],"limit":50})
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
        json!([{"id":shared_id,"model":"openai/fake-model","provider":"custom","weight":1}])
    );
    assert_eq!(
        keys(&view["shared"][0]),
        ["id", "model", "provider", "weight"]
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
