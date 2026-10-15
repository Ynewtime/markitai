//! The composer's defaults (`/api/convert-defaults`): the admin saves the
//! server's into `config.json` with the revision rule, a user token keeps its
//! own over them, and Cloudflare is never a default.
use super::*;

const ADMIN: &str = "admin-test-token-0002";

fn call(server: &Server, method: &str, token: &str, body: Option<Value>) -> Reply {
    let authorization = format!("Bearer {token}");
    let mut headers = vec![("Authorization", authorization.as_str())];
    let body = body.map(|value| value.to_string()).unwrap_or_default();
    if !body.is_empty() {
        headers.push(("Content-Type", "application/json"));
    }
    server.request(method, "/api/convert-defaults", &headers, body.as_bytes())
}

#[test]
fn defaults_are_saved_by_the_admin_and_kept_per_user() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start_with(
        temp.path(),
        &[("MARKITAI_SERVE_TOKEN", std::ffi::OsStr::new(ADMIN))],
    );
    let view = call(&server, "GET", ADMIN, None);
    assert_eq!(view.status, 200, "{}", view.text());
    assert_eq!(
        view.headers.get("cache-control").map(String::as_str),
        Some("no-store")
    );
    let view = view.json();
    assert_eq!(view["server"]["llm"], false);
    assert_eq!(view["server"]["strategy"], "static");
    assert_eq!(
        (view["scope"].as_str(), view["can_save"].as_bool()),
        (Some("server"), Some(true))
    );
    let capabilities = server
        .request(
            "GET",
            "/api/capabilities",
            &[("Authorization", &format!("Bearer {ADMIN}"))],
            b"",
        )
        .json();
    assert_eq!(capabilities["features"]["convert_defaults"], true);

    // Cloudflare cannot be a default, nor can an unknown option.
    for defaults in [
        json!({"strategy":"cloudflare"}),
        json!({"backend":"cloudflare"}),
        json!({"llm":1}),
    ] {
        let reply = call(
            &server,
            "PUT",
            ADMIN,
            Some(json!({"expected_revision":view["revision"],"defaults":defaults})),
        );
        assert_eq!(
            (reply.status, reply.json()["reason"].clone()),
            (422, json!("invalid_convert_defaults"))
        );
    }
    // The admin's save goes into config.json at the CLI's keys.
    let saved = call(
        &server,
        "PUT",
        ADMIN,
        Some(
            json!({"expected_revision":view["revision"],"defaults":{"ocr":true,"profile":"rag","no_compress":false}}),
        ),
    );
    assert_eq!(saved.status, 200, "{}", saved.text());
    let saved = saved.json();
    assert_eq!(
        (
            saved["defaults"]["ocr"].clone(),
            saved["defaults"]["profile"].clone()
        ),
        (json!(true), json!("rag"))
    );
    let file: Value =
        serde_json::from_str(&std::fs::read_to_string(temp.path().join("config.json")).unwrap())
            .unwrap();
    assert_eq!(
        (
            file["ocr"]["enabled"].clone(),
            file["output"]["profile"].clone(),
            file["image"]["compress"].clone()
        ),
        (json!(true), json!("rag"), json!(true))
    );
    // A save against the old revision is refused.
    let stale = call(
        &server,
        "PUT",
        ADMIN,
        Some(json!({"expected_revision":view["revision"],"defaults":{"ocr":false}})),
    );
    assert_eq!(stale.status, 409);
    assert_eq!(
        stale.json()["detail"]["current_revision"],
        saved["revision"]
    );

    // A user's own defaults over the server's; null follows the server's again.
    let created = server.request(
        "POST",
        "/api/access-tokens",
        &[
            ("Authorization", &format!("Bearer {ADMIN}")),
            ("Content-Type", "application/json"),
        ],
        br#"{"label":"Tablet"}"#,
    );
    assert_eq!(created.status, 201, "{}", created.text());
    let token = created.json()["token"].as_str().unwrap().to_owned();
    let mine = call(&server, "GET", &token, None).json();
    assert_eq!(
        (
            mine["scope"].as_str(),
            mine["own"].clone(),
            mine["defaults"]["ocr"].clone()
        ),
        (Some("user"), Value::Null, json!(true))
    );
    let own = call(&server, "PUT", &token, Some(json!({"expected_revision":mine["revision"],"defaults":{"ocr":false,"strategy":"playwright"}}))).json();
    assert_eq!(
        (
            own["origin"].as_str(),
            own["defaults"]["ocr"].clone(),
            own["defaults"]["profile"].clone()
        ),
        (Some("user"), json!(false), json!("rag"))
    );
    assert_eq!(
        call(&server, "GET", ADMIN, None).json()["defaults"]["ocr"],
        true,
        "the admin's stay"
    );
    let back = call(
        &server,
        "PUT",
        &token,
        Some(json!({"expected_revision":own["revision"],"defaults":null})),
    )
    .json();
    assert_eq!(
        (
            back["origin"].as_str(),
            back["own"].clone(),
            back["defaults"]["strategy"].clone()
        ),
        (Some("server"), Value::Null, json!("static"))
    );
    // The admin cannot remove the server's defaults.
    let refused = call(
        &server,
        "PUT",
        ADMIN,
        Some(json!({"expected_revision":saved["revision"],"defaults":null})),
    );
    assert_eq!(refused.status, 422);
}
