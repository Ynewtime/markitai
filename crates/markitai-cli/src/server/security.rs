use super::{
    State,
    types::{ApiError, ApiResult, MAX_REQUEST},
};
use axum::{
    extract::{ConnectInfo, Request, State as ExtractState},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

#[derive(Clone, Copy)]
pub(super) struct Trusted(pub bool);

fn equal_secret(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for (index, byte) in right.iter().enumerate() {
        difference |= usize::from(left.get(index).copied().unwrap_or(0) ^ byte);
    }
    difference == 0
}
fn hostname(value: &str) -> Option<String> {
    let url = url::Url::parse(&format!("http://{value}/")).ok()?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    Some(
        url.host_str()?
            .trim_matches(['[', ']'])
            .to_ascii_lowercase(),
    )
}
fn origin_url(value: &str) -> Option<url::Url> {
    let url = url::Url::parse(value).ok()?;
    (["http", "https"].contains(&url.scheme())
        && url.host_str().is_some()
        && url.username().is_empty()
        && url.password().is_none()
        && url.path() == "/"
        && url.query().is_none()
        && url.fragment().is_none())
    .then_some(url)
}

fn same_authority(origin: &url::Url, host: &str) -> bool {
    // Host carries no scheme. For a TLS-terminating proxy, use the browser's
    // scheme to interpret an omitted default port; never trust forwarding headers.
    origin_url(&format!("{}://{host}/", origin.scheme())).is_some_and(|target| {
        origin.host_str() == target.host_str()
            && origin.port_or_known_default() == target.port_or_known_default()
    })
}
pub(super) fn allowed_host(value: &str) -> ApiResult<String> {
    let host = hostname(value)
        .ok_or_else(|| ApiError::new(400, "invalid_allowed_host", "invalid allowed host"))?;
    if host.is_empty() {
        return Err(ApiError::new(
            400,
            "invalid_allowed_host",
            "invalid allowed host",
        ));
    }
    Ok(host)
}

pub(super) async fn guard(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
    next: Next,
) -> Response {
    let settings = settings_path(request.uri().path());
    let mut response = guard_inner(state, request, next).await;
    if settings {
        response.headers_mut().insert(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store"),
        );
    }
    response
}

fn settings_path(path: &str) -> bool {
    path == "/api/settings/llm" || path.starts_with("/api/settings/llm/")
}

async fn guard_inner(state: Arc<State>, mut request: Request, next: Next) -> Response {
    let peer = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| info.0.ip());
    let forwarded = ["forwarded", "x-forwarded-for", "x-real-ip"]
        .iter()
        .any(|name| request.headers().contains_key(*name));
    let loopback = !forwarded && peer.is_some_and(|ip| ip.to_canonical().is_loopback());
    let header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
        .map(|(_, value)| value.trim().to_owned());
    // Owned, so nothing borrowed from the request lives across `next.run`.
    let query_string = request.uri().query().map(str::to_owned);
    let parameter = |name: &str| {
        query_string.as_deref().and_then(|s| {
            url::form_urlencoded::parse(s.as_bytes())
                .filter(|(key, _)| key == name)
                .map(|(_, v)| v.into_owned())
                .last()
        })
    };
    let query = parameter("token");
    let mut authenticated = state.token.as_ref().is_some_and(|token| {
        [header.as_deref(), query.as_deref()]
            .into_iter()
            .flatten()
            .any(|value| equal_secret(value.as_bytes(), token.as_bytes()))
    });
    let api = request.uri().path().starts_with("/api/");
    // A download ticket stands in for the token on exactly one GET of the path
    // it was issued for (see `tickets`); any other use spends it for nothing.
    if api
        && state.token.is_some()
        && !authenticated
        && let Some(ticket) = parameter("ticket")
    {
        let redeemed =
            state
                .tickets
                .redeem(&ticket, request.uri().path(), std::time::Instant::now());
        if redeemed && request.method() == axum::http::Method::GET {
            authenticated = true;
        } else {
            return ApiError::new(
                401,
                "ticket_invalid",
                "download ticket is invalid, expired, already used or issued for another path",
            )
            .into_response();
        }
    }
    if api && state.token.is_some() && !authenticated {
        return ApiError::new(
            401,
            "token_required",
            "authentication required: send the startup token in Authorization: Bearer or ?token=",
        )
        .into_response();
    }
    let authority = request.headers().get("host").and_then(|v| v.to_str().ok());
    let host = authority.and_then(hostname);
    if host.as_deref().is_none_or(|host| {
        !(host == "localhost"
            || host.parse::<IpAddr>().is_ok()
            || state.allowed_hosts.contains(host))
    }) {
        return ApiError::new(
            400,
            "host_not_allowed",
            "host is not allowed; use localhost, an IP address, or --allowed-host",
        )
        .into_response();
    }
    if !matches!(request.method().as_str(), "GET" | "HEAD" | "OPTIONS")
        && let Some(origin) = request.headers().get("origin")
    {
        let origin = origin.to_str().ok().and_then(origin_url);
        if origin.as_ref().is_none_or(|origin| {
            let name = origin
                .host_str()
                .unwrap_or_default()
                .trim_matches(['[', ']']);
            !(state.allowed_hosts.contains(name)
                || authority.is_some_and(|host| same_authority(origin, host)))
        }) {
            return ApiError::new(
                403,
                "origin_not_allowed",
                "cross-site request from this origin is not allowed",
            )
            .into_response();
        }
    }
    let trusted = authenticated || (state.token.is_none() && loopback);
    if settings_path(request.uri().path()) && !trusted {
        return ApiError::new(
            403,
            "settings_forbidden",
            "settings access requires loopback or token authentication",
        )
        .into_response();
    }
    if request
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .is_some_and(|size| size > MAX_REQUEST as u64)
    {
        return ApiError::new(413, "request_too_large", "request exceeds upload limit")
            .into_response();
    }
    request.extensions_mut().insert(Trusted(trusted));
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_authorities_and_secret_comparison_are_not_prefix_checks() {
        assert_eq!(hostname("[::1]:3600").as_deref(), Some("::1"));
        assert_eq!(hostname("LOCALHOST:3600").as_deref(), Some("localhost"));
        for bad in [
            "evil.com/path",
            "user@localhost",
            "localhost?query=1",
            "localhost#fragment",
        ] {
            assert!(hostname(bad).is_none(), "{bad}");
        }
        assert!(equal_secret(b"token", b"token"));
        assert!(!equal_secret(b"token-more", b"token"));
        assert!(!equal_secret(b"toke", b"token"));
    }
}

#[cfg(test)]
mod router_tests {
    use super::*;
    use axum::{
        Json, Router,
        body::{Body, to_bytes},
        extract::Extension,
        http::Request,
        middleware,
        routing::{get, post},
    };
    use serde_json::{Value, json};
    use std::{
        collections::{HashMap, HashSet},
        sync::{Mutex, atomic::AtomicBool},
    };
    use tokio::sync::{Semaphore, watch};
    use tower::ServiceExt;

    fn app(root: &std::path::Path, token: Option<&str>) -> Router {
        super::super::store::private_dir(root).unwrap();
        let (shutdown, _) = watch::channel(false);
        let state = Arc::new(State {
            settings: super::super::settings::Store::new(
                markitai_core::config::normalize(&json!({"llm":{"enabled":false}})).unwrap(),
                super::super::SettingsSource {
                    path: root.join("config.json"),
                    origin: "default".into(),
                    overrides: None,
                },
            )
            .unwrap(),
            root: root.into(),
            jobs: Mutex::new(HashMap::new()),
            file_slots: Arc::new(Semaphore::new(1)),
            url_slots: Arc::new(Semaphore::new(1)),
            closing: AtomicBool::new(false),
            persistence_failed: AtomicBool::new(false),
            shutdown,
            tasks: Mutex::new(Vec::new()),
            token: token.map(str::to_owned),
            allowed_hosts: HashSet::from(["trusted.example".into()]),
            tickets: Default::default(),
        });
        Router::new()
            .route(
                "/api/probe",
                get(|Extension(trust): Extension<Trusted>| async move {
                    Json(json!({"trusted":trust.0}))
                })
                .post(|| async { Json(json!({"ok":true})) }),
            )
            .route(
                "/api/settings/llm",
                get(|| async { Json(json!({"ok":true})) })
                    .post(|| async { Json(json!({"ok":true})) }),
            )
            .route("/api/jobs", post(super::super::http::create))
            .route(
                "/api/jobs/{job_id}/archive",
                get(|Extension(trust): Extension<Trusted>| async move {
                    Json(json!({"trusted":trust.0}))
                })
                .post(|| async { Json(json!({"ok":true})) }),
            )
            .route("/api/download-tickets", post(super::super::tickets::issue))
            .layer(middleware::from_fn_with_state(state.clone(), guard))
            .with_state(state)
    }
    fn request(
        method: &str,
        path: &str,
        peer: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> Request<Body> {
        let mut builder = Request::builder()
            .method(method)
            .uri(path)
            .header("host", "127.0.0.1:3600");
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let mut request = builder.body(Body::from(body.to_owned())).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(peer.parse::<SocketAddr>().unwrap()));
        request
    }
    async fn checked(router: Router, request: Request<Body>, status: u16) -> Value {
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status().as_u16(), status);
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap()).unwrap()
    }

    #[tokio::test]
    async fn settings_always_require_trust_and_never_cache_success_or_errors() {
        let temp = tempfile::tempdir().unwrap();
        let authenticated = app(temp.path(), Some("settings-test-token"));
        let open = app(temp.path(), None);
        for (router, method, path, peer, headers, expected) in [
            (
                &authenticated,
                "GET",
                "/api/settings/llm",
                "203.0.113.40:1234",
                vec![],
                401,
            ),
            (
                &open,
                "GET",
                "/api/settings/llm",
                "203.0.113.40:1234",
                vec![],
                403,
            ),
            (
                &authenticated,
                "GET",
                "/api/settings/llm",
                "203.0.113.40:1234",
                vec![("authorization", "Bearer settings-test-token")],
                200,
            ),
            (
                &open,
                "GET",
                "/api/settings/llm",
                "127.0.0.1:1234",
                vec![],
                200,
            ),
            (
                &open,
                "POST",
                "/api/settings/llm",
                "127.0.0.1:1234",
                vec![("origin", "https://evil.invalid")],
                403,
            ),
            (
                &open,
                "GET",
                "/api/settings/llm/missing",
                "127.0.0.1:1234",
                vec![],
                404,
            ),
            (
                &open,
                "PUT",
                "/api/settings/llm",
                "127.0.0.1:1234",
                vec![],
                405,
            ),
            (
                &open,
                "POST",
                "/api/settings/llm",
                "127.0.0.1:1234",
                vec![("content-length", "99999999999")],
                413,
            ),
        ] {
            let response = router
                .clone()
                .oneshot(request(method, path, peer, &headers, ""))
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected, "{method} {path}");
            assert_eq!(response.headers()["cache-control"], "no-store");
        }
        let mut invalid_host = request("GET", "/api/settings/llm", "127.0.0.1:1234", &[], "");
        invalid_host
            .headers_mut()
            .insert("host", "evil.invalid".parse().unwrap());
        let response = open.oneshot(invalid_host).await.unwrap();
        assert_eq!(response.status().as_u16(), 400);
        assert_eq!(response.headers()["cache-control"], "no-store");
        assert!(!settings_path("/api/settings/llm-other"));
    }

    #[tokio::test]
    async fn remote_authentication_uses_peer_and_complete_secret_not_forwarded_headers() {
        let temp = tempfile::tempdir().unwrap();
        let router = app(temp.path(), Some("private-test-token"));
        let peer = "203.0.113.40:4321";
        for headers in [
            vec![],
            vec![("x-forwarded-for", "127.0.0.1")],
            vec![("authorization", "Bearer private-test-token-extra")],
        ] {
            assert_eq!(
                checked(
                    router.clone(),
                    request("GET", "/api/probe", peer, &headers, ""),
                    401
                )
                .await["code"],
                "unauthorized"
            );
        }
        assert_eq!(
            checked(
                router.clone(),
                request(
                    "GET",
                    "/api/probe",
                    peer,
                    &[("authorization", "Bearer private-test-token")],
                    ""
                ),
                200
            )
            .await["trusted"],
            true
        );
        assert_eq!(
            checked(
                router.clone(),
                request("GET", "/api/probe?token=private-test-token", peer, &[], ""),
                200
            )
            .await["trusted"],
            true
        );
        assert_eq!(
            checked(
                router.clone(),
                request("GET", "/api/probe", "127.0.0.1:4321", &[], ""),
                401
            )
            .await["reason"],
            "token_required"
        );
        assert_eq!(
            checked(
                router.clone(),
                request(
                    "POST",
                    "/api/probe",
                    peer,
                    &[
                        ("authorization", "Bearer private-test-token"),
                        ("origin", "https://evil.invalid")
                    ],
                    ""
                ),
                403
            )
            .await["code"],
            "forbidden"
        );
        checked(
            router,
            request(
                "POST",
                "/api/probe",
                peer,
                &[
                    ("authorization", "Bearer private-test-token"),
                    ("origin", "https://trusted.example"),
                ],
                "",
            ),
            200,
        )
        .await;
    }

    #[tokio::test]
    async fn local_and_proxied_clients_authenticate_even_on_a_loopback_listener() {
        let temp = tempfile::tempdir().unwrap();
        let router = app(temp.path(), Some("private-test-token"));
        for peer in ["127.0.0.1:4321", "[::1]:4321", "[::ffff:127.0.0.1]:4321"] {
            for forwarding in [
                None,
                Some("forwarded"),
                Some("x-forwarded-for"),
                Some("x-real-ip"),
            ] {
                let mut headers = Vec::new();
                if let Some(name) = forwarding {
                    headers.push((name, "127.0.0.1"));
                }
                for path in ["/api/probe", "/api/settings/llm"] {
                    let error = checked(
                        router.clone(),
                        request("GET", path, peer, &headers, ""),
                        401,
                    )
                    .await;
                    assert_eq!(error["reason"], "token_required");
                }
                headers.push(("authorization", "Bearer private-test-token"));
                let value = checked(
                    router.clone(),
                    request("GET", "/api/probe", peer, &headers, ""),
                    200,
                )
                .await;
                assert_eq!(value["trusted"], true);
            }
        }
    }

    #[tokio::test]
    async fn no_auth_trust_is_canonical_loopback_without_forwarding_headers() {
        let temp = tempfile::tempdir().unwrap();
        let router = app(temp.path(), None);
        let peer = "[::ffff:127.0.0.1]:4321";
        assert_eq!(
            checked(
                router.clone(),
                request("GET", "/api/probe", peer, &[], ""),
                200
            )
            .await["trusted"],
            true
        );
        for name in ["forwarded", "x-forwarded-for", "x-real-ip"] {
            let headers = [(name, "127.0.0.1")];
            assert_eq!(
                checked(
                    router.clone(),
                    request("GET", "/api/probe", peer, &headers, ""),
                    200
                )
                .await["trusted"],
                false
            );
            checked(
                router.clone(),
                request("GET", "/api/settings/llm", peer, &headers, ""),
                403,
            )
            .await;
        }
    }

    #[tokio::test]
    async fn mutation_origins_are_complete_authorities_not_arbitrary_localhost_ports() {
        let temp = tempfile::tempdir().unwrap();
        let router = app(temp.path(), None);
        for origin in [
            "http://127.0.0.1:3000",
            "http://localhost:3600",
            "http://[::1]:3600",
            "null",
            "file://localhost/",
            "http://user@127.0.0.1:3600",
            "http://127.0.0.1:3600/other",
            "http://127.0.0.1:3600/?x=1",
            "http://127.0.0.1:3600/#fragment",
            "https://user@trusted.example",
        ] {
            let error = checked(
                router.clone(),
                request(
                    "POST",
                    "/api/probe",
                    "127.0.0.1:4321",
                    &[("origin", origin)],
                    "",
                ),
                403,
            )
            .await;
            assert_eq!(error["reason"], "origin_not_allowed", "{origin}");
        }
        for (origin, host) in [
            ("http://127.0.0.1:3600", "127.0.0.1:3600"),
            ("http://localhost", "localhost:80"),
            ("http://[::1]:3600", "[::1]:3600"),
            ("https://trusted.example", "trusted.example"),
        ] {
            let mut req = request(
                "POST",
                "/api/probe",
                "127.0.0.1:4321",
                &[("origin", origin)],
                "",
            );
            req.headers_mut().insert("host", host.parse().unwrap());
            checked(router.clone(), req, 200).await;
        }
    }

    #[tokio::test]
    async fn a_download_ticket_admits_one_remote_get_of_its_path_only() {
        let temp = tempfile::tempdir().unwrap();
        let router = app(temp.path(), Some("private-test-token"));
        let peer = "203.0.113.40:4321";
        let json = [("content-type", "application/json")];
        let bearer = [
            ("content-type", "application/json"),
            ("authorization", "Bearer private-test-token"),
        ];
        let archive = "/api/jobs/0123456789ab/archive";
        let body = format!(r#"{{"path":"{archive}"}}"#);
        // Issuing a ticket needs the token itself.
        checked(
            router.clone(),
            request("POST", "/api/download-tickets", peer, &json, &body),
            401,
        )
        .await;
        let issue = |router: Router| {
            let body = body.clone();
            async move {
                checked(
                    router,
                    request("POST", "/api/download-tickets", peer, &bearer, &body),
                    201,
                )
                .await
            }
        };
        let issued = issue(router.clone()).await;
        let url = issued["url"].as_str().unwrap().to_owned();
        let ticket = issued["ticket"].as_str().unwrap().to_owned();
        assert_eq!(url, format!("{archive}?ticket={ticket}"));
        assert_eq!(issued["expires_in"], 60);
        assert!(!url.contains("private-test-token"));
        assert_eq!(
            checked(router.clone(), request("GET", &url, peer, &[], ""), 200).await["trusted"],
            true
        );
        let reused = checked(router.clone(), request("GET", &url, peer, &[], ""), 401).await;
        assert_eq!(reused["reason"], "ticket_invalid");

        // Shown to another path, or with another method, a ticket is spent.
        let ticket = issue(router.clone()).await["ticket"]
            .as_str()
            .unwrap()
            .to_owned();
        let elsewhere = format!("/api/probe?ticket={ticket}");
        assert_eq!(
            checked(
                router.clone(),
                request("GET", &elsewhere, peer, &[], ""),
                401
            )
            .await["reason"],
            "ticket_invalid"
        );
        let spent = format!("{archive}?ticket={ticket}");
        checked(router.clone(), request("GET", &spent, peer, &[], ""), 401).await;
        let url = issue(router.clone()).await["url"]
            .as_str()
            .unwrap()
            .to_owned();
        checked(router.clone(), request("POST", &url, peer, &[], ""), 401).await;
        checked(router.clone(), request("GET", &url, peer, &[], ""), 401).await;

        // Loopback also redeems a ticket exactly once.
        let url = issue(router.clone()).await["url"]
            .as_str()
            .unwrap()
            .to_owned();
        checked(
            router.clone(),
            request("GET", &url, "127.0.0.1:4321", &[], ""),
            200,
        )
        .await;
        checked(router.clone(), request("GET", &url, peer, &[], ""), 401).await;

        // Only download routes can be named.
        for path in [
            "/api/settings/llm",
            "/api/settings/llm/providers/x/credentials",
            "/api/jobs/0123456789ab/events",
            "/api/history",
        ] {
            let value = checked(
                router.clone(),
                request(
                    "POST",
                    "/api/download-tickets",
                    peer,
                    &bearer,
                    &format!(r#"{{"path":"{path}"}}"#),
                ),
                422,
            )
            .await;
            assert_eq!(value["reason"], "invalid_ticket_path", "{path}");
        }
        for body in [
            "",
            "[]",
            r#"{"path":1}"#,
            r#"{"path":"/api/history/archive","x":1}"#,
        ] {
            checked(
                router.clone(),
                request("POST", "/api/download-tickets", peer, &bearer, body),
                422,
            )
            .await;
        }
        // A path needs a few bytes: a body far past one is refused unread.
        let large = format!(
            r#"{{"path":"/api/history/archive","x":"{}"}}"#,
            "a".repeat(1 << 20)
        );
        let refused = checked(
            router.clone(),
            request("POST", "/api/download-tickets", peer, &bearer, &large),
            413,
        )
        .await;
        assert_eq!(refused["reason"], "request_too_large");
    }

    #[tokio::test]
    async fn no_auth_does_not_grant_remote_url_fetch_trust_and_rolls_back_stage() {
        let temp = tempfile::tempdir().unwrap();
        let router = app(temp.path(), None);
        let peer = "203.0.113.40:4321";
        assert_eq!(
            checked(
                router.clone(),
                request("GET", "/api/probe", peer, &[], ""),
                200
            )
            .await["trusted"],
            false
        );
        let body = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("urls", "[\"http://127.0.0.1/private\"]")
            .finish();
        let result = checked(
            router,
            request(
                "POST",
                "/api/jobs",
                peer,
                &[("content-type", "application/x-www-form-urlencoded")],
                &body,
            ),
            403,
        )
        .await;
        assert!(
            result["detail"]
                .as_str()
                .unwrap()
                .contains("requires loopback or token")
        );
        assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 0);
    }
}
