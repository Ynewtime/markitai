//! `GET /api/openapi.json`: an OpenAPI 3.1 description of the REST API.
//!
//! The document is written by hand (`openapi.json`) and tested against the
//! route table every module registers through `api_routes!`, so a route
//! cannot be added, removed or renamed without the document following. The
//! served copy carries this build's version.
use axum::{
    http::{HeaderValue, header},
    response::{IntoResponse, Response},
};
use serde_json::Value;
use std::sync::OnceLock;

const DOCUMENT: &str = include_str!("openapi.json");
/// `info.version` in the file; replaced in place so the served text keeps the
/// file's order and layout.
const VERSION_PLACEHOLDER: &str = "\"set from the build when served\"";

fn document_bytes() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        let version = Value::from(markitai_core::VERSION).to_string();
        DOCUMENT
            .replacen(VERSION_PLACEHOLDER, &version, 1)
            .into_bytes()
    })
}

pub(super) async fn document() -> Response {
    let mut response = document_bytes().into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    const METHODS: [&str; 5] = ["get", "post", "put", "patch", "delete"];

    fn parsed() -> Value {
        serde_json::from_str(DOCUMENT).unwrap()
    }

    /// axum's `{*rest}` catch-all is an ordinary `{rest}` parameter in OpenAPI.
    fn openapi_path(route: &str) -> String {
        route.replace("{*", "{")
    }

    #[test]
    fn every_registered_route_is_documented_and_nothing_else() {
        let document = parsed();
        let documented: BTreeSet<(String, String)> = document["paths"]
            .as_object()
            .unwrap()
            .iter()
            .flat_map(|(path, item)| {
                item.as_object()
                    .unwrap()
                    .keys()
                    .filter(|key| METHODS.contains(&key.as_str()))
                    .map(move |method| (method.clone(), path.clone()))
            })
            .collect();
        let routes = super::super::api_table();
        let registered: BTreeSet<(String, String)> = routes
            .iter()
            .map(|(method, path)| ((*method).to_owned(), openapi_path(path)))
            .collect();
        assert_eq!(
            registered.len(),
            routes.len(),
            "a route is registered twice"
        );
        let missing: Vec<_> = registered.difference(&documented).collect();
        let stale: Vec<_> = documented.difference(&registered).collect();
        assert!(
            missing.is_empty() && stale.is_empty(),
            "undocumented: {missing:?}; documented but not served: {stale:?}"
        );
    }

    #[test]
    fn the_document_is_well_formed_openapi_3_1() {
        let document = parsed();
        assert_eq!(document["openapi"], "3.1.0");
        assert_eq!(DOCUMENT.matches(VERSION_PLACEHOLDER).count(), 1);
        assert_eq!(
            format!("\"{}\"", document["info"]["version"].as_str().unwrap()),
            VERSION_PLACEHOLDER
        );
        assert!(document["info"]["title"].is_string());
        let mut ids = HashSet::new();
        for (path, item) in document["paths"].as_object().unwrap() {
            assert!(path.starts_with("/api/"), "{path}");
            let parameters: BTreeSet<&str> = path
                .split('/')
                .filter_map(|segment| segment.strip_prefix('{')?.strip_suffix('}'))
                .collect();
            for (method, operation) in item.as_object().unwrap() {
                assert!(METHODS.contains(&method.as_str()), "{path}: {method}");
                let id = operation["operationId"].as_str().unwrap_or_default();
                assert!(!id.is_empty() && ids.insert(id), "{path} {method}: {id}");
                assert!(operation["summary"].is_string(), "{id}");
                let responses = operation["responses"].as_object().unwrap();
                assert!(!responses.is_empty(), "{id}");
                for (status, response) in responses {
                    assert!(
                        status.len() == 3 && status.parse::<u16>().is_ok(),
                        "{id}: {status}"
                    );
                    let response = match response["$ref"].as_str() {
                        Some(target) => document.pointer(&target[1..]).unwrap(),
                        None => response,
                    };
                    assert!(response["description"].is_string(), "{id}: {status}");
                }
                // Path parameters are declared exactly.
                let declared: BTreeSet<&str> = operation["parameters"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .chain(item["parameters"].as_array().into_iter().flatten())
                    .map(|parameter| match parameter["$ref"].as_str() {
                        Some(target) => document.pointer(&target[1..]).unwrap(),
                        None => parameter,
                    })
                    .filter(|parameter| parameter["in"] == "path")
                    .map(|parameter| {
                        assert_eq!(parameter["required"], true, "{id}");
                        parameter["name"].as_str().unwrap()
                    })
                    .collect();
                assert_eq!(declared, parameters, "{id}");
            }
        }
        // Every reference resolves.
        fn references<'a>(value: &'a Value, found: &mut Vec<&'a str>) {
            match value {
                Value::Object(map) => {
                    if let Some(Value::String(target)) = map.get("$ref") {
                        found.push(target);
                    }
                    map.values().for_each(|value| references(value, found));
                }
                Value::Array(items) => items.iter().for_each(|value| references(value, found)),
                _ => {}
            }
        }
        let mut found = Vec::new();
        references(&document, &mut found);
        assert!(!found.is_empty());
        for target in found {
            let pointer = target.strip_prefix('#').unwrap();
            assert!(document.pointer(pointer).is_some(), "{target}");
        }
    }

    #[test]
    fn the_error_schema_lists_the_reasons_the_service_writes() {
        // Every `reason` literal in the service sources appears in the
        // document's error schema, so clients can rely on that list.
        let document = parsed();
        let listed: HashSet<&str> = document
            .pointer("/components/schemas/Error/properties/reason/examples")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        let sources = [
            include_str!("http.rs"),
            include_str!("files.rs"),
            include_str!("jobs.rs"),
            include_str!("rerun.rs"),
            include_str!("security.rs"),
            include_str!("store.rs"),
            include_str!("tickets.rs"),
            include_str!("types.rs"),
            include_str!("providers.rs"),
            include_str!("settings.rs"),
            include_str!("settings/handlers.rs"),
            include_str!("settings/identity.rs"),
            include_str!("settings/model.rs"),
            include_str!("settings/store.rs"),
            include_str!("settings/views.rs"),
        ];
        // `ApiError::new(422, "reason", …)` and its `structured`/`Self::` forms.
        let mut written = HashSet::new();
        for source in sources {
            for pattern in [
                "ApiError::new(",
                "ApiError::structured(",
                "Self::new(",
                "Self::structured(",
            ] {
                for (index, _) in source.match_indices(pattern) {
                    let rest = source[index + pattern.len()..].trim_start();
                    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
                    if digits == 0 {
                        continue;
                    }
                    let Some(rest) = rest[digits..].trim_start().strip_prefix(',') else {
                        continue;
                    };
                    let Some(rest) = rest.trim_start().strip_prefix('"') else {
                        continue;
                    };
                    if let Some(end) = rest.find('"') {
                        written.insert(rest[..end].to_owned());
                    }
                }
            }
        }
        // Built from a status variable in `ApiError::multipart`.
        written.extend([
            "request_too_large".to_owned(),
            "invalid_multipart".to_owned(),
        ]);
        assert!(written.len() > 20, "{written:?}");
        let missing: Vec<_> = written
            .iter()
            .filter(|reason| !listed.contains(reason.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "reasons missing from openapi.json: {missing:?}"
        );
    }

    #[tokio::test]
    async fn the_served_document_carries_this_builds_version() {
        let response = document().await;
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let served: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(served["info"]["version"], markitai_core::VERSION);
        assert_eq!(served["paths"], parsed()["paths"]);
    }
}
