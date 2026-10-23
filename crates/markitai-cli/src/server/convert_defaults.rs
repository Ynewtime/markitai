//! `/api/convert-defaults`: the conversion options the workbench's composer
//! starts from. The server's are its configuration (`config.json` over the
//! built-in defaults); a user token may keep its own over them. The admin
//! saves the server's into `config.json` (the same keys the CLI reads); a user
//! saves its own into its file under `serve/user-models/`. Cloudflare is never
//! a default: it needs the caller's confirmation in each request.
use super::{
    State,
    security::Principal,
    types::{ApiError, ApiResult, FORMATS, REMOTE_STRATEGIES, ocr_languages},
};
use axum::{
    Json, Router,
    extract::{State as ExtractState, rejection::JsonRejection},
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;

api_routes! {
    get "/api/convert-defaults" => view;
    put "/api/convert-defaults" => replace;
}

pub(super) fn routes() -> Router<Arc<State>> {
    api_routes().layer(axum::extract::DefaultBodyLimit::max(64 * 1024))
}

/// One option a caller may make a default.
enum Kind {
    Flag,
    /// A flag stored inverted (`no_compress` is `image.compress: false`).
    Inverted,
    /// One of these words; `null` allowed when `nullable`.
    Word(&'static [&'static str], bool),
    /// An `ocr.lang` code; a save checks it against the offered ones
    /// (`types::ocr_languages`).
    Language,
}

/// The options, their configuration keys (both backends' `config.json`), and their kinds.
const OPTIONS: [(&str, &str, Kind); 15] = [
    ("llm", "/llm/enabled", Kind::Flag),
    ("ocr", "/ocr/enabled", Kind::Flag),
    ("alt", "/image/alt_enabled", Kind::Flag),
    ("desc", "/image/desc_enabled", Kind::Flag),
    ("screenshot", "/screenshot/enabled", Kind::Flag),
    ("screenshot_only", "/screenshot/screenshot_only", Kind::Flag),
    ("pure", "/llm/pure", Kind::Flag),
    ("no_cache", "/cache/no_cache", Kind::Flag),
    ("no_compress", "/image/compress", Kind::Inverted),
    (
        "profile",
        "/output/profile",
        Kind::Word(&["rag", "obsidian", "okf"], true),
    ),
    (
        "strategy",
        "/fetch/strategy",
        Kind::Word(&["auto", "static", "playwright", "defuddle", "jina"], false),
    ),
    // v1.20
    ("format", "/output/format", Kind::Word(&FORMATS, false)),
    ("ocr_lang", "/ocr/lang", Kind::Language),
    ("ocr_all_pages", "/ocr/per_page_routing", Kind::Inverted),
    ("page_markers", "/output/page_markers", Kind::Flag),
];

fn invalid(detail: impl Into<String>) -> ApiError {
    ApiError::new(422, "invalid_convert_defaults", detail)
}

/// The defaults a configuration gives (missing or wrong values read as the built-in ones).
pub(super) fn of_config(cfg: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    for (name, pointer, kind) in &OPTIONS {
        let value = cfg.pointer(pointer);
        let shown = match kind {
            Kind::Flag => json!(value.and_then(Value::as_bool).unwrap_or(false)),
            Kind::Inverted => json!(!value.and_then(Value::as_bool).unwrap_or(true)),
            Kind::Word(words, nullable) => match value.and_then(Value::as_str) {
                Some(word) if words.contains(&word) => json!(word),
                // A server set to Cloudflare starts its composer on auto.
                _ if *nullable => Value::Null,
                _ => json!(words[0]),
            },
            Kind::Language => json!(value.and_then(Value::as_str).unwrap_or("en")),
        };
        out.insert((*name).to_owned(), shown);
    }
    out
}

/// A caller's `defaults` object: only known options, each of its kind.
pub(super) fn checked(value: &Value) -> ApiResult<Map<String, Value>> {
    let map = value
        .as_object()
        .ok_or_else(|| invalid("defaults must be an object"))?;
    for (key, entry) in map {
        let Some((_, _, kind)) = OPTIONS.iter().find(|(name, _, _)| name == key) else {
            let names: Vec<&str> = OPTIONS.iter().map(|(name, _, _)| *name).collect();
            let shown: String = key.chars().take(48).collect();
            return Err(invalid(format!(
                "'{shown}' cannot be a default; these can: {}",
                names.join(", ")
            )));
        };
        let fits = match kind {
            Kind::Flag | Kind::Inverted => entry.is_boolean(),
            Kind::Word(words, nullable) => {
                (entry.is_null() && *nullable)
                    || entry.as_str().is_some_and(|word| words.contains(&word))
            }
            Kind::Language => entry.as_str().is_some_and(|code| {
                !code.is_empty()
                    && code.len() <= 48
                    && code
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
            }),
        };
        if !fits {
            return Err(invalid(match kind {
                Kind::Word(words, _) => {
                    format!("default '{key}' must be one of: {}", words.join(", "))
                }
                Kind::Language => format!("default '{key}' must be an OCR language code"),
                _ => format!("default '{key}' must be true or false"),
            }));
        }
    }
    Ok(map.clone())
}

/// The built-in defaults: what a configuration without any of these keys gives.
fn builtin() -> Map<String, Value> {
    of_config(&markitai_core::config::normalize(&json!({})).unwrap_or_else(|_| json!({})))
}

/// Write `defaults` into the configuration file's JSON: a key the file already
/// has is set, a value the built-in default already gives is not added, and an
/// `output` section left empty by removing the profile goes too.
pub(super) fn write_into_file(raw: &mut Value, defaults: &Map<String, Value>) {
    let builtin = builtin();
    let wanted: Map<String, Value> = defaults
        .iter()
        .filter(|(name, value)| {
            let pointer = OPTIONS
                .iter()
                .find(|(option, _, _)| option == name)
                .map(|(_, p, _)| *p);
            pointer.is_some_and(|p| raw.pointer(p).is_some()) || builtin.get(*name) != Some(value)
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    let removing = wanted.get("profile").is_some_and(Value::is_null);
    write_into(raw, &wanted);
    if removing && raw["output"].as_object().is_some_and(Map::is_empty) {
        raw.as_object_mut().map(|fields| fields.remove("output"));
    }
}

/// Write `defaults` into a configuration (the file's JSON or the running one).
pub(super) fn write_into(target: &mut Value, defaults: &Map<String, Value>) {
    for (name, pointer, kind) in &OPTIONS {
        let Some(value) = defaults.get(*name) else {
            continue;
        };
        let stored = match kind {
            Kind::Inverted => json!(!value.as_bool().unwrap_or(false)),
            _ => value.clone(),
        };
        let mut parts = pointer.trim_start_matches('/').split('/').peekable();
        let mut node = &mut *target;
        while let Some(part) = parts.next() {
            if !node.is_object() {
                *node = json!({});
            }
            let fields = node.as_object_mut().expect("an object");
            if parts.peek().is_none() {
                if stored.is_null() {
                    fields.remove(part);
                } else {
                    fields.insert(part.to_owned(), stored.clone());
                }
                break;
            }
            node = fields.entry(part.to_owned()).or_insert_with(|| json!({}));
        }
    }
}

/// `"sha256:" + hex` over the defaults a caller saves against.
fn revision(value: &Value) -> String {
    let text = serde_json::to_string(value).unwrap_or_default();
    format!(
        "sha256:{}",
        markitai_core::hex(Sha256::digest(text.as_bytes()))
    )
}

/// What a caller sees: the effective defaults, the server's, its own (a user's), where
/// they come from, whether it may save, and the revision a save names.
fn answer(
    server: &Map<String, Value>,
    own: Option<&Map<String, Value>>,
    principal: &Principal,
) -> Value {
    let (scope, can_save, rev) = match principal {
        Principal::Admin => ("server", true, revision(&json!(server))),
        Principal::User { .. } => ("user", true, revision(&json!(own))),
        Principal::Anonymous => ("server", false, revision(&json!(server))),
    };
    // A user's conversions read URLs locally (a remote strategy runs as
    // auto for its jobs), so a remote strategy is not what they start from.
    let user = principal.user().is_some();
    let local = |options: &Map<String, Value>| {
        let mut options = options.clone();
        if user
            && options
                .get("strategy")
                .and_then(Value::as_str)
                .is_some_and(|strategy| REMOTE_STRATEGIES.contains(&strategy))
        {
            options.insert("strategy".into(), json!("auto"));
        }
        options
    };
    let server = &local(server);
    let own = own.map(local);
    let own = own.as_ref();
    let mut effective = server.clone();
    if let Some(own) = own {
        effective.extend(own.iter().map(|(key, value)| (key.clone(), value.clone())));
    }
    json!({
        "defaults": effective,
        "server": server,
        "own": own,
        "origin": if own.is_some() { "user" } else { "server" },
        "scope": scope,
        "can_save": can_save,
        "revision": rev,
    })
}

async fn view(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
) -> ApiResult<Json<Value>> {
    let server = of_config(&state.settings.snapshot());
    let own = match principal.user() {
        Some(id) => state.users.convert_defaults(id),
        None => None,
    };
    Ok(Json(answer(&server, own.as_ref(), &principal)))
}

fn stale(current: String) -> ApiError {
    ApiError::structured(
        409,
        "stale_revision",
        json!({"code":"stale_revision","current_revision":current}),
    )
}

/// `PUT /api/convert-defaults` `{expected_revision, defaults}`: the admin's go into
/// `config.json` (only the options sent); a user's replace its own (`null`: none,
/// the server's apply again).
async fn replace(
    ExtractState(state): ExtractState<Arc<State>>,
    principal: Principal,
    body: Result<Json<Value>, JsonRejection>,
) -> ApiResult<Json<Value>> {
    let shape =
        || invalid("send a JSON body {\"expected_revision\": string, \"defaults\": object}");
    let Json(body) = body.map_err(|rejection| {
        if rejection.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::new(413, "request_too_large", "request exceeds 64 KiB")
        } else {
            shape()
        }
    })?;
    let fields = body
        .as_object()
        .filter(|fields| {
            fields.len() == 2
                && fields.contains_key("expected_revision")
                && fields.contains_key("defaults")
        })
        .ok_or_else(shape)?;
    let expected = fields["expected_revision"]
        .as_str()
        .ok_or_else(shape)?
        .to_owned();
    let wanted = match &fields["defaults"] {
        Value::Null => None,
        value => Some(checked(value)?),
    };
    if let Some(code) = wanted
        .as_ref()
        .and_then(|wanted| wanted.get("ocr_lang"))
        .and_then(Value::as_str)
    {
        let offered = ocr_languages(&state.settings.snapshot());
        if !offered.iter().any(|offer| offer == code) {
            return Err(invalid(format!(
                "default 'ocr_lang' must be one of: {}",
                offered.join(", ")
            )));
        }
    }
    match principal.clone() {
        Principal::Anonymous => Err(ApiError::new(
            403,
            "settings_forbidden",
            "only the server's administrator or a user token may save defaults",
        )),
        Principal::Admin => {
            let Some(wanted) = wanted else {
                return Err(invalid(
                    "the server's defaults cannot be removed; send the options to use",
                ));
            };
            let task = state.clone();
            let cfg = crate::task::blocking(move || {
                task.settings.edit_defaults(
                    |running, raw| {
                        let current = revision(&json!(of_config(running)));
                        if current != expected {
                            return Err(stale(current));
                        }
                        write_into_file(raw, &wanted);
                        Ok(())
                    },
                    |running| write_into(running, &wanted),
                )
            })
            .await
            .map_err(|_| super::settings::io_failure())??;
            Ok(Json(answer(&of_config(&cfg), None, &principal)))
        }
        Principal::User { id, .. } => {
            if wanted.as_ref().is_some_and(|wanted| {
                wanted
                    .get("strategy")
                    .and_then(Value::as_str)
                    .is_some_and(|strategy| REMOTE_STRATEGIES.contains(&strategy))
            }) {
                return Err(invalid(
                    "a user token's conversions never use remote services; the default strategy must be auto, static or playwright",
                ));
            }
            let task = state.clone();
            let user = id.clone();
            let own = crate::task::blocking(move || {
                let alive = || task.access.exists(&user);
                task.users.set_convert_defaults(&user, &alive, |current| {
                    let now = revision(&json!(current));
                    if now != expected {
                        return Err(stale(now));
                    }
                    Ok(wanted)
                })
            })
            .await
            .map_err(|_| super::settings::io_failure())??;
            let server = of_config(&state.settings.snapshot());
            Ok(Json(answer(&server, own.as_ref(), &principal)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_configuration_reads_as_defaults_and_back() {
        let cfg = json!({"llm":{"enabled":true,"pure":false},"ocr":{"enabled":false},
            "image":{"alt_enabled":true,"desc_enabled":false,"compress":false},
            "screenshot":{"enabled":true,"screenshot_only":false},"cache":{"no_cache":true},
            "output":{"profile":"rag"},"fetch":{"strategy":"cloudflare"}});
        let defaults = of_config(&cfg);
        assert_eq!(defaults["llm"], true);
        assert_eq!(defaults["no_compress"], true);
        assert_eq!(defaults["profile"], "rag");
        // Cloudflare needs a confirmation per request: never a default.
        assert_eq!(defaults["strategy"], "auto");
        let mut raw = json!({"llm":{"model_list":[]}});
        write_into(
            &mut raw,
            &checked(&json!({"llm":false,"no_compress":false,"profile":null,"strategy":"jina"}))
                .unwrap(),
        );
        assert_eq!(
            raw,
            json!({"llm":{"model_list":[],"enabled":false},"image":{"compress":true},"output":{},"fetch":{"strategy":"jina"}})
        );
        // In the file: only keys it has or values the built-in defaults do not give; no empty section.
        let mut file =
            json!({"llm":{"model_list":[]},"output":{"profile":"rag"},"cache":{"no_cache":true}});
        write_into_file(
            &mut file,
            &checked(
                &json!({"llm":true,"ocr":false,"profile":null,"no_cache":false,"strategy":"auto"}),
            )
            .unwrap(),
        );
        assert_eq!(
            file,
            json!({"llm":{"model_list":[],"enabled":true},"cache":{"no_cache":false}})
        );
    }

    #[test]
    fn only_known_options_of_their_kind_can_be_defaults() {
        for bad in [
            json!({"backend":"cloudflare"}),
            json!({"strategy":"cloudflare"}),
            json!({"llm":"yes"}),
            json!({"profile":"html"}),
            json!({"strategy":null}),
            json!([]),
        ] {
            assert_eq!(
                checked(&bad).unwrap_err().reason,
                "invalid_convert_defaults",
                "{bad}"
            );
        }
        assert!(checked(&json!({"profile":null,"ocr":true})).is_ok());
        for bad in [
            json!({"format":"yaml"}),
            json!({"format":null}),
            json!({"ocr_lang":"../en"}),
            json!({"ocr_lang":null}),
            json!({"ocr_all_pages":"yes"}),
        ] {
            assert!(checked(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn json_output_ocr_language_all_pages_and_page_markers_are_defaults_too() {
        let built_in = builtin();
        assert_eq!(built_in["format"], "markdown");
        assert_eq!(built_in["ocr_lang"], "en");
        assert_eq!(built_in["ocr_all_pages"], false);
        assert_eq!(built_in["page_markers"], false);
        let cfg = json!({"output":{"format":"json","page_markers":true},
            "ocr":{"lang":"ko","per_page_routing":false}});
        let defaults = of_config(&cfg);
        assert_eq!(
            (
                &defaults["format"],
                &defaults["ocr_lang"],
                &defaults["ocr_all_pages"],
                &defaults["page_markers"]
            ),
            (&json!("json"), &json!("ko"), &json!(true), &json!(true))
        );
        let mut file = json!({});
        write_into_file(
            &mut file,
            &checked(
                &json!({"format":"json","ocr_lang":"en","ocr_all_pages":true,"page_markers":false}),
            )
            .unwrap(),
        );
        assert_eq!(
            file,
            json!({"output":{"format":"json"},"ocr":{"per_page_routing":false}})
        );
    }

    #[test]
    fn a_user_starts_from_a_local_strategy() {
        let server = checked(&json!({"strategy":"jina","llm":true})).unwrap();
        let own = checked(&json!({"strategy":"defuddle"})).unwrap();
        let user = Principal::User {
            id: "u".into(),
            label: "u".into(),
        };
        let seen = answer(&server, Some(&own), &user);
        assert_eq!(seen["defaults"]["strategy"], "auto");
        assert_eq!(seen["server"]["strategy"], "auto");
        assert_eq!(seen["own"]["strategy"], "auto");
        assert_eq!(seen["defaults"]["llm"], true);
        // The revision is the stored one's, so a save against it still matches.
        assert_eq!(seen["revision"], revision(&json!(Some(&own))));
        // The admin sees the server's own choice.
        let admin = answer(&server, None, &Principal::Admin);
        assert_eq!(admin["defaults"]["strategy"], "jina");
    }
}
