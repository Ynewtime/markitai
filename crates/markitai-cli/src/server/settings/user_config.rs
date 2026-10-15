//! `/api/me/config`: a user token's Edit config, over its own models file
//! only. The text is a document the server writes from the user's lists
//! (`models`, `providers`, and `shared`: the models the admin shares, with
//! the user's weight for each), with every secret masked by the admin
//! editor's own rules; a saved text goes through the user rules the Models
//! routes apply, and its revision is the Models page's.
use super::{
    ApiError, ApiResult, editor, failure, identity, model, pool,
    store::Store,
    users::{self, Lists, Shared, Users},
};
use serde_json::{Map, Value, json};
use std::collections::HashSet;

fn refused(detail: impl Into<String>) -> ApiError {
    ApiError::new(422, "invalid_config", detail.into())
}

/// The document of `lists`: pretty JSON, two-space indent, LF line breaks,
/// a final line break, keys `models`, `providers`, `shared` in that order.
fn document(lists: &Lists, shared: &[Shared]) -> String {
    let shared: Vec<Value> = shared
        .iter()
        .map(|model| {
            let mut entry = Map::new();
            entry.insert("id".into(), json!(model.id));
            entry.insert("model".into(), model.model.clone());
            entry.insert(
                "weight".into(),
                json!(users::effective_weight(lists, model)),
            );
            Value::Object(entry)
        })
        .collect();
    let mut root = Map::new();
    root.insert("models".into(), json!(lists.models));
    root.insert("providers".into(), json!(lists.providers));
    root.insert("shared".into(), json!(shared));
    let mut text = serde_json::to_string_pretty(&Value::Object(root)).expect("JSON values");
    text.push('\n');
    text
}

/// The Edit config view of `lists`.
fn view(lists: &Lists, shared: &[Shared], exists: bool) -> Value {
    let (text, masked) = editor::mask(&document(lists, shared));
    json!({"path":null,"origin":"user","exists":exists,"revision":users::revision(lists),
        "text":text,"masked":masked,"problem":null,"read_only":false})
}

/// `GET /api/me/config` for the user `id`.
pub(in crate::server) fn config_view(store: &Store, users: &Users, id: &str) -> Value {
    view(
        &users.lists(id),
        &users::shared_models(store),
        users.exists(id),
    )
}

/// A list of objects under `key` of `document`, empty when absent.
fn objects(document: &Map<String, Value>, key: &str) -> ApiResult<Vec<Value>> {
    match document.get(key) {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => {
            if let Some(index) = items.iter().position(|item| !item.is_object()) {
                return Err(refused(format!("{key}/{index} must be an object")));
            }
            Ok(items.clone())
        }
        Some(_) => Err(refused(format!("{key} must be a list"))),
    }
}

/// The lists `document` describes, beside the `current` ones and the models
/// the admin shares now (`shared`).
fn parse(document: &Value, current: &Lists, shared: &[Shared]) -> ApiResult<Lists> {
    let fields = document
        .as_object()
        .ok_or_else(|| refused("the configuration must be a JSON object"))?;
    if let Some(key) = fields
        .keys()
        .find(|key| !matches!(key.as_str(), "models" | "providers" | "shared"))
    {
        return Err(refused(format!(
            "{key} is not part of your models file; it holds only models, providers and shared"
        )));
    }
    let models = objects(fields, "models")?;
    for (index, entry) in models.iter().enumerate() {
        if !entry["litellm_params"]["model"].is_string() {
            return Err(refused(format!(
                "models/{index} needs litellm_params.model, the model ID"
            )));
        }
    }
    model::check_weights(&models, &current.models, |index, entry| {
        format!(
            "models/{index} ({})",
            entry["litellm_params"]["model"]
                .as_str()
                .unwrap_or_default()
        )
    })?;
    let providers = objects(fields, "providers")?;
    for (index, connection) in providers.iter().enumerate() {
        if !connection["id"].is_string() || !connection["provider"].is_string() {
            return Err(refused(format!(
                "providers/{index} needs an id and a provider"
            )));
        }
    }
    for (index, entry) in models.iter().enumerate() {
        if let Some(link) = entry.pointer("/model_info/provider_id")
            && !link.is_null()
            && !providers.iter().any(|p| p["id"] == *link)
        {
            return Err(refused(format!(
                "models/{index}: its model_info.provider_id is not one of the providers"
            )));
        }
    }
    let mut next = Lists {
        models,
        providers,
        // Weights for models no longer shared stay, for when they are again.
        shared_weights: current
            .shared_weights
            .iter()
            .filter(|(id, _)| !shared.iter().any(|model| model.id == **id))
            .map(|(id, weight)| (id.clone(), *weight))
            .collect(),
    };
    let mut seen = HashSet::new();
    for (index, entry) in objects(fields, "shared")?.iter().enumerate() {
        let entry = entry.as_object().unwrap();
        if let Some(key) = entry
            .keys()
            .find(|key| !matches!(key.as_str(), "id" | "weight" | "model" | "default_weight"))
        {
            return Err(refused(format!(
                "shared/{index}: {key} is not a field of a shared model; it takes id and weight"
            )));
        }
        let id = entry
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| refused(format!("shared/{index} needs its id")))?;
        let model = shared
            .iter()
            .find(|model| model.id == id)
            .ok_or_else(|| refused(format!("{id} is not a model the admin shares")))?;
        if !seen.insert(id.to_owned()) {
            return Err(refused(format!("shared lists {id} twice")));
        }
        let unchanged = entry.get("weight").and_then(Value::as_u64)
            == Some(users::effective_weight(current, model));
        let weight = match entry.get("weight") {
            None | Some(Value::Null) => None,
            // A weight saved before the limit may stay as it is.
            Some(value) if unchanged => value.as_u64(),
            Some(value) => Some(model::weight(value).map_err(|_| {
                model::invalid_weight(format!("shared/{index} ({id}): {}", model::WEIGHT_RULE))
            })?),
        };
        // An unchanged weight keeps what the user had; another one follows
        // the rule of the Models page.
        if unchanged {
            if let Some(kept) = current.shared_weights.get(id) {
                next.shared_weights.insert(id.to_owned(), *kept);
            }
        } else {
            users::set_weight(&mut next, model, weight);
        }
    }
    Ok(next)
}

/// Refuse two entries of one list with the same id.
fn unique<'a>(list: &str, ids: impl Iterator<Item = Option<&'a str>>) -> ApiResult<()> {
    let mut seen = HashSet::new();
    for id in ids.flatten() {
        if !seen.insert(id) {
            return Err(refused(format!("{list}: the id {id} is used twice")));
        }
    }
    Ok(())
}

/// `PUT /api/me/config` for the user `id`: the lists `text` describes,
/// saved if the user's lists are still at `expected`.
pub(in crate::server) fn replace_config(
    store: &Store,
    users: &Users,
    id: &str,
    alive: &dyn Fn() -> bool,
    expected: &str,
    text: &str,
) -> ApiResult<Value> {
    let shared = users::shared_models(store);
    let lists = users.write(id, alive, |lists| {
        let current = users::revision(lists);
        if expected != current {
            return Err(ApiError::structured(
                409,
                "stale_revision",
                json!({"code":"stale_revision","current_revision":current}),
            ));
        }
        serde_json::from_str::<Value>(text).map_err(|error| refused(editor::position(&error)))?;
        let saved = editor::unmask(text, Some(&document(lists, &shared))).map_err(refused)?;
        let parsed: Value = serde_json::from_str(&saved).map_err(|_| failure())?;
        let mut next = parse(&parsed, lists, &shared)?;
        for entry in &next.models {
            users::precheck(entry)?;
        }
        users::check(&mut next)?;
        identity::backfill(&mut next.models);
        unique(
            "models",
            next.models
                .iter()
                .map(|entry| entry.pointer("/model_info/id").and_then(Value::as_str)),
        )?;
        unique(
            "providers",
            next.providers.iter().map(|entry| entry["id"].as_str()),
        )?;
        store
            .with_data(|admin| pool::build(&admin.cfg, &next, admin))
            .map_err(refused)?;
        *lists = next;
        Ok(())
    })?;
    let mut answer = view(&lists, &shared, users.exists(id));
    answer["restart_required"] = json!(false);
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared() -> Vec<Shared> {
        vec![
            Shared {
                id: "s1".into(),
                model: json!("gemini/fixture-model"),
                provider: "gemini".into(),
                weight: 2,
            },
            Shared {
                id: "s2".into(),
                model: json!("openai/other-fixture"),
                provider: "custom".into(),
                weight: 1,
            },
        ]
    }

    #[test]
    fn the_document_has_a_fixed_layout_and_masks_every_key() {
        let lists = Lists {
            models: vec![
                json!({"model_name":"default","litellm_params":{"model":"openai/own",
                "api_key":"user-a-fixture-key-123","use_environment_credentials":false},"model_info":{"id":"m1"}}),
            ],
            providers: vec![],
            shared_weights: [("s2".into(), 0), ("gone".into(), 4)].into(),
        };
        let text = document(&lists, &shared());
        assert!(text.starts_with("{\n  \"models\": [\n"), "{text}");
        assert!(text.ends_with("}\n") && !text.contains('\r'));
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            value["shared"],
            json!([{"id":"s1","model":"gemini/fixture-model","weight":2},
                {"id":"s2","model":"openai/other-fixture","weight":0}])
        );
        let keys: Vec<_> = value.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys, ["models", "providers", "shared"]);
        let shown = view(&lists, &shared(), true);
        assert_eq!(shown["masked"], 1);
        assert!(!shown["text"].as_str().unwrap().contains("fixture-key"));
        assert_eq!(shown["revision"], users::revision(&lists));
    }

    #[test]
    fn a_documents_shared_list_sets_overrides_by_the_models_rule() {
        let current = Lists {
            shared_weights: [("s1".into(), 5), ("gone".into(), 4)].into(),
            ..Default::default()
        };
        // s1 unchanged (kept), s2 set to 0; `gone` stays.
        let next = parse(
            &json!({"shared":[{"id":"s1","weight":5,"model":"ignored"},{"id":"s2","weight":0,"default_weight":9}]}),
            &current,
            &shared(),
        )
        .unwrap();
        assert_eq!(
            next.shared_weights,
            [("s1".into(), 5), ("s2".into(), 0), ("gone".into(), 4)].into()
        );
        // The admin's weight, null or absence: no override.
        let next = parse(
            &json!({"shared":[{"id":"s1","weight":2},{"id":"s2","weight":null}]}),
            &current,
            &shared(),
        )
        .unwrap();
        assert_eq!(next.shared_weights, [("gone".into(), 4)].into());
        let next = parse(&json!({}), &current, &shared()).unwrap();
        assert_eq!(next.shared_weights, [("gone".into(), 4)].into());
        for (document, detail) in [
            (json!([]), "the configuration must be a JSON object"),
            (
                json!({"llm":{}}),
                "llm is not part of your models file; it holds only models, providers and shared",
            ),
            (json!({"models":{}}), "models must be a list"),
            (json!({"providers":[1]}), "providers/0 must be an object"),
            (
                json!({"models":[{"litellm_params":{}}]}),
                "models/0 needs litellm_params.model, the model ID",
            ),
            (
                json!({"shared":[{"id":"other","weight":1}]}),
                "other is not a model the admin shares",
            ),
            (
                json!({"shared":[{"id":"s1"},{"id":"s1"}]}),
                "shared lists s1 twice",
            ),
            (
                json!({"shared":[{"id":"s1","api_key":"x"}]}),
                "shared/0: api_key is not a field of a shared model; it takes id and weight",
            ),
        ] {
            let error = parse(&document, &current, &shared()).unwrap_err();
            assert_eq!(
                (error.reason, error.detail.as_str()),
                ("invalid_config", detail)
            );
        }
        // A weight is a whole number from 0 to 100; one the user already
        // had (from before the limit) may stay as it is.
        for (document, detail) in [
            (
                json!({"shared":[{"id":"s1","weight":-1}]}),
                "shared/0 (s1): Routing weight must be a whole number from 0 to 100",
            ),
            (
                json!({"shared":[{"id":"s2","weight":101}]}),
                "shared/0 (s2): Routing weight must be a whole number from 0 to 100",
            ),
            (
                json!({"models":[{"litellm_params":{"model":"openai/own","weight":1.5}}]}),
                "models/0 (openai/own): Routing weight must be a whole number from 0 to 100",
            ),
        ] {
            let error = parse(&document, &current, &shared()).unwrap_err();
            assert_eq!(
                (error.reason, error.detail.as_str()),
                ("invalid_weight", detail)
            );
        }
        let old = Lists {
            models: vec![
                json!({"model_name":"default","litellm_params":{"model":"openai/own","weight":300},"model_info":{"id":"m1"}}),
            ],
            shared_weights: [("s1".into(), 500)].into(),
            ..Default::default()
        };
        let next = parse(
            &json!({"models":old.models,"shared":[{"id":"s1","weight":500}]}),
            &old,
            &shared(),
        )
        .unwrap();
        assert_eq!(next.shared_weights, [("s1".into(), 500)].into());
        assert_eq!(next.models[0]["litellm_params"]["weight"], 300);
    }
}
