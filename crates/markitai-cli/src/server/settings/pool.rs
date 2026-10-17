//! The models a user token's conversions route to: its own, then the ones
//! the admin shares, in one pool. Computed from the current lists on every
//! request, so a change to either applies to the next conversion.
use super::{identity, model, store::Data, users, views};
use serde_json::{Value, json};

/// The model of an empty pool. A configuration without models would fall
/// back to the server's environment (`MODEL` and the provider key
/// variables); this one never routes and never reads it.
pub(super) const PLACEHOLDER: &str = "openai/markitai-no-model";

fn placeholder() -> Value {
    json!({"model_name":"default","litellm_params":{"model":PLACEHOLDER,"weight":0,
        "api_key":model::NO_KEY,"use_environment_credentials":false}})
}

pub(super) fn weight(entry: &Value) -> u64 {
    entry["litellm_params"]["weight"].as_u64().unwrap_or(1)
}

/// The admin's saved models that are shared and on (weight above 0), with
/// their index in the saved list, in its order.
pub(super) fn shared(data: &Data) -> Vec<(usize, &Value)> {
    data.models
        .iter()
        .enumerate()
        .filter(|(_, entry)| views::shared(entry) && weight(entry) > 0)
        .collect()
}

/// An entry of the pool: one routing group, so the admin's group names and
/// fallbacks mean nothing here.
fn grouped(entry: &Value) -> Value {
    let mut entry = entry.clone();
    entry["model_name"] = json!("default");
    entry
}

/// The pool's id of a user's connection and of an admin's: two namespaces,
/// so that a connection a user names after the admin's can never stand in
/// for it (and send a shared model's key to the user's address).
fn pooled_id(namespace: &str, id: &str) -> Value {
    json!(format!("{namespace}:{id}"))
}

/// `entry` linked to its connection's pool id, when it links to one.
fn relinked(mut entry: Value, namespace: &str) -> Value {
    if let Some(id) = model::linked(&entry).map(str::to_owned) {
        entry["model_info"]["provider_id"] = pooled_id(namespace, &id);
    }
    entry
}

/// `base` (the admin's runtime configuration) routing to `own` models that
/// are on, then the admin's shared ones with the connections they link to,
/// and nothing else. An empty pool turns model processing off and holds only
/// the placeholder.
pub(super) fn build(base: &Value, own: &users::Lists, admin: &Data) -> Result<Value, String> {
    let mut models: Vec<Value> = Vec::new();
    let mut providers: Vec<Value> = Vec::new();
    for connection in &own.providers {
        let mut connection = connection.clone();
        connection["use_environment_credentials"] = json!(false);
        if let Some(id) = connection["id"].as_str().map(str::to_owned) {
            connection["id"] = pooled_id("user", &id);
        }
        providers.push(connection);
    }
    for entry in &own.models {
        // Saved through the rules; checked again, so that no file can make a
        // user's model reach the server's credentials.
        if weight(entry) == 0 || users::check_entry(entry, &own.providers).is_err() {
            continue;
        }
        let mut entry = relinked(grouped(entry), "user");
        entry["litellm_params"]["use_environment_credentials"] = json!(false);
        models.push(entry);
    }
    for (index, saved) in shared(admin) {
        // The user's own weight for this shared model, if it set one; 0
        // leaves it out of this user's pool only.
        let effective = own
            .shared_weights
            .get(&identity::id(saved, index))
            .copied()
            .unwrap_or_else(|| weight(saved));
        if effective == 0 {
            continue;
        }
        let mut entry = grouped(saved);
        entry["litellm_params"]["weight"] = json!(effective);
        // A failure a user sees names a shared model, never where the admin
        // serves it.
        if !entry["model_info"].is_object() {
            entry["model_info"] = json!({});
        }
        entry["model_info"]["hidden_endpoint"] = json!(true);
        models.push(relinked(entry, "admin"));
        if let Some(id) = model::linked(saved)
            && !providers.iter().any(|p| p["id"] == pooled_id("admin", id))
            && let Some(connection) = admin.providers.iter().find(|p| p["id"] == id)
        {
            let mut connection = connection.clone();
            connection["id"] = pooled_id("admin", id);
            providers.push(connection);
        }
    }
    if models.is_empty() {
        return Ok(empty(base));
    }
    let mut cfg = base.clone();
    cfg["llm"]["model_list"] = json!(models);
    cfg["llm"]["providers"] = json!(providers);
    cfg["llm"]["router_settings"]["fallbacks"] = json!([]);
    markitai_core::config::normalize(&cfg).map_err(|error| error.to_string())
}

/// `base` with an empty pool: model processing and image descriptions off,
/// and only the placeholder model. Not normalized again: `base` already is,
/// and nothing here can make it invalid.
pub(super) fn empty(base: &Value) -> Value {
    let mut cfg = base.clone();
    cfg["llm"]["enabled"] = json!(false);
    cfg["image"]["alt_enabled"] = json!(false);
    cfg["image"]["desc_enabled"] = json!(false);
    cfg["llm"]["model_list"] = json!([placeholder()]);
    cfg["llm"]["providers"] = json!([]);
    cfg["llm"]["router_settings"]["fallbacks"] = json!([]);
    cfg
}

/// The model ids a pool routes to; none for an empty one.
pub(super) fn models(pool: &Value) -> Vec<String> {
    pool["llm"]["model_list"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry["litellm_params"]["model"].as_str())
        .filter(|model| *model != PLACEHOLDER)
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn admin(models: Vec<Value>, providers: Vec<Value>) -> Data {
        Data {
            cfg: Arc::new(Value::Null),
            revision: super::super::identity::revision(&models, &providers),
            models,
            providers,
            detected: Vec::new(),
        }
    }
    fn base() -> Value {
        markitai_core::config::normalize(&json!({"llm":{"enabled":true,
            "model_list":[{"model_name":"fast","litellm_params":{"model":"openai/admin-only","api_key":"admin-fixture-key"}}],
            "router_settings":{"fallbacks":[{"fast":["slow"]}]}},
            "image":{"alt_enabled":true,"desc_enabled":true}}))
        .unwrap()
    }

    #[test]
    fn an_empty_pool_is_sealed_and_never_routes() {
        let pool = build(&base(), &users::Lists::default(), &admin(vec![], vec![])).unwrap();
        assert_eq!(pool["llm"]["enabled"], false);
        assert_eq!(pool["image"]["alt_enabled"], false);
        assert_eq!(pool["image"]["desc_enabled"], false);
        assert_eq!(
            pool["llm"]["model_list"],
            json!([{"model_name":"default","litellm_params":{"model":PLACEHOLDER,"weight":0,
                "api_key":model::NO_KEY,"use_environment_credentials":false}}])
        );
        assert_eq!(pool["llm"]["router_settings"]["fallbacks"], json!([]));
        assert!(models(&pool).is_empty());
        assert!(!markitai_core::llm_capabilities(&pool).routable);
        // A private model, and a shared one turned off, are not in it either.
        let off = admin(
            vec![
                json!({"model_name":"a","litellm_params":{"model":"openai/private","api_key":"admin-fixture-key"}}),
                json!({"model_name":"b","litellm_params":{"model":"openai/off","api_key":"admin-fixture-key","weight":0},"model_info":{"share":"shared"}}),
                json!({"model_name":"c","litellm_params":{"model":"openai/other","api_key":"admin-fixture-key"},"model_info":{"share":"public"}}),
            ],
            vec![],
        );
        let pool = build(&base(), &users::Lists::default(), &off).unwrap();
        assert!(models(&pool).is_empty());
        assert!(shared(&off).is_empty());
    }

    #[test]
    fn own_models_come_first_then_shared_ones_in_one_default_group() {
        let own = users::Lists {
            models: vec![
                json!({"model_name":"mine","litellm_params":{"model":"openai/own","weight":2,"use_environment_credentials":true},"model_info":{"id":"m1","provider_id":"p-own"}}),
                json!({"model_name":"mine","litellm_params":{"model":"openai/own-off","api_key":"user-a-fixture-key","weight":0}}),
            ],
            providers: vec![
                json!({"id":"p-own","provider":"openai","api_key":"user-a-fixture-key","api_base":"https://models.example/v1","use_environment_credentials":true}),
            ],
            ..Default::default()
        };
        let data = admin(
            vec![
                json!({"model_name":"fast","litellm_params":{"model":"openai/shared","weight":3},"model_info":{"id":"s1","share":"shared","provider_id":"p-admin"}}),
                json!({"model_name":"slow","litellm_params":{"model":"openai/private","api_key":"admin-fixture-key"},"model_info":{"provider_id":"p-private"}}),
            ],
            vec![
                json!({"id":"p-admin","provider":"openai","api_key":"admin-fixture-key"}),
                json!({"id":"p-private","provider":"openai","api_key":"admin-other-key"}),
            ],
        );
        let pool = build(&base(), &own, &data).unwrap();
        assert_eq!(models(&pool), ["openai/own", "openai/shared"]);
        let list = pool["llm"]["model_list"].as_array().unwrap();
        assert!(list.iter().all(|entry| entry["model_name"] == "default"));
        assert_eq!(list[0]["litellm_params"]["weight"], 2);
        assert_eq!(
            list[0]["litellm_params"]["use_environment_credentials"],
            false
        );
        assert_eq!(list[1]["litellm_params"]["weight"], 3);
        assert_eq!(list[1]["model_info"]["hidden_endpoint"], true);
        assert!(list[0]["model_info"].get("hidden_endpoint").is_none());
        let ids: Vec<_> = pool["llm"]["providers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["user:p-own", "admin:p-admin"]);
        assert_eq!(list[0]["model_info"]["provider_id"], "user:p-own");
        assert_eq!(list[1]["model_info"]["provider_id"], "admin:p-admin");
        assert_eq!(
            pool["llm"]["providers"][0]["use_environment_credentials"],
            false
        );
        assert_eq!(pool["llm"]["router_settings"]["fallbacks"], json!([]));
        assert_eq!(pool["llm"]["enabled"], true);
        assert!(!pool.to_string().contains("admin-other-key"));
        assert!(markitai_core::llm_capabilities(&pool).routable);
        // A user entry that breaks the rules (here: a subscription runtime)
        // is left out even if a file holds it.
        let mut broken = own.clone();
        broken.models[0]["litellm_params"]["model"] = json!("claude-agent/sonnet");
        let pool = build(&base(), &broken, &admin(vec![], vec![])).unwrap();
        assert!(models(&pool).is_empty());
    }

    #[test]
    fn a_user_connection_named_after_the_admins_never_stands_in_for_it() {
        // The shared model carries the admin's key; its address comes from
        // the admin's connection. A user connection with the same id, at the
        // user's address, must not supply that address (nor its key).
        let data = admin(
            vec![
                json!({"model_name":"fast","litellm_params":{"model":"openai/shared","api_key":"admin-fixture-key"},
                "model_info":{"id":"s1","share":"shared","provider_id":"p-admin"}}),
            ],
            vec![
                json!({"id":"p-admin","provider":"openai","api_base":"https://models.admin.example/v1"}),
            ],
        );
        let own = users::Lists {
            models: vec![
                json!({"model_name":"mine","litellm_params":{"model":"openai/own"},
                "model_info":{"id":"m1","provider_id":"p-admin"}}),
            ],
            providers: vec![
                json!({"id":"p-admin","provider":"openai","api_key":"user-a-fixture-key",
                "api_base":"https://capture.user.example/v1"}),
            ],
            ..Default::default()
        };
        let pool = build(&base(), &own, &data).unwrap();
        let connection = |entry: &Value| -> Value {
            let id = &entry["model_info"]["provider_id"];
            let found: Vec<_> = pool["llm"]["providers"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|p| p["id"] == *id)
                .collect();
            assert_eq!(found.len(), 1, "{id}");
            found[0].clone()
        };
        let list = pool["llm"]["model_list"].as_array().unwrap();
        assert_eq!(models(&pool), ["openai/own", "openai/shared"]);
        assert_eq!(
            connection(&list[0])["api_base"],
            "https://capture.user.example/v1"
        );
        assert_eq!(
            connection(&list[1])["api_base"],
            "https://models.admin.example/v1"
        );
    }

    #[test]
    fn a_users_own_weight_for_a_shared_model_applies_to_its_pool_only() {
        let data = admin(
            vec![
                json!({"model_name":"fast","litellm_params":{"model":"openai/first","api_key":"admin-fixture-key","weight":3},"model_info":{"id":"s1","share":"shared"}}),
                json!({"model_name":"slow","litellm_params":{"model":"openai/second","api_key":"admin-fixture-key"},"model_info":{"id":"s2","share":"shared"}}),
            ],
            vec![],
        );
        let weights = |pool: &Value| -> Vec<u64> {
            pool["llm"]["model_list"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| entry["litellm_params"]["weight"].as_u64().unwrap())
                .collect()
        };
        let plain = build(&base(), &users::Lists::default(), &data).unwrap();
        assert_eq!(models(&plain), ["openai/first", "openai/second"]);
        assert_eq!(weights(&plain), [3, 1]);
        // A heavier weight for one, 0 for the other: only the first routes,
        // with the user's weight; an override for an id no longer shared
        // changes nothing.
        let own = users::Lists {
            shared_weights: [("s1".into(), 7), ("s2".into(), 0), ("gone".into(), 5)].into(),
            ..Default::default()
        };
        let pool = build(&base(), &own, &data).unwrap();
        assert_eq!(models(&pool), ["openai/first"]);
        assert_eq!(weights(&pool), [7]);
        assert!(markitai_core::llm_capabilities(&pool).routable);
        // Every shared model off for this user: an empty, sealed pool.
        let off = users::Lists {
            shared_weights: [("s1".into(), 0), ("s2".into(), 0)].into(),
            ..Default::default()
        };
        let pool = build(&base(), &off, &data).unwrap();
        assert!(models(&pool).is_empty());
        assert_eq!(pool["llm"]["enabled"], false);
        assert!(!markitai_core::llm_capabilities(&pool).routable);
        // The admin's own view of its models is untouched.
        assert_eq!(shared(&data).len(), 2);
    }
}
