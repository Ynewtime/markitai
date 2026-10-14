//! The models a user token's conversions route to: its own, then the ones
//! the admin shares, in one pool. Computed from the current lists on every
//! request, so a change to either applies to the next conversion.
use super::{model, store::Data, users, views};
use serde_json::{Value, json};

/// The model of an empty pool. A configuration without models would fall
/// back to the server's environment (`MODEL` and the provider key
/// variables); this one never routes and never reads it.
pub(super) const PLACEHOLDER: &str = "openai/markitai-no-model";

fn placeholder() -> Value {
    json!({"model_name":"default","litellm_params":{"model":PLACEHOLDER,"weight":0,
        "api_key":model::NO_KEY,"use_environment_credentials":false}})
}

fn weight(entry: &Value) -> u64 {
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
        providers.push(connection);
    }
    for entry in &own.models {
        // Saved through the rules; checked again, so that no file can make a
        // user's model reach the server's credentials.
        if weight(entry) == 0 || users::check_entry(entry, &own.providers).is_err() {
            continue;
        }
        let mut entry = grouped(entry);
        entry["litellm_params"]["use_environment_credentials"] = json!(false);
        models.push(entry);
    }
    for (_, saved) in shared(admin) {
        let mut entry = grouped(saved);
        // A failure a user sees names a shared model, never where the admin
        // serves it.
        if !entry["model_info"].is_object() {
            entry["model_info"] = json!({});
        }
        entry["model_info"]["hidden_endpoint"] = json!(true);
        models.push(entry);
        if let Some(id) = model::linked(saved)
            && !providers.iter().any(|p| p["id"] == id)
            && let Some(connection) = admin.providers.iter().find(|p| p["id"] == id)
        {
            providers.push(connection.clone());
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
        assert_eq!(ids, ["p-own", "p-admin"]);
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
}
