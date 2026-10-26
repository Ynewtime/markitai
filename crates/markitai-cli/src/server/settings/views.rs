use super::{SettingsSource, identity, model, store::Data};
use markitai_core::provider_management::{
    catalog, environment_key_variable, held_key_variable, provider_default_base,
    provider_endpoints, provider_entry as entry, provider_label, provider_regions,
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

fn set(variable: &str) -> bool {
    std::env::var(variable).is_ok_and(|value| !value.trim().is_empty())
}
/// What a card says about its provider beyond the connection: the picker
/// group, brand mark and search names, the documented endpoint, the key
/// variable, whether a key is needed, whether models can be listed, and the
/// regions a key may belong to.
fn facts(name: &str, card: &mut Value) {
    let entry = entry(name);
    if let Some(entry) = &entry {
        card["group"] = json!(entry.group);
        card["icon"] = json!(entry.icon);
        card["aliases"] = json!(entry.aliases);
    }
    let variable = entry.as_ref().and_then(|entry| {
        held_key_variable(entry.key_variables, set).or(entry.key_variables.first().copied())
    });
    card["default_base"] = json!(entry.as_ref().and_then(|entry| entry.default_base));
    // A key pasted for a provider that issues keys per region is tried at each.
    let regions = provider_regions(name);
    if !regions.is_empty() {
        card["regions"] = json!(regions);
    }
    // Its documented addresses, labelled, for the address field's list (v1.27).
    let endpoints = provider_endpoints(name);
    if !endpoints.is_empty() {
        card["endpoints"] = json!(endpoints);
    }
    card["key_variable"] = json!(variable);
    card["key_optional"] = json!(entry.as_ref().is_some_and(|entry| entry.key_optional));
    if card
        .get("supports_discovery")
        .is_none_or(|value| value == true)
    {
        card["supports_discovery"] = json!(entry.as_ref().is_none_or(|entry| entry.discovery));
    }
}
fn nonempty(value: &Value) -> bool {
    value.as_str().is_some_and(|s| !s.is_empty())
}
fn resolved(value: &Value) -> Option<String> {
    value.as_str().and_then(|s| {
        if let Some(key) = s.strip_prefix("env:") {
            std::env::var(key).ok()
        } else {
            Some(s.to_owned())
        }
    })
}
fn origin(value: Option<&str>) -> Option<String> {
    let value = value?;
    let value = if let Some(key) = value.strip_prefix("env:") {
        std::env::var(key).ok()?
    } else {
        value.to_owned()
    };
    let url = url::Url::parse(&value).ok()?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return None;
    }
    Some(url.origin().ascii_serialization())
}
/// Whether a conversion could route an LLM request: what the conversion side
/// reports (subscription runtimes, models detected from the environment), or
/// a configured deployment whose key a linked provider record holds.
fn routable(data: &Data) -> bool {
    markitai_core::llm_capabilities(&data.cfg).routable || keyed_deployment(data)
}
fn keyed_deployment(data: &Data) -> bool {
    data.cfg["llm"]["model_list"]
        .as_array()
        .is_some_and(|models| {
            models.iter().any(|entry| {
                if entry["litellm_params"]["weight"].as_u64() == Some(0) {
                    return false;
                }
                let name = model::provider(entry);
                // Subscription runtimes and the clouds' own credentials are
                // the conversion side's to report.
                let Some(known) = self::entry(&name).filter(|known| {
                    !matches!(known.provider, "custom" | "bedrock" | "vertex_ai")
                        && !model::local(known.provider)
                }) else {
                    return false;
                };
                let linked = model::linked(entry)
                    .and_then(|id| data.providers.iter().find(|p| p["id"] == id));
                let explicit = |field: &str| {
                    entry["litellm_params"]
                        .get(field)
                        .filter(|v| nonempty(v))
                        .or_else(|| linked.and_then(|p| p.get(field).filter(|v| nonempty(v))))
                };
                let usable = |value: &Value| resolved(value).is_some_and(|value| !value.is_empty());
                // As conversion resolves it: the deployment's own choice, else its
                // connection's; a connection saved through this API without a key
                // never takes the environment's.
                let environment = entry["litellm_params"]
                    .get("use_environment_credentials")
                    .filter(|value| !value.is_null())
                    .or_else(|| linked.and_then(|p| p.get("use_environment_credentials")))
                    .and_then(Value::as_bool)
                    != Some(false);
                // A local server needs no key, only an address (vLLM has no default one).
                if known.key_optional {
                    return known.default_base.is_some()
                        || explicit("api_base").is_some_and(usable)
                        || environment
                            && known.base_variables.iter().any(|variable| set(variable));
                }
                // An explicit key, even an unresolvable reference, is the only candidate.
                match explicit("api_key") {
                    Some(value) => usable(value),
                    None => environment && known.key_variables.iter().any(|variable| set(variable)),
                }
            })
        })
}
fn deployment(entry: &Value, index: usize, persisted: bool) -> Value {
    let params = &entry["litellm_params"];
    let local = model::local(&model::provider(entry));
    json!({"deployment_id":identity::id(entry,index),"routing_group":entry["model_name"].as_str().unwrap_or("default"),"model":params["model"],"weight":params["weight"].as_u64().unwrap_or(1),"api_key_configured":!local&&model::real_key(params["api_key"].as_str()),"api_base_configured":!local&&nonempty(&params["api_base"]),"api_base":if local{None}else{origin(params["api_base"].as_str())},"persisted":persisted})
}
/// The settings view; `batch_remove`: the batch route also removes saved
/// models (`remove`), so the page saves additions and removals as one change.
pub(super) fn payload(data: &Data, source: &SettingsSource) -> Value {
    let path = display_path(source);
    json!({"configured":!data.models.is_empty(),"routable":routable(data),"source":if !data.models.is_empty(){"config"}else if !data.detected.is_empty(){"detected"}else{"none"},"config_path":path,"config_origin":source.origin,"revision":data.revision,"deployments":data.models.iter().enumerate().map(|(index,entry)|{let mut row=deployment(entry,index,true);row["share"]=json!(if shared(entry){"shared"}else{"private"});row}).collect::<Vec<_>>(),"detected":data.detected.iter().enumerate().map(|(index,entry)|deployment(entry,index,false)).collect::<Vec<_>>(),"batch_remove":true})
}
/// Whether the admin shares this saved model with users' conversions.
pub(super) fn shared(entry: &Value) -> bool {
    entry.pointer("/model_info/share").and_then(Value::as_str) == Some("shared")
}
/// A user token's own models (`data`, its lists) and the admin's shared ones
/// (`shared`, already shaped), with whether the pool of both can route.
pub(super) fn user_payload(data: &Data, shared: Vec<Value>, routable: bool) -> Value {
    json!({"configured":!data.models.is_empty(),"routable":routable,"source":"user",
        "config_path":null,"config_origin":"user","revision":data.revision,
        "deployments":data.models.iter().enumerate().map(|(index,entry)|deployment(entry,index,true)).collect::<Vec<_>>(),
        "detected":[],"shared":shared,"limit":super::users::LIMIT,"batch_remove":true})
}
/// The configuration file's path as the settings show it (`~/…` under home).
pub(super) fn display_path(source: &SettingsSource) -> String {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .and_then(|home| {
            source
                .path
                .strip_prefix(home)
                .ok()
                .map(|rest| format!("~/{}", rest.display()))
        })
        .unwrap_or_else(|| source.path.to_string_lossy().into_owned())
}
/// How a saved connection authenticates, when the environment supplies its
/// key: `env:VAR` for a key that is an `env:VAR` reference, or for a
/// connection with no key at all (hand-written; never the keyless
/// placeholder) that may use the environment's credentials while one of its
/// provider's key variables is set.
fn credential(
    provider: &str,
    key: &Value,
    use_environment: &Value,
    env: &HashMap<String, String>,
) -> Option<String> {
    match key.as_str().filter(|key| !key.is_empty()) {
        Some(key) => key
            .strip_prefix("env:")
            .filter(|name| !name.is_empty())
            .map(|name| format!("env:{name}")),
        None if use_environment.as_bool() != Some(false) => {
            environment_key_variable(provider, env).map(|name| format!("env:{name}"))
        }
        None => None,
    }
}
/// Whether a connection's address is its provider's own endpoint.
fn own_endpoint(provider: &str, base: &Value) -> bool {
    match base.as_str().filter(|base| !base.is_empty()) {
        None => true,
        Some(base) => provider_default_base(provider) == Some(base),
    }
}
/// The cards of saved connections (`configured`) and of hand-written
/// model entries without one (`legacy`), each with whether it is aimed at
/// its provider's own endpoint.
fn configured_cards(data: &Data, env: &HashMap<String, String>) -> Vec<(Value, bool)> {
    let mut cards = Vec::new();
    for saved in &data.providers {
        let name = saved["provider"].as_str().unwrap_or_default();
        let id = saved["id"].as_str().unwrap_or_default();
        let count = data
            .models
            .iter()
            .filter(|entry| model::linked(entry) == Some(id))
            .count();
        let mut card = json!({"id":format!("provider:{id}"),"provider_id":id,"provider":name,"label":provider_label(name),"kind":"configured","status":if !saved["api_key"].is_null()||!saved["api_base"].is_null(){"ready"}else{"needs_credentials"},"source":"config","api_key_configured":model::real_key(saved["api_key"].as_str()),"api_base_configured":nonempty(&saved["api_base"]),"api_base":origin(saved["api_base"].as_str().filter(|s|!s.is_empty()).or_else(||provider_default_base(name))),"model_count":count,"supports_discovery":!model::local(name)});
        if let Some(credential) = credential(
            name,
            &saved["api_key"],
            &saved["use_environment_credentials"],
            env,
        ) {
            card["credential"] = json!(credential);
        }
        cards.push((card, own_endpoint(name, &saved["api_base"])));
    }
    let mut seen = HashSet::new();
    for (index, entry) in data.models.iter().enumerate() {
        let name = model::provider(entry);
        let params = &entry["litellm_params"];
        if model::local(&name)
            || data.providers.iter().any(|p| {
                model::linked(entry) == p["id"].as_str()
                    || p["provider"] == name
                        && p["api_key"] == params["api_key"]
                        && p["api_base"] == params["api_base"]
            })
        {
            continue;
        }
        let connection = json!([name, params["api_key"], params["api_base"]]).to_string();
        if !seen.insert(connection) {
            continue;
        }
        let count = data
            .models
            .iter()
            .filter(|other| {
                model::provider(other) == name
                    && other["litellm_params"]["api_key"] == params["api_key"]
                    && other["litellm_params"]["api_base"] == params["api_base"]
            })
            .count();
        let id = identity::id(entry, index);
        let mut card = json!({"id":format!("provider:legacy:{id}"),"provider_id":format!("legacy:{id}"),"deployment_id":id,"provider":name,"label":provider_label(&name),"kind":"configured","status":if params["weight"].as_u64()==Some(0){"disabled"}else{"ready"},"source":"config","api_key_configured":model::real_key(params["api_key"].as_str()),"api_base_configured":nonempty(&params["api_base"]),"api_base":origin(params["api_base"].as_str().filter(|s|!s.is_empty()).or_else(||provider_default_base(&name))),"model_count":count,"supports_discovery":true});
        if let Some(credential) = credential(
            &name,
            &params["api_key"],
            &params["use_environment_credentials"],
            env,
        ) {
            card["credential"] = json!(credential);
        }
        cards.push((card, own_endpoint(&name, &params["api_base"])));
    }
    cards
}
pub(super) fn providers(data: &Data) -> Value {
    let mut result = Vec::new();
    let catalog = catalog();
    let env = markitai_core::config::environment();
    let configured = configured_cards(data, &env);
    for known in &catalog {
        let name = known.provider;
        let Some(variable) = held_key_variable(known.key_variables, set) else {
            continue;
        };
        let reference = format!("env:{variable}");
        // A saved connection that already uses this variable at the
        // provider's own endpoint stands for it.
        let served = configured.iter().any(|(card, own)| {
            *own && card["provider"] == name && card["credential"] == reference.as_str()
        });
        if !served
            && !data
                .providers
                .iter()
                .filter(|p| p["provider"] == name)
                .any(|p| p["api_key"] == reference)
        {
            result.push(json!({"id":format!("env:{name}"),"provider":name,"label":provider_label(name),"kind":"environment","status":"ready","source":variable,"credential":format!("env:{variable}"),"supports_discovery":known.discovery}));
        }
    }
    // The reference's providers first, then the further OpenAI-compatible ones.
    for known in &catalog {
        let name = known.provider;
        // A saved connection hides its provider's card, except that another
        // OpenAI-compatible endpoint, or OpenAI itself, can always be added,
        // and a subscription runtime, which keeps no saved connection, can
        // always add more of its models.
        let repeatable = matches!(name, "openai" | "custom") || model::local(name);
        if !repeatable
            && (data.providers.iter().any(|p| p["provider"] == name)
                || data.models.iter().any(|p| model::provider(p) == name))
            || result.iter().any(|p| p["provider"] == name)
        {
            continue;
        }
        let label = known.label;
        // A local server at its default address, a subscription runtime or a
        // cloud's own credentials may already work; only a check tells.
        let status = if known.key_optional
            && (known.default_base.is_some() || matches!(known.group, "subscriptions" | "clouds"))
        {
            "unknown"
        } else {
            "needs_credentials"
        };
        result.push(json!({"id":format!("common:{name}"),"provider":name,"label":label,"kind":if known.compatible{"compatible"}else{"common"},"status":status,"source":"built_in","supports_discovery":known.discovery}));
    }
    result.extend(configured.into_iter().map(|(card, _)| card));
    for card in &mut result {
        let name = card["provider"].as_str().unwrap_or_default().to_owned();
        facts(&name, card);
    }
    json!({"providers":result})
}

/// The provider cards a user token sees: the built-in ones it may connect
/// (none that signs in with the server's own account or cloud login, except
/// Bedrock with a key of the user's own), then its own saved connections.
/// Never an environment card, a key variable or a credential, and no default
/// address on the server's own machine.
pub(super) fn user_providers(data: &Data) -> Value {
    let mut result = Vec::new();
    for known in &catalog() {
        let name = known.provider;
        if model::server_login(name) {
            continue;
        }
        let repeatable = matches!(name, "openai" | "custom");
        if !repeatable
            && (data.providers.iter().any(|p| p["provider"] == name)
                || data.models.iter().any(|p| model::provider(p) == name))
        {
            continue;
        }
        let status =
            if known.key_optional && known.group != "clouds" && user_default_base(name).is_some() {
                "unknown"
            } else {
                "needs_credentials"
            };
        result.push(json!({"id":format!("common:{name}"),"provider":name,"label":known.label,"kind":if known.compatible{"compatible"}else{"common"},"status":status,"source":"built_in","supports_discovery":known.discovery}));
    }
    result.extend(
        configured_cards(data, &HashMap::new())
            .into_iter()
            .map(|(card, _)| card),
    );
    for card in &mut result {
        let name = card["provider"].as_str().unwrap_or_default().to_owned();
        facts(&name, card);
        let fields = card.as_object_mut().unwrap();
        fields.remove("credential");
        fields.insert("key_variable".into(), Value::Null);
        fields.insert("default_base".into(), json!(user_default_base(&name)));
        // A user's Bedrock connection needs its own key.
        if name == "bedrock" {
            fields.insert("key_optional".into(), json!(false));
        }
    }
    json!({"providers":result})
}
/// A provider's documented address, unless it is on the server's own machine.
pub(super) fn user_default_base(provider: &str) -> Option<&'static str> {
    model::default_base(provider).filter(|base| !model::server_machine(base))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_connection_names_the_environment_key_it_authenticates_with() {
        let env = HashMap::from([
            ("DEEPSEEK_API_KEY".to_owned(), "set".to_owned()),
            ("GROQ_API_KEY".to_owned(), " ".to_owned()),
        ]);
        let on = json!(true);
        let unset = Value::Null;
        for (provider, key, environment, expected) in [
            // A reference is named as written, set or not.
            (
                "openai",
                json!("env:OPENAI_API_KEY"),
                &unset,
                Some("env:OPENAI_API_KEY"),
            ),
            (
                "deepseek",
                json!("env:OTHER"),
                &json!(false),
                Some("env:OTHER"),
            ),
            // No key at all: the provider's key variable that holds a value.
            (
                "deepseek",
                Value::Null,
                &unset,
                Some("env:DEEPSEEK_API_KEY"),
            ),
            ("deepseek", json!(""), &on, Some("env:DEEPSEEK_API_KEY")),
            // Not when the connection never uses the environment, nor for a
            // stored key, the keyless placeholder, or an empty variable.
            ("deepseek", Value::Null, &json!(false), None),
            ("deepseek", json!("sk-literal"), &on, None),
            ("deepseek", json!(model::NO_KEY), &on, None),
            ("groq", Value::Null, &on, None),
            ("ollama", Value::Null, &on, None),
            ("custom", Value::Null, &on, None),
        ] {
            assert_eq!(
                credential(provider, &key, environment, &env).as_deref(),
                expected,
                "{provider} {key} {environment}"
            );
        }
        assert!(own_endpoint("deepseek", &Value::Null));
        assert!(own_endpoint("deepseek", &json!("")));
        assert!(own_endpoint("deepseek", &json!("https://api.deepseek.com")));
        assert!(!own_endpoint(
            "deepseek",
            &json!("https://proxy.example/v1")
        ));
        assert!(!own_endpoint("custom", &json!("http://127.0.0.1:8000/v1")));
    }
}
