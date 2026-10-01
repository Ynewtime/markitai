use super::{SettingsSource, identity, model, store::Data};
use markitai_core::provider_management::{provider_default_base, provider_label};
use serde_json::{Value, json};
use std::collections::HashSet;

const PROVIDERS: [(&str, &str); 8] = [
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
    ("azure", "AZURE_API_KEY"),
    ("ollama", "OLLAMA_API_KEY"),
    ("custom", ""),
];
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
                if matches!(name.as_str(), "ollama" | "ollama_chat") {
                    return true;
                }
                let Some((_, variable)) = PROVIDERS.iter().find(|(provider, _)| *provider == name)
                else {
                    return false;
                };
                let linked = model::linked(entry)
                    .and_then(|id| data.providers.iter().find(|p| p["id"] == id));
                let key = entry["litellm_params"]
                    .get("api_key")
                    .filter(|v| nonempty(v))
                    .or_else(|| linked.and_then(|p| p.get("api_key").filter(|v| nonempty(v))));
                match key {
                    Some(value) => resolved(value).is_some_and(|key| !key.is_empty()),
                    None => std::env::var(variable).is_ok_and(|v| !v.is_empty()),
                }
            })
        })
}
fn deployment(entry: &Value, index: usize, persisted: bool) -> Value {
    let params = &entry["litellm_params"];
    let local = model::local(&model::provider(entry));
    json!({"deployment_id":identity::id(entry,index),"routing_group":entry["model_name"].as_str().unwrap_or("default"),"model":params["model"],"weight":params["weight"].as_u64().unwrap_or(1),"api_key_configured":!local&&nonempty(&params["api_key"]),"api_base_configured":!local&&nonempty(&params["api_base"]),"api_base":if local{None}else{origin(params["api_base"].as_str())},"persisted":persisted})
}
pub(super) fn payload(data: &Data, source: &SettingsSource) -> Value {
    let path = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .and_then(|home| {
            source
                .path
                .strip_prefix(home)
                .ok()
                .map(|rest| format!("~/{}", rest.display()))
        })
        .unwrap_or_else(|| source.path.to_string_lossy().into_owned());
    json!({"configured":!data.models.is_empty(),"routable":routable(data),"source":if !data.models.is_empty(){"config"}else if !data.detected.is_empty(){"detected"}else{"none"},"config_path":path,"config_origin":source.origin,"revision":data.revision,"deployments":data.models.iter().enumerate().map(|(index,entry)|deployment(entry,index,true)).collect::<Vec<_>>(),"detected":data.detected.iter().enumerate().map(|(index,entry)|deployment(entry,index,false)).collect::<Vec<_>>()})
}
pub(super) fn providers(data: &Data) -> Value {
    let mut result = Vec::new();
    for (name, variable) in PROVIDERS {
        let saved: Vec<_> = data
            .providers
            .iter()
            .filter(|p| p["provider"] == name)
            .collect();
        let environment =
            !variable.is_empty() && std::env::var(variable).is_ok_and(|v| !v.is_empty());
        if environment
            && !saved
                .iter()
                .any(|p| p["api_key"] == format!("env:{variable}"))
        {
            result.push(json!({"id":format!("env:{name}"),"provider":name,"label":provider_label(name),"kind":"environment","status":"ready","source":variable,"credential":format!("env:{variable}"),"supports_discovery":true}));
        }
    }
    for name in [
        "openai",
        "anthropic",
        "gemini",
        "ollama",
        "deepseek",
        "openrouter",
        "azure",
        "custom",
    ] {
        if data.providers.iter().any(|p| p["provider"] == name)
            || data.models.iter().any(|p| model::provider(p) == name)
            || result.iter().any(|p| p["provider"] == name)
        {
            continue;
        }
        let label = match name {
            "azure" => "Azure OpenAI".to_owned(),
            "custom" => "OpenAI-compatible endpoint".to_owned(),
            _ => provider_label(name),
        };
        result.push(json!({"id":format!("common:{name}"),"provider":name,"label":label,"kind":"common","status":if name=="ollama"{"unknown"}else{"needs_credentials"},"source":"built_in","supports_discovery":true}));
    }
    for saved in &data.providers {
        let name = saved["provider"].as_str().unwrap_or_default();
        let id = saved["id"].as_str().unwrap_or_default();
        let count = data
            .models
            .iter()
            .filter(|entry| model::linked(entry) == Some(id))
            .count();
        result.push(json!({"id":format!("provider:{id}"),"provider_id":id,"provider":name,"label":provider_label(name),"kind":"configured","status":if !saved["api_key"].is_null()||!saved["api_base"].is_null(){"ready"}else{"needs_credentials"},"source":"config","api_key_configured":nonempty(&saved["api_key"]),"api_base_configured":nonempty(&saved["api_base"]),"api_base":origin(saved["api_base"].as_str().filter(|s|!s.is_empty()).or_else(||provider_default_base(name))),"model_count":count,"supports_discovery":!model::local(name)}));
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
        result.push(json!({"id":format!("provider:legacy:{id}"),"provider_id":format!("legacy:{id}"),"deployment_id":id,"provider":name,"label":provider_label(&name),"kind":"configured","status":if params["weight"].as_u64()==Some(0){"disabled"}else{"ready"},"source":"config","api_key_configured":nonempty(&params["api_key"]),"api_base_configured":nonempty(&params["api_base"]),"api_base":origin(params["api_base"].as_str().filter(|s|!s.is_empty()).or_else(||provider_default_base(&name))),"model_count":count,"supports_discovery":true}));
    }
    json!({"providers":result})
}
