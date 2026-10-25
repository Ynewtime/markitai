//! What a conversion ran with, as its history entry keeps it (`settings`):
//! the configuration values that shape a result, by dotted key, where they
//! differ from markitai's built-in defaults. The workbench's Details tab
//! shows them as the equivalent command line. Only the keys listed here are
//! ever read, so no credential, address, header or prompt text is kept (a
//! custom prompt is recorded as `"custom"`).

use serde_json::{Map, Value, json};

/// Keys that matter for every conversion.
const ALWAYS: &[&str] = &[
    "llm.enabled",
    "llm.pure",
    "ocr.enabled",
    "screenshot.enabled",
    "output.profile",
    "output.format",
    "output.page_markers",
    "output.text_only",
    "output.wikilinks",
    "cache.enabled",
    "cache.no_cache",
    "image.compress",
    "image.format",
    "image.quality",
    "image.max_width",
    "image.max_height",
    "image.filter.min_width",
    "image.filter.min_height",
    "image.filter.min_area",
    "image.filter.deduplicate",
    "security.pdf_sanitize",
    "fetch.strategy",
    // `never` is `--no-remote-fetch`: a copied command keeps the prohibition.
    "fetch.remote_consent",
    "fetch.cloudflare.convert_enabled",
];

/// Keys that matter only while the switch before them is on.
const WHEN: &[(&str, &[&str])] = &[
    (
        "llm.enabled",
        &[
            "llm.keep_base",
            "image.alt_enabled",
            "image.desc_enabled",
            "llm.on_failure",
            "llm.max_requests_per_document",
            "llm.max_cost_per_document_usd",
            "llm.max_vision_pages_per_document",
        ],
    ),
    ("ocr.enabled", &["ocr.lang", "ocr.per_page_routing"]),
    (
        "screenshot.enabled",
        &[
            "screenshot.screenshot_only",
            "screenshot.quality",
            "screenshot.viewport_width",
            "screenshot.viewport_height",
            "screenshot.max_height",
        ],
    ),
];

fn pointer(key: &str) -> String {
    format!("/{}", key.replace('.', "/"))
}

/// `cfg`'s values for the listed keys that differ from the defaults.
pub(crate) fn of(cfg: &Value) -> Map<String, Value> {
    let defaults = markitai_core::config::defaults();
    let value = |key: &str| cfg.pointer(&pointer(key)).cloned().unwrap_or(Value::Null);
    let default = |key: &str| {
        defaults
            .pointer(&pointer(key))
            .cloned()
            .unwrap_or(Value::Null)
    };
    let mut keys: Vec<&str> = ALWAYS.to_vec();
    for (switch, more) in WHEN {
        if value(switch) == true {
            keys.extend_from_slice(more);
        }
    }
    let mut settings = Map::new();
    for key in keys {
        let current = value(key);
        // OCR's language is free text in the configuration: kept only when it
        // is a language identifier OCR would read, never an arbitrary string.
        if key == "ocr.lang"
            && current
                .as_str()
                .is_none_or(|lang| markitai_core::ocr_lang_spelling(lang).is_none())
        {
            continue;
        }
        if current != default(key) {
            settings.insert(key.to_owned(), current);
        }
    }
    if value("llm.enabled") == true
        && let Some(prompts) = cfg["prompts"].as_object()
    {
        for (name, text) in prompts {
            if name != "dir" && text.as_str().is_some_and(|text| !text.trim().is_empty()) {
                settings.insert(format!("prompts.{name}"), json!("custom"));
            }
        }
    }
    settings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_changed_keys_that_shape_the_result_are_kept() {
        let defaults = markitai_core::config::defaults();
        assert!(of(&defaults).is_empty());
        let mut cfg = defaults.clone();
        cfg["image"]["format"] = json!("webp");
        cfg["image"]["quality"] = json!(85);
        cfg["cache"]["no_cache"] = json!(true);
        // Off-switch details are left out: OCR's language without OCR.
        cfg["ocr"]["lang"] = json!("ja");
        assert_eq!(
            Value::Object(of(&cfg)),
            json!({"image.format":"webp","image.quality":85,"cache.no_cache":true})
        );
        cfg["ocr"]["enabled"] = json!(true);
        assert_eq!(of(&cfg)["ocr.lang"], "ja");
    }

    #[test]
    fn a_run_that_keeps_urls_off_remote_services_says_so() {
        // `--no-remote-fetch` (or `fetch.remote_consent: never`): a copied
        // `--strategy jina` must not lose the prohibition.
        let mut cfg = markitai_core::config::defaults();
        cfg["fetch"]["strategy"] = json!("jina");
        cfg["fetch"]["remote_consent"] = json!("never");
        cfg["fetch"]["no_remote"] = json!(true);
        assert_eq!(
            Value::Object(of(&cfg)),
            json!({"fetch.strategy":"jina","fetch.remote_consent":"never"})
        );
        cfg["fetch"]["remote_consent"] = json!("ask");
        assert_eq!(of(&cfg)["fetch.remote_consent"], "ask");
    }

    #[test]
    fn an_ocr_language_is_kept_only_as_a_language_identifier() {
        let mut cfg = markitai_core::config::defaults();
        cfg["ocr"]["enabled"] = json!(true);
        cfg["ocr"]["lang"] = json!("zh_Hant");
        assert_eq!(of(&cfg)["ocr.lang"], "zh_Hant");
        // Anything OCR would refuse (an address with credentials) is not recorded.
        for bad in [
            "https://user:secret@example.test/private",
            "en zh",
            "",
            "x".repeat(49).as_str(),
        ] {
            cfg["ocr"]["lang"] = json!(bad);
            assert!(!of(&cfg).contains_key("ocr.lang"), "{bad}");
        }
    }

    #[test]
    fn credentials_addresses_and_prompt_text_are_never_kept() {
        let mut cfg = markitai_core::config::defaults();
        cfg["llm"]["enabled"] = json!(true);
        cfg["llm"]["model_list"] = json!([{"model_name":"m","litellm_params":{"model":"openai/x","api_key":"sk-secret-value","api_base":"https://relay.example.test/v1"}}]);
        cfg["fetch"]["jina"]["api_key"] = json!("jina-secret-value");
        cfg["fetch"]["playwright"]["extra_http_headers"] = json!({"Authorization":"Bearer t0ken"});
        cfg["prompts"]["cleaner_system"] = json!("Private instructions that name a client.");
        let settings = of(&cfg);
        let text = serde_json::to_string(&settings).unwrap();
        for secret in [
            "sk-secret",
            "relay.example",
            "jina-secret",
            "t0ken",
            "Private",
        ] {
            assert!(!text.contains(secret), "{text}");
        }
        assert_eq!(settings["llm.enabled"], true);
        assert_eq!(settings["prompts.cleaner_system"], "custom");
    }
}
