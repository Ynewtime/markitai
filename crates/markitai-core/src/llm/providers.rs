//! The model prefixes this build sends over HTTP, and the explicit refusals
//! for prefixes it cannot serve.
//!
//! Each entry was checked against the provider's own documentation on
//! 2026-10-02; `docs/llm.md` lists the page behind every base URL. Key
//! variables keep the names the reference's LiteLLM 1.100.1 reads, so an
//! environment prepared for the reference keeps working, with the
//! provider's own documented name first when the two differ.
use super::Protocol;
use crate::Error;

#[derive(Debug)]
pub(super) struct Provider {
    /// The text before the first `/` of `litellm_params.model`.
    pub prefix: &'static str,
    pub protocol: Protocol,
    /// The documented default endpoint, or `None` when every server has its
    /// own address and `api_base` (or a base variable) must name it.
    pub base: Option<&'static str>,
    /// Environment variables that replace `base`, in order.
    pub base_vars: &'static [&'static str],
    /// Environment variables holding the API key, in order.
    pub key_vars: &'static [&'static str],
    /// A deployment without a key can still be routed (local servers).
    pub key_optional: bool,
}

const fn chat(
    prefix: &'static str,
    base: &'static str,
    base_vars: &'static [&'static str],
    key_vars: &'static [&'static str],
) -> Provider {
    Provider {
        prefix,
        protocol: Protocol::Chat,
        base: Some(base),
        base_vars,
        key_vars,
        key_optional: false,
    }
}

static PROVIDERS: [Provider; 24] = [
    chat(
        "openai",
        "https://api.openai.com/v1",
        &["OPENAI_API_BASE", "OPENAI_BASE_URL"],
        &["OPENAI_API_KEY"],
    ),
    Provider {
        prefix: "anthropic",
        protocol: Protocol::Anthropic,
        base: Some("https://api.anthropic.com/v1"),
        base_vars: &["ANTHROPIC_API_BASE"],
        key_vars: &["ANTHROPIC_API_KEY"],
        key_optional: false,
    },
    chat(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        &["GEMINI_API_BASE"],
        &["GEMINI_API_KEY"],
    ),
    chat(
        "deepseek",
        "https://api.deepseek.com/v1",
        &["DEEPSEEK_API_BASE"],
        &["DEEPSEEK_API_KEY"],
    ),
    chat(
        "openrouter",
        "https://openrouter.ai/api/v1",
        &["OPENROUTER_API_BASE"],
        &["OPENROUTER_API_KEY"],
    ),
    Provider {
        prefix: "azure",
        protocol: Protocol::Azure,
        base: None,
        base_vars: &["AZURE_API_BASE"],
        key_vars: &["AZURE_API_KEY"],
        key_optional: false,
    },
    Provider {
        key_optional: true,
        ..chat(
            "ollama",
            "http://localhost:11434/v1",
            &["OLLAMA_API_BASE"],
            &["OLLAMA_API_KEY"],
        )
    },
    Provider {
        key_optional: true,
        ..chat(
            "ollama_chat",
            "http://localhost:11434/v1",
            &["OLLAMA_CHAT_API_BASE", "OLLAMA_API_BASE"],
            &["OLLAMA_API_KEY"],
        )
    },
    // OpenAI-compatible hosted APIs.
    chat(
        "groq",
        "https://api.groq.com/openai/v1",
        &["GROQ_API_BASE"],
        &["GROQ_API_KEY"],
    ),
    chat(
        "mistral",
        "https://api.mistral.ai/v1",
        &["MISTRAL_API_BASE"],
        &["MISTRAL_API_KEY"],
    ),
    chat(
        "xai",
        "https://api.x.ai/v1",
        &["XAI_API_BASE"],
        &["XAI_API_KEY"],
    ),
    chat(
        "together_ai",
        "https://api.together.ai/v1",
        &["TOGETHER_AI_API_BASE"],
        &[
            "TOGETHER_API_KEY",
            "TOGETHER_AI_API_KEY",
            "TOGETHERAI_API_KEY",
            "TOGETHER_AI_TOKEN",
        ],
    ),
    // Perplexity ended Sonar chat completions on 2026-09-27; its Router
    // keeps the Chat Completions schema.
    chat(
        "perplexity",
        "https://api.perplexity.ai/router/v1",
        &["PERPLEXITY_API_BASE"],
        &["PERPLEXITY_API_KEY", "PERPLEXITYAI_API_KEY"],
    ),
    chat(
        "cerebras",
        "https://api.cerebras.ai/v1",
        &["CEREBRAS_API_BASE"],
        &["CEREBRAS_API_KEY"],
    ),
    chat(
        "fireworks_ai",
        "https://api.fireworks.ai/inference/v1",
        &["FIREWORKS_API_BASE"],
        &[
            "FIREWORKS_API_KEY",
            "FIREWORKS_AI_API_KEY",
            "FIREWORKSAI_API_KEY",
            "FIREWORKS_AI_TOKEN",
        ],
    ),
    chat(
        "deepinfra",
        "https://api.deepinfra.com/v1/openai",
        &["DEEPINFRA_API_BASE"],
        &["DEEPINFRA_API_KEY"],
    ),
    chat(
        "nebius",
        "https://api.tokenfactory.nebius.com/v1",
        &["NEBIUS_API_BASE"],
        &["NEBIUS_API_KEY"],
    ),
    chat(
        "moonshot",
        "https://api.moonshot.ai/v1",
        &["MOONSHOT_API_BASE"],
        &["MOONSHOT_API_KEY"],
    ),
    chat(
        "sambanova",
        "https://api.sambanova.ai/v1",
        &["SAMBANOVA_API_BASE"],
        &["SAMBANOVA_API_KEY"],
    ),
    chat(
        "zai",
        "https://api.z.ai/api/paas/v4",
        &["ZAI_API_BASE"],
        &["ZAI_API_KEY"],
    ),
    chat(
        "nvidia_nim",
        "https://integrate.api.nvidia.com/v1",
        &["NVIDIA_NIM_API_BASE"],
        &["NVIDIA_NIM_API_KEY"],
    ),
    chat(
        "novita",
        "https://api.novita.ai/openai",
        &["NOVITA_API_BASE"],
        &["NOVITA_API_KEY"],
    ),
    // Self-hosted OpenAI-compatible servers. vLLM listens wherever it was
    // started, so it has no default; LM Studio's local server is documented
    // at port 1234. Neither checks a key unless configured to.
    Provider {
        prefix: "hosted_vllm",
        protocol: Protocol::Chat,
        base: None,
        base_vars: &["HOSTED_VLLM_API_BASE"],
        key_vars: &["HOSTED_VLLM_API_KEY"],
        key_optional: true,
    },
    Provider {
        key_optional: true,
        ..chat(
            "lm_studio",
            "http://localhost:1234/v1",
            &["LM_STUDIO_API_BASE"],
            &["LM_STUDIO_API_KEY"],
        )
    },
];

/// Prefixes served by an installed official runtime instead of HTTP.
pub(super) const SUBSCRIPTIONS: [&str; 3] = ["copilot", "claude-agent", "chatgpt"];

pub(super) fn find(prefix: &str) -> Option<&'static Provider> {
    PROVIDERS.iter().find(|provider| provider.prefix == prefix)
}

/// Whether this build can route a model with this prefix.
pub(crate) fn supported(prefix: &str) -> bool {
    find(prefix).is_some() || SUBSCRIPTIONS.contains(&prefix)
}

/// Whether a deployment needs no API key to be usable: local servers and the
/// subscription runtimes, which carry their own sign-in.
pub(super) fn key_optional(prefix: &str) -> bool {
    SUBSCRIPTIONS.contains(&prefix) || find(prefix).is_some_and(|provider| provider.key_optional)
}

/// The refusal for a prefix outside the table. Every refusal names the
/// workaround that does work: an OpenAI-compatible endpoint behind
/// `openai/<model>` with `api_base`.
pub(super) fn unsupported(prefix: &str) -> Error {
    let reason = match prefix {
        "bedrock" | "bedrock_converse" | "sagemaker" | "sagemaker_chat" => format!(
            "LLM provider '{prefix}' (AWS) needs AWS Signature Version 4 request signing, which this build does not implement"
        ),
        "vertex_ai" | "vertex_ai_beta" => format!(
            "LLM provider '{prefix}' (Google Vertex AI) needs Google Cloud service-account authentication, which this build does not implement; gemini/<model> with GEMINI_API_KEY reaches Gemini directly"
        ),
        _ => format!("LLM provider '{prefix}' is not implemented in this build"),
    };
    Error::Unsupported(format!(
        "{reason}. For an OpenAI-compatible endpoint (a gateway or proxy in front of the provider), use openai/<model> with api_base"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn the_table_has_one_entry_per_prefix_with_valid_https_or_loopback_bases() {
        let mut seen = HashSet::new();
        for provider in &PROVIDERS {
            assert!(seen.insert(provider.prefix), "{}", provider.prefix);
            assert!(!provider.key_vars.is_empty(), "{}", provider.prefix);
            assert!(!provider.base_vars.is_empty(), "{}", provider.prefix);
            for name in provider.key_vars.iter().chain(provider.base_vars) {
                assert!(
                    name.bytes().all(|byte| byte.is_ascii_uppercase()
                        || byte.is_ascii_digit()
                        || byte == b'_'),
                    "{name}"
                );
            }
            let Some(base) = provider.base else {
                assert!(provider.prefix == "azure" || provider.key_optional);
                continue;
            };
            let url = url::Url::parse(base).unwrap();
            let host = url.host_str().unwrap();
            if provider.key_optional {
                assert_eq!((url.scheme(), host), ("http", "localhost"), "{base}");
            } else {
                assert_eq!(url.scheme(), "https", "{base}");
            }
            assert!(!base.ends_with('/'), "{base}");
            assert!(!base.ends_with("/chat/completions"), "{base}");
        }
        assert!(SUBSCRIPTIONS.iter().all(|prefix| find(prefix).is_none()));
    }

    #[test]
    fn local_servers_and_subscriptions_need_no_key_but_hosted_apis_do() {
        for prefix in [
            "ollama",
            "ollama_chat",
            "lm_studio",
            "hosted_vllm",
            "copilot",
            "chatgpt",
        ] {
            assert!(supported(prefix) && key_optional(prefix), "{prefix}");
        }
        for prefix in [
            "openai",
            "groq",
            "mistral",
            "together_ai",
            "perplexity",
            "azure",
        ] {
            assert!(supported(prefix) && !key_optional(prefix), "{prefix}");
        }
        assert!(!supported("bedrock") && !supported("vertex_ai") && !supported("cohere"));
    }

    #[test]
    fn refusals_name_the_openai_compatible_workaround() {
        for prefix in ["bedrock", "vertex_ai", "cohere"] {
            let Error::Unsupported(message) = unsupported(prefix) else {
                panic!("{prefix} must be unsupported");
            };
            assert!(message.contains(&format!("'{prefix}'")), "{message}");
            assert!(
                message.contains("openai/<model> with api_base"),
                "{message}"
            );
        }
        let Error::Unsupported(bedrock) = unsupported("bedrock") else {
            unreachable!()
        };
        assert!(bedrock.contains("Signature Version 4"), "{bedrock}");
        let Error::Unsupported(vertex) = unsupported("vertex_ai") else {
            unreachable!()
        };
        assert!(vertex.contains("gemini/<model>"), "{vertex}");
    }
}
