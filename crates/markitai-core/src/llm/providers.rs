//! The model prefixes this build sends over HTTP, and the explicit refusals
//! for prefixes it cannot serve.
//!
//! The first entries were checked against the provider's own documentation
//! on 2026-10-02, the OpenAI-compatible prefixes after them on 2026-10-11;
//! `docs/llm.md` lists the page behind every base URL, or says that a
//! provider documents none and LiteLLM's default is used. Key variables keep
//! the names the reference's LiteLLM reads, so an environment prepared for
//! the reference keeps working, with the provider's own documented name
//! first when the two differ.
use super::Protocol;
use crate::Error;
use std::collections::HashMap;

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
    /// How the base URL or the model sent depend on the model or environment.
    pub special: Special,
}

/// Base URLs and model names that a plain table entry cannot express.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Special {
    None,
    /// Amazon Bedrock's OpenAI-compatible ("Mantle") endpoint: the region
    /// comes from a `<region>/` model prefix or the AWS region variables,
    /// and some models are served under `/openai/v1`.
    BedrockMantle,
    /// Workers AI: the base URL names the Cloudflare account.
    Cloudflare,
    /// Hugging Face's router: `<provider>/<org>/<model>` is sent as
    /// `<org>/<model>:<provider>`.
    HuggingFace,
    /// An eight-character Baseten model ID is a dedicated deployment.
    Baseten,
    /// A DataRobot API endpoint gains the LLM gateway path.
    DataRobot,
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
        special: Special::None,
    }
}

/// An OpenAI-compatible API added on 2026-10-11.
const fn more(
    prefix: &'static str,
    base: &'static str,
    base_vars: &'static [&'static str],
    key_vars: &'static [&'static str],
) -> Provider {
    chat(prefix, base, base_vars, key_vars)
}

/// A server that listens wherever it was started and checks a key only
/// when configured to.
const fn local(
    prefix: &'static str,
    base: &'static str,
    base_vars: &'static [&'static str],
    key_vars: &'static [&'static str],
) -> Provider {
    Provider {
        key_optional: true,
        ..more(prefix, base, base_vars, key_vars)
    }
}

/// An endpoint with no fixed host: `api_base` or a base variable names it.
const fn hosted(
    prefix: &'static str,
    base_vars: &'static [&'static str],
    key_vars: &'static [&'static str],
) -> Provider {
    Provider {
        base: None,
        ..more(prefix, "", base_vars, key_vars)
    }
}

static PROVIDERS: [Provider; 94] = [
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
        // An `ANTHROPIC_AUTH_TOKEN` (a gateway's token) is sent as a bearer
        // token, as Anthropic's SDK and LiteLLM send it.
        key_vars: &["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
        key_optional: false,
        special: Special::None,
    },
    chat(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        &["GEMINI_API_BASE"],
        // Google's SDK reads `GOOGLE_API_KEY` first (see `held_key_variable`).
        &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
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
        &["OPENROUTER_API_KEY", "OR_API_KEY"],
    ),
    Provider {
        prefix: "azure",
        protocol: Protocol::Azure,
        base: None,
        base_vars: &["AZURE_API_BASE"],
        key_vars: &["AZURE_API_KEY"],
        key_optional: false,
        special: Special::None,
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
    // The Kimi Code membership's API (checked 2026-10-24). Its Anthropic
    // Messages endpoint, the one Kimi's guides give Claude Code: its Chat
    // Completions endpoint is reported to admit only the User-Agents of
    // listed coding agents, and Kimi counts a changed User-Agent as a
    // violation. `KIMI_API_KEY` is the name models.dev and OpenCode read,
    // `KIMI_CODE_BASE_URL` the one Kimi Code CLI reads.
    Provider {
        prefix: "kimi_code",
        protocol: Protocol::Anthropic,
        base: Some("https://api.kimi.com/coding/v1"),
        base_vars: &["KIMI_CODE_API_BASE", "KIMI_CODE_BASE_URL"],
        key_vars: &["KIMI_API_KEY"],
        key_optional: false,
        special: Special::None,
    },
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
        special: Special::None,
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
    // The further OpenAI-compatible prefixes LiteLLM routes (checked
    // 2026-10-11). Where a provider documents a newer host or key name than
    // LiteLLM 1.104.2, the documented one is used or listed first.
    more(
        "ai21_chat",
        "https://api.ai21.com/studio/v1",
        &["AI21_API_BASE"],
        &["AI21_API_KEY"],
    ),
    // LiteLLM sends Jamba models named `ai21/…` to the same chat API.
    more(
        "ai21",
        "https://api.ai21.com/studio/v1",
        &["AI21_API_BASE"],
        &["AI21_API_KEY"],
    ),
    more(
        "volcengine",
        "https://ark.cn-beijing.volces.com/api/v3",
        &["VOLCENGINE_API_BASE"],
        &["ARK_API_KEY", "VOLCENGINE_API_KEY"],
    ),
    more(
        "codestral",
        "https://codestral.mistral.ai/v1",
        &["CODESTRAL_API_BASE"],
        &["CODESTRAL_API_KEY"],
    ),
    more(
        "tencent",
        "https://tokenhub-intl.tencentcloudmaas.com/v1",
        &["TENCENT_API_BASE"],
        &["TOKENHUB_API_KEY", "TENCENT_API_KEY"],
    ),
    more(
        "friendliai",
        "https://api.friendli.ai/serverless/v1",
        &["FRIENDLI_API_BASE"],
        &["FRIENDLIAI_API_KEY", "FRIENDLI_TOKEN"],
    ),
    more(
        "galadriel",
        "https://api.galadriel.com/v1/verified",
        &["GALADRIEL_API_BASE"],
        &["GALADRIEL_API_KEY"],
    ),
    Provider {
        special: Special::Baseten,
        ..more(
            "baseten",
            "https://inference.baseten.co/v1",
            &["BASETEN_API_BASE"],
            &["BASETEN_API_KEY"],
        )
    },
    more(
        "meta",
        "https://api.meta.ai/v1",
        &["META_API_BASE"],
        &["MODEL_API_KEY", "META_API_KEY"],
    ),
    more(
        "featherless_ai",
        "https://api.featherless.ai/v1",
        &["FEATHERLESS_AI_API_BASE", "FEATHERLESS_API_BASE"],
        &["FEATHERLESS_API_KEY", "FEATHERLESS_AI_API_KEY"],
    ),
    more(
        "nscale",
        "https://inference.api.nscale.com/v1",
        &["NSCALE_API_BASE"],
        &["NSCALE_SERVICE_TOKEN", "NSCALE_API_KEY"],
    ),
    more(
        "dashscope",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        &["DASHSCOPE_API_BASE"],
        &["DASHSCOPE_API_KEY"],
    ),
    more(
        "qwencloud",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        &["QWENCLOUD_API_BASE"],
        &["QWENCLOUD_API_KEY", "DASHSCOPE_API_KEY"],
    ),
    more(
        "qwen_ai_platform",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        &["QWEN_AI_PLATFORM_API_BASE"],
        &["QWEN_AI_PLATFORM_API_KEY", "DASHSCOPE_API_KEY"],
    ),
    more(
        "modelscope",
        "https://api-inference.modelscope.cn/v1",
        &["MODELSCOPE_API_BASE"],
        &["MODELSCOPE_API_KEY"],
    ),
    more(
        "morph",
        "https://api.morphllm.com/v1",
        &["MORPH_API_BASE"],
        &["MORPH_API_KEY"],
    ),
    more(
        "inception",
        "https://api.inceptionlabs.ai/v1",
        &["INCEPTION_API_BASE"],
        &["INCEPTION_API_KEY"],
    ),
    more(
        "aiml",
        "https://api.aimlapi.com/v1",
        &["AIML_API_BASE"],
        &["AIML_API_KEY"],
    ),
    more(
        "edenai",
        "https://api.edenai.run/v3",
        &["EDENAI_API_BASE"],
        &["EDENAI_API_KEY"],
    ),
    more(
        "wandb",
        "https://api.inference.wandb.ai/v1",
        &["WANDB_API_BASE"],
        &["WANDB_API_KEY"],
    ),
    more(
        "publicai",
        "https://api.publicai.co/v1",
        &["PUBLICAI_API_BASE"],
        &["PUBLICAI_API_KEY"],
    ),
    more(
        "veniceai",
        "https://api.venice.ai/api/v1",
        &["VENICEAI_API_BASE"],
        &["VENICE_API_KEY", "VENICE_AI_API_KEY"],
    ),
    more(
        "xiaomi_mimo",
        "https://api.xiaomimimo.com/v1",
        &["XIAOMI_MIMO_API_BASE"],
        &["MIMO_API_KEY", "XIAOMI_MIMO_API_KEY"],
    ),
    more(
        "scaleway",
        "https://api.scaleway.ai/v1",
        &["SCALEWAY_API_BASE"],
        &["SCW_SECRET_KEY"],
    ),
    more(
        "synthetic",
        "https://api.synthetic.new/openai/v1",
        &["SYNTHETIC_API_BASE"],
        &["SYNTHETIC_API_KEY"],
    ),
    more(
        "apertis",
        "https://api.apertis.ai/v1",
        &["APERTIS_API_BASE"],
        &["STIMA_API_KEY"],
    ),
    more(
        "nano-gpt",
        "https://nano-gpt.com/api/v1",
        &["NANOGPT_API_BASE"],
        &["NANOGPT_API_KEY"],
    ),
    more(
        "poe",
        "https://api.poe.com/v1",
        &["POE_API_BASE"],
        &["POE_API_KEY"],
    ),
    more(
        "chutes",
        "https://llm.chutes.ai/v1",
        &["CHUTES_API_BASE"],
        &["CHUTES_API_KEY"],
    ),
    more(
        "abliteration",
        "https://api.abliteration.ai/v1",
        &["ABLITERATION_API_BASE"],
        &["ABLIT_KEY", "ABLITERATION_API_KEY"],
    ),
    more(
        "gmi",
        "https://api.gmi-serving.com/v1",
        &["GMI_API_BASE"],
        &["GMI_API_KEY"],
    ),
    more(
        "sarvam",
        "https://api.sarvam.ai/v1",
        &["SARVAM_API_BASE"],
        &["SARVAM_API_KEY"],
    ),
    more(
        "assemblyai",
        "https://llm-gateway.assemblyai.com/v1",
        &["ASSEMBLYAI_API_BASE"],
        &["ASSEMBLYAI_API_KEY"],
    ),
    more(
        "aihubmix",
        "https://aihubmix.com/v1",
        &["AIHUBMIX_API_BASE"],
        &["AIHUBMIX_API_KEY"],
    ),
    more(
        "crusoe",
        "https://api.inference.crusoecloud.com/v1",
        &["CRUSOE_API_BASE"],
        &["CRUSOE_API_KEY"],
    ),
    more(
        "darkbloom",
        "https://api.darkbloom.dev/v1",
        &["DARKBLOOM_API_BASE"],
        &["DARKBLOOM_API_KEY"],
    ),
    more(
        "neosantara",
        "https://api.neosantara.xyz/v1",
        &["NEOSANTARA_API_BASE"],
        &["NEOSANTARA_API_KEY"],
    ),
    more(
        "tensormesh",
        "https://serverless.tensormesh.ai/v1",
        &["TENSORMESH_SERVERLESS_BASE_URL"],
        &["TENSORMESH_INFERENCE_API_KEY"],
    ),
    more(
        "parasail",
        "https://api.parasail.io/v1",
        &["PARASAIL_API_BASE"],
        &["PARASAIL_API_KEY"],
    ),
    more(
        "libertai",
        "https://api.libertai.io/v1",
        &["LIBERTAI_API_BASE"],
        &["LIBERTAI_API_KEY"],
    ),
    more(
        "empiriolabs",
        "https://api.empiriolabs.ai/v1",
        &["EMPIRIOLABS_API_BASE"],
        &["EMPIRIOLABS_API_KEY"],
    ),
    more(
        "cognition",
        "https://api.cognition.ai/v1",
        &["COGNITION_API_BASE"],
        &["COGNITION_API_KEY"],
    ),
    more(
        "pinstripes",
        "https://api.pinstripes.io/v1",
        &["PINSTRIPES_API_BASE"],
        &["PINSTRIPES_API_KEY"],
    ),
    more(
        "scx-ai",
        "https://api.scx.ai/v1",
        &["SCX_API_BASE"],
        &["SCX_API_KEY"],
    ),
    more(
        "nadir",
        "https://api.getnadir.com/v1",
        &["NADIR_API_BASE"],
        &["NADIR_API_KEY"],
    ),
    more(
        "amazon_nova",
        "https://api.nova.amazon.com/v1",
        &["AMAZON_NOVA_API_BASE"],
        &["NOVA_API_KEY", "AMAZON_NOVA_API_KEY"],
    ),
    more(
        "minimax",
        "https://api.minimax.io/v1",
        &["MINIMAX_API_BASE"],
        &["MINIMAX_API_KEY"],
    ),
    more(
        "ovhcloud",
        "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        &["OVHCLOUD_API_BASE"],
        &["OVH_AI_ENDPOINTS_ACCESS_TOKEN", "OVHCLOUD_API_KEY"],
    ),
    more(
        "compactifai",
        "https://api.compactif.ai/v1",
        &["COMPACTIFAI_API_BASE"],
        &["COMPACTIFAI_API_KEY"],
    ),
    more(
        "gradient_ai",
        "https://inference.do-ai.run/v1",
        &["GRADIENT_AI_API_BASE"],
        &[
            "MODEL_ACCESS_KEY",
            "GRADIENT_AI_API_KEY",
            "DIGITALOCEAN_TOKEN",
        ],
    ),
    more(
        "helicone",
        "https://ai-gateway.helicone.ai",
        &["HELICONE_API_BASE"],
        &["HELICONE_API_KEY"],
    ),
    more(
        "vercel_ai_gateway",
        "https://ai-gateway.vercel.sh/v1",
        &["VERCEL_AI_GATEWAY_API_BASE"],
        &[
            "AI_GATEWAY_API_KEY",
            "VERCEL_AI_GATEWAY_API_KEY",
            "VERCEL_OIDC_TOKEN",
        ],
    ),
    more(
        "cometapi",
        "https://api.cometapi.com/v1",
        &["COMETAPI_API_BASE", "COMETAPI_BASE_URL"],
        &["COMETAPI_KEY", "COMETAPI_API_KEY"],
    ),
    // Cohere's Compatibility API serves the models LiteLLM's native Cohere
    // route reaches, under the same key.
    more(
        "cohere",
        "https://api.cohere.ai/compatibility/v1",
        &["COHERE_API_BASE"],
        &["COHERE_API_KEY", "CO_API_KEY"],
    ),
    more(
        "cohere_chat",
        "https://api.cohere.ai/compatibility/v1",
        &["COHERE_API_BASE"],
        &["COHERE_API_KEY", "CO_API_KEY"],
    ),
    Provider {
        special: Special::HuggingFace,
        ..more(
            "huggingface",
            "https://router.huggingface.co/v1",
            &["HF_API_BASE", "HUGGINGFACE_API_BASE"],
            &["HF_TOKEN", "HUGGINGFACE_API_KEY"],
        )
    },
    Provider {
        special: Special::BedrockMantle,
        ..more(
            "bedrock_mantle",
            "https://bedrock-mantle.us-east-1.api.aws/v1",
            &["BEDROCK_MANTLE_API_BASE"],
            &["AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_MANTLE_API_KEY"],
        )
    },
    Provider {
        base: None,
        special: Special::Cloudflare,
        ..more(
            "cloudflare",
            "",
            &["CLOUDFLARE_API_BASE"],
            &["CLOUDFLARE_API_KEY", "CLOUDFLARE_API_TOKEN"],
        )
    },
    Provider {
        special: Special::DataRobot,
        ..more(
            "datarobot",
            "https://app.datarobot.com/api/v2/genai/llmgw",
            &["DATAROBOT_ENDPOINT"],
            &["DATAROBOT_API_TOKEN"],
        )
    },
    // Endpoints of the user's own workspace, account or app.
    hosted(
        "databricks",
        &["DATABRICKS_API_BASE"],
        &["DATABRICKS_TOKEN", "DATABRICKS_API_KEY"],
    ),
    hosted(
        "heroku",
        &["HEROKU_API_BASE"],
        &["INFERENCE_KEY", "HEROKU_API_KEY"],
    ),
    hosted(
        "azure_ai",
        &["AZURE_AI_API_BASE"],
        &["AZURE_INFERENCE_CREDENTIAL", "AZURE_AI_API_KEY"],
    ),
    Provider {
        key_optional: true,
        ..hosted(
            "litellm_proxy",
            &["LITELLM_PROXY_API_BASE"],
            &["LITELLM_PROXY_API_KEY"],
        )
    },
    Provider {
        key_optional: true,
        ..hosted(
            "openai_like",
            &["OPENAI_LIKE_API_BASE"],
            &["OPENAI_LIKE_API_KEY"],
        )
    },
    // Local servers at their documented default addresses.
    local(
        "llamafile",
        "http://127.0.0.1:8080/v1",
        &["LLAMAFILE_API_BASE"],
        &["LLAMAFILE_API_KEY"],
    ),
    local(
        "docker_model_runner",
        "http://localhost:12434/engines/v1",
        &["DOCKER_MODEL_RUNNER_API_BASE"],
        &["DOCKER_MODEL_RUNNER_API_KEY"],
    ),
    local(
        "lemonade",
        "http://localhost:13305/api/v1",
        &["LEMONADE_API_BASE"],
        &["LEMONADE_API_KEY"],
    ),
    local(
        "xinference",
        "http://127.0.0.1:9997/v1",
        &["XINFERENCE_API_BASE"],
        &["XINFERENCE_API_KEY"],
    ),
    local(
        "oobabooga",
        "http://127.0.0.1:5000/v1",
        &["OOBABOOGA_API_BASE"],
        &["OOBABOOGA_API_KEY"],
    ),
];

/// The key variable a request reads, among `variables`: the first one that
/// holds a value, except that `GOOGLE_API_KEY` comes before the others, as
/// Google's SDK and LiteLLM read it before `GEMINI_API_KEY`. `None` when none
/// holds one; messages then name the first variable, the documented one.
pub(crate) fn held_key_variable(
    variables: &[&'static str],
    env: &HashMap<String, String>,
) -> Option<&'static str> {
    held_key_variable_by(variables, |name| {
        env.get(name).is_some_and(|value| !value.trim().is_empty())
    })
}

/// [`held_key_variable`] with the caller's test of whether a variable holds
/// a value.
pub(crate) fn held_key_variable_by(
    variables: &[&'static str],
    held: impl Fn(&str) -> bool,
) -> Option<&'static str> {
    variables
        .iter()
        .find(|name| **name == "GOOGLE_API_KEY" && held(name))
        .or_else(|| variables.iter().find(|name| held(name)))
        .copied()
}

/// What provider settings need to know about one HTTP prefix, without the
/// request protocol itself.
pub(crate) struct Listing {
    pub prefix: &'static str,
    pub base: Option<&'static str>,
    pub base_vars: &'static [&'static str],
    pub key_vars: &'static [&'static str],
    pub key_optional: bool,
    /// The provider speaks OpenAI Chat Completions (and may list `/models`).
    pub openai_compatible: bool,
}

/// Every HTTP prefix, in table order.
pub(crate) fn listings() -> impl Iterator<Item = Listing> {
    PROVIDERS.iter().map(|provider| Listing {
        prefix: provider.prefix,
        base: provider.base,
        base_vars: provider.base_vars,
        key_vars: provider.key_vars,
        key_optional: provider.key_optional,
        openai_compatible: provider.protocol == Protocol::Chat,
    })
}

/// One region of a provider whose keys are issued per region: its endpoint
/// and the site (or console region) that issues the keys it accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    pub base: &'static str,
    pub console: &'static str,
}

/// A provider whose key works only at the endpoint of the region that issued
/// it (each answers a key it does not know with HTTP 401), its regions with
/// the documented default first, and what a refusal adds about keys of the
/// same look that belong elsewhere.
struct Regional {
    prefix: &'static str,
    regions: &'static [Region],
    sibling: Option<&'static str>,
}

const fn region(base: &'static str, console: &'static str) -> Region {
    Region { base, console }
}

/// The providers whose documentation says so (checked 2026-10-26, pages in
/// `docs/llm.md`). Moonshot's open platform issues keys on platform.kimi.ai
/// (once platform.moonshot.ai) and platform.kimi.com (once
/// platform.moonshot.cn); Kimi Code's membership has a kimi.com and a kimi.ai
/// endpoint. MiniMax keys are Global or Mainland China ones. Model Studio
/// keys belong to a region; its Coding Plan keys (`sk-sp-`) work only at the
/// plan's own endpoints, which list models without checking the key, so they
/// are named, not tried.
const REGIONAL: &[Regional] = &[
    Regional {
        prefix: "moonshot",
        regions: &[
            region("https://api.moonshot.ai/v1", "platform.kimi.ai"),
            region("https://api.moonshot.cn/v1", "platform.kimi.com"),
        ],
        sibling: Some("a Kimi Code membership key is used as kimi_code/<model>"),
    },
    Regional {
        prefix: "kimi_code",
        regions: &[
            region("https://api.kimi.com/coding/v1", "kimi.com"),
            region("https://api.kimi.ai/coding/v1", "kimi.ai"),
        ],
        sibling: Some("a Kimi platform key is used as moonshot/<model>"),
    },
    Regional {
        prefix: "minimax",
        regions: &[
            region("https://api.minimax.io/v1", "platform.minimax.io"),
            region("https://api.minimax.cn/v1", "platform.minimax.cn"),
        ],
        sibling: None,
    },
    Regional {
        prefix: "dashscope",
        regions: &[
            region(
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "Model Studio's Beijing region",
            ),
            region(
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "Model Studio's Singapore region",
            ),
            region(
                "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
                "Model Studio's US (Virginia) region",
            ),
            region(
                "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
                "Model Studio's Hong Kong region",
            ),
        ],
        sibling: Some(
            "a Coding Plan key (sk-sp-…) works only at the Coding Plan's endpoints, in coding tools",
        ),
    },
];

fn regional(prefix: &str) -> Option<&'static Regional> {
    REGIONAL.iter().find(|entry| entry.prefix == prefix)
}

/// The regions of a provider that issues keys per region, the documented
/// default first; empty for any other provider.
pub(crate) fn regions(prefix: &str) -> &'static [Region] {
    regional(prefix).map_or(&[], |entry| entry.regions)
}

fn host(url: &str) -> Option<String> {
    url::Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
}

/// What to try when a regional endpoint refuses a key, for an English
/// message: the region the endpoint (a base or a request URL) serves, the
/// others' endpoints, and the keys of the same look that belong elsewhere
/// (Moonshot's open platform and Kimi Code, Model Studio's Coding Plan).
/// `None` elsewhere.
pub(crate) fn region_hint(prefix: &str, endpoint: &str) -> Option<String> {
    let entry = regional(prefix)?;
    let at = host(endpoint)?;
    let here = entry
        .regions
        .iter()
        .find(|region| host(region.base).as_deref() == Some(at.as_str()))?;
    let mut others = entry
        .regions
        .iter()
        .filter(|region| *region != here)
        .map(|region| {
            format!(
                "a key from {} needs api_base {}",
                region.console, region.base
            )
        })
        .collect::<Vec<_>>();
    let last = entry.sibling.map(str::to_owned).or_else(|| others.pop());
    let others = match last {
        Some(last) if !others.is_empty() => format!("{}, and {last}", others.join(", ")),
        Some(last) => last,
        None => String::new(),
    };
    Some(format!(
        "{at} accepts keys from {} only: {others}",
        here.console
    ))
}

/// Prefixes served by an installed official runtime instead of HTTP.
pub(super) const SUBSCRIPTIONS: [&str; 3] = ["copilot", "claude-agent", "chatgpt"];

/// Cloud platforms with their own authentication (`llm/cloud.rs`).
pub(super) const CLOUDS: [&str; 3] = ["bedrock", "vertex_ai", "vertex_ai_beta"];

pub(super) fn find(prefix: &str) -> Option<&'static Provider> {
    PROVIDERS.iter().find(|provider| provider.prefix == prefix)
}

/// Whether this build can route a model with this prefix.
pub(crate) fn supported(prefix: &str) -> bool {
    find(prefix).is_some() || SUBSCRIPTIONS.contains(&prefix) || CLOUDS.contains(&prefix)
}

/// Whether a deployment needs no API key to be usable: local servers and the
/// subscription runtimes, which carry their own sign-in.
pub(super) fn key_optional(prefix: &str) -> bool {
    SUBSCRIPTIONS.contains(&prefix) || find(prefix).is_some_and(|provider| provider.key_optional)
}

fn aws_region(value: &str) -> bool {
    static REGION: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"^[a-z]{2}(?:-gov|-iso[a-z]?)?-[a-z]+-[0-9]{1,2}$")
            .expect("valid AWS region pattern")
    });
    REGION.is_match(value)
}

/// The model name a deployment sends: the text after the prefix, except
/// where the provider names models another way.
pub(super) fn wire_model(provider: &Provider, model: &str) -> String {
    match provider.special {
        // `bedrock_mantle/<region>/<model>` chooses the region.
        Special::BedrockMantle => match model.split_once('/') {
            Some((region, rest)) if aws_region(region) => rest.to_owned(),
            _ => model.to_owned(),
        },
        // `huggingface/<provider>/<org>/<model>` asks the router for one
        // inference provider; `huggingface/<org>/<model>` lets it choose.
        Special::HuggingFace => match model.split_once('/') {
            Some((inference, rest)) if rest.contains('/') => format!("{rest}:{inference}"),
            _ => model.to_owned(),
        },
        _ => model.to_owned(),
    }
}

/// The base URL used when neither `api_base` nor a base variable names one.
pub(super) fn default_base(
    provider: &Provider,
    model: &str,
    env: &HashMap<String, String>,
    allow_environment: bool,
) -> crate::Result<Option<String>> {
    let variable = |names: &[&str]| {
        names.iter().find_map(|name| {
            allow_environment
                .then(|| env.get(*name))
                .flatten()
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
    };
    match provider.special {
        Special::BedrockMantle => {
            let region = match model.split_once('/') {
                Some((region, _)) if aws_region(region) => region.to_owned(),
                _ => variable(&["BEDROCK_MANTLE_REGION", "AWS_REGION_NAME", "AWS_REGION"])
                    .unwrap_or_else(|| "us-east-1".into()),
            };
            if !aws_region(&region) {
                return Err(Error::Config(
                    "LLM provider 'bedrock_mantle' needs a valid AWS region".into(),
                ));
            }
            // Per model card (LiteLLM 1.104.2's price map flags the same
            // models): these are served on the OpenAI path, the rest on /v1.
            let model = wire_model(provider, model);
            let openai = [
                "openai.gpt-5",
                "openai.gpt-6",
                "openai.gpt-daybreak",
                "google.gemma-4",
                "xai.grok-4",
            ]
            .iter()
            .any(|prefix| model.starts_with(prefix));
            Ok(Some(format!(
                "https://bedrock-mantle.{region}.api.aws/{}",
                if openai { "openai/v1" } else { "v1" }
            )))
        }
        Special::Cloudflare => {
            let Some(account) = variable(&["CLOUDFLARE_ACCOUNT_ID"]) else {
                return Err(Error::Config(
                    "LLM provider 'cloudflare' needs CLOUDFLARE_ACCOUNT_ID, api_base or CLOUDFLARE_API_BASE".into(),
                ));
            };
            if account.len() != 32 || !account.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(Error::Config(
                    "CLOUDFLARE_ACCOUNT_ID must be a 32-character account ID".into(),
                ));
            }
            Ok(Some(format!(
                "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
            )))
        }
        // An eight-character model ID names a dedicated deployment.
        Special::Baseten
            if model.len() == 8 && model.bytes().all(|byte| byte.is_ascii_alphanumeric()) =>
        {
            Ok(Some(format!(
                "https://model-{model}.api.baseten.co/environments/production/sync/v1"
            )))
        }
        _ => Ok(provider.base.map(str::to_owned)),
    }
}

/// Whether `azure_ai/<model>` is Claude on Microsoft Foundry, which is
/// served by the Anthropic Messages API: a name containing `claude`, the
/// test LiteLLM (Python markitai) routes by.
pub(crate) fn foundry_claude(prefix: &str, model: &str) -> bool {
    prefix == "azure_ai" && model.to_ascii_lowercase().contains("claude")
}

/// The Messages endpoint of Claude on Foundry, as LiteLLM forms it: a path
/// that ends in `/v1/messages` is kept; otherwise everything after the first
/// `/anthropic` is cut (or `/anthropic` is appended) and `/v1/messages`
/// follows.
pub(super) fn foundry_messages_base(base: String) -> String {
    let Ok(mut url) = url::Url::parse(&base) else {
        return base;
    };
    let path = url.path().trim_end_matches('/');
    let path = if path.ends_with("/v1/messages") {
        path.to_owned()
    } else if let Some(at) = path.find("/anthropic") {
        format!("{}/v1/messages", &path[..at + "/anthropic".len()])
    } else {
        format!("{path}/anthropic/v1/messages")
    };
    url.set_path(&path);
    url.into()
}

/// A DataRobot API endpoint (`…/api/v2`, as `DATAROBOT_ENDPOINT` usually
/// holds) gains the LLM gateway path, as LiteLLM adds it.
pub(super) fn normalize_base(provider: &Provider, base: String) -> String {
    if provider.special != Special::DataRobot {
        return base;
    }
    let Ok(mut url) = url::Url::parse(&base) else {
        return base;
    };
    let path = url.path().trim_end_matches('/').to_owned();
    if path.is_empty() {
        url.set_path("/api/v2/genai/llmgw");
    } else if path.ends_with("/api/v2") {
        url.set_path(&format!("{path}/genai/llmgw"));
    } else {
        return base;
    }
    url.into()
}

/// Prefixes LiteLLM still knows whose services have ended or moved.
const RETIRED: [(&str, &str); 9] = [
    ("github", "GitHub Models was retired on 2026-07-30"),
    (
        "meta_llama",
        "Meta's Llama API was replaced by the Meta Model API, which is meta/<model>",
    ),
    (
        "lambda_ai",
        "Lambda's Inference API was shut down on 2025-09-25",
    ),
    (
        "hyperbolic",
        "Hyperbolic retired its serverless inference API",
    ),
    ("anyscale", "Anyscale Endpoints no longer exist"),
    ("empower", "Empower no longer documents an API"),
    ("clarifai", "Clarifai's API domain no longer resolves"),
    (
        "llamagate",
        "LlamaGate's domain no longer serves the provider, so no key is sent there",
    ),
    ("v0", "Vercel no longer documents the v0 Model API"),
];

/// The refusal for a prefix outside the table. Every refusal names the
/// workaround that does work: an OpenAI-compatible endpoint behind
/// `openai/<model>` with `api_base`.
pub(super) fn unsupported(prefix: &str) -> Error {
    if let Some((_, why)) = RETIRED.iter().find(|(name, _)| *name == prefix) {
        return Error::Unsupported(format!(
            "LLM provider '{prefix}' is not routed: {why}. For an OpenAI-compatible endpoint (a gateway or proxy in front of the provider), use openai/<model> with api_base"
        ));
    }
    let reason = match prefix {
        "sagemaker" | "sagemaker_chat" | "sagemaker_nova" => format!(
            "LLM provider '{prefix}' (AWS SageMaker endpoints) is not implemented in this build; a SageMaker endpoint with an OpenAI-compatible route works as openai/<model> with its api_base, and Bedrock models as bedrock/<model>"
        ),
        "github_copilot" => format!(
            "LLM provider '{prefix}' signs in through LiteLLM; this build reaches GitHub Copilot as copilot/<model> through the official Copilot runtime"
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
                assert!(
                    matches!(
                        provider.prefix,
                        "azure" | "databricks" | "heroku" | "azure_ai"
                    ) || provider.key_optional
                        || provider.special == Special::Cloudflare,
                    "{}",
                    provider.prefix
                );
                continue;
            };
            let url = url::Url::parse(base).unwrap();
            let host = url.host_str().unwrap();
            if provider.key_optional {
                assert_eq!(url.scheme(), "http", "{base}");
                assert!(matches!(host, "localhost" | "127.0.0.1"), "{base}");
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
        assert!(supported("bedrock") && supported("vertex_ai") && !supported("watsonx"));
        assert!(!key_optional("bedrock") && !key_optional("vertex_ai"));
    }

    #[test]
    fn every_openai_compatible_prefix_litellm_routes_is_served_or_refused_with_a_reason() {
        // LiteLLM 1.104.2's OpenAI-compatible prefixes (constants.py and its
        // JSON-configured providers), minus the subscription logins.
        for prefix in [
            "groq",
            "nvidia_nim",
            "cerebras",
            "baseten",
            "sambanova",
            "ai21_chat",
            "ai21",
            "volcengine",
            "codestral",
            "deepseek",
            "tencent",
            "deepinfra",
            "perplexity",
            "xinference",
            "xai",
            "zai",
            "together_ai",
            "fireworks_ai",
            "friendliai",
            "azure_ai",
            "litellm_proxy",
            "hosted_vllm",
            "llamafile",
            "lm_studio",
            "galadriel",
            "novita",
            "meta",
            "publicai",
            "synthetic",
            "tensormesh",
            "apertis",
            "nano-gpt",
            "poe",
            "chutes",
            "parasail",
            "libertai",
            "featherless_ai",
            "nscale",
            "nebius",
            "dashscope",
            "qwencloud",
            "qwen_ai_platform",
            "modelscope",
            "moonshot",
            "helicone",
            "morph",
            "inception",
            "vercel_ai_gateway",
            "aiml",
            "edenai",
            "wandb",
            "cometapi",
            "docker_model_runner",
            "pinstripes",
            "darkbloom",
            "cognition",
            "scx-ai",
            "veniceai",
            "xiaomi_mimo",
            "scaleway",
            "abliteration",
            "gmi",
            "sarvam",
            "assemblyai",
            "aihubmix",
            "crusoe",
            "neosantara",
            "empiriolabs",
            "mistral",
            "openai_like",
            "lemonade",
            "datarobot",
            "gradient_ai",
            "heroku",
            "nadir",
            "amazon_nova",
            "minimax",
            "ovhcloud",
            "compactifai",
            "oobabooga",
            "cohere",
            "cohere_chat",
            "huggingface",
            "databricks",
            "cloudflare",
            "bedrock_mantle",
        ] {
            assert!(supported(prefix), "{prefix}");
        }
        for prefix in [
            "github",
            "meta_llama",
            "lambda_ai",
            "hyperbolic",
            "anyscale",
            "empower",
            "clarifai",
            "llamagate",
            "v0",
        ] {
            let Error::Unsupported(message) = unsupported(prefix) else {
                unreachable!()
            };
            assert!(!supported(prefix), "{prefix}");
            assert!(message.contains("is not routed: "), "{message}");
        }
        assert_eq!(listings().count(), PROVIDERS.len());
    }

    #[test]
    fn claude_on_foundry_reaches_the_messages_endpoint_litellm_forms() {
        assert!(foundry_claude("azure_ai", "Claude-Sonnet-5-5"));
        assert!(foundry_claude("azure_ai", "my-claude-deployment"));
        assert!(!foundry_claude("azure_ai", "gpt-x"));
        assert!(!foundry_claude("anthropic", "claude-x"));
        let resource = "https://example-resource.services.ai.azure.com";
        for (base, path) in [
            ("", "/anthropic/v1/messages"),
            ("/", "/anthropic/v1/messages"),
            ("/anthropic", "/anthropic/v1/messages"),
            ("/anthropic/v1", "/anthropic/v1/messages"),
            ("/anthropic/v1/messages/", "/anthropic/v1/messages"),
            ("/gateway/v1/messages", "/gateway/v1/messages"),
            ("/models", "/models/anthropic/v1/messages"),
        ] {
            assert_eq!(
                foundry_messages_base(format!("{resource}{base}")),
                format!("{resource}{path}"),
                "{base}"
            );
        }
    }

    #[test]
    fn regions_accounts_and_model_routes_follow_the_providers_conventions() {
        let env = |pairs: &[(&str, &str)]| {
            pairs
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect::<HashMap<_, _>>()
        };
        let mantle = find("bedrock_mantle").unwrap();
        assert_eq!(
            wire_model(mantle, "us-west-2/openai.gpt-oss-120b"),
            "openai.gpt-oss-120b"
        );
        assert_eq!(
            default_base(mantle, "us-west-2/openai.gpt-oss-120b", &env(&[]), true).unwrap(),
            Some("https://bedrock-mantle.us-west-2.api.aws/v1".into())
        );
        assert_eq!(
            default_base(
                mantle,
                "openai.gpt-5.5",
                &env(&[("AWS_REGION", "eu-central-1")]),
                true
            )
            .unwrap(),
            Some("https://bedrock-mantle.eu-central-1.api.aws/openai/v1".into())
        );
        // An environment the deployment may not read leaves the default region.
        assert_eq!(
            default_base(
                mantle,
                "google.gemma-4-31b",
                &env(&[("AWS_REGION", "eu-central-1")]),
                false
            )
            .unwrap(),
            Some("https://bedrock-mantle.us-east-1.api.aws/openai/v1".into())
        );
        assert!(default_base(mantle, "m", &env(&[("AWS_REGION", "../evil")]), true).is_err());
        let cloudflare = find("cloudflare").unwrap();
        assert!(default_base(cloudflare, "@cf/meta/llama", &env(&[]), true).is_err());
        let account = "0123456789abcdef0123456789abcdef";
        assert_eq!(
            default_base(
                cloudflare,
                "@cf/meta/llama",
                &env(&[("CLOUDFLARE_ACCOUNT_ID", account)]),
                true
            )
            .unwrap(),
            Some(format!(
                "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
            ))
        );
        assert!(
            default_base(
                cloudflare,
                "m",
                &env(&[("CLOUDFLARE_ACCOUNT_ID", "x/../y")]),
                true
            )
            .is_err()
        );
        let hf = find("huggingface").unwrap();
        assert_eq!(
            wire_model(hf, "together/deepseek-ai/DeepSeek-R1"),
            "deepseek-ai/DeepSeek-R1:together"
        );
        assert_eq!(
            wire_model(hf, "meta-llama/Llama-3.3-70B-Instruct"),
            "meta-llama/Llama-3.3-70B-Instruct"
        );
        let baseten = find("baseten").unwrap();
        assert_eq!(
            default_base(baseten, "abcd1234", &env(&[]), true).unwrap(),
            Some("https://model-abcd1234.api.baseten.co/environments/production/sync/v1".into())
        );
        assert_eq!(
            default_base(baseten, "deepseek-ai/DeepSeek-V3", &env(&[]), true).unwrap(),
            Some("https://inference.baseten.co/v1".into())
        );
        let datarobot = find("datarobot").unwrap();
        for (base, expected) in [
            (
                "https://app.datarobot.com",
                "https://app.datarobot.com/api/v2/genai/llmgw",
            ),
            (
                "https://app.eu.datarobot.com/api/v2/",
                "https://app.eu.datarobot.com/api/v2/genai/llmgw",
            ),
            (
                "https://app.datarobot.com/api/v2/deployments/abc",
                "https://app.datarobot.com/api/v2/deployments/abc",
            ),
        ] {
            assert_eq!(normalize_base(datarobot, base.into()), expected);
        }
    }

    #[test]
    fn refusals_name_the_openai_compatible_workaround() {
        for prefix in ["sagemaker", "watsonx", "replicate"] {
            let Error::Unsupported(message) = unsupported(prefix) else {
                panic!("{prefix} must be unsupported");
            };
            assert!(message.contains(&format!("'{prefix}'")), "{message}");
            assert!(
                message.contains("openai/<model> with api_base"),
                "{message}"
            );
        }
        let Error::Unsupported(sagemaker) = unsupported("sagemaker") else {
            unreachable!()
        };
        assert!(sagemaker.contains("bedrock/<model>"), "{sagemaker}");
    }
}
