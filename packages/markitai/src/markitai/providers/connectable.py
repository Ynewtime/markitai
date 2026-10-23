"""The providers the serve workbench can connect, as this build routes them.

markitai sends every API provider's requests through LiteLLM, so a provider
is offered when LiteLLM routes its model prefix (``routable_providers``) and
this table knows how a connection to it is made: the address LiteLLM takes
as ``api_base`` (in LiteLLM's own form, which is not always the OpenAI-style
``/v1`` base: Cohere takes the chat URL, oobabooga and Heroku their host),
the variable LiteLLM reads the key from, whether a key is needed at all, and
how the provider lists its models.

Checked on 2026-10-11 against LiteLLM 1.104.2 by sending a request for every
prefix through a LiteLLM ``Router`` with the transport stubbed (the URL and
the key header each one really gets), and against the addresses markitai-rust
documents for the same providers (its ``docs/llm.md`` names the provider page
behind each). Where LiteLLM's own default is missing, another host, or one
the provider has moved away from, ``pin_base`` makes a new connection store
the documented address, so requests go where the card says: without it
``ai21/``, ``lm_studio/`` and ``xinference/`` would send the key to
api.openai.com, and Perplexity models would go to the Sonar endpoint
Perplexity announced it would retire instead of its Router API.

Left out on purpose: ``gradient_ai`` (LiteLLM 1.104.2 refuses every request
the router sends it: "does not support parameter 'max_retries'"), ``morph``
(LiteLLM accepts only messages and stream for it, and markitai sends
``max_tokens`` for the models LiteLLM knows, Morph's included), and the
subscription runtimes (``claude-agent``, ``copilot``, ``chatgpt``), which
markitai routes itself and ``discovery.detect_provider_connections`` offers.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import cache
from typing import Literal

from markitai.constants import PROVIDER_API_KEY_ENV as _KEYS

Discovery = Literal["builtin", "openai", "cohere", "manual"]


@dataclass(frozen=True, slots=True)
class Connectable:
    """How one provider is connected and routed."""

    provider: str
    # The address in LiteLLM's form; None when every server has its own (one
    # must be entered), or the provider takes none (Bedrock, Vertex AI)
    base: str | None
    # The variable LiteLLM reads the key from when none is stored; None when
    # it reads none for this prefix
    key_variable: str | None
    key_optional: bool = False
    # "builtin": discovery.py's own branch; "openai": GET <base>/<models_path>
    # with a bearer key; "cohere": Cohere's native list; "manual": the
    # provider publishes no model list, model IDs are entered by hand
    discovery: Discovery = "openai"
    models_path: str = "models"
    pin_base: bool = False
    # A module the route needs besides LiteLLM (Vertex AI signs with
    # google-auth, which markitai does not install)
    requires: str | None = None
    # Takes the server's own cloud credentials, no address (the workbench
    # knows Bedrock and Vertex AI by name)
    cloud: bool = False


CONNECTABLE: tuple[Connectable, ...] = (
    # The providers markitai offered before 1.3.0; discovery.py lists their
    # models in their own formats
    Connectable(
        "openai", "https://api.openai.com/v1", _KEYS["openai"], discovery="builtin"
    ),
    Connectable(
        "anthropic",
        "https://api.anthropic.com",
        _KEYS["anthropic"],
        discovery="builtin",
    ),
    Connectable(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        _KEYS["gemini"],
        discovery="builtin",
    ),
    Connectable(
        "deepseek", "https://api.deepseek.com", _KEYS["deepseek"], discovery="builtin"
    ),
    Connectable(
        "openrouter",
        "https://openrouter.ai/api/v1",
        _KEYS["openrouter"],
        discovery="builtin",
    ),
    Connectable("azure", None, "AZURE_API_KEY", discovery="builtin"),
    Connectable(
        "ollama",
        "http://127.0.0.1:11434",
        "OLLAMA_API_KEY",
        key_optional=True,
        discovery="builtin",
    ),
    Connectable("custom", None, None, discovery="builtin"),
    # Model makers' own APIs
    Connectable("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    Connectable("ai21", "https://api.ai21.com/studio/v1", None, pin_base=True),
    Connectable("amazon_nova", "https://api.nova.amazon.com/v1", "AMAZON_NOVA_API_KEY"),
    Connectable("codestral", "https://codestral.mistral.ai/v1", "CODESTRAL_API_KEY"),
    Connectable("cognition", "https://api.cognition.ai/v1", "COGNITION_API_KEY"),
    # LiteLLM's Cohere route posts to this URL itself (v2 chat)
    Connectable(
        "cohere", "https://api.cohere.com/v2/chat", "COHERE_API_KEY", discovery="cohere"
    ),
    Connectable("compactifai", "https://api.compactif.ai/v1", "COMPACTIFAI_API_KEY"),
    Connectable("inception", "https://api.inceptionlabs.ai/v1", None),
    Connectable("meta", "https://api.meta.ai/v1", "META_API_KEY"),
    Connectable("minimax", "https://api.minimax.io/v1", "MINIMAX_API_KEY"),
    Connectable("moonshot", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    # Perplexity announced the end of Sonar chat completions for 2026-09-27;
    # LiteLLM still posts there, the Router API keeps the same schema
    Connectable(
        "perplexity",
        "https://api.perplexity.ai/router/v1",
        "PERPLEXITYAI_API_KEY",
        discovery="manual",
        pin_base=True,
    ),
    Connectable("sarvam", "https://api.sarvam.ai/v1", "SARVAM_API_KEY"),
    Connectable("xai", "https://api.x.ai/v1", "XAI_API_KEY"),
    Connectable("xiaomi_mimo", "https://api.xiaomimimo.com/v1", "XIAOMI_MIMO_API_KEY"),
    Connectable(
        "zai", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY", discovery="manual"
    ),
    # Cloud platforms
    Connectable(
        "dashscope",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
    ),
    Connectable(
        "bedrock",
        None,
        "AWS_BEARER_TOKEN_BEDROCK",
        key_optional=True,
        discovery="manual",
        cloud=True,
    ),
    Connectable("azure_ai", None, "AZURE_AI_API_KEY"),
    Connectable(
        "bedrock_mantle",
        "https://bedrock-mantle.us-east-1.api.aws/v1",
        "BEDROCK_MANTLE_API_KEY",
    ),
    # https://api.cloudflare.com/client/v4/accounts/<account>/ai/v1
    Connectable("cloudflare", None, "CLOUDFLARE_API_KEY"),
    # https://<workspace>/serving-endpoints
    Connectable("databricks", None, "DATABRICKS_API_KEY"),
    # LiteLLM adds the LLM gateway path to the host
    Connectable(
        "datarobot",
        "https://app.datarobot.com",
        "DATAROBOT_API_TOKEN",
        models_path="api/v2/genai/llmgw/models",
    ),
    # https://us.inference.heroku.com: LiteLLM adds /v1
    Connectable("heroku", None, "HEROKU_API_KEY", models_path="v1/models"),
    Connectable(
        "ovhcloud",
        "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        "OVHCLOUD_API_KEY",
    ),
    Connectable(
        "qwen_ai_platform",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "QWEN_AI_PLATFORM_API_KEY",
    ),
    Connectable(
        "qwencloud",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "QWENCLOUD_API_KEY",
    ),
    Connectable("scaleway", "https://api.scaleway.ai/v1", "SCW_SECRET_KEY"),
    Connectable(
        "tencent", "https://tokenhub-intl.tencentcloudmaas.com/v1", "TENCENT_API_KEY"
    ),
    Connectable(
        "vertex_ai",
        None,
        None,
        key_optional=True,
        discovery="manual",
        requires="google.auth",
        cloud=True,
    ),
    Connectable(
        "volcengine", "https://ark.cn-beijing.volces.com/api/v3", "VOLCENGINE_API_KEY"
    ),
    # Inference hosts
    Connectable(
        "abliteration", "https://api.abliteration.ai/v1", "ABLITERATION_API_KEY"
    ),
    Connectable("baseten", "https://inference.baseten.co/v1", "BASETEN_API_KEY"),
    Connectable("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    Connectable("chutes", "https://llm.chutes.ai/v1", "CHUTES_API_KEY"),
    Connectable(
        "crusoe",
        "https://api.inference.crusoecloud.com/v1",
        "CRUSOE_API_KEY",
        pin_base=True,
    ),
    Connectable("darkbloom", "https://api.darkbloom.dev/v1", "DARKBLOOM_API_KEY"),
    Connectable(
        "deepinfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"
    ),
    Connectable("empiriolabs", "https://api.empiriolabs.ai/v1", "EMPIRIOLABS_API_KEY"),
    Connectable(
        "featherless_ai", "https://api.featherless.ai/v1", "FEATHERLESS_AI_API_KEY"
    ),
    Connectable(
        "fireworks_ai",
        "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY",
        discovery="manual",
    ),
    Connectable(
        "friendliai", "https://api.friendli.ai/serverless/v1", "FRIENDLIAI_API_KEY"
    ),
    Connectable(
        "galadriel",
        "https://api.galadriel.com/v1/verified",
        "GALADRIEL_API_KEY",
        pin_base=True,
    ),
    Connectable("gmi", "https://api.gmi-serving.com/v1", "GMI_API_KEY"),
    Connectable("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    Connectable("libertai", "https://api.libertai.io/v1", "LIBERTAI_API_KEY"),
    Connectable(
        "modelscope", "https://api-inference.modelscope.cn/v1", "MODELSCOPE_API_KEY"
    ),
    Connectable(
        "nebius",
        "https://api.tokenfactory.nebius.com/v1",
        "NEBIUS_API_KEY",
        pin_base=True,
    ),
    Connectable("neosantara", "https://api.neosantara.xyz/v1", "NEOSANTARA_API_KEY"),
    Connectable(
        "novita", "https://api.novita.ai/openai", "NOVITA_API_KEY", pin_base=True
    ),
    Connectable("nscale", "https://inference.api.nscale.com/v1", "NSCALE_API_KEY"),
    Connectable(
        "nvidia_nim", "https://integrate.api.nvidia.com/v1", "NVIDIA_NIM_API_KEY"
    ),
    Connectable("parasail", "https://api.parasail.io/v1", "PARASAIL_API_KEY"),
    Connectable(
        "pinstripes",
        "https://api.pinstripes.io/v1",
        "PINSTRIPES_API_KEY",
        pin_base=True,
    ),
    Connectable("publicai", "https://api.publicai.co/v1", "PUBLICAI_API_KEY"),
    Connectable("sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY"),
    Connectable("scx-ai", "https://api.scx.ai/v1", "SCX_API_KEY"),
    Connectable(
        "synthetic", "https://api.synthetic.new/openai/v1", "SYNTHETIC_API_KEY"
    ),
    Connectable(
        "tensormesh",
        "https://serverless.tensormesh.ai/v1",
        "TENSORMESH_INFERENCE_API_KEY",
    ),
    Connectable("together_ai", "https://api.together.ai/v1", "TOGETHER_API_KEY"),
    Connectable("veniceai", "https://api.venice.ai/api/v1", "VENICE_AI_API_KEY"),
    Connectable("wandb", "https://api.inference.wandb.ai/v1", "WANDB_API_KEY"),
    # Gateways
    Connectable("aiml", "https://api.aimlapi.com/v1", "AIML_API_KEY"),
    Connectable("aihubmix", "https://aihubmix.com/v1", "AIHUBMIX_API_KEY"),
    Connectable("apertis", "https://api.apertis.ai/v1", "STIMA_API_KEY", pin_base=True),
    Connectable(
        "assemblyai", "https://llm-gateway.assemblyai.com/v1", "ASSEMBLYAI_API_KEY"
    ),
    Connectable("cometapi", "https://api.cometapi.com/v1", "COMETAPI_KEY"),
    Connectable("edenai", "https://api.edenai.run/v3", "EDENAI_API_KEY"),
    Connectable("helicone", "https://ai-gateway.helicone.ai", "HELICONE_API_KEY"),
    # Model IDs are <org>/<model>; LiteLLM picks the inference provider
    Connectable("huggingface", "https://router.huggingface.co/v1", "HF_TOKEN"),
    # LiteLLM sends no request to a proxy without a key
    Connectable("litellm_proxy", None, "LITELLM_PROXY_API_KEY"),
    Connectable("nadir", "https://api.getnadir.com/v1", "NADIR_API_KEY"),
    Connectable("nano-gpt", "https://nano-gpt.com/api/v1", "NANOGPT_API_KEY"),
    Connectable("poe", "https://api.poe.com/v1", "POE_API_KEY"),
    Connectable(
        "vercel_ai_gateway",
        "https://ai-gateway.vercel.sh/v1",
        "VERCEL_AI_GATEWAY_API_KEY",
    ),
    # Local servers, at their documented default addresses
    Connectable(
        "docker_model_runner",
        "http://localhost:12434/engines/v1",
        "DOCKER_MODEL_RUNNER_API_KEY",
        key_optional=True,
        pin_base=True,
    ),
    Connectable(
        "lemonade",
        "http://localhost:13305/api/v1",
        "LEMONADE_API_KEY",
        key_optional=True,
        pin_base=True,
    ),
    Connectable(
        "llamafile",
        "http://127.0.0.1:8080/v1",
        "LLAMAFILE_API_KEY",
        key_optional=True,
    ),
    Connectable(
        "lm_studio",
        "http://localhost:1234/v1",
        "LM_STUDIO_API_KEY",
        key_optional=True,
        pin_base=True,
    ),
    # LiteLLM adds /v1 to the host
    Connectable(
        "oobabooga",
        "http://127.0.0.1:5000",
        None,
        key_optional=True,
        models_path="v1/models",
        pin_base=True,
    ),
    Connectable("hosted_vllm", None, "HOSTED_VLLM_API_KEY", key_optional=True),
    # LiteLLM sends no request to Xinference without a key: any text works
    # for a server that checks none
    Connectable("xinference", "http://127.0.0.1:9997/v1", None, pin_base=True),
)

_BY_PROVIDER = {row.provider: row for row in CONNECTABLE}

# Offered before 1.3.0, without asking LiteLLM: their cards stay as they were
BUILT_IN = frozenset(row.provider for row in CONNECTABLE if row.discovery == "builtin")


def connectable(provider: str) -> Connectable | None:
    """The table row of *provider*, or None."""
    return _BY_PROVIDER.get(provider.lower())


def default_base(provider: str) -> str | None:
    """The address a connection to *provider* uses when none is entered."""
    row = connectable(provider)
    return row.base if row is not None else None


def pinned_base(provider: str) -> str | None:
    """The address a new connection stores when none is entered: the
    documented one, where LiteLLM's own default would go elsewhere."""
    row = connectable(provider)
    return row.base if row is not None and row.pin_base else None


def installed(row: Connectable) -> bool:
    """Whether the modules the provider's route needs besides LiteLLM are here."""
    return row.requires is None or importlib.util.find_spec(row.requires) is not None


@cache
def routable_providers() -> frozenset[str]:
    """The table's providers this LiteLLM routes under their own prefix.

    Imports LiteLLM (most of a second): call it on a worker thread. A
    LiteLLM older than the one the table was checked with may lack some
    prefixes; those cards are not offered.
    """
    from markitai.utils.litellm_import import import_litellm

    try:
        # The package before its submodule: serve's prewarm thread may be
        # importing it right now (see litellm_import)
        import_litellm()
        from litellm.litellm_core_utils.get_llm_provider_logic import (
            get_llm_provider,
        )
    except Exception:
        return frozenset()
    routable = set()
    for row in CONNECTABLE:
        prefix = "openai" if row.provider == "custom" else row.provider
        try:
            _model, provider, _key, _base = get_llm_provider(f"{prefix}/markitai-probe")
        except Exception:
            continue
        if provider == prefix and installed(row):
            routable.add(row.provider)
    return frozenset(routable)
