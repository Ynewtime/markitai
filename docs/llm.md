# Native LLM execution

API-key model routes send text and image requests directly over HTTP from Rust;
they do not start Python, load LiteLLM or require a provider CLI. The separately
selected [subscription routes](subscriptions.md) use their supported official
runtimes. Request construction, model selection and retry accounting share the
native conversion pipeline.
Configuration validation remains separate from runtime capability: an accepted
configuration can still request a provider or routing strategy that this build
explicitly rejects.

## Getting started

```sh
export OPENAI_API_KEY=...           # or another provider key, or a .env file
markitai init --yes                 # records the detected model; LLM stays off
markitai doctor                     # checks credentials without a model request
markitai report.pdf --llm -o out/   # writes out/report.pdf.llm.md
```

A pinned model in the configuration file looks like this:

```json
{
  "llm": {
    "enabled": true,
    "model_list": [
      {"model_name": "default",
       "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "env:ANTHROPIC_API_KEY"}}
    ]
  }
}
```

`--keep-base` also keeps the unenhanced `report.pdf.md`; `--alt`/`--desc` add
[image captions](image-enrichment.md). See [configuration](configuration.md)
for the related keys and environment variables and [pricing](pricing.md) for
costs.

## Model selection

A nonempty `llm.model_list` owns the model pool. Entries with `weight: 0` are
excluded. Missing explicit `env:NAME` credentials or endpoint references make
that deployment unavailable; usable siblings can still serve the request. An
entirely disabled or unavailable configured pool does not fall back to ambient
credentials.

A deployment whose provider needs a key that nothing supplies (no `api_key`,
none of the provider's key variables, an `env:NAME` reference to an unset or
empty variable) is left out before anything is sent, with one warning per
process naming the variable to set; with no deployment left, the conversion
fails with an error naming the variables (code `no_model_configured`, exit 1
on the command line, before converting). A deployment aimed at an address of
its own (`api_base` or a base variable) is still tried, since a gateway or a
local server may check no key; Azure, which has no default address, needs its
key. `markitai doctor` applies the same rule. A key the provider refuses (HTTP
401, or Gemini's `API_KEY_INVALID`) is reported with the variable it was read
from.

With an empty model list, a nonempty `MODEL` selects one model. Otherwise, as
in Python markitai, the [subscription runtimes](subscriptions.md) signed in on
this machine join the pool first (Claude Code as `claude-agent/haiku`, Copilot
as `copilot/auto`, ChatGPT through Codex as `chatgpt/` and the lightest
current model Codex lists, `gpt-6-luna` today), then each available API credential: Anthropic, OpenAI, Gemini,
DeepSeek and OpenRouter. Each runtime is asked once per process for its own
sign-in status (`claude auth status`, Copilot's status call, `codex login
status`); none of their token files is read, and a runtime signed in only with
an API key does not count. When several providers are detected, a conversion
says so once (`Auto-detected 3 LLM providers; requests are spread across all of
them: …`). The model aliases are derived from the reference checkout's
provider-default table, not an independent assertion about current provider
availability; DeepSeek's is `deepseek/deepseek-flash`, the id DeepSeek's model
list names (the reference's `deepseek-v4-flash` is an alias of it). Pin `MODEL` or a configured deployment to control which provider
receives documents. Keys of the [OpenAI-compatible
prefixes](#openai-compatible-prefixes) do not join this pool, because no default
model is known for them; name one with `MODEL` (for example
`MODEL=groq/<model>`) or in `llm.model_list`. `markitai init` writes the same
models, except that an Anthropic key's model is left out when Claude Code serves
the same Claude Haiku on a subscription.

`simple-shuffle` chooses deployments in proportion to positive integer weights.
A randomized process seed and an atomic sequence provide independent selection
without a runtime random-number dependency. The selection calculation uses a
128-bit weight sum. A failed deployment is avoided during the current request
while another eligible deployment remains; after every candidate has failed,
retries may revisit the pool.

`least-busy` chooses the eligible deployment with the fewest active HTTP
attempts in the same `LlmRuntime`. Equal counts use configured candidate order;
positive weights do not change the comparison, while weight zero still disables
a deployment. Selection and reservation happen atomically after acquiring the
runtime's shared request capacity and passing cancellation. The selected
identity then passes document request/dollar admission before any network call;
a refused admission releases its reservation without a routing observation.
The reservation lasts through response reading and decoding, and is
released on success, error or unwind. Backoff, queueing, cached answers and
coalesced waiters do not reserve a deployment.

Ties preserve candidate order. This follows the normal healthy-pool path in the
reference checkout's locked LiteLLM 1.100.1:
`litellm/router_strategy/least_busy.py::_get_available_deployments` updates its
choice only for a strictly smaller count. Its random choice handles a missing
or unhealthy selected ID; it does not randomize ordinary equal-count ties.

Runtime clones share these counts. Independent runtimes, independent serialized
binding calls and conversions without a caller-supplied shared runtime do not
coordinate across calls. The CLI batch, REST job and MCP job already supply
their own shared runtime. This is neither process-global nor distributed load
balancing. Counts cover adaptive-routing attempts; simultaneous conversions configured
with `simple-shuffle` retain their separate selection behavior.

Resolved endpoint, credentials, model, protocol, group and explicit
`model_info.id` form a private per-runtime salted identity. Different credentials
or deployment IDs do not share occupancy accidentally. Identical entries with
the same identity share their actual occupancy. These fingerprints are not
logged or persisted. Only active occupancy identities remain in the table, bounded by
the runtime's existing concurrency limit. Metric strategies retain a separate
bounded set of observations as described below. The authentication exclusion
list described under retries holds only these salted identities and is
discarded with the runtime.

Without configured fallbacks, all model names share the `default` pool. With
fallbacks, configured group names are preserved and requests enter `default`.
For example:

```json
{
  "llm": {
    "model_list": [
      {"model_name":"default","litellm_params":{"model":"openai/my-primary","api_key":"env:PRIMARY_KEY"}},
      {"model_name":"backup","litellm_params":{"model":"openai/my-backup","api_key":"env:BACKUP_KEY"}}
    ],
    "router_settings":{"fallbacks":[{"default":["backup"]}],"num_retries":2}
  }
}
```

Fallbacks are traversed in declared order, including nested fallback groups.
Each reachable group is attempted once; cycles, malformed target lists and
references to unavailable groups fail before requests begin. Each group gets
its configured transport retry allowance. Adaptive strategies change deployment
selection without adding another retry loop. Apart from the run-scoped
authentication exclusion described under retries, there is no cross-document
deployment cooldown or health-history database.

### Usage and latency observations

`usage-based-routing` selects the smallest successful provider token total in the
current wall-clock minute. Known zero usage stays zero; missing token data stays
unknown and does not create a sample. A provider's explicit `total_tokens` takes
precedence. Otherwise a total is derived only when both protocol input and output
counts are present. For Anthropic, known input counts include reported cache-read
and cache-creation input tokens. Positive deployment weights do not scale scores.
Ties prefer the first observed entry; deployments without an observation follow
in candidate order. There is no TPM/RPM quota configuration in this interface.

`latency-based-routing` compares the mean of each deployment's last ten samples.
A successful sample measures HTTP send through complete bounded response decode,
divided by output tokens when the known count is positive. Zero or missing
output-token counts retain raw elapsed seconds. A timed-out attempt, including an explicit HTTP 408, contributes
the reference's 1000-second penalty; other failed attempts do not add a latency
sample. New deployments score zero. Equal minimum scores are selected randomly,
without positive-weight scaling. This mode is not a promise of lower latency or
fairness for every workload.

Observations are scoped by the existing runtime, strategy, effective model group
pool and text-versus-visual requests. Resolved credentials and endpoints remain
part of private deployment identity. Cached answers and coalesced waiters have no
HTTP attempt and contribute no metric sample. Valid provider envelopes are
observed before the returned document data is validated: a paid response with an
invalid document can affect routing once, while a paid HTTP error still affects
reported usage without becoming a successful routing observation. Reporting and
billing are independent of metric samples.

Usage buckets use a full epoch minute and expire after 60 monotonic seconds without
a qualifying sample, even if the wall clock moves. Each qualifying sample refreshes
that group deadline. A minute change clears the previous bucket instead of
reusing a repeating `HH-MM` key. Latency group state expires after 3,600 monotonic
seconds without a sample; a new group sample refreshes that group deadline.
Retained state allows at most 4,096 idle deployment observations plus currently
active metric identities. Expired entries are removed first; least-recently-used
idle entries are evicted next and return to a cold score. Active reservations are
never evicted. Dropping the runtime discards everything; no metrics are written
to disk or shared between independent binding calls.

These defaults follow the reference's locked LiteLLM 1.100.1 usage-v1 and latency
handlers. Native bounded retention, full-minute identity, explicit text/visual
pool separation and atomic concurrent updates are documented implementation
choices. They do not reproduce LiteLLM's Redis support, streaming TTFT routing,
or optional quota settings that Markitai does not expose.

## Providers and request parameters

| Model prefix | Native request protocol |
|---|---|
| `openai/`, or no prefix | OpenAI Chat Completions |
| `anthropic/` | Anthropic Messages |
| `gemini/`, `deepseek/`, `openrouter/` | Provider's OpenAI-compatible endpoint |
| `azure/` | Azure deployment Chat Completions, with `api-version` |
| `ollama/`, `ollama_chat/` | Ollama's OpenAI-compatible `/v1` endpoint |
| The prefixes in the next table | The provider's OpenAI-compatible Chat Completions endpoint |
| `copilot/`, `claude-agent/`, `chatgpt/` | Installed official subscription runtime; see [subscriptions](subscriptions.md) |

The five built-in API providers read their keys from `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` (a gateway token, sent as
`Authorization: Bearer`), `GEMINI_API_KEY` or `GOOGLE_API_KEY` (which wins when
both are set, as in Google's SDK), `DEEPSEEK_API_KEY`, and `OPENROUTER_API_KEY`
or `OR_API_KEY`, as LiteLLM does.

Anthropic Messages requests (`anthropic/`, including a gateway, proxy or
Microsoft Foundry address in `api_base`, `azure_ai/` Claude models, and Claude
on Vertex AI) use API version `2026-10-21`: the `anthropic-version` header,
Vertex AI's `anthropic_version` body field, Message Batches, model discovery
and the connection test. `litellm_params` keeps only the settings listed in
[configuration](configuration.md), so `temperature`, `top_p` and `top_k` are
never sent (Python markitai drops them too); an Anthropic model that sets them
gets one warning naming them, since that API version refuses them on every
model. A gateway in `api_base` must accept that version.

### OpenAI-compatible prefixes

These prefixes are plain Chat Completions endpoints that the reference reaches
through LiteLLM's routing. The prefix is the text before the first `/`; the rest
of the model name, slashes included, is sent as the model
(`together_ai/meta-llama/…` sends `meta-llama/…`). The base URL of every entry
was checked against the provider's own documentation on 2026-10-02 (the page is
in the last column); `/chat/completions` is appended to it. The key variables
are the ones the reference's LiteLLM 1.100.1 reads, so an environment prepared
for the reference keeps working; where they differ, the provider's documented
name comes first and the first variable that holds a value is used.

| Prefix | Default base URL | Key variables | Base variable | Checked against |
|---|---|---|---|---|
| `groq/` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | `GROQ_API_BASE` | groq-python SDK client (default host and `/openai/v1/chat/completions` path); console.groq.com refused automated reads |
| `mistral/` | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` | `MISTRAL_API_BASE` | docs.mistral.ai/api/endpoint/chat |
| `xai/` | `https://api.x.ai/v1` | `XAI_API_KEY` | `XAI_API_BASE` | docs.x.ai/docs/api-reference |
| `together_ai/` | `https://api.together.ai/v1` | `TOGETHER_API_KEY`, `TOGETHER_AI_API_KEY`, `TOGETHERAI_API_KEY`, `TOGETHER_AI_TOKEN` | `TOGETHER_AI_API_BASE` | docs.together.ai/docs/openai-api-compatibility |
| `perplexity/` | `https://api.perplexity.ai/router/v1` | `PERPLEXITY_API_KEY`, `PERPLEXITYAI_API_KEY` | `PERPLEXITY_API_BASE` | docs.perplexity.ai/api-reference/gateway-chat-completions-post |
| `cerebras/` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` | `CEREBRAS_API_BASE` | inference-docs.cerebras.ai/resources/openai |
| `fireworks_ai/` | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY`, `FIREWORKS_AI_API_KEY`, `FIREWORKSAI_API_KEY`, `FIREWORKS_AI_TOKEN` | `FIREWORKS_API_BASE` | docs.fireworks.ai/getting-started/quickstart |
| `deepinfra/` | `https://api.deepinfra.com/v1/openai` | `DEEPINFRA_API_KEY` | `DEEPINFRA_API_BASE` | docs.deepinfra.com/chat/overview |
| `nebius/` | `https://api.tokenfactory.nebius.com/v1` | `NEBIUS_API_KEY` | `NEBIUS_API_BASE` | docs.tokenfactory.nebius.com/quickstart |
| `moonshot/` | `https://api.moonshot.ai/v1` | `MOONSHOT_API_KEY` | `MOONSHOT_API_BASE` | platform.kimi.ai/docs/guide/start-using-kimi-api |
| `sambanova/` | `https://api.sambanova.ai/v1` | `SAMBANOVA_API_KEY` | `SAMBANOVA_API_BASE` | docs.sambanova.ai/docs/en/get-started/api-keys-urls |
| `zai/` | `https://api.z.ai/api/paas/v4` | `ZAI_API_KEY` | `ZAI_API_BASE` | docs.z.ai/guides/develop/openai/python |
| `nvidia_nim/` | `https://integrate.api.nvidia.com/v1` | `NVIDIA_NIM_API_KEY` | `NVIDIA_NIM_API_BASE` | docs.api.nvidia.com/nim/reference/llm-apis |
| `novita/` | `https://api.novita.ai/openai` | `NOVITA_API_KEY` | `NOVITA_API_BASE` | docs.novita.ai/guides/llm-api |
| `hosted_vllm/` | none: set `api_base` or the base variable | `HOSTED_VLLM_API_KEY` (optional) | `HOSTED_VLLM_API_BASE` | docs.vllm.ai quickstart (a key is checked only when the server was started with one) |
| `lm_studio/` | `http://localhost:1234/v1` | `LM_STUDIO_API_KEY` (optional) | `LM_STUDIO_API_BASE` | lmstudio.ai/docs/app/api/endpoints/openai |

Notes on individual entries:

- Perplexity ended support for its Sonar chat completions on 2026-09-27; its
  documented OpenAI Chat Completions route is now the Router, so the default
  differs from LiteLLM 1.100.1's `https://api.perplexity.ai`. Router model
  names are passed through unchanged.
- Nebius and Novita document newer hosts than LiteLLM 1.100.1 uses
  (`api.studio.nebius.ai`, `api.novita.ai/v3/openai`); the documented ones are
  used. Novita's and NVIDIA's pages name no key variable, so LiteLLM's names
  apply.
- `vLLM` listens wherever it was started; without `api_base` or
  `HOSTED_VLLM_API_BASE` the deployment is unavailable with an error naming
  both. LM Studio's documented local server is the default.
- `hosted_vllm/`, `lm_studio/`, `ollama/` and `ollama_chat/` deployments are
  usable without a key; the hosted APIs count as routable only with one.
  `ollama_chat/` also reads `OLLAMA_API_BASE`, as LiteLLM does.

### Keys that belong to a region

Some providers issue a key for one region (or service) only; any other endpoint
answers it with HTTP 401 (checked 2026-10-26). Set the region's address as
`api_base` or the base variable, e.g. `MOONSHOT_API_BASE=https://api.moonshot.cn/v1`
beside `MOONSHOT_API_KEY`. A refused key's error names the provider's other
regions and the keys that belong elsewhere. The workbench finds the region from
the key: with no address typed, it tries the default first, then each other
region, and saves the models with the address that listed them. Moonshot's open
platform and Kimi Code are one Kimi card there, tried in turn (Kimi Code first for
a key in its documented `sk-kimi-` form).

| Prefix | Region: Base URL (default first) | Key from | Docs |
|---|---|---|---|
| `moonshot/` | International: `https://api.moonshot.ai/v1`; China: `https://api.moonshot.cn/v1` | platform.kimi.ai; platform.kimi.com (开放平台) | platform.kimi.ai/docs/guide/start-using-kimi-api |
| `kimi_code/` | kimi.com: `https://api.kimi.com/coding/v1`; kimi.ai: `https://api.kimi.ai/coding/v1` | a Kimi Code membership, kimi.com; kimi.ai | Kimi Code's guide for Claude Code |
| `minimax/` | Global: `https://api.minimax.io/v1`; Mainland China: `https://api.minimax.cn/v1` | platform.minimax.io; platform.minimax.cn | platform.minimax.io/docs/coding-plan/faq |
| `zai/` | International: `https://api.z.ai/api/paas/v4`; China: `https://open.bigmodel.cn/api/paas/v4` | z.ai; bigmodel.cn (智谱开放平台) | docs.z.ai/guides/develop/http/introduction; docs.bigmodel.cn/cn/api/introduction |
| `dashscope/` | Beijing: `https://dashscope.aliyuncs.com/compatible-mode/v1`; Singapore: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`; US (Virginia): `https://dashscope-us.aliyuncs.com/compatible-mode/v1`; Hong Kong: `https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1` | Model Studio, per region | alibabacloud.com/help/en/model-studio/base-url |

Kimi Code takes `KIMI_API_KEY`, its address `KIMI_CODE_API_BASE` or
`KIMI_CODE_BASE_URL` (model IDs such as `k3.1-pro`, `k3`,
`kimi-for-coding`; the workbench lists the ones the key may use). Z.ai's and
Zhipu's Coding Plan endpoints (`…/api/coding/paas/v4`) take either kind of
key, as MiniMax's addresses take its M Plan keys (`sk-cp-…`). Model Studio's
Coding Plan keys (`sk-sp-…`) work only at the plan's own endpoints
(`https://coding.dashscope.aliyuncs.com/v1`), for coding tools, and are not tried
there (the workbench's API address lists every address above and these Coding
Plan endpoints, with that note); a workspace-dedicated Model Studio host
(`<workspace>.<region>.maas.aliyuncs.com`) is set as `api_base` and tried alone.

Kimi Code is reached at its Anthropic-compatible Messages endpoint
(`…/coding/v1/messages`, `x-api-key`), the one Kimi documents for Claude Code;
its Chat Completions endpoint is reported to admit only the User-Agents of
listed coding agents, and markitai sends its own. Requests ask for at most
32,768 output tokens unless `max_tokens` says otherwise, since its models think
before answering. Kimi's community guidelines describe the membership as
personal, interactive use and ask not to use it for non-interactive automation
or batch pipelines; use an open-platform key for batch conversions. Its cost
shows as unknown.

### Further OpenAI-compatible prefixes

The other OpenAI-compatible prefixes the reference reaches through LiteLLM
1.104.2 are routed the same way, with LiteLLM's model naming and key
variables. Their bases were checked against each provider's documentation on
2026-10-11; where the documented host or key name is newer than LiteLLM's,
the documented one is used or listed first, and where a provider documents
none, LiteLLM's default is used and the last column says so. The `serve`
provider picker offers them like the prefixes of the first table.

| Prefix | Default base URL | Key variables | Base variable | Checked against |
|---|---|---|---|---|
| `ai21_chat/`, `ai21/` | `https://api.ai21.com/studio/v1` | `AI21_API_KEY` | `AI21_API_BASE` | docs.ai21.com/reference/jamba-1-6-api-ref |
| `volcengine/` | `https://ark.cn-beijing.volces.com/api/v3` | `ARK_API_KEY`, `VOLCENGINE_API_KEY` | `VOLCENGINE_API_BASE` | volcengine.com/docs/82379/1330626 (BytePlus: `https://ark.ap-southeast.bytepluses.com/api/v3` as `api_base`) |
| `codestral/` | `https://codestral.mistral.ai/v1` | `CODESTRAL_API_KEY` | `CODESTRAL_API_BASE` | LiteLLM's default; Mistral's current docs name only `api.mistral.ai` |
| `tencent/` | `https://tokenhub-intl.tencentcloudmaas.com/v1` | `TOKENHUB_API_KEY`, `TENCENT_API_KEY` | `TENCENT_API_BASE` | tencentcloud.com/document/product/1300/78941 |
| `friendliai/` | `https://api.friendli.ai/serverless/v1` | `FRIENDLIAI_API_KEY`, `FRIENDLI_TOKEN` | `FRIENDLI_API_BASE` | friendli.ai/docs/guides/openai-compatibility |
| `galadriel/` | `https://api.galadriel.com/v1/verified` | `GALADRIEL_API_KEY` | `GALADRIEL_API_BASE` | docs.galadriel.com/for-agents-developers/quickstart (LiteLLM omits `/verified`) |
| `baseten/` | `https://inference.baseten.co/v1`; an 8-character model ID is a dedicated deployment at `https://model-<id>.api.baseten.co/environments/production/sync/v1` | `BASETEN_API_KEY` | `BASETEN_API_BASE` | docs.baseten.co/inference/model-apis/overview |
| `meta/` | `https://api.meta.ai/v1` | `MODEL_API_KEY`, `META_API_KEY` | `META_API_BASE` | dev.meta.ai/docs/api-reference |
| `featherless_ai/` | `https://api.featherless.ai/v1` | `FEATHERLESS_API_KEY`, `FEATHERLESS_AI_API_KEY` | `FEATHERLESS_AI_API_BASE`, `FEATHERLESS_API_BASE` | featherless.ai/docs/getting-started |
| `nscale/` | `https://inference.api.nscale.com/v1` | `NSCALE_SERVICE_TOKEN`, `NSCALE_API_KEY` | `NSCALE_API_BASE` | docs.nscale.com/docs/use-cases/chat |
| `dashscope/` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | `DASHSCOPE_API_BASE` | alibabacloud.com/help/en/model-studio/base-url; other regions under [Keys that belong to a region](#keys-that-belong-to-a-region) |
| `qwencloud/` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `QWENCLOUD_API_KEY`, `DASHSCOPE_API_KEY` | `QWENCLOUD_API_BASE` | LiteLLM's default (Model Studio's Singapore host) |
| `qwen_ai_platform/` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `QWEN_AI_PLATFORM_API_KEY`, `DASHSCOPE_API_KEY` | `QWEN_AI_PLATFORM_API_BASE` | LiteLLM's default |
| `modelscope/` | `https://api-inference.modelscope.cn/v1` | `MODELSCOPE_API_KEY` | `MODELSCOPE_API_BASE` | LiteLLM's default (modelscope.cn renders its docs only in a browser) |
| `morph/` | `https://api.morphllm.com/v1` | `MORPH_API_KEY` | `MORPH_API_BASE` | docs.morphllm.com/llms.txt |
| `inception/` | `https://api.inceptionlabs.ai/v1` | `INCEPTION_API_KEY` | `INCEPTION_API_BASE` | docs.inceptionlabs.ai/api-reference/chat/create-a-chat-completion |
| `aiml/` | `https://api.aimlapi.com/v1` | `AIML_API_KEY` | `AIML_API_BASE` | docs.aimlapi.com/quickstart/simple-model |
| `edenai/` | `https://api.edenai.run/v3` | `EDENAI_API_KEY` | `EDENAI_API_BASE` | edenai.co/docs/v3/llms/chat-completions |
| `wandb/` | `https://api.inference.wandb.ai/v1` | `WANDB_API_KEY` | `WANDB_API_BASE` | docs.coreweave.com/products/inference/serverless/api-reference |
| `publicai/` | `https://api.publicai.co/v1` | `PUBLICAI_API_KEY` | `PUBLICAI_API_BASE` | platform.publicai.co/docs |
| `veniceai/` | `https://api.venice.ai/api/v1` | `VENICE_API_KEY`, `VENICE_AI_API_KEY` | `VENICEAI_API_BASE` | docs.venice.ai/getting-started/quick-start |
| `xiaomi_mimo/` | `https://api.xiaomimimo.com/v1` | `MIMO_API_KEY`, `XIAOMI_MIMO_API_KEY` | `XIAOMI_MIMO_API_BASE` | mimo.mi.com/static/docs/quick-start/summary/first-api-call.md |
| `scaleway/` | `https://api.scaleway.ai/v1` | `SCW_SECRET_KEY` | `SCALEWAY_API_BASE` | scaleway.com/en/docs/generative-apis/api-cli/using-chat-api |
| `synthetic/` | `https://api.synthetic.new/openai/v1` | `SYNTHETIC_API_KEY` | `SYNTHETIC_API_BASE` | dev.synthetic.new/docs/api/overview |
| `apertis/` | `https://api.apertis.ai/v1` | `STIMA_API_KEY` | `APERTIS_API_BASE` | docs.apertis.ai/getting-started/quick-start (LiteLLM: `api.stima.tech`) |
| `nano-gpt/` | `https://nano-gpt.com/api/v1` | `NANOGPT_API_KEY` | `NANOGPT_API_BASE` | docs.nano-gpt.com/api-reference/endpoint/chat-completion |
| `poe/` | `https://api.poe.com/v1` | `POE_API_KEY` | `POE_API_BASE` | creator.poe.com/docs/external-applications/openai-compatible-api |
| `chutes/` | `https://llm.chutes.ai/v1` | `CHUTES_API_KEY` | `CHUTES_API_BASE` | chutes.ai/docs/api-reference/authentication |
| `abliteration/` | `https://api.abliteration.ai/v1` | `ABLIT_KEY`, `ABLITERATION_API_KEY` | `ABLITERATION_API_BASE` | abliteration.ai/docs |
| `gmi/` | `https://api.gmi-serving.com/v1` | `GMI_API_KEY` | `GMI_API_BASE` | docs.gmicloud.ai/inference-engine/api-reference/llm-api-reference |
| `sarvam/` | `https://api.sarvam.ai/v1` | `SARVAM_API_KEY` | `SARVAM_API_BASE` | docs.sarvam.ai/api-reference-docs/chat/chat-completions (Bearer accepted) |
| `assemblyai/` | `https://llm-gateway.assemblyai.com/v1` | `ASSEMBLYAI_API_KEY` | `ASSEMBLYAI_API_BASE` | assemblyai.com/docs/llm-gateway/quickstart (EU: `https://llm-gateway.eu.assemblyai.com/v1`) |
| `aihubmix/` | `https://aihubmix.com/v1` | `AIHUBMIX_API_KEY` | `AIHUBMIX_API_BASE` | docs.aihubmix.com/cn/quick-start |
| `crusoe/` | `https://api.inference.crusoecloud.com/v1` | `CRUSOE_API_KEY` | `CRUSOE_API_BASE` | docs.crusoecloud.com/quickstart/getting-started-with-serverless-inference (LiteLLM's host is older) |
| `darkbloom/` | `https://api.darkbloom.dev/v1` | `DARKBLOOM_API_KEY` | `DARKBLOOM_API_BASE` | docs.darkbloom.dev/api/chat-completions |
| `neosantara/` | `https://api.neosantara.xyz/v1` | `NEOSANTARA_API_KEY` | `NEOSANTARA_API_BASE` | docs.neosantara.xyz/id/gateway/chat-completions |
| `tensormesh/` | `https://serverless.tensormesh.ai/v1` | `TENSORMESH_INFERENCE_API_KEY` | `TENSORMESH_SERVERLESS_BASE_URL` | LiteLLM's default (Tensormesh documents no serverless API) |
| `parasail/` | `https://api.parasail.io/v1` | `PARASAIL_API_KEY` | `PARASAIL_API_BASE` | docs.parasail.io/parasail-docs/api-reference/authentication |
| `libertai/` | `https://api.libertai.io/v1` | `LIBERTAI_API_KEY` | `LIBERTAI_API_BASE` | docs.libertai.io/apis/text/usage.html |
| `empiriolabs/` | `https://api.empiriolabs.ai/v1` | `EMPIRIOLABS_API_KEY` | `EMPIRIOLABS_API_BASE` | docs.empiriolabs.ai/getting-started |
| `cognition/` | `https://api.cognition.ai/v1` | `COGNITION_API_KEY` | `COGNITION_API_BASE` | LiteLLM's default (Cognition documents no chat API) |
| `pinstripes/` | `https://api.pinstripes.io/v1` | `PINSTRIPES_API_KEY` | `PINSTRIPES_API_BASE` | pinstripes.io (LiteLLM omits `api.`) |
| `scx-ai/` | `https://api.scx.ai/v1` | `SCX_API_KEY` | `SCX_API_BASE` | scx.ai model pages |
| `nadir/` | `https://api.getnadir.com/v1` | `NADIR_API_KEY` | `NADIR_API_BASE` | getnadir.com/docs |
| `amazon_nova/` | `https://api.nova.amazon.com/v1` | `NOVA_API_KEY`, `AMAZON_NOVA_API_KEY` | `AMAZON_NOVA_API_BASE` | github.com/amazon-nova-api/getting-started-with-nova-api |
| `minimax/` | `https://api.minimax.io/v1` | `MINIMAX_API_KEY` | `MINIMAX_API_BASE` | platform.minimax.io/docs/api-reference/text-openai-api (China: [Keys that belong to a region](#keys-that-belong-to-a-region)) |
| `ovhcloud/` | `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1` | `OVH_AI_ENDPOINTS_ACCESS_TOKEN`, `OVHCLOUD_API_KEY` | `OVHCLOUD_API_BASE` | docs.ovhcloud.com AI Endpoints guides |
| `compactifai/` | `https://api.compactif.ai/v1` | `COMPACTIFAI_API_KEY` | `COMPACTIFAI_API_BASE` | docs.compactif.ai/api_reference |
| `gradient_ai/` | `https://inference.do-ai.run/v1` | `MODEL_ACCESS_KEY`, `GRADIENT_AI_API_KEY`, `DIGITALOCEAN_TOKEN` | `GRADIENT_AI_API_BASE` | docs.digitalocean.com/products/inference/how-to/use-chat-completions-api (LiteLLM omits `/v1`) |
| `helicone/` | `https://ai-gateway.helicone.ai` | `HELICONE_API_KEY` | `HELICONE_API_BASE` | docs.helicone.ai/gateway/overview |
| `vercel_ai_gateway/` | `https://ai-gateway.vercel.sh/v1` | `AI_GATEWAY_API_KEY`, `VERCEL_AI_GATEWAY_API_KEY`, `VERCEL_OIDC_TOKEN` | `VERCEL_AI_GATEWAY_API_BASE` | vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions |
| `cometapi/` | `https://api.cometapi.com/v1` | `COMETAPI_KEY`, `COMETAPI_API_KEY` | `COMETAPI_API_BASE`, `COMETAPI_BASE_URL` | apidoc.cometapi.com/guides/use-cometapi-with-openai-sdk |
| `cohere/`, `cohere_chat/` | `https://api.cohere.ai/compatibility/v1` | `COHERE_API_KEY`, `CO_API_KEY` | `COHERE_API_BASE` | docs.cohere.com/docs/compatibility-api |
| `huggingface/` | `https://router.huggingface.co/v1` | `HF_TOKEN`, `HUGGINGFACE_API_KEY` | `HF_API_BASE`, `HUGGINGFACE_API_BASE` | huggingface.co/docs/inference-providers |
| `bedrock_mantle/` | `https://bedrock-mantle.<region>.api.aws/v1` (`/openai/v1` for some models) | `AWS_BEARER_TOKEN_BEDROCK`, `BEDROCK_MANTLE_API_KEY` | `BEDROCK_MANTLE_API_BASE` | docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html |
| `cloudflare/` | `https://api.cloudflare.com/client/v4/accounts/<CLOUDFLARE_ACCOUNT_ID>/ai/v1` | `CLOUDFLARE_API_KEY`, `CLOUDFLARE_API_TOKEN` | `CLOUDFLARE_API_BASE` | developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility |
| `datarobot/` | `https://app.datarobot.com/api/v2/genai/llmgw` | `DATAROBOT_API_TOKEN` | `DATAROBOT_ENDPOINT` | docs.datarobot.com/en/docs/agentic-ai/genai-code/dr-llm-gateway.html |
| `databricks/` | none: `https://<workspace>/serving-endpoints` | `DATABRICKS_TOKEN`, `DATABRICKS_API_KEY` | `DATABRICKS_API_BASE` | docs.databricks.com/aws/en/large-language-models/llm-serving-intro |
| `heroku/` | none: `<INFERENCE_URL>/v1` | `INFERENCE_KEY`, `HEROKU_API_KEY` | `HEROKU_API_BASE` | devcenter.heroku.com/articles/heroku-inference-api-v1-chat-completions |
| `azure_ai/` | none: `https://<resource>.openai.azure.com/openai/v1`; Claude: `https://<resource>.services.ai.azure.com/anthropic` | `AZURE_INFERENCE_CREDENTIAL`, `AZURE_AI_API_KEY` | `AZURE_AI_API_BASE` | learn.microsoft.com/azure/foundry/openai/api-version-lifecycle |
| `litellm_proxy/` | none (key optional) | `LITELLM_PROXY_API_KEY` | `LITELLM_PROXY_API_BASE` | LiteLLM proxy |
| `openai_like/` | none (key optional) | `OPENAI_LIKE_API_KEY` | `OPENAI_LIKE_API_BASE` | LiteLLM |
| `llamafile/` | `http://127.0.0.1:8080/v1` | `LLAMAFILE_API_KEY` (optional) | `LLAMAFILE_API_BASE` | github.com/mozilla-ai/llamafile docs/api.md |
| `docker_model_runner/` | `http://localhost:12434/engines/v1` | `DOCKER_MODEL_RUNNER_API_KEY` (optional) | `DOCKER_MODEL_RUNNER_API_BASE` | docs.docker.com/ai/model-runner/api-reference (LiteLLM's port 22088 is not documented) |
| `lemonade/` | `http://localhost:13305/api/v1` | `LEMONADE_API_KEY` (optional) | `LEMONADE_API_BASE` | github.com/lemonade-sdk/lemonade docs/api/openai.md (LiteLLM: port 8000) |
| `xinference/` | `http://127.0.0.1:9997/v1` | `XINFERENCE_API_KEY` (optional) | `XINFERENCE_API_BASE` | inference.readthedocs.io, getting started |
| `oobabooga/` | `http://127.0.0.1:5000/v1` | `OOBABOOGA_API_KEY` (optional) | `OOBABOOGA_API_BASE` | github.com/oobabooga/textgen wiki, OpenAI API |

How some of them name models and endpoints:

- `huggingface/<provider>/<org>/<model>` is sent as `<org>/<model>:<provider>`,
  the router's way of choosing an inference provider; `huggingface/<org>/<model>`
  lets the router choose. Unlike LiteLLM, no provider mapping is fetched first.
- `bedrock_mantle/<region>/<model>` chooses the AWS region; otherwise
  `BEDROCK_MANTLE_REGION`, `AWS_REGION_NAME`, `AWS_REGION`, then `us-east-1`.
  GPT-5/6, Gemma 4 and Grok 4 models are served under `/openai/v1`, the others
  under `/v1` (the routes LiteLLM's price map records). This is Amazon
  Bedrock with a Bedrock API key; see [Bedrock and Vertex AI](#bedrock-and-vertex-ai)
  for `bedrock/` itself.
- `cloudflare/@cf/<publisher>/<model>` needs `CLOUDFLARE_ACCOUNT_ID` (or an
  `api_base`); the key is a Workers AI API token.
- `datarobot/` accepts `DATAROBOT_ENDPOINT` as the usual API endpoint
  (`…/api/v2`) and adds the LLM gateway path, as LiteLLM does.
- `azure_ai/` expects the v1 base (`…/openai/v1`): Microsoft retired the Azure
  AI Inference route LiteLLM's default used. Azure OpenAI deployments keep
  `azure/`. A model whose name contains `claude` is Claude on Foundry and is
  sent, as LiteLLM sends it, to the Anthropic Messages API at
  `<base>/anthropic/v1/messages` (a base ending in `/v1/messages` is used as
  is; one containing `/anthropic` is cut after it) with an `api-key` header and
  the `anthropic-version` above. Provider Batch does not take it.
- `databricks/` and `heroku/` have no fixed host; set `api_base` (for Heroku,
  `INFERENCE_URL` with `/v1`) or the base variable.
- Prefixes of services that ended are refused with the reason: `github/`
  (GitHub Models retired 2026-07-30), `meta_llama/` (replaced by `meta/`),
  `lambda_ai/`, `hyperbolic/`, `anyscale/`, `empower/`, `clarifai/`, `v0/` and
  `llamagate/` (its domain no longer serves the provider, so no key is sent
  there). `github_copilot/` names LiteLLM's own sign-in; Copilot is
  `copilot/<model>` here.

None of these prefixes has a structured-output capability entry, so their
documents use the JSON-text mode described below, and none is in the [pricing
catalog](pricing.md): with `llm.max_cost_per_document_usd` above zero they are
refused before any request, like every other unpriced model.

### Bedrock and Vertex AI

`bedrock/<model ID>` calls Amazon Bedrock's Converse API
(`POST https://bedrock-runtime.<region>.amazonaws.com/model/<ID>/converse`),
which serves every chat model on Bedrock: foundation model IDs, cross-region
inference profiles (`us.anthropic.…`) and ARNs. LiteLLM's route segments
`bedrock/converse/<ID>` and `bedrock/invoke/<ID>` are accepted and also use
Converse.

- Credentials, first match wins, as the AWS SDKs (and so the reference's
  LiteLLM) look for them: the deployment's `api_key` (a Bedrock API key, `env:`
  references allowed); `AWS_BEARER_TOKEN_BEDROCK`; `AWS_ACCESS_KEY_ID` and
  `AWS_SECRET_ACCESS_KEY` (with `AWS_SESSION_TOKEN`); then the profile
  `AWS_PROFILE` or `default` of `~/.aws/credentials` and `~/.aws/config`
  (`AWS_SHARED_CREDENTIALS_FILE`, `AWS_CONFIG_FILE`): static keys,
  `credential_process`, or, for SSO and role profiles, the AWS CLI's
  `aws configure export-credentials`. Credential programs run only when a
  request needs them, and their answer is reused until five minutes before it
  expires. API keys are sent as Bearer tokens; AWS keys sign each request with
  Signature Version 4 (service `bedrock`). Like the reference's configuration,
  `litellm_params` holds no other AWS setting.
- Region: `AWS_REGION_NAME`, `AWS_REGION`, `AWS_DEFAULT_REGION`, the profile's
  `region`, else `us-east-1`. Endpoint: `api_base` or
  `AWS_BEDROCK_RUNTIME_ENDPOINT` (for a VPC endpoint), else the region's
  runtime host.
- The request carries the system prompt, the text and images (PNG, JPEG, GIF,
  WebP) and `max_tokens` as `inferenceConfig.maxTokens`, and no Anthropic API
  version (Converse has none, so Claude models before 4.7 keep working); structured documents
  use the JSON-text mode. Usage includes cache read and write tokens. A
  guardrail intervention or content filter is a refusal, a `max_tokens` or
  context-window stop a truncation.
- Not implemented: web-identity, instance and container credentials directly
  (an AWS CLI profile that exports them works), Bedrock batch inference and
  SageMaker endpoints.

`vertex_ai/<model>` (or `vertex_ai_beta/`) calls Google Vertex AI: Gemini and
the other Model Garden models through its OpenAI-compatible endpoint
(`https://<location>-aiplatform.googleapis.com/v1/projects/<project>/locations/<location>/endpoints/openapi/chat/completions`,
model `google/<model>` unless the name has its publisher), and Claude models
(`vertex_ai/claude-…@<version>`) through Anthropic's Messages API behind
`…/publishers/anthropic/models/<model>:rawPredict` with
`anthropic_version: 2026-10-21`.

- Credentials, in Application Default Credentials order:
  `GOOGLE_APPLICATION_CREDENTIALS` (a service-account or user credentials
  file), gcloud's application default credentials (`gcloud auth
  application-default login`; `CLOUDSDK_CONFIG` moves them), else
  `gcloud auth application-default print-access-token`. A service account
  signs a token request (RS256 JWT); user credentials use their refresh token.
  Access tokens are cached in the process until a minute before they expire.
  All of these are the server's own, so a deployment that may not use the
  environment's credentials (`use_environment_credentials: false`, or the
  `markitai-no-key` placeholder) is refused rather than sent the gcloud login.
- Project: `VERTEXAI_PROJECT`, `VERTEX_PROJECT`, `GOOGLE_CLOUD_PROJECT`, or the
  credentials' project. Location: `VERTEXAI_LOCATION`, `VERTEX_LOCATION`, else
  `us-central1`; `global` uses `aiplatform.googleapis.com`. `api_base` replaces
  the host.
- Not implemented: workload identity federation (`external_account`
  credentials), the Compute Engine metadata server, and Vertex AI's native
  `generateContent` API, which LiteLLM uses for Gemini (the OpenAI-compatible
  endpoint serves the same models).

Neither cloud is in the [pricing catalog](pricing.md): usage is recorded and
its cost is unknown, as for other unpriced models. Both were tested offline:
the signer against the AWS Signature Version 4 test suite and curl's
`--aws-sigv4`, the Converse mapping against the API reference's shapes, and
token minting against a local token endpoint. Acceptance by AWS and Google
has not been checked live.

### Unsupported prefixes

Any other prefix fails when no usable deployment remains, with an error that
names the working route for an OpenAI-compatible endpoint, `openai/<model>`
with `api_base` (a LiteLLM proxy in front of the provider works this way).
SageMaker (`sagemaker/`, `sagemaker_chat/`) is refused: a SageMaker endpoint
with an OpenAI-compatible route works as `openai/<model>` with its `api_base`.
Gemini itself is reachable directly as `gemini/<model>` with
`GEMINI_API_KEY`.

These tables describe implemented request shapes, not live compatibility tests
against every vendor. No provider network call is required by the test suite.
`markitai doctor` accepts every prefix of this build, and so do `markitai init`'s
guided and interactive model prompts, except `bedrock/` and `vertex_ai/`, whose
cloud credentials are configured in the configuration file or environment.
The browser workbench of `serve` offers every routed provider in its provider
picker (95: both tables above, Bedrock, Vertex AI, the subscription runtimes
and any OpenAI-compatible endpoint; `ollama_chat/`, `ai21_chat/`, `cohere_chat/`,
`vertex_ai_beta/` and `openai_like/` are found under the provider they route
like), grouped as common providers, subscriptions, model makers, clouds,
inference hosts, gateways and local servers, each with its default base URL and
first key variable. Model discovery lists a provider's models through
`GET <base>/models` with the key as Bearer (Together AI's bare-array answer
included; Helicone at `/v1/models`; Cohere through its native
`/v1/models?endpoint=chat`), using the base variable when no `api_base` is
given; a provider whose list request fails, or answers in another shape, can
still take model IDs by hand. Codestral (its endpoint answers `/models` with
404), Cloudflare Workers AI and Galadriel (documented for chat completions only),
Bedrock and Vertex AI are not listed (checked 2026-10-26), so discovery answers
`unavailable` with `source: "manual"` without a request, and their model IDs are
entered by hand.
The subscription runtimes list their models through the installed runtime. The
connection check works for every prefix.

Deployment `api_key` and `api_base` take precedence. A saved provider selected
by `model_info.provider_id` can supply missing deployment credentials and base
URL. Provider-specific environment variables then supply missing values: the
key variables above, and `<PREFIX>_API_BASE` for the built-in providers
(`OPENAI_API_BASE`, then `OPENAI_BASE_URL`, for OpenAI), so an entry you write
without `api_key` sends that key variable to whatever `api_base` it names.
Explicit missing environment references do not fall through to other
credentials. An entry whose key is `markitai-no-key` (what the `serve` settings
page saves for a connection without a key at an address, see
[service settings](service-settings.md#credentials-and-partial-updates)) sends
that word and never reads a key, endpoint or cloud credential from the
environment; nor does one with `use_environment_credentials: false`. Azure requires `api_version` or
`AZURE_API_VERSION`. Endpoint construction preserves query parameters and
encodes the Azure deployment name as one URL segment.

`litellm_params.max_tokens`, followed by explicit `model_info.max_tokens`, sets
the output cap; a cap above a reviewed model's output limit is lowered to that
limit. Requests to OpenAI itself, and GPT-5-or-later and o-series names at other
OpenAI or Azure addresses, name it `max_completion_tokens` (OpenAI deprecated
`max_tokens`, and its reasoning models refuse it); other chat requests use
`max_tokens`. Without a configured cap, Gemini, DeepSeek and Anthropic models in
the reviewed [price catalog](pricing.md) (and Claude on Foundry under the same
name) are asked for their listed output limit within the room their context
window leaves after the prompt, as Python
markitai does (an estimate of the prompt plus 1,600 tokens per image and a
fifth, at least 2,000, of headroom; at most 128,000 and at least 1,000, or
4,000 for a prompt with more than 20 table rows): Gemini Flash gets 65,536
instead of a provider default that cut a ten-page visual batch off mid-answer.
Anthropic requires a cap; a model not in the catalog gets 8,192. OpenAI and
other providers use their own default (OpenAI's is the model's limit, and
OpenAI counts the cap against its per-minute token limit). Nonpositive explicit
deployment caps fail before a request. A truncated answer names its output token count and
`litellm_params.max_tokens`. This implementation does not ship LiteLLM's
tokenizer; a declared `model_info.max_input_tokens` sizes document chunks (see
[below](#structured-documents-and-complete-long-text)).

`litellm_params.reasoning_effort` controls how much a reasoning model thinks
before it answers: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`
(LiteLLM's name for the setting). Reasoning tokens are output tokens, so a model
that thinks at length can spend the whole `max_tokens` budget before writing the
cleaned document; the request then fails with `LLM output was truncated by its
token limit: reasoning used all N output tokens; lower
litellm_params.reasoning_effort or raise max_tokens`.

| Deployment | Without the setting | With a value |
|---|---|---|
| `deepseek/` | `thinking: {"type": "disabled"}`: clean-up, transcription and descriptions need no reasoning | `none` disables thinking; other values enable it and send `reasoning_effort`, which DeepSeek maps to `low`, `high` or `max` |
| `openai/` models that accept `none` (GPT-5.1 to GPT-5.6, `gpt-6-luna`, `gpt-6-sol`, as LiteLLM 1.104.2 lists them) | `reasoning_effort: "none"`: they reason by default | sent as `reasoning_effort` |
| Other OpenAI-compatible endpoints (other `openai/` models, `gemini/`, `openrouter/`, `azure/`, …) | nothing; the model's own default applies | sent as `reasoning_effort`; the provider decides which values a model accepts and rejects others with HTTP 400 |
| `anthropic/` | Claude Haiku 5.5, which thinks unless told not to: `thinking: {"type": "disabled"}` (not on a forced tool call, which does not think). Other models: nothing (Opus 5.5 and Fable refuse the switch; older models do not think by default; Sonnet 5.5's own switch, `between_tools`, is not verified here) | only `none` is accepted; other values are a configuration error |
| `azure_ai/` Claude models | nothing (Python markitai sends the Haiku 5.5 switch to `anthropic/` only) | only `none` is accepted; other values are a configuration error |
| `copilot/`, `claude-agent/`, `chatgpt/` | the runtime's default | not accepted, like `max_tokens` |

```json
{"model_name": "default", "litellm_params": {"model": "deepseek/deepseek-flash", "max_tokens": 8192, "reasoning_effort": "low"}}
```

Image requests reuse routing, authentication, budgets and retry behavior.
OpenAI-compatible requests carry a data URL; Anthropic receives native base64
image blocks. JPEG, PNG, WebP and GIF MIME types are accepted. Deployments
explicitly marked `supports_vision: false` are excluded from image requests.
The caller supplies encoded image bytes; compression and image decoding belong
to the image conversion stage. The orchestrator supplies complete PDF/TIFF page
sets and screenshots; [image enrichment](image-enrichment.md) adds structured
caption/description analysis through this same transport.

### Request failures

The client applies configured request timeouts and a connect timeout capped at
15 seconds. Redirects are rejected so credentials cannot be forwarded to an
unexpected endpoint. Provider error bodies are inspected for retry classification
and structured token usage.

The client goes through the proxy decision of [fetching](fetch.md#proxies): an HTTP(S)
proxy from the environment, else the manual system proxy (re-read at most every
30 seconds). A SOCKS environment proxy is refused by name, and a failure to reach a
proxy names it (host, port, where it is set; never credentials).

A failed request names its cause in the CLI message, the report and `--json`
error and the log, and ends with the deployment and its host:

- no response: `LLM request failed: connection refused`, `…: host name not
  resolved (…)`, `…: TLS handshake failed (…)`, `…: connection closed without a
  response (…)`, or `LLM request timed out: no response within 120 s`; the
  parenthesis is the operating system's or TLS library's reason;
- an HTTP refusal: `LLM returned HTTP 400 (invalid_request_error): Invalid
  'max_tokens': … (deployment openai/gpt-6-luna at api.openai.com)`, with the
  provider's error type or code and its message, cut at 200 characters;
- a response that is not JSON: its status, content type, size and an HTML
  page's title, never its text.

Provider messages are cleaned before they are shown: the deployment's key and
other credential-like words (`sk-…`, `Bearer …`, long opaque tokens) become
`[REDACTED]`, URLs lose userinfo, query and fragment, quotations longer than 40
characters become `'…'`, and a message that repeats any stretch of the prompts
or document becomes `[message withheld: it repeats the request]`. Errors never
contain the endpoint's path or query, authorization headers or response
payloads. A refusal whose body names a recognized cause gets a fixed phrase
after its status instead of the provider's message:
`LLM returned HTTP 403: the model is not available in this region`,
`…: the account's quota or billing does not allow this request` or
`…: the model is unavailable`. The model connection test in `serve` names a
missing response's cause the same way (`Model connection request failed:
connection refused`) and uses the same fixed phrases, but of a provider's
refusal it shows only the error type or code, never its message:
`Model connection returned HTTP 401 (invalid_request_error/invalid_api_key)`. A failed test
also answers a `reason`: `no_key` (with `key_variable`, the variable that is not
set on the server; found before any request is sent), `key_rejected` (401/403),
`model_not_found`, `rate_limited` (429 or an exhausted quota or balance),
`network`, `timeout` or `provider_error`; the workbench explains each in the
reader's language.

## Retries, budgets and usage

`router_settings.num_retries` means additional attempts after the first attempt
in a group. Connection failures, temporary HTTP failures (408, 409, 5xx) and rate
limits may retry. A request is never sent again unchanged after the provider
answered it: a response without any text (the error names its finish reason)
moves only to a deployment of the group not yet tried, without backoff, and an
answer rejected by validation is followed by a request whose system prompt names
the rejection. A request that timed out after it was sent (no response, or a
successful response that stopped arriving) is sent again once, after the usual
backoff and within `num_retries`, never more than once per request: the
provider may already have completed and billed the first one without a usage
report, so its deployment's usage row counts it in
`incomplete_request_observations` and `timed_out_requests`, the cost is
incomplete (`Cost is incomplete: 1 request to … timed out.`), a positive
`llm.max_cost_per_document_usd` stops admitting further requests, and the
document gets the warning `LLM request to … timed out (…) and was sent again
once; the provider may have billed the first request, so the cost is
incomplete`. A connection that could not be opened in time never reached the
provider and retries like any connection failure. A successful response cut off
while being read for another reason is not sent again: the provider completed
and billed it. This applies to HTTP API deployments; a subscription runtime's
timeout is not resent. A configured fallback group can still run.
Billing/payment/insufficient-quota failures stop the entire operation. A 429 is
a rate limit, retried even when its message mentions billing or payment, unless
it carries the `insufficient_quota` code; an HTTP 402 always stops. Truncated
output is rejected instead of being accepted as a complete document.

An authentication or permission refusal excludes that deployment for the rest
of the `LlmRuntime`, so later requests and documents of the run skip it. The
refusals are HTTP 401/403 responses without a billing or quota marker, other
non-temporary responses that name a missing, unavailable or regionally blocked
model, and a [subscription runtime](subscriptions.md) that is signed out, reports a
non-subscription account or reports an authentication failure. The request then
moves at once to another eligible deployment of the same group under the group's
routing strategy: there is no backoff and `num_retries` is not consumed, while
each attempt still counts toward `llm.max_requests_per_document` and paid usage
on the refused response is recorded as usual. Each excluded deployment produces
one warning per run that names its configured model, never a credential or
endpoint, for example `LLM deployment openai/gpt-6-luna failed authentication
and is skipped for this run`; a deployment refused for a regional block is
`… is not available in this region and is skipped for this run`, and one
refused for a missing or unavailable model is `… is unavailable and is skipped
for this run`. Only when every deployment of the group is excluded does the
authentication error stand; configured fallback groups then run as
usual, and later requests fail that group without a network call. A group with a
single deployment identity keeps the earlier rule: its authentication failure is
not excluded, does not retry the same group, and a fallback group can still run.

The warning accompanies the document whose request observed the refusal,
including pure-mode enhancement and image analysis; a document whose
enhancement still fails with `on_failure: fail` reports only its final error. LiteLLM 1.100.1's
`Router.should_retry_this_error` likewise moves an authentication or permission
error to another deployment only when the group has more than one, but spends a
retry on the move and cools a 401 deployment down only for `cooldown_time`
(5 seconds by default) rather than for the run.

Backoff starts at one second and doubles. A `Retry-After` value replaces that
delay: whole or decimal seconds, a duration such as `1.204s`, `750ms` or
`6m0s`, or an HTTP date; without the header, a 429 or 503 body that says when
to retry (Gemini's `retryDelay`, "try again in 3.5s") is used the same way.
Delays round up to whole seconds. Individual delays and total backoff sleep
across all groups are capped at 60 seconds; exhausting the sleep allowance
returns the last failure. HTTP request timeouts are separate from this sleep
allowance. The request budget is checked before requests and before scheduling
another sleep, so an exhausted budget does not wait needlessly; its error then
keeps the failure that spent the last request (`LLM per-document request budget
exhausted after a failed request: …`).

Every request sent again is reported as a warning of its document, and each
counts toward `llm.max_requests_per_document`: a transport retry (`LLM request
to openai/gpt-5.6-luna failed (LLM returned HTTP 503) and was sent again`), a move
to another deployment after an empty answer, a structured answer discarded by
validation together with the mode of the next request, a rejected visual
batch, and image analysis falling back to separate caption and description
requests. A discarded answer is paid; its usage stays in the totals. The
structured resend is reported once its outcome is known: when a later answer
was used, as an information note (`The model's tool-call answer was not used
(…); the request was sent again in JSON-schema mode and a later answer was
used`, notice `llm.answer_retried`, shown on the terminal with `-v` and always in
reports and JSON); when none was, as a warning (`LLM tool-call answer was
rejected (…); the request was sent again in JSON-schema mode`).

`llm.max_requests_per_document` counts every HTTP attempt, including failed
requests, transport retries and fallback groups. Zero disables this budget.
One conversion-scoped counter is shared by document enhancement, image analysis
and structured-answer fallback calls. Parallel document chunks explicitly share an
`Arc<Mutex>` accounting context; creating a worker does not reset its allowance.
Nested conversions restore the caller's
counter on return or unwinding; each independent conversion starts a new budget.

`LlmRuntime` supplies the request capacity for one caller-controlled run. A CLI
run shares one instance across its file and URL workers, using `llm.concurrency`
(default 10); `--llm-concurrency` changes this cap independently of batch worker
concurrency. Text and image requests use the same capacity. Native callers can
pass a borrowed runtime through `ConvertContext.llm_runtime`; cloning a runtime
shares its capacity. A supplied runtime's constructor limit is authoritative,
even if individual conversions have different configuration values. Separately
constructed runtimes are independent. Calls without a supplied runtime create
a local instance for their enhancement; independent binding calls therefore do
not acquire a process-wide or cross-process limit automatically.

Waiting requests enter in arrival order. A permit covers the HTTP attempt,
including reading and interpreting its full response, and is released on success
or error. Retries reacquire capacity; backoff sleep holds no permit. Persistent
cache hits return without acquiring one. Queue waiting is separate from the
configured HTTP timeout, which applies after admission to a request. This
blocking limiter introduces no async runtime dependency and does not cancel
already queued conversions. Hosts with an async event loop should dispatch the
blocking conversion outside that loop. CLI interruption continues to use the
batch scheduler's drain behavior.

Clones also merge identical active typed text chunks and first visual batches.
Identity includes the complete rendered prompts, ordered image bytes and MIME,
resolved credentials/endpoints/model pool, routing and budget policy, and the
semantic cache key. A per-runtime salted digest stays in memory and is excluded
from Debug and disk records. Changing credentials, endpoint, prompt, image bytes
or policy prevents joining an active request. Ordinary persistent cache lookup
keeps its existing credential-independent contract.

Only the owner sends HTTP and records paid usage. Waiters consume a successful
semantic answer after applying their own source/metadata checks; they acquire
no HTTP permit and record zero new usage. The whole conversion reports a cache
hit only when every needed chunk/batch was reused and no other stage made a paid
request. An owner finishes its attempted cache write before publishing to
waiters. An unwritable cache still warns; a validated answer can be shared with
already waiting callers without claiming durable persistence.

Owners release and wake waiters on error, cancellation or unwinding. Errors,
attempt counters and usage are never shared; a surviving waiter may become a new
owner and recheck the cache under its own budget. Visual cancellation is checked
while waiting at bounded intervals and detaches only that caller. This adds no
new public cancellation API or force-cancellation of an active HTTP call.

The temporary table admits at most 128 active identities; each shared answer is
limited to 16 MiB of serialized JSON and retained answers together to 64 MiB.
These bound serialized payload retention, not Rust object overhead or process
RSS. Overflow bypasses merging and runs the existing request path. Completed
entries leave the table immediately; outstanding waiters retain only their
answer until consumption. There is no new long-lived in-memory result cache.
Cache-disabled, refresh and matching bypass-pattern requests do not merge.
Later visual cleaner batches, pure processing and caption/description analysis
remain independent; unrelated runtimes and different processes do not merge.

The reference also scopes its shared runtime to a batch or service job; its
independent processors may own separate semaphores. This implementation does
not add provider-specific concurrency, rate-per-minute quotas or a budget shared
across unrelated documents. The native adaptive metrics described above remain
scoped to the caller-owned runtime.

Public usage retains `requests`, `input_tokens`, `output_tokens` and `cost_usd`
per model. `usage.requests` counts parsed successful HTTP responses and HTTP
error responses containing structured usage, including paid invalid answers and
responses preceding retries or image-analysis fallback. It differs from the
attempt budget. Response model identifiers are used when present. Anthropic
cache-read and cache-creation input tokens remain part of public input totals.

The [bounded offline pricing catalog](pricing.md) estimates reviewed first-party
token charges for current OpenAI, Anthropic, Gemini and DeepSeek models, and an
OpenRouter request costs the charge OpenRouter reports. Numeric `cost_usd` is
the known priced subtotal; additive `priced_requests`, `unpriced_requests`,
`cost_status` and snapshot provenance make incomplete coverage explicit. Unknown is not free. Standard and Batch
classes apply per observed attempt before semantic validation. No live pricing
lookup or new custom-rate configuration is introduced.

`llm.max_cost_per_document_usd` is a shared conversion continuation limit. Zero
disables it. Positive limits require a verified selected tariff (or OpenRouter,
which reports its charge) before network admission. Exact fixed-point observed spend greater than the cap, or incomplete
pricing after an observed response, stops subsequent admissions. The crossing
response and already admitted concurrent work are retained. The cap does not
reserve speculative dollars or promise a hard invoice ceiling. See the pricing
page for conservative float-to-fixed conversion, supported identities and
Provider Batch limitations.

Rust's additive `convert_detailed` and detailed context/publication entrypoints
retain the original error plus its document's already recorded usage. The old
`convert` signatures and error variants stay unchanged. The C JSON envelope and
Node/Python/Go errors now carry optional terminal usage, including late publication
failures. Pre-model errors carry no JSON usage; a recorded zero-token response is
still distinguishable. Embedded image analysis may retain the established successful
fallback with warnings; standalone image terminal errors retain their usage too.
CLI/report, persisted history, REST and MCP retain the shared attempt diagnostics
contract. Pricing completeness accompanies observed rows; native panics still do
not recover unobserved provider usage.

## Prompt selection

The caller supplies a source label separately from the document. Text requests
choose `cleaner_*` in pure mode, `url_enhance_*` for HTTP(S) sources, and
`document_process_*` for files. Image requests choose `document_vision_*`;
with `--screenshot-only`, `prompts.dir/screenshot_extract_system.md` and
`screenshot_extract_user.md` (Python markitai's names) come first when present.
The root conversion layer can redact source labels before passing them here.

For each system/user template, precedence is its explicit `prompts.<name>` path,
`prompts.dir/<name>.md`, then freshly authored built-in text. A missing explicit
file falls through to the next source. A present unreadable or non-UTF-8 file
fails with a configuration diagnostic. Default `~/.markitai/...` paths honor
`MARKITAI_HOME` isolation.

Templates support `{source}`, `{timestamp}`, `{mode_rules}`, `{content}` and
`{metadata_section}`. For non-pure text processing the metadata placeholder carries
the typed JSON contract. The same contract is appended to custom system templates;
custom prompts do not bypass response validation. The first non-pure visual batch
also supplies its typed contract through this placeholder. Later visual cleaner
batches and pure requests leave it empty; pure behavior is unchanged.
Document content is inserted last so literal braces inside a document are not
interpreted as template placeholders. The default user template is exactly
`{content}`: pure mode sends the supplied Markdown unchanged. Its fresh system
instructions request byte-preserving frontmatter behavior. Non-pure formatting
and profile application remain the responsibility of the output pipeline.

## Structured documents and complete long text

Non-pure text from files and URLs uses a typed document response:

```json
{"cleaned_markdown":"Faithful Markdown", "frontmatter":{"description":"A concise description", "tags":["topic"]}}
```

The description and each tag must be nonblank strings. Description whitespace is
collapsed and values over 150 Unicode characters are shortened to 147 plus `...`.
Tags remove quotation marks, convert whitespace and colons to hyphens and are
limited to 30 Unicode characters each. An empty normalized tag collection fails.
Other model fields are ignored: the application owns title, source and processing
time. Only enhanced output receives generated description/tags; base metadata is
retained separately. A source identified as a social post requests metadata while
preserving its body verbatim.

Typed document requests use the capability ladder below. The final JSON-text
mode accepts a bare JSON object or a complete Markdown JSON fence. Arbitrary
plain Markdown is not promoted into successful metadata, including with custom
document prompts. Every decoded mode passes the same metadata and source-content
checks. Pure text retains its separate plain contract; image analysis uses its
caption/description/text schema through the same ladder.

Before a request, fenced and indented code, inline code, math, images, link
addresses, reference definitions and HTML/comment markup are replaced by
collision-safe source-owned tokens. Their values never become system instructions.
A link keeps its words in front of the model and hides only its address
(`[North pier tables](⟦…⟧)`); an answer that takes the address out of its link
is not used. Attribute-less styling tags (`<u>`, `<ins>`, `<mark>`, `<small>`,
`<b>`, `<strong>`, `<i>`, `<em>`) stay as they are, as in Python markitai: hidden
around a phrase, small models dropped them and every drop cost a second request,
while losing one changes no word. `<sup>`, `<sub>`, `<s>`, `<br>` and tags with
attributes stay protected. The answer
must retain the tokens exactly once in source order; original bytes are restored
only after successful validation. An oversized protected code block remains one
opaque token, so it cannot be truncated or have its internal blank lines changed
by chunking. This conservative scanner does not claim full CommonMark parsing.

Remaining text is packed at blank-line or line boundaries into at most 32,000
Unicode scalar values per chunk; an oversized prose line splits at scalar
boundaries. No tail is dropped.

When an enabled deployment declares `model_info.max_input_tokens`, a chunk must
also fit the smallest declared window among them:

- The request may fill 90% of the window. The prompt without the document (the
  system prompt with its schema, the user template and message framing) is
  estimated and subtracted; what remains is the chunk's token allowance.
- Tokens are estimated without a tokenizer, conservatively: two and a half ASCII
  characters, or one other character, per token. Prose usually needs fewer
  tokens (about four ASCII characters each), Markdown tables and digits come
  closest to the estimate.
- Deployments without the key keep the 32,000-character chunks, and a window
  never makes chunks larger than that: the answer repeats the chunk, so output
  caps and the request timeout bound the useful size more than the input window
  does.
- A window that leaves fewer than 256 tokens after the prompt, or a value that is
  not a positive integer, is a configuration error before any request.
- Visual batches are not affected; they are sized by pages and text (below).
- For servers whose window holds both the prompt and the answer (Ollama's
  `num_ctx`, llama.cpp's context size), declare about half the window: LiteLLM
  defines `max_input_tokens` as the input side only, and so does this rule.
- Provider Batch planning uses the same limit; a document that needs more than
  one chunk is still refused there. Bounded worker threads process chunks concurrently
through the same caller runtime and shared document accounting. Results merge in
source order and only the first chunk supplies metadata. For multi-chunk plans,
uncached calls plus `ceil(uncached / 5)` retry headroom must fit the remaining
request allowance before any model request. Zero disables the allowance.

Successful chunks are cached independently. All admitted workers settle before
an error is returned; no partial enhanced document is published. A retry can reuse
previously successful chunks. Empty boilerplate chunks are permitted, but the
merged document must remain nonempty and pass complete-document plausibility:
for sources of at least 200 non-whitespace characters, keep at least 20% of their
length and 30% of distinct character bigrams. Chunk checks are more permissive
when the response's character four-grams substantially come from that chunk.
These are deletion safeguards, not proof of semantic accuracy. An answer that
fails them, or changes a protected marker, is not used. When the request sent
again (it names the markers the answer lost) gets an answer that passes, the
document keeps the info notice `A model answer could not be used, so the request
was sent again; a later answer was used` (`llm.answer_retried`; the terminal
shows it with `-v`, the workbench not at all; the cost counts both requests).
When no answer passes, the warning says so in plain words: `LLM enhancement failed: the model's answer left out or changed
too much of the document, so it was not used; the converted Markdown is kept as
it was.`

Both local and ordinary URL text use the [typed persistent cache](cache.md).
Cache hits require no model request and have zero new usage; bypass controls still
refresh successful answers. Pure text and standalone caption/description analysis
bypass this document cache. Non-pure page/browser vision uses its own image-aware
batch namespace, described below. The cache stores validated semantic data, never
a provider tool-call envelope. The prompt-contract fingerprint excludes older
incompatible protocol rows from cache admission without deleting them. The randomly
selected model deployment and successful protocol rung do not enter this semantic
fingerprint.

## Structured provider protocols

Document chunks, the first non-pure visual batch and image caption/description
analysis select a common wire mode from the actual reachable model pool. Positive
weight, resolved deployment credentials, configured fallback groups and explicit
vision exclusions retain their existing routing behavior. A missing referenced
environment variable removes that deployment; a credential-free configured
endpoint remains a candidate because it may be a local server. Capability
selection itself sends no requests, and a valid cache hit does not need it.

The initial exact capability table is deliberately small:

| Provider and exact model IDs | Available modes before JSON text |
|---|---|
| OpenAI `gpt-4.1`, `gpt-4.1-2025-04-14` | Named tools, JSON schema |
| OpenAI GPT-5 and later (`gpt-5*`, `gpt-6-luna`, `gpt-6.1-sol`, …) | JSON schema (`response_format`), as Python markitai sends it to these reasoning models; LiteLLM's catalog lists structured outputs for all of them |
| Anthropic `claude-haiku-5-5` | Named tools, native JSON schema; in a live check (2026-10-09) its forced tool answer kept every protected marker of the presentation that Haiku 4.5 failed |
| Anthropic `claude-haiku-4-5`, `claude-haiku-4-5-20251001` | Native JSON schema; Haiku 4.5's forced tool answers dropped the protected markers in live checks, costing a second request |
| Anthropic `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-fable-5-1`, `claude-mythos-5-1` | Native JSON schema; these models restrict forced named tools |
| Gemini `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-flash-latest` through the OpenAI-compatible endpoint | Named tools (the forced function call LiteLLM sends; in a live check of 2026-10-17 it got a scan through that a plain-text answer lost to a recitation stop), JSON schema |
| Other or unknown IDs, including Azure deployment aliases | JSON text |

This fixed table was based on the official [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-4.1),
[OpenAI structured-output guide](https://developers.openai.com/api/docs/guides/structured-outputs),
[Anthropic structured-output guide](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
[Anthropic tool-choice rules](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)
and [Gemini OpenAI compatibility guide](https://ai.google.dev/gemini-api/docs/openai).
These are the table's protocol sources, not live verification of an account,
custom endpoint or the current provider catalog. Neither a model-name prefix nor
`supports_vision` establishes structured support. Tool and schema support are
independent bits: intersect them across all reachable candidates, then visit
supported modes in the order tools → schema → text. A pool containing an unknown
model therefore starts at text. No new public capability options are introduced.
A custom endpoint may reject the optional fields even for a documented model.

Chat tools use a forced, named function and decode exactly one expected
`tool_calls[].function.arguments`. Anthropic uses `input_schema`, native tool
choice and exactly one matching `tool_use.input`. Tool data is never executed.
Missing, wrong-name or multiple results fail validation. Chat schema mode uses
`response_format.json_schema`; Anthropic uses `output_config.format`. Both then
validate the decoded object with the same application-owned contract. Tool
responses may have no text body; this is valid when their typed data is valid.

Each non-final mode gets one schema-validation attempt. Rejection of tools/schema
parameters with HTTP 400/422 descends without resending the same shape. A provider
content filter (a finish reason starting with `content_filter`, such as Gemini's
`content_filter: RECITATION`) is an explicit refusal and its error names the
reason. The one exception is a request that transcribes scanned pages no reader
recovered (in any batch of ten pages: a batch with scanned pages is asked for as a structured document too): a recitation stop there is sent once more in the next mode of the
ladder (counted in the document's budget, named in a warning), and fails with the
same error if it is withheld again. Unrelated invalid-input errors, explicit refusal, token-limit truncation, quota and budget
failures stop the ladder; authentication failures first move to unexcluded
sibling deployments and then configured routing fallbacks before stopping.
Transport retries stay inside the router. Exhausted network/HTTP transport errors
stop the ladder in every mode and do not trigger image caption/description fallback.
The final JSON-text mode gets at most three validation attempts; each attempt
after a rejected answer adds the rejection to the system prompt, so the
attempts end early when the same rejection repeats (the next request would be
the one just answered), and a response without any text ends the ladder after
that one request. Response-size
limits are terminal resource errors, not invalid JSON to retry or downgrade. All actual HTTP attempts
use the same document budget, runtime permit and paid-usage accounting. Structured
fatal responses publish cancellation before releasing their permit.

After the final invalid JSON-text candidate, a local repair may remove trailing
commas immediately before existing object/array closing tokens, outside strings.
It runs only for responses up to 1 MiB and must pass all schema and content guards.
It adds no HTTP request. It never invents quotes, missing closers, fields, metadata
or document content; truncated/refused responses never enter repair. More general
Python-style JSON, free-text extraction and incomplete-object repair remain
unsupported. Image analysis retains its existing plain caption/description
fallback after nonfatal ladder exhaustion, within the same accounting scope.

## Structured visual documents

Non-pure PDF/Office page images and standalone image/TIFF transcription use
`MARKITAI_VISION_JSON_V1` for the first batch. The response has the same
`cleaned_markdown` plus `frontmatter.description/tags` shape as typed text.
Canonical title, source and time remain application-owned. The original base
body does not receive model metadata. Standalone alt/description analysis keeps
its separate existing image-analysis schema.

Each image is sent right after a text label naming its page (`Page 3:`, or
`Frame 2:` for a web capture's frames), as Python markitai labels them; subscription
runtimes, which receive the images as attachments, keep only the order listed in
the system prompt.

Paged documents use at most ten images per request, fewer when the pages' text
is dense: a batch carries about 8,000 estimated tokens of extracted text at most
(at least one page), since its answer repeats that text and adds what it reads in
the images, unless the smaller batches would no longer fit the remaining request
budget where ten-page ones do. The first batch completes
before later batches begin; later requests use `MARKITAI_VISION_CLEAN_V1` and
return plain Markdown without metadata. Workers share the caller's runtime and
conversion accounting context. They may complete out of order, but their bodies
are assembled in frame order and only the first batch supplies metadata.
Browser screenshot tiles describe a single web page and remain one bounded
visual request; they are not assigned fictitious document page boundaries.
Pure requests retain their existing path.

The caller supplies positive increasing frame numbers, actual image MIME and
bytes. Existing page-count limits apply before requests, and the complete image
set must be nonempty and at most 100 MiB. Complete ordered `Page number:` or
`Slide number:` comments align text to pages. Preamble and trailing material are
retained, and code examples containing those comments do not create pages.
Unless `output.page_markers=true`, page and slide comments are removed only from
final output, after this alignment and enhancement; this does not change the page mapping.
If a complete map is absent, all source text is retained once, split at safe
line/literal boundaries across batches; a warning explicitly declines precise
text-to-page alignment. No frame is dropped in that fallback.

Source literals use the same protected markers as text processing, separately
for each visual batch. Each marker must survive exactly once and in order. A
marker that stood alone on its line and that the answer dropped is put back
where only one place fits, as Python markitai does: at the start or end of the
chunk (models drop the page marker that opens a one-page PDF, or the screenshot
comment that closes it, as noise) or between neighbouring lines that survived.
A moved, altered or repeated marker, a lost inline one (code, a link address, an
inline picture) or one whose neighbours were rewritten still rejects the answer.
Scanning an image with no extracted text is valid: the output need not overlap
empty text. For substantial extracted prose, the existing deletion safeguard
also checks the visible text after protected literals are removed. Short explicit
refusals, repetitive degraded output, blank results and token-limit truncation
are rejected. These are conservative safeguards, not proof of correct OCR or
semantic fidelity; model accuracy still needs independent evaluation.

The first typed visual batch follows the provider ladder above. Later plain
cleaning batches retain at most three validation attempts for malformed output
or violated content guards, within the shared request budget; as in the ladder,
a retry names the rejection in its system prompt. The known minimum number
of uncached batches must fit the remaining budget before the first request.
Transport retries, visual validation, document fallback and image analysis use
that same budget; zero still means unlimited. Paid error and invalid responses
retain their parsed usage exactly once.

A fatal first-batch failure prevents later dispatch. Fatal authentication after
available sibling deployments and routing fallbacks, quota/billing failure and
exhausted budgets stop queued batches before HTTP admission. Already active
calls finish. A nonfatal batch failure may leave valid sibling batches cached,
but it never returns a partial enhanced document. Ordinary rendered web pages may use the remaining
budget for typed text fallback; visual-only/empty pages, PDF/Office documents
and fatal failures do not use that fallback. Existing output failure policies
govern retained base files and screenshots.

The native vision-model diagnostic uses the actual deployment and credential
resolver, reachable groups, positive weights and explicit `supports_vision`
exclusions. It reports eligible identities without a provider probe or guessing
capability from model names. An unspecified capability is eligible, not evidence
that the provider can process images.

## Repeated tails

A model, most often a vision model reading a page, can fall into a loop and end
its answer with the same passage over and over. Every complete answer is checked
for such a tail before the content guards judge it: document chunks (live,
cached and Provider Batch results), visual batches, pure text and pure image
requests, and the description and transcribed text of image analysis. A
token-limit truncation is still rejected before this check, as before.

- Two linear scans read the end of the answer: a run of identical lines (blank
  lines between copies are skipped, leading indentation counts), and a
  periodic tail of at least 16 bytes repeated back to back (found with the
  prefix function of the reversed tail; the last 256 KiB are scanned and the run
  is then followed backwards to its start). A passage within a line counts too.
- A tail counts as a loop with at least six copies that add up to 120 visible
  characters (a character outside ASCII weighs two). Passages made mostly of
  punctuation or table cells (blank form rows, rules, fill-in lines, closing
  braces) need 64 copies and 1,024 characters, more than a page holds.
- Rows that differ are not repetition: tables, lists of similar items, logs with
  timestamps and code keep every row. Repetition that ends before the answer
  does (a byte array before its closing brace) is not a tail.
- When the text sent to the model already holds the passage as often, it is the
  source's own and stays. When the answer has more, the source's number of copies
  is kept, otherwise one. Words are compared without regard to whitespace layout.
- The cut keeps whole copies: after a line break, or for a passage within a
  line, after a space or a sentence end. Up to three passes run, so a repeated
  line that itself repeats one sentence is reduced to one sentence.

The salvaged answer must still pass the marker, plausibility and refusal
checks; if too little is left, it is rejected and retried like any invalid
answer. An accepted salvage adds the warning `The model's answer ended by
repeating one passage N times: M repeated characters were removed and one copy
was kept. The answer is not cached, so a later run asks again.` and is never
written to the persistent cache. The reference applies the same idea (four
copies of a 20-character unit, or six identical lines) to visual answers only and
compares lines after trimming both ends; this build is stricter about
indentation and structure and covers the text paths as well.

## Privacy notices

The CLI shows a one-time note on stderr before data first goes to a party the
user may not have in mind. Each note is recorded as an empty marker file under
`MARKITAI_HOME/notices`, the store the remote-fetch disclosure already uses, and
is shown again only in a new `MARKITAI_HOME`. A `--quiet` run shows nothing and
records nothing, so the note comes in a later run. `serve`, `mcp` and the
language bindings install no notice host and show none.

| Marker | Shown before |
|---|---|
| `remote-images` | the first request that carries images (page renders, screenshots, pictures for `--alt`/`--desc`, OCR pages) to a deployment off this machine. Loopback endpoints (`localhost`, `*.localhost`, `127.0.0.0/8`, `::1`) are local; subscription runtimes and every other host, a LAN address included, are not. The note names the configured model. |
| `remote-strategy-<service>` | a selected remote fetch strategy (`-s defuddle`, `-s jina`, `-s cloudflare`, or the same in `fetch.strategy`) first sends a URL to its service, once per service; a URL refused before sending (local, private or credentialed) shows nothing. |
| `remote-fetch` | `auto` first falls back to remote services under an explicit `fetch.remote_consent: always` (see [fetch](fetch.md#strategy-order-and-remote-fallback)). |

With `MARKITAI_NO_VLM_OCR` set, OCR pages are read locally and only text is
sent, so no image request and no image note follows from OCR alone. The notes
inform; they never grant or record consent.
