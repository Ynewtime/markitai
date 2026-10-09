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
       "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "env:ANTHROPIC_API_KEY"}}
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

With an empty model list, a nonempty `MODEL` selects one model. Otherwise each
available API credential joins the pool: Anthropic, OpenAI, Gemini, DeepSeek
and OpenRouter. The model aliases are derived from the reference checkout's
provider-default table, not an independent assertion about current provider
availability; DeepSeek's is `deepseek/deepseek-flash`, the id DeepSeek's model
list names (the reference's `deepseek-v4-flash` is an alias of it). Pin `MODEL` or a configured deployment to control which provider
receives documents. Keys of the [OpenAI-compatible
prefixes](#openai-compatible-prefixes) do not join this pool, because no default
model is known for them; name one with `MODEL` (for example
`MODEL=groq/<model>`) or in `llm.model_list`. [Subscription providers](subscriptions.md) are never
auto-detected; configure them explicitly in `llm.model_list`.

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
- Left out because a fixed default could not be verified: DashScope (its
  current documentation gives workspace-specific hosts), GitHub Models, and the
  smaller LiteLLM entries (Codestral, llamafile, Featherless, Hyperbolic, Lambda,
  Nscale, Volcengine and others). Each still works as `openai/<model>` with its
  `api_base`.

None of these prefixes has a structured-output capability entry, so their
documents use the JSON-text mode described below, and none is in the [pricing
catalog](pricing.md): with `llm.max_cost_per_document_usd` above zero they are
refused before any request, like every other unpriced model.

### Unsupported prefixes

Any other prefix fails when no usable deployment remains, with an error that
names the working route for an OpenAI-compatible endpoint, `openai/<model>`
with `api_base`. Bedrock (`bedrock/`, `bedrock_converse/`, `sagemaker/`) is
refused because it needs AWS Signature Version 4 signing, and Vertex AI
(`vertex_ai/`, `vertex_ai_beta/`) because it needs Google Cloud service-account
authentication; neither is implemented. Gemini itself is reachable directly as
`gemini/<model>` with `GEMINI_API_KEY`.

These tables describe implemented request shapes, not live compatibility tests
against every vendor. No provider network call is required by the test suite.
`markitai doctor` accepts every prefix of this build, and so do `markitai init`'s
guided and interactive model prompts. The browser workbench of `serve` offers
every prefix of the table above as a provider card with its default base URL and
first key variable; model discovery lists a provider's models through
`GET <base>/models` with the key as Bearer (Together AI's bare-array answer
included), using the base variable when no `api_base` is given. Perplexity, Z.ai
and Fireworks AI document no OpenAI-compatible model list (checked 2026-10-02:
Fireworks lists models only through its account API), so discovery answers
`unavailable` with `source: "manual"` without a request, and their model IDs are
entered by hand. The connection check works for every prefix.

Deployment `api_key` and `api_base` take precedence. A saved provider selected
by `model_info.provider_id` can supply missing deployment credentials and base
URL. Provider-specific environment variables then supply missing values: the
key variables above, and `<PREFIX>_API_BASE` for the built-in providers
(`OPENAI_API_BASE`, then `OPENAI_BASE_URL`, for OpenAI). Explicit missing
environment references do not fall through to other credentials. Azure requires `api_version` or
`AZURE_API_VERSION`. Endpoint construction preserves query parameters and
encodes the Azure deployment name as one URL segment.

`litellm_params.max_tokens`, followed by explicit `model_info.max_tokens`, sets
the output cap. OpenAI/Azure GPT-5 and o1/o3/o4 names use
`max_completion_tokens`; other chat requests use `max_tokens`. Anthropic
requires a cap and defaults to 8192 when none is supplied. Other providers use
their own default when no cap is configured. Nonpositive explicit deployment
caps fail before a request. This implementation does not ship LiteLLM's model
catalog or tokenizer and does not size output caps from a context window; a
declared `model_info.max_input_tokens` sizes document chunks instead (see
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
| Other OpenAI-compatible endpoints (`openai/`, `gemini/`, `openrouter/`, `azure/`, …) | nothing; the model's own default applies | sent as `reasoning_effort`; the provider decides which values a model accepts and rejects others with HTTP 400 |
| `anthropic/` | nothing; extended thinking stays off | only `none` is accepted; other values are a configuration error |
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

The client applies configured request timeouts and a connect timeout capped at
15 seconds. Redirects are rejected so credentials cannot be forwarded to an
unexpected endpoint. Provider error bodies are inspected for retry classification
and structured token usage; they are never included in public errors. Public request errors
omit URLs, authorization headers, document text and response payloads. A refusal
whose body names a recognized cause gets a fixed phrase after its status instead:
`LLM returned HTTP 403: the model is not available in this region`,
`…: the account's quota or billing does not allow this request` or
`…: the model is unavailable`. The model connection test in `serve` words its
refusals the same way.

## Retries, budgets and usage

`router_settings.num_retries` means additional attempts after the first attempt
in a group. Connection failures, temporary HTTP failures (408, 409, 5xx) and rate
limits may retry. A request is never sent again unchanged after the provider
answered it: a response without any text (the error names its finish reason)
moves only to a deployment of the group not yet tried, without backoff, and an
answer rejected by validation is followed by a request whose system prompt names
the rejection. A request that timed out, or whose
successful response was cut off while being read, is not sent again: the
provider may already have completed and billed it, and that usage cannot be
recorded. A configured fallback group can still run.
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
endpoint, for example `LLM deployment openai/gpt-5.6-luna failed authentication
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

Backoff starts at one second and doubles. A numeric `Retry-After` value can
replace that delay. Individual delays and total backoff sleep across all groups
are capped at 60 seconds; exhausting the sleep allowance returns the last
failure. HTTP request timeouts are separate from this sleep allowance. The
request budget is checked before requests and before scheduling another sleep,
so an exhausted budget does not wait needlessly.

Every request sent again is reported as a warning of its document, and each
counts toward `llm.max_requests_per_document`: a transport retry (`LLM request
to openai/gpt-5.6-luna failed (LLM returned HTTP 503) and was sent again`), a move
to another deployment after an empty answer, a structured answer discarded by
validation together with the mode of the next request (`LLM tool-call answer was
rejected (…); the request was sent again in JSON-schema mode`), a rejected visual
batch, and image analysis falling back to separate caption and description
requests. A discarded answer is paid; its usage stays in the totals.

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
token charges. Numeric `cost_usd` is the known priced subtotal; additive
`priced_requests`, `unpriced_requests`, `cost_status` and snapshot provenance
make incomplete coverage explicit. Unknown is not free. Standard and Batch
classes apply per observed attempt before semantic validation. No live pricing
lookup or new custom-rate configuration is introduced.

`llm.max_cost_per_document_usd` is a shared conversion continuation limit. Zero
disables it. Positive limits require a verified selected tariff before network
admission. Exact fixed-point observed spend greater than the cap, or incomplete
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
`document_process_*` for files. Image requests choose `document_vision_*`.
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

Before a request, fenced and indented code, inline code, math, links/images,
reference definitions and HTML/comment markup are replaced by collision-safe
source-owned tokens. Their values never become system instructions. The answer
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
- Visual batches are not affected; they keep ten images each.
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
These are deletion safeguards, not proof of semantic accuracy.

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
| Anthropic `claude-haiku-4-5`, `claude-haiku-4-5-20251001` | Native JSON schema; its forced tool answers dropped the protected markers in live checks, costing a second request |
| Anthropic `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-fable-5-1`, `claude-mythos-5-1` | Native JSON schema; these models restrict forced named tools |
| Gemini `gemini-3.8-flash` through its OpenAI-compatible endpoint | JSON schema |
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
reason. Unrelated invalid-input errors, explicit refusal, token-limit truncation, quota and budget
failures stop the ladder; authentication failures first move to unexcluded
sibling deployments and then configured routing fallbacks before stopping.
Transport retries stay inside the router. Exhausted network/HTTP transport errors
stop the ladder in every mode and do not trigger image caption/description fallback.
The final JSON-text mode gets three validation attempts total; each attempt
after a rejected answer adds the rejection to the system prompt, and a response
without any text ends the ladder after that one request. Response-size
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

Paged documents use at most ten images per request. The first batch completes
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
for each visual batch. Each marker must survive exactly once and in order.
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
