# Token prices and dollar limits

`cost_usd` is the subtotal of requests whose price can be established from the
bundled public-list catalog and the provider's observed token counters. It is
not an invoice. Unknown prices remain explicitly unknown; a numeric zero alone
does not establish a free request. No provider call or pricing download occurs
when looking up a tariff.

## Reviewed catalog

Snapshot `litellm-1.106.0.dev2-selected-2026-10-11` derives 37 exact source keys
from the backup data in LiteLLM 1.106.0.dev2's wheel. That is a development
pre-release, chosen because it is the newest PyPI release with the
`claude-haiku-5-5` and `claude-sonnet-5-5` rows; every rate used equals the
provider's official price on 2026-10-11, and each row names its official pages.
Three additional Claude aliases are verified against official model pages.
[Provenance](../licenses/pricing/provenance.json) records original source digest,
exact byte ranges, decimal rates, wheel RECORD identity and verification links.
[Source objects](../licenses/pricing/source-rows.json) and the
[original notice](../licenses/pricing/LiteLLM-LICENSE) remain intact. The capture
date does not establish when every original price became effective.

All prices below are USD per million tokens, Standard class. OpenAI rows with two
values take the second for a whole request whose prompt is over 272,000 tokens:

| Exact model family | Input | Cached read | Output | Cache write | Source |
|---|---:|---:|---:|---:|---|
| OpenAI `gpt-6-astra` | 10 / 20 | 1 / 2 | 50 / 75 | 12.50 / 25 (30m) | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI `gpt-6.1-sol` | 2 / 4 | 0.10 / 0.20 | 10 / 15 | 2.50 / 5 (30m) | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI `gpt-6-sol` | 2 / 4 | 0.20 / 0.40 | 10 / 15 | 2.50 / 5 (30m) | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI `gpt-6-luna` | 0.10 / 0.20 | 0.01 / 0.02 | 0.50 / 0.75 | 0.125 / 0.25 (30m) | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI `gpt-5.6-sol`, `gpt-5.6` | 4 / 8 | 0.40 / 0.80 | 20 / 30 | 5 / 10 (30m) | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI `gpt-5.6-terra` | 2 / 4 | 0.20 / 0.40 | 12 / 18 | 2.50 / 5 (30m) | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI `gpt-5.6-luna` | 0.20 / 0.40 | 0.02 / 0.04 | 1.20 / 1.80 | 0.25 / 0.50 (30m) | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI `gpt-4.1` | 2 | 0.50 | 8 | 2 (as input) | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI `gpt-4.1-mini` | 0.40 | 0.10 | 1.60 | 0.40 (as input) | [pricing](https://developers.openai.com/api/docs/pricing) |
| OpenAI `gpt-4.1-nano` | 0.10 | 0.025 | 0.40 | 0.10 (as input) | [pricing](https://developers.openai.com/api/docs/pricing) |
| Anthropic `claude-fable-5-1` | 10 | 0.25 | 50 | 12.50 / 20 (5m / 1h) | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-fable-5` | 10 | 1 | 50 | 12.50 / 20 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-opus-5-5` | 4 | 0.20 | 20 | 5 / 8 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-opus-5`, `-4-8`, `-4-7`, `-4-6` | 5 | 0.50 | 25 | 6.25 / 10 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-opus-4-5-20251101` | 5 | 0.50 | 25 | 6.25 / 10 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-sonnet-5-5` | 2 | 0.10 | 10 | 2.50 / 4 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-sonnet-5` | 2 | 0.20 | 10 | 2.50 / 4 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-sonnet-4-6` | 3 | 0.30 | 15 | 3.75 / 6 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-sonnet-4-5-20250929` | 3 | 0.30 | 15 | 3.75 / 6 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-haiku-5-5`, prompt ≤ 100,000 | 0.10 | 0.01 | 0.50 | 0.125 / 0.20 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-haiku-5-5`, prompt > 100,000 | 0.50 | 0.05 | 2.50 | 0.625 / 1 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Anthropic `claude-haiku-4-5-20251001` | 1 | 0.10 | 5 | 1.25 / 2 | [pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| Gemini `gemini-3.5-flash-lite`, `gemini-flash-lite-latest` | 0.30 | 0.03 | 2.50 | none | [pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite) |
| Gemini `gemini-3.1-flash-lite` | 0.25 | 0.025 | 1.50 | none | [pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite) |
| Gemini `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-flash-latest`, through 2026 | 0.75 | 0.075 | 3.75 | none | [pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-flash) |
| DeepSeek `deepseek-flash`, `deepseek-v4-flash`, peak / off-peak | 0.30 / 0.15 | 0.006 / 0.003 | 1.20 / 0.60 | none | [pricing](https://api-docs.deepseek.com/quick_start/pricing) |
| DeepSeek `deepseek-v4-pro`, peak / off-peak | 1.32 / 0.66 | 0.044 / 0.022 | 3.96 / 1.98 | none | [pricing](https://api-docs.deepseek.com/quick_start/pricing) |
| OpenRouter, any model | the charge OpenRouter reports | | | | [usage accounting](https://openrouter.ai/docs/use-cases/usage-accounting) |

Each listed GPT-4.1 ID also accepts its exact `-2025-04-14` snapshot ID; the
GPT-6 and GPT-5.6 IDs are their own snapshots, and OpenAI's
[model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol) routes
the `gpt-5.6` alias to GPT-5.6 Sol. Their model pages and the pricing page price
a prompt over "272K" input tokens, cached reads and writes included, at the
long-context rates for the whole request. OpenAI's K is 1,000 there: the pricing
page splits short (≤272K) from long (>272K) input, the
[GPT-5 page](https://developers.openai.com/api/docs/models/gpt-5) the figure
comes from lists "Maximum input tokens: 272,000", and the same model pages write
a 1,050,000-token window as "1.05M". The long rates therefore start at 272,001
tokens, as LiteLLM applies them.

`claude-sonnet-4-5`, `claude-haiku-4-5` and `claude-opus-4-5` are the only added
Claude aliases, each listed with its dated snapshot on its official model page;
unknown suffixes are not matched. From the 4.6 generation on, every Claude ID is
itself a pinned snapshot ([models overview](https://platform.claude.com/docs/en/about-claude/models/overview)).
Those models keep their listed rates across the whole 1M-token window, except
`claude-haiku-5-5`, which prices a request by its whole prompt: over 100,000
input tokens, cache reads and writes included, every token of that request
takes the higher rates. Claude aliases may move; a returned model outside the
reviewed family makes that response unpriced.

Gemini rows are the paid tier's Standard rates for text, image and video input.
`gemini-flash-lite-latest` and `gemini-flash-latest` are Google's moving aliases:
Google does not name their current targets, so each is priced as its own family
at its LiteLLM row, which equals the Gemini 3.5 Flash-Lite and 3.8 Flash prices,
and a later swap keeps that price until this catalog is reviewed again. Google
lists the 3.8 and 3.6 Flash prices "through December 31, 2026" and doubles them
from January 1, 2027, so a request that ends at or after 2027-01-01T00:00Z is
unpriced.

DeepSeek bills Beijing time 09:00–12:00 and 14:00–18:00, Monday to Friday, at
its peak rates and every other hour, weekends and Chinese public holidays
included, at half. A request is priced when it was sent and answered within one
period. Whether a weekday peak hour falls on a public holiday is known only for
dates reviewed against the State Council's schedule, now 2026-10-08 to
2026-12-31; outside them a weekday peak-hour request is unpriced. The legacy
`deepseek-v4-flash` name is served and billed as `deepseek-flash`.

OpenRouter returns the amount it charged in every response's `usage.cost`
(USD credits); that figure, rounded to the nearest picodollar, is the cost for
any model it routes to. A bring-your-own-key response (`is_byok: true`) is also
billed by the upstream provider and stays unpriced, as does one without `cost`.

Prices apply only to these exact final endpoints:

- `https://api.openai.com/v1/chat/completions`
- `https://api.anthropic.com/v1/messages`
- `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
- `https://api.deepseek.com/chat/completions` and `https://api.deepseek.com/v1/chat/completions`
- `https://openrouter.ai/api/v1/chat/completions`

Custom/proxy endpoints, alternate hosts/ports/paths, regional hosting (such as
OpenAI's `us.`/`eu.` data-residency hosts, billed 10% higher), other providers
(including Bedrock and Vertex AI) or models are unpriced. An Anthropic response's
`usage.inference_geo` keeps the listed rates only when it is `not_available` or
`global`; US-only inference is billed at a premium and stays unpriced. An
accepted model configuration is not proof of pricing coverage. Explicit service
tiers other than Standard/default or the transport's Batch class are unpriced.
Claude 4.5 models are priced up to 200,000 total input tokens, cache reads and
writes included: that is their context window on the Claude API
([context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)).
Sonnet 4.5's former 1M-token beta and its above-200K rates are no longer
offered, so a larger prompt is not billable there and stays unpriced. No
user-supplied pricing fields are added to configuration.

## Response accounting

The actual transport class accompanies each raw response. Batch uses half the
reviewed Standard rates for the OpenAI and Anthropic rows, including prompt
caching; a later live fallback is Standard. Gemini, DeepSeek and OpenRouter have
no reviewed Batch class. The discount never applies to an already accumulated
run total. This follows the providers' [Batch](https://developers.openai.com/api/docs/guides/batch)
and [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing)
documentation; billing category does not imply that every provider's Batch API
is implemented.

OpenAI prompt totals include cached reads and cache writes
(`prompt_tokens_details.cache_write_tokens`), so both subsets are subtracted
once before applying the ordinary input rate, as OpenAI's
[prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
computes input cost. Cache writes cost 1.25× ordinary input on GPT-5.6 and later;
the guide gives earlier models, GPT-4.1 included, no additional cache-write
charge, so their written tokens are priced as ordinary input. Text and
image counts only split the prompt by modality. Reasoning, prediction and text
output counts are already part of output and are not added again. Anthropic base input excludes
cache reads and writes. Its cache creation totals and explicit 5m/1h breakdown
must agree; positive creation without enough TTL detail remains unpriced.
`output_tokens_details.thinking_tokens` is part of output, and a null counter
reports nothing; any server tool request leaves the response unpriced.
Gemini's OpenAI-compatible usage leaves thinking out of `completion_tokens` but
counts it in `total_tokens`, so a Gemini response needs `total_tokens`, and the
tokens it adds to prompt and completion are billed as output, as Google bills
thinking. DeepSeek's `prompt_cache_hit_tokens` (also reported as
`prompt_tokens_details.cached_tokens`) take the cache-hit rate and the remaining
prompt tokens the cache-miss rate; hits and misses must add up to the prompt.
Unknown billing categories, audio charges, contradictory counters, missing
usage, unreviewed response models and arithmetic overflow cannot yield a known
price. Image requests use reported metered tokens; pixels never become invented
billing tokens. Tool charges, credits, taxes, discounts and subscriptions are
outside this catalog.

Raw successful responses and error JSON containing usage are recorded before
semantic validation. A paid invalid document therefore keeps its observed
usage. A transport failure without observed usage does not fabricate a token
count or charge. Local cache hits and coalesced waiters add no paid event;
only the actual request owner records the response.

Per-model usage keeps its existing requests/input/output/cost fields and adds:

```json
{
  "requests": 2,
  "input_tokens": 1002,
  "output_tokens": 103,
  "cost_usd": 0.0028,
  "priced_requests": 1,
  "unpriced_requests": 1,
  "cost_status": "partial",
  "pricing_snapshot": "litellm-1.106.0.dev2-selected-2026-10-11"
}
```

Rows can also carry `cached_input_tokens`, `cache_creation_input_tokens` and
`incomplete_request_observations` (subscription runtimes that report tokens but
not a request count); `pricing_snapshot` appears only on priced rows; OpenRouter's reported charges
carry `openrouter-reported-cost`.
`complete` means every observed request in that row was priced; `partial` means
both priced and unpriced requests; `unknown` means none was priced. An explicit
zero-token response can be completely priced at zero. No model event is invented
when no response was observed. Old persisted rows without these fields remain
unpriced even if their old numeric subtotal is nonzero. Aggregates with multiple
sources retain sorted unique `pricing_snapshots`; they do not overwrite the
source with the latest one. A usage delta retains conservative observed source
provenance rather than claiming an itemized ledger across historical catalogs.
Old consumers that ignore these additions cannot distinguish unknown from zero.

## Where costs appear

Whenever model requests were observed, a summary object
`pricing: {priced_requests, unpriced_requests, cost_status, pricing_snapshots}`
(plus `incomplete_request_observations` when nonzero) accompanies the existing
`cost_usd`:

- CLI `--json`: each item, and `totals.pricing` for the run; per-model rows stay
  under `llm_usage`.
- Batch reports: the usage blocks ([reports](reports.md)).
- REST job items ([REST service](serve.md)).
- MCP single-source results, `batch_convert` items, and an aggregate `pricing`
  and `cost_usd` on `job_status` ([MCP](mcp.md)).

When any observed request to a priced-API model is unpriced, the CLI also prints
a warning naming each model and the reason, for example `! Cost is
incomplete: 2 requests to deepseek-chat have no reviewed price. cost_usd is the
known priced subtotal; the complete cost is unknown.` The model is the one the
provider reports, the same key the cost breakdown uses. Up to three models are
named and the rest are counted (`and 2 more models`). A row whose provider
reported no token counts says so instead (`reported no usage counts`), because
that is a different root cause. Documents served by a
[subscription runtime](subscriptions.md) carry its own notice instead; requests
through one never carry a dollar quote.

## Dollar continuation budget

`llm.max_cost_per_document_usd: 0` disables the dollar limit. Positive values
share one conversion scope across document chunks, visual batches and image
analysis. Internal sums use checked `u128` picodollars; public costs retain their
existing finite `f64` representation. The actual binary configuration value is
floored to picodollars, so the comparison can stop less than one picodollar early.
A positive sub-picodollar cap stays enabled at zero. Invalid, nonfinite or
unrepresentable budgets fail rather than saturate.

Before an outgoing request, the selected deployment must have a verified tariff
(or be OpenRouter, which reports its charge) when the dollar cap is enabled. Unknown identity rejects before network activity
and without consuming an attempt. After a response, exact observed cost is added.
The response crossing the cap is retained; subsequent admissions fail when the
sum is strictly greater than the cap. Equality permits a further attempt. If an
observed response cannot be priced completely, subsequent budgeted requests stop.

Already admitted concurrent requests finish and still contribute their usage.
The cap is therefore a continuation circuit breaker, not a hard invoice ceiling:
there is no speculative token estimate or dollar reservation. Request-count
limits, cancellation, retry and output-failure policies still apply. A refused
admission releases its routing reservation without a metric observation.
Provider Batch rejects positive dollar caps before cloud submission because a
submitted batch cannot enforce this per-document continuation policy mid-run.

## Packaging and verification scope

The official packaging scripts include all four files under `licenses/pricing`
in CLI, native, npm, installed-wheel and static-Go bundles. Wheel supplementation
places the directory below its `dist-info/licenses` tree and rebuilds RECORD;
package checks compare the exact nested paths and bytes. Direct ad hoc Cargo,
Maturin or npm builds are not the verified distribution workflow. The pricing
notice does not resolve unrelated dependency-license gaps or constitute legal
review.
