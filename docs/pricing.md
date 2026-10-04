# Token prices and dollar limits

`cost_usd` is the subtotal of requests whose price can be established from the
bundled public-list catalog and the provider's observed token counters. It is
not an invoice. Unknown prices remain explicitly unknown; a numeric zero alone
does not establish a free request. No provider call or pricing download occurs
when looking up a tariff.

## Reviewed catalog

Snapshot `litellm-1.100.1-selected-2026-09-29` derives eight exact source keys
from LiteLLM 1.100.1's installed backup data. Two additional Claude aliases are
verified against official model pages. [Provenance](../licenses/pricing/provenance.json)
records original source digest, exact byte ranges, decimal rates, wheel RECORD
identity and verification links. [Source objects](../licenses/pricing/source-rows.json)
and the [original notice](../licenses/pricing/LiteLLM-LICENSE) remain intact.
The capture date does not establish when every original price became effective.

All prices below are USD per million tokens, Standard class:

| Exact model family | Input | Cached read | Output | Cache write 5m / 1h |
|---|---:|---:|---:|---:|
| OpenAI `gpt-4.1` | 2 | 0.50 | 8 | unsupported |
| OpenAI `gpt-4.1-mini` | 0.40 | 0.10 | 1.60 | unsupported |
| OpenAI `gpt-4.1-nano` | 0.10 | 0.025 | 0.40 | unsupported |
| Anthropic `claude-sonnet-4-5-20250929` | 3 | 0.30 | 15 | 3.75 / 6 |
| Anthropic `claude-haiku-4-5-20251001` | 1 | 0.10 | 5 | 1.25 / 2 |

Each listed OpenAI ID also accepts its exact `-2025-04-14` snapshot ID.
`claude-sonnet-4-5` and `claude-haiku-4-5` are the only added Claude aliases;
unknown suffixes are not matched. The official
[Haiku page](https://platform.claude.com/docs/en/models/haiku-4-5/overview) and
[Sonnet page](https://platform.claude.com/docs/zh-CN/models/sonnet-4-5/overview)
identify their dated snapshots and tariffs. Claude aliases may move; a returned
model outside the reviewed family makes that response unpriced.
[OpenAI's model announcement](https://openai.com/index/gpt-4-1/) supplies the
three GPT-4.1 tariffs and Batch discount; these remain a bounded dated catalog,
not a claim to cover every current or default model.

Prices apply only to these exact final endpoints:

- `https://api.openai.com/v1/chat/completions`
- `https://api.anthropic.com/v1/messages`

Custom/proxy endpoints, alternate hosts/ports/paths, regional hosting, other
providers or models are unpriced. An Anthropic response's `usage.inference_geo` keeps
the listed rates only when it is `not_available` or `global`; US-only inference is
billed at a premium and stays unpriced. An accepted model configuration is not proof
of pricing coverage. Explicit service tiers other than Standard/default or the
transport's Batch class are unpriced. The reviewed Claude band is at most
200,000 total input tokens, including cache reads and writes. Larger contexts
are unpriced. No user-supplied pricing fields are added to configuration.

## Response accounting

The actual transport class accompanies each raw response. Batch uses half the
reviewed Standard rates for these rows, including prompt caching; a later live
fallback is Standard. The discount never applies to an already accumulated
run total. This follows the providers' [Batch](https://developers.openai.com/api/docs/guides/batch)
and [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing)
documentation; billing category does not imply that every provider's Batch API
is implemented.

OpenAI prompt totals include cached reads, so their cached subset is subtracted
once before applying the ordinary input rate. Reasoning and prediction counts
are already part of output and are not added again. Anthropic base input excludes
cache reads and writes. Its cache creation totals and explicit 5m/1h breakdown
must agree; positive creation without enough TTL detail remains unpriced.
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
  "pricing_snapshot": "litellm-1.100.1-selected-2026-09-29"
}
```

Rows can also carry `cached_input_tokens`, `cache_creation_input_tokens` and
`incomplete_request_observations` (subscription runtimes that report tokens but
not a request count); `pricing_snapshot` appears only on priced rows.
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

When any observed request is unpriced, the CLI also prints
`Warning: Some observed LLM requests could not be priced. cost_usd is the known
priced subtotal; the complete cost is unknown.`

## Dollar continuation budget

`llm.max_cost_per_document_usd: 0` disables the dollar limit. Positive values
share one conversion scope across document chunks, visual batches and image
analysis. Internal sums use checked `u128` picodollars; public costs retain their
existing finite `f64` representation. The actual binary configuration value is
floored to picodollars, so the comparison can stop less than one picodollar early.
A positive sub-picodollar cap stays enabled at zero. Invalid, nonfinite or
unrepresentable budgets fail rather than saturate.

Before an outgoing request, the selected deployment must have a verified tariff
when the dollar cap is enabled. Unknown identity rejects before network activity
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
