# Provider discovery and connection checks

The native core exposes `provider_management::discover(&Value)` and
`provider_management::probe(&Value)`. The service uses these for
`POST /api/settings/llm/model-discovery` and `POST /api/settings/llm/test`.
Settings resolve saved connection/deployment references before network work and
release their lock. Requests carry private snapshots; discovery and probes do
not save credentials or change running conversions.

Discovery supports OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, Azure,
Ollama, custom OpenAI-compatible endpoints and every OpenAI-compatible prefix of
the routing table (`docs/llm.md`) through `GET <base>/models`, Together AI's bare
array included; Perplexity, Z.ai and Fireworks AI answer `unavailable` with
`source: "manual"` without a request, since they document no OpenAI-compatible
model list, and their model IDs are typed by hand. The subscription providers
`copilot` and `claude-agent` ask their installed official runtime for its model
list on every request (never cached, because a stored login can change);
`chatgpt` returns the adapter's fixed `gpt-5.5` allowlist marked
non-authoritative. They reject an API key or base URL and report `unavailable`
when the runtime or its login is missing. Markitai itself reads no browser or
CLI authentication store; see [subscriptions](subscriptions.md).
Quick-add `/api/settings/llm/detected` is an array derived from nonempty process
environment keys. It does not contact providers; its candidates already have an
environment credential, so `requires_api_key` is false.

Each provider uses its model-list protocol, including Anthropic headers, Gemini
query authentication, Ollama tags and Azure versioned regional models. Discovery
bases for Gemini/Ollama differ from their OpenAI-compatible inference bases.
Azure results are partial and non-authoritative: base model names do not prove a
deployment with that name exists. Pagination indicators also produce an explicit
partial first-page result. At most 1,000 returned records and 2 MiB response bytes
are accepted; excess data fails instead of silently truncating the list.

A process cache holds at most 32 connection identities. Its key hashes the
provider, full endpoint and resolved credential; raw keys are not cache labels.
Successful results are fresh for 300 seconds. A refresh failure may return a
previous result less than 24 hours old with `partial`, `cached` and `stale` flags.
Failures do not renew that result's age. Requests overlapping an active refresh
share its result, including failures. Distinct credentials never share a model
list. Cache contents are never persisted.

HTTP discovery has a three-second connection timeout and ten-second request
limit; the service uses a twenty-second backstop. Probes have a fifteen-second
request limit and thirty-second service backstop. Redirects are never followed.
The service admits at most eight provider network operations at once, separately
from document conversion. A timed-out service wait does not falsely release a
slot while its blocking request is still alive. Response/status errors contain
fixed descriptions, never upstream response text or credential-bearing URLs.

A connection check reuses native deployment resolution, endpoint construction
and provider payload construction. It requests exactly `Reply with exactly OK.`
with a 16-token output budget, using the provider's applicable token field. It
makes one call to the selected model: no fallback, retry, document conversion,
prompt files, cost lookup or persistent LLM cache. The response must be a valid
completion envelope; successful wording need not be exactly `OK`. Provider
failure returns HTTP 200 with `ok:false`; invalid request/reference errors keep
the settings API's error status. The response detail is bounded to 300
characters. Azure probes use `AZURE_API_VERSION` or `2024-10-21` with the existing
native Azure deployment protocol.

The core reads process environment variables and resolves explicit `env:NAME`
references. It deliberately does not load `.env`, user configuration or auth
files during these network calls. Environment variables already supplied to the
service are usable; a missing explicit reference is an error. Endpoints must be
HTTP(S), without URL user information or fragments. Trusted callers may choose
loopback/private endpoints, supporting local providers; this is not a general
untrusted URL-fetch service. Service guard checks and `Cache-Control: no-store`
apply to success and error responses.

The authored tests use only loopback servers: provider paths/headers, model
prefixes, short probe payloads, no fallback/retry, redirect rejection, malformed
and oversized replies, cache isolation/single-flight/stale expiry, and actual
service reference resolution without configuration mutation. Gate execution and
platform claims belong to the coordinator's recorded validation, not this
implementation description.
