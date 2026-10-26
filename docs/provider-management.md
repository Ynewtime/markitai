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
array included, and Kimi Code (`kimi_code`, whose inference uses Kimi's
Messages endpoint) through the same bearer-key `GET <base>/models`; Perplexity, Z.ai and Fireworks AI answer `unavailable` with
`source: "manual"` without a request, since they document no OpenAI-compatible
model list, and their model IDs are typed by hand. The subscription providers
`copilot` and `claude-agent` ask their installed official runtime for its model
list on every request (never cached, because a stored login can change);
`chatgpt` returns the models the installed Codex offers for the signed-in
account (`codex debug models`), its default first. They reject an API key or base URL and report `unavailable`
when the runtime or its login is missing (or Codex cannot list its models). Markitai itself reads no browser or
CLI authentication store; see [subscriptions](subscriptions.md).
Quick-add `/api/settings/llm/detected` lists the default models whose API keys
are set in the conversion environment (process variables plus `.env` files), in
the pool order of [model selection](llm.md#model-selection). It does not contact
providers; its candidates already have an environment credential, so
`requires_api_key` is false.

Each provider uses its model-list protocol, including Anthropic headers, the Gemini
`x-goog-api-key` header (never a `?key=` query), Ollama tags and Azure versioned regional models. Discovery
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
slot while its blocking request is still alive. Errors never contain
credential-bearing URLs.

A discovery that fails answers HTTP 200 with `status: "unavailable"`, no models,
and why, in the connection check's categories: `reason` `no_key` (found before
any request: the key it would send is an unset or empty `env:` reference or key
variable, for a provider that needs one; `key_variable` names the variable),
`key_rejected` (401/403; refused without a key it is `no_key`), `rate_limited`
(429), `network` (no answer: DNS, connection, TLS, or a redirect, which discovery
does not follow), `timeout` (the request limit or the service backstop) or
`provider_error` (404, which for a list means the address has none, any other
status, or an answer without a readable list). A subscription runtime (Copilot,
Claude Code, Codex) says its cause instead: `runtime_missing` (no program on the
server's PATH, or the path variable names no file), `runtime_unsupported` (a
release the adapter does not speak; `runtime.installed` has it when the runtime
could be asked), `not_signed_in` (also when the Claude catalog loads but its
account is not a subscription login), `runtime_failed` (the adapter's own fixed
message is the `detail`) or `timeout`. The answer adds `runtime`
(`{name, required, installed?}`) and, for the first three, `next_step`
(`{kind: "install" | "sign_in", command}`: the pinned `npm install -g` command, or
`markitai auth <provider> login`); nothing the runtime printed is repeated. `detail` reads
`Model discovery returned HTTP 401: <the provider's message>` (the message from
its JSON `error.message`, `message`, `error` or `detail`, else a short text
excerpt without markup, at most 300 characters, with the key that was sent
replaced by "…"), `Model discovery timed out`, `Model discovery request failed:
<cause>`, or `DEEPSEEK_API_KEY is not set on the server, so nothing was sent`.
Discovery no longer answers 503 for these. A `key_rejected` answer also names
the endpoints that refused the key as `endpoints`, in the order tried.

Some providers issue keys per region, and each endpoint refuses another
region's key with HTTP 401: Moonshot AI, Kimi Code, MiniMax, Z.ai and Model
Studio (`dashscope`); their regions are listed under
[LLM](llm.md#keys-that-belong-to-a-region) and on the provider's card as
`regions` (contract v1.25, the default first). A discovery for one of them with
a key but no address (no `api_base`, no base variable) that the default endpoint
refuses tries the other regions in that order; when one lists the models, the
answer carries its address as `api_base` (contract v1.21), which the workbench
fills into the new connection's address so its models are saved with it. That
includes a key the server reads from the provider's variable (`MOONSHOT_API_KEY`,
`KIMI_API_KEY`, `MINIMAX_API_KEY`, `ZAI_API_KEY`, `DASHSCOPE_API_KEY`): its models are added
with `credential_provider_id: "env:<provider>"` and that `api_base`, the one
address other than its own that the settings API lets the server's key go to.
A request that names an address is sent there only.

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

This is the core API contract, not permission for an HTTP client to choose a
server environment variable. The service rejects client-supplied `env:` values
and binds saved/environment credentials to their configured provider and full
endpoint. Custom endpoints use the caller's literal key, without borrowing an
ambient or unrelated saved key; see [settings credential rules](service-settings.md#credentials-and-partial-updates).

The environment-backed calls read the same environment snapshot as conversion:
process variables, then `.env` in the current directory, then
`MARKITAI_HOME/.env` (`~/.markitai/.env`), with existing values taking
precedence. They resolve explicit `env:NAME` references and provider fallback
variables from it, and do not load user configuration or auth files. A missing
explicit reference fails a probe and is `no_key` for discovery. A model written
by hand without a key uses its provider's key variable for discovery and its
credentials view (`env:VAR`, `api_key_source: "environment"`), as conversions
do. The explicit variants read no environment, nor
the server's gcloud login. Discovery sends no key for the keyless placeholder
`markitai-no-key`; a probe sends it as written. Endpoints must be
HTTP(S), without URL user information or fragments. Trusted callers may choose
loopback/private endpoints, supporting local providers; this is not a general
untrusted URL-fetch service. Service guard checks and `Cache-Control: no-store`
apply to success and error responses.
