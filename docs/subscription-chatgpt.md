# Restricted ChatGPT subscription adapter

The `chatgpt/gpt-5.5` route uses a separately installed official Codex CLI 0.159.0.
It supports text, typed document JSON and ordered in-memory image inputs through
the existing native conversion APIs. Other runtime versions and models fail
explicitly. It never silently changes to API-key billing, downloads a runtime,
starts login during conversion or implements a private OAuth endpoint.

```text
markitai auth chatgpt status --json
markitai auth chatgpt login
```

`CODEX_CLI_PATH` selects the executable, otherwise PATH is searched. Official
HOME and optional CODEX_HOME remain unchanged. The runtime owns authentication,
refresh and its credential store; Markitai does not parse or copy token files.
Status accepts only the exact official ChatGPT login result. An API-key login
is unavailable for this route, and its raw status message is never printed.
JSON status retains exit zero when unavailable; human status exits one. Explicit
Unix login replaces the process with official `codex login`, preserving the
terminal, signals and exact exit status without a post-login model request.
Doctor checks runtime version and status, not inference or model entitlement.
Discovery returns the tested adapter allowlist and labels it nonauthoritative.

Set `llm.model_list[].litellm_params.model` to `chatgpt/gpt-5.5`. Do not supply
api_key, api_base or max_tokens; the subscription runtime cannot enforce those
HTTP settings. Its ordered requests share the normal concurrency, document
request admission, cancellation, content validation and configured fallback
rules. Native transport failures do not automatically retry the same deployment.
An explicitly configured fallback is still permitted and preserves earlier
observations. Pools that contain this route bypass persistent response caches
and cross-call merging, including when another HTTP model is first in the pool.
A stable runtime home path does not establish which account is currently logged in.

Each conversion uses a private temporary workspace, custom system prompt,
ephemeral exec, ignored user configuration/rules and a restricted official model
catalog. The selected catalog changes only `apply_patch_tool_type` to null and
omits unvalidated models. Shell, file editing, MCP, applications, search and other
tools are disabled using the pinned official interface. Unexpected tool events
fail the conversion. Success requires a completed turn and successful process
exit; partial output, malformed events or a late nonzero exit cannot become a
successful document. The complete Apache-2.0 catalog license, modification notice
and data provenance are distributed under `licenses/codex/`.

On Unix, the adapter bounds concurrent runtime processes, input, lines, total
stdout, stderr and events, then kills and reaps its process group on failure or
cancellation. One deadline covers version, status and completion. Workspaces and
request files are private. These are resource/lifecycle controls, not an OS
sandbox for a hostile replacement executable. Windows remains unsupported until
equivalent process cleanup is implemented and tested.

## Usage and unavoidable limits

Official exec reports aggregate turn tokens without a reliable API request count.
Input already includes cached input; it is not added twice. The native usage
keeps actual token totals, requests zero and an
`incomplete_request_observations` marker. Zero does not mean no request occurred.
All-zero totals can also mean the official runtime omitted usage. Failed turns
may omit totals entirely; Markitai invents neither counts nor dollars. Aggregate
usage observed before a later protocol/exit error is retained. Reasoning tokens
are included in total output tokens but are not separately exposed by the current
public per-model accounting schema. Price is unknown, and a positive dollar
budget rejects before launching the runtime.

The official CLI can internally retry provider requests/streams. The shared
request budget bounds admitted native attempts, not those hidden internal calls.
A separate no-side-effect login status precedes exec; another process can change
the account between them. There is no atomic account-mode lock. The
`forced_login_method` option is deliberately unused because an incompatible login
can trigger logout and modify the token store. Official system/managed policy
also remains applicable; Markitai does not override mandatory administrative
policy or claim that post-event tool rejection prevents every possible side effect.

## What is actually established

Unmodified official Codex 0.159.0 on an isolated Linux guest sent one request per
local fake-Responses probe. The selected catalog produced `tools: []` with no
hidden `additional_tools`; user/project instruction canaries were absent and a
fake user MCP was not started. A second probe used the exact compiled catalog,
custom system instructions and two authored PNGs, preserving system text and
ordered image bytes. This establishes an offline protocol boundary, not a real
ChatGPT login, model availability or subscription inference. JPEG/WebP paths
have not received equivalent actual-runtime request inspection.

Authored fake subprocess tests cover the native status/login/doctor, conversion,
budget, fallback, cache bypass and terminal-accounting paths; the optimized-CLI
workflow with an authored fake runtime passed 52 checks in
[delivery R36](validation/delivery-round36.md). An actual native-to-official
signed-out status check also passed in R36: the Linux single binary ran the
unmodified official Codex 0.159.0 as a dedicated guest user with loopback-only
networking and no credentials, reporting `authenticated: false` with JSON exit
0 and text exit 1 and creating no auth file
([Linux R36](validation/platform-linux-round36/summary.md)). This establishes
the signed-out interaction only, not login, entitlement or inference.
