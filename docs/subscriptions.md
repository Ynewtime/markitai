# Subscription runtimes

Three model prefixes route requests through a subscription instead of an API
key, using the provider's separately installed official command-line runtime:

| Model | Runtime | Status and login |
|---|---|---|
| `copilot/MODEL` | GitHub Copilot CLI `1.0.90-2` | `markitai auth copilot status`, `markitai auth copilot login` |
| `claude-agent/MODEL` | Claude Code CLI `2.1.284` | `markitai auth claude status`, `markitai auth claude login` |
| `chatgpt/gpt-5.5` | Codex CLI `0.159.0` | `markitai auth chatgpt status`, `markitai auth chatgpt login` |

`markitai auth` alone reports all three. Other runtime versions fail explicitly
before a model request. Login is never started during a conversion, and these
routes are available on macOS and Linux only. Ordinary document conversion
needs none of these runtimes. This page separates the native adapter contract
from verification with a real subscription; see [Evidence](#evidence-boundaries-and-next-provider).

## Copilot

The adapter targets the official GitHub Copilot CLI `1.0.90-2`, protocol 3, pinned
against [SDK source f5d9685](https://github.com/github/copilot-sdk/tree/f5d9685f55286061e12763ed15a73c4469609881).
It uses the Node SDK's actual `--headless --stdio` startup and Content-Length JSON
RPC framing. An incompatible runtime fails explicitly before a model request.

```text
markitai auth
markitai auth copilot status --json
markitai auth copilot login
```

Status asks the official runtime for authentication, without creating a model
session. Its public fields are provider/authenticated/user/expires_at/error/details.
JSON status exits zero even when unauthenticated, matching the reference command;
human status returns nonzero. Login replaces the Unix CLI process with the official
`copilot login`, retaining terminal ownership, signals and the exact exit status.
The official runtime owns login, refresh and storage. Markitai reads no token file,
prints no credential and does not rewrite the configured model list.

Configure the existing `llm.model_list[].litellm_params.model` as `copilot/MODEL`.
`COPILOT_CLI_PATH` selects an executable; `COPILOT_HOME` selects its official auth
home when provided; `COPILOT_CACHE_HOME` independently selects its cache directory.
The connect handshake uses the pinned `editorName`/`editorVersion` fields. Explicit token precedence is COPILOT_GITHUB_TOKEN, GH_TOKEN,
GITHUB_TOKEN. Otherwise the official runtime resolves existing authentication.
No HTTP api_key/api_base or strict max_tokens setting is accepted for this route;
the pinned subscription protocol cannot enforce those settings. Missing login is
an error during conversion; it never initiates an interactive login command.

Text and ordered in-memory image attachments use the shared native LLM pipeline:
concurrency permits, document request admission, cancellation, model groups and
semantic validation. Structured output uses JSON text. Session tools, skills,
plugins, MCP, hooks, instruction discovery and persistent session storage are
disabled in the requested configuration. Unexpected permission/tool callbacks
fail. These controls are protocol settings, not an OS sandbox for an arbitrary
replacement executable.

Successful output requires an authoritative terminal idle event. Partial text at
EOF, terminal failure, refusal or truncation cannot become a successful document.
Every observed API call is counted once, including observations before a failure;
duplicate usage events do not multiply the count. Unreported usage stays unknown,
and subscription multipliers are never treated as dollars. A positive dollar
budget refuses before starting inference because no verified subscription dollar
tariff exists. The official runtime may internally make multiple API calls per
one admitted native request; its internal calls cannot be capped individually.

Pools containing Copilot bypass persistent response caching and cross-call request
merging, including when an explicit token exists. An official stored account may
change without changing its home path; a path hash is not account identity. Model
discovery likewise asks the runtime each time. HTTP-only pools retain existing
cache behavior.

Headless process execution has bounded concurrency, frame/event/input sizes and
an overall deadline. Stderr is drained without forwarding; it has no cumulative
byte-limit claim. On Unix, process-group cleanup ends children before temporary
workspaces are removed. Windows execution and login remain unsupported until its
process-tree boundary is implemented and tested.

## Claude

`claude-agent/MODEL` runs through the separately installed official Claude Code
CLI 2.1.284. The protocol reference is official TypeScript SDK 0.3.284
and CLI release `8364969e9f5234ef3d9743cf7c790e9aab0ac3b1`; neither Python nor Node SDK
is required by the Rust adapter. `CLAUDE_CLI_PATH` selects the executable and
`CLAUDE_CONFIG_DIR` selects the official runtime's state. HOME is preserved.
Markitai never parses a token file or implements its own OAuth refresh.

```text
markitai auth claude status --json
markitai auth claude login
```

Status accepts only the official subscription account method. Login replaces the
Unix process with official `claude auth login`, preserving terminal, PID and exit
status. API key/base and OAuth-token environment overrides are removed from the
child; API settings and strict max_tokens are not accepted on this subscription
route. JSON status remains zero-exit when signed out, matching the public command.
The retained `sdk_installed` status field is false because a Python SDK is not
used; `details.native_adapter` identifies the native adapter.

Text, structured JSON text and ordered in-memory images use the shared request
budget, concurrency, cancellation, model groups and content validation. Model
listing initializes the runtime without a user prompt; its catalog has no vision
capability field, so discovery does not advertise vision automatically. A caller
may explicitly configure the supported image path.

The runtime receives a fresh private workspace, safe/restricted mode, empty tools,
no permission prompts, strict empty MCP, empty setting sources, no Chrome/slash
commands/session persistence and a one-turn limit. Four pinned built-in agent
names are permitted as catalog metadata; they grant no Agent tool authority.
Unexpected tools, callbacks, background/subagent events, provider changes and
model changes fail. Managed administrative policy can still run hooks; Markitai
cannot override it. These controls are not an OS sandbox for a replacement binary.

Success requires a valid terminal result and successful process exit, not partial
assistant text. One deadline covers startup and response; process groups are killed
and reaped on Unix. Input, events, stdout and stderr are bounded. Interrupted pipe
reads retry within that original deadline. Stdout EOF waits for the bounded stderr
drain, so a late flood cannot become a successful response. Other platforms remain
unsupported until equivalent process cleanup is implemented.

Because each runtime has its own process group, a terminal interrupt does not
reach it. The Unix CLI therefore kills every runtime, Chromium and LibreOffice
group it started before terminating on SIGINT, SIGTERM or SIGHUP; conversion
keeps its default signal exit, and a batch's second interrupt does the same before
exit 130. `markitai serve` keeps its SIGINT/SIGTERM drain and MCP its SIGINT
drain; their remaining terminating signals clean up. SIGKILL and host processes
embedding the bindings cannot run this cleanup. The core exposes the
async-signal-safe `terminate_child_process_groups` for Rust hosts.

Actual assistant message IDs count observed requests. Terminal usage and per-model
usage overlap with those calls; they are reconciled, never added twice. Per-model
terminal totals include more than main-loop totals. Aggregate-only tokens remain
visible with zero observed requests and `incomplete_request_observations`; that
zero does not mean no requests happened. Conflicting totals retain known evidence
and fail. Shared ledgers, reports/history and the web UI preserve unknown request
counts. Runtime dollar estimates are ignored and subscription price is unknown.
A positive dollar budget refuses before the runtime starts inference.

Pools containing either Claude or Copilot bypass persistent LLM cache and active
request merging because a stable runtime state path does not establish account
identity. Transport failures are not replayed by the adapter. Explicit configured
fallback groups and semantic content retries retain the shared budget and all
observed usage; neither is a guarantee of one provider-internal paid call.

## Evidence boundaries and next provider

Optimized-CLI Copilot process fixtures passed in
[R31](validation/subscription-recovery-round31.md). Round 32 additionally exercised
both exact official macOS arm64 runtimes under OS-denied network, real HOME files
and keychain access. Version, Copilot connection/status and Claude initialization
were checked; the native debug CLI also successfully reports both as signed out.
That official check exposed and corrected Copilot's connect field names and
separate cache-home routing. It does not establish a real login or authenticated
inference. Authored fixtures cover actual native conversion and failure accounting
without sending provider requests. Round 32 optimized CLI and installed dynamic/static bindings pass at `dc0343b`;
see [delivery evidence](validation/integration-round32.md). Completed-item resume
preserves terminal diagnostics independently of the old minimal success aggregate.

The restricted `chatgpt/gpt-5.5` adapter targets official Codex 0.159.0. Its
[contract and evidence boundaries](subscription-chatgpt.md) distinguish actual
official offline request inspection from native fixture validation and real
subscription inference. It imports no private OAuth protocol or invented dollar
tariff. Models outside this validated capability allowlist fail explicitly.
