# Subscription runtimes

Three model prefixes route requests through a subscription instead of an API
key, using the provider's separately installed official command-line runtime:

| Model | Runtime | Status and login |
|---|---|---|
| `copilot/MODEL` | GitHub Copilot CLI `1.0.90-2` | `markitai auth copilot status`, `markitai auth copilot login` |
| `claude-agent/MODEL` | Claude Code CLI `2.1.284` or a later 2.1 patch release | `markitai auth claude status`, `markitai auth claude login` |
| `chatgpt/gpt-5.5` | Codex CLI `0.159.0` | `markitai auth chatgpt status`, `markitai auth chatgpt login` |

`markitai auth` alone reports all three. Other runtime versions fail explicitly
before a model request. Login is never started during a conversion, and these
adapters support Unix and Windows process management. A compatible official
runtime and its own platform prerequisites are still required; fixture coverage
does not establish authenticated inference. Ordinary document conversion
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
Windows cannot replace a process: the CLI starts the official login on its own
console, ignores Ctrl-C and Ctrl-Break while the login runs (the runtime receives
them itself), waits, and exits with the runtime's exit status.
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
an overall deadline. Stderr is drained without forwarding; more than 1 MiB fails
the request. Process-tree cleanup ends children before temporary workspaces
are removed (see [Runtime processes](#runtime-processes)).

## Runtime processes

`COPILOT_CLI_PATH`, `CLAUDE_CLI_PATH` and `CODEX_CLI_PATH` name a runtime
directly; otherwise the first `copilot`, `claude` or `codex` on PATH is used.
Only absolute PATH entries are searched, and on Unix only files with execute
permission count. On Windows the search tries the PATHEXT extensions in order,
limited to `.com`, `.exe`, `.bat` and `.cmd`, so npm's `copilot.cmd`,
`claude.cmd` and `codex.cmd` shims are found and its extension-less shell
scripts are not; a configured path without an extension is completed the same
way. Variable names are matched without regard to case there (`Path` serves as
PATH). A `.cmd` or `.bat` shim runs through the command processor
(`cmd.exe /d /c`) with the standard library's argument quoting for batch files,
which refuses arguments it cannot pass safely; the native `copilot.exe`,
`claude.exe` and `codex.exe` avoid that extra process. Status reports a resolved
path without the `\\?\` prefix.

Each runtime starts as the root of its own process tree: a new process group on
Unix; on Windows a new process group and a hidden console, suspended until it
has been placed in a Job Object that kills the tree when closed. A timeout,
cancellation or failure kills the whole tree, and on Windows waits until it is
empty before the private workspace is removed. Besides HOME, PATH, temporary
directories, locale and each runtime's home, Windows runtimes receive the
standard system variables that command shims and Node.js need (PATHEXT,
ComSpec, SystemRoot, the profile and program folders, and similar); none holds
a credential. On Windows Claude Code also receives `CLAUDE_CODE_GIT_BASH_PATH`
when it is set.

Each runtime runs at most eight processes at once, counted separately for
Copilot, Claude and Codex; a further request waits for a slot within its own
deadline. Stderr is never forwarded, and more than 1 MiB of it fails the request.

## Claude

`claude-agent/MODEL` runs through the separately installed official Claude Code
CLI 2.1.284 or a later 2.1 patch release (the runtime updates itself; patch releases
keep the stream-json protocol of their line, and every message is still validated).
An older patch or a new minor or major line fails before any request;
`markitai auth claude status --json` reports the installed `cli_version` and the
`supported_from` release. The protocol reference is official TypeScript SDK 0.3.284
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
status; on Windows it runs as a waited-for child, as for Copilot. API key/base and OAuth-token environment overrides are removed from the
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
assistant text. One deadline covers startup and response; the runtime's process
tree is killed and its root reaped. Input, events, stdout and stderr are bounded.
Interrupted pipe reads retry within that original deadline. Stdout EOF waits for
the bounded stderr drain, so a late flood cannot become a successful response.
Platforms other than Unix and Windows refuse to start a runtime.

Because each runtime has its own process tree, a terminal interrupt does not
reach it. The Unix CLI therefore kills every runtime, Chromium and LibreOffice
group it started before terminating on SIGINT, SIGTERM or SIGHUP; conversion
keeps its default signal exit, and a batch's second interrupt does the same before
exit 130. `markitai serve` keeps its SIGINT/SIGTERM drain and MCP its SIGINT
drain; their remaining terminating signals clean up. SIGKILL and host processes
embedding the bindings cannot run this cleanup. The core exposes
`terminate_child_process_groups` (async-signal-safe on Unix) for Rust hosts.

On Windows, Ctrl-C and Ctrl-Break are the interrupt: they kill the runtime trees
and exit with 130 outside a controlled batch. A batch with recovery state stops
admission and drains active work on the first interrupt; a second kills its trees
and exits immediately. Windows batches use the same native recovery protocol
as Unix. Closing the console window, logging off or shutting down kills the
trees and exits with 143 at once, because Windows allows only a few seconds; the
batch state already written stays valid. `markitai serve` and MCP keep their
Ctrl-C drain; Ctrl-Break and the closing events clean up. A Ctrl-C that the
parent disabled (as a new process group starts) stays disabled, as an inherited
ignored SIGINT does on Unix. A runtime's job also ends its tree when Markitai is
terminated without any cleanup.

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

A runtime that is signed out, reports a non-subscription account or reports an
authentication failure is an authentication refusal. When its model group has
another deployment, that request moves there at once and the runtime deployment
is skipped for the rest of the run with one warning; see
[retries](llm.md#retries-budgets-and-usage). Policy violations such as an
unexpected tool or callback remain fatal.

## Evidence boundaries and next provider

Native fixture tests exercise all three runtime protocols, cleanup and failure
accounting on supported platforms, including Windows command shims. Separate
official-runtime checks establish only their recorded version, initialization
or signed-out behavior. They do not establish a real login, model entitlement,
authenticated inference or billing.

The restricted `chatgpt/gpt-5.5` adapter targets official Codex 0.159.0; its
[contract](subscription-chatgpt.md) describes its model allowlist and aggregate
usage limits. No adapter imports a private OAuth protocol or invents dollar
prices for a subscription.
