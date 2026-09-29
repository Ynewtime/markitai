# Native MCP service

`markitai mcp` exposes four conversion tools over stdin/stdout. The executable
calls Rust directly; Python and a second conversion process are not required.
The maintained Rust MCP SDK handles framing, protocol negotiation and request
dispatch. Both the initialization-based 2025-11-25 protocol and the
2026-07-28 discovery/per-request metadata protocol are supported by this SDK;
the service advertises tools, without adding an HTTP MCP endpoint.

```sh
markitai --config /absolute/path/config.json mcp
```

Configure an MCP host to launch that command. Keep provider credentials in the
host's environment or the isolated Markitai configuration. Normal tool results
are MCP messages on stdout; diagnostics belong on stderr. The existing public
`markitai-mcp` launcher name selects the same service, including a no-argument
client launch. Unix distributions expose it as a relative link to the one CLI
executable. `markitai-mcp --config /absolute/path/config.json` accepts the same
global configuration options. A direct Cargo build can use `markitai mcp` or
create that link; it does not compile a third executable.

## Tools and results

`convert_document` requires an absolute local `path`; `~` is expanded. A modern
`.numbers` directory package is one document, with a case-insensitive extension;
ordinary directories are rejected. The same package paths can be individual
`batch_convert` sources. This does not add recursive directory traversal or a
directory-upload protocol. Legacy XML Numbers packages, Numbers OCR and full
worksheet screenshots remain unsupported; [the reader contract](numbers.md)
describes the table-only scope.

`convert_url` requires an `http://` or `https://` URL. Both single-source tools accept
`output_dir`, `llm`, `ocr`, `screenshot`, `alt`, `desc` and `profile`. Feature flags
accept true, false or null: omitted/null values follow configuration. Profiles
are `rag`, `obsidian` or `okf`; null follows configuration.

An explicit output directory must be absolute. Omitting it, passing null, or
passing an empty string creates a temporary directory named `markitai-mcp-*`.
It remains on disk after the call and after server shutdown so clients can read
the returned files. The caller is responsible for later cleanup.

Each single-source result contains:

| Field | Meaning |
|---|---|
| `source` | The conversion's source identifier |
| `markdown` | Enhanced Markdown when available, otherwise base Markdown |
| `truncated` | Whether the inline body was shortened |
| `markdown_file` | Complete enhanced/base output path, or null |
| `output_dir` | The selected output directory |
| `assets`, `screenshots` | Written paths from the core result |
| `cost_usd` | Core usage cost; currently zero without native pricing |
| `skip_reason` | Core skip reason, or null |
| `duration_s` | Core duration rounded to two decimal places |
| `warnings` | Nonfatal conversion notices |
| `diagnostics` | Optional recorded work for the terminating conversion attempt |

Bodies longer than 40,000 Unicode characters are reduced to a 2,000-character
preview. UTF-8 byte counts do not control truncation. The full output remains on
disk. Successful tool responses carry the same object in `structuredContent`
and in a text content block containing JSON; tools also publish output schemas.

With `output.on_conflict=skip`, the tool reads the existing Markdown body and
returns its path. An existing enhanced file takes precedence even when the call
sets `llm=false`; no model request is needed. YAML frontmatter is excluded from
the inline body. Existing files must satisfy the output symlink policy and the
native 500 MiB per-file read limit.

Expected errors are tool results with `isError: true`, including meaningful
validation, fetch or conversion diagnostics. Missing models add guidance about
`MODEL`, provider keys and the MCP host's `mcpServers` environment. An image
without enabled content extraction is an error, matching the public API.
Unsupported OCR, screenshot or other core features remain explicit errors;
exposing a tool does not implement every conversion backend.

When the core records model work, successful results additionally contain
`diagnostics.last_attempt`, with `operation: "convert"`, `status: "done"`,
`error: null` and `usage`. Usage contains `requests`, `input_tokens`,
`output_tokens`, `cost_usd` and `by_model`; it uses the same accounting as the
core detailed conversion API. A recorded request with zero tokens is still an
observation. Results without an observation retain the original eleven fields.

Failed tools retain their existing `isError: true` and text beginning
`Error executing tool <name>: `. If recorded work exists, `structuredContent`
also contains `{error, diagnostics}`, where `error` is the original message and
the attempt has `status: "error"` with the same error string. These remain tool
errors, not JSON-RPC protocol errors. Output schemas permit both success and
structured error values. Input validation, missing models, unstarted work and
panic/worker failures do not invent usage. The missing-model configuration hint
is unchanged.

These are per-attempt observations, not a lifetime billing ledger. A failed
conversion can have consumed tokens; missing diagnostics mean unknown or no
observation, not proof of a free call. Provider pricing is not implemented, so
zero `cost_usd` does not establish zero cost. Raw credentials and provider error
bodies are not added to diagnostics.

## Background batches

`batch_convert` requires a nonempty `sources` array of absolute local paths and/or
HTTP(S) URLs, with the same optional conversion flags. It accepts `concurrency`:
null/omitted means 10, and values at or below zero mean one. As JSON Schema
`integer` allows, an integral number such as `2.0` is accepted; `2.5` is not. Relative sources are
rejected before a job is created. Missing files and other conversion failures
are recorded per item and do not stop the remaining items.

The immediate result is `{job_id, status: "running", total, output_dir}`. Each
item writes under `output_dir/batch-<job_id>/0001/`, `0002/`, and so on. Duplicate
filenames and duplicate sources therefore have distinct output directories.
Each batch shares an [LLM runtime](llm.md); the conversion concurrency and the
LLM request cap are separate limits.

`job_status` takes `job_id` and returns `job_id`, `status`, `total`, `done`,
`failed`, `output_dir` and `results`. `done` counts both successes and failures.
The results array contains finished items in input order, omitting unfinished
slots while running. Once completed, each result corresponds to the same input
index. A completed job may contain failures, including failure of every item.

- Success items contain `source`, `status: "ok"`, `markdown_file`, `cost_usd`
  and `warnings`.
- Failure items contain `source`, `status: "error"` and `error`.

Either slot adds the same optional `diagnostics.last_attempt` when that
conversion recorded work. A success reused through the shared runtime does not
inherit the HTTP owner's usage; only the caller that actually sent the request
is charged. A failed owner and a waiter that subsequently sends its own request
keep separate observations. No new aggregate is added to `job_status`.

This task table is independent of REST history. Jobs exist only in this server
process, with 100 finished jobs retained and up to 500 forgotten IDs remembered
for a distinct expiration diagnostic. Like the reference implementation, the
eviction scan uses creation order among finished jobs. Running jobs are not
evicted. Output files survive eviction and restart.

Native admission bounds are 100 concurrently running jobs and 10,000 sources
per job; exceeding either returns a tool error. These limits are deliberate
additions to the reference's unbounded running-job admission. Protocol argument
types follow the published JSON schemas; Python/Pydantic's incidental coercion
of values outside those schemas is not a compatibility guarantee.

## Configuration and shutdown

Every conversion reloads configuration, with command-line `--config` and
`--config-json` retained as configuration sources. Explicit tool flags apply on
top. Model credential resolution uses the core's environment precedence:
existing environment, the working-directory `.env`, then the isolated
`MARKITAI_HOME/.env`. This reads dotenv as data without mutating the host
environment. Set `MARKITAI_HOME` when running isolated instances.

EOF or SIGINT stops new admissions and drains already started blocking
conversions. Batch items that have not started are left unfinished and the job
is marked cancelled rather than completed. The service offers no public
cancel-job tool. Cancelling an individual protocol request cannot forcibly stop
Rust extraction or an HTTP request already executing in a blocking worker;
written output may still appear. Process termination is not an exactly-once
publication guarantee, and MCP does not enable CLI resume/state reports.
Draining protects already admitted work; it cannot deliver a single-call result
to a disconnected client or persist the in-memory job table. Hard termination
and unreadable provider responses remain gaps in complete accounting.

## Verification scope

Authored process tests launch the real CLI with private MARKITAI_HOME,
configuration, working directory and temporary output while preserving HOME.
The child environment excludes provider credentials; model fixtures use only
loopback endpoints. They exercise both
protocol eras, schemas, errors followed by continued requests, single-source
outputs, Unicode previews, configuration reload, duplicate batch names, failure
slots and polling while a slower URL is still in flight. Additional terminal
usage cases cover both protocol eras, paid HTTP failures including known zero
tokens, early errors without observations, ordered mixed-result batches,
shared-request owner accounting, failed-owner recovery and EOF draining.
Module tests cover
finished-job retention and forgotten-ID bounds. Executed gate results belong in
the project control record; this document does not claim a completed reference
wire differential or live-provider validation.

A 2026-07-28 `tools/list` result carries the required `ttlMs: 0` and
`cacheScope: "private"`, the reference SDK's defaults; 2025-11-25 listings keep
their original shape. The actual Python SDK 2.2.0 differential and its recorded
differences are summarized in the project validation records.

Protocol references: [MCP 2025-11-25 lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle),
[MCP 2026-07-28 versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning),
[official Rust SDK](https://github.com/modelcontextprotocol/rust-sdk).
