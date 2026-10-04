# Native REST service

`markitai serve --host 127.0.0.1 --port 3600 --no-open` runs an HTTP API inside
the Rust binary. Conversion calls the same Rust core as the native bindings.
The root URL serves an embedded browser workbench for conversion, preview,
history and model settings; no Node runtime or CDN is required. Without
`--no-open`, the service asks the system URL handler to open it.
See [browser workbench](web-ui.md), [settings](service-settings.md) and
[provider discovery and probes](provider-management.md) for their contracts.

The configuration destination is fixed at startup: explicit `--config`, then
`MARKITAI_CONFIG`, project `markitai.json`, user `config.json`, or a new file
under the isolated Markitai home. Service reads reject nonregular files,
symlinks and files larger than 8 MiB. Settings saves change the configuration
snapshot for newly admitted jobs and retries; an active job keeps its original
snapshot. CLI model/provider session overrides remain effective and make
settings writes unavailable until restart without those overrides.

## Machine-readable API description

`GET /api/openapi.json` returns an OpenAPI 3.1 document of every `/api/` route
(jobs, history, downloads, settings, provider discovery and tests) with its
parameters, request bodies, response schemas, the error body and its `reason`
codes, and the three ways to authenticate (`bearerAuth`, `tokenQuery`,
`ticketQuery`). It is the same document a code generator or API explorer can
read; its `info.version` is the running build's version. It needs the same
authentication as the rest of `/api/`, including loopback clients. The document is
written by hand (`crates/markitai-cli/src/server/openapi.json`); module tests fail
when a route is served but not described, or described but not served, when a
path parameter is undeclared, a reference dangles, or the service writes a
`reason` the error schema does not list. Response schemas describe the fields the
workbench relies on; they are not a byte-for-byte contract of every payload.

## Job workflow

1. `GET /api/capabilities` returns version, effective LLM configuration, presets,
   SVG/browser availability, Cloudflare local readiness, and the 1,000-item limit.
2. `POST /api/jobs` accepts form fields `urls` (a JSON array) and `options` (a JSON
   object), plus repeated multipart `files`. URL-only requests also accept
   `application/x-www-form-urlencoded`. A successful submission returns
   HTTP 201 with `job_id` and ordered `{item_id,name,kind}` entries. File entries
   precede URL entries. Upload names are sanitized and made unique, including
   case-folded collisions. Base and enhanced Markdown names are reserved as a
   pair, so an upload named `notes.llm` cannot overwrite another item's enhanced
   result.
3. `GET /api/jobs/{job_id}` returns the current snapshot. Items move from
   `queued` to `running` to `done` or `error`; failed conversions remain in the
   job with their error and nullable output. A job containing failed items
   still completes with `status: "done"` and separate success/failure counts.
4. `GET /api/jobs/{job_id}/events` streams a first `snapshot`, subsequent `item`
   and `job` events, and idle keepalives. A terminal job event closes the stream.
   Slow subscribers receive a fresh snapshot after a bounded event-buffer
   overrun.
5. `GET /api/jobs/{job_id}/items/{item_id}/result` returns `name`, selected
   `variant`, `markdown`, and downloadable `{relpath,size}` artifacts. Fetch an
   artifact through `GET /api/jobs/{job_id}/files/{relpath}`. Download all output
   files through `GET /api/jobs/{job_id}/archive` after completion.
6. `POST /api/jobs/{job_id}/cancel` stops the job's original items that are still
   waiting for a conversion slot. It returns HTTP 202 with `job_id` and `stopping`,
   the number of such items when the request arrived. Each becomes an `error` item
   with `cancelled (stopped by request)` and `error_code: "cancelled"`, publishes
   an item event and stays retryable; items already converting finish, and the
   job completes normally.
   Queued retries and enhancements are not affected. A job that is not running,
   or has no waiting original item, returns 409; the request needs no body.

The `options` object supports the existing tri-state preset, LLM, OCR, profile,
image description/alt text, screenshot, pure, cache, compression, strategy, and
backend fields. Unsupported conversion features remain explicit core errors.
Requested LLM processing uses core environment/model resolution, including
`MODEL`; the server does not silently disable LLM because `model_list` is empty.
Unlike the reference service's no-model fallback, which converts without the
model and reports success, a request that itself asks for model processing is
refused when no model is routable. That means `llm: true`, or a preset whose
definition turns the LLM on (`standard`, `rich`, or a configured one) unless
`llm: false` overrides it. `POST /api/jobs` answers 422 with
`reason: "llm_unavailable"` before anything is stored, instead of creating one
identical failing item per input; a retry whose supplied `options` ask for model
processing is refused the same way and leaves the item unchanged, while
`enhance` keeps its 409 `llm_unavailable`. `alt` and `desc` alone are not a
request (they only act while the LLM is on). Model processing enabled only by
the server configuration is not a request either: it keeps following the core's
configured failure policy, item by item.
`GET /api/capabilities` reports `llm.routable`, which the
[browser workbench](web-ui.md#layout-and-interaction) uses to switch
those choices off. Remote consent cannot request interactive terminal input.
When LLM is enabled, base output is retained.

## Cloudflare request authorization

`GET /api/capabilities` includes `remote_services.cloudflare` with `configured`,
`available`, `reason`, `browser_rendering`, `file_conversion` and
`file_extensions`. These describe local prerequisites only. The check makes no
network request and returns no credentials, account ID or endpoint. It does not
verify cloud permissions, quota or service availability. `reason` is null when
ready, otherwise `not_configured`, `invalid_configuration`, `disabled_by_policy`
or `client_not_trusted`. An untrusted caller gets a fixed unavailable response
without resolving credentials.

A trusted request must select `strategy: "cloudflare"` or
`backend: "cloudflare"` and explicitly include `remote_processing: "cloudflare"`
in its `options`. For example, the options form field for a URL job can be:

```json
{"strategy":"cloudflare","remote_processing":"cloudflare","llm":false}
```

The field authorizes this request only. Server Cloudflare defaults do not select
the service implicitly, and saved selections never carry permission into a new
request. The server uses its own Cloudflare configuration; options cannot supply
Cloudflare credentials, environment references or endpoint overrides. The same
checks apply to creation, retry and enhancement, including inherited Cloudflare
selections. A client repeating an item must show its saved selections, obtain
fresh confirmation, then send those options with `remote_processing` added for
that request. A confirmation for Cloudflare cannot authorize Jina or Defuddle;
combining either URL strategy with the Cloudflare file backend returns 422
`invalid_options`.

`fetch.remote_consent=never` and `MARKITAI_NO_REMOTE_FETCH` remain hard stops.
Accepted requests use `ask` with the explicitly selected Cloudflare service;
there is no terminal prompt and no implicit authorization of remote fallback in
`auto`. The server returns 403 `remote_processing_forbidden` for an untrusted
caller, 422 `remote_processing_confirmation_required` without current consent,
422 `remote_processing_disabled` when policy forbids it, and 422
`cloudflare_unavailable` when local prerequisites are missing. Unknown enum
values, or consent without a Cloudflare selection, return 422 `invalid_options`.

## Error responses

Every API error body has the same three fields:

```json
{"detail": "file exceeds upload limit", "code": "payload_too_large", "reason": "file_too_large"}
```

`detail` is the service's English message (an object for the settings conflicts
that already carry `detail.code`), and `code` is the category derived from the
HTTP status (`bad_request`, `unauthorized`, `forbidden`, `not_found`,
`method_not_allowed`, `conflict`, `payload_too_large`, `invalid_request`,
`rate_limited`, `unavailable`, `server_error`). `reason` is an additive, stable
machine code for the specific cause, for example `job_not_found`,
`file_too_large`, `request_too_large`, `empty_job`, `unsupported_url_scheme`,
`remote_url_forbidden`, `invalid_options`, `job_running`, `nothing_to_stop`,
`llm_unavailable`, `upload_missing`, `history_empty`, `token_required`,
`host_not_allowed`, `provider_busy`, `stale_revision`, `config_changed` or
`settings_read_only`. Clients that read only `detail` and `code` are unaffected.
Clients should present text by `reason`, then `code`, and keep `detail` for
diagnosis; the [browser workbench](web-ui.md#language-appearance-and-stored-values) does so in
English and Chinese.

`detail` for malformed input is always the service's own sentence. A body that is
not multipart form data (and not a form-encoded URL list) is 400
`invalid_multipart` and names the accepted forms; the multipart reader's wording
about boundaries is never passed on. A POST with neither a content type nor a body
is an empty job (422 `empty_job`). A bad `options` field is 422 `invalid_options`
and names the option: `unknown option 'x'; supported options: …`,
`option 'llm' must be true or false`, `option 'profile' must be one of: rag,
obsidian, okf`, or `options must be a JSON object`. A retry body is checked the
same way (`invalid_retry_body` for its own fields). JSON parser positions such as
"at line 1 column 8" never appear.

A failed item adds `error_code` beside its `error` string. It is the core's
conversion error code, the same vocabulary as the bindings' error envelope
(`unsupported`, `fetch_error`, `no_model_configured`, `conversion_error`,
`invalid_input`, `config_error`, `io_error`, `not_found`, `is_directory`,
`invalid_json`), or a service cause: `cancelled` (stop request), `shutdown`,
`internal_error`, and for a failed rerun the `reason` of its publication error
(for example `output_conflict`, `enhancement_failed` or `no_output`). Skipped
items keep `skip_reason` instead and have no `error_code`; a successful or queued
attempt clears it. The field is omitted when absent, so items saved before it
existed, and CLI-recorded history items, simply have none. The code names a
category; the precise cause, such as an HTTP status or a timeout, remains in
`error`.

## Output cost and pricing coverage

Items report model cost in two fields. `cost_usd` is the subtotal of requests
priced from the bundled catalog ([token prices](pricing.md)), not an invoice. It
is null until a conversion succeeds; a conversion without model requests reports
`0` and omits `pricing`. `pricing` states how much of the subtotal is established:

```json
{"priced_requests": 1, "unpriced_requests": 1, "cost_status": "partial",
 "pricing_snapshots": ["litellm-1.100.1-selected-2026-09-29"]}
```

`cost_status` is `complete` when every recorded request was priced, `partial`
when some were not, and `unknown` when none was. A subscription turn that reports
tokens without a request count adds `incomplete_request_observations` and keeps
the status from `complete`. A zero `cost_usd` therefore establishes zero cost for
recorded model requests only together with `cost_status: "complete"`; it does
not cover external-service charges. The fields appear in job snapshots, item
SSE events and saved metadata. History summaries add `cost_usd` over the retained
outputs and an aggregated `pricing`, which is omitted when coverage cannot be
established for every retained output (for example an older entry with a numeric
cost but no request counters). A failed attempt's usage is reported separately
under `diagnostics`, described below.

Items whose accepted attempt requested Cloudflare also carry a separate optional
`remote_processing` object in snapshots, SSE events and saved metadata:

```json
{"provider":"cloudflare","requested":true,"execution":"unknown",
 "external_charges":"not_included",
 "notice":"Cloudflare requested; external charges not included"}
```

This is request scope, not execution or billing telemetry. Actual cloud calls,
request counts and charges remain unknown; clients must not infer them from the
selection or the document's metadata. The marker persists across failed reruns
that retain a previous output and later native attempts. History summaries
include it when any item has it; older histories without the field make no claim.
The LLM ledger above stays separate and excludes Cloudflare charges, even when
its known subtotal is zero. Clients can localize the stable fields rather than
displaying the English `notice` verbatim.

## Retry, enhancement and item deletion

`POST /api/jobs/{job_id}/items/{item_id}/retry` returns HTTP 202 with the original
job ID and the one queued `{item_id,name,kind}` entry. Its optional JSON body is:

```json
{"operation":"retry","options":{"llm":false}}
```

`operation` accepts `retry` (default) or `enhance`. An empty body, JSON null, or
omitted/null `options` inherits the item's last conversion options, falling back
to the job's options for older histories. A supplied options object replaces those
options; omitted fields then use the server configuration rather than the item's
previous overrides. Unknown fields and invalid options return 422.

Each full item in a job snapshot or item SSE event includes an `options` object
containing that item's saved repeat selections, with a job-level fallback for
older history. Only supported option keys are exposed; `remote_processing` is
omitted. This object is not authorization. Clients supporting older servers may
use `item.options ?? job.options`, remove any consent field and obtain fresh
[Cloudflare confirmation](#cloudflare-request-authorization) before a rerun.
The small queued-item response returned by creation or retry is unchanged.

Both operations reconvert the original retained upload or refetch the original
URL. Enhancement is an explicit one-off attempt: effective options must enable a
routable LLM, and success must contain an actual enhanced Markdown output. It does
not silently enable LLM, and a fallback base conversion is not reported as a
successful enhancement. Enhancement does not replace the saved options for future
ordinary retries. Ordinary retry updates both per-item options and the job's latest
options; sibling items retain their own options.

Only `done` and `error` items can be queued. A duplicate pending retry returns 409;
a missing original upload returns 404. CLI-recorded file entries without retained
uploads remain nonretryable (409). Original URLs pass the same trusted-peer and
HTTP/HTTPS checks used for new jobs. Retrying a terminal item while an initial
sibling still runs is supported. Retries run in admission order within their job,
share the job's LLM runtime, and use the service's file/URL concurrency limits.
Only the final active conversion or retry can publish terminal job completion.

A retry writes to a private temporary output directory. Publication validates the
item's reserved Markdown pair and preserves any asset another item claims. A
failed rerun of a previous successful, non-skipped item restores its previous
public result, including operation, warnings, cost and timing. Its actual Markdown
and assets remain unchanged. Success reuses the same output names; a successful
plain retry removes a stale enhanced variant. Old unreferenced extracted assets
may remain in the job archive until item/job cleanup; they are not fabricated into
the new result's artifact list.

If a failed retry or enhancement keeps the previous successful result, the item
also exposes optional `rerun_failure`: `operation` (`retry` or `enhance`),
`error_code`, `error` and RFC3339 `failed_at`. Its status, output, original finish
time and price still describe the retained result; this field describes the
latest failed operation. It is available even when no model usage was recorded.
The workbench shows the failure in its shared notification card and keeps preview,
download and retry available. Clicking the item's status icon reopens the full
details. Warnings belonging to the retained output are grouped as previous-result
warnings; the latest attempt's cost is shown only when usage was actually recorded.
New admission or successful completion clears the operation failure.
SSE events, snapshots and saved metadata carry the same outcome; normal restart
preserves it. Old history has no field. Invalid stored outcomes are ignored with
a fixed warning while the retained row remains available, without rewriting the
history. Failed metadata persistence or abrupt termination can still prevent an
outcome from being saved. This field does not establish request counts or costs.

### Recorded attempt usage

An item with recorded model work additionally exposes `diagnostics.last_attempt`:

```json
{
  "operation": "retry",
  "status": "error",
  "error": "the existing conversion or publication error",
  "usage": {
    "cost_usd": 0.0,
    "requests": 1,
    "input_tokens": 7,
    "output_tokens": 5,
    "by_model": {
      "example": {"requests": 1, "input_tokens": 7, "output_tokens": 5, "cost_usd": 0.0,
                  "priced_requests": 0, "unpriced_requests": 1, "cost_status": "unknown"}
    }
  }
}
```

`operation` is `convert`, `retry` or `enhance`; `status` is `done` or `error`.
A successful attempt has a null error. Existing item errors remain strings or
null. Diagnostics are omitted when no usage was recorded, including an unknown
provider response; one recorded request with zero tokens still produces them.
Missing diagnostics do not establish a free call. Each `by_model` row carries the
same coverage counters as [token prices](pricing.md) describes
(`priced_requests`, `unpriced_requests`, `cost_status` and, once a request is
priced, `pricing_snapshot`), so a zero `cost_usd` means zero cost only when that
row's `cost_status` is `complete`.

A newly queued attempt clears previous diagnostics. After it finishes, only its
own observation is published; this is not a cumulative retry ledger. If a retry
restores the previous successful output, its new diagnostics still describe the
attempt that failed. The retained item's original output, status, cost and timing
keep their established meaning. Usage recorded by a successful core conversion
is also retained when the subsequent service asset checks or file publication
fail. An unknown later failure never resurrects old usage.

The same optional field is included in item SSE events, job snapshots and saved
metadata. It survives normal restart, including observations imported from CLI
history. Older metadata without it remains unchanged. Invalid stored diagnostics
are ignored with a fixed warning while the existing output row remains available;
reading that history does not rewrite its bytes. An abrupt process termination,
panic without a returned observation, or failed metadata persistence can still
leave accounting unavailable. This field does not provide a durable billing ledger.

`DELETE /api/jobs/{job_id}/items/{item_id}` returns 204 for a terminal job. It removes
that ledger row, its retained upload, Markdown pair and owned assets/screenshots,
while keeping files claimed by another item. Image metadata rows for assets actually
removed are pruned in the same recoverable transaction; shared rows remain. The last item's deletion removes the
whole job. Deleting while any job work runs returns 409; repeated deletion returns
404. Native ownership indexes are preferred; legacy histories use exact Markdown
references and bounded converter filename suffixes. Arbitrary filename substrings
do not grant ownership.

## History and persistence

Only one `serve` process may use a given `MARKITAI_HOME` at a time. A private
OS lock is held through shutdown and worker draining; a second process fails
before recovery or upload cleanup. Use distinct state directories for independent
servers. Stop all old-version servers before upgrading: older releases do not
participate in this lifetime-lock protocol.

At startup, a persisted running job becomes visible terminal history. Its queued
or running items receive `error_code: "interrupted"` and can be retried or deleted;
recorded successful and failed items retain their results. Uploaded originals and
all existing output bytes are kept. Unrecorded files do not establish a successful
conversion or recover missing usage/cost observations. Jobs are never rerun
automatically.

Startup also removes service-owned upload stages older than 24 hours, while
holding the lifetime lock. Only stages with the current ownership marker and
safe, owned regular contents qualify. Recent stages, unknown or unsafe entries,
and old unmarked `.upload-*` directories are retained; their names alone are not
proof of ownership. Normal publication removes the staging marker.

Jobs live below `MARKITAI_HOME/serve/jobs/<12-hex-id>/`, containing private
`uploads/`, `out/`, and an atomically replaced `meta.json`. Job, upload and output root directories use mode
0700 and generated metadata/uploads use mode 0600 on Unix. Current server
metadata uses the existing version-2 shape, with additive native output-base and
asset indexes, per-item options, and internal file-transaction commit identifiers. Existing CLI and reference terminal histories are readable. Legacy artifact lists
recover assets referenced in Markdown, including visible `assets/` profiles;
assets without references and without a native item index are still downloadable
by path and included in the ZIP, but are not guessed into an item artifact list.
`GET /api/history` refreshes jobs added by other local processes and returns the
existing summaries in reverse creation order. `GET /api/history/archive` streams
a ZIP of completed histories. `DELETE /api/history/{job_id}` removes a terminal
job and its files; running jobs return 409. This implementation does not apply
an automatic age-based deletion policy.

Metadata is flushed and atomically published before reporting durable terminal
completion. If inventory or final metadata publication fails, the live snapshot
and terminal event use `status: "error"` with `persistence_error`. This is an
explicit extension to the ordinary reference job schema. Already completed
item outputs remain downloadable during that process, the job is excluded from
saved-history listings, and shutdown returns failure. Incomplete jobs are not
automatically rerun after restart.

Rerun and item-deletion publication has a durable undo journal beneath the private
job directory. Previous regular files are streamed into bounded backups; a prepared
journal is synced before replacements or removals. The corresponding transaction
identifier in atomically published metadata commits those bytes. On restart,
uncommitted transactions restore the previous files in reverse publication order;
committed transactions keep the new files and discard their backup. This protects
the last persisted result across publication-before-metadata interruption. Journals
reject traversal/symlinks and bound backups at 5 GiB, 100,000 members and 16 MiB of
metadata. A failed rollback is reported as a persistence error; new retry admission
is rejected until restart recovery, rather than claiming the old result is intact.
These are local private-state integrity measures, not authentication
against a local actor able to forge both metadata and recovery files. A full process
or machine failure during filesystem recovery can require another recovery attempt.

SIGINT and SIGTERM stop admission and queued conversion dispatch, close event
streams, and drain active blocking conversions before writing final metadata and
exiting. Active native work is not forcibly interrupted. Pending items record a
shutdown cancellation error. Queued reruns instead restore a prior successful result
when one exists; active retries drain before the service exits. A second forceful process termination can still
leave an incomplete job.

## Network and file boundaries

Every API client requires the startup token through `Authorization: Bearer …`
or `?token=`, including clients on this computer and same-host reverse proxies.
`MARKITAI_SERVE_TOKEN` can supply that token. `--no-auth` explicitly disables
this requirement. In that mode only a direct canonical loopback peer without
`Forwarded`, `X-Forwarded-For` or `X-Real-IP` headers receives URL/settings trust;
other clients cannot submit URL conversions: a full DNS/redirect-aware public-network
fetch policy has not been implemented. Such clients may still upload files and
access/delete history, so expose this mode only as intended by its operator.

`?token=` stays accepted for scripts and older clients, but a URL can end up in
proxy and server logs, shell history and browser history, so prefer the header.
For a download that a browser must open itself (a navigation cannot send
`Authorization`), an authenticated client asks `POST /api/download-tickets` with
`{"path": "/api/jobs/{job_id}/archive"}` (or a job file,
`/api/jobs/{job_id}/files/{relpath}`, or `/api/history/archive`, written exactly as
it will be requested) and receives 201 `{"ticket", "url", "expires_in": 60}`.
`url` is that path with `?ticket=…`: 64 random hexadecimal characters that admit
one GET of that path within 60 seconds. Only the ticket's SHA-256 is held, at most
64 tickets are outstanding (429 `too_many_tickets`), and a path outside those three
download routes is 422 `invalid_ticket_path`. A ticket presented a second time,
after it expired, for another path or with another method is spent and refused
with 401 `ticket_invalid`; loopback clients also redeem tickets exactly once. A ticket never grants
settings access and never lets the startup token into a URL. The event stream
`GET /api/jobs/{job_id}/events` takes the header like any other request (a
fetch-based reader can send it; the browser's EventSource cannot).

All model settings endpoints require the token by default. With `--no-auth`,
only direct loopback peers without forwarding headers can access settings. Every settings response, including rejected requests,
carries `Cache-Control: no-store`. Static UI bootstrap remains accessible under
the Host policy so a remote user can enter a service token. Browser launch places
the token in the URL fragment; the UI removes it immediately and sends every
request, event streams and preview images included, with Bearer authentication;
an archive download uses a ticket, so no URL the workbench requests carries the
token. Credentials are retrieved only through the explicit connection-edit route
and are never stored by the browser workbench.

Host validation accepts localhost, IP literals, and explicit `--allowed-host`
entries. A state-changing request with an Origin must match the request Host
and effective port, or an explicitly allowed hostname. Another localhost port
is a separate origin; userinfo, non-root paths, query strings, fragments and
non-HTTP(S) origins are rejected. Forwarding headers never grant authentication.
A proxy that strips all forwarding information cannot be detected in `--no-auth`
mode; keep token authentication enabled when proxying the service.
File paths reject traversal, absolute paths, and symlink components. Downloads
use attachment responses with `nosniff`. HTML and SVG artifacts remain untrusted
content; downloading them does not execute them on the server.

Uploads are bounded at 100 MiB per file, 1,000 items, and 5 GiB plus 64 MiB per
request. Text form fields are limited to 1 MiB. Upload names are flattened to
their final path segment (a folder's structure is not kept; same-named files from
different folders become `name (2).ext`), so a client that uploads a folder sends
its files individually.

Uploads are made durable once per job, not once per file. A job is published only
after every retained upload has been handed to the drive and one full flush has
covered them all, followed by the metadata and parent-directory syncs as before. On
macOS, where a full flush per file (`F_FULLFSYNC`) costs milliseconds, each file gets
an ordinary `fsync` and a single full flush follows; on other platforms each file is
synced once, in one batch, instead of as it arrives. Measured with the build and
commands below, creating a job of 1,000 small HTML files fell from about 4.5 s to
under a second:

| Run (alternating) | Before | After |
|---|---|---|
| 1 | 4.40 s | 0.58 s |
| 2 | 4.49 s | 1.97 s (the previous job's 1,000 conversions were still running) |
| 3 | 4.58 s | 0.76 s |

Command: `curl -F files=@f0.html … -F files=@f999.html http://127.0.0.1:PORT/api/jobs`
(`time_total`, HTTP 201) against `target/debug/markitai serve` instances started with
`env -i` and an isolated `MARKITAI_HOME`. The files are authored fixtures
(`<h1>Doc N</h1><p>small file N</p>`), the build is the unoptimized `dev` profile,
and the platform is macOS on APFS. Only macOS was measured; no Linux or Windows
figure is claimed, and the numbers are one machine's single-digit samples. JSON Markdown results are limited
to 64 MiB; larger output remains available through file downloads. Files and ZIPs
stream in bounded chunks. Every archive request builds its own private temporary
ZIP, retained until the response completes or disconnects. It never rewrites a
shared job ZIP, so concurrent downloads do not invalidate each other. Internal
`.images.lock` files are excluded from downloads and ZIPs. Rerun publication rebases
image-description metadata paths from its staging directory to the final job output
and merges sibling entries while holding the same stable lock as the core writer.

The root URL serves the embedded [browser workbench](web-ui.md), and `/jobs`
serves the same page for its workspace view, so a reload or Back stays there.
Other paths keep JSON 404 errors; the workbench's own files live under fixed
`/ui/` routes. The bundle and the vendored Markdown libraries are embedded gzip
compressed and sent that way to clients that accept gzip (decompressed once for
others); every workbench response revalidates against a content ETag
(`Cache-Control: no-cache`, 304 on a match). Without `--no-open` the service asks
the system to open the workbench, passing the startup token in the URL fragment;
with `--no-open` it does not open anything.

At startup the service prints, on stderr: `Markitai server listening on http://ADDRESS`
and, unless `--no-auth`, `Remote access token: …` (these two lines are scripted
against and stay English in every language); the address to open in a browser
(with the token in its fragment for local and remote browsers alike, unless
`--no-auth` was explicitly selected); the directory
that holds jobs and history (`MARKITAI_HOME/serve/jobs`, shown absolute); and a
Ctrl-C hint. The other sentences follow the terminal language (`MARKITAI_LANG`,
then `LANG`, then `LC_ALL`, Chinese for a `zh` prefix). When the listener is not
loopback-only (`--host 0.0.0.0`, a network address) it first warns that the service
is reachable from the network and that the token is the credential; with
`--no-auth` the warning says anyone who can reach the address can convert files and
read, download or delete the whole history, and what to do instead. A port that is
already taken stops the service with a message naming the address and suggesting
`--port` (`--port 0` picks a free one) instead of the bare operating-system error.
The workbench reports an unreachable service, checks again
every five seconds and says when it is connected again. It shows the percentage
and bytes of an upload while it is sent, can abort it, and offers Stop remaining
(the cancel route above) while original items wait for a slot.

## Validation scope

The independent Unix CLI process suite in
[`tests/serve.rs`](../crates/markitai-cli/tests/serve.rs) uses private configuration
and `MARKITAI_HOME`, authored text/EML fixtures, and loopback HTTP gates. It covers
submission and name collisions, public response types, SSE, downloads, concurrent
ZIPs, source-upload removal, restart and late CLI history import, malformed form
rollback, host/origin/path protection, shutdown queue cancellation, stop requests
for waiting items, and explicit
metadata-publication failure. Additional cases in
[`tests/serve/rerun.rs`](../crates/markitai-cli/tests/serve/rerun.rs) cover per-item option
inheritance/replacement, enhancement and failure preservation, sibling overlap,
queued cancellation, shared-asset deletion, and retry metadata failure followed by
restart. Module tests exercise committed/uncommitted file recovery, the error
body shape, and router-level `error_code` values for failed, retried, stopped and
shutdown items and request `reason`s. Separate synthetic-peer router tests exercise
remote token and trust decisions without relying on a host network interface,
including download tickets (issued only with the token, one GET of their own path,
spent when shown elsewhere or with another method, single-use on loopback too, refused for
settings and other non-download paths); `server::tickets` tests the path rule,
expiry, single use and the 64-ticket bound, and `server::openapi` the document
against the route table.
[`tests/serve_terminal_usage.rs`](../crates/markitai-cli/tests/serve_terminal_usage.rs)
contains private loopback cases for paid authentication errors, zero-token recorded
responses, SSE/GET/restart agreement, a post-core publication obstruction retaining
old bytes, subsequent unknown usage clearing, and invalid history diagnostics.
[`tests/serve/gates.rs`](../crates/markitai-cli/tests/serve/gates.rs) covers the
up-front model refusal for creation and retry, the service's wording for malformed
bodies, options and retry bodies, and a 250-file job that keeps every upload for
retry; [`tests/serve_startup.rs`](../crates/markitai-cli/tests/serve_startup.rs)
runs the binary for the startup lines (English and Chinese), the network-listener
warning and the taken-port message. Module tests check the no-authentication
warning text and the option parser.
These scoped checks do not establish complete REST/UI compatibility, production
load limits, remote-provider behavior, or cross-platform acceptance.
