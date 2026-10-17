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
under the isolated Markitai home. Service reads follow a symlinked
configuration to its file and reject nonregular files and files larger than
8 MiB, as the CLI does. Settings saves change the configuration
snapshot for newly admitted jobs and retries; an active job keeps its original
snapshot. CLI model/provider session overrides remain effective and make
settings writes unavailable until restart without those overrides.

## Machine-readable API description

Each finished item's [document JSON](json-output.md), the same object `markitai -f json`
writes, is served by `GET /api/jobs/{job_id}/items/{item_id}/document`
(`features.document_json`).

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
   SVG/browser availability, Cloudflare local readiness, the 1,000-item limit and
   the seven-day history retention (`limits`), and, for clients shared with other
   backends, `implementation: "rust"`, `docs_url`, `source_url` and `features` (`cancel`,
   `download_tickets`, `url_lists`, `history_items`, `original`,
   `result_excerpt`, `job_labels`, `item_assets`, `item_archive`, `config_edit`,
   `batch_delete`, all true, and `token_regenerate` and `user_tokens`, true unless
   `--no-auth`), and `access`: `{"role": "admin"}` for the server token (or a
   loopback client with `--no-auth`), `{"role": "user", "label": …}` for a user
   token, `{"role": "user"}` for another `--no-auth` client.
2. `POST /api/jobs` accepts form fields `urls` (a JSON array), `options` (a JSON
   object) and `label`, plus repeated multipart `files`. `label` names the
   submission in history (a folder or `.urls` file name): plain text without
   control characters, at most 120 characters after trimming, else 422
   `invalid_label`; it is returned as `label` (null when absent) in snapshots and
   history entries. URL-only requests also accept
   `application/x-www-form-urlencoded`. A successful submission returns
   HTTP 201 with `job_id` and ordered `{item_id,name,kind}` entries. File entries
   precede URL entries. Upload names are sanitized and made unique, including
   case-folded collisions. Base and enhanced Markdown names are reserved as a
   pair, so an upload named `notes.llm` cannot overwrite another item's enhanced
   result. An uploaded `*.urls` file is read as a URL list rather than converted
   as a document: each entry becomes a URL item, a line starting with `#` and a
   blank line are skipped, an entry that names its output (`https://example.com/x
   Report`, or `{"url": …, "output_name": …}` inside a JSON array) reserves that
   name, and a non-HTTP(S) entry is skipped the way the CLI skips it. A list that
   cannot be read is 422 with the filename in `detail`: `invalid_url_list` (not
   UTF-8, or a JSON array the parser rejects), `empty_url_list` (no usable URL),
   `url_list_too_large` (over 1 MiB) or `invalid_output_name` (a name with a
   directory component).
3. `GET /api/jobs/{job_id}` returns the current snapshot. Items move from
   `queued` to `running` to `done` or `error`; failed conversions remain in the
   job with their error and nullable output. A job containing failed items
   still completes with `status: "done"` and separate success/failure counts.
   An item reports `source_size`, the bytes of its uploaded original, and
   `models`, the models whose requests produced its retained result (named as
   its usage records them, in name order); each is omitted when unknown or
   empty.
4. `GET /api/jobs/{job_id}/events` streams a first `snapshot`, subsequent `item`
   and `job` events, and idle keepalives. A terminal job event closes the stream.
   Slow subscribers receive a fresh snapshot after a bounded event-buffer
   overrun.
5. `GET /api/jobs/{job_id}/items/{item_id}/result` returns `name`, selected
   `variant`, `markdown`, downloadable `{relpath,size}` artifacts, `truncated`
   and `markdown_bytes` (the size of the whole Markdown). `?max_bytes=N`
   (N ≥ 1024, else 422 `invalid_max_bytes`) returns only the start of the
   Markdown, at most N bytes, ending at a line break in its last quarter when
   there is one and never inside a UTF-8 sequence; `artifacts` stays complete.
   Fetch an artifact through `GET /api/jobs/{job_id}/files/{relpath}`. Download
   all output files through `GET /api/jobs/{job_id}/archive` after completion, or
   one item's through `GET /api/jobs/{job_id}/items/{item_id}/archive`: every
   file its result lists, at the same paths, plus its kept original under
   `original/` (404 `result_unavailable` without a result, 409 `item_busy` while
   it runs or waits for a rerun, 413 `archive_too_large` as for a job).
   `GET /api/jobs/{job_id}/items/{item_id}/assets/{relpath}` shows one of that
   item's own extracted images in a page: only a PNG, JPEG, GIF, WebP or AVIF
   that the item's result lists, sent inline with
   `Content-Security-Policy: default-src 'none'; sandbox`, `nosniff`,
   `Cache-Control: private, no-cache` and an `ETag` (a matching `If-None-Match`
   is 304); anything else, another item's image or a path leaving the job is
   404 `file_not_found`. No download ticket names it.
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

## Originals for Compare

Four routes show an item's source beside its Markdown, for items in any status
while the source is kept. `GET /api/jobs/{job_id}/items/{item_id}/original`
describes it without drawing anything: `kind` is `pdf`, `image` (png, jpeg, gif,
webp, avif, bmp, tiff, heic/heif), `html`, `text` (md, markdown, txt, csv, tsv,
json, xml, yaml, yml, log), chosen by the kept upload's name; `html` for a URL
item whose page was converted from HTML in this process (the static or browser
fetch), whose screenshots, when the conversion took some, are listed as its
pages too; else `screenshots` when the conversion saved page screenshots; else
`none` with `reason` `not_kept` (a CLI-recorded job, a removed upload, a URL page
read through Jina, Defuddle or Cloudflare or from the page cache),
`unsupported_format` (no browser view and no screenshots, for example DOCX or
SVG) or `render_unavailable` (a PDF or image this build cannot open). It also
gives the original's `name`, `media_type`, `size`, `download` path, `pages`
(page boxes: PDF points of the visible box after rotation, image and screenshot
pixels; at most 2,000), `page_count`, `text_layer` (PDF only), for `html`
`view`: the path of the kept page for a sandboxed frame, and for a URL item
`capture` (null for uploads): `{page, screenshot}`, how usable the kept page and
the screenshots were when they were captured, each `{state, text_length, title,
reason}` or null when that view was not kept. `state` is `challenge` when the page
was a verification, captcha, bot check or login page in place of the content
(`reason` `challenge`, `verification` or `login`, from the same site rules the
fetch uses), `empty` when it only asked for JavaScript (`javascript`) or showed
fewer than 80 characters of text and no picture, video or audio player
(`blank`; a picture with a one-line caption is `ok`), else `ok`; `text_length` counts the
visible text with white space runs as one (the kept page as a frame without
scripts shows it, `<noscript>` included; the screenshots as the browser showed
them), `title` is the page's title (at most 200 characters). A job saved before
this field had it reads the kept page when asked; its `screenshot` is null.
`blocks` is the path of `…/original/blocks` when the screenshots have block
boxes, else null.

`…/original/view` sends that page as `text/html` for the workbench's frame: the
kept bytes with one insertion after `<head>` (else `<html>`, else the start),
`<meta http-equiv="x-dns-prefetch-control" content="off">` and
`<base href="<the item's URL>" target="_blank">` (an upload gets no `href`),
under `Content-Security-Policy: sandbox allow-same-origin allow-popups allow-popups-to-escape-sandbox;
default-src 'none'; img-src https: data:; style-src https: 'unsafe-inline';
font-src https: data:; media-src https:; form-action 'none'; frame-ancestors 'self'`,
with `nosniff`, `Referrer-Policy: no-referrer`, `X-DNS-Prefetch-Control: off` and
`Cache-Control: no-store`: it runs no script (there is no `allow-scripts` and no
`script-src`, so `default-src 'none'` refuses every script, inline handler and
`javascript:` URL) and sends no form, loads only the site's pictures, style
sheets, fonts and media over HTTPS (relative addresses resolve to the site
through the `<base>`), sends no referrer, and only this service's own pages may
frame it. The frame keeps the service's origin (`allow-same-origin`) so the
workbench can read the page it shows; since nothing in it can run, nothing in it
reaches the service. The page is sent as kept, not rewritten. Other kinds are 404
`original_unavailable`, a page over 10 MiB 413 `result_too_large`. It is opened
with a download ticket, since a frame cannot send the header; the static pages'
policy allows `frame-src 'self'` for it.

`…/original/file` sends the kept upload or page snapshot as an attachment with
`nosniff` and `Content-Security-Policy: sandbox; default-src 'none'`; HTML and SVG
are sent as `text/plain; charset=utf-8`, so a direct visit never renders them.
It may be opened with a download ticket. `…/original/pages/{n}?width=` returns
page `n` as an image `width` pixels wide (clamped to 160–2,400, default 1,200):
a PDF page drawn on white (narrower when the 32-megapixel limit requires), the
image itself when browsers show its type and it is no wider, otherwise a scaled
PNG (always for TIFF and HEIC/HEIF), or screenshot `n`. Pages carry
`Cache-Control: private, max-age=86400` and an ETag, and recently drawn ones are
kept in memory (32 MiB). `…/original/pages/{n}/text` returns a PDF page's text
as up to 5,000 lines in reading order, each a run of text on one baseline, with
`bbox` `[x0, y0, x1, y1]` in points from the page's top-left corner, and
`images`, the boxes its pictures are drawn in (same frame, drawing order, at
most 500, clipped to the page, none under 4 points on a side). At most
two pages are drawn at a time; another request gets 429 `render_busy` with
`Retry-After`. Errors: 404 `original_unavailable`, 404 `page_not_found`, 422
`invalid_page_request` (a page number or width that is not a whole number) and
422 `render_failed`.
`…/original/outline` (`features.pdf_outline`) returns a PDF's own bookmarks as
`{items: [{title, page, level}], truncated}`: outline order, titles on one line
(at most 300 characters), 1-based pages and levels, an entry whose destination is
not a page of this document taking the next entry's page, at most 2,000 entries;
`items: []` when the PDF has none or it cannot be read; 404
`original_unavailable` for anything but a PDF.

`…/original/blocks` (no ticket) answers `application/json`, `Cache-Control:
no-store`: `{version: 1, width, height, blocks: [{tag, text, src?, bbox}]}`, the
page's headings, paragraphs, list items, code, cells, quotes, captions, tables,
pictures and wordy containers in document order (at most 5,000) with their boxes
in CSS pixels from the page's top-left corner, read in the browser just before
it took the screenshots; `width` and `height` are the CSS size the screenshots
cover (they cut the page into tiles from the top). Else 404 `original_unavailable`.

A URL item's page snapshot is the HTML the item's Markdown was converted from,
at most 10 MiB, stored privately as `snapshots/<item_id>.html` in the job folder;
when the Markdown came from elsewhere (an API extractor) but the browser loaded
the page for its screenshots, that browser's page is kept instead. Beside it:
`<item_id>.capture.json` (the two qualities), `<item_id>.blocks.json` (the block
boxes, at most 4 MiB) and, when the browser loaded the page but no screenshots
were asked for, compare screenshots in `<item_id>.shots/<n>.jpg` (JPEG quality 70,
2,000 px tiles of the page as `screenshot` configures them, at most 20; about
85–230 KB per tile, so a long page adds up to about 2 MB). Compare screenshots are
not part of the result (not in `out/`, its file list or archives) and are never
sent to a model or OCR; the descriptor lists them as `pages` when the item has no
screenshots of its own. A retry that kept any of them replaces all of them (an
old file it did not write again goes); a retry that loaded no page (a fetch-cache
hit, or a remote reader without a browser load) keeps the old ones, as does a
failed retry; deleting the item deletes them.

## Cloudflare request authorization

`GET /api/capabilities` includes `remote_services.cloudflare` with `configured`,
`available`, `reason`, `browser_rendering`, `file_conversion` and
`file_extensions`. These describe local prerequisites only. The check makes no
network request and returns no credentials, account ID or endpoint. It does not
verify cloud permissions, quota or service availability. `reason` is null when
ready, otherwise `not_configured`, `invalid_configuration`, `disabled_by_policy`
or `client_not_trusted`. Cloudflare bills the server's account, so only the
admin (the server token, or a loopback client of a `--no-auth` server) may use
it: a user token or an untrusted caller gets a fixed `client_not_trusted`
response without resolving credentials.

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
`auto`. Remote fallback of `auto` (defuddle.md, Jina Reader, and Cloudflare when
configured) runs only when the server's configuration file or `--config-json`
writes `fetch.remote_consent: always`, as in the CLI ([consent](fetch.md#consent));
the default leaves `auto` local. The server returns 403 `remote_processing_forbidden` for any caller
other than the admin (a user token included), 422 `remote_processing_confirmation_required` without current consent,
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
 "pricing_snapshots": ["litellm-1.106.0.dev2-selected-2026-10-11"]}
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
omitted. The snapshot's job-level `options` is projected the same way. This
object is not authorization. Clients supporting older servers may
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
The workbench's conversion window shows the failure above the retained result's facts
and keeps preview, download and retry available, with the service's full wording
behind Details.
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
that item, its retained upload, Markdown pair and owned assets/screenshots,
while keeping files claimed by another item. Image metadata rows for assets actually
removed are pruned in the same recoverable transaction; shared rows remain. The last item's deletion removes the
whole job. Deleting while any job work runs returns 409; repeated deletion returns
404. Native ownership indexes are preferred; legacy histories use exact Markdown
references and bounded converter filename suffixes. Arbitrary filename substrings
do not grant ownership.

`POST /api/history/delete` deletes several saved jobs and items in one request:
a JSON body `{"jobs": [job_id, …], "items": [{"job_id", "item_id"}, …]}` with
only those keys and 1 to 1,000 entries in all (else 422
`invalid_delete_request`; not JSON 422 `json_required` or `invalid_json`; over
256 KiB 413 `request_too_large`). Each entry is handled on its own, in order
(jobs, then items), exactly as `DELETE /api/history/{job_id}` and
`DELETE /api/jobs/{job_id}/items/{item_id}` would (deleting a job's last item
deletes the job), and the answer is 200 `{"deleted", "failed", "results"}` with
one `{job_id, item_id, ok, reason?, detail?}` per entry in that order (`item_id`
null for a job; `reason` and `detail` are what the single route would answer,
such as `job_not_found`, `item_not_found` or `job_running`). Deletion is final;
entries that succeeded are not rolled back when others fail.

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
`uploads/`, `out/`, URL page `snapshots/` (pages, capture records, block boxes,
compare screenshots), and an atomically replaced `meta.json`. Job, upload and output root directories use mode
0700 and generated metadata/uploads use mode 0600 on Unix. Current server
metadata uses the existing version-2 shape, with additive native output-base and
asset indexes, per-item options, and internal file-transaction commit identifiers. Existing CLI and reference terminal histories are readable. Legacy artifact lists
recover assets referenced in Markdown, including visible `assets/` profiles;
assets without references and without a native item index are still downloadable
by path and included in the ZIP, but are not guessed into an item artifact list.
`GET /api/history` refreshes jobs added by other local processes and returns the
existing summaries in reverse creation order; `?include=items` adds each job's
`options` and its items as the snapshot shows them. A job folder that cannot be read
(for example, one with a damaged retry journal) is skipped with one
`Serve: skipped unreadable history job …` line on standard error and left in
place; startup, history and the other jobs are unaffected, and the folder is
tried again on the next scan. `GET /api/history/archive` streams
a ZIP of completed histories. `DELETE /api/history/{job_id}` removes a terminal
job and its files; running jobs return 409.

History is kept for seven days (`limits.history_retention_days: 7` in the
capabilities), as in the reference service. A job is deleted seven days after its
`finished_at`; a folder without a readable one, such as a damaged record, is aged
by its modification time. The service sweeps at startup, after recovery and while
holding the lifetime lock, and then every hour until shutdown; CLI-recorded
archives in the same directory are swept too. An expired job is no longer listed,
served by any job route (404 `job_not_found`) or included in an archive, even
before the sweep removes it. Running jobs and jobs with a queued or running retry
are never expired; a retry's completion starts the seven days again. A sweep
deletes a job the way `DELETE /api/history/{job_id}` does; a failure is logged and
retried at the next sweep.

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
`/api/jobs/{job_id}/files/{relpath}`, an item's archive,
`/api/jobs/{job_id}/items/{item_id}/archive`, an item's original,
`/api/jobs/{job_id}/items/{item_id}/original/file` or its framed page
`…/original/view`, or `/api/history/archive`,
written exactly as it will be requested) and receives 201 `{"ticket", "url",
"expires_in": 60}`.
`url` is that path with `?ticket=…`: 64 random hexadecimal characters that admit
one GET of that path within 60 seconds. Only the ticket's SHA-256 is held, at most
64 tickets are outstanding (429 `too_many_tickets`), a path outside those
routes is 422 `invalid_ticket_path`, and a request body over 8 KiB is 413
`request_too_large`. A ticket presented a second time,
after it expired, for another path or with another method is spent and refused
with 401 `ticket_invalid`; loopback clients also redeem tickets exactly once. A ticket never grants
settings access and never lets the startup token into a URL. The event stream
`GET /api/jobs/{job_id}/events` takes the header like any other request (a
fetch-based reader can send it; the browser's EventSource cannot).

`POST /api/access-token/regenerate` (no body, or `{}`) replaces the token with a
new random one, trusted like the model settings (the token itself): every request
with the old token is then 401, open event streams end, and outstanding download
tickets are revoked. It answers `{"token", "network_url", "persistent": false,
"startup_variable"}` with `Cache-Control: no-store` (`network_url` is the
network address with the new token when the server listens on all interfaces;
`startup_variable` is `MARKITAI_SERVE_TOKEN` when that set the token) and prints
`Access token regenerated from the web workbench; other tabs and devices must
sign in again.` The new token lives in memory: a restart uses
`MARKITAI_SERVE_TOKEN` again, or makes a new one. User tokens are kept. With
`--no-auth` it is 409 `token_disabled`.

### Access levels and user tokens

The server token is the **admin** token. An admin can create **user** tokens in
the workbench (Settings → Access token) for other people or devices:
`POST /api/access-tokens` with `{"label": …}` (trimmed, 1 to 60 characters
without control characters, else 422 `invalid_label`; at most 50 tokens, else 409
`token_limit`) answers 201 `{"id", "label", "hint", "created_at",
"last_used_at": null, "token"}`, the only answer that ever carries the token
(`mku_` and 64 hex characters; `hint` is its last four). `GET /api/access-tokens`
lists `{"tokens": [{"id", "label", "hint", "created_at", "last_used_at",
"model_count"}]}`, oldest first (`model_count`: how many models of its own the
user saved), and `DELETE /api/access-tokens/{id}` revokes one at once (204;
unknown 404 `token_not_found`): its next request is 401 `token_required`, its
download tickets are refused, its open event streams end and its own models are
deleted. The server prints `Access token "<label>" created from the web
workbench.` or `… revoked …` (with `; its N models were deleted.` when it had
any), never the token. These routes, `POST /api/access-token/regenerate` and every
`/api/settings/` route are admin only: a user token gets 403
`settings_forbidden`; with `--no-auth` there are no tokens and the token routes
answer 409 `token_disabled`. All their answers carry `Cache-Control: no-store`.

A user token reaches every other route, URL conversion included, but sees only
the jobs it created: any other job is 404 `job_not_found` on every job route,
in `DELETE /api/history/{job_id}`, a batch deletion and a download-ticket
request, and `GET /api/history` (with or without `include=items`) and the
history archive list only its own jobs. Admins see every job; a revoked user's
jobs stay, for admins. The job's owner is saved in its `meta.json` and never
sent by the API.

Only each user token's SHA-256 is kept, in `MARKITAI_HOME/serve/access.json`
(`{"version": 1, "tokens": [{"id", "label", "sha256", "hint", "created_at"}]}`,
owner-only, replaced atomically), so user tokens survive restarts and admin-token
regeneration; `last_used_at` is kept in memory only. A file that cannot be read
is left alone (with one warning, and no user token works) until an admin creates
or revokes a token.

### Users' own models and shared models

A user token manages models of its own under `/api/me/models` (with the
`/api/settings/llm` suffixes: `/providers`, `/providers/{id}/credentials`,
`/model-discovery`, `/test`, `/deployments/batch`, `/deployments/{id}`,
`/providers/{id}`; any other caller gets 403 `user_token_required`; every answer
is `no-store`). They are kept per token in
`MARKITAI_HOME/serve/user-models/<token id>.json` (owner-only, replaced
atomically), never in the configuration file, and are deleted when the token is
revoked (a file left by a failed deletion is removed at the next start). A user
has at most 50 (409 `model_limit`). Each needs a key of the user's own, or an
address of its own for a server that takes none (422 `key_or_address_required`);
a provider that signs in with the server's own account or cloud login
(subscriptions, Vertex AI, watsonx, GigaChat, Bedrock without a key) is 422
`server_credentials`; an address on the server's machine (`localhost`, loopback,
`0.0.0.0`, `::`, also a provider's documented local address, so Ollama or LM
Studio need another address) is 422 `server_address`. A user's models never use
the server's environment keys or detected models, and name only the user's own
connections.

An admin shares a saved model with every user by `share: "shared"` on
`PATCH /api/settings/llm/deployments/{id}` (saved as `model_info.share` in the
configuration file; `private` is the default). A user's conversions, retries and
enhancements route to one pool: its own models, then the shared ones (weight
above 0, one routing group, no fallbacks), from the current settings on every
request. With neither, a request for model processing is 422 `llm_unavailable`
and conversions run without models. `GET /api/me/models` lists the shared
models as `{id, model, provider, weight, default_weight}` only: no user answer
carries a shared model's key, address or connection, and a failure names the
model without its host. A revoked user's jobs route to the shared models only.

A user's own models take the same batch (`remove` with `deployments`, all or
nothing) and the same weight rule (a whole number from 0 to 100, else 422
`invalid_weight`). A user may give a shared model a weight of its own with
`PATCH /api/me/models/shared/{id}` `{expected_revision, weight}` (`weight` is the
user's, `default_weight` the admin's; 0 turns it off for that user only, null or
the admin's weight goes back to the admin's), kept in its own file. `POST
/api/me/models/test` with `{shared_id}` tests a shared model on the server with
the admin's connection and answers only whether it passed and the failure's
`reason` (one at a time and 6 a minute per user, else 429 `provider_busy` with
`Retry-After`). `GET`/`PUT /api/me/config` is the user's Edit config: a JSON
document of its own `models`, `providers` and the `shared` weights, secrets
masked as in the admin's editor, saved through the same rules
(`features.user_config`).

### Convert defaults

The workbench's Convert options start from `GET /api/convert-defaults`
(`features.convert_defaults`), for every caller: `server` is the server's
defaults read from its running configuration (`llm` `llm.enabled`, `ocr`
`ocr.enabled`, `alt` `image.alt_enabled`, `desc` `image.desc_enabled`,
`screenshot` `screenshot.enabled`, `screenshot_only`, `pure` `llm.pure`,
`no_cache` `cache.no_cache`, `no_compress` the opposite of `image.compress`,
`profile` `output.profile`, `strategy` `fetch.strategy`), `own` a user token's
own, `defaults` the two together, plus `scope`, `can_save` and `revision`.
`PUT /api/convert-defaults` `{expected_revision, defaults}` saves them: the
admin's into `config.json` at those keys (a key the file already has is set, a
value the built-in default gives is not added) and into the running
configuration, so requests that leave an option out follow them too; a user
token's into its own file under `serve/user-models/` (`defaults: null` removes
them). Cloudflare can never be a default (it needs a confirmation in each
request): `strategy: "cloudflare"` or any other option is 422
`invalid_convert_defaults`; a stale revision is 409 `stale_revision`; an
anonymous caller gets 403 `settings_forbidden`. Answers carry
`Cache-Control: no-store`.

All settings endpoints require the admin token by default. With `--no-auth`,
only direct loopback peers without forwarding headers can access settings. Every settings response, including rejected requests,
carries `Cache-Control: no-store`. Static UI bootstrap remains accessible under
the Host policy so a remote user can enter a service token. Browser launch places
the token in the URL fragment; the UI removes it immediately and sends every
request, event streams and preview images included, with Bearer authentication;
an archive download uses a ticket, so no URL the workbench requests carries the
token. A saved connection's key is never returned: the connection-edit route
(`GET /api/settings/llm/providers/{id}/credentials`) shows it masked, as the
reference does (`sk…7890`: the first two and last four characters of a key of 12
or more, the last two of a shorter one), and an `env:NAME` reference by name. Its
address keeps its text but not its userinfo or the value of a secret query or
fragment parameter (`https://…@host/v1?token=…`); no settings route returns
either. An update without `api_key` keeps the saved key, and an `api_base` sent
back exactly as shown keeps the saved address; any other masked value is
refused. The browser workbench never stores credentials. The server's own keys
never go to an address a settings caller names: a connection saved without a
key stores the placeholder `markitai-no-key` and uses no environment
credential, and a keyed connection cannot be made keyless at another host (see
[service settings](service-settings.md#credentials-and-partial-updates)).

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

Uploads are synchronized before durable job publication. An interrupted or
failed upload does not establish a completed conversion. JSON Markdown results are limited
to 64 MiB (an excerpt by `max_bytes` of an item with an asset index is not);
larger output remains available through file downloads. Files and ZIPs
stream in bounded chunks. Every archive request builds its own private temporary
ZIP, retained until the response completes or disconnects. It never rewrites a
shared job ZIP, so concurrent downloads do not invalidate each other. Internal
`.images.lock` files are excluded from downloads and ZIPs. Rerun publication rebases
image-description metadata paths from its staging directory to the final job output
and merges sibling entries while holding the same stable lock as the core writer.

The root URL serves the embedded [browser workbench](web-ui.md), and `/jobs`,
`/settings` and its pages serve the same page for their views, so a reload or
Back stays there. A browser asking for any other page (a GET or HEAD that
accepts `text/html`, outside `/api/` and `/ui/`) gets the workbench with status
404, which says the page does not exist; API clients, other methods and unknown
`/ui/` files keep JSON 404 errors. The workbench's own files live under fixed
`/ui/` routes. The bundle and the vendored Markdown libraries are embedded gzip
compressed and sent that way to clients that accept gzip (decompressed once for
others); every workbench response revalidates against a content ETag
(`Cache-Control: no-cache`, 304 on a match). Without `--no-open` the service asks
the system to open the workbench, passing the startup token in the URL fragment;
with `--no-open` it does not open anything.

At startup the service prints, on stderr: `Markitai server listening on http://ADDRESS`
and, unless `--no-auth`, `Remote access token: …` (these two lines are scripted
against, stay English in every language and are never wrapped); the address to
open in a browser, on a line of its own under `Open in your browser:` so it copies
whole (with the token in its fragment for local and remote browsers alike, unless
`--no-auth` was explicitly selected); the directory
that holds jobs and history (`MARKITAI_HOME/serve/jobs`, shown absolute); and a
Ctrl-C hint. The other sentences follow the terminal language (`MARKITAI_LANG`,
then `LANG`, then `LC_ALL`, Chinese for a `zh` prefix). When the listener is not
loopback-only (`--host 0.0.0.0`, a network address) it first warns
(`! Binding to …`) that the service is reachable from the network and that the token is the credential; with
`--no-auth` the warning says anyone who can reach the address can convert files and
read, download or delete the whole history, and what to do instead. A port that is
already taken stops the service with a message naming the address and suggesting
`--port` (`--port 0` picks a free one) instead of the bare operating-system error.
The workbench reports an unreachable service, checks again
every five seconds and says when it is connected again. It shows the percentage
and bytes of an upload while it is sent, can abort it, and offers Stop remaining
(the cancel route above) while original items wait for a slot.
