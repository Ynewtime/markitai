# Embedded document workspace

`markitai serve` includes a locally bundled web interface at `/`. The page and
its JavaScript, CSS, Markdown renderer and sanitizer are part of the native
binary. No Node runtime, frontend build, CDN or external font is needed to run it.
`--no-open` leaves browser opening to the operator. The REST API remains available;
unknown API paths keep JSON errors instead of returning the application shell.

The workspace accepts multiple files, whole folders and one URL per line in the same
job. Presets come from service capabilities; individual tri-state overrides preserve
server defaults until explicitly selected. It distinguishes uploading/creating a
job from conversion progress, subscribes to the service's snapshots/item/job SSE
events, and reconnects by refreshing the authoritative snapshot. A selected job
is retained in the address bar so reloading can restore it.

Results expose base/enhanced Markdown variants, safe rendered preview, original
source text (the rendered preview and print omit a leading YAML frontmatter
block, as the reference preview does; Source shows the complete text), copying, per-file downloads and job ZIPs. Retry inherits the item's
stored options; Enhance explicitly submits the current job options with LLM
enabled. Both use the existing service operation routes. Item and history removal
ask for confirmation and report service failures; the UI does not invent a
successful state. History search and pagination are local views of the service's
saved job summaries, not claims of a new server pagination API.

## Selecting input and following a job

A status line under Preset says whether model features can work: "LLM ready" with
the number of models, or "LLM not configured" with a link to Connections. While the
service reports no routable model, the presets whose definition turns the LLM on
(Standard and Rich) stay listed but disabled and marked "needs a model", and the
LLM, image alt text and image description options are disabled with the reason
shown beneath them, in English and Chinese. The status refreshes when a connection
is saved. The service enforces the same rule: a job or retry that asks for model
processing without a model is refused with 422 `llm_unavailable`
([error responses](serve.md#job-workflow)); if the page's view was stale it reads
the capabilities again and shows that refusal. An item that failed with
`no_model_configured` additionally offers Retry without LLM, which resubmits the
job's options with the LLM off.

Every conversion option has a one-sentence explanation: a tooltip on the field, and
visible text beneath it on screens without hover (and wherever the option is
disabled). The options panel's hint counts the choices that differ from the server
defaults. The last chosen options (preset, profile, strategy, backend and the
tri-state overrides) are remembered in `localStorage` (`markitai.options`) and
applied on the next visit; a remembered choice that needs a model is not applied
while none is available. Reset options clears them. This is a per-viewer
convenience: a blocked or unreadable store simply means the page starts from the
server defaults, and the stored value is validated on every read.

Files dropped anywhere on the Convert view are added to the selection; a drop
elsewhere is ignored rather than letting the browser open the file in place of
the workspace. Choose a folder (a `webkitdirectory` input) adds a folder's files,
and a folder dropped on the page is walked through the browser's directory entry
API (the entries are read during the drop event); the list then shows each file's
folder-relative path. Hidden and system files inside the folder (names starting with
a dot, `Thumbs.db`, `desktop.ini`, and everything under a dot-folder such as `.git`)
are skipped and counted in the message; the folder you chose, and files you pick one
by one, are never filtered. A selection is capped at the service's item limit (1,000
files and URLs per job): the first files that fit are kept and the message says the
rest need another job. The service flattens upload names, so folder structure is not
kept in the results (same-named files become `name (2).ext`). The same file chosen twice (same name, size and modification time)
is listed once, with a short notice. A file over the 100 MiB upload limit is
marked in its row as soon as it is chosen, and submitting names it instead of
uploading anything. The selection shows its file count and total size and can be
cleared at once. Each URL line is checked before upload: a bare domain such as
`example.com/page` gets `https://`, and the first unusable line is quoted in the
message, with the address field focused.

While files are being sent, the form shows the upload's own progress under the
submit button: the percentage and the bytes sent of the total, in a progress bar
labelled by that text (`aria-valuetext` repeats it). The multipart submission uses
XMLHttpRequest because fetch reports no upload progress; redraws are limited to
one per 100 ms and the completion is never dropped. Once every byte is sent the
line says the service is saving the files and creating the job. Conversion
progress stays in the results panel, which appears only after the job exists.
Cancel upload aborts the request and keeps the selection. If the service had
already accepted the job, it continues and appears in History. While original
items of a running job wait for a conversion slot, Stop remaining calls the
service's cancel route: waiting items end as stopped (and can be retried), items
already converting finish. On narrow screens the results panel is scrolled into
view after a job starts or a saved job is opened.

Each row states its kind and status in words (queued, converting, done, failed,
skipped). An image skipped because no text was extracted offers Retry with OCR, which
resubmits the job's options with Local OCR on (and says it can also be enhanced with
a model). An unsupported file type offers only Delete: converting it again cannot
help. A job with two or more items has a filter (All, Done, Failed, Skipped, with
counts; a skipped item is counted apart from done) and Retry all failed (N), which
queues each retryable failed item again with its own options, one request each, and
shows progress on the button; items that cannot be fixed by converting again are not
counted. Download ZIP is hidden while a job has no item with output, since its
archive would be empty; History hides the row's ZIP for the same reason, and counts
items as "1 item", "3 items", "2 done", "1 skipped", "1 failed". The unsupported-format message keeps its list
of accepted extensions folded under Supported formats. Other failures are stated
by cause, such as a page that answered HTTP 404, a website that refused the
connection, a timeout, a missing model or unavailable local OCR, an input over a
size limit, or a stop request; the service's original message is folded under
Details (see [Language and appearance](#language-and-appearance)). Enhance is
offered only while the service reports a routable model; otherwise the action is
omitted rather than shown disabled. Opening a result moves keyboard focus to its
title; a result with a single Markdown version has no version chooser.

After a job starts or a saved job is opened, keyboard focus moves to the results
heading. When the focused row action disappears (Retry while the item is queued
again, a deleted row), focus moves to that item's next control, or to the results
heading, instead of being lost. An address whose `?job=` names a job that no
longer exists reports that once and drops the parameter, so reloading does not
repeat the error. Download all is hidden while History is empty.

Messages appear in one fixed region at the bottom of the window, so they are
visible wherever the action happened. Confirmations fade after six seconds;
errors stay until dismissed or replaced. When a request cannot reach the service,
the header shows Offline (also on narrow screens), the page checks the service
every five seconds and reports when it is connected again. Copy uses the
Clipboard API and, where a browser withholds it (for example plain HTTP on a LAN
address), the selection copy command; if both are refused it says so.

## Language and appearance

The interface is available in English and Chinese. Without a stored choice, a
browser whose first language starts with `zh` gets Chinese; the EN/中 control
switches immediately, including job rows, history, connections and comparison
summaries. The theme control cycles through automatic (following the operating
system), light and dark. Printed and PDF output always uses the light palette.

These two preferences and the last chosen conversion options (`markitai.options`,
described under [Selecting input](#selecting-input-and-following-a-job)) are the only
values this application writes to `localStorage` (`markitai.lang`, `markitai.theme`).
Nothing else, in particular no token or credential, is stored there. A small classic script,
`/ui/boot.js`, applies them in the document head before first paint; it runs
under the same `script-src 'self'` policy as the module scripts. The page's static
text is English, so when the resolved language is Chinese boot.js also sets the
Chinese tab title and marks the document (`data-i18n-pending`): the stylesheet
keeps translatable text, placeholders and select values transparent until
`i18n.js` has translated them and removes the mark. Layout is unchanged while
marked. If the modules have not run after three seconds, boot.js removes the mark
itself, so the page is never left blank; without JavaScript the mark is never set.
The served HTML stays a fixed embedded file: the stored choice lives in the
browser, where the server cannot see it.

The service answers in English. Its errors carry stable codes: an API error's
`reason` (then its status-derived `code`) and a failed item's `error_code`, which
is the core's conversion error code or a service cause (see
[error responses](serve.md#error-responses)). The page maps them to text in the
interface language, refining recognizable message shapes such as `HTTP 404`,
refused connections, timeouts, `LLM returned HTTP 401`, unavailable local OCR or
`exceeds the 500 MiB limit`. Provider probe and discovery details, which the
core words as fixed phrases, are translated the same way; provider names are
proper nouns and stay as they are, while generic labels (OpenAI compatible,
Unknown provider, the discovery status) are translated. Wherever a translation
replaces the service's wording, the original is kept: folded under Details in
job rows, messages and settings status, or as the element's tooltip for the job
summary and discovery line. A message that matches no known code or shape, for
example from an older history without `error_code`, is shown as written. The
access-token control is shown only when this tab uses a token or the service has
answered 401.

## Views and the address bar

Convert, History and Connections have their own addresses: `/`, `/?view=history` and
`/?view=settings`, next to `?job=<id>` for the open job. Switching views adds a
browser history entry, so Back returns to the previous view and a reload stays where
you were (on History or Connections the open job still loads in the background; Open
in History leaves a Back entry to the list). An unknown `view` value shows Convert.
The service serves the same page at `/` for all of them; only the query changes.

## Connections and concurrent editing

The Connections view lists configured and available providers, saved deployments
and detected session models. It can discover models, select several at once,
manually enter identifiers, append an atomic batch, edit group/model/weight,
perform a real transient model probe, and delete deployments or connections.
Unsupported discovery still leaves manual entry available.

Collection views contain only the server's redacted fields. The raw credentials
endpoint is called only when a connection is explicitly opened for editing.
API key and base URL each have Keep, Replace and Clear choices, preserving the
server's omitted/null semantics. Add connection (and Reset draft) starts a new draft
with the API key in Replace, so the field can be typed in at once; submitting it empty
sends no key, and the provider's environment variable is used, as with Keep. Editing a
saved connection, and setting up a provider whose credentials already exist, keep the
key locked in Keep until you choose Replace or Clear. Password controls hold values in memory; provider
credentials are never written to sessionStorage/localStorage by this application.
Credentials are cleared when a draft is reset or successfully saved.

Mutations include the revision captured by that draft. A conflict refreshes the
visible connection/model lists but retains the draft and old revision. The user
must explicitly acknowledge the current revision after review and submit again;
there is no automatic overwrite/retry. Deployment dialogs use the same behavior.
Only a revision conflict (`stale_revision`, or `config_changed` when the file
changed during the save) opens this review; a server-side session override
(`settings_read_only`) or other write restriction remains visible as an error;
the page does not pretend it saved a blocked configuration.

## Preview and token boundaries

Markdown rendering uses locally vendored [marked 18.0.14](https://github.com/markedjs/marked/releases/tag/v18.0.14)
and [DOMPurify 3.4.16](https://github.com/cure53/DOMPurify/releases/tag/3.4.16).
The pinned official npm archives were checked against their registry SHA512
integrity before extracting the runtime files. Sources, file SHA256 hashes and
package integrity values are in `vendor/web/provenance.json`; MIT and Apache-2.0
license texts accompany them. Root release packaging includes these notices.

The renderer never assigns arbitrary Markdown directly to the live DOM.
DOMPurify returns an HTML-only fragment, with scripts, style, forms, frames,
SVG/MathML and embedded media excluded. Image and attachment URLs must resolve to
an exact entry in the selected result's artifact list. Only raster PNG/JPEG/GIF/
WebP/AVIF artifacts are previewed; external images and SVG appear as labeled
placeholders. External links use only HTTP(S)/mailto with no userinfo, open with
noopener/noreferrer and never receive the service token. Literal source is shown
with textContent. CSS, arbitrary scripts and browser plugins cannot be enabled by
the document.

The token is read first from `#token=`, with `?token=` compatibility, then both URL
copies are removed immediately using replaceState. It is stored only for the tab
session, with a memory fallback when storage access is blocked. API fetch uses
Bearer; EventSource, artifact images and downloads use a token query parameter
only on this service's `/api/` URLs. Cross-origin/credentialed URLs are rejected,
and fetch redirects are refused. No provider API request is issued directly by
browser code; discovery, probes and conversion go through the service.

Every embedded resource uses an explicit fixed route and no-cache, because its
URL is not content-hashed. Responses carry nosniff, no-referrer and a restrictive
Content Security Policy allowing local scripts/styles/connections and local
raster images. No filesystem directory is served. The existing service guard
still controls Host/Origin and API authentication, including remote settings
restrictions.

## Authored validation and current boundaries

`crates/markitai-cli/tests/serve_web.rs` exercises real native HTTP serving:
exact embedded bytes, MIME/HEAD/cache/security headers, unknown API and asset
404s, filesystem non-exposure and Host/method rejection. The pure JavaScript
`web/api.test.mjs` tests fragment/query cleanup, blocked storage, token destination
restriction, artifact identity, authenticated request redirect policy, the
offline error, URL-line parsing, duplicate files, the copy fallback, error
localization by reason and code, the XMLHttpRequest upload (progress, token,
cancellation, service, network and redirect failures), the progress text and
the redraw throttle. `web/i18n.test.mjs` checks that both languages define the
same keys and placeholders, that every key the page, scripts and message tables
use exists, that every `reason` in the Rust service sources and every core error
code has localized text, item-error and provider-phrase translations, language
detection, plural forms, the theme cycle and boot.js's first-paint mark (run in a
`node:vm` context); `web/result-tools.test.mjs` also checks Chinese comparison
and print messages. `web/workspace.test.mjs` covers the remembered options (including a
blocked or damaged store), the model-dependent preset rule, folder selection (hidden
file rules, batched directory entries, the item limit, same-named files in different
folders), the ledger filter and retry rules, history counts and plurals, view
addresses, and that every option has help text in both languages; `web/settings.test.mjs`
runs the connection editor against a recording document to check that Add connection
leaves the key input enabled in Replace. Run them with
`node --test crates/markitai-cli/src/server/web/*.test.mjs`; Node is only a
development test tool. Syntax checks use `node --input-type=module --check` with
each authored JS file on stdin. The embedded resources are `index.html`,
`style.css`, `icon.svg` (the tab icon), `boot.js`, `app.js`, `api.js`, `i18n.js`,
`preview.js`, `result-tools.js`, `settings.js`, `workspace.js` and the two vendored libraries.

These tests and actual release-browser acceptance passed in
[round twenty-one](validation/service-ui-round21.md), including mixed file/URL
jobs, sanitization, downloads, two-tab revision conflicts and mobile layout.
Current scope excludes OAuth login, source editing, embedded
video/audio playback, external-image preview and the reference React UI's exact
appearance. Model responses remain untrusted document content.


## Compare and print a document

When one item lists an exact base/enhanced Markdown pair, Changes compares those
versions in source order, including final newline differences. It uses only the
item's returned artifact paths. Ambiguous inventories do not enable the action.
The view renders every line as text, including HTML/code examples. It does not
execute source markup or save edits. Requests use the existing authenticated,
same-service file endpoint and reject redirects.

Comparison stops with a download suggestion beyond 6,000 lines, 2,097,152 UTF-16
code units per side, or 1,000,000 middle-line comparison cells after equal edges
are removed. Downloads are additionally bounded to 8 MiB per side. No partial
comparison is presented as complete. Choosing another result or view invalidates
pending comparisons; no source text is retained in browser storage.

Print / PDF prints the currently selected version's existing safe rendered
preview, even when Source or Changes is visible. The browser's own dialog can
save a PDF. It uses a temporary clone, not an iframe, popup, new backend or relaxed
CSP. Local manifest-approved raster images must finish loading within ten seconds;
a broken or stalled image prevents printing with an explicit notice. Untrusted
scripts, external images and embedded media remain blocked as in Preview. Link
URLs are removed from the printed clone so authenticated download URLs are not
embedded in generated PDFs. This also removes external clickable PDF links.
A direct browser print of the workspace itself hides every link that carries
the service token, so such a PDF embeds no authenticated URL either.

Print styling includes wrapping code, repeating table headers and constrained
images. It does not promise paginated Office fidelity or a deterministic PDF
renderer. The selected title and document-only layout are restored after the
print dialog, a thrown print error, selection/navigation cancellation, or a
120-second fallback for browsers missing the completion event. Root's actual
browser acceptance must separately verify pagination, image loading, sanitization
and cancellation; the candidate's JS checks alone do not prove these outcomes.
