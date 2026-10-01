# Embedded document workspace

`markitai serve` includes a locally bundled web interface at `/`. The page and
its JavaScript, CSS, Markdown renderer and sanitizer are part of the native
binary. No Node runtime, frontend build, CDN or external font is needed to run it.
`--no-open` leaves browser opening to the operator. The REST API remains available;
unknown API paths keep JSON errors instead of returning the application shell.

The workspace accepts multiple files and one URL per line in the same job.
Presets come from service capabilities; individual tri-state overrides preserve
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

Files dropped anywhere on the Convert view are added to the selection; a drop
elsewhere is ignored rather than letting the browser open the file in place of
the workspace. The same file chosen twice (same name, size and modification time)
is listed once, with a short notice. A file over the 100 MiB upload limit is
marked in its row as soon as it is chosen, and submitting names it instead of
uploading anything. The selection shows its file count and total size and can be
cleared at once. Each URL line is checked before upload: a bare domain such as
`example.com/page` gets `https://`, and the first unusable line is quoted in the
message, with the address field focused.

While files are being sent, Cancel upload aborts the request and keeps the
selection. If the service had already accepted the job, it continues and appears
in History. While original items of a running job wait for a conversion slot,
Stop remaining calls the service's cancel route: waiting items end as stopped
(and can be retried), items already converting finish. On narrow screens the
results panel is scrolled into view after a job starts or a saved job is opened.

Each row states its kind and status in words (queued, converting, done, failed,
skipped). An image skipped because no text was extracted says to convert it again
with Local OCR or LLM enhancement. The unsupported-format message keeps its list
of accepted extensions folded under Supported formats. Transport failures of a
URL (`error sending request`, `HTTP 404`, timeouts) are prefixed with "Could not
fetch this page". Enhance is offered only while the service reports a routable
model; otherwise the action is omitted rather than shown disabled. Opening a
result moves keyboard focus to its title; a result with a single Markdown version
has no version chooser.

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

These two preferences are the only values this application writes to
`localStorage` (`markitai.lang`, `markitai.theme`). A small classic script,
`/ui/boot.js`, applies them in the document head before first paint; it runs
under the same `script-src 'self'` policy as the module scripts. Messages that
come from the service, such as conversion errors and provider probe details, are
shown as the service wrote them, inside localized context where the page knows
the meaning. The access-token control is shown only when this tab uses a token
or the service has answered 401.

## Connections and concurrent editing

The Connections view lists configured and available providers, saved deployments
and detected session models. It can discover models, select several at once,
manually enter identifiers, append an atomic batch, edit group/model/weight,
perform a real transient model probe, and delete deployments or connections.
Unsupported discovery still leaves manual entry available.

Collection views contain only the server's redacted fields. The raw credentials
endpoint is called only when a connection is explicitly opened for editing.
API key and base URL each have Keep, Replace and Clear choices, preserving the
server's omitted/null semantics. Password controls hold values in memory; provider
credentials are never written to sessionStorage/localStorage by this application.
Credentials are cleared when a draft is reset or successfully saved.

Mutations include the revision captured by that draft. A conflict refreshes the
visible connection/model lists but retains the draft and old revision. The user
must explicitly acknowledge the current revision after review and submit again;
there is no automatic overwrite/retry. Deployment dialogs use the same behavior.
A server-side session override or other write restriction remains visible as an
error; the page does not pretend it saved a blocked configuration.

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
offline error, URL-line parsing, duplicate files and the copy fallback.
`web/i18n.test.mjs` checks that both languages define the same keys and
placeholders, that every key the page and scripts use exists, language
detection, plural forms and the theme cycle; `web/result-tools.test.mjs` also
checks Chinese comparison and print messages. Run them with
`node --test crates/markitai-cli/src/server/web/*.test.mjs`; Node is only a
development test tool. Syntax checks use `node --input-type=module --check` with
each authored JS file on stdin. The embedded resources are `index.html`,
`style.css`, `icon.svg` (the tab icon), `boot.js`, `app.js`, `api.js`, `i18n.js`,
`preview.js`, `result-tools.js`, `settings.js` and the two vendored libraries.

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
