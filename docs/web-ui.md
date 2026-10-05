# Browser workbench

`markitai serve` includes a browser workbench at `/` (home) and `/jobs` (the
workspace). It follows the reference project's web interface: the same layout,
interaction model and visual language, with the Rust service's extra
capabilities placed in that vocabulary. Every file it needs is compiled into the
binary; running it needs no Node runtime, CDN or external font, and the page
makes no request to another origin.

## Layout and interaction

The **home view** is one centered column: a one-line headline, one sentence and
the source card. The source card holds the URL line (Enter converts, Shift+Enter
starts a new line, a pasted list grows the field up to six rows), a bottom row
with Options and Upload (a menu for files or a folder) on the left and Convert
on the right, and the options drawer, which ends with the equivalent CLI command
line. Each option row's ⓘ explains the row and lists its choices; the choices
themselves show no hover tooltip.
Under it, one stack of monospaced lines reports, in this order: the upload in
progress (percentage and bytes, with Cancel), a session that could not be
restored (with Retry restore), a refusal from the service, an unreachable
service, a rejected input, a job that could not be saved, a folder notice, and
"LLM not configured · Configure LLM to enable enhancement" while no model is
routable. Files dropped anywhere on the page, or chosen with Upload or its folder choice,
start a job at once, except when the selected Cloudflare service needs the
confirmation described below; a hairline veil with "Drop to convert" shows while
files are dragged over the page, and drops are ignored while a dialog is open.

The **workspace view** (`/jobs`, kept across reloads and Back) shows
"Conversions" with the session counters (`Current session · 6/8 Done · 1 Skipped
· 1 Failed`) and the actions Stop remaining (only
while original items wait for a conversion slot), Retry all failed (N) and Clear
all (Clear completed while something runs). Below the compact source card is one
ledger for the session's jobs and the service's saved jobs, ordered by latest
activity with running jobs first and numbered 01, 02, … in that order. Columns
are Name, Duration (a running row counts live), Finished, LLM / Cost (Base, or
LLM with its cost) and Status (✓, ×, a warning mark for skipped rows, a spinner,
Queued). Row actions are Download .md, Enhance with LLM (disabled with the reason
until LLM enhancement is available and switched on), Retry and Delete, which asks
for confirmation in a card anchored to the row. Failed operations and conversion
warnings use the shared notification card at the top right (docked at the bottom
on phones). Clicking the status icon, or pressing Enter or Space on it, reopens
the complete notice; clicking the row still opens the document preview. A failed
enhancement that retains an earlier result keeps its download and retry available.
Warnings from that result are labelled separately from the latest attempt's
recorded cost. Above ten rows, a name filter and
All/Done/Failed/Skipped chips appear. A Total row closes a ledger without saved
jobs, and Download all (.zip) sits under the ledger.

Saved jobs use the same row language: names (`a.pdf, b.docx +3 more`), a CLI tag
for jobs recorded by the command line, a status mark for one item or a dark pill
such as `4 Done · 1 Skipped` for several. Clicking a saved job of one item opens
its preview; a saved job of several items is brought into the session ledger so
each of its items can be previewed, retried or deleted (an extension; the
reference previews only the first result). Retry and Enhance on a saved
single-item job also bring it into the session.

The **preview** is a 1120px dialog (a full-screen sheet on phones): "Preview
mode", the file name (a URL links to its page), the conversion warnings, then a
panel with the tabs Rendered, Source, Diff (only for an exact base/LLM pair) and
Files. The bar on the right shows words and bytes, a Base | LLM switch for a
paired result, PDF settings (custom header and footer), Export PDF and Download
.md. Rendered shows the document without its YAML front matter; Source is an
always-dark terminal card with the front matter dimmed and a Copy pill; Diff
lists both line numbers with added and removed lines tinted; Files lists the
result's files with sizes and downloads.

**Settings** is a 760px dialog with a breadcrumb (`Settings / Add models /
OpenAI`); Escape steps back one level before it closes. The first level lists
configured models (`routing group · model · Routing weight 1` with icon buttons
for Test, Edit and Delete), models detected for this session (Save to config)
and the configuration source (clicking the path asks the service to open it).
Edit opens the model's own page (`Settings / OpenAI`): the provider's Connection
section above the routing fields (routing group, model, routing weight). Add
models shows provider cards grouped as configured environment credentials, saved
providers (with their model count, Edit and Delete; Edit opens the provider's
page), common providers (the
reference's eight) and "More OpenAI-compatible providers": the sixteen further
prefixes of the routing table ([LLM providers](llm.md#openai-compatible-prefixes)),
each card naming its documented host (vLLM: "Server address required"). A
provider's page shows the documented endpoint and key variable of a built-in
provider and asks for what it needs: an API key (marked optional for LM Studio and
vLLM, absent for Ollama), a base URL for Azure, OpenAI-compatible endpoints and vLLM,
an optional custom base (prefilled hint: the documented endpoint). It loads the
catalogue through the provider's `/models` (automatically for Ollama and LM
Studio) and opens the model picker: search, Select visible, Vision and Configured
badges, a manual model ID (also when the catalogue cannot be loaded), routing
group and weight, `N selected`; the list scrolls after about four models, and the
selection count with Add and Cancel stays pinned at the bottom of the dialog.
A saved provider's page starts with a Connection section that manages the key
and the address separately: the key shows as saved (with its last four
characters), from an environment variable or not set, with Replace and Remove; a
key the service finds in the environment (a process variable or a dotenv file)
is reported as from that variable, with Add storing a literal key instead; the
address shows as custom, from an environment variable or the default, with
Edit and Reset to default. Perplexity, Z.ai and Fireworks AI
document no OpenAI-compatible model list, so their page opens the manual entry
directly. A hand-typed ID gets the page's prefix unless it already has it, so an
ID with slashes of its own (`meta-llama/…` on Together AI, `google/…` on
OpenRouter) still routes to that provider. Test turns the button into ✓ or a
warning and reports in a notification. API key fields accept literal keys;
`env:` references belong in the server configuration, not browser requests.
Environment and saved-provider cards use server-issued connection IDs, allowing
model discovery without returning the key to the browser. These credentials stay
bound to their configured endpoint: the address form therefore also asks for
the key to use with the new address (required while a key is saved; left blank,
the provider connects without one).

The header holds the brand and version, Docs and GitHub (a footer band on
phones) and two icon buttons: Appearance (language EN/中 and theme
Auto/Light/Dark) and Settings. The conversion tasks (the workspace) open from
the entry under the composer on the home view. Notifications appear
at the top right (bottom on phones) and stack: an image skipped for lack of text
offers Enable OCR and retry, which also leaves OCR switched on for new jobs; an
item that asked for LLM processing without a model offers Retry without LLM; the
service becoming unreachable or reachable again, a failed model test and a
failed archive download are reported the same way.

## Job controls

- **Uploads** use XMLHttpRequest so the page can show the percentage and bytes
  sent, redrawn at most ten times a second, and abort them. Files over 100 MiB are
  named and refused before anything is sent. A selection or a dropped folder is
  capped at the service's item limit (`limits.max_job_items`). Hidden and system
  files inside a folder (dot names, `Thumbs.db`, `desktop.ini`) are skipped and
  counted; unreadable entries are counted.
- **Stop remaining** calls `POST /api/jobs/{id}/cancel` for every running job
  with waiting original items; stopped items show "Stopped before conversion ·
  retry to convert it" and can be retried. A 409 means nothing waited any more.
- **Retry** of an image skipped for lack of text resubmits the item's saved
  options with OCR on. An unsupported file type offers no Retry: converting it again
  cannot help, and Retry all failed leaves such rows out.
- **Errors** are worded from the service's stable codes in the interface
  language: an API error's `reason`, then a settings conflict's `detail.code`,
  then the status-derived `code`; a failed item's `error_code` or a recognizable
  message shape (HTTP statuses, refused connections, timeouts, model refusals,
  missing OCR, size limits). The service's full wording stays available in the
  notification's expandable details, and every warning remains readable in its
  scrollable list. The status icon reopens a dismissed notice. Provider probe and
  discovery phrases are translated the same way. A job whose history could not
  be saved (`persistence_error`) reports the failure through the same notice.
- **Offline**: a request that cannot reach the service shows an error line and a
  notification; the page asks again every five seconds and says when the
  service is back, then refreshes capabilities, history and the session. A broken
  event stream triggers the same check and, once closed, reconciles the job from
  its snapshot.
- **Settings conflicts**: every write carries the revision its draft started
  from. A `stale_revision` or `config_changed` refusal reloads the lists but keeps
  the draft and its revision; "Use current revision" adopts the new one and the
  next submit uses it. Other refusals (for example `settings_read_only`) stay
  plain errors. A saved provider's key is never placed in an input; Replace,
  Remove and the address form each send only the field they change. An
  endpoint change with a retained server key is rejected, including when
  linked model deployments have their own credential overrides.
- **Session restore**: the rows of jobs created in this tab are kept in
  `sessionStorage` (`markitai.session`); after a reload each job is asked for
  again. A job the service no longer knows is dropped; an unreachable service
  keeps the rows with Retry restore.

## Preview safety and printing

Markdown is rendered with the vendored [marked 18.0.14](https://github.com/markedjs/marked/releases/tag/v18.0.14)
and sanitized with [DOMPurify 3.4.16](https://github.com/cure53/DOMPurify/releases/tag/3.4.16),
both loaded from `/ui/marked.js` and `/ui/purify.js` the first time a preview
opens. The sanitized fragment never contains scripts, styles, forms, frames or
embedded media. Images and attachment links must resolve to an exact entry of the
result's own artifact list; only raster images are shown, and anything else (an
external or SVG image) becomes a labelled placeholder, a deliberate difference
from the reference, which loads external images. Other links keep only
http(s)/mailto targets and open in a new tab without referrer. Syntax highlighting
of code blocks is not included.

Export PDF prints a temporary clone of the rendered document through the
browser's print dialog, on A4 with the reference's print styles (light palette,
repeated table headers, wrapped code, the optional Markitai header and
`Prepared with Markitai · Source: …` footer). Links are removed from the clone, so
a generated PDF embeds no service URL, and printing waits up to ten seconds for
every image and refuses rather than produce an incomplete document. A direct
browser print of the page also hides any link that carries `token=` (none is
produced any more; the rule stays as a guard).

## Access token

By default, all API requests require the access token, including requests from
this computer. Open the link printed by `markitai serve`, rather than entering
only the host and port. Static page assets can load without a token, but jobs,
settings and results remain protected.

The launch link carries the token in its fragment (`#token=`; `?token=` is
accepted). The page removes it from the address bar at once and keeps it for the
tab in `sessionStorage` (`markitai.service-token`), with a memory fallback. No
URL the page requests or shows carries it:

- Every request sends `Authorization: Bearer`, the job event streams included:
  they are read with `fetch` and a server-sent-events parser (`src/api/events.ts`)
  instead of EventSource, which cannot send headers. The reader keeps
  EventSource's behaviour: named events, a dropped connection retried after the
  stream's `retry:` delay (3 s by default) while a refused one (an HTTP error or
  another content type) is final, and no event after `close()`.
- With a token, preview images are fetched with the header (four at a time) and
  shown through `blob:` object URLs, which the policy already allows
  (`img-src 'self' blob:`) and which are released when the preview changes;
  Export PDF waits for them. An image that cannot be fetched becomes the
  labelled placeholder.
- Download links (rows, Files, Download .md, file links inside the rendered
  document) point at the plain `/api/` path. With a token, a click or a
  middle-click fetches the file with the header and saves it as a Blob, so a
  refusal is reported. Plain download links work without a token only when the
  server was explicitly started with `--no-auth`.
- Download all (.zip), which can be larger than memory, is left to the browser:
  with a token the page first asks `POST /api/download-tickets` for a
  single-use ticket and opens the returned `?ticket=` URL, valid for one GET of
  that archive within a minute (see [serve](serve.md#network-and-file-boundaries)),
  so what stays in the download list is a spent ticket.

A 401 shows "Not authorized · reload the page with the access token link · Enter
token"; Enter token opens a small dialog that stores a typed token for the tab.
After a server restart that generates a new token, reopen its new launch link
or use Enter token. `--no-auth` explicitly disables API token enforcement; only
direct loopback clients without forwarding headers receive the additional trust
needed for settings and URL jobs. A loopback connection through a reverse proxy
is not an authentication mechanism. For mutations, an Origin must match the
request Host's hostname and effective port, or an operator-configured
`--allowed-host`; another localhost port is not automatically allowed. See
[serve boundaries](serve.md#network-and-file-boundaries).

## Language, appearance and stored values

English and Chinese follow the reference's product copy. Without a stored
choice, a browser whose first language starts with `zh` gets Chinese. The
classic script `/ui/boot.js` applies the stored theme and language before first
paint (the Content Security Policy allows no inline script). Values kept in this
browser's `localStorage`: `markitai.lang`, `markitai.theme`, `markitai.options`
(preset, LLM, OCR, output profile and the image overrides; fetch strategy,
backend, cache and source choices are never remembered, so a revisit cannot
silently re-enable a remote service), `markitai.pdf.custom-header-footer` and
`markitai.notify-denied` (a refused desktop-notification permission, which is
then never asked again). A finished job raises a system notification only while
the tab is hidden and permission was granted; permission is asked on the first
submission, never on load. Provider API keys are not retained in browser storage;
the service access token is held for the tab as described above.

## Cloudflare processing

The URL strategy and file backend selectors offer Cloudflare when the service
reports its local configuration ready. That check makes no cloud request: it
does not verify the token's permissions, account quota or Cloudflare availability.
Configure credentials on the server as described under
[remote services](fetch.md#remote-services); the workbench neither accepts nor
receives Cloudflare tokens, account IDs or endpoint overrides.

Once confirmed, the Cloudflare URL strategy sends selected URLs to Browser
Rendering; its file backend allows supported uploaded files to use Workers AI
`toMarkdown`. The choices are independent. File eligibility is decided from the
content the core reads, not just its name; any filename-based estimate in the
confirmation is only an estimate. OCR or screenshot requests keep the native
file reader, and the page does not turn those choices off. Cloudflare's file
backend cannot be combined with the Jina or Defuddle URL strategy: the page
explains the conflict and waits for you to choose compatible options.

Before submission, a dialog explains the selected sources and asks for consent
for this request. Cloudflare may charge separately; its charges are not included
in the displayed LLM subtotal. If the selected sources do not use the chosen
Cloudflare route (for example only static URLs with the file backend selected),
the dialog says so instead of implying they will be sent. Cancel leaves the
sources and existing results in place. The service can still refuse a request
because of its remote-processing policy or configuration.

Retry, Enhance with LLM and Retry all failed each ask again when their effective
options select Cloudflare. They use each item's saved options, falling back to
the job's options only for older history; Enhance also reconverts the original
input. Neither history nor browser storage remembers permission. The API's
[request authorization](serve.md#cloudflare-request-authorization) is explicit
and applies equally to new jobs and reruns.

A Cloudflare notice on a row or history entry means an accepted attempt requested
that service. It can remain after a failed rerun preserves an earlier result or
a later native conversion succeeds. It does not establish that Cloudflare was
called, how many requests ran or what they cost; LLM cost and pricing coverage
continue to describe recorded model work only.

## Differences from the reference

- A request that asks for LLM processing while no model is routable is refused by
  the service (422 `llm_unavailable`) instead of silently converting without the
  model; the page then reloads the capabilities.
- Folder upload, Stop remaining, upload progress and cancellation, the Files
  tab, the Base | LLM switch, the token dialog, offline detection, the pricing
  coverage in the cost cell's tooltip (`$0.012300 · all recorded requests
  priced`, `Price unknown · …`) and the replayable failed-attempt notice are
  additions. Model discovery that fails still leaves manual model entry.
- Provider groups exist only as the service reports them (environment, saved,
  common, compatible); the reference's local-CLI and OAuth groups do not occur,
  and the compatible group (sixteen OpenAI-compatible prefixes) is an addition.
- Previews block external images (see above); the reference renders them.
- Event streams are read with `fetch` and images fetched as Blobs when a token is
  held; the reference puts no token in its page at all (it has no remote
  access token).
- `node scripts/check-contrast.mjs` and the slightly deeper light `--text-3`
  (`#6b6b73` for the reference's `#71717a`) and `.diff-no` colour are
  accessibility fixes the reference does not have.

## Building and checking the workbench

The workbench is embedded in the CLI. End users do not need Node; contributors
who change it run `npm ci && node build.mjs` in
`crates/markitai-cli/src/server/web` and commit the generated `dist/` files.
`npm test`, `npm run typecheck`, `npm run css` and `npm run contrast` check
specific source properties. Browser interaction, focus, mobile layout and
assistive-technology behavior require their own actual checks; passing a source
test is not proof of those interactions. See [development](development.md).
