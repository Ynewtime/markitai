# Browser workbench

`markitai serve` includes a browser workbench at `/` (home) and `/jobs` (the
workspace). It follows the reference project's web interface: the same layout,
interaction model and visual language, with the Rust service's extra
capabilities placed in that vocabulary. Every file it needs is compiled into the
binary; running it needs no Node runtime, CDN or external font, and the page
makes no request to another origin.

## Layout and interaction

The **home view** is one centered column: a one-line headline, one sentence, a
tool row (Options, CLI, Upload, Folder) and the source card. The source card
holds the URL line (Enter converts, Shift+Enter starts a new line, a pasted list
grows the field up to six rows), the options drawer and the CLI command line.
Under it, one stack of monospaced lines reports, in this order: the upload in
progress (percentage and bytes, with Cancel), a session that could not be
restored (with Retry restore), a refusal from the service, an unreachable
service, a rejected input, a job that could not be saved, a folder notice, and
"LLM not configured · Configure LLM to enable enhancement" while no model is
routable. Files dropped anywhere on the page, or chosen with Upload or Folder,
start a job at once; a hairline veil with "Drop to convert" shows while files are
dragged over the page, and drops are ignored while a dialog is open.

The **workspace view** (`/jobs`, kept across reloads and Back) shows
"Conversions" with the session counters (`Current session · 6/8 Done · 1 Skipped
· 1 Failed`), the same tools beside them, and the actions Stop remaining (only
while original items wait for a conversion slot), Retry all failed (N) and Clear
all (Clear completed while something runs). Below the compact source card is one
ledger for the session's jobs and the service's saved jobs, ordered by latest
activity with running jobs first and numbered 01, 02, … in that order. Columns
are Name, Duration (a running row counts live), Finished, LLM / Cost (Base, or
LLM with its cost) and Status (✓, ×, a warning mark for skipped rows, a spinner,
Queued). Row actions are Download .md, Enhance with LLM (disabled with the reason
until LLM enhancement is available and switched on), Retry and Delete, which asks
for confirmation in a card anchored to the row. A failed row shows its cause in
one red line; clicking the row unfolds the service's original message. Rows with
conversion warnings add an amber line. Above ten rows, a name filter and
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
configured models (`routing group · model · Routing weight 1 · Test | Edit |
Delete`), models detected for this session (Save to config) and the
configuration source (clicking the path asks the service to open it). Add models
shows provider cards grouped as configured environment credentials, saved
providers (with their model count, Edit and Delete), common providers (the
reference's eight) and "More OpenAI-compatible providers": the sixteen further
prefixes of the routing table ([LLM providers](llm.md#openai-compatible-prefixes)),
each card naming its documented host (vLLM: "Server address required"). A
provider's page shows the documented endpoint and key variable of a built-in
provider and asks for what it needs: an API key (marked optional for LM Studio and
vLLM, absent for Ollama; its placeholder names the provider's variable, such as
`env:GROQ_API_KEY`), a base URL for Azure, OpenAI-compatible endpoints and vLLM,
an optional custom base (prefilled hint: the documented endpoint). It loads the
catalogue through the provider's `/models` (automatically for Ollama and LM
Studio) and opens the model picker: search, Select visible, Vision and Configured
badges, a manual model ID (also when the catalogue cannot be loaded), routing
group and weight, `N selected · 50 max`. Perplexity, Z.ai and Fireworks AI
document no OpenAI-compatible model list, so their page opens the manual entry
directly. A hand-typed ID gets the page's prefix unless it already has it, so an
ID with slashes of its own (`meta-llama/…` on Together AI, `google/…` on
OpenRouter) still routes to that provider. Test turns the button into ✓ or a
warning and reports in a notification.

The header holds the brand and version, Docs and GitHub (a footer band on
phones) and three icon buttons: Appearance (language EN/中 and theme
Auto/Light/Dark), Conversions (the workspace) and Settings. Notifications appear
at the top right (bottom on phones) and stack: an image skipped for lack of text
offers Enable OCR and retry, which also leaves OCR switched on for new jobs; an
item that asked for LLM processing without a model offers Retry without LLM; the
service becoming unreachable or reachable again, a failed model test and a
failed archive download are reported the same way.

## Behaviour kept from the earlier Rust page

- **Uploads** use XMLHttpRequest so the page can show the percentage and bytes
  sent, redrawn at most ten times a second, and abort them. Files over 100 MiB are
  named and refused before anything is sent. A selection or a dropped folder is
  capped at the service's item limit (`limits.max_job_items`). Hidden and system
  files inside a folder (dot names, `Thumbs.db`, `desktop.ini`) are skipped and
  counted; unreadable entries are counted.
- **Stop remaining** calls `POST /api/jobs/{id}/cancel` for every running job
  with waiting original items; stopped items show "Stopped before conversion ·
  retry to convert it" and can be retried. A 409 means nothing waited any more.
- **Retry** of an image skipped for lack of text resubmits the item's job options
  with OCR on. An unsupported file type offers no Retry: converting it again
  cannot help, and Retry all failed leaves such rows out.
- **Errors** are worded from the service's stable codes in the interface
  language: an API error's `reason`, then a settings conflict's `detail.code`,
  then the status-derived `code`; a failed item's `error_code` or a recognizable
  message shape (HTTP statuses, refused connections, timeouts, model refusals,
  missing OCR, size limits). The service's own wording stays available as the
  line's tooltip or unfolded under the row. Provider probe and discovery phrases
  are translated the same way. A job whose history could not be saved
  (`persistence_error`) shows a red line.
- **Offline**: a request that cannot reach the service shows an error line and a
  notification; the page asks again every five seconds and says when the
  service is back, then refreshes capabilities, history and the session. A broken
  event stream triggers the same check and, once closed, reconciles the job from
  its snapshot.
- **Settings conflicts**: every write carries the revision its draft started
  from. A `stale_revision` or `config_changed` refusal reloads the lists but keeps
  the draft and its revision; "Use current revision" adopts the new one and the
  next submit uses it. Other refusals (for example `settings_read_only`) stay
  plain errors. Editing a saved provider shows its stored references; unchanged
  fields are not sent, a field emptied on purpose is sent as `null` (cleared).
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
  refusal is reported; without one (a loopback visitor) the link works as is.
- Download all (.zip), which can be larger than memory, is left to the browser:
  with a token the page first asks `POST /api/download-tickets` for a
  single-use ticket and opens the returned `?ticket=` URL, valid for one GET of
  that archive within a minute (see [serve](serve.md#network-and-file-boundaries)),
  so what stays in the download list is a spent ticket.

A 401 shows "Not authorized · reload the page with the access token link · Enter
token"; Enter token opens a small dialog that stores a typed token for the tab.
Loopback visitors need no token.

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
submission, never on load. No credential is stored by the page.

## Differences from the reference

- Cloudflare URL rendering and Cloudflare file conversion are shown but disabled
  ("not supported by this server yet"); the Rust core returns explicit errors for
  them. The fetch strategy help describes the Rust behaviour (Auto: a direct
  request, then the local browser).
- A request that asks for LLM processing while no model is routable is refused by
  the service (422 `llm_unavailable`) instead of silently converting without the
  model; the page then reloads the capabilities.
- The Folder tool, Stop remaining, upload progress and cancellation, the Files
  tab, the Base | LLM switch, the token dialog, offline detection, the pricing
  coverage in the cost cell's tooltip (`$0.012300 · all recorded requests
  priced`, `Price unknown · …`) and the amber line for a failed last attempt are
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

## Architecture and build

The source lives in `crates/markitai-cli/src/server/web/`: Preact 10 components
in TypeScript (`src/components/`), state hooks (`src/hooks/`), pure modules with
their tests (`src/lib/`, `src/api/`, `src/i18n/`), one stylesheet
(`src/styles/app.css`) and the static shell (`public/index.html`, `boot.js`,
`logo.svg`). `build.mjs` bundles and minifies with esbuild into `dist/`, which is
committed: `app.js`, `app.css`, their gzip twins, gzip copies of the vendored
marked and DOMPurify modules, the shell files and `manifest.json` (sizes,
SHA-256, tool versions and a digest of the sources). `cargo build` embeds `dist/`
with `include_bytes!` and never runs Node.

```sh
cd crates/markitai-cli/src/server/web
npm ci                      # esbuild 0.28.2, preact 10.29.8, typescript 6.0.3 (exact pins)
node build.mjs              # rewrite dist/; commit it with the sources
node build.mjs --check      # fail when dist/ differs from a fresh build
node --test "src/**/*.test.ts"
node scripts/check-css-scale.mjs
node scripts/check-contrast.mjs
npx tsc --noEmit
```

The stylesheet uses one type ladder (10/11/12/14/16/18/20/24 px plus the 30–44 px
headline), the radii 6/8/14/16/999, the durations 120/140/160 ms (spinner loops
700/1500 ms) and two shadows; `scripts/check-css-scale.mjs` fails on any other
value and checks the reset. There is one phone tier at 780px (with a further
step at 400px); at 375px the page has no horizontal scroll. Positions of
popovers are set through the CSSOM, never through `style` attributes, which the
policy (`style-src 'self'`) would block.

The server serves fixed routes only: `/` and `/jobs` (the shell), `/ui/app.js`,
`/ui/app.css`, `/ui/boot.js`, `/ui/logo.svg`, `/ui/inter-latin-wght.woff2`,
`/ui/marked.js` and `/ui/purify.js`. Compressed files are sent with
`Content-Encoding: gzip` and `Vary: accept-encoding` to clients that accept gzip
and decompressed once for others. Every response carries `Cache-Control:
no-cache`, a weak content ETag (a matching `If-None-Match` answers 304),
`nosniff`, `no-referrer` and the unchanged Content Security Policy. Unknown paths,
including other `/ui/` names, keep JSON 404 errors.

Third-party files: Preact 10.29.8 (MIT) is bundled into `app.js`; the Inter
variable font, Latin subset of `@fontsource-variable/inter` 5.3.0 (OFL-1.1), is
served as is; the path data of 26 Phosphor icons from `@phosphor-icons/react`
2.1.10 (MIT) is in `src/components/icon-paths.ts`. Their licences are in
`vendor/web/` (`preact-LICENSE`, `Inter-OFL.txt`, `phosphor-LICENSE`) and their
sources, package integrity values and file hashes in `vendor/web/provenance.json`,
beside marked and DOMPurify.

## Colour contrast

Every text colour the page draws, on every background it is drawn on, meets
WCAG 2.2 AA in both palettes: 4.5:1 for text (placeholders, tooltips and the
phone tier's `·` separators included) and 3:1 for large text and for icons and
glyphs that carry meaning. The pairs were found by auditing the rendered page in
headless Chrome 154 on macOS against a debug build of this binary: home with
the options drawer and CLI line, the workspace with session rows (done, failed,
skipped, a warning notification) and saved jobs, the delete confirmation, every
preview tab (with a base/LLM pair from a loopback mock model), PDF settings,
appearance, and the settings levels (models, a model test, provider cards, a
provider page, the model picker), each in light and dark, once as drawn and once
with every `:hover` rule applied, at 1440 and 375 px. For each element with its
own text, each `::before`/`::after` with content, each placeholder and each
icon, the audit composited the backgrounds behind it and computed the ratio.

Two pairs were below the minimum and are fixed. `.diff-no` line numbers used
`--text-4` (light 2.56:1, 2.27–2.42:1 on the diff tints; dark 3.67:1, 2.68–2.97:1
on the tints) and now use `--text-3`. Light `--text-3` on `--strip` (provider
card and header link hover) was 4.40:1, and on the removed-line tint 4.27:1, so
light `--text-3` is now `#6b6b73`, zinc-500 one step deeper (the print palette
follows). The phone tier's fact separators moved from `--text-4` to `--text-3`.
Disabled controls (`--text-4`, reduced opacity) and the logo are exempt (WCAG
1.4.3) and are the only colours below the minimum.

`node scripts/check-contrast.mjs` recomputes these pairs from the token blocks of
`app.css` and fails when one drops below its minimum or when the automatic and
the explicit dark palettes differ; `--table` prints the table below (ratio, then
the composited text and background colours). A new colour pair needs a row there.

| Text | Background | Minimum | Light | Dark | Where |
|---|---|---|---|---|---|
| `--text-1` | `--bg` | 4.5:1 | 16.97 (#18181b on #fafafa) | 19.06 (#fafafa on #09090b) | page text, headline, ledger names |
| `--text-1` | `--surface` | 4.5:1 | 17.72 (#18181b on #ffffff) | 16.97 (#fafafa on #18181b) | cards, dialogs, rendered preview |
| `--text-1` | `--strip` | 4.5:1 | 16.12 (#18181b on #f4f4f5) | 16.97 (#fafafa on #18181b) | hovered tools and menu items, selected chips |
| `--text-1` | `--row-hover` | 4.5:1 | 16.59 (#18181b on #f7f7f8) | 18.24 (#fafafa on #101012) | hovered ledger rows |
| `--text-1` | `color-mix(in srgb, --accent 10%, --surface)` | 4.5:1 | 14.45 (#18181b on #e8e8e8) | 12.86 (#fafafa on #2f2f31) | selected segment and chip |
| `--text-2` | `--bg` | 4.5:1 | 7.41 (#52525b on #fafafa) | 7.76 (#a1a1aa on #09090b) | secondary text on the page |
| `--text-2` | `--surface` | 4.5:1 | 7.73 (#52525b on #ffffff) | 6.91 (#a1a1aa on #18181b) | secondary text in cards and dialogs |
| `--text-2` | `--row-hover` | 4.5:1 | 7.24 (#52525b on #f7f7f8) | 7.43 (#a1a1aa on #101012) | secondary text in hovered rows |
| `--text-2` | `color-mix(in srgb, --accent 7%, transparent)` over `--surface` | 4.5:1 | 6.71 (#52525b on #efefef) | 5.75 (#a1a1aa on #28282b) | Convert button |
| `--text-3` | `--bg` | 4.5:1 | 5.06 (#6b6b73 on #fafafa) | 7.76 (#a1a1aa on #09090b) | ledger facts, counters, separators |
| `--text-3` | `--surface` | 4.5:1 | 5.28 (#6b6b73 on #ffffff) | 6.91 (#a1a1aa on #18181b) | hints, metadata, diff line numbers |
| `--text-3` | `--strip` | 4.5:1 | 4.81 (#6b6b73 on #f4f4f5) | 6.91 (#a1a1aa on #18181b) | provider card and link hover (was 4.40:1) |
| `--text-3` | `--row-hover` | 4.5:1 | 4.94 (#6b6b73 on #f7f7f8) | 7.43 (#a1a1aa on #101012) | facts in hovered rows |
| `--text-3` | `--diff-add-bg` | 4.5:1 | 5.00 (#6b6b73 on #ecfdf2) | 5.06 (#a1a1aa on #2c3331) | line numbers of added lines (was --text-4) |
| `--text-3` | `--diff-del-bg` | 4.5:1 | 4.67 (#6b6b73 on #feeded) | 5.61 (#a1a1aa on #31282a) | line numbers and marks of removed lines |
| `--text-1` | `--diff-add-bg` | 4.5:1 | 16.76 (#18181b on #ecfdf2) | 12.43 (#fafafa on #2c3331) | added lines |
| `--text-1` | `--diff-del-bg` | 4.5:1 | 15.66 (#18181b on #feeded) | 13.77 (#fafafa on #31282a) | removed lines |
| `--accent-contrast` | `--accent` | 4.5:1 | 17.72 (#ffffff on #18181b) | 16.97 (#18181b on #fafafa) | primary buttons |
| `--accent-contrast` | `--accent-hover` | 4.5:1 | 14.89 (#ffffff on #27272a) | 13.96 (#18181b on #e4e4e7) | hovered primary buttons |
| `--selection-fg` | `--selection-bg` | 4.5:1 | 16.97 (#fafafa on #18181b) | 16.97 (#18181b on #fafafa) | text selection |
| `--danger-on-fill` | `--danger-fill` | 4.5:1 | 6.47 (#ffffff on #b91c1c) | 4.71 (#18181b on #ef4444) | Delete in a confirmation card |
| `--danger-on-fill` | `--danger-fill-hover` | 4.5:1 | 8.31 (#ffffff on #991b1b) | 6.40 (#18181b on #f87171) | hovered Delete |
| `--err` | `--bg` | 4.5:1 | 6.20 (#b91c1c on #fafafa) | 10.48 (#fca5a5 on #09090b) | error lines on the page |
| `--err` | `--surface` | 4.5:1 | 6.47 (#b91c1c on #ffffff) | 9.33 (#fca5a5 on #18181b) | error lines in cards and dialogs |
| `--err` | `--row-hover` | 4.5:1 | 6.06 (#b91c1c on #f7f7f8) | 10.03 (#fca5a5 on #101012) | failed-row cause in a hovered row |
| `--warning` | `--bg` | 4.5:1 | 4.81 (#b45309 on #fafafa) | 11.92 (#fbbf24 on #09090b) | conversion warnings in the ledger |
| `--warning` | `--surface` | 4.5:1 | 5.02 (#b45309 on #ffffff) | 10.61 (#fbbf24 on #18181b) | warnings in the preview |
| `--warning` | `--row-hover` | 4.5:1 | 4.70 (#b45309 on #f7f7f8) | 11.40 (#fbbf24 on #101012) | warnings in hovered rows |
| `--success` | `--bg` | 4.5:1 | 4.81 (#15803d on #fafafa) | 14.17 (#86efac on #09090b) | done mark, LLM tag |
| `--success` | `--surface` | 4.5:1 | 5.02 (#15803d on #ffffff) | 12.62 (#86efac on #18181b) | test passed, success notices |
| `--success` | `--row-hover` | 4.5:1 | 4.70 (#15803d on #f7f7f8) | 13.56 (#86efac on #101012) | done mark and LLM tag in hovered rows |
| `--err` | `--strip` | 3:1 | 5.89 (#b91c1c on #f4f4f5) | 9.33 (#fca5a5 on #18181b) | hovered delete icon |
| `--err` | `color-mix(in srgb, --err 10%, --surface)` | 3:1 | 5.46 (#b91c1c on #f8e8e8) | 7.74 (#fca5a5 on #2f2629) | confirmation card icon |
| `--text-1` | `color-mix(in srgb, --text-1 6%, transparent)` over `--surface` | 4.5:1 | 15.71 (#18181b on #f1f1f1) | 14.54 (#fafafa on #262628) | hovered text buttons (Test, Edit) |
| `--text-1` | `color-mix(in srgb, --warning 13%, --surface)` | 4.5:1 | 14.82 (#18181b on #f5e9df) | 12.92 (#fafafa on #362e1c) | hovered notice action (warning) |
| `--text-1` | `color-mix(in srgb, --err 13%, --surface)` | 4.5:1 | 14.20 (#18181b on #f6e1e1) | 13.16 (#fafafa on #362a2d) | hovered notice action (error) |
| `color-mix(in srgb, --text-1 70%, transparent)` | `--bg` | 4.5:1 | 6.41 (#5c5c5e on #fafafa) | 9.36 (#b2b2b2 on #09090b) | hovered brand name (70% opacity) |
| `--t-text` | `--t-bg` | 4.5:1 | 16.97 (#fafafa on #18181b) | 16.97 (#fafafa on #18181b) | CLI line, Source card |
| `--t-dim` | `--t-bg` | 4.5:1 | 6.91 (#a1a1aa on #18181b) | 6.91 (#a1a1aa on #18181b) | dimmed front matter, CLI prompt |
| `--t-badge-text` | `--t-bg` | 4.5:1 | 11.99 (#d4d4d8 on #18181b) | 11.99 (#d4d4d8 on #18181b) | Copy pill, status pills, active filter chip |
| `--t-green` | `--t-bg` | 4.5:1 | 14.62 (#bbf7d0 on #18181b) | 14.62 (#bbf7d0 on #18181b) | status pills: done |
| `--t-red` | `--t-bg` | 4.5:1 | 9.33 (#fca5a5 on #18181b) | 9.33 (#fca5a5 on #18181b) | status pills: failed |
| `--t-amber` | `--t-bg` | 4.5:1 | 12.29 (#fcd34d on #18181b) | 12.29 (#fcd34d on #18181b) | status pills: skipped |

## Validation

- `crates/markitai-cli/tests/serve_web.rs` runs the binary: both shell
  addresses, the exact compressed and plain bytes of every asset, 304
  revalidation, HEAD, the exact policy header, and JSON 404 for unknown,
  traversal and old paths.
- Module tests in `server/web.rs` check gzip negotiation, ETag comparison, that
  every compressed asset decodes, and that the field names of `ItemPayload`,
  `JobSnapshot` and `HistoryEntry` in `src/api/types.ts` match what the service
  serializes.
- `node --test` runs 51 tests: token handling and URL restrictions (no token in
  any URL; download tickets checked for path, origin and parameters), the
  fetch-based event stream (parsing at every chunk boundary and line ending,
  header token, reconnection after a dropped connection, a refused stream as
  final, close), archive downloads by ticket and other files as Blobs, rendered
  images waiting for a header fetch, their bounded loading and release, manual
  model IDs, the XMLHttpRequest upload (progress, token, cancellation, refusal,
  network and redirect failures), the redraw throttle, request policy, both
  dictionaries,
  that every `reason` in the Rust service sources and every core error code has
  localized text, that the English phrases recognized are still written by the
  Rust sources, option resolution and storage, the CLI line, URL parsing,
  formatting, artifact path resolution, the base/LLM pair, the line comparison,
  folder walking, the ledger model (merging, ordering, filters, retry rules,
  stop eligibility, session seeds), cost labels, printing, copying and boot.js.
- `node scripts/test_ui_pricing.cjs` checks the cost labels (17 checks).
- `scripts/test_web_dist.py` (part of the Python gate) verifies `dist/` against
  its manifest and the sources' digest without Node, the compressed twins, the
  vendored files against their provenance, and with Node installed rebuilds the
  bundle and runs the workbench tests, the CSS check and the contrast check.

Browser behaviour (layout at 1440 and 375 px, both themes and languages, real
conversions, previews, settings flows) is verified by hand in a browser; these
tests do not establish it.
