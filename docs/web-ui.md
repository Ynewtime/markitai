# Browser workbench

`markitai serve` includes a browser workbench. It is the one frontend of both
markitai backends: this Rust server embeds it, and Python markitai 1.2.1 serves
the same build (see [one frontend, two backends](#one-frontend-two-backends)).
Every file it needs is compiled into the binary; running it needs no Node
runtime, CDN or external font, and the page makes no request to another origin.

## Layout and interaction

A **sidebar** frames every view: Convert and Conversions (with the number of
conversions and a pulsing dot while something converts), the Settings sections
(Models, Appearance and language, Access token), and Docs, GitHub and the
version at the bottom. Its button collapses it to a 60px icon rail; each entry
keeps its name for assistive technology and shows it as a tooltip on hover and
keyboard focus, and the choice is remembered in this browser. On phones the
sidebar gives way to a top bar and a bottom tab bar (Convert, Conversions,
Settings). Every view is an address (`/`, `/jobs`, `/settings/models`,
`/settings/appearance`, `/settings/access`), so reloads, Back and new tabs keep
it. Any other address opens the workbench (with status 404) on "Page not found",
with the way to Convert and Conversions. On a wide window every page is one
column centred between the sidebar and the window's edge (Convert 800 px, the
Settings pages 760 px, Conversions 1280 px); Conversions with the window docked
beside the list takes the whole width.

Controls share one scale: buttons 28 px with 13 px labels (36 px on a phone),
one filled button per area, secondary actions as outlined buttons and the rest
in "⋯"; every choice (status filters, provider groups, the options' segments,
Base | LLM, language and theme, the option toggles) is the same segmented group;
on a phone the short text actions (Full details, Show items, breadcrumbs) answer
taps across 44 px.

**Convert** (`/`) is a headline, one sentence and the composer: the URL field
(Enter converts, Shift+Enter starts a new line, a pasted list grows the field up
to six rows), Options, Upload (files) and Folder on the left, Convert on the
right, and the options drawer, which ends with the equivalent CLI command line.
The latest three submissions follow, each one click from its details. One file
starts a job at once; a folder, several files or a `.urls` list waits for
Convert as one staged line (`sample-folder · 43 files`) that can be removed,
and is sent together with any typed URLs as one job whose name is the folder or
list. A `.urls` list becomes one URL item per entry (`#` lines and blank lines
skipped, `https://example.com/report  Report` naming the output).

**Conversions** (`/jobs`) states what the list adds up to ("18 of 21 converted,
3 failed.", with Summary for the notices and failures of everything listed)
beside Stop remaining (while original items wait for a slot), New conversion
(to Convert) and "⋯" (Retry failed (N), Download all), then the filters: a name
filter and All / Failed / Notices with their counts. Below sits the list at the
page's width. Dropping files anywhere still converts them (a folder or several
files wait on Convert); typing URLs is Convert's job.

- The **list** shows every saved and running item, newest activity first,
  running first: a status glyph (one shape per state: ✓ done, × failed, a
  dashed minus for stopped, a warning sign for skipped, a spinner, a ring for
  waiting), the name with its extension always visible (a long name loses its
  middle, `blog-japanese-…ning.html`; a URL dims its host), a warning count, and
  the duration or the state in words. A submission of several items is a header
  (what was submitted: `report-materials`, `14 files`, `3 web pages`; the preset;
  the time) with its items under it, indented, so a submission of one item is one
  row at the outer edge. It is
  an ARIA tree with no controls inside rows: Up and Down move, Right and Left
  open and close a submission, Home and End jump, and a click, Enter or Space
  opens the conversion's window (docked beside the list, the window follows the
  arrows instead).
- The **peek window** opens over the dimmed list: previous and next with the
  position (`8 of 14`), the name, the actions that matter (Download; Retry, Add a
  model or Enable OCR and retry when that is the fix) and "⋯" (Enhance with LLM,
  Run again, Retry without LLM, Delete), then Details, Preview and Compare as
  tabs. ◫ docks it beside the list instead, and ⤢ opens it as a window again;
  the choice is remembered in this browser. Up and Down (outside the scrolling
  document) move between items, C switches Compare, Escape, ✕ or a click on the
  dimmed list closes it and returns to the row.
- **Details** shows the item: its state in a sentence
  ("Converted with LLM in 14.6 s · 12 images", or the failure in plain words), the
  facts (source and its size, output, what else was saved, models, cost,
  options, finished, kept until; a long output name stays on one line and gives
  up its middle, `2025年度…披露全文.pdf.llm.md`), its notices grouped by kind, and
  the start of the result with its front matter and its own pictures. A
  submission's window shows its progress as one bar (done, failed, stopped,
  converting, waiting) with Retry failed, Download .zip, Stop remaining and Delete,
  the cost, its failures and the notices of all its items; Summary opens the same
  overview for the whole list. Docked, a click on the page's empty space or
  Escape in the list lets go of the selection.
- **Cost** is "unknown" whenever a model ran without a complete price, never
  `$0`: complete coverage shows the amount, partial coverage "At least $x · N
  requests without a price", a model that ran without pricing "Unknown". An item
  that used no model says so.
- **Notices** are counted by kind instead of repeated per row: 31 "image
  analysis failed" warnings of one document read once as "31 images kept their
  original alt text", an incomplete cost names the models without a reviewed
  price and links to how to add one, and other warnings group by their wording
  with a count. Each keeps its full original text behind Details.

**Preview** is the window's second tab; its "⋯" adds Copy, Export PDF and its
header-and-footer choice, and Open the page for a URL. The window has one fixed
height, so the tabs never move it: Rendered, Source, Diff (only for an exact base/LLM pair) and
Files, with the Base | LLM switch beside them for a pair. Rendered starts with
the conversion's notices (collapsed to one line) and a front-matter card (title,
description, tags). Tables over 500 rows show their first 500 rows under a
sticky header with "Showing the first 500 of 20,000 rows … Show all rows";
showing all of them adds the rest in steps of 1,000 so the page keeps answering,
and the download and Source always hold every row. Numeric columns align right
in tabular figures. Code blocks are highlighted by a 3 KB module loaded only when
a preview contains code. While a result loads, Rendered, Source and Files show
its shape in grey lines inside the same frame (with the size of a large result),
so nothing moves when it arrives.

**Compare** puts the original beside the Markdown: a PDF's pages (drawn by the
server, loaded as they scroll into view), an uploaded image, the HTML the server
kept for a web page (sanitized: no scripts, styles, frames, forms, links or
remote images), a text file, or the page screenshots a conversion saved.
Hovering a heading or a table on either side outlines it on both, in the one
accent colour; PDF positions come from the page's text layer, HTML by matching
elements. An original that cannot be shown (a DOCX, a page the server did not
keep) says why and offers its download.

**On phones** the window fills the screen, over the tab bar: back, the name,
previous and next, and "⋯" (Download and every other action, Copy and Export
PDF on Preview), then the same three tabs; Compare places the original above the
Markdown. A long press on a shortened file name shows it whole.

**Notices** are transient: minimal strips at the bottom, on a phone full width
directly above the tab bar (or at the screen's edge under the window), on a wide screen a
380 px stack 24 px from the bottom-right corner. Newest at the bottom, at most
three (a fourth removes the oldest). Each is one sentence on one line with a mark
(✕ error, ! warning, ✓ done) and at most one text action (Retry, View, Cancel,
Enter token); square, on the surface colour with hairlines, no close button. Done
and warning strips leave after 4 s, errors after 6 s, never while pressed, hovered
or focused; a swipe down dismisses one. An upload in progress is a strip with
Cancel until it ends. Screen readers hear every notice (errors at once, others
politely). Lasting conditions are not notices: a conversion that ran without a
model or whose history could not be saved says so in its window's Details, and a list
that cannot load shows a panel (below).

**Empty and blocked states** are one framed panel: an icon, a title, one line
and at most one primary and one secondary action, centred on one axis. It is a
message, not a control (no hover, no pointer). The cases: no conversions yet
(and how long they are kept), no models (Add models), a filter without matches
(Clear filter), a link to a conversion that no longer exists (above the list,
Show all conversions), conversions that cannot be shown (no token: Enter
token; no server: Try now; an error: Retry, also in place of Recent on Convert),
and an unknown address. Alone on a page the panel is 480 px wide; in a column
(the list, a settings page, a phone) it takes the
column's width.

**Settings** are pages. Models lists the configured models (Test, Edit, Delete),
models detected for this session (Save to config) and the configuration source;
Add models opens the provider picker: one search over every provider (name,
model prefix and other names), saved connections and environment credentials as
rows, the eight common providers as cards, then every other provider by name
under a filter (All, Clouds, Inference, Gateways, Local, Subscriptions, Model
makers) that scrolls sideways on a phone. Providers show their monochrome brand
mark (vendored from lobehub icons, drawn in the text colour) or a monogram.
Choosing one leads to the provider's page: the key and the address side by side
(the address optional, its default as the placeholder), then the model catalogue
and, in plain sight under it, Model ID and routing (add a model by its ID, the
routing group and weight), with a breadcrumb that Escape steps back through.
Every settings page keeps its primary action in one place, the pinned bar at the
bottom (Load models, then Save, beside Cancel); a subscription needs no key or
address, and Bedrock and Vertex AI take model IDs by hand. A backend that sends
no provider `group` gets the common eight by name and everything else under All. Appearance and language chooses English, 中文 or 日本語 and the
theme (Auto, Light, Dark); Access token says whether this tab holds a token
(with Forget beside it) and takes a new one in one field with Save; it is kept in
this tab only.

## Job controls

- **Uploads** use XMLHttpRequest so the page can show the percentage and bytes
  sent, redrawn at most ten times a second, and abort them. Files over 100 MiB are
  named and refused before anything is sent. A selection or a dropped folder is
  capped at the service's item limit (`limits.max_job_items`). Hidden and system
  files inside a folder (dot names, `Thumbs.db`, `desktop.ini`) are skipped and
  counted; unreadable entries are counted.
- **Stop remaining** calls `POST /api/jobs/{id}/cancel` for every running job
  with waiting original items; stopped items read "Stopped before conversion.
  Retry to convert it." and can be retried. A 409 means nothing waited any more.
- **Retry** of an image skipped for lack of text resubmits the item's saved
  options with OCR on. An unsupported file type offers no Retry: converting it
  again cannot help, and Retry failed leaves such items out.
- **Errors** are worded from the service's stable codes in the interface
  language: an API error's `reason`, then a settings conflict's `detail.code`,
  then the status-derived `code`; a failed item's `error_code` or a recognizable
  message shape (HTTP statuses, refused connections, timeouts, model refusals,
  missing OCR, size limits). The service's full wording stays behind Details.
  Provider probe and discovery phrases are translated the same way. A job whose
  history could not be saved (`persistence_error`) reports it in the page.
- **A failed action on one conversion** (Retry, Delete, Download, Enhance, Copy;
  for a submission Retry failed, Stop, ZIP and Delete) adds one line at the top of
  its window (under the header, on every tab), in the style of the conversion's own
  failure reason: a red ✕, one sentence, Full details. It stays until the action
  is tried again or another conversion is chosen, the polite live region reads
  it, and no page notice is shown for it.
- **Offline**: no banner. A list that cannot be loaded shows the Can't reach the
  server panel with Try now, and an action that cannot reach the service says so
  in its conversion's window. The page asks again every five seconds and,
  when the service is back, shows a ✓ strip and refreshes capabilities, the list
  and the session. A broken event stream
  triggers the same check and, once closed, reconciles the job from its
  snapshot.
- **Settings conflicts**: every write carries the revision its draft started
  from. A `stale_revision` or `config_changed` refusal reloads the lists but keeps
  the draft and its revision; "Use current revision" adopts the new one and the
  next submit uses it. Other refusals (for example `settings_read_only`) stay
  plain errors. A saved provider's key is never returned by the server, only
  masked (`sk…7890`), and never placed in an input; its address comes back with
  userinfo and secret query parameters masked (`https://…@host/v1?token=…`).
  Replace, Remove and the address form each send only the field they change; an
  address left as shown keeps the stored one, and an edited address that still
  holds a mask is refused before anything is sent. An
  endpoint change with a retained server key is rejected, including when
  linked model deployments have their own credential overrides.
- **Saved conversions** come from `GET /api/history?include=items`, refreshed
  when a watched job ends and when the window regains focus (at most every 20
  seconds). The server keeps a job for `limits.history_retention_days` (7) after
  it finished; the conversion's Details show the day it is kept until. Running jobs started
  in this tab are kept in `sessionStorage` (`markitai.session`) so a reload
  watches them again.

## Preview safety and printing

Markdown is rendered with the vendored [marked 18.0.14](https://github.com/markedjs/marked/releases/tag/v18.0.14)
and sanitized with [DOMPurify 3.4.16](https://github.com/cure53/DOMPurify/releases/tag/3.4.16),
both loaded from `/ui/marked.js` and `/ui/purify.js` the first time a result is
shown. The sanitized fragment never contains scripts, styles, forms, frames or
embedded media. Images and attachment links must resolve to an exact entry of the
result's own artifact list (an image whose folder does not match, such as an
e-book's own `../Images/x.png`, resolves by file name to the one listed image of
that name); only raster images are shown, and anything else (an external or SVG
image) becomes a labelled placeholder. Images are read from the item's own asset
route (`GET /api/jobs/{job}/items/{item}/assets/{relpath}`, inline, that item's
raster images only) when the server offers it, otherwise from the file route. Other links keep only
http(s)/mailto targets and open in a new tab without referrer; a link left
without a target is plain text. An inline SVG drawing (a chart or diagram a web
page kept as markup) is drawn: its drawing properties from `style` become
attributes, and every reference stays inside the drawing. `id`s get a prefix of
their own drawing, `href`/`xlink:href` and `url()` resolve only to elements of the
same drawing, and `image`, `foreignObject`, `style`, scripts, animations, filters'
images and `use` of anything outside are removed, so a drawing loads, runs and
reaches nothing; its links follow the link rule above. Unpainted parts take the
text colour, so labels read in dark mode. Highlighting
builds its spans from text and never reparses HTML. An HTML original in Compare
is sanitized the same way and also loses its links and images, so it loads and
navigates nowhere; the server sends originals as attachments with
`Content-Security-Policy: sandbox` and HTML as plain text, so opening one
directly never runs it.

Export PDF prints a temporary clone of the rendered document through the
browser's print dialog, on A4 with light print styles (repeated table headers,
wrapped code, the optional Markitai header and `Prepared with Markitai · Source:
…` footer). Links are removed from the clone, so a generated PDF embeds no
service URL, and printing waits up to ten seconds for every image and refuses
rather than produce an incomplete document.

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
- With a token, preview images and Compare's page images are fetched with the
  header and shown through `blob:` object URLs, which the policy already allows
  (`img-src 'self' blob:`) and which are released when the view changes; Export
  PDF waits for them. An image that cannot be fetched becomes the labelled
  placeholder.
- Download links (the window's Download, Files, file links inside the
  rendered document) point at the plain `/api/` path. With a token, a click or a
  middle-click fetches the file with the header and saves it as a Blob, so a
  refusal is reported. Plain download links work without a token only when the
  server was explicitly started with `--no-auth`.
- Download all and a submission's .zip, which can be larger than memory, are
  left to the browser:
  with a token the page first asks `POST /api/download-tickets` for a
  single-use ticket and opens the returned `?ticket=` URL, valid for one GET of
  that archive within a minute (see [serve](serve.md#network-and-file-boundaries)),
  so what stays in the download list is a spent ticket.

A 401 shows the Access token needed panel in place of the conversions (and a
refused submission a strip), both with Enter token; it opens a small dialog that
stores a typed token for the tab. Settings → Access token shows
whether the tab holds a token, and replaces or forgets it.
After a server restart that generates a new token, reopen its new launch link
or use Enter token. `--no-auth` explicitly disables API token enforcement; only
direct loopback clients without forwarding headers receive the additional trust
needed for settings and URL jobs. A loopback connection through a reverse proxy
is not an authentication mechanism. For mutations, an Origin must match the
request Host's hostname and effective port, or an operator-configured
`--allowed-host`; another localhost port is not automatically allowed. See
[serve boundaries](serve.md#network-and-file-boundaries).

## Language, appearance and stored values

The interface speaks English, Chinese and Japanese. Without a stored choice, a
browser whose first language starts with `zh` gets Chinese and one with `ja`
Japanese. English is part of the application; each other language is its own
file (`/ui/i18n-zh.js`, `/ui/i18n-ja.js`, about 13 KB compressed), which the
classic script `/ui/boot.js` asks for beside the application, so a reader
downloads only their language. `boot.js` also applies the stored theme, the
sidebar's collapsed state and the language before first paint (the Content
Security Policy allows no inline script). Values kept in this browser's
`localStorage`: `markitai.lang`, `markitai.theme`, `markitai.sidebar`,
`markitai.peek` (`docked` or `window`: how a conversion opens),
`markitai.options` (preset, LLM, OCR, output profile and the image overrides;
fetch strategy, backend, cache and source choices are never remembered, so a
revisit cannot silently re-enable a remote service),
`markitai.pdf.custom-header-footer` and `markitai.notify-denied` (a refused
desktop-notification permission, which is then never asked again). A finished
job raises a system notification only while the tab is hidden and permission was
granted; permission is asked on the first submission, never on load. Provider
API keys are not retained in browser storage; the service access token is held
for the tab as described above.

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

## One frontend, two backends

Python markitai 1.2.1 serves this workbench before Rust 1.3.0 replaces it, so the
page depends only on a written API contract that both servers implement: the
routes, payloads, event stream, authentication and error `reason` codes it
reads, described by the server's [OpenAPI document](serve.md#machine-readable-api-description).
What a backend offers beyond the base job API is announced in
`GET /api/capabilities` under `features` (`cancel`, `download_tickets`,
`url_lists`, `history_items`, `original`, `result_excerpt`, `job_labels`,
`item_assets`); a missing feature hides its control instead of failing (no Stop
remaining, no Compare, archives saved through the page, history read job by job,
images read from the file route).
`docs_url` points the Docs link at the backend's own documentation.

`node scripts/package.mjs OUTDIR` packages a build for the other backend: a
reproducible tar.gz with `index.html`, every file served under `/ui/`, the
licences, and a manifest naming the source commit, each file's SHA-256, the
build command, the Content Security Policy and the page paths. It refuses a
`dist/` that does not match its sources or uncommitted workbench changes.

## Building and checking the workbench

The workbench is embedded in the CLI. End users do not need Node; contributors
who change it run `npm ci && node build.mjs` in
`crates/markitai-cli/src/server/web` and commit the generated `dist/` files.
`npm test`, `npm run typecheck`, `npm run css` and `npm run contrast` check
specific source properties. Browser interaction, focus, mobile layout and
assistive-technology behavior require their own actual checks; passing a source
test is not proof of those interactions. See [development](development.md).
