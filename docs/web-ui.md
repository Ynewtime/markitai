# Browser workbench

`markitai serve` includes a browser workbench. It is the one frontend of both
markitai backends: this Rust server embeds it, and Python markitai 1.3.0 serves
the same build (see [one frontend, two backends](#one-frontend-two-backends)).
Every file it needs is compiled into the binary; running it needs no Node
runtime, CDN or external font, and the page makes no request to another origin.

## Layout and interaction

A **sidebar** frames every view: Convert and Conversions (with the number of
conversions and a grey pulsing dot while something converts, which turns green and still once
everything is done and fades six seconds later), the Settings sections
(Appearance, Models, Access token, Edit config; a user token sees Appearance,
its own Models and Edit config for them, see [Access token](#access-token)), and Docs and GitHub at the bottom, with the version
(`2.0.0-dev`) in small muted text beside Docs and the commit it was built from beside GitHub (`-dirty` after
it when built with uncommitted changes), and under them, as small muted text centred rather than a row,
`Build: 20261017 19:54`, the build's date and time in the viewer's time zone (the label stays `Build:` in every
language; the full stamp with seconds and offset is the tooltip). The Settings list on a phone ends with the Docs
and GitHub rows and the same centred `Build:` line under the list; the icon rail leaves it out. Its button collapses it to a 60px icon rail; each entry
keeps its name for assistive technology and shows it as a tooltip on hover and
keyboard focus, and the choice is remembered in this browser. On phones the
sidebar gives way to a top bar and a bottom tab bar (Convert, Conversions,
Settings); Settings there is one list (the same sections, Docs,
GitHub) with the version centred below it. Every view is an address (`/`, `/jobs`, `/settings/models`,
`/settings/appearance`, `/settings/access`, `/settings/config`), so reloads, Back and new tabs keep
it. Any other address opens the workbench (with status 404) on "Page not found",
with the way to Convert and Conversions. On a wide window every page is one
column centred between the sidebar and the window's edge (Convert 800 px, the
Settings pages 760 px, Conversions 1280 px).

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
The options start from the server's defaults (its configuration: `llm.enabled`,
`ocr.enabled`, `image.alt_enabled`, `image.desc_enabled`, `screenshot.enabled`,
`screenshot.screenshot_only`, `llm.pure`, `cache.no_cache`, `image.compress`,
`output.profile`, `fetch.strategy`, and with contract v1.20 `output.format`,
`ocr.lang`, `ocr.per_page_routing`, `output.page_markers`), or for a user token its own defaults over
them. A row above the command line says whether the choices are those defaults;
Reset to defaults returns to them, and Save as default makes the current choices
the defaults: the admin's go into `config.json` (the command line reads them
too), a user token's into its own settings (Use the server's defaults removes
them). Cloudflare is never a default: it asks for a confirmation each time.
Other also offers Text only (images and PDFs as the recognised text alone, see
[images](images.md#text-only)) when the backend reports `features.text_only`;
it is sent as `text_only` only when chosen and is not one of the saved defaults.
Contract v1.20 fills in the rest of what a conversion can produce, each shown only
when the backend reports its feature: a `{ } JSON` switch after the Output
profiles (`format: "json"`, `features.document_json`) also saves each result's
[document JSON](json-output.md) as `<name>.json` beside its Markdown, in the
result's files, its .zip and the command line (`-f json`). While OCR is on and LLM
off, a picker after OCR names the local engine and its language (`Local ·
Automatic language`, `ocr_lang`, from the server's `ocr_languages`; the automatic
`en` reads Chinese, English and Japanese, see [OCR](ocr.md#languages)). Advanced
gains a Pages row: Page Markers (`page_markers`, `--page-markers`; on and locked
while the profile is RAG, which keeps its own) and OCR Every Page
(`ocr_all_pages`, `ocr.per_page_routing: false`, only while OCR is on, see
[PDF OCR](pdf-ocr.md)). The OCR settings appear in the command line as one
`--config-json '{"ocr":{…}}'`. Image Descriptions (`desc`) is a picture
analysis, not the JSON output.
Files and folders waiting for Convert appear in it relative to the folder that
holds them (`./notes.pdf`, a picked folder as `./Field trip/`, quoted for the
shell), with an (i) beside Copy: run it in that folder, since a browser never
tells a page where a chosen file lives.
On a phone the composer is lighter: the tools are bare 32 px icons (each keeps
its name for assistive technology and as a tooltip) and Convert is a square arrow.
The latest three submissions follow, each one click from its details. Files
chosen with Upload or Folder, or dropped on the page, never start a job by
themselves: they wait above the URL field in a staged list (a folder or list
named with its count and size and Remove all, then one row per file with its
own remove button; choosing more adds to it, the same file once) until Convert
sends them together with any typed URLs as one job, whose name is the folder or
list when it is only that. A `.urls` list becomes one URL item per entry (`#` lines and blank lines
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
  middle, `blog-japanese-…ning.html`; a URL dims its host, which a phone leaves
  out), a warning count, and
  the duration or the state in words. A submission of several items is a header
  (what was submitted: `report-materials`, `14 files`, `3 web pages`; the preset;
  the time) with its items under it, indented, so a submission of one item is one
  row at the outer edge. It is
  an ARIA tree: Up and Down move, Right and Left open and close a submission,
  Home and End jump, and a click or Enter opens the conversion's window. A box
  at the left edge of every row and submission header (one column; a header's
  box stands for its items and shows a dash while only some are checked) selects
  conversions to delete together: the rows carry `aria-checked`, Space checks
  the focused row, Shift (with Space or a click) extends from the last one,
  Ctrl/⌘+A checks every listed row, Escape clears and Delete asks. A bar at the
  bottom of the window then says how many are checked, with Select all, Clear,
  Delete and, to its right, Re-convert (below); Delete asks first ("Delete 3 conversions? Their Markdown, pictures and
  kept originals are deleted too.") and one notice reports the result, naming
  what could not be deleted. Items still converting cannot be checked.
  **Re-convert** takes the checked items back to Convert with their originals
  staged and the options open, so they can be converted again with other options
  (LLM enhancement included: with no model set up, an inline note and a notice
  say so and where to add one); Convert then records a new submission and leaves the
  first results as they were. Each kept upload is read back through the item's
  original route (`GET …/original`, then its `download`), a URL item is added
  by its address, and an original that is no longer kept (a CLI-recorded job, a
  removed upload) is named in a notice and left out. A conversion's "⋯" menu
  has the same Re-convert for that one item. On a
  phone the boxes appear in Select mode (beside the filters; a tap then checks a
  row, Done leaves). The server deletes them in one request
  (`POST /api/history/delete`), or, on a backend without it, the page asks for
  each, four at a time. A submission's chevron sits in the status marks'
  column, so the header and a lone conversion below it share one left edge.
- **Names** of files, pages and folders are shown decoded wherever the page
  shows them (rows, the window's title, Details, Files, download names the page
  sets): `20260314-%E5%B1%B1%E9%97%B4.md` reads `20260314-山间.md`. Only valid
  UTF-8 escapes are decoded; separators (`%2F`, `%5C`, `?`, `#`, `%`), controls
  and invisible or direction-changing marks stay escaped, and links and the
  names the server keeps are unchanged.
- The **peek window** opens over the dimmed list: previous and next with the
  position (`8 of 14`), the name (all on one centre line with the buttons; a
  long name gives up its middle at about 34 characters and shows in full on
  hover), the actions that matter (Download, which saves the Markdown, with a
  caret listing Markdown .md, JSON .json (`…/document`, when the backend reports
  `features.document_json`), All files .zip and Choose files…, which opens
  Preview's Files; Retry,
  Add a model or Enable OCR and retry when that is the fix) and "⋯" (Enhance
  with LLM, Run again, Re-convert, Retry without LLM, Delete), then Preview, Compare and
  Details as tabs; a conversion opens on Preview (a submission, or a
  conversion without a result, on Details; one just submitted shows Details while
  it converts and turns to Preview when it is done, or stays on Details if it
  failed; `?view=details` is Details' own address). The expand button (or F) grows the window to the whole
  viewport and back, remembered in this browser. Up and Down (outside the
  scrolling document) move between items, C switches Compare, Escape, ✕ or a
  click on the dimmed list closes it and returns to the row; so do the browser's
  Back button and the trackpad's back swipe (opening the window from the list adds
  one history entry, so Back returns to the list instead of leaving the page; a
  window opened from a link or from another page closes without adding one).
  Summary's window does the same. Each Back goes one level back: an item opened
  from the summary or a submission's window (a notice's name) returns to that
  window, and so does the list narrowed by Show items; Escape and ✕ close every
  level at once. (The docked
  side-panel layout of earlier builds is gone; a browser that chose it opens the
  window.)
- **Details** shows the item: its state in a sentence
  ("Converted with LLM in 14.6 s · 12 images", or the failure in plain words), the
  facts (source and its size, output, what else was saved, models, cost,
  options, finished, kept until; a long output name stays on one line and gives
  up its middle, `2025年度…披露全文.pdf.llm.md`), its notices grouped by kind, and
  the start of the result with its front matter and its own pictures, in the same
  centred reading column as Preview (the whole card opens Preview). A
  submission's window shows its progress as one bar (done, failed, stopped,
  converting, waiting) with Retry failed, Download .zip, Stop remaining and Delete,
  the cost, its failures and the notices of all its items (each notice names the
  items it covers, the first three as buttons that open that item and "+N more" for
  the rest; Show item opens the one item, and Show items closes the window and narrows
  the list to exactly those items, with a thin "N items" bar over the list (its count drops
  as items are deleted; × shows all again); Summary opens the same
  overview for the whole list.
- **Cost** is "unknown" whenever a model ran without a complete price, never
  `$0`: complete coverage shows the amount, partial coverage "At least $x · N
  requests without a price", a model that ran without pricing "Unknown". An item
  that used no model says so. Before there is an amount the line says what will
  happen: "Counted once it starts" (waiting), "Counting…" (converting with a
  model), "Waiting for the model…" (Enhance), "No model requested", and after the
  end "None: no model answered", "None: no model ran" (failed) or "None: stopped
  before a model ran"; a submission still converting shows "Counting…" or the
  amount "so far".
- **Notices** are counted by kind instead of repeated per row: 31 "image
  analysis failed" warnings of one document read once as "31 images kept their
  original alt text", an incomplete cost names the models without a reviewed
  price and links to how to add one, and other warnings group by their wording
  with a count. Page numbers read as ranges everywhere (`pages 1–12, 15, 20–31`,
  one warning per page with the same words as one); a PDF whose pages (nearly all,
  90 % or more) have no readable text says "This PDF is scanned: turn on OCR and
  convert again", and one with some such pages names them with the same action. A
  model answer a check refused says in plain words that it was not used and that
  the converted Markdown is kept: it left out or changed too much of the document,
  or (Python's page drift guard) it did not match the document page by page. Each keeps its full original text behind Details.

**Preview** is the window's first tab; its "⋯" adds Copy and Export PDF…,
which opens a small card with the one choice it asks (Markitai's own header and
footer, explained under it) and Print…, and Open the page for a URL (on Compare
the page opens from the Original header instead, so it is offered once). **Files** lists the original first (the upload, or the page as the server
fetched it), then every file of the result, each with its own download, their
count and size, and Download all (.zip) at the bottom right: one ZIP of the
item's files and its original (servers without the item ZIP zip a one-item
submission's job instead, without the original, and offer nothing for an item of
a larger submission). The window has one fixed
height, so the tabs never move it: Rendered, Source, Diff (only for an exact base/LLM pair) and
Files, with the contents button and the Base | LLM switch beside them, sit at the
end of the window's own tab row (Preview, Compare, Details), so the window has one
tab row. Where they do not fit (a phone) they become one button naming the view
and version on screen, whose menu holds the views and versions. Each keeps its reading
column centred in the window: 820 px, wider as the window grows (56 % of its
width, at most 1120 px); every table keeps to that column (its width and left
edge), and one whose cells need more scrolls sideways inside its own box. Rendered starts with
the conversion's notices (collapsed to one line) and the front matter as one light
line above the body: a tag icon, `Properties · N` (the number of properties), the
tags inline (`gardening · community beds · …`; else the summary) and Summary at its
end when there is one, with no title of its own (the body's heading is the title).
The whole line is one button (a click anywhere on it, Enter or Space) that opens it into a
two-column properties table (closing with its own rule; a rule that opens the body right after the front matter is not drawn a second time) (Summary, Tags as chips, Source, Processed and every
other key; one key width with a rule between key and value, long keys and values
wrapped) whose `Properties · N` header row, anywhere on it, folds it again; the
choice holds for the tab's session (`sessionStorage['markitai.front']`) and in
Preview, Compare and the Details excerpt alike. Tables over 500 rows show their first 500 rows under a
sticky header with "Showing the first 500 of 20,000 rows … Show all rows";
showing all of them adds the rest in steps of 1,000 so the page keeps answering,
and the download and Source always hold every row. Numeric columns align right
in tabular figures. A table that a line break inside a cell broke (a raw line break,
or a Unicode line or paragraph separator, which ends a table for the parser) is
put back together for drawing: the break becomes a line break in the cell and the
rows after it stay in the table (Source keeps the text as written). A video written
as a linked poster (`[![Video](poster.jpg)](https://…/clip.mp4)`) plays in place
(`https` only, loaded only when played, no referrer, with its poster: the link's
picture, from the web or the item's own copy through its asset route);
one that cannot play turns back into its link. Code blocks are highlighted by a 3 KB module loaded only when
a preview contains code. Formulas, `$…$` inline and `$$…$$` as a display (also
on lines of their own, blank lines and `aligned` environments included), are
drawn by KaTeX 0.19.0 in a separate module (`/ui/math.js`, with `/ui/math.css`
and the woff2 fonts `/ui/KaTeX_*.woff2`, all vendored and served by the binary)
fetched only when a preview holds one; they also draw in headings and in the
Contents. A `$` follows Pandoc's rules, so `$5 and $6` stays text, and a formula
that does not parse stays as its source. A fenced block marked `mermaid` is drawn
as its diagram (every diagram type of Mermaid 12.1) by Pele 1.x in a separate
module (`/ui/mermaid.js`, about 260 KB compressed) fetched only when a preview
holds one; Source, the downloads and the Markdown itself keep the code. The
diagram's SVG is rebuilt from an allowlist of drawing elements and attributes,
its styles set through the page's script under the same Content Security Policy;
its links open in a new tab without a referrer, and it loads no pictures. Its
colours follow the theme without being drawn again, and a label on a series colour (a git
branch, a pie slice, a timeline period) takes whichever text colour reads better on it. A block that does not parse,
holds a diagram type Pele does not know, or is too large (over 50,000 characters,
2,000 flowchart edges or 2,000,000 characters of SVG) stays a code block with a
note saying why. While a result loads, Rendered, Source and Files show
its shape in grey lines inside the same frame (with the size of a large result),
so nothing moves when it arrives.

A long document stays responsive. Rendered reads the Markdown a piece at a time
(cut only between top-level blocks, so every piece reads as it does in the whole
document; a document with reference-style link definitions is read at once),
draws its first screen at once and the rest a few chunks per frame; the browser
lays out and paints only the chunks near the viewport, and pictures load as they
come near. A drawn document is kept for its next view (Details, Compare and back,
Base and LLM), with its scroll position. Source is built the first time it is
opened, in blocks the browser also skips off screen. Diff compares the two
versions line by line (Myers' algorithm, bounded so a document changed
everywhere is refused rather than freezing the page) and folds unchanged
stretches into "Show N unchanged lines". On a wide window the versions sit side
by side (base left, LLM right, each with its line numbers), a change's removed
and added lines face each other and the words that changed between them are
marked; a narrow window shows one column of changes. In Diff the Base | LLM
switch shows both on (it compares both); choosing one shows that version. A document with four or more headings
gets its **contents**: its top three heading levels (a lone title does not count
as a level; each entry is the heading's plain words, without inline HTML tags or
private-use glyphs, on one line cut with an ellipsis, the whole heading in its
tooltip; formulas in it are drawn) beside the document on a wide window, marking the
heading at the top of the reading area as it scrolls; a click goes to the heading
and marks it until the reader scrolls (pictures loading below it do not move the mark). The contents button beside
Base | LLM hides or shows them (remembered in this browser); on a phone or a
narrow window a round button over the document's corner opens them as a sheet.
Export PDF waits for the whole document and every picture before it prints.

**Compare** puts the original beside the Markdown: a PDF's pages (drawn by the
server, loaded as they scroll into view), an uploaded image, an uploaded HTML
file (sanitized: no scripts, styles, frames, forms, links or remote images), a
text file, or the page screenshots a conversion saved. A PDF has a viewer's
toolbar in the original's header: the outline button, previous and next page
with a page box (`12 / 492`, Enter jumps; PageUp and PageDown turn pages; the box
follows scrolling), and zoom (Fit width, Fit page, 50–200 %, − and +, and
Ctrl/⌘ + − 0 or Ctrl with the wheel while the pane has the focus or the pointer;
the zoom is kept per conversion for the session, and the page being read stays
in place). Pages are drawn at the zoom's width near the view only. The outline
panel (beside the pages, over them in a narrow pane, a sheet on a phone) shows
the PDF's own bookmarks, else the Markdown's headings, else page thumbnails, and
says which. Blocks link at every zoom, on scanned pages too (by the lines OCR
read there, when the server reports them); the page text of a long PDF is read in
the background, the first 40 pages first, so linking reaches its last page. A web page shows the page
the server kept, else its screenshot, with Screenshot | Page when both are
usable: the server says when either was a verification or captcha page, only a
request for JavaScript, or blank (almost no text and no pictures), and such a
view is not offered; with neither, the Original side says why the page could not be kept as it looked and
offers Open the page. A server that loads a page in its browser keeps a
screenshot of it even when none was asked for (the Rust server: JPEG tiles, about
85–230 KB per 2000 px of page, at most 20 tiles, beside the job, never in the
result). Page shows the page as the server fetched it, in a sandboxed frame served by the
item's own route, with the site's own pictures, style sheets and fonts (over
HTTPS) but none of its scripts and no form, so nothing in it can reach the
workbench (links open in a new tab). The frame keeps the server's origin, so
the workbench can read the page it shows; its policy still refuses every script,
inline handler and `javascript:` URL (see [serve](serve.md#originals-for-compare)).
Its blocks pair with the Markdown's by their words (headings, paragraphs, list
items, code, table cells and captions, in order; pictures by address, else the
page's picture between the blocks paired around it), and the outline is a style
the workbench adds to the frame. The workbench reads the page as soon as it is
parsed (a page without scripts loads all its pictures at once, which can take
long) and follows the pointer over it by the page's own hover state (Safari runs
no handler in a page that may run no script). Screenshot links the same blocks
where the browser drew them when it took the screenshot (servers that record the
boxes, contract v1.14), outlined on the picture. Open the page sits in the
Original header, with an (i) that says what the frame shows; a web page's
header has no download button (the kept page is in Files). Live frames of the real address are not used: most sites
refuse them. Hovering a heading, table, paragraph, list item, code block or
picture on either side outlines it on both, in the one accent colour. PDF
positions come from the page's text layer, and a picture's from where its page
draws pictures (when the page draws as many as the Markdown places on it, and
the server reports them); a framed web page pairs by words as above (a backend
whose frame policy is still opaque shows the page without outlines); an
uploaded HTML original pairs its headings and tables, a text
file its headings. A block the original side cannot show (nothing pairs it, or the
side shows a screenshot without boxes or nothing) is not outlined, and the
outline never moves the text. An original that cannot be shown (a DOCX, a page the server did not
keep) says why and offers its download.

**On phones** the window fills the screen, over the tab bar: back, previous and
next, and "⋯" (Download's formats and every other action, Copy and Export PDF on
Preview), the name on a row of its own below them, then the same three tabs. While items wait, Stop remaining
stays a button in the header, since the window covers the list's own: a
submission's window stops that submission, an item's window the submission it
belongs to, and Summary everything listed; Compare places the original above the
Markdown. A long name gives up its middle, cut to the width it has with the "…"
right against what follows: a path or web address loses whole folders first
(`slow/annual-reports/…/s5.html`), keeping its host and last segment (a query
goes first), and only then the middle of that segment; a file keeps its
extension and the four characters before it. The whole name is in the tooltip
and read aloud; a long press on a shortened name shows it whole. On a phone the Conversions header
keeps its title, New conversion and "⋯" on one row (the two buttons go below
the title when they do not fit), the status sentence below, and Stop remaining
on a row of its own.

**Notices** are transient: every one in the workbench (a model test, a saved
configuration, a failed download or action, a copied link, a retry) is a toast
card at the bottom, on a wide screen in a 380 px stack 24 px from the
bottom-right corner, on a phone centred directly above the tab bar (or the
selection bar); fixed, so nothing on the page moves. Newest at the bottom, at most
three (a fourth removes the oldest). Each has a mark (✕ error, ! warning, ✓ done),
its sentence (a short lead in bold when it has one), at most one text action
(Retry, View, Cancel, Enter token), the service's own wording folded under Details
when there is any, and a close button. Done and warning toasts leave after 5 s,
never while pressed, hovered or focused; errors stay until closed; a swipe down
dismisses one. An upload in progress is a toast with
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

**Settings** are pages. Models lists the configured models (Test, Edit, Delete;
a model's routing weight is edited in place: click the pill, type, Enter or
leaving the field saves, Escape cancels; a weight is a whole number from 0 to 100
(0 = off: the model stays saved but gets no requests; the empty field shows
`0–100`, a value outside it is said under the field and Enter keeps the field
open, the arrow keys stay within it, and every form checks the same rule);
a test's result is a toast, and a failed test says why, under the model and in its toast: no API key,
naming the server's key variable when it is not set, key rejected, model not
found, rate limited or out of quota, provider unreachable, no answer in time, or
a provider error, with what to do and the provider's own message folded below),
models detected for this session (Save to config) and the configuration source;
on servers with user tokens each saved model has a Private | Shared toggle: a
shared model also serves users' conversions in this workbench (never any other
route), its key and address staying on the server;
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
Every model already saved for the connection shows in its list, ticked and
marked Configured: on the listed row when the provider lists it under another
name ("Saved as claude-agent/sonnet" for Claude's `sonnet`, `opus`, `haiku`
aliases, an undated Anthropic ID on its newest snapshot), else as its own row
("Saved, not in the provider's current list"). Ticking some models and unticking
others saves in one request: the additions and removals succeed or fail
together.
Every settings page keeps its primary action in one place, the pinned bar at the
bottom (Load models, then Save, beside Cancel); a subscription needs no key or
address, and Bedrock and Vertex AI (also Perplexity, Z.ai and Fireworks AI) take
model IDs by hand, saved connections included, whatever a backend's card says. Vertex AI (like
watsonx and GigaChat, when a backend offers them) takes no address, here or on
its saved connection: it signs in with the server's own login, which the page
names in a muted line, at its own endpoint, and takes no key either (a saved
connection offers no Add key; a key a config entry still holds can be removed).
A saved subscription offers no key or address. Bedrock keeps its optional key
and address. A subscription provider (Claude, Copilot, ChatGPT) whose official
runtime cannot list models says the cause instead of one generic error
(contract v1.18): not installed on the server (or its path variable names no
file), not a release the adapter speaks (naming the installed one when known),
not signed in, or installed but failing. It shows the command to run on the
machine where the server runs (install the pinned release, or the sign-in
command) with a copy button, a link to the setup guide and the server's own
sentence under Provider's message. Adding a model by hand is switched off, with
the reason beside it, for the first three: they are certain to fail until
someone runs a command, so a saved model would only fail every conversion that
routes to it, and reloading the list (the ↻ button) re-checks once the command
has run. After a failure of unknown cause entry stays open with a caution that
the model may not work until the runtime answers, because a transient failure
should not stop a deliberate choice. A backend that sends
no provider `group` gets the common eight by name and everything else under All. Appearance (its page is titled Appearance and language, first in the list) chooses English, 中文 or 日本語 and the
theme (Auto, Light, Dark); for a user token it also says which token the tab uses, with Sign out (another token signs in from the sign-in screen that follows). Edit config (admin) shows the configuration file the Models page names as its source, in an editor that follows the theme and colours the JSON (keys, strings, numbers, `true`/`false`/`null`, punctuation; the text field itself stays a plain text area for the keyboard and screen readers, and Windows high-contrast mode shows the plain text): secrets masked (a masked value left where it is keeps the stored secret), the JSON checked as it is typed with the line and column of a mistake, Review changes as a line diff, then Save config, sent with the revision the edit started from and with the file's own line endings (CRLF on Windows; a byte order mark kept): Review shows only real changes, an unchanged line is saved byte for byte, a new line gets the file's ending, and a file that mixes endings says so, keeps each line's own and gives new lines the most common one (a file changed meanwhile offers Load current file; a configuration the server cannot use is refused with its reason). The model list applies at once; other settings after a restart, which the page says. Access token says whether this tab is signed in
(with Sign out of this tab beside it, which forgets the token here only and
opens the sign-in screen: switching tokens is Sign out, then sign in), and
Regenerate token (below); a token is kept in this tab only. Only a refused or
missing token shows a token field here.

In **Add models** a provider appears once among the saved rows: the server's
key for it (`Using OPENAI_API_KEY`) and its connection at its own endpoint are
one row, which says how it authenticates (`Using DEEPSEEK_API_KEY`, `Key saved`,
`No key`) and its model count; a connection at another address is its own row.
Common and All mark a saved provider as Added, and choosing it opens that row.
A provider's key field says where else the key can come from (`Or set
MOONSHOT_API_KEY where markitai serve runs …`). A hand-written model that the
server serves with its key variable shows `From DEEPSEEK_API_KEY` in Edit, and
a model list that cannot be loaded says why in the model test's categories (no
key, key rejected, rate limited, unreachable, no answer in time, provider
error), with the provider's message folded.

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
  for a submission Retry failed, Stop, ZIP and Delete) is an error toast: a red ✕,
  one sentence, the service's wording under Details. It stays until it is closed
  or the action is tried again, and the alert live region reads it.
- **Offline**: no banner. A list that cannot be loaded shows the Can't reach the
  server panel with Try now, and an action that cannot reach the service says so
  in its toast. The page asks again every five seconds and,
  when the service is back, shows a ✓ toast and refreshes capabilities, the list
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
that name); only raster images are shown. A picture on the web (an absolute
`https://` address without user information, as a fetched page's Markdown has)
loads from its site lazily and without a referrer, and becomes the labelled
placeholder only if it fails to load; any other image (`http:`, `data:`, SVG)
becomes the placeholder. Images are read from the item's own asset
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
  (`img-src 'self' blob: https:`: the page's own and `blob:` pictures, and
  remote pictures over HTTPS only; `http:` and `data:` pictures are refused; `media-src https:` lets a linked video play from its site) and which are released when the view changes; Export
  PDF waits for them. An image that cannot be fetched becomes the labelled
  placeholder.
- Download links (Files, file links inside the rendered document) point at the
  plain `/api/` path. With a token, a click or a
  middle-click fetches the file with the header and saves it as a Blob, so a
  refusal is reported. Plain download links work without a token only when the
  server was explicitly started with `--no-auth`.
- Download all, a submission's or an item's .zip and an item's original, which
  can be larger than memory, are left to the browser:
  with a token the page first asks `POST /api/download-tickets` for a
  single-use ticket and opens the returned `?ticket=` URL, valid for one GET of
  that archive within a minute (see [serve](serve.md#network-and-file-boundaries)),
  so what stays in the download list is a spent ticket.
- Compare's framed web page is opened the same way: a frame cannot send the
  header, so it loads `…/original/view` with a single-use ticket (the page's
  policy allows `frame-src 'self'` for it).

A 401 shows the Access token needed panel in place of the conversions (and a
refused submission a toast), both with Enter token; it opens a small dialog that
stores a typed token for the tab. Settings → Access token shows
whether the tab is signed in and signs this tab out (another token then signs in
from the sign-in screen).
**Regenerate token** (on servers that offer it) asks first, then replaces the
server's token: every other tab and device is signed out at once (open event
streams end, unused download tickets stop working) and must sign in again,
while this tab keeps going with the new token. The new token is shown once,
with a link and its QR code to sign in on another device (the server's network
address when it listens on the network). Rust keeps the new token until the
server stops; at the next start it uses `MARKITAI_SERVE_TOKEN` again when that is
set, or makes a new one.
User tokens are kept when the server's token is regenerated. The server's
token is the admin's; on servers that offer user tokens an admin can create
labelled tokens for other people or devices and revoke them, and a user token
reaches everything but the settings (including Edit config) and the token
routes, and sees only its own conversions (see
[access levels](serve.md#access-levels-and-user-tokens)).
Access token (admin) lists them (name, the token's last four characters,
created or last used, Revoke with a confirmation) and creates one from a name,
showing it once with Copy, a sign-in link and its QR code; it joins the list after
Done (at the right); a token with models of its own says how many ("3 models"),
and Revoke says they are deleted with their keys. A user token sees
Appearance and Models among the settings (an admin page says "Admin token needed").
Its Models page is the admin's (Add models, Test, Edit, Delete, weights) for models
of its own, kept on the server for that token alone and used only by its own
conversions; a model there needs the user's own API key, or the address of a
server that takes none (not the server's own machine), and subscriptions or cloud
logins that would sign in with the server's account are refused. Below, "Shared
by the admin" lists the shared models (model, provider, weight), each with Test
(it runs on the server with the admin's settings and says only whether it passed
and why not) and a weight of the user's own: it changes only how often that
user's conversions pick the model (0 turns it off for them alone), and "Use the
admin's 2" follows the admin's weight again. A user's conversions route over its
models and the shared ones as one group, by weight; with neither, the Convert
page's LLM options are off and say to add a model in Settings → Models. Its Edit
config is the admin's page for its own models only: one JSON document of its
models, their connections and its weights for the shared models (keys masked,
kept when left as they are; Review, Save with the revision), never the server's
configuration.
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
`markitai.peek` (`full` or `window`: whether a conversion opens at full screen;
an older `docked` reads as `window`),
`markitai.options` (only with a backend without convert defaults: preset, LLM,
OCR, output profile and the image overrides; fetch strategy, backend, cache and
source choices are never remembered, so a revisit cannot silently re-enable a
remote service),
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

A tab signed in with a user token is offered only the local strategies (Auto,
Static, Playwright) and no backend choice: a user token's conversions never use a
remote service, and both backends refuse those choices for it (403
`remote_processing_forbidden`; see [serve](serve.md)).

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

Python markitai 1.3.0 serves this workbench before Rust 2.0.0 replaces it, so the
page depends only on a written API contract that both servers implement: the
routes, payloads, event stream, authentication and error `reason` codes it
reads, described by the server's [OpenAPI document](serve.md#machine-readable-api-description).
What a backend offers beyond the base job API is announced in
`GET /api/capabilities` under `features` (`cancel`, `download_tickets`,
`url_lists`, `history_items`, `original`, `result_excerpt`, `job_labels`,
`item_assets`, `document_json`, `text_only`, `ocr_lang`, `ocr_all_pages`, `page_markers`;
`document_json` and `text_only` arrived with contract v1.19: the item menu's JSON entry
saves `…/document`, the options' Text only sets `text_only`; contract v1.20 adds the
job options `format` (under `document_json`), `ocr_lang` (with `ocr_languages` and
`ocr_lang` in the capabilities), `ocr_all_pages` and `page_markers`, and the same
keys as saved defaults); a missing feature hides its control instead of failing (no Stop
remaining, no Compare, archives saved through the page, history read job by job,
images read from the file route).
`docs_url` points the Docs link at the backend's own documentation. The guide links
(pricing, subscriptions) open the page beside an `index.md` address, or `guide/<page>`
under a site root (`https://markitai.dev` or `https://markitai.dev/`).

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
specific source properties. The other libraries are pinned exactly; Pele, the
Mermaid renderer, takes any 1.x from 1.9.0 so it follows upstream fixes, and
`package-lock.json` decides the version a build bundles. During dependency
updates run `npm update pele`, then `node build.mjs` and the checks above. When the
locked version changes, copy the new package's `LICENSE` to `vendor/web/pele-LICENSE`
and update its `vendor/web/provenance.json` entry and the Pele line of `NOTICE`
(`scripts/test_web_dist.py` fails until they match the lock). Browser interaction, focus, mobile layout and
assistive-technology behavior require their own actual checks; passing a source
test is not proof of those interactions. See [development](development.md).
