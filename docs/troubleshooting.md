# Troubleshooting

Start with `markitai doctor`. It checks the browser, LibreOffice, local OCR,
the built-in readers and service, and the configured models, without sending a
model request or opening a remote page. `markitai doctor --json` prints the same
checks for scripts. Add `-c FILE` to check a particular configuration.

## Exit status

| Status | Meaning |
|---|---|
| 0 | Success, including dry runs and skipped items |
| 1 | A single input failed, or a runtime error such as a missing input file or model |
| 2 | Usage error: unknown or removed option, missing `-c` file, `--json` without `-o`, interactive mode without a terminal |
| 10 | A directory or `.urls` batch finished with at least one failed item |
| 130 / 143 | Interrupted by Ctrl-C / SIGTERM |

Usage errors are printed on stderr only; `--json` prints its JSON document only
after arguments are accepted. Failed batch items are listed in the JSON output
and in the batch report under `out/.markitai/reports/`.

Standard output behaves as with other Unix tools. A reader that stops early
(`markitai big.txt | head -1`) ends the run quietly with the status it already
had, 0 after a successful conversion. Any other failure to write it, such as a
full disk behind `> file`, prints `Error: Cannot write to standard output: …`
and exits 1.

## Common problems

**An image produces no output.** A standalone image has no text unless you ask
for recognition; Markitai says on stderr that it skipped the image, by file
name (`Skipped photo.png: an image has no text to extract …`), and exits 0
because nothing went wrong. Add `--ocr` (on-device OCR, macOS) or `--llm`
(vision model). With `--json` the item shows `"skip_reason": "image_only"`; the
batch summary words it as `images with no text`. A skipped image creates no
output directory and no ownership files. A file that only has an image's
extension is an error instead: `File is empty (0 bytes)` or `File is not a valid
image: …` (also `the file appears to be damaged or truncated` when the format is
recognized but the data does not decode).

**`Error: File is empty (0 bytes)`.** The input has no content. An empty `.txt`,
`.md`, `.csv` or `.tsv` is still converted to an empty document; every other
format needs content to read.

**`Error: The file appears to be damaged or truncated (…)`.** The reader could
not make sense of the container: a ZIP-based document (`.docx`, `.xlsx`,
`.pptx`, `.odt`, `.epub`) whose end is missing, or a PDF without a usable
cross-reference table. The reader's own wording follows in parentheses. A
download that stopped early or a file that was saved twice is the usual cause;
open it in its application, or fetch it again.

**`Warning: The content is a PDF document although the file name ends in .docx; …`.**
The file name and the content disagree between document types (a PDF saved as
`.docx`, a Word file saved as `.pdf`, a modern package saved as `.doc`). Markitai
reads the content as what it is and says so once; rename the file to silence
the warning. Page screenshots and page OCR need the right extension.

**`Warning: PDF pages 2-3, 5: native text was not recovered …`.** These pages
have no extractable text (scans, or pages made only of pictures). Run again with
`--ocr` to read them (macOS). The terminal shows one line per document; the
report and `--json` keep one warning per page.

**The first OCR takes half a minute.** Vision compiles its recognition models
for this executable the first time it runs and caches them, so the first OCR
after installing or updating Markitai takes about 25–45 seconds and later ones
well under a second per page. A sandbox that forbids writing
`~/Library/Caches/markitai` makes OCR fail with `missingError`.

**Images are missing when I print to stdout.** Without `-o`, images are saved
under `MARKITAI_HOME/assets/blobs/` (`~/.markitai/assets/blobs/` by default) and
linked with `file://` URIs, which open on this machine only. If the output
still shows `.markitai/assets/…` references, either `image.stdout_persist` is
`false` (stderr says so) or the store could not be written; the warning names
the directory and the reason, often a symbolic link refused while
`output.allow_symlinks` is off. Use `-o out/` for Markdown with portable
relative image paths. See [images on stdout](images.md#images-on-stdout).

**The output is named `….v2.md`.** The target already existed and the default
conflict policy is `rename`. A single conversion says `Wrote out/a.docx.v2.md
(a.docx.md already exists)`, a batch lists the renamed results in its summary
(`Renamed 3 items (output already exists): …`), and `--dry-run` shows the name
in advance. Choose `overwrite` or `skip` with
`markitai config set output.on_conflict overwrite`, or for one run with
`--config-json '{"output":{"on_conflict":"overwrite"}}'`.

**`Error: No model configured; set MODEL and a provider API key, or llm.model_list`.**
`--llm` (or `llm.enabled`) needs a model. Set an API key such as
`OPENAI_API_KEY` and run `markitai init --yes`, or configure `llm.model_list`;
see [quick start](quickstart.md#7-enhance-with-a-language-model) and [LLM](llm.md).

**`Warning: LLM enhancement failed; base Markdown retained: Environment variable not found: …`.**
A configured model refers to `env:NAME` and that variable is not set. The base
Markdown is still written because `llm.on_failure` defaults to `fallback`; set
it to `fail` to make this an error. Put the key in the environment, in `.env`
in the current directory, or in `MARKITAI_HOME/.env` (`~/.markitai/.env`).

**`LLM returned HTTP 403: the model is not available in this region`.** The
provider refuses this model for the network's region; the key itself may be
valid. OpenRouter does this for some models. Use another provider or model, or
pin one with `MODEL`. When ambient keys form the model pool, the blocked
deployment is skipped for the run with one warning and the others serve the
documents. A bare `LLM returned HTTP 401` or `403` usually means a rejected key
or a missing permission.

**`Warning: Some observed LLM requests could not be priced`.** The model or
endpoint is not in the bundled [price catalog](pricing.md), so `cost_usd` only
counts the requests whose price is known. It does not mean the call was free.

**`error: --playwright has been removed, use '-s playwright' instead.`** The old
`--playwright`, `--static`, `--jina`, `--defuddle` and `--cloudflare` switches
are now values of `-s/--strategy`. `--kreuzberg` is gone because RTF is read
natively. Like other usage errors, it is printed in the argument parser's own
form: `error: …` (lower case, not `Error: …`), then the usage line, exit status 2.

**`Error: HTTP 404 for https://…: the page may have been removed or is not public`.**
The site answered with that status. The message names the page (without
credentials, and with secret-looking query values replaced) and, for the common
statuses, says what it usually means: 404/410, removed or not public; 401/403,
the site refused access, may block automated clients or need a login (Markitai
sends no cookies or login unless you configure a browser identity, see
[browser contracts](browser.md)); 429, rate limited, try again later; 5xx, a
server error on the site. Only a connection that could not be established
because the peer cut it off (such as `tls handshake eof`) is retried, twice; a
status is reported at once. See
[URL fetching](fetch.md#failures-retries-and-redirects).

**`Error: The page needs JavaScript; use -s playwright or the default auto strategy`.**
`-s static` read a page that is empty or asks for JavaScript. Use `-s playwright`
or the default `-s auto`, which renders it with a local Chrome or Chromium. With
`auto` and no browser installed the message says that none was found: install
one, set `MARKITAI_BROWSER_EXECUTABLE` or run `markitai doctor --fix`. A short
page whose text is little next to a large script (the `quotes.toscrape.com/js/`
example) is converted with a warning under `-s static` and rendered by `auto`;
see [pages that need JavaScript](fetch.md#pages-that-need-javascript).

**A page's text is garbled, or a warning says its encoding could not be determined.**
The server's `Content-Type` and the page's `<meta>` agree with the bytes in nearly
every case. When neither fits, as with a GBK page labelled `utf-8` or one with no
label, Markitai reads the bytes like a local text file and warns when it had to
choose Windows-1252 over a plausible East Asian reading. Fetch the page with
`-s playwright` (the browser applies its own detection), or save it and convert
the file. See [character encodings](fetch.md#character-encodings).

**An output file for a URL has an unexpected name, or two URLs share one.**
A URL is named by its last path segment, with the value of an identifying query
parameter (`?id=…`, `?v=…`, `?p=…`) appended, so `…/item?id=8863` becomes
`news_ycombinator_com_item_8863.md`, and an X/Twitter post becomes
`<user>-status-<id>.md`. See [names from URLs](output.md#names-from-urls). For a
single URL, `-o chosen.md` sets the exact file name; an existing name is renamed
`….v2.md` unless `output.on_conflict` says otherwise.

**`Fetch strategy 'cloudflare' is not implemented` / `Cloudflare file conversion is not implemented`.**
The Cloudflare strategy and `-b cloudflare` backend are not available in this
build. Use `-s auto`, `static`, `playwright`, `jina` or `defuddle`, and the
default native file backend.

**`Remote fetching is disabled by policy`.** `-s jina` and `-s defuddle` send the
URL to a third-party service. They are refused when `--no-remote-fetch` or
`MARKITAI_NO_REMOTE_FETCH=1` is set or `fetch.remote_consent` is not `always`.
Local, private and credential-bearing URLs are never sent to these services.

**`Chromium is not installed; install Chrome/Chromium or set MARKITAI_BROWSER_EXECUTABLE …`.**
JavaScript pages, `-s playwright` and web screenshots need Chrome or Chromium.
Install one, set `MARKITAI_BROWSER_EXECUTABLE` to its executable, or run
`markitai doctor --fix` to download the official headless shell into
`MARKITAI_HOME/browsers/native`. An explicit `MARKITAI_BROWSER_EXECUTABLE` that
points to a missing file is not replaced automatically; fix or unset it first.
See [browser installation](browser-installation.md).

**`Warning: … need LibreOffice (soffice on PATH) …`.** Page images and page OCR
of Word, PowerPoint and spreadsheet files are exported through an installed
LibreOffice and the macOS PDF renderer. Without it the document's own text is
still converted and this one warning appears; install LibreOffice (macOS:
`brew install --cask libreoffice`) to get the pages, or pass `--no-screenshot`
/ `--no-ocr` (also for presets such as `rich`) to silence it. Only
`--screenshot-only` fails, with `Office screenshots require an installed
LibreOffice (soffice on PATH) …`, because nothing else would be written. See
[Office page rendering](office-rendering.md).

**OCR or PDF page images fail on Linux or Windows.** On-device OCR, PDF page
rendering and HEIF/AVIF decoding use macOS system frameworks and return an
explicit unsupported error elsewhere. A standalone PNG, JPEG, WebP, GIF, BMP,
TIFF or SVG image can still be read by a vision model with `--llm`; PDF page
images need macOS.

**`Error: ocr.lang is not supported by the installed macOS Vision text recognizer`.**
`ocr.lang` must be a language the macOS Vision recognizer supports, such as
`en`, `zh`, `ja` or `de`; see [local OCR](ocr.md#language-selection).

**`Error: Interactive configuration requires a terminal`.** `-I`, `config edit`
and `init` without `--yes` need an interactive terminal. In scripts use
`config set` or `init --yes`.

**`Error: Configuration file does not exist: …`.** `-c` must name an existing
file (except for `config set` and `config edit`, which create it). Run
`markitai config path` to see which file is in use; `config validate FILE`
checks a file. See [configuration](configuration.md).

**A batch was interrupted.** On macOS and Linux, run the same command again with
`--resume`; completed items are kept and unfinished items are retried. Resume
is not available on Windows. After Ctrl-C or SIGTERM the closing summary lists
what was done and then the items that were never started, for example
`Not processed 72 items: a.pdf, b.pdf and 70 more. Run the same command with --resume to continue.`
(exit status 130 or 143; `-v` lists every name). The resumed run starts with
`Resuming: 6 already done, 72 remaining` and its closing line counts all items
(`Done: 78/78 files`). `--quiet` leaves the summary out, and `--json` carries
only the items that finished.

**`Error: Native PDF conversion failed: the PDF is encrypted and needs a password to open …`.**
The PDF asks for a password before it can be read. Markitai has no password
option: remove the password with the tool that created the file, or print the
document to a new PDF, and convert that. A PDF that only restricts printing or
copying (an owner password, no password to open it) converts normally.

**A failed conversion left `.markitai/ownership/` in the output directory.**
Before converting, Markitai claims the output names with empty lock files under
`.markitai/ownership/members/`. They stay after a failed or a successful run on
purpose: removing a lock file while another run may be opening it would put the
two runs in different lock domains (see [output ownership](output-ownership.md)).
The files are empty and harmless; delete the whole output directory if it holds
nothing else you want. An input that can only end as a skip or an error before
anything is written, such as an image converted without `--ocr` or `--llm`, is
decided before the claim and leaves no such files.

**Hidden files and folders are not converted.** A directory batch skips
dot-files and dot-directories (`.git`, `.cache`), `node_modules` and Office lock
files (`~$report.docx`). Give a glob that spells the name out to include them,
for example `-g '.notes/**/*.md'` or `-g '**/node_modules/**/*.md'`; pointing the
command at such a directory itself (`markitai .notes -o out`) also works.

**A directory called `~` appeared, or `Error: Cannot expand ~ …`.** A leading
`~` in `-o` and in `output.dir` means your home directory, whether or not the
shell expanded it (`--output=~/x` and a value in `config.json` are never
expanded by the shell). Markitai expands it once, before anything is created; it
refuses a `~` it cannot expand (no `HOME`) instead of creating a directory with
that name. To use a directory that really is called `~`, write `./~`.

**There is no progress line.** A batch shows one status line, `[12/340] name  ETA
0:41`, only on Unix when stderr is a terminal that understands cursor control,
and not with `-q`, `--json` or `TERM=dumb`. Piped or redirected output carries only the
lines described above, unchanged. A single conversion that takes longer than two
seconds shows a spinner line the same way.

**`Error: Unsupported file format: '.xyz'`.** The message lists every recognized
extension. Rename files that have the wrong extension.

## Caches, logs and state

- `markitai cache stats` shows the model-answer and web-page caches;
  `markitai cache clear -y` empties them. `--no-cache` skips cached answers for
  one run but still saves fresh ones. See [cache](cache.md).
- File logs are off by default. Enable them with
  `--config-json '{"log":{"dir":"./logs"}}'` (or `MARKITAI_LOG_DIR`) and choose
  the level with `--log-level DEBUG` (any case: `debug` works too); logs never
  go to stdout.
- Everything Markitai stores lives under `MARKITAI_HOME` (default `~/.markitai`):
  `config.json`, `.env`, `cache.db`, `fetch_cache.db`, `learned_spa_domains.db`,
  `browsers/` and `serve/jobs/` history; the three databases follow
  `cache.global_dir` when you change it. Deleting a trial `MARKITAI_HOME`
  resets everything; batch state and reports stay in each output directory
  under `.markitai/`.
- Proxies come from `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` (and lowercase
  forms) and `NO_PROXY`, otherwise from the system's manual proxy setting. See
  [URL fetching](fetch.md#proxies).

## Reporting a problem

Include `markitai --version`, the platform, `markitai doctor --json`, the exact
command and the error text. `config list` hides keys and tokens by default; do
not add `--show-secrets` to a report.
