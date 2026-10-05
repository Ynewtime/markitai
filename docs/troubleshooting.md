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
| 2 | Usage error (unknown option, missing `-c`, `--json` without `-o`, nonterminal interactive mode), or a provider Batch job still pending |
| 10 | A directory or `.urls` batch finished with at least one failed item |
| 130 / 143 | Unix interruption by Ctrl-C / SIGTERM |

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
because nothing went wrong. Add `--ocr` (Vision on macOS; Paddle on Windows/Linux
with prepared models) or `--llm`
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
`--ocr` to read them. Windows/Linux require the local Paddle models first.
The terminal shows one line per document; the
report and `--json` keep one warning per page.

**`Warning: PDF pages 1-3: the text was read from the invisible OCR text layer laid over each page image …`.**
The PDF is a searchable scan: a scanning or OCR program (named in brackets when
the file says which) laid its recognized words, invisibly, over each page image.
Markitai used those words as the pages' text after checking that they sit on
the page and, when it can read the image, on its print. It did not recognize
the pages itself, so the program's recognition errors stay, and words that sit
on the print but say something else would not be noticed. When the warning says
the layer `could not be checked against the image`, the image is in a format
Markitai does not decode (JBIG2, CCITT fax, JPEG 2000) or the page has several
images. To read such pages with Markitai's own OCR instead, run with `--ocr` and
`ocr.per_page_routing` set to `false`; see [PDF OCR](pdf-ocr.md).

**`Warning: PDF page 2: an invisible OCR text layer lies over the page image, but it does not line up with the text in the image …`.**
The page carries hidden words that are not where the image's print is (a layer
from another page, a misplaced layer, or text hidden on purpose), so they were
not used and the page is treated as a scan; run with `--ocr` to read it.

**`Warning: PDF page 4: 2 text runs (37 characters) could not be decoded and were omitted; …`.**
Some text on the page is in a font whose characters cannot be identified (no
character map the reader knows, or one that maps its codes to nothing usable).
That text is left out and the rest of the page is kept. A variant says that
characters `show as U+FFFD`: they stay in place as the replacement character;
another that a font on the page names its glyphs by index only. Run again with
`--ocr` to read such a page from its image instead. When most of a page
cannot be decoded, it is reported as `native text was not recovered
(suspected_garbled_text)` instead, as above. Japanese, Chinese and Korean PDFs
whose fonts are not embedded (Shift-JIS, GBK, Big5, UHC and Unicode CMaps) are
read directly and need neither.

**The first local OCR is slow.** On macOS, Vision may compile and cache
recognition models after installation or an update. Startup and per-page time
depend on the OS, image and language; earlier measurements are not a timing
guarantee. A sandbox that blocks the Vision cache can cause `missingError`.
On Windows/Linux, run `markitai doctor` to check Paddle models and explicitly
use `markitai doctor --fix` to prepare missing models before offline use.
See [local OCR](ocr.md) for backend and language limits.

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
Documents served by a subscription runtime carry its own notice instead of this
one.

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

**`Error: HTTP 403 for https://www.zhihu.com/…: Zhihu refuses automated clients; open the page in your browser and save it …`.**
The site is known to turn away programs (Zhihu, WeChat, Douban, Bilibili, Weibo,
Toutiao, Reddit, Quora, Stack Overflow, Medium and Hashnode are named; any site behind a
Cloudflare bot check says so). Markitai does not pretend to be a browser, solve
the check or sign requests, so the page stays refused. What works: open the
page in your own browser, save it (File > Save Page As…, "Webpage, HTML Only"
or "Webpage, Complete"; MHTML is not read) and convert the saved file, which
Markitai reads with the site's own reader when it has one (Zhihu, WeChat,
cnblogs, Jianshu, OSCHINA, Bilibili, Douban reviews; see [reading sites](html.md#reading-sites));
or, for a site that wants your account, give the local browser your own
signed-in cookies with `fetch.playwright.cookies` and use `-s playwright` (see
[your own cookies](fetch.md#your-own-cookies-for-the-local-browser)). A JSON body
the site sent with the refusal is quoted in the message as `the site said: …`.

**`Error: <Site> served a verification page instead of the content`** (or **`a login page`**).
The site answered with status 200 but its page is a security or human check
(WeChat's `环境异常`, Douban's `sec.douban.com`, Weibo's visitor check, Reddit's
"Prove your humanity", Toutiao's script challenge, Zhihu's `安全验证` login
check) or only a request to log in, not the article. `auto` tries the local
browser first (WeChat articles are read that way); when that is shown the same
check, or fails as well (the message then ends with the browser's own reason),
the page can only be read from a copy you saved in your browser, as above.

**`The jina service was shown Zhihu's verification page instead of the content`** (or `The defuddle service …`, `… a challenge page …`, `… received HTTP 403 from the site instead of the content`).
A remote service read the site's refusal, not the page; that service counts as
failing, the next one is tried, and nothing is written. When none reads the page
the message says once what works, the same advice as for the static refusal
above: save the page from your own browser, or use your own cookies with the
local browser. See
[remote readings that are refusals](fetch.md#remote-readings-that-are-refusals).

**A warning `The jina service said: This is a cached snapshot of the original page …`.**
Jina answered from its own cache, which can be old or wrong (on 2026-10-02 its
snapshot of `example.com` was a test page). Run with `--no-cache` (or
`--no-cache-for <pattern>`, or set `fetch.jina.no_cache`): when Markitai's page
cache is bypassed for a URL, Jina is sent `X-No-Cache: true` too. defuddle.md
has no such opt-out; its answers may be up to five minutes old.

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

**`Cloudflare needs an API token and an account ID: …`.** `-s cloudflare`
(Browser Rendering) and `-b cloudflare` (Workers AI file conversion) run in your
own Cloudflare account. Create a token at dash.cloudflare.com/profile/api-tokens
with Account / Browser Rendering / Edit and Account / Workers AI / Read, copy the
account ID from the account's home page, then set `fetch.cloudflare.api_token`
and `fetch.cloudflare.account_id` (a value or an `env:NAME` reference) or export
`CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`. `Environment variable not
found: NAME` means an `env:NAME` reference names a variable that is not set.
`HTTP 401/403 from the cloudflare service: …` means Cloudflare refused the token
(check its permissions and the account ID). `-b cloudflare` uploads the file;
formats Workers AI does not read, and files converted with `--ocr` or
`--screenshot`, keep the native reader. See [URL fetching](fetch.md#remote-services).

**`Remote fetching is disabled by policy`.** This policy blocks remote
extraction services, not all network access: source URLs, the local browser
and enabled model requests may still use the network. `-s defuddle`, `-s jina` and
`-s cloudflare` send the URL to a third-party service. They are refused when
`--no-remote-fetch` or `MARKITAI_NO_REMOTE_FETCH=1` is set or
`fetch.remote_consent` is `never`; under `ask` a strategy chosen with `-s` runs,
while one set in `fetch.strategy` needs a yes at the terminal. Local, private and
credential-bearing URLs are never sent to these services, and a configured remote
strategy also honours `fetch.policy.local_only_patterns`.

**A page fails with the default `auto`, and a remote service might read it.**
`auto` never sends a URL to a remote service by default, although `config list`
shows `fetch.remote_consent` as `always` (the reference's default, which this
build does not take as an opt-in). Write it yourself to opt in:
`markitai config set fetch.remote_consent ask` asks once per run on a terminal
before the first remote attempt (without a terminal, or with `--quiet`, the run
skips them and says so once); `always` tries defuddle, Jina and (with
credentials) Cloudflare after the local strategies fail, with a one-time notice.
They are not tried for a 404 or 410, or for local, private or credentialed URLs.
A service that is shown the site's verification, login or challenge page fails
like any other. When all fail, the message keeps the local failure first and
adds `remote services failed as well (…)`, with what works said once. See
[strategy order and remote fallback](fetch.md#strategy-order-and-remote-fallback).

**`Chromium is not installed; install Chrome/Chromium or set MARKITAI_BROWSER_EXECUTABLE …`.**
JavaScript pages, `-s playwright` and web screenshots need Chrome or Chromium.
Install one, set `MARKITAI_BROWSER_EXECUTABLE` to its executable, or run
`markitai doctor --fix` to download the official headless shell into
`MARKITAI_HOME/browsers/native`. An explicit `MARKITAI_BROWSER_EXECUTABLE` that
points to a missing file is not replaced automatically; fix or unset it first.
See [browser installation](browser-installation.md).

**`Warning: … need LibreOffice (soffice on PATH) …`.** Page images and page OCR
of Word, PowerPoint and spreadsheet files are exported through an installed
LibreOffice and the selected PDF renderer (CoreGraphics on macOS by default,
built-in hayro on Windows/Linux). Without it the document's own text is
still converted and this one warning appears; install LibreOffice (macOS:
`brew install --cask libreoffice`) to get the pages, or pass `--no-screenshot`
/ `--no-ocr` (also for presets such as `rich`) to silence it. Only
`--screenshot-only` fails, with `Office screenshots require an installed
LibreOffice (soffice on PATH) …`, because nothing else would be written. See
[Office page rendering](office-rendering.md).

**OCR or PDF page images fail on Linux or Windows.** PDF rendering uses the
built-in hayro renderer; it does not need a separately installed PDF program.
Local OCR uses Paddle and requires prepared model files. Check `markitai doctor`
and [model preparation](ocr.md#the-portable-engine-windows-and-linux).
Page-size limits, damaged/encrypted PDFs and unsupported model languages remain
explicit errors. Office pages additionally need LibreOffice. HEIF/AVIF decoding
remains macOS-only; convert those images to PNG/JPEG first on other platforms.

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

**A batch was interrupted.** On macOS, Linux and Windows, run the same command
again with `--resume`; completed items are kept and unfinished items are retried.
Keep the same command and configuration. Resume compares saved feature switches,
paths and directory discovery settings; it does not compare every model, prompt
or output-profile setting. To reprocess after such a change, use a new output
directory without `--resume`. On Unix, after Ctrl-C or SIGTERM the
closing summary lists
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

**There is no progress line.** A batch shows one status line, `[12/340] name  elapsed
0:41` (the time taken so far), when stderr is a supported terminal (including Windows consoles),
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
- Default application state lives under `MARKITAI_HOME` (default `~/.markitai`):
  `config.json`, `.env`, `cache.db`, `fetch_cache.db`, `learned_spa_domains.db`,
  `browsers/` and `serve/jobs/` history; the three databases follow
  `cache.global_dir` when you change it. Deleting a trial `MARKITAI_HOME`
  resets that trial state; configured custom paths, output assets, batch state
  and reports outside it remain. Reports and recovery files normally live in
  each output directory under `.markitai/`.
- Proxies come from `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` (and lowercase
  forms) and `NO_PROXY`, otherwise from the system's manual proxy setting. See
  [URL fetching](fetch.md#proxies).

## Reporting a problem

Include `markitai --version`, the platform, `markitai doctor --json`, the exact
command and the error text. `config list` hides keys and tokens by default; do
not add `--show-secrets` to a report.
