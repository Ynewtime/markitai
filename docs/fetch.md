# URL fetching and extracted-page caching

The native `static` strategy and anonymous `auto` strategy fetch HTTP(S) directly. They extract
HTML, plain text and Markdown, or dispatch supported downloaded documents to
the local format readers. Requests negotiate
`Accept: text/markdown, text/html;q=0.9, */*;q=0.5`, follow at most ten redirects
and bound response bodies to 100 MiB. HTML links resolve against the final
response URL. Both initial and conditional responses use the same charset and
format decoder. Explicit plain-text and Markdown MIME types keep literal HTML
examples as text rather than triggering HTML challenge detection.

For an unknown anonymous authority, `auto` starts with the static path when screenshots are not requested, with
local browser fallback for recognized JavaScript/challenge or empty-HTML quality
failures and for script-rendered shells (see
[pages that need JavaScript](#pages-that-need-javascript)). Capture requests first perform one bounded, unconditional static GET
to distinguish PDF downloads from browser pages. A PDF is handed directly to
the native document pipeline, including for extensionless and redirected URLs;
other successful responses continue through browser capture. Ordinary HTML
capture therefore adds a static request. Probe transport, HTTP-status, body-read
and size-limit failures propagate instead of being hidden by browser fallback.
The probe does not reuse or admit HTML cache entries, so a fresh cached HTML row
cannot conceal a URL that now downloads a PDF. The `playwright` strategy uses
[native Chromium CDP](browser.md) for JavaScript and captures. Learned routing is
described below; the complete reference fallback policy remains unfinished. `jina` and
`defuddle` keep explicit remote-consent and target checks and do not use this
cache. Explicit browser retrieval also hands initial PDF document responses
to the native reader using bytes streamed from the same CDP session. Remote
strategies retain their selected extraction service and do not use that handoff. Unsupported strategies fail before cache lookup, so an old page cannot
make an unsupported strategy appear to work.

## X post addresses

A post or Article on X is fetched from X itself. Before any strategy runs, a
status or Article address on `twitter.com`, `mobile.twitter.com`, `www.x.com`,
`www.twitter.com` or one of the mirrors `fxtwitter.com`, `vxtwitter.com`,
`fixupx.com`, `fixvx.com` and `twittpr.com` becomes
`https://x.com/<user>/status/<id>` (`/article/<id>` for an Article), without
share-tracking query parameters, a fragment or a trailing `/photo/1` or language
segment, and `/i/web/status/<id>` becomes `/i/status/<id>`. A mirror answers a
non-browser client with a redirect page, which would otherwise be written as a
17-word success. The output keeps the address the caller gave as its `source`;
the cache is keyed by the canonical address. Other X pages (profiles, search)
and addresses with userinfo or an explicit port are left alone. The mirrors'
hosts are only recognized in order to avoid contacting them: no X address is
sent to FxTwitter, oEmbed, defuddle or Jina by any strategy but an explicit
`defuddle`/`jina` request. The post reader is described in
[HTML conversion](html.md).

## Character encodings

HTML bytes become text by a BOM, then the first declaration they fit: the HTTP
`charset`, then a `<meta>` declaration found by the HTML standard's prescan (see
[text encodings](formats.md#text-encodings)), then UTF-8. A declaration that
fits always wins, so correct pages are read as before; a single-byte HTTP
charset such as `windows-1252` is trusted because every byte sequence fits it.
When no declaration fits, which is typically a server that sends `charset=utf-8`
for a GBK page or a page that declares nothing and is not UTF-8:

- a declared legacy encoding (an invalid sequence in GBK, say) is kept, the bad
  sequences become U+FFFD, and a warning says so;
- UTF-8 with only a sprinkling of bad bytes (fewer than 1 % of its non-ASCII
  bytes, as in the reference) stays UTF-8, with a warning;
- anything else is read like a local text file: GB18030 (GBK, GB2312), Big5,
  Shift_JIS, EUC-JP and EUC-KR by the scoring described under text encodings,
  with a Latin-1 page read as Windows-1252. The result is silent when the
  reading is decisive and carries a warning when Windows-1252 was kept although
  an East Asian reading was plausible.

A page read with a warning is not cached (see below). Very short pages are often
undecidable, and there is no option to name the encoding. Plain-text responses
(`text/*` other than HTML) follow a BOM, then the HTTP charset, then UTF-8, with
replacement characters and no detection.

## Failures, retries and redirects

A failed status reads `HTTP <status> for <URL>` and, for the common ones, a hint:
404 and 410 "the page may have been removed or is not public", 401 and 403 "the
site refused access; it may block automated clients or need a login", 429 "rate
limited; try again later" and 5xx "the site had a server error". The URL has no
credentials or fragment, and query values are replaced by `REDACTED` for the
usual secret names (`token`, `key`, `secret`, `password`, `signature`,
`credential`, `auth`, `sig`, `sid`, `session`, `code` and similar) and for long
opaque values. It is cut at 200 characters. The message keeps the status first
and the `fetch_error` code, so the [web interface](web-ui.md) still recognizes
it. A remote service's own failure reads `HTTP <status> from the <service> service`
without a hint, because its status says nothing about the page.

A GET whose connection cannot be established because the peer reset it or cut
it off, such as a TLS handshake that ends early (`tls handshake eof`) or
`Connection reset by peer` while connecting, is repeated up to twice, after
250 ms and 750 ms. Nothing had been sent, so a repeat cannot duplicate work.
Nothing else is retried: not a timeout, a refused connection, a name that does
not resolve, a certificate failure, a connection that fails after the request
went out (a reset or a clean close without an answer) or any HTTP status, so a
4xx or 5xx answer is reported at once. The retry covers the page request, the capture probe and the
remote strategies; the conditional revalidation request falls back to its
unconditional request, which has the retry.

A `<meta http-equiv="refresh">` page is followed like an HTTP redirect when it
refreshes within two seconds to another http(s) URL and its own content is
nearly empty (an empty body or fewer than 30 words, such as "Redirecting…"). The
target resolves against the response URL and must be `http` or `https`; a
slower refresh, a refresh to the page itself, a `<noscript>` refresh and other
schemes are not followed, and a page with real content keeps it. Up to five
refreshes are followed; more fail with `Too many <meta> refresh redirects`.
Unlike an HTTP redirect within one origin, a refresh carries none of the
credentials in the original URL, even to the same origin, and credentials
written in the target are dropped, because a page's markup is not the reader's
to speak for. Nothing is kept of the first response's validators, because they do
not describe the page finally read; the entry expires by TTL. JavaScript
redirects are not followed.

## Pages that need JavaScript

Static extraction recognizes three kinds of page that scripts fill in. Their
text may say so (`Please enable JavaScript`) or be empty, or the page may keep
a little text (a title and a `Login` link) next to a payload of script. A page
is a *script-rendered shell* when its Markdown has fewer than 30 words (link
destinations do not count) and, beside at least one script, its markup has
one of: inline script text of 3,000 bytes or more (structured `ld+json` and
templates excluded), an empty mount point (`#root`, `#app`, `#__next`,
`#__nuxt`, `#___gatsby`, `#svelte`, `#react-root`, `#q-app` or `<app-root>`
holding no text of its own), or a `<noscript>` text about JavaScript.

- `auto` renders such a page with the local browser, as it does an empty page
  or one that asks for JavaScript. Only a page that says so in its own text
  teaches [learned routing](#learned-browser-routing); a page recognized by the
  size of its scripts never does. Without a browser, or when rendering fails
  with a fetch error, the static text is returned with a warning (and the
  reason); a page that asks for JavaScript, or is empty beside its scripts,
  fails without a browser with `The page needs JavaScript, and no local browser
  (Chrome or Chromium) was found; …`.
- An explicit `-s static` fails such a page with `The page needs JavaScript; use
  -s playwright or the default auto strategy`, and keeps the text of a shell
  that has some, with a warning saying the same. An empty page without any
  script evidence keeps its plain `HTML contains no extractable content`.
- A shell result carries a warning, so it is not stored in the page cache.

The thresholds are conservative but heuristic: a short page with a large inline
script, such as a parked domain, is also sent to the browser by `auto`, which
returns the same text slower. `-s static` still converts it.

## Learned browser routing

An anonymous automatic request can remember that an authority serves a short
JavaScript shell. Learning requires both that static classification and a
successful browser response containing useful extracted text. Empty pages,
challenges, failed navigation, PDF downloads and capture-only empty output do
not teach a route. Unlike the reference's early recording, native learning waits
for the browser result so a failed fallback cannot make the routing decision
persistent.

Later automatic requests to that authority start with the available browser,
avoiding another static request or anonymous page-cache lookup. Once selected,
browser HTTP, timeout and resource failures remain errors. If no browser is
available, ordinary static retrieval remains available. Explicit `static`,
`playwright` and remote strategies keep their selected behavior. The CLI treats
`-s auto` as automatic intent, exactly like its default, and both use learned
hints (reference `cli/main.py` also clears explicit provenance for auto). A
low-level caller supplying `ConvertContext.explicit_fetch_strategy=Some("auto")`
can suppress hints while still recording a successful fallback; this matches the
reference fetch function's separate explicit-intent argument.

Authorities are normalized host plus nondefault port, including IPv6 brackets;
different ports remain separate. Following the reference, scheme is not part of
this routing identity. URL paths and queries are not persisted. Configured
browser credentials, cookies, extra headers and non-isolated session mode bypass
anonymous learning and decisions. URLs with userinfo or recognizable credential
query keys also bypass learning; query-key screening is not a universal secret
detector. No cookies, HTML, Markdown, screenshots or authenticated sessions are
saved in this store.

The independent `learned_spa_domains.db` uses the configured state root and honors
`MARKITAI_HOME` for the default path. Document-cache disabling, read bypass and
patterns do not disable route knowledge. A hit advances its count and last-use
time; an entry expires only after more than 30 days without a hit. Listing shows
expired entries until a lookup or successful learning prunes them. Storage is
limited to 4,096 authorities of at most 1,024 UTF-8 bytes each and a 16 MiB SQLite
database; this is not a bound on process memory or all temporary filesystem
overhead. New files are private, and symlink or multiply linked database files
are rejected. There is no migration from the reference JSON store.

Unavailable storage leaves successful fetching intact with a fixed warning.
Management failures are explicit and sanitized. [Cache commands](cache.md#learned-domain-management)
inspect and clear routing knowledge independently. This implementation adds no
browser session pool or browser-document cache.

## Downloaded PDF media

When OCR, screenshots or the screenshot-only flag is requested, the static path
passes an owned, bounded byte buffer and the final response URL to the core PDF
pipeline. It performs no second GET and does not serialize those bytes into
metadata or cache rows. The original URL remains the conversion/cache identity;
the core redacts the final URL before exposing it. The screenshot-only flag alone
does not enable PDF rendering: classification can still return native text.
Ordinary PDF conversion without these flags retains its existing reader path.

Explicit text MIME and HTML responses take precedence over PDF-looking names
or bytes. `application/pdf` and `application/x-pdf` identify a PDF representation
even when it is corrupt. Generic/absent MIME or a final `.pdf` path require a PDF
version header within the first 1,024 bytes. A path suffix alone is insufficient.
This preserves Rust's literal text behavior; the Python reference instead treats
`text/plain` as generic and also uses Content-Disposition filenames. That broader
filename precedence is not implemented here.

An accepted 200 PDF invalidates the old HTML cache entry before native parsing,
OCR or rendering. Later PDF failure therefore neither redownloads the body nor
restores obsolete HTML. PDF bytes, OCR text and screenshots are never admitted to
the extracted-page cache. Default static conditional-failure retry behavior for
responses that have not been accepted as a deferred PDF remains unchanged.

For screenshot-only requests without LLM or an output directory, static/auto
must classify the response before deciding whether the request is a PDF text
conversion or an unobservable browser capture. Non-PDF content is rejected before
starting the browser. This makes classification HTTP errors observable before
the missing-output error. Explicit browser and credentialed auto requests must
first classify their authenticated response; a rendered HTML result then enforces
the same output requirement, while PDF bytes follow the native PDF contract.
Remote services reject that combination before making their selected fetch.

## Stored results and identity

The cache stores extracted Markdown and page metadata before output names,
profiles, frontmatter generation and optional LLM processing. A cache hit goes
through those later stages normally. It never represents an LLM cache hit or
removes the cost of a subsequent model request. The internal fetch-hit flag
feeds the CLI's existing `fetch_cache_hit` field without adding a field to the
bindings conversion JSON.

Only successful HTML and `text/*` responses other than XML are eligible.
Admission also requires nonblank Markdown, no owned asset bytes and no
extraction warnings. The warning restriction avoids replaying content without
its original warning because warnings are not stored in the reference table
shape. Downloaded PDF, Office, email and other local-reader documents bypass
this page cache, including when they happen to contain no images. Their assets
remain available to normal output processing. A successful refresh that changes
into an ineligible representation removes its old page-cache row, preventing a
later 304 or still-fresh TTL from replaying the previous representation. Failed
or rejected refreshes retain the old row instead, except that an accepted deferred
PDF has already changed the representation even if its later reader fails.

The key hashes the extraction namespace `native-fetch-<package version>-r<revision>`
(now `native-fetch-1.3.0-r2`), a NUL separator and the exact original URL. Rows
hold extracted Markdown, so a new release, or a revision bumped when extraction
changes between releases, does not replay an older extraction; older rows stay
readable in statistics and expire by TTL or capacity.
Query strings and fragments therefore remain part of identity. An explicitly
selected non-`auto` strategy adds a NUL-separated strategy suffix; a strategy
inherited from configuration does not. Default/configured `auto` and `static`
share a key, explicit `-s static` has its own key, and explicit `-s auto` stays
unscoped. Redirect URLs do not replace the original identity. The native
namespace prevents reuse of reference-Python extracted results.

## Reuse and revalidation

| Saved state | Behavior on the next invocation |
| --- | --- |
| No HTTP validator, younger than TTL | Return the saved page without an HTTP request. |
| No validator, expired or zero TTL | Fetch normally and save an admissible result. |
| ETag or Last-Modified | Send every available conditional header on every invocation, regardless of TTL. |
| Conditional 304 | Return the saved page as a fetch hit; update access time only. |
| Conditional usable 200 | Decode and replace the page; this is a fetch miss. |
| Conditional failure or unusable response | Retry without conditional headers; do not silently serve the old page. |

The TTL defaults to 86,400 seconds and measures age from creation, with expiry
at `age >= TTL`. Reads update access time, not creation time; a 304 does not
replace saved validators or restart the TTL. An unsolicited 304 is an error.
Expired rows remain visible in cache statistics until replaced or evicted.

This application cache does not apply HTTP `Cache-Control`, `Expires` or `Vary`
directives, including `no-store`. Set `cache.enabled=false` when fetched content
must not be persisted; `--no-cache` still saves admissible fresh results.

Empty content, failed extraction and recognized HTML challenges cannot replace
a good row. Challenge checks examine HTML structure and visible non-code text;
literal vendor names in plain text, HTML code examples or an ordinary CAPTCHA
widget are insufficient. Pages that ask for JavaScript or are empty fail static
extraction, and `auto` may then select the local browser; a short page whose
scripts outweigh its text is converted with a warning and so is never stored
(see [pages that need JavaScript](#pages-that-need-javascript)).
These checks do not implement complete quality classification. If a conditional
request and its unconditional retry both fail, the operation fails while the
old row remains intact. The reference's narrower stale-page fallback after a
failed full strategy chain is not implemented.

## Controls, capacity and privacy

| Setting | Effect |
| --- | --- |
| `cache.enabled=false` | No conversion-time fetch-cache reads, writes or creation. |
| `cache.no_cache=true` | Bypass reads and validators; still refresh from admissible fresh content. |
| `cache.no_cache_patterns` | Matching URLs use the same bypass-and-refresh behavior. |
| `cache.fetch_ttl_seconds` | Reuse window for entries without validators. |
| `cache.global_dir` | Parent of `fetch_cache.db`; default state paths honor `MARKITAI_HOME`. |
| `cache.max_size_bytes` | Logical Markdown capacity of this database, default 512 MiB. |

URL patterns match the full URL, scheme-free URL, authority/hostname and final
path segment. Scheme and hostname comparisons ignore case; path and filename
case is retained. Patterns are trimmed and support an optional leading `**/`.
URL colons are not treated as document-stage suffixes.

The SQLite store uses WAL, normal synchronization and a 30-second busy timeout.
Replacement and LRU eviction share an immediate transaction. Logical capacity
counts UTF-8 Markdown bytes, excluding metadata, SQLite files and WAL overhead.
Fetch and LLM stores each have their own capacity; this is not a combined disk
quota. The store bounds row allocation and lazily creates the database only
when saving an admissible page. Cache errors add a fixed warning to successful
content instead of exposing paths, SQL, headers, URLs or page data.
Conversion reads require writable access to update `accessed_at`. If opening the
store or updating that timestamp fails, even an otherwise reusable page becomes
a cache miss and fetching continues over the network.

The database is plaintext. It stores original/final URLs and metadata as well
as Markdown; URL credentials and query tokens can therefore be stored. A
hashed key does not make its contents secret-free. Direct local fetching keeps
its existing trusted-session behavior; this cache is not suitable for reuse
across anonymous service authority boundaries without additional policy.

## Verification boundary

The loopback tests in `fetch.rs` check actual request counts and headers, TTL and
zero TTL, 304 and replacement behavior, redirects and character encoding,
bypass refresh, strategy provenance, disabled and broken storage, malformed
status/empty content, challenge preservation, legitimate CAPTCHA mentions and
asset-bearing downloaded documents. `fetch/robustness_tests.rs` adds, against
loopback servers, the failure messages and their redaction, the retry of a TLS
handshake that ends early (a listener that accepts and closes) and the failures
that are never retried (statuses, resets and clean closes after the request,
refusals, timeouts), wrong and missing character
encodings (GBK under a `utf-8` header, none at all, nearly valid UTF-8, Latin-1),
script-rendered shells and JavaScript-only pages under `static` and, through the
decision function with a stand-in browser, under `auto`, and `<meta>` refresh
following, its limits and credential handling. They use temporary configured
state paths;
the older direct-fetch test explicitly disables caching. Storage and CLI/API
integration tests exercise their respective boundaries separately. No tests
need provider credentials or the user's real state directory.

The PDF handoff tests additionally check byte identity, exact GET counts,
redirects, MIME/header precedence, unchanged default extraction, stale HTML
invalidation before downstream failure, auto-probe TTL bypass, output capability
checks and status/size rejection. The public conversion tests separately cover
native page media and model input after this private handoff.

[Decision 0003](decisions/0003-persistent-fetch-cache.md) records the detailed
reference comparison, storage contract and deferred browser, remote-cache,
owned-asset, stale-fallback and request-coalescing work.

## Browser HTTP credentials

When `fetch.playwright.http_credentials` is non-null, or browser cookies/extra
HTTP headers are nonempty, `auto` selects the browser
before consulting anonymous page cache or sending a PDF classification probe.
This avoids reusing a public page for a configured private identity. Browser
responses continue to bypass static cache reads and writes; removing credentials
can reuse an existing anonymous row. Explicit `static` does not consume browser
credentials and does not switch strategy on HTTP 401. Authentication failure
does not send the page or credentials to a remote fetch service.

The browser answers bounded Basic challenges only within the configured exact
origin. Omitting origin binds credentials to the initial scheme/host/effective
port; this intentionally narrows the reference's unrestricted default. The
reference also tries anonymous static retrieval first in auto mode; the native
authenticated path intentionally selects its configured identity first. Initial
PDF document responses, including redirects and extensionless attachments, stream
from that session without a second GET and retain `playwright` strategy metadata.
Proxy authentication and persistent sessions remain separate capabilities. See [browser contracts](browser.md).

## Proxies

Static fetch, remote extraction requests and the native browser share one
decision, following the reference. The first nonempty of `HTTPS_PROXY`,
`HTTP_PROXY`, `ALL_PROXY`, `https_proxy`, `http_proxy` and `all_proxy` is used
for every scheme; a value without a scheme means HTTP. An environment proxy
prevents any operating-system read. Otherwise the manual system proxy is read
once per process, only when a request is not already direct or when a browser
launches: macOS `scutil --proxy` by key (the reference's line-order parser misses
the usual sorted output), the Windows user Internet Settings registry values,
and on Linux KDE (`kreadconfig6`/`kreadconfig5`, preferred for mixed markers) or
GNOME/Unity through individual `gsettings get` keys that exclude stored
passwords. Automatic/PAC/WPAD modes, SOCKS-only and authenticated desktop
proxies, network probes and credential stores are never used. One second bounds
all setting subprocesses, whose output is limited and process group reaped. Reads through
the real desktop tools are recorded for GNOME and KDE 5
([R38](validation/linux-desktop-proxy-round38.md)) and KDE 6
([R49](validation/linux-kde6-proxy-round49.md)).

`NO_PROXY` (or `no_proxy` when it is unset or empty) always applies; a system
exception list applies only with its system proxy. `*` matches everything,
`.name`/`*.name` match subdomains only, other names and addresses match exactly,
and CIDR blocks match addresses. Each unsupported entry, such as `<local>`, a
port-qualified host, another wildcard or malformed CIDR, is ignored on its own,
as the reference ignores it. Loopback hosts are always direct. Proxy settings
above 64 KiB are rejected. Static fetch accepts credentials in an environment
proxy URL; a SOCKS proxy there is an explicit unsupported error, because only
the browser can use it. Model and Provider Batch clients keep their provider
environment handling. The Windows registry reader follows the documented API but
has not been compiled or executed on Windows in this project.
