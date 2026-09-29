# URL fetching and extracted-page caching

The native `static` strategy and anonymous `auto` strategy fetch HTTP(S) directly. They extract
HTML, plain text and Markdown, or dispatch supported downloaded documents to
the local format readers. Requests negotiate
`Accept: text/markdown, text/html;q=0.9, */*;q=0.5`, follow at most ten redirects
and bound response bodies to 100 MiB. HTML links resolve against the final
response URL. Both initial and conditional responses use the same charset and
format decoder. Text decoding follows a BOM or HTTP charset, otherwise UTF-8;
malformed sequences use replacement characters. HTML meta-charset prescanning
and heuristic encoding detection are not implemented. Explicit plain-text and
Markdown MIME types keep literal HTML examples as text rather than triggering
HTML challenge detection.

For an unknown anonymous authority, `auto` starts with the static path when screenshots are not requested, with
local browser fallback for recognized JavaScript/challenge or empty-HTML quality
failures. Capture requests first perform one bounded, unconditional static GET
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

The key hashes `native-fetch-v1`, a NUL separator and the exact original URL.
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
widget are insufficient. Obvious short JavaScript shells fail static extraction;
`auto` may then select the local browser.
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
asset-bearing downloaded documents. They use temporary configured state paths;
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
all setting subprocesses, whose output is limited and process group reaped.

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
