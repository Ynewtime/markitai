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
[native Chromium CDP](browser.md) for JavaScript and captures. Learned routing,
the [strategy order](#strategy-order-and-remote-fallback) and the opt-in remote
fallback are described below. `defuddle`, `jina` and `cloudflare` keep their
consent and target checks and do not use this cache. Explicit browser retrieval
also hands initial PDF document responses
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
sent to FxTwitter or oEmbed, and it reaches defuddle, Jina or Cloudflare only
through an explicit remote strategy or an `auto` fallback the user opted in to
(below). `auto` reads an X post from the static page with the X post reader,
usually in one or two seconds; the `x.com` and `twitter.com` entries of the
default `fetch.fallback_patterns` do not make it start with the browser (see
[strategy order](#strategy-order-and-remote-fallback)). The post reader is
described in [HTML conversion](html.md).

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
it. A remote service's own failure reads `HTTP <status> from the <service> service`,
then the reason the service gave in JSON (cut to 160 characters) and a hint for
the common refusals (a Jina key for 401/402/429/451, the Cloudflare token's
permissions for 401/403, `rate limited; try again later` for 429); it names
neither the page nor the endpoint, and any token or account id the service
echoed is replaced by `REDACTED`.

A site that is known to turn automated clients away is named in the hint
instead of the generic text, with what does work: for a 401, 403, 418 or 429
from Zhihu, WeChat, Douban, Weibo, Toutiao, Reddit, Quora, Stack Overflow,
Medium or Hashnode the message reads, for instance, `HTTP 403 for
https://www.zhihu.com/…: Zhihu refuses automated clients; open the page in
your browser and save it (File > Save Page As…, 'Webpage, HTML Only'), then
convert the saved file; or give the local browser your own logged-in cookies
for zhihu.com (fetch.playwright.cookies, see docs/fetch.md)`. The same text
is given when the local browser is refused (`-s playwright`, or a browser
fallback of `auto`), whose own message `Browser navigation returned HTTP 403` is
rewritten to this form. An answer with `cf-mitigated: challenge` from any other
site says that its Cloudflare bot check refused the client. A client error
other than 404 and 410 whose body (read up to 8 KiB) is a JSON refusal, such as
Zhihu's `{"error":{"message":"…","code":40362}}`, adds the site's own words,
cleaned and cut to 120 characters, as `; the site said: … (code 40362)`. The
status stays first in all of them.

A page that is a site's verification or security check but is served with
status 200 is a failure as well, never a short Markdown "page": WeChat's
`环境异常 / 去验证` page (the answer to a static client), Douban's `sec.douban.com`
script check, Weibo's `Sina Visitor System`, Reddit's `Prove your humanity`,
Toutiao's script challenge page and Zhihu's `zse-ck` interstitial, its
`account/unhuman` check and its `安全验证 - 知乎` security check. The message
reads `<Site> served a verification page instead of the content: it … ; open the
page in your browser and save it …`. A site that expects a login (Zhihu, Douban,
Weibo, Reddit) and shows little more than a request to log in (`请您登录后查看…`,
`Log in to continue`; at most 60 words) reads `<Site> served a login page instead
of the content: …`. `auto` renders such a page with the local browser first,
which is how WeChat articles are read; when the browser fails or is shown the
same check, the site's message is the failure (a browser failure of another kind
is added in parentheses).

The same judgement applies to every reader: the static client, the local
browser, Cloudflare Browser Rendering (all on the page's markup) and defuddle
and Jina Reader (on the Markdown and title they return, and the address Jina says
it read). Markup also shows a challenge widget; Markdown cannot, so there a
challenge title (`Just a moment…`, `Attention Required`, `Security verification`,
`Verify you are human`) over at most 60 words of text counts as a challenge,
while an article with such a title stays a page. See
[remote readings](#remote-readings-that-are-refusals).

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

## Sites that refuse automated clients

Markitai reads what a site serves to an ordinary client and what its own
browser rendering shows. It does not make a client look like a person's
browser, solve a challenge or verification, sign requests, rotate identities or
get past a login or paywall, so a site that refuses automated access stays
refused, with a message that says so. A page survey of 2026-10-02 (public
pages, one or two per site, no login and no cookies, 2 s apart; about 90
requests) found:

| Site | Default `auto` reads | Why not, and what works |
|---|---|---|
| Juejin, CSDN, Jianshu, cnblogs, SegmentFault, sspai, 36Kr, Huxiu, InfoQ China, V2EX, dev.to, GitHub Discussions, Hacker News, Weibo (`weibo.com/2/detail/comos:…` article pages) | yes, from the static page | [site readers](html.md#reading-sites) clean up cnblogs, Jianshu, 36Kr and (with the browser) WeChat |
| WeChat articles (`mp.weixin.qq.com`) | yes, with the local browser | a static client is sent to a verification page; `auto` renders the page in the local browser and reads it with the WeChat reader; `-s static` fails naming the verification |
| Bilibili columns and `opus`, OSCHINA, Yuque, public Notion pages | yes, with the local browser | rendered by scripts; the text is read but may carry page chrome (menus, comments, recommendations) |
| Zhihu (answers, `zhuanlan.zhihu.com/p/…`, questions) | no | a static client gets a 403 script interstitial (`zse-ck`); the local browser gets a JSON refusal (code 40362) or a redirect to a security check that asks for a login; Jina Reader was shown the same check (`安全验证 - 知乎`, "请您登录后查看更多专业优质内容"), which is a failure, not the page. Save the page from your own browser and convert the file ([Zhihu reader](html.md#reading-sites)), or give the local browser your own cookies (below) |
| Douban notes | no | a script check (`sec.douban.com`) that the local browser fails in this version |
| Toutiao | no | a bytecode script challenge; the local browser fails in this version |
| Reddit | no | the local browser is sent to `Prove your humanity` after a script redirect (later requests from the survey machine did not connect at all) |
| Quora, Stack Overflow, Hashnode, Medium | no | Cloudflare's bot check answers 403, also to the local browser (`-s playwright` waited and was refused again) |

The table records one day's answers; sites change theirs without notice, and an
answer may depend on the network and the load. Weibo posts other than the
article pages above, X and Facebook-style pages that need a login are read only
as far as the site serves them to an anonymous client. MHTML files are not read;
save a page as "Webpage, HTML Only" or "Webpage, Complete".

### Your own cookies, for the local browser

Markitai sends no cookies unless you configure them. The only place to
configure them is `fetch.playwright.cookies` (a JSON list in the configuration
file, or `--config-json` for one run), which the local browser uses for
`-s playwright` and for `auto`, which then selects the browser and neither
reads nor writes the page cache (see [Browser HTTP credentials](#browser-http-credentials)).
Static fetching (`-s static`, and the static request of `auto` when no cookies
are configured) stays anonymous: it does not read these cookies, which is the
policy for every browser credential. Copy the cookies of your own signed-in
session from your browser's developer tools; they are credentials, so keep the
file private and never paste them into a shared place:

```json
{"fetch": {"playwright": {"cookies": [
  {"name": "session", "value": "<value from your browser>", "domain": ".example.com", "path": "/", "secure": true}
]}}}
```

Give every cookie a `domain` (a leading dot covers the site and its
subdomains) or a `url`. Chromium then sends it only to hosts that domain
covers, including after a redirect, never to the other sites you convert. A
cookie with neither is set for the page being fetched, whatever its host. The
configuration display hides cookie values (`config list` and `get`), error
messages never include them, and a page read with cookies is neither served
from nor stored in the page cache nor learned as a browser route. Whether a
site accepts the session is up to the site: Zhihu, for one, may still answer
the local browser with its security check, in which case saving the page from
your own browser is the way.

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
browser HTTP, timeout and resource failures remain errors (an opted-in remote
fallback may still read the page, see
[strategy order](#strategy-order-and-remote-fallback)). A configured strategy
priority or `fetch.policy.enabled=false` replaces learned routes. If no browser is
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

## Strategy order and remote fallback

`auto` tries strategies in an order decided per URL, as the reference policy
does. The first rule that applies wins:

1. A URL whose host is local or private (`localhost`, `.local`, `.internal`,
   `.lan`, `.home`, `.corp`, single-label names, non-public address literals,
   userinfo), or that `fetch.policy.local_only_patterns` names, uses the local
   steps only. With `fetch.policy.inherit_no_proxy` (the default) the entries
   of `NO_PROXY` (or `no_proxy`) count as local-only patterns too. Patterns use
   the `NO_PROXY` grammar: `*`, `.name` and `*.name` (subdomains only), CIDR
   blocks (`10.0.0.0/8`, `192.168.1.0/255.255.255.0`, `fd00::/8`), and exact
   hosts or addresses.
2. `fetch.domain_profiles."<host[:port]>".strategy_priority` (the exact
   authority, as the browser profile is looked up) replaces the order.
3. That profile's `prefer_strategy` goes first, followed by the default order
   without it.
4. `fetch.policy.strategy_priority` replaces the order.
5. With `fetch.policy.enabled=false`, the default order, with no browser-first
   domains and no learned routes.
6. An authority with a [learned browser route](#learned-browser-routing) starts
   with the browser and does not retry static: a browser failure stays the
   failure, as before.
7. A domain in a `fetch.fallback_patterns` list the user wrote (configuration
   file or `--config-json`), or one of its subdomains, starts with the browser,
   then static. The contract's default list (`twitter.com`, `x.com`,
   `instagram.com`, `facebook.com`, `linkedin.com`, `threads.net`), which
   `config list` shows, is not applied: the static path and the X post reader
   read X posts well and faster than a browser, which is slower and more
   fragile there. As with `remote_consent`, the CLI tells the two apart from the
   configuration before defaults are filled in; `serve`, `mcp` and the bindings
   apply no list.
8. Otherwise: static, browser, then the remote services defuddle, jina and
   cloudflare.

The list is cut to `fetch.policy.max_strategy_hops` (default 5, at most 6).
Remote services are then removed for a URL with credential material: userinfo,
a query or fragment parameter named like a secret (`token`, `key`, `secret`,
`password`, `signature`, `credential`, `auth`, `code`, `sid`, `sig`, `ticket`,
`session`, `jwt`, `otp`, `apiKey`, `access_key` and similar spellings), a
`token=…` path segment, a JWT or a long random-looking path segment, or a
token-like segment after `/reset/`, `/verify/`, `/token/`, `/invite/`,
`/session/` and similar routes. A list left without a step that can run becomes
the local default. Local steps keep their rules: a page that needs JavaScript,
a challenge or a verification page goes on to the browser; a script-rendered
shell keeps its static text when nothing better comes; any other static failure
(an HTTP status, a transport error) ends the local steps.

Remote services are tried only after the local steps could not read the page,
unless a configured priority puts one first, and never for a 404 or 410, a
configuration or input error, or a host that resolves to a non-public address
(checked once, before the first remote request; the failure then says so).
Cloudflare is passed over when its credentials are not set. The first remote
service that reads the page wins; its result names it in `fetch_strategy` and
is not stored in the page cache. When every step fails, the local failure comes
first (its `HTTP <status>` stays at the start) and the remote failures follow:
`…; remote services failed as well (HTTP 429 from the defuddle service: …; …)`.
A kept shell gets `Remote extraction failed (…); the static text was kept.`

### Remote readings that are refusals

A service's reading that is the site's refusal rather than the page counts as
that service failing, judged as the local readers judge a page (see
[failures](#failures-retries-and-redirects)): a verification or login page, a
challenge, a page that only asks for JavaScript, a JSON error answer that is
the whole reading, or the page's own refusal status as the service reports it
(Jina's `httpStatus` 401, 403, 418 or 429, or a warning `Target URL returned
error 403`; Cloudflare's `meta.status`, the HTTP status the origin returned).
The next service in the order is tried; nothing is written. The service's
failure names it and what it was shown:
`The jina service was shown Zhihu's verification page instead of the content`,
`The defuddle service was shown a challenge page instead of the content`,
`The jina service received HTTP 403 from the site instead of the content`.
Another failed status of the page (400 and above) is the page's failure:
`The jina service received HTTP 404 from the site: the page may have been
removed or is not public`, `The cloudflare service received HTTP 503 from the
site: the site had a server error`.

Cloudflare's `meta` object, and each of its fields, is optional in the
Browser Rendering API reference (read 2026-10-02) for `/content`, the endpoint
Markitai calls, and for `/markdown`: `status`,
`finalUrl`, `title`, `headers`, `redirectChain`. Without `status` the rendered
markup alone is judged, as before; `finalUrl`, when it is an http(s) address,
is where the page is judged to have been read, as Jina's `url` is.

A JSON error answer is recognized only when it is the whole reading: the
Markdown raw, or the only content of its only fenced code block (with or
without an info string such as `json`); or markup whose only visible text is
the object, or holds it in a preformatted block with nothing else beside a
browser's `Pretty-print` switch (how a browser shows a JSON answer). The
object is at most 8 KiB, as much as local fetching reads of a refused answer,
and has an error's shape: an `error` object or text, an `errors` list with a
`message`, a `message` (`msg`, `errmsg`, `error_description`, `detail`) with a
failing `code` or `status` (400 and above), a non-zero `errcode`, `success` or
`ok` false or `status: "error"`, or a known site's refusal code (Zhihu's
`40362`, which counts alone). An object that also holds `data`, `result`,
`results` or `items` is content. An article or API document that shows a JSON
error as an example, a heading over the block, a second block or any other
text keeps the page. The failure quotes the service's reading as local
fetching quotes a refused answer (cleaned, at most 120 characters, then the
code): `The cloudflare service was shown Zhihu's JSON refusal instead of the
content, which said: 您当前请求存在异常，暂时限制本次访问。… (code 40362)`; on a
site that is not known, `… was shown a JSON error instead of the content, …`.

When no step reads the page, what works instead is said once: the local failure
already says it for a site that refused the static client (`HTTP 403 for
https://www.zhihu.com/…: Zhihu refuses automated clients; open the page in your
browser and save it …; remote services failed as well (HTTP 502 from the defuddle
service: …; The jina service was shown Zhihu's verification page instead of the
content)`); otherwise it is added after the failures, or, when only remote
services ran, leads: `Zhihu refuses automated clients; open the page …; the
remote services tried failed (…)`. For a site that is not known it reads `the
site turns automated readers away; open the page in your browser and save it …`.
A strategy selected with `-s` reports its service's failure the same way:
`The jina service was shown Zhihu's login page instead of the content; Zhihu
refuses automated clients; …`.

### Consent

`auto` never sends a URL to a remote service unless the user opted in. The
contract's default for `fetch.remote_consent` is `always`, the reference's
value, and `config list` shows it; in this build that default does not opt in,
because the configuration a conversion receives cannot tell it from a value the
user wrote. The CLI reads the selected configuration file and `--config-json`
before defaults are filled in:

| `fetch.remote_consent` | `auto` fallback | `-s defuddle/jina/cloudflare` | `fetch.strategy` set to one of them |
|---|---|---|---|
| default (`always`, not written) | never | runs | runs |
| `always`, written by the user | runs; a one-line disclosure the first time | runs | runs |
| `ask` | asks once per run on a terminal before the first remote fallback; otherwise skipped with one note | runs (choosing it is the answer) | asks like the fallback; refused without a terminal |
| `never`, or `--no-remote-fetch` | never | refused | refused |

`MARKITAI_NO_REMOTE_FETCH=1` (also `true`, `yes`, `on`) refuses everything, as
`never` does. The question names the page and the services the run may try and
defaults to no; one answer serves the whole run, other conversions of a batch
wait for it, and the status line pauses while it is on screen. It needs a
terminal on stdin and stderr and a run without `--quiet`. Without one, `ask`
counts as `never` and the run prints once: `Note: remote extraction services were
skipped: fetch.remote_consent is ask and there is no terminal to ask. …`. The
disclosure for `always` is printed once per `MARKITAI_HOME` (a marker file
`notices/remote-fetch`, which holds nothing, records it; a `--quiet` run does not
print it and leaves it due). Refusals of a selected strategy start with
`Remote fetching is disabled by policy`. A configured remote strategy also
honours local-only patterns; `-s` for the run overrides them for a public URL.
Credential material and private hosts are refused whatever was chosen.

`serve`, `mcp` and the language bindings install no consent host: `auto` there
never falls back to a remote service (as before), and `ask` there counts as
`never` (`serve` already rewrites it so). Explicitly selected strategies behave
as in the table.

Differences from the reference, kept on purpose: the reference's default
`always` sends URLs to the remote services whenever local strategies fail, and
its default `fallback_patterns` make X and the other listed sites browser-first;
a browser-first domain there tries the remote services before static, and a
learned route retries static last. A refused explicit remote strategy does not
fall back to the `auto` chain here; it fails with the service's reason.

## Remote services

| Service | Request | Options |
|---|---|---|
| defuddle | `GET https://defuddle.md/<URL, percent-encoded>`; Markdown with YAML frontmatter, which becomes metadata | `fetch.defuddle.timeout` (seconds, default 30), `rpm` (default 20) |
| jina | `GET https://r.jina.ai/<URL>` with `Accept: application/json` | `fetch.jina.api_key` (a value or `env:NAME`, else `JINA_API_KEY`) as a bearer token; `timeout` (seconds), `rpm`; `no_cache` sends `X-No-Cache: true`; `target_selector` and `wait_for_selector` send `X-Target-Selector` and `X-Wait-For-Selector` |
| cloudflare | `POST https://api.cloudflare.com/client/v4/accounts/<account>/browser-rendering/content` with your token | see below |

Each service has its own sliding one-minute window of `rpm` requests, shared by
every conversion of the process; a request waits for a slot. An `env:NAME` key
whose variable is not set sends no key (and does not fall back to `JINA_API_KEY`).

Jina answers JSON (`data.title`, `url`, `content`, `warning`, `httpStatus`). Its
text form, which it sends when the JSON is not honoured, is read as well: the
header lines `Title:`, `URL Source:`, `Published Time:`, `Warning:` (each
optional, blank lines between them allowed, `\r\n` accepted), then a
`Markdown Content:` line and the page; the same header lines at the start of a
JSON `content` are removed. A body that is neither stays a failure (`The jina
service returned an answer that is not JSON`), and a page whose own first line
merely looks like a header is kept whole. A `Warning:` (such as `This is a
cached snapshot of the original page, consider retry with caching opt-out.`)
becomes a conversion warning, `The jina service said: …`, with `Run with
--no-cache (or set fetch.jina.no_cache) to ask Jina for a fresh reading.` added
to a cached-snapshot warning when no opt-out was sent.

When this conversion bypasses Markitai's own page cache for the URL
(`--no-cache` or `cache.no_cache`, or a matching `--no-cache-for` /
`cache.no_cache_patterns` entry), Jina is sent `X-No-Cache: true` as well, so a
stale or poisoned snapshot of Jina's (seen for `example.com` on 2026-10-02) is
not returned instead of a fresh reading; `fetch.jina.no_cache` sends it always.
`cache.enabled=false` alone does not. defuddle.md documents no cache opt-out (its
answers carry `Cache-Control: s-maxage=300`, so a reading may be up to five
minutes old); nothing is sent for it. Cloudflare's `cacheTTL` is only what
`fetch.cloudflare.cache_ttl` sets.

Cloudflare uses your own account. `fetch.cloudflare.api_token` and
`account_id` are values or `env:NAME` references, else `CLOUDFLARE_API_TOKEN`
and `CLOUDFLARE_ACCOUNT_ID`; for `-s cloudflare` an `env:NAME` whose variable is
missing is an error naming the variable. The token needs Account / Browser
Rendering / Edit for `-s cloudflare` and Account / Workers AI / Read for
`-b cloudflare`. The request body carries the URL, `gotoOptions` with
`timeout` (milliseconds, default 30,000) and `waitUntil` (`wait_until`, default
`networkidle0`), `rejectRequestPattern` (`reject_resource_patterns`; by default
style sheets and fonts), and, when set, `userAgent`, `cookies`,
`waitForSelector` and `authenticate` (`http_credentials`; a value may be
`env:NAME`); `cache_ttl` above 0 adds `?cacheTTL=`. At most two renders run at
a time, and a 429 is repeated twice after 2 and 4 seconds. The rendered HTML goes
through the native extraction and [site readers](html.md#reading-sites) like any
other page, and a verification, login or challenge page is a failure (see
[remote readings](#remote-readings-that-are-refusals)). `renderer` and
`browser_ms_used` (Cloudflare's `X-Browser-Ms-Used`, rounded to a whole number
of milliseconds and written as a number, like `duration_ms`) are recorded in the
metadata. The cookies and HTTP credentials
are sent to Cloudflare's browser; configure them only for sites you accept that
for.

### The Cloudflare file backend

`-b cloudflare` (`fetch.cloudflare.convert_enabled=true`; `-b native` turns it
off) uploads local PDF, DOCX, XLSX/XLSM/XLSB, XLS/ET, ODS, ODT, Numbers (a single
file), CSV, XML, JPEG, PNG, WebP and SVG files to Workers AI `toMarkdown` in your
account and uses its Markdown instead of the native reader's; other formats keep
the native readers. The result records `converter: cloudflare-tomarkdown` and
the `tokens` Cloudflare reports.

Workers AI frames every document: a `# <file name>` heading, a `## Metadata`
list of the file's properties (`- PDFFormatVersion=1.4`, `- Creator=Writer`,
`- Producer=…`, `- CreationDate=D:20170816144228+02'00'`, …) and a `## Contents`
heading over the content, and a PDF's pages as `### Page N` headings. The frame
is removed and the content kept. Of the properties, `Title` becomes `title`
(unless it only names a file, such as `report.docx` or `Microsoft Word - …`),
`Author` becomes `author` and `CreationDate` becomes `date` in RFC 3339
(`2017-08-16T14:42:28+02:00`), the names the native readers use; the others
describe the file and are dropped. `### Page N` headings, numbered from 1 in
order, become the native PDF reader's page markers, `<!-- Page number: N -->`
followed by a blank line and the page's text, and their count is `pages`: a page
is not a section of the document, so it is no heading (it would otherwise be
taken for the title and misplace the document's own headings), and a PDF then
splits by page alike with `-b native` and `-b cloudflare`. A heading that is not
the next page number is the document's own and stays. Markdown without the frame
is used as it came. As for every local file, only `title` reaches the output
frontmatter; the other fields stay in the conversion's metadata, as the native
readers' `pages` does. Images are converted by a model that uses the
account's Neurons allowance, and say so in a warning. A file for which OCR or
screenshots are requested keeps the native reader, which renders its pages,
with a warning. The backend is refused under `--no-remote-fetch`,
`MARKITAI_NO_REMOTE_FETCH` and `fetch.remote_consent=never`, and without
credentials. `-s cloudflare` does not turn on the file backend (the reference's
does); `-b` alone decides. URLs that download a document keep the native reader.

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
(now `native-fetch-1.3.0-dev-r3`; r3 since the site readers), a NUL separator and the exact original URL. Rows
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
following, its limits and credential handling. `fetch/remote_tests.rs` runs the
remote fallback, Defuddle, Jina, Cloudflare Browser Rendering and Workers AI
`toMarkdown` against loopback services that read the request bodies: the order,
priorities and hops, each consent setting with a stand-in terminal (one question
per process, the once-per-home disclosure, the note without a terminal),
local-only patterns and `NO_PROXY`, the options each service receives, pacing,
429 repetition, and that failures carry no token, account id or endpoint. They
also replay the Jina answer recorded for a Zhihu question in the real-service
check of 2026-10-02 (its `安全验证 - 知乎` security check) through the chain after
the recorded 403 and through `-s jina`, defuddle's Markdown of the same page, a
challenge and an article titled like one, Jina's warnings, text form, header lines
in its JSON content and page statuses, `X-No-Cache` under `--no-cache` and
matching patterns, and the Workers AI answers recorded in that check for the
reference's public `sample.pdf` and `sample.docx`
(`crates/markitai-core/tests/fixtures/cloudflare-tomarkdown/`).
`fetch/policy.rs`, `fetch/chain.rs` and `fetch/consent.rs` test the order, the
chain's decisions and the gate on their own. Credentials in these tests come
from injected maps; no real service is contacted. They use temporary configured
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
To read a site that wants your own signed-in session, see
[your own cookies](#your-own-cookies-for-the-local-browser).

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
