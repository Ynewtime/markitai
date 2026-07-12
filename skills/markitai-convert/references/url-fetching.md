# URL fetching: strategy order, tuning, and privacy

Read this when a URL conversion fails, returns junk (login wall, CAPTCHA, empty page), or must satisfy privacy constraints. Long-form docs: <https://markitai.dev/guide/fetch-policy>.

## How `auto` picks strategies

- Standard domains: `static → playwright → defuddle → jina → cloudflare`. Static's native extractor matches remote Defuddle quality on the benchmark corpus and never sends the URL off-machine, so it goes first.
- Known JS-heavy domains (`x.com`, `instagram.com`, other `fallback_patterns` entries, and domains learned into the SPA cache): `playwright → defuddle → jina → cloudflare → static`.
- Each result is quality-validated (too-short content, login walls, Geetest/Cloudflare/reCAPTCHA/hCaptcha challenge pages) before being accepted; a failed check falls through to the next strategy.
- When static fetch succeeds but the page signals JS-required rendering, the domain enters the SPA cache for 30 days and future runs skip straight to the browser. Inspect or reset with `markitai cache spa-domains [--clear]`.
- An explicit `-s <strategy>` uses only that strategy — except explicit `-s defuddle`/`-s jina`/`-s cloudflare`, which fall back to the full `auto` chain if the remote service refuses (rate limit, auth).

## Privacy rules

- localhost, private IPs, intranet hostnames, and URLs carrying credentials/tokens/signatures are always local-only (static/playwright), regardless of flags.
- `fetch.policy.local_only_patterns` (NO_PROXY syntax) and inherited `NO_PROXY` entries stay local in the `auto` chain; an explicit CLI `-s` on a public URL overrides only this pattern-based rule.
- The first remote attempt in a process prints a stderr disclosure naming every service it may use (defuddle.md, Jina, Cloudflare, FxTwitter, Twitter oEmbed). Services are tried one at a time.
- `fetch.remote_consent`: `always` (default) / `ask` (one interactive confirmation; non-interactive runs skip remote) / `never` (config-driven remote off; explicit CLI `-s` still allowed).
- Absolute guarantee for sensitive runs: `MARKITAI_NO_REMOTE_FETCH=1` — blocks remote services even for explicit `-s` flags.

## Failure recovery, in order

1. **Empty/partial content on a JS-heavy site** → `-s playwright`. Ensure Chromium is ready via `markitai doctor --fix` (needs the `browser` extra).
2. **403 or bot-blocked on static fetch, browser undesired** → install `extra-fetch` extra, set `MARKITAI_STATIC_HTTP=curl_cffi`.
3. **Site needs time or a specific element** → domain profile with `wait_for_selector` / `extra_wait_ms` (below).
4. **Local strategies exhausted** → let `auto` fall through to Defuddle/Jina/Cloudflare, or force one with `-s defuddle` / `-s jina` / `-s cloudflare` (credentials: `JINA_API_KEY`; `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`).
5. **Everything extracts garbage but the page renders** → `--llm --screenshot-only`: the LLM reads full-page screenshots instead of the DOM.

## Domain profiles (`fetch.domain_profiles` in markitai.json)

```json
{
  "fetch": {
    "domain_profiles": {
      "x.com": { "wait_for_selector": "[data-testid=tweetText]", "extra_wait_ms": 1200 },
      "docs.example.com": { "prefer_strategy": "static" }
    }
  }
}
```

| Key | Meaning |
|---|---|
| `wait_for_selector` | CSS selector to await before extraction |
| `wait_for` | Load event: `load` / `domcontentloaded` (global default) / `networkidle` |
| `extra_wait_ms` | Extra wait after the load event (global default 3000) |
| `prefer_strategy` | First-choice strategy for this domain |
| `strategy_priority` | Full custom order for this domain (overrides `prefer_strategy` and global order) |
| `skip_auto_scroll` | Skip auto-scroll on single-content pages (tweets, issues) |
| `reject_resource_patterns` | Block matching resource URLs during Playwright navigation |

Built-in profiles exist for `x.com`/`twitter.com` and `github.com`. Defining your own profile for one of these domains **replaces the built-in entirely** — repeat any built-in tuning you still want.

Global knobs: `fetch.policy.strategy_priority` (custom global order), `fetch.policy.max_strategy_hops` (default 5), `fetch.playwright.session_mode: "domain_persistent"` + `session_ttl_seconds` to reuse browser state across requests to one site.
