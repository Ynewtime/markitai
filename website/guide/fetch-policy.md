# Fetch Policy Engine

When you give markitai a URL, it tries several ways of fetching the page, cheapest first, until one returns usable content. This page explains the order, what stays on your machine, and how to tune it.

## Strategy Selection Logic

Three things decide the order:

1. **An explicit strategy**: `-s static`, `-s playwright`, `-s defuddle`, `-s jina` or `-s cloudflare` uses that one strategy. If a remote service refuses (rate limit, auth failure), the run falls back to the `auto` chain instead of failing.
2. **A domain profile**: per-domain settings such as a selector to wait for, extra wait time, or a custom strategy order.
3. **Adaptive fallback** (`auto`, the default): one of the two orders below, adjusted by what markitai has learned about the domain.

### Default Order (Standard Domains)

Local first. The built-in static fetcher matches the remote readers on extraction quality and never sends the URL anywhere.

<StrategyChain />

### SPA/Heavy-JS Order

Domains known to need JavaScript go straight to the browser: `x.com`, `instagram.com`, anything in `fallback_patterns`, and domains learned into the SPA cache after a failed static fetch.

<StrategyChain mode="spa" />

### Remote Fallback and Local-only URLs

By default (`fetch.remote_consent: always`) a public URL that fails locally may be sent to Defuddle, Jina or Cloudflare. Before the first remote attempt in a process, markitai prints a notice on stderr naming every service it may use later: defuddle.md, Jina, Cloudflare, FxTwitter and Twitter oEmbed. Services are tried one at a time, so a URL only reaches the service currently being attempted.

For X/Twitter posts, Playwright may also call FxTwitter and then Twitter oEmbed after local extraction fails. They follow the same process-wide consent decision as every other remote service: under `ask` they can raise the one shared prompt, reuse an earlier answer, and are skipped when the run cannot prompt.

These URLs never leave your machine, whatever strategy is selected:

- localhost, private IPs and common intranet hostnames
- URLs carrying credentials: userinfo, tokens, signatures, passwords, API keys or authorization codes

In the `auto` chain, domains matched by `fetch.policy.local_only_patterns` and by `NO_PROXY` (when `inherit_no_proxy` is on) also stay local.

| Setting | Effect | Can an explicit `-s` override it? |
|---------|--------|-----------------------------------|
| `remote_consent: always` (default) | Remote fallback for public URLs, disclosed on stderr | — |
| `remote_consent: ask` | One prompt per process on a TTY; non-interactive runs skip every remote service | — |
| `remote_consent: never` | Automatic and config-selected strategies stay local | Yes |
| `local_only_patterns` / `NO_PROXY` | Matching domains stay local in the `auto` chain | Yes |
| Private, intranet or credential-bearing URL | Always local | **No** |
| `MARKITAI_NO_REMOTE_FETCH=1` | Hard local-only guarantee | **No** |

A remote `fetch.strategy` set in the config file is not an explicit opt-in. It stays governed by `remote_consent` and prints the same first-use notice.

## Configuration

Tune the policy in `markitai.json`:

```json
{
  "fetch": {
    "policy": {
      "enabled": true,
      "max_strategy_hops": 5
    },
    "domain_profiles": {
      "x.com": {
        "wait_for_selector": "[data-testid=tweetText]",
        "wait_for": "domcontentloaded",
        "extra_wait_ms": 1200
      }
    },
    "playwright": {
      "session_mode": "domain_persistent",
      "session_ttl_seconds": 600
    }
  }
}
```

### Policy Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Turn strategy ordering on or off |
| `max_strategy_hops` | integer | `5` | Strategies to try before giving up |
| `strategy_priority` | list | `null` | Custom global strategy order |
| `local_only_patterns` | list | `[]` | Domains and IPs restricted to local strategies (`NO_PROXY` syntax) |
| `inherit_no_proxy` | boolean | `true` | Also treat `NO_PROXY` entries as local-only |

### Domain Profiles

Per-domain overrides:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `wait_for_selector` | string | `null` | CSS selector to wait for before extracting |
| `wait_for` | string | `null` | Page load event: `load`, `domcontentloaded`, `networkidle`. Unset inherits `fetch.playwright.wait_for` |
| `extra_wait_ms` | integer | `null` | Extra wait after the load event. Unset inherits `fetch.playwright.extra_wait_ms` |
| `prefer_strategy` | string | `null` | Strategy to try first for this domain |
| `strategy_priority` | list | `null` | Full strategy order for this domain (overrides global order and `prefer_strategy`) |
| `skip_auto_scroll` | boolean | `false` | Skip auto-scrolling on single-content pages (tweets, issues, docs) |
| `reject_resource_patterns` | list | `null` | Block browser requests matching these URL patterns, e.g. `["**/analytics/**"]` |

::: warning
markitai ships built-in profiles for `x.com`, `twitter.com` and `github.com`. Your own profile for the same domain replaces the built-in one entirely, so repeat any built-in tuning you want to keep.
:::

```json
{
  "fetch": {
    "domain_profiles": {
      "x.com": { "wait_for_selector": "[data-testid=tweetText]", "extra_wait_ms": 1200 },
      "instagram.com": { "wait_for": "networkidle", "extra_wait_ms": 2000 },
      "docs.example.com": { "prefer_strategy": "static" }
    }
  }
}
```

### Playwright Session Persistence

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `session_mode` | string | `"isolated"` | `isolated`: a fresh browser context per request. `domain_persistent`: reuse the context per domain |
| `session_ttl_seconds` | integer | `600` | How long a persistent session is kept |

`domain_persistent` reuses cookies and local storage, which makes repeated requests to the same site much faster.

## Static HTTP Adapters

The static strategy uses httpx, which works for most sites. If a site rejects it with a 403 or empty content because of TLS fingerprinting, `auto` falls back to Playwright or Cloudflare on its own. To get past the fingerprint check without launching a browser, switch to curl-cffi:

```bash
uv pip install markitai[extra-fetch]
export MARKITAI_STATIC_HTTP=curl_cffi
```

If curl-cffi is not installed, markitai silently uses httpx.

## How It Works

Strategies are attempted one at a time, up to `max_strategy_hops`. The first result that passes validation ends the run.

### Result validation

Empty or too-short content, login walls, and anti-bot or CAPTCHA pages (Geetest, Cloudflare, reCAPTCHA, hCaptcha) all fail validation and fall through to the next strategy.

### SPA learning

When a static fetch succeeds but the page says it needs JavaScript, the domain is added to the SPA cache for 30 days and later requests skip straight to the browser. Only that signal teaches the cache; CAPTCHAs, login walls and network errors do not. Inspect or clear it with `markitai cache spa-domains`.
