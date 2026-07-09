# Fetch Policy Engine

Markitai uses a policy-driven Fetch Policy Engine to determine the best strategy for retrieving URL content. It is designed to be resilient, fast, and user-friendly.

## Strategy Selection Logic

The engine follows a policy-driven approach to select the order of fetching strategies:

1. **Explicit Strategy**: If you provide an explicit strategy (e.g., `-s playwright`, `-s defuddle`, or `-s jina`), the engine uses only that strategy — with two caveats: explicit `-s defuddle`/`-s jina`/`-s cloudflare` can still gracefully fall back to the full `auto` chain if the remote service refuses the request (rate limit, auth, etc.); and domain-profile content settings (`wait_for_selector`, `skip_auto_scroll`, etc.) still apply on top of an explicitly-chosen `playwright` strategy.
2. **Domain Profiles**: You can configure specific settings for individual domains, such as custom selectors to wait for, extra wait times, or a full custom strategy order.
3. **Adaptive Fallback**: In `auto` mode (default), the engine intelligently orders strategies based on the domain and previous success history.

### Default Order (Standard Domains)

Markitai is local-first: for most websites, the native local pipeline is tried before any remote, consent-gated service:

```
Static (HTTP) → Playwright (Browser) → Defuddle → Jina → Cloudflare
```

Static's native webextract pipeline matches remote Defuddle's quality on the extraction benchmark corpus (and does better on CJK spacing), so it goes first — and unlike the remote strategies, it never sends the URL off-machine or requires fetch consent.

### SPA/Heavy-JS Order

For domains known to require JavaScript (like `x.com`, `instagram.com`, domains listed in `fallback_patterns`, or domains that have previously failed static fetching and were learned into the SPA cache), Markitai skips straight to the browser:

```
Playwright (Browser) → Defuddle → Jina → Cloudflare → Static
```

Static goes last here since it has already failed (or is expected to fail) to produce usable content for these domains.

## Configuration

You can tune the fetch policy in your `markitai.json`:

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
| `enabled` | boolean | `true` | Enable or disable intelligent strategy ordering |
| `max_strategy_hops` | integer | `5` | Maximum number of strategies to attempt before giving up |
| `strategy_priority` | list | `null` | Custom global strategy order (overrides default priority) |
| `local_only_patterns` | list | `[]` | Domain/IP patterns restricted to local strategies (NO_PROXY syntax) |
| `inherit_no_proxy` | boolean | `true` | Merge `NO_PROXY` env var into `local_only_patterns` |

### Domain Profiles

Domain profiles allow per-domain overrides for fetch behavior:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `wait_for_selector` | string | `null` | CSS selector to wait for before extracting content |
| `wait_for` | string | `null` | Page load event override (`load`, `domcontentloaded`, `networkidle`); unset inherits the global `fetch.playwright.wait_for` (default `domcontentloaded`) |
| `extra_wait_ms` | integer | `null` | Extra milliseconds to wait after page load event; unset inherits the global `fetch.playwright.extra_wait_ms` (default `3000`) |
| `prefer_strategy` | string | `null` | Preferred strategy for this domain (`static`, `defuddle`, `playwright`, `cloudflare`, `jina`) |
| `strategy_priority` | list | `null` | Custom strategy order for this domain (overrides global and `prefer_strategy`) |
| `skip_auto_scroll` | boolean | `false` | Skip auto-scrolling for single-content pages (tweets, issues, docs) |
| `reject_resource_patterns` | list | `null` | Block Playwright-navigation resources matching these URL patterns (e.g. `["**/analytics/**"]`) |

Markitai ships built-in profiles for `x.com`/`twitter.com` and `github.com`. Configuring your own profile for the same domain **replaces it entirely** rather than merging field-by-field — any built-in tuning (like the x.com profile's `skip_auto_scroll`/`reject_resource_patterns`) is lost unless you repeat it yourself.

Example with multiple domains:

```json
{
  "fetch": {
    "domain_profiles": {
      "x.com": {
        "wait_for_selector": "[data-testid=tweetText]",
        "extra_wait_ms": 1200
      },
      "instagram.com": {
        "wait_for": "networkidle",
        "extra_wait_ms": 2000
      },
      "docs.example.com": {
        "prefer_strategy": "static"
      }
    }
  }
}
```

### Playwright Session Persistence

Control how Playwright manages browser contexts:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `session_mode` | string | `"isolated"` | `isolated`: new context per request; `domain_persistent`: reuse contexts per domain |
| `session_ttl_seconds` | integer | `600` | How long to keep persistent sessions alive (in seconds) |

Using `domain_persistent` mode significantly speeds up multiple requests to the same site by reusing cookies, localStorage, and other browser state.

## Static HTTP Adapters

The Static strategy uses **httpx** by default, which works for the vast majority of websites. For sites with TLS fingerprint detection, you can optionally enable the **curl-cffi** adapter.

| Adapter | Installation | Description |
|---------|-------------|-------------|
| **httpx** (default) | Built-in, works out of the box | Fast and reliable, covers most use cases |
| **curl-cffi** (optional) | `uv pip install markitai[extra-fetch]` | Mimics Chrome TLS/HTTP signatures to bypass some anti-bot protections |

::: tip When do you need curl-cffi?
In most cases, you don't. If the Static strategy returns 403/empty content for a site, the Policy Engine automatically falls back to Playwright or Cloudflare. You only need curl-cffi when you want to bypass TLS fingerprint detection **without launching a browser**.
:::

To enable `curl-cffi`:

```bash
# Install
uv pip install markitai[extra-fetch]

# Activate via environment variable
export MARKITAI_STATIC_HTTP=curl_cffi
```

If the environment variable is set but curl-cffi is not installed, Markitai silently falls back to httpx without errors.

## How It Works

```
URL Request
    │
    ├─ Explicit strategy (-s static/playwright/defuddle/jina/cloudflare)?
    │       └─ Yes → Use only that strategy
    │
    ├─ Domain in SPA cache or known JS-heavy (fallback_patterns)?
    │       └─ Yes → SPA order (Playwright → Defuddle → Jina → Cloudflare → Static)
    │
    └─ Default → Standard order (Static → Playwright → Defuddle → Jina → Cloudflare)
            │
            ├─ Try strategy #1 → Success? → Done
            ├─ Try strategy #2 → Success? → Done
            ├─ Try strategy #3 → Success? → Done
            ├─ Try strategy #4 → Success? → Done
            └─ Try strategy #5 → Success? → Done / Give up
```

Domain profiles (`strategy_priority` or `prefer_strategy`) and a global `strategy_priority` override can reorder this chain per-domain or globally, ahead of the SPA/default fallback — see [Domain Profiles](#domain-profiles) below. Private/local/intranet domains and `local_only_patterns` matches are restricted to local-only strategies (`static`, `playwright`) regardless of the above.

Each strategy validates content quality before accepting the result — checking for empty/too-short content, login walls, and anti-bot/CAPTCHA challenge pages (Geetest, Cloudflare, reCAPTCHA, hCaptcha). If validation fails, it falls through to the next strategy.

::: tip
When static fetching succeeds but the content indicates JavaScript rendering is required (or the page is empty), the domain is added to the SPA cache for 30 days. Future requests to that domain will skip directly to browser rendering, saving time. Other failure modes (CAPTCHA, login walls, network errors) don't trigger this — only the JS-required signal does.
:::
