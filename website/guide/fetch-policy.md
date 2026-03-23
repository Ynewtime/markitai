# Fetch Policy Engine

Markitai uses a policy-driven Fetch Policy Engine to determine the best strategy for retrieving URL content. It is designed to be resilient, fast, and user-friendly.

## Strategy Selection Logic

The engine follows a policy-driven approach to select the order of fetching strategies:

1. **Explicit Strategy**: If you provide an explicit strategy (e.g., `--playwright`, `--defuddle`, or `--jina`), the engine will use only that strategy.
2. **Domain Profiles**: You can configure specific settings for individual domains, such as custom selectors to wait for or extra wait times.
3. **Adaptive Fallback**: In `auto` mode (default), the engine intelligently orders strategies based on the domain and previous success history.

### Default Order (Standard Domains)

For most websites, Markitai prioritizes free, fast strategies first:

```
Defuddle → Jina → Static (HTTP) → Playwright (Browser) → Cloudflare
```

### SPA/Heavy-JS Order

For domains known to require JavaScript (like `x.com`, `instagram.com`, or domains that have failed static fetching before):

```
Defuddle → Jina → Playwright (Browser) → Cloudflare → Static
```

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
| `wait_for` | string | `"domcontentloaded"` | Page load event to wait for (`load`, `domcontentloaded`, `networkidle`) |
| `extra_wait_ms` | integer | `3000` | Extra milliseconds to wait after page load event |
| `prefer_strategy` | string | `null` | Preferred strategy for this domain (`static`, `defuddle`, `playwright`, `cloudflare`, `jina`) |
| `strategy_priority` | list | `null` | Custom strategy order for this domain (overrides global and `prefer_strategy`) |

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
    ├─ Explicit strategy (--static/--defuddle/--playwright/--jina/--cloudflare)?
    │       └─ Yes → Use only that strategy
    │
    ├─ Domain in SPA cache or known JS-heavy?
    │       └─ Yes → SPA order (Defuddle → Jina → Playwright → Cloudflare → Static)
    │
    └─ Default → Standard order (Defuddle → Jina → Static → Playwright → Cloudflare)
            │
            ├─ Try strategy #1 → Success? → Done
            ├─ Try strategy #2 → Success? → Done
            ├─ Try strategy #3 → Success? → Done
            ├─ Try strategy #4 → Success? → Done
            └─ Try strategy #5 → Success? → Done / Give up
```

Each strategy validates content quality before accepting the result. If the content appears empty or too short, it falls through to the next strategy.

::: tip
When a domain fails static fetching, it's automatically added to the SPA domain cache. Future requests to that domain will skip directly to browser rendering, saving time.
:::
