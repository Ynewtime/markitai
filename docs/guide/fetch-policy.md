# Fetch Policy Engine

Markitai uses a sophisticated Fetch Policy Engine to determine the best strategy for retrieving URL content. This engine is inspired by modern web scraping architectures and is designed to be resilient, fast, and user-friendly.

## Strategy Selection Logic

The engine follows a policy-driven approach to select the order of fetching strategies:

1.  **Explicit Strategy**: If you provide an explicit strategy (e.g., `--playwright` or `--jina`), the engine will use only that strategy.
2.  **Domain Profiles**: You can configure specific settings for individual domains, such as custom selectors to wait for or extra wait times.
3.  **Adaptive Fallback**: In `auto` mode (default), the engine intelligently orders strategies based on the domain and previous success history.

### Default Order (Standard Domains)
For most websites, Markitai prioritizes speed:
`Static (HTTP) -> Playwright (Browser) -> Cloudflare -> Jina`

### SPA/Heavy-JS Order
For domains known to require JavaScript (like `x.com`, `instagram.com`, or domains that have failed static fetching before):
`Playwright (Browser) -> Cloudflare -> Jina -> Static`

## Configuration

You can tune the fetch policy in your `markitai.json`:

```json
{
  "fetch": {
    "policy": {
      "enabled": true,
      "max_strategy_hops": 4
    },
    "domain_profiles": {
      "x.com": {
        "wait_for_selector": "[data-testid="tweetText"]",
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

### Options

- `policy.enabled`: Enable or disable the intelligent strategy ordering.
- `policy.max_strategy_hops`: Maximum number of strategies to attempt before giving up.
- `domain_profiles`: A map of domain-specific overrides.
- `playwright.session_mode`: 
    - `isolated`: (Default) New browser context for every request.
    - `domain_persistent`: Reuses browser contexts for the same domain, significantly speeding up multiple requests to the same site.

## Static HTTP Adapters

Markitai supports multiple backend adapters for static fetching:

- **httpx**: The default, fast and reliable Python HTTP client.
- **curl-cffi**: (Optional) Uses `curl-impersonate` to mimic real browser TLS/HTTP signatures, helping bypass some anti-bot protections.

To enable `curl-cffi`, install it (`pip install curl-cffi`) and set the environment variable:
`MARKITAI_STATIC_HTTP=curl_cffi`
