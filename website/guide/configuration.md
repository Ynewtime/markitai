# Configuration

## Configuration Priority

Markitai uses the following priority order (highest to lowest):

1. Command-line arguments
2. Environment variables
3. Configuration file
4. Default values

## Configuration File

Markitai looks for configuration files in the following locations:

1. Path specified by `--config` argument
2. `MARKITAI_CONFIG` environment variable
3. `./markitai.json` (current directory)
4. `~/.markitai/config.json` (user home)

### Initialize Configuration

```bash
# Interactive setup wizard (recommended)
markitai init

# Quick mode (generate default config)
markitai init --yes

# Create in specific location
markitai init --local  # creates ./markitai.json
```

### View Configuration

```bash
# List all settings
markitai config list

# Get specific value
markitai config get llm.enabled

# Set value
markitai config set llm.enabled true
```

### Full Configuration Example

```json
{
  "llm": {
    "enabled": false,
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-2.5-flash",
          "api_key": "env:GEMINI_API_KEY"
        }
      }
    ],
    "router_settings": {
      "routing_strategy": "simple-shuffle",
      "num_retries": 2,
      "timeout": 120
    },
    "concurrency": 10
  },
  "image": {
    "alt_enabled": false,
    "desc_enabled": false,
    "compress": true,
    "quality": 75,
    "format": "jpeg",
    "max_width": 1920,
    "max_height": 99999,
    "filter": {
      "min_width": 50,
      "min_height": 50,
      "min_area": 5000,
      "deduplicate": true
    }
  },
  "ocr": {
    "enabled": false,
    "lang": "en"
  },
  "screenshot": {
    "enabled": false,
    "screenshot_only": false,
    "viewport_width": 1920,
    "viewport_height": 1080,
    "quality": 75,
    "max_height": 10000
  },
  "cache": {
    "enabled": true,
    "no_cache_patterns": [],
    "max_size_bytes": 536870912,
    "global_dir": "~/.markitai"
  },
  "batch": {
    "concurrency": 10,
    "url_concurrency": 5,
    "scan_max_depth": 5,
    "scan_max_files": 10000,
    "state_flush_interval_seconds": 10,
    "heavy_task_limit": 0
  },
  "fetch": {
    "strategy": "auto",
    "playwright": {
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 3000,
      "wait_for_selector": null,
      "cookies": null,
      "reject_resource_patterns": null,
      "extra_http_headers": null,
      "user_agent": null,
      "http_credentials": null
    },
    "jina": {
      "api_key": null,
      "timeout": 30,
      "rpm": 20,
      "no_cache": false,
      "target_selector": null,
      "wait_for_selector": null
    },
    "cloudflare": {
      "api_token": null,
      "account_id": null,
      "timeout": 30000,
      "wait_until": "networkidle0",
      "cache_ttl": 0,
      "reject_resource_patterns": null,
      "convert_enabled": false
    },
    "policy": {
      "enabled": true,
      "max_strategy_hops": 4
    },
    "domain_profiles": {},
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  },
  "output": {
    "dir": "./output",
    "on_conflict": "rename",
    "allow_symlinks": false
  },
  "log": {
    "level": "INFO",
    "format": "text",
    "dir": "~/.markitai/logs",
    "rotation": "10 MB",
    "retention": "7 days"
  },
  "prompts": {
    "dir": "~/.markitai/prompts"
  }
}
```

::: tip
Use `env:VAR_NAME` syntax to reference environment variables in the config file. For `JINA_API_KEY`, `CLOUDFLARE_API_TOKEN`, and `CLOUDFLARE_ACCOUNT_ID`, you can also just set the environment variable (or add it to `.env`) without configuring anything in the config file — markitai reads them automatically.
:::

## Environment Variables

### API Keys

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `JINA_API_KEY` | Jina Reader API key |
| `CLOUDFLARE_API_TOKEN` | Cloudflare API token (Browser Rendering / Workers AI) |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID |

### Markitai Settings

| Variable | Description |
|----------|-------------|
| `MARKITAI_CONFIG` | Path to configuration file |
| `MARKITAI_LOG_DIR` | Directory for log files |
| `MARKITAI_LOG_FORMAT` | Log format override (`text` or `json`) |
| `MARKITAI_STATIC_HTTP` | Static HTTP backend: `httpx` (default) or `curl_cffi` (TLS impersonation) |

## LLM Configuration

### Supported Providers

Markitai supports multiple LLM providers through [LiteLLM](https://docs.litellm.ai/):

- OpenAI (GPT-5.2, GPT-5-mini)
- Anthropic (Claude 3.5/4)
- Google (Gemini 2.x)
- DeepSeek
- OpenRouter
- Ollama (local models)

#### Local Providers (Subscription-based)

Markitai also supports local providers that use CLI authentication and subscription credits:

- **Claude Agent** (`claude-agent/`): Uses [Claude Agent SDK](https://github.com/anthropics/claude-code) with Claude Code CLI authentication
- **GitHub Copilot** (`copilot/`): Uses [GitHub Copilot SDK](https://github.com/github/copilot-sdk) with Copilot CLI authentication
- **ChatGPT** (`chatgpt/`): Uses ChatGPT subscription via OAuth Device Code Flow and Responses API. No extra SDK required.
- **Gemini CLI** (`gemini-cli/`): Uses Google's Gemini CLI OAuth credentials (`~/.gemini/oauth_creds.json`) with automatic token refresh.

These providers require:
1. The respective CLI tool installed and authenticated (or environment variable auth — see below)
2. Optional SDK package: `uv add markitai[claude-agent]`, `uv add markitai[copilot]`, or `uv add markitai[gemini-cli]`

**Install Claude Code CLI:**
```bash
# macOS/Linux/WSL
curl -fsSL https://claude.ai/install.sh | bash

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex
```

**Install GitHub Copilot CLI:**
```bash
# macOS/Linux/WSL
curl -fsSL https://gh.io/copilot-install | bash

# Windows
winget install GitHub.Copilot
```

**ChatGPT (no CLI needed):**

ChatGPT provider authenticates via OAuth Device Code Flow on first use — just configure the model and follow the browser prompt.

**Install Gemini CLI (optional):**

Gemini CLI provider can reuse existing Gemini CLI credentials. Install the CLI and authenticate, or let the built-in OAuth flow handle it:

```bash
# Install Gemini CLI (optional — provider has built-in OAuth)
npm install -g @anthropic-ai/gemini-cli
gemini  # Triggers OAuth login on first run

# Install google-auth for automatic token refresh
uv add 'markitai[gemini-cli]'
```

### Model Naming

Use the LiteLLM model naming convention:

```
provider/model-name
```

Examples:
- `openai/gpt-4o`
- `anthropic/claude-sonnet-4-20250514`
- `gemini/gemini-2.5-flash`
- `deepseek/deepseek-chat`
- `ollama/llama3.2`
- `claude-agent/sonnet` (local, requires Claude Code CLI)
- `copilot/gpt-5.2` (local, requires Copilot CLI)
- `chatgpt/gpt-5.2` (local, requires ChatGPT subscription)
- `gemini-cli/gemini-2.5-pro` (local, requires Gemini CLI or OAuth)

Claude Agent SDK supported models:
- Aliases (recommended): `sonnet`, `opus`, `haiku`, `inherit`
- Full model strings: `claude-sonnet-4-5-20250929`, `claude-opus-4-6`, `claude-opus-4-5-20251101`

GitHub Copilot SDK supported models:
- OpenAI: `gpt-5.2`, `gpt-5.1`, `gpt-5-mini`, `gpt-5.1-codex`
- Anthropic: `claude-sonnet-4.5`, `claude-opus-4.6`, `claude-haiku-4.5`
- Google: `gemini-2.5-pro`, `gemini-3-flash`
- Availability depends on your Copilot subscription

ChatGPT supported models:
- `gpt-5.2`, `gpt-5.2-codex`, `codex-mini`, etc.

Gemini CLI supported models:
- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-3-flash-preview`, etc.

::: warning Model Deprecation Notice
The following models will be **retired on February 13, 2025**:
- `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`, `o4-mini`, `gpt-5`

Please migrate to `gpt-5.2` or other supported models before the deadline.
:::

::: tip Local Providers Support Vision
Local providers (`claude-agent/`, `copilot/`, `chatgpt/`, `gemini-cli/`) support image analysis (`--alt`, `--desc`) via file attachments. Make sure to use a vision-capable model (e.g., `copilot/gpt-5.2`, `gemini-cli/gemini-2.5-pro`).
:::

::: tip Troubleshooting Local Providers
Common errors and solutions:

| Error | Solution |
|-------|----------|
| "SDK not installed" | `uv add markitai[copilot]`, `uv add markitai[claude-agent]`, or `uv add markitai[gemini-cli]` |
| "CLI not found" | Install and authenticate the CLI tool ([Copilot CLI](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli), [Claude Code](https://claude.ai/code)) |
| "Not authenticated" | Run `copilot auth login` or `claude auth login`. Alternatively: set `GH_TOKEN`/`GITHUB_TOKEN` for Copilot, or `CLAUDE_CODE_USE_BEDROCK=1`/`CLAUDE_CODE_USE_VERTEX=1`/`CLAUDE_CODE_USE_FOUNDRY=1` for Claude. ChatGPT auto-triggers OAuth on first use. Gemini CLI reuses `~/.gemini/oauth_creds.json`. |
| "Rate limit" | Wait and retry, or check your subscription quota |
| "Request timeout" | Timeout is adaptive; for very large documents, processing may take longer |

Use `markitai doctor` to check authentication status and get resolution hints.
:::

### Custom API Endpoint

Use `api_base` to override a provider's default API endpoint. This value is passed directly to [LiteLLM](https://docs.litellm.ai/) and works with any LiteLLM-supported provider (OpenAI, Anthropic, Gemini, Azure, etc.). Supports `env:VAR_NAME` syntax just like `api_key`:

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "openai/your-model-name",
          "api_key": "env:YOUR_API_KEY",
          "api_base": "https://your-api-endpoint.com/v1"
        }
      }
    ]
  }
}
```

Examples:

```json
// Local Ollama
{
  "model": "ollama/llama3.2",
  "api_base": "http://localhost:11434"
}

// Azure OpenAI
{
  "model": "azure/gpt-4o",
  "api_key": "env:AZURE_API_KEY",
  "api_base": "https://your-resource.openai.azure.com"
}

// DeepSeek
{
  "model": "deepseek/deepseek-chat",
  "api_key": "env:DEEPSEEK_API_KEY",
  "api_base": "https://api.deepseek.com/v1"
}

// Any OpenAI-compatible provider
{
  "model": "openai/custom-model",
  "api_key": "env:CUSTOM_API_KEY",
  "api_base": "https://your-proxy-or-provider.com/v1"
}

// Reference environment variable
{
  "model": "anthropic/claude-sonnet-4-5-20250929",
  "api_key": "env:ANTHROPIC_API_KEY",
  "api_base": "env:ANTHROPIC_BASE_URL"
}
```

::: tip
Common use cases include self-hosted inference servers (vLLM, Ollama, LocalAI), regional API proxies, and third-party API gateways.
:::

::: warning Local Providers and `api_base`
The `api_base` config field does **not** apply to local providers (`claude-agent/`, `copilot/`, `chatgpt/`, `gemini-cli/`). These providers run as CLI subprocesses or use OAuth and manage API endpoints internally:

- **Claude Agent**: Set `ANTHROPIC_BASE_URL` to override the API endpoint. If `ANTHROPIC_API_KEY` is also set, the CLI will use it for direct API access instead of subscription authentication. Other routing options: `CLAUDE_CODE_USE_BEDROCK=1`, `CLAUDE_CODE_USE_VERTEX=1`, `CLAUDE_CODE_USE_FOUNDRY=1`.
- **GitHub Copilot**: Endpoint is managed by the Copilot CLI internally and cannot be overridden. For token-based auth, set `GH_TOKEN` or `GITHUB_TOKEN` with a personal access token that has the "Copilot Requests" permission.
- **ChatGPT**: Uses OpenAI's Responses API endpoint. Authentication handled via LiteLLM's built-in OAuth Device Code Flow.
- **Gemini CLI**: Uses Google's Code Assist API endpoint. Credentials read from `~/.gemini/oauth_creds.json`.
:::

### Vision Models

For image analysis (`--alt`, `--desc`), Markitai automatically routes to vision-capable models. Vision capability is **auto-detected** from litellm by default - no configuration needed for most models.

To explicitly override auto-detection, set `supports_vision`:

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-2.5-flash",
          "api_key": "env:GEMINI_API_KEY"
        },
        "model_info": {
          "supports_vision": true  // Optional: auto-detected if omitted
        }
      }
    ]
  }
}
```

### Router Settings

Configure how Markitai routes requests across multiple models:

```json
{
  "llm": {
    "router_settings": {
      "routing_strategy": "simple-shuffle",
      "num_retries": 2,
      "timeout": 120,
      "fallbacks": []
    },
    "concurrency": 10
  }
}
```

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| `routing_strategy` | `simple-shuffle`, `least-busy`, `usage-based-routing`, `latency-based-routing` | `simple-shuffle` | How to select models |
| `num_retries` | 0-10 | 2 | Retry count on failure |
| `timeout` | seconds | 120 | Request timeout (base value for adaptive calculation) |
| `concurrency` | 1-20 | 10 | Max concurrent LLM requests |

#### Model Weight

Each model in `model_list` accepts a `weight` parameter in `litellm_params` to control traffic distribution:

```json
{
  "model_name": "default",
  "litellm_params": {
    "model": "gemini/gemini-2.5-flash",
    "api_key": "env:GEMINI_API_KEY",
    "weight": 10
  }
}
```

| Value | Behavior |
|-------|----------|
| `weight: 0` | **Disabled** — model is excluded from routing entirely |
| `weight: 1` (default) | Normal priority |
| `weight: 10` | 10x more likely to be selected than weight=1 models |

Set `weight: 0` to temporarily disable a model without removing its configuration. At least one model must have `weight > 0`.

### Adaptive Timeout

Local providers (`claude-agent/`, `copilot/`, `chatgpt/`, `gemini-cli/`) use **adaptive timeout calculation** based on request complexity:

- Base timeout: 60 seconds minimum, 600 seconds maximum
- Factors considered: prompt length, number of images, expected output length
- Formula: `base_timeout + (prompt_chars / 500) + (images * 30) + (expected_output / 200)`

This prevents timeouts on large documents while keeping short requests responsive.

### Prompt Caching (Claude Agent)

Claude Agent provider automatically enables **prompt caching** for system prompts longer than 4KB. This reduces API costs by caching frequently-used system prompt prefixes.

::: tip
Prompt caching is transparent - no configuration needed. View cache statistics with `markitai cache stats --verbose`.
:::

## Image Configuration

Control how images are processed and compressed:

```json
{
  "image": {
    "alt_enabled": false,
    "desc_enabled": false,
    "compress": true,
    "quality": 75,
    "format": "jpeg",
    "max_width": 1920,
    "max_height": 99999,
    "filter": {
      "min_width": 50,
      "min_height": 50,
      "min_area": 5000,
      "deduplicate": true
    }
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `alt_enabled` | `false` | Generate alt text via LLM |
| `desc_enabled` | `false` | Generate description files |
| `compress` | `true` | Compress images |
| `quality` | `75` | JPEG/WebP quality (1-100) |
| `format` | `jpeg` | Output format: `jpeg`, `png`, `webp` |
| `max_width` | `1920` | Max width in pixels |
| `max_height` | `99999` | Max height in pixels (effectively unlimited) |
| `filter.min_width` | `50` | Skip images smaller than this |
| `filter.min_height` | `50` | Skip images shorter than this |
| `filter.min_area` | `5000` | Skip images with area below this |
| `filter.deduplicate` | `true` | Remove duplicate images |

## Screenshot Configuration

Enable screenshot capture for documents and URLs:

```json
{
  "screenshot": {
    "enabled": false,
    "screenshot_only": false,
    "viewport_width": 1920,
    "viewport_height": 1080,
    "quality": 75,
    "max_height": 10000
  }
}
```

When enabled (`--screenshot` or `--preset rich`):

- **PDF/PPTX**: Renders each page/slide as a JPEG image
- **URLs**: Captures full-page screenshots using Playwright

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Enable screenshot capture |
| `screenshot_only` | `false` | Capture screenshots only, skip content extraction (see `--screenshot-only` CLI flag) |
| `viewport_width` | `1920` | Browser viewport width for URL screenshots |
| `viewport_height` | `1080` | Browser viewport height for URL screenshots |
| `quality` | `75` | JPEG compression quality (1-100) |
| `max_height` | `10000` | Maximum screenshot height in pixels |

Screenshots are saved to `output/screenshots/` directory.

::: tip
For URLs, enabling `--screenshot` automatically upgrades the fetch strategy to `playwright` if needed. This ensures the page is fully rendered before capturing.
:::

## Presets

Markitai includes three built-in presets (`rich`, `standard`, `minimal`). You can also define **custom presets** in the config file:

```json
{
  "presets": {
    "my-preset": {
      "llm": true,
      "ocr": false,
      "alt": true,
      "desc": false,
      "screenshot": true
    }
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | boolean | `false` | Enable LLM enhancement |
| `ocr` | boolean | `false` | Enable OCR for scanned documents |
| `alt` | boolean | `false` | Generate image alt text |
| `desc` | boolean | `false` | Generate image descriptions |
| `screenshot` | boolean | `false` | Enable screenshot capture |

Use custom presets via the `--preset` CLI flag:

```bash
markitai document.pdf --preset my-preset
```

## OCR Configuration

Configure Optical Character Recognition for scanned documents. Markitai uses [RapidOCR](https://github.com/RapidAI/RapidOCR) (ONNX Runtime + OpenCV) for OCR processing.

```json
{
  "ocr": {
    "enabled": false,
    "lang": "en"
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Enable OCR for PDF |
| `lang` | `en` | RapidOCR language code |

Supported language codes:
- `en` - English
- `zh` / `ch` - Chinese (Simplified)
- `ja` / `japan` - Japanese
- `ko` / `korean` - Korean
- `ar` / `arabic` - Arabic
- `th` - Thai
- `latin` - Latin languages

::: tip
RapidOCR is included as a dependency and works out of the box. No additional installation required.
:::

## Batch Configuration

Control parallel processing:

```json
{
  "batch": {
    "concurrency": 10,
    "url_concurrency": 5,
    "scan_max_depth": 5,
    "scan_max_files": 10000,
    "state_flush_interval_seconds": 10,
    "heavy_task_limit": 0
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `concurrency` | `10` | Max concurrent file conversions |
| `url_concurrency` | `5` | Max concurrent URL fetches (separate from files) |
| `scan_max_depth` | `5` | Max directory depth to scan |
| `scan_max_files` | `10000` | Max files to process in one run |
| `state_flush_interval_seconds` | `10` | Interval for persisting batch state to disk |
| `heavy_task_limit` | `0` | Limit for CPU-intensive tasks (0 = auto-detect based on RAM) |

::: tip
URL fetching uses a separate concurrency pool because URLs can have high latency (e.g., browser-rendered pages). This prevents slow URLs from blocking local file processing.
:::

## URL Fetch Configuration

Configure how URLs are fetched:

```json
{
  "fetch": {
    "strategy": "auto",
    "playwright": {
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 3000
    },
    "jina": {
      "api_key": "env:JINA_API_KEY",
      "timeout": 30,
      "rpm": 20,
      "no_cache": false,
      "target_selector": null,
      "wait_for_selector": null
    },
    "cloudflare": {
      "api_token": "env:CLOUDFLARE_API_TOKEN",
      "account_id": "env:CLOUDFLARE_ACCOUNT_ID"
    },
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  }
}
```

### Fetch Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | Auto-detect: use playwright for patterns in `fallback_patterns`, static otherwise |
| `static` | Use MarkItDown's built-in URL converter (fast, no JS) |
| `playwright` | Use Playwright for JS-rendered pages (SPA support) |
| `jina` | Use Jina Reader API |
| `cloudflare` | Use Cloudflare Browser Rendering `/markdown` API |

### Playwright Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `timeout` | `30000` | Page load timeout (ms) |
| `wait_for` | `domcontentloaded` | Wait condition: `load`, `domcontentloaded`, `networkidle` |
| `extra_wait_ms` | `3000` | Extra wait time for JS rendering |
| `session_mode` | `isolated` | Session mode: `isolated` (new context per request), `domain_persistent` (reuse context per domain) |
| `session_ttl_seconds` | `600` | TTL for persistent sessions in seconds |
| `wait_for_selector` | `null` | CSS selector to wait for before extraction |
| `cookies` | `null` | Cookies to set: `[{name, value, domain, path}]` |
| `reject_resource_patterns` | `null` | Block resources matching patterns: `["**/*.css"]` |
| `extra_http_headers` | `null` | Additional HTTP headers: `{"Accept-Language": "zh-CN"}` |
| `user_agent` | `null` | Custom User-Agent string |
| `http_credentials` | `null` | HTTP auth credentials: `{username, password}` |

### Jina Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `api_key` | `null` | Jina Reader API key (supports `env:` syntax) |
| `timeout` | `30` | Request timeout in seconds |
| `rpm` | `20` | Rate limit in requests per minute |
| `no_cache` | `false` | Disable Jina's server-side cache |
| `target_selector` | `null` | CSS selector to target specific page content |
| `wait_for_selector` | `null` | CSS selector to wait for before extraction |

### Cloudflare Settings

Cloudflare provides two capabilities through a unified `--cloudflare` flag:

1. **Browser Rendering** (`/markdown` API) for URL-to-markdown conversion
2. **Workers AI toMarkdown** for file-to-markdown conversion (PDF, Office, CSV, XML, images)

```json
{
  "fetch": {
    "cloudflare": {
      "api_token": "env:CLOUDFLARE_API_TOKEN",
      "account_id": "env:CLOUDFLARE_ACCOUNT_ID",
      "timeout": 30000,
      "wait_until": "networkidle0",
      "cache_ttl": 0,
      "reject_resource_patterns": null,
      "user_agent": null,
      "cookies": null,
      "wait_for_selector": null,
      "http_credentials": null,
      "convert_enabled": false
    }
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `api_token` | `null` | Cloudflare API token (supports `env:` syntax) |
| `account_id` | `null` | Cloudflare account ID (supports `env:` syntax) |
| `timeout` | `30000` | Browser Rendering timeout (ms) |
| `wait_until` | `networkidle0` | Wait event for BR: `load`, `domcontentloaded`, `networkidle0` |
| `cache_ttl` | `0` | BR cache TTL in seconds (0 = no cache) |
| `reject_resource_patterns` | `null` | Block resources matching regex patterns: `["/\\.css$/"]` |
| `user_agent` | `null` | Custom User-Agent string for Browser Rendering |
| `cookies` | `null` | Cookies to set before navigation: `[{"name": "k", "value": "v", "url": "..."}]` |
| `wait_for_selector` | `null` | CSS selector to wait for after page load (e.g. `"#content"`) |
| `http_credentials` | `null` | HTTP Basic Auth: `{"username": "u", "password": "p"}` |
| `convert_enabled` | `false` | Enable Workers AI toMarkdown for file conversion |

::: tip
Browser Rendering is available on the Free plan. Workers AI toMarkdown is free for PDF/Office/CSV/XML conversions; image conversion uses Neurons quota.
:::

**How to obtain credentials:**

1. **Account ID** — Log in to the [Cloudflare Dashboard](https://dash.cloudflare.com/). The Account ID is shown in the URL (`dash.cloudflare.com/<account_id>/...`) or on the right sidebar of any zone's **Overview** page.

2. **API Token** — Go to [My Profile → API Tokens](https://dash.cloudflare.com/profile/api-tokens) and click **Create Token**. Use the *Custom token* template with the following permissions:

   | Permission | Access | Required for |
   |------------|--------|--------------|
   | Account / Cloudflare Workers AI | Read | `toMarkdown` file conversion |
   | Account / Browser Rendering | Edit | `/markdown` URL rendering |

   Set **Account Resources** to your target account, then create and copy the token.

3. **Enable Browser Rendering** — In your Cloudflare dashboard, go to **Workers & Pages → Browser Rendering** and follow the prompts to enable it (available on Free plan).

```bash
export CLOUDFLARE_API_TOKEN="your-api-token"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
```

::: warning Limitations & Caveats
- **Concurrency**: Free plan allows **2 concurrent browser instances**. Markitai automatically serializes CF BR requests and retries on 429 rate-limit errors with exponential backoff, so high `url_concurrency` values are safe but won't speed up CF BR fetching.
- **Site compatibility**: Sites with aggressive anti-bot protection (e.g. x.com, twitter.com) may return 400 errors via CF BR. For these sites, use `--playwright` or `--jina` instead.
- **File conversion quality**: For formats that have a local converter (PDF, DOCX, XLSX, etc.), CF Workers AI `toMarkdown` generally produces **lower quality** output than local converters (e.g. less accurate formatting, no image extraction). `--cloudflare` will warn when a better local converter is available. CF `toMarkdown` is most useful for formats without a local converter (`.numbers`, `.ods`, `.svg`, etc.).
:::

### Fetch Policy Engine

The policy engine intelligently orders fetch strategies based on domain characteristics and history. See the [Fetch Policy Guide](/guide/fetch-policy) for details.

```json
{
  "fetch": {
    "policy": {
      "enabled": true,
      "max_strategy_hops": 4
    }
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable intelligent strategy ordering |
| `max_strategy_hops` | `4` | Maximum number of strategies to attempt before giving up |

### Domain Profiles

Configure per-domain fetch overrides for sites with specific requirements:

```json
{
  "fetch": {
    "domain_profiles": {
      "x.com": {
        "wait_for_selector": "[data-testid=tweetText]",
        "wait_for": "domcontentloaded",
        "extra_wait_ms": 1200,
        "prefer_strategy": "playwright"
      }
    }
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `wait_for_selector` | `null` | CSS selector to wait for before content extraction |
| `wait_for` | `null` | Wait condition override: `load`, `domcontentloaded`, `networkidle` |
| `extra_wait_ms` | `null` | Extra wait time override (ms) |
| `prefer_strategy` | `null` | Preferred strategy: `static`, `playwright`, `cloudflare`, `jina` |

### Fallback Patterns

Sites matching these patterns automatically use browser strategy:

```json
{
  "fetch": {
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  }
}
```

## Cache Configuration

Markitai uses a global cache stored at `~/.markitai/cache.db`.

```json
{
  "cache": {
    "enabled": true,
    "no_cache_patterns": [],
    "max_size_bytes": 536870912,
    "global_dir": "~/.markitai"
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable LLM result caching |
| `no_cache` | `false` | Skip reading cache but still write (like `--no-cache` flag) |
| `no_cache_patterns` | `[]` | Glob patterns to skip cache |
| `max_size_bytes` | `536870912` (512MB) | Max cache size |
| `global_dir` | `~/.markitai` | Global cache directory |

### Cache Commands

```bash
# View cache statistics
markitai cache stats

# View detailed statistics (entries, by model)
markitai cache stats --verbose

# View with limit
markitai cache stats --verbose --limit 50

# Clear cache
markitai cache clear
markitai cache clear -y  # Skip confirmation
```

### Disable Cache

```bash
# Disable for entire run
markitai document.pdf --no-cache

# Disable for specific files/patterns
markitai ./docs --no-cache-for "*.pdf"
markitai ./docs --no-cache-for "file1.pdf,reports/**"
```

## Output Configuration

Control output file handling:

```json
{
  "output": {
    "on_conflict": "rename"
  }
}
```

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| `dir` | - | `./output` | Output directory |
| `on_conflict` | `rename`, `overwrite`, `skip` | `rename` | How to handle existing files |
| `allow_symlinks` | - | `false` | Allow symlinks in output paths |

## Log Configuration

Configure logging behavior:

```json
{
  "log": {
    "level": "INFO",
    "format": "text",
    "dir": "~/.markitai/logs",
    "rotation": "10 MB",
    "retention": "7 days"
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `level` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `format` | `text` | Log format: `text` (human-readable) or `json` (structured) |
| `dir` | `~/.markitai/logs` | Log file directory |
| `rotation` | `10 MB` | Rotate when file exceeds this size |
| `retention` | `7 days` | Delete logs older than this |

## Custom Prompts

Customize LLM prompts for different tasks. Each prompt is split into **system** (role definition) and **user** (content template) parts:

```json
{
  "prompts": {
    "dir": "~/.markitai/prompts",
    "cleaner_system": null,
    "cleaner_user": null,
    "image_caption_system": null,
    "image_caption_user": null,
    "image_description_system": null,
    "image_description_user": null,
    "image_analysis_system": null,
    "image_analysis_user": null,
    "page_content_system": null,
    "page_content_user": null,
    "document_enhance_system": null,
    "document_enhance_user": null,
    "document_enhance_complete_system": null,
    "document_enhance_complete_user": null,
    "document_process_system": null,
    "document_process_user": null,
    "document_vision_system": null,
    "document_vision_user": null,
    "url_enhance_system": null,
    "url_enhance_user": null
  }
}
```

Create custom prompt files in the prompts directory:

```
~/.markitai/prompts/
├── cleaner_system.md          # Document cleaning role & rules
├── cleaner_user.md            # Document cleaning content template
├── image_caption_system.md    # Alt text generation role
├── image_caption_user.md      # Alt text content template
├── document_enhance_system.md # Document enhancement role
└── document_process_system.md # Document processing role
```

Set a specific prompt file path:

```json
{
  "prompts": {
    "cleaner_system": "/path/to/my-cleaner-system.md",
    "cleaner_user": "/path/to/my-cleaner-user.md"
  }
}
```

::: tip
The system/user split prevents LLM from accidentally including prompt instructions in its output. System prompts define the role and rules, while user prompts contain the actual content to process.
:::

