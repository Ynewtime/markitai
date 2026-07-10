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
markitai config list --format json    # JSON (default)
markitai config list --format table   # Rich table view
markitai config list --format yaml    # YAML (requires pyyaml: uv add pyyaml)
markitai config list --show-secrets   # Reveal original secret values

# Get specific value
markitai config get llm.enabled

# Set value
markitai config set llm.enabled true

# Interactive editor (guided menu)
markitai config edit

# Validate configuration
markitai config validate
markitai config validate ./markitai.json    # Validate specific file
```

`config list` recursively redacts secrets by default, including nested API keys, tokens, cookies, credentials, and every custom HTTP header value. Custom `api_base` values are reduced to their origin. Use `--show-secrets` only for local inspection, and never paste its complete output into an issue, chat, CI log, or other shared channel.

### Full Configuration Example

```json
{
  "llm": {
    "enabled": false,
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-3.1-flash-lite-preview",
          "api_key": "env:GEMINI_API_KEY"
        }
      }
    ],
    "router_settings": {
      "routing_strategy": "simple-shuffle",
      "num_retries": 2,
      "timeout": 120
    },
    "concurrency": 10,
    "pure": false,
    "keep_base": false
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
    },
    "stdout_persist": true,
    "stdout_persist_dir": "~/.markitai/assets",
    "stdout_fetch_external": false
  },
  "ocr": {
    "enabled": false,
    "lang": "en",
    "per_page_routing": true
  },
  "office": {
    "macos_fallback": true
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
    "no_cache": false,
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
    "remote_consent": "always",
    "kreuzberg_convert_enabled": false,
    "defuddle": {
      "timeout": 30,
      "rpm": 20
    },
    "playwright": {
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 3000,
      "wait_for_selector": null,
      "cookies": null,
      "reject_resource_patterns": null,
      "extra_http_headers": null,
      "user_agent": null,
      "http_credentials": null,
      "session_mode": "isolated",
      "session_ttl_seconds": 600
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
      "convert_enabled": false,
      "user_agent": null,
      "cookies": null,
      "wait_for_selector": null,
      "http_credentials": null
    },
    "policy": {
      "enabled": true,
      "max_strategy_hops": 5,
      "strategy_priority": null,
      "local_only_patterns": [],
      "inherit_no_proxy": true
    },
    "domain_profiles": {},
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  },
  "output": {
    "dir": null,
    "on_conflict": "rename",
    "allow_symlinks": false,
    "report": null
  },
  "log": {
    "level": "INFO",
    "format": "text",
    "dir": null,
    "rotation": "10 MB",
    "retention": "7 days"
  },
  "security": {
    "pdf_sanitize": "warn"
  },
  "prompts": {
    "dir": "~/.markitai/prompts"
  }
}
```

::: tip
Use `env:VAR_NAME` syntax to reference environment variables in the config file. For `JINA_API_KEY`, `CLOUDFLARE_API_TOKEN`, and `CLOUDFLARE_ACCOUNT_ID`, you can also just set the environment variable (or add it to `.env`) without configuring anything in the config file; markitai reads them automatically.
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
| `MARKITAI_LANG` | CLI language override (`en` or `zh`) |
| `MARKITAI_PURE` | Enable pure mode (`1`, `true`, or `yes`) |
| `MARKITAI_NO_REMOTE_FETCH` | Hard-disable remote extraction, including explicit remote `-s` strategies (`1`, `true`, or `yes`) |
| `MODEL` | Single-model override when no `model_list` configured |

### `.env` File Loading

Markitai automatically loads `.env` files in the following order (first-loaded values win):

1. `./.env` (current working directory, project-level)
2. `~/.markitai/.env` (user home, global fallback)

Project-level `.env` takes priority, allowing per-project overrides of global settings.

## LLM Configuration

### Supported Providers

Markitai supports multiple LLM providers through [LiteLLM](https://docs.litellm.ai/):

- OpenAI (GPT-5.4)
- Anthropic (Claude Sonnet 4.6)
- Google (Gemini 3.1)
- DeepSeek
- OpenRouter
- Ollama (local models)

#### Local Providers (Subscription-based)

Markitai also supports local providers that use CLI authentication and subscription credits:

- **Claude Agent** (`claude-agent/`): Uses [Claude Agent SDK](https://github.com/anthropics/claude-code) with Claude Code CLI authentication
- **GitHub Copilot** (`copilot/`): Uses [GitHub Copilot SDK](https://github.com/github/copilot-sdk) with Copilot CLI authentication
- **ChatGPT** (`chatgpt/`): Uses ChatGPT subscription via OAuth Device Code Flow and Responses API. No extra SDK required.

These providers require:
1. The respective CLI tool installed and authenticated (or environment variable auth; see below)
2. Optional SDK package: `uv add markitai[claude-agent]` or `uv add markitai[copilot]`

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

ChatGPT provider authenticates via OAuth Device Code Flow on first use. Just configure the model and follow the browser prompt.

::: tip Gemini Access
Gemini is not a local/CLI provider. Use a direct API key (`gemini/`, see [Model Naming](#model-naming)) or route through OpenRouter (`openrouter/google/...`).
:::

### Model Naming

Use the LiteLLM model naming convention:

```
provider/model-name
```

Examples:
- `openai/gpt-5.4`
- `anthropic/claude-sonnet-4-6`
- `gemini/gemini-3.1-flash-lite-preview`
- `deepseek/deepseek-chat`
- `ollama/llama3.2`
- `claude-agent/sonnet` (local, requires Claude Code CLI)
- `copilot/gpt-5.4` (local, requires Copilot CLI)
- `chatgpt/gpt-5.4` (local, requires ChatGPT subscription)

Claude Agent SDK supported models:
- Aliases (recommended): `sonnet`, `opus`, `haiku`, `inherit`
- Full model strings: `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-opus-4-5-20251101`

GitHub Copilot SDK supported models:
- Supports all models available to your Copilot subscription (except o1/o3 reasoning models)
- Examples: `gpt-5.4`, `claude-sonnet-4.6`, `gemini-3.1-pro-preview`, etc.
- Availability depends on your Copilot subscription plan

ChatGPT supported models:
- `gpt-5.4`, `gpt-5.4-codex`, `codex-mini`, etc.

::: warning Deprecated Models
The following models were **retired on February 13, 2025** and are no longer available:
- `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`, `o4-mini`, `gpt-5`, `gpt-5.1`, `gpt-5.2`

Please use `gpt-5.4` or other currently supported models.
:::

::: tip Local Providers Support Vision
Local providers (`claude-agent/`, `copilot/`, `chatgpt/`) support image analysis (`--alt`, `--desc`) via file attachments. Make sure to use a vision-capable model (e.g., `copilot/gpt-5.4`, `chatgpt/gpt-5.4`).
:::

::: tip Troubleshooting Local Providers
Common errors and solutions:

| Error | Solution |
|-------|----------|
| "SDK not installed" | `uv add markitai[copilot]` or `uv add markitai[claude-agent]` |
| "CLI not found" | Install and authenticate the CLI tool ([Copilot CLI](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli), [Claude Code](https://claude.ai/code)) |
| "Not authenticated" | Run `copilot auth login` or `claude auth login`. Alternatively: set `COPILOT_GITHUB_TOKEN`/`GH_TOKEN`/`GITHUB_TOKEN` for Copilot, or `CLAUDE_CODE_USE_BEDROCK=1`/`CLAUDE_CODE_USE_VERTEX=1`/`CLAUDE_CODE_USE_FOUNDRY=1` for Claude. ChatGPT auto-triggers OAuth on first use. |
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

// Azure OpenAI — "azure/<...>" is your Azure deployment name (an alias you chose in Azure Portal), not a model ID
{
  "model": "azure/your-deployment-name",
  "api_key": "env:AZURE_API_KEY",
  "api_base": "https://your-resource.openai.azure.com",
  "api_version": "2025-02-01-preview"
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
  "model": "anthropic/claude-sonnet-4-6",
  "api_key": "env:ANTHROPIC_API_KEY",
  "api_base": "env:ANTHROPIC_BASE_URL"
}
```

::: tip
Common use cases include self-hosted inference servers (vLLM, Ollama, LocalAI), regional API proxies, and third-party API gateways.
:::

::: warning Local Providers and `api_base`
The `api_base` config field does **not** apply to local providers (`claude-agent/`, `copilot/`, `chatgpt/`). These providers run as CLI subprocesses or use OAuth and manage API endpoints internally:

- **Claude Agent**: Set `ANTHROPIC_BASE_URL` to override the API endpoint. If `ANTHROPIC_API_KEY` is also set, the CLI will use it for direct API access instead of subscription authentication. Other routing options: `CLAUDE_CODE_USE_BEDROCK=1`, `CLAUDE_CODE_USE_VERTEX=1`, `CLAUDE_CODE_USE_FOUNDRY=1`.
- **GitHub Copilot**: Endpoint is managed by the Copilot CLI internally and cannot be overridden. For token-based auth, set `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `GITHUB_TOKEN` with a personal access token that has the "Copilot Requests" permission.
- **ChatGPT**: Uses OpenAI's Responses API endpoint. Authentication handled via LiteLLM's built-in OAuth Device Code Flow.
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
          "model": "gemini/gemini-3.1-flash-lite-preview",
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

### Model Token Limits

Both `litellm_params` and `model_info` accept optional token-limit overrides:

```json
{
  "model_name": "default",
  "litellm_params": {
    "model": "gemini/gemini-3.1-flash-lite-preview",
    "api_key": "env:GEMINI_API_KEY",
    "max_tokens": 8192
  },
  "model_info": {
    "max_tokens": 8192,
    "max_input_tokens": 1000000
  }
}
```

| Field | Default | Description |
|-------|---------|--------------|
| `litellm_params.max_tokens` | `null` | Overrides the max **output** tokens requested per call for this model |
| `model_info.max_tokens` | `null` | Max output tokens metadata; auto-detected from litellm if omitted |
| `model_info.max_input_tokens` | `null` | Max input/context tokens metadata; auto-detected from litellm if omitted |

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
| `num_retries` | ≥0 | 2 | Retry count on failure |
| `timeout` | seconds | 120 | Request timeout (base value for adaptive calculation) |
| `fallbacks` | list | `[]` | LiteLLM Router fallback model groups |
| `concurrency` | ≥1 | 10 | Max concurrent LLM requests |

#### Model Weight

Each model in `model_list` accepts a `weight` parameter in `litellm_params` to control traffic distribution:

```json
{
  "model_name": "default",
  "litellm_params": {
    "model": "gemini/gemini-3.1-flash-lite-preview",
    "api_key": "env:GEMINI_API_KEY",
    "weight": 10
  }
}
```

| Value | Behavior |
|-------|----------|
| `weight: 0` | **Disabled**: model is excluded from routing entirely |
| `weight: 1` (default) | Normal priority |
| `weight: 10` | 10x more likely to be selected than weight=1 models |

Set `weight: 0` to temporarily disable a model without removing its configuration. At least one model must have `weight > 0`. This is enforced when the LLM router actually starts (first LLM use), not by `markitai config validate`, so an all-zero-weight config will validate cleanly but fail at first use.

### Adaptive Timeout

Local providers (`claude-agent/`, `copilot/`, `chatgpt/`) use **adaptive timeout calculation** based on request complexity:

- Base timeout: 60 seconds minimum, 600 seconds maximum
- Factors: prompt length, image presence/count, expected output tokens
- Formula:
  1. `timeout = 60 + (prompt_chars / 500)`
  2. If expected output tokens provided: add `tokens / 4`
  3. If images: `timeout *= 1.5` (this also scales the output-token term above), then add `(extra_images - 1) * 10s` for multiple images
  4. Clamp to [60, 600] seconds

This prevents timeouts on large documents while keeping short requests responsive.

### Prompt Caching (Claude Agent)

Claude Agent provider automatically enables **prompt caching** for system prompts of 4096 characters (~4KB) or more. This reduces API costs by caching frequently-used system prompt prefixes.

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
| `stdout_persist` | `true` | Save piped images to persistent asset store |
| `stdout_persist_dir` | `~/.markitai/assets` | Directory for persistent image storage |
| `stdout_fetch_external` | `false` | Download external image URLs in stdout mode |

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

Screenshots are saved to the `.markitai/screenshots/` subdirectory within the output directory.

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
    "lang": "en",
    "per_page_routing": true
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Enable OCR for PDF and standalone images |
| `lang` | `en` | RapidOCR language code |
| `per_page_routing` | `true` | With `--ocr`, keep the native text layer for pages that don't look scanned/garbled and OCR only the remaining pages. Disable to OCR every page |

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

## Office Configuration

Control the macOS MS Office fallback used when LibreOffice is not installed.

```json
{
  "office": {
    "macos_fallback": true
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `macos_fallback` | `true` | On macOS without LibreOffice, drive installed MS Office apps (Word/PowerPoint/Excel) via AppleScript for legacy `.doc`/`.ppt`/`.xls` conversion and PPTX slide rendering |

::: tip
The first conversion triggers a one-time macOS Automation consent dialog per app. Disable this fallback in headless sessions (SSH, CI) where the dialog cannot be answered.
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
    "remote_consent": "always",
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
    "fallback_patterns": ["twitter.com", "x.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  }
}
```

### Fetch Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | Auto-detect: local-first priority order (static → playwright → defuddle → jina → cloudflare); known SPA/JS-heavy domains use playwright → defuddle → jina → cloudflare → static instead. See [Fetch Policy Guide](/guide/fetch-policy) |
| `static` | Use static HTTP fetch with native webextract (fast, no JS) |
| `defuddle` | Use Defuddle API for clean content extraction (free, no auth) |
| `playwright` | Use Playwright for JS-rendered pages (SPA support) |
| `jina` | Use Jina Reader API |
| `cloudflare` | Use Cloudflare Browser Rendering `/content` API (rendered HTML, extracted locally) |

`fetch.kreuzberg_convert_enabled` (default `false`) forces the kreuzberg converter for **file** conversion. It is the config equivalent of the `-b kreuzberg` CLI flag (see [CLI Reference](/guide/cli#b-backend-name)).

### Remote Fetch Consent

For public URLs, `auto` may fall back to a remote extraction service without asking. Local strategies still run first on standard domains. When a process reaches its first remote attempt, Markitai writes a disclosure to stderr before sending the requested URL to the next service in the chain. The notice names the complete service set covered by the process-wide decision: defuddle.md, Jina, Cloudflare, FxTwitter, and Twitter oEmbed. Services are tried one at a time; the URL is not broadcast to all of them.

For public X/Twitter status or article URLs, Playwright may try FxTwitter and then Twitter oEmbed after local DOM extraction fails. This public-URL enrichment does not open its own `ask` prompt, but it uses the same one-time stderr disclosure and honors a process-wide decline already given to the complete-service prompt. Both `fetch.remote_consent=never` and `MARKITAI_NO_REMOTE_FETCH=1` disable it.

Private, local, intranet, and credential-bearing URLs never use remote extraction, even when a remote strategy is selected explicitly. Credential-bearing includes URL userinfo and sensitive query/fragment parameters such as tokens, signatures, credentials, passwords, API keys, and authorization codes. In the `auto` policy chain, domains matched by `fetch.policy.local_only_patterns` or `NO_PROXY` (when `inherit_no_proxy` is enabled) also stay local. For an otherwise public URL, explicitly passing a non-`auto` remote `-s` flag is an intentional override of those pattern-based rules. A remote `fetch.strategy` set only in config remains governed by `fetch.remote_consent` and emits the same first-use disclosure.

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| `fetch.remote_consent` | `ask`, `always`, `never` | `always` | `always`: allow remote fallback for public URLs without asking and disclose the first remote attempt on stderr; `ask`: prompt once per process on an interactive TTY, otherwise skip remote extraction services (the public X/Twitter enrichment exception above only discloses); `never`: local strategies only |

`MARKITAI_NO_REMOTE_FETCH=1` (or `true`/`yes`) is the hard opt-out: it blocks remote extraction even when `-s defuddle`, `-s jina`, or `-s cloudflare` is passed. Without that environment override, explicitly passing one of those CLI flags opts into that service for the run and can override `fetch.remote_consent=never` plus `local_only_patterns`/`NO_PROXY` for an otherwise public URL. The private/local/credential-bearing URL safeguard still applies.

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

### Defuddle Settings

[Defuddle](https://defuddle.md) extracts clean article content from web pages, removing clutter like ads, sidebars, and navigation. Returns Markdown with rich YAML frontmatter (title, author, published, description, word_count).

| Setting | Default | Description |
|---------|---------|-------------|
| `timeout` | `30` | Request timeout in seconds |
| `rpm` | `20` | Rate limit in requests per minute |

```json
{
  "fetch": {
    "defuddle": {
      "timeout": 30,
      "rpm": 20
    }
  }
}
```

::: tip
Defuddle is free and requires no API key or authentication. It's a good default for article-heavy websites.
:::

### Cloudflare Settings

Cloudflare provides two capabilities through a unified `--cloudflare` flag:

1. **Browser Rendering** (`/content` API, which fetches rendered HTML and then extracts locally through the same native webextract pipeline as every other strategy) for URL-to-markdown conversion
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

1. **Account ID**: Log in to the [Cloudflare Dashboard](https://dash.cloudflare.com/). The Account ID is shown in the URL (`dash.cloudflare.com/<account_id>/...`) or on the right sidebar of any zone's **Overview** page.

2. **API Token**: Go to [My Profile → API Tokens](https://dash.cloudflare.com/profile/api-tokens) and click **Create Token**. Use the *Custom token* template with the following permissions:

   | Permission | Access | Required for |
   |------------|--------|--------------|
   | Account / Cloudflare Workers AI | Read | `toMarkdown` file conversion |
   | Account / Browser Rendering | Edit | `/content` URL rendering |

   Set **Account Resources** to your target account, then create and copy the token.

3. **Enable Browser Rendering**: In your Cloudflare dashboard, go to **Workers & Pages → Browser Rendering** and follow the prompts to enable it (available on Free plan).

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
      "max_strategy_hops": 5
    }
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Enable intelligent strategy ordering |
| `max_strategy_hops` | `5` | Maximum number of strategies to attempt before giving up |
| `strategy_priority` | `null` | Custom global strategy order (overrides default priority) |
| `local_only_patterns` | `[]` | Domain/IP patterns restricted to local strategies (NO_PROXY syntax) |
| `inherit_no_proxy` | `true` | Merge `NO_PROXY` env var into `local_only_patterns` |

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
| `wait_for` | `null` | Wait condition override: `load`, `domcontentloaded`, `networkidle` (unset inherits the global `fetch.playwright.wait_for`) |
| `extra_wait_ms` | `null` | Extra wait time override in ms (unset inherits the global `fetch.playwright.extra_wait_ms`) |
| `prefer_strategy` | `null` | Preferred strategy: `static`, `defuddle`, `playwright`, `cloudflare`, `jina` |
| `strategy_priority` | `null` | Custom strategy order for this domain (overrides global and `prefer_strategy`) |
| `skip_auto_scroll` | `false` | Skip auto-scrolling for single-content pages (tweets, issues, docs) |
| `reject_resource_patterns` | `null` | Block Playwright-navigation resources matching these URL patterns (e.g. `["**/analytics/**"]`) |

Markitai ships built-in profiles for `x.com`/`twitter.com` and `github.com`. Setting your own `domain_profiles` entry for the same domain **replaces it entirely** rather than merging field-by-field; any built-in tuning is lost unless you repeat it yourself.

### Fallback Patterns

Sites matching these patterns are treated as SPA/JS-heavy, promoting browser rendering in the strategy order:

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
| `dir` | - | `null` | Output directory |
| `on_conflict` | `rename`, `overwrite`, `skip` | `rename` | How to handle existing files |
| `allow_symlinks` | - | `false` | Allow symlinks in output paths |
| `report` | `true`, `false`, `null` | `null` | Write a JSON conversion report. `null` (default) writes one for batch/URL-batch runs only; `true`/`false` force it on/off for every run |

## Log Configuration

Configure logging behavior:

```json
{
  "log": {
    "level": "INFO",
    "format": "text",
    "dir": null,
    "rotation": "10 MB",
    "retention": "7 days"
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `level` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `format` | `text` | Log format: `text` (human-readable) or `json` (structured) |
| `dir` | `null` | Log file directory (auto-detected if not set) |
| `rotation` | `10 MB` | Rotate when file exceeds this size |
| `retention` | `7 days` | Delete logs older than this |

## Security Configuration

Control PDF hidden-text handling. This is a prompt-injection vector for LLM pipelines (invisible text such as white-on-white, near-zero-size, zero-opacity, or off-page content that would otherwise be silently included in the extracted markdown):

```json
{
  "security": {
    "pdf_sanitize": "warn"
  }
}
```

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| `pdf_sanitize` | `off`, `warn`, `remove` | `warn` | `warn` logs a consolidated advisory naming affected pages; `remove` also strips the matched hidden text from the output; `off` disables detection |

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
├── cleaner_system.md            # Document cleaning role & rules
├── cleaner_user.md              # Document cleaning content template
├── image_caption_system.md      # Alt text generation role
├── image_caption_user.md        # Alt text content template
├── document_process_system.md   # Document processing role
└── url_enhance_system.md        # URL enhancement role
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
