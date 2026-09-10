# Configuration

## Configuration Priority

Highest to lowest:

1. Command-line flags
2. Environment variables
3. Configuration file
4. Built-in defaults

## Configuration File

markitai reads the first config file it finds:

1. The path given with `--config`
2. `MARKITAI_CONFIG`
3. `./markitai.json` in the current directory
4. `~/.markitai/config.json`

### Initialize Configuration

```bash
markitai init            # guided setup
markitai init --yes      # defaults without prompts
markitai init --local    # write ./markitai.json
```

### View Configuration

```bash
markitai config list                  # effective settings, secrets redacted
markitai config get llm.enabled
markitai config set llm.enabled true
markitai config edit                  # guided menu
markitai config validate
```

Secrets are redacted in `config list`, including nested API keys, tokens, cookies and custom headers. `--show-secrets` reveals them; keep that output on your machine.

### Full Configuration Example

Every setting with its default value:

:::: details markitai.json with all defaults

```json
{
  "llm": {
    "enabled": false,
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "gemini/gemini-flash-lite-latest",
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
    "max_requests_per_document": 50,
    "max_cost_per_document_usd": 0,
    "max_vision_pages_per_document": 0,
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
  "history": {
    "record": false
  },
  "prompts": {
    "dir": "~/.markitai/prompts"
  }
}
```

::::

Any string value can reference an environment variable with `env:VAR_NAME`. `JINA_API_KEY`, `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` are picked up from the environment even without a config entry.

## Environment Variables

### API Keys

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |
| `GEMINI_API_KEY` | Google Gemini |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `OPENROUTER_API_KEY` | OpenRouter |
| `JINA_API_KEY` | Jina Reader |
| `CLOUDFLARE_API_TOKEN` | Cloudflare (Browser Rendering, Workers AI) |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID |

### Markitai Settings

| Variable | Description |
|----------|-------------|
| `MODEL` | The model to use when no `model_list` is configured |
| `MARKITAI_CONFIG` | Path to the config file |
| `MARKITAI_LOG_DIR` | Directory for log files |
| `MARKITAI_LOG_FORMAT` | `text` or `json` |
| `MARKITAI_LANG` | CLI language, `en` or `zh` |
| `MARKITAI_PURE` | Enable pure mode (`1`, `true`, `yes`) |
| `MARKITAI_RECORD_HISTORY` | Record CLI runs to the web workspace history (`1`, `true`, `yes`, `on`) |
| `MARKITAI_NO_REMOTE_FETCH` | Never send URLs to remote services, even with an explicit `-s` (`1`, `true`, `yes`) |
| `MARKITAI_NO_VLM_OCR` | With `--ocr --llm`, use local RapidOCR instead of the vision model (`1`, `true`, `yes`) |
| `MARKITAI_STATIC_HTTP` | Static fetch client: `httpx` (default) or `curl_cffi` |
| `MARKITAI_SERVE_TOKEN` | Fixed access token for `markitai serve` |
| `MARKITAI_INSTALL_OPTIONAL` | Setup script: install optional components without prompting |
| `MARKITAI_USE_MIRROR` | Setup script: `1` always offers a package mirror, `0` never asks |
| `MARKITAI_VERSION` | Setup script: version to install |

### `.env` File Loading

markitai loads `./.env` first and `~/.markitai/.env` second. Values from the first file win, so a project can override global settings.

## LLM Configuration

### Supported Providers

Any [LiteLLM](https://docs.litellm.ai/) provider works with an API key: OpenAI, Anthropic, Google, DeepSeek, OpenRouter, Ollama and more.

Three subscription providers sign in through their own CLI or OAuth instead:

| Provider | Prefix | Sign-in | Extra |
|----------|--------|---------|-------|
| Claude Code | `claude-agent/` | `markitai auth claude login` | `markitai[claude-agent]` |
| GitHub Copilot | `copilot/` | `markitai auth copilot login` | `markitai[copilot]` |
| ChatGPT | `chatgpt/` | OAuth device code on first use | — |

The Claude Code and Copilot CLIs must be installed first: `curl -fsSL https://claude.ai/install.sh | bash` and `curl -fsSL https://gh.io/copilot-install | bash` (Windows: `irm https://claude.ai/install.ps1 | iex` and `winget install GitHub.Copilot`).

Gemini has no subscription sign-in. Use an API key (`gemini/`) or go through OpenRouter (`openrouter/google/...`).

### Model Naming

Models are named `provider/model`, following LiteLLM:

- `openai/gpt-5.6`
- `anthropic/claude-sonnet-4-6`
- `gemini/gemini-flash-lite-latest`
- `deepseek/deepseek-v4-flash`
- `ollama/llama3.2`
- `claude-agent/sonnet`, `copilot/gpt-5.6`, `chatgpt/gpt-5.6` (subscription providers)

#### Defaults markitai picks for you

`markitai init` and the automatic key detection pick the cheap, fast tier of whichever provider they find:

| Provider | Default model |
|---|---|
| Claude Code | `claude-agent/sonnet` |
| GitHub Copilot | `copilot/claude-haiku-4.5` |
| ChatGPT | `chatgpt/gpt-5.6` |
| Anthropic | `anthropic/claude-haiku-4-5` |
| OpenAI | `openai/gpt-5.6-luna` |
| Gemini | `gemini/gemini-flash-lite-latest` |
| DeepSeek | `deepseek/deepseek-v4-flash` |
| OpenRouter | `openrouter/google/gemini-3.1-flash-lite` |

Set `model_list` to use anything else. A model the provider has retired only produces a startup warning; markitai never rewrites your choice.

Image analysis (`--alt`, `--desc`) needs a vision-capable model. The subscription providers support it through file attachments.

Common errors:

| Error | Fix |
|-------|-----|
| "SDK not installed" | Install `markitai[copilot]` or `markitai[claude-agent]` |
| "CLI not found" | Install the [Copilot CLI](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli) or [Claude Code](https://claude.ai/code) |
| "Not authenticated" | Run `markitai auth copilot login` or `markitai auth claude login`. ChatGPT signs in on first use |
| "Rate limit" | Wait and retry, or check your subscription quota |

`markitai doctor` reports sign-in status with hints.

### Custom API Endpoint

`api_base` points a provider at another endpoint: a self-hosted server, a regional proxy or an API gateway. It accepts `env:VAR_NAME` like `api_key`:

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

Two more shapes:

```json
// Local Ollama
{ "model": "ollama/llama3.2", "api_base": "http://localhost:11434" }

// Azure OpenAI: the model is your deployment name, not a model ID
{
  "model": "azure/your-deployment-name",
  "api_key": "env:AZURE_API_KEY",
  "api_base": "https://your-resource.openai.azure.com",
  "api_version": "2025-02-01-preview"
}
```

`api_base` does not apply to the subscription providers. Claude Code honours `ANTHROPIC_BASE_URL`; Copilot and ChatGPT manage their endpoints themselves.

### Vision Models

Vision capability is detected automatically from LiteLLM. To override it, set `model_info.supports_vision` on the model entry:

```json
{
  "model_name": "default",
  "litellm_params": { "model": "gemini/gemini-flash-lite-latest", "api_key": "env:GEMINI_API_KEY" },
  "model_info": { "supports_vision": true }
}
```

### Model Token Limits

| Field | Default | Description |
|-------|---------|-------------|
| `litellm_params.max_tokens` | `null` | Max output tokens requested per call |
| `model_info.max_tokens` | `null` | Max output tokens metadata; detected from LiteLLM if omitted |
| `model_info.max_input_tokens` | `null` | Max context tokens metadata; detected from LiteLLM if omitted |

### Router Settings

`llm.router_settings` and its siblings control how requests are spread over several models and how much one document may consume:

| Setting | Default | Description |
|---------|---------|-------------|
| `routing_strategy` | `simple-shuffle` | `simple-shuffle`, `least-busy`, `usage-based-routing` or `latency-based-routing`. Subscription providers always use weighted random |
| `num_retries` | `2` | Retries per request |
| `timeout` | `120` | Request timeout in seconds |
| `fallbacks` | `[]` | Group fallbacks, e.g. `[{"default": ["backup"]}]`. Models not in a fallback group only receive traffic through fallback |
| `concurrency` | `10` | Concurrent LLM requests |
| `max_requests_per_document` | `50` | Stop enhancing a document after this many requests and keep the plain output. `0` disables |
| `max_cost_per_document_usd` | `0` | Stop enhancing a document once it has spent this much. `0` disables |
| `max_vision_pages_per_document` | `0` | Max page images sent to a vision model per document. Oversized documents convert without vision. `0` disables |

#### Model Weight

Each entry in `model_list` accepts `weight` inside `litellm_params`. `1` is normal, `10` is ten times as likely to be picked, `0` disables the model without deleting it. At least one model must have a weight above zero; this is checked at first use, not by `config validate`.

```json
{
  "model_name": "default",
  "litellm_params": { "model": "gemini/gemini-flash-lite-latest", "api_key": "env:GEMINI_API_KEY", "weight": 10 }
}
```

### Adaptive Timeout

The subscription providers scale the request timeout with prompt length, image count and expected output, between 60 and 600 seconds, so large documents do not time out and short requests stay responsive.

### Prompt Caching (Claude Agent)

Claude Code caches system prompts of 4 KB or more automatically. Nothing to configure; `markitai cache stats --verbose` shows the numbers.

## Image Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `alt_enabled` | `false` | Generate alt text with the LLM |
| `desc_enabled` | `false` | Generate image descriptions |
| `compress` | `true` | Compress extracted images |
| `quality` | `75` | JPEG/WebP quality (1–100) |
| `format` | `jpeg` | `jpeg`, `png` or `webp` |
| `max_width` | `1920` | Max width in pixels |
| `max_height` | `99999` | Max height in pixels |
| `filter.min_width` | `50` | Skip narrower images |
| `filter.min_height` | `50` | Skip shorter images |
| `filter.min_area` | `5000` | Skip smaller images |
| `filter.deduplicate` | `true` | Drop duplicate images |
| `stdout_persist` | `true` | In stdout mode, keep images in a persistent store |
| `stdout_persist_dir` | `~/.markitai/assets` | Where that store lives |
| `stdout_fetch_external` | `false` | Download external image URLs in stdout mode |

## Screenshot Configuration

Screenshots render PDF pages and PPTX slides as JPEG, and capture full-page images of URLs with Playwright. They land in `.markitai/screenshots/`.

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Same as `--screenshot` |
| `screenshot_only` | `false` | Same as `--screenshot-only` |
| `viewport_width` | `1920` | Browser viewport width for URLs |
| `viewport_height` | `1080` | Browser viewport height for URLs |
| `quality` | `75` | JPEG quality (1–100) |
| `tile_height` | `2000` | Tall URL screenshots are cut into tiles of this height so a vision model can read them |
| `max_height` | `10000` | Height cap used only when `tile_height` is `0` |

## Presets

Three presets are built in (`minimal`, `standard`, `rich`). Define your own under `presets`:

```json
{
  "presets": {
    "my-preset": { "llm": true, "ocr": false, "alt": true, "desc": false, "screenshot": true }
  }
}
```

Each of the five keys defaults to `false`. Use it with `markitai document.pdf --preset my-preset`.

## OCR Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Same as `--ocr` |
| `lang` | `en` | Language: `en`, `zh`, `ja`, `ko`, `ar`, `th` or `latin` |
| `per_page_routing` | `true` | Keep the native text layer on pages that look fine and OCR only the rest. `false` OCRs every page |

Local OCR uses [RapidOCR](https://github.com/RapidAI/RapidOCR) from the `ocr` extra:

```bash
uv tool install "markitai[ocr]" --force
```

With `--ocr --llm` and a vision-capable model, no extra is needed: the model reads the page images. `MARKITAI_NO_VLM_OCR=1` forces the local path.

## Office Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `macos_fallback` | `true` | On a Mac without LibreOffice, drive an installed PowerPoint to render PPTX slides |

The first render pops a one-time macOS permission dialog. Set this to `false` on headless Macs (SSH, CI), where nobody can answer it.

## Batch Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `concurrency` | `10` | Concurrent file conversions |
| `url_concurrency` | `5` | Concurrent URL fetches, separate so slow pages never block files |
| `scan_max_depth` | `5` | Directory scan depth |
| `scan_max_files` | `10000` | Max files per run |
| `state_flush_interval_seconds` | `10` | How often batch state is saved for `--resume` |
| `heavy_task_limit` | `0` | Cap on CPU-heavy tasks; `0` picks one from available RAM |

## URL Fetch Configuration

```json
{
  "fetch": {
    "strategy": "auto",
    "remote_consent": "always",
    "playwright": { "timeout": 30000, "wait_for": "domcontentloaded", "extra_wait_ms": 3000 },
    "jina": { "api_key": "env:JINA_API_KEY" },
    "cloudflare": { "api_token": "env:CLOUDFLARE_API_TOKEN", "account_id": "env:CLOUDFLARE_ACCOUNT_ID" },
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  }
}
```

### Fetch Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | Local first: static, then Playwright, then Defuddle, Jina, Cloudflare. Known JavaScript-heavy domains start with Playwright. See [Fetch Policy](/guide/fetch-policy) |
| `static` | Plain HTTP with the built-in extractor. Fast, no JavaScript |
| `playwright` | Browser rendering for JavaScript pages |
| `defuddle` | Defuddle API, free, no key |
| `jina` | Jina Reader API |
| `cloudflare` | Cloudflare Browser Rendering; the rendered HTML is extracted locally |

### Remote Fetch Consent

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| `fetch.remote_consent` | `always`, `ask`, `never` | `always` | `always`: fall back to remote services for public URLs and print a notice on stderr before the first attempt. `ask`: prompt once per process on a terminal, skip every remote service otherwise. `never`: local strategies only |

Private, intranet and credential-bearing URLs never go to a remote service, whatever this is set to. Domains in `fetch.policy.local_only_patterns` and `NO_PROXY` stay local in the `auto` chain. An explicit `-s defuddle`, `-s jina` or `-s cloudflare` overrides `never` and the pattern rules for a public URL; `MARKITAI_NO_REMOTE_FETCH=1` blocks even that.

The X/Twitter enrichment path (FxTwitter, Twitter oEmbed) follows the same consent decision as every other remote service.

### Playwright Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `timeout` | `30000` | Page load timeout in ms |
| `wait_for` | `domcontentloaded` | `load`, `domcontentloaded` or `networkidle` |
| `extra_wait_ms` | `3000` | Extra wait for JavaScript after the load event |
| `session_mode` | `isolated` | `isolated` (fresh context per request) or `domain_persistent` (reuse per domain) |
| `session_ttl_seconds` | `600` | Lifetime of a persistent session |
| `wait_for_selector` | `null` | CSS selector to wait for |
| `cookies` | `null` | `[{name, value, domain, path}]` |
| `reject_resource_patterns` | `null` | Block matching requests, e.g. `["**/*.css"]` |
| `extra_http_headers` | `null` | `{"Accept-Language": "zh-CN"}` |
| `user_agent` | `null` | Custom User-Agent |
| `http_credentials` | `null` | `{username, password}` for HTTP auth |

### Jina Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `api_key` | `null` | Jina Reader API key (`env:` works) |
| `timeout` | `30` | Request timeout in seconds |
| `rpm` | `20` | Requests per minute |
| `no_cache` | `false` | Bypass Jina's server-side cache |
| `target_selector` | `null` | CSS selector for the content to extract |
| `wait_for_selector` | `null` | CSS selector to wait for |

### Defuddle Settings

[Defuddle](https://defuddle.md) extracts clean article content and returns Markdown with rich frontmatter. It is free and needs no key.

| Setting | Default | Description |
|---------|---------|-------------|
| `timeout` | `30` | Request timeout in seconds |
| `rpm` | `20` | Requests per minute |

### Cloudflare Settings

Cloudflare offers two things, chosen independently: **Browser Rendering** (`-s cloudflare`) fetches rendered HTML for URLs, and **Workers AI toMarkdown** (`-b cloudflare`) converts files.

| Setting | Default | Description |
|---------|---------|-------------|
| `api_token` | `null` | API token (`env:` works) |
| `account_id` | `null` | Account ID (`env:` works) |
| `timeout` | `30000` | Browser Rendering timeout in ms |
| `wait_until` | `networkidle0` | `load`, `domcontentloaded` or `networkidle0` |
| `cache_ttl` | `0` | Browser Rendering cache TTL in seconds |
| `reject_resource_patterns` | `null` | Block matching requests, e.g. `["/\\.css$/"]` |
| `user_agent` | `null` | Custom User-Agent |
| `cookies` | `null` | `[{"name": "k", "value": "v", "url": "..."}]` |
| `wait_for_selector` | `null` | CSS selector to wait for |
| `http_credentials` | `null` | `{"username": "u", "password": "p"}` |
| `convert_enabled` | `false` | Enable Workers AI toMarkdown for files |

To get credentials:

1. **Account ID**: shown in the [dashboard](https://dash.cloudflare.com/) URL, `dash.cloudflare.com/<account_id>/...`.
2. **API token**: [My Profile → API Tokens](https://dash.cloudflare.com/profile/api-tokens), *Create Token*, custom token with *Browser Rendering: Edit* and *Workers AI: Read* on your account.
3. **Enable Browser Rendering** under *Workers & Pages → Browser Rendering*. It is available on the Free plan.

```bash
export CLOUDFLARE_API_TOKEN="your-api-token"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
```

The Free plan allows two concurrent browser sessions; markitai serializes its requests and retries on rate limits. Sites with aggressive anti-bot protection (x.com, for example) may fail through Cloudflare; use `-s playwright` or `-s jina` there. For files, the native converters usually give better results; toMarkdown is most useful for formats markitai cannot convert locally.

### Fetch Policy, Domain Profiles and Fallback Patterns {#fetch-policy-domain-profiles}

The policy engine orders strategies per domain and remembers which domains need a browser. Its options, the domain-profile fields and the built-in profiles are documented in the [Fetch Policy guide](/guide/fetch-policy#configuration).

Two rules surprise people: a custom `domain_profiles` entry replaces the built-in profile for that domain instead of merging with it, and `auto` treats every domain in `fallback_patterns` as JavaScript-heavy and starts with the browser.

### Proxies

`HTTPS_PROXY`, `HTTP_PROXY` and `ALL_PROXY` are honoured, with `NO_PROXY` as the bypass list. When none is set, the operating system proxy is used: Windows internet settings, macOS network settings, and the manual HTTP proxy of a GNOME or KDE desktop on Linux. PAC, SOCKS-only and authenticated desktop proxies are not imported; set the environment variables instead.

## Cache Configuration

LLM results are cached in `~/.markitai/cache.db`, so converting the same document again is free.

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `true` | Cache LLM results |
| `no_cache` | `false` | Skip reads but keep writing (like `--no-cache`) |
| `no_cache_patterns` | `[]` | Globs that bypass the cache |
| `max_size_bytes` | `536870912` | Max cache size (512 MB) |
| `global_dir` | `~/.markitai` | Cache directory |

```bash
markitai cache stats --verbose         # what is cached, by model
markitai cache clear
markitai document.pdf --no-cache       # bypass for one run
markitai ./docs --no-cache-for "*.pdf"
```

## Output Configuration

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| `dir` | — | `null` | Output directory |
| `on_conflict` | `rename`, `overwrite`, `skip` | `rename` | What to do when the output file exists |
| `allow_symlinks` | — | `false` | Allow symlinks in output paths |
| `report` | `true`, `false`, `null` | `null` | Write a JSON report. `null` writes one for batch runs only |
| `profile` | `rag`, `obsidian`, `okf`, `null` | `null` | [Output profile](./output-profiles.md) |
| `wikilinks` | `true`, `false` | `false` | With `obsidian`, write image links as `![[assets/x.png]]` |

## Log Configuration

File logging starts when `dir` is set.

| Setting | Default | Description |
|---------|---------|-------------|
| `level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |
| `format` | `text` | `text` or `json` |
| `dir` | `null` | Log directory |
| `rotation` | `10 MB` | Rotate when a file exceeds this size |
| `retention` | `7 days` | Delete older logs |

## Security Configuration

PDFs can carry invisible text (white on white, zero size, off page) that would silently end up in the Markdown and in any LLM prompt built from it.

| Setting | Options | Default | Description |
|---------|---------|---------|-------------|
| `pdf_sanitize` | `off`, `warn`, `remove` | `warn` | `warn` logs which pages carry hidden text, `remove` also strips it, `off` skips the check |

## Custom Prompts

Every LLM task has a system prompt (role and rules) and a user prompt (the content template). Override either by dropping a Markdown file into the prompts directory, or by pointing a setting at a file:

```text
~/.markitai/prompts/
├── cleaner_system.md            # document cleaning
├── cleaner_user.md
├── image_caption_system.md      # alt text
├── image_description_system.md  # image descriptions
├── document_process_system.md   # document processing
└── url_enhance_system.md        # URL enhancement
```

```json
{
  "prompts": {
    "dir": "~/.markitai/prompts",
    "cleaner_system": "/path/to/my-cleaner-system.md"
  }
}
```

The available keys are `cleaner`, `image_caption`, `image_description`, `image_analysis`, `document_process`, `document_vision` and `url_enhance`, each with a `_system` and a `_user` variant.
