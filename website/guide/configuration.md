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
# Create config file in current directory
markitai config init

# Create in specific location
markitai config init -o ~/.markitai/
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
    "scan_max_files": 10000
  },
  "fetch": {
    "strategy": "auto",
    "agent_browser": {
      "command": "agent-browser",
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 1000
    },
    "jina": {
      "api_key": null,
      "timeout": 30
    },
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  },
  "output": {
    "on_conflict": "rename"
  },
  "log": {
    "level": "INFO",
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
Use `env:VAR_NAME` syntax to reference environment variables in the config file.
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

### Markitai Settings

| Variable | Description |
|----------|-------------|
| `MARKITAI_CONFIG` | Path to configuration file |
| `MARKITAI_LOG_DIR` | Directory for log files |

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

These providers require:
1. The respective CLI tool installed and authenticated
2. Optional SDK package: `pip install markitai[claude-agent]` or `pip install markitai[copilot]`

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

Claude Agent SDK supported models:
- Aliases (recommended): `sonnet`, `opus`, `haiku`, `inherit`
- Full model strings: `claude-sonnet-4-5-20250929`, `claude-opus-4-5-20251101`, `claude-opus-4-1-20250805`

GitHub Copilot SDK supported models:
- OpenAI: `gpt-5.2`, `gpt-5.1`, `gpt-5-mini`, `gpt-5.1-codex`
- Anthropic: `claude-sonnet-4.5`, `claude-opus-4.5`, `claude-haiku-4.5`
- Google: `gemini-2.5-pro`, `gemini-3-flash`
- Availability depends on your Copilot subscription

::: warning Model Deprecation Notice
The following models will be **retired on February 13, 2025**:
- `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`, `o4-mini`, `gpt-5`

Please migrate to `gpt-5.2` or other supported models before the deadline.
:::

::: tip Local Providers Support Vision
Local providers (`claude-agent/`, `copilot/`) support image analysis (`--alt`, `--desc`) via file attachments. Make sure to use a vision-capable model (e.g., `copilot/gpt-5.2`, `copilot/claude-sonnet-4.5`).
:::

::: tip Troubleshooting Local Providers
Common errors and solutions:

| Error | Solution |
|-------|----------|
| "SDK not installed" | `pip install markitai[copilot]` or `pip install markitai[claude-agent]` |
| "CLI not found" | Install and authenticate the CLI tool ([Copilot CLI](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli), [Claude Code](https://claude.ai/code)) |
| "Not authenticated" | Run `copilot auth login` or `claude auth login` |
| "Rate limit" | Wait and retry, or check your subscription quota |
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
| `timeout` | seconds | 120 | Request timeout |
| `concurrency` | 1-20 | 10 | Max concurrent LLM requests |

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
    "viewport_width": 1920,
    "viewport_height": 1080,
    "quality": 75,
    "max_height": 10000
  }
}
```

When enabled (`--screenshot` or `--preset rich`):

- **PDF/PPTX**: Renders each page/slide as a JPEG image
- **URLs**: Captures full-page screenshots using agent-browser

| Setting | Default | Description |
|---------|---------|-------------|
| `enabled` | `false` | Enable screenshot capture |
| `viewport_width` | `1920` | Browser viewport width for URL screenshots |
| `viewport_height` | `1080` | Browser viewport height for URL screenshots |
| `quality` | `75` | JPEG compression quality (1-100) |
| `max_height` | `10000` | Maximum screenshot height in pixels |

Screenshots are saved to `output/screenshots/` directory.

::: tip
For URLs, enabling `--screenshot` automatically upgrades the fetch strategy to `browser` if needed. This ensures the page is fully rendered before capturing.
:::

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
    "scan_max_files": 10000
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `concurrency` | `10` | Max concurrent file conversions |
| `url_concurrency` | `5` | Max concurrent URL fetches (separate from files) |
| `scan_max_depth` | `5` | Max directory depth to scan |
| `scan_max_files` | `10000` | Max files to process in one run |

::: tip
URL fetching uses a separate concurrency pool because URLs can have high latency (e.g., browser-rendered pages). This prevents slow URLs from blocking local file processing.
:::

## URL Fetch Configuration

Configure how URLs are fetched:

```json
{
  "fetch": {
    "strategy": "auto",
    "agent_browser": {
      "command": "agent-browser",
      "timeout": 30000,
      "wait_for": "domcontentloaded",
      "extra_wait_ms": 1000
    },
    "jina": {
      "api_key": "env:JINA_API_KEY",
      "timeout": 30
    },
    "fallback_patterns": ["x.com", "twitter.com", "instagram.com", "facebook.com", "linkedin.com", "threads.net"]
  }
}
```

### Fetch Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | Auto-detect: use browser for patterns in `fallback_patterns`, static otherwise |
| `static` | Use MarkItDown's built-in URL converter (fast, no JS) |
| `browser` | Use agent-browser for JS-rendered pages (SPA support) |
| `jina` | Use Jina Reader API |

### Browser Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `command` | `agent-browser` | Path to agent-browser |
| `timeout` | `30000` | Page load timeout (ms) |
| `wait_for` | `domcontentloaded` | Wait condition: `load`, `domcontentloaded`, `networkidle` |
| `extra_wait_ms` | `1000` | Extra wait time for JS rendering |

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
| `no_cache_patterns` | `[]` | Glob patterns to skip cache |
| `max_size_bytes` | `536870912` (512MB) | Max cache size |
| `global_dir` | `~/.markitai` | Global cache directory |

### Cache Commands

```bash
# View cache statistics
markitai cache stats

# View detailed statistics (entries, by model)
markitai cache stats -v

# View with limit
markitai cache stats -v --limit 50

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
| `on_conflict` | `rename`, `overwrite`, `skip` | `rename` | How to handle existing files |

## Log Configuration

Configure logging behavior:

```json
{
  "log": {
    "level": "INFO",
    "dir": "~/.markitai/logs",
    "rotation": "10 MB",
    "retention": "7 days"
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `level` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
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
    "frontmatter_system": null,
    "frontmatter_user": null,
    "image_caption_system": null,
    "image_caption_user": null,
    "image_description_system": null,
    "image_description_user": null,
    "document_process_system": null,
    "document_process_user": null
  }
}
```

Create custom prompt files in the prompts directory:

```
~/.markitai/prompts/
├── cleaner_system.md          # Document cleaning role & rules
├── cleaner_user.md            # Document cleaning content template
├── frontmatter_system.md      # Metadata extraction role
├── frontmatter_user.md        # Metadata extraction template
├── image_caption_system.md    # Alt text generation role
├── image_caption_user.md      # Alt text content template
└── document_enhance_system.md # Vision enhancement role
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
