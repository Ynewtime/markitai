# Getting Started

## Prerequisites

### Required

- **Python 3.11+** - Required runtime
- **[uv](https://docs.astral.sh/uv/)** - Package manager (recommended) or pip

### Optional Dependencies

These are required for specific features:

| Dependency | Required For | Installation |
|------------|--------------|--------------|
| **[Node.js 22+](https://nodejs.org/)** | `--agent-browser` (SPA rendering) | See [nodejs.org](https://nodejs.org/) |
| **[agent-browser](https://www.npmjs.com/package/agent-browser)** | `--agent-browser` | `pnpm add -g agent-browser && agent-browser install` |
| **Jina API Key** | `--jina` (URL conversion) | Set `JINA_API_KEY` env var |
| **LLM API Key** | `--llm` (AI enhancement) | Set `OPENAI_API_KEY` or provider-specific key |

::: tip Browser Automation
For SPA websites (Twitter, React apps, etc.), install `agent-browser`:
```bash
pnpm add -g agent-browser
agent-browser install              # Download Chromium browser
agent-browser install --with-deps  # Linux: also install system dependencies
```
Then use `--agent-browser` flag to enable browser rendering.
:::

## Installation

```bash
# Using uv (recommended)
uv add markitai

# Or using pip
pip install markitai
```

## Quick Start

### Basic Conversion

Convert a single document to Markdown:

```bash
markitai document.docx
```

This saves the output to the current directory and prints it to stdout. Use `-o` to specify a different output directory:

```bash
markitai document.docx -o output/
```

### URL Conversion

Convert web pages directly:

```bash
markitai https://example.com/article
```

### LLM Enhancement

Enable AI-powered format cleaning and optimization:

```bash
markitai document.docx --llm
```

This requires setting up an API key (see [Configuration](/guide/configuration)).

### Using Presets

Markitai provides three presets for common use cases:

```bash
# Rich: LLM + alt text + descriptions + screenshots
markitai document.pdf --preset rich

# Standard: LLM + alt text + descriptions
markitai document.pdf --preset standard

# Minimal: Basic conversion only
markitai document.pdf --preset minimal
```

### Batch Processing

Convert multiple files in a directory:

```bash
markitai ./docs -o ./output
```

Resume interrupted batch processing:

```bash
markitai ./docs -o ./output --resume
```

## Output Structure

```
output/
├── document.docx.md        # Basic Markdown output
├── document.docx.llm.md    # LLM-enhanced version (when --llm is used)
├── assets/
│   ├── document.docx.0001.jpg
│   └── images.json         # Image descriptions
```

## Supported Formats

| Format | Extensions |
|--------|------------|
| Word | `.docx`, `.doc` |
| PowerPoint | `.pptx`, `.ppt` |
| Excel | `.xlsx`, `.xls` |
| PDF | `.pdf` |
| Text | `.txt`, `.md` |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp` |
| URLs | `http://`, `https://` |

## Next Steps

- [Configuration](/guide/configuration) - Configure LLM providers and other settings
- [CLI Reference](/guide/cli) - Full command reference
