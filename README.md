# MarkIt

[中文文档](docs/README_ZH.md)

Convert documents to Markdown with optional LLM enhancement.

## Features

- **Multi-format Support**: Word (.docx/.doc), PowerPoint (.pptx/.ppt), Excel (.xlsx/.xls), PDF, HTML, Images
- **LLM Enhancement**: Clean headers/footers, fix headings, add frontmatter, generate summaries
- **Image Processing**: Auto compression, format conversion, deduplication, LLM-powered analysis
- **Batch Processing**: Recursive directory conversion with resume capability
- **Multi-Provider LLM**: OpenAI, Anthropic, Google Gemini, Ollama, OpenRouter with concurrent fallback
- **Capability-Based Routing**: Automatically route text tasks to text models, vision tasks to vision models

## Install

```bash
pip install markit
```

**System dependencies** (for .doc/.ppt/.xls):

```bash
# Ubuntu/Debian
sudo apt install libreoffice-core

# macOS
brew install --cask libreoffice

# Windows
scoop install libreoffice
```

**Optional** (better performance):

```bash
# Ubuntu/Debian
sudo apt install pandoc
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
cargo install oxipng
# or: brew install pandoc oxipng

# macOS
brew install pandoc oxipng

# Windows
scoop install pandoc oxipng
#   or download from:
#   Pandoc: https://pandoc.org/installing.html
#   oxipng: https://github.com/shssoichiro/oxipng/releases
```

## Usage

```bash
# Basic conversion
markit convert document.docx

# With LLM enhancement (format cleanup, frontmatter, summary)
markit convert document.docx --llm

# With image analysis (generate alt text)
markit convert document.pdf --analyze-image

# With image description markdown files
markit convert document.pdf --analyze-image-with-md

# Batch convert directory
markit batch ./docs -o ./output -r

# Resume interrupted batch
markit batch ./docs -o ./output --resume

# Fast mode (skip validation, minimal retries)
markit batch ./docs -o ./output --fast
```

## Commands

| Command | Description |
|---------|-------------|
| `markit convert <file>` | Convert single file |
| `markit batch <dir>` | Batch convert directory |
| `markit config init` | Create config file |
| `markit config test` | Validate configuration |
| `markit config list` | Display current settings |
| `markit config locations` | Show config file search paths |
| `markit provider add` | Add LLM provider credential |
| `markit provider test` | Test LLM connectivity |
| `markit provider list` | List configured credentials |
| `markit provider fetch` | Fetch available models from providers |
| `markit model add` | Add a model to config |
| `markit model list` | List configured models |

## Key Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output directory |
| `--llm` | Enable LLM Markdown enhancement |
| `--analyze-image` | Generate image alt text via LLM |
| `--analyze-image-with-md` | Also generate `.md` description files |
| `--no-compress` | Disable image compression |
| `--pdf-engine` | PDF engine: pymupdf4llm (default), pymupdf, pdfplumber |
| `--llm-provider` | Override provider: openai, anthropic, gemini, ollama, openrouter |
| `--llm-model` | Override model name |
| `-r, --recursive` | Process subdirectories (batch) |
| `--resume` | Resume interrupted batch |
| `--fast` | Fast mode (skip validation, minimal retries) |
| `--dry-run` | Show plan without executing |
| `-v, --verbose` | Verbose logging |

## Supported Formats

| Format | Extensions | Engine |
|--------|------------|--------|
| Word | .docx, .doc | MarkItDown (+ LibreOffice for .doc) |
| PowerPoint | .pptx, .ppt | MarkItDown (+ LibreOffice for .ppt) |
| Excel | .xlsx, .xls | MarkItDown (+ LibreOffice for .xls) |
| PDF | .pdf | PyMuPDF4LLM / PyMuPDF / pdfplumber |
| HTML | .html, .htm | MarkItDown |
| Text | .txt | MarkItDown |
| Images | .png, .jpg, .gif, .webp, .bmp | LLM analysis |

## Configuration

Create `markit.yaml` with `markit config init`:

```yaml
log_level: "INFO"
state_file: ".markit-state.json"

image:
  enable_compression: true
  png_optimization_level: 2  # 0-6, higher = slower
  jpeg_quality: 85
  max_dimension: 2048
  filter_small_images: true

concurrency:
  file_workers: 4      # Concurrent file conversions
  image_workers: 8     # Concurrent image processing
  llm_workers: 5       # Concurrent LLM requests

pdf:
  engine: "pymupdf4llm"  # pymupdf4llm, pymupdf, pdfplumber

enhancement:
  enabled: false
  remove_headers_footers: true
  fix_heading_levels: true
  add_frontmatter: true
  generate_summary: true

output:
  default_dir: "output"
  on_conflict: "rename"  # skip, overwrite, rename

prompt:
  output_language: "zh"  # zh, en, auto
  # prompts_dir: "prompts"  # Custom prompts directory (relative to cwd, optional)
  # Files should be named: {type}_{lang}.md
  #   type: enhancement, image_analysis, summary
  #   lang: zh, en (based on output_language)
  #   e.g., enhancement_zh.md, image_analysis_en.md
  # If not found, falls back to built-in prompts
  # Custom prompt file paths (highest priority, override prompts_dir):
  # image_analysis_prompt: "my_prompts/image_analysis.md"
  # enhancement_prompt: "my_prompts/enhancement.md"
  # summary_prompt: "my_prompts/summary.md"

# LLM Configuration
# Credentials and models are defined separately for flexibility
llm:
  concurrent_fallback_enabled: true
  concurrent_fallback_timeout: 60  # Seconds before starting backup model
  max_request_timeout: 300

  credentials:
    - id: "openai-main"
      provider: "openai"
      # api_key: "sk-..."  # Or use OPENAI_API_KEY env var
    - id: "deepseek"
      provider: "openai"
      base_url: "https://api.deepseek.com"
      api_key_env: "DEEPSEEK_API_KEY"

  models:
    - name: "GPT-4o"
      model: "gpt-4o"
      credential_id: "openai-main"
      capabilities: ["text", "vision"]
      cost:  # Optional: for cost tracking
        input_per_1m: 2.50
        output_per_1m: 10.00
    - name: "deepseek-chat"
      model: "deepseek-chat"
      credential_id: "deepseek"
      capabilities: ["text"]  # text-only model
```

**Environment variables:**

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="..."
export DEEPSEEK_API_KEY="..."
```

## Output Structure

```
output/
  document.docx.md              # Converted markdown
  assets/
    document.docx.001.png       # Extracted images
    document.docx.001.png.md    # Image description (with --analyze-image-with-md)
```

## Contributing

See [Contributing Guide](docs/CONTRIBUTING.md) for architecture details, setup instructions, and contribution guidelines.

## License

MIT
