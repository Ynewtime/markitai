# MarkIt

Convert documents to Markdown with optional LLM enhancement.

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

# With LLM enhancement (format cleanup, frontmatter)
markit convert document.docx --llm

# With image analysis
markit convert document.pdf --analyze-image

# Batch convert
markit batch ./docs -o ./output -r
```

## Commands

| Command | Description |
|---------|-------------|
| `markit convert <file>` | Convert single file |
| `markit batch <dir>` | Batch convert directory |
| `markit config show` | Show current config |
| `markit provider test` | Test LLM connectivity |

## Key Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output directory |
| `--llm` | Enable LLM Markdown enhancement |
| `--analyze-image` | Generate image alt text via LLM |
| `--analyze-image-with-md` | Also generate `.md` description files |
| `--pdf-engine` | PDF engine: pymupdf4llm, pymupdf, pdfplumber |
| `--llm-provider` | openai, anthropic, gemini, ollama, openrouter |
| `-r, --recursive` | Process subdirectories (batch) |
| `--resume` | Resume interrupted batch |

## Supported Formats

| Format | Extensions | Engine |
|--------|------------|--------|
| Word | .docx, .doc | MarkItDown (+ LibreOffice for .doc) |
| PowerPoint | .pptx, .ppt | MarkItDown (+ LibreOffice for .ppt) |
| Excel | .xlsx, .xls | MarkItDown (+ LibreOffice for .xls) |
| PDF | .pdf | PyMuPDF4LLM / PyMuPDF / pdfplumber |
| HTML | .html, .htm | MarkItDown |
| Images | .png, .jpg, .gif, .webp | LLM analysis |

## Configuration

Create `markit.toml` with `markit config init`:

```toml
log_level = "INFO"

[output]
default_dir = "output"

[image]
enable_compression = true

# New Schema (recommended for multiple models)
# Define credentials separately from models
[[llm.credentials]]
id = "openai-main"
provider = "openai"
# api_key = "sk-..."  # Or use OPENAI_API_KEY env var

[[llm.models]]
name = "GPT-4o"
model = "gpt-4o"
credential_id = "openai-main"
capabilities = ["text", "vision"]

# Legacy Schema (simpler, still supported)
[[llm.providers]]
provider = "ollama"
model = "llama3.2-vision"
base_url = "http://localhost:11434"
```

**Environment variables:**

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Output Structure

```
output/
  document.docx.md
  assets/
    document.docx.001.png
    document.docx.001.png.md    # with --analyze-image-with-md
```

## License

MIT
