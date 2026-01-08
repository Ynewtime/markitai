# MarkIt

Intelligent document to Markdown conversion tool with LLM enhancement.

## Table of Contents

- [Quick Start](#quick-start)
- [Detailed Guide](#detailed-guide)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Command Reference](#command-reference)
  - [Supported Formats](#supported-formats)
  - [LLM Providers](#llm-providers)
  - [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

---

## Quick Start

### Install

```bash
# Using uv (recommended)
uv pip install markit

# Or using pip
pip install markit
```

**System Dependencies** (for legacy Office formats):

```bash
# Ubuntu/Debian
sudo apt install libreoffice-core pandoc

# macOS
brew install --cask libreoffice
brew install pandoc

# Windows
# Install LibreOffice from https://www.libreoffice.org/
# Install Pandoc from https://pandoc.org/
```

### Basic Usage

```bash
# Convert a single file
markit convert document.docx

# Convert with LLM enhancement
markit convert document.docx --llm

# Batch convert a directory
markit batch ./documents -o ./output

# Batch convert recursively
markit batch ./documents -o ./output -r

# Show help
markit -h
markit convert -h
markit batch -h
```

### Output

Converted files are saved with the format `<original_filename>.md`:

```
input/
  report.docx
  data.xlsx

output/
  report.docx.md
  data.xlsx.md
  assets/
    image_001.png
    image_002.png
```

With `--analyze-image-with-md`, detailed description files are also generated:

```
output/
  report.pdf.md
  assets/
    image_001.png
    image_001.png.md    # Detailed image description
    image_002.png
    image_002.png.md
```

---

## Detailed Guide

### Installation

#### Python Package

Requires Python 3.12+

```bash
# Using uv (recommended)
uv pip install markit

# Using pip
pip install markit

# Install with all optional dependencies
pip install "markit[all]"
```

#### System Dependencies

MarkIt requires external tools for certain conversions:

| Dependency | Required For | Installation |
|------------|--------------|--------------|
| LibreOffice | .doc, .ppt, .xls (legacy formats) | See below |
| Pandoc | Fallback converter | See below |

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install -y libreoffice-core pandoc

# For image compression (optional)
sudo apt install -y oxipng
```

**macOS:**

```bash
brew install --cask libreoffice
brew install pandoc oxipng
```

**Windows:**

1. Download and install [LibreOffice](https://www.libreoffice.org/download/download/)
2. Download and install [Pandoc](https://pandoc.org/installing.html)

#### Verify Installation

```bash
# Check markit
markit --version

# Check dependencies
which soffice || where soffice
which pandoc || where pandoc
```

---

### Configuration

MarkIt can be configured via configuration file, environment variables, or command-line arguments.

**Priority (highest to lowest):**
1. Command-line arguments
2. Environment variables
3. Configuration file (`markit.toml`)
4. Default values

#### Configuration File

Create `markit.toml` in your project directory:

```bash
# Copy the example configuration
cp markit.example.toml markit.toml
```

**Basic configuration:**

```toml
log_level = "INFO"

[output]
default_dir = "output"
on_conflict = "rename"

[image]
enable_compression = true
filter_small_images = true
```

**Full configuration example:** See [markit.example.toml](markit.example.toml)

#### Environment Variables

All settings can be overridden via environment variables with `MARKIT_` prefix:

```bash
# Logging
export MARKIT_LOG_LEVEL="DEBUG"
export MARKIT_LOG_FILE="logs/markit.log"

# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."

# Concurrency
export MARKIT_CONCURRENCY__FILE_WORKERS=8
```

#### Log File Behavior

By default, logs are output to console (stderr) only. To save logs to a file:

```toml
# In markit.toml
log_file = "logs/markit.log"
```

Or via environment variable:

```bash
export MARKIT_LOG_FILE="logs/markit.log"
```

---

### Command Reference

#### `markit convert`

Convert a single document to Markdown.

```bash
markit convert [OPTIONS] INPUT_FILE
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `INPUT_FILE` | Path to the input document |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output PATH` | Output directory | `./output` |
| `--llm` | Enable LLM Markdown enhancement | Disabled |
| `--analyze-image` | Enable LLM image analysis (alt text only) | Disabled |
| `--analyze-image-with-md` | Enable LLM image analysis with detailed `.md` description files | Disabled |
| `--no-compress` | Disable image compression | Enabled |
| `--pdf-engine ENGINE` | PDF engine (pymupdf4llm, pymupdf, pdfplumber, markitdown) | `pymupdf4llm` |
| `--llm-provider PROVIDER` | LLM provider to use | First available |
| `--llm-model MODEL` | LLM model name | Provider default |
| `-v, --verbose` | Enable verbose output | Disabled |
| `--dry-run` | Show plan without executing | Disabled |
| `-h, --help` | Show help message | - |

**Examples:**

```bash
# Basic conversion
markit convert report.pdf

# With LLM enhancement
markit convert report.pdf --llm --llm-provider openai

# With image analysis (generates alt text)
markit convert report.pdf --analyze-image

# With detailed image descriptions (generates .md files for each image)
markit convert report.pdf --analyze-image-with-md

# Custom output directory
markit convert presentation.pptx -o ./converted

# Verbose mode for debugging
markit convert document.docx -v
```

#### `markit batch`

Batch convert documents in a directory.

```bash
markit batch [OPTIONS] INPUT_DIR
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `INPUT_DIR` | Path to the input directory |

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output PATH` | Output directory | `INPUT_DIR/output` |
| `-r, --recursive` | Process subdirectories | Disabled |
| `--include PATTERN` | Include file pattern (glob) | All supported |
| `--exclude PATTERN` | Exclude file pattern (glob) | None |
| `--file-concurrency N` | Concurrent file processing | 4 |
| `--image-concurrency N` | Concurrent image processing | 8 |
| `--llm-concurrency N` | Concurrent LLM requests | 5 |
| `--on-conflict MODE` | Conflict handling (skip, overwrite, rename) | `rename` |
| `--resume` | Resume from last interrupted batch | Disabled |
| `--state-file PATH` | State file for resume | `.markit-state.json` |
| `--llm` | Enable LLM enhancement | Disabled |
| `--analyze-image` | Enable image analysis (alt text only) | Disabled |
| `--analyze-image-with-md` | Enable image analysis with detailed `.md` description files | Disabled |
| `-v, --verbose` | Enable verbose output | Disabled |
| `--dry-run` | Show plan without executing | Disabled |
| `-h, --help` | Show help message | - |

**Examples:**

```bash
# Basic batch conversion
markit batch ./documents

# Recursive with custom output
markit batch ./docs -o ./markdown -r

# Filter by file type
markit batch ./docs --include "*.docx" --exclude "*draft*"

# With LLM enhancement and detailed image descriptions
markit batch ./docs --llm --analyze-image-with-md

# Resume interrupted batch
markit batch ./docs --resume

# High concurrency
markit batch ./docs --file-concurrency 8 --image-concurrency 16
```

#### `markit config`

Configuration management commands.

```bash
markit config show    # Show current configuration
markit config init    # Initialize configuration file
markit config validate  # Validate configuration
```

#### `markit provider`

Provider management and testing commands.

```bash
markit provider test      # Test connectivity to all configured LLM providers
markit provider models    # List available models from all providers
```

**Options for `markit provider models`:**

| Option | Description | Default |
|--------|-------------|---------|
| `--no-cache` | Don't save models to local cache | Save to cache |
| `-o, --output PATH` | Custom output file path | `~/.cache/markit/models.json` |
| `-p, --provider NAME` | Filter by provider name | All providers |

**Examples:**

```bash
# Test all configured providers
markit provider test

# List models from all providers (auto-cached)
markit provider models

# List models from specific provider
markit provider models -p openrouter

# List models without caching
markit provider models --no-cache
```

---

### Supported Formats

| Format | Extension | Primary Engine | Notes |
|--------|-----------|----------------|-------|
| Word | .docx | MarkItDown | Full support |
| Word (Legacy) | .doc | LibreOffice + MarkItDown | Requires LibreOffice |
| PowerPoint | .pptx | MarkItDown | Full support |
| PowerPoint (Legacy) | .ppt | LibreOffice + MarkItDown | Requires LibreOffice |
| Excel | .xlsx | MarkItDown | Tables preserved |
| Excel (Legacy) | .xls | LibreOffice + MarkItDown | Requires LibreOffice |
| PDF | .pdf | PyMuPDF4LLM / PyMuPDF / pdfplumber | Configurable engine |
| CSV | .csv | MarkItDown | Converted to table |
| HTML | .html, .htm | MarkItDown | Full support |
| Text | .txt | Direct | Passthrough |
| Images | .png, .jpg, .gif, .webp, .bmp | LLM Analysis | With --analyze-image |

### Image Analysis Options

MarkIt provides two levels of LLM-powered image analysis:

#### `--analyze-image` (Alt Text Only)

Generates concise alt text for each extracted image and embeds it in the Markdown:

```markdown
![A flowchart showing the data processing pipeline](assets/diagram_001.png)
```

#### `--analyze-image-with-md` (Detailed Descriptions)

In addition to alt text, generates a detailed `.md` description file for each image in the `assets/` directory:

```
output/
  document.pdf.md
  assets/
    diagram_001.png
    diagram_001.png.md    # Detailed description file
```

The description file includes:

```markdown
---
source_image: diagram_001.png
image_type: diagram
generated_at: 2026-01-08T12:00:00Z
---

# Image Description

## Alt Text

A flowchart showing the data processing pipeline.

## Detailed Description

This diagram illustrates a three-stage data processing pipeline...

## Detected Text

"Input" -> "Process" -> "Output"
```

**Note:** Both options use a single LLM request per image. When `--analyze-image-with-md` is used, it automatically enables `--analyze-image` behavior.

---

### LLM Providers

MarkIt supports multiple LLM providers for Markdown enhancement and image analysis.

#### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
markit convert doc.pdf --llm --llm-provider openai --llm-model gpt-5.2
```

#### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="..."
markit convert doc.pdf --llm --llm-provider anthropic --llm-model claude-sonnet-4-5
```

#### Google Gemini

```bash
export GOOGLE_API_KEY="..."
markit convert doc.pdf --llm --llm-provider gemini --llm-model gemini-3-flash-preview
```

#### Ollama (Local)

```bash
# Start Ollama server
ollama serve

# Pull a model
ollama pull llama3.2-vision

# Use with markit
markit convert doc.pdf --llm --llm-provider ollama --llm-model llama3.2-vision
```

#### OpenRouter

```bash
export OPENROUTER_API_KEY="..."
markit convert doc.pdf --llm --llm-provider openrouter --llm-model google/gemini-3-flash-preview
```

#### Configuration File

Configure multiple providers in `markit.toml`:

```toml
[[llm.providers]]
provider = "openai"
model = "gpt-5.2"

[[llm.providers]]
provider = "anthropic"
model = "claude-sonnet-4-5"

[[llm.providers]]
provider = "ollama"
model = "llama3.2-vision"
base_url = "http://localhost:11434"
```

The first available provider will be used. If a provider fails, the next one is tried automatically.

---

### Troubleshooting

#### LibreOffice Conversion Fails

**Symptom:** Legacy formats (.doc, .ppt, .xls) fail to convert with "LibreOffice error" message.

**Solutions:**

1. Verify LibreOffice is installed:
   ```bash
   which soffice  # Linux/macOS
   where soffice  # Windows
   ```

2. Try running LibreOffice manually:
   ```bash
   soffice --headless --convert-to docx --outdir /tmp test.doc
   ```

3. Check LibreOffice version (4.0+ recommended):
   ```bash
   soffice --version
   ```

#### Concurrent Conversion Issues

**Symptom:** Batch conversion fails intermittently for .doc, .ppt, .xls files.

**Solution:** MarkIt uses isolated user profiles for LibreOffice to prevent lock conflicts. Additionally, v1.0.1+ fixes a race condition in LLM provider initialization that could cause redundant API calls during concurrent processing. Ensure you're using the latest version.

#### LLM Connection Errors

**Symptom:** "LLM provider not available" or timeout errors.

**Solutions:**

1. Verify API key is set:
   ```bash
   echo $OPENAI_API_KEY
   ```

2. Test connectivity:
   ```bash
   curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

3. Check firewall/proxy settings.

#### Out of Memory

**Symptom:** Process killed during large file conversion.

**Solutions:**

1. Reduce concurrency:
   ```bash
   markit batch ./docs --file-concurrency 2 --image-concurrency 4
   ```

2. Process files individually.

#### Image Extraction Issues

**Symptom:** Images missing from converted output.

**Solutions:**

1. Check image filter settings in `markit.toml`:
   ```toml
   [image]
   filter_small_images = false  # Disable to keep all images
   ```

2. Verify PDF engine supports image extraction:
   ```bash
   markit convert doc.pdf --pdf-engine pymupdf4llm
   ```

#### AutoShape/VML Graphics Not Converted

**Symptom:** Word documents (.docx) or PowerPoint files (.pptx) contain shapes or diagrams that don't appear in the Markdown output.

**Explanation:** MarkItDown (the underlying conversion library) does not support AutoShape/VML/DrawingML graphics. This is a known limitation of the library.

**What MarkIt Does:**
- Detects AutoShapes in the source document
- Adds a note at the end of the Markdown file indicating that AutoShapes were found
- The note reminds you to refer to the original source file

**Workaround:** For documents with important diagrams, export them manually as images and include them in your Markdown.

#### PowerPoint Headers/Footers in Output

**Symptom:** Repetitive footer text (company names, dates, slide numbers) appears on every slide in the converted Markdown.

**Solutions:**

1. **Automatic Filtering (Default):** MarkIt automatically detects and removes footer/header placeholders from PPTX files.

2. **LLM-Powered Cleanup:** Use the `--llm` flag for more intelligent filtering:
   ```bash
   markit convert presentation.pptx --llm
   ```
   The LLM will identify and remove repetitive footer content more accurately.

#### LLM Initialization Slow or Redundant API Calls

**Symptom:** Multiple "Provider initialized successfully" log messages, or many concurrent API calls to `/models` endpoint during batch processing.

**Explanation:** This was a race condition bug in versions prior to v1.0.1 where concurrent LLM tasks would each trigger provider initialization.

**Solution:** Upgrade to v1.0.1+ which implements proper async locking to ensure providers are initialized only once, even under high concurrency.

#### OpenAI API "Invalid max_tokens" Error

**Symptom:** Error message: `Invalid type for 'max_tokens': expected an unsupported value, but got null instead.`

**Explanation:** This was a bug in versions prior to v1.0.2 where `max_tokens=None` was passed to the OpenAI API, which rejects null values.

**Solution:** Upgrade to v1.0.2+ which properly filters out None values from API request parameters.

---

## Development

```bash
# Clone the repository
git clone https://github.com/Ynewtime/markit
cd markit

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=markit

# Run linter
ruff check .

# Run type checker
mypy markit

# Format code
ruff format .
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
