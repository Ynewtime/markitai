"""Config command for configuration management."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from markit.config import get_settings
from markit.config.constants import CONFIG_LOCATIONS, DEFAULT_CONFIG_FILE

# Create config sub-app
config_app = typer.Typer(help="Configuration management.")
console = Console()


@config_app.command("show")
def show() -> None:
    """Show current configuration."""
    settings = get_settings()

    console.print("\n[bold blue]Current Configuration[/bold blue]\n")

    # Create table for settings
    table = Table(show_header=True, header_style="bold")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # Global settings
    table.add_row("Log Level", settings.log_level)
    table.add_row("Log Directory", settings.log_dir)
    table.add_row("State File", settings.state_file)

    # Output settings
    table.add_row("Output Directory", settings.output.default_dir)
    table.add_row("On Conflict", settings.output.on_conflict)
    table.add_row("Create Assets Subdir", str(settings.output.create_assets_subdir))

    # Image settings
    table.add_row("Image Compression", str(settings.image.enable_compression))
    table.add_row("Image Analysis", str(settings.image.enable_analysis))
    table.add_row("Max Image Dimension", str(settings.image.max_dimension))

    # PDF settings
    table.add_row("PDF Engine", settings.pdf.engine)
    table.add_row("Extract Images", str(settings.pdf.extract_images))

    # Enhancement settings
    table.add_row("LLM Enhancement", str(settings.enhancement.enabled))
    table.add_row("Add Frontmatter", str(settings.enhancement.add_frontmatter))

    # Concurrency settings
    table.add_row("File Workers", str(settings.concurrency.file_workers))
    table.add_row("Image Workers", str(settings.concurrency.image_workers))
    table.add_row("LLM Workers", str(settings.concurrency.llm_workers))

    # LLM providers
    if settings.llm.providers:
        providers = ", ".join(p.provider for p in settings.llm.providers)
        table.add_row("LLM Providers", providers)
    else:
        table.add_row("LLM Providers", "None configured")

    console.print(table)
    console.print()


# Default configuration template - kept in sync with markit.example.toml
DEFAULT_CONFIG_TEMPLATE = '''# MarkIt Configuration Example
# ============================
# Copy this file to markit.toml and modify as needed.
# All settings shown here are optional - defaults will be used if not specified.
#
# Configuration priority (highest to lowest):
#   1. Command line arguments
#   2. Environment variables (MARKIT_*)
#   3. Configuration file (markit.toml)
#   4. Default values

# ============================================================================
# Global Settings
# ============================================================================

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_level = "INFO"

# Log directory for task-level log files
# Each task (convert/batch) creates a unique timestamped log file.
# Default: ".logs" (hidden folder to keep project directory clean)
# log_dir = ".logs"

# State file for batch resume functionality
# Used to track progress and enable --resume option
state_file = ".markit-state.json"

# ============================================================================
# LLM Configuration
# ============================================================================
# MarkIt supports two configuration schemas for LLM providers:
#
# 1. Legacy Schema (simpler): [[llm.providers]]
#    - Each provider entry contains both credentials and model config
#    - Good for simple setups with few models
#
# 2. New Schema (recommended for multiple models): [[llm.credentials]] + [[llm.models]]
#    - Separate credential storage from model configuration
#    - Allows multiple models to share the same credentials
#    - Better for managing many models across providers
#
# Both schemas can be mixed in the same config file.
#
# API keys can be set via environment variables:
#   - OPENAI_API_KEY
#   - ANTHROPIC_API_KEY
#   - GOOGLE_API_KEY
#   - OPENROUTER_API_KEY
#   - Or custom env var via api_key_env field
#
# To use LLM features, run commands with --llm or --analyze-image flags.
# Use `markit provider select` to interactively add models to your config.

# ============================================================================
# New Schema: Credentials + Models (Recommended)
# ============================================================================
# Step 1: Define credentials (API keys and endpoints)
# Step 2: Define models that reference those credentials
#
# Benefits:
#   - Share one credential across multiple models
#   - Easier to manage API keys in one place
#   - Cleaner config when using many models

# Supported provider types: openai, anthropic, gemini, ollama, openrouter
#
# Note: "openai" provider type is compatible with any OpenAI-compatible API
# (e.g., DeepSeek, Azure OpenAI, local LLM servers with OpenAI API)

# Credential: OpenAI
# [[llm.credentials]]
# id = "openai-main"
# provider = "openai"
# # api_key = "sk-..."  # Or use OPENAI_API_KEY env var (default)

# Credential: DeepSeek (OpenAI-compatible)
# [[llm.credentials]]
# id = "deepseek"
# provider = "openai"
# base_url = "https://api.deepseek.com"
# api_key_env = "DEEPSEEK_API_KEY"  # Custom env var

# Credential: Anthropic
# [[llm.credentials]]
# id = "anthropic-main"
# provider = "anthropic"
# # api_key = "..."  # Or use ANTHROPIC_API_KEY env var (default)

# Credential: Google Gemini
# [[llm.credentials]]
# id = "gemini-main"
# provider = "gemini"
# # api_key = "..."  # Or use GOOGLE_API_KEY env var (default)

# Credential: Ollama (Local)
# [[llm.credentials]]
# id = "ollama-local"
# provider = "ollama"
# base_url = "http://localhost:11434"
# # No API key needed for local Ollama

# Credential: OpenRouter (API aggregator)
# [[llm.credentials]]
# id = "openrouter-main"
# provider = "openrouter"
# base_url = "https://openrouter.ai/api/v1"
# # api_key = "..."  # Or use OPENROUTER_API_KEY env var (default)

# Model: GPT-4o (Vision capable)
# [[llm.models]]
# name = "GPT-4o"
# model = "gpt-4o"
# credential_id = "openai-main"
# capabilities = ["text", "vision"]
# timeout = 60
# max_retries = 3

# Model: GPT-4o-mini (Fast, cheaper)
# [[llm.models]]
# name = "GPT-4o Mini"
# model = "gpt-4o-mini"
# credential_id = "openai-main"
# capabilities = ["text", "vision"]
# timeout = 30

# Model: DeepSeek Reasoner (Text only)
# [[llm.models]]
# name = "DeepSeek Reasoner"
# model = "deepseek-reasoner"
# credential_id = "deepseek"
# capabilities = ["text"]
# timeout = 120

# Model: Claude Sonnet (Vision capable)
# [[llm.models]]
# name = "Claude Sonnet 4.5"
# model = "claude-sonnet-4-5"
# credential_id = "anthropic-main"
# capabilities = ["text", "vision"]
# timeout = 60
# max_retries = 3

# ============================================================================
# Legacy Schema: Combined Provider Config
# ============================================================================
# Simpler but less flexible. Each entry contains both credentials and model.
# Still fully supported for backward compatibility.

# Provider: OpenAI
# [[llm.providers]]
# provider = "openai"
# name = "OpenAI GPT-5.2"  # Custom display name
# model = "gpt-5.2"
# timeout = 60
# max_retries = 3
# capabilities = ["text", "vision"]  # Optional: Explicitly declare capabilities
# api_key = "sk-..."  # Or use OPENAI_API_KEY env var

# Provider: DeepSeek (Text Only, Custom API Key Env)
# [[llm.providers]]
# provider = "openai"
# name = "DeepSeek Reasoner"
# model = "deepseek-reasoner"
# base_url = "https://api.deepseek.com"
# api_key_env = "DEEPSEEK_API_KEY"  # Read from DEEPSEEK_API_KEY env var
# capabilities = ["text"]           # Skip image analysis tasks
# timeout = 120

# Provider: Anthropic Claude
# [[llm.providers]]
# provider = "anthropic"
# model = "claude-sonnet-4-5"
# timeout = 60
# max_retries = 3
# api_key = "..."  # Or use ANTHROPIC_API_KEY env var

# Provider: Google Gemini
# [[llm.providers]]
# provider = "gemini"
# model = "gemini-3-flash-preview"
# timeout = 60
# max_retries = 3
# # api_key = "..."  # Or use GOOGLE_API_KEY env var

# Provider: Ollama (Local)
# [[llm.providers]]
# provider = "ollama"
# model = "llama3.2-vision"
# base_url = "http://localhost:11434"
# timeout = 120
# # No API key needed for local Ollama

# Provider: OpenRouter (API aggregator)
# [[llm.providers]]
# provider = "openrouter"
# model = "google/gemini-3-flash-preview"
# base_url = "https://openrouter.ai/api/v1"
# timeout = 60
# max_retries = 3
# # api_key = "..."  # Or use OPENROUTER_API_KEY env var

# ============================================================================
# Image Processing
# ============================================================================

[image]
# Enable image compression (PNG optimization, JPEG quality reduction)
enable_compression = true

# PNG optimization level (0-6, higher = more compression but slower)
png_optimization_level = 2

# JPEG quality (0-100, higher = better quality but larger file)
jpeg_quality = 85

# Maximum image dimension in pixels (larger images will be resized)
max_dimension = 2048

# Enable LLM-based image analysis (generates alt text and descriptions)
# Requires --analyze-image flag or this setting to be true
enable_analysis = false

# Filter out small decorative images (icons, bullets, etc.)
filter_small_images = true

# Minimum image width or height in pixels to keep
min_dimension = 100

# Minimum image area in pixels squared (e.g., 200x200 = 40000)
min_area = 40000

# Minimum image file size in bytes to keep (e.g., 10KB = 10240)
min_file_size = 10240

# ============================================================================
# Concurrency Settings
# ============================================================================

[concurrency]
# Number of files to process concurrently in batch mode
file_workers = 4

# Number of images to process concurrently
image_workers = 8

# Number of concurrent LLM API requests
llm_workers = 5

# ============================================================================
# PDF Processing
# ============================================================================

[pdf]
# PDF processing engine:
#   - pymupdf4llm: Best for LLM/RAG applications, good table/heading detection (default)
#   - pymupdf: Fast, good for most PDFs
#   - pdfplumber: Better table extraction
#   - markitdown: Uses Microsoft's MarkItDown library
engine = "pymupdf4llm"

# Extract images from PDFs
extract_images = true

# Enable OCR for scanned PDFs (requires additional dependencies)
ocr_enabled = false

# ============================================================================
# Markdown Enhancement (LLM-powered)
# ============================================================================
# These features are only active when --llm flag is used.

[enhancement]
# Enable LLM-based markdown enhancement by default
# Can be overridden with --llm flag
enabled = false

# Remove headers and footers from converted content
remove_headers_footers = true

# Fix heading levels (ensure proper hierarchy starting from h2)
fix_heading_levels = true

# Add YAML frontmatter to output files
add_frontmatter = true

# Generate document summary in frontmatter
generate_summary = true

# Chunk size for processing large documents (in tokens)
# Optimized for large context models like Gemini 3 Flash (1.05M context)
# Larger chunks = fewer LLM calls = faster processing
# Example: 75K token doc with chunk_size=32000 → ~3 chunks instead of 19+
chunk_size = 32000

# Overlap between chunks for context continuity (in tokens)
chunk_overlap = 500

# ============================================================================
# Output Settings
# ============================================================================

[output]
# Default output directory (relative to input directory in batch mode)
default_dir = "output"

# Conflict resolution strategy:
#   - skip: Skip if output file exists
#   - overwrite: Overwrite existing files
#   - rename: Add numeric suffix (e.g., file_1.md, file_2.md)
on_conflict = "rename"

# Create assets subdirectory for extracted images
create_assets_subdir = true

# Generate markdown description files for images (image.png.md)
generate_image_descriptions = true
'''


@config_app.command("init")
def init(
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            "-p",
            help="Path to create config file.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing config file.",
        ),
    ] = False,
) -> None:
    """Initialize a configuration file."""
    config_path = path or Path.cwd() / DEFAULT_CONFIG_FILE

    if config_path.exists() and not force:
        console.print(f"[red]Error:[/red] Config file already exists at {config_path}")
        console.print("Use --force to overwrite.")
        raise typer.Exit(1)

    config_path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    console.print(f"[green]Created config file at:[/green] {config_path}")


@config_app.command("validate")
def validate() -> None:
    """Validate current configuration."""
    try:
        settings = get_settings()

        console.print("\n[bold blue]Configuration Validation[/bold blue]\n")

        # Check LLM providers
        if settings.llm.providers:
            console.print("[green]LLM Providers:[/green]")
            for provider in settings.llm.providers:
                has_key = provider.api_key or provider.provider == "ollama"
                status = "[green]OK[/green]" if has_key else "[yellow]No API key[/yellow]"
                console.print(f"  - {provider.provider}: {status}")
        else:
            console.print("[yellow]No LLM providers configured.[/yellow]")
            console.print("  LLM features will be disabled.")

        console.print()
        console.print("[green]Configuration is valid![/green]")

    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1) from e


@config_app.command("locations")
def locations() -> None:
    """Show configuration file search locations."""
    console.print("\n[bold blue]Configuration File Locations[/bold blue]\n")
    console.print("MarkIt searches for configuration files in the following order:\n")

    for i, loc in enumerate(CONFIG_LOCATIONS, 1):
        exists = "[green]exists[/green]" if loc.exists() else "[dim]not found[/dim]"
        console.print(f"  {i}. {loc} ({exists})")

    console.print()
    console.print("[dim]Environment variables with MARKIT_ prefix are also supported.[/dim]")
    console.print()
