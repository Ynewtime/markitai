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


# Default configuration template - kept in sync with markit.example.yaml
DEFAULT_CONFIG_TEMPLATE = """# MarkIt Configuration
# Run `markit provider add` to configure LLM providers interactively.

log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
state_file: ".markit-state.json"  # State file for batch resume

image:
  enable_compression: true  # PNG/JPEG compression
  png_optimization_level: 2  # 0-6, higher = slower
  jpeg_quality: 85  # 0-100
  max_dimension: 2048  # Max image size in pixels
  enable_analysis: false  # LLM-based image analysis
  filter_small_images: true  # Filter decorative images
  min_dimension: 100  # Min width/height to keep
  min_area: 40000  # Min area (e.g., 200x200)
  min_file_size: 10240  # Min file size in bytes

concurrency:
  file_workers: 4  # Concurrent files in batch mode
  image_workers: 8  # Concurrent image processing
  llm_workers: 5  # Concurrent LLM requests

pdf:
  engine: "pymupdf4llm"  # pymupdf4llm, pymupdf, pdfplumber, markitdown
  extract_images: true
  ocr_enabled: false

enhancement:  # LLM-powered features (requires --llm flag)
  enabled: false
  remove_headers_footers: true
  fix_heading_levels: true
  add_frontmatter: true
  generate_summary: true
  chunk_size: 32000  # Tokens per chunk
  chunk_overlap: 500

output:
  default_dir: "output"
  on_conflict: "rename"  # skip, overwrite, rename
  create_assets_subdir: true
  generate_image_descriptions: true

prompt:
  output_language: "zh"  # zh, en, auto
  # prompts_dir: "prompts"  # Custom prompts directory (relative to cwd, optional)
  # Files should be named: {type}_{lang}.md
  #   type: enhancement, image_analysis, summary
  #   lang: zh, en (based on output_language)
  #   e.g., enhancement_zh.md, image_analysis_en.md
  # If not found, falls back to built-in prompts
  # Custom prompt file paths (highest priority, override prompts_dir):
  # image_analysis_prompt_file: "my_prompts/image_analysis.md"
  # enhancement_prompt_file: "my_prompts/enhancement.md"
  # summary_prompt_file: "my_prompts/summary.md"
  # Description generation strategy for chunked documents:
  # description_strategy: "first_chunk"  # first_chunk (default), separate_call, none

# Execution mode
# execution:
#   mode: "default"  # default, fast (skip validation, limit fallbacks)
#   fast_max_fallback: 1  # Max fallback attempts in fast mode
#   fast_skip_validation: true  # Skip provider validation in fast mode

# LibreOffice profile pool (for .doc/.ppt/.xls conversion)
# libreoffice:
#   pool_size: 4  # Number of concurrent LibreOffice profiles
#   reset_after_failures: 3  # Reset profile after N consecutive failures
#   reset_after_uses: 100  # Reset profile after N conversions

# LLM Configuration - Use `markit provider add` to add credentials
# llm:
#   # Concurrent fallback settings
#   concurrent_fallback_enabled: true  # Enable smart concurrent fallback
#   concurrent_fallback_timeout: 60  # Seconds before starting backup model
#   max_request_timeout: 300  # Absolute timeout (5 min), force interrupt
#
#   # Routing strategy (how to select models for requests)
#   routing:
#     strategy: "least_pending"  # cost_first, least_pending, round_robin
#     cost_weight: 0.3  # Weight for cost in least_pending (0-1)
#     load_weight: 0.7  # Weight for load balancing in least_pending (0-1)
#
#   # AIMD rate limiting (per credential, auto-adjusts concurrency)
#   adaptive:
#     enabled: true
#     initial_concurrency: 3
#     max_concurrency: 10
#     min_concurrency: 1
#     success_threshold: 5  # Increase concurrency after N successes
#     multiplicative_decrease: 0.5  # Halve on failure
#     cooldown_seconds: 30.0  # Cooldown after decrease
#
#   # Provider validation
#   validation:
#     enabled: true
#     retry_count: 2
#     on_failure: "warn"  # warn, skip, fail
#
#   credentials:
#     - id: "deepseek"
#       provider: "openai"  # openai, anthropic, gemini, ollama, openrouter
#       base_url: "https://api.deepseek.com"
#       api_key_env: "DEEPSEEK_API_KEY"
#     - id: "openai-main"
#       provider: "openai"
#     - id: "anthropic-main"
#       provider: "anthropic"
#   models:
#     - name: "deepseek-chat"
#       model: "deepseek-chat"
#       credential_id: "deepseek"
#       capabilities: ["text"]  # text-only model
#       # Optional cost config for statistics (USD per 1M tokens)
#       cost:
#         input_per_1m: 0.28
#         output_per_1m: 0.42
#         cached_input_per_1m: 0.028  # 90% cache discount
#     - name: "GPT-4o"
#       model: "gpt-4o"
#       credential_id: "openai-main"
#       capabilities: ["text", "vision"]
#       cost:
#         input_per_1m: 2.50
#         output_per_1m: 10.00
#         cached_input_per_1m: 1.25  # 50% cache discount
"""


# Command order: init -> test -> list -> locations


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


@config_app.command("test")
def test_config() -> None:
    """Test current configuration."""
    try:
        settings = get_settings()

        console.print("\n[bold blue]Configuration Validation[/bold blue]\n")

        # Check LLM configuration (new format: credentials + models)
        if settings.llm.credentials:
            console.print("[green]LLM Credentials:[/green]")
            for cred in settings.llm.credentials:
                console.print(f"  - {cred.id} ({cred.provider})")

        if settings.llm.models:
            console.print("[green]LLM Models:[/green]")
            for model in settings.llm.models:
                caps = ", ".join(model.capabilities) if model.capabilities else "text"
                console.print(f"  - {model.name}: {model.model} [{caps}]")

        # Check legacy LLM providers
        if settings.llm.providers:
            console.print("[green]LLM Providers (Legacy):[/green]")
            for provider in settings.llm.providers:
                has_key = provider.api_key or provider.provider == "ollama"
                status = "[green]OK[/green]" if has_key else "[yellow]No API key[/yellow]"
                console.print(f"  - {provider.provider}: {status}")

        if not (settings.llm.credentials or settings.llm.providers):
            console.print("[yellow]No LLM providers configured.[/yellow]")
            console.print("  LLM features will be disabled.")

        console.print()
        console.print("[green]Configuration is valid![/green]")

    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1) from e


@config_app.command("list")
def list_config() -> None:
    """List current configuration."""
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

    # Prompt settings
    table.add_row("Output Language", settings.prompt.output_language)
    table.add_row("Prompts Directory", settings.prompt.prompts_dir)

    # Concurrency settings
    table.add_row("File Workers", str(settings.concurrency.file_workers))
    table.add_row("Image Workers", str(settings.concurrency.image_workers))
    table.add_row("LLM Workers", str(settings.concurrency.llm_workers))

    # LLM providers
    if settings.llm.credentials:
        creds = ", ".join(f"{c.id} ({c.provider})" for c in settings.llm.credentials)
        table.add_row("LLM Credentials", creds)

    if settings.llm.models:
        models = ", ".join(f"{m.name}" for m in settings.llm.models)
        table.add_row("LLM Models", models)

    if settings.llm.providers:
        providers = ", ".join(p.provider for p in settings.llm.providers)
        table.add_row("LLM Providers (Legacy)", providers)

    if not (settings.llm.credentials or settings.llm.providers):
        table.add_row("LLM Config", "None configured")

    console.print(table)
    console.print()


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


# Backward compatibility aliases (hidden from help)


@config_app.command("show", hidden=True)
def show() -> None:
    """Alias for 'list' (deprecated)."""
    console.print("[yellow]Note: 'show' is deprecated, use 'list' instead[/yellow]\n")
    list_config()


@config_app.command("validate", hidden=True)
def validate() -> None:
    """Alias for 'test' (deprecated)."""
    console.print("[yellow]Note: 'validate' is deprecated, use 'test' instead[/yellow]\n")
    test_config()
