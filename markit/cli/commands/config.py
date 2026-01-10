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

# LLM Configuration - Use `markit provider add` to add credentials
# llm:
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
#     - name: "GPT-4o"
#       model: "gpt-4o"
#       credential_id: "openai-main"
#       capabilities: ["text", "vision"]
"""


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
