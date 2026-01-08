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
    table.add_row("Log File", str(settings.log_file) if settings.log_file else "None")
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

    # Default configuration template
    config_content = """# MarkIt Configuration File
# https://github.com/Ynewtime/markit

log_level = "INFO"
# log_file = "markit.log"
state_file = ".markit-state.json"

# LLM Providers (first valid one is used as default)
# Uncomment and configure the providers you want to use

# [[llm.providers]]
# provider = "openai"
# model = "gpt-5.2"
# timeout = 60
# max_retries = 3
# # api_key is read from OPENAI_API_KEY environment variable

# [[llm.providers]]
# provider = "anthropic"
# model = "claude-sonnet-4-5"
# # api_key is read from ANTHROPIC_API_KEY environment variable

# [[llm.providers]]
# provider = "gemini"
# model = "gemini-3-flash-preview"
# # api_key is read from GOOGLE_API_KEY environment variable

# [[llm.providers]]
# provider = "ollama"
# model = "llama3.2-vision"
# base_url = "http://localhost:11434"

[image]
enable_compression = true
png_optimization_level = 2
jpeg_quality = 85
max_dimension = 2048
enable_analysis = false  # Enable to use LLM for image descriptions

[concurrency]
file_workers = 4
image_workers = 8
llm_workers = 5

[pdf]
engine = "pymupdf"  # Options: pymupdf, pdfplumber, markitdown
extract_images = true
ocr_enabled = false

[enhancement]
enabled = false  # Enable to use LLM for Markdown optimization
remove_headers_footers = true
fix_heading_levels = true
add_frontmatter = true
generate_summary = true
chunk_size = 4000

[output]
default_dir = "output"
on_conflict = "rename"  # Options: skip, overwrite, rename
create_assets_subdir = true
generate_image_descriptions = true
"""

    config_path.write_text(config_content, encoding="utf-8")
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
