"""Main CLI application using Typer."""

from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console

from markit import __version__
from markit.cli.commands.batch import batch
from markit.cli.commands.config import config_app
from markit.cli.commands.convert import convert
from markit.cli.commands.provider import provider_app

# Load environment variables from .env file
load_dotenv()

# Create main Typer app
app = typer.Typer(
    name="markit",
    help="Intelligent document to Markdown conversion tool with LLM enhancement.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Console for output
console = Console()

# Register commands
app.command(name="convert", help="Convert a single document to Markdown.")(convert)
app.command(name="batch", help="Batch convert documents in a directory.")(batch)
app.add_typer(config_app, name="config")
app.add_typer(provider_app, name="provider")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]MarkIt[/bold blue] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """MarkIt - Intelligent document to Markdown conversion tool.

    Convert various document formats to high-quality Markdown with optional
    LLM-powered enhancement for format optimization and image analysis.
    """
    pass


if __name__ == "__main__":
    app()
