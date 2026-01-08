"""CLI callback functions."""

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def validate_output_dir(value: Path | None) -> Path | None:
    """Validate and create output directory if needed."""
    if value is None:
        return None

    if value.exists() and not value.is_dir():
        raise typer.BadParameter(f"Output path exists but is not a directory: {value}")

    return value


def validate_input_file(value: Path) -> Path:
    """Validate input file exists and is readable."""
    if not value.exists():
        raise typer.BadParameter(f"File not found: {value}")

    if not value.is_file():
        raise typer.BadParameter(f"Path is not a file: {value}")

    return value


def validate_pdf_engine(value: str | None) -> str | None:
    """Validate PDF engine option."""
    from markit.config.constants import PDF_ENGINES

    if value is not None and value not in PDF_ENGINES:
        raise typer.BadParameter(f"Invalid PDF engine '{value}'. Options: {', '.join(PDF_ENGINES)}")

    return value


def validate_llm_provider(value: str | None) -> str | None:
    """Validate LLM provider option."""
    from markit.config.constants import LLM_PROVIDERS

    if value is not None and value not in LLM_PROVIDERS:
        raise typer.BadParameter(
            f"Invalid LLM provider '{value}'. Options: {', '.join(LLM_PROVIDERS)}"
        )

    return value
