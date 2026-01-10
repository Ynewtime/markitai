"""Conversion execution context - holds initialized state for commands."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from markit.cli.shared.options import ConversionOptions
from markit.config import get_settings
from markit.config.constants import LLM_PROVIDERS, PDF_ENGINES
from markit.utils.logging import get_logger, setup_task_logging

if TYPE_CHECKING:
    from markit.config.settings import MarkitSettings
    from markit.core.pipeline import ConversionPipeline

log = get_logger(__name__)


@dataclass
class ConversionContext:
    """Execution context for conversion commands.

    This class encapsulates all the initialization logic that is shared
    between convert and batch commands, including:
    - Settings loading
    - Logging setup
    - Validation
    - Pipeline creation
    """

    settings: "MarkitSettings"
    options: ConversionOptions
    task_id: str
    log_path: Path
    output_dir: Path
    console: Console = field(default_factory=Console)

    @classmethod
    def create(
        cls,
        options: ConversionOptions,
        command_prefix: str = "task",
        console: Console | None = None,
        base_path: Path | None = None,
    ) -> "ConversionContext":
        """Create and initialize a conversion context.

        Args:
            options: Conversion options from CLI
            command_prefix: Prefix for log files (e.g., "convert", "batch")
            console: Optional Rich console instance
            base_path: Optional base path for resolving relative output dir

        Returns:
            Initialized ConversionContext

        Raises:
            typer.Exit: If validation fails
        """
        settings = get_settings()
        console = console or Console()

        # Setup task-level logging
        task_id, log_path = setup_task_logging(
            log_dir=settings.log_dir,
            prefix=command_prefix,
            verbose=options.verbose,
        )

        # Log verbose hint
        if options.verbose:
            log.info("Logs will be saved to", log_file=str(log_path))

        # Log masked configuration
        config_dump = _mask_config(settings.model_dump())
        log.info("Task Configuration", task_id=task_id, config=config_dump)

        # Validate options
        _validate_options(options, console)

        # Resolve output directory
        output_dir = options.resolve_output_dir(settings, base_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            settings=settings,
            options=options,
            task_id=task_id,
            log_path=log_path,
            output_dir=output_dir,
            console=console,
        )

    def create_pipeline(self) -> "ConversionPipeline":
        """Create a ConversionPipeline with the current context settings."""
        from markit.core.pipeline import ConversionPipeline

        return ConversionPipeline(
            settings=self.settings,
            llm_enabled=self.options.llm,
            analyze_image=self.options.effective_analyze_image,
            analyze_image_with_md=self.options.analyze_image_with_md,
            compress_images=self.options.compress_images,
            pdf_engine=self.options.pdf_engine,
            llm_provider=self.options.llm_provider,
            llm_model=self.options.llm_model,
        )

    def log_start(self, **kwargs) -> None:
        """Log conversion start with additional context."""
        log.info(
            "Starting conversion",
            output_dir=str(self.output_dir),
            llm_enabled=self.options.llm,
            analyze_image=self.options.effective_analyze_image,
            analyze_image_with_md=self.options.analyze_image_with_md,
            **kwargs,
        )


def _mask_config(config_dump: dict) -> dict:
    """Mask sensitive information in config dump."""
    if "llm" in config_dump and "providers" in config_dump["llm"]:
        for provider in config_dump["llm"]["providers"]:
            if "api_key" in provider and provider["api_key"]:
                provider["api_key"] = "***"
    return config_dump


def _validate_options(options: ConversionOptions, console: Console) -> None:
    """Validate conversion options.

    Raises:
        typer.Exit: If validation fails
    """
    # Validate PDF engine
    if options.pdf_engine and options.pdf_engine not in PDF_ENGINES:
        console.print(
            f"[red]Error:[/red] Invalid PDF engine '{options.pdf_engine}'. "
            f"Options: {', '.join(PDF_ENGINES)}"
        )
        raise typer.Exit(1)

    # Validate LLM provider
    if options.llm_provider and options.llm_provider not in LLM_PROVIDERS:
        console.print(
            f"[red]Error:[/red] Invalid LLM provider '{options.llm_provider}'. "
            f"Options: {', '.join(LLM_PROVIDERS)}"
        )
        raise typer.Exit(1)
