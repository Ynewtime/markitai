"""Shared conversion options for convert and batch commands."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markit.config.settings import MarkitSettings


@dataclass
class ConversionOptions:
    """Shared conversion options between convert and batch commands.

    This dataclass captures the common CLI parameters that are used
    by both single-file and batch conversion operations.
    """

    # Output settings
    output_dir: Path | None = None

    # LLM settings
    llm: bool = False
    analyze_image: bool = False
    analyze_image_with_md: bool = False
    llm_provider: str | None = None
    llm_model: str | None = None

    # Processing settings
    no_compress: bool = False
    pdf_engine: str | None = None

    # Runtime settings
    verbose: bool = False
    dry_run: bool = False

    @property
    def effective_analyze_image(self) -> bool:
        """Get effective analyze_image setting (with_md implies analyze_image)."""
        return self.analyze_image or self.analyze_image_with_md

    @property
    def compress_images(self) -> bool:
        """Get compress_images setting (inverse of no_compress)."""
        return not self.no_compress

    @property
    def use_phased_pipeline(self) -> bool:
        """Check if phased pipeline should be used (LLM features enabled)."""
        return self.llm or self.effective_analyze_image

    def resolve_output_dir(self, settings: "MarkitSettings", base_path: Path | None = None) -> Path:
        """Resolve output directory with fallback to settings default.

        Args:
            settings: Application settings
            base_path: Optional base path for relative output dir

        Returns:
            Resolved output directory path
        """
        if self.output_dir:
            return self.output_dir
        if base_path:
            return base_path / settings.output.default_dir
        return Path(settings.output.default_dir)
