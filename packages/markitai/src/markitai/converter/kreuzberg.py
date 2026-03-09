"""Kreuzberg-based converter for formats without native converters.

Kreuzberg is an optional dependency (pure Rust wheel) that can extract
text/markdown from 75+ file formats. This module registers it only for
formats that markitai doesn't already handle natively.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    _converter_registry,
)

# Formats that kreuzberg should handle — only those without native converters.
# NUMBERS: handled by markitdown_ext.
KREUZBERG_FORMATS: list[FileFormat] = [
    # Structured data (CSV handled by markitdown_ext)
    FileFormat.TSV,
    FileFormat.XML,
    # OpenDocument
    FileFormat.ODS,
    FileFormat.ODT,
    # Rich text / markup
    FileFormat.RTF,
    FileFormat.RST,
    FileFormat.ORG,
    FileFormat.TEX,
    # Email (MSG handled by markitdown_ext)
    FileFormat.EML,
]


class KreuzbergConverter(BaseConverter):
    """Converter using kreuzberg for formats without native support.

    Kreuzberg is a pure Rust wheel that can extract text/markdown from
    75+ file formats. This converter wraps its synchronous API and maps
    results to the standard ConvertResult.
    """

    supported_formats = KREUZBERG_FORMATS

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert a document to Markdown using kreuzberg.

        Args:
            input_path: Path to the input file.
            output_dir: Unused (present for BaseConverter interface
                compatibility).

        Returns:
            ConvertResult with markdown content and metadata.

        Raises:
            ImportError: If kreuzberg is not installed.
            RuntimeError: If kreuzberg fails to extract content.
        """
        try:
            from kreuzberg import ExtractionConfig, extract_file_sync
        except ImportError:
            raise ImportError(
                "kreuzberg is required for this file format but is not "
                "installed. Install it with: uv pip install kreuzberg"
            )

        input_path = Path(input_path)
        logger.debug("[Kreuzberg] Converting: {}", input_path.name)

        try:
            result = extract_file_sync(
                input_path,
                config=ExtractionConfig(output_format="markdown"),
            )
        except Exception as e:
            raise RuntimeError(
                f"kreuzberg failed to extract '{input_path}': {e}"
            ) from e

        metadata: dict = {
            "source": str(input_path),
            "format": input_path.suffix.lstrip(".").upper(),
            "converter": "kreuzberg",
        }
        if result.metadata:
            metadata["kreuzberg_metadata"] = result.metadata

        return ConvertResult(
            markdown=result.content,
            images=[],
            metadata=metadata,
        )


def register_kreuzberg_converters() -> None:
    """Register KreuzbergConverter for all supported formats.

    Only registers for formats that don't already have a converter in the
    registry, and only if kreuzberg is installed.
    """
    if importlib.util.find_spec("kreuzberg") is None:
        return

    registered = []
    for fmt in KREUZBERG_FORMATS:
        if _converter_registry.get(fmt) is None:
            _converter_registry[fmt] = KreuzbergConverter
            registered.append(fmt.value)

    if registered:
        logger.trace("Kreuzberg registered for: {}", ", ".join(registered))
