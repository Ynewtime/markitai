"""Legacy Office format converters (DOC, PPT - Office 97-2003) via anydoc.

Since 1.0.0 these formats are handled by the bundled-Rust anydoc backend
(``firecrawl-anydoc``, opt in via the ``markitai[legacy]`` extra) instead
of driving Microsoft Office / LibreOffice: no Office installation is
required on any platform, and conversion is millisecond-scale.

Two behavior notes versus the retired Office-automation path:

- Embedded images are not extracted (anydoc's markdown carries no image
  anchors), and PPT tables arrive as plain text lines.
- Encrypted files raise a clear error; the old path could sometimes open
  them through a locally installed Office suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)
from markitai.utils.errors import (
    ConversionError,
    MissingDependencyError,
    extra_install_command,
)


def _load_anydoc() -> Any:
    """Import the anydoc backend or raise an actionable error."""
    try:
        import anydoc
    except ImportError:
        raise MissingDependencyError(
            "Legacy Office formats (.doc, .ppt) need the anydoc backend. "
            f"Install it with: {extra_install_command('legacy')}"
        ) from None
    return anydoc


class _AnyDocLegacyConverter(BaseConverter):
    """Convert a legacy Office file directly with anydoc (sync, local)."""

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert a .doc/.ppt file to Markdown.

        Args:
            input_path: Path to the input file.
            output_dir: Unused (anydoc's markdown carries no extractable
                image assets); accepted for interface compatibility.

        Returns:
            ConvertResult with the converted markdown.

        Raises:
            MissingDependencyError: ``markitai[legacy]`` extra not installed.
            ConversionError: The file is encrypted, malformed, or otherwise
                rejected by anydoc.
        """
        anydoc = _load_anydoc()
        input_path = Path(input_path)
        suffix = input_path.suffix.lower()

        try:
            markdown = anydoc.to_markdown(str(input_path))
        except anydoc.EncryptedError:
            raise ConversionError(
                f"{input_path.name} is password-protected; decrypt it before converting"
            ) from None
        except anydoc.ConvertError as e:
            raise ConversionError(f"Failed to convert {input_path.name}: {e}") from None

        logger.debug(f"[Legacy] Converted {input_path.name} via anydoc")
        return ConvertResult(
            markdown=markdown,
            metadata={
                "original_format": suffix.lstrip(".").upper(),
                "source": str(input_path),
                "backend": "anydoc",
            },
        )


@register_converter(FileFormat.DOC)
class DocConverter(_AnyDocLegacyConverter):
    """Converter for legacy DOC (Word 97-2003) documents."""

    supported_formats = [FileFormat.DOC]


@register_converter(FileFormat.PPT)
class PptConverter(_AnyDocLegacyConverter):
    """Converter for legacy PPT (PowerPoint 97-2003) documents."""

    supported_formats = [FileFormat.PPT]
