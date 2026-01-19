"""Legacy Office document converters (DOC, PPT, XLS)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from markit.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)
from markit.converter.office import OfficeConverter, PptxConverter
from markit.utils.office import find_libreoffice

if TYPE_CHECKING:
    from markit.config import MarkitConfig


class LegacyOfficeConverter(BaseConverter):
    """Base converter for legacy Office documents (DOC, PPT, XLS).

    Converts legacy formats to modern formats using LibreOffice CLI,
    then processes with MarkItDown.
    """

    # Mapping of legacy format to target format
    TARGET_FORMAT: dict[str, str] = {
        ".doc": "docx",
        ".ppt": "pptx",
        ".xls": "xlsx",
    }

    def __init__(self, config: MarkitConfig | None = None) -> None:
        super().__init__(config)
        self._office_converter = OfficeConverter(config)
        self._pptx_converter = PptxConverter(config)
        self._soffice_path = find_libreoffice()

    def _convert_with_libreoffice(
        self,
        input_path: Path,
        target_format: str,
        output_dir: Path,
    ) -> Path:
        """Convert legacy format using LibreOffice CLI.

        Uses isolated user profile to support concurrent LibreOffice processes.
        """
        if not self._soffice_path:
            raise RuntimeError(
                "LibreOffice not found. Install LibreOffice to convert "
                f"{input_path.suffix} files."
            )

        # Create isolated user profile for concurrent execution
        # LibreOffice uses a shared user config directory by default,
        # which causes conflicts when multiple processes run simultaneously
        with tempfile.TemporaryDirectory(prefix="lo_profile_") as profile_dir:
            profile_url = Path(profile_dir).as_uri()

            # Run LibreOffice conversion with isolated profile
            cmd = [
                self._soffice_path,
                "--headless",
                f"-env:UserInstallation={profile_url}",
                "--convert-to",
                target_format,
                "--outdir",
                str(output_dir),
                str(input_path),
            ]

            logger.debug(f"Running LibreOffice: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"LibreOffice conversion failed: {result.stderr}"
                    )

            except subprocess.TimeoutExpired:
                raise RuntimeError("LibreOffice conversion timed out")

        # Find converted file
        converted_name = input_path.stem + "." + target_format
        converted_path = output_dir / converted_name

        if not converted_path.exists():
            raise RuntimeError(f"Converted file not found: {converted_path}")

        return converted_path

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert legacy Office document to Markdown.

        Args:
            input_path: Path to the input file
            output_dir: Optional output directory for extracted images

        Returns:
            ConvertResult containing markdown and extracted images
        """
        input_path = Path(input_path)
        suffix = input_path.suffix.lower()

        target_format = self.TARGET_FORMAT.get(suffix)
        if not target_format:
            raise ValueError(f"Unsupported format: {suffix}")

        # Create temp directory for conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Convert to modern format
            logger.info(f"Converting {input_path.name} to {target_format}...")
            converted_path = self._convert_with_libreoffice(
                input_path, target_format, temp_path
            )

            # Process with appropriate converter based on target format
            if target_format == "pptx":
                result = self._pptx_converter.convert(converted_path, output_dir)
            else:
                result = self._office_converter.convert(converted_path, output_dir)

            # Update metadata
            result.metadata["original_format"] = suffix.lstrip(".").upper()
            result.metadata["source"] = str(input_path)

            return result


@register_converter(FileFormat.DOC)
class DocConverter(LegacyOfficeConverter):
    """Converter for legacy DOC (Word 97-2003) documents."""

    supported_formats = [FileFormat.DOC]


@register_converter(FileFormat.PPT)
class PptConverter(LegacyOfficeConverter):
    """Converter for legacy PPT (PowerPoint 97-2003) documents."""

    supported_formats = [FileFormat.PPT]


@register_converter(FileFormat.XLS)
class XlsConverter(LegacyOfficeConverter):
    """Converter for legacy XLS (Excel 97-2003) documents."""

    supported_formats = [FileFormat.XLS]
