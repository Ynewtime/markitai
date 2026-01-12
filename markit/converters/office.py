"""Office document preprocessor for legacy formats.

Converts legacy Office formats (.doc, .ppt, .xls) to modern formats
(.docx, .pptx, .xlsx) using MS Office (Windows) or LibreOffice (cross-platform).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

from markit.converters.base import BaseProcessor

if TYPE_CHECKING:
    from markit.converters.libreoffice_pool import LibreOfficeProfilePool
from markit.exceptions import ConversionError
from markit.utils.logging import get_logger

log = get_logger(__name__)


# Format mappings: legacy -> modern
LEGACY_TO_MODERN = {
    ".doc": ".docx",
    ".ppt": ".pptx",
    ".xls": ".xlsx",
    ".wps": ".docx",  # WPS Writer
    ".et": ".xlsx",  # WPS Spreadsheet
    ".dps": ".pptx",  # WPS Presentation
}


class OfficePreprocessor(BaseProcessor):
    """Preprocessor that converts legacy Office formats to modern formats.

    On Windows, prefers MS Office COM automation if available.
    Falls back to LibreOffice on all platforms.
    """

    name = "office_preprocessor"

    def __init__(
        self,
        prefer_ms_office: bool = True,
        libreoffice_path: str | None = None,
        timeout: int = 120,
    ) -> None:
        """Initialize the Office preprocessor.

        Args:
            prefer_ms_office: Prefer MS Office over LibreOffice on Windows
            libreoffice_path: Custom path to LibreOffice
            timeout: Conversion timeout in seconds
        """
        self.prefer_ms_office = prefer_ms_office
        self.libreoffice_path = libreoffice_path
        self.timeout = timeout
        self._converter: _BaseOfficeConverter | None = None
        self._converted_dir: Path | None = None  # Directory for converted intermediate files

    def set_converted_dir(self, converted_dir: Path) -> None:
        """Set the directory for storing converted intermediate files.

        Args:
            converted_dir: Directory path for storing .docx/.pptx/.xlsx files
        """
        self._converted_dir = converted_dir

    async def process(self, file_path: Path) -> Path:
        """Convert legacy Office format to modern format.

        Args:
            file_path: Path to the legacy format file

        Returns:
            Path to the converted modern format file
        """
        suffix = file_path.suffix.lower()

        if suffix not in LEGACY_TO_MODERN:
            log.debug("Not a legacy format, skipping", file=str(file_path))
            return file_path

        log.info(
            "Converting legacy Office format",
            from_format=suffix,
            to_format=LEGACY_TO_MODERN[suffix],
            file=str(file_path),
        )

        try:
            # Get appropriate converter
            converter = self._get_converter(file_path)

            # Convert file
            result = await anyio.to_thread.run_sync(
                lambda: converter.convert(file_path, LEGACY_TO_MODERN[suffix], self._converted_dir),
            )

            log.info(
                "Legacy format converted",
                original=str(file_path),
                converted=str(result),
            )

            return result

        except Exception as e:
            log.error(
                "Legacy format conversion failed",
                file=str(file_path),
                error=str(e),
            )
            raise ConversionError(file_path, f"Office conversion failed: {e}", cause=e) from e

    def _get_converter(self, file_path: Path | None = None) -> _BaseOfficeConverter:
        """Get the appropriate Office converter.

        Args:
            file_path: Optional file path for logging context (first file being converted)
        """
        if self._converter is not None:
            return self._converter

        # On Windows, try MS Office first
        if sys.platform == "win32" and self.prefer_ms_office:
            try:
                self._converter = MSOfficeConverter()
                log.info(
                    "Using MS Office for conversion", file=str(file_path) if file_path else None
                )
                return self._converter
            except Exception as e:
                log.warning("MS Office not available", error=str(e))

        # Fall back to LibreOffice
        self._converter = LibreOfficeConverter(
            soffice_path=self.libreoffice_path,
            timeout=self.timeout,
        )
        log.info("Using LibreOffice for conversion", file=str(file_path) if file_path else None)
        return self._converter


class _BaseOfficeConverter:
    """Base class for Office converters."""

    def convert(
        self, file_path: Path, target_format: str, converted_dir: Path | None = None
    ) -> Path:
        """Convert file to target format.

        Args:
            file_path: Input file path
            target_format: Target format extension (e.g., '.docx')
            converted_dir: Optional directory for storing converted files

        Returns:
            Path to converted file
        """
        raise NotImplementedError


class LibreOfficeConverter(_BaseOfficeConverter):
    """Convert Office documents using LibreOffice."""

    def __init__(
        self,
        soffice_path: str | None = None,
        timeout: int = 120,
        profile_pool: LibreOfficeProfilePool | None = None,
    ) -> None:
        """Initialize LibreOffice converter.

        Args:
            soffice_path: Path to soffice executable
            timeout: Conversion timeout in seconds
            profile_pool: Optional profile pool for concurrent conversions.
                          If provided, use pooled profiles instead of temp directories.
        """
        self.soffice_path = soffice_path or self._find_soffice()
        self.timeout = timeout
        self._profile_pool = profile_pool

        if not self.soffice_path:
            raise RuntimeError("LibreOffice not found")

    def _find_soffice(self) -> str | None:
        """Find LibreOffice soffice executable."""
        # Try common locations
        if sys.platform == "win32":
            common_paths = [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            ]
            for path in common_paths:
                if Path(path).exists():
                    return path

        elif sys.platform == "darwin":
            app_path = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            if Path(app_path).exists():
                return app_path

        # Try PATH
        return shutil.which("soffice") or shutil.which("libreoffice")

    def convert(
        self, file_path: Path, target_format: str, converted_dir: Path | None = None
    ) -> Path:
        """Convert file using LibreOffice with isolated user profile.

        Uses a separate user profile directory for each conversion to allow
        concurrent execution without lock conflicts.

        Args:
            file_path: Input file path
            target_format: Target format extension (e.g., '.docx')
            converted_dir: Optional directory for storing converted files

        Returns:
            Path to converted file
        """
        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create isolated user profile directory to allow concurrent execution
            # This prevents lock conflicts when multiple LibreOffice processes run
            with tempfile.TemporaryDirectory() as profile_dir:
                # Build cross-platform file URI for user profile
                if sys.platform == "win32":
                    # Windows: file:///C:/path/to/profile
                    profile_uri = f"file:///{profile_dir.replace(os.sep, '/')}"
                else:
                    # Linux/macOS: file:///path/to/profile
                    profile_uri = f"file://{profile_dir}"

                # Determine output filter
                filter_name = self._get_filter_name(target_format)

                # Build command with isolated user profile
                cmd = [
                    self.soffice_path,
                    "--headless",
                    f"-env:UserInstallation={profile_uri}",
                    "--convert-to",
                    filter_name,
                    "--outdir",
                    str(temp_path),
                    str(file_path),
                ]

                log.debug("Running LibreOffice", command=" ".join(cmd))

                # Run conversion
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        timeout=self.timeout,
                    )
                except subprocess.TimeoutExpired as e:
                    raise ConversionError(
                        file_path,
                        f"LibreOffice conversion timed out after {self.timeout}s",
                    ) from e
                except subprocess.CalledProcessError as e:
                    raise ConversionError(
                        file_path,
                        f"LibreOffice error: {e.stderr.decode() if e.stderr else 'Unknown error'}",
                    ) from e

            # Profile directory is automatically cleaned up here

            # Find output file
            output_name = file_path.stem + target_format
            output_path = temp_path / output_name

            if not output_path.exists():
                # LibreOffice might use different naming
                for f in temp_path.iterdir():
                    if f.suffix.lower() == target_format:
                        output_path = f
                        break

            if not output_path.exists():
                raise ConversionError(
                    file_path,
                    "LibreOffice did not produce output file",
                )

            # Determine final output location
            if converted_dir:
                # Store in specified converted directory (e.g., output/converted/)
                converted_dir.mkdir(parents=True, exist_ok=True)
                final_path = converted_dir / output_name
            else:
                # Fall back to source directory (legacy behavior)
                final_path = file_path.parent / output_name

            # Copy converted file to final location (overwrites if exists)
            shutil.copy2(output_path, final_path)

            return final_path

    def _get_filter_name(self, target_format: str) -> str:
        """Get LibreOffice filter name for target format."""
        filters = {
            ".docx": "docx",
            ".xlsx": "xlsx",
            ".pptx": "pptx",
            ".pdf": "pdf",
            ".html": "html",
        }
        return filters.get(target_format, target_format.lstrip("."))

    async def convert_async(
        self, file_path: Path, target_format: str, converted_dir: Path | None = None
    ) -> Path:
        """Convert file asynchronously, using profile pool if available.

        This method uses the profile pool for concurrent conversions if one was
        provided during initialization. Otherwise, it falls back to the sync
        convert method wrapped in a thread.

        Args:
            file_path: Input file path
            target_format: Target format extension (e.g., '.docx')
            converted_dir: Optional directory for storing converted files

        Returns:
            Path to converted file
        """
        if self._profile_pool:
            # Use profile pool for concurrent conversion
            async with self._profile_pool.acquire() as profile_dir:
                return await self._convert_with_profile(
                    file_path, target_format, profile_dir, converted_dir
                )
        else:
            # Fall back to sync conversion in thread
            return await anyio.to_thread.run_sync(
                lambda: self.convert(file_path, target_format, converted_dir)
            )

    async def _convert_with_profile(
        self,
        file_path: Path,
        target_format: str,
        profile_dir: Path,
        converted_dir: Path | None = None,
    ) -> Path:
        """Convert file using a specific profile directory.

        Args:
            file_path: Input file path
            target_format: Target format extension
            profile_dir: Profile directory to use
            converted_dir: Optional directory for storing converted files

        Returns:
            Path to converted file
        """
        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Build cross-platform file URI for user profile
            if sys.platform == "win32":
                profile_uri = f"file:///{str(profile_dir).replace(os.sep, '/')}"
            else:
                profile_uri = f"file://{profile_dir}"

            # Determine output filter
            filter_name = self._get_filter_name(target_format)

            # Build command with pool profile
            cmd = [
                self.soffice_path,
                "--headless",
                f"-env:UserInstallation={profile_uri}",
                "--convert-to",
                filter_name,
                "--outdir",
                str(temp_path),
                str(file_path),
            ]

            log.debug("Running LibreOffice (pooled)", command=" ".join(cmd))

            # Run conversion in thread pool
            try:
                await anyio.to_thread.run_sync(
                    lambda: subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        timeout=self.timeout,
                    )
                )
            except subprocess.TimeoutExpired as e:
                raise ConversionError(
                    file_path,
                    f"LibreOffice conversion timed out after {self.timeout}s",
                ) from e
            except subprocess.CalledProcessError as e:
                raise ConversionError(
                    file_path,
                    f"LibreOffice error: {e.stderr.decode() if e.stderr else 'Unknown error'}",
                ) from e

            # Find output file
            output_name = file_path.stem + target_format
            output_path = temp_path / output_name

            if not output_path.exists():
                # LibreOffice might use different naming
                for f in temp_path.iterdir():
                    if f.suffix.lower() == target_format:
                        output_path = f
                        break

            if not output_path.exists():
                raise ConversionError(
                    file_path,
                    "LibreOffice did not produce output file",
                )

            # Determine final output location
            if converted_dir:
                converted_dir.mkdir(parents=True, exist_ok=True)
                final_path = converted_dir / output_name
            else:
                final_path = file_path.parent / output_name

            # Copy converted file to final location
            shutil.copy2(output_path, final_path)

            return final_path


class MSOfficeConverter(_BaseOfficeConverter):
    """Convert Office documents using MS Office COM automation (Windows only)."""

    def __init__(self) -> None:
        """Initialize MS Office converter."""
        if sys.platform != "win32":
            raise RuntimeError("MS Office converter only works on Windows")

        # Try to import win32com
        try:
            import pythoncom  # noqa: F401
            import win32com.client  # noqa: F401
        except ImportError as e:
            raise RuntimeError("pywin32 is required for MS Office conversion") from e

    def convert(
        self, file_path: Path, target_format: str, converted_dir: Path | None = None
    ) -> Path:
        """Convert file using MS Office COM automation.

        Args:
            file_path: Input file path
            target_format: Target format extension
            converted_dir: Optional directory for storing converted files

        Returns:
            Path to converted file
        """
        import pythoncom
        from pywintypes import com_error

        # Initialize COM for this thread - required when called from worker threads
        # (e.g., via anyio.to_thread.run_sync)
        pythoncom.CoInitialize()

        try:
            suffix = file_path.suffix.lower()

            # Determine output location
            if converted_dir:
                converted_dir.mkdir(parents=True, exist_ok=True)
                output_path = converted_dir / (file_path.stem + target_format)
            else:
                output_path = file_path.parent / (file_path.stem + target_format)

            try:
                if suffix == ".doc":
                    return self._convert_word(file_path, output_path)
                elif suffix == ".xls":
                    return self._convert_excel(file_path, output_path)
                elif suffix == ".ppt":
                    return self._convert_powerpoint(file_path, output_path)
                else:
                    raise ConversionError(
                        file_path,
                        f"Unsupported format for MS Office conversion: {suffix}",
                    )
            except com_error as e:
                raise ConversionError(
                    file_path,
                    f"MS Office COM error: {e}",
                ) from e
        finally:
            # Always uninitialize COM when done
            pythoncom.CoUninitialize()

    def _convert_word(self, input_path: Path, output_path: Path) -> Path:
        """Convert Word document."""
        import win32com.client

        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False

        try:
            doc = word.Documents.Open(str(input_path.absolute()))
            # 16 = wdFormatDocumentDefault (docx)
            doc.SaveAs2(str(output_path.absolute()), FileFormat=16)
            doc.Close()
            return output_path
        finally:
            word.Quit()

    def _convert_excel(self, input_path: Path, output_path: Path) -> Path:
        """Convert Excel workbook."""
        import win32com.client

        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False

        try:
            workbook = excel.Workbooks.Open(str(input_path.absolute()))
            # 51 = xlOpenXMLWorkbook (xlsx)
            workbook.SaveAs(str(output_path.absolute()), FileFormat=51)
            workbook.Close()
            return output_path
        finally:
            excel.Quit()

    def _convert_powerpoint(self, input_path: Path, output_path: Path) -> Path:
        """Convert PowerPoint presentation."""
        import win32com.client

        powerpoint = win32com.client.Dispatch("PowerPoint.Application")

        try:
            presentation = powerpoint.Presentations.Open(
                str(input_path.absolute()),
                WithWindow=False,
            )
            # 24 = ppSaveAsOpenXMLPresentation (pptx)
            presentation.SaveAs(str(output_path.absolute()), FileFormat=24)
            presentation.Close()
            return output_path
        finally:
            powerpoint.Quit()


def check_office_available() -> dict[str, bool]:
    """Check availability of Office converters.

    Returns:
        Dictionary with converter availability status
    """
    result = {
        "libreoffice": False,
        "ms_office": False,
    }

    # Check LibreOffice
    try:
        LibreOfficeConverter()  # Just verify initialization works
        result["libreoffice"] = True
    except Exception:
        pass

    # Check MS Office (Windows only)
    if sys.platform == "win32":
        try:
            import win32com.client

            # Try to create Word application
            word = win32com.client.Dispatch("Word.Application")
            word.Quit()
            result["ms_office"] = True
        except Exception:
            pass

    return result
