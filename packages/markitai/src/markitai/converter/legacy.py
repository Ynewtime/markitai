"""Legacy Office format converters (DOC, PPT, XLS - Office 97-2003).

These formats require conversion to modern formats first.
Conversion priority:
1. MS Office COM (Windows) - faster and more accurate
2. LibreOffice CLI (cross-platform) - fallback
"""

from __future__ import annotations

import platform
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)
from markitai.converter.office import OfficeConverter, PptxConverter
from markitai.utils.office import (
    check_ms_excel_available,
    check_ms_powerpoint_available,
    check_ms_word_available,
    find_libreoffice,
)

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig


# =============================================================================
# COM Application Configuration
# =============================================================================


@dataclass
class COMAppConfig:
    """Configuration for a COM Office application."""

    name: str  # Display name (Word, PowerPoint, Excel)
    com_class: str  # COM ProgID (Word.Application, etc.)
    input_ext: str  # Source extension (.doc, .ppt, .xls)
    output_ext: str  # Target extension (.docx, .pptx, .xlsx)
    save_format: int  # SaveAs format code
    init_script: str  # PowerShell initialization lines
    open_script: str  # PowerShell document open command (uses {input})
    save_script: str  # PowerShell save command (uses {output}, {format})
    close_script: str  # PowerShell close command
    cleanup_script: str  # PowerShell cleanup lines
    availability_check: Callable[[], bool]  # Function to check if app is available


# PowerPoint configuration
POWERPOINT_CONFIG = COMAppConfig(
    name="PowerPoint",
    com_class="PowerPoint.Application",
    input_ext=".ppt",
    output_ext=".pptx",
    save_format=24,  # ppSaveAsOpenXMLPresentation
    init_script="$app.Visible = [Microsoft.Office.Core.MsoTriState]::msoFalse",
    open_script="$doc = $app.Presentations.Open('{input}', $true, $false, $false)",
    save_script="$doc.SaveAs('{output}', {format})",
    close_script="$doc.Close()",
    cleanup_script="",
    availability_check=check_ms_powerpoint_available,
)

# Word configuration
WORD_CONFIG = COMAppConfig(
    name="Word",
    com_class="Word.Application",
    input_ext=".doc",
    output_ext=".docx",
    save_format=16,  # wdFormatDocumentDefault
    init_script="$app.Visible = $false",
    open_script="$doc = $app.Documents.Open('{input}')",
    save_script="$doc.SaveAs2('{output}', {format})",
    close_script="$doc.Close()",
    cleanup_script="",
    availability_check=check_ms_word_available,
)

# Excel configuration
EXCEL_CONFIG = COMAppConfig(
    name="Excel",
    com_class="Excel.Application",
    input_ext=".xls",
    output_ext=".xlsx",
    save_format=51,  # xlOpenXMLWorkbook
    init_script="$app.Visible = $false\n$app.DisplayAlerts = $false",
    open_script="$doc = $app.Workbooks.Open('{input}')",
    save_script="$doc.SaveAs('{output}', {format})",
    close_script="$doc.Close($false)",
    cleanup_script="",
    availability_check=check_ms_excel_available,
)

# Map extension to config
COM_CONFIGS: dict[str, COMAppConfig] = {
    ".ppt": POWERPOINT_CONFIG,
    ".doc": WORD_CONFIG,
    ".xls": EXCEL_CONFIG,
}


# =============================================================================
# Single File COM Conversion
# =============================================================================


def _build_single_file_script(
    config: COMAppConfig, input_path: str, output_path: str
) -> str:
    """Build PowerShell script for single file conversion.

    Args:
        config: COM application configuration
        input_path: Escaped input file path
        output_path: Escaped output file path

    Returns:
        PowerShell script string
    """
    open_cmd = config.open_script.format(input=input_path)
    save_cmd = config.save_script.format(output=output_path, format=config.save_format)

    return f"""
$app = New-Object -ComObject {config.com_class}
{config.init_script}
try {{
    {open_cmd}
    {save_cmd}
    {config.close_script}
    Write-Host "SUCCESS"
}} catch {{
    Write-Host "FAILED: $_"
}} finally {{
    $app.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($app) | Out-Null
}}
"""


def _convert_with_com(
    input_file: Path,
    output_dir: Path,
    config: COMAppConfig,
) -> Path | None:
    """Convert a file using MS Office COM (Windows only).

    Uses PowerShell subprocess for COM access, which provides:
    - Process isolation (safe for concurrent execution)
    - No pywin32 dependency required
    - Automatic COM object cleanup

    Args:
        input_file: Path to the source file
        output_dir: Directory for the converted file
        config: COM application configuration

    Returns:
        Path to the converted file, or None if conversion failed
    """
    if platform.system() != "Windows":
        return None

    output_file = output_dir / (input_file.stem + config.output_ext)

    # Escape single quotes for PowerShell string
    input_path = str(input_file.resolve()).replace("'", "''")
    output_path = str(output_file.resolve()).replace("'", "''")

    ps_script = _build_single_file_script(config, input_path, output_path)

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if "SUCCESS" in result.stdout and output_file.exists():
            logger.debug(f"MS {config.name} conversion succeeded: {output_file}")
            return output_file
        else:
            logger.warning(f"MS {config.name} conversion failed: {result.stdout}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning(f"MS {config.name} conversion timed out")
        return None
    except Exception as e:
        logger.warning(f"MS {config.name} conversion error: {e}")
        return None


# =============================================================================
# Batch COM Conversion
# =============================================================================


def _build_batch_script(config: COMAppConfig, files_array: str) -> str:
    """Build PowerShell script for batch file conversion.

    Args:
        config: COM application configuration
        files_array: PowerShell array string of file entries

    Returns:
        PowerShell script string
    """
    # Build the loop body with proper variable substitution
    open_cmd = config.open_script.replace("'{input}'", "$file.Input")
    save_cmd = config.save_script.replace("'{output}'", "$file.Output").replace(
        "{format}", str(config.save_format)
    )

    return f"""
$files = {files_array}
$app = New-Object -ComObject {config.com_class}
{config.init_script}
$results = @()
try {{
    foreach ($file in $files) {{
        try {{
            {open_cmd}
            {save_cmd}
            {config.close_script}
            $results += "OK:" + $file.Input
        }} catch {{
            $results += "FAIL:" + $file.Input + ":" + $_
        }}
    }}
}} finally {{
    $app.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($app) | Out-Null
}}
$results -join "`n"
"""


def _run_batch_conversion(
    ps_script: str,
    files: list[Path],
    output_dir: Path,
    new_ext: str,
    app_name: str,
) -> dict[Path, Path]:
    """Execute batch conversion PowerShell script and parse results.

    Args:
        ps_script: PowerShell script to execute
        files: List of input files
        output_dir: Output directory for converted files
        new_ext: New file extension
        app_name: Application name for logging

    Returns:
        Dict mapping original file path to converted file path
    """
    results: dict[Path, Path] = {}

    try:
        logger.info(f"Batch converting {len(files)} files with MS {app_name}...")
        proc_result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=120 * len(files),  # Scale timeout with file count
        )

        # Parse results
        for line in proc_result.stdout.strip().split("\n"):
            if line.startswith("OK:"):
                input_path = Path(line[3:].strip())
                output_path = output_dir / (input_path.stem + new_ext)
                if output_path.exists():
                    # Find original file in list (case-insensitive match)
                    for f in files:
                        if (
                            f.resolve() == input_path
                            or str(f.resolve()).lower() == line[3:].strip().lower()
                        ):
                            results[f] = output_path
                            break
            elif line.startswith("FAIL:"):
                parts = line[5:].split(":", 1)
                logger.warning(
                    f"MS {app_name} failed for {parts[0]}: {parts[1] if len(parts) > 1 else 'unknown'}"
                )

        logger.info(
            f"MS {app_name} batch conversion: {len(results)}/{len(files)} succeeded"
        )

    except subprocess.TimeoutExpired:
        logger.warning(f"MS {app_name} batch conversion timed out")
    except Exception as e:
        logger.warning(f"MS {app_name} batch conversion error: {e}")

    return results


def _batch_convert_with_com(
    files: list[Path],
    output_dir: Path,
    config: COMAppConfig,
) -> dict[Path, Path]:
    """Batch convert files using a single COM session.

    Args:
        files: List of files to convert
        output_dir: Output directory
        config: COM application configuration

    Returns:
        Dict mapping original file path to converted file path
    """
    if not files:
        return {}

    # Build file list for PowerShell
    file_entries = []
    for f in files:
        input_path = str(f.resolve()).replace("'", "''")
        output_path = str(
            (output_dir / (f.stem + config.output_ext)).resolve()
        ).replace("'", "''")
        file_entries.append(f"@{{Input='{input_path}'; Output='{output_path}'}}")

    files_array = "@(" + ", ".join(file_entries) + ")"
    ps_script = _build_batch_script(config, files_array)

    return _run_batch_conversion(
        ps_script, files, output_dir, config.output_ext, config.name
    )


def batch_convert_legacy_files(
    files: list[Path],
    output_dir: Path,
) -> dict[Path, Path]:
    """Batch convert legacy Office files using a single COM session per app.

    This significantly reduces overhead by:
    - Starting each Office application only once
    - Processing all files of the same type in one session
    - Running Word, PowerPoint, and Excel conversions in parallel
    - Reducing PowerShell process spawn overhead

    Args:
        files: List of legacy format files (.doc, .ppt, .xls)
        output_dir: Directory for converted files

    Returns:
        Dict mapping original file path to converted file path.
        Files that failed conversion are not included.
    """
    if platform.system() != "Windows":
        return {}

    import concurrent.futures

    # Group files by type
    files_by_ext: dict[str, list[Path]] = {}
    for f in files:
        ext = f.suffix.lower()
        if ext in COM_CONFIGS:
            if ext not in files_by_ext:
                files_by_ext[ext] = []
            files_by_ext[ext].append(f)

    results: dict[Path, Path] = {}

    # Run conversions for different Office apps in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        for ext, file_list in files_by_ext.items():
            config = COM_CONFIGS[ext]
            if file_list and config.availability_check():
                futures.append(
                    executor.submit(
                        _batch_convert_with_com, file_list, output_dir, config
                    )
                )

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                converted = future.result()
                results.update(converted)
            except Exception as e:
                logger.warning(f"Batch conversion failed: {e}")

    return results


# =============================================================================
# Legacy Office Converter Class
# =============================================================================


class LegacyOfficeConverter(BaseConverter):
    """Base converter for legacy Office documents (DOC, PPT, XLS).

    Conversion priority:
    1. MS Office COM (Windows) - faster and more accurate
    2. LibreOffice CLI (cross-platform) - fallback
    """

    # Mapping of legacy format to target format
    TARGET_FORMAT: dict[str, str] = {
        ".doc": "docx",
        ".ppt": "pptx",
        ".xls": "xlsx",
    }

    def __init__(self, config: MarkitaiConfig | None = None) -> None:
        super().__init__(config)
        self._office_converter = OfficeConverter(config)
        self._pptx_converter = PptxConverter(config)
        self._soffice_path = find_libreoffice()

    def _convert_legacy_format(
        self,
        input_path: Path,
        target_format: str,
        output_dir: Path,
    ) -> Path:
        """Convert legacy format to modern format.

        Tries MS Office COM first (Windows), falls back to LibreOffice.

        Args:
            input_path: Path to the legacy format file
            target_format: Target format (docx, pptx, xlsx)
            output_dir: Directory for converted file

        Returns:
            Path to the converted file

        Raises:
            RuntimeError: If conversion fails with all methods
        """
        suffix = input_path.suffix.lower()
        converted_path: Path | None = None

        # Try MS Office COM first (Windows only)
        config = COM_CONFIGS.get(suffix)
        if config and config.availability_check():
            logger.info(f"Converting {input_path.name} with MS {config.name}...")
            converted_path = _convert_with_com(input_path, output_dir, config)
            if converted_path:
                return converted_path
            logger.warning(f"MS {config.name} conversion failed, trying LibreOffice...")

        # Fallback to LibreOffice
        if self._soffice_path:
            logger.info(f"Converting {input_path.name} with LibreOffice...")
            return self._convert_with_libreoffice(input_path, target_format, output_dir)

        # No conversion method available
        if platform.system() == "Windows":
            raise RuntimeError(
                f"Cannot convert {suffix} files. "
                "Install Microsoft Office (recommended) or LibreOffice."
            )
        else:
            raise RuntimeError(f"Cannot convert {suffix} files. Install LibreOffice.")

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

            except subprocess.TimeoutExpired as e:
                raise RuntimeError("LibreOffice conversion timed out") from e

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

            # Convert to modern format (COM first, LibreOffice fallback)
            converted_path = self._convert_legacy_format(
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


# =============================================================================
# Registered Converters
# =============================================================================


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
