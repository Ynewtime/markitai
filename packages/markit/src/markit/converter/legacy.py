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
from markit.utils.office import (
    check_ms_excel_available,
    check_ms_powerpoint_available,
    check_ms_word_available,
    find_libreoffice,
)

if TYPE_CHECKING:
    from markit.config import MarkitConfig


def _convert_with_ms_powerpoint(input_file: Path, output_dir: Path) -> Path | None:
    """Convert .ppt to .pptx using MS Office PowerPoint (Windows only).

    Uses PowerShell subprocess for COM access, which provides:
    - Process isolation (safe for concurrent execution)
    - No pywin32 dependency required
    - Automatic COM object cleanup

    Args:
        input_file: Path to the .ppt file
        output_dir: Directory for the converted .pptx file

    Returns:
        Path to the converted .pptx file, or None if conversion failed
    """
    if platform.system() != "Windows":
        return None

    output_file = output_dir / (input_file.stem + ".pptx")

    # Escape single quotes for PowerShell string
    input_path = str(input_file.resolve()).replace("'", "''")
    output_path = str(output_file.resolve()).replace("'", "''")

    ps_script = f"""
$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = [Microsoft.Office.Core.MsoTriState]::msoFalse
try {{
    $presentation = $ppt.Presentations.Open('{input_path}', $true, $false, $false)
    $presentation.SaveAs('{output_path}', 24)  # 24 = ppSaveAsOpenXMLPresentation
    $presentation.Close()
    Write-Host "SUCCESS"
}} catch {{
    Write-Host "FAILED: $_"
}} finally {{
    $ppt.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
}}
"""

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if "SUCCESS" in result.stdout and output_file.exists():
            logger.debug(f"MS PowerPoint conversion succeeded: {output_file}")
            return output_file
        else:
            logger.warning(f"MS PowerPoint conversion failed: {result.stdout}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("MS PowerPoint conversion timed out")
        return None
    except Exception as e:
        logger.warning(f"MS PowerPoint conversion error: {e}")
        return None


def _convert_with_ms_word(input_file: Path, output_dir: Path) -> Path | None:
    """Convert .doc to .docx using MS Office Word (Windows only).

    Uses PowerShell subprocess for COM access, which provides:
    - Process isolation (safe for concurrent execution)
    - No pywin32 dependency required
    - Automatic COM object cleanup

    Args:
        input_file: Path to the .doc file
        output_dir: Directory for the converted .docx file

    Returns:
        Path to the converted .docx file, or None if conversion failed
    """
    if platform.system() != "Windows":
        return None

    output_file = output_dir / (input_file.stem + ".docx")

    # Escape single quotes for PowerShell string
    input_path = str(input_file.resolve()).replace("'", "''")
    output_path = str(output_file.resolve()).replace("'", "''")

    ps_script = f"""
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {{
    $doc = $word.Documents.Open('{input_path}')
    $doc.SaveAs2('{output_path}', 16)  # 16 = wdFormatDocumentDefault (.docx)
    $doc.Close()
    Write-Host "SUCCESS"
}} catch {{
    Write-Host "FAILED: $_"
}} finally {{
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}}
"""

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if "SUCCESS" in result.stdout and output_file.exists():
            logger.debug(f"MS Word conversion succeeded: {output_file}")
            return output_file
        else:
            logger.warning(f"MS Word conversion failed: {result.stdout}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("MS Word conversion timed out")
        return None
    except Exception as e:
        logger.warning(f"MS Word conversion error: {e}")
        return None


def _convert_with_ms_excel(input_file: Path, output_dir: Path) -> Path | None:
    """Convert .xls to .xlsx using MS Office Excel (Windows only).

    Uses PowerShell subprocess for COM access, which provides:
    - Process isolation (safe for concurrent execution)
    - No pywin32 dependency required
    - Automatic COM object cleanup

    Args:
        input_file: Path to the .xls file
        output_dir: Directory for the converted .xlsx file

    Returns:
        Path to the converted .xlsx file, or None if conversion failed
    """
    if platform.system() != "Windows":
        return None

    output_file = output_dir / (input_file.stem + ".xlsx")

    # Escape single quotes for PowerShell string
    input_path = str(input_file.resolve()).replace("'", "''")
    output_path = str(output_file.resolve()).replace("'", "''")

    ps_script = f"""
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
try {{
    $workbook = $excel.Workbooks.Open('{input_path}')
    $workbook.SaveAs('{output_path}', 51)  # 51 = xlOpenXMLWorkbook (.xlsx)
    $workbook.Close($false)
    Write-Host "SUCCESS"
}} catch {{
    Write-Host "FAILED: $_"
}} finally {{
    $excel.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
}}
"""

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if "SUCCESS" in result.stdout and output_file.exists():
            logger.debug(f"MS Excel conversion succeeded: {output_file}")
            return output_file
        else:
            logger.warning(f"MS Excel conversion failed: {result.stdout}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("MS Excel conversion timed out")
        return None
    except Exception as e:
        logger.warning(f"MS Excel conversion error: {e}")
        return None


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
    doc_files = [f for f in files if f.suffix.lower() == ".doc"]
    ppt_files = [f for f in files if f.suffix.lower() == ".ppt"]
    xls_files = [f for f in files if f.suffix.lower() == ".xls"]

    results: dict[Path, Path] = {}

    # Run conversions for different Office apps in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        if doc_files and check_ms_word_available():
            futures.append(
                executor.submit(_batch_convert_with_word, doc_files, output_dir)
            )

        if ppt_files and check_ms_powerpoint_available():
            futures.append(
                executor.submit(_batch_convert_with_powerpoint, ppt_files, output_dir)
            )

        if xls_files and check_ms_excel_available():
            futures.append(
                executor.submit(_batch_convert_with_excel, xls_files, output_dir)
            )

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                converted = future.result()
                results.update(converted)
            except Exception as e:
                logger.warning(f"Batch conversion failed: {e}")

    return results


def _batch_convert_with_word(files: list[Path], output_dir: Path) -> dict[Path, Path]:
    """Batch convert .doc files using a single Word COM session."""
    if not files:
        return {}

    # Build file list for PowerShell
    file_entries = []
    for f in files:
        input_path = str(f.resolve()).replace("'", "''")
        output_path = str((output_dir / (f.stem + ".docx")).resolve()).replace(
            "'", "''"
        )
        file_entries.append(f"@{{Input='{input_path}'; Output='{output_path}'}}")

    files_array = "@(" + ", ".join(file_entries) + ")"

    ps_script = f"""
$files = {files_array}
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$results = @()
try {{
    foreach ($file in $files) {{
        try {{
            $doc = $word.Documents.Open($file.Input)
            $doc.SaveAs2($file.Output, 16)
            $doc.Close()
            $results += "OK:" + $file.Input
        }} catch {{
            $results += "FAIL:" + $file.Input + ":" + $_
        }}
    }}
}} finally {{
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}}
$results -join "`n"
"""

    return _run_batch_conversion(ps_script, files, output_dir, ".docx", "Word")


def _batch_convert_with_powerpoint(
    files: list[Path], output_dir: Path
) -> dict[Path, Path]:
    """Batch convert .ppt files using a single PowerPoint COM session."""
    if not files:
        return {}

    file_entries = []
    for f in files:
        input_path = str(f.resolve()).replace("'", "''")
        output_path = str((output_dir / (f.stem + ".pptx")).resolve()).replace(
            "'", "''"
        )
        file_entries.append(f"@{{Input='{input_path}'; Output='{output_path}'}}")

    files_array = "@(" + ", ".join(file_entries) + ")"

    ps_script = f"""
$files = {files_array}
$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = [Microsoft.Office.Core.MsoTriState]::msoFalse
$results = @()
try {{
    foreach ($file in $files) {{
        try {{
            $presentation = $ppt.Presentations.Open($file.Input, $true, $false, $false)
            $presentation.SaveAs($file.Output, 24)
            $presentation.Close()
            $results += "OK:" + $file.Input
        }} catch {{
            $results += "FAIL:" + $file.Input + ":" + $_
        }}
    }}
}} finally {{
    $ppt.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
}}
$results -join "`n"
"""

    return _run_batch_conversion(ps_script, files, output_dir, ".pptx", "PowerPoint")


def _batch_convert_with_excel(files: list[Path], output_dir: Path) -> dict[Path, Path]:
    """Batch convert .xls files using a single Excel COM session."""
    if not files:
        return {}

    file_entries = []
    for f in files:
        input_path = str(f.resolve()).replace("'", "''")
        output_path = str((output_dir / (f.stem + ".xlsx")).resolve()).replace(
            "'", "''"
        )
        file_entries.append(f"@{{Input='{input_path}'; Output='{output_path}'}}")

    files_array = "@(" + ", ".join(file_entries) + ")"

    ps_script = f"""
$files = {files_array}
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$results = @()
try {{
    foreach ($file in $files) {{
        try {{
            $workbook = $excel.Workbooks.Open($file.Input)
            $workbook.SaveAs($file.Output, 51)
            $workbook.Close($false)
            $results += "OK:" + $file.Input
        }} catch {{
            $results += "FAIL:" + $file.Input + ":" + $_
        }}
    }}
}} finally {{
    $excel.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($excel) | Out-Null
}}
$results -join "`n"
"""

    return _run_batch_conversion(ps_script, files, output_dir, ".xlsx", "Excel")


def _run_batch_conversion(
    ps_script: str,
    files: list[Path],
    output_dir: Path,
    new_ext: str,
    app_name: str,
) -> dict[Path, Path]:
    """Execute batch conversion PowerShell script and parse results."""
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

    def __init__(self, config: MarkitConfig | None = None) -> None:
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
        if suffix == ".ppt" and check_ms_powerpoint_available():
            logger.info(f"Converting {input_path.name} with MS PowerPoint...")
            converted_path = _convert_with_ms_powerpoint(input_path, output_dir)
            if converted_path:
                return converted_path
            logger.warning("MS PowerPoint conversion failed, trying LibreOffice...")

        if suffix == ".doc" and check_ms_word_available():
            logger.info(f"Converting {input_path.name} with MS Word...")
            converted_path = _convert_with_ms_word(input_path, output_dir)
            if converted_path:
                return converted_path
            logger.warning("MS Word conversion failed, trying LibreOffice...")

        if suffix == ".xls" and check_ms_excel_available():
            logger.info(f"Converting {input_path.name} with MS Excel...")
            converted_path = _convert_with_ms_excel(input_path, output_dir)
            if converted_path:
                return converted_path
            logger.warning("MS Excel conversion failed, trying LibreOffice...")

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
