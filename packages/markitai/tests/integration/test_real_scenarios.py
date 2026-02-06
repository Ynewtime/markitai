"""Real-world scenario tests using actual fixture files.

This module tests realistic business scenarios with actual document files.
Tests are organized to minimize redundant conversions using shared fixtures.

Note: These tests are slower due to real file processing.
Use `pytest -m "not slow"` to skip slow tests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from markitai.cli import app
from markitai.utils.office import find_libreoffice

_HAS_LIBREOFFICE = bool(find_libreoffice())

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Return a CLI test runner."""
    return CliRunner()


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def converted_fixtures(tmp_path_factory) -> dict:
    """Convert fixtures once and share results across tests.

    This session-scoped fixture converts the fixtures directory once
    and returns paths to output files for verification.

    Note: Excludes .urls files to avoid network-dependent URL processing
    in CI environments. URL processing is tested separately.
    """
    import shutil

    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    output_dir = tmp_path_factory.mktemp("converted")

    # Create temp input dir excluding .urls files (network-dependent)
    input_dir = tmp_path_factory.mktemp("fixtures_no_urls")
    for item in fixtures_dir.iterdir():
        if item.suffix == ".urls":
            continue
        if item.is_dir():
            shutil.copytree(item, input_dir / item.name)
        else:
            shutil.copy2(item, input_dir / item.name)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [str(input_dir), "-o", str(output_dir)],
    )

    return {
        "exit_code": result.exit_code,
        "output": result.output,
        "output_dir": output_dir,
        "fixtures_dir": fixtures_dir,
    }


# =============================================================================
# Test: Batch Conversion Results (uses shared fixture)
# =============================================================================


@pytest.mark.slow
class TestBatchConversionResults:
    """Tests that verify batch conversion results using shared fixture."""

    @pytest.mark.skipif(not _HAS_LIBREOFFICE, reason="LibreOffice not installed")
    def test_batch_conversion_succeeds(self, converted_fixtures: dict):
        """Test batch conversion completes successfully."""
        assert converted_fixtures["exit_code"] == 0

    def test_pdf_converted(self, converted_fixtures: dict):
        """Test PDF file was converted."""
        output_dir = converted_fixtures["output_dir"]
        pdf_output = output_dir / "file-example_PDF_500_kB.pdf.md"
        assert pdf_output.exists(), "PDF should be converted"

        content = pdf_output.read_text(encoding="utf-8")
        assert content.startswith("---"), "Should have frontmatter"
        assert len(content) > 200, "Should have substantial content"

    def test_xlsx_converted(self, converted_fixtures: dict):
        """Test Excel file was converted with table structure."""
        output_dir = converted_fixtures["output_dir"]
        xlsx_output = output_dir / "file_example_XLSX_100.xlsx.md"
        assert xlsx_output.exists(), "XLSX should be converted"

        content = xlsx_output.read_text(encoding="utf-8")
        assert "|" in content, "Should contain markdown table syntax"

    def test_pptx_converted(self, converted_fixtures: dict):
        """Test PowerPoint file was converted."""
        output_dir = converted_fixtures["output_dir"]
        pptx_output = output_dir / "Free_Test_Data_500KB_PPTX.pptx.md"
        assert pptx_output.exists(), "PPTX should be converted"

    def test_jpg_converted(self, converted_fixtures: dict):
        """Test image file was converted."""
        output_dir = converted_fixtures["output_dir"]
        jpg_output = output_dir / "candy.JPG.md"
        assert jpg_output.exists(), "JPG should be converted"

    @pytest.mark.skipif(not _HAS_LIBREOFFICE, reason="LibreOffice not installed")
    def test_subdirectory_preserved(self, converted_fixtures: dict):
        """Test subdirectory structure is preserved."""
        output_dir = converted_fixtures["output_dir"]
        sub_output = output_dir / "sub_dir"
        assert sub_output.exists(), "Subdirectory should be preserved"

        # Check legacy format files in subdirectory
        doc_output = sub_output / "file-sample_100kB.doc.md"
        assert doc_output.exists(), "DOC in subdirectory should be converted"

    def test_report_generated(self, converted_fixtures: dict):
        """Test conversion report was generated."""
        output_dir = converted_fixtures["output_dir"]
        reports_dir = output_dir / "reports"
        assert reports_dir.exists(), "Reports directory should exist"

        report_files = list(reports_dir.glob("*.json"))
        assert len(report_files) == 1, "Should have one report file"

    @pytest.mark.skipif(not _HAS_LIBREOFFICE, reason="LibreOffice not installed")
    def test_report_structure(self, converted_fixtures: dict):
        """Test report has correct structure."""
        output_dir = converted_fixtures["output_dir"]
        reports_dir = output_dir / "reports"
        report_files = list(reports_dir.glob("*.json"))
        report = json.loads(report_files[0].read_text(encoding="utf-8"))

        assert "version" in report
        assert "summary" in report
        assert "documents" in report

        summary = report["summary"]
        assert summary["total_documents"] >= 5
        assert summary["failed_documents"] == 0

    def test_frontmatter_structure(self, converted_fixtures: dict):
        """Test converted files have proper frontmatter."""
        output_dir = converted_fixtures["output_dir"]
        pdf_output = output_dir / "file-example_PDF_500_kB.pdf.md"
        content = pdf_output.read_text(encoding="utf-8")

        parts = content.split("---", 2)
        assert len(parts) >= 3, "Should have frontmatter"
        frontmatter = parts[1]

        assert "title:" in frontmatter or "source:" in frontmatter
        assert "markitai_processed:" in frontmatter


# =============================================================================
# Test: Single File Scenarios (individual conversions)
# =============================================================================


class TestSingleFileScenarios:
    """Tests for specific single file conversion scenarios."""

    @pytest.mark.slow
    def test_pdf_with_ocr(self, runner: CliRunner, fixtures_dir: Path, tmp_path: Path):
        """Test PDF conversion with OCR enabled."""
        pdf_file = fixtures_dir / "file-example_PDF_500_kB.pdf"
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(pdf_file), "-o", str(output_dir), "--ocr"],
        )

        assert result.exit_code == 0
        assert (output_dir / "file-example_PDF_500_kB.pdf.md").exists()

    def test_pdf_with_screenshot(
        self, runner: CliRunner, fixtures_dir: Path, tmp_path: Path
    ):
        """Test PDF conversion with screenshots."""
        pdf_file = fixtures_dir / "file-example_PDF_500_kB.pdf"
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(pdf_file), "-o", str(output_dir), "--screenshot"],
        )

        assert result.exit_code == 0


# =============================================================================
# Test: Preset Scenarios
# =============================================================================


class TestPresetScenarios:
    """Tests for preset configuration with real files."""

    def test_minimal_preset(
        self, runner: CliRunner, fixtures_dir: Path, tmp_path: Path
    ):
        """Test minimal preset produces clean output."""
        pdf_file = fixtures_dir / "file-example_PDF_500_kB.pdf"
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(pdf_file), "-o", str(output_dir), "--preset", "minimal"],
        )

        assert result.exit_code == 0
        assert (output_dir / "file-example_PDF_500_kB.pdf.md").exists()

    def test_rich_preset_dry_run(
        self, runner: CliRunner, fixtures_dir: Path, tmp_path: Path
    ):
        """Test rich preset in dry run mode."""
        pdf_file = fixtures_dir / "file-example_PDF_500_kB.pdf"
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(pdf_file), "-o", str(output_dir), "--preset", "rich", "--dry-run"],
        )

        assert result.exit_code == 0


# =============================================================================
# Test: URL List Scenarios
# =============================================================================


class TestURLListScenarios:
    """Tests for .urls file processing."""

    def test_urls_file_dry_run(
        self, runner: CliRunner, fixtures_dir: Path, tmp_path: Path
    ):
        """Test .urls file dry run."""
        urls_file = fixtures_dir / "test.urls"
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(urls_file), "-o", str(output_dir), "--dry-run"],
        )

        assert result.exit_code == 0

    @pytest.mark.network
    def test_urls_file_conversion(
        self, runner: CliRunner, fixtures_dir: Path, tmp_path: Path
    ):
        """Test .urls file batch conversion (requires network)."""
        urls_file = fixtures_dir / "test.urls"
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(urls_file), "-o", str(output_dir)],
        )

        # May partially fail due to network
        assert result.exit_code in (0, 1)


# =============================================================================
# Test: Batch Options
# =============================================================================


class TestBatchOptions:
    """Tests for batch conversion options."""

    def test_batch_dry_run(self, runner: CliRunner, fixtures_dir: Path, tmp_path: Path):
        """Test batch dry-run shows files without converting."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(fixtures_dir), "-o", str(output_dir), "--dry-run"],
        )

        assert result.exit_code == 0
        # No files should be created
        md_files = list(output_dir.rglob("*.md")) if output_dir.exists() else []
        assert len(md_files) == 0

    def test_batch_with_concurrency(
        self, runner: CliRunner, fixtures_dir: Path, tmp_path: Path
    ):
        """Test batch conversion with custom concurrency."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(fixtures_dir), "-o", str(output_dir), "-j", "1", "--dry-run"],
        )

        assert result.exit_code == 0


# =============================================================================
# Test: Error Recovery
# =============================================================================


class TestErrorRecovery:
    """Tests for error handling and recovery scenarios."""

    def test_partial_batch_failure_continues(self, runner: CliRunner, tmp_path: Path):
        """Test batch processing continues after individual file failures."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        (input_dir / "valid.txt").write_text("Valid content")
        (input_dir / "invalid.xyz").write_text("Invalid format")
        (input_dir / "also_valid.md").write_text("# Also valid")

        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(input_dir), "-o", str(output_dir)],
        )

        assert result.exit_code in (0, 1)
        assert (output_dir / "valid.txt.md").exists()
        assert (output_dir / "also_valid.md.md").exists()


# =============================================================================
# Test: Report Verification
# =============================================================================


@pytest.mark.slow
class TestReportVerification:
    """Tests for report content using shared fixture."""

    def test_report_timing(self, converted_fixtures: dict):
        """Test report contains timing information."""
        output_dir = converted_fixtures["output_dir"]
        reports_dir = output_dir / "reports"
        report_files = list(reports_dir.glob("*.json"))
        report = json.loads(report_files[0].read_text(encoding="utf-8"))

        assert "generated_at" in report
        assert re.match(r"\d{4}-\d{2}-\d{2}", report["generated_at"])

    def test_report_documents(self, converted_fixtures: dict):
        """Test report documents list."""
        output_dir = converted_fixtures["output_dir"]
        reports_dir = output_dir / "reports"
        report_files = list(reports_dir.glob("*.json"))
        report = json.loads(report_files[0].read_text(encoding="utf-8"))

        assert len(report["documents"]) >= 5
        for doc in report["documents"].values():
            assert "duration" in doc or "error" in doc
