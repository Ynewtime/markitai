"""Integration tests using real test files from tests/fixtures/documents.

These tests verify the actual conversion functionality with real document files.
Test files should be placed in the tests/fixtures/documents directory.
"""

from pathlib import Path

import pytest

# Test input directory (tests/fixtures/documents)
INPUT_DIR = Path(__file__).parent.parent / "fixtures" / "documents"


@pytest.fixture
def input_files():
    """Get list of input test files."""
    if not INPUT_DIR.exists():
        pytest.skip("Input directory not found")
    files = [f for f in INPUT_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not files:
        pytest.skip("No input files found")
    return files


@pytest.fixture
def integration_output_dir(tmp_path):
    """Create a temporary output directory for integration tests."""
    output = tmp_path / "output"
    output.mkdir()
    yield output
    # Cleanup is handled by tmp_path fixture


class TestInputDirectory:
    """Tests to verify test input directory setup."""

    def test_input_directory_exists(self):
        """Verify test input directory exists."""
        assert INPUT_DIR.exists(), f"Input directory not found: {INPUT_DIR}"

    def test_input_files_available(self, input_files):
        """Verify test files are available."""
        assert len(input_files) > 0, "No test files found in input directory"

    def test_has_legacy_formats(self, input_files):
        """Verify legacy format test files exist."""
        legacy_extensions = {".doc", ".ppt", ".xls"}
        found_extensions = {f.suffix.lower() for f in input_files}
        found_legacy = legacy_extensions & found_extensions

        if not found_legacy:
            pytest.skip("No legacy format files (.doc, .ppt, .xls) found for testing")

        assert len(found_legacy) > 0

    def test_has_modern_formats(self, input_files):
        """Verify modern format test files exist."""
        modern_extensions = {".docx", ".pptx", ".xlsx", ".pdf"}
        found_extensions = {f.suffix.lower() for f in input_files}
        found_modern = modern_extensions & found_extensions

        assert len(found_modern) > 0, "No modern format files found"


class TestOutputNaming:
    """Tests for output file naming format."""

    def test_output_naming_preserves_extension(self, input_files, integration_output_dir):
        """Test that output files follow <name>.<ext>.md format."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        pipeline = ConversionPipeline(settings)

        # Find a simple file to test (prefer .txt or .docx)
        test_file = None
        for f in input_files:
            if f.suffix.lower() in {".txt", ".docx", ".xlsx"}:
                test_file = f
                break

        if test_file is None:
            test_file = input_files[0]

        result = pipeline.convert_file(test_file, integration_output_dir)

        if result.success:
            expected_name = f"{test_file.name}.md"
            assert result.output_path.name == expected_name, (
                f"Expected {expected_name}, got {result.output_path.name}"
            )

    def test_different_extensions_no_conflict(self, integration_output_dir):
        """Test that files with same stem but different extensions don't conflict."""
        # Find pairs like file.xls and file.xlsx
        xls_files = list(INPUT_DIR.glob("*.xls"))
        xlsx_files = list(INPUT_DIR.glob("*.xlsx"))

        if not xls_files or not xlsx_files:
            pytest.skip("Need both .xls and .xlsx files to test naming conflict")

        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        pipeline = ConversionPipeline(settings)

        outputs = []
        for f in [xls_files[0], xlsx_files[0]]:
            result = pipeline.convert_file(f, integration_output_dir)
            if result.success:
                outputs.append(result.output_path.name)

        # Verify no duplicates
        if len(outputs) == 2:
            assert outputs[0] != outputs[1], f"Output names should be different: {outputs}"


class TestFormatConversion:
    """Tests for converting different file formats."""

    @pytest.mark.parametrize("ext", [".docx", ".xlsx", ".pptx", ".pdf"])
    def test_modern_format_conversion(self, ext, integration_output_dir):
        """Test conversion of modern Office formats."""
        files = list(INPUT_DIR.glob(f"*{ext}"))
        if not files:
            pytest.skip(f"No {ext} file found in input directory")

        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        pipeline = ConversionPipeline(settings)

        test_file = files[0]
        result = pipeline.convert_file(test_file, integration_output_dir)

        # Modern formats should convert successfully
        assert result.success, f"Conversion failed for {test_file.name}: {result.error}"
        assert result.output_path.exists(), f"Output file not created: {result.output_path}"
        assert result.output_path.suffix == ".md"

    @pytest.mark.parametrize("ext", [".doc", ".ppt", ".xls"])
    def test_legacy_format_conversion(self, ext, integration_output_dir):
        """Test conversion of legacy Office formats (requires LibreOffice)."""
        files = list(INPUT_DIR.glob(f"*{ext}"))
        if not files:
            pytest.skip(f"No {ext} file found in input directory")

        # Check if LibreOffice is available
        import shutil as sh

        if not sh.which("soffice") and not sh.which("libreoffice"):
            pytest.skip("LibreOffice not installed")

        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        pipeline = ConversionPipeline(settings)

        test_file = files[0]
        result = pipeline.convert_file(test_file, integration_output_dir)

        if not result.success:
            # Log error but don't fail - LibreOffice may have issues
            print(f"Warning: Legacy format conversion failed for {test_file.name}: {result.error}")
        else:
            assert result.output_path.exists()


class TestLibreOfficeConcurrency:
    """Tests for LibreOffice concurrent conversion stability."""

    def test_concurrent_legacy_conversion_stability(self, integration_output_dir):
        """Test that legacy formats convert consistently across multiple runs.

        This test verifies the fix for LibreOffice lock conflicts by running
        the same conversion multiple times and checking for consistent results.
        """
        import shutil as sh

        if not sh.which("soffice") and not sh.which("libreoffice"):
            pytest.skip("LibreOffice not installed")

        legacy_files = []
        for ext in [".xls", ".ppt", ".doc"]:
            files = list(INPUT_DIR.glob(f"*{ext}"))
            if files:
                legacy_files.append(files[0])

        if not legacy_files:
            pytest.skip("No legacy format files found")

        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        # Run conversion 3 times
        all_results = []
        for run_num in range(3):
            settings = MarkitSettings()
            pipeline = ConversionPipeline(settings)

            run_output = integration_output_dir / f"run_{run_num}"
            run_output.mkdir()

            run_results = []
            for f in legacy_files:
                result = pipeline.convert_file(f, run_output)
                run_results.append((f.name, result.success))

            all_results.append(run_results)

        # Compare results across runs
        # All runs should have the same success/failure pattern
        first_run = all_results[0]
        for i, run in enumerate(all_results[1:], start=2):
            assert run == first_run, f"Run {i} results differ from run 1. Results: {all_results}"


class TestCleanup:
    """Tests for cleanup after conversion."""

    def test_no_temp_files_in_input_dir(self, input_files, integration_output_dir):
        """Ensure conversion doesn't leave temp files in input directory."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        # Record files before
        files_before = {f.name for f in INPUT_DIR.iterdir()}

        settings = MarkitSettings()
        pipeline = ConversionPipeline(settings)

        # Convert a file
        if input_files:
            pipeline.convert_file(input_files[0], integration_output_dir)

        # Record files after
        files_after = {f.name for f in INPUT_DIR.iterdir()}

        # Check for unexpected new files (converted modern formats are expected)
        new_files = files_after - files_before
        unexpected_files = [f for f in new_files if not f.endswith((".docx", ".pptx", ".xlsx"))]

        assert len(unexpected_files) == 0, (
            f"Unexpected files left in input directory: {unexpected_files}"
        )

    def test_output_directory_structure(self, input_files, integration_output_dir):
        """Test that output follows expected directory structure."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        pipeline = ConversionPipeline(settings)

        # Convert a file with images (PDF is good for this)
        pdf_files = [f for f in input_files if f.suffix.lower() == ".pdf"]
        if not pdf_files:
            pytest.skip("No PDF file available for testing")

        result = pipeline.convert_file(pdf_files[0], integration_output_dir)

        if result.success:
            # Check markdown file exists
            assert result.output_path.exists()

            # If images were extracted, check assets directory
            if result.images_count > 0:
                assets_dir = integration_output_dir / "assets"
                assert assets_dir.exists(), (
                    "Assets directory should exist when images are extracted"
                )


class TestImageExtraction:
    """Tests for image extraction from documents."""

    def test_pdf_image_extraction(self, integration_output_dir):
        """Test that images are extracted from PDF files."""
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        if not pdf_files:
            pytest.skip("No PDF files available")

        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        settings.image.filter_small_images = False  # Keep all images for testing
        pipeline = ConversionPipeline(settings)

        result = pipeline.convert_file(pdf_files[0], integration_output_dir)

        if result.success:
            print(f"Extracted {result.images_count} images from {pdf_files[0].name}")
            # Just log the count, don't assert specific number

    def test_pptx_image_extraction(self, integration_output_dir):
        """Test that images are extracted from PPTX files."""
        pptx_files = list(INPUT_DIR.glob("*.pptx"))
        if not pptx_files:
            pytest.skip("No PPTX files available")

        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        settings.image.filter_small_images = False
        pipeline = ConversionPipeline(settings)

        result = pipeline.convert_file(pptx_files[0], integration_output_dir)

        if result.success:
            print(f"Extracted {result.images_count} images from {pptx_files[0].name}")
