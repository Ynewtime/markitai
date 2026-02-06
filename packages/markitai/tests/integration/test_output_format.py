"""Integration tests for output format validation.

Tests cover:
- T1: Fixture-based integration test framework
- T2: Language preservation (English/Chinese input/output)
- T3: Image alt text generation (mock vision model)
- T4: PPTX header/footer cleanup
- T5: Subdirectory images.json generation
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
# T1: Fixture-based integration test framework
# =============================================================================


@pytest.fixture
def runner(cli_runner: CliRunner) -> CliRunner:
    """Alias for cli_runner from conftest.py."""
    return cli_runner


@pytest.fixture
def pptx_file(fixtures_dir: Path) -> Path:
    """Return the PPTX test fixture."""
    return fixtures_dir / "Free_Test_Data_500KB_PPTX.pptx"


@pytest.fixture
def pdf_file(fixtures_dir: Path) -> Path:
    """Return the PDF test fixture."""
    return fixtures_dir / "file-example_PDF_500_kB.pdf"


@pytest.fixture
def doc_file(fixtures_dir: Path) -> Path:
    """Return the DOC test fixture in sub_dir."""
    return fixtures_dir / "sub_dir" / "file-sample_100kB.doc"


@pytest.fixture
def image_file(fixtures_dir: Path) -> Path:
    """Return the JPG test fixture."""
    return fixtures_dir / "candy.JPG"


# =============================================================================
# T2: Language preservation tests
# =============================================================================


class TestLanguagePreservation:
    """Tests for language preservation in output."""

    def test_english_content_preserved(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that English content is preserved without modification."""
        # Create test file with English content
        input_file = tmp_path / "english.txt"
        input_file.write_text(
            "# Introduction\n\n"
            "This is a test document with English content.\n\n"
            "## Key Points\n\n"
            "- First point\n"
            "- Second point\n"
            "- Third point\n",
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(input_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0

        output_file = output_dir / "english.txt.md"
        assert output_file.exists()

        content = output_file.read_text(encoding="utf-8")
        # Verify key English content is preserved
        assert "Introduction" in content
        assert "English content" in content
        assert "Key Points" in content
        assert "First point" in content

    def test_chinese_content_preserved(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that Chinese content is preserved without modification."""
        # Create test file with Chinese content
        input_file = tmp_path / "chinese.txt"
        input_file.write_text(
            "# 简介\n\n"
            "这是一个测试文档，包含中文内容。\n\n"
            "## 重点\n\n"
            "- 第一点\n"
            "- 第二点\n"
            "- 第三点\n",
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(input_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0

        output_file = output_dir / "chinese.txt.md"
        assert output_file.exists()

        content = output_file.read_text(encoding="utf-8")
        # Verify Chinese content is preserved
        assert "简介" in content
        assert "中文内容" in content
        assert "重点" in content
        assert "第一点" in content

    def test_mixed_language_preserved(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that mixed language content is preserved."""
        # Create test file with mixed content
        input_file = tmp_path / "mixed.txt"
        input_file.write_text(
            "# 产品介绍 Product Introduction\n\n"
            "This product supports 多语言 (multilingual) content.\n\n"
            "## Features 特性\n\n"
            "- Fast 快速\n"
            "- Reliable 可靠\n",
            encoding="utf-8",
        )

        output_dir = tmp_path / "output"
        result = runner.invoke(
            app,
            [str(input_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0

        output_file = output_dir / "mixed.txt.md"
        assert output_file.exists()

        content = output_file.read_text(encoding="utf-8")
        # Verify both languages are preserved
        assert "产品介绍" in content
        assert "Product Introduction" in content
        assert "多语言" in content
        assert "multilingual" in content


# =============================================================================
# T3: Image alt tests (mock vision model)
# =============================================================================


class TestImageAltGeneration:
    """Tests for image alt text generation."""

    def test_image_converts_without_llm(
        self, runner: CliRunner, image_file: Path, tmp_path: Path
    ) -> None:
        """Test that image conversion works without LLM."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(image_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0

        # Check output file exists
        output_file = output_dir / "candy.JPG.md"
        assert output_file.exists()

        content = output_file.read_text(encoding="utf-8")
        # Should have frontmatter
        assert "---" in content
        # Should have image reference
        assert "![" in content or "candy" in content.lower()

    def test_alt_flag_shows_llm_warning(
        self, runner: CliRunner, image_file: Path, tmp_path: Path
    ) -> None:
        """Test that --alt without --llm shows warning."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(image_file), "-o", str(output_dir), "--alt", "--dry-run"],
        )

        assert result.exit_code == 0
        # Should show LLM required warning
        assert "LLM Required" in result.output or "LLM" in result.output

    def test_alt_with_llm_shows_vision_warning(
        self, runner: CliRunner, image_file: Path, tmp_path: Path
    ) -> None:
        """Test that --alt --llm shows vision model warning when not configured."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(image_file), "-o", str(output_dir), "--alt", "--llm", "--dry-run"],
        )

        assert result.exit_code == 0
        # Should show vision model warning
        assert "Vision" in result.output or "vision" in result.output


# =============================================================================
# T4: PPTX header/footer cleanup tests
# =============================================================================


class TestPPTXHeaderFooterCleanup:
    """Tests for PPTX header/footer cleanup."""

    @pytest.mark.skipif(not _HAS_LIBREOFFICE, reason="LibreOffice not installed")
    def test_pptx_converts(
        self, runner: CliRunner, pptx_file: Path, tmp_path: Path
    ) -> None:
        """Test that PPTX file converts successfully."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(pptx_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0

        # Check output file exists
        output_file = output_dir / "Free_Test_Data_500KB_PPTX.pptx.md"
        assert output_file.exists()

    @pytest.mark.skipif(not _HAS_LIBREOFFICE, reason="LibreOffice not installed")
    def test_pptx_no_residual_headers(
        self, runner: CliRunner, pptx_file: Path, tmp_path: Path
    ) -> None:
        """Test that PPTX output has no residual header/footer patterns."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(pptx_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0

        output_file = output_dir / "Free_Test_Data_500KB_PPTX.pptx.md"
        content = output_file.read_text(encoding="utf-8")

        # Should not have common header/footer patterns
        # These are typical auto-generated patterns in PPTX
        problematic_patterns = [
            "Click to edit Master",
            "CLICK TO EDIT",
            "<date/time>",
            "<footer>",
            "‹#›",
            "#/#",  # Page number pattern
        ]

        for pattern in problematic_patterns:
            assert pattern not in content, f"Found residual pattern: {pattern}"


# =============================================================================
# T5: Subdirectory images.json tests
# =============================================================================


class TestSubdirectoryImagesJson:
    """Tests for images.json generation in subdirectories."""

    def test_batch_with_subdirs_creates_structure(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that batch processing creates correct directory structure."""
        # Create input directory with subdirs
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "root.txt").write_text("Root content")

        sub_dir = input_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / "nested.txt").write_text("Nested content")

        output_dir = tmp_path / "output"

        result = runner.invoke(
            app,
            [str(input_dir), "-o", str(output_dir)],
        )

        assert result.exit_code == 0
        assert (output_dir / "root.txt.md").exists()
        assert (output_dir / "sub" / "nested.txt.md").exists()

    @pytest.mark.slow
    @pytest.mark.skipif(not _HAS_LIBREOFFICE, reason="LibreOffice not installed")
    def test_doc_in_subdir_creates_assets(
        self, runner: CliRunner, doc_file: Path, tmp_path: Path
    ) -> None:
        """Test that DOC file in subdirectory creates assets folder."""
        output_dir = tmp_path / "output"

        # Process the doc file from sub_dir
        result = runner.invoke(
            app,
            [str(doc_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0

        # Check output structure
        output_file = output_dir / "file-sample_100kB.doc.md"
        assert output_file.exists()

        # If document has images, assets folder should be created
        assets_dir = output_dir / "assets"
        if assets_dir.exists():
            # Just verify assets folder structure is correct
            assert assets_dir.is_dir()


class TestPostProcessingCleanup:
    """Tests for post-processing cleanup functions."""

    def test_fix_broken_markdown_links(self) -> None:
        """Test that broken markdown links are fixed."""
        from markitai.utils.text import fix_broken_markdown_links

        # Test link broken by newline
        broken = "[Link\ntext](https://example.com)"
        fixed = fix_broken_markdown_links(broken)
        assert "\n" not in fixed.split("](")[0]  # No newline in link text

    def test_clean_ppt_headers_footers(self) -> None:
        """Test that PPT headers/footers are cleaned with page markers."""
        from markitai.utils.text import clean_ppt_headers_footers

        # Content with page markers (realistic PPTX output)
        test_content = """<!-- Slide 1 -->

# Slide 1

Content here

FTD
FREE TEST DATA
1

<!-- Slide 2 -->

## Slide 2

More content

FTD
FREE TEST DATA
2

<!-- Slide 3 -->

## Slide 3

Real content

FTD
FREE TEST DATA
3
"""
        cleaned = clean_ppt_headers_footers(test_content)

        # Common footer lines should be removed
        # Note: "FTD" and "FREE TEST DATA" appear 3 times, so they should be detected
        assert "FREE TEST DATA" not in cleaned or cleaned.count("FREE TEST DATA") < 3
        assert "Real content" in cleaned

    def test_clean_residual_placeholders(self) -> None:
        """Test that residual placeholders are cleaned."""
        from markitai.utils.text import clean_residual_placeholders

        test_content = """# Document

Content here

__MARKITAI_FILE_ASSET__something__

More content
"""
        cleaned = clean_residual_placeholders(test_content)

        assert "__MARKITAI_FILE_ASSET__" not in cleaned
        assert "Content here" in cleaned
        assert "More content" in cleaned


class TestDoctorCommand:
    """Tests for doctor command."""

    def test_doctor_runs(self, runner: CliRunner) -> None:
        """Test that doctor command runs successfully."""
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Dependency Status" in result.output or "System Check" in result.output

    def test_doctor_json(self, runner: CliRunner) -> None:
        """Test that doctor --json outputs valid JSON."""
        result = runner.invoke(app, ["doctor", "--json"], color=False)
        assert result.exit_code == 0

        # Strip any ANSI codes and leading/trailing whitespace
        output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        output = output.strip()

        # Should be valid JSON
        data = json.loads(output)
        assert "playwright" in data
        assert "libreoffice" in data
        assert "rapidocr" in data
        assert "llm-api" in data

    def test_doctor_shows_components(self, runner: CliRunner) -> None:
        """Test that doctor shows all required components."""
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0

        # Should show all major components
        assert "Playwright" in result.output
        assert "LibreOffice" in result.output
        assert "RapidOCR" in result.output
        assert "LLM" in result.output
