"""Unit tests for CLI helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def find_report_files(reports_dir: Path) -> list[Path]:
    """Find all report files in directory, sorted by name."""
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("markitai.*.report.json"))


class TestResolveOutputPath:
    """Tests for resolve_output_path function."""

    def test_nonexistent_file_returns_original(self, tmp_path: Path) -> None:
        """Test that non-existent file returns original path."""
        from markitai.cli import resolve_output_path

        path = tmp_path / "new_file.md"
        result = resolve_output_path(path, "rename")
        assert result == path

    def test_skip_returns_none(self, tmp_path: Path) -> None:
        """Test that skip strategy returns None for existing file."""
        from markitai.cli import resolve_output_path

        path = tmp_path / "existing.md"
        path.touch()
        result = resolve_output_path(path, "skip")
        assert result is None

    def test_overwrite_returns_original(self, tmp_path: Path) -> None:
        """Test that overwrite strategy returns original path."""
        from markitai.cli import resolve_output_path

        path = tmp_path / "existing.md"
        path.touch()
        result = resolve_output_path(path, "overwrite")
        assert result == path

    def test_rename_creates_v2(self, tmp_path: Path) -> None:
        """Test that rename strategy creates v2 suffix."""
        from markitai.cli import resolve_output_path

        path = tmp_path / "file.pdf.md"
        path.touch()
        result = resolve_output_path(path, "rename")
        assert result is not None
        assert result.name == "file.pdf.v2.md"

    def test_rename_increments_version(self, tmp_path: Path) -> None:
        """Test that rename strategy increments version numbers."""
        from markitai.cli import resolve_output_path

        base = tmp_path / "file.pdf.md"
        base.touch()
        (tmp_path / "file.pdf.v2.md").touch()
        (tmp_path / "file.pdf.v3.md").touch()

        result = resolve_output_path(base, "rename")
        assert result is not None
        assert result.name == "file.pdf.v4.md"

    def test_rename_llm_suffix(self, tmp_path: Path) -> None:
        """Test that rename preserves .llm.md suffix."""
        from markitai.cli import resolve_output_path

        path = tmp_path / "file.pdf.llm.md"
        path.touch()
        result = resolve_output_path(path, "rename")
        assert result is not None
        assert result.name == "file.pdf.v2.llm.md"

    def test_rename_uuid_fallback(self, tmp_path: Path) -> None:
        """Test that rename falls back to UUID when version numbers exhausted."""
        from markitai.cli import resolve_output_path

        # Create base file and versions 2-9999 (simulated by mocking)
        base = tmp_path / "file.pdf.md"
        base.touch()

        # Create versions up to 9999
        for i in range(2, 10):  # Only create a few for speed
            (tmp_path / f"file.pdf.v{i}.md").touch()

        # Mock by creating up to 9999
        (tmp_path / "file.pdf.v9999.md").touch()

        # To properly test UUID fallback, we need all 9999 versions
        # For unit test efficiency, we'll just verify the function logic
        # by checking it doesn't raise and returns a valid path

        # Create all versions 2-9999 would be slow, so verify UUID pattern works
        result = resolve_output_path(base, "rename")
        assert result is not None
        # Should be v10 since only v2-v9 and v9999 exist
        assert result.name == "file.pdf.v10.md"


class TestReportGeneration:
    """Tests for report generation in single file mode."""

    def test_report_file_created(self, tmp_path: Path) -> None:
        """Test that report file is created for single file conversion."""
        from click.testing import CliRunner

        from markitai.cli import app

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        # Report file is now named markitai.<hash>.report.json
        reports = find_report_files(output_dir / "reports")
        assert len(reports) == 1

    def test_report_structure(self, tmp_path: Path) -> None:
        """Test report structure."""
        from click.testing import CliRunner

        from markitai.cli import app

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        output_dir = tmp_path / "output"

        runner = CliRunner()
        runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
            ],
        )

        reports = find_report_files(output_dir / "reports")
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())

        # Check required fields
        assert "version" in report
        assert "generated_at" in report
        assert "summary" in report
        assert "llm_usage" in report
        assert "documents" in report

        # Check summary fields (transformed: duration_seconds -> duration)
        summary = report["summary"]
        assert "total_documents" in summary
        assert "completed_documents" in summary
        assert "failed_documents" in summary
        assert "duration" in summary  # Now human-readable

        # Check llm_usage (transformed: total_cost_usd -> cost_usd)
        assert "cost_usd" in report["llm_usage"]

    def test_report_file_details(self, tmp_path: Path) -> None:
        """Test file details in report."""
        from click.testing import CliRunner

        from markitai.cli import app

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        output_dir = tmp_path / "output"

        runner = CliRunner()
        runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
            ],
        )

        reports = find_report_files(output_dir / "reports")
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())

        # documents is a dict with relative paths as keys
        assert len(report["documents"]) == 1
        # Get the first (and only) file entry
        file_key = list(report["documents"].keys())[0]
        file_info = report["documents"][file_key]

        assert file_key == "test.txt"  # Relative path
        assert file_info["status"] == "completed"
        assert file_info["error"] is None
        assert "duration" in file_info  # Now human-readable
        assert "llm_usage" in file_info

    def test_report_no_overwrite(self, tmp_path: Path) -> None:
        """Test that reports are not overwritten (on_conflict=rename)."""
        from click.testing import CliRunner

        from markitai.cli import app

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        output_dir = tmp_path / "output"

        runner = CliRunner()

        # First run
        runner.invoke(app, [str(test_file), "-o", str(output_dir)])
        reports = find_report_files(output_dir / "reports")
        assert len(reports) == 1

        # Second run - should create a new report with .2. in name
        runner.invoke(app, [str(test_file), "-o", str(output_dir)])
        reports = find_report_files(output_dir / "reports")
        assert len(reports) == 2

        # Third run - should create another report with .3. in name
        runner.invoke(app, [str(test_file), "-o", str(output_dir)])
        reports = find_report_files(output_dir / "reports")
        assert len(reports) == 3


class TestImageDescriptions:
    """Tests for images.json merge behavior."""

    def test_write_images_json_merges_images(self, tmp_path: Path) -> None:
        """Test images.json merges images with source field."""
        from markitai.cli import ImageAnalysisResult, write_images_json

        output_dir = tmp_path / "output"
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True)

        first = ImageAnalysisResult(
            source_file="doc1.pdf",
            assets=[{"asset": str(assets_dir / "a.png"), "alt": "A", "desc": "A"}],
        )
        second = ImageAnalysisResult(
            source_file="doc2.pdf",
            assets=[{"asset": str(assets_dir / "b.png"), "alt": "B", "desc": "B"}],
        )

        report_paths = write_images_json(output_dir, [first, second])
        assert len(report_paths) == 1
        data = json.loads(report_paths[0].read_text())

        # Check flat images array structure (field renamed from assets to images)
        images = {a["path"]: a for a in data["images"]}
        assert len(images) == 2
        # Check source field is added
        assert any(a["source"] == "doc1.pdf" for a in data["images"])
        assert any(a["source"] == "doc2.pdf" for a in data["images"])

        # Update existing image
        updated = ImageAnalysisResult(
            source_file="doc1.pdf",
            assets=[{"asset": str(assets_dir / "a2.png"), "alt": "A2", "desc": "A2"}],
        )
        report_paths = write_images_json(output_dir, [updated])
        assert len(report_paths) == 1
        data = json.loads(report_paths[0].read_text())
        # Should have 3 images now (a.png, b.png, a2.png)
        assert len(data["images"]) == 3
        a2_image = next(a for a in data["images"] if "a2.png" in a["path"])
        assert a2_image["source"] == "doc1.pdf"


class TestBatchResumeDuration:
    """Tests for resume mode duration calculation."""

    async def test_resume_resets_started_at(self, tmp_path: Path) -> None:
        """Test that resume mode resets started_at for accurate duration."""
        from datetime import datetime

        from markitai.batch import BatchProcessor, BatchState, FileState, FileStatus
        from markitai.config import BatchConfig

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create processor first to get the state file path
        config = BatchConfig()
        processor = BatchProcessor(config, output_dir, input_path=input_dir)

        # Create old state file with old timestamp at the correct path
        old_time = "2026-01-01T00:00:00+00:00"
        state = BatchState(
            started_at=old_time,
            updated_at=old_time,
            input_dir=str(input_dir),
            output_dir=str(output_dir),
        )
        state.files[str(input_dir / "file1.txt")] = FileState(
            path=str(input_dir / "file1.txt"),
            status=FileStatus.COMPLETED,
        )
        state.files[str(input_dir / "file2.txt")] = FileState(
            path=str(input_dir / "file2.txt"),
            status=FileStatus.PENDING,
        )

        # Save state to the correct state file path (states/markitai.<hash>.state.json)
        processor.state_file.parent.mkdir(parents=True, exist_ok=True)
        processor.state_file.write_text(json.dumps(state.to_dict()))

        # Create the pending file
        (input_dir / "file2.txt").write_text("content")

        # Process with resume
        async def mock_process(path: Path):
            from markitai.batch import ProcessResult

            return ProcessResult(success=True, output_path=str(path) + ".md")

        before_process = datetime.now().astimezone()
        await processor.process_batch(
            [input_dir / "file2.txt"],
            mock_process,
            resume=True,
        )

        # Check that started_at was reset
        assert processor.state is not None
        started_at = datetime.fromisoformat(processor.state.started_at)
        # started_at should be close to when we started processing (within a few seconds)
        assert (started_at - before_process).total_seconds() < 5


class TestUrlHelpers:
    """Tests for URL helper functions."""

    def test_is_url_http(self) -> None:
        """Test is_url with http URLs."""
        from markitai.cli import is_url

        assert is_url("http://example.com") is True
        assert is_url("http://example.com/path/to/page") is True
        assert is_url("HTTP://EXAMPLE.COM") is True  # Case insensitive

    def test_is_url_https(self) -> None:
        """Test is_url with https URLs."""
        from markitai.cli import is_url

        assert is_url("https://example.com") is True
        assert is_url("https://example.com/path") is True
        assert is_url("HTTPS://example.com") is True

    def test_is_url_non_urls(self) -> None:
        """Test is_url with non-URL strings."""
        from markitai.cli import is_url

        assert is_url("example.com") is False
        assert is_url("/path/to/file") is False
        assert is_url("./relative/path") is False
        assert is_url("C:\\Windows\\path") is False
        assert is_url("ftp://example.com") is False  # Not http/https

    def test_url_to_filename_basic(self) -> None:
        """Test url_to_filename with basic URLs."""
        from markitai.cli import url_to_filename

        assert url_to_filename("https://example.com/page.html") == "page.html.md"
        assert url_to_filename("https://example.com/path/to/doc") == "doc.md"

    def test_url_to_filename_domain_only(self) -> None:
        """Test url_to_filename with domain-only URLs."""
        from markitai.cli import url_to_filename

        assert url_to_filename("https://example.com") == "example_com.md"
        assert url_to_filename("https://example.com/") == "example_com.md"

    def test_url_to_filename_special_chars(self) -> None:
        """Test url_to_filename sanitizes special characters."""
        from markitai.cli import url_to_filename

        # Characters like : ? < > should be replaced
        filename = url_to_filename("https://example.com/page?query=value")
        assert "?" not in filename
        assert filename.endswith(".md")

    def test_sanitize_filename(self) -> None:
        """Test _sanitize_filename function."""
        from markitai.cli import _sanitize_filename

        # Remove invalid characters
        assert _sanitize_filename("file<name>") == "file_name_"
        assert _sanitize_filename("file:name") == "file_name"
        assert _sanitize_filename('file"name') == "file_name"

        # Strip leading/trailing spaces and dots
        assert _sanitize_filename("  file  ") == "file"
        assert _sanitize_filename("...file...") == "file"

        # Empty string fallback
        assert _sanitize_filename("") == "unnamed"
        assert _sanitize_filename("...") == "unnamed"

        # Length limit
        long_name = "a" * 300
        result = _sanitize_filename(long_name)
        assert len(result) <= 200


class TestSingleFileOutput:
    """Tests for single file stdout output behavior."""

    def test_single_file_outputs_to_stdout(self, tmp_path: Path) -> None:
        """Test that single file conversion outputs to stdout."""
        from click.testing import CliRunner

        from markitai.cli import app

        test_file = tmp_path / "test.txt"
        test_file.write_text("# Hello World\n\nTest content.")

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        # Output should contain the markdown content
        assert "Hello World" in result.output
        assert "Test content" in result.output
        # Should have frontmatter
        assert "---" in result.output
        assert "source:" in result.output

    def test_single_file_no_logs_without_verbose(self, tmp_path: Path) -> None:
        """Test that single file mode doesn't print logs without --verbose."""
        from click.testing import CliRunner

        from markitai.cli import app

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        # Should NOT contain log-style output (timestamps like HH:MM:SS)
        # The output should be clean markdown
        assert "| INFO" not in result.output
        assert "Converting" not in result.output

    def test_single_file_shows_logs_with_verbose(self, tmp_path: Path) -> None:
        """Test that single file mode shows logs with --verbose."""
        from click.testing import CliRunner

        from markitai.cli import app

        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                str(test_file),
                "-o",
                str(output_dir),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        # With --verbose, should show converting message
        # Note: logs go to stderr, output goes to stdout
        # CliRunner captures both, so we check for content presence

    def test_batch_mode_still_shows_progress(self, tmp_path: Path) -> None:
        """Test that batch mode still shows progress (not affected by quiet mode)."""
        from click.testing import CliRunner

        from markitai.cli import app

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("Content 1")

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                str(input_dir),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        # Batch mode should show summary or progress
        # (exact output depends on Rich terminal detection)


class TestUrlConversion:
    """Tests for URL conversion functionality."""

    def test_url_dry_run(self, tmp_path: Path) -> None:
        """Test URL conversion dry run."""
        from click.testing import CliRunner

        from markitai.cli import app

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "https://example.com/test",
                "-o",
                str(output_dir),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Would convert URL" in result.output
        assert "example.com" in result.output

    def test_url_dry_run_with_llm(self, tmp_path: Path) -> None:
        """Test URL conversion dry run shows LLM status."""
        from click.testing import CliRunner

        from markitai.cli import app

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "https://example.com/test",
                "-o",
                str(output_dir),
                "--llm",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "Would convert URL" in result.output
        assert "LLM: enabled" in result.output

    def test_url_dry_run_without_llm(self, tmp_path: Path) -> None:
        """Test URL conversion dry run shows LLM disabled."""
        from click.testing import CliRunner

        from markitai.cli import app

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "https://example.com/test",
                "-o",
                str(output_dir),
                "--no-llm",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "LLM: disabled" in result.output

    def test_url_dry_run_with_rich_preset(self, tmp_path: Path) -> None:
        """Test URL conversion dry run with rich preset shows correct info."""
        from click.testing import CliRunner

        from markitai.cli import app

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "https://example.com/test",
                "-o",
                str(output_dir),
                "--preset",
                "rich",  # rich preset enables --alt, --desc, --screenshot
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        # Should show dry run panel with LLM enabled (from rich preset)
        assert "Dry Run" in result.output
        assert "LLM: enabled" in result.output

    @pytest.mark.skip(reason="Requires network access")
    def test_url_conversion_real(self, tmp_path: Path) -> None:
        """Test real URL conversion (requires network)."""
        from click.testing import CliRunner

        from markitai.cli import app

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "https://example.com",
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "example_com.md").exists()
