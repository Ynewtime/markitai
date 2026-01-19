"""Unit tests for CLI helper functions."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path


class TestReportGeneration:
    """Tests for report generation in single file mode."""

    def test_report_file_created(self, tmp_path: Path) -> None:
        """Test that report file is created for single file conversion."""
        from click.testing import CliRunner

        from markit.cli import app

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
        report_path = output_dir / "reports" / "test.txt.report.json"
        assert report_path.exists()

    def test_report_structure(self, tmp_path: Path) -> None:
        """Test report structure."""
        from click.testing import CliRunner

        from markit.cli import app

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

        report_path = output_dir / "reports" / "test.txt.report.json"
        report = json.loads(report_path.read_text())

        # Check required fields
        assert "version" in report
        assert "generated_at" in report
        assert "summary" in report
        assert "llm_usage" in report
        assert "files" in report

        # Check summary fields
        summary = report["summary"]
        assert "total_files" in summary
        assert "completed" in summary
        assert "failed" in summary
        assert "duration_seconds" in summary

        # Check llm_usage
        assert "total_cost_usd" in report["llm_usage"]

    def test_report_file_details(self, tmp_path: Path) -> None:
        """Test file details in report."""
        from click.testing import CliRunner

        from markit.cli import app

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

        report_path = output_dir / "reports" / "test.txt.report.json"
        report = json.loads(report_path.read_text())

        assert len(report["files"]) == 1
        file_info = report["files"][0]

        assert file_info["input"] == "test.txt"
        assert file_info["status"] == "completed"
        assert file_info["error"] is None
        assert "duration_seconds" in file_info
        assert "llm_usage" in file_info
        assert "cost_usd" in file_info["llm_usage"]

    def test_report_no_overwrite(self, tmp_path: Path) -> None:
        """Test that reports are not overwritten."""
        from click.testing import CliRunner

        from markit.cli import app

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        output_dir = tmp_path / "output"

        runner = CliRunner()

        # First run
        runner.invoke(app, [str(test_file), "-o", str(output_dir)])
        report1 = output_dir / "reports" / "test.txt.report.json"
        assert report1.exists()

        # Second run
        runner.invoke(app, [str(test_file), "-o", str(output_dir)])
        report2 = output_dir / "reports" / "test.txt.report.1.json"
        assert report2.exists()
        assert report1.exists()  # First report still exists

        # Third run
        runner.invoke(app, [str(test_file), "-o", str(output_dir)])
        report3 = output_dir / "reports" / "test.txt.report.2.json"
        assert report3.exists()


class TestAssetDescriptions:
    """Tests for assets.desc.json merge behavior."""

    def test_write_assets_desc_json_merges_sources(self, tmp_path: Path) -> None:
        """Test assets.desc.json merges by source file."""
        from markit.cli import ImageAnalysisResult, write_assets_desc_json

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        first = ImageAnalysisResult(
            source_file="doc1.pdf",
            assets=[{"asset": "a.png", "alt": "A", "desc": "A"}],
        )
        second = ImageAnalysisResult(
            source_file="doc2.pdf",
            assets=[{"asset": "b.png", "alt": "B", "desc": "B"}],
        )

        report_path = write_assets_desc_json(output_dir, [first, second])
        assert report_path is not None
        data = json.loads(report_path.read_text())

        sources = {s["file"]: s for s in data["sources"]}
        assert "doc1.pdf" in sources
        assert "doc2.pdf" in sources

        # Update existing source
        updated = ImageAnalysisResult(
            source_file="doc1.pdf",
            assets=[{"asset": "a2.png", "alt": "A2", "desc": "A2"}],
        )
        report_path = write_assets_desc_json(output_dir, [updated])
        data = json.loads(report_path.read_text())
        sources = {s["file"]: s for s in data["sources"]}
        assert sources["doc1.pdf"]["assets"][0]["asset"] == "a2.png"


class TestBatchResumeDuration:
    """Tests for resume mode duration calculation."""

    async def test_resume_resets_started_at(self, tmp_path: Path) -> None:
        """Test that resume mode resets started_at for accurate duration."""
        from datetime import datetime

        from markit.batch import BatchProcessor, BatchState, FileState, FileStatus
        from markit.config import BatchConfig

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create old state file with old timestamp
        old_time = "2026-01-01T00:00:00+00:00"
        state = BatchState(
            started_at=old_time,
            updated_at=old_time,
            input_dir=str(tmp_path),
            output_dir=str(output_dir),
        )
        state.files[str(tmp_path / "file1.txt")] = FileState(
            path=str(tmp_path / "file1.txt"),
            status=FileStatus.COMPLETED,
        )
        state.files[str(tmp_path / "file2.txt")] = FileState(
            path=str(tmp_path / "file2.txt"),
            status=FileStatus.PENDING,
        )

        # Save state
        state_file = output_dir / ".markit-state.json"
        state_file.write_text(json.dumps(state.to_dict()))

        # Create processor and load state with resume
        config = BatchConfig()
        processor = BatchProcessor(config, output_dir)

        # Create the pending file
        (tmp_path / "file2.txt").write_text("content")

        # Process with resume
        async def mock_process(path: Path):
            from markit.batch import ProcessResult

            return ProcessResult(success=True, output_path=str(path) + ".md")

        before_process = datetime.now(UTC)
        await processor.process_batch(
            [tmp_path / "file2.txt"],
            mock_process,
            resume=True,
        )

        # Check that started_at was reset
        assert processor.state is not None
        started_at = datetime.fromisoformat(processor.state.started_at)
        # started_at should be close to when we started processing (within a few seconds)
        assert (started_at - before_process).total_seconds() < 5
