"""Unit tests for CLI helper functions."""

from __future__ import annotations

import json
from pathlib import Path


def find_report_files(reports_dir: Path) -> list[Path]:
    """Find all report files in directory, sorted by name."""
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.glob("markit.*.report.json"))


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
        # Report file is now named markit.<hash>.report.json
        reports = find_report_files(output_dir / "reports")
        assert len(reports) == 1

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

        reports = find_report_files(output_dir / "reports")
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())

        # Check required fields
        assert "version" in report
        assert "generated_at" in report
        assert "summary" in report
        assert "llm_usage" in report
        assert "files" in report

        # Check summary fields (total_files renamed to total)
        summary = report["summary"]
        assert "total" in summary
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

        reports = find_report_files(output_dir / "reports")
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())

        # files is now a dict with relative paths as keys
        assert len(report["files"]) == 1
        # Get the first (and only) file entry
        file_key = list(report["files"].keys())[0]
        file_info = report["files"][file_key]

        assert file_key == "test.txt"  # Relative path
        assert file_info["status"] == "completed"
        assert file_info["error"] is None
        assert "duration_seconds" in file_info
        assert "llm_usage" in file_info

    def test_report_no_overwrite(self, tmp_path: Path) -> None:
        """Test that reports are not overwritten (on_conflict=rename)."""
        from click.testing import CliRunner

        from markit.cli import app

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

        # Save state to the correct state file path (states/markit.<hash>.state.json)
        processor.state_file.parent.mkdir(parents=True, exist_ok=True)
        processor.state_file.write_text(json.dumps(state.to_dict()))

        # Create the pending file
        (input_dir / "file2.txt").write_text("content")

        # Process with resume
        async def mock_process(path: Path):
            from markit.batch import ProcessResult

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
