"""Tests for batch processing module."""

import os
from pathlib import Path

import pytest

from markitai.batch import (
    BatchProcessor,
    BatchState,
    FileState,
    FileStatus,
    ProcessResult,
)
from markitai.config import BatchConfig


class TestFileState:
    """Tests for FileState dataclass."""

    def test_default_state(self) -> None:
        """Test default file state."""
        state = FileState(path="/path/to/file.pdf")

        assert state.status == FileStatus.PENDING
        assert state.output is None
        assert state.error is None

    def test_state_transitions(self) -> None:
        """Test state transitions."""
        state = FileState(path="/path/to/file.pdf")

        state.status = FileStatus.IN_PROGRESS
        assert state.status == FileStatus.IN_PROGRESS

        state.status = FileStatus.COMPLETED
        state.output = "/output/file.pdf.md"
        assert state.status == FileStatus.COMPLETED
        assert state.output is not None


class TestBatchState:
    """Tests for BatchState dataclass."""

    def test_empty_state(self) -> None:
        """Test empty batch state."""
        state = BatchState()

        assert state.total == 0
        assert state.completed_count == 0
        assert state.failed_count == 0
        assert state.pending_count == 0

    def test_state_with_files(self) -> None:
        """Test state with files."""
        state = BatchState()
        state.files = {
            "/path/file1.pdf": FileState(
                path="/path/file1.pdf", status=FileStatus.COMPLETED
            ),
            "/path/file2.pdf": FileState(
                path="/path/file2.pdf", status=FileStatus.FAILED
            ),
            "/path/file3.pdf": FileState(
                path="/path/file3.pdf", status=FileStatus.PENDING
            ),
        }

        assert state.total == 3
        assert state.completed_count == 1
        assert state.failed_count == 1
        assert state.pending_count == 2  # pending + failed

    def test_get_pending_files(self) -> None:
        """Test getting pending files."""
        state = BatchState()
        state.files = {
            "/path/file1.pdf": FileState(
                path="/path/file1.pdf", status=FileStatus.COMPLETED
            ),
            "/path/file2.pdf": FileState(
                path="/path/file2.pdf", status=FileStatus.PENDING
            ),
            "/path/file3.pdf": FileState(
                path="/path/file3.pdf", status=FileStatus.FAILED
            ),
        }

        pending = state.get_pending_files()

        assert len(pending) == 2
        assert Path("/path/file2.pdf") in pending
        assert Path("/path/file3.pdf") in pending

    def test_to_dict(self) -> None:
        """Test converting state to dictionary.

        Note: After refactoring, input_dir/output_dir are stored in options (not root),
        and files keys are relative paths. stats section is removed (merged into summary).
        """
        state = BatchState(
            version="1.0",
            started_at="2026-01-15T10:00:00Z",
            input_dir="/input",
            output_dir="/output",
            log_file="/nonexistent/path/markitai.log",  # Non-existent, kept as-is
        )
        state.files["/input/test.pdf"] = FileState(
            path="/input/test.pdf",
            status=FileStatus.COMPLETED,
            output="/output/test.pdf.md",
        )

        data = state.to_dict()

        assert data["version"] == "1.0"
        # input_dir is no longer in root level (only in options)
        assert "input_dir" not in data
        # log_file is kept as-is when file doesn't exist
        assert data["log_file"] == "/nonexistent/path/markitai.log"
        # stats is removed (merged into summary computed by processor)
        assert "stats" not in data
        # documents keys are now relative paths (renamed from 'files')
        assert "test.pdf" in data["documents"]
        assert data["documents"]["test.pdf"]["status"] == "completed"

    def test_from_dict(self, tmp_path: Path) -> None:
        """Test creating state from dictionary."""
        # Use real paths for cross-platform compatibility
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create a test file
        test_file = input_dir / "test.pdf"
        test_file.write_text("test")

        data = {
            "version": "1.0",
            "started_at": "2026-01-15T10:00:00Z",
            "updated_at": "2026-01-15T10:30:00Z",
            "log_file": str(tmp_path / "markitai.log"),
            "options": {
                "llm": True,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
            },
            "documents": {
                # Use relative path (used by to_dict)
                "test.pdf": {
                    "status": "completed",
                    "output": "test.pdf.md",
                }
            },
        }

        state = BatchState.from_dict(data)

        assert state.version == "1.0"
        assert state.input_dir == str(input_dir)
        assert len(state.files) == 1
        # from_dict reconstructs absolute path from relative path + input_dir
        expected_key = str(input_dir / "test.pdf")
        assert expected_key in state.files
        assert state.files[expected_key].status == FileStatus.COMPLETED


class TestFileStateSkipReason:
    """Tests for FileState skip_reason field."""

    def test_skip_reason_defaults_to_none(self) -> None:
        """FileState should have skip_reason=None by default."""
        state = FileState(path="/path/to/file.pdf")
        assert state.skip_reason is None

    def test_skip_reason_can_be_set(self) -> None:
        """FileState skip_reason can be set for skipped files."""
        state = FileState(path="/path/to/image.png")
        state.skip_reason = "image_only"
        assert state.skip_reason == "image_only"


class TestBatchStateSkippedCount:
    """Tests for BatchState skipped file counting."""

    def test_skipped_count_zero_when_no_skips(self) -> None:
        """skipped_count should be 0 when no files have skip_reason."""
        state = BatchState()
        state.files = {
            "/path/file1.pdf": FileState(
                path="/path/file1.pdf", status=FileStatus.COMPLETED
            ),
        }
        assert state.skipped_count == 0

    def test_skipped_count_counts_files_with_skip_reason(self) -> None:
        """skipped_count should count COMPLETED files that have a skip_reason."""
        state = BatchState()
        state.files = {
            "/path/file1.pdf": FileState(
                path="/path/file1.pdf", status=FileStatus.COMPLETED
            ),
            "/path/image.png": FileState(
                path="/path/image.png",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
            "/path/image2.jpg": FileState(
                path="/path/image2.jpg",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
        }
        assert state.skipped_count == 2

    def test_completed_count_includes_skipped(self) -> None:
        """completed_count should still include skipped files (backwards compat)."""
        state = BatchState()
        state.files = {
            "/path/file1.pdf": FileState(
                path="/path/file1.pdf", status=FileStatus.COMPLETED
            ),
            "/path/image.png": FileState(
                path="/path/image.png",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
        }
        assert state.completed_count == 2


class TestBatchProcessorSkipInProcessing:
    """Tests for skip_reason being stored during batch processing."""

    async def test_process_batch_stores_skip_reason_for_image_only(
        self, tmp_path: Path
    ) -> None:
        """process_batch should store skip_reason when ProcessResult has skipped error."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "image.png").write_text("fake")
        (input_dir / "doc.pdf").write_text("fake")

        output_dir = tmp_path / "output"
        config = BatchConfig()
        processor = BatchProcessor(config, output_dir)

        files = [input_dir / "image.png", input_dir / "doc.pdf"]

        async def mock_process(path: Path) -> ProcessResult:
            if path.suffix == ".png":
                return ProcessResult(success=True, error="skipped (image_only)")
            return ProcessResult(
                success=True,
                output_path=str(output_dir / f"{path.name}.md"),
            )

        state = await processor.process_batch(files, mock_process)

        # Find the image file state
        image_key = str(input_dir / "image.png")
        assert state.files[image_key].skip_reason == "image_only"
        assert state.files[image_key].status == FileStatus.COMPLETED

        # Doc should have no skip_reason
        doc_key = str(input_dir / "doc.pdf")
        assert state.files[doc_key].skip_reason is None

    async def test_process_batch_stores_skip_reason_for_exists(
        self, tmp_path: Path
    ) -> None:
        """process_batch should store skip_reason='exists' for already-existing output."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "doc.pdf").write_text("fake")

        output_dir = tmp_path / "output"
        config = BatchConfig()
        processor = BatchProcessor(config, output_dir)

        files = [input_dir / "doc.pdf"]

        async def mock_process(path: Path) -> ProcessResult:
            return ProcessResult(
                success=True,
                output_path=str(output_dir / f"{path.name}.md"),
                error="skipped (exists)",
            )

        state = await processor.process_batch(files, mock_process)

        doc_key = str(input_dir / "doc.pdf")
        assert state.files[doc_key].skip_reason == "exists"


class TestPrintSummaryWithSkips:
    """Tests for print_summary displaying skipped file information."""

    def test_summary_shows_skipped_count(self, tmp_path: Path, capsys) -> None:
        """print_summary should show skipped count separately from completed."""
        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path)
        processor.state = BatchState(
            started_at="2026-01-15T10:00:00Z",
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )
        # 3 files: 1 completed, 2 skipped
        processor.state.files = {
            "/path/doc.pdf": FileState(
                path="/path/doc.pdf", status=FileStatus.COMPLETED
            ),
            "/path/image.png": FileState(
                path="/path/image.png",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
            "/path/image2.jpg": FileState(
                path="/path/image2.jpg",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
        }

        processor.print_summary()

        captured = capsys.readouterr()
        # Should show "1/3 ✓" (only 1 truly completed) and mention 2 skipped
        assert "1/3" in captured.err
        assert "2 skipped" in captured.err

    def test_skip_warnings_show_two_example_filenames(
        self, tmp_path: Path, capsys
    ) -> None:
        """Skip warnings should list up to 2 example filenames."""
        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path)
        processor.state = BatchState(
            started_at="2026-01-15T10:00:00Z",
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )
        processor.state.files = {
            "/path/a.bmp": FileState(
                path="/path/a.bmp",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
            "/path/b.gif": FileState(
                path="/path/b.gif",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
            "/path/c.jpg": FileState(
                path="/path/c.jpg",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
            "/path/d.svg": FileState(
                path="/path/d.svg",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
            "/path/e.tiff": FileState(
                path="/path/e.tiff",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
        }

        processor.print_summary()

        captured = capsys.readouterr()
        # Should be ONE grouped line with 2 example names + "..."
        assert "5 files skipped (image_only)" in captured.err
        # Only 2 example filenames shown, plus "..."
        assert "..." in captured.err

    def test_skip_warnings_show_hint_for_image_only(
        self, tmp_path: Path, capsys
    ) -> None:
        """image_only skip warnings should suggest --llm or --ocr."""
        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path)
        processor.state = BatchState(
            started_at="2026-01-15T10:00:00Z",
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )
        processor.state.files = {
            "/path/a.png": FileState(
                path="/path/a.png",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
        }

        processor.print_summary()

        captured = capsys.readouterr()
        assert "--llm" in captured.err
        assert "--ocr" in captured.err

    def test_skip_warnings_many_files_no_individual_names(
        self, tmp_path: Path, capsys
    ) -> None:
        """Many skipped files should show count and 2 examples, not all names."""
        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path)
        processor.state = BatchState(
            started_at="2026-01-15T10:00:00Z",
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )
        processor.state.files = {
            f"/path/img{i:02d}.png": FileState(
                path=f"/path/img{i:02d}.png",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            )
            for i in range(10)
        }

        processor.print_summary()

        captured = capsys.readouterr()
        assert "10 files skipped (image_only)" in captured.err
        assert "..." in captured.err

    def test_skip_warnings_multiple_reasons_separate_lines(
        self, tmp_path: Path, capsys
    ) -> None:
        """Different skip reasons should each get their own grouped line."""
        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path)
        processor.state = BatchState(
            started_at="2026-01-15T10:00:00Z",
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )
        processor.state.files = {
            "/path/a.png": FileState(
                path="/path/a.png",
                status=FileStatus.COMPLETED,
                skip_reason="image_only",
            ),
            "/path/b.pdf": FileState(
                path="/path/b.pdf",
                status=FileStatus.COMPLETED,
                skip_reason="exists",
            ),
        }

        processor.print_summary()

        captured = capsys.readouterr()
        assert "1 file skipped (image_only)" in captured.err
        assert "1 file skipped (exists)" in captured.err


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_discover_files_respects_scan_limits(self, tmp_path: Path) -> None:
        """Test scan_max_depth and scan_max_files limits."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        nested_dir = input_dir / "level1" / "level2"
        nested_dir.mkdir(parents=True)

        file_root = input_dir / "root.pdf"
        file_nested = nested_dir / "nested.pdf"
        file_root.write_text("root")
        file_nested.write_text("nested")

        config = BatchConfig(scan_max_depth=1, scan_max_files=1)
        processor = BatchProcessor(config, tmp_path / "out", input_path=input_dir)

        files = processor.discover_files(input_dir, {".pdf"})

        assert len(files) == 1
        assert file_root in files

    def test_discover_files(self, tmp_path: Path) -> None:
        """Test file discovery."""
        # Create test files
        (tmp_path / "doc1.docx").touch()
        (tmp_path / "doc2.pdf").touch()
        (tmp_path / "doc3.txt").touch()
        (tmp_path / "other.xyz").touch()

        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path / "output")

        files = processor.discover_files(
            tmp_path,
            extensions={".docx", ".pdf", ".txt"},
        )

        assert len(files) == 3
        assert any(f.name == "doc1.docx" for f in files)
        assert any(f.name == "doc2.pdf" for f in files)
        assert any(f.name == "doc3.txt" for f in files)

    def test_discover_files_uppercase_extensions(self, tmp_path: Path) -> None:
        """Test that uppercase file extensions are discovered on case-sensitive systems."""
        # Create test files with mixed case extensions
        (tmp_path / "image1.jpg").touch()
        (tmp_path / "image2.JPG").touch()
        (tmp_path / "image3.JPEG").touch()
        (tmp_path / "document.PDF").touch()
        (tmp_path / "document2.pdf").touch()
        (tmp_path / "other.xyz").touch()

        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path / "output")

        files = processor.discover_files(
            tmp_path,
            extensions={".jpg", ".jpeg", ".pdf"},
        )

        assert len(files) == 5
        assert any(f.name == "image1.jpg" for f in files)
        assert any(f.name == "image2.JPG" for f in files)
        assert any(f.name == "image3.JPEG" for f in files)
        assert any(f.name == "document.PDF" for f in files)
        assert any(f.name == "document2.pdf" for f in files)

    def test_discover_files_nested_uppercase(self, tmp_path: Path) -> None:
        """Test uppercase extensions in nested directories."""
        nested = tmp_path / "subdir" / "nested"
        nested.mkdir(parents=True)

        (tmp_path / "root.JPG").touch()
        (nested / "deep.PNG").touch()
        (nested / "normal.png").touch()

        config = BatchConfig(scan_max_depth=5)
        processor = BatchProcessor(config, tmp_path / "output", input_path=tmp_path)

        files = processor.discover_files(
            tmp_path,
            extensions={".jpg", ".png"},
        )

        assert len(files) == 3
        assert any(f.name == "root.JPG" for f in files)
        assert any(f.name == "deep.PNG" for f in files)
        assert any(f.name == "normal.png" for f in files)

    def test_discover_files_applies_positive_globs(self, tmp_path: Path) -> None:
        """Test directory glob filters require a positive match when configured."""
        reports_dir = tmp_path / "reports"
        notes_dir = tmp_path / "notes"
        reports_dir.mkdir()
        notes_dir.mkdir()

        report = reports_dir / "quarterly.pdf"
        note = notes_dir / "meeting.pdf"
        report.touch()
        note.touch()

        config = BatchConfig(scan_max_depth=3)
        processor = BatchProcessor(config, tmp_path / "output", input_path=tmp_path)

        files = processor.discover_files(
            tmp_path,
            extensions={".pdf"},
            glob_patterns=["reports/**/*.pdf"],
        )

        assert files == [report]

    def test_discover_files_applies_negative_globs_last(self, tmp_path: Path) -> None:
        """Test exclusion globs win after inclusion globs match."""
        public_dir = tmp_path / "reports" / "public"
        private_dir = tmp_path / "reports" / "private"
        public_dir.mkdir(parents=True)
        private_dir.mkdir(parents=True)

        public_report = public_dir / "summary.pdf"
        private_report = private_dir / "summary.pdf"
        public_report.touch()
        private_report.touch()

        config = BatchConfig(scan_max_depth=4)
        processor = BatchProcessor(config, tmp_path / "output", input_path=tmp_path)

        files = processor.discover_files(
            tmp_path,
            extensions={".pdf"},
            glob_patterns=["reports/**/*.pdf", "!reports/private/**"],
        )

        assert files == [public_report]

    def test_discover_files_with_relative_input_path_and_globs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test glob filtering works when the input directory is passed relatively."""
        input_dir = tmp_path / "input"
        nested_dir = input_dir / "legacy"
        input_dir.mkdir()
        nested_dir.mkdir()

        sample_file = nested_dir / "sample.xls"
        sample_file.touch()

        monkeypatch.chdir(tmp_path)
        relative_input = Path("input")

        config = BatchConfig(scan_max_depth=3)
        processor = BatchProcessor(
            config, tmp_path / "output", input_path=relative_input
        )

        files = processor.discover_files(
            relative_input,
            extensions={".xls"},
            glob_patterns=["!**/*.org"],
        )

        assert files == [relative_input / "legacy" / "sample.xls"]

    def test_discover_files_globs_work_without_glob_translate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Glob filtering should not depend on glob.translate being available."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        report = reports_dir / "quarterly.pdf"
        report.touch()

        # Replace batch.py's module reference with a stub lacking translate —
        # deleting glob.translate itself would break pathlib on Python 3.13+
        import types

        monkeypatch.setattr("markitai.batch.glob_module", types.SimpleNamespace())

        config = BatchConfig(scan_max_depth=3)
        processor = BatchProcessor(config, tmp_path / "output", input_path=tmp_path)

        files = processor.discover_files(
            tmp_path,
            extensions={".pdf"},
            glob_patterns=["reports/**/*.pdf"],
        )

        assert files == [report]

    def test_discover_files_globs_support_question_and_char_classes(
        self, tmp_path: Path
    ) -> None:
        """Fallback glob matching should preserve basic glob semantics."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        alpha = reports_dir / "report-a.pdf"
        beta = reports_dir / "report-b.pdf"
        other = reports_dir / "report-aa.pdf"
        alpha.touch()
        beta.touch()
        other.touch()

        config = BatchConfig(scan_max_depth=3)
        processor = BatchProcessor(config, tmp_path / "output", input_path=tmp_path)

        files = processor.discover_files(
            tmp_path,
            extensions={".pdf"},
            glob_patterns=["reports/report-[ab].pdf", "!reports/report-?.pdf"],
        )

        assert files == []

    def test_discover_single_file(self, tmp_path: Path) -> None:
        """Test discovery of single file."""
        test_file = tmp_path / "test.docx"
        test_file.touch()

        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path / "output")

        files = processor.discover_files(test_file, extensions={".docx"})

        assert len(files) == 1
        assert files[0] == test_file

    def test_init_state(self, tmp_path: Path) -> None:
        """Test state initialization."""
        files = [
            tmp_path / "file1.pdf",
            tmp_path / "file2.pdf",
        ]
        for f in files:
            f.touch()

        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path / "output")

        state = processor.init_state(
            input_dir=tmp_path,
            files=files,
            options={"llm": True},
        )

        assert state.total == 2
        assert state.pending_count == 2
        assert str(tmp_path) == state.input_dir

    def test_init_state_with_log_file(self, tmp_path: Path) -> None:
        """Test state initialization with log file path."""
        files = [tmp_path / "file1.pdf"]
        files[0].touch()

        log_file = tmp_path / "logs" / "markitai_test.log"
        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path / "output", log_file=log_file)

        state = processor.init_state(
            input_dir=tmp_path,
            files=files,
            options={"llm": True},
        )

        assert state.log_file == str(log_file)

    def test_save_and_load_state(self, tmp_path: Path) -> None:
        """Test saving and loading state."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = BatchConfig()
        processor = BatchProcessor(config, output_dir)

        # Create and save state
        processor.state = BatchState(
            started_at="2026-01-15T10:00:00Z",
            input_dir=str(tmp_path),
            output_dir=str(output_dir),
        )
        processor.state.files["/test/file.pdf"] = FileState(
            path="/test/file.pdf",
            status=FileStatus.COMPLETED,
        )

        processor.save_state(force=True)

        # Load state
        loaded = processor.load_state()

        assert loaded is not None
        assert loaded.total == 1
        assert loaded.completed_count == 1

    async def test_process_batch(self, tmp_path: Path) -> None:
        """Test batch processing."""
        # Create test files
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.txt").write_text("content1")
        (input_dir / "file2.txt").write_text("content2")

        output_dir = tmp_path / "output"
        config = BatchConfig(concurrency=2)
        processor = BatchProcessor(config, output_dir)

        files = [input_dir / "file1.txt", input_dir / "file2.txt"]

        async def mock_process(path: Path) -> ProcessResult:
            return ProcessResult(
                success=True,
                output_path=str(output_dir / f"{path.name}.md"),
            )

        state = await processor.process_batch(files, mock_process)

        assert state.total == 2
        assert state.completed_count == 2
        assert state.failed_count == 0

    def test_progress_tracks_multiple_active_files(self, tmp_path: Path) -> None:
        """File progress should show multiple active items instead of last-writer-wins."""

        class FakeProgress:
            def __init__(self) -> None:
                self.updates: list[dict[str, str]] = []
                self.advanced: list[int] = []

            def update(self, task_id: int, **kwargs) -> None:
                self.updates.append(kwargs)

            def advance(self, task_id: int) -> None:
                self.advanced.append(task_id)

        processor = BatchProcessor(BatchConfig(), tmp_path / "output")
        processor._progress = FakeProgress()  # type: ignore[reportAttributeAccessIssue]
        processor._overall_task_id = 1  # type: ignore[reportAttributeAccessIssue]

        processor.set_current_file("alpha.txt")
        processor.set_current_file("beta.txt")

        assert processor._progress is not None
        current = processor._progress.updates[-1]["current"]  # type: ignore[reportAttributeAccessIssue]
        assert "alpha.txt" in current
        assert "beta.txt" in current

        processor.advance_progress(current_item="alpha.txt")

        current = processor._progress.updates[-1]["current"]  # type: ignore[reportAttributeAccessIssue]
        assert "beta.txt" in current
        assert "alpha.txt" not in current

    def test_progress_tracks_multiple_active_urls(self, tmp_path: Path) -> None:
        """URL progress should show all active domains, not just the last one."""

        class FakeProgress:
            def __init__(self) -> None:
                self.updates: list[dict[str, str]] = []
                self.advanced: list[int] = []

            def update(self, task_id: int, **kwargs) -> None:
                self.updates.append(kwargs)

            def advance(self, task_id: int) -> None:
                self.advanced.append(task_id)

        processor = BatchProcessor(BatchConfig(), tmp_path / "output")
        processor._progress = FakeProgress()  # type: ignore[reportAttributeAccessIssue]
        processor._url_task_id = 2  # type: ignore[reportAttributeAccessIssue]

        processor.update_url_status("https://a.example.com/path")
        processor.update_url_status("https://b.example.com/path")

        assert processor._progress is not None
        current = processor._progress.updates[-1]["current"]  # type: ignore[reportAttributeAccessIssue]
        assert "a.example.com" in current
        assert "b.example.com" in current

        processor.update_url_status("https://a.example.com/path", completed=True)

        current = processor._progress.updates[-1]["current"]  # type: ignore[reportAttributeAccessIssue]
        assert "b.example.com" in current
        assert "a.example.com" not in current

    async def test_process_batch_with_failures(self, tmp_path: Path) -> None:
        """Test batch processing with some failures."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "good.txt").write_text("content")
        (input_dir / "bad.txt").write_text("content")

        output_dir = tmp_path / "output"
        config = BatchConfig()
        processor = BatchProcessor(config, output_dir)

        files = [input_dir / "good.txt", input_dir / "bad.txt"]

        async def mock_process(path: Path) -> ProcessResult:
            if "bad" in path.name:
                return ProcessResult(success=False, error="Simulated error")
            return ProcessResult(success=True, output_path=str(path) + ".md")

        state = await processor.process_batch(files, mock_process)

        assert state.total == 2
        assert state.completed_count == 1
        assert state.failed_count == 1

    def test_generate_report(self, tmp_path: Path) -> None:
        """Test report generation."""
        config = BatchConfig()
        processor = BatchProcessor(config, tmp_path)

        processor.state = BatchState(
            started_at="2026-01-15T10:00:00Z",
            input_dir="/input",
            output_dir=str(tmp_path),
        )
        processor.state.files["/input/test.pdf"] = FileState(
            path="/input/test.pdf",
            status=FileStatus.COMPLETED,
            output="/output/test.pdf.md",
            duration=5.5,
            images=3,
            cost_usd=0.01,
        )

        report = processor.generate_report()

        assert "summary" in report
        # Summary fields use *_documents suffix for clarity
        assert report["summary"]["total_documents"] == 1
        assert report["summary"]["completed_documents"] == 1
        assert "documents" in report
        # documents keys are now relative paths (renamed from 'files')
        assert len(report["documents"]) == 1
        assert "test.pdf" in report["documents"]


class TestDiscoveryNeverFeedsOnItsOwnOutput:
    """discover_files prunes what markitai wrote and matches suffixes loosely."""

    @staticmethod
    def _tree(root: Path, *names: str) -> None:
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")

    def test_output_dir_inside_input_is_pruned(self, tmp_path: Path) -> None:
        """`markitai in -o in/out` run twice must not convert out/*.md."""
        inp = tmp_path / "in"
        self._tree(
            inp,
            "doc.txt",
            "out/doc.txt.md",
            "out/.markitai/assets/doc.txt.0001.jpg",
            "out/sub/x.pdf.md",
        )
        processor = BatchProcessor(BatchConfig(), inp / "out", input_path=inp)

        files = processor.discover_files(inp, {".txt", ".md", ".jpg", ".pdf"})

        assert files == [inp / "doc.txt"]

    def test_default_output_dir_under_cwd_is_pruned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`markitai init` (output.dir=./output) followed by `markitai .`."""
        self._tree(tmp_path, "a.docx", "output/a.docx.md")
        monkeypatch.chdir(tmp_path)
        processor = BatchProcessor(BatchConfig(), Path("output"), input_path=Path("."))

        files = processor.discover_files(Path("."), {".docx", ".md"})

        assert files == [Path("a.docx")]

    def test_meta_dir_is_always_skipped(self, tmp_path: Path) -> None:
        """.markitai/ holds assets/states/reports wherever it sits."""
        self._tree(
            tmp_path,
            "keep.png",
            ".markitai/assets/a.jpg",
            "nested/.markitai/screenshots/b.jpg",
            "nested/keep2.png",
        )
        processor = BatchProcessor(BatchConfig(), tmp_path, input_path=tmp_path)

        files = processor.discover_files(tmp_path, {".png", ".jpg"})

        assert files == [tmp_path / "keep.png", tmp_path / "nested" / "keep2.png"]

    def test_profile_assets_dir_of_the_output_tree_is_skipped(
        self, tmp_path: Path
    ) -> None:
        """--profile rag/obsidian writes visible assets/ next to outputs."""
        self._tree(tmp_path, "a.pdf", "assets/a.pdf-0001-01.jpg", "sub/assets/b.jpg")
        processor = BatchProcessor(BatchConfig(), tmp_path, input_path=tmp_path)

        with_profile = processor.discover_files(
            tmp_path, {".pdf", ".jpg"}, visible_assets=True
        )
        without_profile = processor.discover_files(tmp_path, {".pdf", ".jpg"})

        assert with_profile == [tmp_path / "a.pdf"]
        # Without an asset-visible profile an assets/ folder is ordinary input
        assert tmp_path / "assets" / "a.pdf-0001-01.jpg" in without_profile

    def test_input_assets_dir_outside_the_output_tree_is_kept(
        self, tmp_path: Path
    ) -> None:
        inp = tmp_path / "in"
        self._tree(inp, "assets/diagram.png")
        processor = BatchProcessor(BatchConfig(), tmp_path / "out", input_path=inp)

        files = processor.discover_files(inp, {".png"}, visible_assets=True)

        assert files == [inp / "assets" / "diagram.png"]

    def test_mixed_case_suffixes_match(self, tmp_path: Path) -> None:
        """Single-file mode lowercases the suffix; batch must too."""
        self._tree(tmp_path, "Mixed.Docx", "Report.Pdf", "skip.xyz")
        processor = BatchProcessor(BatchConfig(), tmp_path / "out")

        files = processor.discover_files(tmp_path, {".docx", ".pdf"})

        assert files == [tmp_path / "Mixed.Docx", tmp_path / "Report.Pdf"]

    def test_tree_is_walked_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One walk, not two per extension (39 extensions = 78 walks)."""
        import os

        import markitai.batch as batch_module

        self._tree(tmp_path, "a/b/c.pdf", "d.docx")
        calls: list[object] = []
        real_walk = os.walk

        def counting_walk(*args, **kwargs):
            calls.append(args[0])
            return real_walk(*args, **kwargs)

        monkeypatch.setattr(batch_module.os, "walk", counting_walk)
        processor = BatchProcessor(BatchConfig(), tmp_path / "out")

        files = processor.discover_files(
            tmp_path, {".pdf", ".docx", ".txt", ".md", ".png"}
        )

        assert len(calls) == 1
        assert files == [tmp_path / "a" / "b" / "c.pdf", tmp_path / "d.docx"]

    def test_scan_max_files_keeps_the_first_sorted_paths(self, tmp_path: Path) -> None:
        """Truncation must not depend on set hash order (PYTHONHASHSEED)."""
        self._tree(tmp_path, "z.pdf", "b.docx", "a.txt", "m/c.pdf", "c.png")
        processor = BatchProcessor(BatchConfig(scan_max_files=3), tmp_path / "out")

        files = processor.discover_files(tmp_path, {".pdf", ".docx", ".txt", ".png"})

        assert files == sorted(
            [tmp_path / "a.txt", tmp_path / "b.docx", tmp_path / "c.png"]
        )

    def test_scan_max_files_subset_is_stable_across_hash_seeds(
        self, tmp_path: Path
    ) -> None:
        import subprocess
        import sys

        for i in range(12):
            (tmp_path / f"f{i:02d}{['.pdf', '.docx', '.txt'][i % 3]}").write_text("x")
        script = (
            "import sys; from pathlib import Path;"
            "from markitai.batch import BatchProcessor;"
            "from markitai.config import BatchConfig;"
            "from markitai.converter.base import EXTENSION_MAP;"
            "p = BatchProcessor(BatchConfig(scan_max_files=4), Path(sys.argv[1]) / 'o');"
            "print([f.name for f in p.discover_files("
            "Path(sys.argv[1]), set(EXTENSION_MAP))])"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script, str(tmp_path)],
                env={**__import__("os").environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for seed in ("1", "2", "3")
        }
        assert len(outputs) == 1


class TestReservedTargetSurvivesTheState:
    """The output an unfinished item reserved is what --resume overwrites."""

    def test_target_round_trips_through_the_sidecar_and_base_file(
        self, tmp_path: Path
    ) -> None:
        from markitai.batch import UrlState

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        processor = BatchProcessor(BatchConfig(), tmp_path / "out")
        processor.state = processor.init_state(input_dir, [], {})
        done = FileState(
            path=str(input_dir / "done.pdf"),
            status=FileStatus.COMPLETED,
            output="out/done.pdf.llm.md",
            target="out/done.pdf.md",
        )
        busy = FileState(
            path=str(input_dir / "busy.pdf"),
            status=FileStatus.IN_PROGRESS,
            target="out/busy.pdf.v2.md",
        )
        processor.state.files = {done.path: done, busy.path: busy}
        processor.state.urls["https://x/y"] = UrlState(
            url="https://x/y",
            source_file="l.urls",
            status=FileStatus.FAILED,
            error="boom",
            target="out/y.md",
        )
        processor.save_state(force=True)

        # Stored absolute (a resume may run from another cwd)
        minimal = processor.state.to_minimal_dict()
        assert "target" not in minimal["documents"]["done.pdf"]
        assert minimal["documents"]["done.pdf"]["output"] == os.path.abspath(
            "out/done.pdf.llm.md"
        )
        assert minimal["documents"]["busy.pdf"]["target"] == os.path.abspath(
            "out/busy.pdf.v2.md"
        )
        assert minimal["urls"]["https://x/y"]["target"] == os.path.abspath("out/y.md")

        # A later incremental (sidecar) save changes the target
        busy.target = "out/busy.pdf.v3.md"
        processor._dirty_keys.add(busy.path)
        processor._last_state_save = None  # past the flush interval
        processor.save_state()
        assert processor.state_file.with_suffix(".jsonl").exists()

        loaded = processor.load_state()
        assert loaded is not None
        resumed = loaded.files[str(input_dir / "busy.pdf")]
        assert resumed.status == FileStatus.FAILED  # interrupted -> re-queued
        assert resumed.target == os.path.abspath("out/busy.pdf.v3.md")
        assert loaded.urls["https://x/y"].target == os.path.abspath("out/y.md")

    def test_paths_stay_valid_when_resumed_from_another_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative -o recorded in one cwd must not move with the next one.

        The state hash uses absolute paths, so ``--resume`` from another
        directory finds the same state; its cwd-relative targets/outputs
        then pointed the resumed items at ``<new cwd>/out``.
        """
        (tmp_path / "other").mkdir()
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        monkeypatch.chdir(tmp_path)
        processor = BatchProcessor(BatchConfig(), Path("out"), input_path=Path("in"))
        processor.state = processor.init_state(Path("in"), [], {})
        processor.state.files[str(input_dir / "a.pdf")] = FileState(
            path=str(input_dir / "a.pdf"),
            status=FileStatus.COMPLETED,
            output="out/a.pdf.md",
        )
        processor.state.files[str(input_dir / "b.pdf")] = FileState(
            path=str(input_dir / "b.pdf"),
            status=FileStatus.IN_PROGRESS,
            target="out/b.pdf.md",
        )
        processor.save_state(force=True)

        monkeypatch.chdir(tmp_path / "other")
        resumer = BatchProcessor(
            BatchConfig(), Path("../out"), input_path=Path("../in")
        )
        loaded = resumer.load_state()

        assert loaded is not None
        files = {Path(k).name: v for k, v in loaded.files.items()}
        assert files["a.pdf"].output == str(tmp_path / "out" / "a.pdf.md")
        assert files["b.pdf"].target == str(tmp_path / "out" / "b.pdf.md")

    def test_legacy_cwd_relative_paths_are_read_back_unchanged(
        self, tmp_path: Path
    ) -> None:
        """States written before paths were anchored keep their old values."""
        state = BatchState.from_dict(
            {
                "options": {"input_dir": str(tmp_path), "output_dir": "out"},
                "documents": {
                    "a.pdf": {"status": "completed", "output": "out/a.pdf.md"}
                },
                "urls": {},
            }
        )

        (file_state,) = state.files.values()
        assert file_state.output == "out/a.pdf.md"


class TestUrlStateKeys:
    """Each (url, output_name) of a URL list is its own resumable item."""

    def test_named_entries_get_their_own_key(self) -> None:
        from markitai.batch import url_state_key

        assert url_state_key("https://x/y") == "https://x/y"
        assert url_state_key("https://x/y", None) == "https://x/y"
        assert url_state_key("https://x/y", "first") == "https://x/y first"
        assert url_state_key("https://x/y", "first") != url_state_key(
            "https://x/y", "second"
        )

    def test_named_key_round_trips_with_its_url(self, tmp_path: Path) -> None:
        from markitai.batch import UrlState, url_state_key

        processor = BatchProcessor(BatchConfig(), tmp_path / "out")
        processor.state = processor.init_state(tmp_path, [], {})
        for name in ("first", "second"):
            key = url_state_key("https://x/y", name)
            processor.state.urls[key] = UrlState(
                url="https://x/y",
                source_file="l.urls",
                status=FileStatus.COMPLETED if name == "first" else FileStatus.FAILED,
                output=str(tmp_path / "out" / f"{name}.md"),
                error=None if name == "first" else "boom",
            )
        processor.save_state(force=True)

        loaded = processor.load_state()

        assert loaded is not None
        assert set(loaded.urls) == {"https://x/y first", "https://x/y second"}
        assert {u.url for u in loaded.urls.values()} == {"https://x/y"}
        assert loaded.urls["https://x/y first"].status == FileStatus.COMPLETED
        assert loaded.urls["https://x/y second"].status == FileStatus.FAILED

    def test_legacy_bare_url_state_is_adopted_by_a_named_entry(self) -> None:
        from markitai.batch import UrlState

        state = BatchState()
        state.urls["https://x/y"] = UrlState(
            url="https://x/y", source_file="l.urls", status=FileStatus.COMPLETED
        )

        state.adopt_legacy_url_keys(
            [
                ("https://x/y first", "https://x/y"),
                ("https://x/y second", "https://x/y"),
            ]
        )

        # The first named entry takes the old shared state; the other one is
        # new work (the old state never said it was done)
        assert set(state.urls) == {"https://x/y first"}
        assert state.urls["https://x/y first"].status == FileStatus.COMPLETED

    def test_bare_key_stays_with_an_unnamed_entry(self) -> None:
        from markitai.batch import UrlState

        state = BatchState()
        state.urls["https://x/y"] = UrlState(url="https://x/y", source_file="")

        state.adopt_legacy_url_keys(
            [("https://x/y", "https://x/y"), ("https://x/y named", "https://x/y")]
        )

        assert set(state.urls) == {"https://x/y"}
