"""Tests for batch processing module."""

from pathlib import Path

from markit.batch import (
    BatchProcessor,
    BatchState,
    FileState,
    FileStatus,
    ProcessResult,
)
from markit.config import BatchConfig


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
        """Test converting state to dictionary."""
        state = BatchState(
            version="1.0",
            started_at="2026-01-15T10:00:00Z",
            input_dir="/input",
            output_dir="/output",
        )
        state.files["/input/test.pdf"] = FileState(
            path="/input/test.pdf",
            status=FileStatus.COMPLETED,
            output="/output/test.pdf.md",
        )

        data = state.to_dict()

        assert data["version"] == "1.0"
        assert data["input_dir"] == "/input"
        assert "stats" in data
        assert data["stats"]["total"] == 1

    def test_from_dict(self) -> None:
        """Test creating state from dictionary."""
        data = {
            "version": "1.0",
            "started_at": "2026-01-15T10:00:00Z",
            "updated_at": "2026-01-15T10:30:00Z",
            "input_dir": "/input",
            "output_dir": "/output",
            "options": {"llm": True},
            "files": {
                "/input/test.pdf": {
                    "status": "completed",
                    "output": "/output/test.pdf.md",
                }
            },
        }

        state = BatchState.from_dict(data)

        assert state.version == "1.0"
        assert state.input_dir == "/input"
        assert len(state.files) == 1
        assert state.files["/input/test.pdf"].status == FileStatus.COMPLETED


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

        processor.save_state()

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
            duration_seconds=5.5,
            images_extracted=3,
            llm_cost_usd=0.01,
        )

        report = processor.generate_report()

        assert "summary" in report
        assert report["summary"]["total_files"] == 1
        assert report["summary"]["completed"] == 1
        assert "files" in report
        assert len(report["files"]) == 1
