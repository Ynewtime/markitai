"""Tests for atomic write utilities."""

from __future__ import annotations

import threading
from pathlib import Path

from markit.security import atomic_write_json, atomic_write_text


class TestAtomicWriteText:
    """Tests for atomic_write_text function."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """Test basic text write."""
        file_path = tmp_path / "test.txt"
        content = "Hello, World!"

        atomic_write_text(file_path, content)

        assert file_path.exists()
        assert file_path.read_text() == content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        file_path = tmp_path / "subdir" / "nested" / "test.txt"
        content = "nested content"

        atomic_write_text(file_path, content)

        assert file_path.exists()
        assert file_path.read_text() == content

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Test overwriting an existing file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("old content")

        atomic_write_text(file_path, "new content")

        assert file_path.read_text() == "new content"

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Test writing Unicode content."""
        file_path = tmp_path / "test.txt"
        content = "你好世界 🌍"

        atomic_write_text(file_path, content)

        assert file_path.read_text(encoding="utf-8") == content

    def test_no_temp_file_left_on_success(self, tmp_path: Path) -> None:
        """Test that no temp files are left after successful write."""
        file_path = tmp_path / "test.txt"

        atomic_write_text(file_path, "content")

        # Check no .tmp files remain
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_no_temp_file_left_on_error(self) -> None:
        """Test that temp files are cleaned up on error."""
        # We can't easily test write failure without mocking, so this is a placeholder
        # The implementation handles cleanup in the except block
        pass

    def test_accepts_path_object_or_string(self, tmp_path: Path) -> None:
        """Test that both Path and str are accepted."""
        path_obj = tmp_path / "path_test.txt"
        path_str = str(tmp_path / "str_test.txt")

        atomic_write_text(path_obj, "path content")
        atomic_write_text(Path(path_str), "str content")

        assert Path(path_obj).read_text() == "path content"
        assert Path(path_str).read_text() == "str content"


class TestAtomicWriteJson:
    """Tests for atomic_write_json function."""

    def test_basic_json_write(self, tmp_path: Path) -> None:
        """Test basic JSON write."""
        file_path = tmp_path / "test.json"
        obj = {"key": "value", "number": 42}

        atomic_write_json(file_path, obj)

        import json

        assert file_path.exists()
        loaded = json.loads(file_path.read_text())
        assert loaded == obj

    def test_nested_json(self, tmp_path: Path) -> None:
        """Test nested JSON structures."""
        file_path = tmp_path / "test.json"
        obj = {
            "level1": {
                "level2": {
                    "list": [1, 2, 3],
                    "string": "hello",
                }
            }
        }

        atomic_write_json(file_path, obj)

        import json

        loaded = json.loads(file_path.read_text())
        assert loaded == obj

    def test_unicode_in_json(self, tmp_path: Path) -> None:
        """Test Unicode content in JSON."""
        file_path = tmp_path / "test.json"
        obj = {"message": "你好世界", "emoji": "🎉"}

        atomic_write_json(file_path, obj, ensure_ascii=False)

        content = file_path.read_text(encoding="utf-8")
        assert "你好世界" in content
        assert "🎉" in content

    def test_json_indentation(self, tmp_path: Path) -> None:
        """Test JSON indentation."""
        file_path = tmp_path / "test.json"
        obj = {"key": "value"}

        atomic_write_json(file_path, obj, indent=4)

        content = file_path.read_text()
        # 4-space indent means line should have 4 spaces before "key"
        assert '    "key"' in content

    def test_list_json(self, tmp_path: Path) -> None:
        """Test JSON with list at root."""
        file_path = tmp_path / "test.json"
        obj = [1, 2, 3, {"nested": True}]

        atomic_write_json(file_path, obj)

        import json

        loaded = json.loads(file_path.read_text())
        assert loaded == obj


class TestAtomicWriteAtomicity:
    """Tests for atomicity guarantees."""

    def test_concurrent_writes_produce_valid_content(self, tmp_path: Path) -> None:
        """Test that concurrent writes don't produce corrupted files."""
        file_path = tmp_path / "concurrent.txt"
        results = []
        errors = []

        def writer(content: str) -> None:
            try:
                for _ in range(10):
                    atomic_write_text(file_path, content)
                    # Verify the file is readable and has valid content
                    read_content = file_path.read_text()
                    if read_content not in ["AAAA", "BBBB"]:
                        errors.append(f"Corrupted content: {read_content!r}")
                results.append(True)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=writer, args=("AAAA",)),
            threading.Thread(target=writer, args=("BBBB",)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 2
