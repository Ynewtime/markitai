"""Unit tests for security utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from markitai.constants import MAX_DOCUMENT_SIZE, MAX_STATE_FILE_SIZE
from markitai.security import (
    atomic_write_json,
    atomic_write_text,
    check_symlink_safety,
    escape_glob_pattern,
    sanitize_error_message,
    validate_file_size,
    validate_path_within_base,
)


class TestAtomicWriteText:
    """Tests for atomic_write_text function."""

    def test_write_basic_content(self, tmp_path: Path) -> None:
        """Test basic text write."""
        test_file = tmp_path / "test.txt"
        content = "Hello, World!"

        atomic_write_text(test_file, content)

        assert test_file.exists()
        assert test_file.read_text() == content

    def test_write_unicode_content(self, tmp_path: Path) -> None:
        """Test writing unicode content."""
        test_file = tmp_path / "unicode.txt"
        content = "Hello, 世界! 🌍 Привет мир"

        atomic_write_text(test_file, content)

        assert test_file.read_text(encoding="utf-8") == content

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        test_file = tmp_path / "a" / "b" / "c" / "test.txt"

        atomic_write_text(test_file, "content")

        assert test_file.exists()
        assert test_file.read_text() == "content"

    def test_write_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that existing file is overwritten."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("old content")

        atomic_write_text(test_file, "new content")

        assert test_file.read_text() == "new content"

    def test_write_custom_encoding(self, tmp_path: Path) -> None:
        """Test writing with custom encoding."""
        test_file = tmp_path / "latin1.txt"
        content = "café"

        atomic_write_text(test_file, content, encoding="latin-1")

        assert test_file.read_text(encoding="latin-1") == content

    def test_write_cleans_temp_on_error(self, tmp_path: Path) -> None:
        """Test that temp file is cleaned up on write error."""
        test_file = tmp_path / "test.txt"

        # Mock os.fdopen to raise an exception
        with (
            patch("os.fdopen", side_effect=OSError("Simulated write error")),
            pytest.raises(IOError, match="Simulated write error"),
        ):
            atomic_write_text(test_file, "content")

        # Verify no temp files left behind
        temp_files = list(tmp_path.glob(".*tmp*"))
        assert len(temp_files) == 0

    def test_write_empty_content(self, tmp_path: Path) -> None:
        """Test writing empty content."""
        test_file = tmp_path / "empty.txt"

        atomic_write_text(test_file, "")

        assert test_file.exists()
        assert test_file.read_text() == ""

    def test_write_large_content(self, tmp_path: Path) -> None:
        """Test writing large content."""
        test_file = tmp_path / "large.txt"
        content = "x" * 1024 * 1024  # 1 MB

        atomic_write_text(test_file, content)

        assert test_file.read_text() == content

    def test_write_multiline_content(self, tmp_path: Path) -> None:
        """Test writing multiline content preserves line endings."""
        test_file = tmp_path / "multiline.txt"
        content = "line1\nline2\nline3\n"

        atomic_write_text(test_file, content)

        assert test_file.read_text() == content


class TestAtomicWriteJson:
    """Tests for atomic_write_json function."""

    def test_write_basic_json(self, tmp_path: Path) -> None:
        """Test basic JSON write."""
        test_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        atomic_write_json(test_file, data)

        import json

        assert json.loads(test_file.read_text()) == data

    def test_write_json_with_unicode(self, tmp_path: Path) -> None:
        """Test JSON with unicode characters."""
        test_file = tmp_path / "unicode.json"
        data = {"message": "你好世界", "emoji": "🎉"}

        atomic_write_json(test_file, data, ensure_ascii=False)

        content = test_file.read_text(encoding="utf-8")
        assert "你好世界" in content
        assert "🎉" in content

    def test_write_json_with_indent(self, tmp_path: Path) -> None:
        """Test JSON with custom indentation."""
        test_file = tmp_path / "indented.json"
        data = {"key": "value"}

        atomic_write_json(test_file, data, indent=4)

        content = test_file.read_text()
        assert "    " in content  # 4-space indent

    def test_write_json_with_order_func(self, tmp_path: Path) -> None:
        """Test JSON with custom ordering function."""
        test_file = tmp_path / "ordered.json"
        data = {"z": 1, "a": 2, "m": 3}

        def sort_keys(d: dict) -> dict:
            return dict(sorted(d.items()))

        atomic_write_json(test_file, data, order_func=sort_keys)

        import json

        content = test_file.read_text()
        result = json.loads(content)
        assert list(result.keys()) == ["a", "m", "z"]

    def test_write_json_list(self, tmp_path: Path) -> None:
        """Test writing JSON list (order_func ignored for non-dict)."""
        test_file = tmp_path / "list.json"
        data = [1, 2, 3, "four"]

        atomic_write_json(test_file, data)

        import json

        assert json.loads(test_file.read_text()) == data

    def test_write_json_nested(self, tmp_path: Path) -> None:
        """Test writing nested JSON structures."""
        test_file = tmp_path / "nested.json"
        data = {
            "level1": {"level2": {"level3": {"value": 42}}},
            "list": [1, {"nested": True}],
        }

        atomic_write_json(test_file, data)

        import json

        assert json.loads(test_file.read_text()) == data


class TestEscapeGlobPattern:
    """Tests for escape_glob_pattern function."""

    def test_escape_asterisk(self) -> None:
        """Test escaping asterisk."""
        assert escape_glob_pattern("file*.txt") == "file[*].txt"

    def test_escape_question_mark(self) -> None:
        """Test escaping question mark."""
        assert escape_glob_pattern("file?.txt") == "file[?].txt"

    def test_escape_brackets(self) -> None:
        """Test escaping square brackets."""
        # Only [ and ] are escaped, - is not a glob special char
        assert escape_glob_pattern("file[0-9].txt") == "file[[]0-9[]].txt"

    def test_escape_multiple_chars(self) -> None:
        """Test escaping multiple special characters."""
        result = escape_glob_pattern("*?[test]*")
        assert "[*]" in result
        assert "[?]" in result
        assert "[[]" in result

    def test_no_special_chars(self) -> None:
        """Test string without special characters."""
        assert escape_glob_pattern("normal_file.txt") == "normal_file.txt"

    def test_empty_string(self) -> None:
        """Test empty string."""
        assert escape_glob_pattern("") == ""


class TestValidatePathWithinBase:
    """Tests for validate_path_within_base function."""

    def test_valid_path(self, tmp_path: Path) -> None:
        """Test valid path within base."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = subdir / "test.txt"
        test_file.touch()

        result = validate_path_within_base(test_file, tmp_path)
        assert result == test_file.resolve()

    def test_path_traversal_attack(self, tmp_path: Path) -> None:
        """Test path traversal is detected."""
        malicious_path = tmp_path / ".." / ".." / "etc" / "passwd"

        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path_within_base(malicious_path, tmp_path)

    def test_absolute_path_outside_base(self, tmp_path: Path) -> None:
        """Test absolute path outside base is rejected."""
        outside_path = Path("/tmp/outside")

        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_path_within_base(outside_path, tmp_path)

    def test_relative_path_within_base(self, tmp_path: Path) -> None:
        """Test relative path resolved correctly."""
        os.chdir(tmp_path)
        test_file = tmp_path / "test.txt"
        test_file.touch()

        result = validate_path_within_base(Path("test.txt"), tmp_path)
        assert result == test_file.resolve()


class TestCheckSymlinkSafety:
    """Tests for check_symlink_safety function."""

    def test_regular_file(self, tmp_path: Path) -> None:
        """Test regular file passes check."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        # Should not raise
        check_symlink_safety(test_file, allow_symlinks=False)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlink creation requires admin privileges on Windows",
    )
    def test_symlink_not_allowed(self, tmp_path: Path) -> None:
        """Test symlink raises when not allowed."""
        target = tmp_path / "target.txt"
        target.touch()
        link = tmp_path / "link.txt"
        link.symlink_to(target)

        with pytest.raises(ValueError, match="Symlink not allowed"):
            check_symlink_safety(link, allow_symlinks=False)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlink creation requires admin privileges on Windows",
    )
    def test_symlink_allowed(self, tmp_path: Path) -> None:
        """Test symlink allowed when flag set."""
        target = tmp_path / "target.txt"
        target.touch()
        link = tmp_path / "link.txt"
        link.symlink_to(target)

        # Should not raise, just warn
        check_symlink_safety(link, allow_symlinks=True)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlink creation requires admin privileges on Windows",
    )
    def test_symlink_directory_not_allowed(self, tmp_path: Path) -> None:
        """Test symlink directory itself is detected."""
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()

        # Create symlink directory: tmp_path/link_dir -> tmp_path/real_dir
        link_dir = tmp_path / "link_dir"
        link_dir.symlink_to(real_dir)

        # Checking the symlink directory itself should fail
        with pytest.raises(ValueError, match="Symlink not allowed"):
            check_symlink_safety(link_dir, allow_symlinks=False)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlink creation requires admin privileges on Windows",
    )
    def test_file_through_symlink_dir_resolved(self, tmp_path: Path) -> None:
        """Test that file accessed through symlink dir is safe after resolution.

        System-level symlinks (e.g. /tmp -> /private/tmp on macOS) and
        user symlinks that resolve to real paths are handled by resolving
        the path before checking parent components.
        """
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        real_file = real_dir / "file.txt"
        real_file.touch()

        link_dir = tmp_path / "link_dir"
        link_dir.symlink_to(real_dir)

        # Accessing file through symlink dir: after resolution it's just
        # real_dir/file.txt, so no symlink detected in parents
        nested_path = link_dir / "file.txt"
        check_symlink_safety(nested_path, allow_symlinks=False)  # Should not raise

    def test_regular_nested_path(self, tmp_path: Path) -> None:
        """Test deeply nested regular paths pass check."""
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "file.txt"
        deep_path.parent.mkdir(parents=True)
        deep_path.touch()

        # Should not raise
        check_symlink_safety(deep_path, allow_symlinks=False)


class TestSanitizeErrorMessage:
    """Tests for sanitize_error_message function."""

    def test_sanitize_unix_path(self) -> None:
        """Test Unix paths are sanitized."""
        error = Exception("File not found: /home/user/secret/file.txt")
        result = sanitize_error_message(error)
        assert "/home/user/secret" not in result
        assert "[PATH]" in result or "[USER]" in result

    def test_sanitize_windows_path(self) -> None:
        """Test Windows paths are sanitized."""
        error = Exception("Cannot access C:\\Users\\admin\\Documents\\secret.docx")
        result = sanitize_error_message(error)
        assert "admin" not in result
        assert "[PATH]" in result or "[USER]" in result

    def test_sanitize_username_in_path(self) -> None:
        """Test username in path is sanitized."""
        error = Exception("Error in /home/sensitive_user/file.txt")
        result = sanitize_error_message(error)
        assert "sensitive_user" not in result

    def test_no_paths_unchanged(self) -> None:
        """Test message without paths is unchanged."""
        error = Exception("Generic error occurred")
        result = sanitize_error_message(error)
        assert result == "Generic error occurred"


class TestValidateFileSize:
    """Tests for validate_file_size function."""

    def test_file_within_limit(self, tmp_path: Path) -> None:
        """Test file within size limit passes."""
        test_file = tmp_path / "small.txt"
        test_file.write_text("small content")

        # Should not raise
        validate_file_size(test_file, 1024 * 1024)  # 1 MB limit

    def test_file_exceeds_limit(self, tmp_path: Path) -> None:
        """Test file exceeding limit raises error."""
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * 1000)  # 1000 bytes

        with pytest.raises(ValueError, match="File too large"):
            validate_file_size(test_file, 100)  # 100 bytes limit

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test nonexistent file passes (no error)."""
        test_file = tmp_path / "missing.txt"

        # Should not raise for missing files
        validate_file_size(test_file, 100)

    def test_exact_limit(self, tmp_path: Path) -> None:
        """Test file at exact limit passes."""
        test_file = tmp_path / "exact.txt"
        test_file.write_bytes(b"x" * 100)

        # Should not raise (equal to limit is OK)
        validate_file_size(test_file, 100)

    def test_one_byte_over_limit(self, tmp_path: Path) -> None:
        """Test file one byte over limit fails."""
        test_file = tmp_path / "over.txt"
        test_file.write_bytes(b"x" * 101)

        with pytest.raises(ValueError, match="File too large"):
            validate_file_size(test_file, 100)


class TestSizeConstants:
    """Tests for size limit constants."""

    def test_max_state_file_size(self) -> None:
        """Test MAX_STATE_FILE_SIZE is reasonable."""
        assert MAX_STATE_FILE_SIZE == 10 * 1024 * 1024  # 10 MB
        assert MAX_STATE_FILE_SIZE > 0

    def test_max_document_size(self) -> None:
        """Test MAX_DOCUMENT_SIZE is reasonable."""
        assert MAX_DOCUMENT_SIZE == 500 * 1024 * 1024  # 500 MB
        assert MAX_DOCUMENT_SIZE > MAX_STATE_FILE_SIZE


class TestPathTraversalInBatch:
    """Tests for path traversal protection in batch processing."""

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlink creation requires admin privileges on Windows",
    )
    def test_discover_files_rejects_symlink_outside(self, tmp_path: Path) -> None:
        """Test batch discover_files rejects symlinks pointing outside."""
        from markitai.batch import BatchProcessor
        from markitai.config import BatchConfig

        # Create structure
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a file outside input_dir
        outside_file = tmp_path / "outside.docx"
        outside_file.touch()

        # Create symlink inside input_dir pointing outside
        link = input_dir / "sneaky.docx"
        link.symlink_to(outside_file)

        # Create a legitimate file
        legit_file = input_dir / "legit.docx"
        legit_file.touch()

        config = BatchConfig()
        processor = BatchProcessor(config, output_dir)
        files = processor.discover_files(input_dir, {".docx"})

        # Should only find the legitimate file, not the symlink
        assert len(files) == 1
        assert files[0].name == "legit.docx"


class TestGlobPatternInjection:
    """Tests for glob pattern injection prevention."""

    def test_malicious_filename_pattern(self, tmp_path: Path) -> None:
        """Test that malicious filename patterns don't match other files."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        # Create files
        (assets_dir / "normal.jpg").touch()
        (assets_dir / "secret.jpg").touch()

        # Simulate a malicious filename that tries to use glob patterns
        malicious_name = "*"  # Would match everything if not escaped

        escaped = escape_glob_pattern(malicious_name)
        matches = list(assets_dir.glob(f"{escaped}*"))

        # Should not match any files since there's no file starting with literal *
        assert len(matches) == 0

    def test_bracket_injection(self, tmp_path: Path) -> None:
        """Test bracket injection is prevented."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        (assets_dir / "a.jpg").touch()
        (assets_dir / "b.jpg").touch()
        (assets_dir / "c.jpg").touch()

        # This would match a, b, or c if not escaped
        malicious_name = "[abc]"

        escaped = escape_glob_pattern(malicious_name)
        matches = list(assets_dir.glob(f"{escaped}.jpg"))

        # Should not match any files
        assert len(matches) == 0


class TestAtomicWriteTextAsync:
    """Tests for atomic_write_text_async function."""

    @pytest.mark.asyncio
    async def test_async_write_basic_content(self, tmp_path: Path) -> None:
        """Test basic async text write."""
        from markitai.security import atomic_write_text_async

        test_file = tmp_path / "async_test.txt"
        content = "Hello, Async World!"

        await atomic_write_text_async(test_file, content)

        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_async_write_unicode(self, tmp_path: Path) -> None:
        """Test async writing unicode content."""
        from markitai.security import atomic_write_text_async

        test_file = tmp_path / "async_unicode.txt"
        content = "こんにちは 世界 🌸"

        await atomic_write_text_async(test_file, content)

        assert test_file.read_text(encoding="utf-8") == content

    @pytest.mark.asyncio
    async def test_async_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test async write creates parent directories."""
        from markitai.security import atomic_write_text_async

        test_file = tmp_path / "deep" / "nested" / "dir" / "file.txt"

        await atomic_write_text_async(test_file, "nested content")

        assert test_file.exists()
        assert test_file.read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_async_write_overwrites_existing(self, tmp_path: Path) -> None:
        """Test async write overwrites existing file."""
        from markitai.security import atomic_write_text_async

        test_file = tmp_path / "existing_async.txt"
        test_file.write_text("old content")

        await atomic_write_text_async(test_file, "new async content")

        assert test_file.read_text() == "new async content"

    @pytest.mark.asyncio
    async def test_async_write_custom_encoding(self, tmp_path: Path) -> None:
        """Test async write with custom encoding."""
        from markitai.security import atomic_write_text_async

        test_file = tmp_path / "async_latin1.txt"
        content = "résumé"

        await atomic_write_text_async(test_file, content, encoding="latin-1")

        assert test_file.read_text(encoding="latin-1") == content


class TestAtomicWriteJsonAsync:
    """Tests for atomic_write_json_async function."""

    @pytest.mark.asyncio
    async def test_async_write_json(self, tmp_path: Path) -> None:
        """Test basic async JSON write."""
        import json

        from markitai.security import atomic_write_json_async

        test_file = tmp_path / "async.json"
        data = {"async": True, "value": 123}

        await atomic_write_json_async(test_file, data)

        assert json.loads(test_file.read_text()) == data

    @pytest.mark.asyncio
    async def test_async_write_json_unicode(self, tmp_path: Path) -> None:
        """Test async JSON with unicode."""
        from markitai.security import atomic_write_json_async

        test_file = tmp_path / "async_unicode.json"
        data = {"greeting": "Здравствуй"}

        await atomic_write_json_async(test_file, data, ensure_ascii=False)

        content = test_file.read_text(encoding="utf-8")
        assert "Здравствуй" in content

    @pytest.mark.asyncio
    async def test_async_write_json_indent(self, tmp_path: Path) -> None:
        """Test async JSON with custom indent."""
        from markitai.security import atomic_write_json_async

        test_file = tmp_path / "async_indented.json"
        data = {"key": "value"}

        await atomic_write_json_async(test_file, data, indent=4)

        content = test_file.read_text()
        assert "    " in content


class TestWriteBytesAsync:
    """Tests for write_bytes_async function."""

    @pytest.mark.asyncio
    async def test_async_write_bytes(self, tmp_path: Path) -> None:
        """Test basic async bytes write."""
        from markitai.security import write_bytes_async

        test_file = tmp_path / "bytes.bin"
        data = b"\x00\x01\x02\x03\xff\xfe"

        await write_bytes_async(test_file, data)

        assert test_file.read_bytes() == data

    @pytest.mark.asyncio
    async def test_async_write_bytes_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test async bytes write creates parent directories."""
        from markitai.security import write_bytes_async

        test_file = tmp_path / "nested" / "dir" / "data.bin"

        await write_bytes_async(test_file, b"binary data")

        assert test_file.exists()
        assert test_file.read_bytes() == b"binary data"

    @pytest.mark.asyncio
    async def test_async_write_empty_bytes(self, tmp_path: Path) -> None:
        """Test async writing empty bytes."""
        from markitai.security import write_bytes_async

        test_file = tmp_path / "empty.bin"

        await write_bytes_async(test_file, b"")

        assert test_file.exists()
        assert test_file.read_bytes() == b""

    @pytest.mark.asyncio
    async def test_async_write_large_bytes(self, tmp_path: Path) -> None:
        """Test async writing large binary content."""
        from markitai.security import write_bytes_async

        test_file = tmp_path / "large.bin"
        data = bytes(range(256)) * 4096  # ~1 MB

        await write_bytes_async(test_file, data)

        assert test_file.read_bytes() == data


class TestPathTraversalAdvanced:
    """Advanced tests for path traversal prevention."""

    def test_double_encoded_traversal(self, tmp_path: Path) -> None:
        """Test double-encoded path traversal attempt."""
        # %2e%2e = .. when URL decoded
        malicious_path = tmp_path / "%2e%2e" / "%2e%2e" / "etc" / "passwd"

        # The path should still be within tmp_path since we're not URL decoding
        result = validate_path_within_base(malicious_path, tmp_path)
        assert result == malicious_path.resolve()

    def test_null_byte_in_path(self, tmp_path: Path) -> None:
        """Test null byte injection in path."""
        # Create the test file first
        test_file = tmp_path / "test.txt"
        test_file.touch()

        # Path with null byte - should be handled by the filesystem
        # On most systems this will raise or create a truncated path
        try:
            malicious = tmp_path / "test.txt\x00.jpg"
            validate_path_within_base(malicious, tmp_path)
        except (ValueError, OSError):
            # Expected - null bytes in paths are invalid
            pass

    def test_unicode_normalization_attack(self, tmp_path: Path) -> None:
        """Test unicode normalization in path traversal."""
        # Some systems might normalize these characters
        malicious = tmp_path / "．．" / "secret"  # Full-width periods

        # Should either resolve within tmp_path or raise
        try:
            result = validate_path_within_base(malicious, tmp_path)
            # If it didn't raise, verify it's still within base
            assert str(result).startswith(str(tmp_path.resolve()))
        except ValueError:
            # Path traversal was detected - this is acceptable
            pass

    def test_case_sensitivity_traversal(self, tmp_path: Path) -> None:
        """Test case sensitivity in path validation."""
        subdir = tmp_path / "SubDir"
        subdir.mkdir()
        test_file = subdir / "file.txt"
        test_file.touch()

        # On case-insensitive systems, this should still work
        result = validate_path_within_base(test_file, tmp_path)
        assert result.exists()


class TestAtomicWriteAtomicity:
    """Tests to verify atomic write behavior."""

    def test_concurrent_writes_dont_corrupt(self, tmp_path: Path) -> None:
        """Test that concurrent writes don't produce corrupt files."""
        import threading

        test_file = tmp_path / "concurrent.txt"
        results: list[str] = []

        def write_content(content: str) -> None:
            for _ in range(10):
                atomic_write_text(test_file, content)
            results.append(content)

        threads = [
            threading.Thread(target=write_content, args=(f"content_{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # File should contain one of the valid contents, not a mix
        final_content = test_file.read_text()
        assert final_content in [f"content_{i}" for i in range(5)]

    def test_write_preserves_file_on_partial_failure(self, tmp_path: Path) -> None:
        """Test that original file is preserved if write fails."""
        test_file = tmp_path / "preserve.txt"
        original_content = "original content"
        test_file.write_text(original_content)

        # Try to write with a failing operation
        with (
            patch("os.fdopen", side_effect=OSError("Simulated failure")),
            pytest.raises(IOError),
        ):
            atomic_write_text(test_file, "new content")

        # Original file should be preserved
        assert test_file.read_text() == original_content

    def test_cleanup_failure_on_write_error(self, tmp_path: Path) -> None:
        """Test handling when both write and cleanup fail."""
        test_file = tmp_path / "test.txt"

        # Mock os.fdopen to fail, and os.unlink to also fail
        with (
            patch("os.fdopen", side_effect=OSError("Write failed")),
            patch("os.unlink", side_effect=OSError("Cleanup failed")),
            pytest.raises(IOError, match="Write failed"),
        ):
            atomic_write_text(test_file, "content")

        # Even if cleanup fails, the original exception should be raised


class TestAtomicWriteAsyncErrorHandling:
    """Tests for async atomic write error handling."""

    @pytest.mark.asyncio
    async def test_async_write_error_cleanup(self, tmp_path: Path) -> None:
        """Test async write cleans up temp file on error."""

        from markitai.security import atomic_write_text_async

        test_file = tmp_path / "test.txt"

        # Mock aiofiles.open to raise an error
        with (
            patch("aiofiles.open", side_effect=OSError("Async write failed")),
            pytest.raises(IOError, match="Async write failed"),
        ):
            await atomic_write_text_async(test_file, "content")

    @pytest.mark.asyncio
    async def test_async_write_replace_error(self, tmp_path: Path) -> None:
        """Test async write handles replace failure."""
        from markitai.security import atomic_write_text_async

        test_file = tmp_path / "test.txt"

        # Mock aiofiles.os.replace to fail
        with (
            patch("aiofiles.os.replace", side_effect=OSError("Replace failed")),
            pytest.raises(OSError, match="Replace failed"),
        ):
            await atomic_write_text_async(test_file, "content")

    @pytest.mark.asyncio
    async def test_async_write_cleanup_failure(self, tmp_path: Path) -> None:
        """Test async write handles cleanup failure gracefully."""
        from markitai.security import atomic_write_text_async

        test_file = tmp_path / "test.txt"

        # Mock both replace and remove to fail
        with (
            patch("aiofiles.os.replace", side_effect=OSError("Replace failed")),
            patch("aiofiles.os.remove", side_effect=OSError("Cleanup failed")),
            pytest.raises(OSError, match="Replace failed"),
        ):
            await atomic_write_text_async(test_file, "content")


class TestFilenameSecurityEdgeCases:
    """Edge case tests for filename and path security."""

    def test_very_long_path(self, tmp_path: Path) -> None:
        """Test handling of very long paths."""
        # Create a path that might exceed filesystem limits
        long_name = "a" * 200
        test_file = tmp_path / long_name

        try:
            atomic_write_text(test_file, "content")
            assert test_file.exists()
        except OSError:
            # Expected on some filesystems with name length limits
            pass

    def test_special_characters_in_filename(self, tmp_path: Path) -> None:
        """Test filenames with special characters."""
        special_chars = ["test file.txt", "test\ttab.txt", "test'quote.txt"]

        for name in special_chars:
            test_file = tmp_path / name
            try:
                atomic_write_text(test_file, "content")
                assert test_file.exists()
            except OSError:
                # Some chars might not be allowed on certain systems
                pass

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows doesn't allow these characters in filenames",
    )
    def test_unix_special_filenames(self, tmp_path: Path) -> None:
        """Test Unix-specific special filenames."""
        special_names = [".hidden", "..file", "-dash"]

        for name in special_names:
            test_file = tmp_path / name
            atomic_write_text(test_file, "content")
            assert test_file.exists()

    def test_whitespace_only_dirname(self, tmp_path: Path) -> None:
        """Test path with whitespace-only directory component."""
        # This should work on most systems
        test_file = tmp_path / " " / "file.txt"

        try:
            atomic_write_text(test_file, "content")
            assert test_file.exists()
        except OSError:
            # Might fail on some systems
            pass
