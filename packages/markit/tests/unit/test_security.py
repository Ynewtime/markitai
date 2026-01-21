"""Unit tests for security utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from markit.security import (
    MAX_DOCUMENT_SIZE,
    MAX_STATE_FILE_SIZE,
    check_symlink_safety,
    escape_glob_pattern,
    sanitize_error_message,
    validate_file_size,
    validate_path_within_base,
)


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
    def test_nested_symlink_not_allowed(self, tmp_path: Path) -> None:
        """Test nested symlink in parent directory is detected."""
        # Create: tmp_path/real_dir/file.txt
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        real_file = real_dir / "file.txt"
        real_file.touch()

        # Create symlink directory: tmp_path/link_dir -> tmp_path/real_dir
        link_dir = tmp_path / "link_dir"
        link_dir.symlink_to(real_dir)

        # Access file through symlink: tmp_path/link_dir/file.txt
        nested_path = link_dir / "file.txt"

        with pytest.raises(ValueError, match="Nested symlink not allowed"):
            check_symlink_safety(nested_path, allow_symlinks=False)

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
        from markit.batch import BatchProcessor
        from markit.config import BatchConfig

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
