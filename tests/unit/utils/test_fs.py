"""Tests for filesystem utilities module."""

from pathlib import Path

import pytest

from markit.utils.fs import (
    atomic_write,
    clean_empty_directories,
    compute_file_hash,
    compute_quick_hash,
    copy_file_safe,
    discover_files,
    ensure_directory,
    format_size,
    get_file_size_human,
    get_relative_path,
    get_unique_path,
    is_hidden,
    iter_files,
    move_file_safe,
    safe_filename,
    temporary_directory,
)


class TestEnsureDirectory:
    """Tests for ensure_directory function."""

    def test_creates_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "new_dir"
        result = ensure_directory(new_dir)
        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_creates_nested_directories(self, tmp_path):
        """Test creating nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        result = ensure_directory(nested_dir)
        assert result == nested_dir
        assert nested_dir.exists()

    def test_existing_directory(self, tmp_path):
        """Test with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        result = ensure_directory(existing_dir)
        assert result == existing_dir
        assert existing_dir.exists()


class TestSafeFilename:
    """Tests for safe_filename function."""

    def test_removes_path_separators(self):
        """Test removing path separators."""
        assert safe_filename("file/name") == "file_name"
        assert safe_filename("file\\name") == "file_name"

    def test_removes_special_characters(self):
        """Test removing special characters."""
        assert safe_filename("file:name") == "file_name"
        assert safe_filename("file*name") == "file_name"
        assert safe_filename("file?name") == "file_name"
        assert safe_filename('file"name') == "file_name"
        assert safe_filename("file<name") == "file_name"
        assert safe_filename("file>name") == "file_name"
        assert safe_filename("file|name") == "file_name"

    def test_removes_null_character(self):
        """Test removing null character."""
        assert safe_filename("file\0name") == "filename"

    def test_strips_dots_and_spaces(self):
        """Test stripping leading/trailing dots and spaces."""
        assert safe_filename("  filename  ") == "filename"
        assert safe_filename("..filename..") == "filename"
        assert safe_filename(". filename .") == "filename"

    def test_truncates_long_filename(self):
        """Test truncating long filenames."""
        long_name = "a" * 300 + ".txt"
        result = safe_filename(long_name, max_length=255)
        assert len(result) <= 255
        assert result.endswith(".txt")

    def test_preserves_valid_filename(self):
        """Test that valid filenames are preserved."""
        assert safe_filename("valid_filename.txt") == "valid_filename.txt"

    def test_handles_empty_string(self):
        """Test handling empty string."""
        assert safe_filename("") == ""


class TestGetUniquePath:
    """Tests for get_unique_path function."""

    def test_returns_original_if_not_exists(self, tmp_path):
        """Test returning original path if it doesn't exist."""
        path = tmp_path / "file.txt"
        result = get_unique_path(path)
        assert result == path

    def test_adds_counter_if_exists(self, tmp_path):
        """Test adding counter suffix if path exists."""
        path = tmp_path / "file.txt"
        path.write_text("content")

        result = get_unique_path(path)
        assert result == tmp_path / "file_1.txt"

    def test_increments_counter(self, tmp_path):
        """Test incrementing counter for multiple existing files."""
        path = tmp_path / "file.txt"
        path.write_text("content")
        (tmp_path / "file_1.txt").write_text("content")
        (tmp_path / "file_2.txt").write_text("content")

        result = get_unique_path(path)
        assert result == tmp_path / "file_3.txt"


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_sha256_hash(self, tmp_path):
        """Test SHA256 hash computation."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")

        hash_result = compute_file_hash(file_path, "sha256")
        assert len(hash_result) == 64  # SHA256 hex length

    def test_md5_hash(self, tmp_path):
        """Test MD5 hash computation."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")

        hash_result = compute_file_hash(file_path, "md5")
        assert len(hash_result) == 32  # MD5 hex length

    def test_same_content_same_hash(self, tmp_path):
        """Test that same content produces same hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("identical content")
        file2.write_text("identical content")

        assert compute_file_hash(file1) == compute_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path):
        """Test that different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")

        assert compute_file_hash(file1) != compute_file_hash(file2)


class TestComputeQuickHash:
    """Tests for compute_quick_hash function."""

    def test_returns_hash(self, tmp_path):
        """Test that quick hash returns a hash string."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        result = compute_quick_hash(file_path)
        assert len(result) == 32  # MD5 hex length

    def test_different_files_different_hash(self, tmp_path):
        """Test that files with different metadata have different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("short")
        file2.write_text("much longer content here")

        # Different sizes should produce different hashes
        assert compute_quick_hash(file1) != compute_quick_hash(file2)


class TestDiscoverFiles:
    """Tests for discover_files function."""

    def test_discover_files_in_directory(self, tmp_path):
        """Test discovering files in a directory."""
        (tmp_path / "doc.pdf").write_text("pdf")
        (tmp_path / "doc.docx").write_text("docx")
        (tmp_path / "other.txt").write_text("txt")

        files = discover_files(tmp_path, extensions={".pdf", ".docx"})
        assert len(files) == 2

    def test_discover_files_recursive(self, tmp_path):
        """Test recursive file discovery."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.pdf").write_text("pdf")
        (subdir / "nested.pdf").write_text("pdf")

        # Non-recursive
        files = discover_files(tmp_path, recursive=False, extensions={".pdf"})
        assert len(files) == 1

        # Recursive
        files = discover_files(tmp_path, recursive=True, extensions={".pdf"})
        assert len(files) == 2

    def test_discover_files_with_include_pattern(self, tmp_path):
        """Test file discovery with include pattern."""
        (tmp_path / "report.pdf").write_text("pdf")
        (tmp_path / "test.pdf").write_text("pdf")

        files = discover_files(
            tmp_path,
            extensions={".pdf"},
            include_pattern="report*",
        )
        assert len(files) == 1
        assert files[0].name == "report.pdf"

    def test_discover_files_with_exclude_pattern(self, tmp_path):
        """Test file discovery with exclude pattern."""
        (tmp_path / "report.pdf").write_text("pdf")
        (tmp_path / "test.pdf").write_text("pdf")

        files = discover_files(
            tmp_path,
            extensions={".pdf"},
            exclude_pattern="test*",
        )
        assert len(files) == 1
        assert files[0].name == "report.pdf"

    def test_discover_files_sorted(self, tmp_path):
        """Test that discovered files are sorted."""
        (tmp_path / "z.pdf").write_text("pdf")
        (tmp_path / "a.pdf").write_text("pdf")
        (tmp_path / "m.pdf").write_text("pdf")

        files = discover_files(tmp_path, extensions={".pdf"})
        names = [f.name for f in files]
        assert names == ["a.pdf", "m.pdf", "z.pdf"]


class TestIterFiles:
    """Tests for iter_files function."""

    def test_iter_files_basic(self, tmp_path):
        """Test basic file iteration."""
        (tmp_path / "doc.pdf").write_text("pdf")
        (tmp_path / "doc.docx").write_text("docx")

        files = list(iter_files(tmp_path, extensions={".pdf", ".docx"}))
        assert len(files) == 2

    def test_iter_files_recursive(self, tmp_path):
        """Test recursive file iteration."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.pdf").write_text("pdf")
        (subdir / "nested.pdf").write_text("pdf")

        files = list(iter_files(tmp_path, recursive=True, extensions={".pdf"}))
        assert len(files) == 2

    def test_iter_files_non_recursive(self, tmp_path):
        """Test non-recursive file iteration."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.pdf").write_text("pdf")
        (subdir / "nested.pdf").write_text("pdf")

        files = list(iter_files(tmp_path, recursive=False, extensions={".pdf"}))
        assert len(files) == 1


class TestGetRelativePath:
    """Tests for get_relative_path function."""

    def test_relative_path(self, tmp_path):
        """Test getting relative path."""
        file_path = tmp_path / "subdir" / "file.txt"
        result = get_relative_path(file_path, tmp_path)
        assert result == Path("subdir/file.txt") or result == Path("subdir\\file.txt")

    def test_not_relative(self):
        """Test when path is not relative to base."""
        file_path = Path("/some/other/path/file.txt")
        base_path = Path("/different/base")
        result = get_relative_path(file_path, base_path)
        assert result == file_path


class TestCopyFileSafe:
    """Tests for copy_file_safe function."""

    def test_copy_file(self, tmp_path):
        """Test copying a file."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("content")

        result = copy_file_safe(src, dst)
        assert result == dst
        assert dst.exists()
        assert dst.read_text() == "content"
        assert src.exists()  # Source still exists

    def test_copy_creates_parent_dirs(self, tmp_path):
        """Test that copy creates parent directories."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "new_dir" / "dest.txt"
        src.write_text("content")

        result = copy_file_safe(src, dst)
        assert result == dst
        assert dst.exists()

    def test_copy_raises_if_exists(self, tmp_path):
        """Test that copy raises if destination exists."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("content")
        dst.write_text("existing")

        with pytest.raises(FileExistsError):
            copy_file_safe(src, dst, overwrite=False)

    def test_copy_overwrites_if_allowed(self, tmp_path):
        """Test that copy overwrites if allowed."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("new content")
        dst.write_text("old content")

        copy_file_safe(src, dst, overwrite=True)
        assert dst.read_text() == "new content"


class TestMoveFileSafe:
    """Tests for move_file_safe function."""

    def test_move_file(self, tmp_path):
        """Test moving a file."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("content")

        result = move_file_safe(src, dst)
        assert result == dst
        assert dst.exists()
        assert not src.exists()  # Source removed

    def test_move_creates_parent_dirs(self, tmp_path):
        """Test that move creates parent directories."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "new_dir" / "dest.txt"
        src.write_text("content")

        result = move_file_safe(src, dst)
        assert result == dst
        assert dst.exists()

    def test_move_raises_if_exists(self, tmp_path):
        """Test that move raises if destination exists."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("content")
        dst.write_text("existing")

        with pytest.raises(FileExistsError):
            move_file_safe(src, dst, overwrite=False)

    def test_move_overwrites_if_allowed(self, tmp_path):
        """Test that move overwrites if allowed."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "dest.txt"
        src.write_text("new content")
        dst.write_text("old content")

        move_file_safe(src, dst, overwrite=True)
        assert dst.read_text() == "new content"
        assert not src.exists()


class TestAtomicWrite:
    """Tests for atomic_write context manager."""

    def test_atomic_write_text(self, tmp_path):
        """Test atomic write for text files."""
        file_path = tmp_path / "test.txt"

        with atomic_write(file_path) as f:
            f.write("test content")

        assert file_path.exists()
        assert file_path.read_text() == "test content"

    def test_atomic_write_binary(self, tmp_path):
        """Test atomic write for binary files."""
        file_path = tmp_path / "test.bin"

        with atomic_write(file_path, mode="wb") as f:
            f.write(b"binary content")

        assert file_path.exists()
        assert file_path.read_bytes() == b"binary content"

    def test_atomic_write_creates_parent_dirs(self, tmp_path):
        """Test that atomic write creates parent directories."""
        file_path = tmp_path / "new_dir" / "test.txt"

        with atomic_write(file_path) as f:
            f.write("content")

        assert file_path.exists()

    def test_atomic_write_cleans_up_on_error(self, tmp_path):
        """Test that atomic write cleans up temp file on error."""
        file_path = tmp_path / "test.txt"

        with pytest.raises(ValueError), atomic_write(file_path) as f:
            f.write("partial")
            raise ValueError("test error")

        assert not file_path.exists()


class TestTemporaryDirectory:
    """Tests for temporary_directory context manager."""

    def test_creates_temp_dir(self):
        """Test that temporary directory is created."""
        with temporary_directory() as temp_dir:
            assert temp_dir.exists()
            assert temp_dir.is_dir()

    def test_cleans_up_temp_dir(self):
        """Test that temporary directory is cleaned up."""
        with temporary_directory() as temp_dir:
            temp_path = temp_dir
            # Create some files
            (temp_dir / "file.txt").write_text("content")

        assert not temp_path.exists()


class TestGetFileSizeHuman:
    """Tests for get_file_size_human function."""

    def test_small_file(self, tmp_path):
        """Test human-readable size for small file."""
        file_path = tmp_path / "small.txt"
        file_path.write_text("x" * 100)

        result = get_file_size_human(file_path)
        assert "B" in result


class TestFormatSize:
    """Tests for format_size function."""

    def test_bytes(self):
        """Test formatting bytes."""
        assert format_size(500) == "500.0 B"

    def test_kilobytes(self):
        """Test formatting kilobytes."""
        result = format_size(1536)
        assert "KB" in result

    def test_megabytes(self):
        """Test formatting megabytes."""
        result = format_size(1024 * 1024 * 2)
        assert "MB" in result

    def test_gigabytes(self):
        """Test formatting gigabytes."""
        result = format_size(1024 * 1024 * 1024 * 3)
        assert "GB" in result


class TestIsHidden:
    """Tests for is_hidden function."""

    def test_unix_hidden_file(self, tmp_path):
        """Test detecting Unix hidden files."""
        hidden_file = tmp_path / ".hidden"
        hidden_file.write_text("content")

        assert is_hidden(hidden_file) is True

    def test_regular_file(self, tmp_path):
        """Test that regular files are not hidden."""
        regular_file = tmp_path / "regular.txt"
        regular_file.write_text("content")

        assert is_hidden(regular_file) is False

    def test_hidden_directory(self, tmp_path):
        """Test detecting hidden directories."""
        hidden_dir = tmp_path / ".hidden_dir"
        hidden_dir.mkdir()

        assert is_hidden(hidden_dir) is True


class TestCleanEmptyDirectories:
    """Tests for clean_empty_directories function."""

    def test_removes_empty_directories(self, tmp_path):
        """Test removing empty directories."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        removed = clean_empty_directories(tmp_path, recursive=True)
        assert removed == 1
        assert not empty_dir.exists()

    def test_preserves_non_empty_directories(self, tmp_path):
        """Test that non-empty directories are preserved."""
        non_empty_dir = tmp_path / "non_empty"
        non_empty_dir.mkdir()
        (non_empty_dir / "file.txt").write_text("content")

        removed = clean_empty_directories(tmp_path, recursive=True)
        assert removed == 0
        assert non_empty_dir.exists()

    def test_recursive_cleanup(self, tmp_path):
        """Test recursive cleanup of nested empty directories."""
        nested = tmp_path / "level1" / "level2" / "level3"
        nested.mkdir(parents=True)

        removed = clean_empty_directories(tmp_path, recursive=True)
        assert removed == 3  # All three levels removed
        assert not (tmp_path / "level1").exists()
