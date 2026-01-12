"""Tests for Office preprocessor module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from markit.converters.office import (
    LEGACY_TO_MODERN,
    LibreOfficeConverter,
    OfficePreprocessor,
    check_office_available,
)
from markit.exceptions import ConversionError


class TestLegacyToModernMapping:
    """Tests for LEGACY_TO_MODERN mapping."""

    def test_doc_to_docx(self):
        """Test .doc to .docx mapping."""
        assert LEGACY_TO_MODERN[".doc"] == ".docx"

    def test_ppt_to_pptx(self):
        """Test .ppt to .pptx mapping."""
        assert LEGACY_TO_MODERN[".ppt"] == ".pptx"

    def test_xls_to_xlsx(self):
        """Test .xls to .xlsx mapping."""
        assert LEGACY_TO_MODERN[".xls"] == ".xlsx"

    def test_wps_formats(self):
        """Test WPS format mappings."""
        assert LEGACY_TO_MODERN[".wps"] == ".docx"
        assert LEGACY_TO_MODERN[".et"] == ".xlsx"
        assert LEGACY_TO_MODERN[".dps"] == ".pptx"


class TestOfficePreprocessorInit:
    """Tests for OfficePreprocessor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        preprocessor = OfficePreprocessor()

        assert preprocessor.prefer_ms_office is True
        assert preprocessor.libreoffice_path is None
        assert preprocessor.timeout == 120
        assert preprocessor._converter is None
        assert preprocessor._converted_dir is None

    def test_custom_init(self):
        """Test custom initialization."""
        preprocessor = OfficePreprocessor(
            prefer_ms_office=False,
            libreoffice_path="/custom/path",
            timeout=60,
        )

        assert preprocessor.prefer_ms_office is False
        assert preprocessor.libreoffice_path == "/custom/path"
        assert preprocessor.timeout == 60

    def test_set_converted_dir(self, tmp_path):
        """Test setting converted directory."""
        preprocessor = OfficePreprocessor()
        converted_dir = tmp_path / "converted"

        preprocessor.set_converted_dir(converted_dir)

        assert preprocessor._converted_dir == converted_dir


class TestOfficePreprocessorProcess:
    """Tests for OfficePreprocessor.process method."""

    @pytest.mark.asyncio
    async def test_process_non_legacy_format(self, tmp_path):
        """Test processing non-legacy format returns original."""
        docx_path = tmp_path / "test.docx"
        docx_path.touch()

        preprocessor = OfficePreprocessor()
        result = await preprocessor.process(docx_path)

        assert result == docx_path

    @pytest.mark.asyncio
    async def test_process_legacy_format_success(self, tmp_path):
        """Test processing legacy format."""
        doc_path = tmp_path / "test.doc"
        doc_path.touch()
        docx_path = tmp_path / "test.docx"

        mock_converter = MagicMock()
        mock_converter.convert.return_value = docx_path

        preprocessor = OfficePreprocessor()
        preprocessor._converter = mock_converter

        result = await preprocessor.process(doc_path)

        assert result == docx_path
        mock_converter.convert.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_legacy_format_failure(self, tmp_path):
        """Test processing legacy format handles errors."""
        doc_path = tmp_path / "test.doc"
        doc_path.touch()

        mock_converter = MagicMock()
        mock_converter.convert.side_effect = Exception("Conversion failed")

        preprocessor = OfficePreprocessor()
        preprocessor._converter = mock_converter

        with pytest.raises(ConversionError) as exc_info:
            await preprocessor.process(doc_path)

        assert "Office conversion failed" in str(exc_info.value)


class TestOfficePreprocessorGetConverter:
    """Tests for _get_converter method."""

    def test_get_converter_cached(self):
        """Test that cached converter is returned."""
        preprocessor = OfficePreprocessor()
        mock_converter = MagicMock()
        preprocessor._converter = mock_converter

        result = preprocessor._get_converter()

        assert result == mock_converter

    def test_get_converter_libreoffice_fallback(self):
        """Test LibreOffice fallback when MS Office not preferred."""
        preprocessor = OfficePreprocessor(prefer_ms_office=False)

        mock_libre = MagicMock()
        with patch("markit.converters.office.LibreOfficeConverter", return_value=mock_libre):
            result = preprocessor._get_converter()

            assert result == mock_libre

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_get_converter_ms_office_windows(self):
        """Test MS Office preference on Windows."""
        preprocessor = OfficePreprocessor(prefer_ms_office=True)

        mock_ms = MagicMock()
        with patch("markit.converters.office.MSOfficeConverter", return_value=mock_ms):
            result = preprocessor._get_converter()

            assert result == mock_ms


class TestLibreOfficeConverterInit:
    """Tests for LibreOfficeConverter initialization."""

    def test_init_with_path(self):
        """Test initialization with custom path."""
        converter = LibreOfficeConverter(soffice_path="/usr/bin/soffice")

        assert converter.soffice_path == "/usr/bin/soffice"

    def test_init_auto_discover(self):
        """Test automatic path discovery."""
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()

            assert converter.soffice_path is not None

    def test_init_not_found(self):
        """Test initialization when LibreOffice not found."""
        with (
            patch.object(LibreOfficeConverter, "_find_soffice", return_value=None),
            pytest.raises(RuntimeError, match="LibreOffice not found"),
        ):
            LibreOfficeConverter()


class TestLibreOfficeConverterFindSoffice:
    """Tests for _find_soffice method."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_find_soffice_windows(self):
        """Test finding soffice on Windows."""
        converter = LibreOfficeConverter.__new__(LibreOfficeConverter)

        with patch("pathlib.Path.exists", return_value=True):
            result = converter._find_soffice()

            assert result is not None
            assert "soffice.exe" in result

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_find_soffice_macos(self):
        """Test finding soffice on macOS."""
        converter = LibreOfficeConverter.__new__(LibreOfficeConverter)

        with patch("pathlib.Path.exists", return_value=True):
            result = converter._find_soffice()

            assert result is not None
            assert "soffice" in result

    def test_find_soffice_in_path(self):
        """Test finding soffice in PATH."""
        converter = LibreOfficeConverter.__new__(LibreOfficeConverter)

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            result = converter._find_soffice()

            assert result == "/usr/bin/soffice"


class TestLibreOfficeConverterGetFilterName:
    """Tests for _get_filter_name method."""

    def test_get_filter_name_docx(self):
        """Test filter name for docx."""
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()
            assert converter._get_filter_name(".docx") == "docx"

    def test_get_filter_name_xlsx(self):
        """Test filter name for xlsx."""
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()
            assert converter._get_filter_name(".xlsx") == "xlsx"

    def test_get_filter_name_pptx(self):
        """Test filter name for pptx."""
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()
            assert converter._get_filter_name(".pptx") == "pptx"

    def test_get_filter_name_unknown(self):
        """Test filter name for unknown format."""
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()
            assert converter._get_filter_name(".xyz") == "xyz"


class TestLibreOfficeConverterConvert:
    """Tests for LibreOfficeConverter.convert method."""

    def test_convert_success(self, tmp_path):
        """Test successful conversion."""
        input_file = tmp_path / "test.doc"
        input_file.touch()

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()

            # Mock subprocess.run to create the output file
            def mock_run(cmd, **_kwargs):
                # Find --outdir argument
                for i, arg in enumerate(cmd):
                    if arg == "--outdir" and i + 1 < len(cmd):
                        temp_dir = Path(cmd[i + 1])
                        (temp_dir / "test.docx").write_bytes(b"converted content")
                        break
                return MagicMock(returncode=0)

            with patch("subprocess.run", side_effect=mock_run):
                result = converter.convert(input_file, ".docx")

                assert result.exists()
                assert result.suffix == ".docx"

    def test_convert_timeout(self, tmp_path):
        """Test conversion timeout."""
        import subprocess

        input_file = tmp_path / "test.doc"
        input_file.touch()

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter(timeout=1)

            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
                with pytest.raises(ConversionError) as exc_info:
                    converter.convert(input_file, ".docx")

                assert "timed out" in str(exc_info.value)

    def test_convert_error(self, tmp_path):
        """Test conversion error handling."""
        import subprocess

        input_file = tmp_path / "test.doc"
        input_file.touch()

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()

            error = subprocess.CalledProcessError(1, "soffice")
            error.stderr = b"Error message"

            with patch("subprocess.run", side_effect=error):
                with pytest.raises(ConversionError) as exc_info:
                    converter.convert(input_file, ".docx")

                assert "LibreOffice error" in str(exc_info.value)

    def test_convert_no_output(self, tmp_path):
        """Test conversion when no output is produced."""
        input_file = tmp_path / "test.doc"
        input_file.touch()

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()

            # Mock subprocess.run but don't create output
            with patch("subprocess.run", return_value=MagicMock(returncode=0)):
                with pytest.raises(ConversionError) as exc_info:
                    converter.convert(input_file, ".docx")

                assert "did not produce output" in str(exc_info.value)


class TestLibreOfficeConverterConvertAsync:
    """Tests for LibreOfficeConverter.convert_async method."""

    @pytest.mark.asyncio
    async def test_convert_async_without_pool(self, tmp_path):
        """Test async conversion without profile pool."""
        input_file = tmp_path / "test.doc"
        input_file.touch()
        output_file = tmp_path / "test.docx"

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()

            # Mock the sync convert method
            with patch.object(converter, "convert", return_value=output_file):
                result = await converter.convert_async(input_file, ".docx")

                assert result == output_file

    @pytest.mark.asyncio
    async def test_convert_async_with_pool(self, tmp_path):
        """Test async conversion with profile pool."""
        input_file = tmp_path / "test.doc"
        input_file.touch()
        output_file = tmp_path / "test.docx"

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock()

        # Create async context manager mock
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        class MockContextManager:
            async def __aenter__(self):
                return profile_dir

            async def __aexit__(self, *_args):
                pass

        mock_pool.acquire.return_value = MockContextManager()

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter(profile_pool=mock_pool)

            # Mock the _convert_with_profile method
            with patch.object(converter, "_convert_with_profile", return_value=output_file):
                result = await converter.convert_async(input_file, ".docx")

                assert result == output_file


class TestLibreOfficeConverterConvertWithProfile:
    """Tests for LibreOfficeConverter._convert_with_profile method.

    This tests the profile pool integration that allows concurrent
    LibreOffice conversions using isolated user profiles.
    """

    @pytest.mark.asyncio
    async def test_convert_with_profile_success(self, tmp_path):
        """Test successful conversion with profile directory."""
        input_file = tmp_path / "test.doc"
        input_file.write_bytes(b"test content")
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        captured_commands: list[list[str]] = []

        def mock_run(cmd, **_kwargs):
            captured_commands.append(cmd)
            # Find --outdir argument and create output file
            for i, arg in enumerate(cmd):
                if arg == "--outdir" and i + 1 < len(cmd):
                    temp_dir = Path(cmd[i + 1])
                    (temp_dir / "test.docx").write_bytes(b"converted content")
                    break
            return MagicMock(returncode=0)

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()

            with patch("subprocess.run", side_effect=mock_run):
                result = await converter._convert_with_profile(input_file, ".docx", profile_dir)

                # Verify output file exists and is correct
                assert result.exists()
                assert result.suffix == ".docx"

                # Verify command includes profile directory
                assert len(captured_commands) == 1
                cmd = captured_commands[0]

                # Verify soffice executable
                assert cmd[0] == "/usr/bin/soffice"
                # Verify headless mode
                assert "--headless" in cmd
                # Verify profile URI is included
                profile_args = [arg for arg in cmd if "UserInstallation" in arg]
                assert len(profile_args) == 1
                assert str(profile_dir) in profile_args[0] or profile_dir.name in profile_args[0]

    @pytest.mark.asyncio
    async def test_convert_with_profile_custom_output_dir(self, tmp_path):
        """Test conversion with profile and custom output directory."""
        input_file = tmp_path / "test.doc"
        input_file.write_bytes(b"test content")
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        converted_dir = tmp_path / "converted"
        converted_dir.mkdir()

        def mock_run(cmd, **_kwargs):
            # Find --outdir argument and create output file
            for i, arg in enumerate(cmd):
                if arg == "--outdir" and i + 1 < len(cmd):
                    temp_dir = Path(cmd[i + 1])
                    (temp_dir / "test.docx").write_bytes(b"converted content")
                    break
            return MagicMock(returncode=0)

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()

            with patch("subprocess.run", side_effect=mock_run):
                result = await converter._convert_with_profile(
                    input_file, ".docx", profile_dir, converted_dir
                )

                # Verify output is in converted directory
                assert result.parent == converted_dir
                assert result.name == "test.docx"

    @pytest.mark.asyncio
    async def test_convert_with_profile_timeout(self, tmp_path):
        """Test timeout handling in _convert_with_profile."""
        import subprocess

        input_file = tmp_path / "test.doc"
        input_file.write_bytes(b"test content")
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter(timeout=1)

            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
                with pytest.raises(ConversionError) as exc_info:
                    await converter._convert_with_profile(input_file, ".docx", profile_dir)

                assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_convert_with_profile_subprocess_error(self, tmp_path):
        """Test subprocess error handling in _convert_with_profile."""
        import subprocess

        input_file = tmp_path / "test.doc"
        input_file.write_bytes(b"test content")
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()

            error = subprocess.CalledProcessError(1, "soffice")
            error.stderr = b"LibreOffice crashed"

            with patch("subprocess.run", side_effect=error):
                with pytest.raises(ConversionError) as exc_info:
                    await converter._convert_with_profile(input_file, ".docx", profile_dir)

                assert "LibreOffice error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_convert_with_profile_no_output(self, tmp_path):
        """Test handling when no output is produced."""
        input_file = tmp_path / "test.doc"
        input_file.write_bytes(b"test content")
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeConverter()

            # Mock subprocess.run but don't create output
            with patch("subprocess.run", return_value=MagicMock(returncode=0)):
                with pytest.raises(ConversionError) as exc_info:
                    await converter._convert_with_profile(input_file, ".docx", profile_dir)

                assert "did not produce output" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_convert_with_profile_windows_uri(self, tmp_path):
        """Test Windows-style profile URI generation."""
        input_file = tmp_path / "test.doc"
        input_file.write_bytes(b"test content")
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        captured_commands: list[list[str]] = []

        def mock_run(cmd, **_kwargs):
            captured_commands.append(cmd)
            for i, arg in enumerate(cmd):
                if arg == "--outdir" and i + 1 < len(cmd):
                    temp_dir = Path(cmd[i + 1])
                    (temp_dir / "test.docx").write_bytes(b"converted content")
                    break
            return MagicMock(returncode=0)

        with (
            patch("shutil.which", return_value="/usr/bin/soffice"),
            patch("sys.platform", "win32"),
        ):
            converter = LibreOfficeConverter()

            with patch("subprocess.run", side_effect=mock_run):
                await converter._convert_with_profile(input_file, ".docx", profile_dir)

                # Verify command was issued
                assert len(captured_commands) == 1
                cmd = captured_commands[0]

                # On Windows, should use file:/// format
                profile_args = [arg for arg in cmd if "UserInstallation" in arg]
                assert len(profile_args) == 1
                # Windows path should be converted to URI format
                assert "file:///" in profile_args[0]


class TestCheckOfficeAvailable:
    """Tests for check_office_available function."""

    def test_check_office_libreoffice_available(self):
        """Test when LibreOffice is available."""
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            result = check_office_available()

            assert result["libreoffice"] is True

    def test_check_office_libreoffice_not_available(self):
        """Test when LibreOffice is not available."""
        with patch.object(LibreOfficeConverter, "__init__", side_effect=RuntimeError("Not found")):
            result = check_office_available()

            assert result["libreoffice"] is False

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    def test_check_office_ms_office_available(self):
        """Test when MS Office is available on Windows."""
        mock_word = MagicMock()

        with patch("win32com.client.Dispatch", return_value=mock_word):
            result = check_office_available()

            # Result depends on LibreOffice availability too
            assert "ms_office" in result

    def test_check_office_returns_dict(self):
        """Test that function returns proper dict structure."""
        with patch.object(LibreOfficeConverter, "__init__", side_effect=RuntimeError):
            result = check_office_available()

            assert isinstance(result, dict)
            assert "libreoffice" in result
            assert "ms_office" in result
