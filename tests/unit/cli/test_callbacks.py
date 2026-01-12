import pytest
import typer

from markit.cli.callbacks import (
    validate_input_file,
    validate_llm_provider,
    validate_output_dir,
    validate_pdf_engine,
)


class TestCallbacks:
    def test_validate_output_dir_valid(self, tmp_path):
        """Test validation of valid output directory."""
        assert validate_output_dir(tmp_path) == tmp_path

    def test_validate_output_dir_none(self):
        """Test validation of None output directory."""
        assert validate_output_dir(None) is None

    def test_validate_output_dir_file(self, tmp_path):
        """Test validation fails if output path is a file."""
        file_path = tmp_path / "test.txt"
        file_path.touch()
        with pytest.raises(typer.BadParameter, match="Output path exists but is not a directory"):
            validate_output_dir(file_path)

    def test_validate_input_file_valid(self, tmp_path):
        """Test validation of valid input file."""
        file_path = tmp_path / "test.txt"
        file_path.touch()
        assert validate_input_file(file_path) == file_path

    def test_validate_input_file_not_found(self, tmp_path):
        """Test validation fails if input file doesn't exist."""
        with pytest.raises(typer.BadParameter, match="File not found"):
            validate_input_file(tmp_path / "nonexistent.txt")

    def test_validate_input_file_is_dir(self, tmp_path):
        """Test validation fails if input path is a directory."""
        with pytest.raises(typer.BadParameter, match="Path is not a file"):
            validate_input_file(tmp_path)

    def test_validate_pdf_engine_valid(self):
        """Test validation of valid PDF engines."""
        from markit.config.constants import PDF_ENGINES

        for engine in PDF_ENGINES:
            assert validate_pdf_engine(engine) == engine

    def test_validate_pdf_engine_none(self):
        """Test validation of None PDF engine."""
        assert validate_pdf_engine(None) is None

    def test_validate_pdf_engine_invalid(self):
        """Test validation fails for invalid PDF engine."""
        with pytest.raises(typer.BadParameter, match="Invalid PDF engine"):
            validate_pdf_engine("invalid_engine")

    def test_validate_llm_provider_valid(self):
        """Test validation of valid LLM providers."""
        from markit.config.constants import LLM_PROVIDERS

        for provider in LLM_PROVIDERS:
            assert validate_llm_provider(provider) == provider

    def test_validate_llm_provider_none(self):
        """Test validation of None LLM provider."""
        assert validate_llm_provider(None) is None

    def test_validate_llm_provider_invalid(self):
        """Test validation fails for invalid LLM provider."""
        with pytest.raises(typer.BadParameter, match="Invalid LLM provider"):
            validate_llm_provider("invalid_provider")
