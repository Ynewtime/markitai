"""Tests for exceptions module."""

from pathlib import Path


class TestExceptions:
    """Tests for custom exceptions."""

    def test_markit_error(self):
        """Test base MarkitError."""
        from markit.exceptions import MarkitError

        error = MarkitError("Test error")
        assert str(error) == "Test error"

    def test_conversion_error(self):
        """Test ConversionError."""
        from markit.exceptions import ConversionError

        file_path = Path("/test/file.docx")
        error = ConversionError(file_path, "Invalid format")

        assert "file.docx" in str(error)
        assert "Invalid format" in str(error)
        assert error.file_path == file_path
        assert error.cause is None

    def test_conversion_error_with_cause(self):
        """Test ConversionError with cause."""
        from markit.exceptions import ConversionError

        file_path = Path("/test/file.docx")
        cause = ValueError("Original error")
        error = ConversionError(file_path, "Conversion failed", cause=cause)

        assert error.cause == cause

    def test_converter_not_found_error(self):
        """Test ConverterNotFoundError."""
        from markit.exceptions import ConverterNotFoundError

        file_path = Path("/test/file.xyz")
        error = ConverterNotFoundError(file_path, ".xyz")

        assert ".xyz" in str(error)
        assert error.extension == ".xyz"

    def test_fallback_exhausted_error(self):
        """Test FallbackExhaustedError."""
        from markit.exceptions import ConversionError, FallbackExhaustedError

        file_path = Path("/test/file.docx")
        errors: list[Exception] = [
            ConversionError(file_path, "Primary failed"),
            ConversionError(file_path, "Fallback failed"),
        ]
        error = FallbackExhaustedError(file_path, errors)

        assert "All conversion attempts failed" in str(error)
        assert len(error.errors) == 2

    def test_llm_error(self):
        """Test LLMError."""
        from markit.exceptions import LLMError

        error = LLMError("API error")
        assert str(error) == "API error"

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        from markit.exceptions import RateLimitError

        error = RateLimitError(retry_after=30)
        assert error.retry_after == 30
        assert "30" in str(error)

    def test_rate_limit_error_no_retry(self):
        """Test RateLimitError without retry_after."""
        from markit.exceptions import RateLimitError

        error = RateLimitError()
        assert error.retry_after is None
        assert "Rate limited" in str(error)

    def test_provider_not_found_error(self):
        """Test ProviderNotFoundError."""
        from markit.exceptions import ProviderNotFoundError

        error = ProviderNotFoundError()
        assert "No valid LLM provider" in str(error)

    def test_image_processing_error(self):
        """Test ImageProcessingError."""
        from markit.exceptions import ImageProcessingError

        error = ImageProcessingError("Compression failed")
        assert str(error) == "Compression failed"

    def test_state_error(self):
        """Test StateError."""
        from markit.exceptions import StateError

        error = StateError("State corrupted")
        assert str(error) == "State corrupted"

    def test_configuration_error(self):
        """Test ConfigurationError."""
        from markit.exceptions import ConfigurationError

        error = ConfigurationError("Invalid config")
        assert str(error) == "Invalid config"
