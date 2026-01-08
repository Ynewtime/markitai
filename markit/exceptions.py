"""Custom exceptions for MarkIt."""

from pathlib import Path


class MarkitError(Exception):
    """Base exception class for MarkIt."""

    pass


class ConversionError(MarkitError):
    """Error during document conversion."""

    def __init__(self, file_path: Path, message: str, cause: Exception | None = None) -> None:
        self.file_path = file_path
        self.cause = cause
        super().__init__(f"Conversion failed for {file_path}: {message}")


class ConverterNotFoundError(ConversionError):
    """No suitable converter found for the file format."""

    def __init__(self, file_path: Path, extension: str) -> None:
        super().__init__(file_path, f"No converter found for extension: {extension}")
        self.extension = extension


class FallbackExhaustedError(ConversionError):
    """All fallback conversion attempts failed."""

    def __init__(self, file_path: Path, errors: list[Exception]) -> None:
        messages = [str(e) for e in errors]
        super().__init__(file_path, f"All conversion attempts failed: {messages}")
        self.errors = errors


class LLMError(MarkitError):
    """LLM-related error."""

    pass


class RateLimitError(LLMError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: int | None = None) -> None:
        self.retry_after = retry_after
        message = f"Rate limited, retry after {retry_after}s" if retry_after else "Rate limited"
        super().__init__(message)


class ProviderNotFoundError(LLMError):
    """No valid LLM provider available."""

    def __init__(self, message: str = "No valid LLM provider available") -> None:
        super().__init__(message)


class ImageProcessingError(MarkitError):
    """Error during image processing."""

    pass


class StateError(MarkitError):
    """State management error."""

    pass


class ConfigurationError(MarkitError):
    """Configuration error."""

    pass
