"""User-facing error messages must not leak useless exception class names.

A message like ``OCRBackendMissing: --ocr requires ... Install with: ...`` is
worse than the same sentence without the prefix: the class name adds nothing
when the message already states the problem and the fix. Errors that opt in
via :class:`markitai.utils.errors.SelfExplanatoryError` render bare.

The flip side matters just as much: an *unexpected* exception's class name is
the best diagnostic hint available, so anything outside the marker hierarchy
must keep its prefix, and the DEBUG log must retain the full type and
traceback that the user-facing string drops.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from loguru import logger

from markitai.config import MarkitaiConfig, OCRConfig
from markitai.ocr import OCR_INSTALL_HINT, OCRBackendMissing, OCRProcessor
from markitai.security import validate_file_size
from markitai.utils.errors import (
    ConversionError,
    FileTooLargeError,
    MissingDependencyError,
    SelfExplanatoryError,
)
from markitai.utils.text import format_error_message
from markitai.workflow.core import (
    ConversionContext,
    convert_document,
    validate_and_detect_format,
)


@pytest.fixture
def ocr_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an environment where the `ocr` extra was never installed."""
    monkeypatch.delitem(sys.modules, "rapidocr", raising=False)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None):
        if name == "rapidocr" or name.startswith("rapidocr."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def _context_for(path: Path, tmp_path: Path) -> ConversionContext:
    return ConversionContext(
        input_path=path,
        output_dir=tmp_path / "output",
        config=MarkitaiConfig(),
    )


class TestMissingOcrBackendMessage:
    """The real OCR dead end renders as one actionable sentence."""

    def teardown_method(self) -> None:
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_message_has_install_hint_but_no_class_name(
        self, ocr_missing: None
    ) -> None:
        with pytest.raises(ImportError) as excinfo:
            OCRProcessor._create_engine_impl(OCRConfig(enabled=True))

        message = format_error_message(excinfo.value)
        assert "OCRBackendMissing" not in message
        assert "ImportError" not in message
        assert OCR_INSTALL_HINT in message

    def test_chained_backend_error_keeps_the_actionable_wording(self) -> None:
        """The root-cause walk must not trade the install hint for the terse
        ``No module named 'rapidocr'`` the error was raised from."""
        try:
            try:
                raise ModuleNotFoundError("No module named 'rapidocr'")
            except ImportError as inner:
                raise OCRBackendMissing(
                    f"--ocr requires the optional OCR backend (RapidOCR), "
                    f"which is not installed. {OCR_INSTALL_HINT}"
                ) from inner
        except OCRBackendMissing as e:
            message = format_error_message(e)

        assert OCR_INSTALL_HINT in message
        assert "OCRBackendMissing" not in message
        assert "No module named" not in message


class TestFileTooLargeMessage:
    """The size guard names the file and the limit — nothing else."""

    def test_message_has_no_class_name(self, tmp_path: Path) -> None:
        big = tmp_path / "big.txt"
        big.write_bytes(b"x" * 100)

        with pytest.raises(ValueError) as excinfo:
            validate_file_size(big, max_size_bytes=10)

        assert isinstance(excinfo.value, FileTooLargeError)
        message = format_error_message(excinfo.value)
        assert "FileTooLargeError" not in message
        assert "ValueError" not in message
        assert "File too large" in message

    def test_step_error_stays_bare_in_the_workflow(self, tmp_path: Path) -> None:
        big = tmp_path / "big.txt"
        big.write_bytes(b"x" * 100)

        result = validate_and_detect_format(_context_for(big, tmp_path), max_size=10)

        assert result.success is False
        assert result.error is not None
        assert result.error.startswith("File too large")


class TestUnsupportedFormatMessage:
    """The unsupported-format message is built bare and must stay bare."""

    def test_step_error_has_no_class_name(self, tmp_path: Path) -> None:
        odd = tmp_path / "file.xyz"
        odd.write_text("content")

        result = validate_and_detect_format(_context_for(odd, tmp_path), max_size=1024)

        assert result.success is False
        assert result.error is not None
        assert result.error.startswith("Unsupported file format")


class TestUnexpectedErrorsKeepClassName:
    """Anything outside the marker hierarchy keeps its diagnostic prefix."""

    def test_plain_valueerror_keeps_prefix(self) -> None:
        assert format_error_message(ValueError("boom")) == "ValueError: boom"

    def test_plain_runtimeerror_keeps_prefix(self) -> None:
        assert (
            format_error_message(RuntimeError("unexpected state"))
            == "RuntimeError: unexpected state"
        )

    def test_marker_with_empty_message_falls_back_to_class_name(self) -> None:
        class Odd(SelfExplanatoryError):
            pass

        assert format_error_message(Odd()) == "Odd"


class TestPreRenderedStepErrors:
    """A step error re-raised as an exception is not re-prefixed."""

    def test_conversion_error_renders_verbatim(self) -> None:
        step_error = (
            "Conversion failed: --ocr requires the optional OCR backend "
            f"(RapidOCR), which is not installed. {OCR_INSTALL_HINT}"
        )
        message = format_error_message(ConversionError(step_error))
        assert message == step_error

    def test_file_processor_rewraps_step_errors_as_conversion_error(self) -> None:
        """The single-file CLI path must not fall back to a plain
        ``RuntimeError(result.error)``, which would resurrect the
        ``RuntimeError:`` prefix in the final output."""
        import inspect

        from markitai.cli.processors import file as file_processor

        source = inspect.getsource(file_processor)
        assert "ConversionError(result.error)" in source
        assert "RuntimeError(result.error)" not in source


class TestConvertStepRendering:
    """convert_document flattens exceptions; check both sides of the line."""

    async def test_self_explanatory_error_renders_bare(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        ctx = _context_for(doc, tmp_path)
        ctx.converter = MagicMock()
        ctx.converter.convert.side_effect = MissingDependencyError(
            f"--ocr requires the optional OCR backend (RapidOCR), "
            f"which is not installed. {OCR_INSTALL_HINT}"
        )

        result = await convert_document(ctx)

        assert result.success is False
        assert result.error is not None
        assert "MissingDependencyError" not in result.error
        assert "ImportError" not in result.error
        assert OCR_INSTALL_HINT in result.error

    async def test_unexpected_error_keeps_class_name(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        ctx = _context_for(doc, tmp_path)
        ctx.converter = MagicMock()
        ctx.converter.convert.side_effect = ValueError("kaboom")

        result = await convert_document(ctx)

        assert result.success is False
        assert result.error is not None
        assert "ValueError: kaboom" in result.error

    async def test_debug_log_retains_type_and_traceback(self, tmp_path: Path) -> None:
        """The user-facing string drops the class name; the DEBUG log (the
        --verbose / log-file diagnosis path) must keep type and traceback."""
        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        ctx = _context_for(doc, tmp_path)
        ctx.converter = MagicMock()
        ctx.converter.convert.side_effect = MissingDependencyError(
            f"backend missing. {OCR_INSTALL_HINT}"
        )

        records = []
        sink_id = logger.add(lambda m: records.append(m), level="DEBUG")
        try:
            result = await convert_document(ctx)
        finally:
            logger.remove(sink_id)

        assert result.success is False
        with_exception = [
            m.record for m in records if m.record["exception"] is not None
        ]
        assert with_exception, "conversion failure must be logged with traceback"
        exc_types = {r["exception"].type for r in with_exception}
        assert MissingDependencyError in exc_types
