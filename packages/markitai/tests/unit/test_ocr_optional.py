"""OCR is an opt-in capability: every dead end must name `markitai[ocr]`.

Moving `rapidocr` into an extra is only half the job. The failure modes a
user can now hit — asking for `--ocr` without the backend, or being told a
PDF looks scanned — must each hand back a command that fixes the situation,
instead of leaving the user to guess or to make two round trips.

"Not installed" is always simulated (monkeypatched import machinery); the
tests never uninstall anything.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from markitai.config import MarkitaiConfig, OCRConfig
from markitai.ocr import OCRProcessor

#: The one command every OCR dead end must print.
INSTALL_COMMAND = 'uv tool install "markitai[ocr]"'


def _command_module(name: str) -> ModuleType:
    """Import a command *module*, not the click command that shadows it.

    ``markitai.cli.commands.__init__`` re-exports each command object under
    its module's own name, so ``from markitai.cli.commands import doctor``
    yields a RichCommand and every ``hasattr`` assertion against it passes
    vacuously.
    """
    return importlib.import_module(f"markitai.cli.commands.{name}")


@pytest.fixture
def ocr_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an environment where the `ocr` extra was never installed."""
    import sys

    monkeypatch.delitem(sys.modules, "rapidocr", raising=False)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None):
        if name == "rapidocr" or name.startswith("rapidocr."):
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


@pytest.fixture
def ocr_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an environment where the `ocr` extra *is* installed."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package: str | None = None):
        if name == "rapidocr":
            return object()
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


class TestAvailabilityProbe:
    def test_reports_unavailable_without_the_extra(self, ocr_missing: None) -> None:
        from markitai.ocr import is_ocr_available

        assert is_ocr_available() is False

    def test_reports_available_with_the_extra(self, ocr_present: None) -> None:
        from markitai.ocr import is_ocr_available

        assert is_ocr_available() is True

    def test_probe_does_not_import_rapidocr(self, monkeypatch: pytest.MonkeyPatch):
        """Importing rapidocr costs seconds; the probe must stay metadata-only."""
        import sys

        from markitai.ocr import is_ocr_available

        monkeypatch.delitem(sys.modules, "rapidocr", raising=False)
        is_ocr_available()
        assert "rapidocr" not in sys.modules


class TestInstallHint:
    def test_hint_names_the_ocr_extra(self) -> None:
        from markitai.ocr import OCR_INSTALL_HINT

        assert INSTALL_COMMAND in OCR_INSTALL_HINT

    def test_hint_does_not_suggest_installing_rapidocr_directly(self) -> None:
        """`uv add rapidocr` puts the wheel somewhere markitai cannot see."""
        from markitai.ocr import OCR_INSTALL_HINT

        assert "uv add rapidocr" not in OCR_INSTALL_HINT


class TestEngineCreationError:
    def teardown_method(self) -> None:
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_missing_backend_raises_actionable_importerror(
        self, ocr_missing: None
    ) -> None:
        with pytest.raises(ImportError) as excinfo:
            OCRProcessor._create_engine_impl(OCRConfig(enabled=True))

        message = str(excinfo.value)
        assert INSTALL_COMMAND in message
        assert "--ocr" in message

    def test_shared_engine_surfaces_the_same_error(self, ocr_missing: None) -> None:
        with pytest.raises(ImportError) as excinfo:
            OCRProcessor.get_shared_engine(OCRConfig(enabled=True))

        assert INSTALL_COMMAND in str(excinfo.value)


class TestScannedPdfAdvisory:
    """The advisory must not send the user on a two-hop detour.

    Before: warn "consider --ocr" -> user adds --ocr -> ImportError -> user
    installs -> re-runs. The install command belongs in the first message.
    """

    @staticmethod
    def _converter() -> object:
        from markitai.converter.pdf import PdfConverter

        return PdfConverter(config=MarkitaiConfig())

    @staticmethod
    def _fixture() -> Path:
        return Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"

    def _warn(
        self, monkeypatch: pytest.MonkeyPatch, pages: tuple[list[int], list[int]]
    ) -> list[str]:
        from loguru import logger

        import markitai.converter.pdf as pdf_module

        monkeypatch.setattr(
            pdf_module, "collect_page_advisories", lambda _doc: pages, raising=False
        )
        captured: list[str] = []
        sink_id = logger.add(lambda message: captured.append(message), level="WARNING")
        try:
            self._converter()._warn_scanned_or_garbled(self._fixture())
        finally:
            logger.remove(sink_id)
        return captured

    def test_advisory_includes_install_command_when_ocr_missing(
        self, monkeypatch: pytest.MonkeyPatch, ocr_missing: None
    ) -> None:
        messages = self._warn(monkeypatch, ([1, 2], []))
        assert messages, "expected a scanned-page advisory"
        text = "".join(messages)
        assert "--ocr" in text
        assert INSTALL_COMMAND in text

    def test_advisory_omits_install_command_when_ocr_present(
        self, monkeypatch: pytest.MonkeyPatch, ocr_present: None
    ) -> None:
        messages = self._warn(monkeypatch, ([1, 2], []))
        assert messages
        text = "".join(messages)
        assert "--ocr" in text
        assert INSTALL_COMMAND not in text

    def test_no_advisory_when_no_pages_flagged(
        self, monkeypatch: pytest.MonkeyPatch, ocr_missing: None
    ) -> None:
        assert self._warn(monkeypatch, ([], [])) == []


class TestDoctorTreatsOcrAsOptional:
    def test_missing_ocr_is_reported_as_an_unused_optional_capability(
        self, ocr_missing: None
    ) -> None:
        from markitai.cli.commands.doctor import _check_rapidocr

        result = _check_rapidocr(MarkitaiConfig())
        assert result["status"] == "missing"
        assert result["optional"] is True
        assert INSTALL_COMMAND in result["install_hint"]

    def test_ocr_is_not_a_required_check(self) -> None:
        import inspect

        source = inspect.getsource(_command_module("doctor")._doctor_impl)
        assert 'required_deps = ["rapidocr"]' not in source

    def test_missing_ocr_does_not_fail_doctor(self, ocr_missing: None) -> None:
        """The exit-code contract: a bare install is healthy without OCR."""
        import json

        from click.testing import CliRunner

        from markitai.cli.commands.doctor import doctor

        result = CliRunner().invoke(doctor, ["--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["rapidocr"]["status"] == "missing"
        assert "ffmpeg" not in payload

    def test_rendered_install_hint_keeps_the_extra_name(
        self, ocr_missing: None
    ) -> None:
        """Rich reads `[ocr]` as a style tag and swallows it.

        Unescaped, `markitai[ocr]` reaches the terminal as plain `markitai` —
        a hint that installs the wrong thing. Also bites `[serve]` and
        `[browser]`, so the escape belongs at the render site, not in each
        string (the --json payload must stay free of markup escapes).
        """
        import json

        from click.testing import CliRunner

        from markitai.cli.commands.doctor import doctor

        rendered = CliRunner().invoke(doctor, []).output
        assert INSTALL_COMMAND in rendered, rendered

        payload = json.loads(CliRunner().invoke(doctor, ["--json"]).output)
        assert "\\[" not in payload["rapidocr"]["install_hint"]

    def test_suggest_extras_offers_ocr(self) -> None:
        """Guided-installer users keep the OCR they had when it was core."""
        from markitai.cli.commands.doctor import suggest_extras

        assert "ocr" in suggest_extras()


class TestFFmpegSurfaceRemoved:
    """markitai has never supported audio/video: EXTENSION_MAP has no such format."""

    def test_extension_map_really_has_no_media_formats(self) -> None:
        from markitai.converter.base import EXTENSION_MAP

        media = {".mp3", ".mp4", ".wav", ".m4a", ".mov", ".avi", ".flac", ".ogg"}
        assert media & set(EXTENSION_MAP) == set()

    def test_doctor_has_no_ffmpeg_check(self) -> None:
        assert not hasattr(_command_module("doctor"), "_check_ffmpeg")

    def test_doctor_install_hints_drop_ffmpeg(self) -> None:
        from markitai.cli.commands.doctor import INSTALL_HINTS

        assert "ffmpeg" not in INSTALL_HINTS

    def test_init_has_no_ffmpeg_check(self) -> None:
        assert not hasattr(_command_module("init"), "_check_ffmpeg_dep")

    def test_doctor_never_advertises_audio_video(self) -> None:
        for name in ("doctor", "init"):
            module = _command_module(name)
            source = Path(module.__file__).read_text(encoding="utf-8").lower()
            assert "ffmpeg" not in source, name
            assert "audio/video" not in source, name
