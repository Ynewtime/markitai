"""Tests for the macOS MS Office AppleScript fallback (utils/office_mac.py)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from markitai.config import MarkitaiConfig, OfficeConfig
from markitai.converter.legacy import LegacyOfficeConverter
from markitai.converter.office import PptxConverter
from markitai.utils import office_mac

# office_mac locks via fcntl and stages files with POSIX permission bits;
# neither exists on Windows, where the fallback is unreachable anyway.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="office_mac uses Unix-only fcntl and POSIX permissions",
)


@pytest.fixture(autouse=True)
def _clear_detection_cache():
    office_mac.find_ms_office_app.cache_clear()
    yield
    # monkeypatch may still hold a plain-function replacement at teardown
    cache_clear = getattr(office_mac.find_ms_office_app, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


class TestDetection:
    def test_non_darwin_returns_false(self) -> None:
        with patch("markitai.utils.office_mac.platform.system", return_value="Linux"):
            assert office_mac.find_ms_office_app("Microsoft Word") is False

    def test_darwin_app_present(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "Microsoft Word.app").mkdir()
        monkeypatch.setattr(office_mac, "_APP_SEARCH_BASES", (tmp_path,))
        with patch("markitai.utils.office_mac.platform.system", return_value="Darwin"):
            assert office_mac.find_ms_office_app("Microsoft Word") is True

    def test_darwin_app_missing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(office_mac, "_APP_SEARCH_BASES", (tmp_path,))
        with patch("markitai.utils.office_mac.platform.system", return_value="Darwin"):
            assert office_mac.find_ms_office_app("Microsoft Word") is False

    def test_legacy_app_available_unknown_suffix(self) -> None:
        assert office_mac.legacy_app_available(".txt") is False

    def test_xls_is_not_an_automation_format(self) -> None:
        # .xls converts in pure Python (xlrd via markitdown); Excel
        # automation was removed and must not come back silently.
        assert ".xls" not in office_mac.APP_BY_SUFFIX
        assert office_mac.legacy_app_available(".xls") is False


class TestScriptBuilding:
    def test_legacy_scripts_use_verified_enums(self) -> None:
        cases = {
            "Microsoft Word": "format document default",
            "Microsoft PowerPoint": "save as Open XML presentation",
        }
        for app, enum in cases.items():
            script = office_mac._build_legacy_script(
                app, Path("/tmp/in.doc"), Path("/tmp/out.docx")
            )
            assert enum in script
            assert 'tell application "' + app + '"' in script
            assert "close" in script and "saving no" in script
            assert "msoAutomationSecurityForceDisable" in script
            # Regression lock (openedItem-not-defined cascade): Word's and
            # PowerPoint's `open` returns nothing, so the script must never
            # bind from open's return value — it binds by staged name after
            # an existence poll, and cleanup never references the variable.
            assert "set openedItem to open" not in script
            assert 'set openedItem to document "' in script or (
                'set openedItem to presentation "' in script
            )
            assert "repeat until (exists" in script
            assert "active document" not in script
            assert "active presentation" not in script
            assert "on error errorMessage number errorNumber" in script
            assert "if openedItem is not missing value then" not in script
            # ForceDisable must never survive a cold-launch missing value read
            assert "msoAutomationSecurityByUI" in script

    def test_word_opens_without_links_or_recent_file_side_effects(self) -> None:
        script = office_mac._build_legacy_script(
            "Microsoft Word", Path("/tmp/in.doc"), Path("/tmp/out.docx")
        )
        assert "read only true" in script
        assert "add to recent files false" in script
        assert "set update links at open of settings to false" in script
        assert "set update links at open of settings to previousExtraSetting" in script

    def test_security_is_restored_before_save(self) -> None:
        script = office_mac._build_pdf_script(
            Path("/tmp/in.pptx"), Path("/tmp/out.pdf")
        )
        force = script.index("msoAutomationSecurityForceDisable")
        opened = script.index("open (POSIX file")
        bound = script.index("set openedItem to presentation ", opened)
        restored = script.index(
            "set automation security to previousAutomationSecurity", bound
        )
        saved = script.index("save openedItem", restored)
        assert force < opened < bound < restored < saved

    def test_pdf_script_uses_save_as_pdf(self) -> None:
        script = office_mac._build_pdf_script(
            Path("/tmp/in.pptx"), Path("/tmp/out.pdf")
        )
        assert "save as PDF" in script
        assert "msoAutomationSecurityForceDisable" in script
        assert "active presentation" not in script

    def test_unsupported_app_raises(self) -> None:
        with pytest.raises(ValueError):
            office_mac._build_legacy_script("Microsoft Outlook", Path("/a"), Path("/b"))

    def test_paths_are_escaped(self) -> None:
        script = office_mac._build_legacy_script(
            "Microsoft Word", Path('/tmp/we"ird.doc'), Path("/tmp/out.docx")
        )
        assert 'we\\"ird' in script


class TestRunAppleScript:
    def test_success(self) -> None:
        ok = MagicMock(returncode=0, stdout="", stderr="")
        with patch(
            "markitai.utils.office_mac.subprocess.run", return_value=ok
        ) as run_mock:
            office_mac._run_applescript("script", timeout=10, app="Microsoft Word")
        cmd = run_mock.call_args[0][0]
        assert cmd[0] == "osascript"
        assert "with timeout of 10 seconds" in cmd[2]

    def test_tcc_denial_maps_to_actionable_error(self) -> None:
        denied = MagicMock(
            returncode=1, stdout="", stderr="execution error: Not authorized. (-1743)"
        )
        with (
            patch("markitai.utils.office_mac.subprocess.run", return_value=denied),
            pytest.raises(RuntimeError, match="Automation"),
        ):
            office_mac._run_applescript("s", timeout=10, app="Microsoft Word")

    def test_generic_failure_includes_stderr(self) -> None:
        failed = MagicMock(returncode=1, stdout="", stderr="boom")
        with (
            patch("markitai.utils.office_mac.subprocess.run", return_value=failed),
            pytest.raises(RuntimeError, match="boom"),
        ):
            office_mac._run_applescript("s", timeout=10, app="Microsoft Word")

    def test_timeout_mentions_dialog(self) -> None:
        with (
            patch(
                "markitai.utils.office_mac.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="osascript", timeout=10),
            ),
            pytest.raises(RuntimeError, match="dialog"),
        ):
            office_mac._run_applescript("s", timeout=10, app="Microsoft Word")


class TestConvertLegacy:
    def test_happy_path_moves_output_and_cleans_staging(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        monkeypatch.setattr(office_mac, "_make_staging_dir", lambda: staging)
        monkeypatch.setattr(office_mac, "find_ms_office_app", lambda _app: True)

        def fake_run(script: str, *, timeout: int, app: str) -> None:
            staged_inputs = list(staging.glob("*.doc"))
            assert len(staged_inputs) == 1
            assert staged_inputs[0].name == f"{staging.name}.doc"
            assert staged_inputs[0].stat().st_mode & 0o777 == 0o400
            (staging / f"{staging.name}.docx").write_bytes(b"PK")

        monkeypatch.setattr(office_mac, "_run_applescript", fake_run)

        src = tmp_path / "sample.doc"
        src.write_bytes(b"legacy")
        output_dir = tmp_path / "out"

        result = office_mac.convert_legacy(src, "docx", output_dir)

        assert result == output_dir / "sample.docx"
        assert result.read_bytes() == b"PK"
        assert not staging.exists()

    def test_no_output_raises_and_cleans_staging(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        monkeypatch.setattr(office_mac, "_make_staging_dir", lambda: staging)
        monkeypatch.setattr(office_mac, "find_ms_office_app", lambda _app: True)
        monkeypatch.setattr(
            office_mac, "_run_applescript", lambda *_args, **_kwargs: None
        )

        src = tmp_path / "sample.doc"
        src.write_bytes(b"legacy")

        with pytest.raises(RuntimeError, match="did not produce"):
            office_mac.convert_legacy(src, "docx", tmp_path / "out")
        assert not staging.exists()

    def test_app_unavailable_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(office_mac, "find_ms_office_app", lambda _app: False)
        with pytest.raises(RuntimeError, match="No Microsoft Office app"):
            office_mac.convert_legacy(Path("x.doc"), "docx", Path("."))


class TestPptxToPdf:
    def test_happy_path(self, tmp_path: Path, monkeypatch) -> None:
        staging = tmp_path / "staging"
        staging.mkdir()
        monkeypatch.setattr(office_mac, "_make_staging_dir", lambda: staging)
        monkeypatch.setattr(office_mac, "find_ms_office_app", lambda _app: True)

        def fake_run(script: str, *, timeout: int, app: str) -> None:
            assert app == "Microsoft PowerPoint"
            (staging / f"{staging.name}.pdf").write_bytes(b"%PDF")

        monkeypatch.setattr(office_mac, "_run_applescript", fake_run)

        src = tmp_path / "deck.pptx"
        src.write_bytes(b"pptx")

        result = office_mac.pptx_to_pdf(src, tmp_path / "out")
        assert result == tmp_path / "out" / "deck.pdf"
        assert result.read_bytes() == b"%PDF"

    def test_powerpoint_missing_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(office_mac, "find_ms_office_app", lambda _app: False)
        with pytest.raises(RuntimeError, match="PowerPoint not found"):
            office_mac.pptx_to_pdf(Path("deck.pptx"), Path("."))


class TestStagingDir:
    def test_uses_group_container_when_present(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        container = tmp_path / "UBF8T346G9.Office"
        container.mkdir()
        monkeypatch.setattr(office_mac, "_OFFICE_GROUP_CONTAINER", container)
        monkeypatch.setattr(office_mac, "_STAGING_ROOT", container / "markitai")

        work = office_mac._make_staging_dir()
        try:
            assert work.parent == container / "markitai"
            assert work.is_dir()
            assert work.stat().st_mode & 0o777 == 0o700
        finally:
            import shutil

            shutil.rmtree(work, ignore_errors=True)

    def test_falls_back_to_tempdir_without_container(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(office_mac, "_OFFICE_GROUP_CONTAINER", tmp_path / "missing")
        monkeypatch.setattr(
            office_mac, "_FALLBACK_STAGING_ROOT", tmp_path / "private-fallback"
        )
        work = office_mac._make_staging_dir()
        try:
            assert work.is_dir()
            assert work.parent == tmp_path / "private-fallback"
            assert work.parent.stat().st_mode & 0o777 == 0o700
        finally:
            import shutil

            shutil.rmtree(work, ignore_errors=True)

    def test_purges_only_old_owned_uuid_directories(self, tmp_path: Path) -> None:
        root = tmp_path / "markitai"
        root.mkdir()
        stale = root / ("0" * 32)
        fresh = root / ("1" * 32)
        unrelated = root / "keep-me"
        stale.mkdir()
        fresh.mkdir()
        unrelated.mkdir()
        stale_time = time.time() - office_mac._STALE_STAGING_AGE_SECONDS - 10
        os.utime(stale, (stale_time, stale_time))

        office_mac._purge_stale_staging_dirs(root)

        assert not stale.exists()
        assert fresh.exists()
        assert unrelated.exists()


class TestOfficeAppLock:
    def test_uses_cross_process_flock(self, tmp_path: Path) -> None:
        with (
            patch("fcntl.flock") as flock_mock,
            office_mac._office_app_lock("Microsoft Word", root=tmp_path),
        ):
            pass

        import fcntl

        assert flock_mock.call_args_list[0].args[1] == fcntl.LOCK_EX
        assert flock_mock.call_args_list[-1].args[1] == fcntl.LOCK_UN
        assert (tmp_path / ".locks" / "word.lock").stat().st_mode & 0o777 == 0o600


class TestLegacyChainWiring:
    """_convert_legacy_format falls back to office_mac on macOS."""

    def _converter(self, config: MarkitaiConfig | None = None) -> LegacyOfficeConverter:
        converter = LegacyOfficeConverter(config)
        converter._soffice_path = None  # simulate LibreOffice absent
        return converter

    def test_darwin_uses_office_mac(self, tmp_path: Path) -> None:
        converter = self._converter()
        sentinel = tmp_path / "sample.docx"
        with (
            patch("platform.system", return_value="Darwin"),
            patch.object(office_mac, "legacy_app_available", return_value=True),
            patch.object(
                office_mac, "convert_legacy", return_value=sentinel
            ) as convert_mock,
        ):
            result = converter._convert_legacy_format(
                tmp_path / "sample.doc", "docx", tmp_path
            )
        assert result == sentinel
        convert_mock.assert_called_once()

    def test_darwin_office_missing_raises_with_both_options(
        self, tmp_path: Path
    ) -> None:
        converter = self._converter()
        with (
            patch("platform.system", return_value="Darwin"),
            patch.object(office_mac, "legacy_app_available", return_value=False),
            pytest.raises(
                RuntimeError, match="Install LibreOffice or Microsoft Office"
            ),
        ):
            converter._convert_legacy_format(tmp_path / "sample.doc", "docx", tmp_path)

    def test_darwin_fallback_disabled_by_config(self, tmp_path: Path) -> None:
        config = MarkitaiConfig(office=OfficeConfig(macos_fallback=False))
        converter = self._converter(config)
        with (
            patch("platform.system", return_value="Darwin"),
            patch.object(
                office_mac, "legacy_app_available", return_value=True
            ) as available_mock,
            pytest.raises(
                RuntimeError,
                match=r"enable office\.macos_fallback to use Microsoft Office",
            ),
        ):
            converter._convert_legacy_format(tmp_path / "sample.doc", "docx", tmp_path)
        available_mock.assert_not_called()

    def test_linux_error_unchanged(self, tmp_path: Path) -> None:
        converter = self._converter()
        with (
            patch("platform.system", return_value="Linux"),
            pytest.raises(RuntimeError, match="Install LibreOffice."),
        ):
            converter._convert_legacy_format(tmp_path / "sample.doc", "docx", tmp_path)


class TestPptxRenderWiring:
    """_render_slides_via_pdf uses PowerPoint PDF export when soffice is absent."""

    def test_darwin_renders_via_powerpoint_pdf(self, tmp_path: Path) -> None:
        pymupdf = pytest.importorskip("pymupdf")

        def fake_pptx_to_pdf(input_path: Path, output_dir: Path) -> Path:
            pdf_path = output_dir / f"{input_path.stem}.pdf"
            doc = pymupdf.open()
            doc.new_page(width=720, height=540)
            doc.save(pdf_path)
            doc.close()
            return pdf_path

        converter = PptxConverter(None)
        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()
        input_path = tmp_path / "deck.pptx"
        input_path.write_bytes(b"pptx")

        with (
            patch("platform.system", return_value="Darwin"),
            patch("markitai.converter.office.find_libreoffice", return_value=None),
            patch.object(office_mac, "powerpoint_available", return_value=True),
            patch.object(office_mac, "pptx_to_pdf", side_effect=fake_pptx_to_pdf),
        ):
            images, slide_infos = converter._render_slides_via_pdf(
                input_path, screenshots_dir, "jpg"
            )

        assert len(images) == 1
        assert len(slide_infos) == 1
        assert images[0].path.exists()

    def test_darwin_powerpoint_failure_returns_empty(self, tmp_path: Path) -> None:
        converter = PptxConverter(None)
        with (
            patch("platform.system", return_value="Darwin"),
            patch("markitai.converter.office.find_libreoffice", return_value=None),
            patch.object(office_mac, "powerpoint_available", return_value=True),
            patch.object(
                office_mac,
                "pptx_to_pdf",
                side_effect=RuntimeError("dialog pending"),
            ),
        ):
            images, slide_infos = converter._render_slides_via_pdf(
                tmp_path / "deck.pptx", tmp_path, "jpg"
            )
        assert images == []
        assert slide_infos == []

    def test_darwin_no_powerpoint_warns_and_returns_empty(self, tmp_path: Path) -> None:
        converter = PptxConverter(None)
        with (
            patch("platform.system", return_value="Darwin"),
            patch("markitai.converter.office.find_libreoffice", return_value=None),
            patch.object(office_mac, "powerpoint_available", return_value=False),
        ):
            images, slide_infos = converter._render_slides_via_pdf(
                tmp_path / "deck.pptx", tmp_path, "jpg"
            )
        assert images == []
        assert slide_infos == []

    def test_disabled_fallback_message_does_not_claim_powerpoint_missing(
        self, tmp_path: Path
    ) -> None:
        config = MarkitaiConfig(office=OfficeConfig(macos_fallback=False))
        converter = PptxConverter(config)
        with (
            patch("platform.system", return_value="Darwin"),
            patch("markitai.converter.office.find_libreoffice", return_value=None),
            patch.object(
                office_mac, "powerpoint_available", return_value=True
            ) as available_mock,
            patch("markitai.converter.office.logger.warning") as warning_mock,
        ):
            images, slide_infos = converter._render_slides_via_pdf(
                tmp_path / "deck.pptx", tmp_path, "jpg"
            )

        assert images == []
        assert slide_infos == []
        available_mock.assert_not_called()
        message = warning_mock.call_args.args[0]
        assert "office.macos_fallback is disabled" in message
        assert "Neither LibreOffice nor Microsoft PowerPoint found" not in message


class TestDoctorFallbackMessage:
    def test_darwin_reports_office_fallback_as_warning(self) -> None:
        from markitai.cli.commands.doctor import _check_libreoffice

        with (
            patch("sys.platform", "darwin"),
            patch("markitai.utils.office.find_libreoffice", return_value=None),
            patch("markitai.utils.office_mac.find_ms_office_app", return_value=True),
        ):
            result = _check_libreoffice()

        assert result["status"] == "warning"
        assert "MS Office fallback available" in result["message"]
        assert "Word, PowerPoint" in result["message"]
        assert "Excel" not in result["message"]

    def test_darwin_without_office_stays_missing(self) -> None:
        from markitai.cli.commands.doctor import _check_libreoffice

        with (
            patch("sys.platform", "darwin"),
            patch("markitai.utils.office.find_libreoffice", return_value=None),
            patch("markitai.utils.office_mac.find_ms_office_app", return_value=False),
        ):
            result = _check_libreoffice()

        assert result["status"] == "missing"
        assert result["message"] == "soffice/libreoffice command not found"

    def test_disabled_fallback_does_not_claim_office_is_available(self) -> None:
        from markitai.cli.commands.doctor import _check_libreoffice

        with (
            patch("sys.platform", "darwin"),
            patch("markitai.utils.office.find_libreoffice", return_value=None),
            patch(
                "markitai.utils.office_mac.find_ms_office_app", return_value=True
            ) as office_probe,
        ):
            result = _check_libreoffice(False)

        assert result["status"] == "missing"
        assert "fallback available" not in result["message"]
        office_probe.assert_not_called()

    def test_linux_stays_missing(self) -> None:
        from markitai.cli.commands.doctor import _check_libreoffice

        with (
            patch("sys.platform", "linux"),
            patch("markitai.utils.office.find_libreoffice", return_value=None),
        ):
            result = _check_libreoffice()

        assert result["status"] == "missing"


class TestConfig:
    def test_office_config_defaults(self) -> None:
        config = MarkitaiConfig()
        assert config.office.macos_fallback is True

    def test_office_config_in_schema(self) -> None:
        import json

        schema_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "markitai"
            / "config.schema.json"
        )
        schema = json.loads(schema_path.read_text())
        assert "OfficeConfig" in schema["$defs"]
        assert schema["properties"]["office"]["$ref"] == "#/$defs/OfficeConfig"
