"""Tests for single file processor — stdout mode asset reference handling."""

from __future__ import annotations

from pathlib import Path

from markitai.cli.processors.file import resolve_asset_references


class TestResolveAssetReferences:
    """resolve_asset_references() replaces image refs based on available tiers."""

    def test_placeholder_fallback_format(self) -> None:
        """Without protocol or store, produces ![image: filename]() placeholder."""
        markdown = "# Title\n\n![diagram](.markitai/assets/image1.png)\n\nText."
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))

        assert ".markitai/assets/" not in result
        assert "![image: image1.png]()" in result

    def test_strips_screenshot_reference(self) -> None:
        markdown = "![page1](.markitai/screenshots/doc.pdf.page0001.jpg)\n"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))

        assert ".markitai/screenshots/" not in result
        assert "![image: doc.pdf.page0001.jpg]()" in result

    def test_preserves_non_asset_content(self) -> None:
        markdown = "# Title\n\nSome text with [a link](https://example.com).\n"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert result == markdown

    def test_strips_multiple_references(self) -> None:
        markdown = (
            "![img1](.markitai/assets/a.png)\n"
            "Text between.\n"
            "![img2](.markitai/assets/b.jpg)\n"
            "![page](.markitai/screenshots/c.png)\n"
        )
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))

        assert "![image: a.png]()" in result
        assert "![image: b.jpg]()" in result
        assert "![image: c.png]()" in result

    def test_handles_empty_alt_text(self) -> None:
        markdown = "![](.markitai/assets/image.png)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert "![image: image.png]()" in result

    def test_preserves_external_urls(self) -> None:
        markdown = "![photo](https://example.com/photo.jpg)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert result == markdown

    def test_handles_no_images(self) -> None:
        markdown = "# Hello\n\nJust text."
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert result == markdown

    def test_handles_backslash_paths(self) -> None:
        """Windows-style backslash separators should also be matched."""
        markdown = "![img](.markitai\\assets\\image.png)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert "![image: image.png]()" in result

    def test_handles_mixed_separators(self) -> None:
        """Mixed forward/backslash separators should be matched."""
        markdown = "![img](.markitai/assets\\image.png)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert "![image: image.png]()" in result

    def test_alt_text_containing_assets_does_not_misroute_screenshots(
        self, tmp_path: Path
    ) -> None:
        """Alt text 'assets diagram' must not cause a screenshot to be looked up
        under .markitai/assets/ instead of .markitai/screenshots/."""
        # Set up a temp dir with only a screenshot file
        ss_dir = tmp_path / ".markitai" / "screenshots"
        ss_dir.mkdir(parents=True)
        (ss_dir / "page.png").write_bytes(b"screenshot-data")

        markdown = "![assets diagram](.markitai/screenshots/page.png)"
        from markitai.utils.asset_store import AssetStore

        store = AssetStore(tmp_path / "store")
        result = resolve_asset_references(
            markdown,
            temp_dir=tmp_path,
            asset_store=store,
            source_name="test.pdf",
        )

        # Should persist via asset store, NOT fall back to placeholder
        assert "![image:" not in result
        assert "file://" in result

    def test_asset_store_tier_persists_image(self, tmp_path: Path) -> None:
        """When asset_store is provided and image exists, should produce file:// URI."""
        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "chart.jpg").write_bytes(b"chart-data")

        from markitai.utils.asset_store import AssetStore

        store = AssetStore(tmp_path / "store")
        markdown = "![chart](.markitai/assets/chart.jpg)"
        result = resolve_asset_references(
            markdown,
            temp_dir=tmp_path,
            asset_store=store,
            source_name="doc.pdf",
        )

        assert "file://" in result
        assert "chart.jpg" in result
        assert "![image:" not in result  # not placeholder

    def test_protocol_tier_renders_escape_sequence(self, tmp_path: Path) -> None:
        """When protocol is provided and image exists, should produce escape sequence."""
        import io

        from PIL import Image

        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        # Create a real PNG file (Pillow needs valid image data)
        img = Image.new("RGB", (10, 10), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        (assets_dir / "img.png").write_bytes(buf.getvalue())

        from markitai.utils.terminal_image import Protocol

        markdown = "![diagram](.markitai/assets/img.png)"
        result = resolve_asset_references(
            markdown,
            temp_dir=tmp_path,
            protocol=Protocol.KITTY,
        )

        assert "\033_G" in result  # Kitty escape sequence
        assert "![image:" not in result  # not placeholder


import pytest

from markitai.config import MarkitaiConfig


class TestImageOnlySkip:
    """Tests for image-only format skip behavior."""

    @pytest.mark.asyncio
    async def test_image_file_skipped_without_llm_ocr(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Image file should be skipped when neither LLM nor OCR is enabled."""
        from markitai.cli.processors.file import process_single_file

        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        # Neither LLM nor OCR enabled (defaults)

        # Should return normally (no error, no SystemExit)
        await process_single_file(
            input_path=input_path,
            output_dir=output_dir,
            cfg=cfg,
            dry_run=False,
        )

        # No output file should be created
        output_files = list(output_dir.glob("*.md"))
        assert len(output_files) == 0

    @pytest.mark.asyncio
    async def test_image_file_skipped_stdout_mode(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Image file should be skipped in stdout mode too."""
        from markitai.cli.processors.file import process_single_file

        input_path = fixtures_dir / "sample.bmp"
        cfg = MarkitaiConfig()

        # stdout mode: output_dir=None. Should return normally.
        await process_single_file(
            input_path=input_path,
            output_dir=None,
            cfg=cfg,
            dry_run=False,
            quiet=True,  # quiet to suppress output
        )


class TestFinalOutputFileDetermination:
    """Tests for the final output file selection logic."""

    def test_llm_mode_prefers_llm_md(self, tmp_path: Path) -> None:
        """When .llm.md exists, it should be preferred in LLM mode."""
        output_file = tmp_path / "test.md"
        llm_file = tmp_path / "test.llm.md"
        output_file.write_text("base content")
        llm_file.write_text("llm content")

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        final = output_file.with_suffix(".llm.md") if cfg.llm.enabled else output_file
        if not final.exists() and cfg.llm.enabled:
            final = output_file

        assert final == llm_file
        assert final.read_text() == "llm content"

    def test_llm_mode_falls_back_to_md(self, tmp_path: Path) -> None:
        """When .llm.md doesn't exist (LLM failed), should fall back to .md."""
        output_file = tmp_path / "test.md"
        output_file.write_text("base content")
        # No .llm.md file

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        final = output_file.with_suffix(".llm.md") if cfg.llm.enabled else output_file
        if not final.exists() and cfg.llm.enabled:
            final = output_file

        assert final == output_file
        assert final.read_text() == "base content"

    def test_non_llm_uses_md(self, tmp_path: Path) -> None:
        """Without LLM, should always use .md."""
        output_file = tmp_path / "test.md"
        output_file.write_text("base content")

        cfg = MarkitaiConfig()

        final = output_file.with_suffix(".llm.md") if cfg.llm.enabled else output_file
        if not final.exists() and cfg.llm.enabled:
            final = output_file

        assert final == output_file


class TestStdoutPersistDefault:
    """stdout image persistence defaults and opt-out behavior."""

    def test_stdout_persist_defaults_to_true(self) -> None:
        """image.stdout_persist defaults to true so stdout links outlive temp dir."""
        from markitai.config import ImageConfig

        assert ImageConfig().stdout_persist is True

    @pytest.mark.asyncio
    async def test_opt_out_warns_links_ephemeral(self, tmp_path: Path) -> None:
        """stdout_persist=false emits a warning when output has asset refs."""
        from loguru import logger

        from markitai.cli.processors.file import process_single_file

        cfg = MarkitaiConfig()
        cfg.image.stdout_persist = False

        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n\n![img](.markitai/assets/doc.png)\n")

        messages: list[str] = []
        handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            await process_single_file(
                input_path=doc,
                output_dir=None,
                cfg=cfg,
                dry_run=False,
                quiet=False,
            )
        finally:
            logger.remove(handler_id)

        assert any("ephemeral" in m for m in messages), messages

    @pytest.mark.asyncio
    async def test_opt_out_warning_reaches_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The ephemeral warning is written to stderr directly (stdout-mode
        console handlers drop WARNING-level logs)."""
        from markitai.cli.processors.file import process_single_file

        cfg = MarkitaiConfig()
        cfg.image.stdout_persist = False

        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n\n![img](.markitai/assets/doc.png)\n")

        await process_single_file(
            input_path=doc,
            output_dir=None,
            cfg=cfg,
            dry_run=False,
            quiet=False,
        )

        assert "ephemeral" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_default_persist_no_ephemeral_warning(self, tmp_path: Path) -> None:
        """With persistence on (default), no ephemeral-links warning is emitted."""
        from loguru import logger

        from markitai.cli.processors.file import process_single_file

        cfg = MarkitaiConfig()
        # Redirect the asset store away from the user's home directory
        cfg.image.stdout_persist_dir = str(tmp_path / "store")

        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n\n![img](.markitai/assets/doc.png)\n")

        messages: list[str] = []
        handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            await process_single_file(
                input_path=doc,
                output_dir=None,
                cfg=cfg,
                dry_run=False,
                quiet=False,
            )
        finally:
            logger.remove(handler_id)

        assert not any("ephemeral" in m for m in messages), messages


class TestNormalizeTempAssetRefs:
    """normalize_temp_asset_refs rewrites absolute temp refs to relative."""

    def test_rewrites_absolute_temp_ref(self, tmp_path: Path) -> None:
        """Absolute refs into temp_dir become relative .markitai refs."""
        from markitai.cli.processors.file import normalize_temp_asset_refs

        markdown = f"![img]({tmp_path.as_posix()}/.markitai/assets/a.jpg)"
        result = normalize_temp_asset_refs(markdown, tmp_path)
        assert result == "![img](.markitai/assets/a.jpg)"

    def test_rewrites_resolved_symlink_form(self, tmp_path: Path) -> None:
        """Refs using the canonicalized (resolved) temp path are rewritten too.

        On macOS, tempfile returns /var/... while converters may canonicalize
        to /private/var/...; both spellings must normalize.
        """
        from markitai.cli.processors.file import normalize_temp_asset_refs

        resolved = tmp_path.resolve()
        markdown = f"![img]({resolved.as_posix()}/.markitai/assets/a.jpg)"
        result = normalize_temp_asset_refs(markdown, tmp_path)
        assert result == "![img](.markitai/assets/a.jpg)"

    def test_leaves_other_paths_alone(self, tmp_path: Path) -> None:
        """Unrelated absolute or relative refs are untouched."""
        from markitai.cli.processors.file import normalize_temp_asset_refs

        markdown = "![a](.markitai/assets/a.jpg) ![b](/elsewhere/b.jpg)"
        assert normalize_temp_asset_refs(markdown, tmp_path) == markdown
