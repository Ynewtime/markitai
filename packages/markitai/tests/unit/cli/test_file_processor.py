"""Tests for single file processor — stdout mode asset reference handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

    def test_asset_store_tier_preserves_alt_text(self, tmp_path: Path) -> None:
        """LLM-generated alt text must survive the asset-store rewrite.

        Regression: the rewrite hardcoded the filename as alt, overwriting
        captions produced by image analysis in stdout mode.
        """
        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "chart.jpg").write_bytes(b"chart-data")

        from markitai.utils.asset_store import AssetStore

        store = AssetStore(tmp_path / "store")
        markdown = "![A bar chart of Q3 revenue](.markitai/assets/chart.jpg)"
        result = resolve_asset_references(
            markdown,
            temp_dir=tmp_path,
            asset_store=store,
            source_name="doc.pdf",
        )

        assert "![A bar chart of Q3 revenue](file://" in result

    def test_asset_store_resolves_uri_encoded_unicode_filename(
        self, tmp_path: Path
    ) -> None:
        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "截屏 下午.png").write_bytes(b"image-data")

        from markitai.utils.asset_store import AssetStore

        store = AssetStore(tmp_path / "store")
        markdown = (
            "![截图](.markitai/assets/%E6%88%AA%E5%B1%8F%20%E4%B8%8B%E5%8D%88.png)"
        )
        result = resolve_asset_references(
            markdown,
            temp_dir=tmp_path,
            asset_store=store,
            source_name="doc.png",
        )

        assert "![截图](file://" in result
        assert "![image:" not in result

    def test_asset_store_tier_uses_filename_for_empty_alt(self, tmp_path: Path) -> None:
        """Empty alt text falls back to the filename."""
        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "chart.jpg").write_bytes(b"chart-data")

        from markitai.utils.asset_store import AssetStore

        store = AssetStore(tmp_path / "store")
        markdown = "![](.markitai/assets/chart.jpg)"
        result = resolve_asset_references(
            markdown,
            temp_dir=tmp_path,
            asset_store=store,
            source_name="doc.pdf",
        )

        assert "![chart.jpg](file://" in result

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

        assert "![chart](file://" in result
        # Linked to the immutable content-addressed blob, not refs/
        assert "/blobs/" in result
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


class TestQuietDryRun:
    """Quiet single-file dry runs should be validation-only."""

    @pytest.mark.asyncio
    async def test_quiet_dry_run_suppresses_preview(
        self,
        tmp_path: Path,
        fixtures_dir: Path,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        from markitai.cli.processors.file import process_single_file

        output_dir = tmp_path / "output"
        with pytest.raises(SystemExit) as exc_info:
            await process_single_file(
                input_path=fixtures_dir / "sample.txt",
                output_dir=output_dir,
                cfg=MarkitaiConfig(),
                dry_run=True,
                quiet=True,
            )

        captured = capfd.readouterr()
        assert exc_info.value.code == 0
        assert captured.out == ""
        assert captured.err == ""
        assert not output_dir.exists()


class TestImageOnlySkip:
    """Tests for image-only format skip behavior."""

    @pytest.mark.asyncio
    async def test_image_file_skipped_without_llm_ocr(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """A single image without extraction features is an actionable failure."""
        from markitai.cli.processors.file import process_single_file

        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        # Neither LLM nor OCR enabled (defaults)

        with pytest.raises(SystemExit) as exc_info:
            await process_single_file(
                input_path=input_path,
                output_dir=output_dir,
                cfg=cfg,
                dry_run=False,
            )

        assert exc_info.value.code == 1

        # No output file should be created
        output_files = list(output_dir.glob("*.md"))
        assert len(output_files) == 0

    @pytest.mark.asyncio
    async def test_image_file_skipped_stdout_mode(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Stdout mode must not report success when no image payload exists."""
        from markitai.cli.processors.file import process_single_file

        input_path = fixtures_dir / "sample.bmp"
        cfg = MarkitaiConfig()

        with pytest.raises(SystemExit) as exc_info:
            await process_single_file(
                input_path=input_path,
                output_dir=None,
                cfg=cfg,
                dry_run=False,
                quiet=True,  # quiet to suppress diagnostics
            )

        assert exc_info.value.code == 1


class TestFinalOutputFileDetermination:
    """The reported output is the file the pipeline says it produced."""

    @staticmethod
    def _fake_core(*, llm_written: bool, cache_hit: bool = False) -> Any:
        from markitai.workflow.core import ConversionStepResult

        async def core(ctx: Any, _max_size: int) -> ConversionStepResult:
            ctx.output_dir.mkdir(parents=True, exist_ok=True)
            ctx.output_file = ctx.output_dir / f"{ctx.input_path.name}.md"
            if llm_written:
                ctx.llm_output_file = ctx.output_file.with_suffix(".llm.md")
                ctx.llm_output_file.write_text("# enhanced", encoding="utf-8")
                ctx.cache_hit = cache_hit
            return ConversionStepResult(success=True)

        return core

    @pytest.mark.asyncio
    async def test_llm_output_and_cache_hit_reach_the_outcome(
        self, tmp_path: Path
    ) -> None:
        """Regression: single-file --json always reported cache_hit false,
        even for a rerun served entirely from cache at zero cost."""
        from unittest.mock import patch

        from markitai.cli.processors.file import process_single_file
        from markitai.runs import Outcome

        doc = tmp_path / "doc.txt"
        doc.write_text("hello", encoding="utf-8")
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        history: list[Outcome] = []

        with patch(
            "markitai.workflow.core.convert_document_core",
            self._fake_core(llm_written=True, cache_hit=True),
        ):
            await process_single_file(
                input_path=doc,
                output_dir=tmp_path / "out",
                cfg=cfg,
                dry_run=False,
                quiet=True,
                history=history,
            )

        [outcome] = history
        assert outcome.status == "completed"
        assert outcome.output_path == tmp_path / "out" / "doc.txt.llm.md"
        assert outcome.cache_hit is True
        assert outcome.llm_cache_hit is True

    @pytest.mark.asyncio
    async def test_missing_llm_output_fails_instead_of_printing_a_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Regression: with no .llm.md the CLI fell back to a .md that did
        not exist either, printed its path and exited 0."""
        from unittest.mock import patch

        from markitai.cli.processors.file import process_single_file
        from markitai.runs import Outcome

        doc = tmp_path / "doc.txt"
        doc.write_text("hello", encoding="utf-8")
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        history: list[Outcome] = []

        with (
            patch(
                "markitai.workflow.core.convert_document_core",
                self._fake_core(llm_written=False),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await process_single_file(
                input_path=doc,
                output_dir=tmp_path / "out",
                cfg=cfg,
                dry_run=False,
                history=history,
            )

        assert exc_info.value.code == 1
        assert [o.status for o in history] == ["failed"]
        assert "doc.txt.md" not in capsys.readouterr().out


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

    def test_rewrites_absolute_visible_asset_ref(self, tmp_path: Path) -> None:
        """Absolute refs into temp_dir/assets (asset-visible profiles) too."""
        from markitai.cli.processors.file import normalize_temp_asset_refs

        markdown = f"![img]({tmp_path.resolve().as_posix()}/assets/a.jpg)"
        result = normalize_temp_asset_refs(markdown, tmp_path)
        assert result == "![img](assets/a.jpg)"


class TestVisibleAssetRefs:
    """Asset-visible profiles (rag/obsidian) write ``assets/`` refs in stdout
    mode; they point into the temp dir just like ``.markitai/`` refs.

    Regression: ASSET_REF_PATTERN only matched ``.markitai/...``, so under
    ``--profile rag|obsidian`` the stdout links were neither persisted nor
    rewritten and pointed at the deleted temp dir, without any warning.
    """

    @staticmethod
    def _asset(tmp_path: Path, name: str, data: bytes = b"img") -> None:
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / name).write_bytes(data)

    def test_pattern_detects_visible_and_wikilink_refs(self) -> None:
        from markitai.runs import ASSET_REF_PATTERN

        assert ASSET_REF_PATTERN.search("![x](assets/a.png)")
        assert ASSET_REF_PATTERN.search("![[assets/a.png]]")
        assert ASSET_REF_PATTERN.search("![[assets/a.png|cap]]")
        assert not ASSET_REF_PATTERN.search("![x](https://h/assets/a.png)")
        assert not ASSET_REF_PATTERN.search("![x](img/assets/a.png)")

    def test_visible_ref_persisted_to_store(self, tmp_path: Path) -> None:
        from markitai.utils.asset_store import AssetStore

        self._asset(tmp_path, "doc.pdf-0001-01.jpg", b"chart")
        store = AssetStore(tmp_path / "store")
        result = resolve_asset_references(
            "![A chart](assets/doc.pdf-0001-01.jpg)",
            temp_dir=tmp_path,
            asset_store=store,
            source_name="doc.pdf",
        )

        assert result.startswith("![A chart](file://")
        assert "/blobs/" in result
        assert "](assets/" not in result

    def test_escaped_alt_text_is_kept_verbatim(self, tmp_path: Path) -> None:
        from markitai.utils.asset_store import AssetStore

        self._asset(tmp_path, "a.jpg")
        store = AssetStore(tmp_path / "store")
        result = resolve_asset_references(
            r"![see \[1\]](assets/a.jpg)",
            temp_dir=tmp_path,
            asset_store=store,
        )

        assert result.startswith(r"![see \[1\]](file://")

    def test_wikilink_persisted_as_markdown_image(self, tmp_path: Path) -> None:
        """A wikilink cannot carry a file:// target: emit a markdown image."""
        from markitai.utils.asset_store import AssetStore

        self._asset(tmp_path, "a b.jpg", b"chart")
        store = AssetStore(tmp_path / "store")
        result = resolve_asset_references(
            "before ![[assets/a b.jpg| Fig *1* ]] after ![[assets/a b.jpg]]",
            temp_dir=tmp_path,
            asset_store=store,
        )

        assert "![[" not in result
        assert "before ![Fig *1*](file://" in result
        assert "![a b.jpg](file://" in result

    def test_visible_ref_without_store_becomes_placeholder(
        self, tmp_path: Path
    ) -> None:
        self._asset(tmp_path, "a.jpg")
        result = resolve_asset_references(
            "![x](assets/a.jpg) ![[assets/a.jpg]]", temp_dir=tmp_path
        )
        assert result == "![image: a.jpg]() ![image: a.jpg]()"

    def test_foreign_relative_assets_ref_left_untouched(self, tmp_path: Path) -> None:
        """A source document's own ``assets/`` image is not an extracted asset."""
        from markitai.utils.asset_store import AssetStore

        store = AssetStore(tmp_path / "store")
        markdown = "![logo](assets/logo.png)"
        result = resolve_asset_references(
            markdown, temp_dir=tmp_path, asset_store=store
        )
        assert result == markdown

    def test_same_named_documents_keep_their_own_images(self, tmp_path: Path) -> None:
        """Regression: stdout links pointed at the mutable refs/<name>/<file>
        symlink, so converting b/report.pdf after a/report.pdf silently
        swapped the image shown by the earlier output."""
        from markitai.utils.asset_store import AssetStore

        store = AssetStore(tmp_path / "store")
        outputs = []
        for sub, data in (("a", b"red"), ("b", b"blue")):
            temp = tmp_path / sub
            (temp / ".markitai" / "assets").mkdir(parents=True)
            (temp / ".markitai" / "assets" / "report.pdf-0001-01.jpg").write_bytes(data)
            outputs.append(
                resolve_asset_references(
                    "![](.markitai/assets/report.pdf-0001-01.jpg)",
                    temp_dir=temp,
                    asset_store=store,
                    source_name="report.pdf",
                )
            )

        from urllib.parse import unquote, urlparse

        def _target(md: str) -> Path:
            return Path(unquote(urlparse(md[md.index("(") + 1 : -1]).path))

        assert _target(outputs[0]).read_bytes() == b"red"
        assert _target(outputs[1]).read_bytes() == b"blue"


class TestStdoutProfileAssets:
    """End to end: stdout mode + asset-visible profile persists images."""

    @staticmethod
    def _pdf_with_image(path: Path) -> None:
        import io

        import pymupdf
        from PIL import Image

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Report with an image")
        buf = io.BytesIO()
        Image.new("RGB", (200, 120), "red").save(buf, format="PNG")
        page.insert_image(pymupdf.Rect(72, 100, 272, 220), stream=buf.getvalue())
        doc.save(path)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("profile", "wikilinks"),
        [("rag", False), ("obsidian", False), ("obsidian", True)],
    )
    async def test_profile_stdout_links_point_at_persisted_blobs(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        profile: str,
        wikilinks: bool,
    ) -> None:
        from urllib.parse import unquote, urlparse

        from markitai.cli.processors.file import process_single_file

        doc = tmp_path / "report.pdf"
        self._pdf_with_image(doc)
        cfg = MarkitaiConfig()
        cfg.output.profile = profile  # type: ignore[assignment]
        cfg.output.wikilinks = wikilinks
        cfg.image.stdout_persist_dir = str(tmp_path / "store")

        await process_single_file(
            input_path=doc, output_dir=None, cfg=cfg, dry_run=False, quiet=True
        )

        out = capsys.readouterr().out
        assert "](assets/" not in out
        assert "![[assets/" not in out
        [uri] = [
            line[line.index("](") + 2 : -1]
            for line in out.splitlines()
            if line.startswith("![")
        ]
        target = Path(unquote(urlparse(uri).path))
        assert target.parent == (tmp_path / "store" / "blobs").resolve()
        assert target.is_file()
