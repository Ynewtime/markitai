"""Tests for output profiles (rag/obsidian/okf).

Covers the profile transforms in isolation, the CLI wiring, and the
downstream-acceptance properties the rag profile promises (visible assets,
resolvable relative references, parseable frontmatter) — asserted directly
on the file layout, without an ingestor dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from markitai.config import MarkitaiConfig
from markitai.output_profiles import (
    RAG_TABLE_PROMPT_RULES,
    __version__,
    apply_profile_to_file,
    assets_visible,
    extra_cleaning_rules,
    okf_frontmatter,
    relocate_analysis_asset_path,
    table_column_warnings,
    visible_asset_names,
)


def _config(profile: str | None = None, wikilinks: bool = False) -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.output.profile = profile  # type: ignore[assignment]
    cfg.output.wikilinks = wikilinks
    return cfg


class TestTableColumnWarnings:
    """Pipe-table column consistency detection (rag profile)."""

    def test_consistent_table_passes(self) -> None:
        markdown = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
        assert table_column_warnings(markdown) == []

    def test_inconsistent_row_is_reported(self) -> None:
        markdown = "| A | B |\n|---|---|\n| 1 | 2 | 3 |\n"
        warnings = table_column_warnings(markdown)
        assert len(warnings) == 1
        assert "2 header column(s)" in warnings[0]

    def test_alignment_delimiter_row_is_not_a_mismatch(self) -> None:
        markdown = "| A | B |\n| :--- | ---: |\n| 1 | 2 |\n"
        assert table_column_warnings(markdown) == []

    def test_tables_inside_code_fences_are_ignored(self) -> None:
        markdown = "```\n| A | B |\n| 1 | 2 | 3 |\n```\n"
        assert table_column_warnings(markdown) == []

    def test_escaped_pipes_do_not_count_as_cells(self) -> None:
        markdown = "| A | B |\n|---|---|\n| a \\| b | 2 |\n"
        assert table_column_warnings(markdown) == []

    def test_two_separate_tables_reported_separately(self) -> None:
        markdown = (
            "| A | B |\n| 1 |\n"  # first table, mismatch
            "\ntext\n\n"
            "| C | D |\n| 3 | 4 | 5 |\n"  # second table, mismatch
        )
        assert len(table_column_warnings(markdown)) == 2


class TestOkfFrontmatter:
    """markitai -> OKF frontmatter field mapping."""

    def test_full_mapping(self) -> None:
        result = okf_frontmatter(
            {
                "title": "Doc",
                "source": "sample.pdf",
                "description": "A doc",
                "tags": ["a", "b"],
                "markitai_processed": "2026-08-25T09:42:09.460+08:00",
                "fetch_strategy": "static",
            }
        )
        assert list(result)[0] == "type"
        assert result["type"] == "Document"
        assert result["title"] == "Doc"
        assert result["resource"] == "sample.pdf"
        assert result["description"] == "A doc"
        assert result["tags"] == ["a", "b"]
        assert result["generated"]["by"] == f"markitai/{__version__}"
        # Timestamp converted to the OKF UTC "Z" form
        assert result["generated"]["at"] == "2026-08-25T01:42:09Z"
        assert "markitai_processed" not in result
        assert "source" not in result
        # Fields without an OKF equivalent keep their names (spec allows
        # unknown keys)
        assert result["fetch_strategy"] == "static"

    def test_empty_frontmatter_still_gets_type_and_generated(self) -> None:
        result = okf_frontmatter({})
        assert result["type"] == "Document"
        assert result["generated"]["by"] == f"markitai/{__version__}"
        assert "at" not in result["generated"]

    def test_unparseable_timestamp_drops_at(self) -> None:
        result = okf_frontmatter({"markitai_processed": "not-a-date"})
        assert "at" not in result["generated"]


class TestHelpers:
    def test_assets_visible_by_profile(self) -> None:
        assert not assets_visible(_config(None))
        assert assets_visible(_config("rag"))
        assert assets_visible(_config("obsidian"))
        assert not assets_visible(_config("okf"))

    def test_extra_cleaning_rules_only_for_rag(self) -> None:
        assert extra_cleaning_rules(_config(None)) == ""
        assert extra_cleaning_rules(_config("obsidian")) == ""
        assert extra_cleaning_rules(_config("rag")) == RAG_TABLE_PROMPT_RULES
        assert "column" in RAG_TABLE_PROMPT_RULES.lower()

    def test_visible_asset_names_extracts_and_decodes(self) -> None:
        markdown = (
            "![a](assets/x.png)\n![b](assets/with%20space.jpg)\n![c](other/y.png)"
        )
        assert visible_asset_names(markdown) == ["x.png", "with space.jpg"]

    def test_relocate_analysis_asset_path(self) -> None:
        moved = relocate_analysis_asset_path("/out/.markitai/assets/x.png")
        assert moved == "/out/assets/x.png"
        untouched = relocate_analysis_asset_path("/out/elsewhere/x.png")
        assert untouched == "/out/elsewhere/x.png"


def _write_output(tmp_path: Path, body: str, frontmatter: str | None = None) -> Path:
    """Write a fake conversion output with one hidden asset."""
    assets = tmp_path / ".markitai" / "assets"
    assets.mkdir(parents=True)
    (assets / "img.png").write_bytes(b"png")
    md_file = tmp_path / "doc.pdf.md"
    content = body if frontmatter is None else f"---\n{frontmatter}\n---\n\n{body}"
    md_file.write_text(content, encoding="utf-8")
    return md_file


class TestApplyProfileToFile:
    def test_no_profile_leaves_file_and_assets_untouched(self, tmp_path: Path) -> None:
        md_file = _write_output(
            tmp_path, "![x](.markitai/assets/img.png)\n", "title: T"
        )
        before = md_file.read_bytes()
        apply_profile_to_file(md_file, tmp_path, _config(None))
        assert md_file.read_bytes() == before
        assert (tmp_path / ".markitai" / "assets" / "img.png").is_file()

    def test_rag_relocates_assets_and_rewrites_markers(self, tmp_path: Path) -> None:
        body = "<!-- Page number: 3 -->\n\n![x](.markitai/assets/img.png)\n"
        md_file = _write_output(tmp_path, body, "title: T")
        apply_profile_to_file(md_file, tmp_path, _config("rag"))
        content = md_file.read_text(encoding="utf-8")
        assert "<!-- page: 3 -->" in content
        assert "Page number" not in content
        assert "](assets/img.png)" in content
        assert ".markitai" not in content
        assert (tmp_path / "assets" / "img.png").is_file()
        # Emptied hidden dirs are pruned entirely
        assert not (tmp_path / ".markitai").exists()

    def test_rag_logs_table_warning(self, tmp_path: Path, caplog) -> None:
        from loguru import logger as loguru_logger

        body = "| A | B |\n|---|---|\n| 1 | 2 | 3 |\n"
        md_file = _write_output(tmp_path, body, "title: T")
        handler_id = loguru_logger.add(caplog.handler, level="WARNING")
        try:
            apply_profile_to_file(md_file, tmp_path, _config("rag"))
        finally:
            loguru_logger.remove(handler_id)
        assert any("pipe table" in r.message for r in caplog.records)
        # Detection only: the table itself is not rewritten
        assert "| 1 | 2 | 3 |" in md_file.read_text(encoding="utf-8")

    def test_obsidian_wikilinks_toggle(self, tmp_path: Path) -> None:
        body = "![Alt text](.markitai/assets/img.png)\n"
        md_file = _write_output(tmp_path, body, "title: T")
        apply_profile_to_file(md_file, tmp_path, _config("obsidian", wikilinks=True))
        content = md_file.read_text(encoding="utf-8")
        assert "![[assets/img.png|Alt text]]" in content

    def test_obsidian_without_wikilinks_keeps_standard_refs(
        self, tmp_path: Path
    ) -> None:
        body = "![Alt](.markitai/assets/img.png)\n"
        md_file = _write_output(tmp_path, body, "title: T")
        apply_profile_to_file(md_file, tmp_path, _config("obsidian"))
        content = md_file.read_text(encoding="utf-8")
        assert "![Alt](assets/img.png)" in content
        assert "![[" not in content

    def test_okf_rewrites_frontmatter_only(self, tmp_path: Path) -> None:
        body = "![x](.markitai/assets/img.png)\n"
        md_file = _write_output(
            tmp_path,
            body,
            "title: T\nsource: doc.pdf\nmarkitai_processed: '2026-01-01T00:00:00+00:00'",
        )
        apply_profile_to_file(md_file, tmp_path, _config("okf"))
        content = md_file.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content.split("---\n")[1])
        assert parsed["type"] == "Document"
        assert parsed["resource"] == "doc.pdf"
        assert parsed["generated"]["at"] == "2026-01-01T00:00:00Z"
        # Body and asset layout untouched
        assert "](.markitai/assets/img.png)" in content
        assert (tmp_path / ".markitai" / "assets" / "img.png").is_file()

    def test_okf_injects_frontmatter_when_missing(self, tmp_path: Path) -> None:
        md_file = _write_output(tmp_path, "Just a body\n")
        apply_profile_to_file(md_file, tmp_path, _config("okf"))
        content = md_file.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        parsed = yaml.safe_load(content.split("---\n")[1])
        assert parsed["type"] == "Document"

    def test_second_file_referencing_moved_asset(self, tmp_path: Path) -> None:
        """A sibling output referencing an already-moved asset still rewrites."""
        md_file = _write_output(tmp_path, "![x](.markitai/assets/img.png)\n", "t: 1")
        sibling = tmp_path / "doc2.pdf.md"
        sibling.write_text("![y](.markitai/assets/img.png)\n", encoding="utf-8")
        cfg = _config("rag")
        apply_profile_to_file(md_file, tmp_path, cfg)
        apply_profile_to_file(sibling, tmp_path, cfg)
        assert "](assets/img.png)" in sibling.read_text(encoding="utf-8")
        assert (tmp_path / "assets" / "img.png").is_file()


class TestWriteImagesJsonVisibleAssets:
    def test_entries_remapped_to_visible_dir(self, tmp_path: Path) -> None:
        from markitai.workflow.helpers import write_images_json
        from markitai.workflow.single import ImageAnalysisResult

        visible = tmp_path / "assets"
        visible.mkdir()
        hidden_path = tmp_path / ".markitai" / "assets" / "img.png"
        result = ImageAnalysisResult(
            source_file="doc.pdf",
            assets=[
                {
                    "asset": str(hidden_path),
                    "alt": "a",
                    "desc": "d",
                    "text": "",
                    "created": "2026-01-01T00:00:00+00:00",
                }
            ],
        )
        created = write_images_json(tmp_path, [result], visible_assets=True)
        assert created == [visible / "images.json"]
        import json

        data = json.loads((visible / "images.json").read_text(encoding="utf-8"))
        assert data["images"][0]["path"] == str(visible / "img.png")


class TestProfileCliEndToEnd:
    """End-to-end CLI conversions of the PDF fixture per profile."""

    @pytest.fixture
    def pdf_file(self, fixtures_dir: Path) -> Path:
        return fixtures_dir / "sample.pdf"

    def _convert(
        self, cli_runner: CliRunner, pdf_file: Path, out: Path, *args: str
    ) -> None:
        from markitai.cli import app

        result = cli_runner.invoke(
            app, [str(pdf_file), "-o", str(out), *args], catch_exceptions=False
        )
        assert result.exit_code == 0, result.output

    def test_default_output_is_unchanged(
        self, cli_runner: CliRunner, pdf_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        self._convert(cli_runner, pdf_file, out)
        content = (out / "sample.pdf.md").read_text(encoding="utf-8")
        assert "](.markitai/assets/" in content
        assert "<!-- Page number: 1 -->" in content
        assert not (out / "assets").exists()
        assert (out / ".markitai" / "assets").is_dir()

    def test_rag_profile_layout_is_ingestor_friendly(
        self, cli_runner: CliRunner, pdf_file: Path, tmp_path: Path
    ) -> None:
        """The acceptance properties behind the LlamaIndex recipe.

        SimpleDirectoryReader skips hidden paths, so the rag layout must
        contain no hidden directories, every image reference must resolve
        relative to the markdown file, and the frontmatter must parse.
        """
        out = tmp_path / "out"
        self._convert(cli_runner, pdf_file, out, "--profile", "rag")
        md_file = out / "sample.pdf.md"
        content = md_file.read_text(encoding="utf-8")

        # No hidden directories anywhere in the output tree
        hidden = [p for p in out.rglob("*") if p.name.startswith(".")]
        assert hidden == []

        # Every image reference resolves relative to the markdown file
        names = visible_asset_names(content)
        assert names, "expected image references in the converted PDF"
        for name in names:
            assert (out / "assets" / name).is_file()
        assert "](.markitai/" not in content

        # Page markers rewritten and aligned with the pymupdf page count
        assert "<!-- page: 1 -->" in content
        assert "<!-- Page number:" not in content

        # Frontmatter parses as YAML
        frontmatter = yaml.safe_load(content.split("---\n")[1])
        assert frontmatter["source"] == "sample.pdf"

    def test_okf_profile_frontmatter(
        self, cli_runner: CliRunner, pdf_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        self._convert(cli_runner, pdf_file, out, "--profile", "okf")
        content = (out / "sample.pdf.md").read_text(encoding="utf-8")
        frontmatter = yaml.safe_load(content.split("---\n")[1])
        assert frontmatter["type"] == "Document"
        assert frontmatter["resource"] == "sample.pdf"
        assert frontmatter["generated"]["by"] == f"markitai/{__version__}"
        assert frontmatter["generated"]["at"].endswith("Z")
        # okf does not relocate assets
        assert (out / ".markitai" / "assets").is_dir()
