"""Batch-scoped output-name reservations (utils.output).

Items of a batch resolve their output name before their conversion or LLM
call is awaited and write it only afterwards, so a disk-only conflict check
let two items that derive the same name both take it: two ``.urls`` entries
``page?a=1`` / ``page?a=2`` under ``--llm`` (which writes no base ``.md`` to
hold the name) shared one ``.llm.md``, and ``report.pdf`` next to a URL for
``http://host/report.pdf`` shared ``report.pdf.md`` — one result silently
lost.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from markitai.utils.output import (
    OutputNameReservations,
    output_claim_scope,
    resolve_item_output_path,
)


class TestClaim:
    def test_second_claim_of_one_name_is_renamed(self, tmp_path: Path) -> None:
        table = OutputNameReservations()
        base = tmp_path / "page.md"

        assert table.claim(base, "rename") == base
        assert table.claim(base, "rename") == tmp_path / "page.v2.md"
        assert table.claim(base, "rename") == tmp_path / "page.v3.md"

    @pytest.mark.parametrize("on_conflict", ["skip", "overwrite"])
    def test_batch_siblings_are_renamed_whatever_the_strategy(
        self, tmp_path: Path, on_conflict: str
    ) -> None:
        """skip/overwrite apply to earlier runs, never to a sibling item."""
        table = OutputNameReservations()
        base = tmp_path / "page.md"

        assert table.claim(base, on_conflict) == base
        assert table.claim(base, on_conflict) == tmp_path / "page.v2.md"

    def test_md_and_llm_md_are_one_name(self, tmp_path: Path) -> None:
        table = OutputNameReservations()
        table.reserve(tmp_path / "report.pdf.llm.md")

        assert table.claim(tmp_path / "report.pdf.md", "rename") == (
            tmp_path / "report.pdf.v2.md"
        )

    def test_disk_conflicts_still_follow_on_conflict(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("earlier run")
        table = OutputNameReservations()

        assert table.claim(tmp_path / "a.md", "skip") is None
        assert table.claim(tmp_path / "a.md", "overwrite") == tmp_path / "a.md"
        assert table.claim(tmp_path / "a.md", "rename") == tmp_path / "a.v2.md"

    def test_reuse_claims_the_items_own_earlier_output(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("earlier")
        (tmp_path / "a.v2.md").write_text("this item's interrupted output")
        table = OutputNameReservations()

        assert (
            table.claim(tmp_path / "a.md", "rename", reuse=tmp_path / "a.v2.md")
            == tmp_path / "a.v2.md"
        )

    def test_reuse_held_by_a_sibling_falls_back_to_resolution(
        self, tmp_path: Path
    ) -> None:
        table = OutputNameReservations()
        table.reserve(tmp_path / "a.md")

        assert (
            table.claim(tmp_path / "a.md", "rename", reuse=tmp_path / "a.md")
            == tmp_path / "a.v2.md"
        )

    def test_case_insensitive_directory_folds_case(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.utils.output._probe_case_insensitive", lambda _d: True
        )
        table = OutputNameReservations()

        assert table.claim(tmp_path / "Report.md", "rename") == tmp_path / "Report.md"
        assert table.claim(tmp_path / "report.md", "rename") == (
            tmp_path / "report.v2.md"
        )

    def test_case_sensitive_directory_keeps_case(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.utils.output._probe_case_insensitive", lambda _d: False
        )
        table = OutputNameReservations()

        assert table.claim(tmp_path / "Report.md", "rename") == tmp_path / "Report.md"
        assert table.claim(tmp_path / "report.md", "rename") == tmp_path / "report.md"


class TestScope:
    def test_without_scope_it_is_the_plain_disk_check(self, tmp_path: Path) -> None:
        assert resolve_item_output_path(tmp_path / "a.md", "rename") == (
            tmp_path / "a.md"
        )
        assert resolve_item_output_path(tmp_path / "a.md", "rename") == (
            tmp_path / "a.md"
        )

    def test_scope_is_idempotent_and_reports_the_claim(self, tmp_path: Path) -> None:
        """The CLI URL worker and the cascade it delegates to both resolve."""
        table = OutputNameReservations()
        claimed: list[Path] = []
        with output_claim_scope(table, on_claimed=claimed.append):
            first = resolve_item_output_path(tmp_path / "a.md", "rename")
            second = resolve_item_output_path(tmp_path / "a.md", "rename")

        assert first == second == tmp_path / "a.md"
        assert claimed == [tmp_path / "a.md"]

    def test_per_item_override_redoes_over_the_default_name(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a.md").write_text("interrupted earlier")
        table = OutputNameReservations()
        with output_claim_scope(table, on_conflict="overwrite"):
            assert resolve_item_output_path(tmp_path / "a.md", "rename") == (
                tmp_path / "a.md"
            )

    @pytest.mark.asyncio
    async def test_concurrent_items_never_share_a_name(self, tmp_path: Path) -> None:
        """Resolve early, await, write late — the batch worker shape."""
        table = OutputNameReservations()

        async def item() -> Path:
            with output_claim_scope(table):
                resolved = resolve_item_output_path(tmp_path / "page.md", "rename")
                await asyncio.sleep(0.01)  # conversion / LLM call
                assert resolved is not None
                resolved.with_name(resolved.name[:-3] + ".llm.md").write_text("x")
                return resolved

        results = await asyncio.gather(*(item() for _ in range(5)))

        assert len(set(results)) == 5
        assert len(list(tmp_path.glob("*.llm.md"))) == 5


# ---------------------------------------------------------------------------
# Batch drivers wire the reservation table through files and URLs alike
# ---------------------------------------------------------------------------


class _Entry:
    def __init__(self, url: str, output_name: str | None = None) -> None:
        self.url = url
        self.output_name = output_name


def _fetch_result(url: str):
    from markitai.fetch_types import FetchResult

    return FetchResult(
        content=f"# Page\n\nBody of {url}\n",
        strategy_used="static",
        title="Page",
        url=url,
    )


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    async def _fetch(url, *args, **kwargs):
        return _fetch_result(url)

    monkeypatch.setattr("markitai.fetch.fetch_url", _fetch)
    monkeypatch.setattr(
        "markitai.fetch._get_playwright_renderer", AsyncMock(return_value=None)
    )


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the URL document LLM call; it writes <output>.llm.md."""
    from unittest.mock import MagicMock

    async def _document_llm(markdown, url, cfg, output_file, *args, **kwargs):
        await asyncio.sleep(0.01)  # both URLs are in flight at once
        output_file.with_suffix(".llm.md").write_text(f"LLM for {url}\n")
        return "", 0.0, {"m": {"requests": 1}}

    monkeypatch.setattr(
        "markitai.cli.processors.url.run_url_document_llm", _document_llm
    )
    monkeypatch.setattr(
        "markitai.workflow.helpers.create_llm_processor",
        MagicMock(return_value=MagicMock()),
    )


@pytest.mark.asyncio
async def test_url_list_llm_outputs_with_one_derived_name_both_survive(
    tmp_path: Path, fake_fetch: None, fake_llm: None
) -> None:
    """page?a=1 and page?a=2 derive one name; --llm writes no base .md."""
    from markitai.cli.processors.url import process_url_batch
    from markitai.config import MarkitaiConfig

    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.cache.enabled = False
    out = tmp_path / "out"
    urls = ["https://example.com/page?a=1", "https://example.com/page?a=2"]

    await process_url_batch(
        [_Entry(u) for u in urls], out, cfg, dry_run=False, verbose=False, quiet=True
    )

    written = sorted(out.glob("*.llm.md"))
    assert len(written) == 2
    assert {p.read_text() for p in written} == {f"LLM for {u}\n" for u in urls}


@pytest.mark.asyncio
async def test_directory_batch_file_and_url_never_share_an_output(
    tmp_path: Path, fake_fetch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """report.pdf and http://host/report.pdf both derive report.pdf.md."""
    import markitai.workflow.core as core
    from markitai.cli.processors.batch import process_batch
    from markitai.config import MarkitaiConfig

    real_convert = core.convert_document

    async def slow_convert(ctx):
        # The file has resolved its name; the URL resolves and writes while
        # the (thread-pooled) conversion is still running
        await asyncio.sleep(0.2)
        return await real_convert(ctx)

    async def slow_fetch(url, *args, **kwargs):
        await asyncio.sleep(0.05)
        return _fetch_result(url)

    monkeypatch.setattr(core, "convert_document", slow_convert)
    monkeypatch.setattr("markitai.fetch.fetch_url", slow_fetch)

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "report.txt").write_text("local file body")
    (docs / "more.urls").write_text("https://example.com/report.txt\n")
    out = tmp_path / "out"
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False

    await process_batch(docs, out, cfg, resume=False, dry_run=False, quiet=True)

    written = {p.name: p.read_text() for p in out.glob("*.md")}
    assert sorted(written) == ["report.txt.md", "report.txt.v2.md"]
    bodies = "\n".join(written.values())
    assert "local file body" in bodies
    assert "Body of https://example.com/report.txt" in bodies
