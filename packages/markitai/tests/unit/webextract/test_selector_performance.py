"""Differential contracts for the indexed simple-selector execution path."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import soupsieve
from bs4 import BeautifulSoup

from markitai.webextract.constants import (
    EXACT_SELECTORS_JOINED,
    FOOTNOTE_INLINE_REFERENCES,
    FOOTNOTE_LIST_SELECTORS,
)

FIXTURES = Path(__file__).parents[2] / "defuddle_fixtures/fixtures"


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.html")), ids=lambda p: p.stem)
@pytest.mark.parametrize(
    "selector",
    [EXACT_SELECTORS_JOINED, FOOTNOTE_INLINE_REFERENCES, FOOTNOTE_LIST_SELECTORS],
    ids=["removals", "footnote-references", "footnote-lists"],
)
def test_indexed_removal_selection_matches_css_reference(path, selector):
    from markitai.webextract.selectors import select

    root = BeautifulSoup(path.read_text(), "lxml")
    expected = root.select(selector)
    actual = select(root, selector)
    assert [id(el) for el in actual] == [id(el) for el in expected]


@pytest.mark.parametrize(
    "selector",
    [
        ".ad, #Header, [hidden]",
        "div[class*='cover-']",
        "input:not([type='checkbox'])",
        '[aria-hidden="true"]:not(math):not([class*="math" i])',
        '[rel="tag"]',
        '[aria-label="Close" i]',
        "p > a",
        ":scope > div",
        "input:checked",
        "div:has(span)",
        "[data-x='a,b']",
        "svg[data-icon]",
    ],
)
def test_selector_semantics_order_deduplication_and_limit(selector):
    from markitai.webextract.selectors import select

    root = BeautifulSoup(
        """<main><div class="ad cover-page" id="Header" hidden>
    <span>nested</span></div><input type="CHECKBOX" checked><input type="text">
    <p aria-hidden="true"><a rel="tag" href="#">tag</a></p>
    <math aria-hidden="true">x</math><div class="MATH" aria-hidden="true">x</div>
    <button aria-label="CLOSE">Close</button><p data-x="a,b">Text</p>
    <svg data-icon="x"></svg></main>""",
        "lxml",
    ).main
    assert root is not None
    for limit in [0, 1, 2]:
        assert [id(e) for e in select(root, selector, limit=limit)] == [
            id(e) for e in root.select(selector, limit=limit)
        ]


def test_simple_removals_avoid_general_css_tree_matching():
    from markitai.webextract.selectors import select

    root = BeautifulSoup('<main><p>body</p><div class="ad">noise</div></main>', "lxml")
    with patch.object(
        soupsieve.SoupSieve, "select", side_effect=AssertionError("general CSS scan")
    ):
        result = select(root, EXACT_SELECTORS_JOINED)
    assert len(result) == 1 and result[0].get_text() == "noise"


def test_relationship_selectors_match_only_indexed_candidates():
    from markitai.webextract.selectors import select

    root = BeautifulSoup(
        '<article><div><span class="reference">yes</span></div>'
        '<p><span class="reference">no</span></p>'
        + "<section>unrelated</section>" * 100
        + "</article>",
        "lxml",
    ).article
    assert root is not None
    selector = "div > span.reference"
    expected = root.select(selector)
    with (
        patch.object(
            soupsieve.SoupSieve, "select", side_effect=AssertionError("full CSS scan")
        ),
        patch.object(
            soupsieve.SoupSieve,
            "match",
            autospec=True,
            side_effect=soupsieve.SoupSieve.match,
        ) as match,
    ):
        actual = select(root, selector)
    assert [id(e) for e in actual] == [id(e) for e in expected]
    assert match.call_count == 2


@pytest.mark.parametrize(
    "selector",
    [
        "article div > span.reference",
        "div + p span",
        "div ~ p > span",
        "div:not(.excluded) span, p span",
        "p:not(div > p) span",
        "body span, #extra",
        ":scope span.reference",
        "div span:has(a)",
        "p > span:nth-child(1)",
    ],
)
def test_relationship_candidates_keep_context_order_and_mutations(selector):
    from markitai.webextract.selectors import select

    root = BeautifulSoup(
        '<body><article><div><span class="reference"><a>reference</a></span></div>'
        '<p><span>next</span></p><p><span id="extra">last</span></p>'
        "</article></body>",
        "lxml",
    ).article
    assert root is not None
    for limit in (0, 1, 2):
        assert [id(e) for e in select(root, selector, limit=limit)] == [
            id(e) for e in root.select(selector, limit=limit)
        ]
    div, paragraph, span = root.div, root.p, root.span
    assert div is not None and paragraph is not None and span is not None
    cast(Any, div)["class"] = ["excluded"]
    paragraph.extract()
    span["class"] = "reference new"
    assert [id(e) for e in select(root, selector)] == [
        id(e) for e in root.select(selector)
    ]


def test_mutations_are_visible_and_xml_keeps_reference_semantics():
    from markitai.webextract.selectors import select

    for parser in ["lxml", "xml"]:
        root = BeautifulSoup(
            '<main><p id="X">body</p><p id="Y">body</p></main>', parser
        )
        assert select(root, "#X")[0] is root.p
        first = root.p
        assert first is not None
        first["id"] = "Z"
        assert select(root, "#X") == []
        first.decompose()
        assert [id(e) for e in select(root, "p")] == [id(e) for e in root.select("p")]
        second = root.p
        assert second is not None
        second.name = "P"
        assert [id(e) for e in select(root, "p")] == [id(e) for e in root.select("p")]


def test_candidate_selection_is_reused_without_sharing_mutable_retry_trees():
    from markitai.webextract import pipeline

    html = "<body><article><p>A short complete note.</p></article><p>Outside context remains available.</p></body>"
    with patch.object(pipeline, "_pick_root", wraps=pipeline._pick_root) as pick:
        result = pipeline.extract_web_content(html, "https://example.com")
    assert "A short complete note." in result.markdown
    assert "Outside context remains available." in result.markdown
    assert pick.call_count == 1


def test_clean_short_note_does_not_repeat_unchanged_removal_passes():
    from markitai.webextract import pipeline

    with patch.object(
        pipeline, "_extract_once", wraps=pipeline._extract_once
    ) as extract:
        result = pipeline.extract_web_content(
            "<body><article><p>A short complete note.</p></article></body>",
            "https://example.com",
        )
    assert result.markdown == "A short complete note."
    # One selected root plus, at most, a broader body fallback. Disabling
    # stages that removed nothing cannot recover any additional content.
    assert extract.call_count <= 2


def test_short_body_does_not_repeat_an_identical_body_fallback():
    from markitai.webextract import pipeline

    with patch.object(
        pipeline, "_extract_once", wraps=pipeline._extract_once
    ) as extract:
        result = pipeline.extract_web_content(
            "<body><p>A brief useful note.</p></body>", "https://example.com"
        )
    assert result.markdown == "A brief useful note."
    assert extract.call_count == 1


def test_body_retry_runs_relaxed_removals_only_once():
    from markitai.webextract import pipeline

    with patch.object(
        pipeline, "_extract_once", wraps=pipeline._extract_once
    ) as extract:
        result = pipeline.extract_web_content(
            '<body>Small note. <div class="related-content">A longer related '
            "note, more words than the initial note.</div></body>",
            "https://example.com",
        )
    assert "A longer related note" in result.markdown
    relaxed_body_calls = [
        call
        for call in extract.call_args_list
        if call.args[0].name == "body"
        and call.kwargs.get("use_scoring") is False
        and call.kwargs.get("use_content_patterns") is False
    ]
    assert len(relaxed_body_calls) == 1


def test_custom_extractor_body_still_retries_the_unmodified_original():
    from types import SimpleNamespace

    from markitai.webextract import pipeline

    def extract_root(soup):
        soup.find(id="extra").decompose()
        return soup.body

    extractor = SimpleNamespace(name="custom", extract_root=extract_root)
    with patch.object(pipeline, "find_extractor", return_value=extractor):
        result = pipeline.extract_web_content(
            '<body><p>A brief useful note.</p><p id="extra">Additional context '
            "from the original body remains available.</p></body>",
            "https://example.com",
        )
    assert "A brief useful note." in result.markdown
    assert (
        "Additional context from the original body remains available."
        in result.markdown
    )
    assert result.diagnostics["retry_level"] == "body_fallback"
