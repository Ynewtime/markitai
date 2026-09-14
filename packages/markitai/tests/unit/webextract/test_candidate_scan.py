"""Shared entry-point scans must preserve independent CSS query semantics."""

from typing import Any, cast
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup, Tag

from markitai.webextract import scoring, selectors


def identities(groups):
    return [[id(node) for node in group] for group in groups]


@pytest.mark.parametrize(
    "queries",
    [
        scoring.ENTRY_POINT_SELECTORS,
        (".post, #post", "#post", "article", "*", "[role]", "#post"),
        ("article > p", ":scope > article", "p:has(a)", "p:nth-child(1)"),
        ("[role='MAIN' i]", ":not(p)", "article:is(.post, #post)"),
        (),
    ],
)
@pytest.mark.parametrize("parser", ["lxml", "xml"])
def test_shared_selection_matches_css_and_live_mutations(queries, parser):
    soup = BeautifulSoup(
        '<main role="main"><article id="post" class="post post-content">'
        '<p><a>hello</a></p><p>world</p></article><article class="post">'
        "<p>other</p></article></main>",
        parser,
    )
    article = soup.article
    assert isinstance(article, Tag)
    for mutation in range(4):
        if mutation == 1:
            article["class"] = "post post-content"
        elif mutation == 2:
            cast(Any, article.attrs).update(id=42, role=["main", "article"])
        elif mutation == 3:
            article.extract()
        assert identities(selectors.select_many(soup, queries)) == identities(
            [soup.select(query) for query in queries]
        )


def test_entry_points_share_one_dom_traversal():
    soup = BeautifulSoup(
        '<main><article id="post" class="post post-content"><p>text</p>'
        "</article></main>",
        "lxml",
    )
    expected = [soup.select(query) for query in scoring.ENTRY_POINT_SELECTORS]
    original = Tag.descendants.fget
    assert original is not None
    visits = []

    def descendants(node):
        visits.append(id(node))
        return original(node)

    with patch.object(Tag, "descendants", property(descendants)):
        actual = selectors.select_many(soup, scoring.ENTRY_POINT_SELECTORS)
    assert identities(actual) == identities(expected)
    assert visits == [id(soup)]


@pytest.mark.parametrize(
    "attributes",
    [
        {"class": ["post", "post"], "role": None},
        {"class": ("post", "post-content"), "id": None},
        {"class": None, "id": ["post"]},
        {"role": 42, "CLASS": "post"},
        {"data-x": ["a", 2], "role": ["main", "article"]},
    ],
)
def test_mutated_attributes_and_scoped_roots_keep_reference_semantics(attributes):
    soup = BeautifulSoup("<main><article><p>body</p></article></main>", "lxml")
    article = soup.article
    assert isinstance(article, Tag)
    article.attrs.update(attributes)
    queries = (*scoring.ENTRY_POINT_SELECTORS, "[role]", "[data-x]", "*")
    for root in (soup, soup.main, soup.article):
        assert isinstance(root, Tag)
        assert identities(selectors.select_many(root, queries)) == identities(
            [root.select(query) for query in queries]
        )


def test_candidate_score_is_reused_only_within_one_selection():
    soup = BeautifulSoup(
        '<main><article id="post" class="post post-content article-body">'
        "<p>" + "Content, " * 80 + "</p></article></main>",
        "lxml",
    )
    original = scoring.score_candidate
    article = soup.article
    assert isinstance(article, Tag)
    paragraph = article.p
    assert isinstance(paragraph, Tag)
    for _ in range(2):
        visited = []

        def score(node, visited=visited):
            visited.append(id(node))
            return original(node)

        with patch.object(scoring, "score_candidate", side_effect=score):
            assert scoring.select_best_candidate(soup) is soup.article
        assert len(visited) == len(set(visited))
        assert id(soup.article) in visited
        paragraph.append(" Fresh text" * 30)
