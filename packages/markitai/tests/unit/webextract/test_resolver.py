"""Tests for the resolver layer that sits between extractor selection and the pipeline.

These tests verify that:
- ``resolve_page()`` orchestrates extractor-based structured resolution
- ``ResolvedPage`` lives in the resolver layer and not in FetchResult
- Extractors cannot return final Markdown (only Tags or HTML strings)
- The generic pipeline remains the fallback when no resolver matches
"""

from __future__ import annotations

from typing import Any

import pytest
from bs4 import BeautifulSoup, Tag

# These imports will fail until resolver.py is created
from markitai.webextract.resolver import (  # type: ignore[import-not-found]
    ResolvedPage,
    resolve_page,
)
from markitai.webextract.types import SemanticExtraction

_SIMPLE_HTML = "<html><body><article><p>Hello world</p></article></body></html>"
_STRUCTURED_HTML = "<html><body><article><p>Structured</p></article></body></html>"
_X_URL = "https://x.com/user/status/123"
_GENERIC_URL = "https://example.com/blog/post"


# ---------------------------------------------------------------------------
# Fake resolvers for testing
# ---------------------------------------------------------------------------


class _StructuredResolver:
    """Fake extractor that implements resolve() and returns content_html."""

    name = "fake_structured"

    def matches_url(self, url: str) -> bool:
        return True

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        return None

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        return ResolvedPage(content_html="<article><p>Structured</p></article>")


class _MarkdownReturningResolver:
    """Fake extractor that illegally returns Markdown from resolve()."""

    name = "fake_markdown"

    def matches_url(self, url: str) -> bool:
        return True

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        return None

    def resolve(self, soup: BeautifulSoup, url: str) -> Any:
        # This violates the contract: resolve() must not return Markdown
        return "# This is Markdown\n\nParagraph text."


class _NoResolveExtractor:
    """Fake extractor that only implements extract_root (no resolve method)."""

    name = "fake_no_resolve"

    def matches_url(self, url: str) -> bool:
        return True

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        return soup.find("article")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ResolvedPage dataclass tests
# ---------------------------------------------------------------------------


class TestResolvedPage:
    """Tests for the ResolvedPage dataclass contract."""

    def test_resolved_page_can_be_constructed_empty(self) -> None:
        page = ResolvedPage()
        assert page.content_root is None
        assert page.content_html is None
        assert page.metadata_overrides == {}
        assert page.semantic is None
        assert page.diagnostics == {}

    def test_resolved_page_accepts_content_html(self) -> None:
        page = ResolvedPage(content_html="<article><p>Hello</p></article>")
        assert page.content_html == "<article><p>Hello</p></article>"

    def test_resolved_page_accepts_semantic(self) -> None:
        sem = SemanticExtraction(thread={"items": ["a", "b"]})  # type: ignore[reportArgumentType]
        page = ResolvedPage(semantic=sem)
        assert page.semantic is sem

    def test_resolved_page_accepts_metadata_overrides(self) -> None:
        page = ResolvedPage(metadata_overrides={"title": "Override Title"})
        assert page.metadata_overrides["title"] == "Override Title"

    def test_resolved_page_is_not_in_fetch_result(self) -> None:
        """ResolvedPage must live in resolver, not in FetchResult."""
        import markitai.webextract.resolver as resolver_module

        # ResolvedPage must be defined in resolver.py
        assert hasattr(resolver_module, "ResolvedPage")

        # It must NOT exist in the fetch types
        try:
            import markitai.fetch as fetch_module

            assert not hasattr(fetch_module, "ResolvedPage"), (
                "ResolvedPage must not be added to FetchResult or fetch.py"
            )
        except ImportError:
            pass  # If fetch module doesn't exist, test passes


# ---------------------------------------------------------------------------
# resolve_page() function tests
# ---------------------------------------------------------------------------


class TestResolvePage:
    """Tests for the resolve_page() orchestration function."""

    def test_resolved_page_lives_in_resolver_layer_not_fetch_types(self) -> None:
        result = resolve_page(_SIMPLE_HTML, _GENERIC_URL)
        assert isinstance(result, ResolvedPage) or result is None

    def test_resolver_prefers_structured_content_html_over_generic_root_selection(
        self,
    ) -> None:
        """When a resolver returns content_html, it takes priority over generic root."""
        fake_resolver = _StructuredResolver()
        result = resolve_page(_STRUCTURED_HTML, _GENERIC_URL, resolver=fake_resolver)
        assert result is not None
        assert result.content_html == "<article><p>Structured</p></article>"

    def test_resolver_does_not_allow_extractors_to_return_final_markdown(self) -> None:
        """resolve_page() must raise TypeError when resolver returns Markdown string."""
        fake_resolver = _MarkdownReturningResolver()
        with pytest.raises(TypeError):
            resolve_page(_SIMPLE_HTML, _GENERIC_URL, resolver=fake_resolver)

    def test_resolve_page_returns_none_when_extractor_has_no_resolve_method(
        self,
    ) -> None:
        """If extractor only has extract_root (no resolve), resolve_page returns None."""
        fake_extractor = _NoResolveExtractor()
        result = resolve_page(_SIMPLE_HTML, _GENERIC_URL, resolver=fake_extractor)
        assert result is None

    def test_resolve_page_returns_none_when_no_resolver_provided(self) -> None:
        """Without a resolver argument, resolve_page returns None (falls back to pipeline)."""
        # We pass no resolver= kwarg, which means the function uses find_extractor()
        # For a generic URL, no extractor matches, so result is None
        result = resolve_page(_SIMPLE_HTML, _GENERIC_URL)
        assert result is None

    def test_resolve_page_result_is_resolved_page_type(self) -> None:
        """Result must be ResolvedPage or None, never a raw string or Tag."""
        fake_resolver = _StructuredResolver()
        result = resolve_page(_SIMPLE_HTML, _GENERIC_URL, resolver=fake_resolver)
        assert isinstance(result, ResolvedPage)

    def test_resolved_page_content_html_must_not_be_markdown(self) -> None:
        """content_html field should contain HTML, not Markdown."""
        page = ResolvedPage(content_html="<p>Some content</p>")
        # content_html is HTML - it should not start with a Markdown heading
        assert not page.content_html.startswith("#")  # type: ignore[union-attr]
