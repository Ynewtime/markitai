"""Tests for review-discovered bugs (P3-D review fixes + parity review)."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from markitai.webextract.dom import parse_html
from markitai.webextract.removals import apply_removals

FIXTURES = Path(__file__).parents[2] / "fixtures" / "web"


class TestAlertCalloutConflict:
    """ISSUE-5/6: .alert and aside in EXACT_SELECTORS kills callouts."""

    def test_bootstrap_alert_survives_removals_and_becomes_callout(self):
        """Bootstrap .alert should be converted to blockquote, not removed."""
        soup = parse_html(
            '<div><div class="alert alert-warning">'
            "<p>Warning: this is important.</p></div>"
            "<p>Main content here.</p></div>"
        )
        root = soup.find("div")
        apply_removals(root)
        # After removals, the alert content should still exist
        assert "important" in root.get_text()

    def test_aside_callout_survives_removals(self):
        """aside with callout class should not be removed by selectors."""
        soup = parse_html(
            '<div><aside class="callout callout-info">'
            "<p>Helpful note here.</p></aside>"
            "<p>Main content.</p></div>"
        )
        root = soup.find("div")
        apply_removals(root)
        assert "Helpful note" in root.get_text()


class TestMathProtection:
    """ISSUE-8: [aria-hidden] selector bypasses math protection."""

    def test_math_katex_survives_full_removal_pipeline(self):
        """KaTeX elements with aria-hidden should survive all removals."""
        soup = parse_html(
            "<div><p>The formula is "
            '<span class="katex" aria-hidden="true">x^2 + y^2</span></p></div>'
        )
        root = soup.find("div")
        apply_removals(root)
        assert "x^2" in root.get_text()

    def test_mathjax_survives_full_removal_pipeline(self):
        soup = parse_html(
            "<div><p>Equation: "
            '<span class="MathJax" aria-hidden="true">E=mc^2</span></p></div>'
        )
        root = soup.find("div")
        apply_removals(root)
        assert "E=mc" in root.get_text()

    def test_math_element_survives(self):
        soup = parse_html(
            "<div><p>See <math aria-hidden='true'><mi>x</mi></math></p></div>"
        )
        root = soup.find("div")
        apply_removals(root)
        assert root.find("math") is not None


class TestBodyFallbackFreshParse:
    """BUG-4: body fallback should use fresh HTML, not mutated soup."""

    def test_body_fallback_preserves_all_content(self):
        from markitai.webextract.pipeline import extract_web_content

        # Page where initial root is very short, triggering retry + body fallback.
        # The body has content that should NOT be stripped by Level 1 mutations.
        html = """
        <html><body>
            <div id="wrapper">
                <p>Short.</p>
            </div>
            <section>
                <p>This is extra content that lives outside the main wrapper
                and should be captured by the body fallback path with enough
                words to pass the sparse threshold.</p>
            </section>
        </body></html>
        """
        result = extract_web_content(html, "https://example.com/page")
        assert "extra content" in result.markdown


class TestSelectorIdentity:
    """ISSUE-13: el in to_remove should use identity, not equality."""

    def test_identical_elements_both_removed(self):
        """Two different elements with same content should both be removed."""
        soup = parse_html("<div><nav>links</nav><p>content</p><nav>links</nav></div>")
        root = soup.find("div")
        apply_removals(root)
        # Both nav elements should be removed
        assert root.find("nav") is None


class TestBylineRegex:
    """BUG-3: IGNORECASE makes [A-Z] meaningless in byline regex."""

    def test_byline_requires_capitalized_name(self):
        from markitai.webextract.removals.content_patterns import (
            remove_content_patterns,
        )

        # Lowercase "by someone" with date should NOT match
        soup = parse_html("<div><p>by someone on Jan 15</p><p>Real content.</p></div>")
        root = soup.find("div")
        remove_content_patterns(root)
        # Should not be removed (lowercase name)
        assert "by someone" in root.get_text()


# ---------------------------------------------------------------------------
# Parity review regression tests
# ---------------------------------------------------------------------------


class TestResolverSanitization:
    """High-1: Resolver path must sanitize/standardize HTML."""

    def test_resolver_output_does_not_leak_script_tags(self) -> None:
        """Script tags in resolver HTML must be stripped by sanitize_tag_tree."""
        from markitai.webextract.pipeline import _build_from_resolved
        from markitai.webextract.resolver import ResolvedPage

        resolved = ResolvedPage(
            content_html=(
                "<article><script>alert(1)</script><p>Hello world</p></article>"
            ),
            metadata_overrides={"title": "Test"},
        )
        result = _build_from_resolved(
            "<html><body></body></html>",
            "https://example.com/page",
            resolved,
        )
        assert "<script" not in result.clean_html

    def test_resolver_output_resolves_relative_links(self) -> None:
        """Relative links in resolver HTML must be resolved by standardize."""
        from markitai.webextract.pipeline import _build_from_resolved
        from markitai.webextract.resolver import ResolvedPage

        resolved = ResolvedPage(
            content_html='<article><a href="/about">About</a></article>',
            metadata_overrides={"title": "Test"},
        )
        result = _build_from_resolved(
            "<html><body></body></html>",
            "https://example.com/page",
            resolved,
        )
        assert "example.com/about" in result.clean_html


class TestContentRootContract:
    """High-2: ResolvedPage.content_root must be consumed by pipeline."""

    def test_content_root_resolver_produces_markdown(self) -> None:
        """A root-only ResolvedPage must produce markdown, not fallback."""
        from markitai.webextract.pipeline import _build_from_resolved
        from markitai.webextract.resolver import ResolvedPage

        soup = BeautifulSoup(
            "<article><p>Root content here</p></article>", "html.parser"
        )
        root = soup.find("article")

        resolved = ResolvedPage(
            content_root=root,
            metadata_overrides={"title": "Override Title"},
        )
        result = _build_from_resolved(
            "<html><body></body></html>",
            "https://example.com",
            resolved,
        )
        assert "Root content here" in result.markdown
        assert result.metadata.title == "Override Title"

    def test_content_root_triggers_resolver_path_in_extract_web_content(
        self,
    ) -> None:
        """extract_web_content gate condition must accept content_root."""
        from unittest.mock import patch

        from markitai.webextract.pipeline import extract_web_content
        from markitai.webextract.resolver import ResolvedPage

        root_soup = BeautifulSoup(
            "<article><p>Resolver root content</p></article>", "html.parser"
        )
        root_tag = root_soup.find("article")
        fake_resolved = ResolvedPage(
            content_root=root_tag,
            metadata_overrides={"title": "Root Override"},
        )

        with patch(
            "markitai.webextract.pipeline.resolve_page",
            return_value=fake_resolved,
        ):
            result = extract_web_content(
                "<html><body><p>Generic fallback</p></body></html>",
                "https://example.com",
            )

        # Must come from the resolver path, not the generic path
        assert "Resolver root content" in result.markdown
        assert "Generic fallback" not in result.markdown
        assert result.metadata.title == "Root Override"


class TestRedditNestedReplies:
    """High-3: Reddit nested replies with .child as sibling of .entry."""

    def test_real_reddit_fixture_captures_nested_reply(self) -> None:
        """Nested reply in real old Reddit structure must be collected."""
        from markitai.webextract.pipeline import extract_web_content

        html = (FIXTURES / "reddit_post.playwright.html").read_text(encoding="utf-8")
        result = extract_web_content(
            html,
            "https://old.reddit.com/r/rust/comments/abc123/whats_the_best_way/",
        )
        assert result.semantic is not None
        assert result.semantic.thread is not None
        ids = [item.id for item in result.semantic.thread.items]
        assert "t1_xyz002" in ids, f"Nested reply missing. Got: {ids}"
        nested = next(i for i in result.semantic.thread.items if i.id == "t1_xyz002")
        assert nested.parent_id == "t1_xyz001"

    def test_child_inside_entry_also_works(self) -> None:
        """Fallback: .child inside .entry should also collect nested replies."""
        from markitai.webextract.extractors.reddit_post import (
            _collect_comment_nodes,
        )
        from markitai.webextract.semantics import ConversationItem

        html = """
        <div class="sitetable nestedlisting">
          <div class="thing comment" data-fullname="t1_parent">
            <div class="entry">
              <p class="tagline"><a class="author">user1</a></p>
              <div class="usertext-body"><div class="md"><p>Parent</p></div></div>
              <div class="child">
                <div class="sitetable nestedlisting">
                  <div class="thing comment" data-fullname="t1_child">
                    <div class="entry">
                      <p class="tagline"><a class="author">user2</a></p>
                      <div class="usertext-body"><div class="md"><p>Child</p></div></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        root = soup.find("div", class_="sitetable")
        items: list[ConversationItem] = []
        _collect_comment_nodes(root, parent_id=None, items=items)  # type: ignore[arg-type]
        assert len(items) == 2
        assert items[1].parent_id == "t1_parent"


# ---------------------------------------------------------------------------
# Low-2: Thread policy filters third-party replies in X extraction
# ---------------------------------------------------------------------------


class TestXThreadPolicyFiltering:
    """X extractor must use thread policy to exclude third-party replies."""

    def test_x_fixture_excludes_third_party_replies_by_default(self) -> None:
        """Default policy (include_third_party_replies=False) filters replies."""
        from markitai.webextract.pipeline import extract_web_content

        html = (FIXTURES / "x_status_2030105637204676808.playwright.html").read_text(
            encoding="utf-8"
        )
        result = extract_web_content(
            html, "https://x.com/ixiaowenz/status/2030105637204676808"
        )
        assert result.semantic is not None
        assert result.semantic.thread is not None
        # Default policy: include_third_party_replies=False, so reply_user_1
        # and random_person from the <section> should NOT appear in items.
        assert result.semantic.thread.items == []

    def test_x_author_self_reply_would_be_included(self) -> None:
        """If the replies section had an author self-reply, policy would keep it."""
        from markitai.webextract.thread_policy import get_thread_policy

        policy = get_thread_policy("https://x.com/ixiaowenz/status/2030105637204676808")
        assert policy is not None
        assert policy.include_author_thread is True
        assert policy.include_third_party_replies is False


# ---------------------------------------------------------------------------
# Low-3: build_source_frontmatter integration in fetch paths
# ---------------------------------------------------------------------------


class TestFrontmatterIntegration:
    """Fetch paths must produce enriched frontmatter with content_profile."""

    def test_native_extraction_returns_content_profile_in_frontmatter(self) -> None:
        """When info is populated, frontmatter must include content_profile."""
        from markitai.webextract.frontmatter import build_source_frontmatter
        from markitai.webextract.pipeline import extract_web_content

        html = (FIXTURES / "x_status_2030105637204676808.playwright.html").read_text(
            encoding="utf-8"
        )
        result = extract_web_content(
            html, "https://x.com/ixiaowenz/status/2030105637204676808"
        )
        assert result.info is not None

        fm = build_source_frontmatter(result)
        assert "content_profile" in fm
        assert fm["content_profile"] == "social_post"
        assert "word_count" in fm
        assert isinstance(fm["word_count"], int)

    def test_fetch_prefers_build_over_coerce_when_info_present(self) -> None:
        """fetch.py integration: info-bearing result triggers build path."""

        from markitai.webextract.frontmatter import build_source_frontmatter
        from markitai.webextract.types import (
            ContentProfile,
            ExtractedWebContent,
            ExtractionInfo,
            QualityAssessment,
            WebMetadata,
        )

        result = ExtractedWebContent(
            clean_html="<p>Hello</p>",
            markdown="Hello",
            metadata=WebMetadata(title="Test"),
            word_count=1,
            info=ExtractionInfo(
                content_profile=ContentProfile.SOCIAL_POST,
                extractor_name="test",
                word_count=1,
            ),
            quality=QualityAssessment(accepted=True, score=1.0),
        )
        fm = build_source_frontmatter(result)
        # Verify the enriched fields that coerce_source_frontmatter would miss
        assert fm["content_profile"] == "social_post"
        assert fm["word_count"] == 1
        assert fm["title"] == "Test"
