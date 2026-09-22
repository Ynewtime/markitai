"""Behavior ported from defuddle 0.19.3 -> 0.19.4 (issue #41).

Each test pins one upstream change against the port; the fixture-backed
ones read the resynced corpus.
"""

from __future__ import annotations

from pathlib import Path

from markitai.webextract import extract_web_content

FIXTURES = Path(__file__).parents[2] / "defuddle_fixtures" / "fixtures"

_PARA = (
    "<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo.</p>"
)


def _extract(body: str) -> str:
    html = (
        "<html><head><title>T</title></head><body><article><h1>Title</h1>"
        f"{body}{_PARA}{_PARA}{_PARA}</article></body></html>"
    )
    return extract_web_content(html, "https://example.com/post").markdown


def test_code_fence_grows_past_backtick_runs_instead_of_escaping():
    """defuddle#359: backslash escapes are literal in a fence."""
    md = _extract(
        '<pre><code class="language-md">Use `x` here\n```\nfenced inside\n```'
        "</code></pre>"
    )
    assert "````md\nUse `x` here\n```\nfenced inside\n```\n````" in md
    assert "\\`" not in md


def test_code_fence_stays_three_backticks_without_a_closing_run():
    md = _extract("<pre><code>a `b` c</code></pre>")
    assert "```\na `b` c\n```" in md


def test_sub_and_sup_keep_their_tags_and_hug_neighbours():
    """defuddle#379: flattening merged the script into the word (20215ya)."""
    md = _extract(
        "<p>Water is H<sub>2</sub>O and 10<sup>n</sup>; built in "
        "1989<sub>37ya</sub> ok.</p>"
    )
    assert "H<sub>2</sub>O" in md
    assert "10<sup>n</sup>;" in md
    assert "1989<sub>37ya</sub> ok" in md


def test_footnote_sup_still_becomes_a_reference():
    md = _extract(
        '<p>Claim<sup id="fnref:1"><a href="#fn:1">1</a></sup> made.</p>'
        '<div id="footnotes"><ol><li id="fn:1">Note text here.</li></ol></div>'
    )
    assert "Claim[^1] made." in md
    assert "<sup>" not in md


def test_date_inside_a_labeled_row_is_not_stripped_from_the_row():
    """An email-style header keeps "Date: ..." instead of an orphaned label."""
    md = extract_web_content(
        (FIXTURES / "metadata--email-style-header-block.html").read_text(
            encoding="utf-8"
        ),
        "https://example.com/posts/announcement-reflection/",
    ).markdown
    assert "Date: Wed, 08 Apr 2026" in md
    assert "From: Example Team <hello@example.com>" in md
    assert "\nDate:\n" not in md


def test_arxiv_hidden_note_outer_spans_are_dropped():
    md = extract_web_content(
        (FIXTURES / "issues--144-arxiv-footnote-marks.html").read_text(
            encoding="utf-8"
        ),
        "https://arxiv.org/html/2305.18290",
    ).markdown
    assert "Rafael Rafailov<sup>2</sup>, Archit Sharma<sup>1</sup>" in md
    assert "footnotemark" not in md


def test_closed_and_nested_declarative_shadow_roots_are_content():
    md = _extract(
        '<div><template shadowrootmode="closed"><section>'
        '<template shadowroot="open"><p>Nested shadow content that should be '
        "visible in the output.</p></template></section>"
        "<p>Outer shadow content that should be visible too.</p></template></div>"
    )
    assert "Nested shadow content that should be visible in the output." in md
    assert "Outer shadow content that should be visible too." in md
    assert "<template" not in md


def test_plain_template_content_never_reaches_the_output():
    md = _extract(
        "<p>Before</p><template><p>Template leak paragraph long enough to look "
        "like real article content for scoring.</p></template><p>After</p>"
    )
    assert "Template leak" not in md
    assert "Before" in md and "After" in md
