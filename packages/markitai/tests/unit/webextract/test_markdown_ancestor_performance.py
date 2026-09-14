"""Render-only ancestor shortcuts must follow the live DOM exactly."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from bs4 import BeautifulSoup, Tag

from markitai.webextract.html_to_markdown import (
    WebExtractHtmlConverter,
    WebExtractMarkdownConverter,
)


@pytest.mark.parametrize("kind", ["html", "docx"])
def test_rendering_does_not_build_general_queries_for_pre_ancestors(
    kind, fixtures_dir, monkeypatch
):
    original = Tag.find_parents
    calls = []

    def track(self, *args, **kwargs):
        if args and args[0] == "pre":
            calls.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Tag, "find_parents", track)
    if kind == "docx":
        from markitai.converter.office import DocxConverter

        result = DocxConverter().convert(fixtures_dir / "sample.docx")
    else:
        result = WebExtractHtmlConverter().convert_string(
            "<article>" + "<p>Text <a href='/x'>link</a></p>" * 100 + "</article>"
        )
    assert result.markdown
    # A bare soup root is BeautifulSoup itself, not one of the parsed Tags.
    # Its single check is independent of the number of rendered elements.
    assert calls in ([], ["[document]"])


@pytest.mark.parametrize("parser", ["html.parser", "lxml"])
def test_render_dom_and_markdown_are_identical(parser):
    from markitai.webextract.markdownify_compat import parse_markdown_html

    bodies = [
        "<p>A <strong>B</strong> C<br>D</p><hr><p>E</p>",
        "<pre> a\n<span><a href='/x'>b</a>\n\nc</span> </pre>",
        "<table><tr><th>A</th><td> B &amp; C </td></tr></table>",
        "<div><pre><pre><code>x</code></pre><p>y</p></pre></div>",
        "<pre id='a'><span>one</pre><p>two</span><pre>three",
        "<!-- c --><style>body{}</style><p>&nbsp;&copy;&#13;&#10;</p>",
        "<pre><img src='x' alt='y'><kbd>A</kbd><samp>B</samp></pre>",
        "<div><pre class='x'><code>\n\n```\n\n</code></pre></div>",
    ]
    for depth in range(8):
        for body in bodies:
            markup = "<div>" * depth + body + "</div>" * depth
            old = BeautifulSoup(markup, parser)
            new = parse_markdown_html(markup, parser)
            assert str(new) == str(old)
            assert WebExtractMarkdownConverter().convert_soup(new) == (
                WebExtractMarkdownConverter().convert_soup(old)
            )


def test_other_queries_and_reparenting_keep_bs4_semantics():
    from markitai.webextract.markdownify_compat import parse_markdown_html

    soup = parse_markdown_html(
        "<div id='root'><pre id='outer'><pre id='inner'><b>x</b></pre></pre>"
        "<section><pre id='new'></pre></section></div>"
    )
    tag = soup.find("b")
    assert isinstance(tag, Tag)
    for args, kwargs in [
        (("pre",), {}),
        (("pre", {"id": "outer"}), {}),
        ((), {"name": "pre", "id": "outer"}),
        ((["pre", "div"],), {}),
        ((lambda node: node.name == "div",), {}),
        ((), {}),
    ]:
        assert tag.find_parent(*args, **kwargs) is Tag.find_parent(tag, *args, **kwargs)
    parent = tag.find_parent("pre")
    assert isinstance(parent, Tag) and parent["id"] == "inner"
    tag.extract()
    assert tag.find_parent("pre") is None
    new_parent = soup.find(id="new")
    assert isinstance(new_parent, Tag)
    new_parent.append(tag)
    parent = tag.find_parent("pre")
    assert isinstance(parent, Tag) and parent["id"] == "new"


def test_parallel_rendering_does_not_change_global_tag_class():
    from markitai.webextract.markdownify_compat import parse_markdown_html

    original = Tag.find_parent
    documents = [f"<pre><code>{i}\n\nx</code></pre><p>y</p>" for i in range(30)]

    def render(markup):
        return WebExtractMarkdownConverter().convert_soup(parse_markdown_html(markup))

    expected = [
        WebExtractMarkdownConverter().convert_soup(BeautifulSoup(s, "html.parser"))
        for s in documents
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(render, documents)) == expected
    assert Tag.find_parent is original
    assert type(BeautifulSoup("<p>x</p>", "html.parser").p) is Tag
