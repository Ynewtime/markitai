from __future__ import annotations


def test_normalize_code_blocks_wraps_preformatted_code_and_keeps_language() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.code import normalize_code_blocks

    soup = parse_html(
        """
        <article>
          <code class="language-python" style="white-space: pre;">
          print("hello")
          </code>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    normalize_code_blocks(article)

    html = str(article)
    assert "<pre>" in html
    assert "language-python" in html
    assert 'print("hello")' in html


def test_numeric_gutters_go_but_numeric_literals_stay() -> None:
    """Only rows that announce themselves as line rows lose a leading number."""
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.code import _strip_inline_numeric_gutters

    soup = parse_html(
        "<pre><code>"
        '<span style="display:flex"><span>1</span><span>p = 61</span></span>\n'
        '<span><span class="mi">42</span><span> + x</span></span>'
        "</code></pre>"
    )
    pre = soup.pre
    assert pre is not None
    _strip_inline_numeric_gutters(pre)
    text = pre.get_text()
    assert "1p" not in text and "p = 61" in text
    assert "42 + x" in text


def test_names_drawn_as_buttons_in_a_code_line_stay_in_the_code() -> None:
    """Twoslash (nodejs.org) wraps names with type popups in buttons inside
    the highlighted lines: ``const { createServer } = require('node:http');``
    came out as ``const {  } = ('node:http');``. A copy button beside the
    lines, a hidden one and a popup's type text are still left out, as in
    markitai-rust (0f44d83)."""
    from markitai.webextract.pipeline import extract_web_content

    prose = (
        "<p>This invented page explains a small server example in enough "
        "words for the extractor to take it for an article worth keeping.</p>"
    )
    html = (
        f"<html><head><title>Server</title></head><body><article><h1>Server</h1>"
        f"{prose}<pre class='shiki twoslash language-cjs'>"
        "<button class='copy'>Copy</button><code>"
        "<span class='line'><span>const {</span><span> </span><span>"
        "<button class='twoslash-hover'>createServer</button></span>"
        "<span> } = </span><span><button data-state='closed' "
        "class='twoslash-hover'><span class='twoslash-popup-container'>"
        "function require(id: string): any</span>require</button></span>"
        "<span>('node:http');</span></span>\n"
        "<span class='line'><span>const </span><span role='button'>port</span>"
        "<span> = 3000;</span><button hidden>secret</button>"
        "<button><svg></svg>icon</button></span></code></pre>"
        f"{prose}</article></body></html>"
    )

    markdown = extract_web_content(html, "https://example.com/server").markdown

    assert "const { createServer } = require('node:http');" in markdown
    assert "const port = 3000;" in markdown
    for gone in ("Copy", "function require", "secret", "icon"):
        assert gone not in markdown
