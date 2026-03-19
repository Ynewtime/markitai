from __future__ import annotations

from importlib.util import find_spec

from bs4 import BeautifulSoup

from markitai.webextract.preprocess import preprocess_html


def parse_html(html: str) -> BeautifulSoup:
    """Parse HTML using the best available parser.

    Applies raw HTML preprocessing (shadow DOM flattening, ``<wbr>`` removal,
    ``<noscript>`` promotion) before handing the markup to BeautifulSoup.

    Args:
        html: Raw HTML content.

    Returns:
        Parsed BeautifulSoup document.
    """
    html = preprocess_html(html)
    parser = "lxml" if find_spec("lxml") is not None else "html.parser"
    return BeautifulSoup(html, parser)
