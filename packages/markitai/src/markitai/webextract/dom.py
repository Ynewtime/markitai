from __future__ import annotations

from importlib.util import find_spec

from bs4 import BeautifulSoup


def parse_html(html: str) -> BeautifulSoup:
    """Parse HTML using the best available parser.

    Args:
        html: Raw HTML content.

    Returns:
        Parsed BeautifulSoup document.
    """

    parser = "lxml" if find_spec("lxml") is not None else "html.parser"
    return BeautifulSoup(html, parser)
