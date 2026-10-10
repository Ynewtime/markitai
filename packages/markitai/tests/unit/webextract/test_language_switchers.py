"""Language switchers: lists of the page's other-language editions."""

from __future__ import annotations

from bs4 import BeautifulSoup

from markitai.webextract.pipeline import extract_web_content
from markitai.webextract.removals.language_switchers import remove_language_switchers

_LANGS = [
    ("de", "Deutsch"),
    ("fr", "Français"),
    ("ja", "日本語"),
    ("nb", "Norsk bokmål"),
    ("zh", "中文"),
]
_PROSE = " ".join(
    ["Markdown is a lightweight markup language for formatting text."] * 30
)


def _items() -> str:
    return "".join(
        f'<li class="interlanguage-link"><a href="https://{code}.example.org/wiki/X" '
        f'hreflang="{code}" lang="{code}">{name}</a></li>'
        for code, name in _LANGS
    )


def _root(body: str) -> BeautifulSoup:
    return BeautifulSoup(f"<main>{body}<p>{_PROSE}</p></main>", "lxml")


def test_a_vector_language_dropdown_goes_with_its_controls() -> None:
    """MediaWiki's Vector skin: checkbox, "5 languages", list, "Edit links"."""
    soup = _root(
        "<header><h1>X</h1>"
        '<div id="p-lang-btn"><input type="checkbox"/>'
        "<label><span>5 languages</span></label>"
        f'<div class="vector-menu-content"><ul>{_items()}</ul>'
        '<div><a href="https://www.wikidata.org/wiki/Q1">Edit links</a></div>'
        "</div></div></header>"
    )
    root = soup.main
    assert root is not None
    assert remove_language_switchers(root) == 1
    assert soup.find(id="p-lang-btn") is None
    assert soup.h1 is not None and soup.h1.get_text() == "X"
    assert soup.header is not None


def test_the_article_starts_with_the_article() -> None:
    html = (
        "<html><head><title>X</title></head><body><main><header><h1>X</h1>"
        f'<div id="p-lang-btn"><input type="checkbox"/><ul>{_items()}</ul></div>'
        f"</header><p>{_PROSE}</p></main></body></html>"
    )
    markdown = extract_web_content(html, "https://en.example.org/wiki/X").markdown
    assert "[ ]" not in markdown
    assert "Deutsch" not in markdown and "日本語" not in markdown
    assert "Markdown is a lightweight markup language" in markdown


def test_a_translation_bar_goes() -> None:
    links = " • ".join(
        f'<a href="/x/{code}" hreflang="{code}">{name}</a>' for code, name in _LANGS
    )
    soup = _root(f'<div class="bar">Languages: English • {links}</div>')
    assert soup.main is not None
    assert remove_language_switchers(soup.main) == 1
    assert soup.find(class_="bar") is None


def test_prose_that_names_translations_stays() -> None:
    soup = _root(
        "<p>The guide is also available in "
        '<a href="/de" hreflang="de">German</a>, '
        '<a href="/fr" hreflang="fr">French</a> and '
        '<a href="/es" hreflang="es">Spanish</a>, '
        "translated by volunteers over many years.</p>"
    )
    assert soup.main is not None
    assert remove_language_switchers(soup.main) == 0


def test_a_reading_list_of_foreign_articles_stays() -> None:
    items = "".join(
        f'<li><a href="/{code}" hreflang="{code}">A long article title about the '
        f"history of markup in {name}</a></li>"
        for code, name in _LANGS
    )
    soup = _root(f"<ul>{items}</ul>")
    assert soup.main is not None
    assert remove_language_switchers(soup.main) == 0


def test_a_mixed_link_list_stays() -> None:
    items = _items() + "".join(
        f'<li><a href="/page/{n}">Page {n}</a></li>' for n in range(4)
    )
    soup = _root(f"<ul>{items}</ul>")
    assert soup.main is not None
    assert remove_language_switchers(soup.main) == 0


def test_a_portal_whose_content_is_the_list_keeps_it() -> None:
    soup = BeautifulSoup(f"<main><h1>Wiki</h1><ul>{_items()}</ul></main>", "lxml")
    assert soup.main is not None
    assert remove_language_switchers(soup.main) == 0
    assert len(soup.find_all("a", hreflang=True)) == len(_LANGS)
