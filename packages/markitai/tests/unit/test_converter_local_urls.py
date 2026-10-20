"""Relative references in a local HTML file or email resolve against its path."""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

from markitai.converter.eml import EmlConverter
from markitai.converter.markitdown_ext import HtmlConverter

_PROSE = (
    "<p>Paragraph text that is long enough to be the main content of the "
    "page, so the extractor keeps it as the article body.</p>"
)


def test_a_hash_in_the_folder_name_is_part_of_the_path(tmp_path: Path) -> None:
    """The base URL was f"file://{path}": "#" started a fragment, so the
    images and links of "C# notes/page.html" resolved one folder up (and
    could name another file of the same name there)."""
    folder = tmp_path / "C# notes 100%"
    folder.mkdir()
    page = folder / "page.html"
    page.write_text(
        f"<html><head><title>T</title></head><body><article><h1>T</h1>{_PROSE}"
        '<p><img src="cat.png" alt="a cat"> <a href="other.html">next</a></p>'
        f"{_PROSE}</article></body></html>",
        encoding="utf-8",
    )

    markdown = HtmlConverter().convert(page).markdown

    assert f"]({(folder / 'cat.png').as_uri()})" in markdown
    assert f"[next]({(folder / 'other.html').as_uri()})" in markdown


def test_an_emails_relative_link_resolves_against_its_folder(tmp_path: Path) -> None:
    folder = tmp_path / "mail #2"
    folder.mkdir()
    message = EmailMessage()
    message["Subject"] = "Hello"
    message["From"] = "a@example.com"
    message.set_content(
        f"<html><body>{_PROSE}<p><a href='notes.html'>notes</a></p>{_PROSE}"
        "</body></html>",
        subtype="html",
    )
    path = folder / "hello.eml"
    path.write_bytes(message.as_bytes())

    markdown = EmlConverter().convert(path).markdown

    assert f"[notes]({(folder / 'notes.html').as_uri()})" in markdown
