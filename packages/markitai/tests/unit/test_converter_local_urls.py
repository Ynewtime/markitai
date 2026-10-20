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


_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01"
    b"\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_a_local_pages_own_pictures_are_embedded(tmp_path: Path) -> None:
    """A local page's pictures kept absolute file:// addresses, which break
    once the Markdown is anywhere else; a DOCX or EPUB picture becomes an
    asset. Only a picture inside the page's own folder is read: not one
    reached through ``..``, an absolute path or a symlink out of it, and not
    a file that only has a picture's name."""
    import base64
    import os

    folder = tmp_path / "site dir"
    (folder / "img").mkdir(parents=True)
    (folder / "img" / "red one.png").write_bytes(_PNG)
    (folder / "img" / "fake.png").write_text("ssh key", encoding="utf-8")
    outside = tmp_path / "other"
    outside.mkdir()
    (outside / "out.png").write_bytes(_PNG)
    symlink = folder / "img" / "link.png"
    try:
        os.symlink(outside / "out.png", symlink)
    except OSError:
        symlink = None
    page = folder / "page.html"
    page.write_text(
        f"<html><head><title>T</title></head><body><article><h1>T</h1>{_PROSE}"
        '<p><img src="img/red%20one.png" alt="inside">'
        ' <img src="img/link.png" alt="linked">'
        ' <img src="../other/out.png" alt="parent">'
        f' <img src="{(outside / "out.png").as_uri()}" alt="absolute">'
        ' <img src="img/fake.png" alt="fake">'
        ' <img src="img/missing.png" alt="missing">'
        ' <img src="https://example.com/x.png" alt="remote"></p>'
        f"{_PROSE}</article></body></html>",
        encoding="utf-8",
    )

    markdown = HtmlConverter().convert(page).markdown

    encoded = base64.b64encode(_PNG).decode("ascii")
    assert f"![inside](data:image/png;base64,{encoded})" in markdown
    assert markdown.count("data:image/") == 1
    for kept in ("parent", "absolute", "fake", "missing"):
        assert f"![{kept}](file://" in markdown
    if symlink is not None:
        assert "![linked](file://" in markdown
    assert "![remote](https://example.com/x.png)" in markdown


def test_relative_pictures_of_the_fallback_reader_are_embedded(
    tmp_path: Path,
) -> None:
    # Too little text for the extractor: markitdown's reader keeps the
    # relative reference, which is resolved the same way
    (tmp_path / "a.png").write_bytes(_PNG)
    page = tmp_path / "tiny.html"
    page.write_text('<p>Hi <img src="a.png" alt="dot"></p>', encoding="utf-8")

    markdown = HtmlConverter().convert(page).markdown

    assert "![dot](data:image/png;base64," in markdown
