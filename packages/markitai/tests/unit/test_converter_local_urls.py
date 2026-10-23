"""Relative references in a local HTML file or email resolve against its path,
and what is left of them stays relative in the Markdown: never an absolute
``file:///home/<user>/…`` address of the user's folder."""

from __future__ import annotations

import re
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

    assert "](cat.png)" in markdown
    assert "[next](other.html)" in markdown
    assert "file:" not in markdown


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

    assert "[notes](notes.html)" in markdown
    assert "file:" not in markdown


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
    assert "![parent](../other/out.png)" in markdown
    assert "![absolute](../other/out.png)" in markdown
    assert "![fake](img/fake.png)" in markdown
    assert "![missing](img/missing.png)" in markdown
    if symlink is not None:
        assert "![linked](img/link.png)" in markdown
    assert "file:" not in markdown
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


def test_a_local_pages_links_stay_relative(tmp_path: Path) -> None:
    """Relative links of a local page became absolute ``file:///home/<user>/…``
    addresses, putting the user's folder into the output; markitai-rust
    keeps them as written. A link into the page keeps its fragment only, an
    address quoted in the text or in code is left alone, and a share on
    another host keeps its address."""
    folder = tmp_path / "notes (draft)"
    (folder / "sub").mkdir(parents=True)
    page = folder / "page.html"
    page.write_text(
        f"<html><head><title>T</title></head><body><article><h1>T</h1>{_PROSE}"
        '<p><a href="other.html">same folder</a>, <a href="sub/deep%20page.html'
        '?v=2#part">deeper</a>, <a href="../up.html">up</a>, '
        '<a href="page.html#intro">intro</a>, <a href="file://server/share/doc.html">share'
        '</a>, <a href="https://example.com/x">web</a>.</p>'
        "<p>Type file:///etc/hosts in the address bar.</p>"
        f"{_PROSE}</article></body></html>",
        encoding="utf-8",
    )

    markdown = HtmlConverter().convert(page).markdown

    assert "[same folder](other.html)" in markdown
    assert "[deeper](sub/deep%20page.html?v=2#part)" in markdown
    assert "[up](../up.html)" in markdown
    assert "[intro](#intro)" in markdown
    assert "[share](file://server/share/doc.html)" in markdown
    assert "[web](https://example.com/x)" in markdown
    assert "file:///etc/hosts" in markdown
    assert str(tmp_path) not in markdown
    assert tmp_path.as_uri() not in markdown


def test_a_windows_pages_links_stay_relative() -> None:
    """On Windows the page's address is ``file:///C:/Users/<user>/…`` and the
    link writer quotes the drive's colon (``file:///C%3A/Users/…``), which
    Windows' url2pathname does not read as a drive: every link stayed an
    absolute address of the user's folder. Windows' own path rules (ntpath,
    nturl2path) run here on any system, on the extractor's real output."""
    import ntpath
    import nturl2path
    from pathlib import PureWindowsPath

    from markitai.converter.local_links import relative_file_links
    from markitai.webextract import extract_web_content

    page = PureWindowsPath(r"C:\Users\dev\AppData\Local\Temp\C# notes 100%\page.html")
    html = (
        f"<html><head><title>T</title></head><body><article><h1>T</h1>{_PROSE}"
        '<p><img src="cat.png" alt="a cat"> <a href="other.html">same folder</a>, '
        '<a href="sub/deep%20page.html?v=2#part">deeper</a>, <a href="../up.html">'
        'up</a>, <a href="../C%23%20notes%20100%25/back.html">back</a>, '
        '<a href="page.html#intro">intro</a>, <a href="PAGE.HTML#case">case</a>, '
        '<a href="file:///D:/data/x.html">other drive</a>, '
        '<a href="file://server/share/doc.html">share</a>, '
        '<a href="https://example.com/x">web</a>.</p>'
        f"{_PROSE}</article></body></html>"
    )
    extracted = extract_web_content(html, page.as_uri()).markdown
    assert "file:///C%3A/Users/" in extracted  # the shape that broke

    markdown = relative_file_links(
        extracted, str(page), ntpath, nturl2path.url2pathname
    )

    assert "![a cat](cat.png)" in markdown
    assert "[same folder](other.html)" in markdown
    assert "[deeper](sub/deep%20page.html?v=2#part)" in markdown
    assert "[up](../up.html)" in markdown
    assert "[back](back.html)" in markdown
    assert "[intro](#intro)" in markdown
    assert "[case](#case)" in markdown  # Windows names ignore case
    # Not relative to anything here: kept as an address (its colon as the
    # link writer quoted it)
    assert re.search(r"\[other drive\]\(file:///D(:|%3A)/data/x\.html\)", markdown)
    assert "[share](file://server/share/doc.html)" in markdown
    assert "[web](https://example.com/x)" in markdown
    assert "Users" not in markdown
    assert "C:" not in markdown and "C%3A" not in markdown


def test_a_windows_drive_is_read_in_every_spelling() -> None:
    """``C:``, ``c%3a``, ``C%3A`` and the old ``C|`` all name the drive."""
    import ntpath
    import nturl2path

    from markitai.converter.local_links import relative_file_links

    own = r"C:\docs\page.html"
    for drive in ("C:", "c%3a", "C%3A", "C|"):
        markdown = f"[a](file:///{drive}/docs/img/a.png) [b](file:///{drive}/b.html)"
        assert (
            relative_file_links(markdown, own, ntpath, nturl2path.url2pathname)
            == "[a](img/a.png) [b](../b.html)"
        ), drive


def test_a_windows_emails_relative_link_stays_relative() -> None:
    """The same for an email's HTML body, which has its own renderer."""
    import ntpath
    import nturl2path
    from pathlib import PureWindowsPath

    from markitai.converter.eml import _render_message
    from markitai.converter.local_links import relative_file_links

    path = PureWindowsPath(r"C:\Users\dev\Mail\mail #2\hello.eml")
    message = EmailMessage()
    message["Subject"] = "Hello"
    message["From"] = "a@example.com"
    message.set_content(
        f"<html><body>{_PROSE}<p><a href='notes.html'>notes</a> "
        f"<a href='../old/a.html'>older</a></p>{_PROSE}</body></html>",
        subtype="html",
    )

    rendered = _render_message(message, source=path.as_uri())
    markdown = relative_file_links(rendered, str(path), ntpath, nturl2path.url2pathname)

    assert "[notes](notes.html)" in markdown
    assert "[older](../old/a.html)" in markdown
    assert "file:" not in markdown and "Users" not in markdown
