"""Relative references of a local document, kept relative in its Markdown."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from urllib.parse import quote, urlsplit
from urllib.request import url2pathname

# A link's or picture's ``file:`` target (``](file:///…)``, ``](<file:///…>)``);
# a ``file:`` address quoted in the text or in code is left as written.
_FILE_TARGET = re.compile(r"(\]\(<?)(file://[^\s()<>\"']*)", re.IGNORECASE)

# A Windows drive at the start of a file: URL's path, its colon percent-encoded
# (``/C%3A/Users/…``): the link writer quotes every ":" of a path, and
# Windows' url2pathname finds the drive only by a literal ":" (or "|").
_ENCODED_DRIVE = re.compile(r"^(/*[A-Za-z])%3A(?=/|$)", re.IGNORECASE)


def relative_local_links(markdown: str, page: Path) -> str:
    """Write the ``file:`` addresses left in a local page's Markdown relative
    to the page's folder, as the page wrote them (an HTML file, an email's
    HTML body).

    The page is extracted against its own ``file:`` address so its relative
    links and pictures resolve; what is left of them (links, pictures that
    were not saved as assets) would otherwise put the absolute path of the
    user's folder (``file:///home/<user>/…``, ``file:///C:/Users/<user>/…``)
    into the output. A link to the page itself keeps only its ``#fragment``.
    A share on another host, and a file on another Windows drive, are left
    as they are.
    """
    if "file:" not in markdown.lower():
        return markdown
    try:
        own = str(page.resolve())
    except OSError:
        return markdown
    return relative_file_links(markdown, own, os.path, url2pathname)


def relative_file_links(
    markdown: str,
    own: str,
    paths: ModuleType,
    to_path: Callable[[str], str],
) -> str:
    """:func:`relative_local_links` for the page at the absolute path *own*,
    with the platform's path rules (``os.path`` and ``url2pathname``; tests
    pass ``ntpath`` and ``nturl2path.url2pathname`` to check Windows')."""
    folder = paths.dirname(own)
    same = paths.normcase(paths.normpath(own))

    def replace(match: re.Match[str]) -> str:
        opening, address = match.groups()
        parts = urlsplit(address)
        if parts.netloc not in ("", "localhost"):
            return match.group(0)
        try:
            target = to_path(_ENCODED_DRIVE.sub(r"\1:", parts.path))
            relative = paths.relpath(target, folder)
        except (OSError, ValueError):  # not a path, or another Windows drive
            return match.group(0)
        if paths.normcase(paths.normpath(target)) == same and parts.fragment:
            return f"{opening}#{parts.fragment}"
        written = quote(relative.replace(paths.sep, "/"), safe="/-._~!$&'*+,;=:@")
        if parts.query:
            written += f"?{parts.query}"
        if parts.fragment:
            written += f"#{parts.fragment}"
        return opening + written

    return _FILE_TARGET.sub(replace, markdown)
