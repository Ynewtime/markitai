"""Relative references of a local document, kept relative in its Markdown."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import url2pathname

# A link's or picture's ``file:`` target (``](file:///…)``, ``](<file:///…>)``);
# a ``file:`` address quoted in the text or in code is left as written.
_FILE_TARGET = re.compile(r"(\]\(<?)(file://[^\s()<>\"']*)", re.IGNORECASE)


def relative_local_links(markdown: str, page: Path) -> str:
    """Write the ``file:`` addresses left in a local page's Markdown relative
    to the page's folder, as the page wrote them (an HTML file, an email's
    HTML body).

    The page is extracted against its own ``file:`` address so its relative
    links and pictures resolve; what is left of them (links, pictures that
    were not saved as assets) would otherwise put the absolute path of the
    user's folder (``file:///home/<user>/…``) into the output. A link to the
    page itself keeps only its ``#fragment``. A share on another host is
    left as it is.
    """
    if "file:" not in markdown.lower():
        return markdown
    try:
        own = page.resolve()
    except OSError:
        return markdown
    folder = own.parent

    def replace(match: re.Match[str]) -> str:
        opening, address = match.groups()
        parts = urlsplit(address)
        if parts.netloc not in ("", "localhost"):
            return match.group(0)
        target = Path(url2pathname(parts.path))
        try:
            relative = os.path.relpath(target, folder)
        except ValueError:  # another drive on Windows
            return match.group(0)
        if target == own and parts.fragment:
            return f"{opening}#{parts.fragment}"
        written = quote(relative.replace(os.sep, "/"), safe="/-._~!$&'*+,;=:@")
        if parts.query:
            written += f"?{parts.query}"
        if parts.fragment:
            written += f"#{parts.fragment}"
        return opening + written

    return _FILE_TARGET.sub(replace, markdown)
