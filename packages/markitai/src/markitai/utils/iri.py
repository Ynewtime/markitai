"""Web addresses as people read them: IRIs (RFC 3987) in front matter.

A page address arrives percent-encoded (a browser copies
``/posts/20251223-ai-%E5%BA%94%E7%94%A8``, and a page's ``rel=canonical``
is written that way), which hides every non-ASCII word of it. Front matter
shows it with those characters decoded (``/posts/20251223-ai-应用``): the
same address, since turning an IRI back into a URI percent-encodes each
non-ASCII character as its UTF-8 bytes (RFC 3987 §3.1), which gives the
original back. Only escapes of non-ASCII characters are decoded; ``%20``,
``%2F``, ``%3F``, ``%25`` and every other ASCII escape stay, so the path,
query and fragment keep their boundaries. Characters RFC 3987 leaves out
of an IRI (bidirectional controls, spaces and other separators, control
and private-use characters) and bytes that are not UTF-8 stay encoded too.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit

_ESCAPES = re.compile(r"(?:%[0-9A-Fa-f]{2})+")


def _readable(char: str) -> bool:
    """A non-ASCII character an IRI may hold as itself."""
    # Separators (Z*: no-break and ideographic spaces), controls, format
    # characters (Cf: the bidirectional controls RFC 3987 §4.1 forbids),
    # private use, surrogates, unassigned
    if unicodedata.category(char)[0] in "ZC":
        return False
    # RFC 3987 ucschar leaves out the last two code points of every plane
    return (ord(char) & 0xFFFE) != 0xFFFE


def _decode_run(run: str) -> str:
    """One run of ``%XX`` escapes with its non-ASCII characters decoded."""
    escapes = [run[i : i + 3] for i in range(0, len(run), 3)]
    data = bytes(int(escape[1:], 16) for escape in escapes)
    out: list[str] = []
    i = 0
    while i < len(data):
        lead = data[i]
        size = (
            2
            if 0xC2 <= lead <= 0xDF
            else 3
            if 0xE0 <= lead <= 0xEF
            else 4
            if 0xF0 <= lead <= 0xF4
            else 0
        )
        char = ""
        if size and i + size <= len(data):
            try:
                char = data[i : i + size].decode("utf-8")
            except UnicodeDecodeError:
                char = ""
        if not char:
            # ASCII or a byte that starts no UTF-8 character
            out.append(escapes[i])
            i += 1
            continue
        # A whole character: shown, or all of its escapes kept
        out.extend([char] if _readable(char) else escapes[i : i + size])
        i += size
    return "".join(out)


def readable_url(url: str) -> str:
    """*url* with its percent-encoded non-ASCII characters shown, if HTTP(S).

    Anything else (a file name, a ``mailto:``, a value that is no URL) comes
    back unchanged, as does the host: a punycode name (``xn--``) is what
    the address says.
    """
    if not isinstance(url, str) or "%" not in url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return url
    # Everything after the authority; the authority itself is left alone
    start = url.find(parts.netloc) + len(parts.netloc)
    tail = _ESCAPES.sub(lambda match: _decode_run(match.group(0)), url[start:])
    return url[:start] + tail
