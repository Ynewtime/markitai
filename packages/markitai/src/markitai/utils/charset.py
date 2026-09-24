"""Character-set resolution for fetched web bodies.

Follows the WHATWG Encoding Standard the way a browser does, because the
pages markitai fetches are authored against browsers, not against Python's
codec registry:

1. A byte-order mark wins.
2. The ``charset`` parameter of the ``Content-Type`` header.
3. For HTML only, a ``<meta charset>`` / ``http-equiv`` prescan of the
   first 1024 bytes.
4. Detection: strict UTF-8, then a statistical guess.

A body that fits its encoding except for a few corrupt bytes keeps that
encoding and loses only those bytes: a late ``<meta>`` declaration, UTF-8
with under 1% bad bytes, and a detected encoding all decode with
replacement instead of dropping the whole page to windows-1252.

Labels are mapped the way the standard maps them, which is usually to a
superset of what the label says: ``gb2312``/``gbk`` decode as GB18030,
``shift_jis`` as Windows-31J (CP932), ``iso-8859-1``/``ascii`` as
Windows-1252, ``euc-kr`` as CP949. A page labelled ``gb2312`` that contains
a GBK-only character, or an ``iso-8859-1`` page with curly quotes, then
decodes the way the browser shows it instead of as mojibake.
"""

from __future__ import annotations

import codecs
import re

# WHATWG encoding labels whose Python codec differs from the label, or that
# Python's registry does not know. Keys are lowercase labels; values are
# Python codec names. Labels mapped to None are the standard's "replacement"
# encodings (legacy ISO-2022 variants disallowed for security reasons).
_WHATWG_LABELS: dict[str, str | None] = {
    # UTF-8
    "unicode-1-1-utf-8": "utf-8",
    "unicode11utf8": "utf-8",
    "unicode20utf8": "utf-8",
    "utf-8": "utf-8",
    "utf8": "utf-8",
    "x-unicode20utf8": "utf-8",
    # windows-1252 (covers every latin1/ascii label)
    "ansi_x3.4-1968": "cp1252",
    "ascii": "cp1252",
    "cp1252": "cp1252",
    "cp819": "cp1252",
    "csisolatin1": "cp1252",
    "ibm819": "cp1252",
    "iso-8859-1": "cp1252",
    "iso-ir-100": "cp1252",
    "iso8859-1": "cp1252",
    "iso88591": "cp1252",
    "iso_8859-1": "cp1252",
    "iso_8859-1:1987": "cp1252",
    "l1": "cp1252",
    "latin1": "cp1252",
    "us-ascii": "cp1252",
    "windows-1252": "cp1252",
    "x-cp1252": "cp1252",
    # windows-1254 (covers latin5/iso-8859-9)
    "cp1254": "cp1254",
    "csisolatin5": "cp1254",
    "iso-8859-9": "cp1254",
    "iso-ir-148": "cp1254",
    "iso8859-9": "cp1254",
    "iso88599": "cp1254",
    "iso_8859-9": "cp1254",
    "iso_8859-9:1989": "cp1254",
    "l5": "cp1254",
    "latin5": "cp1254",
    "windows-1254": "cp1254",
    "x-cp1254": "cp1254",
    # windows-874 (covers tis-620/iso-8859-11)
    "dos-874": "cp874",
    "iso-8859-11": "cp874",
    "iso8859-11": "cp874",
    "iso885911": "cp874",
    "tis-620": "cp874",
    "windows-874": "cp874",
    # GBK / GB18030: the standard decodes every GBK label with GB18030
    "chinese": "gb18030",
    "csgb2312": "gb18030",
    "csiso58gb231280": "gb18030",
    "gb2312": "gb18030",
    "gb_2312": "gb18030",
    "gb_2312-80": "gb18030",
    "gbk": "gb18030",
    "iso-ir-58": "gb18030",
    "x-gbk": "gb18030",
    "gb18030": "gb18030",
    # Big5 (the standard's Big5 is Big5-HKSCS)
    "big5": "big5hkscs",
    "big5-hkscs": "big5hkscs",
    "cn-big5": "big5hkscs",
    "csbig5": "big5hkscs",
    "x-x-big5": "big5hkscs",
    # Japanese
    "cseucpkdfmtjapanese": "euc_jp",
    "euc-jp": "euc_jp",
    "x-euc-jp": "euc_jp",
    "csiso2022jp": "iso2022_jp",
    "iso-2022-jp": "iso2022_jp",
    "csshiftjis": "cp932",
    "ms932": "cp932",
    "ms_kanji": "cp932",
    "shift-jis": "cp932",
    "shift_jis": "cp932",
    "sjis": "cp932",
    "windows-31j": "cp932",
    "x-sjis": "cp932",
    # Korean (the standard's EUC-KR is Windows-949)
    "cseuckr": "cp949",
    "csksc56011987": "cp949",
    "euc-kr": "cp949",
    "iso-ir-149": "cp949",
    "korean": "cp949",
    "ks_c_5601-1987": "cp949",
    "ks_c_5601-1989": "cp949",
    "ksc5601": "cp949",
    "ksc_5601": "cp949",
    "windows-949": "cp949",
    # Cyrillic / Mac
    "cskoi8r": "koi8_r",
    "koi": "koi8_r",
    "koi8": "koi8_r",
    "koi8-r": "koi8_r",
    "koi8_r": "koi8_r",
    "koi8-ru": "koi8_u",
    "koi8-u": "koi8_u",
    "csmacintosh": "mac_roman",
    "mac": "mac_roman",
    "macintosh": "mac_roman",
    "x-mac-roman": "mac_roman",
    "x-mac-cyrillic": "mac_cyrillic",
    "x-mac-ukrainian": "mac_cyrillic",
    # UTF-16 (a bare "utf-16" label means little-endian)
    "csunicode": "utf-16-le",
    "iso-10646-ucs-2": "utf-16-le",
    "ucs-2": "utf-16-le",
    "unicode": "utf-16-le",
    "unicodefeff": "utf-16-le",
    "utf-16": "utf-16-le",
    "utf-16le": "utf-16-le",
    "unicodefffe": "utf-16-be",
    "utf-16be": "utf-16-be",
    # Replacement encodings: never decode with these
    "csiso2022kr": None,
    "hz-gb-2312": None,
    "iso-2022-cn": None,
    "iso-2022-cn-ext": None,
    "iso-2022-kr": None,
    "replacement": None,
}

# Python canonical codec names (codecs.lookup(...).name) that the standard
# widens to a superset. Catches labels outside the table above that Python
# still resolves (e.g. "latin-1", "shiftjis", "euc_kr").
_SUPERSETS: dict[str, str] = {
    "ascii": "cp1252",
    "latin-1": "cp1252",
    "iso8859-1": "cp1252",
    "iso8859-9": "cp1254",
    "iso8859-11": "cp874",
    "tis-620": "cp874",
    "gb2312": "gb18030",
    "gbk": "gb18030",
    "shift_jis": "cp932",
    "euc_kr": "cp949",
    "big5": "big5hkscs",
    "utf-16": "utf-16-le",
}

# Single-byte Windows code pages leave a few bytes undefined; the standard
# maps them to the matching C1 control instead of failing.
_SINGLE_BYTE_WINDOWS = {"cp874", "cp1250", "cp1251", "cp1252", "cp1253", "cp1254"}
_SINGLE_BYTE_WINDOWS.update({"cp1255", "cp1256", "cp1257", "cp1258"})

_C1_ERROR_HANDLER = "markitai-c1"


def _c1_fallback(error: UnicodeError) -> tuple[str, int]:
    """Decode an undefined single-byte code point as its C1 control."""
    if not isinstance(error, UnicodeDecodeError):
        raise error
    chunk = error.object[error.start : error.end]
    return "".join(chr(b) for b in bytes(chunk)), error.end


codecs.register_error(_C1_ERROR_HANDLER, _c1_fallback)

_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
)

_CHARSET_PARAM_RE = re.compile(r"charset\s*=\s*[\"']?\s*([^\s\"';,]+)", re.IGNORECASE)
_META_TAG_RE = re.compile(rb"<meta\b[^>]*>", re.IGNORECASE)
_COMMENT_RE = re.compile(rb"<!--.*?(?:-->|$)", re.DOTALL)
_META_ATTR_RE = re.compile(
    rb"""([^\s=/>"']+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?"""
)

#: Bytes of an HTML document the ``<meta>`` prescan looks at (WHATWG).
META_PRESCAN_BYTES = 1024
#: A later ``<meta charset>`` is still a better hint than statistics.
_LATE_META_SCAN_BYTES = 16384
#: Bytes handed to the statistical detector.
_DETECT_SAMPLE_BYTES = 65536
#: Fewer non-ASCII bytes than this are too few for a statistical guess.
_MIN_DETECT_NON_ASCII = 20
#: A decode whose bad bytes stay under this share of the non-ASCII bytes is
#: a corrupt document in that encoding, not a different encoding.
_NEAR_MISS_RATIO = 0.01

_ASCII_BYTES = bytes(range(0x80))
_LONE_SURROGATE_RE = re.compile("[\udc80-\udcff]")

#: Legacy multi-byte encodings the detector can tell apart reliably.
_MULTIBYTE_LEGACY = ("gb18030", "big5hkscs", "cp932", "euc_jp", "cp949")

#: Latin-script single-byte code pages the detector confuses with each other.
#: windows-1254 is left out: Turkish letters decode as other letters under
#: windows-1252 and would always look plausible.
_LATIN_SINGLE_BYTE = frozenset(
    {
        "cp1250",
        "cp1257",
        "cp1258",
        "cp437",
        "cp775",
        "cp850",
        "cp852",
        "cp858",
        "hp-roman8",
        "iso8859-2",
        "iso8859-3",
        "iso8859-4",
        "iso8859-10",
        "iso8859-13",
        "iso8859-14",
        "iso8859-15",
        "iso8859-16",
        "mac-iceland",
        "mac-latin2",
        "mac-roman",
    }
)

#: Non-letters that Western text uses. Superscripts, ``¾`` and the like are
#: left out on purpose: they are what Polish or Slovak letters (``ą``, ``ł``,
#: ``ľ``) turn into when windows-1250 bytes are read as windows-1252.
_WESTERN_PUNCTUATION = frozenset("\u00a0\u00ad¡¢£§©«®°±·»½¿×÷‚„…†‡‰‹›‘’“”•–—™€")


def resolve_charset_label(label: str | None) -> str | None:
    """Map an encoding label to the Python codec a browser would decode with.

    Returns None for unknown labels and for the standard's "replacement"
    encodings, so callers fall through to the next source.
    """
    if not label:
        return None
    normalized = label.strip().strip("\"'").strip().lower()
    if not normalized:
        return None
    if normalized in _WHATWG_LABELS:
        return _WHATWG_LABELS[normalized]
    try:
        canonical = codecs.lookup(normalized).name
    except LookupError:
        return None
    return _SUPERSETS.get(canonical, canonical)


def charset_from_content_type(content_type: str | None) -> str | None:
    """Return the resolved codec of a ``Content-Type`` ``charset`` parameter."""
    if not content_type:
        return None
    match = _CHARSET_PARAM_RE.search(content_type)
    if not match:
        return None
    return resolve_charset_label(match.group(1))


def _meta_attributes(tag: bytes) -> dict[str, str]:
    """Parse a ``<meta ...>`` tag's attributes (first occurrence wins)."""
    attrs: dict[str, str] = {}
    body = tag[len(b"<meta") :].rstrip(b">").rstrip(b"/")
    for match in _META_ATTR_RE.finditer(body):
        name = match.group(1).decode("ascii", errors="ignore").lower()
        value = next((g for g in match.groups()[1:] if g is not None), b"")
        attrs.setdefault(name, value.decode("ascii", errors="ignore"))
    return attrs


def _meta_label(tag: bytes) -> str | None:
    """Return the encoding label a single ``<meta>`` tag declares, if any."""
    attrs = _meta_attributes(tag)
    # <meta charset="gbk">
    if attrs.get("charset"):
        return attrs["charset"]
    # <meta http-equiv="Content-Type" content="text/html; charset=gbk">
    if attrs.get("http-equiv", "").strip().lower() == "content-type":
        param = _CHARSET_PARAM_RE.search(attrs.get("content", ""))
        if param:
            return param.group(1)
    return None


def sniff_meta_charset(data: bytes, limit: int = META_PRESCAN_BYTES) -> str | None:
    """Prescan the start of an HTML document for a declared encoding.

    Returns the resolved codec. As in the standard, a UTF-16 declaration
    inside the document itself means UTF-8 (the bytes were readable as
    ASCII, so they cannot be UTF-16).
    """
    head = _COMMENT_RE.sub(b"", bytes(data[:limit]))
    for tag in _META_TAG_RE.findall(head):
        label = _meta_label(tag)
        if not label:
            continue
        codec = resolve_charset_label(label)
        if codec is None:
            continue
        if codec.startswith("utf-16"):
            return "utf-8"
        return codec
    return None


def _bom_encoding(data: bytes) -> tuple[str, int] | None:
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return encoding, len(bom)
    return None


def _decode(data: bytes, codec: str, *, strict: bool) -> str:
    if codec in _SINGLE_BYTE_WINDOWS:
        return data.decode(codec, errors=_C1_ERROR_HANDLER)
    return data.decode(codec, errors="strict" if strict else "replace")


def _utf8_invalid_ratio(data: bytes) -> float:
    """Share of the non-ASCII bytes that are not part of valid UTF-8.

    ``surrogateescape`` turns every undecodable byte into one lone surrogate,
    and real UTF-8 can never produce those, so the count is exact.
    """
    non_ascii = len(data.translate(None, _ASCII_BYTES))
    if not non_ascii:
        return 0.0
    invalid = len(_LONE_SURROGATE_RE.findall(data.decode("utf-8", "surrogateescape")))
    return invalid / non_ascii


def _is_near_miss(text: str, non_ascii: int) -> bool:
    """Whether a replace-decoded text has only a sprinkling of bad bytes."""
    bad = text.count("\ufffd")
    return 0 < bad < non_ascii * _NEAR_MISS_RATIO


def _plausible_windows_1252(data: bytes) -> bool:
    """Whether every non-ASCII byte reads as a Western letter or punctuation."""
    text = _decode(data, "cp1252", strict=False)
    return all(
        ch.isalpha() or ch in _WESTERN_PUNCTUATION for ch in text if ord(ch) >= 0x80
    )


def _detect_raw(sample: bytes) -> str | None:
    try:
        from charset_normalizer import from_bytes
    except ImportError:  # pragma: no cover - transitive dependency
        return None
    try:
        best = from_bytes(sample).best()
    except Exception:
        return None
    if best is None:
        return None
    return resolve_charset_label(best.encoding)


def _detect(data: bytes) -> str | None:
    """Statistical guess for a body with no usable declaration."""
    sample = data[:_DETECT_SAMPLE_BYTES]
    non_ascii = len(sample.translate(None, _ASCII_BYTES))
    # The detector judges only the non-ASCII bytes; a handful of them
    # ("Café", "São Paulo") yields a random guess (UTF-16, Big5, Urdu).
    if non_ascii < _MIN_DETECT_NON_ASCII:
        return None
    detected = _detect_raw(sample) or _detect_despite_corruption(sample, non_ascii)
    if detected is None:
        return None
    # Without a BOM the standard never picks UTF-16/32; Latin-1 text
    # ("Résumé") otherwise comes back as UTF-16-BE.
    if detected.startswith(("utf-16", "utf-32")):
        return None
    # The detector cannot tell the Latin code pages apart (a Portuguese CSV
    # comes back as windows-1250, "São" as "Săo"). The standard's default
    # for an undeclared document is windows-1252, so keep it whenever the
    # bytes read naturally that way.
    if detected in _LATIN_SINGLE_BYTE and _plausible_windows_1252(sample):
        return "cp1252"
    return detected


def _detect_despite_corruption(sample: bytes, non_ascii: int) -> str | None:
    """Find a CJK encoding the detector rejected over a few corrupt bytes.

    The detector discards any encoding that fails to decode, so one stray
    byte in a GBK page leaves it with no answer. Drop the bytes each CJK
    candidate cannot decode and accept the candidate the detector then
    confirms.
    """
    for codec in _MULTIBYTE_LEGACY:
        text = sample.decode(codec, errors="replace")
        if not _is_near_miss(text, non_ascii):
            continue
        cleaned = text.replace("\ufffd", "").encode(codec, errors="ignore")
        if _detect_raw(cleaned) == codec:
            return codec
    return None


def decode_body(
    data: bytes, content_type: str | None = None, *, html: bool = True
) -> tuple[str, str]:
    """Decode a fetched body the way a browser would.

    Args:
        data: Raw response body.
        content_type: The ``Content-Type`` header (may be None).
        html: Whether the body is HTML; only HTML gets the ``<meta>``
            prescan. Plain text, CSV, Markdown and JSON bodies use the BOM,
            the header and detection.

    Returns:
        ``(text, codec)`` — the decoded text and the Python codec used.
    """
    data = bytes(data)
    bom = _bom_encoding(data)
    if bom is not None:
        codec, skip = bom
        return data[skip:].decode(codec, errors="replace"), codec

    declared: list[str] = []
    header_codec = charset_from_content_type(content_type)
    if header_codec:
        declared.append(header_codec)
    if html:
        meta_codec = sniff_meta_charset(data)
        if meta_codec and meta_codec not in declared:
            declared.append(meta_codec)

    # The first declaration wins when it decodes cleanly. A declaration that
    # does not fit the bytes (a server header left at utf-8 for a GBK page)
    # yields to a later one that does, before the replacement fallback.
    for codec in declared:
        try:
            return _decode(data, codec, strict=True), codec
        except (UnicodeDecodeError, LookupError):
            continue

    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    if declared:
        codec = declared[0]
        return _decode(data, codec, strict=False), codec

    if html:
        late = sniff_meta_charset(data, limit=_LATE_META_SCAN_BYTES)
        if late:
            # A late declaration is still a declaration: a corrupt byte
            # costs a character, not the whole page.
            return _decode(data, late, strict=False), late

    # UTF-8 with a stray bad byte (a truncated multi-byte sequence, one
    # Latin-1 character pasted in) is still UTF-8.
    if _utf8_invalid_ratio(data) < _NEAR_MISS_RATIO:
        return data.decode("utf-8", errors="replace"), "utf-8"

    detected = _detect(data)
    if detected:
        try:
            return _decode(data, detected, strict=False), detected
        except LookupError:
            pass

    # The standard's fallback for an undeclared legacy document.
    return _decode(data, "cp1252", strict=False), "cp1252"


__all__ = [
    "META_PRESCAN_BYTES",
    "charset_from_content_type",
    "decode_body",
    "resolve_charset_label",
    "sniff_meta_charset",
]
