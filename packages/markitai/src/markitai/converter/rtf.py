"""Native RTF converter (stdlib only).

An RTF file is a stream of nested groups, control words and raw bytes, and
every piece of structure a Markdown reader wants is already in there: the
outline level of a paragraph, the style it points at, the bullet glyph Word 97
wrote next to a list item, the cells between ``\\trowd`` and ``\\row``. So this
module reads the format rather than stripping backslashes out of it — a
tokenizer feeds a group-scoped formatting state, paragraphs accumulate as runs,
and only at the end are those runs rendered.

The one genuinely fiddly part is text encoding. Bytes written as ``\\'xx`` mean
whatever the document's ``\\ansicpg`` codepage — or the current font's
``\\fcharset`` — says they mean, and a GBK character is two of them, so
consecutive bytes are buffered and decoded together instead of one at a time.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    conversion_failed,
    register_converter,
)
from markitai.converter.delimited import render_markdown_table

# A run of `{` with no `}` is malformed input, but a merely deep document is
# not; Word nests a handful of levels, never hundreds.
_MAX_GROUP_DEPTH = 512
_MAX_HEADING_LEVEL = 6
# A Markdown hard break is two trailing spaces, which the trailing-whitespace
# pass would otherwise eat; carry it as a sentinel and expand it afterwards.
_HARD_BREAK = "\x00BR\x00"

_CONTROL_RE = re.compile(rb"([A-Za-z]+)(-?[0-9]+)? ?")
_TEXT_RE = re.compile(rb"[^\\{}\r\n\x00]+")
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_HEADING_STYLE_RE = re.compile(r"heading\s*([1-9])", re.IGNORECASE)
_TITLE_STYLE_RE = re.compile(r"(?:^|\W)title(?:$|\W)", re.IGNORECASE)
_NUMBER_MARKER_RE = re.compile(r"^\(?(\d+)[.):]?$")
_HYPERLINK_RE = re.compile(r'HYPERLINK\s+"([^"]*)"|HYPERLINK\s+(\S+)')

# \fcharset values whose bytes are not the document codepage. 0 (ANSI) and 1
# (default) deliberately fall through to \ansicpg.
_CHARSET_CODECS: dict[int, str] = {
    77: "mac_roman",
    128: "cp932",
    129: "cp949",
    130: "cp1361",
    134: "gbk",
    136: "big5",
    161: "cp1253",
    162: "cp1254",
    163: "cp1258",
    177: "cp1255",
    178: "cp1256",
    186: "cp1257",
    204: "cp1251",
    222: "cp874",
    238: "cp1250",
}

_CODEPAGE_CODECS: dict[int, str] = {
    874: "cp874",
    932: "cp932",
    936: "gbk",
    949: "cp949",
    950: "big5",
    1250: "cp1250",
    1251: "cp1251",
    1252: "cp1252",
    1253: "cp1253",
    1254: "cp1254",
    1255: "cp1255",
    1256: "cp1256",
    1257: "cp1257",
    1258: "cp1258",
    10000: "mac_roman",
    65001: "utf-8",
}

_DEFAULT_CODEC = "cp1252"

# Control symbols that stand for one character of text.
_CONTROL_SYMBOL_TEXT: dict[bytes, str] = {
    b"\\": "\\",
    b"{": "{",
    b"}": "}",
    b"~": " ",
    b"_": "‑",
}

# Control words that stand for one character of text.
_SYMBOL_WORDS: dict[str, str] = {
    "emdash": "—",
    "endash": "–",
    "lquote": "‘",
    "rquote": "’",
    "ldblquote": "“",
    "rdblquote": "”",
    "bullet": "•",
    "enspace": " ",
    "emspace": " ",
    "qmspace": " ",
    "zwnj": "‌",
    "zwj": "‍",
    "ltrmark": "‎",
    "rtlmark": "‏",
}

# Destinations whose content is not document text. Footnotes are dropped
# rather than inlined: a Markdown footnote needs a reference at the call site,
# and RTF puts none there.
_SKIPPED_DESTINATIONS = frozenset(
    {
        "colortbl",
        "filetbl",
        "revtbl",
        "rsidtbl",
        "xmlnstbl",
        "themedata",
        "datastore",
        "latentstyles",
        "generator",
        "colorschememapping",
        "template",
        "header",
        "headerl",
        "headerr",
        "headerf",
        "footer",
        "footerl",
        "footerr",
        "footerf",
        "footnote",
        "ftnsep",
        "ftnsepc",
        "ftncn",
        "aftnsep",
        "aftnsepc",
        "aftncn",
        "annotation",
        "atnauthor",
        "atnid",
        "atnref",
        "atntime",
        "atrfstart",
        "atrfend",
        "comment",
        "doccomm",
        "company",
        "operator",
        "keywords",
        "subject",
        "category",
        "manager",
        "hlinkbase",
        "password",
        "bkmkstart",
        "bkmkend",
        "object",
        "objdata",
        "objclass",
        "objname",
        "result",
        "shppict",
        "shpinst",
        "do",
        "falt",
        "fname",
        "panose",
        "xe",
        "tc",
        "tcn",
        "mail",
        "private",
        "protusertbl",
        "userprops",
        "factoidname",
        "pntxta",
        "pntxtb",
        "upr",
        "author",
        "creatim",
        "revtim",
        "printim",
        "buptim",
    }
)

# Destinations this module reads instead of skipping, keyed to the sink their
# text goes to. "skip" means the group is entered but its text discarded.
_DESTINATION_SINKS: dict[str, str] = {
    **dict.fromkeys(_SKIPPED_DESTINATIONS, "skip"),
    "fonttbl": "fonttbl",
    "stylesheet": "stylesheet",
    "info": "info",
    "title": "title",
    "listtable": "listtable",
    "listoverridetable": "listoverride",
    "pn": "pn",
    "pntext": "pntext",
    "fldinst": "fldinst",
    "fldrslt": "body",
    "pict": "pict",
    # Word writes the same image twice, as \shppict and \nonshppict; the
    # second copy is skipped without counting so the tally stays honest.
    "nonshppict": "skipalt",
}

# `\*\foo` means "skip this group if you do not know \foo"; these are the ones
# this module does know, and Word writes all of them starred.
_KNOWN_STARRED = frozenset({"pn", "fldinst", "listtable", "listoverridetable"})

# \levelnfc values that mean "bullet", not "number".
_BULLET_NFC = frozenset({23, 255})


class _RtfError(Exception):
    """Malformed RTF: bad header, unbalanced groups, runaway nesting."""


@dataclass
class _Run:
    """One stretch of text sharing a single character format."""

    text: str
    bold: bool = False
    italic: bool = False
    strike: bool = False
    size: int | None = None

    def same_format(self, other: _Run) -> bool:
        return (
            self.bold == other.bold
            and self.italic == other.italic
            and self.strike == other.strike
        )


@dataclass
class _Para:
    """A paragraph, kept as runs until the whole document has been read."""

    runs: list[_Run]
    outline: int | None = None
    style: int | None = None
    list_kind: str | None = None  # "bullet" | "number"
    list_level: int = 0
    list_number: int | None = None


@dataclass
class _Table:
    rows: list[list[str]]


@dataclass
class _Field:
    """One `{\\field ...}` group: the URL its `\\fldinst` named, if any."""

    url: str | None = None


@dataclass
class _State:
    """Formatting and destination state, copied on `{` and restored on `}`."""

    bold: bool = False
    italic: bool = False
    strike: bool = False
    hidden: bool = False
    size: int | None = None
    font: int | None = None
    style: int | None = None
    outline: int | None = None
    intbl: bool = False
    ilvl: int = 0
    ls: int | None = None
    uc: int = 1
    sink: str = "body"
    star: bool = False


@dataclass
class _Document:
    """Everything the renderer needs once parsing is done."""

    blocks: list[_Para | _Table] = field(default_factory=list)
    styles: dict[int, str] = field(default_factory=dict)
    title: str | None = None
    pictures: int = 0


class _Parser:
    """Reads one RTF byte string into paragraphs, tables and metadata."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._doc = _Document()

        self._state = _State()
        self._stack: list[_State] = []
        self._depth = 0
        self._closers: dict[int, list[Callable[[], None]]] = {}

        self._pending = bytearray()
        self._codepage: str | None = None
        self._font_charsets: dict[int, int] = {}
        self._default_font: int | None = None
        self._high_surrogate: int | None = None

        self._runs: list[_Run] = []
        self._cell_lines: list[str] = []
        self._cells: list[str] = []
        self._rows: list[list[str]] = []
        self._in_row = False

        self._styles: dict[int, str] = self._doc.styles
        self._style_name: list[str] = []
        self._style_number = 0

        self._list_levels: dict[int, list[int | None]] = {}
        self._ls_to_listid: dict[int, int] = {}
        self._pending_levels: list[int | None] = []
        self._current_listid: int | None = None

        self._pn_text: str | None = None
        self._pn_kind: str | None = None
        self._pn_level: int | None = None

        self._fields: list[_Field] = []
        self._fldinst: list[str] = []
        self._title_text: list[str] = []
        self._pntext_buf: list[str] = []

    # ---------------------------------------------------------------- driving

    def parse(self) -> _Document:
        data = self._data
        length = len(data)
        index = 0
        while index < length:
            char = data[index : index + 1]
            if char == b"{":
                self._flush_bytes()
                self._open_group()
                index += 1
            elif char == b"}":
                self._flush_bytes()
                self._close_group()
                index += 1
            elif char == b"\\":
                index = self._read_control(index)
            elif char in b"\r\n\x00":
                index += 1
            else:
                match = _TEXT_RE.match(data, index)
                if match is None:  # pragma: no cover - defensive
                    index += 1
                    continue
                self._pending.extend(match.group(0))
                index = match.end()

        self._flush_bytes()
        if self._depth != 0:
            raise _RtfError(f"unbalanced group: {self._depth} unclosed '{{'")
        self._flush_paragraph()
        self._flush_table()
        return self._doc

    def _open_group(self) -> None:
        if self._depth >= _MAX_GROUP_DEPTH:
            raise _RtfError("group nesting deeper than the format allows")
        self._stack.append(self._state)
        self._state = dataclasses.replace(self._state, star=False)
        self._depth += 1
        if self._stack[-1].sink == "stylesheet":
            self._begin_style_definition()

    def _close_group(self) -> None:
        if not self._stack:
            raise _RtfError("unbalanced group: '}' with no matching '{'")
        for closer in self._closers.pop(self._depth, []):
            closer()
        self._state = self._stack.pop()
        self._depth -= 1

    def _on_close(self, closer: Callable[[], None]) -> None:
        self._closers.setdefault(self._depth, []).append(closer)

    # -------------------------------------------------------------- tokenizer

    def _read_control(self, index: int) -> int:
        data = self._data
        start = index + 1
        char = data[start : start + 1]

        if char.isalpha():
            match = _CONTROL_RE.match(data, start)
            if match is None:  # pragma: no cover - isalpha guarantees a match
                return start + 1
            name = match.group(1).decode("ascii")
            raw_param = match.group(2)
            param = int(raw_param) if raw_param is not None else None
            self._flush_bytes()
            if name == "u" and param is not None:
                if self._state.sink not in ("skip", "skipalt"):
                    self._unicode_char(param)
                return self._skip_fallback(match.end(), self._state.uc)
            self._control_word(name, param)
            return match.end()

        if char == b"'":
            try:
                value = int(data[start + 1 : start + 3], 16)
            except ValueError:
                return start + 1
            self._pending.append(value)
            return start + 3

        self._flush_bytes()
        if char in _CONTROL_SYMBOL_TEXT:
            self._text(_CONTROL_SYMBOL_TEXT[char])
        elif char in (b"\r", b"\n"):
            self._flush_paragraph()
        elif char == b"*":
            self._state.star = True
        # `\-` (optional hyphen), `\:`, `\|` and anything unrecognised carry no
        # text of their own.
        return start + 1

    def _skip_fallback(self, index: int, count: int) -> int:
        """Skip the ``\\ucN`` ANSI fallback that follows a ``\\uN``."""
        data = self._data
        length = len(data)
        while count > 0 and index < length:
            char = data[index : index + 1]
            if char in b"{}":
                break
            if char == b"\\":
                if data[index + 1 : index + 2] == b"'":
                    index += 4
                else:
                    match = _CONTROL_RE.match(data, index + 1)
                    index = match.end() if match else index + 2
                count -= 1
                continue
            if char in b"\r\n":
                index += 1
                continue
            index += 1
            count -= 1
        return index

    def _unicode_char(self, value: int) -> None:
        if value < 0:
            value += 0x10000
        if 0xD800 <= value <= 0xDBFF:
            self._high_surrogate = value
            return
        if 0xDC00 <= value <= 0xDFFF and self._high_surrogate is not None:
            high = self._high_surrogate
            self._high_surrogate = None
            value = 0x10000 + ((high - 0xD800) << 10) + (value - 0xDC00)
        else:
            self._high_surrogate = None
        try:
            self._text(chr(value))
        except ValueError:  # pragma: no cover - chr only rejects out-of-range
            pass

    # --------------------------------------------------------------- decoding

    def _codec(self) -> str:
        font = self._state.font if self._state.font is not None else self._default_font
        if font is not None:
            charset = self._font_charsets.get(font)
            if charset is not None and charset in _CHARSET_CODECS:
                return _CHARSET_CODECS[charset]
        return self._codepage or _DEFAULT_CODEC

    def _flush_bytes(self) -> None:
        if not self._pending:
            return
        raw = bytes(self._pending)
        self._pending.clear()
        self._text(raw.decode(self._codec(), errors="replace"))

    # ---------------------------------------------------------------- control

    def _control_word(self, name: str, param: int | None) -> None:
        state = self._state
        starred = state.star
        state.star = False

        if state.sink in ("skip", "skipalt"):
            if name == "pict" and state.sink == "skip":
                self._doc.pictures += 1
            return

        sink = _DESTINATION_SINKS.get(name)
        if sink is not None:
            self._begin_destination(name, sink)
            return
        if starred and name not in _KNOWN_STARRED:
            state.sink = "skip"
            return

        handler = _SINK_HANDLERS.get(state.sink)
        if handler is not None:
            handler(self, name, param)
            return
        self._body_control(name, param)

    def _begin_destination(self, name: str, sink: str) -> None:
        state = self._state
        if name == "title":
            if state.sink != "info":
                state.sink = "skip"
                return
            self._title_text = []
            self._on_close(self._end_title)
            state.sink = "title"
            return
        if name == "pict":
            self._doc.pictures += 1
            state.sink = "skip"
            return
        if name == "pntext":
            self._pntext_buf = []
            self._on_close(self._end_pntext)
            state.sink = "pntext"
            return
        if name == "fldinst":
            self._fldinst = []
            self._on_close(self._end_fldinst)
            state.sink = "fldinst"
            return
        if name == "fldrslt":
            self._begin_field_result()
            state.sink = "body"
            return
        state.sink = sink

    # ------------------------------------------------------------ sink: body

    def _body_control(self, name: str, param: int | None) -> None:
        state = self._state
        flag = param is None or param != 0

        if name == "par":
            self._flush_paragraph()
        elif name == "pard":
            state.outline = None
            state.style = None
            state.intbl = False
            state.ilvl = 0
            state.ls = None
        elif name == "plain":
            state.bold = state.italic = state.strike = state.hidden = False
            state.size = None
        elif name == "b":
            state.bold = flag
        elif name == "i":
            state.italic = flag
        elif name in ("strike", "striked"):
            state.strike = flag
        elif name == "v":
            state.hidden = flag
        elif name == "fs":
            state.size = param
        elif name == "f":
            state.font = param
        elif name == "s":
            state.style = param
        elif name == "outlinelevel":
            state.outline = param if param is not None else 0
        elif name == "intbl":
            state.intbl = True
        elif name == "ilvl":
            state.ilvl = param or 0
        elif name == "ls":
            state.ls = param
        elif name == "uc":
            state.uc = max(0, param or 0)
        elif name == "tab":
            self._text("    ")
        elif name == "line":
            self._line_break()
        elif name in ("sect", "page", "column"):
            self._flush_paragraph()
        elif name == "cell":
            self._end_cell()
        elif name == "nestcell":
            self._flush_paragraph()
        elif name == "row":
            self._end_row()
        elif name == "nestrow":
            self._flush_paragraph()
        elif name == "trowd":
            self._in_row = True
        elif name == "ansicpg":
            self._codepage = _CODEPAGE_CODECS.get(param or 0)
        elif name == "ansi":
            self._codepage = self._codepage or _DEFAULT_CODEC
        elif name == "mac":
            self._codepage = self._codepage or "mac_roman"
        elif name == "deff":
            self._default_font = param
        elif name == "field":
            self._begin_field()
        elif name in _SYMBOL_WORDS:
            self._text(_SYMBOL_WORDS[name])

    # ------------------------------------------------------- sink: side tables

    def _fonttbl_control(self, name: str, param: int | None) -> None:
        if name == "f":
            self._state.font = param
        elif name == "fcharset" and param is not None:
            font = self._state.font
            if font is None:
                font = self._default_font
            if font is not None:
                self._font_charsets[font] = param

    def _stylesheet_control(self, name: str, param: int | None) -> None:
        if name == "s":
            self._style_number = param or 0

    def _listtable_control(self, name: str, param: int | None) -> None:
        if name == "list":
            self._pending_levels = []
        elif name == "listlevel":
            self._pending_levels.append(None)
        elif name in ("levelnfc", "levelnfcn") and param is not None:
            if self._pending_levels:
                self._pending_levels[-1] = param
        elif name == "listid" and param is not None:
            self._list_levels[param] = list(self._pending_levels)
            self._pending_levels = []

    def _listoverride_control(self, name: str, param: int | None) -> None:
        if name == "listid":
            self._current_listid = param
        elif name == "ls" and param is not None and self._current_listid is not None:
            self._ls_to_listid[param] = self._current_listid

    def _pn_control(self, name: str, param: int | None) -> None:
        if name == "pnlvlblt":
            self._pn_kind = "bullet"
        elif name in ("pnlvlbody", "pndec", "pnlvlcont"):
            self._pn_kind = "number"
        elif name == "pnlvl" and param is not None:
            self._pn_level = max(0, param - 1)

    # ----------------------------------------------------------- destinations

    def _begin_style_definition(self) -> None:
        self._style_name = []
        self._style_number = 0
        self._on_close(self._end_style_definition)
        self._state.sink = "styledef"

    def _end_style_definition(self) -> None:
        name = "".join(self._style_name).split(";", 1)[0].strip()
        if name:
            self._styles[self._style_number] = name

    def _end_title(self) -> None:
        title = "".join(self._title_text).strip()
        if title and not self._doc.title:
            self._doc.title = title

    def _end_pntext(self) -> None:
        self._pn_text = "".join(self._pntext_buf)

    def _end_fldinst(self) -> None:
        instruction = "".join(self._fldinst)
        if not self._fields:
            return
        match = _HYPERLINK_RE.search(instruction)
        if match:
            self._fields[-1].url = match.group(1) or match.group(2)

    def _begin_field(self) -> None:
        record = _Field()
        self._fields.append(record)

        def _drop() -> None:
            if record in self._fields:
                self._fields.remove(record)

        self._on_close(_drop)

    def _begin_field_result(self) -> None:
        if not self._fields:
            return
        record = self._fields[-1]
        mark = len(self._runs)

        def _wrap() -> None:
            start = min(mark, len(self._runs))
            captured = self._runs[start:]
            del self._runs[start:]
            text = "".join(run.text for run in captured).strip()
            if not text:
                return
            if record.url:
                self._runs.append(
                    _Run(f"[{text}]({record.url})", size=self._state.size)
                )
            else:
                self._runs.extend(captured)

        self._on_close(_wrap)

    # ------------------------------------------------------------------- text

    def _text(self, value: str) -> None:
        if not value:
            return
        sink = self._state.sink
        if sink == "body":
            if not self._state.hidden:
                self._runs.append(
                    _Run(
                        value,
                        bold=self._state.bold,
                        italic=self._state.italic,
                        strike=self._state.strike,
                        size=self._state.size,
                    )
                )
        elif sink == "title":
            self._title_text.append(value)
        elif sink == "styledef":
            self._style_name.append(value)
        elif sink == "pntext":
            self._pntext_buf.append(value)
        elif sink == "fldinst":
            self._fldinst.append(value)

    def _line_break(self) -> None:
        if self._state.sink == "body":
            self._runs.append(_Run(_HARD_BREAK + "\n", size=self._state.size))

    # ------------------------------------------------------------- paragraphs

    def _flush_paragraph(self) -> None:
        state = self._state
        if state.sink != "body":
            self._runs = []
            self._reset_list_state()
            return

        runs = self._runs
        self._runs = []
        para = _Para(
            runs=runs,
            outline=state.outline,
            style=state.style,
            **self._list_of(state),
        )
        self._reset_list_state()

        if state.intbl or self._in_row:
            text = _render_runs(runs).strip()
            if text:
                self._cell_lines.append(text.replace(_HARD_BREAK, ""))
            return

        if not runs:
            self._flush_table()
            return
        self._flush_table()
        self._doc.blocks.append(para)

    def _list_of(self, state: _State) -> dict:
        """Decide whether this paragraph is a list item, and of what kind."""
        if self._pn_text is not None or self._pn_kind is not None:
            marker = (self._pn_text or "").replace("\t", " ").strip()
            kind = self._pn_kind
            number = None
            # The glyph itself never reaches the output: \pntext is a
            # destination of its own, so all the marker decides is "-" vs "1.".
            match = _NUMBER_MARKER_RE.match(marker)
            if match:
                kind = kind or "number"
                number = int(match.group(1))
            else:
                kind = kind or "bullet"
            level = self._pn_level if self._pn_level is not None else state.ilvl
            return {"list_kind": kind, "list_level": level, "list_number": number}

        if state.ls is not None:
            listid = self._ls_to_listid.get(state.ls, state.ls)
            levels = self._list_levels.get(listid, [])
            nfc = levels[state.ilvl] if state.ilvl < len(levels) else None
            kind = "bullet" if nfc is None or nfc in _BULLET_NFC else "number"
            return {"list_kind": kind, "list_level": state.ilvl, "list_number": None}

        return {"list_kind": None, "list_level": 0, "list_number": None}

    def _reset_list_state(self) -> None:
        self._pn_text = None
        self._pn_kind = None
        self._pn_level = None

    # ------------------------------------------------------------------ tables

    def _end_cell(self) -> None:
        self._flush_paragraph()
        self._cells.append("\n".join(self._cell_lines))
        self._cell_lines = []

    def _end_row(self) -> None:
        self._flush_paragraph()
        if self._cell_lines:
            self._cells.append("\n".join(self._cell_lines))
            self._cell_lines = []
        if self._cells:
            self._rows.append(self._cells)
            self._cells = []
        self._in_row = False

    def _flush_table(self) -> None:
        if self._rows:
            self._doc.blocks.append(_Table(self._rows))
            self._rows = []


_SINK_HANDLERS: dict[str, Callable[[_Parser, str, int | None], None]] = {
    "fonttbl": _Parser._fonttbl_control,
    "stylesheet": _Parser._stylesheet_control,
    "styledef": _Parser._stylesheet_control,
    "listtable": _Parser._listtable_control,
    "listoverride": _Parser._listoverride_control,
    "pn": _Parser._pn_control,
}


# --------------------------------------------------------------- rendering


def _render_runs(runs: list[_Run], *, drop_bold: bool = False) -> str:
    """Merge adjacent runs sharing a format, then apply Markdown emphasis.

    ``drop_bold`` is for headings, which a word processor writes in bold and
    Markdown renders bold on its own; ``# **Title**`` is noise, not fidelity.
    """
    merged: list[_Run] = []
    for run in runs:
        if drop_bold:
            run = dataclasses.replace(run, bold=False)
        if merged and merged[-1].same_format(run):
            merged[-1].text += run.text
        else:
            merged.append(dataclasses.replace(run))

    out: list[str] = []
    for run in merged:
        text = run.text
        if not text:
            continue
        core = text.strip()
        if not core or not (run.bold or run.italic or run.strike):
            out.append(text)
            continue
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()) :]
        emphasis = "*" * ((2 if run.bold else 0) + (1 if run.italic else 0))
        body = f"{emphasis}{core}{emphasis}" if emphasis else core
        if run.strike:
            body = f"~~{body}~~"
        out.append(f"{lead}{body}{trail}")
    return "".join(out)


def _heading_level(para: _Para, styles: dict[int, str]) -> int | None:
    """The heading level from ``\\outlinelevel`` or the paragraph's style."""
    if para.outline is not None and 0 <= para.outline <= 8:
        return min(para.outline + 1, _MAX_HEADING_LEVEL)
    if para.style is not None:
        name = styles.get(para.style, "")
        match = _HEADING_STYLE_RE.search(name)
        if match:
            return min(int(match.group(1)), _MAX_HEADING_LEVEL)
        if _TITLE_STYLE_RE.search(name):
            return 1
    return None


def _body_size(blocks: list[_Para | _Table], styles: dict[int, str]) -> int | None:
    """The dominant ``\\fs`` of ordinary paragraphs, in half-points."""
    sizes: Counter[int] = Counter()
    for block in blocks:
        if not isinstance(block, _Para) or _heading_level(block, styles) is not None:
            continue
        for run in block.runs:
            if run.size is not None and run.text.strip():
                sizes[run.size] += len(run.text)
    return sizes.most_common(1)[0][0] if sizes else None


def _looks_like_heading(para: _Para, text: str, body_size: int | None) -> bool:
    """A short, wholly bold paragraph set at least 4pt above the body size.

    Deliberately narrow: this is the guess that fires when a document carries
    no outline levels and no named styles, and a wrong guess turns a sentence
    into a section title.
    """
    if body_size is None or para.list_kind is not None:
        return False
    visible = [run for run in para.runs if run.text.strip()]
    if not visible or not all(run.bold for run in visible):
        return False
    size = max((run.size for run in visible if run.size is not None), default=None)
    if size is None or size < body_size + 8:  # \fs is half-points
        return False
    return len(text.split()) < 12


def _render_document(doc: _Document, styles: dict[int, str]) -> str:
    body_size = _body_size(doc.blocks, styles)
    counters: dict[int, int] = {}
    pieces: list[tuple[str, str]] = []

    for block in doc.blocks:
        if isinstance(block, _Table):
            counters.clear()
            table = render_markdown_table(block.rows)
            if table:
                pieces.append(("table", table))
            continue

        text = _render_runs(block.runs).strip()
        if not text:
            continue

        if block.list_kind is not None:
            level = max(0, min(block.list_level, 8))
            if block.list_kind == "number":
                if block.list_number is not None:
                    number = block.list_number
                else:
                    number = counters.get(level, 0) + 1
                counters[level] = number
                for deeper in [key for key in counters if key > level]:
                    del counters[deeper]
                marker = f"{number}."
            else:
                marker = "-"
            pieces.append(("list", f"{'  ' * level}{marker} {text}"))
            continue

        counters.clear()
        level = _heading_level(block, styles)
        if level is None and _looks_like_heading(block, text, body_size):
            level = 1 if not any(kind == "heading" for kind, _ in pieces) else 2
        if level is not None:
            heading = _render_runs(block.runs, drop_bold=True).strip()
            pieces.append(("heading", f"{'#' * level} {heading or text}"))
        else:
            pieces.append(("para", text))

    out: list[str] = []
    for index, (kind, text) in enumerate(pieces):
        if index:
            previous = pieces[index - 1][0]
            out.append("\n" if kind == "list" and previous == "list" else "\n\n")
        out.append(text)
    return _normalise("".join(out))


def _normalise(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    joined = "\n".join(lines).replace(_HARD_BREAK, "  ")
    joined = _BLANK_RUN_RE.sub("\n\n", joined).strip()
    return f"{joined}\n" if joined else ""


def rtf_to_markdown(data: bytes) -> tuple[str, str | None, int]:
    """Convert RTF bytes to Markdown, its title and its dropped-image count."""
    header = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    if not header.startswith(b"{\\rt"):
        raise _RtfError("not an RTF document (no '{\\rtf' header)")

    document = _Parser(data).parse()
    return (
        _render_document(document, document.styles),
        document.title,
        document.pictures,
    )


@register_converter(FileFormat.RTF)
class RtfConverter(BaseConverter):
    """Converter for Rich Text Format documents."""

    supported_formats = [FileFormat.RTF]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        logger.debug("[RtfConverter] Converting: {}", input_path.name)

        metadata: dict = {
            "source": str(input_path),
            "format": "RTF",
            "converter": "rtf",
        }

        try:
            data = input_path.read_bytes()
        except OSError as exc:
            logger.warning("[RtfConverter] {}: {}", input_path.name, exc)
            conversion_failed(f"unreadable RTF file: {exc}")

        try:
            markdown, title, pictures = rtf_to_markdown(data)
        except _RtfError as exc:
            logger.warning("[RtfConverter] {}: {}", input_path.name, exc)
            conversion_failed(f"malformed RTF: {exc}")
        except (RecursionError, MemoryError, UnicodeError) as exc:
            logger.warning("[RtfConverter] {}: {}", input_path.name, exc)
            conversion_failed(f"unreadable RTF: {exc}")

        if not title:
            for line in markdown.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        if title:
            metadata["title"] = title
        if pictures:
            metadata["dropped_images"] = pictures

        return ConvertResult(markdown=markdown, images=[], metadata=metadata)
