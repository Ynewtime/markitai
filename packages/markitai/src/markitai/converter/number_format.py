"""Spreadsheet number format codes, applied the way a spreadsheet shows them.

A cell stores ``0.256``; its format ``0.00%`` makes Excel show ``25.60%``.
:func:`format_cell` renders a cell value through its format code
(ISO/IEC 29500-1 §18.8.30/31), following markitai-rust's reader so both
write the same text:

- numbers: digit placeholders ``0 # ?``, decimals, thousands grouping and
  scaling commas, percent, scientific notation, literals (quoted, escaped,
  ``[$€-407]`` currency), ``_x`` padding as one space, colors ignored, up to
  four ``;`` sections with their sign rules and conditions;
- dates and times stay ISO-like (``2024-03-05 14:30``, whose order no reader
  can misread), showing only the parts the format names (no seconds for
  ``h:mm``), elapsed spans (``[h]:mm:ss``) as durations; a format that names
  its month or weekday in English and shows a four-digit year
  (``mmmm d, yyyy``), or labels its month the CJK way
  (``yyyy"年"m"月"d"日"``), is written as it stands;
- a text section (``@``) wraps a text value.

Anything outside that grammar (fractions, locale-specific codes) returns
None and the caller shows the value as a General cell: approximating a
format silently would be worse than not applying it.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

_COLORS = frozenset(
    {"black", "blue", "cyan", "green", "magenta", "red", "white", "yellow"}
)
_PLAIN_LITERALS = frozenset("$-+():/ ")
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_UNIT_SECONDS = {"h": 3600, "m": 60, "s": 1}


class _Unsupported(Exception):
    """The code is outside the implemented grammar."""


# Tokens: ("digit", place), ("decimal",), ("comma",), ("percent",),
# ("literal", text), ("exp", plus), ("at",), ("run", letter, length),
# ("ampm", full), ("slash",)
_Tok = tuple[Any, ...]


@dataclass
class _Section:
    kind: str  # "general", "number", "date", "text", "empty"
    toks: list[_Tok] = field(default_factory=list)
    condition: tuple[str, float] | None = None
    prefix: str = ""  # around a General body
    suffix: str = ""
    # number
    grouping: bool = False
    scale: int = 0
    percents: int = 0
    exp: bool = False
    # date: (date, time, seconds, elapsed unit or "", smallest unit)
    date: bool = False
    time: bool = False
    seconds: bool = False
    elapsed: str = ""
    smallest: str = ""
    spelled: bool = False


def _push_literal(toks: list[_Tok], text: str) -> None:
    if toks and toks[-1][0] == "literal":
        toks[-1] = ("literal", toks[-1][1] + text)
    else:
        toks.append(("literal", text))


def _split_sections(code: str) -> list[str]:
    parts = [""]
    i = 0
    while i < len(code):
        c = code[i]
        if c == ";":
            parts.append("")
            i += 1
        elif c in '"[':
            close = '"' if c == '"' else "]"
            end = code.find(close, i + 1)
            if end < 0:
                raise _Unsupported
            parts[-1] += code[i : end + 1]
            i = end + 1
        elif c in "\\_*":
            if i + 1 >= len(code):
                raise _Unsupported
            parts[-1] += code[i : i + 2]
            i += 2
        else:
            parts[-1] += c
            i += 1
    return parts


def _bracket(inner: str, toks: list[_Tok], section: _Section, runs: list[str]) -> None:
    if not inner:
        raise _Unsupported
    first = inner[0]
    if first in "<>=":
        if section.condition is not None:
            raise _Unsupported
        for op in (">=", "<=", "<>", ">", "<", "="):
            if inner.startswith(op):
                try:
                    section.condition = (op, float(inner[len(op) :].strip()))
                except ValueError:
                    raise _Unsupported from None
                return
        raise _Unsupported
    if first == "$":
        symbol = inner[1:].split("-", 1)[0]
        if symbol:
            _push_literal(toks, symbol)
        return
    lower = inner.lower()
    if lower[0] in "hms" and set(lower) == {lower[0]}:
        section.elapsed = section.elapsed or lower[0]
        runs.append(lower[0])
        toks.append(("elapsed", lower[0]))
        return
    if lower in _COLORS:
        return
    if lower.startswith("color"):
        try:
            if 1 <= int(lower[5:].strip()) <= 56:
                return
        except ValueError:
            pass
    raise _Unsupported


def _parse_section(text: str) -> _Section:
    section = _Section(kind="number")
    toks: list[_Tok] = []
    runs: list[str] = []
    has_date = has_general = False
    general_at = 0
    i = 0
    while i < len(text):
        c = text[i]
        lower = c.lower()
        if c == "[":
            end = text.find("]", i)
            if end < 0:
                raise _Unsupported
            _bracket(text[i + 1 : end], toks, section, runs)
            has_date = has_date or bool(section.elapsed)
            i = end + 1
        elif c == '"':
            end = text.find('"', i + 1)
            if end < 0:
                raise _Unsupported
            if end > i + 1:
                _push_literal(toks, text[i + 1 : end])
            i = end + 1
        elif c == "\\":
            _push_literal(toks, text[i + 1])
            i += 2
        elif c == "_":
            _push_literal(toks, " ")
            i += 2
        elif c == "*":
            i += 2  # fill to the column width: there is none here
        elif c in "0#?":
            toks.append(("digit", c))
            i += 1
        elif c == ".":
            toks.append(("decimal",))
            i += 1
        elif c == ",":
            toks.append(("comma",))
            i += 1
        elif c == "%":
            toks.append(("percent",))
            i += 1
        elif c == "@":
            toks.append(("at",))
            i += 1
        elif lower == "e" and text[i + 1 : i + 2] in ("+", "-"):
            toks.append(("exp", text[i + 1] == "+"))
            i += 2
        elif lower in "ydhsm":
            start = i
            while i < len(text) and text[i].lower() == lower:
                i += 1
            has_date = True
            runs.append(lower)
            toks.append(("run", lower, i - start))
        elif lower == "g":
            if text[i : i + 7].lower() != "general":
                raise _Unsupported
            has_general = True
            general_at = len(_decoration(toks))
            i += 7
        elif lower == "a":
            for word, full in (("am/pm", True), ("a/p", False)):
                if text[i : i + len(word)].lower() == word:
                    toks.append(("ampm", full))
                    runs.append("a")
                    has_date = True
                    i += len(word)
                    break
            else:
                raise _Unsupported
        elif c == "/":
            toks.append(("slash",))
            i += 1
        elif c in _PLAIN_LITERALS:
            _push_literal(toks, c)
            i += 1
        else:
            # Unquoted digits (a fraction's fixed denominator) and any other
            # character are outside the grammar
            raise _Unsupported

    if has_general:
        if any(t[0] not in ("literal", "run", "ampm") for t in toks):
            raise _Unsupported
        decoration = _decoration(toks)
        section.kind = "general"
        section.prefix, section.suffix = (
            decoration[:general_at],
            decoration[general_at:],
        )
    elif has_date:
        if any(t[0] in ("at", "exp") for t in toks):
            raise _Unsupported
        section.kind = "date"
        section.toks = toks
        _date_parts(section, runs)
    elif any(t[0] == "at" for t in toks):
        if any(t[0] in ("digit", "decimal", "exp", "slash") for t in toks):
            raise _Unsupported
        section.kind = "text"
        section.toks = [
            ("literal", "%")
            if t[0] == "percent"
            else ("literal", ",")
            if t[0] == "comma"
            else t
            for t in toks
        ]
    elif not toks:
        section.kind = "empty"
    else:
        _resolve_number(section, toks)
    return section


def _decoration(toks: list[_Tok]) -> str:
    return "".join(t[1] for t in toks if t[0] == "literal")


def _date_parts(section: _Section, runs: list[str]) -> None:
    for i, run in enumerate(runs):
        if run in "yd":
            section.date = True
        elif run in "ha":
            section.time = True
        elif run == "s":
            section.time = section.seconds = True
        elif run == "m":
            if (i and runs[i - 1] == "h") or runs[i + 1 : i + 2] == ["s"]:
                section.time = True
            else:
                section.date = True
    if section.elapsed or not (section.date or section.time):
        section.seconds = not section.elapsed or "s" in runs
        section.date = False
        section.time = True
    if section.elapsed:
        units = [run for run in runs if run in _UNIT_SECONDS]
        section.smallest = max(
            units, key=lambda unit: -_UNIT_SECONDS[unit], default=section.elapsed
        )
    section.spelled = _is_spelled(section)


def _month_or_minute(toks: list[_Tok], index: int) -> str:
    """ "m" (month) or "n" (minute) for the m run at toks[index]."""
    runs = [(i, t[1]) for i, t in enumerate(toks) if t[0] in ("run", "elapsed")]
    k = next(k for k, (at, _) in enumerate(runs) if at == index)
    minute = (k > 0 and runs[k - 1][1] == "h") or (
        k + 1 < len(runs) and runs[k + 1][1] == "s"
    )
    return "n" if minute else "m"


def _is_spelled(section: _Section) -> bool:
    """A date format that reads the same in every locale's order: a four
    digit year, and an English month or weekday name or a CJK month label."""
    toks = section.toks
    if section.elapsed or any(
        t[0] not in ("literal", "slash", "comma", "ampm", "run") for t in toks
    ):
        return False
    year4 = named = labeled = False
    for i, tok in enumerate(toks):
        if tok[0] != "run":
            continue
        letter, length = tok[1], tok[2]
        if letter == "y":
            year4 = year4 or length >= 3
        elif letter == "d":
            named = named or length >= 3
        elif letter == "m" and _month_or_minute(toks, i) == "m":
            if length >= 5:
                return False
            named = named or length >= 3
            after = toks[i + 1] if i + 1 < len(toks) else None
            labeled = labeled or (
                after is not None
                and after[0] == "literal"
                and after[1].startswith(("月", "월"))
            )
    return year4 and (named or labeled)


def _resolve_number(section: _Section, raw: list[_Tok]) -> None:
    region = "int"
    toks: list[_Tok] = []
    for tok in raw:
        if tok[0] == "digit":
            toks.append(("digit", tok[1], region))
        elif tok[0] == "decimal":
            if region != "int":
                raise _Unsupported
            region = "frac"
            toks.append(tok)
        elif tok[0] == "exp":
            if section.exp:
                raise _Unsupported
            section.exp = True
            region = "exp"
            toks.append(tok)
        elif tok[0] == "slash":
            if toks and toks[-1][0] == "digit":
                raise _Unsupported  # a fraction: shown as General
            _push_literal(toks, "/")
        elif tok[0] == "percent":
            section.percents += 1
            toks.append(tok)
        elif tok[0] in ("run", "ampm"):
            raise _Unsupported
        else:
            toks.append(tok)
    digits = [i for i, t in enumerate(toks) if t[0] == "digit"]
    out: list[_Tok] = []
    for i, tok in enumerate(toks):
        if tok[0] != "comma":
            out.append(tok)
        elif digits and i > digits[-1]:
            section.scale += 1
        elif digits and digits[0] < i < digits[-1]:
            section.grouping = True
        else:
            _push_literal(out, ",")
    if section.exp and not any(t[0] == "digit" and t[2] == "exp" for t in out):
        raise _Unsupported
    section.toks = out


@lru_cache(maxsize=256)
def _parse(code: str) -> list[_Section] | None:
    if not code or code.lower() == "general":
        return None
    try:
        sections = [_parse_section(part) for part in _split_sections(code)]
    except (_Unsupported, IndexError):
        return None
    if len(sections) > 4 or sum(s.condition is not None for s in sections) > 2:
        return None
    for i, section in enumerate(sections):
        if section.kind == "text" and (
            i != len(sections) - 1 or section.condition is not None
        ):
            return None
    return sections


def _split_digits(value: float, places: int) -> tuple[str, str]:
    """The integer digits (empty for zero) and *places* fractional digits of
    *value* rounded half away from zero at its 15 significant digits, the
    way a spreadsheet displays it (binary rounding makes 5.255 5.25)."""
    from decimal import ROUND_HALF_UP, Decimal

    rounded = Decimal(f"{value:.15g}").quantize(
        Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP
    )
    text = format(rounded, "f")
    whole, _, fraction = text.partition(".")
    return whole.lstrip("0"), fraction.ljust(places, "0")[:places]


def _assign(digits: str, places: list[str]) -> list[str]:
    """Digits over placeholders: the first takes any excess digits, and the
    ones past the digits pad (``0`` a zero, ``?`` a space, ``#`` nothing)."""
    k = len(places)
    if not k:
        return []
    n = len(digits)
    if n > k:
        return [digits[: n - k + 1], *digits[n - k + 1 :]]
    pads = ["0" if p == "0" else " " if p == "?" else "" for p in places[: k - n]]
    return pads + list(digits)


def _emit(
    section: _Section,
    int_digits: str,
    frac_digits: str,
    exp_digits: str = "",
    exp_negative: bool = False,
) -> str:
    def places(region: str) -> list[str]:
        return [t[1] for t in section.toks if t[0] == "digit" and t[2] == region]

    int_assigned = _assign(int_digits, places("int"))
    exp_assigned = _assign(exp_digits, places("exp"))
    remaining = sum(ch.isdigit() for part in int_assigned for ch in part)
    frac_places = places("frac")
    min_frac = max(
        (i + 1 for i, place in enumerate(frac_places) if place == "0"), default=0
    )
    keep_frac = max(len(frac_digits.rstrip("0")), min_frac)
    show_decimal = keep_frac > 0 or "?" in frac_places or not frac_places
    out: list[str] = []
    int_i = frac_i = exp_i = 0
    for tok in section.toks:
        kind = tok[0]
        if kind == "digit" and tok[2] == "int":
            for ch in int_assigned[int_i]:
                out.append(ch)
                if ch.isdigit():
                    remaining -= 1
                    if section.grouping and remaining > 0 and remaining % 3 == 0:
                        out.append(",")
            int_i += 1
        elif kind == "digit" and tok[2] == "frac":
            if frac_i < keep_frac:
                out.append(frac_digits[frac_i])
            elif tok[1] == "?":
                out.append(" ")
            frac_i += 1
        elif kind == "digit":
            out.append(exp_assigned[exp_i])
            exp_i += 1
        elif kind == "decimal":
            if show_decimal:
                out.append(".")
        elif kind == "percent":
            out.append("%")
        elif kind == "literal":
            out.append(tok[1])
        elif kind == "exp":
            out.append("E" + ("-" if exp_negative else "+" if tok[1] else ""))
    return "".join(out)


def _render_number(section: _Section, value: float, minus: bool) -> str:
    value = value * 100**section.percents / 1000**section.scale
    frac_places = sum(1 for t in section.toks if t[0] == "digit" and t[2] == "frac")
    if section.exp:
        int_places = max(
            1, sum(1 for t in section.toks if t[0] == "digit" and t[2] == "int")
        )
        exponent = 0
        if value:
            # The exponent stays a multiple of the integer places
            # (engineering notation for ##0.0E+0); rounding 9.99 to 10.0
            # moves it on
            exponent = int(f"{value:e}".split("e")[1]) // int_places * int_places
            if len(_split_digits(value / 10.0**exponent, frac_places)[0]) > int_places:
                exponent += int_places
        whole, fraction = _split_digits(value / 10.0**exponent, frac_places)
        exp_places = sum(1 for t in section.toks if t[0] == "digit" and t[2] == "exp")
        body = _emit(
            section,
            whole,
            fraction,
            str(abs(exponent)).rjust(exp_places, "0"),
            exponent < 0,
        )
    else:
        whole, fraction = _split_digits(value, frac_places)
        body = _emit(section, whole, fraction)
    return f"-{body}" if minus else body


def _select(sections: list[_Section], value: float) -> tuple[_Section, float, bool]:
    """The section for *value*, the value it renders, and whether a minus
    sign goes in front (not for the positional negative section, whose code
    writes its own sign)."""
    numeric = [s for s in sections if s.kind != "text"][:3]
    if any(s.condition is not None for s in numeric):
        for section in numeric:
            condition = section.condition
            if condition is None or _matches(condition, value):
                return section, value, True
        raise _Unsupported
    if len(numeric) == 1 or (len(numeric) == 2 and value >= 0):
        index = 0
    elif len(numeric) == 2 or value < 0:
        index = 1
    else:
        index = 0 if value > 0 else 2
    if index == 1:
        return numeric[1], abs(value), False
    return numeric[index], value, True


def _matches(condition: tuple[str, float], value: float) -> bool:
    op, operand = condition
    return {
        ">=": value >= operand,
        "<=": value <= operand,
        "<>": value != operand,
        ">": value > operand,
        "<": value < operand,
        "=": value == operand,
    }[op]


def _general(value: float) -> str:
    from markitai.converter.xlsx import cell_text

    return cell_text(value)


def _clock(seconds: int, show_seconds: bool) -> str:
    seconds %= 86_400
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return (
        f"{hours:02}:{minutes:02}:{secs:02}"
        if show_seconds
        else f"{hours:02}:{minutes:02}"
    )


def _duration(section: _Section, total: float) -> str:
    sign = "-" if total < 0 else ""
    seconds = round(abs(total))
    largest = _UNIT_SECONDS[section.elapsed]
    out = f"{sign}{seconds // largest}"
    rest = seconds % largest
    for unit in ("m", "s"):
        size = _UNIT_SECONDS[unit]
        if size < largest and size >= _UNIT_SECONDS[section.smallest]:
            out += f":{rest // size:02}"
            rest %= size
    return out


def _spelled(section: _Section, moment: datetime.datetime) -> str:
    clock12 = any(t[0] == "ampm" for t in section.toks)
    out: list[str] = []
    for i, tok in enumerate(section.toks):
        kind = tok[0]
        if kind == "literal":
            out.append(tok[1])
        elif kind == "slash":
            out.append("/")
        elif kind == "comma":
            out.append(",")
        elif kind == "ampm":
            text = "AM" if moment.hour < 12 else "PM"
            out.append(text if tok[1] else text[0])
        elif kind == "run":
            letter, length = tok[1], tok[2]
            if letter == "m" and _month_or_minute(section.toks, i) == "n":
                letter = "n"
            value = {
                "y": moment.year,
                "m": moment.month,
                "d": moment.day,
                "h": (moment.hour % 12 or 12) if clock12 else moment.hour,
                "n": moment.minute,
                "s": moment.second,
            }[letter]
            if letter == "y":
                out.append(f"{value % 100:02}" if length <= 2 else f"{value:04}")
            elif letter == "m" and length >= 3:
                name = _MONTHS[value - 1]
                out.append(name[:3] if length == 3 else name)
            elif letter == "d" and length >= 3:
                name = _DAYS[moment.weekday()]
                out.append(name[:3] if length == 3 else name)
            else:
                out.append(f"{value:02}" if length >= 2 else str(value))
        else:
            raise _Unsupported
    return "".join(out).strip()


def _render_moment(section: _Section, value: Any) -> str:
    if isinstance(value, datetime.timedelta):
        if not section.elapsed:
            raise _Unsupported
        return _duration(section, value.total_seconds())
    if isinstance(value, datetime.time):
        if section.date and not section.time:
            raise _Unsupported
        seconds = value.hour * 3600 + value.minute * 60 + value.second
        return _clock(seconds + (value.microsecond >= 500_000), section.seconds)
    if not isinstance(value, datetime.datetime):
        value = datetime.datetime.combine(value, datetime.time())
    if value.microsecond >= 500_000:
        value += datetime.timedelta(microseconds=1_000_000 - value.microsecond)
    value = value.replace(microsecond=0)
    if section.elapsed:
        raise _Unsupported
    if section.spelled:
        return _spelled(section, value)
    seconds = value.hour * 3600 + value.minute * 60 + value.second
    if not section.date:
        return _clock(seconds, section.seconds)
    text = value.date().isoformat()
    if section.time and seconds:
        text += " " + _clock(seconds, section.seconds)
    return text


def format_cell(code: str | None, value: Any) -> str | None:
    """The text a cell holding *value* shows under number format *code*.

    *value* is a number, a ``datetime``/``date``/``time``/``timedelta`` (a
    date-formatted cell as openpyxl and xlrd return it) or a string. None
    when *code* is General or outside the implemented grammar: the caller
    shows the value as a General cell.
    """
    if code is None or isinstance(value, bool) or value is None:
        return None
    sections = _parse(code)
    if sections is None:
        return None
    try:
        if isinstance(value, str):
            last = sections[-1]
            if last.kind == "text":
                return "".join(
                    value if t[0] == "at" else t[1] if t[0] == "literal" else ""
                    for t in last.toks
                )
            return "" if len(sections) == 4 and last.kind == "empty" else None
        if isinstance(value, (datetime.date, datetime.time, datetime.timedelta)):
            section = next(s for s in sections if s.kind == "date")
            return _render_moment(section, value)
        if not isinstance(value, (int, float)):
            return None
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        section, number, minus_allowed = _select(sections, number)
        if section.kind == "general":
            return f"{section.prefix}{_general(number)}{section.suffix}".strip()
        if section.kind == "empty":
            return ""
        if section.kind != "number":
            return None
        return _render_number(
            section, abs(number), minus_allowed and number < 0
        ).strip()
    except (_Unsupported, StopIteration, ValueError, OverflowError):
        return None
