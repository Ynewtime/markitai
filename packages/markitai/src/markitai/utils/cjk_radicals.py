"""CJK radical forms in extracted text, read as the ideographs they stand for.

A PDF's ToUnicode map sometimes names a radical code point instead of the
ideograph the glyph draws: Chrome's "Save as PDF" writes ``使⽤`` (U+2F64
KANGXI RADICAL USE) and ``右下⻆`` (U+2EC6, a CJK Radicals Supplement form)
for 使用 and 右下角. The mapping is the shared workbench's (markitai-rust
``web/src/lib/compare.ts``, ``norm``): NFKC for Kangxi Radicals
(U+2F00–U+2FDF, all of which it maps) and for the two supplement forms it
maps (U+2E9F, U+2EF3), then ``RADICALS`` for the supplement forms NFKC
leaves alone. Other supplement forms (component shapes such as ⺁) stay.
"""

from __future__ import annotations

import re
import unicodedata

#: compare.ts ``RADICALS``, entry for entry.
RADICALS = {
    "⺠": "民",
    "⺰": "纟",
    "⻄": "西",
    "⻅": "见",
    "⻆": "角",
    "⻈": "讠",
    "⻉": "贝",
    "⻋": "车",
    "⻐": "钅",
    "⻓": "长",
    "⻔": "门",
    "⻘": "青",
    "⻙": "韦",
    "⻚": "页",
    "⻛": "风",
    "⻜": "飞",
    "⻝": "食",
    "⻠": "饣",
    "⻢": "马",
    "⻥": "鱼",
    "⻦": "鸟",
    "⻧": "卤",
    "⻨": "麦",
    "⻩": "黄",
    "⻪": "黾",
    "⻬": "齐",
    "⻮": "齿",
    "⻰": "龙",
}

_RADICAL = re.compile("[⺀-⻿⼀-⿟]")


def _ideograph(match: re.Match[str]) -> str:
    radical = match.group()
    folded = unicodedata.normalize("NFKC", radical)
    if folded != radical:
        return folded
    return RADICALS.get(radical, radical)


def fold_cjk_radicals(text: str) -> str:
    """*text* with each mappable radical form replaced by its ideograph."""
    if not text or not _RADICAL.search(text):
        return text
    return _RADICAL.sub(_ideograph, text)
