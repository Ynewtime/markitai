"""Native reStructuredText and Org-mode converters (stdlib only).

Both formats are line-oriented and already close to Markdown: headings,
inline code and code blocks are the only constructs that must be rewritten,
and everything else — bullet lists, field lists, emphasis — is left exactly
as written rather than guessed at.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)

_MAX_HEADING_LEVEL = 6

# --- reStructuredText -------------------------------------------------------

# Section adornments are punctuation runs; require three characters so that
# "--" or "**" in running text is never mistaken for one.
_ADORNMENT_CHARS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_LITERAL_RE = re.compile(r"``(.+?)``")
_CODE_DIRECTIVE_RE = re.compile(
    r"^(?P<indent>\s*)\.\.\s+(?:code|code-block|sourcecode)::\s*(?P<lang>\S*)\s*$"
)
_DIRECTIVE_OPTION_RE = re.compile(r"^\s*:[\w-]+:.*$")


def _rst_inline(text: str) -> str:
    """``literal`` is the one inline construct Markdown spells differently."""
    return _LITERAL_RE.sub(r"`\1`", text)


def _is_adornment(line: str) -> bool:
    stripped = line.strip()
    return (
        len(stripped) >= 3
        and len(set(stripped)) == 1
        and stripped[0] in _ADORNMENT_CHARS
        and line.rstrip() == stripped  # adornments start at column 0
    )


def _dedent(lines: list[str]) -> list[str]:
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    common = min(indents) if indents else 0
    return [line[common:] if line.strip() else "" for line in lines]


def _rst_code_block(
    lines: list[str], start: int, lang: str, indent: int
) -> tuple[list[str], int]:
    """Collect the indented body of a code directive starting after ``start``."""
    index = start
    body: list[str] = []
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            body.append("")
            index += 1
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        if not body and _DIRECTIVE_OPTION_RE.match(line):
            index += 1
            continue
        body.append(line)
        index += 1

    while body and not body[-1].strip():
        body.pop()
    while body and not body[0].strip():
        body.pop(0)

    fence = [f"```{lang}".rstrip(), *_dedent(body), "```"]
    return fence, index


def _rst_to_markdown(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    levels: dict[str, int] = {}
    index = 0

    def heading(title: str, char: str) -> str:
        """Adornment characters map to levels by order of first appearance."""
        level = min(levels.setdefault(char, len(levels) + 1), _MAX_HEADING_LEVEL)
        text = _rst_inline(title.strip())
        return f"{'#' * level} {text}"

    while index < len(lines):
        line = lines[index]

        directive = _CODE_DIRECTIVE_RE.match(line)
        if directive:
            fence, index = _rst_code_block(
                lines,
                index + 1,
                directive.group("lang"),
                len(directive.group("indent")),
            )
            out.extend(fence)
            continue

        # Overline + title + underline.
        if (
            _is_adornment(line)
            and index + 2 < len(lines)
            and lines[index + 1].strip()
            and lines[index + 2].strip() == line.strip()
        ):
            out.append(heading(lines[index + 1], line.strip()[0]))
            index += 3
            continue

        # Title + underline.
        if (
            line.strip()
            and index + 1 < len(lines)
            and _is_adornment(lines[index + 1])
            and len(lines[index + 1].strip()) >= len(line.strip())
        ):
            out.append(heading(line, lines[index + 1].strip()[0]))
            index += 2
            continue

        out.append(_rst_inline(line))
        index += 1

    return "\n".join(out).strip() + "\n"


# --- Org mode ---------------------------------------------------------------

_ORG_HEADING_RE = re.compile(r"^(\*+)\s+(.*)$")
_ORG_TITLE_RE = re.compile(r"^#\+TITLE:\s*(.+)$", re.IGNORECASE)
_ORG_SRC_BEGIN_RE = re.compile(r"^\s*#\+BEGIN_SRC\s*(\S*).*$", re.IGNORECASE)
_ORG_SRC_END_RE = re.compile(r"^\s*#\+END_SRC\s*$", re.IGNORECASE)
_ORG_LINK_DESC_RE = re.compile(r"\[\[([^\]]+)\]\[([^\]]+)\]\]")
_ORG_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
# Org verbatim/code markers: no whitespace just inside the delimiters, which
# is what keeps "a = b + c = d" from being read as one code span.
_ORG_TILDE_RE = re.compile(r"(?<![\w~])~(\S(?:[^~\n]*\S)?)~(?![\w~])")
_ORG_EQUALS_RE = re.compile(r"(?<![\w=])=(\S(?:[^=\n]*\S)?)=(?![\w=])")


def _org_inline(text: str) -> str:
    text = _ORG_LINK_DESC_RE.sub(r"[\2](\1)", text)
    text = _ORG_LINK_RE.sub(r"<\1>", text)
    text = _ORG_TILDE_RE.sub(r"`\1`", text)
    return _ORG_EQUALS_RE.sub(r"`\1`", text)


def _org_to_markdown(text: str) -> tuple[str, str | None]:
    """Return the Markdown body and the ``#+TITLE:`` value, if any."""
    out: list[str] = []
    title: str | None = None
    in_src = False

    for line in text.splitlines():
        if in_src:
            if _ORG_SRC_END_RE.match(line):
                out.append("```")
                in_src = False
            else:
                out.append(line)
            continue

        src = _ORG_SRC_BEGIN_RE.match(line)
        if src:
            out.append(f"```{src.group(1)}".rstrip())
            in_src = True
            continue

        title_match = _ORG_TITLE_RE.match(line)
        if title_match:
            title = title_match.group(1).strip()
            continue

        heading = _ORG_HEADING_RE.match(line)
        if heading:
            level = min(len(heading.group(1)), _MAX_HEADING_LEVEL)
            out.append(f"{'#' * level} {_org_inline(heading.group(2)).strip()}")
            continue

        out.append(_org_inline(line))

    if in_src:
        out.append("```")

    return "\n".join(out).strip() + "\n", title


@register_converter(FileFormat.RST)
class RstConverter(BaseConverter):
    """Converter for reStructuredText documents."""

    supported_formats = [FileFormat.RST]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        logger.debug("[RstConverter] Converting: {}", input_path.name)

        text = input_path.read_text(encoding="utf-8", errors="replace")
        metadata: dict = {
            "source": str(input_path),
            "format": "RST",
            "converter": "markup",
        }
        return ConvertResult(
            markdown=_rst_to_markdown(text), images=[], metadata=metadata
        )


@register_converter(FileFormat.ORG)
class OrgConverter(BaseConverter):
    """Converter for Org-mode documents."""

    supported_formats = [FileFormat.ORG]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        logger.debug("[OrgConverter] Converting: {}", input_path.name)

        text = input_path.read_text(encoding="utf-8", errors="replace")
        markdown, title = _org_to_markdown(text)
        metadata: dict = {
            "source": str(input_path),
            "format": "ORG",
            "converter": "markup",
        }
        if title:
            metadata["title"] = title
        return ConvertResult(markdown=markdown, images=[], metadata=metadata)
