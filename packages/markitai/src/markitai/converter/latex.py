"""Native LaTeX converter (stdlib only).

A .tex file is a program, and rendering it faithfully means running TeX. What
a Markdown reader wants instead is the prose: the preamble is dropped, the
handful of commands that carry document structure (sections, emphasis, lists,
verbatim) are rewritten, and every other command is replaced by the text of
its argument rather than deleted with it.
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

_COMMAND_NAME_RE = re.compile(r"[A-Za-z]+")
_COMMENT_RE = re.compile(r"(?<!\\)%.*$")
_TITLE_RE = re.compile(r"\\title\s*\{")
_VERBATIM_RE = re.compile(
    r"\\begin\{(verbatim|lstlisting)\}(.*?)\\end\{\1\}", re.DOTALL
)
_BEGIN_RE = re.compile(r"^\s*\\begin\{(\w+\*?)\}")
_END_RE = re.compile(r"^\s*\\end\{(\w+\*?)\}")
_ITEM_RE = re.compile(r"^\s*\\item\b\s*(?:\[[^\]]*\])?\s*")
_BLANK_RUN_RE = re.compile(r"\n{3,}")

_HEADINGS = {
    "chapter": 1,
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
}
_INLINE = {
    "textbf": "**{}**",
    "emph": "*{}*",
    "textit": "*{}*",
    "texttt": "`{}`",
}
# Commands whose argument is markup metadata, not prose.
_DROPPED = {
    "maketitle",
    "label",
    "ref",
    "cite",
    "index",
    "documentclass",
    "usepackage",
    "title",
    "author",
    "date",
    "centering",
    "hline",
    "newpage",
    "tableofcontents",
    "bibliographystyle",
    "bibliography",
}
_ESCAPES = set("%&_$#{}")

_LIST_MARKERS = {"itemize": "-", "enumerate": "1.", "description": "-"}


def _braced(text: str, start: int) -> tuple[str, int]:
    """Read the balanced ``{...}`` group at ``start``; return it and the end."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return text[start + 1 :], len(text)


def _render_command(name: str, arg: str | None) -> str:
    body = _convert_commands(arg) if arg else ""
    if name in _HEADINGS:
        return f"\n\n{'#' * _HEADINGS[name]} {body.strip()}\n\n"
    if name in _INLINE:
        return _INLINE[name].format(body.strip()) if body.strip() else ""
    if name in _DROPPED:
        return ""
    return body


def _convert_commands(text: str) -> str:
    """Rewrite commands and escapes; unknown commands keep their argument."""
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue

        index += 1
        if index >= length:
            break
        following = text[index]
        if following == "\\":
            out.append("\n")
            index += 1
            continue
        if following in _ESCAPES:
            out.append(following)
            index += 1
            continue

        match = _COMMAND_NAME_RE.match(text, index)
        if match is None:
            index += 1  # control symbol with no name (\, \/ ...): drop it
            continue

        name = match.group(0)
        index = match.end()
        if index < length and text[index] == "*":
            index += 1
        while index < length and text[index] == "[":
            close = text.find("]", index)
            if close == -1:
                break
            index = close + 1

        arg: str | None = None
        if index < length and text[index] == "{":
            arg, index = _braced(text, index)
        else:
            while index < length and text[index] == " ":
                index += 1

        out.append(_render_command(name, arg))

    return "".join(out)


def _document_body(text: str) -> str:
    """Everything between ``\\begin{document}`` and ``\\end{document}``."""
    start = text.find(r"\begin{document}")
    if start == -1:
        return text
    body = text[start + len(r"\begin{document}") :]
    end = body.find(r"\end{document}")
    return body if end == -1 else body[:end]


def _extract_title(text: str) -> str | None:
    match = _TITLE_RE.search(text)
    if match is None:
        return None
    raw, _ = _braced(text, match.end() - 1)
    title = _convert_commands(raw).strip()
    return title or None


def _convert_lists(lines: list[str]) -> list[str]:
    """Turn itemize/enumerate environments into Markdown lists."""
    out: list[str] = []
    stack: list[str] = []
    for line in lines:
        begin = _BEGIN_RE.match(line)
        if begin:
            name = begin.group(1)
            if name in _LIST_MARKERS:
                stack.append(_LIST_MARKERS[name])
            continue
        end = _END_RE.match(line)
        if end:
            if end.group(1) in _LIST_MARKERS and stack:
                stack.pop()
            continue
        item = _ITEM_RE.match(line)
        if item and stack:
            indent = "  " * (len(stack) - 1)
            out.append(f"{indent}{stack[-1]} {line[item.end() :].strip()}")
            continue
        out.append(line)
    return out


def _tex_to_markdown(text: str) -> tuple[str, str | None]:
    """Return the Markdown body and the ``\\title{}`` value, if any."""
    verbatim: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        verbatim.append(match.group(2).strip("\n"))
        return f"\x00VERBATIM{len(verbatim) - 1}\x00"

    # Verbatim first: a "%" inside it is content, not a comment.
    text = _VERBATIM_RE.sub(_stash, text)
    text = "\n".join(_COMMENT_RE.sub("", line) for line in text.splitlines())

    title = _extract_title(text)
    body = _document_body(text)
    body = "\n".join(_convert_lists(body.splitlines()))
    body = _convert_commands(body)

    for position, block in enumerate(verbatim):
        body = body.replace(f"\x00VERBATIM{position}\x00", f"```\n{block}\n```", 1)

    lines = [line.rstrip() for line in body.splitlines()]
    return _BLANK_RUN_RE.sub("\n\n", "\n".join(lines)).strip() + "\n", title


@register_converter(FileFormat.TEX)
class TexConverter(BaseConverter):
    """Converter for LaTeX documents."""

    supported_formats = [FileFormat.TEX]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        logger.debug("[TexConverter] Converting: {}", input_path.name)

        text = input_path.read_text(encoding="utf-8", errors="replace")
        markdown, title = _tex_to_markdown(text)
        metadata: dict = {
            "source": str(input_path),
            "format": "TEX",
            "converter": "latex",
        }
        if title:
            metadata["title"] = title
        return ConvertResult(markdown=markdown, images=[], metadata=metadata)
