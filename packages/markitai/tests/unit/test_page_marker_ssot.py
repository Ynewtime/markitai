"""Every page boundary is written and read in exactly one spelling.

The marker is a contract between halves of the pipeline that never see each
other: a converter splits a document into pages and writes the marker; LLM
enhancement protects it from being rewritten, page/image alignment counts
it, and output profiles rewrite it for publication.

Nothing enforced the spelling, so it drifted. Screenshot-only mode emitted
``<!-- Page 3 -->``; every reader matched ``<!-- Page number: 3 -->`` and so
matched nothing, and the pages were invisible to all of them — silently,
because a document with no markers at all is a legitimate case. Meanwhile
four near-identical regexes had accumulated across three modules, two of
them byte-for-byte identical inside the same file.

The guards, matching the two ways it broke:

* **One writer.** The marker text is built by ``constants.page_marker``.
* **One reader.** The marker is matched by ``constants.PAGE_MARKER_RE``.

What this cannot catch: a marker carrying the wrong page number.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from markitai.constants import PAGE_MARKER_RE, page_marker

_SRC = Path(__file__).resolve().parents[2] / "src" / "markitai"

# A literal mentioning a page comment that is not a second implementation.
_NOT_A_MARKER_COPY = frozenset(
    {
        # llm/content.py: the model sometimes invents markers of its own, in
        # looser spacing than any converter writes; this strips them.
        r"<!--\s*Page\s+number:\s*\d+\s*-->\s*\n?",
        # llm/content.py: page *image* references, a different comment.
        r"<!--\s*!\[Page\s+\d+\]\([^)]*\)\s*-->",
        r"!\[Page\s+(\d+)\]",
        # llm/content.py: slides, the PPTX counterpart.
        r"<!--\s*Slide\s+(?:number:\s*)?\d+\s*-->",
        r"<!--\s*Slide number:\s*\d+\s*-->",
        r"<!--\s*Slide\s+number:\s*\d+\s*-->\s*\n?",
    }
)

_MENTIONS_PAGE_MARKER = re.compile(r"Page\s*(?:\\s\*)?\s*number", re.IGNORECASE)

# A page *number* marker being built: the literal number, an f-string hole, or
# the "number:" label. Deliberately not "<!-- Page images for reference -->",
# which is a different comment that happens to start the same way.
_BUILDS_PAGE_MARKER = re.compile(r"<!--\s*Page\s+(?:number:|\d|\{)")


def _joined(node: ast.JoinedStr) -> str:
    """An f-string with its holes as ``{}``.

    An f-string reaches the AST already split, so ``f"<!-- Page {i} -->"``
    survives only as the constant ``"<!-- Page "`` — which matches no guard
    below. That is the exact shape of the bug this module exists to catch,
    so the pieces are put back together before matching.
    """
    return "".join(
        part.value
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
        else "{}"
        for part in node.values
    )


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Executable string literals only — docstrings are prose."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        doc
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        )
        for doc in [ast.get_docstring(node, clean=False)]
        if doc
    }
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            found.append((node.lineno, _joined(node)))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value not in docstrings
        ):
            found.append((node.lineno, node.value))
    return found


def test_the_marker_is_written_in_one_place() -> None:
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "constants.py":
            continue
        for lineno, value in _string_literals(path):
            if value in _NOT_A_MARKER_COPY:
                continue
            if _BUILDS_PAGE_MARKER.search(value):
                offenders.append(f"{path.relative_to(_SRC)}:{lineno}: {value!r}")

    assert not offenders, (
        "page markers built by hand instead of constants.page_marker() — this "
        "is how screenshot-only came to emit a spelling no reader matched:\n"
        + "\n".join(offenders)
    )


def test_the_marker_is_matched_in_one_place() -> None:
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "constants.py":
            continue
        for lineno, value in _string_literals(path):
            if value in _NOT_A_MARKER_COPY or "<!--" not in value:
                continue
            if _MENTIONS_PAGE_MARKER.search(value):
                offenders.append(f"{path.relative_to(_SRC)}:{lineno}: {value!r}")

    assert not offenders, (
        "page-marker patterns outside constants.PAGE_MARKER_RE (import it, or "
        "add the literal to _NOT_A_MARKER_COPY with the reason it differs):\n"
        + "\n".join(offenders)
    )


def test_the_writer_and_the_reader_agree() -> None:
    """The pair has to round-trip, or every guard above is decorative."""
    for number in (1, 7, 128):
        match = PAGE_MARKER_RE.fullmatch(page_marker(number))
        assert match is not None, f"PAGE_MARKER_RE does not match page_marker({number})"
        assert int(match.group(1)) == number
