"""Known CSV/notebook inputs need parsers, not binary format detection.

Rendering follows MarkItDown's CSV/notebook rules, including ragged-row
handling (0.1.9: the table is as wide as its longest row, so a row longer
than the header keeps its cells, as markitai-rust and the TSV reader do) and
its final whitespace normalization, except that a UTF-8 BOM is not part of
the first header. See NOTICE (Microsoft, MIT).
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from markitai.converter.base import ConvertResult, conversion_failed
from markitai.webextract.table_cells import text_cell


def _normalize(markdown: str) -> str:
    markdown = "\n".join(line.rstrip() for line in re.split(r"\r?\n", markdown))
    return re.sub(r"\n{3,}", "\n\n", markdown)


def convert_csv(path: Path) -> ConvertResult:
    data = path.read_bytes()
    # UTF-8 is common and unambiguous; other encodings use the same detector
    # as the reference CSV reader, without initializing Magika/ONNX.
    try:
        # -sig: the BOM Excel's "CSV UTF-8" writes stuck to the first header
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        from markitai.utils.charset import decode_body

        # str() of the detector's answer was "None" when it found none
        text = decode_body(data, html=False)[0]
    reader = csv.reader(io.StringIO(text))
    try:
        rows = list(reader)
        lines = []
        if rows:
            # A row longer than the header widened the table instead of
            # losing its last cells
            width = max(len(row) for row in rows)
            header, *body = (row + [""] * (width - len(row)) for row in rows)
            lines = [
                "| " + " | ".join(text_cell(cell) for cell in header) + " |",
                "| " + " | ".join(["---"] * width) + " |",
            ]
            for cells in body:
                lines.append("| " + " | ".join(text_cell(c) for c in cells) + " |")
    except csv.Error as exc:
        conversion_failed(f"Malformed CSV: {exc}")
    return ConvertResult(
        markdown=_normalize("\n".join(lines)),
        metadata={"source": str(path), "format": "CSV", "converter": "delimited"},
    )


def convert_notebook(path: Path) -> ConvertResult:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        output = []
        title = None
        for cell in notebook.get("cells", []):
            kind = cell.get("cell_type", "")
            source = cell.get("source", [])
            text = "".join(source)
            if kind == "markdown":
                output.append(text)
                if title is None:
                    for line in source:
                        if line.startswith("# "):
                            title = line.lstrip("# ").strip()
                            break
            elif kind == "code":
                output.append(f"```python\n{text}\n```")
            elif kind == "raw":
                output.append(f"```\n{text}\n```")
        title = notebook.get("metadata", {}).get("title", title)
    except (ValueError, TypeError, AttributeError) as exc:
        conversion_failed(f"Malformed notebook: {exc}")
    metadata = {"source": str(path), "format": "IPYNB", "converter": "notebook"}
    if title:
        metadata["title"] = title
    return ConvertResult(markdown=_normalize("\n\n".join(output)), metadata=metadata)
