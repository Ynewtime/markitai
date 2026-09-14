"""Known CSV/notebook inputs need parsers, not binary format detection.

Rendering follows MarkItDown 0.1.7's CSV/notebook rules, including ragged-row
handling and its final whitespace normalization. See NOTICE (Microsoft, MIT).
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from markitai.converter.base import ConvertResult, conversion_failed


def _normalize(markdown: str) -> str:
    markdown = "\n".join(line.rstrip() for line in re.split(r"\r?\n", markdown))
    return re.sub(r"\n{3,}", "\n\n", markdown)


def convert_csv(path: Path) -> ConvertResult:
    data = path.read_bytes()
    # UTF-8 is common and unambiguous; other encodings use the same detector
    # as the reference CSV reader, without initializing Magika/ONNX.
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes

        text = str(from_bytes(data).best())
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader, None)
        lines = []
        if header is not None:
            width = len(header)
            lines = [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * width) + " |",
            ]
            for row in reader:
                cells = (row + [""] * max(0, width - len(row)))[:width]
                lines.append("| " + " | ".join(cells) + " |")
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
