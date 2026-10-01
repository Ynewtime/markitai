"""Result records and the type names built on `typing`, loaded on first use.

`dataclasses` pulls in `inspect` (and through it `ast`, `dis`, `tokenize`), so
`import markitai` leaves these definitions alone: `markitai.api` imports this
module when a conversion produces a result or a caller names one of its types.
The classes keep `markitai.api` as their `__module__`, the home they had before
they moved here, so repr, pickles and `typing.get_type_hints` are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

__all__ = ["ConfigModel", "ConversionOutput", "ConversionUsage", "OutputProfileName"]

OutputProfileName = Literal["rag", "obsidian", "okf"]


class ConfigModel(Protocol):
    def model_dump(self, *, mode: str) -> dict[str, Any]: ...


# Assigned after the class: a protocol body may only declare members.
ConfigModel.__module__ = "markitai.api"


@dataclass
class ConversionUsage:
    __module__ = "markitai.api"

    cost_usd: float = 0.0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_usage_dict(
        cls, cost_usd: float, by_model: dict[str, dict[str, Any]]
    ) -> ConversionUsage:
        return cls(
            cost_usd=cost_usd,
            requests=sum(int(row.get("requests", 0)) for row in by_model.values()),
            input_tokens=sum(int(row.get("input_tokens", 0)) for row in by_model.values()),
            output_tokens=sum(int(row.get("output_tokens", 0)) for row in by_model.values()),
            by_model=by_model,
        )


@dataclass
class ConversionOutput:
    __module__ = "markitai.api"

    source: str
    markdown: str
    llm_markdown: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None
    llm_output_path: Path | None = None
    assets: list[Path] = field(default_factory=list)
    screenshots: list[Path] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    usage: ConversionUsage = field(default_factory=ConversionUsage)
    skip_reason: str | None = None
    duration: float = 0.0
    warnings: list[str] = field(default_factory=list)
