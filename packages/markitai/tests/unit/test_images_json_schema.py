"""Frozen schema lock for images.json.

The images.json schema is documented on the website (Output Profiles page,
en/zh) and consumed by downstream tooling. Any field addition, removal, or
rename must update this test AND both documentation pages in the same
change — that is the point of the lock.
"""

from __future__ import annotations

import json
from pathlib import Path

from markitai.json_order import IMAGE_ENTRY_FIELD_ORDER, IMAGES_FIELD_ORDER
from markitai.workflow.helpers import write_images_json
from markitai.workflow.single import ImageAnalysisResult

# Frozen v1.0 schema — see website/guide/output-profiles.md
FROZEN_TOP_LEVEL_KEYS = {"version", "created", "updated", "images"}
FROZEN_ENTRY_KEYS = {"path", "alt", "desc", "text", "created", "source"}
FROZEN_VERSION = "1.0"


def _write_sample(tmp_path: Path) -> dict:
    assets_dir = tmp_path / ".markitai" / "assets"
    assets_dir.mkdir(parents=True)
    image = assets_dir / "img.png"
    image.write_bytes(b"png")
    result = ImageAnalysisResult(
        source_file="/in/doc.pdf",
        assets=[
            {
                "asset": str(image),
                "alt": "caption",
                "desc": "description",
                "text": "extracted",
                "created": "2026-01-01T00:00:00+00:00",
                "llm_usage": {"model": {"requests": 1}},
            }
        ],
    )
    files = write_images_json(tmp_path, [result])
    assert files == [assets_dir / "images.json"]
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_top_level_key_set_is_frozen(tmp_path: Path) -> None:
    data = _write_sample(tmp_path)
    assert set(data) == FROZEN_TOP_LEVEL_KEYS
    assert data["version"] == FROZEN_VERSION


def test_entry_key_set_is_frozen(tmp_path: Path) -> None:
    data = _write_sample(tmp_path)
    (entry,) = data["images"]
    assert set(entry) == FROZEN_ENTRY_KEYS
    # llm_usage is internal tracking and must never leak into the file
    assert "llm_usage" not in entry


def test_field_order_constants_match_frozen_schema() -> None:
    """The ordering constants are part of the same frozen contract."""
    assert set(IMAGES_FIELD_ORDER) == FROZEN_TOP_LEVEL_KEYS
    assert set(IMAGE_ENTRY_FIELD_ORDER) == FROZEN_ENTRY_KEYS
