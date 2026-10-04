#!/usr/bin/env python3
"""Verify reference configuration facts with explicit Rust output extensions.

Run with the reference environment's Python interpreter. This imports only its
configuration models, never its CLI, and reads no user configuration or dotenv.
No reference documentation or implementation source is copied into the output.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--write", action="store_true", help="replace derived fact files")
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference / "packages/markitai/src"))
    import markitai.config as reference
    from pydantic import BaseModel

    schema = reference.MarkitaiConfig.model_json_schema()
    for name, cls in vars(reference).items():
        if not inspect.isclass(cls) or not issubclass(cls, BaseModel):
            continue
        if cls.__module__ != reference.__name__:
            continue
        node = schema if name == "MarkitaiConfig" else schema.get("$defs", {}).get(name)
        if node is None:
            continue
        for key, field in cls.model_fields.items():
            if field.is_required():
                continue
            value = field.get_default(call_default_factory=True)
            node["properties"][key]["default"] = {} if isinstance(value, BaseModel) else value

    def facts(value):
        if isinstance(value, dict):
            return {key: facts(item) for key, item in value.items() if key not in {"title", "description"}}
        if isinstance(value, list):
            return [facts(item) for item in value]
        return value

    # Rust-only page/slide marker preference; the reference remains read-only.
    schema["$defs"]["OutputConfig"]["properties"]["page_markers"] = {"default": False, "type": "boolean"}
    defaults = reference.MarkitaiConfig().model_dump(mode="json")
    defaults["output"]["page_markers"] = False

    root = Path(__file__).resolve().parents[1]
    expected = {
        root / "src/config_contract.json": json.dumps(facts(schema), ensure_ascii=False, separators=(",", ":")) + "\n",
        root / "tests/fixtures/config_defaults.json": json.dumps(defaults, ensure_ascii=False, indent=2) + "\n",
    }
    for path, content in expected.items():
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        elif path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Reference facts differ: {path.name}")
    print(f"Verified {len(expected)} fact files against markitai {reference.MarkitaiConfig.__module__}")


if __name__ == "__main__":
    main()
