#!/usr/bin/env python3
"""Export the ``markitai serve`` OpenAPI schema without starting a server.

The document includes the SSE event payload schemas (``x-sse-events`` on the
events route); see ``markitai/serve/openapi.py``. The contract test
``packages/markitai/tests/unit/serve/test_contract_sync.py`` compares it against the
webapp mirror ``webapp/src/api/types.ts``.

Usage:
    uv run python scripts/export_openapi.py               # print to stdout
    uv run python scripts/export_openapi.py -o openapi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write the schema to this file instead of stdout",
    )
    args = parser.parse_args()

    from markitai.serve.openapi import build_openapi_schema

    payload = json.dumps(build_openapi_schema(), indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
