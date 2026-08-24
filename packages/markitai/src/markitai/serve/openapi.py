"""OpenAPI schema export for ``markitai serve`` (no running server needed).

Builds the FastAPI app hermetically and returns its OpenAPI document with the
SSE event payload models injected. The SSE models (:data:`SSE_EVENTS` in
:mod:`markitai.serve.schemas`) never appear in a route signature — without
the explicit injection the exported contract would miss the event stream the
webapp actually consumes.

Consumers: ``scripts/export_openapi.py`` (CLI dump) and
``tests/unit/serve/test_contract_sync.py`` (contract test against
``webapp/src/api/types.ts``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from markitai.serve.schemas import SSE_EVENTS

_REF_TEMPLATE = "#/components/schemas/{model}"
#: Path whose ``x-sse-events`` extension maps event names to payload schemas.
SSE_ROUTE_PATH = "/api/jobs/{job_id}/events"


def build_openapi_schema() -> dict[str, Any]:
    """Return the serve OpenAPI document, SSE event schemas included.

    The app is built with a never-existing static dir so the export is
    deterministic: the JSON root hint route is always part of the contract,
    whether or not a webapp bundle happens to be built locally.
    """
    from markitai.serve.app import create_app

    app = create_app(
        static_dir=Path(__file__).with_name("__no_static__"),
        configure_logging=False,
    )
    schema = app.openapi()

    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for model in SSE_EVENTS.values():
        model_schema = model.model_json_schema(
            ref_template=_REF_TEMPLATE, mode="serialization"
        )
        # Route response models already register components under the same
        # names; keep those and only fill in whatever is still missing.
        for name, definition in model_schema.pop("$defs", {}).items():
            components.setdefault(name, definition)
        components.setdefault(model.__name__, model_schema)

    schema["paths"][SSE_ROUTE_PATH]["get"]["x-sse-events"] = {
        event: {"$ref": _REF_TEMPLATE.format(model=model.__name__)}
        for event, model in SSE_EVENTS.items()
    }
    return schema
