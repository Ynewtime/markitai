"""Contract sync: exported OpenAPI schema vs the webapp mirror types.

``webapp/src/api/types.ts`` is a hand-written mirror of the serve API. These
tests compare it against the schema exported by
``markitai.serve.openapi.build_openapi_schema`` (the same document
``scripts/export_openapi.py`` dumps), so a field added or removed on either
side fails CI instead of drifting silently.

The TypeScript side is read with a deliberately small parser: it only needs
the ``export interface`` declarations of the mirror file (field names,
``?`` optionality and ``extends`` chains) — no node toolchain in the Python
test suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from markitai.serve.openapi import SSE_ROUTE_PATH, build_openapi_schema
from markitai.serve.schemas import SSE_EVENTS

REPO_ROOT = Path(__file__).resolve().parents[5]
TYPES_TS = REPO_ROOT / "webapp" / "src" / "api" / "types.ts"

#: TS interface name -> OpenAPI component name for response payloads.
#: Field names and optionality must match exactly in both directions.
RESPONSE_MIRRORS: dict[str, str] = {
    "Capabilities": "Capabilities",
    "PresetFeatures": "PresetFeatures",
    "CreatedItem": "CreatedItem",
    "CreateJobResponse": "CreateJobResponse",
    "ItemPayload": "ItemPayload",
    "JobPayload": "JobPayload",
    "JobSnapshot": "JobSnapshot",
    "ItemResult": "ItemResult",
    "LLMDeployment": "LLMDeployment",
    "LLMSettingsPayload": "LLMSettingsPayload",
    "LLMProviderCredentials": "LLMProviderCredentials",
    "ProviderConnection": "ProviderConnection",
    "ModelCandidate": "ModelCandidate",
    "ModelDiscoveryResult": "ModelDiscoveryResult",
    "LLMTestResult": "LLMTestResult",
    "HistoryEntry": "HistoryEntry",
}

#: TS interface name -> OpenAPI component name for request bodies. The client
#: may omit optional fields it never sends, so the rules are one-directional:
#: every TS field must exist server-side, and every server-required field must
#: be required in TS.
REQUEST_MIRRORS: dict[str, str] = {
    "JobOptions": "JobOptions",
    "LLMModelCreate": "LLMModelCreate",
    "LLMModelUpdate": "LLMModelUpdate",
    "LLMProviderUpdate": "LLMProviderUpdate",
    "LLMDeploymentBatch": "LLMDeploymentBatch",
}


# ---------------------------------------------------------------------------
# Minimal TypeScript interface parser
# ---------------------------------------------------------------------------


@dataclass
class TSInterface:
    """One ``export interface``: base names and ``field -> is_optional``."""

    name: str
    extends: list[str]
    fields: dict[str, bool]


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def _read_braced(source: str, open_index: int) -> str:
    """Return the text between the brace at *open_index* and its match."""
    assert source[open_index] == "{"
    depth = 0
    for index in range(open_index, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[open_index + 1 : index]
    raise ValueError("unbalanced braces in types.ts")


def _split_declarations(body: str) -> list[str]:
    """Split an interface body on top-level ``;`` (inline objects nest)."""
    declarations: list[str] = []
    depth = 0
    current: list[str] = []
    for char in body:
        if char in "{[(":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == ";" and depth == 0:
            declarations.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        declarations.append(tail)
    return [decl for decl in declarations if decl]


_FIELD_RE = re.compile(r"^(?:readonly\s+)?([A-Za-z_$][\w$]*)(\?)?\s*:")


def parse_interfaces(source: str) -> dict[str, TSInterface]:
    """Parse every ``export interface`` of *source*."""
    source = _strip_comments(source)
    interfaces: dict[str, TSInterface] = {}
    for match in re.finditer(
        r"\bexport\s+interface\s+(\w+)(?:\s+extends\s+([\w\s,]+?))?\s*\{", source
    ):
        name = match.group(1)
        extends = [
            base.strip() for base in (match.group(2) or "").split(",") if base.strip()
        ]
        body = _read_braced(source, match.end() - 1)
        fields: dict[str, bool] = {}
        for declaration in _split_declarations(body):
            field = _FIELD_RE.match(declaration)
            if field is None:
                raise AssertionError(
                    f"unparsable field in interface {name}: {declaration!r}"
                )
            fields[field.group(1)] = field.group(2) == "?"
        interfaces[name] = TSInterface(name=name, extends=extends, fields=fields)
    return interfaces


def resolved_fields(interfaces: dict[str, TSInterface], name: str) -> dict[str, bool]:
    """Fields of *name* including inherited ones (``extends`` chain)."""
    interface = interfaces[name]
    fields: dict[str, bool] = {}
    for base in interface.extends:
        fields.update(resolved_fields(interfaces, base))
    fields.update(interface.fields)
    return fields


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openapi() -> dict:
    """The exported OpenAPI document (built once per module)."""
    return build_openapi_schema()


@pytest.fixture(scope="module")
def ts_source() -> str:
    assert TYPES_TS.is_file(), TYPES_TS
    return TYPES_TS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ts_interfaces(ts_source: str) -> dict[str, TSInterface]:
    return parse_interfaces(ts_source)


def _component(openapi: dict, name: str) -> dict:
    components = openapi["components"]["schemas"]
    assert name in components, f"component {name} missing from exported schema"
    return components[name]


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestResponseMirrors:
    """Response payload mirrors must match the schema field-for-field."""

    @pytest.mark.parametrize(("ts_name", "schema_name"), RESPONSE_MIRRORS.items())
    def test_field_names_and_optionality(
        self,
        openapi: dict,
        ts_interfaces: dict[str, TSInterface],
        ts_name: str,
        schema_name: str,
    ) -> None:
        assert ts_name in ts_interfaces, f"interface {ts_name} missing from types.ts"
        ts_fields = resolved_fields(ts_interfaces, ts_name)
        component = _component(openapi, schema_name)
        schema_fields = set(component.get("properties", {}))
        schema_required = set(component.get("required", []))

        missing_in_ts = schema_fields - set(ts_fields)
        unknown_in_ts = set(ts_fields) - schema_fields
        assert not missing_in_ts and not unknown_in_ts, (
            f"{ts_name} drifted from {schema_name}: "
            f"missing in types.ts: {sorted(missing_in_ts)}; "
            f"unknown to the server: {sorted(unknown_in_ts)}"
        )

        ts_required = {name for name, optional in ts_fields.items() if not optional}
        assert ts_required == schema_required, (
            f"{ts_name} optionality drifted from {schema_name}: "
            f"required only in schema: {sorted(schema_required - ts_required)}; "
            f"required only in types.ts: {sorted(ts_required - schema_required)}"
        )


class TestRequestMirrors:
    """Request body mirrors may narrow, but never invent or under-require."""

    @pytest.mark.parametrize(("ts_name", "schema_name"), REQUEST_MIRRORS.items())
    def test_fields_are_known_and_required_ones_required(
        self,
        openapi: dict,
        ts_interfaces: dict[str, TSInterface],
        ts_name: str,
        schema_name: str,
    ) -> None:
        assert ts_name in ts_interfaces, f"interface {ts_name} missing from types.ts"
        ts_fields = resolved_fields(ts_interfaces, ts_name)
        component = _component(openapi, schema_name)
        schema_fields = set(component.get("properties", {}))
        schema_required = set(component.get("required", []))

        unknown_in_ts = set(ts_fields) - schema_fields
        assert not unknown_in_ts, (
            f"{ts_name} sends fields the server does not accept: "
            f"{sorted(unknown_in_ts)}"
        )
        ts_required = {name for name, optional in ts_fields.items() if not optional}
        under_required = schema_required - ts_required
        assert not under_required, (
            f"{ts_name} marks server-required fields as optional or omits "
            f"them: {sorted(under_required)}"
        )


class TestSSEContract:
    """The SSE payloads must be part of the exported contract."""

    def test_sse_models_are_exported_and_mapped(self, openapi: dict) -> None:
        mapping = openapi["paths"][SSE_ROUTE_PATH]["get"].get("x-sse-events")
        assert mapping is not None, "events route lost its x-sse-events mapping"
        assert set(mapping) == set(SSE_EVENTS)
        for event, model in SSE_EVENTS.items():
            ref = mapping[event]["$ref"]
            assert ref == f"#/components/schemas/{model.__name__}"
            _component(openapi, model.__name__)  # ref target must exist


class TestServerOwnedConstants:
    """Server-enforced limits reach the UI via capabilities, not copies."""

    def test_max_job_items_is_served_via_capabilities(self, openapi: dict) -> None:
        capabilities = _component(openapi, "Capabilities")
        assert "limits" in capabilities.get("properties", {}), (
            "capabilities must expose server limits (limits.max_job_items)"
        )
        assert "limits" in capabilities.get("required", [])
        limits = _component(openapi, "CapabilitiesLimits")
        assert "max_job_items" in limits.get("properties", {})

    def test_types_ts_does_not_hardcode_max_job_items(self, ts_source: str) -> None:
        assert not re.search(r"\bMAX_JOB_ITEMS\s*=", ts_source), (
            "types.ts re-declares MAX_JOB_ITEMS; the webapp must read "
            "capabilities.limits.max_job_items at runtime instead"
        )

    def test_types_ts_header_points_at_the_contract_mechanism(
        self, ts_source: str
    ) -> None:
        first_line = ts_source.splitlines()[0]
        assert "scripts/export_openapi.py" in first_line, (
            "the types.ts header must reference the OpenAPI export that "
            "this test validates it against (not a nonexistent document)"
        )
