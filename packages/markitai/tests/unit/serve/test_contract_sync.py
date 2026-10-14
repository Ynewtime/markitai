"""Contract sync: exported OpenAPI schema vs the shared workbench's API contract.

The response models in ``markitai.serve.schemas`` are this server's half of
the workbench API contract (the shapes in ``test_workbench_contract``). This
file compares them, as exported by ``markitai.serve.openapi`` (the document
``scripts/export_openapi.py`` dumps), with those shapes: a field the contract
names but the schema lacks, or a schema field the contract does not know,
fails CI instead of drifting silently. The behavioural checks (real answers
against the shapes) live in ``test_workbench_contract``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from markitai.serve.openapi import SSE_ROUTE_PATH, build_openapi_schema
from markitai.serve.schemas import SSE_EVENTS
from tests.unit.serve.test_workbench_contract import OPTION_KEYS, SHAPES

#: Contract shape -> OpenAPI component of this server.
COMPONENTS: dict[str, str] = {
    "Capabilities": "Capabilities",
    "CapabilitiesLLM": "CapabilitiesLLM",
    "Extras": "CapabilitiesExtras",
    "Limits": "CapabilitiesLimits",
    "Features": "CapabilitiesFeatures",
    "RemoteServices": "RemoteServices",
    "CloudflareCapability": "CloudflareCapability",
    "PresetFeatures": "PresetFeatures",
    "Pricing": "Pricing",
    "ItemPayload": "ItemPayload",
    "JobProgress": "JobPayload",
    "JobSnapshot": "JobSnapshot",
    "HistoryEntry": "HistoryEntry",
    "CreatedItem": "CreatedItem",
    "CreatedJob": "CreateJobResponse",
    "CancelResponse": "CancelResponse",
    "Artifact": "ItemArtifact",
    "ItemResult": "ItemResult",
    "DownloadTicket": "DownloadTicket",
    "RegeneratedToken": "RegeneratedToken",
    "Access": "CapabilitiesAccess",
    "AccessToken": "AccessTokenEntry",
    "AccessTokenList": "AccessTokenList",
    "CreatedAccessToken": "CreatedAccessToken",
    "BatchDeleteEntry": "BatchDeleteEntry",
    "BatchDeleteResult": "BatchDeleteResult",
    "ConfigDocument": "ConfigDocument",
    "SavedConfigDocument": "SavedConfigDocument",
    "PageBox": "PageBox",
    "OriginalDescriptor": "OriginalDescriptor",
    "PageCapture": "PageCaptureQualities",
    "CaptureQuality": "CaptureQuality",
    "PageTextLine": "PageTextLine",
    "PageText": "PageText",
    "ProviderCard": "ProviderConnection",
    "ProviderCredentials": "LLMProviderCredentials",
    "SettingsView": "LLMSettingsPayload",
}

#: Contract fields only the Rust server sends (the UI treats them as optional).
RUST_ONLY: dict[str, set[str]] = {
    "JobSnapshot": {"persistence_error"},
}

#: Fields this server sends beyond the contract (the UI ignores them).
PYTHON_EXTRAS: dict[str, set[str]] = {
    # The model a subscription runtime's card starts with
    "ProviderCard": {"default_model"},
    # Python 1.2's name for the credentials display value, kept for 1.2 clients
    "ProviderCredentials": {"api_key_masked"},
}


@pytest.fixture(scope="module")
def openapi() -> dict:
    return build_openapi_schema()


def _component(openapi: dict, name: str) -> dict:
    components = openapi["components"]["schemas"]
    assert name in components, f"component {name} missing from exported schema"
    return components[name]


class TestResponseShapes:
    @pytest.mark.parametrize(("shape", "component"), COMPONENTS.items())
    def test_fields_match_the_contract(
        self, openapi: dict, shape: str, component: str
    ) -> None:
        schema_fields = set(_component(openapi, component).get("properties", {}))
        contract_fields = set(SHAPES[shape]) - RUST_ONLY.get(shape, set())
        missing = contract_fields - schema_fields
        unknown = schema_fields - contract_fields - PYTHON_EXTRAS.get(shape, set())
        assert not missing and not unknown, (
            f"{component} drifted from the contract's {shape}: "
            f"missing here: {sorted(missing)}; not in the contract: {sorted(unknown)}"
        )

    @pytest.mark.parametrize(("shape", "component"), COMPONENTS.items())
    def test_required_contract_fields_are_required(
        self, openapi: dict, shape: str, component: str
    ) -> None:
        """A key the UI must find is never optional in this server's schema.

        Keys with a default are always serialized too, so a schema may list a
        required contract key as optional only when it has a default.
        """
        schema = _component(openapi, component)
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        for name, field in SHAPES[shape].items():
            if not field.required or name in RUST_ONLY.get(shape, set()):
                continue
            assert name in required or "default" in properties.get(name, {}), (
                f"{component}.{name} must always be sent"
            )


class TestRequestShapes:
    def test_job_options_accept_every_contract_key(self, openapi: dict) -> None:
        fields = set(_component(openapi, "JobOptions").get("properties", {}))
        assert fields == {*OPTION_KEYS, "remote_processing"}


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


class TestRoutes:
    """Every route the workbench calls is part of the exported API."""

    @pytest.mark.parametrize(
        ("path", "method"),
        [
            ("/api/capabilities", "get"),
            ("/api/jobs", "post"),
            ("/api/jobs/{job_id}", "get"),
            ("/api/jobs/{job_id}/events", "get"),
            ("/api/jobs/{job_id}/cancel", "post"),
            ("/api/jobs/{job_id}/items/{item_id}/retry", "post"),
            ("/api/jobs/{job_id}/items/{item_id}", "delete"),
            ("/api/jobs/{job_id}/items/{item_id}/result", "get"),
            ("/api/jobs/{job_id}/files/{relpath}", "get"),
            ("/api/jobs/{job_id}/archive", "get"),
            ("/api/download-tickets", "post"),
            ("/api/history", "get"),
            ("/api/history/archive", "get"),
            ("/api/history/{job_id}", "delete"),
            ("/api/jobs/{job_id}/items/{item_id}/original", "get"),
            ("/api/jobs/{job_id}/items/{item_id}/original/file", "get"),
            ("/api/jobs/{job_id}/items/{item_id}/original/pages/{page}", "get"),
            ("/api/jobs/{job_id}/items/{item_id}/original/pages/{page}/text", "get"),
            ("/api/settings/llm", "get"),
            ("/api/settings/llm/providers", "get"),
            ("/api/settings/llm/providers/{provider_id}/credentials", "get"),
            ("/api/settings/llm/detected", "get"),
            ("/api/settings/llm/model-discovery", "post"),
            ("/api/settings/llm/test", "post"),
            ("/api/settings/llm/deployments/batch", "post"),
            ("/api/settings/llm/deployments/{deployment_id}", "patch"),
            ("/api/settings/llm/deployments/{deployment_id}", "delete"),
            ("/api/settings/llm/providers/{provider_id}", "patch"),
            ("/api/settings/llm/providers/{provider_id}", "delete"),
            ("/api/settings/llm/config/open", "post"),
        ],
    )
    def test_route_exists(self, openapi: dict, path: str, method: str) -> None:
        assert method in openapi["paths"].get(path, {}), f"{method.upper()} {path}"
