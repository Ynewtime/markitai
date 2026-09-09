"""Request and response schemas for the serve API.

The response models double as the machine-readable contract consumed by the
webapp: ``scripts/export_openapi.py`` dumps them into an OpenAPI document
(including the SSE event payloads via :data:`SSE_EVENTS`) and
``tests/unit/serve/test_contract_sync.py`` compares that document against the
hand-written mirror in ``webapp/src/api/types.ts``.

Response-model discipline: FastAPI silently drops any response key the model
does not declare. Models for payloads this package builds itself therefore use
``extra="forbid"`` (an undeclared key fails loudly instead of vanishing);
models wrapping detector-produced cards with naturally varying keys use
``extra="allow"`` and their routes serialize with
``response_model_exclude_unset=True`` so absent optional keys stay absent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from markitai.config import ConversionBackend, FetchStrategy, OutputProfile


class JobOptions(BaseModel):
    """Options accepted by ``POST /api/jobs``."""

    model_config = ConfigDict(extra="forbid")

    preset: str | None = None
    llm: bool | None = None
    ocr: bool | None = None
    profile: OutputProfile | None = None
    # Individually settable rather than only as part of a preset. Each is
    # tri-state: None leaves whatever the preset (or the server config)
    # decided, so "not sent" and "explicitly off" stay different answers.
    alt: bool | None = None
    desc: bool | None = None
    screenshot: bool | None = None
    screenshot_only: bool | None = None
    pure: bool | None = None
    no_cache: bool | None = None
    no_compress: bool | None = None
    strategy: FetchStrategy | None = None
    backend: ConversionBackend | None = None


class JobRetryBody(BaseModel):
    """Optional body of ``POST /api/jobs/{job_id}/items/{item_id}/retry``."""

    model_config = ConfigDict(extra="forbid")

    options: JobOptions | None = None
    operation: Literal["retry", "enhance"] = "retry"


def _require_non_blank(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError("must be a non-empty string")
    return value


def _reject_mask_char(value: str | None) -> str | None:
    """Reject values containing U+2026 (mask-writeback guard)."""
    if value is not None and "…" in value:
        raise ValueError(
            "value contains the mask character '…'; "
            "send the real value or omit the field to keep the current one"
        )
    return value


class LLMSettingsUpdate(BaseModel):
    """Transient probe by deployment id, legacy routing group, or ad-hoc values."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str | None = None
    model_name: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None

    @field_validator("deployment_id", "model_name", "model")
    @classmethod
    def _non_blank(cls, value: str | None) -> str | None:
        return _require_non_blank(value)

    @model_validator(mode="after")
    def _one_form_only(self) -> LLMSettingsUpdate:
        references = sum(
            value is not None for value in (self.deployment_id, self.model_name)
        )
        if references > 1:
            raise ValueError("send deployment_id or model_name, not both")
        if references == 1:
            if (
                self.model is not None
                or self.api_key is not None
                or self.api_base is not None
            ):
                raise ValueError(
                    "a stored deployment reference cannot be combined with "
                    "model/api_key/api_base"
                )
        elif self.model is None:
            raise ValueError("deployment_id, model_name, or model is required")
        return self


class LLMModelCreate(BaseModel):
    """One deployment to append to ``llm.model_list``.

    ``model_name`` is a LiteLLM routing group and may be shared by multiple
    deployments. ``model_info.id`` is generated server-side.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    model: str
    provider: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    weight: int = Field(default=1, ge=0)
    credential_provider_id: str | None = None
    credential_deployment_id: str | None = None
    expected_revision: str | None = None

    @field_validator("model_name", "model")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        return _require_non_blank(value) or ""

    @field_validator("provider")
    @classmethod
    def _optional_non_blank(cls, value: str | None) -> str | None:
        return _require_non_blank(value)

    @field_validator(
        "model_name",
        "model",
        "provider",
        "api_key",
        "api_base",
        "credential_provider_id",
        "credential_deployment_id",
        "expected_revision",
    )
    @classmethod
    def _no_mask_char(cls, value: str | None) -> str | None:
        return _reject_mask_char(value)

    @model_validator(mode="after")
    def _one_credential_reference(self) -> LLMModelCreate:
        if (
            self.credential_provider_id is not None
            and self.credential_deployment_id is not None
        ):
            raise ValueError(
                "send credential_provider_id or credential_deployment_id, not both"
            )
        return self


class LLMModelUpdate(BaseModel):
    """Partial deployment update; omitted values keep their stored value."""

    model_config = ConfigDict(extra="forbid")

    model_name: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    weight: int | None = Field(default=None, ge=0)
    expected_revision: str | None = None

    @field_validator("model_name", "model")
    @classmethod
    def _require_non_blank_model(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("value cannot be null; omit the field to keep it")
        return _require_non_blank(value) or ""

    @field_validator("model_name", "model", "api_key", "api_base", "expected_revision")
    @classmethod
    def _no_mask_char(cls, value: str | None) -> str | None:
        return _reject_mask_char(value)


class LLMModelDiscoveryRequest(BaseModel):
    """Connection draft used to discover provider models without persisting it."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    provider_id: str | None = None
    deployment_id: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    refresh: bool = False

    @field_validator("provider")
    @classmethod
    def _provider_not_blank(cls, value: str) -> str:
        return _require_non_blank(value) or ""

    @field_validator("provider_id", "deployment_id", "api_key", "api_base")
    @classmethod
    def _no_mask_char(cls, value: str | None) -> str | None:
        return _reject_mask_char(value)


class LLMProviderUpdate(BaseModel):
    """Partial update for one saved provider connection."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    api_base: str | None = None
    expected_revision: str

    @field_validator("api_key", "api_base", "expected_revision")
    @classmethod
    def _valid_value(cls, value: str | None) -> str | None:
        value = _reject_mask_char(value)
        return _require_non_blank(value) if value is not None else None

    @model_validator(mode="after")
    def _has_update(self) -> LLMProviderUpdate:
        if not ({"api_key", "api_base"} & self.model_fields_set):
            raise ValueError("api_key or api_base is required")
        return self


class LLMDeploymentBatch(BaseModel):
    """Atomic creation of one or more deployments."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: str
    deployments: list[LLMModelCreate] = Field(min_length=1, max_length=50)

    @field_validator("expected_revision")
    @classmethod
    def _revision_not_blank(cls, value: str) -> str:
        return _require_non_blank(value) or ""


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CapabilitiesLLM(BaseModel):
    """LLM availability summary inside ``GET /api/capabilities``."""

    model_config = ConfigDict(extra="forbid")

    configured: bool
    routable: bool
    effective: bool
    models: list[str]


class CapabilitiesExtras(BaseModel):
    """Optional-dependency availability inside ``GET /api/capabilities``."""

    model_config = ConfigDict(extra="forbid")

    browser: bool
    svg: bool


class CapabilitiesLimits(BaseModel):
    """Server-enforced limits the UI mirrors (single source of truth here)."""

    model_config = ConfigDict(extra="forbid")

    max_job_items: int


class PresetFeatures(BaseModel):
    """Resolved preset values, including config-file overrides."""

    model_config = ConfigDict(extra="forbid")

    llm: bool
    ocr: bool
    alt: bool
    desc: bool
    screenshot: bool


class Capabilities(BaseModel):
    """Response of ``GET /api/capabilities``."""

    model_config = ConfigDict(extra="forbid")

    version: str
    llm: CapabilitiesLLM
    presets: list[str]
    preset_options: dict[str, PresetFeatures]
    extras: CapabilitiesExtras
    limits: CapabilitiesLimits


class LLMDeployment(BaseModel):
    """One secret-free deployment view inside the LLM settings payload."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str
    routing_group: str
    model: str
    weight: int
    api_key_configured: bool
    api_base_configured: bool
    api_base: str | None  # sanitized scheme + host + port only
    persisted: bool


class LLMSettingsPayload(BaseModel):
    """Response of ``GET /api/settings/llm`` and every settings mutation."""

    model_config = ConfigDict(extra="forbid")

    configured: bool
    routable: bool
    source: Literal["config", "detected", "none"]
    config_path: str
    config_origin: Literal["explicit", "environment", "project", "user", "default"]
    revision: str
    deployments: list[LLMDeployment]
    detected: list[LLMDeployment]


class LLMProviderCredentials(BaseModel):
    """Response of ``GET /api/settings/llm/providers/{provider_id}/credentials``."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None
    api_base: str | None  # RAW saved base; null when on the provider default
    api_base_placeholder: str | None  # provider default, for the editor only


class ProviderConnection(BaseModel):
    """One provider connection card of ``GET /api/settings/llm/providers``.

    Cards come from the shared provider detector plus config-derived entries;
    their optional keys legitimately vary per card kind, so the route
    serializes with ``response_model_exclude_unset=True`` and unknown future
    detector keys pass through via ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    provider: str
    label: str
    kind: str  # "local_cli" | "oauth" | "environment" | "configured" | "common"
    status: str
    source: str
    supports_discovery: bool
    provider_id: str | None = None
    deployment_id: str | None = None
    default_model: str | None = None
    credential: str | None = None
    api_key_configured: bool | None = None
    api_base_configured: bool | None = None
    api_base: str | None = None
    model_count: int | None = None


class ProviderConnectionList(BaseModel):
    """Response of ``GET /api/settings/llm/providers``."""

    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderConnection]


class DetectedModel(BaseModel):
    """One legacy quick-add candidate of ``GET /api/settings/llm/detected``."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    label: str
    requires_api_key: bool


class ModelCandidate(BaseModel):
    """One discovered model inside a model-discovery result."""

    model_config = ConfigDict(extra="forbid")

    model: str
    label: str
    supports_vision: bool


class ModelDiscoveryResult(BaseModel):
    """Response of ``POST /api/settings/llm/model-discovery``.

    ``detail`` is only present when discovery has something to explain, so the
    route serializes with ``response_model_exclude_unset=True``.
    """

    model_config = ConfigDict(extra="allow")

    provider: str
    status: str  # "ok" | "partial" | "unavailable"
    source: str
    authoritative: bool
    cached: bool
    stale: bool
    models: list[ModelCandidate]
    detail: str | None = None


class LLMTestResult(BaseModel):
    """Response of ``POST /api/settings/llm/test`` (always 200)."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    detail: str


class CreatedItem(BaseModel):
    """One accepted item echoed by job creation and item retry."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    name: str
    kind: str  # "file" | "url"


class CreateJobResponse(BaseModel):
    """Response of ``POST /api/jobs`` and the item retry endpoint."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    items: list[CreatedItem]


class ItemPayload(BaseModel):
    """Payload of SSE ``event: item`` (and of items inside the snapshot).

    Mirror of :meth:`markitai.serve.jobs.JobItem.to_payload`.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    name: str
    kind: str  # "file" | "url"
    status: str  # "queued" | "running" | "done" | "error"
    error: str | None
    output: str | None
    output_name: str | None  # pre-assigned unique output name (url items)
    duration_ms: int | None
    finished_at: str | None
    cost_usd: float | None
    llm_enhanced: bool
    operation: str  # "convert" | "retry" | "enhance"
    skipped: bool
    skip_reason: str | None


class JobPayload(BaseModel):
    """Payload of SSE ``event: job``."""

    model_config = ConfigDict(extra="forbid")

    status: str  # "running" | "done"
    done: int
    failed: int
    total: int


class JobSnapshotOptions(BaseModel):
    """Options echoed inside a job snapshot.

    Jobs rehydrated from meta.json carry an extra ``origin`` key ("web" |
    "cli"); ``extra="allow"`` passes it through untouched.
    """

    model_config = ConfigDict(extra="allow")

    preset: str | None
    llm: bool | None
    ocr: bool | None


class JobSnapshot(JobPayload):
    """Payload of SSE ``event: snapshot`` and ``GET /api/jobs/{job_id}``."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    created_at: str
    finished_at: str | None
    options: JobSnapshotOptions
    items: list[ItemPayload]


class ItemArtifact(BaseModel):
    """One downloadable artifact of an item result."""

    model_config = ConfigDict(extra="forbid")

    relpath: str
    size: int


class ItemResult(BaseModel):
    """Response of ``GET /api/jobs/{job_id}/items/{item_id}/result``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    variant: Literal["llm", "base"]
    markdown: str
    artifacts: list[ItemArtifact]


class HistoryEntry(BaseModel):
    """One entry of ``GET /api/history`` (time-descending)."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    created_at: str
    finished_at: str | None
    status: str  # "running" | "done"
    total: int
    done: int
    failed: int
    skipped: int
    llm_enhanced: int
    cost_usd: float | None
    names_preview: list[str]
    kinds_preview: list[str]
    duration_ms: int | None
    size_bytes: int
    origin: str  # "web" | "cli"


class RootInfo(BaseModel):
    """JSON hint served at ``/`` when no web UI is bundled."""

    model_config = ConfigDict(extra="forbid")

    markitai: str
    hint: str


#: SSE event name -> payload model for ``GET /api/jobs/{job_id}/events``.
#: These models never appear in a route signature, so the OpenAPI export
#: injects them explicitly — without this the contract test would miss the
#: most drift-prone surface (the event stream the webapp actually consumes).
SSE_EVENTS: dict[str, type[BaseModel]] = {
    "snapshot": JobSnapshot,
    "item": ItemPayload,
    "job": JobPayload,
}
