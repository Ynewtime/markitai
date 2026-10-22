"""Request and response schemas for the serve API.

The response models are this server's half of the shared web workbench's API
contract: ``scripts/export_openapi.py`` dumps them into an OpenAPI document
(including the SSE event payloads via :data:`SSE_EVENTS`) and
``tests/unit/serve/test_contract_sync.py`` compares that document against the
contract's shapes (``tests/unit/serve/test_workbench_contract.py``).

Response-model discipline: FastAPI silently drops any response key the model
does not declare. Models for payloads this package builds itself therefore use
``extra="forbid"`` (an undeclared key fails loudly instead of vanishing);
models wrapping detector-produced cards with naturally varying keys use
``extra="allow"`` and their routes serialize with
``response_model_exclude_unset=True`` so absent optional keys stay absent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from markitai.config import ConversionBackend, FetchStrategy, OutputProfile


class JobOptions(BaseModel):
    """Options accepted by ``POST /api/jobs``.

    Strict: an unknown key or a value of the wrong type ("yes" for a switch)
    is refused rather than coerced.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

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
    # Consent to Cloudflare processing for this one request (a Cloudflare
    # strategy or backend in the same options). Never stored or echoed.
    remote_processing: Literal["cloudflare"] | None = None


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


class UserModelTest(LLMSettingsUpdate):
    """``POST /api/me/models/test``: one of the user's own models, a draft,
    or ``{shared_id}`` alone, a model the admin shares (contract v1.16)."""

    shared_id: str | None = None

    @model_validator(mode="after")
    def _one_form_only(self) -> UserModelTest:  # replaces the parent's rule
        if self.shared_id is None:
            return LLMSettingsUpdate._one_form_only(self)  # type: ignore[return-value]
        if len(self.model_fields_set) != 1:
            raise ValueError("shared_id cannot be combined with other probe fields")
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
    # Round 18: a whole number from 0 to 100 (422 ``invalid_weight``)
    weight: StrictInt = Field(default=1, ge=0, le=100)
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
    weight: StrictInt | None = Field(default=None, ge=0, le=100)
    expected_revision: str | None = None
    # v1.15: whether user tokens' conversions may use the admin's model
    # (stored as model_info.share; the user routes refuse it)
    share: Literal["private", "shared"] | None = None

    @field_validator("share")
    @classmethod
    def _share_not_null(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("share must be 'private' or 'shared'")
        return value

    @field_validator("model_name", "model")
    @classmethod
    def _require_non_blank_model(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("value cannot be null; omit the field to keep it")
        return _require_non_blank(value) or ""

    # A masked api_base is checked by the route: sent back as shown, it keeps
    # the stored base
    @field_validator("model_name", "model", "api_key", "expected_revision")
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

    @field_validator("provider_id", "deployment_id", "api_key")
    @classmethod
    def _no_mask_char(cls, value: str | None) -> str | None:
        return _reject_mask_char(value)


class LLMProviderUpdate(BaseModel):
    """Partial update for one saved provider connection."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    api_base: str | None = None
    expected_revision: str

    @field_validator("api_key", "expected_revision")
    @classmethod
    def _valid_value(cls, value: str | None) -> str | None:
        value = _reject_mask_char(value)
        return _require_non_blank(value) if value is not None else None

    @field_validator("api_base")
    @classmethod
    def _valid_base(cls, value: str | None) -> str | None:
        # A masked base is checked by the route: sent back as shown, it keeps
        # the stored base
        return _require_non_blank(value) if value is not None else None

    @model_validator(mode="after")
    def _has_update(self) -> LLMProviderUpdate:
        if not ({"api_key", "api_base"} & self.model_fields_set):
            raise ValueError("api_key or api_base is required")
        return self


class LLMDeploymentBatch(BaseModel):
    """One atomic change (round 18): add ``deployments`` and remove the saved
    deployments ``remove`` lists, every id as the list at
    ``expected_revision`` names it, all or nothing."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: str
    deployments: list[LLMModelCreate] = Field(default_factory=list, max_length=50)
    remove: list[str] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def _has_change(self) -> LLMDeploymentBatch:
        if not self.deployments and not self.remove:
            raise ValueError(
                "deployments must contain 1 to 50 entries, or remove at least "
                "one deployment id"
            )
        if len(set(self.remove)) != len(self.remove):
            raise ValueError("remove lists one deployment twice")
        if any(not deployment_id.strip() for deployment_id in self.remove):
            raise ValueError("remove must list deployment ids")
        return self

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
    history_retention_days: int | None = None  # None: kept until deleted


class CapabilitiesFeatures(BaseModel):
    """What this backend implements beyond the base job API."""

    model_config = ConfigDict(extra="forbid")

    cancel: bool
    download_tickets: bool
    url_lists: bool
    history_items: bool
    original: bool
    result_excerpt: bool
    job_labels: bool
    item_assets: bool  # contract v1.6: GET .../items/{item_id}/assets/{relpath}
    item_archive: bool  # contract v1.11: GET .../items/{item_id}/archive
    token_regenerate: bool  # v1.11: POST /api/access-token/regenerate (not --no-auth)
    user_tokens: bool  # v1.13: /api/access-tokens and access levels (not --no-auth)
    config_edit: bool  # v1.13: GET/PUT /api/settings/config
    batch_delete: bool  # v1.13: POST /api/history/delete
    convert_defaults: bool  # v1.17: GET/PUT /api/convert-defaults
    pdf_outline: bool  # v1.17: GET …/original/outline
    user_models: bool  # v1.15: /api/me/models* and `share` (not --no-auth)
    user_config: bool  # v1.16: GET/PUT /api/me/config (not --no-auth)


class CapabilitiesAccess(BaseModel):
    """Who is asking (contract v1.13)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["admin", "user"]
    label: str | None = None  # a user token's name


class CloudflareCapability(BaseModel):
    """Whether this server can process a request with Cloudflare."""

    model_config = ConfigDict(extra="forbid")

    configured: bool
    available: bool
    reason: (
        Literal[
            "not_configured",
            "invalid_configuration",
            "disabled_by_policy",
            "client_not_trusted",
        ]
        | None
    )
    browser_rendering: bool
    file_conversion: bool
    file_extensions: list[str]


class RemoteServices(BaseModel):
    """Remote services a request may opt into."""

    model_config = ConfigDict(extra="forbid")

    cloudflare: CloudflareCapability


class PresetFeatures(BaseModel):
    """Resolved preset values, including config-file overrides."""

    model_config = ConfigDict(extra="forbid")

    llm: bool
    ocr: bool
    alt: bool
    desc: bool
    screenshot: bool


class BuildInfo(BaseModel):
    """Which build answers (``Capabilities.build``, contract v1.16)."""

    model_config = ConfigDict(extra="forbid")

    version: str
    commit: str | None  # the short commit hash; None when unknown
    built_at: str | None  # RFC 3339 UTC; None when unknown (a source checkout)
    dirty: bool  # the build's tree had uncommitted changes


class Capabilities(BaseModel):
    """Response of ``GET /api/capabilities``."""

    model_config = ConfigDict(extra="forbid")

    version: str
    implementation: Literal["python"]
    docs_url: str
    source_url: str  # this server's source repository (the UI's GitHub link)
    llm: CapabilitiesLLM
    presets: list[str]
    preset_options: dict[str, PresetFeatures]
    extras: CapabilitiesExtras
    limits: CapabilitiesLimits
    features: CapabilitiesFeatures
    remote_services: RemoteServices
    access: CapabilitiesAccess
    build: BuildInfo  # v1.16, for every caller


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
    # v1.15: the configuration file's rows only (absent for detected rows
    # and a user's own)
    share: Literal["private", "shared"] | None = None


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
    # Round 18: the batch route also takes ``remove``
    batch_remove: bool = True


class SharedModel(BaseModel):
    """An admin's shared model as a user sees it (contract v1.15, v1.16):
    nothing else, ever (no key, mask, address, host, group or connection)."""

    model_config = ConfigDict(extra="forbid")

    # The admin's deployment id: taken only by PATCH /api/me/models/shared/{id}
    # and POST /api/me/models/test {shared_id}
    id: str
    model: str
    provider: str  # the connection's provider; "custom" = OpenAI-compatible
    weight: int  # the user's effective weight (its own, else the admin's)
    default_weight: int  # the admin's weight


class UserModelsView(BaseModel):
    """Response of ``GET /api/me/models`` and every user model write."""

    model_config = ConfigDict(extra="forbid")

    configured: bool  # the user has models of its own
    routable: bool  # its pool (own + shared) can serve a conversion now
    source: Literal["user"]
    config_path: None
    config_origin: Literal["user"]
    revision: str  # over its own models, connections and shared weights
    deployments: list[LLMDeployment]  # its own, without `share`
    detected: list[LLMDeployment]  # always empty
    shared: list[SharedModel]  # the admin's shared models with weight > 0
    limit: int  # its own models at most (50)
    batch_remove: bool = True  # round 18: the batch route also takes `remove`


class LLMProviderCredentials(BaseModel):
    """Response of ``GET /api/settings/llm/providers/{provider_id}/credentials``."""

    model_config = ConfigDict(extra="forbid")

    # Masked like every other settings response; ``env:`` references verbatim.
    # The editor omits ``api_key`` on save unless the user typed a new one.
    # ``api_key`` is the same display value under the shared workbench's name.
    api_key: str | None
    api_key_masked: str | None
    # "environment" when no key is stored and the server's own key variable
    # serves the connection: ``api_key`` is then ``env:<VARIABLE>``, the
    # variable conversions and model tests read (as Rust's ``credentials``)
    api_key_source: Literal["environment"] | None = None
    # Saved base without userinfo or secret parameters (``…``); null when on
    # the provider default
    api_base: str | None
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
    # Provider picker (contract v1.5, markitai.serve.provider_catalog): the
    # catalog group, the brand mark's key (None: a monogram) and further
    # model prefixes and searched names
    group: (
        Literal[
            "common",
            "subscriptions",
            "makers",
            "clouds",
            "inference",
            "gateways",
            "local",
        ]
        | None
    ) = None
    icon: str | None = None
    aliases: list[str] | None = None
    # How the provider is connected (markitai.providers.connectable): the
    # address used when none is entered (null: one must be entered, or the
    # provider takes none), the variable LiteLLM reads its key from, and
    # whether no key needs entering (local servers, subscription runtimes,
    # cloud credentials)
    default_base: str | None = None
    key_variable: str | None = None
    key_optional: bool | None = None


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
    # Why the list could not be loaded, with the model test's reasons
    # (contract v1.10): no_key, key_rejected, rate_limited, network,
    # timeout, provider_error; with no_key, when known, the variable the key
    # would come from (a name, never a value)
    reason: str | None = None
    key_variable: str | None = None
    # A subscription runtime (Claude, Copilot, ChatGPT) that gave no list
    # says its cause (contract v1.18): reason is then runtime_missing,
    # not_signed_in, runtime_failed or timeout, ``runtime`` names the program
    # and the release the adapter speaks, ``next_step`` what to run on the
    # machine where the server runs
    runtime: dict[str, str] | None = None
    next_step: dict[str, str] | None = None


class LLMTestResult(BaseModel):
    """Response of ``POST /api/settings/llm/test`` (always 200)."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    detail: str
    # Why a failed test failed (serve.model_test.REASONS, contract v1.10):
    # no_key, key_rejected, model_not_found, rate_limited, network, timeout,
    # provider_error; absent on success
    reason: str | None = None
    # With no_key, when known: the variable the key would come from (a name,
    # never a value)
    key_variable: str | None = None


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


class Pricing(BaseModel):
    """Request coverage of a model cost."""

    model_config = ConfigDict(extra="forbid")

    priced_requests: int
    unpriced_requests: int
    # complete: every request priced; partial: cost_usd is a lower bound;
    # unknown: no request was priced, so cost_usd says nothing (never "$0")
    cost_status: Literal["complete", "partial", "unknown"]


class JobOptionsEcho(BaseModel):
    """Saved options echoed by snapshots, items and history.

    Never carries ``remote_processing``; a CLI-recorded job adds ``origin``.
    """

    model_config = ConfigDict(extra="allow")

    preset: str | None
    llm: bool | None
    ocr: bool | None


class RerunFailure(BaseModel):
    """The last failed rerun of an item whose previous result was kept."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["retry", "enhance"]
    error_code: str
    error: str
    failed_at: str


class PageSource(BaseModel):
    """Where one PDF page's text came from (beyond the workbench contract)."""

    model_config = ConfigDict(extra="forbid")

    page: int  # 1-based
    source: Literal["text", "ocr", "mixed"]
    # Short English note: "text layer trusted", "no text layer", "text layer
    # failed the script check: ...", "OCR forced by --ocr ..."
    reason: str


class ItemPayload(BaseModel):
    """Payload of SSE ``event: item`` (and of items inside the snapshot).

    Mirror of :meth:`markitai.serve.jobs.Job.item_payload`.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    name: str
    kind: str  # "file" | "url"
    status: str  # "queued" | "running" | "done" | "error"
    error: str | None
    # Stable cause of a failure: unsupported, fetch_error, conversion_error,
    # no_output, enhancement_failed, cancelled, shutdown, internal_error, ...
    error_code: str | None = None
    output: str | None
    output_name: str | None  # pre-assigned unique output name (url items)
    duration_ms: int | None
    finished_at: str | None
    cost_usd: float | None
    pricing: Pricing | None = None  # present whenever a model answered requests
    llm_enhanced: bool
    operation: str  # "convert" | "retry" | "enhance"
    skipped: bool
    skip_reason: str | None
    # False when the item cannot be retried/enhanced (CLI-recorded files)
    retryable: bool
    # Actionable notices raised while the item converted (scanned pages,
    # hidden text, OCR found nothing, screenshot not captured, ...)
    warnings: list[str]
    models: list[str] = Field(default_factory=list)  # models that produced it
    source_size: int | None = None  # bytes of the uploaded original
    options: JobOptionsEcho | None = None  # this item's saved selections
    # Contract v1.6, 6.6: a failed retry/enhance kept the previous result
    rerun_failure: RerunFailure | None = None
    # A PDF's pages, each with where its text came from; empty otherwise
    page_sources: list[PageSource] = Field(default_factory=list)


class JobPayload(BaseModel):
    """Payload of SSE ``event: job``."""

    model_config = ConfigDict(extra="forbid")

    status: str  # "running" | "done"
    done: int
    failed: int
    total: int


class JobSnapshot(JobPayload):
    """Payload of SSE ``event: snapshot`` and ``GET /api/jobs/{job_id}``."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    created_at: str
    finished_at: str | None
    options: JobOptionsEcho
    items: list[ItemPayload]
    label: str | None = None  # what the submission is called in the list


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
    # With ``?max_bytes=``: whether the excerpt was cut, and the full size
    truncated: bool | None = None
    markdown_bytes: int | None = None


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
    pricing: Pricing | None = None  # job-level coverage of cost_usd
    names_preview: list[str]
    kinds_preview: list[str]
    duration_ms: int | None
    size_bytes: int
    origin: str  # "web" | "cli"
    retryable: bool  # at least one item can be retried/enhanced
    label: str | None = None
    # With ``?include=items``
    options: JobOptionsEcho | None = None
    items: list[ItemPayload] | None = None


class CancelResponse(BaseModel):
    """Response of ``POST /api/jobs/{job_id}/cancel``."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    stopping: int  # queued items that will not start


class DownloadTicketRequest(BaseModel):
    """Body of ``POST /api/download-tickets``."""

    model_config = ConfigDict(extra="forbid")

    path: str


class RegeneratedToken(BaseModel):
    """Response of ``POST /api/access-token/regenerate``."""

    model_config = ConfigDict(extra="forbid")

    token: str
    network_url: str | None  # http://<network address>:<port>/#token=<token>
    persistent: bool  # survives a restart (Python: never, it lives in memory)
    startup_variable: str | None  # "MARKITAI_SERVE_TOKEN" when it set the token


class AccessTokenEntry(BaseModel):
    """One user token as ``GET /api/access-tokens`` lists it (never the token)."""

    model_config = ConfigDict(extra="forbid")

    id: str  # u_ + 12 hex digits
    label: str
    hint: str  # the token's last 4 characters
    created_at: str
    last_used_at: str | None  # kept in memory only: null after a restart


class ListedAccessToken(AccessTokenEntry):
    """One user token in ``GET /api/access-tokens`` (v1.15: with its models)."""

    model_count: int  # the user's own models (never which)


class AccessTokenList(BaseModel):
    """Response of ``GET /api/access-tokens``."""

    model_config = ConfigDict(extra="forbid")

    tokens: list[ListedAccessToken]  # oldest first


class CreatedAccessToken(AccessTokenEntry):
    """Response of ``POST /api/access-tokens``: the only time ``token`` is sent."""

    token: str  # mku_ + 64 hex digits


class BatchDeleteEntry(BaseModel):
    """One entry's outcome in ``POST /api/history/delete``."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    item_id: str | None  # None: the whole job
    ok: bool
    reason: str | None = None  # what the single route would answer
    detail: str | None = None


class BatchDeleteResult(BaseModel):
    """Response of ``POST /api/history/delete``."""

    model_config = ConfigDict(extra="forbid")

    deleted: int
    failed: int
    results: list[BatchDeleteEntry]  # request order


class ConfigDocument(BaseModel):
    """Response of ``GET /api/settings/config``: the file's text, secrets masked."""

    model_config = ConfigDict(extra="forbid")

    path: str | None  # null for a user's own document (v1.16)
    origin: str
    exists: bool
    revision: str  # "sha256:<hex of the file's bytes>" or "absent"
    text: str | None  # null when the file cannot be shown (see problem)
    masked: int
    problem: str | None
    read_only: bool


class SavedConfigDocument(ConfigDocument):
    """Response of ``PUT /api/settings/config``."""

    restart_required: bool  # something outside llm.model_list/providers changed


class DownloadTicket(BaseModel):
    """Response of ``POST /api/download-tickets``."""

    model_config = ConfigDict(extra="forbid")

    ticket: str
    url: str  # "<path>?ticket=<ticket>"
    expires_in: int  # seconds


class PageBox(BaseModel):
    """Size of one page: points for a PDF, pixels for images."""

    model_config = ConfigDict(extra="forbid")

    width: float
    height: float


class CaptureQuality(BaseModel):
    """How usable a URL item's kept page or screenshot was when captured."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["ok", "empty", "challenge"]
    text_length: int  # visible text, white-space runs as one, at most 1 000 000
    title: str | None  # at most 200 characters
    # challenge | verification | login | javascript | blank; None when ok
    reason: str | None


class PageCaptureQualities(BaseModel):
    """A URL item's ``capture`` (contract v1.14 section 7)."""

    model_config = ConfigDict(extra="forbid")

    page: CaptureQuality | None
    screenshot: CaptureQuality | None


class OriginalDescriptor(BaseModel):
    """Response of ``GET /api/jobs/{job_id}/items/{item_id}/original``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["pdf", "image", "html", "text", "screenshots", "none"]
    name: str
    media_type: str | None
    size: int | None
    download: str | None  # API path of the stored original
    view: str | None  # html only: API path of .../original/view (v1.11)
    pages: list[PageBox]
    page_count: int
    text_layer: bool
    # not_kept | unsupported_format | too_large | render_unavailable
    reason: str | None
    # URL items (v1.14): how usable each kept view was; null for uploads
    capture: PageCaptureQualities | None = None
    # v1.14: API path of .../original/blocks when the screenshots have boxes
    blocks: str | None = None


class PageTextLine(BaseModel):
    """One line of a page's text layer; bbox from the page's top-left, in points."""

    model_config = ConfigDict(extra="forbid")

    text: str
    bbox: tuple[float, float, float, float]


class PageTextImage(BaseModel):
    """Where a page draws one picture; bbox from the page's top-left, in points."""

    model_config = ConfigDict(extra="forbid")

    bbox: tuple[float, float, float, float]


class PageText(BaseModel):
    """Response of ``GET .../original/pages/{n}/text``."""

    model_config = ConfigDict(extra="forbid")

    page: int
    width: float
    height: float
    lines: list[PageTextLine]
    #: The page's pictures in drawing order (at most 500, none under 4
    #: points on a side), for Compare to outline them (contract v1.12).
    images: list[PageTextImage] = []


class PdfOutlineEntry(BaseModel):
    """One bookmark of a PDF: its title, 1-based page and 1-based level."""

    model_config = ConfigDict(extra="forbid")

    title: str
    page: int
    level: int


class PdfOutline(BaseModel):
    """Response of ``GET .../original/outline`` (contract v1.17)."""

    model_config = ConfigDict(extra="forbid")

    items: list[PdfOutlineEntry]
    truncated: bool


class RootInfo(BaseModel):
    """JSON hint served at ``/`` when no web UI is bundled."""

    model_config = ConfigDict(extra="forbid")

    markitai: str
    hint: str


#: SSE event name -> payload model for ``GET /api/jobs/{job_id}/events``.
#: These models never appear in a route signature, so the OpenAPI export
#: injects them explicitly — without this the contract test would miss the
#: most drift-prone surface (the event stream the web UI actually consumes).
SSE_EVENTS: dict[str, type[BaseModel]] = {
    "snapshot": JobSnapshot,
    "item": ItemPayload,
    "job": JobPayload,
}
