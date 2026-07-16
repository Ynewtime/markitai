"""Request schemas for the serve API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobOptions(BaseModel):
    """Options accepted by ``POST /api/jobs``."""

    model_config = ConfigDict(extra="forbid")

    preset: str | None = None
    llm: bool | None = None
    ocr: bool | None = None


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
