"""Request schemas for the serve API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class JobOptions(BaseModel):
    """Options accepted by ``POST /api/jobs``.

    Semantics mirror the CLI: the preset is applied first (five feature
    booleans), then the explicit ``llm`` flag overrides it (None = keep).
    """

    model_config = ConfigDict(extra="forbid")

    preset: str | None = None
    llm: bool | None = None


class LLMSettingsUpdate(BaseModel):
    """Body of ``POST /api/settings/llm/test``.

    ``model`` is usually ``provider/model-id`` but bare model names are
    accepted too (litellm resolves some); only non-blank is required.
    ``api_key``/``api_base`` support the ``env:VAR`` indirection syntax.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    api_key: str | None = None
    api_base: str | None = None

    @field_validator("model")
    @classmethod
    def _require_non_blank_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model must be a non-empty string")
        return value


def _reject_mask_char(value: str | None) -> str | None:
    """Reject values containing U+2026 (mask-writeback guard).

    ``GET /api/settings/llm`` masks keys with ``…``; a client echoing such a
    masked value back would silently corrupt the stored key. No legitimate
    model name, key or base URL contains the character.
    """
    if value is not None and "…" in value:
        raise ValueError(
            "value contains the mask character '…'; "
            "send the real value or omit the field to keep the current one"
        )
    return value


class LLMModelCreate(BaseModel):
    """Body of ``POST /api/settings/llm/models``.

    ``model_name`` is the unique entry key inside ``llm.model_list``;
    ``api_key``/``api_base`` support the ``env:VAR`` indirection syntax and
    are accepted-but-ignored for local providers (claude-agent/, copilot/,
    chatgpt/).
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    model: str
    api_key: str | None = None
    api_base: str | None = None

    @field_validator("model_name", "model")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("model_name", "model", "api_key", "api_base")
    @classmethod
    def _no_mask_char(cls, value: str | None) -> str | None:
        return _reject_mask_char(value)


class LLMModelUpdate(BaseModel):
    """Body of ``PUT /api/settings/llm/models/{model_name}``.

    Field semantics: omitted = keep the current value; for ``api_key`` and
    ``api_base`` an explicit ``null`` clears it and a new string replaces it.
    ``model`` cannot be cleared (an entry always has a model). Which fields
    were actually sent is read from ``model_fields_set``.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None

    @field_validator("model")
    @classmethod
    def _require_non_blank_model(cls, value: str | None) -> str:
        # Runs only when the field was provided: explicit null is rejected,
        # while an omitted field never reaches this validator.
        if value is None:
            raise ValueError("model cannot be null; omit the field to keep it")
        value = value.strip()
        if not value:
            raise ValueError("model must be a non-empty string")
        return value

    @field_validator("model", "api_key", "api_base")
    @classmethod
    def _no_mask_char(cls, value: str | None) -> str | None:
        return _reject_mask_char(value)
