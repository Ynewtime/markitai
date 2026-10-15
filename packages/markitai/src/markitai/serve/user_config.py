"""A user token's Edit config (workbench contract v1.16, section 8).

``GET``/``PUT /api/me/config`` show and replace a user's own models file as
one JSON document the server writes from its lists, never the file's bytes
and never the server's ``config.json``: ``models`` (its ``llm.model_list``
entries as stored), ``providers`` (its connections) and ``shared`` (each model
the admin shares and users see, with the user's weight for it). Secrets are
masked and kept masks restored by the admin editor's own functions
(:mod:`markitai.serve.config_editor`); a saved document goes through the
rules the Models routes apply, and its revision is the Models page's.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from markitai.serve import config_editor, user_models
from markitai.serve.errors import ApiError

if TYPE_CHECKING:
    from markitai.serve.user_models import Lists, Shared


def refused(detail: str) -> ApiError:
    return ApiError(422, "invalid_config", detail)


def document(lists: Lists, shared: list[Shared]) -> str:
    """The text of *lists*: two-space JSON, LF, a final line break, keys
    ``models``, ``providers``, ``shared`` in that order."""
    return (
        json.dumps(
            {
                "models": lists.models,
                "providers": lists.providers,
                "shared": [
                    {
                        "id": model.id,
                        "model": model.model,
                        "weight": user_models.effective_weight(lists, model),
                    }
                    for model in shared
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def view(
    lists: Lists, shared: list[Shared], *, exists: bool, revision: str
) -> dict[str, Any]:
    """The ``ConfigDocument`` of *lists*, every secret masked."""
    text, masked = config_editor.mask(document(lists, shared))
    return {
        "path": None,
        "origin": "user",
        "exists": exists,
        "revision": revision,
        "text": text,
        "masked": masked,
        "problem": None,
        "read_only": False,
    }


def _objects(fields: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The list of objects under *key*, empty when absent."""
    items = fields.get(key)
    if items is None:
        return []
    if not isinstance(items, list):
        raise refused(f"{key} must be a list")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise refused(f"{key}/{index} must be an object")
    return items


def parse(value: Any, current: Lists, shared: list[Shared]) -> Lists:
    """The lists the parsed document *value* describes, beside the user's
    *current* ones and the models the admin shares now (*shared*)."""
    if not isinstance(value, dict):
        raise refused("the configuration must be a JSON object")
    for key in value:
        if key not in ("models", "providers", "shared"):
            raise refused(
                f"{key} is not part of your models file; it holds only models, "
                "providers and shared"
            )
    models = _objects(value, "models")
    for index, entry in enumerate(models):
        params = entry.get("litellm_params")
        if not isinstance(params, dict) or not isinstance(params.get("model"), str):
            raise refused(f"models/{index} needs litellm_params.model, the model ID")
    user_models.check_weights(
        models,
        current.models,
        lambda index, entry: f"models/{index} ({entry['litellm_params']['model']})",
    )
    providers = _objects(value, "providers")
    for index, connection in enumerate(providers):
        if not isinstance(connection.get("id"), str) or not isinstance(
            connection.get("provider"), str
        ):
            raise refused(f"providers/{index} needs an id and a provider")
    ids = {connection["id"] for connection in providers}
    for index, entry in enumerate(models):
        info = entry.get("model_info")
        link = info.get("provider_id") if isinstance(info, dict) else None
        if link is not None and link not in ids:
            raise refused(
                f"models/{index}: its model_info.provider_id is not one of the "
                "providers"
            )
    listed = {model.id for model in shared}
    # Weights for models no longer shared stay, for when they are again
    weights = {
        key: weight
        for key, weight in current.shared_weights.items()
        if key not in listed
    }
    result = user_models.Lists(
        models=models,
        providers=providers,
        shared_weights=weights,
        convert_defaults=current.convert_defaults,
    )
    seen: set[str] = set()
    for index, entry in enumerate(_objects(value, "shared")):
        for key in entry:
            if key not in ("id", "weight", "model", "default_weight"):
                raise refused(
                    f"shared/{index}: {key} is not a field of a shared model; it "
                    "takes id and weight"
                )
        shared_id = entry.get("id")
        if not isinstance(shared_id, str):
            raise refused(f"shared/{index} needs its id")
        weight = entry.get("weight")
        model = next((model for model in shared if model.id == shared_id), None)
        if model is None:
            raise refused(f"{shared_id} is not a model the admin shares")
        if shared_id in seen:
            raise refused(f"shared lists {shared_id} twice")
        seen.add(shared_id)
        # A weight the user already has (one saved before the limit) may stay
        unchanged = user_models.valid_weight(weight) and weight == (
            user_models.effective_weight(current, model)
        )
        if (
            weight is not None
            and not unchanged
            and not user_models.routing_weight(weight)
        ):
            raise user_models.invalid_weight(f"shared/{index} ({shared_id})")
        # An unchanged weight keeps what the user had; another one follows
        # the rule of the Models page
        if weight == user_models.effective_weight(current, model):
            if shared_id in current.shared_weights:
                result.shared_weights[shared_id] = current.shared_weights[shared_id]
        else:
            user_models.set_weight(result, model, weight)
    return result


def unique(name: str, ids: list[Any]) -> None:
    """Refuse two entries of one list with the same id."""
    seen: set[str] = set()
    for value in ids:
        if not isinstance(value, str):
            continue
        if value in seen:
            raise refused(f"{name}: the id {value} is used twice")
        seen.add(value)
