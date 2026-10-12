"""Model entries markitai ships for models its pinned LiteLLM may not list.

A model missing from LiteLLM's cost table is priced as unknown: its requests
land in ``unpriced_requests``, the batch summary says "cost unknown", and
``max_cost_per_document_usd`` cannot count them. LiteLLM 1.104.2 (the newest
stable release on 2026-10-11) has no entry for ``claude-haiku-5-5`` (the
Anthropic default since 1.2.1), ``claude-sonnet-5-5`` or ``gpt-6.1-sol``,
nor for OpenRouter's ``anthropic/claude-haiku-5.5``.

The three first-party entries are in ``litellm_rows.json``: their rows from
LiteLLM 1.106.0.dev2's ``model_prices_and_context_window_backup.json``
(a development release, never a dependency): the same keys and values
markitai-rust prices from. Every rate, Batch and cache rates included,
equals the provider's published price on 2026-10-11:

- claude-haiku-5-5: https://platform.claude.com/docs/en/models/haiku-5-5/overview
  ($0.10 / $0.50 per MTok up to 100k prompt tokens, $0.50 / $2.50 above)
- claude-sonnet-5-5: https://platform.claude.com/docs/en/about-claude/pricing
  ($2 / $10 over the whole 1M context, cache reads $0.10, writes $2.50 / 1 h $4)
- gpt-6.1-sol: https://developers.openai.com/api/docs/pricing ($2 / $10, cached
  input $0.10, writes $2.50; a prompt over 272K bills the request at $4 / $15)

Batch is half of each. LiteLLM's licence (MIT, Berri AI) is in NOTICE.

Its entry for OpenRouter's moving alias ``~anthropic/claude-haiku-latest``,
the OpenRouter default, is kept as it is (Claude Haiku 4.5's limits and
prices): on 2026-10-10 OpenRouter's catalog named Claude Haiku 5.5 as the
alias's target, but requests to the alias were answered and billed as Claude
Haiku 4.5, so the smaller limits are the safe ones, and OpenRouter requests
are charged at the amount OpenRouter reports
(``markitai.llm.models.get_response_cost``). The copy here only fills in for
a LiteLLM that no longer lists the alias (upstream has dropped its OpenRouter
entries).

Each entry is registered only when the installed LiteLLM has no entry of that
name, so a LiteLLM release that adds the model wins and the copy here goes
dormant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_LITELLM_ROWS = Path(__file__).with_name("litellm_rows.json")

_MISSING_FROM_LITELLM: dict[str, dict[str, Any]] = {}

# https://openrouter.ai/api/v1/models on 2026-10-10: anthropic/claude-haiku-5.5
# (the target the catalog names for ~anthropic/claude-haiku-latest) lists $0.10 / $0.50
# per MTok ($0.50 / $2.50 for prompts from 100k tokens), cache reads $0.01,
# writes $0.125 (1 h: $0.20), 1M context, 128k output, image input, tools,
# tool_choice and response_format, and no sampling parameters.
_OPENROUTER_CLAUDE_HAIKU_5_5: dict[str, Any] = {
    "litellm_provider": "openrouter",
    "mode": "chat",
    "max_input_tokens": 1000000,
    "max_output_tokens": 128000,
    "max_tokens": 128000,
    "input_cost_per_token": 1e-07,  # nosec B105 - a price, not a credential
    "output_cost_per_token": 5e-07,
    "input_cost_per_token_above_100k_tokens": 5e-07,
    "output_cost_per_token_above_100k_tokens": 2.5e-06,
    "cache_read_input_token_cost": 1e-08,
    "cache_read_input_token_cost_above_100k_tokens": 5e-08,
    "cache_creation_input_token_cost": 1.25e-07,
    "cache_creation_input_token_cost_above_100k_tokens": 6.25e-07,
    "cache_creation_input_token_cost_above_1hr": 2e-07,
    "cache_creation_input_token_cost_above_1hr_above_100k_tokens": 1e-06,
    "supports_function_calling": True,
    "supports_pdf_input": True,
    "supports_prompt_caching": True,
    "supports_reasoning": True,
    "supports_response_schema": True,
    "supports_tool_choice": True,
    "supports_vision": True,
    "source": "https://openrouter.ai/api/v1/models",
}
_MISSING_FROM_LITELLM["openrouter/anthropic/claude-haiku-5.5"] = (
    _OPENROUTER_CLAUDE_HAIKU_5_5
)

# LiteLLM 1.104.2's own entry for the alias: Claude Haiku 4.5's figures, which
# is what the alias was served as on 2026-10-10 (see the module docstring).
_MISSING_FROM_LITELLM["openrouter/~anthropic/claude-haiku-latest"] = {
    "litellm_provider": "openrouter",
    "mode": "chat",
    "max_input_tokens": 200000,
    "max_output_tokens": 64000,
    "max_tokens": 64000,
    "input_cost_per_token": 1e-06,  # nosec B105 - a price, not a credential
    "output_cost_per_token": 5e-06,
    "cache_read_input_token_cost": 1e-07,
    "cache_creation_input_token_cost": 1.25e-06,
    "cache_creation_input_token_cost_above_1hr": 2e-06,
    "supports_function_calling": True,
    "supports_pdf_input": True,
    "supports_prompt_caching": True,
    "supports_reasoning": True,
    "supports_response_schema": True,
    "supports_tool_choice": True,
    "supports_vision": True,
    "source": "https://openrouter.ai/api/v1/models",
}

_registered = False


def register_missing_models() -> list[str]:
    """Add the entries the installed LiteLLM lacks; return the names added.

    Idempotent and cheap after the first call. Never replaces an entry
    LiteLLM already has.
    """
    global _registered
    if _registered:
        return []
    import litellm

    shipped = {
        **json.loads(_LITELLM_ROWS.read_text(encoding="utf-8")),
        **_MISSING_FROM_LITELLM,
    }
    missing = {
        name: dict(entry)
        for name, entry in shipped.items()
        if name not in litellm.model_cost
    }
    if missing:
        # register_model looks each name up first, and for a name LiteLLM
        # cannot place yet (gpt-6.1-sol) it prints "Provider List: ..." to
        # stdout, into --json output. markitai turns that off for good in
        # markitai.llm.processor, which may not be imported yet.
        quiet = litellm.suppress_debug_info
        litellm.suppress_debug_info = True
        try:
            litellm.register_model(missing)
        finally:
            litellm.suppress_debug_info = quiet
    _registered = True
    return sorted(missing)


# LiteLLM's model table, read without importing LiteLLM: markitai sets
# LITELLM_LOCAL_MODEL_COST_MAP, so this file is the table LiteLLM itself uses
_LITELLM_TABLE = "model_prices_and_context_window_backup.json"
_table: dict[str, dict[str, Any]] | None = None


def _model_table() -> dict[str, dict[str, Any]]:
    """LiteLLM's model rows plus markitai's own, without importing LiteLLM."""
    global _table
    if _table is None:
        import importlib.util

        rows: dict[str, dict[str, Any]] = {}
        spec = importlib.util.find_spec("litellm")
        for folder in getattr(spec, "submodule_search_locations", None) or []:
            path = Path(folder) / _LITELLM_TABLE
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
                break
            except (OSError, ValueError):
                continue
        shipped = {
            **json.loads(_LITELLM_ROWS.read_text(encoding="utf-8")),
            **_MISSING_FROM_LITELLM,
        }
        for name, entry in shipped.items():
            rows.setdefault(name, entry)
        _table = rows
    return _table


def model_row(model: str) -> dict[str, Any] | None:
    """The model table's row for *model* (``provider/id``), or None.

    LiteLLM's own lookup for a prefixed model: the full name, then the id
    alone when that row belongs to the same provider. For ``markitai doctor``,
    which asks what a configured model can do without importing LiteLLM (its
    import takes about 8 s on the Windows machine).
    """
    table = _model_table()
    row = table.get(model)
    if row is not None:
        return row
    provider, sep, name = model.partition("/")
    if not sep:
        return None
    row = table.get(name)
    if row is not None and str(row.get("litellm_provider", "")).startswith(provider):
        return row
    return None


def supports_vision(model: str) -> bool:
    """Whether the model table says *model* reads images (False when unknown)."""
    row = model_row(model)
    return bool(row.get("supports_vision")) if row else False
