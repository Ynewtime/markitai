"""What a failed LLM enhancement does to its item (``llm.on_failure``).

A failed enhancement (invalid key, provider error, timeout, refusal, a
tripped request budget) always leaves the unenhanced ``.md`` on disk. Under
``"fallback"`` (the default) that file is the item's output and the item
succeeds with a warning; under ``"fail"`` the item fails. Every entry point
(single file and URL, both batches, ``--llm-batch``, serve, the Python API
and MCP) decides through this module, so the policy cannot drift between
them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from markitai.notices import user_notice

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig

# Prefixes the pipelines put in front of an LLM error; the warning names the
# cause once instead of repeating "LLM ... failed" twice.
_ERROR_PREFIXES = (
    "Pure LLM processing failed: ",
    "LLM processing failed: ",
    "LLM enhancement failed: ",
    "Vision LLM failed: ",
)

_hinted = False


def llm_failure_fails_item(cfg: MarkitaiConfig) -> bool:
    """Whether a failed LLM enhancement fails the item (``on_failure="fail"``)."""
    return cfg.llm.on_failure == "fail"


def llm_fallback_warning(source: str, error: str) -> str:
    """Report an LLM failure the item survives; return the item's warning.

    The warning is also raised as a user notice (stderr in default mode, the
    batch summary, serve/API/MCP warnings). The first one of a process adds
    how to make such items fail instead.

    Args:
        source: The file name or URL, as the user knows it.
        error: The failure as the pipeline reported it.
    """
    global _hinted
    reason = error
    for prefix in _ERROR_PREFIXES:
        reason = reason.removeprefix(prefix)
    warning = f"LLM enhancement failed ({reason}); kept the unenhanced output"
    user_notice("{}: {}", source, warning)
    if not _hinted:
        _hinted = True
        user_notice(
            "Items whose LLM enhancement fails keep their unenhanced output; "
            'set llm.on_failure to "fail" to fail them instead.'
        )
    return warning
