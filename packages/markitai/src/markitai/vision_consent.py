"""VLM-OCR privacy gate: one-time disclosure + hard env opt-out.

``--ocr --llm`` sends *document page images* to the configured vision
model — strictly more sensitive than the extracted text the normal LLM
path sees, and more sensitive than sending a public URL to a remote
extraction service. Unlike :mod:`markitai.fetch_consent` there is
deliberately no "ask" prompt: the user opted in explicitly by passing both
``--ocr`` and ``--llm``, so a blocking question would be redundant friction
(there, the prompt exists because remote strategies fire implicitly inside
the auto chain). The gate is two things only:

* a one-time, per-process disclosure naming the vision model(s) the page
  images will be sent to, delivered to stderr so ``--quiet`` cannot hide it
  (the same privacy-boundary rule as ``disclose_remote_use``);
* ``MARKITAI_NO_VLM_OCR`` as a hard opt-out: when set truthy, the VLM-OCR
  path never sends page images — it degrades to local RapidOCR when
  installed, or fails with an actionable error.

State is process-wide and module-local (there is no session to attach it to
in the converter layer). Tests reset it with :func:`reset_vlm_ocr_disclosure`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from markitai.ports import get_interaction

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig


class _VlmOcrConsentState:
    """Process-wide VLM-OCR disclosure state."""

    disclosure_emitted: bool = False


_state = _VlmOcrConsentState()


def _env_no_vlm_ocr() -> bool:
    """Return True when MARKITAI_NO_VLM_OCR is set to a truthy value."""
    value = os.environ.get("MARKITAI_NO_VLM_OCR", "").strip().lower()
    return value not in ("", "0", "false", "no")


def reset_vlm_ocr_disclosure() -> None:
    """Reset the cached disclosure flag (mainly for tests)."""
    _state.disclosure_emitted = False


def vlm_ocr_disclosure_emitted() -> bool:
    """Return whether the VLM-OCR disclosure has been emitted this process."""
    return _state.disclosure_emitted


def vlm_ocr_allowed() -> bool:
    """Return False when ``MARKITAI_NO_VLM_OCR`` blocks sending page images.

    Callers must never hand page images to a remote model when this returns
    False: degrade to local OCR (when installed) or fail with a clear error.
    """
    return not _env_no_vlm_ocr()


def _vision_model_names(config: MarkitaiConfig | None) -> list[str]:
    """Best-effort names of vision-capable configured models.

    Mirrors the vision router's capability rule: an explicit
    ``model_info.supports_vision`` wins, local provider models are assumed
    vision-capable, and standard models are auto-detected from the litellm
    registry. Auto-detection is deferred (it pulls litellm) and only runs
    when the LLM path is already active, i.e. at disclosure time.
    """
    if config is None:
        return []
    llm_cfg = getattr(config, "llm", None)
    if llm_cfg is None or not getattr(llm_cfg, "model_list", None):
        return []

    from markitai.llm.models import get_model_info_cached
    from markitai.providers import is_local_provider_model

    names: list[str] = []
    for model in llm_cfg.model_list:
        model_id = getattr(getattr(model, "litellm_params", None), "model", "") or ""
        if not model_id:
            continue
        info = getattr(model, "model_info", None)
        explicit = getattr(info, "supports_vision", None) if info else None
        if explicit is True:
            vision = True
        elif explicit is None:
            vision = is_local_provider_model(model_id) or bool(
                get_model_info_cached(model_id).get("supports_vision", False)
            )
        else:
            vision = False
        if vision:
            names.append(model_id)
    return names


def ensure_vlm_ocr_disclosed(
    config: MarkitaiConfig | None, page_count: int | None = None
) -> None:
    """Emit the one-time VLM-OCR privacy disclosure (once per process).

    Delivered to stderr via the interaction port so ``--quiet`` cannot hide
    it. Call at the point where the VLM-OCR path is actually taken — before
    page images are handed to the vision model.
    """
    if _state.disclosure_emitted:
        return
    model_names = _vision_model_names(config)
    label = ", ".join(model_names) if model_names else "your configured vision model"
    subject = f"{page_count} page image(s)" if page_count else "page images"
    disclosure = (
        "[VLM OCR] This run will send "
        f"{subject} of scanned documents to the vision LLM ({label}) for OCR "
        "reading. Set MARKITAI_NO_VLM_OCR=1 to use local RapidOCR instead."
    )
    # Privacy boundary, not diagnostics: deliver via the interaction port so
    # normal log filtering and --quiet cannot hide it.
    get_interaction().notify(disclosure)
    logger.info(disclosure)
    _state.disclosure_emitted = True
