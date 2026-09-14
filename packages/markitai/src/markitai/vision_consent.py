"""VLM-OCR privacy gate: one-time disclosure + hard env opt-out.

``--ocr --llm`` sends *document page images* to the configured vision
model — strictly more sensitive than the extracted text the normal LLM
path sees, and more sensitive than sending a public URL to a remote
extraction service. Unlike :mod:`markitai.fetch_consent` there is
deliberately no "ask" prompt: the user opted in explicitly by passing both
``--ocr`` and ``--llm``, so a blocking question would be redundant friction
(there, the prompt exists because remote strategies fire implicitly inside
the auto chain). The gate is two things only:

* a short notice, shown once per OS user across runs on stderr
  (the same privacy-boundary rule as ``disclose_remote_use``);
* ``MARKITAI_NO_VLM_OCR`` as a hard opt-out: when set truthy, the VLM-OCR
  path never sends page images — it degrades to local RapidOCR when
  installed, or fails with an actionable error.

The process-local flag avoids repeated disk checks. The notice marker is
persisted under ~/.markitai/notices; neither records user authorization.
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


def ensure_vlm_ocr_disclosed(
    config: MarkitaiConfig | None, page_count: int | None = None
) -> None:
    """Emit the VLM-OCR notice once per user across CLI runs.

    Delivered to stderr via the interaction port so ``--quiet`` cannot hide
    it. Call at the point where the VLM-OCR path is actually taken — before
    page images are handed to the vision model.
    """
    if _state.disclosure_emitted:
        return
    from markitai.notices import notify_once

    disclosure = (
        "[VLM OCR] Page images go to your vision model. Disable: MARKITAI_NO_VLM_OCR=1."
    )
    notify_once("vlm-ocr", disclosure, get_interaction().notify)
    logger.debug("[VLM OCR] Sending {} page image(s)", page_count or "document")
    _state.disclosure_emitted = True
