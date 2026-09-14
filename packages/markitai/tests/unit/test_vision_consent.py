"""Tests for the VLM-OCR privacy gate (``markitai.vision_consent``).

The gate is two things only: a short persistent notice about sending
page images to vision models, and the ``MARKITAI_NO_VLM_OCR`` hard
opt-out. No interactive prompt (``--ocr --llm`` is already an explicit
double opt-in). These tests never call a real model.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from markitai.config import (
    LiteLLMParams,
    LLMConfig,
    MarkitaiConfig,
    ModelConfig,
    ModelInfo,
)
from markitai.vision_consent import (
    ensure_vlm_ocr_disclosed,
    reset_vlm_ocr_disclosure,
    vlm_ocr_allowed,
    vlm_ocr_disclosure_emitted,
)


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    """Each test starts with a clean, undecided disclosure state."""
    reset_vlm_ocr_disclosure()
    yield
    reset_vlm_ocr_disclosure()


def _config_with_models(*models: tuple[str, bool | None]) -> MarkitaiConfig:
    """Build a config whose ``llm.model_list`` mirrors (model_id, supports_vision)."""
    return MarkitaiConfig(
        llm=LLMConfig(
            model_list=[
                ModelConfig(
                    model_name=f"group-{i}",
                    litellm_params=LiteLLMParams(model=model_id),
                    model_info=(
                        ModelInfo(supports_vision=supports)
                        if supports is not None
                        else None
                    ),
                )
                for i, (model_id, supports) in enumerate(models)
            ]
        )
    )


class TestVlmOcrAllowed:
    def test_default_allowed(self) -> None:
        assert vlm_ocr_allowed() is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "anything"])
    def test_env_truthy_blocks(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("MARKITAI_NO_VLM_OCR", value)
        assert vlm_ocr_allowed() is False

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "FALSE"])
    def test_env_falsy_allows(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        if value:
            monkeypatch.setenv("MARKITAI_NO_VLM_OCR", value)
        else:
            monkeypatch.delenv("MARKITAI_NO_VLM_OCR", raising=False)
        assert vlm_ocr_allowed() is True


class TestDisclosure:
    def test_short_notice_once_across_process_state_resets(self):
        config = _config_with_models(("claude-agent/sonnet", None))
        with patch("markitai.vision_consent.get_interaction") as interaction:
            port = interaction.return_value
            ensure_vlm_ocr_disclosed(config, page_count=3)
            ensure_vlm_ocr_disclosed(config, page_count=9)
            reset_vlm_ocr_disclosure()
            ensure_vlm_ocr_disclosed(config)
        port.notify.assert_called_once()
        message = port.notify.call_args.args[0]
        assert "Page images" in message
        assert "MARKITAI_NO_VLM_OCR=1" in message
        assert len(message) <= 80
        assert "\n" not in message
        assert vlm_ocr_disclosure_emitted() is True

    def test_notice_does_not_change_opt_out(self, monkeypatch):
        with patch("markitai.vision_consent.get_interaction"):
            ensure_vlm_ocr_disclosed(MarkitaiConfig())
        monkeypatch.setenv("MARKITAI_NO_VLM_OCR", "1")
        assert vlm_ocr_allowed() is False
