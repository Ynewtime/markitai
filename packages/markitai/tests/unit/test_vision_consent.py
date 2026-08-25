"""Tests for the VLM-OCR privacy gate (``markitai.vision_consent``).

The gate is two things only: a one-time per-process disclosure naming the
vision model(s) page images go to, and the ``MARKITAI_NO_VLM_OCR`` hard
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
    def _emit(self, config: MarkitaiConfig, page_count: int | None = None) -> str:
        """Emit the disclosure and return the stderr text delivered once."""
        with patch("markitai.vision_consent.get_interaction") as mock_get:
            mock_port = mock_get.return_value
            ensure_vlm_ocr_disclosed(config, page_count=page_count)
            return mock_port.notify.call_args.args[0]

    def test_disclosed_once_per_process(self) -> None:
        config = _config_with_models(("claude-agent/sonnet", None))
        with patch("markitai.vision_consent.get_interaction") as mock_get:
            mock_port = mock_get.return_value
            ensure_vlm_ocr_disclosed(config, page_count=3)
            ensure_vlm_ocr_disclosed(config, page_count=9)
            mock_port.notify.assert_called_once()
            assert "3 page image(s)" in mock_port.notify.call_args.args[0]

    def test_names_the_vision_model_and_skips_non_vision(self) -> None:
        config = _config_with_models(
            ("claude-agent/sonnet", None),  # local provider → vision
            ("deepseek/deepseek-chat", False),  # explicit non-vision
        )
        msg = self._emit(config, page_count=1)
        assert "claude-agent/sonnet" in msg
        assert "deepseek/deepseek-chat" not in msg
        assert "MARKITAI_NO_VLM_OCR" in msg

    def test_explicit_supports_vision_is_honored(self) -> None:
        config = _config_with_models(("custom-vision", True))
        assert "custom-vision" in self._emit(config, page_count=2)

    def test_fallback_wording_without_models(self) -> None:
        msg = self._emit(MarkitaiConfig(), page_count=1)
        assert "your configured vision model" in msg

    def test_generic_wording_without_page_count(self) -> None:
        config = _config_with_models(("claude-agent/sonnet", None))
        msg = self._emit(config, page_count=None)
        assert "page images of scanned documents" in msg

    def test_reset_allows_re_disclosure(self) -> None:
        config = _config_with_models(("claude-agent/sonnet", None))
        with patch("markitai.vision_consent.get_interaction") as mock_get:
            mock_port = mock_get.return_value
            ensure_vlm_ocr_disclosed(config)
            assert mock_port.notify.call_count == 1
            reset_vlm_ocr_disclosure()
            ensure_vlm_ocr_disclosed(config)
            assert mock_port.notify.call_count == 2
        assert vlm_ocr_disclosure_emitted() is True
