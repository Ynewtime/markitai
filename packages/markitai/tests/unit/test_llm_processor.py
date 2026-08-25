"""Unit tests for LLMProcessor utility methods and cache classes.

Focuses on non-async utility methods to increase coverage:
- Initialization and configuration
- Cache management (SQLiteCache, PersistentCache, ContentCache)
- Usage tracking methods
- Helper methods (_get_cached_image, _calculate_dynamic_max_tokens)
- Format output methods
- Router creation logic
- Module-level helper functions
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import (
    LiteLLMParams,
    LLMConfig,
    ModelConfig,
    PromptsConfig,
)
from markitai.llm.cache import ContentCache, PersistentCache, SQLiteCache
from markitai.llm.models import (
    context_display_name,
    get_model_info_cached,
    get_model_max_output_tokens,
    get_response_cost,
)
from markitai.llm.processor import LLMProcessor
from markitai.llm.router import (
    STANDARD_POOL_ID,
    MarkitaiRouter,
    RouterCandidate,
    cooldown_seconds_for_error,
    select_weighted_model,
)

# =============================================================================
# Test Module-Level Helper Functions
# =============================================================================


class TestContextDisplayName:
    """Tests for context_display_name function (from models.py)."""

    def test_empty_context(self):
        """Test empty context returns empty."""
        assert context_display_name("") == ""

    def test_simple_filename(self):
        """Test simple filename."""
        assert context_display_name("file.pdf") == "file.pdf"

    def test_unix_path(self):
        """Test Unix path extracts filename."""
        assert context_display_name("/path/to/file.pdf") == "file.pdf"

    def test_windows_style_path_with_forward_slashes(self):
        """Test Windows-style path with forward slashes (normalized).

        Note: On Linux, Path("C:/path/to/file.pdf") returns "C:/path/to/file.pdf"
        because C: is not recognized as a drive letter. The function preserves
        this behavior since it uses Path().name which works correctly on Windows.
        On Windows, this would return "file.pdf".
        """
        import platform

        result = context_display_name("C:/path/to/file.pdf")
        if platform.system() == "Windows":
            assert result == "file.pdf"
        else:
            # On Linux, C: is not recognized as drive letter, so split on ":"
            assert "file.pdf" in result

    def test_path_with_suffix(self):
        """Test path with :suffix preserves suffix."""
        result = context_display_name("/path/to/file.pdf:images")
        assert result == "file.pdf:images"

    def test_windows_path_with_suffix_forward_slash(self):
        """Test Windows path with suffix using forward slashes."""
        result = context_display_name("C:/path/to/file.pdf:images")
        assert result == "file.pdf:images"

    def test_relative_path(self):
        """Test relative path extracts filename."""
        assert context_display_name("subdir/file.pdf") == "file.pdf"


class TestGetModelInfoCached:
    """Tests for get_model_info_cached function."""

    def test_returns_cached_value(self):
        """Test that cached values are returned."""
        from markitai.llm import models

        # Clear cache first
        models._model_info_cache.clear()

        # First call - should query litellm
        with (
            patch(
                "markitai.providers.get_local_provider_model_info", return_value=None
            ),
            patch("litellm.get_model_info") as mock_get_info,
        ):
            mock_get_info.return_value = {
                "max_input_tokens": 100000,
                "max_output_tokens": 8000,
                "supports_vision": True,
            }
            result1 = get_model_info_cached("test/model")
            assert mock_get_info.called

        # Second call - should use cache
        with patch("litellm.get_model_info") as mock_get_info:
            result2 = get_model_info_cached("test/model")
            assert not mock_get_info.called
            assert result2 == result1

        # Clean up
        models._model_info_cache.clear()

    def test_local_provider_model(self):
        """Test local provider model info."""
        from markitai.llm import models

        models._model_info_cache.clear()

        local_info = {
            "max_input_tokens": 200000,
            "max_output_tokens": 16000,
            "supports_vision": True,
        }

        with patch(
            "markitai.providers.get_local_provider_model_info",
            return_value=local_info,
        ):
            result = get_model_info_cached("claude-agent/sonnet")
            assert result["max_input_tokens"] == 200000
            assert result["max_output_tokens"] == 16000
            assert result["supports_vision"] is True

        models._model_info_cache.clear()

    def test_fallback_on_exception(self):
        """Test defaults are returned when litellm fails."""
        from markitai.llm import models

        models._model_info_cache.clear()

        with (
            patch(
                "markitai.providers.get_local_provider_model_info", return_value=None
            ),
            patch("litellm.get_model_info", side_effect=Exception("API Error")),
        ):
            result = get_model_info_cached("unknown/model")
            assert result["max_input_tokens"] == 128000
            assert result["supports_vision"] is False

        models._model_info_cache.clear()


class TestGetModelMaxOutputTokens:
    """Tests for get_model_max_output_tokens function."""

    def test_returns_max_output_tokens(self):
        """Test returns max_output_tokens from model info."""
        with patch(
            "markitai.llm.models.get_model_info_cached",
            return_value={"max_output_tokens": 16384},
        ):
            assert get_model_max_output_tokens("test/model") == 16384


class TestGetResponseCost:
    """Tests for get_response_cost function."""

    def test_cost_from_hidden_params(self):
        """Test cost is extracted from _hidden_params."""
        response = MagicMock()
        response._hidden_params = {"total_cost_usd": 0.0025}

        cost = get_response_cost(response)
        assert cost == 0.0025

    def test_cost_from_litellm_completion_cost(self):
        """Test fallback to litellm.completion_cost."""
        response = MagicMock()
        response._hidden_params = {}

        with patch("markitai.llm.models.completion_cost", return_value=0.005):
            cost = get_response_cost(response)
            assert cost == 0.005

    def test_cost_zero_on_exception(self):
        """Test returns 0.0 when cost calculation fails."""
        response = MagicMock()
        response._hidden_params = {}

        with patch(
            "markitai.llm.models.completion_cost", side_effect=Exception("Error")
        ):
            cost = get_response_cost(response)
            assert cost == 0.0

    def test_no_hidden_params(self):
        """Test when _hidden_params is None."""
        response = MagicMock(spec=[])  # No _hidden_params attribute

        with patch("markitai.llm.models.completion_cost", return_value=0.003):
            cost = get_response_cost(response)
            assert cost == 0.003


# =============================================================================
# Test select_weighted_model (pure selection function)
# =============================================================================


def _cand(model_id: str, weight: float = 1.0, image_capable: bool = True):
    return RouterCandidate(
        model_id=model_id, weight=weight, image_capable=image_capable
    )


class TestSelectWeightedModel:
    """Tests for the pure weighted selection function."""

    def test_empty_returns_none(self):
        """Empty candidate list yields None."""
        assert select_weighted_model([], cooldowns={}, now=0.0) is None

    def test_single_candidate(self):
        """A single candidate is always selected."""
        selected = select_weighted_model(
            [_cand("claude-agent/sonnet")], cooldowns={}, now=0.0
        )
        assert selected == "claude-agent/sonnet"

    def test_skips_zero_weight(self):
        """weight=0 models should never be selected when others have weight > 0."""
        candidates = [
            _cand("claude-agent/haiku", weight=0),
            _cand("copilot/gpt-5", weight=0),
            _cand("chatgpt/gpt-5.3", weight=20),
        ]
        for _ in range(50):
            assert (
                select_weighted_model(candidates, cooldowns={}, now=0.0)
                == "chatgpt/gpt-5.3"
            )

    def test_all_zero_weight_still_selects(self):
        """If all weights are 0, selection falls back to uniform random."""
        candidates = [
            _cand("claude-agent/haiku", weight=0),
            _cand("copilot/gpt-5", weight=0),
        ]
        selected = select_weighted_model(candidates, cooldowns={}, now=0.0)
        assert selected in ("claude-agent/haiku", "copilot/gpt-5")

    def test_skips_cooldown_model(self):
        """Models in cooldown should be skipped during selection."""
        candidates = [
            _cand("claude-agent/sonnet", weight=10),
            _cand("copilot/gemini-3-flash", weight=10),
        ]
        cooldowns = {"copilot/gemini-3-flash": 100.0}
        selections = {
            select_weighted_model(candidates, cooldowns=cooldowns, now=50.0)
            for _ in range(50)
        }
        assert selections == {"claude-agent/sonnet"}

    def test_routes_after_cooldown_expires(self):
        """Models should be routable again after cooldown expires."""
        candidates = [
            _cand("claude-agent/sonnet", weight=10),
            _cand("copilot/gemini-3-flash", weight=10),
        ]
        cooldowns = {"copilot/gemini-3-flash": 100.0}
        selections = {
            select_weighted_model(candidates, cooldowns=cooldowns, now=101.0)
            for _ in range(100)
        }
        assert len(selections) == 2

    def test_picks_soonest_expiring_when_all_in_cooldown(self):
        """When all models are in cooldown, pick the one expiring soonest."""
        candidates = [
            _cand("model-a", weight=10),
            _cand("model-b", weight=10),
        ]
        cooldowns = {"model-a": 120.0, "model-b": 10.0}
        assert (
            select_weighted_model(candidates, cooldowns=cooldowns, now=0.0) == "model-b"
        )

    def test_image_request_prefers_image_capable(self):
        """Vision requests exclude non-image-capable candidates."""
        candidates = [
            _cand("copilot/grok-1", weight=10, image_capable=False),
            _cand("claude-agent/sonnet", weight=10, image_capable=True),
        ]
        selections = {
            select_weighted_model(
                candidates, cooldowns={}, now=0.0, prefer_image_capable=True
            )
            for _ in range(50)
        }
        assert selections == {"claude-agent/sonnet"}

    def test_image_request_without_capable_candidates_proceeds(self):
        """With no image-capable candidate, all candidates stay eligible."""
        candidates = [
            _cand("copilot/grok-1", weight=10, image_capable=False),
            _cand("copilot/grok-2", weight=10, image_capable=False),
        ]
        selected = select_weighted_model(
            candidates, cooldowns={}, now=0.0, prefer_image_capable=True
        )
        assert selected in ("copilot/grok-1", "copilot/grok-2")


# =============================================================================
# Test cooldown_seconds_for_error (unified error classification)
# =============================================================================


class TestCooldownSecondsForError:
    """Tests for the single error-to-cooldown classification."""

    def test_rate_limit_default(self):
        """Rate-limit text without a retry hint uses the 60s default."""
        assert cooldown_seconds_for_error("429 Too Many Requests") == 60.0

    def test_rate_limit_with_retry_after(self):
        """A "retry in Ns" hint overrides the default cooldown."""
        assert (
            cooldown_seconds_for_error("Rate limit: quota will reset after 30s") == 30.0
        )

    def test_model_level_error(self):
        """Model-level errors get the long cooldown."""
        assert (
            cooldown_seconds_for_error("User location is not supported for the API")
            == 3600.0
        )

    def test_model_level_wins_over_rate_limit(self):
        """When both match, the model-level (long) cooldown wins."""
        assert (
            cooldown_seconds_for_error("429 quota: model is not available in 10s")
            == 3600.0
        )

    def test_content_error_no_cooldown(self):
        """Content-specific errors say nothing about routability."""
        assert cooldown_seconds_for_error("Invalid request: content too long") is None


# =============================================================================
# Test MarkitaiRouter — local provider pool
# =============================================================================


def _local_entry(model: str, weight: float = 1.0) -> dict:
    return {
        "model_name": "default",
        "litellm_params": {"model": model, "weight": weight},
    }


def _standard_entry(model: str, weight: float = 1.0) -> dict:
    return {
        "model_name": "default",
        "litellm_params": {"model": model, "weight": weight, "api_key": "test-key"},
    }


class TestMarkitaiRouterLocal:
    """Tests for MarkitaiRouter with only local provider models."""

    def test_init_builds_local_group(self):
        """Local entries land in the default selection group, no LiteLLM Router."""
        router = MarkitaiRouter(
            [
                _local_entry("claude-agent/sonnet", 2.0),
                _local_entry("claude-agent/haiku", 1.0),
            ]
        )
        assert router._standard_router is None
        assert len(router._groups["default"]) == 2

    def test_model_list_property(self):
        """model_list returns the original entries."""
        entries = [_local_entry("claude-agent/sonnet")]
        router = MarkitaiRouter(entries)
        assert router.model_list == entries

    def test_is_image_capable_patterns(self):
        """Image capability follows the local provider pattern table."""
        router = MarkitaiRouter([])
        assert router._is_image_capable_local("claude-agent/sonnet") is True
        assert router._is_image_capable_local("copilot/claude-sonnet-4") is True
        assert router._is_image_capable_local("copilot/gpt-4o-mini") is True
        assert router._is_image_capable_local("chatgpt/gpt-5.3") is True
        assert router._is_image_capable_local("copilot/gpt-3.5-turbo") is False
        assert router._is_image_capable_local("copilot/grok-3") is False

    def test_select_unknown_name_passes_through(self):
        """Unknown model names pass through unchanged (concrete-id calls)."""
        router = MarkitaiRouter([])
        assert router._select("unknown-model", False) == "unknown-model"

    @pytest.mark.asyncio
    async def test_acompletion_calls_registered_handler_directly(self):
        """Should call registered custom handler directly, bypassing litellm.acompletion.

        This is critical for providers like chatgpt/ where LiteLLM has a native
        handler that would take priority over custom_provider_map entries.
        """
        mock_handler = AsyncMock()
        mock_response = MagicMock()
        mock_handler.acompletion.return_value = mock_response

        router = MarkitaiRouter([_local_entry("chatgpt/gpt-5.3")])
        messages = [{"role": "user", "content": "Hello"}]

        with patch("markitai.providers.get_provider", return_value=mock_handler):
            result = await router.acompletion("default", messages)

        assert result is mock_response
        mock_handler.acompletion.assert_called_once_with(
            model="chatgpt/gpt-5.3",
            messages=messages,
        )

    @pytest.mark.asyncio
    async def test_acompletion_falls_back_to_litellm_when_no_handler(self):
        """Should fall back to litellm.acompletion when no handler is registered."""
        mock_response = MagicMock()
        messages = [{"role": "user", "content": "Hello"}]

        with (
            patch("markitai.providers.is_local_provider_model", return_value=True),
            patch("markitai.providers.get_provider", return_value=None),
            patch("markitai.llm.router.litellm") as mock_litellm,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            router = MarkitaiRouter([_local_entry("some-provider/model")])
            result = await router.acompletion("default", messages)

        assert result is mock_response

    @pytest.mark.asyncio
    async def test_acompletion_strips_metadata_before_handler_call(self):
        """Should strip metadata kwarg before calling the handler."""
        mock_handler = AsyncMock()
        mock_handler.acompletion.return_value = MagicMock()

        router = MarkitaiRouter([_local_entry("chatgpt/gpt-5.3")])
        messages = [{"role": "user", "content": "Hello"}]

        with patch("markitai.providers.get_provider", return_value=mock_handler):
            await router.acompletion(
                "default", messages, metadata={"key": "val"}, max_tokens=1000
            )

        # metadata should be stripped, max_tokens preserved
        call_kwargs = mock_handler.acompletion.call_args
        assert "metadata" not in call_kwargs.kwargs
        assert call_kwargs.kwargs["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_rate_limit_error_records_cooldown(self):
        """A rate-limit failure puts the local model in cooldown."""
        mock_handler = AsyncMock()
        mock_handler.acompletion.side_effect = RuntimeError(
            "429 rate limit: retry after 30s"
        )

        router = MarkitaiRouter([_local_entry("claude-agent/sonnet")])

        with (
            patch("markitai.providers.get_provider", return_value=mock_handler),
            pytest.raises(RuntimeError),
        ):
            await router.acompletion("default", [{"role": "user", "content": "Hi"}])

        remaining = router._cooldowns["claude-agent/sonnet"] - time.monotonic()
        assert 25 < remaining <= 30

    @pytest.mark.asyncio
    async def test_model_level_error_records_long_cooldown(self):
        """Model-level errors put the local model in a 3600s cooldown."""
        mock_handler = AsyncMock()
        mock_handler.acompletion.side_effect = RuntimeError(
            "User location is not supported for the API use."
        )

        router = MarkitaiRouter([_local_entry("claude-agent/sonnet")])

        with (
            patch("markitai.providers.get_provider", return_value=mock_handler),
            pytest.raises(RuntimeError),
        ):
            await router.acompletion("default", [{"role": "user", "content": "Hi"}])

        remaining = router._cooldowns["claude-agent/sonnet"] - time.monotonic()
        assert remaining > 3500

    @pytest.mark.asyncio
    async def test_regular_error_records_no_cooldown(self):
        """Content-specific errors do not put the model in cooldown."""
        mock_handler = AsyncMock()
        mock_handler.acompletion.side_effect = RuntimeError(
            "Invalid request: content too long"
        )

        router = MarkitaiRouter([_local_entry("claude-agent/sonnet")])

        with (
            patch("markitai.providers.get_provider", return_value=mock_handler),
            pytest.raises(RuntimeError),
        ):
            await router.acompletion("default", [{"role": "user", "content": "Hi"}])

        assert "claude-agent/sonnet" not in router._cooldowns

    def test_concurrent_cooldown_read_write(self):
        """Cooldown map handles concurrent access safely.

        Defensive against Python 3.13+ free-threaded mode (PEP 703).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        router = MarkitaiRouter(
            [_local_entry(f"claude-agent/model-{i}", 10) for i in range(5)]
        )

        errors: list[str] = []

        def write_cooldowns() -> None:
            for i in range(200):
                router.record_cooldown(f"claude-agent/model-{i % 5}", float(i % 10))

        def read_cooldowns() -> None:
            for _ in range(200):
                try:
                    router._select("default", False)
                except Exception as e:
                    errors.append(str(e))

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(write_cooldowns) for _ in range(3)] + [
                pool.submit(read_cooldowns) for _ in range(3)
            ]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Concurrent cooldown errors: {errors}"


# =============================================================================
# Test MarkitaiRouter — mixed local + standard pool
# =============================================================================


class TestAnthropicCacheBreakpoint:
    """Anthropic prompt-caching injection at the router's unified exits."""

    long_system = "You are a document processing assistant. " * 200  # >4096 chars

    def test_breakpoint_marks_long_system_only(self):
        from markitai.llm.router import _anthropic_cache_breakpoint

        messages = [
            {"role": "system", "content": self.long_system},
            {"role": "system", "content": "short"},
            {"role": "user", "content": "x" * 9000},
        ]
        out = _anthropic_cache_breakpoint(messages)

        assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert out[0]["content"][0]["text"] == self.long_system
        assert out[1] is messages[1]  # short system untouched
        assert out[2] is messages[2]  # user messages never marked

    def test_breakpoint_leaves_block_form_alone(self):
        from markitai.llm.router import _anthropic_cache_breakpoint

        block_msg = {
            "role": "system",
            "content": [{"type": "text", "text": self.long_system}],
        }
        assert _anthropic_cache_breakpoint([block_msg])[0] is block_msg

    @pytest.mark.asyncio
    async def test_standard_pool_all_anthropic_injects(self):
        router = MarkitaiRouter([_standard_entry("anthropic/claude-sonnet-4")])
        assert router._standard_pool_all_anthropic is True
        assert router._standard_router is not None
        router._standard_router.acompletion = AsyncMock(return_value=MagicMock())

        messages = [
            {"role": "system", "content": self.long_system},
            {"role": "user", "content": "doc"},
        ]
        await router._standard_acompletion("default", messages)

        sent = router._standard_router.acompletion.call_args.args[1]
        assert sent[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert sent[1] is messages[1]

    @pytest.mark.asyncio
    async def test_mixed_standard_pool_does_not_inject(self):
        router = MarkitaiRouter(
            [
                _standard_entry("anthropic/claude-sonnet-4"),
                _standard_entry("openai/gpt-5"),
            ]
        )
        assert router._standard_pool_all_anthropic is False
        assert router._standard_router is not None
        router._standard_router.acompletion = AsyncMock(return_value=MagicMock())

        messages = [{"role": "system", "content": self.long_system}]
        await router._standard_acompletion("default", messages)

        sent = router._standard_router.acompletion.call_args.args[1]
        assert sent[0] is messages[0]

    @pytest.mark.asyncio
    async def test_local_bare_litellm_anthropic_injects(self):
        with (
            patch("markitai.providers.get_provider", return_value=None),
            patch("markitai.llm.router.litellm") as mock_litellm,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=MagicMock())
            router = MarkitaiRouter([])
            messages = [{"role": "system", "content": self.long_system}]
            await router._local_acompletion("anthropic/claude-haiku", messages)

        sent = mock_litellm.acompletion.call_args.kwargs["messages"]
        assert sent[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_local_bare_litellm_non_anthropic_skips(self):
        with (
            patch("markitai.providers.get_provider", return_value=None),
            patch("markitai.llm.router.litellm") as mock_litellm,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=MagicMock())
            router = MarkitaiRouter([])
            messages = [{"role": "system", "content": self.long_system}]
            await router._local_acompletion("openai/gpt-5", messages)

        sent = mock_litellm.acompletion.call_args.kwargs["messages"]
        assert sent[0] is messages[0]

    @pytest.mark.asyncio
    async def test_local_handler_path_not_touched(self):
        """claude-agent providers own their cache_control (provider-side)."""
        mock_handler = AsyncMock()
        mock_handler.acompletion.return_value = MagicMock()
        router = MarkitaiRouter([_local_entry("claude-agent/sonnet")])
        messages = [{"role": "system", "content": self.long_system}]

        with patch("markitai.providers.get_provider", return_value=mock_handler):
            await router.acompletion("default", messages)

        sent = mock_handler.acompletion.call_args.kwargs["messages"]
        assert sent[0] is messages[0]


class TestMarkitaiRouterMixed:
    """Tests for MarkitaiRouter with local and standard models."""

    def test_init_splits_entries(self):
        """Standard and local entries are split; a LiteLLM Router is created."""
        router = MarkitaiRouter(
            [
                _standard_entry("openai/gpt-4o"),
                _local_entry("claude-agent/sonnet"),
            ]
        )
        assert router._standard_router is not None
        assert len(router._local_entries) == 1
        assert len(router._standard_entries) == 1
        assert len(router.model_list) == 2

    def test_standard_pool_candidate_aggregates_weight(self):
        """The pool candidate's weight is the sum of standard model weights."""
        router = MarkitaiRouter(
            [
                _standard_entry("openai/gpt-4o", 3),
                _standard_entry("openai/gpt-4.1-mini", 2),
                _local_entry("claude-agent/sonnet", 5),
            ]
        )
        by_id = {c.model_id: c for c in router._groups["default"]}
        assert by_id[STANDARD_POOL_ID].weight == 5
        assert by_id["claude-agent/sonnet"].weight == 5

    def test_inner_router_disables_litellm_retries(self):
        """Transport retries are owned by the engine, not the inner Router."""
        router = MarkitaiRouter(
            [_standard_entry("openai/gpt-4o")],
            router_settings={"num_retries": 7, "timeout": 30},
        )
        assert router._standard_router is not None
        assert router._standard_router.num_retries == 0

    @pytest.mark.asyncio
    async def test_standard_pool_receives_group_name(self):
        """The LiteLLM Router is called with the group name, not a deployment id.

        In-group weighted balancing, cooldown, and fallbacks belong to the
        LiteLLM Router; passing a concrete deployment id would bypass them.
        """
        router = MarkitaiRouter(
            [
                _standard_entry("openai/gpt-4o"),
                _local_entry("claude-agent/sonnet"),
            ]
        )
        mock_standard = MagicMock()
        mock_standard.acompletion = AsyncMock(return_value="ok")
        router._standard_router = mock_standard

        messages = [{"role": "user", "content": "Hello"}]
        with patch.object(router, "_select", return_value=STANDARD_POOL_ID):
            result = await router.acompletion("default", messages, temperature=0.2)

        assert result == "ok"
        mock_standard.acompletion.assert_awaited_once_with(
            "default", messages, temperature=0.2
        )

    @pytest.mark.asyncio
    async def test_local_selection_bypasses_standard_router(self):
        """A selected local model goes straight to its handler."""
        router = MarkitaiRouter(
            [
                _standard_entry("openai/gpt-4o"),
                _local_entry("claude-agent/sonnet"),
            ]
        )
        mock_standard = MagicMock()
        mock_standard.acompletion = AsyncMock()
        router._standard_router = mock_standard

        mock_handler = AsyncMock()
        mock_handler.acompletion.return_value = "local-ok"

        with (
            patch.object(router, "_select", return_value="claude-agent/sonnet"),
            patch("markitai.providers.get_provider", return_value=mock_handler),
        ):
            result = await router.acompletion(
                "default", [{"role": "user", "content": "Hello"}]
            )

        assert result == "local-ok"
        mock_standard.acompletion.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concrete_standard_id_passes_through(self):
        """Addressing a concrete standard deployment id still works."""
        router = MarkitaiRouter(
            [
                _standard_entry("openai/gpt-4o"),
                _local_entry("claude-agent/sonnet"),
            ]
        )
        mock_standard = MagicMock()
        mock_standard.acompletion = AsyncMock(return_value="ok")
        router._standard_router = mock_standard

        messages = [{"role": "user", "content": "Hello"}]
        result = await router.acompletion("openai/gpt-4.1-mini", messages)

        assert result == "ok"
        mock_standard.acompletion.assert_awaited_once_with(
            "openai/gpt-4.1-mini", messages
        )

    @pytest.mark.asyncio
    async def test_pool_exhaustion_cools_down_standard_pool(self):
        """ "No deployments available" puts the whole standard pool in cooldown."""
        router = MarkitaiRouter(
            [
                _standard_entry("openai/gpt-4o"),
                _local_entry("claude-agent/sonnet"),
            ]
        )
        mock_standard = MagicMock()
        mock_standard.acompletion = AsyncMock(
            side_effect=ValueError(
                "No deployments available for selected model group, "
                "Try again in 60 seconds."
            )
        )
        router._standard_router = mock_standard

        with (
            patch.object(router, "_select", return_value=STANDARD_POOL_ID),
            pytest.raises(ValueError),
        ):
            await router.acompletion("default", [{"role": "user", "content": "Hi"}])

        remaining = router._cooldowns[STANDARD_POOL_ID] - time.monotonic()
        assert 55 < remaining <= 60

    @pytest.mark.asyncio
    async def test_standard_error_does_not_cool_down_pool(self):
        """Individual standard-model failures are LiteLLM's cooldown business."""
        from litellm.exceptions import BadRequestError

        router = MarkitaiRouter(
            [
                _standard_entry("gemini/gemini-flash"),
                _local_entry("claude-agent/sonnet"),
            ]
        )
        mock_standard = MagicMock()
        mock_standard.acompletion = AsyncMock(
            side_effect=BadRequestError(
                message="Invalid request: content too long",
                model="gemini/gemini-flash",
                llm_provider="gemini",
            )
        )
        router._standard_router = mock_standard

        with (
            patch.object(router, "_select", return_value=STANDARD_POOL_ID),
            pytest.raises(BadRequestError),
        ):
            await router.acompletion("default", [{"role": "user", "content": "Hi"}])

        assert STANDARD_POOL_ID not in router._cooldowns


# =============================================================================
# Test router_settings.fallbacks actually taking effect
# =============================================================================


class TestRouterFallbacks:
    """Configured fallbacks must switch groups when the primary fails.

    Uses litellm's ``mock_response`` deployment param: the magic string
    "litellm.RateLimitError" makes the deployment raise, and any other
    string is returned as the completion content — so the real LiteLLM
    Router fallback path runs without network access.
    """

    @staticmethod
    def _fallback_router() -> MarkitaiRouter:
        return MarkitaiRouter(
            [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "test-key",
                        "mock_response": "litellm.RateLimitError",
                    },
                },
                {
                    "model_name": "backup",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "test-key",
                        "mock_response": "backup response",
                    },
                },
            ],
            router_settings={"fallbacks": [{"default": ["backup"]}]},
        )

    @pytest.mark.asyncio
    async def test_fallback_group_serves_when_primary_fails(self):
        """The backup group answers when every default-group model fails."""
        router = self._fallback_router()
        response = await router.acompletion(
            "default", [{"role": "user", "content": "Hi"}], max_tokens=10
        )
        assert response.choices[0].message.content == "backup response"

    @pytest.mark.asyncio
    async def test_engine_call_reaches_fallback_group(self):
        """A full engine text call is served by the fallback group."""
        import asyncio

        from markitai.llm.engine import LLMEngine

        router = self._fallback_router()
        engine = LLMEngine(
            router=router,
            semaphore=asyncio.Semaphore(1),
            memory_cache=MagicMock(),
            persistent_cache=MagicMock(),
            track_usage=MagicMock(),
            calculate_max_tokens=MagicMock(return_value=64),
            get_primary_model=MagicMock(return_value=None),
            max_retries=0,
        )
        response = await engine.complete_text(
            model="default",
            messages=[{"role": "user", "content": "Hi"}],
            call_id="fallback-test",
        )
        assert response.content == "backup response"

    def test_create_router_preserves_groups_when_fallbacks_configured(
        self, prompts_config: PromptsConfig
    ):
        """With fallbacks configured, standard model groups survive into LiteLLM."""
        from markitai.config import RouterSettings

        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini", api_key="test-key"
                    ),
                ),
                ModelConfig(
                    model_name="backup",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o", api_key="test-key"
                    ),
                ),
            ],
            router_settings=RouterSettings(fallbacks=[{"default": ["backup"]}]),
        )
        processor = LLMProcessor(config, prompts_config)
        router = processor.router
        assert isinstance(router, MarkitaiRouter)
        assert router._standard_router is not None
        groups = {e["model_name"] for e in router._standard_router.model_list}
        assert groups == {"default", "backup"}
        assert router._standard_router.fallbacks == [{"default": ["backup"]}]

    def test_create_router_pools_groups_when_no_fallbacks(
        self, prompts_config: PromptsConfig
    ):
        """Without fallbacks, all model groups normalize into "default"."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="primary",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini", api_key="test-key"
                    ),
                ),
                ModelConfig(
                    model_name="secondary",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o", api_key="test-key"
                    ),
                ),
            ],
        )
        processor = LLMProcessor(config, prompts_config)
        router = processor.router
        assert router._standard_router is not None
        groups = {e["model_name"] for e in router._standard_router.model_list}
        assert groups == {"default"}

    def test_fallbacks_without_default_group_raises(
        self, prompts_config: PromptsConfig
    ):
        """Fallback routing needs the "default" entry group."""
        from markitai.config import RouterSettings

        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="primary",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini", api_key="test-key"
                    ),
                ),
            ],
            router_settings=RouterSettings(fallbacks=[{"primary": ["backup"]}]),
        )
        processor = LLMProcessor(config, prompts_config)
        with pytest.raises(ValueError, match="'default' model group"):
            _ = processor.router


# =============================================================================
# Test SQLiteCache
# =============================================================================


class TestSQLiteCache:
    """Tests for SQLiteCache class."""

    def test_init_creates_db(self, tmp_path: Path):
        """Test initialization creates database file."""
        db_path = tmp_path / "cache.db"
        _ = SQLiteCache(db_path)
        assert db_path.exists()

    def test_compute_hash_consistency(self, tmp_path: Path):
        """Test hash is consistent for same input."""
        cache = SQLiteCache(tmp_path / "cache.db")
        hash1 = cache._compute_hash("prompt", "content")
        hash2 = cache._compute_hash("prompt", "content")
        assert hash1 == hash2

    def test_compute_hash_uniqueness(self, tmp_path: Path):
        """Test different inputs produce different hashes."""
        cache = SQLiteCache(tmp_path / "cache.db")
        hash1 = cache._compute_hash("prompt1", "content")
        hash2 = cache._compute_hash("prompt2", "content")
        assert hash1 != hash2

    def test_compute_hash_uses_head_tail(self, tmp_path: Path):
        """Test hash uses head and tail for large content."""
        cache = SQLiteCache(tmp_path / "cache.db")
        # Create content longer than 25000 chars
        long_content = "a" * 50000
        hash1 = cache._compute_hash("prompt", long_content)
        # Same length, different tail
        different_tail = "a" * 25000 + "b" * 25000
        hash2 = cache._compute_hash("prompt", different_tail)
        assert hash1 != hash2

    def test_set_and_get(self, tmp_path: Path):
        """Test basic set and get operations."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("prompt", "content", '{"result": "test"}', "test-model")
        result = cache.get("prompt", "content", model="test-model")
        assert result == '{"result": "test"}'

    def test_get_miss(self, tmp_path: Path):
        """Test get returns None for missing entry."""
        cache = SQLiteCache(tmp_path / "cache.db")
        result = cache.get("nonexistent", "content")
        assert result is None

    def test_clear(self, tmp_path: Path):
        """Test clearing cache."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("p1", "c1", "r1")
        cache.set("p2", "c2", "r2")

        count = cache.clear()
        assert count == 2
        assert cache.get("p1", "c1") is None

    def test_stats(self, tmp_path: Path):
        """Test cache statistics."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("prompt", "content", '{"key": "value"}')

        stats = cache.stats()
        assert stats["count"] == 1
        assert stats["size_bytes"] > 0
        assert "db_path" in stats

    def test_stats_by_model(self, tmp_path: Path):
        """Test statistics grouped by model."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("p1", "c1", "r1", "model-a")
        cache.set("p2", "c2", "r2", "model-a")
        cache.set("p3", "c3", "r3", "model-b")

        stats = cache.stats_by_model()
        assert "model-a" in stats
        assert stats["model-a"]["count"] == 2
        assert "model-b" in stats
        assert stats["model-b"]["count"] == 1

    def test_list_entries(self, tmp_path: Path):
        """Test listing cache entries."""
        cache = SQLiteCache(tmp_path / "cache.db")
        cache.set("p1", "c1", '{"caption": "Test image"}', "model")

        entries = cache.list_entries(limit=10)
        assert len(entries) == 1
        assert "key" in entries[0]
        assert entries[0]["model"] == "model"

    def test_parse_value_preview_json(self, tmp_path: Path):
        """Test preview parsing for JSON values."""
        cache = SQLiteCache(tmp_path / "cache.db")

        # Image caption
        preview = cache._parse_value_preview('{"caption": "A beautiful sunset"}')
        assert preview.startswith("image:")

        # Frontmatter
        preview = cache._parse_value_preview('{"title": "My Document"}')
        assert preview.startswith("frontmatter:")

    def test_parse_value_preview_plain(self, tmp_path: Path):
        """Test preview parsing for plain text."""
        cache = SQLiteCache(tmp_path / "cache.db")
        preview = cache._parse_value_preview("Just plain text content")
        assert preview.startswith("text:")

    def test_parse_value_preview_empty(self, tmp_path: Path):
        """Test preview parsing for empty value."""
        cache = SQLiteCache(tmp_path / "cache.db")
        assert cache._parse_value_preview("") == ""
        assert cache._parse_value_preview(None) == ""


# =============================================================================
# Test PersistentCache
# =============================================================================


class TestPersistentCache:
    """Tests for PersistentCache class."""

    def test_init_disabled(self):
        """Test initialization with caching disabled."""
        cache = PersistentCache(enabled=False)
        assert cache._enabled is False
        assert cache._global_cache is None

    def test_init_enabled(self, tmp_path: Path):
        """Test initialization with caching enabled."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        assert cache._enabled is True
        assert cache._global_cache is not None

    def test_get_disabled(self):
        """Test get returns None when disabled."""
        cache = PersistentCache(enabled=False)
        result = cache.get("prompt", "content")
        assert result is None

    def test_get_skip_read(self, tmp_path: Path):
        """Test get returns None when skip_read is True."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True, skip_read=True)
        # Set a value
        cache.set("prompt", "content", {"result": "test"})
        # Get should return None (skip_read mode)
        result = cache.get("prompt", "content")
        assert result is None

    def test_set_and_get(self, tmp_path: Path):
        """Test basic set and get."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        cache.set("prompt", "content", {"key": "value"}, "model")
        result = cache.get("prompt", "content", model="model")
        assert result == {"key": "value"}

    def test_set_disabled(self):
        """Test set does nothing when disabled."""
        cache = PersistentCache(enabled=False)
        cache.set("prompt", "content", {"key": "value"})
        # Should not raise, just no-op

    def test_hit_miss_tracking(self, tmp_path: Path):
        """Test hit/miss tracking."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)

        # Miss
        cache.get("prompt1", "content1")
        assert cache._misses == 1
        assert cache._hits == 0

        # Set and hit
        cache.set("prompt2", "content2", "result")
        cache.get("prompt2", "content2")
        assert cache._hits == 1
        assert cache._misses == 1

    def test_stats(self, tmp_path: Path):
        """Test cache statistics."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        cache.set("p", "c", "r")
        cache.get("p", "c")  # Hit
        cache.get("x", "y")  # Miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0

    def test_clear(self, tmp_path: Path):
        """Test clearing cache."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        cache.set("p1", "c1", "r1")
        cache.set("p2", "c2", "r2")

        count = cache.clear()
        assert count == 2

    def test_glob_match_standard(self, tmp_path: Path):
        """Test standard glob matching."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        assert cache._glob_match("file.pdf", "*.pdf") is True
        assert cache._glob_match("file.txt", "*.pdf") is False

    def test_glob_match_double_star(self, tmp_path: Path):
        """Test ** glob matching for zero-or-more directories."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        # Should match zero directories
        assert cache._glob_match("file.pdf", "**/*.pdf") is True
        # Should match one directory
        assert cache._glob_match("dir/file.pdf", "**/*.pdf") is True
        # Should match multiple directories
        assert cache._glob_match("a/b/c/file.pdf", "**/*.pdf") is True

    def test_extract_matchable_path_simple(self, tmp_path: Path):
        """Test extracting filename from simple path."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        assert cache._extract_matchable_path("file.pdf") == "file.pdf"

    def test_extract_matchable_path_with_suffix(self, tmp_path: Path):
        """Test extracting filename from path with suffix."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        result = cache._extract_matchable_path("/path/to/file.pdf:images")
        assert result == "file.pdf"

    def test_extract_matchable_path_windows(self, tmp_path: Path):
        """Test extracting filename from Windows path."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        result = cache._extract_matchable_path("C:\\Users\\test\\file.pdf")
        assert result == "file.pdf"

    def test_should_skip_cache_no_patterns(self, tmp_path: Path):
        """Test skip check with no patterns."""
        cache = PersistentCache(global_dir=tmp_path, enabled=True)
        assert cache._should_skip_cache("file.pdf") is False

    def test_should_skip_cache_matching_pattern(self, tmp_path: Path):
        """Test skip check with matching pattern."""
        cache = PersistentCache(
            global_dir=tmp_path, enabled=True, no_cache_patterns=["*.pdf"]
        )
        assert cache._should_skip_cache("file.pdf") is True
        assert cache._should_skip_cache("file.txt") is False

    def test_should_skip_cache_full_path(self, tmp_path: Path):
        """Test skip check with full path context."""
        cache = PersistentCache(
            global_dir=tmp_path, enabled=True, no_cache_patterns=["*.JPG"]
        )
        # Should match filename from full path
        assert cache._should_skip_cache("/home/user/project/photo.JPG") is True


# =============================================================================
# Test ContentCache
# =============================================================================


class TestContentCacheExtended:
    """Extended tests for ContentCache class."""

    def test_lru_eviction_order(self):
        """Test LRU eviction removes oldest accessed item."""
        cache = ContentCache(maxsize=3, ttl_seconds=300)

        fake_time = [1000.0]
        with patch("markitai.llm.cache.time.time", side_effect=lambda: fake_time[0]):
            cache.set("p1", "c1", "r1")
            fake_time[0] = 1000.01
            cache.set("p2", "c2", "r2")
            fake_time[0] = 1000.02
            cache.set("p3", "c3", "r3")

            # Access p1 to make it recently used
            cache.get("p1", "c1")
            fake_time[0] = 1000.03

            # Add new item - should evict p2 (oldest accessed)
            cache.set("p4", "c4", "r4")

            assert cache.get("p1", "c1") == "r1"  # Still there
            assert cache.get("p2", "c2") is None  # Evicted
            assert cache.get("p3", "c3") == "r3"  # Still there
            assert cache.get("p4", "c4") == "r4"  # New item

    def test_update_moves_to_end(self):
        """Test updating existing key moves it to end."""
        cache = ContentCache(maxsize=3, ttl_seconds=300)

        fake_time = [1000.0]
        with patch("markitai.llm.cache.time.time", side_effect=lambda: fake_time[0]):
            cache.set("p1", "c1", "r1")
            fake_time[0] = 1000.01
            cache.set("p2", "c2", "r2")
            fake_time[0] = 1000.02
            cache.set("p3", "c3", "r3")

            # Update p1 - should move to end
            cache.set("p1", "c1", "r1_updated")
            fake_time[0] = 1000.03

            # Add new item - should evict p2 (now oldest)
            cache.set("p4", "c4", "r4")

            assert cache.get("p1", "c1") == "r1_updated"
            assert cache.get("p2", "c2") is None

    def test_thread_safety_concurrent_set_get(self):
        """Test ContentCache handles concurrent thread access without corruption.

        Defensive against Python 3.13+ free-threaded mode (PEP 703).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cache = ContentCache(maxsize=50, ttl_seconds=300)
        errors: list[str] = []

        def writer(thread_id: int) -> None:
            for i in range(100):
                cache.set(f"p{thread_id}", f"c{i}", f"result-{thread_id}-{i}")

        def reader(thread_id: int) -> None:
            for i in range(100):
                result = cache.get(f"p{thread_id}", f"c{i}")
                if result is not None and not result.startswith(f"result-{thread_id}-"):
                    errors.append(f"Wrong result for thread {thread_id}: {result}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for t in range(4):
                futures.append(pool.submit(writer, t))
                futures.append(pool.submit(reader, t))
            for f in as_completed(futures):
                f.result()  # Raises if thread had exception

        assert not errors, f"Thread safety violations: {errors}"
        # Cache should still be functional after concurrent access
        cache.set("final", "test", "ok")
        assert cache.get("final", "test") == "ok"


# =============================================================================
# Test LLMProcessor
# =============================================================================


class TestLLMProcessorInit:
    """Tests for LLMProcessor initialization."""

    def test_init_with_no_cache(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test initialization with no_cache flag."""
        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        assert processor._persistent_cache._skip_read is True

    def test_init_with_no_cache_patterns(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test initialization with no_cache_patterns."""
        patterns = ["*.pdf", "reports/**"]
        processor = LLMProcessor(llm_config, prompts_config, no_cache_patterns=patterns)
        assert processor._persistent_cache._no_cache_patterns == patterns

    def test_init_creates_default_usage_tracking(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test default usage tracking structures are created."""
        processor = LLMProcessor(llm_config, prompts_config)
        assert processor._usage is not None
        assert processor._context_usage is not None
        assert processor._call_counter is not None


class TestLLMProcessorUsageTracking:
    """Tests for usage tracking methods."""

    def test_track_usage_basic(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test basic usage tracking."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001)

        usage = processor.get_usage()
        assert "model-a" in usage
        assert usage["model-a"]["requests"] == 1
        assert usage["model-a"]["input_tokens"] == 100
        assert usage["model-a"]["output_tokens"] == 50
        assert usage["model-a"]["cost_usd"] == pytest.approx(0.001)

    def test_track_usage_accumulates(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test usage accumulates across multiple calls."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001)
        processor._track_usage("model-a", 200, 100, 0.002)

        usage = processor.get_usage()
        assert usage["model-a"]["requests"] == 2
        assert usage["model-a"]["input_tokens"] == 300
        assert usage["model-a"]["output_tokens"] == 150
        assert usage["model-a"]["cost_usd"] == pytest.approx(0.003)

    def test_track_usage_with_context(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test usage tracking with context."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001, context="file.pdf")

        context_usage = processor.get_context_usage("file.pdf")
        assert "model-a" in context_usage
        assert context_usage["model-a"]["requests"] == 1

    def test_get_total_cost(self, llm_config: LLMConfig, prompts_config: PromptsConfig):
        """Test getting total cost across all models."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001)
        processor._track_usage("model-b", 200, 100, 0.002)

        total = processor.get_total_cost()
        assert total == pytest.approx(0.003)

    def test_get_context_cost(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test getting cost for specific context."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001, context="file.pdf")
        processor._track_usage("model-a", 200, 100, 0.002, context="file.pdf")
        processor._track_usage("model-b", 50, 25, 0.0005, context="other.pdf")

        cost = processor.get_context_cost("file.pdf")
        assert cost == pytest.approx(0.003)

    def test_get_context_cost_unknown(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test getting cost for unknown context returns 0."""
        processor = LLMProcessor(llm_config, prompts_config)
        cost = processor.get_context_cost("nonexistent")
        assert cost == 0.0

    def test_clear_context_usage(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test clearing usage for specific context."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._track_usage("model-a", 100, 50, 0.001, context="file.pdf")
        processor._call_counter["file.pdf"] = 5

        processor.clear_context_usage("file.pdf")

        assert processor.get_context_usage("file.pdf") == {}
        assert "file.pdf" not in processor._call_counter

    def test_usage_tracking_thread_safety(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test usage tracking is thread-safe."""
        processor = LLMProcessor(llm_config, prompts_config)

        def track_usage():
            for _ in range(100):
                processor._track_usage("model", 10, 5, 0.0001)

        threads = [threading.Thread(target=track_usage) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        usage = processor.get_usage()
        assert usage["model"]["requests"] == 1000


class TestLLMProcessorCallCounter:
    """Tests for call counter methods."""

    def test_get_next_call_index(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test call index increments."""
        processor = LLMProcessor(llm_config, prompts_config)

        idx1 = processor._get_next_call_index("file.pdf")
        idx2 = processor._get_next_call_index("file.pdf")
        idx3 = processor._get_next_call_index("other.pdf")

        assert idx1 == 1
        assert idx2 == 2
        assert idx3 == 1

    def test_reset_call_counter_specific(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test resetting counter for specific context."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._get_next_call_index("file.pdf")
        processor._get_next_call_index("file.pdf")
        processor._get_next_call_index("other.pdf")

        processor.reset_call_counter("file.pdf")

        # file.pdf should start fresh
        assert processor._get_next_call_index("file.pdf") == 1
        # other.pdf should continue
        assert processor._get_next_call_index("other.pdf") == 2

    def test_reset_call_counter_all(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test resetting all counters."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._get_next_call_index("file.pdf")
        processor._get_next_call_index("other.pdf")

        processor.reset_call_counter()

        assert processor._get_next_call_index("file.pdf") == 1
        assert processor._get_next_call_index("other.pdf") == 1


class TestLLMProcessorCacheManagement:
    """Tests for cache management methods."""

    def test_get_cache_stats(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test getting cache statistics."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor.engine._cache_hits = 5
        processor.engine._cache_misses = 10

        stats = processor.get_cache_stats()
        assert stats["memory"]["hits"] == 5
        assert stats["memory"]["misses"] == 10
        assert stats["memory"]["hit_rate"] == pytest.approx(33.33, rel=0.01)

    def test_clear_cache_memory(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test clearing memory cache."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._cache.set("p", "c", "r")
        processor.engine._cache_hits = 5
        processor.engine._cache_misses = 10

        result = processor.clear_cache("memory")

        assert result["memory"] == 1
        assert processor._cache.size == 0
        assert processor.engine._cache_hits == 0
        assert processor.engine._cache_misses == 0

    def test_clear_image_cache(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test clearing image cache."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._image_cache["path"] = (b"data", "base64")
        processor._image_cache_bytes = 1000

        processor.clear_image_cache()

        assert len(processor._image_cache) == 0
        assert processor._image_cache_bytes == 0


class TestLLMProcessorImageCache:
    """Tests for _get_cached_image method."""

    def test_get_cached_image_cache_hit(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig, tmp_path: Path
    ):
        """Test image cache hit returns cached value."""
        processor = LLMProcessor(llm_config, prompts_config)

        img_path = tmp_path / "test.jpg"
        cached_data = (b"image_bytes", "base64_encoded")
        processor._image_cache[str(img_path)] = cached_data

        result = processor._get_cached_image(img_path)
        assert result == cached_data

    def test_get_cached_image_cache_miss(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        tmp_path: Path,
        sample_png_bytes: bytes,
    ):
        """Test image cache miss reads file and caches."""
        processor = LLMProcessor(llm_config, prompts_config)

        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_png_bytes)

        result = processor._get_cached_image(img_path)

        assert result[0] == sample_png_bytes
        assert result[1] == base64.b64encode(sample_png_bytes).decode()
        assert str(img_path) in processor._image_cache

    def test_get_cached_image_lru_eviction(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        tmp_path: Path,
        sample_png_bytes: bytes,
    ):
        """Test LRU eviction when cache is full."""
        processor = LLMProcessor(llm_config, prompts_config)
        processor._image_cache_max_size = 2

        # Create and cache 2 images
        for i in range(2):
            img_path = tmp_path / f"test_{i}.png"
            img_path.write_bytes(sample_png_bytes)
            processor._get_cached_image(img_path)

        assert len(processor._image_cache) == 2

        # Add third image - should evict first
        img_path_3 = tmp_path / "test_3.png"
        img_path_3.write_bytes(sample_png_bytes)
        processor._get_cached_image(img_path_3)

        assert len(processor._image_cache) == 2
        assert str(tmp_path / "test_0.png") not in processor._image_cache


class TestGetCachedImageSVG:
    """Tests for SVG rasterization in _get_cached_image."""

    def test_svg_rasterized_to_png_when_cairosvg_available(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig, tmp_path: Path
    ):
        """SVG bytes should be rasterized to PNG via cairosvg before caching."""
        processor = LLMProcessor(llm_config, prompts_config)

        svg_content = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10" fill="red"/></svg>'
        svg_path = tmp_path / "test.svg"
        svg_path.write_bytes(svg_content)

        fake_png = b"\x89PNG\r\n\x1a\nfake_png_data"
        with patch("markitai.llm.processor.cairosvg") as mock_cairo:
            mock_cairo.svg2png.return_value = fake_png
            raw_bytes, b64_str = processor._get_cached_image(svg_path)

        mock_cairo.svg2png.assert_called_once()
        assert raw_bytes == fake_png
        assert b64_str == base64.b64encode(fake_png).decode()

    def test_svg_skipped_when_cairosvg_missing(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig, tmp_path: Path
    ):
        """SVG should be returned as raw bytes when cairosvg is not installed."""
        processor = LLMProcessor(llm_config, prompts_config)

        svg_content = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        svg_path = tmp_path / "test.svg"
        svg_path.write_bytes(svg_content)

        with patch("markitai.llm.processor.cairosvg", None):
            raw_bytes, _b64_str = processor._get_cached_image(svg_path)

        # Falls through without rasterization — returns raw SVG bytes
        assert raw_bytes == svg_content


class TestLLMProcessorDynamicMaxTokens:
    """Tests for _calculate_dynamic_max_tokens method."""

    def test_returns_none_without_model_info(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test returns None when model info unavailable."""
        processor = LLMProcessor(llm_config, prompts_config)

        with patch(
            "markitai.llm.processor.get_model_info_cached",
            return_value={"max_input_tokens": None, "max_output_tokens": None},
        ):
            result = processor._calculate_dynamic_max_tokens(
                [{"role": "user", "content": "test"}], "unknown/model"
            )
            assert result is None

    def test_calculates_based_on_input_size(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test calculation based on input token count."""
        processor = LLMProcessor(llm_config, prompts_config)

        with (
            patch(
                "markitai.llm.processor.get_model_info_cached",
                return_value={"max_input_tokens": 128000, "max_output_tokens": 8192},
            ),
            patch("litellm.token_counter", return_value=1000),
        ):
            result = processor._calculate_dynamic_max_tokens(
                [{"role": "user", "content": "test"}], "test/model"
            )
            # Should return a reasonable value
            assert result is not None
            assert result >= 1000  # At least minimum floor
            assert result <= 8192  # At most max_output

    def test_table_heavy_content(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test table-heavy content gets more output tokens."""
        processor = LLMProcessor(llm_config, prompts_config)

        # Content with many table rows
        table_content = "|col1|col2|\n" * 30

        with (
            patch(
                "markitai.llm.processor.get_model_info_cached",
                return_value={"max_input_tokens": 128000, "max_output_tokens": 16384},
            ),
            patch("litellm.token_counter", return_value=500),
        ):
            result = processor._calculate_dynamic_max_tokens(
                [{"role": "user", "content": table_content}], "test/model"
            )
            # Should have higher floor for table-heavy content
            assert result is not None and result >= 4000

    def test_uses_router_model_limits(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test uses minimum limits across router models."""
        processor = LLMProcessor(llm_config, prompts_config)

        mock_router = MagicMock()
        mock_router.model_list = [
            {"litellm_params": {"model": "model-a"}},
            {"litellm_params": {"model": "model-b"}},
        ]

        with (
            patch(
                "markitai.llm.processor.get_model_info_cached",
                side_effect=[
                    {"max_input_tokens": 128000, "max_output_tokens": 16384},
                    {"max_input_tokens": 64000, "max_output_tokens": 4096},
                ],
            ),
            patch("litellm.token_counter", return_value=1000),
        ):
            result = processor._calculate_dynamic_max_tokens(
                [{"role": "user", "content": "test"}],
                router=mock_router,
            )
            # Should use minimum max_output (4096)
            assert result is not None and result <= 4096


class TestLLMProcessorRouterHelpers:
    """Tests for router helper methods."""

    def test_get_router_primary_model_returns_highest_weight(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Should return the highest-weight model, not model_list[0]."""
        processor = LLMProcessor(llm_config, prompts_config)

        mock_router = MagicMock()
        mock_router.model_list = [
            {"litellm_params": {"model": "deepseek/deepseek-chat", "weight": 0}},
            {"litellm_params": {"model": "chatgpt/gpt-5.3", "weight": 20}},
            {
                "litellm_params": {
                    "model": "gemini/gemini-3.1-flash-lite-preview",
                    "weight": 0,
                }
            },
        ]

        result = processor._get_router_primary_model(mock_router)
        assert result == "chatgpt/gpt-5.3"

    def test_get_router_primary_model_default_weight(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Models without explicit weight should default to 1.0."""
        processor = LLMProcessor(llm_config, prompts_config)

        mock_router = MagicMock()
        mock_router.model_list = [
            {"litellm_params": {"model": "openai/gpt-4o"}},
            {"litellm_params": {"model": "deepseek/deepseek-chat"}},
        ]

        result = processor._get_router_primary_model(mock_router)
        # Both have default weight 1.0, first one wins
        assert result == "openai/gpt-4o"

    def test_get_router_primary_model_empty(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test returns None for empty model list."""
        processor = LLMProcessor(llm_config, prompts_config)

        mock_router = MagicMock()
        mock_router.model_list = []

        result = processor._get_router_primary_model(mock_router)
        assert result is None

    def test_has_images_true(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test image detection in messages using has_images from providers.common."""
        from markitai.providers.common import has_images

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,x"},
                    },
                ],
            }
        ]

        assert has_images(messages) is True

    def test_has_images_false(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test text-only messages using has_images from providers.common."""
        from markitai.providers.common import has_images

        messages = [{"role": "user", "content": "Hello, world!"}]

        assert has_images(messages) is False


class TestLLMProcessorFormatOutput:
    """Tests for format_llm_output method."""

    def test_format_basic(self, llm_config: LLMConfig, prompts_config: PromptsConfig):
        """Test basic formatting."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output(
            "# Content", "title: Test\nsource: file.md"
        )

        assert result.startswith("---\n")
        assert "title: Test" in result
        # Content should be after frontmatter, may have trailing newline
        assert "\n\n# Content" in result

    def test_format_strips_existing_markers(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test strips existing --- markers."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output("# Content", "---\ntitle: Test\n---")

        # Should have exactly 2 markers
        assert result.count("---") == 2

    def test_format_strips_code_block(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test strips yaml code block."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output("# Content", "```yaml\ntitle: Test\n```")

        assert "```" not in result
        assert "title: Test" in result
        assert "\n\n# Content" in result


class TestLLMProcessorRouterCreation:
    """Tests for router creation logic."""

    def test_no_models_error(self, prompts_config: PromptsConfig):
        """Test error when no models configured."""
        config = LLMConfig(enabled=True, model_list=[])
        processor = LLMProcessor(config, prompts_config)

        with pytest.raises(ValueError, match="No models configured"):
            _ = processor.router

    def test_creates_router_for_local_models(self, prompts_config: PromptsConfig):
        """All-local configs get a MarkitaiRouter without an inner LiteLLM Router."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(model="claude-agent/sonnet"),
                )
            ],
        )
        with (
            patch("markitai.providers.is_local_provider_available", return_value=True),
            patch("markitai.providers.is_local_provider_model", return_value=True),
        ):
            processor = LLMProcessor(config, prompts_config)
            router = processor.router
            assert isinstance(router, MarkitaiRouter)
            assert router._standard_router is None

    def test_router_resolves_api_base_plain_url(self, prompts_config: PromptsConfig):
        """Test router passes plain api_base URL to litellm model list."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                        api_base="https://custom-proxy.example.com/v1",
                    ),
                )
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with patch("markitai.providers.is_local_provider_available", return_value=True):
            router = processor.router
            model_entry = router.model_list[0]
            assert (
                model_entry["litellm_params"]["api_base"]
                == "https://custom-proxy.example.com/v1"
            )

    def test_router_resolves_api_base_env_syntax(
        self, prompts_config: PromptsConfig, monkeypatch: pytest.MonkeyPatch
    ):
        """Test router resolves env:VAR_NAME in api_base before passing to litellm."""
        monkeypatch.setenv(
            "TEST_ROUTER_API_BASE", "https://env-resolved-proxy.example.com/v1"
        )
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                        api_base="env:TEST_ROUTER_API_BASE",
                    ),
                )
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with patch("markitai.providers.is_local_provider_available", return_value=True):
            router = processor.router
            model_entry = router.model_list[0]
            assert (
                model_entry["litellm_params"]["api_base"]
                == "https://env-resolved-proxy.example.com/v1"
            )

    def test_router_omits_api_base_when_none(self, prompts_config: PromptsConfig):
        """Test router does not include api_base when not configured."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                    ),
                )
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with patch("markitai.providers.is_local_provider_available", return_value=True):
            router = processor.router
            model_entry = router.model_list[0]
            assert "api_base" not in model_entry["litellm_params"]

    def test_create_router_filters_weight_zero_models(
        self, prompts_config: PromptsConfig
    ):
        """Test that _create_router excludes weight=0 models from the Router.

        This prevents LiteLLM's simple_shuffle from hitting ZeroDivisionError
        when all deployments have weight=0.
        """
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                        weight=0,
                    ),
                ),
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o",
                        api_key="test-key",
                        weight=10,
                    ),
                ),
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with patch("markitai.providers.is_local_provider_available", return_value=True):
            router = processor.router
            model_ids = [e["litellm_params"]["model"] for e in router.model_list]
            # weight=0 model should be excluded
            assert "openai/gpt-4o-mini" not in model_ids
            assert "openai/gpt-4o" in model_ids

    def test_create_router_logs_disabled_models_once_as_summary(
        self, prompts_config: PromptsConfig
    ) -> None:
        """Disabled models should be logged once as a compact summary."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                        weight=0,
                    ),
                ),
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o",
                        api_key="test-key",
                        weight=0,
                    ),
                ),
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-5.4",
                        api_key="test-key",
                        weight=1,
                    ),
                ),
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with (
            patch("markitai.providers.is_local_provider_available", return_value=True),
            patch("markitai.llm.processor.logger.debug") as mock_debug,
        ):
            _ = processor.router

        debug_messages = [
            call.args[0] for call in mock_debug.call_args_list if call.args
        ]
        assert (
            "[Router] Skipped 2 disabled models (weight=0): "
            "openai/gpt-4o-mini, openai/gpt-4o" in debug_messages
        )
        assert all("Skipping disabled model" not in msg for msg in debug_messages)

    def test_create_router_all_weight_zero_raises(self, prompts_config: PromptsConfig):
        """Test that _create_router raises ValueError when all models are weight=0."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                        weight=0,
                    ),
                ),
            ],
        )

        with (
            patch("markitai.providers.is_local_provider_available", return_value=True),
            pytest.raises(ValueError, match="weight=0.*disabled"),
        ):
            LLMProcessor(config, prompts_config)

    def test_create_router_skips_models_with_missing_env_api_key(
        self, prompts_config: PromptsConfig
    ):
        """Test that _create_router skips models whose env:VAR API key is missing.

        When a model has api_key="env:MISSING_VAR" and the env var is not set,
        the router should skip that model with a warning instead of crashing.
        """
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="gemini/gemini-2.0-flash",
                        api_key="env:NONEXISTENT_TEST_API_KEY_XYZ",
                    ),
                ),
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key-present",
                    ),
                ),
            ],
        )
        processor = LLMProcessor(config, prompts_config)

        with patch("markitai.providers.is_local_provider_available", return_value=True):
            router = processor.router
            model_ids = [e["litellm_params"]["model"] for e in router.model_list]
            # Model with missing env var should be skipped
            assert "gemini/gemini-2.0-flash" not in model_ids
            # Model with plain key should be included
            assert "openai/gpt-4o-mini" in model_ids

    def test_create_router_all_models_missing_env_api_key_raises(
        self, prompts_config: PromptsConfig
    ):
        """Test clear error when all models have missing env var API keys."""
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="gemini/gemini-2.0-flash",
                        api_key="env:NONEXISTENT_TEST_KEY_AAA",
                    ),
                ),
            ],
        )

        with (
            patch("markitai.providers.is_local_provider_available", return_value=True),
            pytest.raises(ValueError, match="NONEXISTENT_TEST_KEY_AAA"),
        ):
            LLMProcessor(config, prompts_config)

    def test_semaphore_property(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test semaphore property creates semaphore."""
        processor = LLMProcessor(llm_config, prompts_config)

        sem = processor.semaphore
        assert sem is not None
        # Should return same instance
        assert processor.semaphore is sem

    def test_semaphore_uses_runtime(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test semaphore uses runtime semaphore when provided."""
        from markitai.llm.types import LLMRuntime

        runtime = LLMRuntime(concurrency=5)
        processor = LLMProcessor(llm_config, prompts_config, runtime=runtime)

        # Should use runtime's semaphore
        assert processor.semaphore is runtime.semaphore

    def test_io_semaphore_returns_same_instance(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test io_semaphore returns the same instance on repeated access.

        Without caching, each access creates a new Semaphore, making
        I/O concurrency control completely ineffective.
        """
        processor = LLMProcessor(llm_config, prompts_config)

        sem1 = processor.io_semaphore
        sem2 = processor.io_semaphore
        assert sem1 is sem2

    def test_io_semaphore_uses_runtime(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test io_semaphore uses runtime's io_semaphore when provided."""
        from markitai.llm.types import LLMRuntime

        runtime = LLMRuntime(concurrency=5)
        processor = LLMProcessor(llm_config, prompts_config, runtime=runtime)

        assert processor.io_semaphore is runtime.io_semaphore


class TestLLMProcessorVisionModel:
    """Tests for vision model detection."""

    def test_is_vision_model_config_override(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test config override for vision support."""
        processor = LLMProcessor(llm_config, prompts_config)

        model_config = MagicMock()
        model_config.litellm_params.model = "some/model"
        model_config.model_info = MagicMock()
        model_config.model_info.supports_vision = True

        assert processor._is_vision_model(model_config) is True

        model_config.model_info.supports_vision = False
        assert processor._is_vision_model(model_config) is False

    def test_is_vision_model_local_provider(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test local providers are always vision capable.

        The function imports is_local_provider_model from markitai.providers
        inside the method body. Local provider models (claude-agent/*, copilot/*)
        are recognized as vision-capable.
        """
        processor = LLMProcessor(llm_config, prompts_config)

        # Use actual local provider model ID that will be recognized
        model_config = MagicMock()
        model_config.litellm_params.model = "claude-agent/sonnet"
        model_config.model_info = None

        # The actual implementation will recognize claude-agent/ as local provider
        result = processor._is_vision_model(model_config)
        assert result is True

    def test_is_vision_model_auto_detect(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test auto-detection from litellm for standard models.

        For non-local provider models, the function uses get_model_info_cached
        to check for vision support. GPT-4o is a known vision model.
        """
        processor = LLMProcessor(llm_config, prompts_config)

        model_config = MagicMock()
        model_config.litellm_params.model = "openai/gpt-4o"
        model_config.model_info = None

        # GPT-4o is known to support vision in litellm
        # The actual get_model_info_cached will be called
        result = processor._is_vision_model(model_config)
        # GPT-4o should be detected as vision capable
        assert result is True


# =============================================================================
# Test _call_llm_with_retry Model-Level Error Retry
# =============================================================================


class TestCallLlmModelLevelRetry:
    """Tests for _call_llm_with_retry retrying on model-level errors."""

    @pytest.mark.asyncio
    async def test_retries_on_model_level_error(self, llm_config, prompts_config):
        """Model-level 400 should be retried, not immediately raised."""
        from litellm.exceptions import BadRequestError

        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # First call: model-level error. Second call: success.
        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "OK"
        success_response.model = "claude-agent/sonnet"
        success_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        success_response._hidden_params = {}

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(
            side_effect=[
                BadRequestError(
                    message="User location is not supported for the API use.",
                    model="gemini/gemini-flash",
                    llm_provider="gemini",
                ),
                success_response,
            ]
        )
        processor._router = mock_router

        # Mock asyncio.sleep to skip the 1s retry backoff delay
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await processor._call_llm_with_retry(
                "default",
                [{"role": "user", "content": "test"}],
                call_id="test:1",
                context="test",
                max_retries=2,
            )

        assert result.content == "OK"
        assert mock_router.acompletion.await_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_regular_bad_request(self, llm_config, prompts_config):
        """Regular 400 should raise immediately, not retry."""
        from litellm.exceptions import BadRequestError

        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(
            side_effect=BadRequestError(
                message="Invalid request body",
                model="gemini/gemini-flash",
                llm_provider="gemini",
            )
        )
        processor._router = mock_router

        with pytest.raises(BadRequestError):
            await processor._call_llm_with_retry(
                "default",
                [{"role": "user", "content": "test"}],
                call_id="test:1",
                context="test",
                max_retries=2,
            )

        # Should NOT retry
        assert mock_router.acompletion.await_count == 1


# =============================================================================
# Test _call_llm_with_retry Rate Limit (429) Retry
# =============================================================================


class TestCallLlmRateLimitRetry:
    """Tests for _call_llm_with_retry retrying rate limits (429)."""

    @pytest.mark.asyncio
    async def test_retries_rate_limit_with_quota_text(self, llm_config, prompts_config):
        """A 429 RateLimitError mentioning 'quota' must be retried.

        Some providers' 429 errors carry text like 'quota will reset after
        57s'. The word 'quota' must not trigger the non-retryable billing
        check for genuine rate limits — the router cooldown routes the
        retry to another model.
        """
        from litellm.exceptions import RateLimitError

        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "OK"
        success_response.model = "claude-agent/sonnet"
        success_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        success_response._hidden_params = {}

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(
            side_effect=[
                RateLimitError(
                    message=(
                        "provider rate limit (429): Your quota will reset after 57s."
                    ),
                    llm_provider="copilot",
                    model="copilot/gemini-3-flash",
                ),
                success_response,
            ]
        )
        processor._router = mock_router

        # Mock asyncio.sleep to skip the retry backoff delay
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await processor._call_llm_with_retry(
                "default",
                [{"role": "user", "content": "test"}],
                call_id="test:1",
                context="test",
                max_retries=2,
            )

        assert result.content == "OK"
        assert mock_router.acompletion.await_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_rate_limit_with_billing_error(
        self, llm_config, prompts_config
    ):
        """RateLimitError with insufficient_quota (billing) raises immediately."""
        from litellm.exceptions import RateLimitError

        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(
            side_effect=RateLimitError(
                message="You exceeded your current quota (insufficient_quota)",
                llm_provider="openai",
                model="openai/gpt-4o",
            )
        )
        processor._router = mock_router

        with pytest.raises(RateLimitError):
            await processor._call_llm_with_retry(
                "default",
                [{"role": "user", "content": "test"}],
                call_id="test:1",
                context="test",
                max_retries=2,
            )

        # Should NOT retry
        assert mock_router.acompletion.await_count == 1
