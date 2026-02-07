"""Tests for LLM models module.

Tests cover:
- Model info caching and retrieval
- Response cost extraction
- Context display name formatting
- MarkitaiLLMLogger callback handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from markitai.llm.models import (
    MarkitaiLLMLogger,
    _model_info_cache,
    context_display_name,
    get_model_info_cached,
    get_model_max_output_tokens,
    get_response_cost,
)


class TestGetModelInfoCached:
    """Tests for get_model_info_cached function."""

    def setup_method(self):
        """Clear cache before each test."""
        _model_info_cache.clear()

    def test_returns_cached_value(self):
        """Test that cached values are returned without re-querying."""
        # Pre-populate cache
        _model_info_cache["test-model"] = {
            "max_input_tokens": 50000,
            "max_output_tokens": 4096,
            "supports_vision": True,
        }

        result = get_model_info_cached("test-model")

        assert result["max_input_tokens"] == 50000
        assert result["max_output_tokens"] == 4096
        assert result["supports_vision"] is True

    def test_local_provider_model(self):
        """Test that local provider models use provider-specific info."""
        with patch(
            "markitai.providers.get_local_provider_model_info"
        ) as mock_get_local:
            mock_get_local.return_value = {
                "max_input_tokens": 200000,
                "max_output_tokens": 64000,
                "supports_vision": True,
            }

            result = get_model_info_cached("claude-agent/sonnet")

            mock_get_local.assert_called_once_with("claude-agent/sonnet")
            assert result["max_input_tokens"] == 200000
            assert result["max_output_tokens"] == 64000
            assert result["supports_vision"] is True
            # Should be cached
            assert "claude-agent/sonnet" in _model_info_cache

    def test_litellm_model_info_success(self):
        """Test successful litellm model info retrieval."""
        with (
            patch(
                "markitai.providers.get_local_provider_model_info", return_value=None
            ),
            patch("markitai.llm.models.litellm.get_model_info") as mock_litellm,
        ):
            mock_litellm.return_value = {
                "max_input_tokens": 128000,
                "max_output_tokens": 16384,
                "supports_vision": True,
            }

            result = get_model_info_cached("openai/gpt-4o")

            assert result["max_input_tokens"] == 128000
            assert result["max_output_tokens"] == 16384
            assert result["supports_vision"] is True

    def test_litellm_model_info_partial(self):
        """Test that missing litellm fields use defaults."""
        with (
            patch(
                "markitai.providers.get_local_provider_model_info", return_value=None
            ),
            patch("markitai.llm.models.litellm.get_model_info") as mock_litellm,
        ):
            # Only max_input_tokens available
            mock_litellm.return_value = {
                "max_input_tokens": 32000,
            }

            result = get_model_info_cached("some/model")

            assert result["max_input_tokens"] == 32000
            # Defaults for missing fields
            assert result["max_output_tokens"] == 8192  # DEFAULT_MAX_OUTPUT_TOKENS
            assert result["supports_vision"] is False

    def test_litellm_model_info_failure(self):
        """Test fallback to defaults when litellm fails."""
        with (
            patch(
                "markitai.providers.get_local_provider_model_info", return_value=None
            ),
            patch("markitai.llm.models.litellm.get_model_info") as mock_litellm,
        ):
            mock_litellm.side_effect = Exception("Model not found")

            result = get_model_info_cached("unknown/model")

            # Should use defaults
            assert result["max_input_tokens"] == 128000
            assert result["max_output_tokens"] == 8192
            assert result["supports_vision"] is False

    def test_caches_result(self):
        """Test that results are cached."""
        with (
            patch(
                "markitai.providers.get_local_provider_model_info", return_value=None
            ),
            patch("markitai.llm.models.litellm.get_model_info") as mock_litellm,
        ):
            mock_litellm.return_value = {
                "max_input_tokens": 100000,
                "max_output_tokens": 8000,
                "supports_vision": False,
            }

            # First call
            get_model_info_cached("test/model")
            # Second call
            get_model_info_cached("test/model")

            # litellm should only be called once
            mock_litellm.assert_called_once()
            assert "test/model" in _model_info_cache

    def test_supports_vision_none_handling(self):
        """Test that supports_vision=None from litellm defaults to False."""
        with (
            patch(
                "markitai.providers.get_local_provider_model_info", return_value=None
            ),
            patch("markitai.llm.models.litellm.get_model_info") as mock_litellm,
        ):
            mock_litellm.return_value = {
                "max_input_tokens": 100000,
                "max_output_tokens": 8000,
                "supports_vision": None,  # Explicitly None
            }

            result = get_model_info_cached("model/without/vision")

            # None should NOT update the default False
            assert result["supports_vision"] is False


class TestGetModelMaxOutputTokens:
    """Tests for get_model_max_output_tokens function."""

    def setup_method(self):
        """Clear cache before each test."""
        _model_info_cache.clear()

    def test_returns_max_output_tokens(self):
        """Test that max_output_tokens is returned correctly."""
        _model_info_cache["cached-model"] = {
            "max_input_tokens": 100000,
            "max_output_tokens": 16384,
            "supports_vision": True,
        }

        result = get_model_max_output_tokens("cached-model")

        assert result == 16384

    def test_uses_get_model_info_cached(self):
        """Test that it delegates to get_model_info_cached."""
        with patch("markitai.llm.models.get_model_info_cached") as mock_get_info:
            mock_get_info.return_value = {
                "max_input_tokens": 100000,
                "max_output_tokens": 32000,
                "supports_vision": False,
            }

            result = get_model_max_output_tokens("test-model")

            mock_get_info.assert_called_once_with("test-model")
            assert result == 32000


class TestGetResponseCost:
    """Tests for get_response_cost function."""

    def test_local_provider_cost_from_hidden_params(self):
        """Test extracting cost from local provider's _hidden_params."""
        response = MagicMock()
        response._hidden_params = {"total_cost_usd": 0.0123}

        result = get_response_cost(response)

        assert result == pytest.approx(0.0123)

    def test_local_provider_cost_zero(self):
        """Test local provider with zero cost."""
        response = MagicMock()
        response._hidden_params = {"total_cost_usd": 0.0}

        result = get_response_cost(response)

        assert result == 0.0

    def test_litellm_completion_cost(self):
        """Test fallback to litellm.completion_cost."""
        response = MagicMock()
        response._hidden_params = {}  # No local provider cost

        with patch("markitai.llm.models.completion_cost") as mock_cost:
            mock_cost.return_value = 0.0456

            result = get_response_cost(response)

            mock_cost.assert_called_once_with(completion_response=response)
            assert result == pytest.approx(0.0456)

    def test_litellm_completion_cost_failure(self):
        """Test fallback to 0.0 when completion_cost fails."""
        response = MagicMock()
        response._hidden_params = {}

        with patch("markitai.llm.models.completion_cost") as mock_cost:
            mock_cost.side_effect = Exception("Cost calculation failed")

            result = get_response_cost(response)

            assert result == 0.0

    def test_no_hidden_params(self):
        """Test when response has no _hidden_params attribute."""
        response = MagicMock(spec=[])  # No attributes

        with patch("markitai.llm.models.completion_cost") as mock_cost:
            mock_cost.return_value = 0.0789

            result = get_response_cost(response)

            assert result == pytest.approx(0.0789)

    def test_hidden_params_not_dict(self):
        """Test when _hidden_params is not a dict."""
        response = MagicMock()
        response._hidden_params = "not a dict"

        with patch("markitai.llm.models.completion_cost") as mock_cost:
            mock_cost.return_value = 0.01

            result = get_response_cost(response)

            assert result == pytest.approx(0.01)


class TestContextDisplayName:
    """Tests for context_display_name function."""

    def test_empty_context(self):
        """Test empty context returns empty string."""
        assert context_display_name("") == ""

    def test_simple_filename(self):
        """Test simple filename without path."""
        assert context_display_name("file.pdf") == "file.pdf"

    def test_unix_path(self):
        """Test Unix-style path extracts filename."""
        assert context_display_name("/path/to/document.pdf") == "document.pdf"

    def test_windows_path(self):
        """Test Windows-style path is handled cross-platform."""
        result = context_display_name("C:\\Users\\test\\doc.pdf")
        assert result == "doc.pdf"

    def test_path_with_suffix(self):
        """Test path with :suffix preserves suffix."""
        result = context_display_name("/path/to/file.pdf:images")
        assert result == "file.pdf:images"

    def test_windows_path_with_suffix(self):
        """Test Windows path with :suffix preserves suffix cross-platform."""
        result = context_display_name("C:\\path\\to\\file.pdf:images")
        assert result == "file.pdf:images"

    def test_windows_drive_letter_preserved(self):
        """Test Windows drive letter handling cross-platform."""
        result = context_display_name("C:\\file.pdf")
        assert result == "file.pdf"

    def test_complex_suffix(self):
        """Test more complex suffix patterns."""
        result = context_display_name("/docs/report.pdf:page1:images")
        # rsplit with maxsplit=1 means only the last colon is considered
        assert result == "report.pdf:page1:images"

    def test_suffix_with_backslash(self):
        """Test that suffix starting with backslash is not split."""
        # This tests the condition: not parts[1].startswith("\\")
        # A path like "prefix:\\something" would have parts[1] = "\\something"
        result = context_display_name("some:path:\\test.txt")
        # Should not split on first colon since suffix starts with backslash
        assert "test.txt" in result


class TestMarkitaiLLMLogger:
    """Tests for MarkitaiLLMLogger callback class."""

    def test_init(self):
        """Test logger initialization."""
        logger = MarkitaiLLMLogger()
        assert logger.last_call_details == {}

    def test_log_success_event(self):
        """Test logging successful LLM call."""
        logger = MarkitaiLLMLogger()

        kwargs = {
            "standard_logging_object": {
                "api_base": "https://api.openai.com",
                "response_time": 1.5,
                "model_id": "gpt-4o",
            },
            "cache_hit": True,
        }
        response_obj = MagicMock()

        logger.log_success_event(kwargs, response_obj, None, None)

        assert logger.last_call_details["api_base"] == "https://api.openai.com"
        assert logger.last_call_details["response_time"] == 1.5
        assert logger.last_call_details["cache_hit"] is True
        assert logger.last_call_details["model_id"] == "gpt-4o"

    def test_log_success_event_no_cache_hit(self):
        """Test logging without cache_hit in kwargs."""
        logger = MarkitaiLLMLogger()

        kwargs = {
            "standard_logging_object": {
                "api_base": "https://api.example.com",
                "response_time": 2.0,
                "model_id": "test-model",
            },
        }
        response_obj = MagicMock()

        logger.log_success_event(kwargs, response_obj, None, None)

        assert logger.last_call_details["cache_hit"] is False

    def test_log_success_event_empty_slo(self):
        """Test logging with empty standard_logging_object."""
        logger = MarkitaiLLMLogger()

        kwargs = {}
        response_obj = MagicMock()

        logger.log_success_event(kwargs, response_obj, None, None)

        assert logger.last_call_details["api_base"] is None
        assert logger.last_call_details["response_time"] is None
        assert logger.last_call_details["model_id"] is None

    @pytest.mark.asyncio
    async def test_async_log_success_event(self):
        """Test async version of success event logging."""
        logger = MarkitaiLLMLogger()

        kwargs = {
            "standard_logging_object": {
                "api_base": "https://api.async.com",
                "response_time": 0.5,
                "model_id": "async-model",
            },
            "cache_hit": False,
        }
        response_obj = MagicMock()

        await logger.async_log_success_event(kwargs, response_obj, None, None)

        assert logger.last_call_details["api_base"] == "https://api.async.com"
        assert logger.last_call_details["cache_hit"] is False

    def test_log_failure_event(self):
        """Test logging failed LLM call."""
        logger = MarkitaiLLMLogger()

        kwargs = {
            "standard_logging_object": {
                "api_base": "https://api.openai.com",
                "error_code": 429,
                "error_class": "RateLimitError",
            },
        }
        response_obj = MagicMock()

        logger.log_failure_event(kwargs, response_obj, None, None)

        assert logger.last_call_details["api_base"] == "https://api.openai.com"
        assert logger.last_call_details["error_code"] == 429
        assert logger.last_call_details["error_class"] == "RateLimitError"

    def test_log_failure_event_empty_slo(self):
        """Test logging failure with empty standard_logging_object."""
        logger = MarkitaiLLMLogger()

        kwargs = {}
        response_obj = MagicMock()

        logger.log_failure_event(kwargs, response_obj, None, None)

        assert logger.last_call_details["api_base"] is None
        assert logger.last_call_details["error_code"] is None
        assert logger.last_call_details["error_class"] is None

    @pytest.mark.asyncio
    async def test_async_log_failure_event(self):
        """Test async version of failure event logging."""
        logger = MarkitaiLLMLogger()

        kwargs = {
            "standard_logging_object": {
                "api_base": "https://api.example.com",
                "error_code": 500,
                "error_class": "InternalServerError",
            },
        }
        response_obj = MagicMock()

        await logger.async_log_failure_event(kwargs, response_obj, None, None)

        assert logger.last_call_details["error_code"] == 500
        assert logger.last_call_details["error_class"] == "InternalServerError"

    def test_success_overwrites_previous_details(self):
        """Test that new calls overwrite previous call details."""
        logger = MarkitaiLLMLogger()

        # First call
        logger.log_success_event(
            {
                "standard_logging_object": {"api_base": "first"},
                "cache_hit": True,
            },
            MagicMock(),
            None,
            None,
        )
        assert logger.last_call_details["api_base"] == "first"

        # Second call
        logger.log_success_event(
            {
                "standard_logging_object": {"api_base": "second"},
                "cache_hit": False,
            },
            MagicMock(),
            None,
            None,
        )
        assert logger.last_call_details["api_base"] == "second"
        assert logger.last_call_details["cache_hit"] is False


class TestModelInfoCacheIsolation:
    """Tests for cache isolation between tests."""

    def setup_method(self):
        """Clear cache before each test."""
        _model_info_cache.clear()

    def test_cache_starts_empty(self):
        """Test that cache is empty at test start."""
        # After setup_method clears it, should be empty
        assert len(_model_info_cache) == 0

    def test_cache_entries_persist_within_test(self):
        """Test that cache entries persist within a single test."""
        _model_info_cache["model-a"] = {"max_input_tokens": 1000}
        _model_info_cache["model-b"] = {"max_input_tokens": 2000}

        assert len(_model_info_cache) == 2
        assert _model_info_cache["model-a"]["max_input_tokens"] == 1000
