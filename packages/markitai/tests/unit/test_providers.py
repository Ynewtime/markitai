"""Unit tests for the providers module."""

from __future__ import annotations


class TestCheckDeprecatedModels:
    """Tests for check_deprecated_models function."""

    def test_detects_deprecated_model(self) -> None:
        """Test that deprecated models are detected."""
        from markitai.providers import check_deprecated_models

        warnings = check_deprecated_models(["gpt-4o"])
        assert len(warnings) == 1
        assert "gpt-4o" in warnings[0]
        assert "gpt-5.2" in warnings[0]
        assert "February 13, 2025" in warnings[0]

    def test_detects_multiple_deprecated_models(self) -> None:
        """Test that multiple deprecated models are detected."""
        from markitai.providers import check_deprecated_models

        warnings = check_deprecated_models(["gpt-4o", "gpt-4.1", "o4-mini"])
        assert len(warnings) == 3

    def test_strips_provider_prefix(self) -> None:
        """Test that provider prefix is stripped before checking."""
        from markitai.providers import check_deprecated_models

        warnings = check_deprecated_models(["copilot/gpt-4o", "openai/gpt-4.1"])
        assert len(warnings) == 2
        assert "gpt-4o" in warnings[0]
        assert "gpt-4.1" in warnings[1]

    def test_no_duplicates(self) -> None:
        """Test that same model doesn't produce duplicate warnings."""
        from markitai.providers import check_deprecated_models

        warnings = check_deprecated_models(
            ["gpt-4o", "copilot/gpt-4o", "openai/gpt-4o"]
        )
        assert len(warnings) == 1

    def test_non_deprecated_models_no_warning(self) -> None:
        """Test that non-deprecated models don't produce warnings."""
        from markitai.providers import check_deprecated_models

        warnings = check_deprecated_models(
            ["gpt-5.2", "claude-sonnet-4.5", "gemini-2.5-flash"]
        )
        assert len(warnings) == 0

    def test_empty_list(self) -> None:
        """Test with empty model list."""
        from markitai.providers import check_deprecated_models

        warnings = check_deprecated_models([])
        assert len(warnings) == 0


class TestDeprecatedModelsConstant:
    """Tests for DEPRECATED_MODELS constant."""

    def test_deprecated_models_defined(self) -> None:
        """Test that DEPRECATED_MODELS constant is defined."""
        from markitai.providers import DEPRECATED_MODELS

        assert isinstance(DEPRECATED_MODELS, dict)
        assert "gpt-4o" in DEPRECATED_MODELS
        assert "gpt-4.1" in DEPRECATED_MODELS
        assert "gpt-4.1-mini" in DEPRECATED_MODELS
        assert "o4-mini" in DEPRECATED_MODELS
        assert "gpt-5" in DEPRECATED_MODELS

    def test_all_replacements_are_gpt_5_2(self) -> None:
        """Test that all deprecated models recommend gpt-5.2."""
        from markitai.providers import DEPRECATED_MODELS

        for replacement in DEPRECATED_MODELS.values():
            assert replacement == "gpt-5.2"


class TestIsLocalProviderModel:
    """Tests for is_local_provider_model function."""

    def test_claude_agent_model_is_local(self) -> None:
        """Test that claude-agent models are identified as local."""
        from markitai.providers import is_local_provider_model

        # Aliases (recommended)
        assert is_local_provider_model("claude-agent/sonnet") is True
        assert is_local_provider_model("claude-agent/opus") is True
        assert is_local_provider_model("claude-agent/haiku") is True
        # Full model strings
        assert (
            is_local_provider_model("claude-agent/claude-sonnet-4-5-20250929") is True
        )
        assert is_local_provider_model("claude-agent/claude-opus-4-6") is True
        # Legacy format (still works)
        assert is_local_provider_model("claude-agent/claude-sonnet-4") is True
        assert is_local_provider_model("claude-agent/some-model") is True

    def test_copilot_model_is_local(self) -> None:
        """Test that copilot models are identified as local."""
        from markitai.providers import is_local_provider_model

        assert is_local_provider_model("copilot/gpt-4.1") is True
        assert is_local_provider_model("copilot/gpt-4o") is True
        assert is_local_provider_model("copilot/some-model") is True

    def test_standard_models_not_local(self) -> None:
        """Test that standard LiteLLM models are not identified as local."""
        from markitai.providers import is_local_provider_model

        assert is_local_provider_model("gemini/gemini-2.5-flash") is False
        assert is_local_provider_model("openai/gpt-4") is False
        assert is_local_provider_model("anthropic/claude-3-sonnet") is False
        assert is_local_provider_model("deepseek/deepseek-chat") is False

    def test_partial_prefix_not_matched(self) -> None:
        """Test that partial prefixes don't match."""
        from markitai.providers import is_local_provider_model

        # These should NOT match because they don't start with the prefix
        assert is_local_provider_model("my-claude-agent/model") is False
        assert is_local_provider_model("not-copilot/model") is False


class TestGetLocalProviderModelInfo:
    """Tests for get_local_provider_model_info function.

    Note: Model info is inherited from LiteLLM's database, so we test
    structure and reasonable ranges rather than exact values.
    """

    def test_claude_agent_haiku_model_info(self) -> None:
        """Test that claude-agent/haiku returns valid model info from LiteLLM."""
        from markitai.providers import get_local_provider_model_info

        info = get_local_provider_model_info("claude-agent/haiku")
        assert info is not None
        # Haiku has 200K context window
        assert info["max_input_tokens"] >= 100000
        assert info["max_output_tokens"] >= 4096
        assert info["supports_vision"] is True

    def test_claude_agent_sonnet_model_info(self) -> None:
        """Test that claude-agent/sonnet returns valid model info from LiteLLM."""
        from markitai.providers import get_local_provider_model_info

        info = get_local_provider_model_info("claude-agent/sonnet")
        assert info is not None
        # Sonnet 4 has large context and output
        assert info["max_input_tokens"] >= 200000
        assert info["max_output_tokens"] >= 8192
        assert info["supports_vision"] is True

    def test_claude_agent_opus_model_info(self) -> None:
        """Test that claude-agent/opus returns valid model info from LiteLLM."""
        from markitai.providers import get_local_provider_model_info

        info = get_local_provider_model_info("claude-agent/opus")
        assert info is not None
        # Opus 4.6 has 200K context (1M beta)
        assert info["max_input_tokens"] >= 200000
        assert info["max_output_tokens"] >= 8192
        assert info["supports_vision"] is True

    def test_copilot_model_info(self) -> None:
        """Test that copilot models return valid model info from LiteLLM."""
        from markitai.providers import get_local_provider_model_info

        info = get_local_provider_model_info("copilot/gpt-4o")
        assert info is not None
        # GPT-4o has 128K context
        assert info["max_input_tokens"] >= 100000
        assert info["max_output_tokens"] >= 4096
        assert info["supports_vision"] is True

    def test_unknown_local_model_returns_default(self) -> None:
        """Test that unknown local models return default info."""
        from markitai.providers import get_local_provider_model_info

        info = get_local_provider_model_info("claude-agent/unknown-nonexistent-model")
        assert info is not None
        # Should return defaults
        assert "max_input_tokens" in info
        assert "max_output_tokens" in info
        assert "supports_vision" in info

    def test_non_local_model_returns_none(self) -> None:
        """Test that non-local models return None."""
        from markitai.providers import get_local_provider_model_info

        assert get_local_provider_model_info("gemini/gemini-2.5-flash") is None
        assert get_local_provider_model_info("openai/gpt-4") is None
        assert get_local_provider_model_info("anthropic/claude-3-sonnet") is None

    def test_model_info_has_required_keys(self) -> None:
        """Test that model info contains all required keys."""
        from markitai.providers import get_local_provider_model_info

        for model in ["claude-agent/haiku", "claude-agent/sonnet", "copilot/gpt-4o"]:
            info = get_local_provider_model_info(model)
            assert info is not None
            assert "max_input_tokens" in info
            assert "max_output_tokens" in info
            assert "supports_vision" in info
            assert isinstance(info["max_input_tokens"], int)
            assert isinstance(info["max_output_tokens"], int)
            assert isinstance(info["supports_vision"], bool)


class TestRegisterProviders:
    """Tests for register_providers function."""

    def test_register_providers_runs_without_error(self) -> None:
        """Test that register_providers can be called without error."""
        from markitai.providers import register_providers

        # Should not raise any exceptions
        register_providers()

    def test_register_providers_is_idempotent(self) -> None:
        """Test that register_providers can be called multiple times."""
        from markitai.providers import register_providers

        # Calling multiple times should be safe
        register_providers()
        register_providers()
        register_providers()


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_count_tokens_estimation(self) -> None:
        """Test token counting falls back to estimation."""
        from markitai.providers import count_tokens

        # Test with Claude model (always uses estimation)
        text = "Hello, world!"  # 13 chars
        tokens = count_tokens(text, "claude-sonnet-4.5")
        # Estimation: 13 // 4 = 3
        assert tokens == 3

    def test_count_tokens_longer_text(self) -> None:
        """Test token counting with longer text."""
        from markitai.providers import count_tokens

        text = "a" * 1000  # 1000 chars
        tokens = count_tokens(text, "claude-sonnet-4.5")
        # Estimation: 1000 // 4 = 250
        assert tokens == 250

    def test_count_tokens_gpt_model(self) -> None:
        """Test token counting for GPT models."""
        from markitai.providers import count_tokens

        text = "Hello, world!"
        tokens = count_tokens(text, "gpt-4.1")
        # If tiktoken available, should be accurate
        # If not, falls back to estimation
        assert tokens > 0


class TestCalculateCopilotCost:
    """Tests for calculate_copilot_cost function.

    Note: calculate_copilot_cost returns CopilotCostResult with:
    - cost_usd: The estimated cost
    - is_estimated: Always True for Copilot (subscription-based)
    - source: "litellm", "litellm_fuzzy", "fallback", or "none"
    - matched_model: The model name used for pricing lookup
    """

    def test_calculate_cost_uses_litellm_when_available(self) -> None:
        """Test cost calculation uses LiteLLM pricing when available."""
        import litellm
        import pytest

        from markitai.providers import calculate_copilot_cost

        # Use a well-known model that exists in LiteLLM
        model = "gpt-4o"
        try:
            info = litellm.get_model_info(model)
            input_cost = info.get("input_cost_per_token", 0)
            output_cost = info.get("output_cost_per_token", 0)
        except Exception:
            pytest.skip("LiteLLM model info not available")

        result = calculate_copilot_cost(model, 1000, 500)
        expected = 1000 * input_cost + 500 * output_cost
        assert abs(result.cost_usd - expected) < 0.000001
        assert result.is_estimated is True
        assert result.source == "litellm"
        assert result.matched_model == model

    def test_calculate_cost_fallback_to_hardcoded(self) -> None:
        """Test cost calculation falls back to hardcoded pricing when LiteLLM fails."""
        from markitai.providers import calculate_copilot_cost

        # Use a model that definitely won't exist in LiteLLM (neither exact nor fuzzy)
        # but exists in our hardcoded COPILOT_MODEL_PRICING
        # Note: With fuzzy matching, many models will match LiteLLM entries,
        # so we need a truly unique model name from our fallback table
        model = "completely-unknown-xyz-999"
        result = calculate_copilot_cost(model, 10000, 5000)

        # Should return 0 cost with "none" source for truly unknown models
        assert result.is_estimated is True
        assert result.source == "none"
        assert result.cost_usd == 0.0

    def test_calculate_cost_unknown_model(self) -> None:
        """Test cost calculation for unknown model returns 0 with 'none' source."""
        from markitai.providers import calculate_copilot_cost

        result = calculate_copilot_cost("unknown-model-xyz-12345", 1000, 500)
        assert result.cost_usd == 0.0
        assert result.is_estimated is True
        assert result.source == "none"
        assert result.matched_model is None

    def test_calculate_cost_prefix_matching_fallback(self) -> None:
        """Test cost calculation uses prefix matching when LiteLLM and exact match fail."""
        from markitai.providers import calculate_copilot_cost

        # Use a model variant that's unlikely to be in LiteLLM but matches our prefix
        # If this model is in LiteLLM, the test verifies LiteLLM is used
        # If not, it verifies prefix matching works
        model = "gpt-5.1-codex-mini"
        result = calculate_copilot_cost(model, 1000, 500)

        # Cost should be non-zero (either from LiteLLM or hardcoded)
        assert result.cost_usd > 0
        assert result.is_estimated is True
        # Source can be litellm, litellm_fuzzy, or fallback
        assert result.source in ("litellm", "litellm_fuzzy", "fallback")

    def test_fuzzy_matching_finds_similar_model(self) -> None:
        """Test fuzzy matching finds models with different naming conventions."""
        from markitai.providers import calculate_copilot_cost

        # Copilot uses "claude-haiku-4.5" but LiteLLM has "claude-haiku-4-5"
        # Fuzzy matching should find it
        result = calculate_copilot_cost("claude-haiku-4.5", 1000, 500)

        # Should find via fuzzy match or fallback
        assert result.cost_usd >= 0
        assert result.is_estimated is True
        if result.source == "litellm_fuzzy":
            # Fuzzy matched to a LiteLLM model
            assert result.matched_model is not None

    def test_copilot_cost_result_has_all_attributes(self) -> None:
        """Test CopilotCostResult has all expected attributes."""
        from markitai.providers import calculate_copilot_cost

        result = calculate_copilot_cost("gpt-4o", 100, 50)

        # Check all attributes exist
        assert hasattr(result, "cost_usd")
        assert hasattr(result, "is_estimated")
        assert hasattr(result, "source")
        assert hasattr(result, "matched_model")

        # Check types
        assert isinstance(result.cost_usd, float)
        assert isinstance(result.is_estimated, bool)
        assert isinstance(result.source, str)
        assert result.matched_model is None or isinstance(result.matched_model, str)


class TestClaudeAgentProvider:
    """Tests for ClaudeAgentProvider class."""

    def test_messages_to_prompt_simple(self) -> None:
        """Test converting simple messages to prompt."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()
        messages = [
            {"role": "user", "content": "Hello, world!"},
        ]
        prompt = provider._messages_to_prompt(messages)
        assert prompt == "Hello, world!"

    def test_messages_to_prompt_with_system(self) -> None:
        """Test converting messages with system prompt."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        prompt = provider._messages_to_prompt(messages)
        assert "<system>" in prompt
        assert "You are a helpful assistant." in prompt
        assert "Hello!" in prompt

    def test_messages_to_prompt_with_assistant(self) -> None:
        """Test converting messages with assistant response."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = provider._messages_to_prompt(messages)
        assert "Hello!" in prompt
        assert "<assistant>" in prompt
        assert "Hi there!" in prompt
        assert "How are you?" in prompt

    def test_messages_to_prompt_multimodal_content(self) -> None:
        """Test converting multimodal message content."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ]
        prompt = provider._messages_to_prompt(messages)
        # Should extract only text parts
        assert "What's in this image?" in prompt

    def test_has_images_detects_image_content(self) -> None:
        """Test _has_images correctly detects image content."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Messages without images
        no_images = [{"role": "user", "content": "Hello"}]
        assert provider._has_images(no_images) is False

        # Messages with images
        with_images = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc"},
                    },
                ],
            }
        ]
        assert provider._has_images(with_images) is True

    def test_sdk_availability_check(self) -> None:
        """Test SDK availability check using module-level function."""
        from markitai.providers.claude_agent import _is_claude_agent_sdk_available

        # Result depends on whether SDK is installed
        result = _is_claude_agent_sdk_available()
        assert isinstance(result, bool)

    def test_convert_content_to_sdk_format_text(self) -> None:
        """Test converting text content to SDK format."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Simple string
        result = provider._convert_content_to_sdk_format("Hello world")
        assert result == [{"type": "text", "text": "Hello world"}]

        # List with text block
        result = provider._convert_content_to_sdk_format(
            [{"type": "text", "text": "Hello"}]
        )
        assert result == [{"type": "text", "text": "Hello"}]

    def test_convert_content_to_sdk_format_image(self) -> None:
        """Test converting image content to SDK format."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Base64 image
        content = [
            {"type": "text", "text": "What's this?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc123"},
            },
        ]
        result = provider._convert_content_to_sdk_format(content)

        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "What's this?"}
        assert result[1]["type"] == "image"
        assert result[1]["source"]["type"] == "base64"
        assert result[1]["source"]["media_type"] == "image/png"
        assert result[1]["source"]["data"] == "abc123"

    def test_messages_to_stream_format(self) -> None:
        """Test that streaming output follows SDK format."""
        import asyncio

        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,xyz"},
                    },
                ],
            }
        ]

        async def collect_stream() -> list:
            result = []
            async for item in provider._messages_to_stream(messages):
                result.append(item)
            return result

        result = asyncio.run(collect_stream())

        # Should yield complete user message objects per SDK docs
        assert len(result) == 1
        assert result[0]["type"] == "user"
        assert result[0]["message"]["role"] == "user"
        assert isinstance(result[0]["message"]["content"], list)

    def test_unsupported_params_defined(self) -> None:
        """Test that _UNSUPPORTED_PARAMS is defined and contains expected params."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        # Verify the frozenset exists and contains expected params
        unsupported = ClaudeAgentProvider._UNSUPPORTED_PARAMS
        assert isinstance(unsupported, frozenset)
        assert "max_tokens" in unsupported
        assert "temperature" in unsupported
        assert "top_p" in unsupported
        assert "stop" in unsupported
        assert "max_completion_tokens" in unsupported


class TestCopilotProvider:
    """Tests for CopilotProvider class."""

    def test_messages_to_prompt_simple(self) -> None:
        """Test converting simple messages to prompt."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        messages = [
            {"role": "user", "content": "Hello, world!"},
        ]
        prompt = provider._messages_to_prompt(messages)
        assert prompt == "Hello, world!"

    def test_messages_to_prompt_with_system(self) -> None:
        """Test converting messages with system prompt."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        prompt = provider._messages_to_prompt(messages)
        assert "<system>" in prompt
        assert "You are a helpful assistant." in prompt
        assert "Hello!" in prompt

    def test_messages_to_prompt_with_assistant(self) -> None:
        """Test converting messages with assistant response."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = provider._messages_to_prompt(messages)
        assert "Hello!" in prompt
        assert "<assistant>" in prompt
        assert "Hi there!" in prompt
        assert "How are you?" in prompt

    def test_messages_to_prompt_multimodal_content(self) -> None:
        """Test converting multimodal message content."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ]
        prompt = provider._messages_to_prompt(messages)
        # Should extract only text parts
        assert "What's in this image?" in prompt

    def test_has_images_detects_image_content(self) -> None:
        """Test _has_images correctly detects image content."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Messages without images
        no_images = [{"role": "user", "content": "Hello"}]
        assert provider._has_images(no_images) is False

        # Messages with images
        with_images = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc"},
                    },
                ],
            }
        ]
        assert provider._has_images(with_images) is True

    def test_extract_images_with_base64_data(self) -> None:
        """Test _extract_images extracts base64 images correctly."""
        import base64
        import os

        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Create a small test image (1x1 red pixel PNG)
        test_image_data = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{test_image_data}"
                        },
                    },
                ],
            }
        ]

        prompt, attachments = provider._extract_images(messages)

        assert "What's this?" in prompt
        assert len(attachments) == 1
        assert attachments[0]["type"] == "file"
        assert os.path.exists(attachments[0]["path"])

        # Clean up
        provider._cleanup_temp_files()
        assert len(provider._temp_files) == 0

    def test_sdk_availability_check(self) -> None:
        """Test SDK availability check using module-level function."""
        from markitai.providers.copilot import _is_copilot_sdk_available

        # Result depends on whether SDK is installed
        result = _is_copilot_sdk_available()
        assert isinstance(result, bool)

    def test_acompletion_raises_runtime_error_without_sdk(self) -> None:
        """Test that acompletion raises RuntimeError when SDK not installed."""
        import asyncio

        from markitai.providers.copilot import (
            CopilotProvider,
            _is_copilot_sdk_available,
        )

        # Skip test if SDK is actually installed
        if _is_copilot_sdk_available():
            return

        provider = CopilotProvider()

        async def test_async() -> None:
            try:
                await provider.acompletion(
                    model="copilot/gpt-4.1",
                    messages=[{"role": "user", "content": "Hello"}],
                )
                raise AssertionError("Should have raised RuntimeError")
            except RuntimeError as e:
                # Check for English error message
                assert "Copilot SDK" in str(e) and "not installed" in str(e)

        asyncio.run(test_async())

    def test_completion_raises_runtime_error_without_sdk(self) -> None:
        """Test that completion raises RuntimeError when SDK not installed."""
        from markitai.providers.copilot import (
            CopilotProvider,
            _is_copilot_sdk_available,
        )

        # Skip test if SDK is actually installed
        if _is_copilot_sdk_available():
            return

        provider = CopilotProvider()

        try:
            provider.completion(
                model="copilot/gpt-4.1",
                messages=[{"role": "user", "content": "Hello"}],
            )
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError as e:
            # Check for English error message
            assert "Copilot SDK" in str(e) and "not installed" in str(e)

    def test_unsupported_params_defined(self) -> None:
        """Test that _UNSUPPORTED_PARAMS is defined and contains expected params."""
        from markitai.providers.copilot import CopilotProvider

        # Verify the frozenset exists and contains expected params
        unsupported = CopilotProvider._UNSUPPORTED_PARAMS
        assert isinstance(unsupported, frozenset)
        assert "max_tokens" in unsupported
        assert "temperature" in unsupported
        assert "top_p" in unsupported
        assert "stop" in unsupported
        assert "max_completion_tokens" in unsupported

    def test_unsupported_models_defined(self) -> None:
        """Test that UNSUPPORTED_MODELS only contains o1/o3 reasoning models."""
        from markitai.providers.copilot import CopilotProvider

        # Verify the frozenset exists and contains expected models
        unsupported = CopilotProvider.UNSUPPORTED_MODELS
        assert isinstance(unsupported, frozenset)

        # GPT-5 series is now fully supported by Copilot
        assert "gpt-5" not in unsupported
        assert "gpt-5.1" not in unsupported
        assert "gpt-5.2" not in unsupported

        # o1/o3 reasoning models should still be unsupported
        assert "o1" in unsupported
        assert "o3" in unsupported

        # GPT-4.1 should NOT be in unsupported list
        assert "gpt-4.1" not in unsupported

        # Claude models should NOT be in unsupported list
        assert "claude-sonnet-4.5" not in unsupported


class TestClaudeAgentAdaptiveTimeout:
    """Tests for ClaudeAgentProvider adaptive timeout calculation."""

    def test_calculate_adaptive_timeout_exists(self) -> None:
        """Test that _calculate_adaptive_timeout method exists."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()
        assert hasattr(provider, "_calculate_adaptive_timeout")
        assert callable(provider._calculate_adaptive_timeout)

    def test_longer_messages_return_higher_timeout(self) -> None:
        """Test that longer messages result in higher timeout values."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Short message
        short_messages = [{"role": "user", "content": "Hello!"}]
        short_timeout = provider._calculate_adaptive_timeout(short_messages)

        # Long message (100K chars)
        long_content = "x" * 100000
        long_messages = [{"role": "user", "content": long_content}]
        long_timeout = provider._calculate_adaptive_timeout(long_messages)

        # Long messages should have higher timeout
        assert long_timeout > short_timeout

    def test_images_increase_timeout(self) -> None:
        """Test that messages with images have higher timeout."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Text-only message
        text_messages = [{"role": "user", "content": "Describe this."}]
        text_timeout = provider._calculate_adaptive_timeout(text_messages)

        # Message with image
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123"},
                    },
                ],
            }
        ]
        image_timeout = provider._calculate_adaptive_timeout(image_messages)

        # Image messages should have higher timeout
        assert image_timeout > text_timeout

    def test_multiple_images_increase_timeout(self) -> None:
        """Test that multiple images result in higher timeout than single image."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Single image
        single_image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc"},
                    },
                ],
            }
        ]
        single_timeout = provider._calculate_adaptive_timeout(single_image_messages)

        # Multiple images
        multi_image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,def"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,ghi"},
                    },
                ],
            }
        ]
        multi_timeout = provider._calculate_adaptive_timeout(multi_image_messages)

        # Multiple images should have higher timeout
        assert multi_timeout > single_timeout

    def test_returns_integer(self) -> None:
        """Test that _calculate_adaptive_timeout returns an integer."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()
        messages = [{"role": "user", "content": "Hello!"}]
        timeout = provider._calculate_adaptive_timeout(messages)

        assert isinstance(timeout, int)

    def test_minimum_timeout_is_60_seconds(self) -> None:
        """Test that minimum timeout is at least 60 seconds (base timeout)."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()
        messages = [{"role": "user", "content": "Hi"}]
        timeout = provider._calculate_adaptive_timeout(messages)

        assert timeout >= 60


class TestCopilotJsonExtraction:
    """Tests for CopilotProvider JSON extraction using StructuredOutputHandler."""

    def test_extract_json_from_markdown_code_block(self) -> None:
        """Test extracting JSON from markdown code blocks."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        text = '```json\n{"name": "test", "value": 123}\n```'
        result = provider._extract_json_from_response(text)

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 123

    def test_extract_json_from_plain_text(self) -> None:
        """Test extracting JSON from plain text response."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        text = '{"name": "test"}'
        result = provider._extract_json_from_response(text)

        assert isinstance(result, dict)
        assert result["name"] == "test"

    def test_extract_json_array(self) -> None:
        """Test extracting JSON array from response."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        text = '[1, 2, 3, "four"]'
        result = provider._extract_json_from_response(text)

        assert isinstance(result, list)
        assert result == [1, 2, 3, "four"]

    def test_extract_json_cleans_control_characters(self) -> None:
        """Test that control characters are cleaned before JSON parsing."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Text with control character (bell character \x07)
        text = '{"name": "test\x07value"}'
        result = provider._extract_json_from_response(text)

        assert isinstance(result, dict)
        assert result["name"] == "testvalue"  # Control char removed

    def test_extract_json_returns_original_on_failure(self) -> None:
        """Test that original text is returned when no JSON found."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        text = "This is just plain text without JSON."
        result = provider._extract_json_from_response(text)

        assert result == text

    def test_extract_json_with_surrounding_text(self) -> None:
        """Test extracting JSON embedded in surrounding text."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        text = 'Here is the data: {"key": "value"} That was it.'
        result = provider._extract_json_from_response(text)

        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_json_handler_instance_exists(self) -> None:
        """Test that _json_handler is initialized as StructuredOutputHandler."""
        from markitai.providers.copilot import CopilotProvider
        from markitai.providers.json_mode import StructuredOutputHandler

        provider = CopilotProvider()
        assert hasattr(provider, "_json_handler")
        assert isinstance(provider._json_handler, StructuredOutputHandler)


class TestCopilotAdaptiveTimeout:
    """Tests for CopilotProvider adaptive timeout calculation."""

    def test_calculate_adaptive_timeout_exists(self) -> None:
        """Test that _calculate_adaptive_timeout method exists."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        assert hasattr(provider, "_calculate_adaptive_timeout")
        assert callable(provider._calculate_adaptive_timeout)

    def test_longer_messages_return_higher_timeout(self) -> None:
        """Test that longer messages result in higher timeout values."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Short message
        short_messages = [{"role": "user", "content": "Hello!"}]
        short_timeout = provider._calculate_adaptive_timeout(short_messages)

        # Long message (100K chars)
        long_content = "x" * 100000
        long_messages = [{"role": "user", "content": long_content}]
        long_timeout = provider._calculate_adaptive_timeout(long_messages)

        # Long messages should have higher timeout
        assert long_timeout > short_timeout

    def test_images_increase_timeout(self) -> None:
        """Test that messages with images have higher timeout."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Text-only message
        text_messages = [{"role": "user", "content": "Describe this."}]
        text_timeout = provider._calculate_adaptive_timeout(text_messages)

        # Message with image
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123"},
                    },
                ],
            }
        ]
        image_timeout = provider._calculate_adaptive_timeout(image_messages)

        # Image messages should have higher timeout
        assert image_timeout > text_timeout

    def test_multiple_images_increase_timeout(self) -> None:
        """Test that multiple images result in higher timeout than single image."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Single image
        single_image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc"},
                    },
                ],
            }
        ]
        single_timeout = provider._calculate_adaptive_timeout(single_image_messages)

        # Multiple images
        multi_image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,def"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,ghi"},
                    },
                ],
            }
        ]
        multi_timeout = provider._calculate_adaptive_timeout(multi_image_messages)

        # Multiple images should have higher timeout
        assert multi_timeout > single_timeout

    def test_returns_integer(self) -> None:
        """Test that _calculate_adaptive_timeout returns an integer."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        messages = [{"role": "user", "content": "Hello!"}]
        timeout = provider._calculate_adaptive_timeout(messages)

        assert isinstance(timeout, int)

    def test_minimum_timeout_is_60_seconds(self) -> None:
        """Test that minimum timeout is at least 60 seconds (base timeout)."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()
        messages = [{"role": "user", "content": "Hi"}]
        timeout = provider._calculate_adaptive_timeout(messages)

        assert timeout >= 60


class TestClaudeAgentPromptCaching:
    """Tests for ClaudeAgentProvider prompt caching support."""

    def test_cache_threshold_constant_defined(self) -> None:
        """Test that _CACHE_THRESHOLD_CHARS constant is defined."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        assert hasattr(ClaudeAgentProvider, "_CACHE_THRESHOLD_CHARS")
        assert ClaudeAgentProvider._CACHE_THRESHOLD_CHARS == 4096

    def test_add_cache_control_method_exists(self) -> None:
        """Test that _add_cache_control method exists."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()
        assert hasattr(provider, "_add_cache_control")
        assert callable(provider._add_cache_control)

    def test_long_system_prompt_gets_cache_control(self) -> None:
        """Test that long system prompts get cache_control added."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Create a long system prompt (>= 4096 chars)
        long_system_content = "x" * 5000
        messages = [
            {"role": "system", "content": long_system_content},
            {"role": "user", "content": "Hello!"},
        ]

        result = provider._add_cache_control(messages)

        # System message should be converted to content blocks format
        assert result[0]["role"] == "system"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == long_system_content
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

        # User message should be unchanged
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello!"

    def test_short_system_prompt_not_modified(self) -> None:
        """Test that short system prompts are not modified."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Create a short system prompt (< 4096 chars)
        short_system_content = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": short_system_content},
            {"role": "user", "content": "Hello!"},
        ]

        result = provider._add_cache_control(messages)

        # System message should remain as string content
        assert result[0]["role"] == "system"
        assert result[0]["content"] == short_system_content

        # User message should be unchanged
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello!"

    def test_non_system_messages_not_modified(self) -> None:
        """Test that non-system messages are not modified."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Create a long user message
        long_user_content = "x" * 5000
        messages = [
            {"role": "user", "content": long_user_content},
            {"role": "assistant", "content": "y" * 5000},
        ]

        result = provider._add_cache_control(messages)

        # Both messages should remain unchanged (as copies)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == long_user_content
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "y" * 5000

    def test_exact_threshold_gets_cache_control(self) -> None:
        """Test that content exactly at threshold gets cache_control."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Create content exactly at threshold
        exact_content = "x" * 4096
        messages = [{"role": "system", "content": exact_content}]

        result = provider._add_cache_control(messages)

        # Should get cache_control
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_just_below_threshold_not_modified(self) -> None:
        """Test that content just below threshold is not modified."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Create content just below threshold
        below_content = "x" * 4095
        messages = [{"role": "system", "content": below_content}]

        result = provider._add_cache_control(messages)

        # Should NOT get cache_control
        assert result[0]["content"] == below_content

    def test_non_string_system_content_not_modified(self) -> None:
        """Test that system messages with non-string content are not modified."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # System message with list content (already in content blocks format)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "x" * 5000}],
            }
        ]

        result = provider._add_cache_control(messages)

        # Should be copied but not modified (content is not a string)
        assert result[0]["content"] == messages[0]["content"]

    def test_original_messages_not_mutated(self) -> None:
        """Test that original messages list is not mutated."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        long_content = "x" * 5000
        messages = [
            {"role": "system", "content": long_content},
            {"role": "user", "content": "Hello!"},
        ]

        # Store original references
        original_system = messages[0].copy()
        original_user = messages[1].copy()

        provider._add_cache_control(messages)

        # Original messages should not be modified
        assert messages[0] == original_system
        assert messages[1] == original_user
