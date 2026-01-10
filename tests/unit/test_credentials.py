"""Tests for shared credential utilities."""

from markit.cli.shared.credentials import get_effective_api_key, get_unique_credentials
from markit.config.settings import (
    LLMConfig,
    LLMCredentialConfig,
    LLMProviderConfig,
    MarkitSettings,
)


class TestGetEffectiveApiKey:
    """Tests for get_effective_api_key function."""

    def test_returns_direct_api_key(self):
        """Test that direct api_key is returned when present."""
        cred = LLMCredentialConfig(
            id="test",
            provider="openai",
            api_key="sk-direct-key",
        )
        result = get_effective_api_key(cred)
        assert result == "sk-direct-key"

    def test_returns_key_from_env_var(self, monkeypatch):
        """Test that api_key_env resolves to environment variable."""
        monkeypatch.setenv("MY_CUSTOM_KEY", "sk-from-env")
        cred = LLMCredentialConfig(
            id="test",
            provider="openai",
            api_key_env="MY_CUSTOM_KEY",
        )
        result = get_effective_api_key(cred)
        assert result == "sk-from-env"

    def test_direct_key_takes_precedence(self, monkeypatch):
        """Test that direct api_key takes precedence over api_key_env."""
        monkeypatch.setenv("MY_CUSTOM_KEY", "sk-from-env")
        cred = LLMCredentialConfig(
            id="test",
            provider="openai",
            api_key="sk-direct-key",
            api_key_env="MY_CUSTOM_KEY",
        )
        result = get_effective_api_key(cred)
        assert result == "sk-direct-key"

    def test_falls_back_to_provider_default_env(self, monkeypatch):
        """Test fallback to provider-specific default env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-default-openai")
        cred = LLMCredentialConfig(
            id="test",
            provider="openai",
        )
        result = get_effective_api_key(cred)
        assert result == "sk-default-openai"

    def test_returns_none_when_no_key_available(self, monkeypatch):
        """Test that None is returned when no key is available."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cred = LLMCredentialConfig(
            id="test",
            provider="openai",
        )
        result = get_effective_api_key(cred)
        assert result is None

    def test_works_with_legacy_provider_config(self, monkeypatch):
        """Test that it works with legacy LLMProviderConfig."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
        config = LLMProviderConfig(
            provider="anthropic",
            model="claude-3",
        )
        result = get_effective_api_key(config)
        assert result == "sk-anthropic"

    def test_legacy_config_with_direct_key(self):
        """Test legacy config with direct api_key."""
        config = LLMProviderConfig(
            provider="openai",
            model="gpt-4",
            api_key="sk-legacy-key",
        )
        result = get_effective_api_key(config)
        assert result == "sk-legacy-key"


class TestGetUniqueCredentials:
    """Tests for get_unique_credentials function."""

    def test_returns_empty_list_for_no_credentials(self):
        """Test that empty list is returned when no credentials configured."""
        settings = MarkitSettings(llm=LLMConfig())
        result = get_unique_credentials(settings)
        assert result == []

    def test_returns_credentials_from_new_schema(self, monkeypatch):
        """Test that credentials are returned from new schema."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        settings = MarkitSettings(
            llm=LLMConfig(
                credentials=[
                    LLMCredentialConfig(id="openai-main", provider="openai"),
                ]
            )
        )
        result = get_unique_credentials(settings)

        assert len(result) == 1
        provider, api_key, base_url, display_name, cred_id = result[0]
        assert provider == "openai"
        assert api_key == "sk-openai"
        assert base_url is None
        assert display_name == "openai-main"
        assert cred_id == "openai-main"

    def test_returns_credentials_with_base_url(self, monkeypatch):
        """Test that base_url is included in credentials."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        settings = MarkitSettings(
            llm=LLMConfig(
                credentials=[
                    LLMCredentialConfig(
                        id="deepseek",
                        provider="openai",
                        base_url="https://api.deepseek.com",
                        api_key_env="DEEPSEEK_API_KEY",
                    ),
                ]
            )
        )
        result = get_unique_credentials(settings)

        assert len(result) == 1
        provider, api_key, base_url, display_name, cred_id = result[0]
        assert provider == "openai"
        assert api_key == "sk-deepseek"
        assert base_url == "https://api.deepseek.com"
        assert display_name == "deepseek"

    def test_returns_legacy_providers(self, monkeypatch):
        """Test that legacy providers are also returned."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
        settings = MarkitSettings(
            llm=LLMConfig(
                providers=[
                    LLMProviderConfig(
                        provider="anthropic",
                        model="claude-3",
                        name="Claude Provider",
                    ),
                ]
            )
        )
        result = get_unique_credentials(settings)

        assert len(result) == 1
        provider, api_key, base_url, display_name, cred_id = result[0]
        assert provider == "anthropic"
        assert api_key == "sk-anthropic"
        assert display_name == "Claude Provider"
        assert cred_id is None  # Legacy providers have no credential ID

    def test_deduplicates_credentials(self, monkeypatch):
        """Test that duplicate credentials are deduplicated."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        settings = MarkitSettings(
            llm=LLMConfig(
                credentials=[
                    LLMCredentialConfig(id="openai-1", provider="openai"),
                    LLMCredentialConfig(id="openai-2", provider="openai"),
                ]
            )
        )
        result = get_unique_credentials(settings)

        # Should be deduplicated since same provider, api_key, and base_url
        assert len(result) == 1

    def test_different_base_urls_are_unique(self, monkeypatch):
        """Test that same provider with different base_url is not deduplicated."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        settings = MarkitSettings(
            llm=LLMConfig(
                credentials=[
                    LLMCredentialConfig(id="openai-default", provider="openai"),
                    LLMCredentialConfig(
                        id="openai-custom",
                        provider="openai",
                        base_url="https://custom.api.com",
                    ),
                ]
            )
        )
        result = get_unique_credentials(settings)

        assert len(result) == 2

    def test_mixed_legacy_and_new_credentials(self, monkeypatch):
        """Test mixing legacy providers and new credentials."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
        settings = MarkitSettings(
            llm=LLMConfig(
                credentials=[
                    LLMCredentialConfig(id="openai-main", provider="openai"),
                ],
                providers=[
                    LLMProviderConfig(provider="anthropic", model="claude-3"),
                ],
            )
        )
        result = get_unique_credentials(settings)

        assert len(result) == 2
        providers = [r[0] for r in result]
        assert "openai" in providers
        assert "anthropic" in providers

    def test_legacy_provider_deduplicated_with_credential(self, monkeypatch):
        """Test that legacy provider is deduplicated if same as new credential."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        settings = MarkitSettings(
            llm=LLMConfig(
                credentials=[
                    LLMCredentialConfig(id="openai-main", provider="openai"),
                ],
                providers=[
                    LLMProviderConfig(provider="openai", model="gpt-4"),
                ],
            )
        )
        result = get_unique_credentials(settings)

        # Should be deduplicated - same provider, api_key, base_url
        assert len(result) == 1

    def test_ollama_provider_no_api_key(self):
        """Test Ollama provider without API key."""
        settings = MarkitSettings(
            llm=LLMConfig(
                credentials=[
                    LLMCredentialConfig(
                        id="ollama-local",
                        provider="ollama",
                        base_url="http://localhost:11434",
                    ),
                ]
            )
        )
        result = get_unique_credentials(settings)

        assert len(result) == 1
        provider, api_key, base_url, display_name, cred_id = result[0]
        assert provider == "ollama"
        assert api_key is None
        assert base_url == "http://localhost:11434"

    def test_multiple_providers_all_returned(self, monkeypatch):
        """Test that multiple different providers are all returned."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
        monkeypatch.setenv("GOOGLE_API_KEY", "gemini-key")
        settings = MarkitSettings(
            llm=LLMConfig(
                credentials=[
                    LLMCredentialConfig(id="openai", provider="openai"),
                    LLMCredentialConfig(id="anthropic", provider="anthropic"),
                    LLMCredentialConfig(id="gemini", provider="gemini"),
                ]
            )
        )
        result = get_unique_credentials(settings)

        assert len(result) == 3
        providers = [r[0] for r in result]
        assert set(providers) == {"openai", "anthropic", "gemini"}
