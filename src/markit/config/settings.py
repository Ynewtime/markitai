"""Configuration settings using pydantic-settings."""

import importlib.resources
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

from markit.config.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONCURRENT_FALLBACK_ENABLED,
    DEFAULT_CONCURRENT_FALLBACK_TIMEOUT,
    DEFAULT_FILE_WORKERS,
    DEFAULT_IMAGE_WORKERS,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_LIBREOFFICE_POOL_SIZE,
    DEFAULT_LIBREOFFICE_PROFILE_DIR,
    DEFAULT_LIBREOFFICE_RESET_AFTER_FAILURES,
    DEFAULT_LIBREOFFICE_RESET_AFTER_USES,
    DEFAULT_LLM_WORKERS,
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_IMAGE_DIMENSION,
    DEFAULT_MAX_REQUEST_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_IMAGE_AREA,
    DEFAULT_MIN_IMAGE_DIMENSION,
    DEFAULT_MIN_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PNG_OPTIMIZATION_LEVEL,
    DEFAULT_PROCESS_POOL_MAX_WORKERS,
    DEFAULT_PROCESS_POOL_THRESHOLD,
    DEFAULT_PROMPTS_DIR,
    DEFAULT_STATE_FILE,
)


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider (Legacy mode)."""

    provider: Literal["openai", "anthropic", "gemini", "ollama", "openrouter"]
    model: str
    name: str | None = None  # User-defined name for display
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    timeout: int = DEFAULT_MAX_REQUEST_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    capabilities: list[str] | None = None  # None implies ["text", "vision"] (optimistic)


class LLMCredentialConfig(BaseModel):
    """Configuration for an LLM provider credential."""

    id: str  # Unique identifier (e.g. "openai-main", "deepseek")
    provider: Literal["openai", "anthropic", "gemini", "ollama", "openrouter"]
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None


class ModelCostConfig(BaseModel):
    """Model cost configuration (optional)."""

    input_per_1m: float = 0.0  # USD per 1M input tokens
    output_per_1m: float = 0.0  # USD per 1M output tokens
    cached_input_per_1m: float | None = None  # USD per 1M cached input tokens


class LLMModelConfig(BaseModel):
    """Configuration for a specific LLM model instance."""

    name: str  # Display name (e.g. "GPT-4o")
    model: str  # Model ID (e.g. "gpt-4o")
    credential_id: str  # Reference to LLMCredentialConfig.id
    capabilities: list[str] | None = None
    timeout: int = DEFAULT_MAX_REQUEST_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    cost: ModelCostConfig | None = None  # Optional cost config for statistics


class ValidationConfig(BaseModel):
    """LLM validation configuration."""

    enabled: bool = True  # Whether to validate providers
    retry_count: int = 2  # Number of retries on validation failure
    on_failure: Literal["warn", "skip", "fail"] = "warn"  # Action on failure


class LLMConfig(BaseModel):
    """LLM configuration - supports multiple providers."""

    # Legacy: Mix of credentials and model info
    providers: list[LLMProviderConfig] = Field(default_factory=list)

    # New: Decoupled credentials and models
    credentials: list[LLMCredentialConfig] = Field(default_factory=list)
    models: list[LLMModelConfig] = Field(default_factory=list)

    default_provider: str | None = None

    # Validation settings
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    # Concurrent fallback settings
    concurrent_fallback_enabled: bool = DEFAULT_CONCURRENT_FALLBACK_ENABLED
    concurrent_fallback_timeout: int = DEFAULT_CONCURRENT_FALLBACK_TIMEOUT
    max_request_timeout: int = DEFAULT_MAX_REQUEST_TIMEOUT


class ImageConfig(BaseModel):
    """Image processing configuration."""

    enable_compression: bool = True
    png_optimization_level: int = Field(default=DEFAULT_PNG_OPTIMIZATION_LEVEL, ge=0, le=6)
    jpeg_quality: int = Field(default=DEFAULT_JPEG_QUALITY, ge=0, le=100)
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION
    enable_analysis: bool = False  # LLM image analysis disabled by default

    # Image filtering (remove small decorative images)
    filter_small_images: bool = True
    min_dimension: int = Field(default=DEFAULT_MIN_IMAGE_DIMENSION, ge=0)
    min_area: int = Field(default=DEFAULT_MIN_IMAGE_AREA, ge=0)
    min_file_size: int = Field(default=DEFAULT_MIN_IMAGE_SIZE, ge=0)

    # Process pool settings for parallel image processing
    use_process_pool: bool = True
    process_pool_threshold: int = Field(default=DEFAULT_PROCESS_POOL_THRESHOLD, ge=1)
    process_pool_max_workers: int = Field(default=DEFAULT_PROCESS_POOL_MAX_WORKERS, ge=1)


class ConcurrencyConfig(BaseModel):
    """Concurrency configuration."""

    file_workers: int = Field(default=DEFAULT_FILE_WORKERS, ge=1)
    image_workers: int = Field(default=DEFAULT_IMAGE_WORKERS, ge=1)
    llm_workers: int = Field(default=DEFAULT_LLM_WORKERS, ge=1)


class PDFConfig(BaseModel):
    """PDF processing configuration."""

    engine: Literal["pymupdf4llm", "pymupdf", "pdfplumber", "markitdown"] = "pymupdf4llm"
    extract_images: bool = True
    ocr_enabled: bool = False


class EnhancementConfig(BaseModel):
    """Markdown enhancement configuration."""

    enabled: bool = False  # LLM enhancement disabled by default
    remove_headers_footers: bool = True
    fix_heading_levels: bool = True
    add_frontmatter: bool = True
    generate_summary: bool = True
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


class OutputConfig(BaseModel):
    """Output configuration."""

    default_dir: str = DEFAULT_OUTPUT_DIR
    on_conflict: Literal["skip", "overwrite", "rename"] = "rename"
    create_assets_subdir: bool = True
    generate_image_descriptions: bool = True


class ExecutionConfig(BaseModel):
    """Execution mode configuration."""

    mode: Literal["default", "fast"] = "default"
    fast_max_fallback: int = 1  # Max fallback attempts in fast mode
    fast_skip_validation: bool = True  # Skip provider validation in fast mode


class LibreOfficeConfig(BaseModel):
    """LibreOffice profile pool configuration for concurrent document conversion."""

    pool_size: int = Field(default=DEFAULT_LIBREOFFICE_POOL_SIZE, ge=1, le=32)
    profile_base_dir: str = DEFAULT_LIBREOFFICE_PROFILE_DIR
    reset_after_failures: int = Field(default=DEFAULT_LIBREOFFICE_RESET_AFTER_FAILURES, ge=1)
    reset_after_uses: int = Field(default=DEFAULT_LIBREOFFICE_RESET_AFTER_USES, ge=1)


class PromptConfig(BaseModel):
    """LLM prompt configuration for output language and customization."""

    output_language: Literal["zh", "en", "auto"] = "zh"
    extract_knowledge_graph_meta: bool = True

    # Prompts directory for external prompt files
    prompts_dir: str = DEFAULT_PROMPTS_DIR

    # Custom prompt file paths (override prompts_dir)
    image_analysis_prompt: str | None = None
    enhancement_prompt: str | None = None
    summary_prompt: str | None = None

    # Legacy: inline custom prompts (deprecated, use file paths instead)
    custom_enhancement_prompt: str | None = None
    custom_summary_prompt: str | None = None
    custom_image_analysis_prompt: str | None = None

    def get_prompt(self, prompt_type: str) -> str | None:
        """Get prompt content for the specified type.

        Lookup order:
        1. Custom file path (e.g., image_analysis_prompt config option)
        2. User's prompts directory with language suffix
        3. Legacy inline custom prompts (deprecated)
        4. Package built-in prompts

        Args:
            prompt_type: One of "image_analysis", "enhancement", "summary"

        Returns:
            Prompt content string, or None if not found
        """
        lang = self.output_language if self.output_language != "auto" else "zh"
        filename = f"{prompt_type}_{lang}.md"

        # 1. Check custom file path
        custom_path_attr = f"{prompt_type}_prompt"
        custom_path = getattr(self, custom_path_attr, None)
        if custom_path:
            path = Path(custom_path)
            if path.exists():
                return path.read_text(encoding="utf-8")

        # 2. Check user's prompts directory with language suffix
        prompts_dir = Path(self.prompts_dir)
        default_path = prompts_dir / filename
        if default_path.exists():
            return default_path.read_text(encoding="utf-8")

        # 3. Check legacy inline custom prompts
        legacy_attr = f"custom_{prompt_type}_prompt"
        legacy_prompt = getattr(self, legacy_attr, None)
        if legacy_prompt:
            return legacy_prompt

        # 4. Fall back to package built-in prompts
        try:
            package_prompts = importlib.resources.files("markit.config.prompts")
            prompt_file = package_prompts.joinpath(filename)
            return prompt_file.read_text(encoding="utf-8")
        except (FileNotFoundError, TypeError):
            return None


class MarkitSettings(BaseSettings):
    """Main configuration class for MarkIt."""

    model_config = SettingsConfigDict(
        env_prefix="MARKIT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        """Customize settings sources to include YAML file."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file="markit.yaml"),
            file_secret_settings,
        )

    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    libreoffice: LibreOfficeConfig = Field(default_factory=LibreOfficeConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)

    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_dir: str = DEFAULT_LOG_DIR
    state_file: str = DEFAULT_STATE_FILE

    def get_output_dir(self, base_path: Path | None = None) -> Path:
        """Get the output directory path."""
        if base_path:
            return base_path / self.output.default_dir
        return Path(self.output.default_dir)


@lru_cache
def get_settings() -> MarkitSettings:
    """Get cached settings instance."""
    return MarkitSettings()


def reload_settings() -> MarkitSettings:
    """Force reload settings (clear cache)."""
    get_settings.cache_clear()
    return get_settings()


class LLMConfigResolver:
    """Resolve LLM configuration by merging config file and CLI arguments.

    This class centralizes the logic for merging CLI-provided LLM options
    (like --llm-provider and --llm-model) with the base configuration from
    markit.yaml. CLI arguments take precedence over config file values.

    Usage:
        resolved = LLMConfigResolver.resolve(
            base_config=settings.llm,
            cli_provider="anthropic",
            cli_model="claude-sonnet-4-5",
        )
    """

    @staticmethod
    def resolve(
        base_config: LLMConfig,
        cli_provider: str | None = None,
        cli_model: str | None = None,
    ) -> LLMConfig:
        """Merge CLI arguments into base LLM configuration.

        Priority: CLI arguments > config file

        Args:
            base_config: Base LLM configuration from settings
            cli_provider: CLI-provided provider override (e.g., "anthropic")
            cli_model: CLI-provided model override (e.g., "claude-sonnet-4-5")

        Returns:
            Resolved LLMConfig with CLI overrides applied
        """
        if not cli_provider and not cli_model:
            return base_config

        from markit.config.constants import DEFAULT_LLM_MODELS

        # Deep copy to avoid modifying original
        resolved = base_config.model_copy(deep=True)

        if cli_provider:
            # Look for existing provider config
            existing_provider = next(
                (p for p in resolved.providers if p.provider == cli_provider),
                None,
            )

            if existing_provider:
                # Update existing provider's model if specified
                if cli_model:
                    existing_provider.model = cli_model
                # Move to front of list for priority
                resolved.providers.remove(existing_provider)
                resolved.providers.insert(0, existing_provider)
            else:
                # Create new provider config
                new_provider = LLMProviderConfig(
                    provider=cli_provider,  # type: ignore[arg-type]
                    model=cli_model or DEFAULT_LLM_MODELS.get(cli_provider, ""),
                )
                resolved.providers.insert(0, new_provider)

        elif cli_model and resolved.providers:
            # Only model specified, update first provider's model
            resolved.providers[0].model = cli_model

        return resolved
