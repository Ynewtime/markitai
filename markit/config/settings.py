"""Configuration settings using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

from markit.config.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FILE_WORKERS,
    DEFAULT_IMAGE_WORKERS,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_LLM_WORKERS,
    DEFAULT_MAX_IMAGE_DIMENSION,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_IMAGE_AREA,
    DEFAULT_MIN_IMAGE_DIMENSION,
    DEFAULT_MIN_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PNG_OPTIMIZATION_LEVEL,
    DEFAULT_STATE_FILE,
)


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    provider: Literal["openai", "anthropic", "gemini", "ollama", "openrouter"]
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = DEFAULT_LLM_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES


class LLMConfig(BaseModel):
    """LLM configuration - supports multiple providers."""

    providers: list[LLMProviderConfig] = Field(default_factory=list)
    default_provider: str | None = None


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
        """Customize settings sources to include TOML file."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file="markit.toml"),
            file_secret_settings,
        )

    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    enhancement: EnhancementConfig = Field(default_factory=EnhancementConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str | None = None
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
