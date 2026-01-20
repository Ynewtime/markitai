"""Configuration management for Markit."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from markit.constants import (
    CONFIG_FILENAME,
    DEFAULT_BATCH_CONCURRENCY,
    DEFAULT_IMAGE_FILTER_MIN_AREA,
    DEFAULT_IMAGE_FILTER_MIN_HEIGHT,
    DEFAULT_IMAGE_FILTER_MIN_WIDTH,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_IMAGE_MAX_HEIGHT,
    DEFAULT_IMAGE_MAX_WIDTH,
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_RETENTION,
    DEFAULT_LOG_ROTATION,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_OCR_LANG,
    DEFAULT_ON_CONFLICT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPTS_DIR,
    DEFAULT_ROUTER_NUM_RETRIES,
    DEFAULT_ROUTER_TIMEOUT,
    DEFAULT_ROUTING_STRATEGY,
    DEFAULT_SCAN_MAX_DEPTH,
    DEFAULT_SCAN_MAX_FILES,
    DEFAULT_STATE_FLUSH_INTERVAL_SECONDS,
)


class EnvVarNotFoundError(ValueError):
    """Raised when an environment variable referenced by env: syntax is not found."""

    def __init__(self, var_name: str) -> None:
        self.var_name = var_name
        super().__init__(f"Environment variable not found: {var_name}")


def resolve_env_value(value: str, strict: bool = True) -> str | None:
    """Resolve env:VAR_NAME syntax to actual environment variable value.

    Args:
        value: The value to resolve. If starts with "env:", looks up environment variable.
        strict: If True, raises EnvVarNotFoundError when variable not found.
                If False, returns None when variable not found.

    Returns:
        The resolved value, or None if env var not found and strict=False.

    Raises:
        EnvVarNotFoundError: If strict=True and environment variable not found.
    """
    if isinstance(value, str) and value.startswith("env:"):
        env_var = value[4:]
        env_value = os.environ.get(env_var)
        if env_value is None:
            if strict:
                raise EnvVarNotFoundError(env_var)
            return None
        return env_value
    return value


class OutputConfig(BaseModel):
    """Output configuration."""

    dir: str = DEFAULT_OUTPUT_DIR
    on_conflict: Literal["skip", "overwrite", "rename"] = DEFAULT_ON_CONFLICT
    allow_symlinks: bool = False


class LiteLLMParams(BaseModel):
    """LiteLLM parameters for a model."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    weight: int = DEFAULT_MODEL_WEIGHT
    max_tokens: int | None = None  # Override max_output_tokens for this model

    def get_resolved_api_key(self, strict: bool = True) -> str | None:
        """Get API key with env: syntax resolved.

        Args:
            strict: If True, raises EnvVarNotFoundError when env var not found.
                    If False, returns None when env var not found.

        Returns:
            The resolved API key, or None if not configured or env var not found.

        Raises:
            EnvVarNotFoundError: If strict=True and environment variable not found.
        """
        if self.api_key:
            return resolve_env_value(self.api_key, strict=strict)
        return None


class ModelInfo(BaseModel):
    """Model metadata."""

    supports_vision: bool = False
    max_tokens: int | None = None


class ModelConfig(BaseModel):
    """Model configuration for LiteLLM Router."""

    model_name: str
    litellm_params: LiteLLMParams
    model_info: ModelInfo | None = None


class RouterSettings(BaseModel):
    """LiteLLM Router settings."""

    routing_strategy: Literal[
        "simple-shuffle", "least-busy", "usage-based-routing", "latency-based-routing"
    ] = DEFAULT_ROUTING_STRATEGY
    num_retries: int = DEFAULT_ROUTER_NUM_RETRIES
    timeout: int = DEFAULT_ROUTER_TIMEOUT
    fallbacks: list[dict[str, Any]] = Field(default_factory=list)


class LLMConfig(BaseModel):
    """LLM configuration."""

    enabled: bool = False
    model_list: list[ModelConfig] = Field(default_factory=list)
    router_settings: RouterSettings = Field(default_factory=RouterSettings)
    concurrency: int = DEFAULT_LLM_CONCURRENCY


class ImageFilterConfig(BaseModel):
    """Image filter configuration."""

    min_width: int = DEFAULT_IMAGE_FILTER_MIN_WIDTH
    min_height: int = DEFAULT_IMAGE_FILTER_MIN_HEIGHT
    min_area: int = DEFAULT_IMAGE_FILTER_MIN_AREA
    deduplicate: bool = True


class ImageConfig(BaseModel):
    """Image processing configuration."""

    alt_enabled: bool = False  # Generate alt text for images via LLM
    desc_enabled: bool = False  # Generate description files for images
    compress: bool = True
    quality: int = Field(default=DEFAULT_IMAGE_QUALITY, ge=1, le=100)
    format: Literal["jpeg", "png", "webp"] = DEFAULT_IMAGE_FORMAT
    max_width: int = DEFAULT_IMAGE_MAX_WIDTH
    max_height: int = DEFAULT_IMAGE_MAX_HEIGHT
    filter: ImageFilterConfig = Field(default_factory=ImageFilterConfig)


class OCRConfig(BaseModel):
    """OCR configuration."""

    enabled: bool = False
    lang: str = DEFAULT_OCR_LANG


class ScreenshotConfig(BaseModel):
    """Screenshot rendering configuration."""

    enabled: bool = False


class PromptsConfig(BaseModel):
    """Prompts configuration."""

    dir: str = DEFAULT_PROMPTS_DIR
    cleaner: str | None = None
    frontmatter: str | None = None
    image_caption: str | None = None
    image_description: str | None = None
    image_analysis: str | None = None  # Combined caption + description
    page_content: str | None = None  # Page content extraction
    document_enhance: str | None = None  # Document enhancement with vision


class BatchConfig(BaseModel):
    """Batch processing configuration."""

    concurrency: int = DEFAULT_BATCH_CONCURRENCY
    state_flush_interval_seconds: int = DEFAULT_STATE_FLUSH_INTERVAL_SECONDS
    scan_max_depth: int = DEFAULT_SCAN_MAX_DEPTH
    scan_max_files: int = DEFAULT_SCAN_MAX_FILES


class LogConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = DEFAULT_LOG_LEVEL
    dir: str | None = DEFAULT_LOG_DIR
    rotation: str = DEFAULT_LOG_ROTATION
    retention: str = DEFAULT_LOG_RETENTION


class PresetConfig(BaseModel):
    """Preset configuration defining which features to enable."""

    llm: bool = False
    ocr: bool = False
    alt: bool = False
    desc: bool = False
    screenshot: bool = False


# Built-in preset definitions
BUILTIN_PRESETS: dict[str, PresetConfig] = {
    "rich": PresetConfig(llm=True, alt=True, desc=True, screenshot=True),
    "standard": PresetConfig(llm=True, alt=True, desc=True),
    "minimal": PresetConfig(),
}


class MarkitConfig(BaseModel):
    """Main configuration model."""

    output: OutputConfig = Field(default_factory=OutputConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    screenshot: ScreenshotConfig = Field(default_factory=ScreenshotConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    presets: dict[str, PresetConfig] = Field(default_factory=dict)


class ConfigManager:
    """Configuration manager for loading and merging configs."""

    CONFIG_FILENAME = CONFIG_FILENAME
    DEFAULT_USER_CONFIG_DIR = Path.home() / ".markit"

    def __init__(self) -> None:
        self._config: MarkitConfig | None = None
        self._config_path: Path | None = None

    @property
    def config(self) -> MarkitConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load()
        return self._config

    @property
    def config_path(self) -> Path | None:
        """Get the path of the loaded configuration file."""
        return self._config_path

    def load(
        self,
        config_path: Path | str | None = None,
        env_override: bool = True,
    ) -> MarkitConfig:
        """
        Load configuration from file with fallback chain.

        Priority (highest to lowest):
        1. Explicit config_path parameter
        2. MARKIT_CONFIG environment variable
        3. ./markit.json (current directory)
        4. ~/.markit/config.json (user directory)
        5. Default values
        """
        config_data: dict[str, Any] = {}

        # Determine config file path
        resolved_path = self._resolve_config_path(config_path, env_override)

        if resolved_path and resolved_path.exists():
            config_data = self._load_json(resolved_path)
            self._config_path = resolved_path

        self._config = MarkitConfig.model_validate(config_data)
        return self._config

    def _resolve_config_path(
        self,
        config_path: Path | str | None,
        env_override: bool,
    ) -> Path | None:
        """Resolve configuration file path based on priority."""
        # 1. Explicit path
        if config_path:
            return Path(config_path)

        # 2. Environment variable
        if env_override:
            env_path = os.environ.get("MARKIT_CONFIG")
            if env_path:
                return Path(env_path)

        # 3. Current directory
        cwd_config = Path.cwd() / self.CONFIG_FILENAME
        if cwd_config.exists():
            return cwd_config

        # 4. User directory
        user_config = self.DEFAULT_USER_CONFIG_DIR / "config.json"
        if user_config.exists():
            return user_config

        return None

    def _load_json(self, path: Path) -> dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save(self, path: Path | str | None = None) -> Path:
        """Save current configuration to file."""
        if self._config is None:
            self._config = MarkitConfig()

        save_path = Path(path) if path else self._config_path
        if save_path is None:
            save_path = self.DEFAULT_USER_CONFIG_DIR / "config.json"
        elif save_path.is_dir():
            # If path is a directory, append default filename
            save_path = save_path / "markit.json"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self._config.model_dump(mode="json"), f, indent=2)

        return save_path

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated key path.

        Example: config_manager.get("llm.enabled")
        """
        parts = key.split(".")
        value: Any = self.config

        for part in parts:
            if isinstance(value, BaseModel):
                value = getattr(value, part, None)
            elif isinstance(value, dict):
                value = value.get(part)
            else:
                return default

            if value is None:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by dot-separated key path.

        Example: config_manager.set("llm.enabled", True)
        """
        parts = key.split(".")
        if len(parts) == 1:
            setattr(self.config, key, value)
            return

        # Navigate to parent
        parent: Any = self.config
        for part in parts[:-1]:
            if isinstance(parent, BaseModel):
                parent = getattr(parent, part)
            elif isinstance(parent, dict):
                parent = parent[part]

        # Set the value
        final_key = parts[-1]
        if isinstance(parent, BaseModel):
            setattr(parent, final_key, value)
        elif isinstance(parent, dict):
            parent[final_key] = value

    def merge_cli_args(self, **kwargs: Any) -> None:
        """Merge CLI arguments into configuration."""
        for key, value in kwargs.items():
            if value is not None:
                # Convert CLI arg names (e.g., output_dir) to config paths (e.g., output.dir)
                config_key = key.replace("_", ".")
                self.set(config_key, value)


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> MarkitConfig:
    """Get the global configuration."""
    return config_manager.config


def get_preset(name: str, config: MarkitConfig | None = None) -> PresetConfig | None:
    """Get a preset by name.

    Looks up in config file first, then falls back to built-in presets.

    Args:
        name: Preset name (e.g., "rich", "standard", "minimal")
        config: Optional config to look up custom presets

    Returns:
        PresetConfig if found, None otherwise
    """
    # Check config file presets first
    if config and name in config.presets:
        return config.presets[name]

    # Fall back to built-in presets
    return BUILTIN_PRESETS.get(name)
