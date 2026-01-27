"""Configuration management for Markitai."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from markitai.constants import (
    CONFIG_FILENAME,
    DEFAULT_AGENT_BROWSER_COMMAND,
    DEFAULT_AGENT_BROWSER_EXTRA_WAIT_MS,
    DEFAULT_AGENT_BROWSER_TIMEOUT,
    DEFAULT_AGENT_BROWSER_WAIT_FOR,
    DEFAULT_BATCH_CONCURRENCY,
    DEFAULT_CACHE_SIZE_LIMIT,
    DEFAULT_FETCH_FALLBACK_PATTERNS,
    DEFAULT_FETCH_STRATEGY,
    DEFAULT_GLOBAL_CACHE_DIR,
    DEFAULT_IMAGE_FILTER_MIN_AREA,
    DEFAULT_IMAGE_FILTER_MIN_HEIGHT,
    DEFAULT_IMAGE_FILTER_MIN_WIDTH,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_IMAGE_MAX_HEIGHT,
    DEFAULT_IMAGE_MAX_WIDTH,
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_JINA_TIMEOUT,
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
    DEFAULT_SCREENSHOT_MAX_HEIGHT,
    DEFAULT_SCREENSHOT_QUALITY,
    DEFAULT_SCREENSHOT_VIEWPORT_HEIGHT,
    DEFAULT_SCREENSHOT_VIEWPORT_WIDTH,
    DEFAULT_STATE_FLUSH_INTERVAL_SECONDS,
    DEFAULT_URL_CONCURRENCY,
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
    """Model metadata. All fields are optional and auto-detected from litellm if not set."""

    supports_vision: bool | None = None
    max_tokens: int | None = None
    max_input_tokens: int | None = None


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
    """Screenshot rendering configuration.

    For PDF/PPTX: Renders pages as JPEG images.
    For URLs: Captures full-page screenshots using agent-browser.
    """

    enabled: bool = False
    # URL screenshot settings
    viewport_width: int = DEFAULT_SCREENSHOT_VIEWPORT_WIDTH
    viewport_height: int = DEFAULT_SCREENSHOT_VIEWPORT_HEIGHT
    quality: int = Field(default=DEFAULT_SCREENSHOT_QUALITY, ge=1, le=100)
    max_height: int = (
        DEFAULT_SCREENSHOT_MAX_HEIGHT  # Max height for full-page URL screenshots
    )


class PromptsConfig(BaseModel):
    """Prompts configuration for custom prompt overrides.

    Each prompt has a system and user variant:
    - system: Contains instructions and context
    - user: Contains the actual request with content placeholders
    """

    dir: str = DEFAULT_PROMPTS_DIR
    # Cleaner prompts
    cleaner_system: str | None = None
    cleaner_user: str | None = None
    # Frontmatter prompts
    frontmatter_system: str | None = None
    frontmatter_user: str | None = None
    # Image prompts
    image_caption_system: str | None = None
    image_caption_user: str | None = None
    image_description_system: str | None = None
    image_description_user: str | None = None
    image_analysis_system: str | None = None
    image_analysis_user: str | None = None
    # Page content prompts
    page_content_system: str | None = None
    page_content_user: str | None = None
    # Document prompts
    document_enhance_system: str | None = None
    document_enhance_user: str | None = None
    document_enhance_complete_system: str | None = None
    document_enhance_complete_user: str | None = None
    document_process_system: str | None = None
    document_process_user: str | None = None
    # URL prompts
    url_enhance_system: str | None = None
    url_enhance_user: str | None = None


class BatchConfig(BaseModel):
    """Batch processing configuration."""

    concurrency: int = Field(default=DEFAULT_BATCH_CONCURRENCY, ge=1)
    url_concurrency: int = Field(
        default=DEFAULT_URL_CONCURRENCY, ge=1
    )  # Separate concurrency for URL fetching
    state_flush_interval_seconds: int = DEFAULT_STATE_FLUSH_INTERVAL_SECONDS
    scan_max_depth: int = Field(default=DEFAULT_SCAN_MAX_DEPTH, ge=1)
    scan_max_files: int = Field(default=DEFAULT_SCAN_MAX_FILES, ge=1)


class LogConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = DEFAULT_LOG_LEVEL
    dir: str | None = DEFAULT_LOG_DIR
    rotation: str = DEFAULT_LOG_ROTATION
    retention: str = DEFAULT_LOG_RETENTION


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    no_cache: bool = False  # Skip reading cache but still write (Bun semantics)
    no_cache_patterns: list[
        str
    ] = []  # Patterns to skip cache (glob, relative to input_dir)
    max_size_bytes: int = DEFAULT_CACHE_SIZE_LIMIT
    global_dir: str = DEFAULT_GLOBAL_CACHE_DIR


class AgentBrowserConfig(BaseModel):
    """agent-browser configuration for JS-rendered pages."""

    command: str = DEFAULT_AGENT_BROWSER_COMMAND
    timeout: int = DEFAULT_AGENT_BROWSER_TIMEOUT  # milliseconds
    wait_for: Literal["load", "domcontentloaded", "networkidle"] = (
        DEFAULT_AGENT_BROWSER_WAIT_FOR
    )
    extra_wait_ms: int = DEFAULT_AGENT_BROWSER_EXTRA_WAIT_MS  # Extra wait after load
    session: str | None = None  # Optional session name for isolated browser instances


class JinaConfig(BaseModel):
    """Jina Reader API configuration."""

    api_key: str | None = None  # Supports env: syntax
    timeout: int = DEFAULT_JINA_TIMEOUT  # seconds

    def get_resolved_api_key(self, strict: bool = False) -> str | None:
        """Get API key with env: syntax resolved.

        Args:
            strict: If True, raises EnvVarNotFoundError when env var not found.
                    If False (default), returns None when env var not found.

        Returns:
            The resolved API key, or None if not configured or env var not found.
        """
        if self.api_key:
            return resolve_env_value(self.api_key, strict=strict)
        return None


class FetchConfig(BaseModel):
    """URL fetch configuration for handling static and JS-rendered pages."""

    strategy: Literal["auto", "static", "browser", "jina"] = DEFAULT_FETCH_STRATEGY
    agent_browser: AgentBrowserConfig = Field(default_factory=AgentBrowserConfig)
    jina: JinaConfig = Field(default_factory=JinaConfig)
    fallback_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FETCH_FALLBACK_PATTERNS)
    )


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


class MarkitaiConfig(BaseModel):
    """Main configuration model."""

    output: OutputConfig = Field(default_factory=OutputConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    screenshot: ScreenshotConfig = Field(default_factory=ScreenshotConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    fetch: FetchConfig = Field(default_factory=FetchConfig)
    presets: dict[str, PresetConfig] = Field(default_factory=dict)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep merge updates into base dict, preserving base structure.

    Only updates keys that exist in updates, preserving other keys in base.
    """
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested_value(data: dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested value in a dict using dot-separated key path.

    Creates intermediate dicts if they don't exist.
    """
    parts = key_path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


class ConfigManager:
    """Configuration manager for loading and merging configs."""

    CONFIG_FILENAME = CONFIG_FILENAME
    DEFAULT_USER_CONFIG_DIR = Path.home() / ".markitai"

    def __init__(self) -> None:
        self._config: MarkitaiConfig | None = None
        self._config_path: Path | None = None
        self._raw_data: dict[str, Any] = {}  # Preserve original JSON structure
        self._modified_keys: set[str] = set()  # Track modified key paths

    @property
    def config(self) -> MarkitaiConfig:
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
    ) -> MarkitaiConfig:
        """
        Load configuration from file with fallback chain.

        Priority (highest to lowest):
        1. Explicit config_path parameter
        2. MARKITAI_CONFIG environment variable
        3. ./markitai.json (current directory)
        4. ~/.markitai/config.json (user directory)
        5. Default values
        """
        config_data: dict[str, Any] = {}

        # Determine config file path
        resolved_path = self._resolve_config_path(config_path, env_override)

        if resolved_path and resolved_path.exists():
            config_data = self._load_json(resolved_path)
            self._config_path = resolved_path

        # Preserve original JSON structure for minimal-diff saves
        self._raw_data = config_data.copy()
        self._modified_keys.clear()

        self._config = MarkitaiConfig.model_validate(config_data)
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
            env_path = os.environ.get("MARKITAI_CONFIG")
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

    def _generate_minimal_config(self) -> dict[str, Any]:
        """Generate minimal template config for init command.

        Only includes essential fields that users typically need to configure.
        Auto-detectable values (max_tokens, supports_vision) are omitted.
        """
        return {
            "output": {"dir": "./output"},
            "llm": {
                "enabled": False,
                "model_list": [
                    {
                        "model_name": "default",
                        "litellm_params": {
                            "model": "gemini/gemini-2.5-flash",
                            "api_key": "env:GEMINI_API_KEY",
                        },
                    }
                ],
            },
            "image": {
                "compress": True,
                "quality": 75,
            },
        }

    def save(
        self,
        path: Path | str | None = None,
        full_dump: bool = False,
        minimal: bool = False,
    ) -> Path:
        """Save current configuration to file.

        Args:
            path: Optional path to save to. If None, uses loaded config path.
            full_dump: If True, dumps entire config including defaults.
                       If False (default), only updates modified keys in original JSON.
            minimal: If True, generates a minimal template config (for init command).
        """
        if self._config is None:
            self._config = MarkitaiConfig()

        save_path = Path(path) if path else self._config_path
        if save_path is None:
            save_path = self.DEFAULT_USER_CONFIG_DIR / "config.json"
        elif save_path.is_dir():
            # If path is a directory, append default filename
            save_path = save_path / "markitai.json"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if minimal:
            # Generate minimal template config for init command
            output_data = self._generate_minimal_config()
        elif full_dump:
            # Full dump for init command or explicit full export
            output_data = self._config.model_dump(mode="json")
        else:
            # Minimal-diff save: only update modified keys in original JSON
            output_data = self._raw_data.copy()
            for key in self._modified_keys:
                _set_nested_value(output_data, key, self.get(key))

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            f.write("\n")  # Trailing newline for POSIX compliance

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
        # Track modified key for minimal-diff save
        self._modified_keys.add(key)

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


def get_config() -> MarkitaiConfig:
    """Get the global configuration."""
    return config_manager.config


def get_preset(name: str, config: MarkitaiConfig | None = None) -> PresetConfig | None:
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
