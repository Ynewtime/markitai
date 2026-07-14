"""Configuration management for Markitai."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator

from markitai.constants import (
    ALL_FETCH_STRATEGIES,
    CONFIG_FILENAME,
    DEFAULT_BATCH_CONCURRENCY,
    DEFAULT_CACHE_SIZE_LIMIT,
    DEFAULT_DEFUDDLE_RPM,
    DEFAULT_DEFUDDLE_TIMEOUT,
    DEFAULT_FETCH_FALLBACK_PATTERNS,
    DEFAULT_FETCH_STRATEGY,
    DEFAULT_GLOBAL_CACHE_DIR,
    DEFAULT_HEAVY_TASK_LIMIT,
    DEFAULT_IMAGE_FILTER_MIN_AREA,
    DEFAULT_IMAGE_FILTER_MIN_HEIGHT,
    DEFAULT_IMAGE_FILTER_MIN_WIDTH,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_IMAGE_MAX_HEIGHT,
    DEFAULT_IMAGE_MAX_WIDTH,
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_JINA_RPM,
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
    DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
    DEFAULT_PLAYWRIGHT_TIMEOUT,
    DEFAULT_PLAYWRIGHT_WAIT_FOR,
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

# Environment variable descriptions for user-friendly error messages
ENV_VAR_DESCRIPTIONS: dict[str, str] = {
    "OPENAI_API_KEY": "OpenAI API (GPT-4o, GPT-4o-mini)",
    "ANTHROPIC_API_KEY": "Anthropic API (Claude models)",
    "GEMINI_API_KEY": "Google Gemini API (Gemini 2.x)",
    "DEEPSEEK_API_KEY": "DeepSeek API",
    "OPENROUTER_API_KEY": "OpenRouter API (multi-provider gateway)",
    "JINA_API_KEY": "Jina Reader API (URL conversion)",
    "MARKITAI_CONFIG": "Markitai configuration file path",
    "MARKITAI_LOG_DIR": "Markitai log directory",
}


class EnvVarNotFoundError(ValueError):
    """Raised when an environment variable referenced by env: syntax is not found."""

    def __init__(self, var_name: str) -> None:
        self.var_name = var_name
        description = ENV_VAR_DESCRIPTIONS.get(var_name, "Required by configuration")
        message = (
            f"Environment variable not found: {var_name}\n\n"
            f"  Purpose: {description}\n\n"
            f"  To set it:\n"
            f"    export {var_name}=your_value_here\n\n"
            f"  Or add to .env file:\n"
            f"    {var_name}=your_value_here\n\n"
            f"  See .env.example for all supported variables."
        )
        super().__init__(message)


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


def _resolve_api_key(api_key: str | None, strict: bool = True) -> str | None:
    """Resolve API key from value with env: syntax support.

    Args:
        api_key: Raw API key value, possibly using env: syntax.
        strict: If True, raises EnvVarNotFoundError when env var not found.
                If False, returns None when env var not found.

    Returns:
        The resolved API key, or None if not configured or env var not found.

    Raises:
        EnvVarNotFoundError: If strict=True and environment variable not found.
    """
    if api_key:
        return resolve_env_value(api_key, strict=strict)
    return None


def _validate_strategy_list(strategies: list[str]) -> None:
    """Validate a list of strategy names."""
    valid = set(ALL_FETCH_STRATEGIES)
    for s in strategies:
        if s not in valid:
            raise ValueError(f"invalid_strategy: '{s}'. Must be one of {sorted(valid)}")
    if len(strategies) != len(set(strategies)):
        raise ValueError("duplicate strategies in strategy_priority")


def _validate_local_only_pattern(pattern: str) -> None:
    """Validate a single local-only pattern (NO_PROXY syntax)."""
    import ipaddress as _ipaddress

    if not pattern or not pattern.strip():
        raise ValueError("empty pattern in local_only_patterns")
    p = pattern.strip()
    if "/" in p:
        try:
            _ipaddress.ip_network(p, strict=False)
        except ValueError as e:
            raise ValueError(f"invalid CIDR in local_only_patterns: '{p}' — {e}") from e


class OutputConfig(BaseModel):
    """Output configuration."""

    dir: str | None = Field(
        default=DEFAULT_OUTPUT_DIR, description="Default output directory"
    )
    on_conflict: Literal["skip", "overwrite", "rename"] = Field(
        default=DEFAULT_ON_CONFLICT,
        description="skip, overwrite, or rename on conflict",
    )
    allow_symlinks: bool = Field(
        default=False, description="Follow symlinks when processing files"
    )
    report: bool | None = Field(
        default=None,
        description="Write JSON report (default: batch/URL-batch runs only)",
    )


class LiteLLMParams(BaseModel):
    """LiteLLM parameters for a model."""

    model: str
    api_key: str | None = None
    api_base: str | None = Field(
        default=None,
        description="Custom API base URL to override the provider's default endpoint. Supports env:VAR_NAME syntax. Passed directly to LiteLLM; works with any LiteLLM-supported provider. Not used by local providers (claude-agent/, copilot/) which manage endpoints via their own environment variables.",
    )
    weight: int = Field(
        default=DEFAULT_MODEL_WEIGHT,
        ge=0,
        description="Routing weight for load balancing. 0 = disabled (model excluded from routing). Higher values get more traffic.",
    )
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
        return _resolve_api_key(self.api_key, strict=strict)

    def get_resolved_api_base(self, strict: bool = True) -> str | None:
        """Get API base URL with env: syntax resolved.

        Args:
            strict: If True, raises EnvVarNotFoundError when env var not found.
                    If False, returns None when env var not found.

        Returns:
            The resolved API base URL, or None if not configured or env var not found.

        Raises:
            EnvVarNotFoundError: If strict=True and environment variable not found.
        """
        if self.api_base:
            return resolve_env_value(self.api_base, strict=strict)
        return None


class ModelInfo(BaseModel):
    """Model metadata. All fields are optional and auto-detected from litellm if not set."""

    # LiteLLM's deployment identity. Routing groups (``model_name``) are not
    # unique: several deployments may legitimately share ``default``.
    id: str | None = None
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
    ] = Field(
        default=DEFAULT_ROUTING_STRATEGY,
        description="LLM router load balancing strategy",
    )
    num_retries: int = Field(
        default=DEFAULT_ROUTER_NUM_RETRIES,
        ge=0,
        description="Max retries on LLM failure",
    )
    timeout: int = Field(
        default=DEFAULT_ROUTER_TIMEOUT,
        ge=1,
        description="LLM request timeout in seconds",
    )
    fallbacks: list[dict[str, Any]] = Field(default_factory=list)


class LLMConfig(BaseModel):
    """LLM configuration."""

    enabled: bool = False
    pure: bool = Field(
        default=False,
        description="Pure mode: raw MD sent directly to LLM, no markitai processing",
    )
    keep_base: bool = Field(
        default=False,
        description="Keep base .md file alongside .llm.md in LLM mode",
    )
    model_list: list[ModelConfig] = Field(default_factory=list)
    router_settings: RouterSettings = Field(default_factory=RouterSettings)
    concurrency: int = Field(
        default=DEFAULT_LLM_CONCURRENCY, ge=1, description="Max parallel LLM requests"
    )


class ImageFilterConfig(BaseModel):
    """Image filter configuration."""

    min_width: int = Field(
        default=DEFAULT_IMAGE_FILTER_MIN_WIDTH,
        description="Skip images narrower than this (px)",
    )
    min_height: int = Field(
        default=DEFAULT_IMAGE_FILTER_MIN_HEIGHT,
        description="Skip images shorter than this (px)",
    )
    min_area: int = Field(
        default=DEFAULT_IMAGE_FILTER_MIN_AREA,
        description="Skip images smaller than this (px²)",
    )
    deduplicate: bool = Field(
        default=True, description="Remove duplicate images by hash"
    )


class ImageConfig(BaseModel):
    """Image processing configuration."""

    alt_enabled: bool = Field(default=False, description="Generate alt text via LLM")
    desc_enabled: bool = Field(
        default=False, description="Generate description files via LLM"
    )
    compress: bool = Field(default=True, description="Compress images before embedding")
    quality: int = Field(
        default=DEFAULT_IMAGE_QUALITY,
        ge=1,
        le=100,
        description="JPEG/WebP quality (1-100)",
    )
    format: Literal["jpeg", "png", "webp"] = Field(
        default=DEFAULT_IMAGE_FORMAT, description="Output image format"
    )
    max_width: int = Field(
        default=DEFAULT_IMAGE_MAX_WIDTH,
        description="Downscale images wider than this (px)",
    )
    max_height: int = Field(
        default=DEFAULT_IMAGE_MAX_HEIGHT,
        description="Downscale images taller than this (px)",
    )
    filter: ImageFilterConfig = Field(default_factory=ImageFilterConfig)
    stdout_persist: bool = Field(default=True, description="Save piped images to disk")
    stdout_persist_dir: str = Field(
        default="~/.markitai/assets", description="Directory for persisted piped images"
    )
    stdout_fetch_external: bool = Field(
        default=False, description="Download external image URLs from stdin"
    )


class OCRConfig(BaseModel):
    """OCR configuration."""

    enabled: bool = False
    lang: str = Field(default=DEFAULT_OCR_LANG, description="OCR language code")
    per_page_routing: bool = Field(
        default=True,
        description=(
            "With --ocr, keep the native text layer for pages that do not "
            "look scanned/garbled and OCR only the remaining pages. "
            "Disable to OCR every page."
        ),
    )


class OfficeConfig(BaseModel):
    """Office document conversion configuration."""

    macos_fallback: bool = Field(
        default=True,
        description=(
            "On macOS without LibreOffice, drive installed MS Office apps "
            "via AppleScript for legacy conversion and PPTX PDF export. "
            "Disable in headless sessions where macOS permission dialogs "
            "cannot be answered."
        ),
    )


class ScreenshotConfig(BaseModel):
    """Screenshot rendering configuration.

    For PDF/PPTX: Renders pages as JPEG images.
    For URLs: Captures full-page screenshots using Playwright.
    """

    enabled: bool = False
    # URL screenshot settings
    viewport_width: int = Field(
        default=DEFAULT_SCREENSHOT_VIEWPORT_WIDTH,
        description="Browser viewport width (px)",
    )
    viewport_height: int = Field(
        default=DEFAULT_SCREENSHOT_VIEWPORT_HEIGHT,
        description="Browser viewport height (px)",
    )
    quality: int = Field(
        default=DEFAULT_SCREENSHOT_QUALITY,
        ge=1,
        le=100,
        description="Screenshot JPEG quality (1-100)",
    )
    max_height: int = Field(
        default=DEFAULT_SCREENSHOT_MAX_HEIGHT,
        description="Max full-page screenshot height (px)",
    )
    screenshot_only: bool = Field(
        default=False, description="LLM reads only screenshots, no text extraction"
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
    document_process_system: str | None = None
    document_process_user: str | None = None
    document_vision_system: str | None = None
    document_vision_user: str | None = None
    # URL prompts
    url_enhance_system: str | None = None
    url_enhance_user: str | None = None


class BatchConfig(BaseModel):
    """Batch processing configuration."""

    concurrency: int = Field(
        default=DEFAULT_BATCH_CONCURRENCY,
        ge=1,
        description="Max parallel file processing tasks",
    )
    url_concurrency: int = Field(
        default=DEFAULT_URL_CONCURRENCY, ge=1, description="Max parallel URL fetches"
    )
    state_flush_interval_seconds: int = Field(
        default=DEFAULT_STATE_FLUSH_INTERVAL_SECONDS,
        description="Seconds between state file writes",
    )
    scan_max_depth: int = Field(
        default=DEFAULT_SCAN_MAX_DEPTH,
        ge=0,
        description="Max directory recursion depth",
    )
    scan_max_files: int = Field(
        default=DEFAULT_SCAN_MAX_FILES,
        ge=1,
        description="Max files to scan per directory",
    )
    heavy_task_limit: int = Field(
        default=DEFAULT_HEAVY_TASK_LIMIT,
        ge=0,
        description="Max heavy tasks (0=unlimited)",
    )


class LogConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default=DEFAULT_LOG_LEVEL, description="Minimum log level"
    )
    dir: str | None = Field(default=DEFAULT_LOG_DIR, description="Log file directory")
    rotation: str = Field(
        default=DEFAULT_LOG_ROTATION, description="Rotate log file at this size"
    )
    retention: str = Field(
        default=DEFAULT_LOG_RETENTION, description="Delete old logs after this period"
    )
    format: Literal["text", "json"] = Field(
        default="text", description="Log output format"
    )


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    no_cache: bool = Field(default=False, description="Skip reading cache, still write")
    no_cache_patterns: list[
        str
    ] = []  # Patterns to skip cache (glob, relative to input_dir)
    max_size_bytes: int = Field(
        default=DEFAULT_CACHE_SIZE_LIMIT, description="Max cache size in bytes"
    )
    global_dir: str = Field(
        default=DEFAULT_GLOBAL_CACHE_DIR, description="Global cache/config directory"
    )


class FetchPolicyConfig(BaseModel):
    """Configuration for fetch strategy policy engine."""

    enabled: bool = True
    max_strategy_hops: int = Field(
        default=5, ge=1, le=6, description="Max strategy fallback attempts"
    )
    strategy_priority: list[str] | None = Field(
        default=None,
        description="Custom global strategy order. Overrides the default priority.",
    )
    local_only_patterns: list[str] = Field(
        default_factory=list,
        description="Domain/IP patterns that must use local-only strategies (NO_PROXY syntax).",
    )
    inherit_no_proxy: bool = Field(
        default=True,
        description="Merge NO_PROXY env var into local_only_patterns at runtime.",
    )

    @field_validator("strategy_priority")
    @classmethod
    def validate_strategy_priority(
        cls,
        v: list[str] | None,
    ) -> list[str] | None:
        if v is None:
            return None
        if len(v) == 0:
            raise ValueError("strategy_priority must not be empty if set")
        _validate_strategy_list(v)
        return v

    @field_validator("local_only_patterns")
    @classmethod
    def validate_local_only_patterns(cls, v: list[str]) -> list[str]:
        for pattern in v:
            _validate_local_only_pattern(pattern)
        return v


class DomainProfileConfig(BaseModel):
    """Domain-specific overrides for fetch settings."""

    wait_for_selector: str | None = None
    wait_for: Literal["load", "domcontentloaded", "networkidle"] | None = None
    extra_wait_ms: int | None = Field(default=None, ge=0, le=30000)
    prefer_strategy: (
        Literal["static", "defuddle", "playwright", "jina", "cloudflare"] | None
    ) = None
    strategy_priority: list[str] | None = Field(
        default=None,
        description="Custom strategy order for this domain. Overrides global and prefer_strategy.",
    )
    skip_auto_scroll: bool = Field(
        default=False,
        description="Skip auto-scrolling for single-content pages (tweets, issues, docs).",
    )
    reject_resource_patterns: list[str] | None = Field(
        default=None,
        description="URL patterns to block during Playwright navigation (e.g. '**/analytics/**').",
    )

    @field_validator("strategy_priority")
    @classmethod
    def validate_strategy_priority(
        cls,
        v: list[str] | None,
    ) -> list[str] | None:
        if v is None:
            return None
        if len(v) == 0:
            raise ValueError("strategy_priority must not be empty if set")
        _validate_strategy_list(v)
        return v


class PlaywrightConfig(BaseModel):
    """Playwright configuration for JS-rendered pages."""

    timeout: int = Field(
        default=DEFAULT_PLAYWRIGHT_TIMEOUT, description="Navigation timeout (ms)"
    )
    wait_for: Literal["load", "domcontentloaded", "networkidle"] = Field(
        default=DEFAULT_PLAYWRIGHT_WAIT_FOR, description="Page load event to wait for"
    )
    extra_wait_ms: int = Field(
        default=DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
        description="Extra wait after page load (ms)",
    )

    session_mode: Literal["isolated", "domain_persistent"] = Field(
        default="isolated", description="Browser session reuse mode"
    )
    session_ttl_seconds: int = Field(
        default=600, ge=60, le=7200, description="Persistent session TTL (seconds)"
    )

    # Advanced browser control (aligned with CF Browser Rendering API capabilities)
    wait_for_selector: str | None = Field(
        default=None, description="CSS selector to wait for"
    )
    cookies: list[dict[str, str]] | None = None  # [{name, value, domain, path}, ...]
    reject_resource_patterns: list[str] | None = None  # ["**/*.css", "**/*.woff2"]
    extra_http_headers: dict[str, str] | None = None  # {"Accept-Language": "zh-CN"}
    user_agent: str | None = Field(default=None, description="Custom User-Agent string")
    http_credentials: dict[str, str] | None = None  # {username, password}


class DefuddleConfig(BaseModel):
    """Defuddle content extraction API configuration.

    Defuddle (https://defuddle.md) extracts clean article content from web pages,
    removing clutter like ads, sidebars, and navigation. Returns Markdown with
    rich YAML frontmatter (title, author, published, description, word_count).

    NOTE: Rate limit is undocumented. The default RPM is conservative.
    NOTE: JS rendering capability of the API is unconfirmed — SPA-heavy sites
    may still need playwright as fallback.
    """

    timeout: int = Field(
        default=DEFAULT_DEFUDDLE_TIMEOUT, description="Request timeout in seconds"
    )
    rpm: int = Field(
        default=DEFAULT_DEFUDDLE_RPM, ge=1, description="Max requests per minute"
    )


class JinaConfig(BaseModel):
    """Jina Reader API configuration."""

    api_key: str | None = Field(
        default=None, description="Jina Reader API key (supports env: syntax)"
    )
    timeout: int = Field(
        default=DEFAULT_JINA_TIMEOUT, description="Request timeout in seconds"
    )
    rpm: int = Field(
        default=DEFAULT_JINA_RPM, ge=1, description="Max requests per minute"
    )
    no_cache: bool = Field(default=False, description="Skip Jina server-side cache")
    target_selector: str | None = Field(
        default=None, description="CSS selector for content extraction"
    )
    wait_for_selector: str | None = Field(
        default=None, description="Wait for element before extraction"
    )

    def get_resolved_api_key(self, strict: bool = False) -> str | None:
        """Get API key with env: syntax resolved.

        Falls back to JINA_API_KEY env var when not set in config.
        """
        if self.api_key:
            return _resolve_api_key(self.api_key, strict=strict)
        return os.environ.get("JINA_API_KEY")


class CloudflareConfig(BaseModel):
    """Cloudflare Browser Rendering + Workers AI configuration.

    Used as a cloud-based fallback for URL fetching (Browser Rendering /markdown API)
    and file conversion (Workers AI toMarkdown API). Requires a Cloudflare account
    with Browser Rendering enabled (available on Free plan).

    Both api_token and account_id support env: syntax for environment variable resolution.
    """

    api_token: str | None = Field(
        default=None, description="CF API token (supports env: syntax)"
    )
    account_id: str | None = Field(
        default=None, description="CF account ID (supports env: syntax)"
    )
    timeout: int = Field(default=30000, description="Browser rendering timeout (ms)")
    wait_until: str = Field(
        default="networkidle0", description="Page load event to wait for"
    )
    cache_ttl: int = Field(
        default=0, description="Browser rendering cache TTL (seconds)"
    )
    reject_resource_patterns: list[str] | None = None  # e.g. ["/\\.css$/"]
    user_agent: str | None = Field(default=None, description="Custom User-Agent string")
    cookies: list[dict[str, str]] | None = None  # Cookies to set before navigation
    wait_for_selector: str | None = Field(
        default=None, description="CSS selector to wait for after load"
    )
    http_credentials: dict[str, str] | None = (
        None  # HTTP Basic Auth {username, password}
    )
    convert_enabled: bool = Field(
        default=False, description="Use Workers AI for file conversion"
    )

    def get_resolved_api_token(self, strict: bool = False) -> str | None:
        """Get API token with env: syntax resolved.

        Falls back to CLOUDFLARE_API_TOKEN env var when not set in config.
        """
        if self.api_token:
            return _resolve_api_key(self.api_token, strict=strict)
        return os.environ.get("CLOUDFLARE_API_TOKEN")

    def get_resolved_account_id(self, strict: bool = False) -> str | None:
        """Get account ID with env: syntax resolved.

        Falls back to CLOUDFLARE_ACCOUNT_ID env var when not set in config.
        """
        if self.account_id:
            return resolve_env_value(self.account_id, strict=strict)
        return os.environ.get("CLOUDFLARE_ACCOUNT_ID")


class FetchConfig(BaseModel):
    """URL fetch configuration for handling static and JS-rendered pages."""

    strategy: Literal[
        "auto", "static", "defuddle", "playwright", "jina", "cloudflare"
    ] = Field(default=DEFAULT_FETCH_STRATEGY, description="Default URL fetch strategy")
    remote_consent: Literal["ask", "always", "never"] = Field(
        default="always",
        description=(
            "Consent for sending URLs to remote extraction services "
            "(defuddle.md, Jina, Cloudflare) in the auto strategy chain, "
            "tried one at a time only after local strategies fail. "
            "Private/local/credentialed URLs never use remote services "
            "regardless of this setting. Public X/Twitter Playwright "
            "enrichment also uses the first-use disclosure. It does not open "
            "its own prompt under ask, but honors a cached process-wide "
            "decline, never, and the hard opt-out. always (default): use them without "
            "asking (a stderr notice discloses the first use); ask: prompt once "
            "per run on an interactive TTY, otherwise skip remote services; "
            "never: local strategies only. Overridden by the "
            "MARKITAI_NO_REMOTE_FETCH hard opt-out, including when an "
            "explicit remote strategy is selected."
        ),
    )
    defuddle: DefuddleConfig = Field(default_factory=DefuddleConfig)
    playwright: PlaywrightConfig = Field(default_factory=PlaywrightConfig)
    jina: JinaConfig = Field(default_factory=JinaConfig)
    cloudflare: CloudflareConfig = Field(default_factory=CloudflareConfig)
    kreuzberg_convert_enabled: bool = Field(
        default=False,
        description="Force kreuzberg converter for all supported file formats.",
    )
    policy: FetchPolicyConfig = Field(default_factory=FetchPolicyConfig)
    domain_profiles: dict[str, DomainProfileConfig] = Field(default_factory=dict)
    fallback_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FETCH_FALLBACK_PATTERNS)
    )


class SecurityConfig(BaseModel):
    """Security-related conversion safeguards."""

    pdf_sanitize: Literal["off", "warn", "remove"] = Field(
        default="warn",
        description=(
            "Hidden-text handling for PDFs: 'warn' logs invisible text "
            "(white-on-white, zero-opacity, <2pt, off-page) that would end "
            "up in the markdown, 'remove' also strips verbatim matches "
            "from the output, 'off' disables detection."
        ),
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
    office: OfficeConfig = Field(default_factory=OfficeConfig)
    screenshot: ScreenshotConfig = Field(default_factory=ScreenshotConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    fetch: FetchConfig = Field(default_factory=FetchConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    presets: dict[str, PresetConfig] = Field(default_factory=dict)


# Matches array index notation in key paths, e.g. "model_list[0]"
_ARRAY_KEY_RE = re.compile(r"^([^\[]+)\[(\d+)\]$")


def _set_nested_value(data: dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested value in a dict using dot-separated key path.

    Supports array index notation (e.g. "llm.model_list[0].model_name").
    Creates intermediate dicts/list entries if they don't exist.
    """
    parts = key_path.split(".")
    current: Any = data
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        array_match = _ARRAY_KEY_RE.match(part)
        if array_match:
            field_name = array_match.group(1)
            index = int(array_match.group(2))
            if field_name not in current or not isinstance(current[field_name], list):
                current[field_name] = []
            items = current[field_name]
            while len(items) <= index:
                items.append({})
            if is_last:
                items[index] = value
            else:
                if not isinstance(items[index], dict):
                    items[index] = {}
                current = items[index]
        else:
            if is_last:
                current[part] = value
            else:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]


class ConfigFileError(Exception):
    """Raised when a config file contains invalid values.

    Carries an actionable message; the CLI layer (MarkitaiGroup.invoke)
    translates it into a ClickException so users see the message instead
    of a raw ValidationError traceback. Kept framework-free so the
    configuration foundation does not depend on the CLI stack.
    """


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` deep-merged over ``base``.

    Nested dicts are merged key by key; any other value (including lists)
    in ``override`` replaces the ``base`` value. Neither input is mutated.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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

    def restore(self, config: MarkitaiConfig) -> None:
        """Replace the in-memory configuration (e.g. to roll back a bad set())."""
        self._config = config

    def load(
        self,
        config_path: Path | str | None = None,
        env_override: bool = True,
        overrides: dict[str, Any] | None = None,
    ) -> MarkitaiConfig:
        """
        Load configuration from file with fallback chain.

        Priority (highest to lowest):
        1. Explicit config_path parameter
        2. MARKITAI_CONFIG environment variable
        3. ./markitai.json (current directory)
        4. ~/.markitai/config.json (user directory)
        5. Default values (applied by caller when no config file is found)

        ``overrides`` (e.g. from ``--config-json``) is a partial config dict
        deep-merged over the file data before validation. It is ephemeral:
        never written back on save().
        """
        config_data: dict[str, Any] = {}

        # Determine config file path
        resolved_path = self._resolve_config_path(config_path, env_override)

        if resolved_path and resolved_path.exists():
            config_data = self._load_json(resolved_path)
            self._config_path = resolved_path

        # Preserve original JSON structure for minimal-diff saves
        # (deliberately WITHOUT the ephemeral overrides)
        self._raw_data = config_data.copy()
        self._modified_keys.clear()

        if overrides:
            config_data = _deep_merge(config_data, overrides)

        try:
            self._config = MarkitaiConfig.model_validate(config_data)
        except ValidationError as e:
            source = str(self._config_path or "config")
            if overrides:
                source += " (with --config-json overrides)"
            lines = [f"Invalid configuration in {source}:"]
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                lines.append(f"  {loc}: {err['msg']}")
            lines.append(
                "Fix the value(s) in that file, or remove the field(s) "
                "to fall back to defaults."
            )
            raise ConfigFileError("\n".join(lines)) from e
        return self._config

    def _resolve_config_path(
        self,
        config_path: Path | str | None,
        env_override: bool,
    ) -> Path | None:
        """Resolve configuration file path based on priority."""
        # 1. Explicit path
        if config_path:
            explicit_path = Path(config_path)
            if not explicit_path.exists():
                logger.warning(
                    f"Config file not found: {explicit_path} (using defaults)"
                )
            return explicit_path

        # 2. Environment variable
        if env_override:
            env_path = os.environ.get("MARKITAI_CONFIG")
            if env_path:
                env_config = Path(env_path).expanduser()
                if not env_config.exists():
                    logger.warning(
                        f"Config file from MARKITAI_CONFIG not found: {env_config} "
                        "(using defaults)"
                    )
                return env_config

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
        Auto-detectable values (max_tokens, supports_vision) and settings that
        match defaults (image.compress, image.quality) are omitted.
        """
        return {
            "output": {"dir": "./output"},
            "llm": {
                "enabled": False,
                "model_list": [
                    {
                        "model_name": "default",
                        "litellm_params": {
                            "model": "gemini/gemini-3.1-flash-lite-preview",
                            "api_key": "env:GEMINI_API_KEY",
                        },
                    }
                ],
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

        from markitai.security import atomic_write_text

        content = json.dumps(output_data, indent=2, ensure_ascii=False) + "\n"
        atomic_write_text(save_path, content, follow_symlinks=True)

        return save_path

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated key path.

        Supports array index notation: llm.model_list[0].model_name

        Example:
            config_manager.get("llm.enabled")
            config_manager.get("llm.model_list[0]")
            config_manager.get("llm.model_list[0].litellm_params.model")
        """
        # Split by dots, but preserve array indices
        # e.g., "llm.model_list[0].name" -> ["llm", "model_list[0]", "name"]
        parts = key.split(".")
        value: Any = self.config

        for part in parts:
            # Check for array index notation: "field[N]"
            array_match = _ARRAY_KEY_RE.match(part)
            if array_match:
                field_name = array_match.group(1)
                index = int(array_match.group(2))

                # First get the field
                if isinstance(value, BaseModel):
                    value = getattr(value, field_name, None)
                elif isinstance(value, dict):
                    value = value.get(field_name)
                else:
                    return default

                if value is None:
                    return default

                # Then get the array element
                if isinstance(value, list):
                    if 0 <= index < len(value):
                        value = value[index]
                    else:
                        return default
                else:
                    return default
            else:
                # Regular field access. Missing keys return the default;
                # existing keys holding None return None (so callers can
                # distinguish "missing" from "set to null").
                if isinstance(value, BaseModel):
                    if not hasattr(value, part):
                        return default
                    value = getattr(value, part)
                elif isinstance(value, dict):
                    if part not in value:
                        return default
                    value = value[part]
                else:
                    return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by dot-separated key path.

        Supports array index notation: llm.model_list[0].litellm_params.weight

        Example: config_manager.set("llm.enabled", True)
        """
        # Track modified key for minimal-diff save
        self._modified_keys.add(key)

        def resolve_child(obj: Any, part: str) -> Any:
            """Resolve one dot-path component (with optional [N] index)."""
            array_match = _ARRAY_KEY_RE.match(part)
            field_name = array_match.group(1) if array_match else part
            if isinstance(obj, BaseModel):
                child = getattr(obj, field_name)
            else:
                child = obj[field_name]
            if array_match:
                index = int(array_match.group(2))
                if not isinstance(child, list) or not 0 <= index < len(child):
                    raise KeyError(f"Index out of range in '{part}'")
                return child[index]
            return child

        parts = key.split(".")

        # Navigate to parent
        parent: Any = self.config
        for part in parts[:-1]:
            parent = resolve_child(parent, part)

        # Set the value
        final_key = parts[-1]
        array_match = _ARRAY_KEY_RE.match(final_key)
        if array_match:
            field_name = array_match.group(1)
            index = int(array_match.group(2))
            container = (
                getattr(parent, field_name)
                if isinstance(parent, BaseModel)
                else parent[field_name]
            )
            if not isinstance(container, list) or not 0 <= index < len(container):
                raise KeyError(f"Index out of range in '{final_key}'")
            container[index] = value
        elif isinstance(parent, BaseModel):
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
