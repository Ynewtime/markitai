"""URL fetch module for handling static and JS-rendered pages.

This module provides a unified interface for fetching web pages using different
strategies:
- defuddle: Free content extraction API (best content cleaning, no auth)
- jina: Jina Reader API (cloud-based, no local dependencies)
- static: Direct HTTP request via httpx/curl-cffi (fastest, no external deps)
- playwright: Headless browser via Playwright Python (JS-rendered pages)
- cloudflare: Cloudflare Browser Rendering API (cloud browser)
- auto: Policy engine orders strategies and falls back through them

For X/Twitter URLs, the playwright strategy includes an oEmbed enricher
fallback (FxTwitter API → X oEmbed) that activates when DOM parsing fails
and remote_consent is allowed.

Example usage:
    from markitai.fetch import fetch_url, FetchStrategy

    # Auto-detect strategy (static → playwright → defuddle → jina → cloudflare)
    result = await fetch_url("https://example.com", FetchStrategy.AUTO, config.fetch)

    # Force Defuddle
    result = await fetch_url("https://example.com", FetchStrategy.DEFUDDLE, config.fetch)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from loguru import logger

from markitai.constants import (
    EXTERNAL_STRATEGIES,
    JS_REQUIRED_PATTERNS,
    LOCAL_STRATEGIES,
)

if TYPE_CHECKING:
    from markitai.config import (
        FetchConfig,
        FetchPolicyConfig,
        ScreenshotConfig,
    )


from markitai.fetch_cache import FetchCache as FetchCache  # noqa: F401
from markitai.fetch_cache import SPADomainCache as SPADomainCache  # noqa: F401

# Shared mutable fetch state (clients, caches, proxy, consent) lives on the
# process-wide FetchSession (markitai.fetch_session); consent logic lives in
# markitai.fetch_consent, screenshot helpers in markitai.fetch_screenshot,
# per-strategy fetch implementations in markitai.fetch_strategies, and
# helpers shared between this orchestrator and the strategies in
# markitai.fetch_support. They are re-exported here so the public
# markitai.fetch API (and its non-patch import surface) stays stable.
from markitai.fetch_consent import (
    _REMOTE_SERVICE_LABELS as _REMOTE_SERVICE_LABELS,  # noqa: F401
)
from markitai.fetch_consent import (
    _env_no_remote_fetch as _env_no_remote_fetch,
)
from markitai.fetch_consent import (
    _remote_service_names as _remote_service_names,  # noqa: F401
)
from markitai.fetch_consent import (
    _should_fallback_after_refusal as _should_fallback_after_refusal,
)
from markitai.fetch_consent import (
    disclose_remote_use as disclose_remote_use,
)
from markitai.fetch_consent import (
    peek_cached_remote_consent as peek_cached_remote_consent,  # noqa: F401
)
from markitai.fetch_consent import (
    peek_remote_consent as peek_remote_consent,
)
from markitai.fetch_consent import (
    reset_explicit_fallback_decision as reset_explicit_fallback_decision,  # noqa: F401
)
from markitai.fetch_consent import (
    reset_remote_consent as reset_remote_consent,  # noqa: F401
)
from markitai.fetch_consent import (
    resolve_remote_consent as resolve_remote_consent,
)
from markitai.fetch_consent import (
    set_remote_consent as set_remote_consent,  # noqa: F401
)
from markitai.fetch_consent import (
    set_remote_consent_prompt_allowed as set_remote_consent_prompt_allowed,  # noqa: F401
)
from markitai.fetch_http import (
    get_static_http_client as get_static_http_client,  # noqa: F401
)
from markitai.fetch_screenshot import (
    _compress_screenshot as _compress_screenshot,  # noqa: F401
)
from markitai.fetch_screenshot import (
    _url_to_screenshot_filename as _url_to_screenshot_filename,  # noqa: F401
)
from markitai.fetch_session import (
    FetchSession as FetchSession,  # noqa: F401
)
from markitai.fetch_session import (
    _get_system_proxy as _get_system_proxy,  # noqa: F401
)
from markitai.fetch_session import (
    _SlidingWindowRateLimiter as _SlidingWindowRateLimiter,  # noqa: F401
)
from markitai.fetch_session import (
    get_default_session as get_default_session,
)
from markitai.fetch_session import (
    reset_default_session as reset_default_session,  # noqa: F401
)
from markitai.fetch_strategies import (
    CloudflareRunner as CloudflareRunner,  # noqa: F401
)
from markitai.fetch_strategies import (
    DefuddleRunner as DefuddleRunner,  # noqa: F401
)
from markitai.fetch_strategies import (
    JinaRunner as JinaRunner,  # noqa: F401
)
from markitai.fetch_strategies import (
    PlaywrightRunner as PlaywrightRunner,  # noqa: F401
)
from markitai.fetch_strategies import (
    StaticRunner as StaticRunner,  # noqa: F401
)
from markitai.fetch_strategies import (
    StrategyContext as StrategyContext,
)
from markitai.fetch_strategies import (
    StrategyRunner as StrategyRunner,  # noqa: F401
)
from markitai.fetch_strategies import (
    fetch_with_cloudflare as fetch_with_cloudflare,  # noqa: F401
)
from markitai.fetch_strategies import (
    fetch_with_defuddle as fetch_with_defuddle,  # noqa: F401
)
from markitai.fetch_strategies import (
    fetch_with_jina as fetch_with_jina,  # noqa: F401
)
from markitai.fetch_strategies import (
    fetch_with_static as fetch_with_static,  # noqa: F401
)
from markitai.fetch_strategies import (
    fetch_with_static_conditional as fetch_with_static_conditional,
)
from markitai.fetch_strategies import (
    get_cf_semaphore as get_cf_semaphore,  # noqa: F401
)
from markitai.fetch_strategies import (
    get_runner as get_runner,
)
from markitai.fetch_strategies._shared import (
    _get_markitdown as _get_markitdown,  # noqa: F401
)
from markitai.fetch_strategies.jina import (
    _extract_jina_error_message as _extract_jina_error_message,  # noqa: F401
)
from markitai.fetch_strategies.jina import (
    _get_jina_client as _get_jina_client,  # noqa: F401
)
from markitai.fetch_strategies.jina import (
    _get_jina_rate_limiter as _get_jina_rate_limiter,  # noqa: F401
)
from markitai.fetch_strategies.static import (
    _extract_markdown_title as _extract_markdown_title,  # noqa: F401
)
from markitai.fetch_support import (
    _detect_proxy as _detect_proxy,
)
from markitai.fetch_support import (
    _get_playwright_advanced_kwargs as _get_playwright_advanced_kwargs,  # noqa: F401
)
from markitai.fetch_support import (
    _get_playwright_fetch_kwargs as _get_playwright_fetch_kwargs,
)
from markitai.fetch_support import (
    _resolve_playwright_profile_overrides as _resolve_playwright_profile_overrides,  # noqa: F401
)
from markitai.fetch_support import (
    _url_to_session_key as _url_to_session_key,  # noqa: F401
)
from markitai.fetch_types import (
    CRITICAL_INVALID_REASONS as CRITICAL_INVALID_REASONS,  # noqa: F401
)
from markitai.fetch_types import (
    ConditionalFetchResult as ConditionalFetchResult,  # noqa: F401
)
from markitai.fetch_types import FetchError as FetchError  # noqa: F401
from markitai.fetch_types import FetchResult as FetchResult  # noqa: F401
from markitai.fetch_types import FetchStrategy as FetchStrategy  # noqa: F401
from markitai.fetch_types import JinaAPIError as JinaAPIError  # noqa: F401
from markitai.fetch_types import JinaRateLimitError as JinaRateLimitError  # noqa: F401


def get_spa_domain_cache() -> SPADomainCache:
    """Get or create the global SPA domain cache instance.

    Returns:
        SPADomainCache instance
    """
    return get_default_session().get_spa_domain_cache()


def get_fetch_cache(
    cache_dir: Path, max_size_bytes: int = 100 * 1024 * 1024
) -> FetchCache:
    """Get or create the global fetch cache instance.

    Rebuilds the cache when configuration (cache_dir or max_size_bytes)
    changes, using a fingerprint to detect config drift.

    Args:
        cache_dir: Directory to store cache database
        max_size_bytes: Maximum cache size

    Returns:
        FetchCache instance
    """
    return get_default_session().get_fetch_cache(cache_dir, max_size_bytes)


def get_proxy_for_url(url: str, auto_proxy: bool = True) -> str:
    """Get proxy URL for a given URL, respecting auto_proxy setting and NO_PROXY.

    This is the unified entry point for proxy resolution. All fetch backends
    should use this instead of calling _detect_proxy() directly.

    Args:
        url: URL being fetched (checked against NO_PROXY patterns)
        auto_proxy: If False, always return empty string (proxy disabled)

    Returns:
        Proxy URL string or empty string if no proxy should be used
    """
    if not auto_proxy:
        return ""

    proxy = _detect_proxy()
    if not proxy:
        return ""

    # Check NO_PROXY bypass patterns
    bypass = get_default_session().detected_proxy_bypass
    if bypass:
        from urllib.parse import urlparse

        from markitai.fetch_policy import match_local_only, parse_no_proxy

        domain = urlparse(url).netloc.lower()
        patterns = parse_no_proxy(bypass)
        if match_local_only(domain, patterns):
            return ""

    return proxy


async def close_shared_clients() -> None:
    """Close shared client instances.

    Call this during cleanup to release resources.
    """
    await get_default_session().close()


async def _get_playwright_renderer(
    proxy: str | None = None, config: FetchConfig | None = None
) -> Any:
    """Get or create the shared PlaywrightRenderer.

    Rebuilds when ``proxy`` or session-mode configuration changes.

    Args:
        proxy: Optional proxy URL
        config: Optional fetch configuration to enable session cache

    Returns:
        PlaywrightRenderer instance
    """
    return await get_default_session().get_playwright_renderer(
        proxy=proxy, config=config
    )


def detect_js_required(content: str) -> bool:
    """Detect if content indicates JavaScript rendering is required.

    Note: This function receives MARKDOWN content (converted by markitdown),
    not raw HTML. Detection strategies must work with Markdown text.

    Uses multiple detection strategies:
    1. Simple string matching for common JS-required messages
    2. Content patterns that survive Markdown conversion
    3. Content length and quality checks

    Args:
        content: Markdown content to check (from markitdown conversion)

    Returns:
        True if content suggests JavaScript is needed
    """
    if not content:
        return True  # Empty content likely means JS-rendered

    content_lower = content.lower()

    # 1. Simple string matching for JS-required messages
    # These text patterns survive Markdown conversion
    for pattern in JS_REQUIRED_PATTERNS:
        if pattern.lower() in content_lower:
            logger.debug(f"JS required: string pattern matched '{pattern}'")
            return True

    # 2. Check for SPA/bot-protection text patterns
    # These patterns are more specific to avoid false positives
    spa_text_patterns = [
        # JS requirement messages (already covered by JS_REQUIRED_PATTERNS, but regex variants)
        r"this (?:page|site|website) requires javascript",
        r"you need (?:to enable )?javascript",
        r"enable javascript to (?:view|continue|access)",
        # Cloudflare/bot protection (only when it's the main content)
        r"^(?:\s*#?\s*)?(?:just a moment|one moment)\.{0,3}\s*$",
        r"checking (?:if the site connection is secure|your browser)",
        r"verifying (?:you are human|your browser)",
        r"ray id:",  # Cloudflare error pages include Ray ID
        # Common SPA loading states (only if very short content)
    ]
    for pattern in spa_text_patterns:
        if re.search(pattern, content_lower, re.MULTILINE):
            logger.debug(f"JS required: SPA text pattern matched '{pattern}'")
            return True

    # 3. Check for very short content (likely a JS-only page)
    # Strip markdown formatting for accurate length check
    text_only = re.sub(r"[#*_\[\]()>`\-|]", "", content)
    text_only = re.sub(r"!\[.*?\]\(.*?\)", "", text_only)  # Remove image refs
    text_only = re.sub(r"\[.*?\]\(.*?\)", "", text_only)  # Remove links
    text_only = " ".join(text_only.split()).strip()

    if len(text_only) < 100:
        logger.debug(f"JS required: content too short ({len(text_only)} chars)")
        return True

    # 4. Check for repetitive/placeholder content (SPA stub pages)
    # Some SPAs return minimal placeholder text
    unique_words = set(text_only.lower().split())
    if len(text_only) < 500 and len(unique_words) < 20:
        logger.debug(
            f"JS required: low content diversity "
            f"({len(unique_words)} unique words in {len(text_only)} chars)"
        )
        return True

    return False


def should_use_browser_for_domain(url: str, fallback_patterns: list[str]) -> bool:
    """Check if URL domain matches fallback patterns that need browser rendering.

    Args:
        url: URL to check
        fallback_patterns: List of domain patterns (e.g., ["twitter.com", "x.com"])

    Returns:
        True if domain matches any pattern
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        for pattern in fallback_patterns:
            pattern_lower = pattern.lower()
            # Match exact domain or subdomain
            if domain == pattern_lower or domain.endswith("." + pattern_lower):
                logger.debug(f"Domain {domain} matches fallback pattern {pattern}")
                return True
    except Exception as e:
        logger.debug("[Fetch] Domain pattern matching failed: {}", e)

    return False


def _merge_screenshot_result(result: FetchResult, pw_result: Any) -> FetchResult:
    """Attach a separately captured screenshot without dropping existing fields."""
    return FetchResult(
        content=result.content,
        strategy_used=result.strategy_used,
        title=result.title or getattr(pw_result, "title", None),
        url=result.url,
        final_url=result.final_url or getattr(pw_result, "final_url", None),
        metadata=result.metadata,
        cache_hit=result.cache_hit,
        screenshot_path=getattr(pw_result, "screenshot_path", None),
        static_content=result.static_content,
        browser_content=result.browser_content,
    )


async def _ensure_external_strategy_allowed(
    url: str,
    strategy_name: str,
    *,
    config: FetchConfig | None = None,
    allow_pattern_override: bool = False,
) -> None:
    """Enforce hard privacy guards before any external-only strategy runs."""
    from markitai.fetch_policy import (
        assess_url_for_remote,
        match_local_only,
    )

    if strategy_name not in {
        FetchStrategy.DEFUDDLE.value,
        FetchStrategy.JINA.value,
        FetchStrategy.CLOUDFLARE.value,
    }:
        return

    if _env_no_remote_fetch():
        raise FetchError(
            f"{strategy_name} is disabled by MARKITAI_NO_REMOTE_FETCH. "
            "Unset it before explicitly selecting a remote strategy."
        )

    from urllib.parse import urlparse

    domain = urlparse(url).netloc.lower()
    if (
        config is not None
        and not allow_pattern_override
        and match_local_only(domain, _build_local_only_patterns(config.policy))
    ):
        raise FetchError(
            f"{strategy_name} cannot fetch a URL matched by local-only policy. "
            "Use static/playwright, or explicitly select the remote strategy "
            "on the CLI to override the pattern for this public URL."
        )

    assessment = await assess_url_for_remote(url)
    if not assessment.allowed:
        reason = assessment.reason or "privacy policy"
        if reason == "non_global_address":
            reason = "hostname resolves to a non-public address"
        elif reason == "hostname_resolution_failed":
            reason = "hostname could not be resolved safely"
        elif reason == "credential_material":
            reason = "URL contains credential material"
        elif reason == "private_or_local_host":
            reason = "URL targets a private/local host"
        raise FetchError(
            f"{strategy_name} cannot fetch this URL: {reason}. "
            "Use static or playwright instead."
        )


async def _dispatch_strategy(
    url: str,
    strategy: FetchStrategy,
    config: FetchConfig,
    explicit_strategy: bool,
    screenshot_kwargs: dict[str, Any],
    screenshot_config: ScreenshotConfig | None,
    screenshot_dir: Path | None,
    renderer: Any | None,
) -> tuple[FetchResult, tuple[str | None, str | None] | None]:
    """Dispatch URL fetch to the appropriate strategy implementation.

    Returns:
        (result, validators_to_write) — validators is non-None for static strategy
    """
    validators_to_write: tuple[str | None, str | None] | None = None

    if strategy.value in EXTERNAL_STRATEGIES:
        await _ensure_external_strategy_allowed(
            url,
            strategy.value,
            config=config,
            allow_pattern_override=explicit_strategy,
        )
        if explicit_strategy:
            disclose_remote_use([strategy.value])
        elif not resolve_remote_consent(config, services=[strategy.value]):
            raise FetchError(
                f"{strategy.value} remote extraction is not allowed by "
                "fetch.remote_consent. Use an explicit -s strategy to opt in, "
                "or set fetch.remote_consent=always."
            )

    if strategy == FetchStrategy.AUTO:
        # Determine if browser should be tried first
        use_browser_first = False
        if not explicit_strategy:
            spa_cache = get_spa_domain_cache()
            if should_use_browser_for_domain(url, config.fallback_patterns):
                use_browser_first = True
            elif spa_cache.is_known_spa(url):
                spa_cache.record_hit(url)
                use_browser_first = True

        result = await _fetch_with_fallback(
            url, config, start_with_browser=use_browser_first, **screenshot_kwargs
        )
        # Propagate HTTP validators captured by the static path so AUTO
        # caching stores them for future conditional revalidation.
        etag = result.metadata.pop("_markitai_etag", None)
        last_modified = result.metadata.pop("_markitai_last_modified", None)
        if etag or last_modified:
            validators_to_write = (etag, last_modified)
    else:
        runner = get_runner(strategy)
        if runner is None:
            raise ValueError(f"Unknown fetch strategy: {strategy}")
        ctx = StrategyContext(
            config=config,
            session=get_default_session(),
            explicit=True,
            screenshot_kwargs={
                "renderer": renderer,
                "screenshot_config": screenshot_config,
                "screenshot_dir": screenshot_dir,
            },
        )
        result = await runner.fetch(url, ctx)
        if strategy == FetchStrategy.STATIC:
            # Fresh fetch went through the conditional variant; hand its
            # validators (possibly None) to the caching layer.
            validators_to_write = (
                result.metadata.pop("_markitai_etag", None),
                result.metadata.pop("_markitai_last_modified", None),
            )

    # AUTO already rejects anti-bot/CAPTCHA challenge pages internally and
    # tries the next strategy (_fetch_with_fallback calls _is_invalid_content
    # per strategy); an explicitly-chosen strategy has no next strategy to
    # fall back to and previously returned such a page as if it were the
    # real content. Only captcha_* reasons raise here — other
    # _is_invalid_content reasons (too_short, login_required, ...) are not
    # new information for an explicit choice and stay non-fatal to avoid
    # changing existing behavior for legitimately short/edge-case pages.
    is_invalid, reason = _is_invalid_content(result.content)
    if is_invalid and reason.startswith("captcha_"):
        raise FetchError(
            f"{strategy.value}: page returned an anti-bot/CAPTCHA challenge "
            f"({reason}) instead of real content. Automated fetching cannot "
            "solve this — try again later, from a different network, or "
            "open the URL in a browser."
        )

    return result, validators_to_write


# Remote strategies that can refuse a request server-side (4xx / auth).
_REMOTE_EXPLICIT_STRATEGIES = {
    FetchStrategy.JINA,
    FetchStrategy.DEFUDDLE,
    FetchStrategy.CLOUDFLARE,
}


def _classify_service_refusal(strategy: FetchStrategy, error: Exception) -> str | None:
    """Return a short reason when a remote service refused the request.

    Covers service-side 4xx refusals: blocked domain, rate limit, auth
    required. Returns None for everything else (network failures, local
    misconfiguration, 5xx, ...), which should propagate unchanged.
    """
    if isinstance(error, JinaRateLimitError):
        return "rate limited (free tier: 20 RPM)"
    if isinstance(error, JinaAPIError):
        if 400 <= error.status_code < 500:
            reason = re.sub(r"^Jina Reader API error \(\d+\): ", "", str(error))
            return f"HTTP {error.status_code}: {reason}"
        return None
    if not isinstance(error, FetchError):
        return None
    msg = str(error)
    if "Cloudflare API token and account ID required" in msg:
        # Local misconfiguration, not a service refusal — the actionable
        # credentials error (see utils.guidance) must surface as-is.
        return None
    if strategy == FetchStrategy.DEFUDDLE:
        if "rate limit" in msg.lower():
            return "rate limited"
        if re.search(r"HTTP 4\d\d", msg):
            return msg
        return None
    if strategy == FetchStrategy.CLOUDFLARE:
        if "rate limit" in msg.lower():
            return "rate limited"
        if re.search(r"\b(401|402|403|429|451)\b", msg) or "CF BR API error" in msg:
            return msg
        return None
    return None


def _jina_refusal_needs_key(error: Exception) -> bool:
    """True when a Jina refusal would be lifted by configuring an API key."""
    if isinstance(error, JinaRateLimitError):
        return True
    if isinstance(error, JinaAPIError):
        return error.status_code in (401, 402, 451) or (
            "anonymous" in str(error).lower()
        )
    return False


async def _fallback_after_refusal(
    *,
    url: str,
    strategy: FetchStrategy,
    refusal: str,
    original_error: Exception,
    config: FetchConfig,
    cache: FetchCache | None,
    skip_read_cache: bool,
    screenshot: bool,
    screenshot_dir: Path | None,
    screenshot_config: ScreenshotConfig | None,
    renderer: Any | None,
) -> FetchResult:
    """Handle a service-side refusal of an explicitly-selected remote strategy.

    Instead of dumping the raw service error, offer (interactive TTY) or
    apply (non-interactive, with a warning) a graceful fallback to the auto
    strategy chain. If the fallback also fails, report both failures compactly.
    """
    from markitai.utils.guidance import format_actionable_error, jina_api_key_hint

    name = strategy.value
    include_key_hint = strategy == FetchStrategy.JINA and _jina_refusal_needs_key(
        original_error
    )

    if not _should_fallback_after_refusal(name, refusal):
        steps: list[str] = []
        if include_key_hint:
            steps.append(jina_api_key_hint())
        steps.append(f"Or retry without '-s {name}' to use the auto strategy chain.")
        raise FetchError(
            format_actionable_error(
                f"{name} cannot fetch this URL ({refusal}): {url}", steps
            )
        ) from original_error

    logger.warning(
        "[Fetch] {} refused the request ({}); falling back to the auto "
        "strategy chain: {}",
        name,
        refusal,
        url,
    )
    try:
        return await fetch_url(
            url,
            FetchStrategy.AUTO,
            config,
            explicit_strategy=False,
            cache=cache,
            skip_read_cache=skip_read_cache,
            screenshot=screenshot,
            screenshot_dir=screenshot_dir,
            screenshot_config=screenshot_config,
            renderer=renderer,
        )
    except Exception as fallback_error:
        summary = [
            f"Failed to fetch {url}:",
            f"  - {name}: {refusal}",
            f"  - auto fallback: {fallback_error}",
        ]
        if include_key_hint:
            summary.extend(["", jina_api_key_hint()])
        raise FetchError("\n".join(summary)) from fallback_error


async def _resolve_cache_and_check(
    url: str,
    strategy: FetchStrategy,
    config: FetchConfig,
    cache: FetchCache | None,
    skip_read_cache: bool,
    cache_strategy: str | None,
) -> tuple[FetchResult | None, tuple[str | None, str | None] | None]:
    """Check cache and optionally try conditional fetch (ETag/Last-Modified).

    Returns:
        (result, None) — cache hit, done
        (None, (etag, last_modified)) — cache miss but got validators from conditional fetch
        (None, None) — no cache or miss without validators
    """
    if cache is None or skip_read_cache:
        return None, None

    # For static/auto strategy, try HTTP conditional request for efficiency
    use_conditional_cache = strategy in (FetchStrategy.STATIC, FetchStrategy.AUTO)

    if use_conditional_cache:
        (
            cached_result,
            cached_etag,
            cached_last_modified,
        ) = await cache.aget_with_validators(url, strategy=cache_strategy)

        # If we have validators, try conditional fetch (static strategy only)
        if cached_result is not None and (cached_etag or cached_last_modified):
            if strategy == FetchStrategy.STATIC or (
                strategy == FetchStrategy.AUTO
                and not should_use_browser_for_domain(url, config.fallback_patterns)
                and not get_spa_domain_cache().is_known_spa(url)
            ):
                try:
                    cond_result = await fetch_with_static_conditional(
                        url, cached_etag, cached_last_modified
                    )
                    if cond_result.not_modified:
                        # 304 Not Modified - use cached content
                        await cache.aupdate_accessed_at(url, strategy=cache_strategy)
                        return cached_result, None
                    elif cond_result.result is not None:
                        return cond_result.result, (
                            cond_result.etag,
                            cond_result.last_modified,
                        )
                except FetchError:
                    logger.debug(
                        f"[ConditionalFetch] Failed, falling back to normal fetch: {url}"
                    )

        # No validators but have cached result - use it directly
        elif cached_result is not None:
            return cached_result, None
    else:
        # Traditional cache check for non-conditional strategies
        cached_result = await cache.aget(url, strategy=cache_strategy)
        if cached_result is not None:
            return cached_result, None

    return None, None


async def fetch_url(
    url: str,
    strategy: FetchStrategy,
    config: FetchConfig,
    explicit_strategy: bool = False,
    cache: FetchCache | None = None,
    skip_read_cache: bool = False,
    *,
    screenshot: bool = False,
    screenshot_dir: Path | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    renderer: Any | None = None,
) -> FetchResult:
    """Fetch URL content using the specified strategy.

    Args:
        url: URL to fetch
        strategy: Fetch strategy to use
        config: Fetch configuration
        explicit_strategy: If True, don't fallback on error (user explicitly chose strategy)
        cache: Optional FetchCache for caching results
        skip_read_cache: If True, skip reading from cache but still write results (--no-cache)
        screenshot: If True, capture full-page screenshot (requires browser strategy)
        screenshot_dir: Directory to save screenshot
        screenshot_config: Screenshot settings (viewport, quality, etc.)
        renderer: Optional shared PlaywrightRenderer

    Returns:
        FetchResult with content and metadata

    Raises:
        FetchError: If fetch fails and no fallback available
        JinaRateLimitError: If --jina used and rate limit exceeded
    """
    # Use provided renderer or get global one if needed
    _renderer = renderer
    if _renderer is None and (
        strategy == FetchStrategy.PLAYWRIGHT
        or (strategy == FetchStrategy.AUTO and screenshot)
        or screenshot
    ):
        # Only initialize global renderer if browser strategy is likely to be used
        proxy = _detect_proxy() if getattr(config, "auto_proxy", True) else None
        _renderer = await _get_playwright_renderer(proxy=proxy, config=config)

    # Screenshot kwargs for browser fetching (used by _fetch_with_fallback)
    screenshot_kwargs: dict[str, Any] = {
        "renderer": _renderer,
        "screenshot_config": screenshot_config,
        "screenshot_dir": screenshot_dir,
    }

    # Include strategy in cache key when an explicit strategy is requested,
    # so that --playwright and --static don't return each other's cached results.
    cache_strategy: str | None = (
        strategy.value if explicit_strategy and strategy != FetchStrategy.AUTO else None
    )

    result, cache_validators_to_write = await _resolve_cache_and_check(
        url, strategy, config, cache, skip_read_cache, cache_strategy
    )

    # Fetch the content if not served from cache
    if result is None:
        try:
            result, new_validators = await _dispatch_strategy(
                url=url,
                strategy=strategy,
                config=config,
                explicit_strategy=explicit_strategy,
                screenshot_kwargs=screenshot_kwargs,
                screenshot_config=screenshot_config,
                screenshot_dir=screenshot_dir,
                renderer=_renderer,
            )
        except Exception as dispatch_error:
            refusal = (
                _classify_service_refusal(strategy, dispatch_error)
                if explicit_strategy and strategy in _REMOTE_EXPLICIT_STRATEGIES
                else None
            )
            if refusal is None:
                raise
            # Graceful fallback: the explicitly-selected remote service
            # refused the request (blocked domain / rate limit / auth).
            return await _fallback_after_refusal(
                url=url,
                strategy=strategy,
                refusal=refusal,
                original_error=dispatch_error,
                config=config,
                cache=cache,
                skip_read_cache=skip_read_cache,
                screenshot=screenshot,
                screenshot_dir=screenshot_dir,
                screenshot_config=screenshot_config,
                renderer=_renderer,
            )
        if new_validators is not None:
            cache_validators_to_write = new_validators

    # Capture screenshot separately if requested and not already captured
    if screenshot and result.screenshot_path is None:
        try:
            from markitai.fetch_playwright import (
                fetch_with_playwright,
                is_playwright_available,
            )

            if is_playwright_available():
                logger.debug("[URL] Capturing screenshot separately via playwright")
                pw_result = await fetch_with_playwright(
                    url,
                    **_get_playwright_fetch_kwargs(
                        url,
                        config,
                        screenshot_config=screenshot_config,
                        output_dir=screenshot_dir,
                        renderer=_renderer,
                    ),
                )
                result = _merge_screenshot_result(result, pw_result)
            else:
                logger.debug("[URL] Screenshot requested but playwright not available")
        except Exception as e:
            logger.warning(f"[URL] Screenshot capture failed: {e}")

    # Cache the result (for non-static strategies that don't use conditional caching)
    if cache is not None and not result.cache_hit:
        if cache_validators_to_write is not None:
            await cache.aset_with_validators(
                url,
                result,
                cache_validators_to_write[0],
                cache_validators_to_write[1],
                strategy=cache_strategy,
            )
        else:
            await cache.aset(url, result, strategy=cache_strategy)

    return result


def _is_invalid_content(content: str) -> tuple[bool, str]:
    """Check if fetched content is invalid (JS error page, login prompt,
    anti-bot/CAPTCHA challenge, etc.).

    Args:
        content: Fetched content to check

    Returns:
        Tuple of (is_invalid, reason). CAPTCHA/anti-bot reasons are
        prefixed ``captcha_`` — callers may treat these as harder failures
        than the others (e.g. raise instead of silently falling back),
        since a solved challenge is definitionally unavailable to an
        automated fetch.
    """
    if not content or not content.strip():
        return True, "empty"

    # Check for common invalid content patterns
    invalid_patterns = [
        (r"JavaScript is (not available|disabled)", "javascript_disabled"),
        (r"Please enable JavaScript", "javascript_required"),
        (r"switch to a supported browser", "unsupported_browser"),
        (r"Something went wrong.*let's give it another shot", "error_page"),
        (r"Log in.*Sign up.*to continue", "login_required"),
        (r"You must be logged in", "login_required"),
        # Geetest (极验) — confirmed against a real bilibili.com challenge
        # page returned during rate-limit-triggered testing (2026-07-07).
        (r"智能验证检测中", "captcha_geetest"),
        (r"由极验提供技术支持", "captcha_geetest"),
        # Widely-documented anti-bot vendor signatures (Cloudflare browser
        # check, reCAPTCHA, hCaptcha) — stable, well-known markers, not
        # captured against a live challenge this session.
        (r"Checking your browser before accessing", "captcha_cloudflare"),
        (r"cf-browser-verification|cf_chl_", "captcha_cloudflare"),
        (r"g-recaptcha|google\.com/recaptcha", "captcha_recaptcha"),
        (r"hcaptcha\.com|h-captcha", "captcha_hcaptcha"),
    ]

    for pattern, reason in invalid_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return True, reason

    # Check content length (after removing markdown links and images)
    clean_content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)  # Remove images
    clean_content = re.sub(r"\[[^\]]*\]\([^)]+\)", "", clean_content)  # Remove links
    clean_content = re.sub(
        r"[#\-*_>\[\]`|]", "", clean_content
    )  # Remove markdown syntax
    clean_content = " ".join(clean_content.split())  # Normalize whitespace

    if len(clean_content) < 30:
        return True, "too_short"

    return False, ""


def _build_local_only_patterns(policy: FetchPolicyConfig) -> list[str]:
    """Build effective local-only patterns from config + NO_PROXY env var.

    When ``inherit_no_proxy`` is True (default), patterns from the NO_PROXY
    environment variable are merged into the configured ``local_only_patterns``
    (deduplicated, config patterns take precedence).
    """
    import os

    from markitai.fetch_policy import parse_no_proxy

    patterns = list(policy.local_only_patterns)
    if policy.inherit_no_proxy:
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
        if no_proxy:
            inherited = parse_no_proxy(no_proxy)
            seen = set(patterns)
            for p in inherited:
                if p not in seen:
                    patterns.append(p)
                    seen.add(p)
    return patterns


async def _fetch_with_fallback(
    url: str,
    config: FetchConfig,
    start_with_browser: bool = False,
    renderer: Any | None = None,
    **screenshot_kwargs: Any,
) -> FetchResult:
    """Fetch URL with automatic fallback between strategies.

    Args:
        url: URL to fetch
        config: Fetch configuration
        start_with_browser: If True, try browser first (for known JS domains)
        renderer: Optional shared PlaywrightRenderer
        **screenshot_kwargs: Screenshot options (screenshot, screenshot_dir, screenshot_config)

    Returns:
        FetchResult from first successful strategy
    """
    from urllib.parse import urlparse

    from markitai.fetch_policy import (
        FetchPolicyEngine,
        is_private_or_local_domain,
        url_contains_credentials,
    )

    errors = []

    domain = urlparse(url).netloc.lower()
    engine = FetchPolicyEngine()
    jina_key = config.jina.get_resolved_api_key()
    profile = config.domain_profiles.get(domain)
    domain_prefer = profile.prefer_strategy if profile else None
    domain_priority = profile.strategy_priority if profile else None

    # Build effective local_only_patterns (config + optional NO_PROXY merge)
    effective_local_only = _build_local_only_patterns(config.policy)

    decision = engine.decide(
        domain=domain,
        known_spa=start_with_browser,
        explicit_strategy=config.strategy if config.strategy != "auto" else None,
        fallback_patterns=config.fallback_patterns,
        policy_enabled=config.policy.enabled,
        has_jina_key=bool(jina_key),
        domain_prefer_strategy=domain_prefer,
        global_strategy_priority=config.policy.strategy_priority,
        domain_strategy_priority=domain_priority,
        local_only_patterns=effective_local_only,
    )
    strategies = decision.order[: config.policy.max_strategy_hops]
    if is_private_or_local_domain(domain) or url_contains_credentials(url):
        strategies = [s for s in strategies if s in {"static", "playwright"}]
        if not strategies:
            strategies = list(LOCAL_STRATEGIES)

    # Remote-fetch consent gate: defuddle/jina/cloudflare send the URL to
    # third-party services. When the answer is already known (config/env/
    # cached), filter up front; when it would need an interactive prompt,
    # DEFER it — the chain is local-first, so most fetches succeed via
    # static/playwright and the user is never asked (lazy consent).
    # A strategy selected in config is still governed by remote_consent.
    # CLI-explicit remote choices seed the process decision to True before
    # reaching this chain, so they remain deliberate opt-ins without creating
    # a second bypass seam here.
    _consent_gated = True
    if _consent_gated and any(s in EXTERNAL_STRATEGIES for s in strategies):
        _peeked = peek_remote_consent(config)
        if _peeked is False:
            strategies = [s for s in strategies if s not in EXTERNAL_STRATEGIES]
            if not strategies:
                strategies = list(LOCAL_STRATEGIES)

    # Resolve domain profile for telemetry
    domain_profile_applied = decision.reason == "spa_or_pattern"

    ctx = StrategyContext(
        config=config,
        session=get_default_session(),
        explicit=False,
        screenshot_kwargs={
            "renderer": renderer,
            "screenshot_config": screenshot_kwargs.get("screenshot_config"),
            "screenshot_dir": screenshot_kwargs.get("screenshot_dir"),
        },
    )

    for strat in strategies:
        runner = get_runner(strat)
        if runner is None:
            continue

        skip_reason = runner.unavailable_reason(ctx)
        if skip_reason is not None:
            if skip_reason:
                errors.append(f"{strat}: {skip_reason}")
            continue

        if runner.requires_remote_consent:
            try:
                await _ensure_external_strategy_allowed(url, strat, config=config)
            except FetchError as e:
                logger.debug(f"[Fetch] Skipping {strat}: {e}")
                errors.append(f"{strat}: {e}")
                continue

            # Lazy consent: only when the chain actually reaches a remote
            # strategy (local ones failed) do we resolve — and possibly
            # prompt for — remote-fetch consent
            if _consent_gated and not resolve_remote_consent(
                config,
                services=[s for s in strategies if s in EXTERNAL_STRATEGIES],
            ):
                logger.debug(f"[Fetch] Skipping {strat}: no remote-fetch consent")
                errors.append(f"{strat}: skipped (no remote-fetch consent)")
                continue
            disclose_remote_use([s for s in strategies if s in EXTERNAL_STRATEGIES])

        try:
            result = await runner.fetch(url, ctx)

            # Static-only follow-up: a JS-rendered page can look like a
            # successful static fetch — learn the domain for future
            # browser-first requests and fall through to the next strategy.
            if strat == "static" and detect_js_required(result.content):
                spa_cache = get_spa_domain_cache()
                spa_cache.record_spa_domain(url)
                errors.append(f"{strat}: page requires JavaScript rendering")
                continue

            # Validate content quality before accepting
            is_invalid, reason = _is_invalid_content(result.content)
            if is_invalid:
                logger.debug(f"Strategy {strat} returned invalid content: {reason}")
                errors.append(f"{strat}: invalid content ({reason})")
                continue

            # Add telemetry
            result.metadata.update(
                {
                    "policy_reason": decision.reason,
                    "policy_order": strategies,
                    "profile_applied": domain_profile_applied,
                }
            )
            return result

        except JinaRateLimitError as e:
            errors.append(str(e))
            logger.warning(str(e))
            continue
        except FetchError as e:
            errors.append(f"{strat}: {e}")
            logger.debug(f"Strategy {strat} failed: {e}")
            continue
        except Exception as e:
            errors.append(f"{strat}: {e}")
            logger.debug(f"Strategy {strat} failed: {e}")
            continue

    # All strategies failed
    detail = "\n".join(f"  - {e}" for e in errors) or (
        "  - no strategy was applicable (all were skipped)"
    )
    raise FetchError(f"All fetch strategies failed for {url}:\n{detail}")
