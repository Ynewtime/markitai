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
import time
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
        ScreenshotConfig,
    )


from markitai.fetch_cache import FetchCache as FetchCache
from markitai.fetch_cache import SPADomainCache as SPADomainCache

# Shared mutable fetch state (clients, caches, proxy, consent) lives on the
# process-wide FetchSession (markitai.fetch_session); consent logic lives in
# markitai.fetch_consent, screenshot helpers in markitai.fetch_screenshot,
# per-strategy fetch implementations in markitai.fetch_strategies, and
# helpers shared between this orchestrator and the strategies in
# markitai.fetch_support. They are re-exported here so the public
# markitai.fetch API (and its non-patch import surface) stays stable.
from markitai.fetch_consent import (
    _REMOTE_SERVICE_LABELS as _REMOTE_SERVICE_LABELS,
)
from markitai.fetch_consent import (
    _env_no_remote_fetch as _env_no_remote_fetch,
)
from markitai.fetch_consent import (
    _remote_service_names as _remote_service_names,
)
from markitai.fetch_consent import (
    _should_fallback_after_refusal as _should_fallback_after_refusal,
)
from markitai.fetch_consent import (
    disclose_remote_use as disclose_remote_use,
)
from markitai.fetch_consent import (
    peek_cached_remote_consent as peek_cached_remote_consent,
)
from markitai.fetch_consent import (
    peek_remote_consent as peek_remote_consent,
)
from markitai.fetch_consent import (
    reset_explicit_fallback_decision as reset_explicit_fallback_decision,
)
from markitai.fetch_consent import (
    reset_remote_consent as reset_remote_consent,
)
from markitai.fetch_consent import (
    resolve_remote_consent as resolve_remote_consent,
)
from markitai.fetch_consent import (
    set_remote_consent as set_remote_consent,
)
from markitai.fetch_consent import (
    set_remote_consent_prompt_allowed as set_remote_consent_prompt_allowed,
)
from markitai.fetch_http import (
    get_static_http_client as get_static_http_client,
)
from markitai.fetch_policy import (
    build_local_only_patterns as _build_local_only_patterns,
)
from markitai.fetch_screenshot import (
    _url_to_screenshot_filename as _url_to_screenshot_filename,
)
from markitai.fetch_session import (
    FetchSession as FetchSession,
)
from markitai.fetch_session import (
    get_default_session as get_default_session,
)
from markitai.fetch_session import (
    reset_default_session as reset_default_session,
)
from markitai.fetch_strategies import (
    CloudflareRunner as CloudflareRunner,
)
from markitai.fetch_strategies import (
    DefuddleRunner as DefuddleRunner,
)
from markitai.fetch_strategies import (
    JinaRunner as JinaRunner,
)
from markitai.fetch_strategies import (
    PlaywrightRunner as PlaywrightRunner,
)
from markitai.fetch_strategies import (
    StaticRunner as StaticRunner,
)
from markitai.fetch_strategies import (
    StrategyContext as StrategyContext,
)
from markitai.fetch_strategies import (
    StrategyRunner as StrategyRunner,
)
from markitai.fetch_strategies import (
    fetch_with_cloudflare as fetch_with_cloudflare,
)
from markitai.fetch_strategies import (
    fetch_with_defuddle as fetch_with_defuddle,
)
from markitai.fetch_strategies import (
    fetch_with_jina as fetch_with_jina,
)
from markitai.fetch_strategies import (
    fetch_with_static as fetch_with_static,
)
from markitai.fetch_strategies import (
    fetch_with_static_conditional as fetch_with_static_conditional,
)
from markitai.fetch_strategies import (
    get_cf_semaphore as get_cf_semaphore,
)
from markitai.fetch_strategies import (
    get_runner as get_runner,
)
from markitai.fetch_strategies._shared import (
    _get_markitdown as _get_markitdown,
)
from markitai.fetch_strategies.jina import (
    _extract_jina_error_message as _extract_jina_error_message,
)
from markitai.fetch_strategies.jina import (
    _get_jina_client as _get_jina_client,
)
from markitai.fetch_strategies.jina import (
    _get_jina_rate_limiter as _get_jina_rate_limiter,
)
from markitai.fetch_strategies.static import (
    _extract_markdown_title as _extract_markdown_title,
)
from markitai.fetch_support import (
    _detect_proxy as _detect_proxy,
)
from markitai.fetch_support import (
    _get_playwright_advanced_kwargs as _get_playwright_advanced_kwargs,
)
from markitai.fetch_support import (
    _get_playwright_fetch_kwargs as _get_playwright_fetch_kwargs,
)
from markitai.fetch_support import (
    _resolve_playwright_profile_overrides as _resolve_playwright_profile_overrides,
)
from markitai.fetch_support import (
    _url_to_session_key as _url_to_session_key,
)
from markitai.fetch_types import (
    CRITICAL_INVALID_REASONS as CRITICAL_INVALID_REASONS,
)
from markitai.fetch_types import (
    ConditionalFetchResult as ConditionalFetchResult,
)
from markitai.fetch_types import FetchError as FetchError
from markitai.fetch_types import FetchResult as FetchResult
from markitai.fetch_types import FetchStrategy as FetchStrategy
from markitai.fetch_types import JinaAPIError as JinaAPIError
from markitai.fetch_types import JinaRateLimitError as JinaRateLimitError


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


def get_proxy_for_url(url: str) -> str:
    """Get the proxy to use for *url*, honoring NO_PROXY.

    This is the unified entry point for proxy resolution in this layer.
    Backends below ``markitai.fetch`` cannot import it (import-linter keeps
    strategies below the orchestrator); they resolve the candidate proxy with
    ``_detect_proxy()`` and the connection-level NO_PROXY bypass is applied
    for them by :func:`markitai.fetch_http.resolve_proxy_for_url`. Both paths
    share :meth:`FetchSession.is_proxy_bypassed`.

    To disable proxying entirely, set ``NO_PROXY=*`` — the standard mechanism,
    which every path here now respects. There is deliberately no bespoke
    config switch duplicating it.

    Args:
        url: URL being fetched (checked against NO_PROXY patterns)

    Returns:
        Proxy URL string or empty string if no proxy should be used
    """
    proxy = _detect_proxy()
    if not proxy:
        return ""

    if get_default_session().is_proxy_bypassed(url):
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


def detect_js_required(content: str, *, allow_short_content: bool = False) -> bool:
    """Detect if content indicates JavaScript rendering is required.

    Note: This function receives MARKDOWN content (converted by markitdown),
    not raw HTML. Detection strategies must work with Markdown text.

    Uses multiple detection strategies:
    1. Simple string matching for common JS-required messages
    2. Content patterns that survive Markdown conversion
    3. Content length and quality checks

    Args:
        content: Markdown content to check (from markitdown conversion)
        allow_short_content: Trust the native extractor's accepted content;
            still reject explicit JS requirements and loading placeholders.

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

    if allow_short_content:
        # Native extraction has already passed a content-aware quality gate.
        # A second arbitrary length/word-diversity threshold would turn valid
        # notes, formulas and short social posts into expensive browser jobs.
        placeholder = re.sub(r"[^\w\s]", " ", text_only.lower())
        return bool(
            re.fullmatch(
                r"\s*(?:(?:loading|please wait|loading please wait|"
                r"正在加载|加载中|请稍候|読み込み中|読み込み中です|로딩 중|로딩 중입니다)\s*)+",
                placeholder,
            )
        )

    if len(text_only) < 100:
        logger.debug(f"JS required: content too short ({len(text_only)} chars)")
        return True

    # 4. Check for repetitive/placeholder content (SPA stub pages)
    # Some SPAs return minimal placeholder text
    # Chinese/Japanese/Korean prose often has no spaces. Count CJK characters
    # individually so a complete short article is not mistaken for a SPA stub.
    tokenized = re.sub(
        r"([\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af])",
        r" \1 ",
        text_only.lower(),
    )
    unique_words = set(tokenized.split())
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
        screenshot_tiles=list(getattr(pw_result, "screenshot_tiles", []) or []),
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
    from markitai.fetch_consent import assess_remote_url
    from markitai.fetch_policy import match_local_only

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

    assessment = await assess_remote_url(
        url,
        config or "always",
        [strategy_name],
        consent_granted=allow_pattern_override,
    )
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
    reason = _challenge_page_reason(result)
    if reason is not None:
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
    cache_ttl_seconds: int | None = None,
    no_cache_patterns: list[str] | None = None,
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
            cache_ttl_seconds=cache_ttl_seconds,
            no_cache_patterns=no_cache_patterns,
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
    max_age_seconds: int | None = None,
) -> tuple[
    FetchResult | None, tuple[str | None, str | None] | None, FetchResult | None
]:
    """Check cache and optionally try conditional fetch (ETag/Last-Modified).

    A conditional request that comes back 200 is held to the same standard
    as a fresh fetch: under AUTO it must pass the static checks (JavaScript
    shell, challenge page, ...), explicit static rejects challenge pages.
    A rejected revalidation falls through to a normal fetch (which can fall
    back to the browser) and never overwrites the cached entry by itself;
    the cached entry is handed back as the stale fallback for when that
    normal fetch fails too.

    Args:
        max_age_seconds: Reuse window for entries without validators
            (``cache.fetch_ttl_seconds``); older ones count as a miss.

    Returns:
        (result, None, None) — cache hit, done
        (result, (etag, last_modified), None) — revalidated with new content
        (None, None, cached) — revalidation rejected; ``cached`` is the
            entry to serve if the normal fetch fails
        (None, None, None) — no cache or miss
    """
    if cache is None or skip_read_cache:
        return None, None, None

    # For static/auto strategy, try HTTP conditional request for efficiency
    use_conditional_cache = strategy in (FetchStrategy.STATIC, FetchStrategy.AUTO)

    if use_conditional_cache:
        (
            cached_result,
            cached_etag,
            cached_last_modified,
        ) = await cache.aget_with_validators(
            url, strategy=cache_strategy, max_age_seconds=max_age_seconds
        )

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
                        return cached_result, None, None
                    elif cond_result.result is not None:
                        fresh = cond_result.result
                        if strategy == FetchStrategy.AUTO:
                            rejection, _ = _static_result_rejection(fresh)
                        else:
                            rejection = _challenge_page_reason(fresh)
                        if rejection is None:
                            return (
                                fresh,
                                (cond_result.etag, cond_result.last_modified),
                                None,
                            )
                        logger.debug(
                            "[ConditionalFetch] Revalidated content rejected "
                            f"({rejection}); running a full fetch: {url}"
                        )
                        return None, None, cached_result
                except FetchError:
                    logger.debug(
                        f"[ConditionalFetch] Failed, falling back to normal fetch: {url}"
                    )

        # No validators but have cached result - use it directly
        elif cached_result is not None:
            return cached_result, None, None
    else:
        # Traditional cache check for non-conditional strategies
        cached_result = await cache.aget(
            url, strategy=cache_strategy, max_age_seconds=max_age_seconds
        )
        if cached_result is not None:
            return cached_result, None, None

    return None, None, None


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
    cache_ttl_seconds: int | None = None,
    no_cache_patterns: list[str] | None = None,
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
        screenshot_config: Screenshot settings (viewport, quality, etc.).
            With ``screenshot_only`` set, a failed text extraction still
            completes from the screenshot.
        renderer: Optional shared PlaywrightRenderer
        cache_ttl_seconds: Reuse window for cached pages without HTTP
            validators (``cache.fetch_ttl_seconds``); defaults to the value
            registered on the fetch session.
        no_cache_patterns: URL globs whose cached fetches are not read
            (``cache.no_cache_patterns`` / ``--no-cache-for``); defaults to
            the patterns registered on the fetch session.

    Returns:
        FetchResult with content and metadata. When a requested screenshot
        could not be captured, ``metadata["screenshot_error"]`` says why.
        When a cached page's revalidation was rejected (it changed into a
        challenge page or a JavaScript shell) and the full fetch failed too,
        the cached copy is returned with ``metadata["stale"] = True`` and
        the reason in ``metadata["fetch_warning"]``; policy refusals
        (private network, local-only, consent) still raise.

    Raises:
        FetchError: If fetch fails and no fallback available
        JinaRateLimitError: If -s jina used and rate limit exceeded
    """
    from markitai.fetch_cache import url_matches_cache_patterns
    from markitai.fetch_policy import public_network_only

    session = get_default_session()
    if cache_ttl_seconds is None:
        cache_ttl_seconds = session.fetch_cache_ttl_seconds
    if no_cache_patterns is None:
        no_cache_patterns = session.fetch_cache_skip_patterns
    if (
        cache is not None
        and not skip_read_cache
        and url_matches_cache_patterns(url, no_cache_patterns)
    ):
        skip_read_cache = True

    if public_network_only.get():
        # Trusted callers may have cached a public URL that redirects inward.
        # Keep that content outside the authority of anonymous remote jobs.
        cache = None
        skip_read_cache = True
    screenshot_only = bool(
        screenshot and getattr(screenshot_config, "screenshot_only", False) is True
    )
    # Use provided renderer or get global one if needed
    _renderer = renderer
    if _renderer is None and (
        strategy == FetchStrategy.PLAYWRIGHT
        or (strategy == FetchStrategy.AUTO and screenshot)
        or screenshot
    ):
        # Only initialize global renderer if browser strategy is likely to be used
        proxy = get_proxy_for_url(url) or None
        _renderer = await _get_playwright_renderer(proxy=proxy, config=config)

    # Screenshot kwargs for browser fetching (used by _fetch_with_fallback)
    screenshot_kwargs: dict[str, Any] = {
        "renderer": _renderer,
        "screenshot": screenshot,
        "screenshot_config": screenshot_config,
        "screenshot_dir": screenshot_dir,
    }

    # Include strategy in cache key when an explicit strategy is requested,
    # so that -s playwright and -s static don't return each other's cached results.
    cache_strategy: str | None = (
        strategy.value if explicit_strategy and strategy != FetchStrategy.AUTO else None
    )

    (
        result,
        cache_validators_to_write,
        stale_fallback,
    ) = await _resolve_cache_and_check(
        url,
        strategy,
        config,
        cache,
        skip_read_cache,
        cache_strategy,
        max_age_seconds=cache_ttl_seconds,
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
            if refusal is not None:
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
                    cache_ttl_seconds=cache_ttl_seconds,
                    no_cache_patterns=no_cache_patterns,
                )
            if stale_fallback is None or _is_policy_refusal(dispatch_error):
                if screenshot_only and isinstance(dispatch_error, FetchError):
                    # Never cached: the text layer is missing, not empty.
                    return await _screenshot_only_fallback(
                        url,
                        dispatch_error,
                        config,
                        screenshot_dir=screenshot_dir,
                        screenshot_config=screenshot_config,
                        renderer=_renderer,
                    )
                raise
            # The page changed into something unusable (a challenge page, a
            # JavaScript shell) and nothing else could fetch it: the last
            # good copy beats an error.
            result = _mark_stale(stale_fallback, url, dispatch_error)
            new_validators = None
        if new_validators is not None:
            cache_validators_to_write = new_validators

    screenshot_error: str | None = None
    if result.cache_hit:
        # A cached screenshot path belongs to the run that captured it; only
        # reuse it when this run asked for one in the same directory.
        result = _reuse_cached_screenshot(result, screenshot, screenshot_dir)
    if screenshot and result.screenshot_path is None:
        # Capture screenshot separately if requested and not already captured
        result, screenshot_error = await _capture_missing_screenshot(
            url,
            result,
            config,
            screenshot_dir=screenshot_dir,
            screenshot_config=screenshot_config,
            renderer=_renderer,
        )
    elif screenshot and not result.screenshot_tiles and result.screenshot_path:
        result.screenshot_tiles = [result.screenshot_path]

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

    if screenshot_error is not None:
        # Recorded after caching: it describes this run, not the page.
        result.metadata["screenshot_error"] = screenshot_error
    return result


_POLICY_REFUSAL_MARKERS = (
    "non-public address",
    "Fetch target refused",
    "local-only policy",
    "private/local host",
    "credential material",
    "remote_consent",
    "hostname could not be resolved safely",
)


def _is_policy_refusal(error: BaseException) -> bool:
    """Whether a fetch failed because policy forbade it, not because it broke.

    A refusal must stay a refusal: serving the cached copy instead would
    let a cache entry written under a looser policy outlive the rule.
    """
    if isinstance(error, PermissionError):
        return True
    message = str(error)
    return any(marker in message for marker in _POLICY_REFUSAL_MARKERS)


def _mark_stale(cached: FetchResult, url: str, error: BaseException) -> FetchResult:
    """Label a cached result served because its revalidation could not."""
    first_line = str(error).strip().splitlines()[0] if str(error).strip() else ""
    warning = (
        "Served the cached copy: the page changed into content that could "
        f"not be used and a fresh fetch failed ({first_line or type(error).__name__})"
    )
    logger.warning("[Fetch] {}: {}", warning, url)
    cached.metadata["stale"] = True
    cached.metadata["fetch_warning"] = warning
    return cached


def _reuse_cached_screenshot(
    result: FetchResult, screenshot: bool, screenshot_dir: Path | None
) -> FetchResult:
    """Keep a cache hit's screenshot only when it is this run's to use."""
    from markitai.fetch_screenshot import existing_screenshot_tiles

    path = result.screenshot_path
    usable = (
        screenshot
        and path is not None
        and path.is_file()
        and (
            screenshot_dir is None
            or path.parent.resolve() == Path(screenshot_dir).resolve()
        )
    )
    if usable and path is not None:
        result.screenshot_tiles = existing_screenshot_tiles(path)
    else:
        result.screenshot_path = None
        result.screenshot_tiles = []
    return result


def _error_reason(error: Exception) -> str:
    """A one-line, user-facing reason for a failed browser capture."""
    from markitai.fetch_playwright import is_playwright_browser_installed

    if "Failed to launch Chromium" in str(error) and not (
        is_playwright_browser_installed(use_cache=False)
    ):
        # The launch error carries a multi-line install guide; one line is
        # enough for a warning or a --json field.
        return "Chromium browser is not installed (run 'markitai doctor --fix')"
    if isinstance(error, FetchError) and str(error):
        return str(error)
    from markitai.utils.text import format_error_message

    return format_error_message(error)


def _screenshot_unavailable_reason() -> str | None:
    """Why a local browser screenshot cannot be taken at all, if it cannot.

    A missing Chromium download surfaces from the launch itself, with the
    install guidance attached.
    """
    from markitai.fetch_playwright import is_playwright_available
    from markitai.utils.errors import extra_install_command

    if not is_playwright_available():
        return (
            "Playwright is not installed "
            f"(install it with: {extra_install_command('browser')})"
        )
    return None


async def _capture_missing_screenshot(
    url: str,
    result: FetchResult,
    config: FetchConfig,
    *,
    screenshot_dir: Path | None,
    screenshot_config: ScreenshotConfig | None,
    renderer: Any | None,
) -> tuple[FetchResult, str | None]:
    """Capture the screenshot the content strategy did not take.

    Returns:
        ``(result, error)`` — ``error`` says why no screenshot was captured
        (None on success), so callers can warn instead of staying silent.
    """
    from markitai.fetch_playwright import fetch_with_playwright

    unavailable = _screenshot_unavailable_reason()
    if unavailable is not None:
        logger.warning(f"[URL] Screenshot not captured: {unavailable}")
        return result, unavailable

    try:
        logger.debug("[URL] Capturing screenshot separately via playwright")
        pw_result = await fetch_with_playwright(
            url,
            **_get_playwright_fetch_kwargs(
                url,
                config,
                screenshot_config=screenshot_config,
                output_dir=screenshot_dir,
                renderer=renderer,
            ),
        )
    except Exception as e:
        reason = _error_reason(e)
        logger.warning(f"[URL] Screenshot capture failed: {reason}")
        return result, reason

    merged = _merge_screenshot_result(result, pw_result)
    if merged.screenshot_path is None:
        reason = str(
            (getattr(pw_result, "metadata", None) or {}).get("screenshot_error")
            or "the browser did not produce a screenshot"
        )
        logger.warning(f"[URL] Screenshot not captured: {reason}")
        return merged, reason
    return merged, None


async def _screenshot_only_fallback(
    url: str,
    extraction_error: Exception,
    config: FetchConfig,
    *,
    screenshot_dir: Path | None,
    screenshot_config: ScreenshotConfig | None,
    renderer: Any | None,
) -> FetchResult:
    """Finish a ``--screenshot-only`` fetch from the screenshot alone.

    Screenshot-only mode exists for pages whose text cannot be extracted
    (canvas apps, image-only pages). When every text strategy failed, the
    page is still captured; only when that capture fails too is the fetch
    an error, reporting both causes.
    """
    unavailable = _screenshot_unavailable_reason()
    if unavailable is not None:
        raise FetchError(
            f"{extraction_error}\n  - screenshot: {unavailable}"
        ) from extraction_error

    from markitai.fetch_playwright import fetch_with_playwright
    from markitai.fetch_strategies.playwright import _playwright_strategy_label

    try:
        pw_result = await fetch_with_playwright(
            url,
            **_get_playwright_fetch_kwargs(
                url,
                config,
                screenshot_config=screenshot_config,
                output_dir=screenshot_dir,
                renderer=renderer,
            ),
        )
    except Exception as e:
        raise FetchError(
            f"{extraction_error}\n  - screenshot: {_error_reason(e)}"
        ) from extraction_error

    if pw_result.screenshot_path is None:
        reason = pw_result.metadata.get("screenshot_error") or (
            "the browser did not produce a screenshot"
        )
        raise FetchError(
            f"{extraction_error}\n  - screenshot: {reason}"
        ) from extraction_error

    logger.warning(
        "[URL] Text extraction failed; continuing from the screenshot "
        f"(--screenshot-only): {url}"
    )
    metadata = dict(pw_result.metadata)
    metadata["extraction_error"] = str(extraction_error)
    return FetchResult(
        content=pw_result.content or "",
        strategy_used=_playwright_strategy_label(pw_result.metadata),
        title=pw_result.title,
        url=url,
        final_url=pw_result.final_url,
        metadata=metadata,
        screenshot_path=pw_result.screenshot_path,
        screenshot_tiles=list(pw_result.screenshot_tiles or []),
    )


def _is_invalid_content(
    content: str, *, allow_short_content: bool = False
) -> tuple[bool, str]:
    """Check if fetched content is invalid (JS error page, login prompt,
    anti-bot/CAPTCHA challenge, etc.).

    Args:
        content: Fetched content to check
        allow_short_content: The native extractor has already accepted this
            content. Skip only the duplicate length gate, never error patterns.

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

    if not allow_short_content and len(clean_content) < 30:
        return True, "too_short"

    return False, ""


def _is_html_result(result: FetchResult) -> bool:
    """Whether a fetch result came from an HTML page (or unknown content).

    Only an HTML page can be a JavaScript shell. A text/plain note, a CSV
    or a PDF is the document itself, however short; the static strategy
    records ``content_kind`` so AUTO can tell them apart.
    """
    kind = result.metadata.get("content_kind")
    return kind is None or kind == "html"


def _challenge_page_reason(result: FetchResult) -> str | None:
    """The ``captcha_*`` reason when an HTML result is an anti-bot page.

    Documents are exempt, as in :func:`_static_result_rejection`: a text
    file or PDF that mentions ``g-recaptcha`` or ``cf_chl_`` (a write-up
    about CAPTCHAs, a code sample) is the document itself.
    """
    if not _is_html_result(result):
        return None
    is_invalid, reason = _is_invalid_content(result.content)
    if is_invalid and reason.startswith("captcha_"):
        return reason
    return None


def _native_accepted(result: FetchResult) -> bool:
    """Whether a content-aware extractor already accepted this result."""
    return result.metadata.get("converter") == "native-html" or (
        result.metadata.get("_enricher_source")
        in {"fxtwitter", "fxtwitter_article", "oembed"}
    )


def _static_result_rejection(result: FetchResult) -> tuple[str | None, bool]:
    """Why AUTO would not accept a static result, if it would not.

    Shared by the fresh static attempt in the AUTO chain and by a
    conditional revalidation that came back 200, so a page that changed
    into a challenge page or a JavaScript shell is rejected either way.

    Returns:
        ``(reason, needs_browser)``. ``reason`` is None when the result is
        acceptable; ``needs_browser`` is True when the page looked
        JavaScript-rendered (the domain is then learned as a SPA).
    """
    if not _is_html_result(result):
        # Documents skip the JS heuristics and the length gate entirely:
        # short plain text is complete, and a PDF quoting "log in to
        # continue" is not a login wall.
        if not result.content or not result.content.strip():
            return "invalid content (empty)", False
        return None, False

    native_accepted = _native_accepted(result)
    if detect_js_required(result.content, allow_short_content=native_accepted):
        return "page requires JavaScript rendering", True
    is_invalid, reason = _is_invalid_content(
        result.content, allow_short_content=native_accepted
    )
    if is_invalid:
        return f"invalid content ({reason})", False
    return None, False


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
    logger.debug("[Fetch] Policy {}: {}", decision.reason, decision.order)
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

        attempt_started = time.perf_counter()

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

            if strat == "static":
                # A JS-rendered page can look like a successful static
                # fetch — learn the domain for future browser-first
                # requests and fall through to the next strategy. Non-HTML
                # documents (text, CSV, PDF, ...) are never JS shells, so
                # they neither trigger this nor teach the domain anything.
                rejection, needs_browser = _static_result_rejection(result)
                if rejection is not None:
                    if needs_browser:
                        get_spa_domain_cache().record_spa_domain(url)
                    logger.debug(f"Strategy {strat} rejected: {rejection}")
                    errors.append(f"{strat}: {rejection}")
                    continue
            else:
                # Validate content quality before accepting
                is_invalid, reason = _is_invalid_content(
                    result.content, allow_short_content=_native_accepted(result)
                )
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
        finally:
            logger.debug(
                "[Fetch] Strategy {} finished in {:.3f}s",
                strat,
                time.perf_counter() - attempt_started,
            )

    # All strategies failed
    detail = "\n".join(f"  - {e}" for e in errors) or (
        "  - no strategy was applicable (all were skipped)"
    )
    raise FetchError(f"All fetch strategies failed for {url}:\n{detail}")
