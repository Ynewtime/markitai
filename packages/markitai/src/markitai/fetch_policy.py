"""Fetch policy engine for determining URL fetch strategies.

Strategy priority rationale (v0.15.0, local-first):
  static → playwright → defuddle → jina → cloudflare

- static: Fast local fetch + native webextract pipeline (full defuddle port:
  scoring, math, footnotes, code protection) — matches remote defuddle
  quality on the ground-truth corpus, no data leaves the machine.
- playwright: Full JS rendering for SPA/JS-heavy pages, local. Includes
  oEmbed enricher fallback for X/Twitter URLs when DOM parsing fails.
- defuddle: Remote extraction API (free, no auth). Consent-gated
  (fetch.remote_consent) since it receives the user's URLs.
- jina: Remote Reader API, free tier (20 RPM), consent-gated; anonymous
  access is blocked for some domains (e.g. github.com → 451).
- cloudflare: Requires CF account credentials, rate-limited, consent-gated.
"""

from __future__ import annotations

import asyncio
import ipaddress
import math
import re
import socket
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, unquote, urlsplit

from markitai.constants import ALL_FETCH_STRATEGIES, LOCAL_STRATEGIES

ALL_STRATEGIES = list(ALL_FETCH_STRATEGIES)
LOCAL_ONLY_STRATEGIES = list(LOCAL_STRATEGIES)


def _extract_host(domain: str) -> str:
    """Extract host from a netloc-like domain string."""
    if domain.startswith("[") and "]" in domain:
        return domain[1 : domain.index("]")]
    # Bare IPv6 (multiple colons, e.g. "fd12::1") — return as-is
    if domain.count(":") > 1:
        return domain
    return domain.split(":", 1)[0]


def _legacy_ipv4_address(host: str) -> ipaddress.IPv4Address | None:
    """Parse deterministic inet_aton-style IPv4 aliases such as ``127.1``."""
    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return None

    values: list[int] = []
    try:
        for part in parts:
            lowered = part.lower()
            if lowered.startswith("0x"):
                value = int(lowered[2:], 16)
            elif len(lowered) > 1 and lowered.startswith("0"):
                value = int(lowered, 8)
            else:
                value = int(lowered, 10)
            values.append(value)
    except ValueError:
        return None

    widths = {
        1: (32,),
        2: (8, 24),
        3: (8, 8, 16),
        4: (8, 8, 8, 8),
    }[len(values)]
    if any(value < 0 or value >= (1 << width) for value, width in zip(values, widths)):
        return None

    packed = 0
    for value, width in zip(values, widths):
        packed = (packed << width) | value
    return ipaddress.IPv4Address(packed)


def _is_non_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not address.is_global or address.is_multicast


def is_private_or_local_domain(domain: str) -> bool:
    """Return True for localhost, private IPs, and common intranet-only hosts."""
    if "@" in domain:
        # Credentials in the netloc (user:pass@host, user@host): the userinfo
        # itself is the secret, so never send such URLs to remote services.
        return True
    host = _extract_host(domain).strip().lower().rstrip(".")
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    if host.endswith((".local", ".internal", ".lan", ".home", ".corp")):
        return True
    if "." not in host:
        return True

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        legacy_address = _legacy_ipv4_address(host)
        return bool(legacy_address and _is_non_public_ip(legacy_address))

    return _is_non_public_ip(address)


def _normalize_query_key(key: str) -> str:
    snake_key = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    snake_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake_key)
    return re.sub(r"[^a-z0-9]+", "_", snake_key.lower()).strip("_")


def _is_sensitive_query_key(key: str) -> bool:
    normalized = _normalize_query_key(key)
    parts = set(normalized.split("_"))
    if normalized in {
        "apikey",
        "auth",
        "code",
        "key",
        "pwd",
        "sid",
        "sig",
        "ticket",
    }:
        return True
    if "api" in parts and "key" in parts:
        return True
    if "key" in parts and ({"auth", "access"} & parts):
        return True
    return bool(
        parts
        & {
            "assertion",
            "authorization",
            "auth",
            "bearer",
            "credential",
            "credentials",
            "jwt",
            "otp",
            "passcode",
            "passwd",
            "password",
            "pwd",
            "saml",
            "session",
            "secret",
            "signature",
            "ticket",
            "token",
        }
    )


_SENSITIVE_PATH_CONTEXTS = {
    "activate",
    "activation",
    "invite",
    "invitation",
    "magic",
    "magic_link",
    "password_reset",
    "reset",
    "secret",
    "session",
    "token",
    "verify",
    "verification",
}
_UUID_TOKEN_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_JWT_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}$")
_URLSAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_~.-]+$")


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in Counter(value).values()
    )


def _looks_high_entropy_path_token(segment: str) -> bool:
    """Return whether a standalone path segment looks like an opaque secret."""
    if _JWT_TOKEN_RE.fullmatch(segment):
        return True
    if len(segment) < 24 or not _URLSAFE_TOKEN_RE.fullmatch(segment):
        return False
    # Hex hashes and UUID-like public identifiers are common URL paths. Treat
    # them as secrets only when a preceding route segment supplies context.
    compact = segment.replace("-", "")
    if compact and all(char in "0123456789abcdefABCDEF" for char in compact):
        return False
    has_mixed_random_alphabet = (
        any(char.islower() for char in segment)
        and any(char.isupper() for char in segment)
        and any(char.isdigit() for char in segment)
    )
    return (
        has_mixed_random_alphabet
        and len(set(segment)) >= 10
        and _shannon_entropy(segment) >= 3.5
    )


def _looks_contextual_path_token(segment: str) -> bool:
    """Return whether a segment following a sensitive route looks token-like."""
    if len(segment) < 8:
        return False
    if _UUID_TOKEN_RE.fullmatch(segment) or _looks_high_entropy_path_token(segment):
        return True
    if re.fullmatch(r"[0-9a-f]{16,}", segment, re.IGNORECASE):
        return True
    if re.fullmatch(r"[a-z]+(?:-[a-z]+)*", segment):
        return False
    return (
        len(segment) >= 12
        and _URLSAFE_TOKEN_RE.fullmatch(segment) is not None
        and (
            any(char.isupper() for char in segment)
            or any(char.isdigit() for char in segment)
        )
        and _shannon_entropy(segment) >= 3.2
    )


def sensitive_path_segment_indexes(path: str) -> frozenset[int]:
    """Locate path segments that should stay local and be hidden in displays."""
    raw_segments = path.split("/")
    decoded_segments = [unquote(segment) for segment in raw_segments]
    sensitive: set[int] = set()

    for index, segment in enumerate(decoded_segments):
        if not segment:
            continue
        if _looks_high_entropy_path_token(segment):
            sensitive.add(index)
            continue

        for separator in ("=", ":"):
            key, found, value = segment.partition(separator)
            if found and value and _is_sensitive_query_key(key):
                sensitive.add(index)
                break

        if index == 0:
            continue
        context = _normalize_query_key(decoded_segments[index - 1])
        if (
            context in _SENSITIVE_PATH_CONTEXTS
            or _is_sensitive_query_key(decoded_segments[index - 1])
        ) and _looks_contextual_path_token(segment):
            sensitive.add(index)

    return frozenset(sensitive)


def url_contains_credentials(url: str) -> bool:
    """Detect userinfo and credential-bearing path/query/fragment material."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None:
        return True
    if sensitive_path_segment_indexes(parsed.path):
        return True

    for component in (parsed.query, parsed.fragment if "=" in parsed.fragment else ""):
        if any(_is_sensitive_query_key(key) for key, _ in parse_qsl(component)):
            return True
    return False


@dataclass(frozen=True)
class RemoteURLAssessment:
    """Decision from the hard URL privacy boundary for remote services."""

    allowed: bool
    reason: str | None = None


HostResolver = Callable[[str], Awaitable[Sequence[str]]]


async def resolve_hostname_addresses(hostname: str) -> tuple[str, ...]:
    """Resolve a hostname without blocking the event loop."""
    loop = asyncio.get_running_loop()
    records = await asyncio.wait_for(
        loop.getaddrinfo(hostname, None, type=socket.SOCK_STREAM),
        timeout=5.0,
    )
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


def _address_is_global(value: str) -> bool:
    """Return False for malformed, scoped, private, or otherwise non-global IPs."""
    unscoped = value.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(unscoped)
    except ValueError:
        return False
    return address.is_global and not address.is_multicast


async def assess_url_for_remote(
    url: str,
    *,
    resolver: HostResolver | None = None,
) -> RemoteURLAssessment:
    """Apply the complete hard privacy policy before sharing a URL remotely."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
    except (TypeError, ValueError):
        return RemoteURLAssessment(False, "invalid_url")

    if parsed.scheme not in {"http", "https"} or not hostname:
        return RemoteURLAssessment(False, "invalid_url")
    if url_contains_credentials(url):
        return RemoteURLAssessment(False, "credential_material")
    if is_private_or_local_domain(parsed.netloc):
        return RemoteURLAssessment(False, "private_or_local_host")

    try:
        direct_address = ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        direct_address = None

    if direct_address is not None:
        addresses: Sequence[str] = (str(direct_address),)
    else:
        resolve = resolver or resolve_hostname_addresses
        try:
            addresses = await resolve(hostname)
        except (TimeoutError, OSError, UnicodeError):
            return RemoteURLAssessment(False, "hostname_resolution_failed")

    if not addresses:
        return RemoteURLAssessment(False, "hostname_resolution_failed")
    if any(not _address_is_global(address) for address in addresses):
        return RemoteURLAssessment(False, "non_global_address")
    return RemoteURLAssessment(True)


def parse_no_proxy(value: str | None) -> list[str]:
    """Parse a NO_PROXY-style comma-separated string into a list of patterns."""
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def match_local_only(domain: str, patterns: list[str]) -> bool:
    """Check if domain matches any local-only exemption pattern.

    Supports NO_PROXY syntax:
    - Domain exact: ``internal.corp.com``
    - Suffix: ``.internal.com`` (matches subdomains only)
    - Wildcard: ``*.internal.com`` (same as ``.internal.com``)
    - IP exact: ``10.0.1.5``
    - CIDR: ``10.0.0.0/8``, ``fd00::/8``
    - Star: ``*`` (matches everything)
    - Special: ``localhost``
    """
    if not patterns:
        return False

    host = _extract_host(domain).strip().lower().rstrip(".")
    if not host:
        return False

    for pattern in patterns:
        p = pattern.strip().lower()
        if not p:
            continue

        # Wildcard: match everything
        if p == "*":
            return True

        # Normalize *.foo.com → .foo.com
        if p.startswith("*."):
            p = p[1:]  # "*.foo.com" → ".foo.com"

        # Suffix match: .foo.com matches sub.foo.com but not foo.com
        if p.startswith("."):
            if host.endswith(p):
                return True
            continue

        # CIDR match
        if "/" in p:
            try:
                network = ipaddress.ip_network(p, strict=False)
                host_ip = ipaddress.ip_address(host)
                if host_ip in network:
                    return True
            except ValueError:
                pass
            continue

        # Exact match (domain or IP)
        if host == p:
            return True

    return False


@dataclass
class FetchDecision:
    """Decision from the fetch policy engine."""

    order: list[str]
    reason: str


class FetchPolicyEngine:
    """Engine to decide the order of fetch strategies based on URL and config."""

    def decide(
        self,
        domain: str,
        known_spa: bool,
        explicit_strategy: str | None,
        fallback_patterns: list[str],
        policy_enabled: bool,
        has_jina_key: bool = False,
        domain_prefer_strategy: str | None = None,
        global_strategy_priority: list[str] | None = None,
        domain_strategy_priority: list[str] | None = None,
        local_only_patterns: list[str] | None = None,
    ) -> FetchDecision:
        """Decide the fetch strategy order.

        Priority chain (highest to lowest):
        1. Explicit strategy (CLI flag) -- single strategy, no fallback
        2. Local-only exemption -- security: restrict to local strategies
        3. Private/local domain -- hardcoded local detection
        4. Domain strategy_priority -- per-domain full override
        5. Domain prefer_strategy -- per-domain single promotion
        6. Global strategy_priority -- global full override
        7. SPA/JS-required pattern -- browser-first order
        8. Default order
        """
        # 1. Explicit strategy always wins
        if explicit_strategy and explicit_strategy != "auto":
            return FetchDecision(
                order=[explicit_strategy], reason=f"explicit_{explicit_strategy}"
            )

        is_fallback_domain = any(
            domain == p or domain.endswith(f".{p}") for p in fallback_patterns
        )

        # 2. Local-only exemption (security: skip external APIs)
        if local_only_patterns and match_local_only(domain, local_only_patterns):
            order = (
                ["playwright", "static"]
                if known_spa or is_fallback_domain
                else LOCAL_ONLY_STRATEGIES.copy()
            )
            return FetchDecision(order=order, reason="local_only_pattern")

        # 3. Private/local domain (hardcoded detection)
        if is_private_or_local_domain(domain):
            order = (
                ["playwright", "static"]
                if known_spa or is_fallback_domain
                else LOCAL_ONLY_STRATEGIES.copy()
            )
            return FetchDecision(order=order, reason="private_or_local")

        # 4. Per-domain strategy_priority (full override)
        if domain_strategy_priority:
            return FetchDecision(
                order=list(domain_strategy_priority), reason="domain_priority"
            )

        # 5. Per-domain prefer_strategy (single promotion)
        if domain_prefer_strategy:
            remaining = [s for s in ALL_STRATEGIES if s != domain_prefer_strategy]
            return FetchDecision(
                order=[domain_prefer_strategy] + remaining,
                reason=f"domain_prefer_{domain_prefer_strategy}",
            )

        # 6. Global strategy_priority (full override)
        if global_strategy_priority:
            return FetchDecision(
                order=list(global_strategy_priority), reason="global_priority"
            )

        if not policy_enabled:
            return FetchDecision(
                order=ALL_STRATEGIES.copy(),
                reason="disabled",
            )

        # 7. SPA/JS-heavy: browser first
        # Known SPAs and fallback-pattern domains need JS rendering — go
        # straight to the local browser; consent-gated remote services are
        # the fallback, and static (which already failed to produce content
        # for such domains) goes last.
        if known_spa or is_fallback_domain:
            return FetchDecision(
                order=["playwright", "defuddle", "jina", "cloudflare", "static"],
                reason="spa_or_pattern",
            )

        # 8. Default
        return FetchDecision(
            order=ALL_STRATEGIES.copy(),
            reason="default",
        )
