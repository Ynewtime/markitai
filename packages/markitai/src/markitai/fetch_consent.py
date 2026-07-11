"""Remote-fetch consent (privacy) state and decision logic.

Whether the auto strategy chain may send URLs to remote extraction services
(defuddle.md, Jina, Cloudflare BR) is a process-wide decision, resolved once
and cached. This module owns both the state shape (:class:`ConsentState`) and
every function that reads or writes it.

The canonical ConsentState instance lives on the process-wide
``markitai.fetch_session.FetchSession``. To keep this module a leaf (it must
not import ``markitai.fetch_session``, which would create an import cycle),
the functions here reach the active state through a module-level provider
that ``markitai.fetch_session`` registers at import time. Before registration
a module-local fallback state is used; in practice every entry point imports
``markitai.fetch`` (which imports ``markitai.fetch_session``) before any
consent function runs.

The public import path for these functions remains ``markitai.fetch``, which
re-exports them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from markitai.ports import get_interaction

if TYPE_CHECKING:
    from markitai.config import FetchConfig


@dataclass
class ConsentState:
    """Process-wide remote-fetch consent state (one per FetchSession)."""

    # Whether remote extraction services may receive URLs (None = undecided).
    decision: bool | None = None
    # Whether the interactive consent prompt is allowed (disabled by --quiet).
    prompt_allowed: bool = True
    # Whether the first-use privacy disclosure has been emitted.
    disclosure_emitted: bool = False
    # Fallback decision when an explicitly-selected remote strategy (jina,
    # defuddle, cloudflare) is refused server-side (4xx / auth-required).
    # Cached for the whole process so batch runs prompt at most once,
    # never once per URL.
    explicit_fallback_decision: bool | None = None


# Provider indirection: fetch_session registers a provider that returns the
# default session's ConsentState. The local fallback keeps this module usable
# (and deterministic) even if fetch_session was never imported.
_fallback_state = ConsentState()
_state_provider: Callable[[], ConsentState] | None = None


def set_consent_state_provider(provider: Callable[[], ConsentState]) -> None:
    """Register the source of the active ConsentState.

    Called by ``markitai.fetch_session`` at import time so consent state
    lives on the process-wide FetchSession.
    """
    global _state_provider
    _state_provider = provider


def _get_state() -> ConsentState:
    """Return the active ConsentState (session-owned once registered)."""
    if _state_provider is None:
        return _fallback_state
    return _state_provider()


def reset_remote_consent() -> None:
    """Reset the cached remote-fetch consent decision (mainly for tests)."""
    state = _get_state()
    state.decision = None
    state.prompt_allowed = True
    state.disclosure_emitted = False


def set_remote_consent(allowed: bool) -> None:
    """Seed the process-wide remote-fetch consent decision.

    Used when the user explicitly selects a remote strategy
    (e.g. ``-s jina``), which counts as consent for the run unless the
    process-wide ``MARKITAI_NO_REMOTE_FETCH`` hard opt-out is active.
    """
    _get_state().decision = False if _env_no_remote_fetch() else allowed


def set_remote_consent_prompt_allowed(allowed: bool) -> None:
    """Allow or disallow the interactive consent prompt (disabled by --quiet)."""
    _get_state().prompt_allowed = allowed


def reset_explicit_fallback_decision() -> None:
    """Reset the cached explicit-strategy fallback decision (mainly for tests)."""
    _get_state().explicit_fallback_decision = None


def _should_fallback_after_refusal(strategy_name: str, reason: str) -> bool:
    """Decide whether to fall back to the auto chain after a service refusal.

    Interactive TTY: prompt once per run (the answer is cached). Otherwise:
    fall back automatically (the caller logs a clear warning to stderr).
    """
    state = _get_state()
    if state.explicit_fallback_decision is not None:
        return state.explicit_fallback_decision

    interaction = get_interaction()
    if state.prompt_allowed and interaction.can_prompt():
        decision = interaction.confirm(
            f"{strategy_name} cannot fetch this URL ({reason}). "
            "Fall back to the auto strategy chain?",
            default=True,
        )
        state.explicit_fallback_decision = decision
        return decision

    # Non-interactive: fall back automatically (warned at the call site).
    state.explicit_fallback_decision = True
    return True


def _env_no_remote_fetch() -> bool:
    """Return True when MARKITAI_NO_REMOTE_FETCH is set to a truthy value."""
    import os

    value = os.environ.get("MARKITAI_NO_REMOTE_FETCH", "").strip().lower()
    return value not in ("", "0", "false", "no")


def peek_cached_remote_consent() -> bool | None:
    """Return only the hard opt-out or an explicit/cached process decision."""
    if _env_no_remote_fetch():
        return False
    return _get_state().decision


def peek_remote_consent(config: FetchConfig) -> bool | None:
    """Non-prompting view of the remote-consent state.

    Returns True/False when the answer is already determined (cached
    decision, env override, config "always"/"never", or "ask" in a
    non-interactive session, which auto-denies). Returns None when an
    interactive prompt would be needed — callers defer that prompt until
    a remote strategy is actually about to run (lazy consent), so users
    whose URLs are satisfied by local strategies are never asked.
    """
    if _env_no_remote_fetch():
        return False
    state = _get_state()
    if state.decision is not None:
        return state.decision
    consent = getattr(config, "remote_consent", "always")
    if consent == "always":
        return True
    if consent == "never":
        return False
    if state.prompt_allowed and get_interaction().can_prompt():
        return None  # would need to prompt
    return False


# Human-readable names for consent-gated remote strategies. Cloudflare runs
# against the user's own account credentials, hence the qualifier.
_REMOTE_SERVICE_LABELS = {
    "defuddle": "defuddle.md",
    "jina": "Jina",
    "cloudflare": "Cloudflare (your account)",
    "fxtwitter": "FxTwitter",
    "twitter-oembed": "Twitter oEmbed",
}


def _remote_service_names(services: list[str] | None) -> str:
    """Render every service covered by the process-wide decision.

    Consent and first-use disclosure are cached for the whole process, so the
    wording must cover services that a later URL may introduce, not only the
    chain that happened to trigger the first decision.
    """
    keys = list(_REMOTE_SERVICE_LABELS)
    for service in services or []:
        if service not in keys:
            keys.append(service)
    return ", ".join(_REMOTE_SERVICE_LABELS.get(s, s) for s in keys)


def disclose_remote_use(services: list[str] | None = None) -> None:
    """Emit the process-wide first-use privacy disclosure directly to stderr."""
    state = _get_state()
    if state.disclosure_emitted:
        return

    disclosure = (
        "[Fetch] This run's remote extraction services may receive URLs "
        "when needed, tried one at a time "
        f"({_remote_service_names(services)}). Disable all remote extraction "
        "with MARKITAI_NO_REMOTE_FETCH=1; fetch.remote_consent=never disables "
        "automatic and config-selected remote use. Private/local/"
        "credential-bearing URLs stay local."
    )
    # This is a privacy boundary, not diagnostic logging. Deliver it via the
    # interaction port (stderr) so normal INFO filtering and --quiet cannot
    # hide it.
    get_interaction().notify(disclosure)
    # Keep the disclosure in diagnostic log files too. The console filter
    # suppresses this copy to avoid duplicating the direct stderr message.
    logger.info(disclosure)
    state.disclosure_emitted = True


def resolve_remote_consent(
    config: FetchConfig, services: list[str] | None = None
) -> bool:
    """Return True if remote extraction services may receive URLs.

    Precedence: MARKITAI_NO_REMOTE_FETCH env var (truthy behaves as a hard
    "never", including explicit remote strategies) > cached/explicit decision
    > ``fetch.remote_consent`` config ("always" / "never" / "ask"). With
    "always" (the default) the first use is disclosed directly to stderr and
    retained in the INFO log; with "ask", prompts once on an interactive TTY
    (unless prompting is disabled, e.g. --quiet), otherwise remote services
    are skipped. The decision is cached for the whole process.
    Private/local/credentialed URLs never reach this gate — the policy layer
    strips remote strategies for them.

    Args:
        config: Fetch configuration.
        services: Remote strategy names that triggered the decision. The
            process-wide disclosure/prompt also names every other service the
            cached decision can authorize later in the run.
    """
    state = _get_state()

    if _env_no_remote_fetch():
        if state.decision is not False:
            logger.info(
                "[Fetch] Remote extraction services disabled by "
                "MARKITAI_NO_REMOTE_FETCH; using local strategies only"
            )
        state.decision = False
        return False
    if state.decision is not None:
        return state.decision

    consent = getattr(config, "remote_consent", "always")
    if consent == "always":
        disclose_remote_use(services)
        state.decision = True
        return True
    if consent == "never":
        logger.info(
            "[Fetch] Remote extraction services disabled "
            "(fetch.remote_consent=never); using local strategies only"
        )
        state.decision = False
        return False

    # "ask": prompt once when interactive, otherwise skip (privacy-safe)
    interaction = get_interaction()
    if state.prompt_allowed and interaction.can_prompt():
        allowed = interaction.confirm(
            "Allow sending public URLs to these remote services when needed?",
            default=False,
            preamble=(
                "This run can try remote services for public URLs, one at a "
                f"time (first success wins): {_remote_service_names(services)}."
            ),
        )
        if not allowed:
            logger.info(
                "[Fetch] Remote extraction services declined; "
                "using local strategies only"
            )
        state.decision = allowed
        return allowed

    logger.info(
        "[Fetch] Skipping remote extraction services (defuddle.md, Jina, Cloudflare): "
        "no consent (fetch.remote_consent=ask, non-interactive). "
        "Use -s <strategy> or set fetch.remote_consent=always to enable them."
    )
    state.decision = False
    return False
