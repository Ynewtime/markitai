"""Thread policy tests: freeze default inclusion rules for threaded page types.

These tests are intentionally written against ``get_thread_policy()`` which
does NOT yet exist.  They will fail until Task 4 introduces the policy model.
The intent is to lock down the desired default behaviour so that future
implementation cannot silently deviate from it.
"""

from __future__ import annotations

X_URL = "https://x.com/ixiaowenz/status/2030105637204676808"


# ---------------------------------------------------------------------------
# Default thread policy for X status pages
# ---------------------------------------------------------------------------


def test_thread_policy_defaults_do_not_include_unrelated_replies() -> None:
    """Default policy for X status pages must exclude third-party replies.

    Verifies:
    - include_main_item is True (the focal post must always be included)
    - include_author_thread is True (the author's own reply chain is relevant)
    - include_third_party_replies is False (noise for most use-cases)
    """
    # This import will raise ImportError until the policy module is created.
    from markitai.webextract.thread_policy import (
        get_thread_policy,  # type: ignore[import-not-found]
    )

    policy = get_thread_policy(X_URL)

    assert policy is not None
    assert policy.include_main_item is True
    assert policy.include_author_thread is True
    assert policy.include_third_party_replies is False


def test_thread_policy_for_generic_url_returns_none() -> None:
    """Non-threaded URLs must not have a thread policy.

    A generic article URL should return None from ``get_thread_policy`` because
    thread policies are only meaningful for social or discussion page types.
    """
    from markitai.webextract.thread_policy import (
        get_thread_policy,  # type: ignore[import-not-found]
    )

    policy = get_thread_policy("https://example.com/blog/async-python")

    assert policy is None
