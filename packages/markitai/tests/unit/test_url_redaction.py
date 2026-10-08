"""Tests for privacy-preserving URL display."""

from markitai.utils.url_redaction import redact_url, redact_urls_in_text


def test_redact_url_masks_sensitive_path_token() -> None:
    url = (
        "https://alice:password@example.com/reset/"
        "550e8400-e29b-41d4-a716-446655440000/complete?source=email"
    )

    assert redact_url(url) == "https://example.com/reset/[REDACTED]/complete"


def test_redact_url_masks_standalone_high_entropy_path_token() -> None:
    url = "https://example.com/download/AbCDef0123456789_-AbCDef0123456789/file.pdf"

    assert redact_url(url) == "https://example.com/download/[REDACTED]/file.pdf"


def test_redact_url_preserves_normal_public_path() -> None:
    url = "https://example.com/articles/understanding-zero-trust-security-models"

    assert redact_url(url) == url


def test_redact_urls_in_text_masks_token_in_bare_request_target() -> None:
    line = '127.0.0.1:5000 - "GET /api/jobs/abc/events?token=mk_SECRET123 HTTP/1.1" 200'

    assert redact_urls_in_text(line) == (
        '127.0.0.1:5000 - "GET /api/jobs/abc/events?token=[REDACTED] HTTP/1.1" 200'
    )
