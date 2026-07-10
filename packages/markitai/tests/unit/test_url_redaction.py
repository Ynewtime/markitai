"""Tests for privacy-preserving URL display."""

from markitai.utils.url_redaction import redact_url


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
