"""Tests for privacy-preserving URL display."""

import pytest

from markitai.utils.url_redaction import (
    mask_url_credentials,
    redact_url,
    redact_urls_in_text,
    url_credential_values,
)


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


def test_redact_urls_in_text_masks_download_ticket_in_request_target() -> None:
    line = '127.0.0.1:5000 - "GET /api/jobs/abc/archive?ticket=0f1e2d3c HTTP/1.1" 200'

    assert redact_urls_in_text(line) == (
        '127.0.0.1:5000 - "GET /api/jobs/abc/archive?ticket=[REDACTED] HTTP/1.1" 200'
    )


@pytest.mark.parametrize(
    ("base", "shown"),
    [
        (
            "https://user:pw@host/v1?token=x&v=2",
            "https://…@host/v1?token=…&v=2",
        ),
        ("https://tok@host:8443/v1", "https://…@host:8443/v1"),
        (
            "https://host/v1?api_key=a&key=b&sig=c&Token=d&api-version=2024-10-01",
            "https://host/v1?api_key=…&key=…&sig=…&Token=…&api-version=2024-10-01",
        ),
        # Azure API Management: a name markitai-rust also reads as secret
        (
            "https://apim.example/openai?subscription-key=k",
            "https://apim.example/openai?subscription-key=…",
        ),
        ("https://host/v1#access_token=abc&x=1", "https://host/v1#access_token=…&x=1"),
        ("https://proxy.example/v1", "https://proxy.example/v1"),
        ("http://[::1]:11434/v1", "http://[::1]:11434/v1"),
        ("https://host/path@v1", "https://host/path@v1"),
        ("env:OPENAI_BASE_URL", "env:OPENAI_BASE_URL"),
    ],
)
def test_mask_url_credentials_keeps_the_address_readable(base: str, shown: str) -> None:
    assert mask_url_credentials(base) == shown


def test_url_credential_values_lists_what_the_mask_hides() -> None:
    values = url_credential_values(
        "https://alice:p%40ss@host/v1?api_key=k-123&api-version=2024#sig=s1"
    )
    assert set(values) == {"alice", "p%40ss", "p@ss", "k-123", "s1"}
