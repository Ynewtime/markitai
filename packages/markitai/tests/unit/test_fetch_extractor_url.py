"""Tests that extractor matching uses original URL, not redirected URL."""

from markitai.webextract.pipeline import extract_web_content


def test_extractor_matches_original_url_not_redirect() -> None:
    """Even if final_url is a login page, the original URL should be used."""
    html = "<html><head><title>T</title></head><body><p>tweet text</p></body></html>"
    original_url = "https://x.com/user/status/123"
    login_url = "https://x.com/i/flow/login"

    result = extract_web_content(html, original_url)
    assert result.info.extractor_name == "x_tweet"

    result2 = extract_web_content(html, login_url)
    assert result2.info.extractor_name == "generic"
