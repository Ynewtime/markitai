"""Verification pages, empty pages and the page kept beside its screenshot."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import FetchConfig, ScreenshotConfig
from markitai.fetch import _capture_missing_screenshot
from markitai.fetch_page_check import is_interstitial, page_problem, visible_text
from markitai.fetch_types import FetchResult

# An invented click-the-characters challenge, shaped like the vendors' pages
CHALLENGE_HTML = """<!DOCTYPE html><html><head><title>验证</title>
<link rel="stylesheet" href="https://static.example-captcha.test/geetest.css"></head>
<body><div class="geetest_panel"><div class="geetest_title">请在下图依次点击：</div>
<img src="https://static.example-captcha.test/bg.jpg"><button>确认</button></div>
</body></html>"""

ARTICLE_HTML = (
    "<html><body><article><h1>Why sites use CAPTCHAs</h1>"
    + "<p>A verification page asks you to verify you are human. " * 60
    + "</p></article></body></html>"
)


class TestPageProblem:
    def test_challenge_page(self):
        assert page_problem(CHALLENGE_HTML) == "challenge"

    @pytest.mark.parametrize(
        "text",
        ["Just a moment...", "Checking your browser before accessing", "安全验证"],
    )
    def test_challenge_phrases_on_a_short_page(self, text):
        assert page_problem(f"<html><body><p>{text}</p></body></html>") == "challenge"

    def test_vendor_markup_on_a_short_page(self):
        html = '<html><body><form id="challenge-form" action="/?__cf_chl_f_tk=x"></form><p>One moment</p></body></html>'
        assert page_problem(html) == "challenge"

    def test_an_article_about_captchas_is_the_page(self):
        assert page_problem(ARTICLE_HTML) is None

    def test_fastly_client_challenge(self):
        html = (
            "<html><head><title>Client Challenge</title></head><body>"
            "<div>A required part of this site couldn't load.</div>"
            "<script>loadScript('/_fs-ch-1T1wmsGaOgGaSxcX/errors.js')</script>"
            "</body></html>"
        )
        assert page_problem(html) == "challenge"
        assert is_interstitial(html)

    def test_a_form_widget_is_not_an_interstitial(self):
        """A short contact page with a reCAPTCHA is the page itself when
        fetched; only the rendered-page check counts it as a challenge."""
        html = (
            "<html><body><h1>Contact</h1><form><textarea></textarea>"
            '<div class="g-recaptcha"></div></form>'
            '<script src="https://www.google.com/recaptcha/api.js"></script>'
            "</body></html>"
        )
        assert page_problem(html) == "challenge"
        assert not is_interstitial(html)
        assert not is_interstitial(ARTICLE_HTML)

    @pytest.mark.parametrize(
        "text",
        [
            "手机号 获取验证码 登录 还没有账号？立即注册",
            "如何关闭短信验证码？进入设置，在账号与安全里关闭登录时的短信验证码即可。",
            "账号登录 手机号 请输入验证码 获取验证码 忘记密码？",
            "安全验证码收不到怎么办？ 关注者 12 被浏览 3,456",
        ],
        ids=["login", "article", "form", "question"],
    )
    def test_a_page_that_mentions_a_verification_code_is_the_page(self, text):
        """验证码 alone is a login form or an article, not a bot check."""
        html = f"<html><body><h1>帮助</h1><p>{text}</p></body></html>"
        assert not is_interstitial(html)
        assert page_problem(html) is None

    @pytest.mark.parametrize(
        "text",
        [
            "请输入验证码以继续访问",
            "为了确认是您本人操作，请输入图中验证码后继续",
            "我们的系统检测到您网络中存在异常访问请求。此验证码用于确认这些请求"
            "是您的正常行为而不是自动程序发出的，需要您协助验证。",
        ],
        ids=["go-on", "picture", "sogou"],
    )
    def test_a_page_that_asks_for_the_code_to_go_on_is_a_challenge(self, text):
        html = f"<html><body><p>{text}</p><form><input name=c></form></body></html>"
        assert is_interstitial(html)
        assert page_problem(html) == "challenge"

    def test_empty_document(self):
        assert (
            page_problem(
                "<html><head><title>X</title></head><body><div id=root></div></body></html>"
            )
            == "blank"
        )
        assert page_problem("") == "blank"

    def test_a_picture_page_is_not_blank(self):
        assert page_problem("<html><body><img src='a.jpg'></body></html>") is None

    def test_a_shadow_root_page_is_not_blank(self):
        assert (
            page_problem(
                "<html><body><app-shell></app-shell></body></html>", shadow_roots=1
            )
            is None
        )

    def test_browser_measured_text_wins(self):
        assert (
            page_problem(
                "<html><body></body></html>",
                text="Hello there, readers of the post",
                media=0,
            )
            is None
        )

    def test_visible_text_skips_scripts_and_styles(self):
        text, media = visible_text(
            "<html><head><style>p{}</style></head><body><script>var a</script>"
            "<p>Shown</p><video></video></body></html>"
        )
        assert (text, media) == ("Shown", 1)


def _pw_result(tmp_path: Path, *, problem: str | None = None, html: str | None = None):
    result = MagicMock()
    result.title = "Page"
    result.final_url = "https://example.com/post"
    result.metadata = {"page_problem": problem} if problem else {}
    shot = None if problem else tmp_path / "shot.jpg"
    result.screenshot_path = shot
    result.screenshot_tiles = [shot] if shot else []
    result.html = None if problem else html
    return result


def _jina_result() -> FetchResult:
    return FetchResult(
        content="# Post\n\nText the reader service returned.",
        strategy_used="jina",
        url="https://example.com/post",
        metadata={"api": "jina-reader"},
    )


async def _capture(result: FetchResult, pw: AsyncMock, tmp_path: Path):
    with (
        patch("markitai.fetch._screenshot_unavailable_reason", return_value=None),
        patch("markitai.fetch_playwright.fetch_with_playwright", pw),
    ):
        return await _capture_missing_screenshot(
            "https://example.com/post",
            result,
            FetchConfig(),
            screenshot_dir=tmp_path,
            screenshot_config=ScreenshotConfig(enabled=True),
            renderer=object(),
        )


class TestSeparateCapture:
    async def test_the_browser_page_is_kept_with_its_screenshot(self, tmp_path):
        drawn = "<html><body><h1>Post</h1><p>As the browser drew it</p></body></html>"
        pw = AsyncMock(return_value=_pw_result(tmp_path, html=drawn))
        merged, error = await _capture(_jina_result(), pw, tmp_path)
        assert error is None
        assert merged.screenshot_path == tmp_path / "shot.jpg"
        assert merged.html == drawn
        assert merged.content.startswith("# Post")

    async def test_a_content_strategy_page_is_not_replaced(self, tmp_path):
        result = _jina_result()
        result.html = "<html><body>fetched</body></html>"
        pw = AsyncMock(return_value=_pw_result(tmp_path, html="<html>drawn</html>"))
        merged, _ = await _capture(result, pw, tmp_path)
        assert merged.html == "<html><body>fetched</body></html>"

    async def test_a_verification_page_is_loaded_once_more(self, tmp_path):
        drawn = "<html><body><p>The real post</p></body></html>"
        pw = AsyncMock(
            side_effect=[
                _pw_result(tmp_path, problem="challenge"),
                _pw_result(tmp_path, html=drawn),
            ]
        )
        merged, error = await _capture(_jina_result(), pw, tmp_path)
        assert pw.await_count == 2
        assert error is None
        assert merged.html == drawn

    async def test_a_page_still_verifying_keeps_nothing(self, tmp_path):
        pw = AsyncMock(return_value=_pw_result(tmp_path, problem="challenge"))
        merged, error = await _capture(_jina_result(), pw, tmp_path)
        assert pw.await_count == 2
        assert merged.screenshot_path is None
        assert merged.html is None
        assert merged.metadata["page_problem"] == "challenge"
        assert error is not None and "verification page" in error

    async def test_the_chains_browser_attempt_counts_as_the_first_load(self, tmp_path):
        result = _jina_result()
        result.metadata["browser_challenge"] = True
        pw = AsyncMock(return_value=_pw_result(tmp_path, problem="challenge"))
        merged, _ = await _capture(result, pw, tmp_path)
        assert pw.await_count == 1
        assert merged.metadata["page_problem"] == "challenge"
