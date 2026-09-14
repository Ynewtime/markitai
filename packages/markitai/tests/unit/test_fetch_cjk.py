"""Complete CJK notes should finish on the static path without SPA learning."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import FetchConfig
from markitai.fetch import detect_js_required, fetch_url
from markitai.fetch_cache import SPADomainCache
from markitai.fetch_types import FetchResult, FetchStrategy

NOTE = (
    "同意。虽然是 Notion 的广告，不过想强调的 Point 是，Harness 一直在变，"
    "且不同大模型在不同 Harness 效果就是很难一致，模型厂商自家的 Harness "
    "肯定做了针对性的调优。因此从普通用户视角，自己在使用各家 Harness 的过程中，"
    "如何维护好个人的 Context，至关重要。"
)


@pytest.mark.parametrize(
    "content",
    [
        NOTE,
        "今天读完这篇文章以后，我重新整理了日常工作中的资料和笔记。不同工具各有所长，"
        "选择时需要结合实际任务，保留必要的背景信息，并记录每次尝试的具体结果。"
        "这样在切换环境之后，也能够快速恢复之前的思路，减少重复操作，让下一次协作更加顺畅。",
        "今日は図書館で新しい本を読みながら、これまでの仕事の進め方について考えました。"
        "道具にはそれぞれ得意なことがあるので、目的に合わせて選ぶことが大切です。"
        "途中で気づいたことを記録しておけば、別の環境でも続きから作業を始められます。",
        "오늘은 도서관에서 새로운 책을 읽으며 지금까지 일하는 방식을 돌아보았습니다. "
        "도구마다 잘하는 일이 다르기 때문에 목적에 맞게 선택하는 것이 중요합니다. "
        "작업 중에 발견한 내용을 기록해 두면 다른 환경에서도 이전에 하던 일을 이어갈 수 있습니다.",
    ],
)
def test_cjk_prose_is_not_a_js_placeholder(content):
    assert not detect_js_required(content)


@pytest.mark.parametrize(
    "content",
    [
        "正在加载，请稍候。" * 15,
        "読み込み中です。" * 20,
        "로딩 중입니다. " * 15,
        "loading please wait " * 10,
        NOTE + "\nPlease enable JavaScript to continue.",
        NOTE + "\nChecking your browser before accessing this page.",
    ],
)
def test_loading_and_challenge_pages_still_require_js(content):
    assert detect_js_required(content)


@pytest.mark.asyncio
async def test_auto_accepts_cjk_static_content_without_browser_or_learning():
    url = "https://example.com/notes/short-note"
    result = FetchResult(content=NOTE, strategy_used="static", url=url)
    spa_cache = MagicMock()
    spa_cache.is_known_spa.return_value = False
    with (
        patch("markitai.fetch.get_spa_domain_cache", return_value=spa_cache),
        patch(
            "markitai.fetch_strategies.static.fetch_with_static",
            AsyncMock(return_value=result),
        ) as static,
        patch(
            "markitai.fetch_strategies.playwright.PlaywrightRunner.fetch",
            new_callable=AsyncMock,
        ) as browser,
    ):
        actual = await fetch_url(
            url, FetchStrategy.AUTO, config=FetchConfig(), skip_read_cache=True
        )
    assert actual.content == NOTE
    assert actual.strategy_used == "static"
    static.assert_awaited_once()
    browser.assert_not_awaited()
    spa_cache.record_spa_domain.assert_not_called()


@pytest.mark.parametrize("old_version", [1, 2])
def test_old_spa_classifications_are_relearned_with_current_detector(
    tmp_path, old_version
):
    path = tmp_path / "spa.json"
    path.write_text(
        json.dumps(
            {
                "version": old_version,
                "domains": {
                    "example.com": {
                        "learned_at": datetime.now().astimezone().isoformat(),
                        "hits": 2,
                    }
                },
            }
        )
    )
    cache = SPADomainCache(path)
    assert not cache.is_known_spa("https://example.com/notes/short-note")
    cache.record_spa_domain("https://actual-spa.example/app")
    reloaded = SPADomainCache(path)
    assert reloaded.is_known_spa("https://actual-spa.example/app")
    assert not reloaded.is_known_spa("https://example.com/notes/short-note")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "A short but complete note.",
        "今天完成了迁移，明天继续整理文档。",
        "Sample note text for testing the extractor. It has multiple paragraphs to verify the content is captured correctly.",
    ],
)
async def test_auto_trusts_accepted_native_short_articles(body):
    from markitai.fetch_strategies._shared import _build_native_fetch_result
    from markitai.fetch_types import FetchError

    url = "https://example.com/notes/complete"
    result = await _build_native_fetch_result(
        html=f"<html><body><article><p>{body}</p></article></body></html>",
        url=url,
        final_url=url,
        strategy_used="static",
    )
    assert result is not None
    spa_cache = MagicMock()
    spa_cache.is_known_spa.return_value = False
    with (
        patch("markitai.fetch.get_spa_domain_cache", return_value=spa_cache),
        patch(
            "markitai.fetch_strategies.static.fetch_with_static",
            AsyncMock(return_value=result),
        ),
        patch(
            "markitai.fetch_strategies.playwright.PlaywrightRunner.fetch",
            AsyncMock(side_effect=FetchError("Unexpected browser fallback")),
        ) as browser,
    ):
        actual = await fetch_url(
            url,
            FetchStrategy.AUTO,
            FetchConfig(remote_consent="never"),
            skip_read_cache=True,
        )
    assert body in actual.content
    assert actual.strategy_used == "static"
    browser.assert_not_awaited()
    spa_cache.record_spa_domain.assert_not_called()


@pytest.mark.parametrize(
    "content",
    [
        "Loading...",
        "# Loading\n\nPlease wait.",
        "正在加载，请稍候。",
        "読み込み中です。",
        "로딩 중입니다.",
        "Please enable JavaScript to continue.",
    ],
)
def test_native_short_content_exemption_still_detects_loading_pages(content):
    assert detect_js_required(content, allow_short_content=True)


@pytest.mark.parametrize(
    "content",
    [
        "You must be logged in",
        "智能验证检测中",
        "cf-browser-verification",
    ],
)
def test_native_short_content_exemption_still_rejects_login_and_captcha(content):
    from markitai.fetch import _is_invalid_content

    invalid, reason = _is_invalid_content(content, allow_short_content=True)
    assert invalid
    assert reason != "too_short"
