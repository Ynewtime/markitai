"""How a kept page or its screenshot reads (workbench contract v1.14).

The same cases as Rust's ``page_capture`` tests, plus each of the contract's
minimum markers and phrases. Invented pages only.
"""

from __future__ import annotations

import json

import pytest

from markitai.page_capture import (
    MAX_BLOCKS_BYTES,
    MAX_TEXT_LENGTH,
    PAGE_BLOCKS_PATH,
    classify,
    page_blocks_script,
    page_quality,
    shown_instead,
    sized_blocks,
)

ARTICLE = (
    "<html><head><title>  An   invented\n article </title><style>p{}</style></head>"
    "<body><article><h1>Heading</h1><p>"
    + "Plain words of an invented article. " * 4
    + "</p></article><script>var hidden = 'not text';</script></body></html>"
)


def _state(url: str, html: str) -> tuple[str, str | None]:
    quality = page_quality(url, html)
    return quality["state"], quality["reason"]


def test_an_article_is_ok_with_its_collapsed_text_and_title() -> None:
    expected = "Heading" + ("Plain words of an invented article. " * 4).rstrip()
    assert page_quality("https://example.com/post", ARTICLE) == {
        "state": "ok",
        "text_length": len(expected),
        "title": "An invented article",
        "reason": None,
    }


def test_a_challenge_widget_a_site_captcha_and_a_login_wall_are_challenges() -> None:
    cloudflare = (
        "<html><head><title>Just a moment...</title></head><body>"
        '<div id="cf-chl-widget"></div>'
        "<noscript>Enable JavaScript and cookies to continue</noscript>"
        '<script src="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1">'
        "</script></body></html>"
    )
    quality = page_quality("https://example.org/a", cloudflare)
    assert (quality["state"], quality["reason"]) == ("challenge", "challenge")
    assert quality["title"] == "Just a moment..."
    # The frame shows the noscript text
    assert quality["text_length"] == len("Enable JavaScript and cookies to continue")
    captcha = (
        "<html><head><title>Video</title></head><body>"
        '<div id="risk-captcha-app"></div></body></html>'
    )
    assert _state("https://www.bilibili.com/video/BV0000000000", captcha) == (
        "challenge",
        "verification",
    )
    # The same markup elsewhere is only a blank page
    assert _state("https://example.com/video", captcha) == ("empty", "blank")
    assert _state(
        "https://www.bilibili.com/video/BV0000000000",
        "<html><head><title>验证码_哔哩哔哩</title></head><body></body></html>",
    ) == ("challenge", "verification")
    login = (
        "<html><head><title>Example</title></head><body>"
        "<p>Log in to continue reading this answer.</p></body></html>"
    )
    assert _state("https://www.zhihu.com/question/1", login) == ("challenge", "login")
    # A site that expects no login, or a long page, is not a login wall
    assert _state("https://www.bilibili.com/read/1", login) == ("empty", "blank")
    long_login = login.replace("</p>", " " + "More words here. " * 30 + "</p>")
    assert _state("https://www.zhihu.com/question/1", long_login) == ("ok", None)


@pytest.mark.parametrize(
    "url",
    [
        "https://mp.weixin.qq.com/mp/wappoc_appmsgcaptcha?x=1",
        "https://sec.douban.com/c?r=1",
        "https://passport.weibo.com/visitor/visitor?a=1",
        "https://www.zhihu.com/account/unhuman?type=1",
    ],
)
def test_known_sites_verification_pages(url: str) -> None:
    assert shown_instead(url, "<html><body><p>x</p></body></html>") == "verification"


def test_more_known_site_markers() -> None:
    assert (
        shown_instead(
            "https://www.reddit.com/r/x",
            "<html><body><h1>Prove your humanity</h1></body></html>",
        )
        == "verification"
    )
    assert (
        shown_instead(
            "https://www.toutiao.com/article/1",
            "<html><body><script>var _$jsvmprt=1</script></body></html>",
        )
        == "verification"
    )
    assert (
        shown_instead(
            "https://zhuanlan.zhihu.com/p/1",
            "<html><head><title>安全验证 - 知乎</title></head><body></body></html>",
        )
        == "verification"
    )
    # A question that starts with the same words is a question
    assert (
        shown_instead(
            "https://www.zhihu.com/question/1",
            "<html><head><title>安全验证怎么做 - 知乎</title></head><body></body></html>",
        )
        is None
    )
    assert (
        shown_instead(
            "https://weibo.com/1",
            "<html><head><title>Sina Visitor System</title></head></html>",
        )
        == "verification"
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "Checking your browser before accessing example.org",
        "Verify you are human by completing the action below.",
        "Verify that you are human",
        "Verifying you are human. This may take a few seconds.",
        "智能验证检测中",
        "由极验提供技术支持",
    ],
)
def test_the_contracts_challenge_phrases(phrase: str) -> None:
    html = f"<html><body><p>{phrase}</p></body></html>"
    assert _state("https://example.org/", html) == ("challenge", "challenge")
    # Inside an article with text, the same words are content
    article = f"<html><body><article><p>{phrase}</p></article></body></html>"
    assert _state("https://example.org/", article)[1] != "challenge"


@pytest.mark.parametrize(
    "markup",
    [
        '<div class="cf-browser-verification"></div>',
        '<div id="cf-chl-widget-x"></div>',
        '<form action="/?__cf_chl_f_tk=1"></form>',
        '<script src="/cdn-cgi/challenge-platform/scripts/x.js"></script>',
        '<script src="https://www.google.com/recaptcha/api.js"></script>',
        '<script src="https://js.hcaptcha.com/1/api.js"></script>',
        '<div class="g-recaptcha"></div>',
        '<div class="h-captcha"></div>',
    ],
)
@pytest.mark.parametrize(
    "title",
    [
        "Just a moment...",
        "Attention Required!",
        "Security verification",
        "Verify you are human",
    ],
)
def test_the_contracts_widget_markers_with_a_challenge_title(
    markup: str, title: str
) -> None:
    html = f"<html><head><title>{title}</title></head><body>{markup}<p>Wait.</p></body></html>"
    assert _state("https://example.org/", html) == ("challenge", "challenge")


def test_a_widget_needs_a_challenge_title_or_no_text() -> None:
    form = '<div class="g-recaptcha"></div>'
    titled = f"<html><head><title>Contact us</title></head><body>{form}"
    assert _state(
        "https://example.org/", titled + "<p>Write to us.</p></body></html>"
    ) == (
        "empty",
        "blank",
    )
    assert _state("https://example.org/", titled + "</body></html>") == (
        "challenge",
        "challenge",
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "Please enable JavaScript to view this page.",
        "JavaScript is disabled in your browser.",
        "JavaScript is not available.",
        "You need to enable JavaScript to run this app.",
        "Enable JavaScript to continue",
    ],
)
def test_a_javascript_notice_is_empty(phrase: str) -> None:
    html = f"<html><head><title>Example</title></head><body><div><p>{phrase}</p></div></body></html>"
    assert _state("https://example.com/status/1", html) == ("empty", "javascript")


def test_a_long_page_is_never_a_notice() -> None:
    text = "Verify you are human " + "and read the long article. " * 100
    assert _state("https://example.org/", f"<body><p>{text}</p></body>") == ("ok", None)


def test_an_empty_body_is_blank_and_80_characters_is_content() -> None:
    for empty in ("", "<html><head><title> </title></head><body>\n \t</body></html>"):
        assert page_quality("https://example.com/", empty) == {
            "state": "empty",
            "text_length": 0,
            "title": None,
            "reason": "blank",
        }

    def text(count: int) -> str:
        return f"<body><p>{'x' * count}</p></body>"

    assert _state("https://example.com/", text(79)) == ("empty", "blank")
    assert _state("https://example.com/", text(80)) == ("ok", None)


def _post(body: str) -> dict:
    return page_quality(
        "https://example.com/p/1",
        '<html><head><title>Post</title><link rel="icon" href="/i.png"></head>'
        f"<body>{body}</body></html>",
    )


@pytest.mark.parametrize(
    "shown",
    [
        '<article><img src="https://pictures.example.com/river.jpg" alt="A river">'
        "<p>Autumn light on the river.</p></article>",
        '<picture><source srcset="/river.avif"><img srcset="/river.jpg 1x"></picture>'
        "<p>Autumn light.</p>",
        '<video controls src="https://media.example.com/clip.mp4"></video>'
        "<p>A short clip.</p>",
        '<video><source src="/clip.webm"></video>',
        '<audio controls src="/song.ogg"></audio><p>Track 1</p>',
    ],
)
def test_a_short_caption_under_a_picture_or_a_player_is_ok(shown: str) -> None:
    """Rust ``page_capture::tests::a_short_caption_under_a_picture_or_a_player_is_ok``."""
    quality = _post(shown)
    assert (quality["state"], quality["reason"]) == ("ok", None)


@pytest.mark.parametrize(
    "unseen",
    [
        '<div id="app"></div><noscript><img height="1" width="1" style="display:none"'
        ' src="https://pixel.example.com/t"></noscript>',
        '<img src="https://pixel.example.com/t" width="1" height="1" alt="">'
        '<div id="root"></div>',
        '<img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" data-src="/river.jpg">',
        '<img src="  "><img alt="No address">',
        '<img hidden src="/river.jpg"><div style="Display : None">'
        '<img src="/river.jpg"></div>',
        '<div style="visibility:hidden"><video src="/clip.mp4"></video></div>',
        '<audio src="/chime.ogg" autoplay></audio>',
        '<template><img src="/river.jpg"></template>'
        '<svg viewBox="0 0 10 10"><circle r="4"/></svg>',
    ],
)
def test_nothing_a_reader_would_see_is_still_blank(unseen: str) -> None:
    quality = _post(unseen)
    assert (quality["state"], quality["reason"]) == ("empty", "blank")


def test_only_the_blank_rule_looks_at_pictures() -> None:
    quality = _post('<img src="/logo.png"><p>Please enable JavaScript to continue.</p>')
    assert (quality["state"], quality["reason"]) == ("empty", "javascript")


def test_more_pictures_that_do_not_count() -> None:
    """Cases beyond Rust's list, from the same rule."""
    for unseen in (
        # A noscript pixel without other hiding, and a 0-sized one
        '<noscript><img src="https://pixel.example.com/t"></noscript>',
        '<img src="/t.gif" width=" 0 ">',
        # An ancestor's ``hidden``; ``DATA:`` in any case
        '<section hidden><figure><img src="/river.jpg"></figure></section>',
        '<img src=" DATA:image/png;base64,AAAA">',
    ):
        assert _post(unseen)["reason"] == "blank", unseen
    for shown in (
        '<img src="/river.jpg" width="10" height="2">',
        '<img src="data:image/png;base64,AAAA" srcset="/river.jpg 2x">',
    ):
        assert _post(shown)["state"] == "ok", shown


def test_text_is_counted_in_characters_and_capped_and_titles_are_bounded() -> None:
    wide = "<body><p>" + "字  \n " * 100 + "</p></body>"
    assert page_quality("https://example.com/", wide)["text_length"] == 199
    long = (
        "<head><title>"
        + "t" * 300
        + "</title></head><body><p>"
        + "word " * 300_000
        + "</p></body>"
    )
    quality = page_quality("https://example.com/", long)
    assert quality["text_length"] == MAX_TEXT_LENGTH
    assert len(quality["title"]) == 200
    # The first title counts; template text never does
    quality = page_quality(
        "https://example.com/",
        "<title>First</title><svg><title>Second</title></svg>"
        "<template><p>hidden</p></template>",
    )
    assert (quality["title"], quality["text_length"]) == ("First", 0)
    assert classify(None, "", 5_000_000, "")["text_length"] == MAX_TEXT_LENGTH
    # Comments are not text; the text around them is
    assert page_quality("u", "<p>a<!-- not shown -->b<b>c</b>d</p>")["text_length"] == 4


def test_pythons_own_verdict_only_adds() -> None:
    """``fetch_page_check``'s verdict adds a challenge, never removes one."""
    page = "<body><p>" + "Plenty of words on a real page. " * 10 + "</p></body>"
    length = 300
    assert classify("https://example.org/", page, length, "T")["state"] == "ok"
    assert (
        classify("https://example.org/", page, length, "T", problem="challenge")[
            "reason"
        ]
        == "challenge"
    )
    # Its blank verdict never overrides the contract's rule: a page that
    # rule calls ok stays ok (v1.14 "Blank pages")
    assert (
        classify("https://example.org/", page, length, "T", problem="blank")["state"]
        == "ok"
    )
    post = '<body><img src="/river.jpg"><p>Autumn light.</p></body>'
    assert classify("https://example.org/", post, 13, "", problem="blank") == {
        "state": "ok",
        "text_length": 13,
        "title": None,
        "reason": None,
    }
    assert (
        classify("https://example.org/", "<body></body>", 0, "", problem="blank")[
            "reason"
        ]
        == "blank"
    )
    login = "<body><p>Log in to continue</p></body>"
    assert (
        classify("https://www.zhihu.com/q", login, 20, "", problem="blank")["reason"]
        == "login"
    )


def test_blocks_take_the_screenshots_size_and_lose_what_is_below_it() -> None:
    value = {
        "version": 1,
        "width": 1300,
        "height": 9000,
        "blocks": [
            {"tag": "p", "text": "top", "bbox": [0, 10, 100, 30]},
            {"tag": "p", "text": "cut", "bbox": [0, 4000, 100, 4020]},
            {
                "tag": "img",
                "text": "",
                "src": "https://example.com/a.png",
                "bbox": [0, 3990, 100, 4100],
            },
        ],
    }
    data = sized_blocks(value, 1280, 4000)
    assert data is not None
    sized = json.loads(data)
    assert (sized["width"], sized["height"]) == (1280, 4000)
    assert [block["tag"] for block in sized["blocks"]] == ["p", "img"]
    assert sized_blocks({"version": 2, "blocks": []}, 1, 1) is None
    assert sized_blocks(None, 1, 1) is None
    huge = {
        "version": 1,
        "blocks": [
            {"tag": "p", "text": "x" * 1000, "bbox": [0, 0, 1, 1]} for _ in range(5000)
        ],
    }
    assert sized_blocks(huge, 1, 1) is None
    assert MAX_BLOCKS_BYTES == 4 * 1024 * 1024


def test_the_block_script_is_the_bundles_file() -> None:
    script = page_blocks_script()
    assert script is not None
    assert PAGE_BLOCKS_PATH.parts[-4:] == (
        "serve",
        "workbench",
        "capture",
        "page-blocks.js",
    )
    manifest = json.loads(
        (PAGE_BLOCKS_PATH.parents[1] / "manifest.json").read_text(encoding="utf-8")
    )
    assert "capture/page-blocks.js" in manifest["files"]
    assert script.rstrip().endswith("})()")
