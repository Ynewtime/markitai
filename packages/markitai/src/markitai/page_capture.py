"""What loading a URL's page left for showing it beside its Markdown.

``markitai serve`` keeps, for a URL item, the page it was converted from,
the browser's screenshots of it, where the page's blocks were drawn on them,
and how usable each view was when it was captured (workbench contract v1.14,
section 7). This module holds the parts that do not know about serve:

- :func:`page_quality` and :func:`classify` judge a view: ``ok``, ``empty``
  (blank, or a page that only asks for JavaScript) or ``challenge`` (a
  verification, captcha, bot check or login page in place of the content),
  by the contract's rules, which are Rust's ``fetch::sites`` rules. Python's
  own markers (:mod:`markitai.fetch_page_check`) may add a verdict, never
  remove one.
- :func:`sized_blocks` fits the block boxes ``capture/page-blocks.js`` read
  to the area the screenshots cover.
- :class:`CaptureSink` collects, for one conversion, what the browser
  showed when it took screenshots. A caller that keeps pages (serve) puts
  one in :data:`CAPTURE` around a conversion; the browser fetch then also
  takes *compare screenshots* when the conversion asked for none. They are
  never conversion output and never reach a model or OCR.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

#: Characters of visible text counted at most.
MAX_TEXT_LENGTH = 1_000_000
#: Characters of a title kept at most.
MAX_TITLE = 200
#: Visible text shorter than this, in characters, is a blank page.
BLANK_BELOW = 80
#: Bytes of block boxes kept at most.
MAX_BLOCKS_BYTES = 4 * 1024 * 1024
#: Compare screenshots: JPEG quality, tiles and pixels at most.
COMPARE_QUALITY = 70
COMPARE_TILES = 20
COMPARE_MAX_PIXELS = 50_000_000
#: A notice page's visible text is at most this long (characters).
_NOTICE_TEXT = 2000
#: "Little else" beside a login request, in words.
_FEW_WORDS = 60

#: The block-box script the workbench bundle ships; evaluated unchanged.
PAGE_BLOCKS_PATH = (
    Path(__file__).resolve().parent
    / "serve"
    / "workbench"
    / "capture"
    / "page-blocks.js"
)

# Visible text a challenge page starts with (the contract's six, then
# Rust's two more).
_CHALLENGE_TEXT = (
    "checking your browser before accessing",
    "verify you are human",
    "verify that you are human",
    "verifying you are human",
    "智能验证检测中",
    "由极验提供技术支持",
    "请在下图依次点击",
    "请完成下列验证后继续",
)
_CHALLENGE_TITLES = (
    "just a moment",
    "attention required",
    "security verification",
    "verify you are human",
)
# A challenge widget, in an element's id, class, src or action (the
# contract's eight, then Rust's and Python's own vendors).
_WIDGET_MARKERS = (
    "cf-browser-verification",
    "cf-chl-",
    "cf_chl_",
    "/cdn-cgi/challenge-platform/",
    "google.com/recaptcha",
    "hcaptcha.com",
    "g-recaptcha",
    "h-captcha",
    "geetest_",
    "static.geetest.com",
    "cf-turnstile",
    "challenges.cloudflare.com/turnstile",
    "captcha-delivery.com",
    "px-captcha",
    "_incapsula_resource",
    "recaptcha/api.js",
)
_JAVASCRIPT_TEXT = (
    "please enable javascript",
    "javascript is disabled",
    "javascript is not available",
    "you need to enable javascript",
    "enable javascript to continue",
)
# Requests to log in a known site shows in place of a page (lower case).
_LOGIN_REQUESTS = (
    "请您登录后查看",
    "登录后查看更多",
    "登录后才能查看",
    "log in to continue",
    "sign in to continue",
    "log in to view",
)


@dataclass(frozen=True)
class _Site:
    name: str
    domains: tuple[str, ...]
    #: Whether the site expects a login (its login wall is a refusal)
    login: bool


_SITES = (
    _Site("Zhihu", ("zhihu.com",), True),
    _Site("WeChat", ("weixin.qq.com",), False),
    _Site("Douban", ("douban.com",), True),
    _Site("Bilibili", ("bilibili.com",), False),
    _Site("Weibo", ("weibo.com", "weibo.cn"), True),
    _Site("Toutiao", ("toutiao.com",), False),
    _Site("Reddit", ("reddit.com",), True),
)

_CJK = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿\U00020000-\U0002a6df]")
_WORD = re.compile(r"[^\W]+")


def _collapsed(text: str) -> str:
    return " ".join(text.split())


def _word_count(text: str) -> int:
    """Words as Rust counts them: each CJK character, each run of others."""
    cjk = len(_CJK.findall(text))
    return cjk + len(_WORD.findall(_CJK.sub(" ", text)))


def _site_of(host: str) -> _Site | None:
    host = host.rstrip(".").lower()
    for site in _SITES:
        if any(host == d or host.endswith("." + d) for d in site.domains):
            return site
    return None


@dataclass
class _Shown:
    """Markup reduced to what tells a refusal from the content."""

    title: str = ""
    #: Text outside head/script/style/template and pre/code, collapsed
    text: str = ""
    widget: bool = False
    article: bool = False


_HIDDEN = frozenset({"head", "script", "style", "template", "title"})


def _parse(html: str) -> Any:
    """An lxml tree of *html* (None for markup lxml cannot read)."""
    import lxml.html

    data = html.encode("utf-8", "replace")
    if not data.strip():
        return None
    try:
        parser = lxml.html.HTMLParser(encoding="utf-8", remove_comments=True)
        return lxml.html.document_fromstring(data, parser=parser)
    except Exception:  # lxml raises on documents it finds empty
        return None


def _texts(root: Any, hidden: frozenset[str]) -> list[tuple[str, bool]]:
    """Text nodes of *root* in order, each with whether it is literal
    (inside ``pre``/``code``); text under a *hidden* element is skipped.
    """
    out: list[tuple[str, bool]] = []
    # (node, hidden, literal, is its tail): a tail is text after the node,
    # inside the node's parent
    stack: list[tuple[Any, bool, bool, bool]] = [(root, False, False, False)]
    while stack:
        element, parent_hidden, parent_literal, is_tail = stack.pop()
        if is_tail:
            if element.tail and not parent_hidden:
                out.append((element.tail, parent_literal))
            continue
        tag = element.tag if isinstance(element.tag, str) else ""
        name = tag.lower()
        # A comment or processing instruction has no tag name: its text is
        # never shown, its tail is
        own_hidden = parent_hidden or name in hidden or not tag
        own_literal = parent_literal or name in ("pre", "code")
        if element.text and not own_hidden:
            out.append((element.text, own_literal))
        inner_hidden = parent_hidden or name in hidden
        # Reversed, so the first child pops first and its tail right after it
        for child in reversed(list(element)):
            stack.append((child, inner_hidden, own_literal, True))
            stack.append((child, inner_hidden, own_literal, False))
    return out


def _title(root: Any) -> str:
    for element in root.iter("title"):
        return element.text_content() or ""
    return ""


def _shown(html: str) -> _Shown:
    """What a reader sees of *html*, and the challenge evidence in it."""
    root = _parse(html)
    if root is None:
        return _Shown()
    shown = _Shown(title=_collapsed(_title(root)))
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        name = element.tag.lower()
        if name == "article" and element.text_content().strip():
            shown.article = True
        if not shown.widget:
            for attr in ("id", "class", "src", "action"):
                value = (element.get(attr) or "").lower()
                if value and any(marker in value for marker in _WIDGET_MARKERS):
                    shown.widget = True
                    break
    shown.text = _collapsed(
        " ".join(
            text for text, literal in _texts(root, _HIDDEN - {"title"}) if not literal
        )
    )
    return shown


def framed_text(html: str) -> tuple[str, str]:
    """The text and first title of *html* as a frame without scripts shows
    it: outside ``head``, ``script``, ``style``, ``template`` and ``title``,
    ``<noscript>`` text included.
    """
    root = _parse(html)
    if root is None:
        return "", ""
    text = "".join(text for text, _ in _texts(root, _HIDDEN))
    return text, _title(root)


def _zhihu_check(title: str) -> bool:
    bare = title
    if title.endswith("知乎"):
        bare = title[: -len("知乎")].rstrip().rstrip("-–|").rstrip()
    return bare == "安全验证"


def _site_refusal(url: str | None, markup: str, shown: _Shown) -> str | None:
    """``verification`` or ``login`` for a known site's own refusal page."""
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    site = _site_of(host) if host else None
    if site is None:
        return None
    path = parts.path

    def says(words: str) -> bool:
        return words in markup or words in shown.text or words in shown.title

    verification = {
        "WeChat": lambda: (
            "wappoc_appmsgcaptcha" in path
            or (host == "mp.weixin.qq.com" and says("环境异常") and says("完成验证"))
        ),
        "Douban": lambda: host == "sec.douban.com",
        "Weibo": lambda: (
            (host == "passport.weibo.com" and path.startswith("/visitor"))
            or shown.title == "Sina Visitor System"
        ),
        "Reddit": lambda: says("Prove your humanity"),
        "Bilibili": lambda: (
            'id="risk-captcha-app"' in markup or shown.title == "验证码_哔哩哔哩"
        ),
        "Toutiao": lambda: "_$jsvmprt" in markup,
        "Zhihu": lambda: (
            path.startswith("/account/unhuman")
            or 'id="zh-zse-ck"' in markup
            or _zhihu_check(shown.title)
        ),
    }[site.name]
    if verification():
        return "verification"
    folded = f"{shown.title} {shown.text}".lower()
    if (
        site.login
        and any(request in folded for request in _LOGIN_REQUESTS)
        and _word_count(shown.text) <= _FEW_WORDS
    ):
        return "login"
    return None


def _notice(shown: _Shown) -> str | None:
    """``challenge`` or ``javascript`` for a notice page on any site."""
    text = shown.text.lower()
    if len(text) > _NOTICE_TEXT:
        return None
    title = shown.title.lower()
    instruction = text.startswith(_CHALLENGE_TEXT)
    if (not shown.article and instruction) or (
        shown.widget and (title.startswith(_CHALLENGE_TITLES) or not text)
    ):
        return "challenge"
    if not shown.article and text.startswith(_JAVASCRIPT_TEXT):
        return "javascript"
    return None


def shown_instead(url: str | None, markup: str) -> str | None:
    """What *markup* shows in place of its content, as the workbench names
    it: ``verification``/``login`` (a known site's page), ``challenge`` (a
    bot check) or ``javascript`` (a page that only asks for JavaScript).
    """
    shown = _shown(markup)
    return _site_refusal(url, markup, shown) or _notice(shown)


def _quality(
    state: str, text_length: int, title: str, reason: str | None
) -> dict[str, Any]:
    shown_title = _collapsed(title)[:MAX_TITLE]
    return {
        "state": state,
        "text_length": min(max(0, text_length), MAX_TEXT_LENGTH),
        "title": shown_title or None,
        "reason": reason,
    }


def text_length(text: str) -> int:
    """Characters of *text*, white space runs as one, ends trimmed, capped."""
    return min(len(_collapsed(text)), MAX_TEXT_LENGTH)


def classify(
    url: str | None,
    markup: str,
    length: int,
    title: str,
    *,
    problem: str | None = None,
) -> dict[str, Any]:
    """Judge a view (contract v1.14 ``CaptureQuality``).

    *markup* is the HTML the view showed (the kept page, or the browser's
    DOM), read from *url*; *length* and *title* are what it showed a reader.
    *problem* is :func:`markitai.fetch_page_check.page_problem`'s verdict on
    the same view, which can only add one: ``challenge`` stays a challenge,
    ``blank`` makes the view empty.
    """
    instead = shown_instead(url, markup)
    if instead == "javascript":
        return _quality("empty", length, title, "javascript")
    if instead is not None:
        return _quality("challenge", length, title, instead)
    if problem == "challenge":
        return _quality("challenge", length, title, "challenge")
    if problem == "blank" or min(length, MAX_TEXT_LENGTH) < BLANK_BELOW:
        return _quality("empty", length, title, "blank")
    return _quality("ok", length, title, None)


def page_quality(url: str | None, html: str) -> dict[str, Any]:
    """How a kept page reads in a frame that runs no script."""
    text, title = framed_text(html)
    return classify(url, html, text_length(text), title)


def sized_blocks(value: Any, width: int, height: int) -> bytes | None:
    """``page-blocks.js``'s result for screenshots that cover *width* x
    *height* CSS pixels from the page's top: that size in place of the
    page's own, no block that starts below it, as JSON; None when it is not
    the script's shape or is larger than ``MAX_BLOCKS_BYTES``.
    """
    if not isinstance(value, dict) or value.get("version") != 1:
        return None
    blocks = value.get("blocks")
    if not isinstance(blocks, list):
        return None

    def above(block: Any) -> bool:
        try:
            top = block["bbox"][1]
        except (TypeError, KeyError, IndexError):
            return False
        return isinstance(top, int | float) and top < height

    sized = dict(value)
    sized["blocks"] = [block for block in blocks if above(block)]
    sized["width"] = int(width)
    sized["height"] = int(height)
    data = json.dumps(sized, ensure_ascii=False, separators=(",", ":")).encode()
    return data if len(data) <= MAX_BLOCKS_BYTES else None


_page_blocks_script: str | None = None


def page_blocks_script() -> str | None:
    """The bundle's ``capture/page-blocks.js`` text (None when missing)."""
    global _page_blocks_script
    if _page_blocks_script is None:
        try:
            _page_blocks_script = PAGE_BLOCKS_PATH.read_text(encoding="utf-8")
        except OSError:
            return None
    return _page_blocks_script


@dataclass
class ShotView:
    """What the page showed in the browser when its screenshots were taken."""

    quality: dict[str, Any]
    #: Sized block boxes (JSON), when they could be read
    blocks: bytes | None = None
    #: The conversion's own screenshot files these belong to (in order)
    screenshots: list[Path] = field(default_factory=list)
    #: Compare screenshots (JPEG tiles from the top) when none were asked for
    compare: list[bytes] = field(default_factory=list)


@dataclass
class CaptureSink:
    """What one conversion's browser loads left, for a caller keeping pages.

    ``tile_height`` is the compare screenshots' tile height (CSS pixels).
    The browser fetch sets :attr:`view` each time it takes screenshots (the
    last load wins, as its screenshots do) and :attr:`refused` when it set a
    page aside as a verification or blank page.
    """

    tile_height: int = 2000
    view: ShotView | None = None
    #: ``(html, verdict)`` of a page the browser showed instead of the
    #: content, kept only when no other page is
    refused: tuple[str, dict[str, Any]] | None = None


#: The sink of the conversion running in this context, if its caller keeps
#: pages (``markitai serve``); None everywhere else.
CAPTURE: ContextVar[CaptureSink | None] = ContextVar("page_capture", default=None)
