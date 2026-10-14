"""Whether a page the browser drew is the page, or a stand-in for it.

A site can answer a browser with a verification page (a CAPTCHA, a "checking
your browser" interstitial) or with nothing at all (a refused request that
leaves an empty document). Neither is worth keeping as the page or as its
screenshot. :func:`page_problem` names the case from the rendered HTML:

- ``"challenge"``: little visible text (under ``CHALLENGE_MAX_TEXT``
  characters) together with a verification marker: a vendor's script or
  element in the HTML, or a "verify you are human" phrase in the text. A long
  article that mentions CAPTCHAs is not one.
- ``"blank"``: under ``BLANK_MAX_TEXT`` visible characters and no picture,
  video, canvas or frame.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

CHALLENGE_MAX_TEXT = 1500
BLANK_MAX_TEXT = 20

_VENDOR_MARKERS = re.compile(
    r"/cdn-cgi/challenge-platform/|cf[-_]chl[-_]|challenges\.cloudflare\.com/turnstile"
    r"|geetest|captcha-delivery\.com|px-captcha|_Incapsula_Resource"
    r"|hcaptcha\.com/1/api\.js|google\.com/recaptcha|recaptcha/api\.js"
    r"|awswaf|aliyun_captcha|nc_1_wrapper|tcaptcha|turing\.captcha",
    re.IGNORECASE,
)
_TEXT_MARKERS = re.compile(
    r"智能验证检测中|由极验提供技术支持|请在下图依次点击|请按顺序点击|拖动滑块|滑动验证"
    r"|请完成(?:安全)?验证|安全验证|人机验证|验证码"
    r"|verify (?:that )?you are (?:a )?human|are you a robot|checking your browser"
    r"|just a moment\.\.\.|attention required|complete the security check"
    r"|unusual traffic from your computer|press (?:&|and) hold",
    re.IGNORECASE,
)
_MEDIA_TAGS = frozenset(
    {"img", "video", "canvas", "picture", "iframe", "embed", "object"}
)
_HIDDEN_TAGS = frozenset({"script", "style", "noscript", "template", "head", "title"})


class _Visible(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.media = 0
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HIDDEN_TAGS:
            self.hidden += 1
        elif tag in _MEDIA_TAGS:
            self.media += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _HIDDEN_TAGS and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.text.append(data)


def visible_text(html: str) -> tuple[str, int]:
    """The document's visible text (whitespace collapsed) and its media count."""
    parser = _Visible()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # html.parser is lenient; never fail a fetch on it
        pass
    return " ".join("".join(parser.text).split()), parser.media


def page_problem(
    html: str | None,
    *,
    text: str | None = None,
    media: int | None = None,
    shadow_roots: int = 0,
) -> str | None:
    """``"challenge"``, ``"blank"`` or None for the rendered page *html*.

    ``text``/``media`` are the live page's visible text and media count when
    the browser measured them (else they are read from *html*). A page built
    from shadow roots is never called blank: its text is not in the HTML.
    """
    if not html:
        return None if shadow_roots else "blank"
    if text is None or media is None:
        parsed_text, parsed_media = visible_text(html)
        text = parsed_text if text is None else " ".join(text.split())
        media = parsed_media if media is None else media
    else:
        text = " ".join(text.split())
    if len(text) < CHALLENGE_MAX_TEXT and (
        _TEXT_MARKERS.search(text) or _VENDOR_MARKERS.search(html)
    ):
        return "challenge"
    if len(text) < BLANK_MAX_TEXT and not media and not shadow_roots:
        return "blank"
    return None


#: What a kept-page problem means, for notices.
PROBLEM_TEXT = {
    "challenge": "the site showed a verification page (CAPTCHA) instead of the page",
    "blank": "the page the browser loaded was empty",
}
