"""A page address in front matter reads as an IRI, not percent-escapes.

Seen in practice: a URL item's ``source`` and ``canonical_url``
were ``https://ynewtime.com/posts/20251223-ai-%E5%BA%94%E7%94%A8…``. They are
written with the characters decoded, the same address: turning the IRI back
into a URI (RFC 3987 §3.1) gives the original.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest
import yaml

from markitai.utils.frontmatter import build_frontmatter_dict, frontmatter_to_yaml
from markitai.utils.iri import readable_url
from markitai.workflow.helpers import add_basic_frontmatter

_ENCODED = (
    "https://ynewtime.com/posts/20251223-ai-%E5%BA%94%E7%94%A8%E6%A1%88%E4%BE%8B"
    "%E7%B3%BB%E5%88%972chrome-%E6%8F%92%E4%BB%B6/"
)
_READABLE = "https://ynewtime.com/posts/20251223-ai-应用案例系列2chrome-插件/"


def _as_uri(iri: str) -> str:
    """RFC 3987 §3.1: each non-ASCII character as its UTF-8 escapes."""
    return quote(iri, safe="".join(chr(c) for c in range(0x21, 0x7F)))


def _frontmatter(text: str) -> dict:
    return yaml.safe_load(text.split("---\n", 2)[1])


class TestReadableUrl:
    def test_the_owners_address(self) -> None:
        assert readable_url(_ENCODED) == _READABLE
        assert _as_uri(readable_url(_ENCODED)) == _ENCODED

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            # ASCII escapes keep their meaning: space, slash, &, %, ?, #
            (
                "https://x.test/a%20b%2Fc?q=%E4%B8%AD%26x#%E6%96%87",
                "https://x.test/a%20b%2Fc?q=中%26x#文",
            ),
            ("https://x.test/100%25", "https://x.test/100%25"),
            # Lower-case escapes decode too
            ("https://x.test/%e4%b8%ad", "https://x.test/中"),
            # Bidi controls, ideographic space, non-UTF-8 bytes, a cut
            # sequence: kept as escapes
            ("https://x.test/%E2%80%8Fa", "https://x.test/%E2%80%8Fa"),
            ("https://x.test/%E3%80%80", "https://x.test/%E3%80%80"),
            ("https://x.test/%FF%E4%B8", "https://x.test/%FF%E4%B8"),
            # The authority stays as written (punycode, userinfo, port)
            ("https://xn--fiq.test:8443/%E4%B8%AD", "https://xn--fiq.test:8443/中"),
            # Not a web address: unchanged
            ("report%E4%B8%AD.pdf", "report%E4%B8%AD.pdf"),
            ("ftp://x.test/%E4%B8%AD", "ftp://x.test/%E4%B8%AD"),
            ("https://x.test/plain", "https://x.test/plain"),
        ],
    )
    def test_rules(self, url: str, expected: str) -> None:
        assert readable_url(url) == expected

    def test_round_trip(self) -> None:
        for url in (
            _ENCODED,
            "https://x.test/%E4%B8%AD/%F0%9F%98%80?a=%E6%97%A5%E6%9C%AC#%C3%A9",
        ):
            assert _as_uri(readable_url(url)) == url


class TestFrontMatter:
    def test_base_output(self) -> None:
        text = add_basic_frontmatter(
            "# Title\n\nBody",
            _ENCODED,
            title="AI 应用案例系列2",
            extra_meta={"canonical_url": _ENCODED, "site": "昨日书"},
        )
        data = _frontmatter(text)
        assert data["source"] == _READABLE
        assert data["canonical_url"] == _READABLE
        assert "%E5" not in text

    def test_llm_output(self) -> None:
        data = build_frontmatter_dict(
            source=_ENCODED,
            description="d",
            tags=["t"],
            title="t",
            extra_meta={"canonical_url": _ENCODED},
        )
        loaded = yaml.safe_load(frontmatter_to_yaml(data))
        assert loaded["source"] == _READABLE
        assert loaded["canonical_url"] == _READABLE

    def test_a_file_source_is_left_alone(self) -> None:
        text = add_basic_frontmatter("Body", "report%20final.pdf", title="Report")
        assert _frontmatter(text)["source"] == "report%20final.pdf"

    def test_an_address_yaml_would_misread_stays_a_string(self) -> None:
        url = "https://x.test/%E4%B8%AD: #not-a-comment"
        text = add_basic_frontmatter("Body", url, title="T")
        assert _frontmatter(text)["source"] == "https://x.test/中: #not-a-comment"
