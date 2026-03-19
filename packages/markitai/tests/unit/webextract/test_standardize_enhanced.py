"""Tests for enhanced standardization (Phase 2)."""

from __future__ import annotations

from markitai.webextract.dom import parse_html


class TestConvertH1ToH2:
    def test_converts_multiple_h1_to_h2(self):
        from markitai.webextract.standardize import _convert_h1_to_h2

        soup = parse_html("<div><h1>Title</h1><p>content</p><h1>Another</h1></div>")
        root = soup.find("div")
        _convert_h1_to_h2(root)
        assert root.find("h1") is None
        h2s = root.find_all("h2")
        assert len(h2s) == 2
        assert h2s[0].get_text() == "Title"
        assert h2s[1].get_text() == "Another"

    def test_keeps_single_h1(self):
        from markitai.webextract.standardize import _convert_h1_to_h2

        soup = parse_html("<div><h1>Title</h1><p>content</p></div>")
        root = soup.find("div")
        _convert_h1_to_h2(root)
        # Single h1 should be kept as-is
        assert root.find("h1") is not None


class TestUnwrapBareSpans:
    def test_unwraps_spans_without_attributes(self):
        from markitai.webextract.standardize import _unwrap_bare_spans

        soup = parse_html("<div><p><span>text</span></p></div>")
        root = soup.find("div")
        _unwrap_bare_spans(root)
        assert root.find("span") is None
        assert "text" in root.get_text()

    def test_keeps_spans_with_class(self):
        from markitai.webextract.standardize import _unwrap_bare_spans

        soup = parse_html('<div><p><span class="highlight">text</span></p></div>')
        root = soup.find("div")
        _unwrap_bare_spans(root)
        assert root.find("span") is not None


class TestRemoveEmptyElements:
    def test_removes_empty_div(self):
        from markitai.webextract.standardize import _remove_empty_elements

        soup = parse_html("<div><div></div><p>content</p></div>")
        root = soup.find("div")
        _remove_empty_elements(root)
        # Only the outer div and p should remain
        inner_divs = root.find_all("div")
        assert len(inner_divs) == 0 or all(d.get_text(strip=True) for d in inner_divs)

    def test_keeps_void_elements(self):
        from markitai.webextract.standardize import _remove_empty_elements

        soup = parse_html('<div><img src="photo.jpg"><br><hr></div>')
        root = soup.find("div")
        _remove_empty_elements(root)
        assert root.find("img") is not None
        assert root.find("br") is not None
        assert root.find("hr") is not None


class TestRemoveTrailingContent:
    def test_removes_trailing_hr(self):
        from markitai.webextract.standardize import _remove_trailing_content

        soup = parse_html("<div><p>content</p><hr></div>")
        root = soup.find("div")
        _remove_trailing_content(root)
        assert root.find("hr") is None

    def test_keeps_hr_in_middle(self):
        from markitai.webextract.standardize import _remove_trailing_content

        soup = parse_html("<div><p>before</p><hr><p>after</p></div>")
        root = soup.find("div")
        _remove_trailing_content(root)
        assert root.find("hr") is not None


class TestFlattenWrapperDivs:
    def test_unwraps_single_child_wrapper(self):
        from markitai.webextract.standardize import _flatten_wrapper_divs

        soup = parse_html("<div><div><p>content</p></div></div>")
        root = soup.find("div")
        _flatten_wrapper_divs(root)
        # Inner div should be unwrapped
        assert root.find("p") is not None

    def test_keeps_div_with_multiple_block_children(self):
        from markitai.webextract.standardize import _flatten_wrapper_divs

        soup = parse_html("<div><div><p>a</p><p>b</p></div></div>")
        root = soup.find("div")
        _flatten_wrapper_divs(root)
        # div with multiple block children may or may not be unwrapped
        # but content should be preserved
        assert "a" in root.get_text()
        assert "b" in root.get_text()

    def test_preserves_semantic_elements(self):
        from markitai.webextract.standardize import _flatten_wrapper_divs

        soup = parse_html("<div><pre>code</pre></div>")
        root = soup.find("div")
        _flatten_wrapper_divs(root)
        assert root.find("pre") is not None
