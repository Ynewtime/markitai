"""Tests for Playwright rendering enhancements (auto-scroll + DOM cleanup)."""


class TestAutoScroll:
    """Test auto-scroll logic for triggering lazy-loaded content."""

    def test_auto_scroll_script_structure(self):
        """Verify auto-scroll JS script has correct structure."""
        from markitai.fetch_playwright import _build_auto_scroll_script

        script = _build_auto_scroll_script(max_steps=8, step_delay_ms=600)
        assert "scrollTo" in script
        assert "scrollHeight" in script
        assert "document.body" in script

    def test_auto_scroll_script_returns_to_top(self):
        """Verify scroll returns to top after scrolling."""
        from markitai.fetch_playwright import _build_auto_scroll_script

        script = _build_auto_scroll_script(max_steps=8, step_delay_ms=600)
        assert "scrollTo(0, 0)" in script or "scrollTo(0,0)" in script

    def test_auto_scroll_custom_params(self):
        """Verify custom parameters are reflected in script."""
        from markitai.fetch_playwright import _build_auto_scroll_script

        script = _build_auto_scroll_script(max_steps=5, step_delay_ms=300)
        assert "5" in script
        assert "300" in script

    def test_auto_scroll_default_params(self):
        """Verify default parameters work."""
        from markitai.fetch_playwright import _build_auto_scroll_script

        script = _build_auto_scroll_script()
        assert "scrollTo" in script


class TestDomCleanup:
    """Test DOM cleanup for removing noise elements before extraction."""

    def test_dom_cleanup_script_removes_noise(self):
        """Verify DOM cleanup script targets known noise selectors."""
        from markitai.fetch_playwright import _build_dom_cleanup_script

        script = _build_dom_cleanup_script()
        assert "script" in script
        assert "noscript" in script
        assert "cookie-banner" in script

    def test_dom_cleanup_script_removes_ad_elements(self):
        """Verify ad-related elements are targeted."""
        from markitai.fetch_playwright import _build_dom_cleanup_script

        script = _build_dom_cleanup_script()
        assert ".advertisement" in script

    def test_dom_cleanup_cleans_attributes(self):
        """Verify inline event handlers and styles are removed."""
        from markitai.fetch_playwright import _build_dom_cleanup_script

        script = _build_dom_cleanup_script()
        assert "onclick" in script
        assert "removeAttribute" in script

    def test_dom_cleanup_converts_urls(self):
        """Verify relative URLs are converted to absolute."""
        from markitai.fetch_playwright import _build_dom_cleanup_script

        script = _build_dom_cleanup_script()
        assert "baseURI" in script
        assert "new URL" in script
