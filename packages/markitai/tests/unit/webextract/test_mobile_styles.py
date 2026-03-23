"""Tests for CSS @media mobile style pruning."""

from __future__ import annotations

from bs4 import BeautifulSoup

from markitai.webextract.mobile_styles import apply_mobile_style_pruning


def test_removes_sidebar_hidden_on_mobile() -> None:
    html = """<html><head><style>
    @media (max-width: 768px) { .sidebar { display: none; } }
    </style></head><body>
    <article><p>Main content.</p></article>
    <div class="sidebar"><p>Sidebar noise.</p></div>
    </body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    removed = apply_mobile_style_pruning(soup)
    assert removed > 0
    assert "Sidebar noise" not in soup.get_text()
    assert "Main content" in soup.get_text()


def test_preserves_elements_not_hidden() -> None:
    html = """<html><head><style>
    @media (max-width: 768px) { .nav { display: none; } }
    </style></head><body>
    <article><p>Content.</p></article>
    <div class="footer"><p>Footer.</p></div>
    </body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    apply_mobile_style_pruning(soup)
    assert "Footer" in soup.get_text()


def test_no_style_tags_returns_zero() -> None:
    html = "<html><body><p>Just text.</p></body></html>"
    soup = BeautifulSoup(html, "html.parser")
    assert apply_mobile_style_pruning(soup) == 0


def test_ignores_large_breakpoint_media_queries() -> None:
    html = """<html><head><style>
    @media (max-width: 1200px) { .wide-sidebar { display: none; } }
    </style></head><body>
    <div class="wide-sidebar"><p>Should stay.</p></div>
    </body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    apply_mobile_style_pruning(soup)
    assert "Should stay" in soup.get_text()
