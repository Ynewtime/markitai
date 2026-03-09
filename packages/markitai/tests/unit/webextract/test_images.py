from __future__ import annotations


def test_normalize_images_upgrades_lazy_sources_and_builds_figures() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.images import normalize_images

    soup = parse_html(
        """
        <article>
          <img data-src="/hero.png" alt="Hero" />
          <p class="caption">Hero caption</p>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    normalize_images(article, base_url="https://example.com/post")

    html = str(article)
    assert 'src="https://example.com/hero.png"' in html
    assert "<figure" in html
    assert "Hero caption" in html


def test_normalize_images_keeps_captioned_figure_in_original_order() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.images import normalize_images

    soup = parse_html(
        """
        <article>
          <p>Before</p>
          <img data-src="/hero.png" alt="Hero" />
          <p class="caption">Hero caption</p>
          <p>After</p>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    normalize_images(article, base_url="https://example.com/post")

    html = str(article)
    assert html.index("Before") < html.index("<figure") < html.index("After")
