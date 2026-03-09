from __future__ import annotations


def test_extract_metadata_prefers_canonical_jsonld_and_meta_layers() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.metadata import extract_metadata

    soup = parse_html(
        """
        <html>
          <head>
            <title>How Arrays Work | MDN</title>
            <link rel="canonical" href="https://example.com/canonical" />
            <meta property="og:site_name" content="MDN" />
            <meta name="author" content="Jane Doe" />
            <script type="application/ld+json">
              {"@type":"Article","headline":"How Arrays Work","datePublished":"2026-02-01"}
            </script>
          </head>
          <body></body>
        </html>
        """
    )

    metadata = extract_metadata(soup, "https://example.com/post")

    assert metadata.title == "How Arrays Work"
    assert metadata.author == "Jane Doe"
    assert metadata.site == "MDN"
    assert metadata.published == "2026-02-01"
    assert metadata.canonical_url == "https://example.com/canonical"


def test_clean_title_removes_site_suffixes() -> None:
    from markitai.webextract.metadata import clean_title

    assert clean_title("How Arrays Work | MDN", site="MDN") == "How Arrays Work"


def test_extract_metadata_reads_article_fields_from_jsonld_graph() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.metadata import extract_metadata

    soup = parse_html(
        """
        <html>
          <head>
            <script type="application/ld+json">
              {
                "@context": "https://schema.org",
                "@graph": [
                  {"@type": "WebSite", "name": "Example Site"},
                  {
                    "@type": "NewsArticle",
                    "headline": "Graph Headline",
                    "author": {"name": "Graph Author"},
                    "datePublished": "2026-03-02"
                  }
                ]
              }
            </script>
          </head>
          <body></body>
        </html>
        """
    )

    metadata = extract_metadata(soup, "https://example.com/post")

    assert metadata.title == "Graph Headline"
    assert metadata.author == "Graph Author"
    assert metadata.published == "2026-03-02"
