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


def _extract(html: str, url: str = "https://example.com/blog/post"):
    from markitai.webextract.dom import parse_html
    from markitai.webextract.metadata import extract_metadata

    return extract_metadata(parse_html(html), url)


class TestSiteNameViaNameAttribute:
    def test_og_site_name_with_name_attribute_is_detected(self) -> None:
        """Jekyll-style ``<meta name="og:site_name">`` must count as site."""
        metadata = _extract(
            """
            <html><head>
              <title>My Post | Y</title>
              <meta name="og:site_name" content="Y" />
            </head><body></body></html>
            """
        )
        assert metadata.site == "Y"
        assert metadata.title == "My Post"


class TestTitleSuffixStripping:
    def test_site_suffix_stripped_when_h1_matches(self) -> None:
        """Without og:site_name, an h1 matching the prefix strips the suffix."""
        metadata = _extract(
            """
            <html><head><title>Great Article | Some Site</title></head>
            <body><h1>Great Article</h1></body></html>
            """
        )
        assert metadata.title == "Great Article"

    def test_suffix_kept_without_matching_headline(self) -> None:
        """No h1/JSON-LD evidence: the title is kept verbatim."""
        metadata = _extract(
            "<html><head><title>Great Article | Some Site</title></head>"
            "<body></body></html>"
        )
        assert metadata.title == "Great Article | Some Site"

    def test_repeated_site_affixes_are_all_stripped(self) -> None:
        """GitHub-style ``Site - text · Site`` loses both affixes."""
        from markitai.webextract.metadata import clean_title

        assert (
            clean_title("GitHub - owner/repo: description · GitHub", site="GitHub")
            == "owner/repo: description"
        )


class TestUntruncatedTitlePreference:
    def test_full_title_tag_preferred_over_truncated_headline(self) -> None:
        metadata = _extract(
            """
            <html><head>
              <title>The Complete And Very Long Article Title</title>
              <script type="application/ld+json">
                {"@type":"Article","headline":"The Complete And Very Long…"}
              </script>
            </head><body></body></html>
            """
        )
        assert metadata.title == "The Complete And Very Long Article Title"

    def test_truncated_title_kept_when_no_longer_source_exists(self) -> None:
        metadata = _extract(
            """
            <html><head>
              <script type="application/ld+json">
                {"@type":"Article","headline":"Truncated headline only…"}
              </script>
            </head><body></body></html>
            """
        )
        assert metadata.title == "Truncated headline only…"


class TestPublishedExtraction:
    def test_article_published_time_meta_wins(self) -> None:
        metadata = _extract(
            """
            <html><head>
              <meta property="article:published_time" content="2026-05-01T08:00:00Z" />
              <script type="application/ld+json">
                {"@type":"Article","datePublished":"2026-05-02"}
              </script>
            </head><body></body></html>
            """
        )
        assert metadata.published == "2026-05-01T08:00:00Z"

    def test_time_element_datetime_used_as_fallback(self) -> None:
        metadata = _extract(
            "<html><head></head><body>"
            '<time datetime="2018-04-16">Apr 16, 2018</time>'
            "</body></html>"
        )
        assert metadata.published == "2018-04-16"

    def test_time_element_with_non_date_value_ignored(self) -> None:
        metadata = _extract(
            "<html><head></head><body>"
            '<time datetime="PT2H30M">2h30m</time>'
            "</body></html>"
        )
        assert metadata.published is None


class TestCanonicalUrl:
    def test_missing_canonical_is_not_replaced_by_source_url(self) -> None:
        metadata = _extract("<html><head></head><body></body></html>")
        assert metadata.canonical_url is None

    def test_site_root_canonical_on_deeper_page_is_dropped(self) -> None:
        """A homepage canonical on an article page is a template artifact."""
        metadata = _extract(
            '<html><head><link rel="canonical" href="https://www.example.com/" />'
            "</head><body></body></html>",
            url="https://blog.example.com/some/deep/post",
        )
        assert metadata.canonical_url is None

    def test_page_specific_canonical_is_kept(self) -> None:
        metadata = _extract(
            '<html><head><link rel="canonical" href="https://example.com/post-canonical" />'
            "</head><body></body></html>",
            url="https://example.com/post?utm_source=x",
        )
        assert metadata.canonical_url == "https://example.com/post-canonical"

    def test_root_canonical_on_root_page_is_kept(self) -> None:
        metadata = _extract(
            '<html><head><link rel="canonical" href="https://example.com/" />'
            "</head><body></body></html>",
            url="https://example.com/",
        )
        assert metadata.canonical_url == "https://example.com/"
