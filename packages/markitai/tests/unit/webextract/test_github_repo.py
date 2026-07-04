"""Tests for the GitHub repository home page extractor (rendered README)."""

from __future__ import annotations

from markitai.webextract.extractors.github_repo import GitHubRepoExtractor

_REPO_PAGE_HTML = """
<html>
<head>
  <title>GitHub - owner/repo: A tool that does useful things for people. · GitHub</title>
  <meta property="og:site_name" content="GitHub" />
</head>
<body>
<main>
  <table aria-labelledby="folders-and-files">
    <tr><td><a href="/owner/repo/tree/main/src">src</a></td>
        <td>Last commit message</td><td>Last commit date</td></tr>
    <tr><td><a href="/owner/repo/blob/main/README.md">README.md</a></td>
        <td>docs update</td><td>yesterday</td></tr>
  </table>
  <div id="readme">
    <article class="markdown-body entry-content container-lg" itemprop="text">
      <h1>Repo</h1>
      <p>This is the README body with the actual documentation content that
      users came to read, explaining what the project does in detail. It
      describes the motivation behind the tool, walks through the main
      concepts one by one, and links to further reading for anyone who wants
      to dig deeper into the architecture and the trade-offs involved.</p>
      <h2>Install</h2>
      <p>Run the installer and follow the printed instructions carefully.
      The installer checks your environment first, downloads the required
      dependencies, and finally verifies the installation by running a quick
      self test so you know everything works before you start using it.</p>
    </article>
  </div>
  <div class="Layout-sidebar">
    <h2>About</h2>
    <p>A tool that does useful things for people.</p>
    <a href="/owner/repo/stargazers"><strong>49.9k</strong> stars</a>
    <a href="/owner/repo/forks"><strong>4k</strong> forks</a>
    <h2>Releases 7</h2>
    <h2>Languages</h2>
    <ul><li><a href="/owner/repo/search?l=python">Python</a></li></ul>
  </div>
</main>
</body>
</html>
"""


class TestMatchesUrl:
    def test_matches_repo_root(self) -> None:
        extractor = GitHubRepoExtractor()
        assert extractor.matches_url("https://github.com/Panniantong/Agent-Reach")
        assert extractor.matches_url("https://github.com/owner/repo/")
        assert extractor.matches_url("https://www.github.com/owner/repo")

    def test_rejects_non_repo_paths(self) -> None:
        extractor = GitHubRepoExtractor()
        assert not extractor.matches_url("https://github.com/owner/repo/issues/56")
        assert not extractor.matches_url("https://github.com/owner/repo/tree/main")
        assert not extractor.matches_url("https://github.com/owner")
        assert not extractor.matches_url("https://github.com/")
        assert not extractor.matches_url("https://example.com/owner/repo")

    def test_rejects_reserved_product_pages(self) -> None:
        extractor = GitHubRepoExtractor()
        assert not extractor.matches_url("https://github.com/topics/python")
        assert not extractor.matches_url("https://github.com/orgs/anthropics")
        assert not extractor.matches_url("https://github.com/sponsors/someone")

    def test_registry_routes_issue_urls_to_thread_extractor(self) -> None:
        """Repo extractor must not shadow the issue/PR thread extractor."""
        from markitai.webextract.extractors.registry import find_extractor

        issue = find_extractor("https://github.com/org/repo/issues/56")
        assert issue is not None and issue.name == "github_thread"

        repo = find_extractor("https://github.com/org/repo")
        assert repo is not None and repo.name == "github_repo"


class TestExtractRoot:
    def test_selects_readme_article(self) -> None:
        from markitai.webextract.dom import parse_html

        root = GitHubRepoExtractor().extract_root(parse_html(_REPO_PAGE_HTML))
        assert root is not None
        text = root.get_text(" ", strip=True)
        assert "README body" in text
        assert "Last commit message" not in text
        assert "49.9k" not in text

    def test_returns_none_without_readme(self) -> None:
        from markitai.webextract.dom import parse_html

        html = "<html><body><main><p>No readme here.</p></main></body></html>"
        assert GitHubRepoExtractor().extract_root(parse_html(html)) is None


class TestFullPipeline:
    def test_repo_page_extraction_keeps_readme_and_drops_chrome(self) -> None:
        from markitai.webextract import extract_web_content

        result = extract_web_content(_REPO_PAGE_HTML, "https://github.com/owner/repo")

        assert result.info is not None
        assert result.info.extractor_name == "github_repo"
        assert "README body" in result.markdown
        assert "Install" in result.markdown
        # Chrome must be gone: file tree, About sidebar, star counts
        assert "Last commit message" not in result.markdown
        assert "49.9k" not in result.markdown
        assert "Releases" not in result.markdown

    def test_repo_title_is_full_and_site_suffix_free(self) -> None:
        from markitai.webextract import extract_web_content

        result = extract_web_content(_REPO_PAGE_HTML, "https://github.com/owner/repo")

        assert (
            result.metadata.title
            == "owner/repo: A tool that does useful things for people."
        )
        assert result.metadata.site == "GitHub"
