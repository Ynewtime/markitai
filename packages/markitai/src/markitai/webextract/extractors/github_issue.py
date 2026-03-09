from __future__ import annotations

from bs4 import BeautifulSoup, Tag


class GitHubIssueExtractor:
    """Extractor for GitHub issue or pull request pages."""

    name = "github_issue"

    def matches_url(self, url: str) -> bool:
        return "/issues/" in url or "/pull/" in url

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        header = soup.find(class_="gh-header-show")
        body = soup.find(class_="comment-body")
        if header and body:
            main = soup.new_tag("article")
            main.append(header)
            main.append(body)
            return main
        return None
