"""Compatibility shim: the renderer moved to markitai.webextract.html_to_markdown.

It always belonged to the webextract domain (it renders extracted HTML to
Markdown and depends on webextract constants); this module survives only to
keep the old import path working.
"""

from markitai.webextract.html_to_markdown import (
    WebExtractHtmlConverter as WebExtractHtmlConverter,
)
from markitai.webextract.html_to_markdown import (
    WebExtractMarkdownConverter as WebExtractMarkdownConverter,
)
