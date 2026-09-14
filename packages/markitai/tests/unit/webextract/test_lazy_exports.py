"""Office rendering must not initialize the unrelated web extraction pipeline."""

import os
import subprocess
import sys

import pytest


def isolated(code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "code",
    [
        "import markitai.webextract",
        'from markitai.webextract.markdownify_compat import CompatibleMarkdownConverter; assert CompatibleMarkdownConverter().convert("<p>text</p>").strip() == "text"',
        'from markitai.webextract import WebMetadata, coerce_source_frontmatter; assert coerce_source_frontmatter(WebMetadata(title="title"))["title"] == "title"',
    ],
)
def test_lightweight_exports_do_not_load_web_pipeline(code):
    isolated(
        code
        + '\nimport sys\nassert "markitai.webextract.pipeline" not in sys.modules\nassert "markitai.webextract.extractors" not in sys.modules'
    )


def test_lazy_extractor_retains_identity_signature_and_discoverability():
    isolated("""
import inspect, sys
import markitai.webextract as public
assert 'extract_web_content' in public.__all__
assert 'extract_web_content' in dir(public)
assert 'markitai.webextract.pipeline' not in sys.modules
from markitai.webextract import extract_web_content
from markitai.webextract.pipeline import extract_web_content as implementation
assert extract_web_content is implementation
assert public.extract_web_content is implementation
assert inspect.signature(extract_web_content) == inspect.signature(implementation)
result=extract_web_content('<article><h1>Title</h1><p>Complete 中文 content that survives extraction.</p></article>', 'https://example.com/article')
assert 'Complete 中文 content' in result.markdown
""")


def test_reload_preserves_reexport_behavior():
    isolated("""
import importlib
import markitai.webextract as public
from markitai.webextract import extract_web_content
from markitai.webextract import pipeline
sentinel=lambda *args: None
pipeline.extract_web_content=sentinel
importlib.reload(public)
assert public.extract_web_content is sentinel
try:
    public.not_a_public_export
except AttributeError:
    pass
else:
    raise AssertionError('Unknown exports must fail normally')
""")
