"""Performance contracts tested in fresh interpreters, without clock thresholds.

These assert real output plus absence of unrelated expensive subsystems. Time
and memory improvements are measured separately by the fixed benchmark suite.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_TIMEOUT_S = 30
# A script still running 5 s before the timeout prints every thread's stack
# and exits, so a hang fails with where it hung (macOS CI timed out with no
# output at all) instead of a bare TimeoutExpired
_WATCHDOG = (
    "import faulthandler\n"
    f"faulthandler.dump_traceback_later({_TIMEOUT_S - 5}, exit=True)\n"
)


def run_isolated(script: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _WATCHDOG + script, *args],
        check=False,
        capture_output=True,
        # UTF-8 as the CLI writes it; text=True decodes ANSI on Windows
        encoding="utf-8",
        errors="replace",
        timeout=_TIMEOUT_S,
        env={**os.environ, "MARKITAI_NO_REMOTE_FETCH": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_html_keeps_file_detectors_and_office_engines_unloaded():
    run_isolated("""
import sys
from markitai.webextract import extract_web_content
result = extract_web_content(
    '<article><h1>Complete note</h1><p>A useful note with '
    '<strong>important information</strong> and '
    '<a href="https://example.com/source">its source</a>.</p></article>',
    'https://example.com/note',
)
assert '**important information**' in result.markdown, result.markdown
assert '[its source](https://example.com/source)' in result.markdown
unexpected = set(sys.modules) & {'markitdown', 'magika', 'onnxruntime', 'pandas', 'openpyxl', 'pptx', 'litellm'}
assert not unexpected, unexpected
""")


def test_cli_without_llm_does_not_initialize_llm_during_cleanup(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("A useful complete note for the CLI performance contract.")
    run_isolated(
        """
import sys
from click.testing import CliRunner
from markitai.cli.main import app
result = CliRunner().invoke(app, [sys.argv[1], '--preset', 'minimal', '--no-cache', '--no-remote-fetch'])
assert result.exit_code == 0, (result.output, result.exception)
assert 'A useful complete note' in result.output, result.output
assert 'litellm' not in sys.modules, 'Teardown imported an unused LLM provider'
assert 'pymupdf' not in sys.modules, 'Noise suppression imported an unused PDF engine'
assert 'markitai.webextract' not in sys.modules, 'File setup imported webpage extraction'
assert 'markitai.image' not in sys.modules, 'Text conversion initialized image processing'
""",
        str(path),
    )


def test_url_without_llm_does_not_load_the_llm_stack(tmp_path):
    """A plain URL conversion never imports LiteLLM.

    Its import takes about a second on Linux and several on Windows; it once
    crept into every single-URL run through the spend tracking.
    """
    page = tmp_path / "page.html"
    page.write_text(
        "<html><head><title>Note</title></head><body><article><h1>Note</h1>"
        "<p>A complete page for the URL performance contract. "
        * 20
        + "</p></article></body></html>",
        encoding="utf-8",
    )
    run_isolated(
        """
import functools, socketserver, sys, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from click.testing import CliRunner
from markitai.cli.main import app

class Server(ThreadingHTTPServer):
    # As tests.local_http.LocalHTTPServer: HTTPServer's bind looks up the
    # address's host name, which took over 25 s on macOS runners
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

def step(name):
    print(name, file=sys.stderr, flush=True)

step("imported")
handler = functools.partial(SimpleHTTPRequestHandler, directory=sys.argv[1])
server = Server(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}/page.html"
step("serving")
result = CliRunner().invoke(app, [url, '-s', 'static', '--preset', 'minimal', '--no-cache', '--no-remote-fetch'])
step("converted")
server.shutdown()
step("stopped")
assert result.exit_code == 0, (result.output, result.exception)
assert 'A complete page' in result.output, result.output
assert 'litellm' not in sys.modules, 'A URL without LLM imported the LLM stack'
""",
        str(tmp_path),
    )


def test_html_file_converter_does_not_load_office_or_detection_models(tmp_path):
    path = tmp_path / "note.html"
    path.write_text(
        "<article><p>A complete note with useful information.</p></article>"
    )
    run_isolated(
        """
import sys
from pathlib import Path
from markitai.converter.base import get_converter
converter = get_converter(Path(sys.argv[1]))
result = converter.convert(Path(sys.argv[1]))
assert 'A complete note with useful information.' in result.markdown
unexpected = set(sys.modules) & {'markitdown', 'magika', 'onnxruntime', 'pandas', 'openpyxl', 'pptx'}
assert not unexpected, unexpected
""",
        str(path),
    )


@pytest.mark.parametrize(
    "extension,content,expected",
    [
        ("csv", "Name,Value\nExample,42\n", "| Example | 42 |"),
        (
            "ipynb",
            '{"cells":[{"cell_type":"markdown","source":["# Useful notebook"]}],"metadata":{},"nbformat":4,"nbformat_minor":5}',
            "# Useful notebook",
        ),
    ],
)
def test_structured_text_does_not_initialize_binary_file_engines(
    tmp_path, extension, content, expected
):
    path = tmp_path / f"sample.{extension}"
    path.write_text(content)
    run_isolated(
        """
import sys
from pathlib import Path
from markitai.converter.base import get_converter
result = get_converter(Path(sys.argv[1])).convert(Path(sys.argv[1]))
assert sys.argv[2] in result.markdown, result.markdown
unexpected = set(sys.modules) & {'markitdown', 'magika', 'onnxruntime', 'pandas', 'openpyxl', 'pptx'}
assert not unexpected, unexpected
""",
        str(path),
        expected,
    )


def test_docx_without_equations_does_not_load_other_office_engines(fixtures_dir):
    run_isolated(
        """
import sys
from pathlib import Path
from markitai.converter.base import get_converter
result = get_converter(Path(sys.argv[1])).convert(Path(sys.argv[1]))
assert 'Markitai Snapshot Fixture' in result.markdown
assert '**bold**' in result.markdown and '*italic*' in result.markdown
assert 'markitai.image' not in sys.modules, 'Plain DOCX initialized image processing'
unexpected = set(sys.modules) & {'markitdown', 'magika', 'onnxruntime', 'pandas', 'openpyxl', 'pptx'}
assert not unexpected, unexpected
""",
        str(fixtures_dir / "sample.docx"),
    )


def test_xlsx_does_not_load_other_format_detectors(fixtures_dir):
    run_isolated(
        """
import sys
from pathlib import Path
from markitai.converter.base import get_converter
result = get_converter(Path(sys.argv[1])).convert(Path(sys.argv[1]))
assert len(result.markdown) > 100
unexpected = set(sys.modules) & {'markitdown', 'magika', 'onnxruntime', 'pptx'}
assert not unexpected, unexpected
""",
        str(fixtures_dir / "sample.xlsx"),
    )


def test_pptx_does_not_load_other_format_detectors(fixtures_dir):
    run_isolated(
        """
import sys
from pathlib import Path
from markitai.converter.base import get_converter
result = get_converter(Path(sys.argv[1])).convert(Path(sys.argv[1]))
assert '<!-- Slide number: 1 -->' in result.markdown
unexpected = set(sys.modules) & {'markitdown', 'magika', 'onnxruntime', 'pandas', 'openpyxl'}
assert not unexpected, unexpected
""",
        str(fixtures_dir / "sample.pptx"),
    )


def test_text_api_does_not_load_pdf_engine_to_suppress_its_logs(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("A complete note through the public API.")
    run_isolated(
        """
import sys
import markitai
result = markitai.convert(sys.argv[1], llm=False, ocr=False, screenshot=False, alt=False, desc=False)
assert result.markdown.strip() == 'A complete note through the public API.', result.markdown
assert 'pymupdf' not in sys.modules, 'Noise suppression imported an unused PDF engine'
""",
        str(path),
    )
