"""Converting one format must not import the machinery of every other.

Importing all converters up front cost ~430ms per process and, through
``markitdown`` -> ``magika`` -> ``onnxruntime``, a native teardown that can
abort a finished run (see ``markitai.utils.shutdown``) — for converting a
``.txt``, which needs none of it.

The checks run in fresh interpreters: once a test session has imported a
converter for any reason, ``sys.modules`` can no longer answer the question.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from markitai.converter.base import _CONVERTER_MODULES, EXTENSION_MAP, FileFormat


def _modules_after(code: str) -> set[str]:
    """Top-level module names loaded by ``code`` in a fresh interpreter."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(code)
            + "\nimport sys"
            + "\nprint(' '.join(sorted({m.split('.')[0] for m in sys.modules})))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return set(proc.stdout.split())


class TestNothingLoadsEverything:
    def test_importing_the_package_pulls_in_no_converter(self) -> None:
        loaded = _modules_after("import markitai.converter")

        assert "markitdown" not in loaded
        assert "pymupdf4llm" not in loaded

    @pytest.mark.parametrize("suffix", [".txt", ".md"])
    def test_text_conversion_stays_clear_of_markitdown(self, suffix: str) -> None:
        """The path that needs the least must pay for the least."""
        loaded = _modules_after(
            f"from markitai.converter import get_converter\n"
            f"assert get_converter('a{suffix}') is not None"
        )

        assert "markitdown" not in loaded
        assert "magika" not in loaded
        assert "onnxruntime" not in loaded
        assert "pymupdf4llm" not in loaded

    def test_resolving_one_format_does_not_import_another(self) -> None:
        loaded = _modules_after(
            "from markitai.converter import get_converter\n"
            "assert get_converter('a.pdf') is not None"
        )

        assert "pymupdf4llm" in loaded  # the one it does need
        assert "markitdown" not in loaded


class TestEveryFormatStillResolves:
    """A missing entry in _CONVERTER_MODULES is a silently unsupported format."""

    def test_every_registered_module_answers_for_its_format(self) -> None:
        from markitai.converter.base import load_converter_class

        for fmt, module in sorted(_CONVERTER_MODULES.items(), key=lambda kv: kv[1]):
            cls = load_converter_class(fmt)
            assert cls is not None, f"{module} does not register {fmt}"
            assert fmt in cls.supported_formats

    def test_known_extensions_resolve_or_fall_through_to_kreuzberg(self) -> None:
        from markitai.converter.kreuzberg import KREUZBERG_FORMATS

        covered = set(_CONVERTER_MODULES) | set(KREUZBERG_FORMATS)
        uncovered = {
            fmt
            for fmt in EXTENSION_MAP.values()
            if fmt is not FileFormat.UNKNOWN and fmt not in covered
        }
        assert not uncovered, f"extensions map to formats nothing converts: {uncovered}"
