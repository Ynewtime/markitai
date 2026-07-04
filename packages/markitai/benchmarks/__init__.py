"""Dev-only conversion-quality benchmarks for markitai.

This directory is developer tooling and is NOT shipped in the wheel
(``[tool.hatch.build.targets.wheel]`` packages only ``src/markitai``).

See ``webextract_quality.py`` for the HTML -> Markdown quality benchmark
runner and ``scorer.py`` for the heuristic scorer.
"""
