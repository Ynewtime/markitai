#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright", "pillow", "pypdf"]
# ///
"""Retake the website's workbench screenshots in one command.

The home page (en, zh) and the serve guide show the workbench in eight files
under ``website/images/``: ``workbench[.zh][.dark].png`` and the same names in
``.webp``, 1600x1000 CSS pixels at 2x. This script makes all eight from the
``markitai serve`` this checkout builds. The pages reference them by relative
path, so the site build gives each a content-hashed URL
(``/assets/workbench.zh.dark.<hash>.webp``): a retaken shot gets a new URL and
no browser or CDN keeps showing the old one.

The shot: a one-page PDF report open in the centred window over the
Conversions list, on Compare, its page on the left and its Markdown on the
right. By default the diagram is hovered so both sides outline it, each pane
from its top (the Markdown from the diagram's section heading when the
diagram would not show otherwise). The report
is printed by Playwright's Chromium from text in this script, taken from the
repository's own documentation (MIT), in the shot's language with type large
enough to read at fit width:

- ``overview`` (default): "Files and URLs to Markdown", from the
  README and the home page: the presets table, a diagram of the conversion
  core, what it converts
- ``performance``: "Conversion performance", from website/guide/performance.md
  (zh: website/zh/guide/performance.md): the recorded results table, a bar
  chart of the 1.2.0 speedups, how it was measured

``--highlight`` hovers another block (``table``, ``heading``) or none.

For each language the script:

1. Gets the other inputs into ``--inputs`` (kept between runs; a file already
   there is reused). All are safe to publish:

   - ``us-state-population-2024.xlsx``: U.S. Census Bureau, NST-EST2024-POP,
     a U.S. government work (17 U.S.C. 105). Its author fields, which name a
     Bureau employee, are blanked; the data is unchanged.
   - ``alice-in-wonderland.epub``: Lewis Carroll, Project Gutenberg eBook #11,
     public domain in the USA.
   - ``getting-started.docx`` and ``markitai-overview.pptx``: this
     repository's own getting-started guide and README intro, built with
     pandoc (MIT).
   - https://en.wikipedia.org/wiki/Markdown, fetched by the server (CC BY-SA
     4.0): only its row's timing shows behind the window, none of its text.

2. Starts ``markitai serve --no-auth`` on a free loopback port with a
   throwaway HOME and MARKITAI_HOME and no other environment, so no
   configuration, model or key of yours is read: nothing calls an LLM.
3. Converts the report and the inputs as one submission ("Reading list",
   Minimal) and shoots the light and dark frames.

Other compositions, for another page sentence: ``compare-download`` (the
same with the header's Download menu open), ``preview`` and
``preview-download`` (the rendered Markdown instead of Compare).

Usage (needs pandoc, network access and Playwright's Chromium:
``uv run --with playwright playwright install chromium``; oxipng, when on PATH,
makes the PNGs smaller, and cwebp, when on PATH, keeps coloured text sharp in
the WebPs)::

    cargo build --release --locked -p markitai-cli
    uv run scripts/website_shots.py                  # into website/images
    uv run scripts/website_shots.py --out .local/shots --document performance --highlight table

``--markitai`` names another binary (default: target/release/markitai, else
target/debug/markitai). Run it from a clean checkout and a binary built from
it: the sidebar shows the commit, with "-dirty" for uncommitted changes to
tracked files (the script refuses to shoot then, and when the binary was built
from another commit). Then look at the eight files, and update the alt texts in
website/index.md, website/zh/index.md and website/{,zh/}guide/serve.md if the
composition changed, and the width and height in both index.md files if the
size did.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OTHERS = [
    "getting-started.docx",
    "markitai-overview.pptx",
    "us-state-population-2024.xlsx",
    "alice-in-wonderland.epub",
]
URLS = ["https://en.wikipedia.org/wiki/Markdown"]
LABEL = {"en": "Reading list", "zh": "阅读清单"}
DOWNLOADS = {
    "us-state-population-2024.xlsx": "https://www2.census.gov/programs-surveys/popest/"
    "tables/2020-2024/state/totals/NST-EST2024-POP.xlsx",
    "alice-in-wonderland.epub": "https://www.gutenberg.org/ebooks/11.epub3.images",
}
# The deck's source: the README's own intro, feature list and install lines
DECK = """\
---
title: Markitai
subtitle: Opinionated Markdown converter with native LLM enhancement support
---

# What it converts

- DOCX, PPTX, XLSX, PDF, EPUB, EML, TXT, MD, images (JPG/PNG/WebP), and URLs → clean Markdown
- Legacy `.doc` / `.ppt` via the `legacy` extra
- Scanned PDFs and images via local RapidOCR, or `--ocr --llm` for VLM-OCR

# LLM enhancement

- Format cleaning, frontmatter metadata, and vision analysis of embedded images
- All through litellm, so any provider works: OpenAI, Anthropic, Gemini, local CLIs, and more

# Batch processing and the web

- Concurrent conversion with progress display and `--resume` for interrupted jobs
- Static HTTP with cache revalidation, or Playwright rendering for JS-heavy pages
- A local web workspace: upload files or folders, submit URLs, compare results, retry failures

# Install

- Guided installer: `curl -fsSL https://markitai.dev/setup.sh | sh`
- Or `uv tool install markitai`, then `markitai doctor` and `markitai init`
- Docs: https://markitai.dev
"""
# The reports, from the documentation, written around what PDF extraction
# gets wrong today: no "a, b" in a table cell (the space after the comma is
# dropped), no bulleted list or close paragraphs (they merge into one), no
# line that starts with a short word and a space (read as a bullet), and
# CJK lines short enough not to wrap (a wrapped line comes back with a space,
# a wrapped heading split in two).
REPORTS: dict[str, dict[str, dict[str, Any]]] = {
    "performance": {
        "en": {
            "file": "conversion-performance.pdf",
            "title": "Conversion performance",
            "lead": "Local file conversion loads only the backend a format needs. "
            "An audit on macOS measured a 6.01× composite speedup against a "
            "frozen pre-optimization snapshot.",
            "table_heading": "Recorded results",
            "table": [
                ("Equally weighted workload", "Speedup"),
                ("Five URL pages over loopback HTTP", "5.83×"),
                ("Office and PDF cold CLI runs", "5.61×"),
                ("30-file TXT/HTML/DOCX batch", "6.64×"),
            ],
            "figure_heading": "1.2.0 against 1.1.0",
            "bars": [
                ("One 300-page text PDF", "4.1×", 4.1),
                ("40 PDFs of 5 pages", "4.1×", 4.1),
                ("240 mixed files", "3.3×", 3.3),
                ("30 documents with --llm", "1.5×", 1.5),
            ],
            "caption": "Figure 1. Cold CLI runs on an 18-core Mac, median of three",
            "notes_heading": "How it was measured",
            "note": "Model enrichment, remote fetching and the cache off, over 49 "
            "fixed cases with three fresh processes each.",
        },
        "zh": {
            "file": "转换性能.pdf",
            "title": "转换性能",
            # One line: a CJK line break comes back as a space
            "lead": "在 macOS 上相对冻结的优化前源码快照，综合提速 6.01×。",
            "table_heading": "实测结果",
            "table": [
                ("等权负载", "提速"),
                ("通过本地回环 HTTP 获取的五个 URL 页面", "5.83×"),
                ("Office 与 PDF 的冷 CLI 转换", "5.61×"),
                ("30 个 TXT/HTML/DOCX 文件的批处理", "6.64×"),
            ],
            "figure_heading": "1.2.0 对比 1.1.0",
            "bars": [
                ("一个 300 页的纯文本 PDF", "4.1×", 4.1),
                ("40 个 5 页的 PDF", "4.1×", 4.1),
                ("240 个混合文件", "3.3×", 3.3),
                ("30 个文档，开启 --llm", "1.5×", 1.5),
            ],
            "caption": "图 1. 18 核 Mac 上的冷启动 CLI 运行，三次取中位数",
            "notes_heading": "测量方法",
            "note": "固定样例共 49 个，关闭模型增强、远程抓取和缓存。",
        },
    },
    "overview": {
        "en": {
            "file": "markitai-at-a-glance.pdf",
            # One line, so the figure shows with the page at its top
            "title": "Files and URLs to Markdown",
            "lead": "Convert documents, images, and web pages with one command. "
            "LLM enhancement stays optional.",
            "table_heading": "Presets",
            "table": [
                ("Preset", "What runs"),
                ("minimal", "Conversion only"),
                ("standard", "LLM cleanup with image alt text and descriptions"),
                ("rich", "standard with page screenshots"),
            ],
            "figure_heading": "One conversion core",
            "diagram": [
                ("Files and URLs", "PDF · Office · EPUB · web"),
                ("markitai", "CLI · serve · MCP"),
                ("Markdown", ".md · .json"),
            ],
            "caption": "Figure 1. The CLI, the web workspace and the MCP server "
            "share one pipeline",
            "notes_heading": "What it converts",
            "note": "DOCX, PPTX, XLSX, PDF, EPUB, HTML, images with local OCR, "
            "and web pages, static or rendered in a browser.",
            # Room below the figure, so the Markdown can scroll its heading up
            "more_heading": "Learn more",
            "more": "Guides for the CLI, the web workspace and the Python API "
            "are at markitai.dev.",
            "license_heading": "License",
            "license": "Markitai is open source under the MIT license.",
        },
        "zh": {
            "file": "markitai-简介.pdf",
            # One line each: a wrapped CJK heading comes back split, a
            # wrapped CJK line with a space
            "title": "文件和网页转成 Markdown",
            "lead": "文档、图片和网页都能转，LLM 增强按需开启。",
            "table_heading": "预设",
            "table": [
                ("预设", "运行内容"),
                ("minimal", "只做转换"),
                ("standard", "LLM 清理，并生成图片替代文本和描述"),
                ("rich", "在 standard 之上加页面截图"),
            ],
            "figure_heading": "同一套转换内核",
            "diagram": [
                ("文件和网页", "PDF · Office · EPUB · 网页"),
                ("markitai", "CLI · serve · MCP"),
                ("Markdown", ".md · .json"),
            ],
            "caption": "图 1. 命令行、网页工作台和 MCP 服务共用一条流程",
            "notes_heading": "能转什么",
            "note": "Office、PDF、EPUB、HTML、图片和网页，扫描页可用本地 OCR。",
            "more_heading": "了解更多",
            "more": "命令行、网页工作台和 Python API 的指南见 markitai.dev。",
            "license_heading": "许可",
            "license": "Markitai 以 MIT 许可证开源。",
        },
    },
}
REPORT_CSS = """
@font-face { font-family: ReportSans; font-weight: 100 900;
  src: url(data:font/woff2;base64,%(inter)s) format('woff2'); }
@page { size: Letter; margin: 0.6in 0.85in; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: ReportSans, 'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei',
  sans-serif; color: #1d2433; font-size: 15pt; line-height: 1.45; margin: 0; }
h1 { font-size: 30pt; line-height: 1.15; margin: 0 0 10pt; color: #0f172a; }
h2 { font-size: 19pt; margin: 18pt 0 8pt; color: #0f172a; }
p { margin: 0 0 8pt; }
table { border-collapse: collapse; width: 100%%; font-size: 14pt; }
th, td { border: 1pt solid #9aa7bb; padding: 5pt 9pt; text-align: left; }
th { background: #e8eef7; }
td + td, th + th { text-align: right; }
.wide td + td, .wide th + th { text-align: left; }
figure { margin: 6pt 0 0; }
figure img { width: 100%%; display: block; }
figcaption { font-size: 12pt; color: #5b6b84; margin-top: 4pt; }
"""
FIGURE_FONT = "font-family: ReportSans, 'Noto Sans CJK SC', 'PingFang SC', sans-serif"
COMPOSITIONS = ("compare", "compare-download", "preview", "preview-download")
HIGHLIGHTS = ("table", "image", "heading", "none")
# The window is at most 1080 px wide and 88% of the height (880 px here):
# Compare shows the whole page and the Markdown down to the figure
WIDTH, HEIGHT = 1600, 1000
WEBP_QUALITY = 82
# Compare with a block outlined: each pane from its top when the block shows
# there, else the Markdown with the block's section heading at its top edge
# (no cut-off row above it). True when it scrolled either pane.
FRAME_PANES = """() => {
  const pane = document.querySelector('#cmp-pane-rendered');
  const original = document.querySelector('.orig-scroll');
  const linked = pane.querySelector('.is-linked');
  const mark = document.querySelector('.pdfv-page .omark.is-on');
  if (!linked || !mark) return false;
  const scroll = (box, top) => {
    const before = box.scrollTop;
    box.scrollTop = Math.max(0, Math.round(top));
    return Math.abs(box.scrollTop - before) > 2;
  };
  const at = (el, box) =>
    el.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
  let moved = false;
  const markBottom = at(mark, original) + mark.getBoundingClientRect().height;
  if (markBottom + 8 <= original.clientHeight) moved = scroll(original, 0) || moved;
  const headings = [...pane.querySelectorAll('.doc-body :is(h1, h2, h3, h4, h5, h6)')];
  const heading = linked.matches('h1, h2, h3, h4, h5, h6') ? linked : headings
    .filter(h => h.compareDocumentPosition(linked) & Node.DOCUMENT_POSITION_FOLLOWING)
    .pop();
  const linkedBottom = at(linked, pane) + linked.getBoundingClientRect().height;
  if (linkedBottom + 16 <= pane.clientHeight) return scroll(pane, 0) || moved;
  let top = heading ? at(heading, pane) - 16 : at(linked, pane) - 16;
  if (linkedBottom + 16 - top > pane.clientHeight) top = at(linked, pane) - 16;
  return scroll(pane, top) || moved;
}"""
# How far below the Markdown pane's top edge the first heading in view sits
# (none when the pane shows its top)
HEADING_GAP = """() => {
  const pane = document.querySelector('#cmp-pane-rendered');
  if (pane.scrollTop < 1) return 0;
  const edge = pane.getBoundingClientRect().top;
  const heading = [...pane.querySelectorAll('.doc-body :is(h1, h2, h3, h4, h5, h6)')]
    .find(h => h.getBoundingClientRect().bottom > edge);
  return heading ? heading.getBoundingClientRect().top - edge : 0;
}"""


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "markitai-website-shots"}
    )
    with urllib.request.urlopen(request, timeout=60) as answer:
        return answer.read()


def _blank_xlsx_authors(data: bytes) -> bytes:
    """The workbook unchanged but for empty creator / last-modified-by fields."""
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data)) as source,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            body = source.read(info)
            if info.filename == "docProps/core.xml":
                text = body.decode("utf-8")
                for tag in ("dc:creator", "cp:lastModifiedBy"):
                    text = re.sub(
                        rf"<{tag}>.*?</{tag}>|<{tag}/>",
                        f"<{tag}></{tag}>",
                        text,
                        flags=re.DOTALL,
                    )
                body = text.encode("utf-8")
            target.writestr(info, body)
    return out.getvalue()


def prepare_inputs(inputs: Path) -> None:
    inputs.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        if not (inputs / name).exists():
            data = _download(url)
            if name.endswith(".xlsx"):
                data = _blank_xlsx_authors(data)
            (inputs / name).write_bytes(data)
            print(f"fetched {name} ({len(data):,} bytes) from {url}")
    pandoc = shutil.which("pandoc")
    builds = {
        "getting-started.docx": ROOT / "website" / "guide" / "getting-started.md",
        "markitai-overview.pptx": None,
    }
    for name, source in builds.items():
        if (inputs / name).exists():
            continue
        if not pandoc:
            raise SystemExit(
                f"pandoc is needed to build {name} (or put one in {inputs})"
            )
        with tempfile.TemporaryDirectory() as tmp:
            if source is None:
                source = Path(tmp) / "markitai-overview.md"
                source.write_text(DECK, encoding="utf-8")
            subprocess.run([pandoc, str(source), "-o", str(inputs / name)], check=True)
        print(f"built {name} with pandoc")


def _svg_start(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="{FIGURE_FONT}"><rect width="{width}" height="{height}" fill="#f4f7fb"/>'
    )


def _bar_chart(bars: list[tuple[str, str, float]]) -> str:
    width, row, left = 900, 64, 400
    height = 20 + row * len(bars)
    scale = (width - left - 100) / max(value for _, _, value in bars)
    parts = [_svg_start(width, height)]
    for i, (label, shown, value) in enumerate(bars):
        y = 10 + i * row
        parts.append(
            f'<text x="{left - 18}" y="{y + 41}" text-anchor="end" font-size="25" '
            f'fill="#1d2433">{html.escape(label)}</text>'
            f'<rect x="{left}" y="{y + 14}" width="{value * scale:.0f}" height="36" '
            f'rx="5" fill="#2f6fdf"/>'
            f'<text x="{left + value * scale + 12:.0f}" y="{y + 42}" font-size="25" '
            f'font-weight="600" fill="#1d2433">{shown}</text>'
        )
    return "".join(parts) + "</svg>"


def _diagram(boxes: list[tuple[str, str]]) -> str:
    width, height, margin, box, arrow = 960, 170, 24, 262, 26
    gap = (width - 2 * margin - len(boxes) * box) / (len(boxes) - 1)
    parts = [_svg_start(width, height)]
    for i, (name, detail) in enumerate(boxes):
        x = margin + i * (box + gap)
        core = i == 1
        parts.append(
            f'<rect class="box" x="{x:.0f}" y="30" width="{box}" height="110" rx="14" '
            f'fill="{"#2f6fdf" if core else "#ffffff"}" stroke="#2f6fdf" stroke-width="3"/>'
            f'<text x="{x + box / 2:.0f}" y="78" text-anchor="middle" font-size="27" '
            f'font-weight="700" fill="{"#ffffff" if core else "#0f172a"}">{html.escape(name)}</text>'
            f'<text x="{x + box / 2:.0f}" y="112" text-anchor="middle" font-size="17" '
            f'fill="{"#dbe7ff" if core else "#5b6b84"}">{html.escape(detail)}</text>'
        )
        if i:
            start, end = x - gap + 10, x - 10
            parts.append(
                f'<path d="M{start:.0f} 85 H{end - 4:.0f}" stroke="#5b6b84" stroke-width="4"/>'
                f'<path d="M{end - arrow / 2:.0f} 73 L{end:.0f} 85 L{end - arrow / 2:.0f} 97" '
                'fill="none" stroke="#5b6b84" stroke-width="4"/>'
            )
    return "".join(parts) + "</svg>"


# Every label inside its box (with room to spare) and every shape inside the
# figure: the empty list when the figure is whole
FIGURE_OVERFLOW = """svg => {
  const out = [];
  const view = svg.getBoundingClientRect();
  const boxes = [...svg.querySelectorAll('rect.box')].map(r => r.getBoundingClientRect());
  for (const shape of svg.querySelectorAll('rect, text, path')) {
    const b = shape.getBoundingClientRect();
    if (b.left < view.left || b.right > view.right) out.push(shape.outerHTML.slice(0, 80));
  }
  for (const label of svg.querySelectorAll('text')) {
    const b = label.getBoundingClientRect();
    const home = boxes.find(r => b.top >= r.top && b.bottom <= r.bottom
      && b.left < r.right && b.right > r.left);
    if (home && (b.left < home.left + 8 || b.right > home.right - 8))
      out.push(label.textContent);
  }
  return out;
}"""


def build_report(
    document: str, lang: str, target: Path, browser: Any, base: str
) -> None:
    """Print the one-page report: title, a table, a figure (a picture in the
    PDF, so the converter keeps it as one) and a closing paragraph, in the
    workbench's own Inter."""
    text = REPORTS[document][lang]
    with urllib.request.urlopen(f"{base}/ui/inter-latin-wght.woff2", timeout=30) as answer:
        inter = base64.b64encode(answer.read()).decode()
    css = REPORT_CSS % {"inter": inter}
    page = browser.new_page(device_scale_factor=2)
    figure = _bar_chart(text["bars"]) if "bars" in text else _diagram(text["diagram"])
    page.set_content(f"<style>{css}</style><div id=figure>{figure}</div>")
    clipped = page.locator("#figure svg").evaluate(FIGURE_OVERFLOW)
    if clipped:
        raise SystemExit(f"the {document} figure does not fit: {clipped}")
    picture = base64.b64encode(page.locator("#figure svg").screenshot()).decode()
    rows = "".join(
        "<tr>"
        + "".join(
            f"<{'th' if i == 0 else 'td'}>{html.escape(cell)}</{'th' if i == 0 else 'td'}>"
            for cell in row
        )
        + "</tr>"
        for i, row in enumerate(text["table"])
    )
    wide = ' class="wide"' if document == "overview" else ""
    page.set_content(
        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
        f"<title>{html.escape(text['title'])}</title><style>{css}</style></head><body>"
        f"<h1>{html.escape(text['title'])}</h1><p>{html.escape(text['lead'])}</p>"
        f"<h2>{html.escape(text['table_heading'])}</h2><table{wide}>{rows}</table>"
        f"<h2>{html.escape(text['figure_heading'])}</h2>"
        f'<figure><img alt="" src="data:image/png;base64,{picture}">'
        f"<figcaption>{html.escape(text['caption'])}</figcaption></figure>"
        f"<h2>{html.escape(text['notes_heading'])}</h2><p>{html.escape(text['note'])}</p>"
        + "".join(
            f"<h2>{html.escape(text[f'{key}_heading'])}</h2><p>{html.escape(text[key])}</p>"
            for key in ("more", "license")
            if key in text
        )
        + "</body></html>"
    )
    page.pdf(path=str(target), print_background=True, prefer_css_page_size=True)
    page.close()
    from pypdf import PdfReader

    pages = len(PdfReader(target).pages)
    if pages != 1:
        raise SystemExit(f"{target.name} has {pages} pages, not one")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(base: str, path: str) -> Any:
    with urllib.request.urlopen(f"{base}{path}", timeout=30) as answer:
        return json.loads(answer.read())


def start_server(
    markitai: Path, home: Path, log: Path
) -> tuple[subprocess.Popen[bytes], str]:
    (home / ".markitai").mkdir(parents=True)
    port = _free_port()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "MARKITAI_HOME": str(home / ".markitai"),
        "LANG": "C.UTF-8",
        "TMPDIR": str(home),
        "NO_PROXY": "127.0.0.1,localhost",
    }
    if os.environ.get("SYSTEMROOT"):
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    command = [
        str(markitai.resolve()),
        "serve",
        "--port",
        str(port),
        "--no-open",
        "--no-auth",
    ]
    server = subprocess.Popen(
        command,
        cwd=home,
        env=env,
        stdout=log.open("wb"),
        stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt",
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"markitai serve exited ({server.returncode}); see {log}")
        try:
            _get(base, "/api/capabilities")
            return server, base
        except OSError:
            time.sleep(0.5)
    stop_server(server)
    raise SystemExit(f"markitai serve did not answer in 90 s; see {log}")


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    if os.name == "nt":
        server.terminate()
    else:
        os.killpg(server.pid, signal.SIGINT)
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            server.kill()
        else:
            os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def convert(base: str, files: list[Path], label: str) -> None:
    boundary = uuid.uuid4().hex
    body = b""
    for path in files:
        body += (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
                f'filename="{path.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
            ).encode()
            + path.read_bytes()
            + b"\r\n"
        )
    fields = {
        "urls": json.dumps(URLS),
        "options": json.dumps({"preset": "minimal"}),
        "label": label,
    }
    for key, value in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
    body += f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{base}/api/jobs",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=60) as answer:
        job = json.loads(answer.read())["job_id"]
    deadline = time.monotonic() + 300
    while (snapshot := _get(base, f"/api/jobs/{job}"))["status"] == "running":
        if time.monotonic() > deadline:
            raise SystemExit("the conversion did not finish in 5 minutes")
        time.sleep(0.5)
    statuses = {item["name"]: item["status"] for item in snapshot["items"]}
    print(statuses)
    if set(statuses.values()) != {"done"}:
        raise SystemExit(f"not every item converted: {statuses}")
    lead = next(item for item in snapshot["items"] if item["name"] == files[0].name)
    if lead.get("warnings"):
        raise SystemExit(f"{files[0].name} has notices: {lead['warnings']}")


def _spaced(name: str) -> re.Pattern[str]:
    # The list draws a name in pieces (a long name is cut in the middle)
    return re.compile(r"\s*".join(re.escape(c) for c in name))


def _save(page: Any, png: Path, oxipng: str | None) -> list[Path]:
    from PIL import Image

    Image.open(io.BytesIO(page.screenshot())).convert("RGB").save(png, optimize=True)
    if oxipng:  # lossless, about a quarter smaller
        subprocess.run(
            [oxipng, "-o", "4", "--strip", "safe", "-q", str(png)], check=True
        )
    webp = png.with_suffix(".webp")
    cwebp = shutil.which("cwebp")
    if cwebp:  # sharp RGB to YUV: coloured text keeps its edges
        subprocess.run(
            [cwebp, "-q", str(WEBP_QUALITY), "-m", "6", "-sharp_yuv", "-quiet",
             str(png), "-o", str(webp)],
            check=True,
        )
    else:
        Image.open(png).save(webp, "WEBP", quality=WEBP_QUALITY, method=6)
    return [png, webp]


def shoot(
    base: str,
    browser: Any,
    out: Path,
    lang: str,
    lead: str,
    composition: str,
    highlight: str,
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    oxipng = shutil.which("oxipng")
    written: list[Path] = []
    for theme in ("light", "dark"):
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=2,
            color_scheme=theme,
            locale="zh-CN" if lang == "zh" else "en-US",
        )
        page = context.new_page()
        page.goto(base)
        page.evaluate(
            "([theme, lang]) => { localStorage.setItem('markitai.theme', theme); "
            "localStorage.setItem('markitai.lang', lang); "
            "localStorage.setItem('markitai.peek', 'window'); }",
            [theme, lang],
        )
        page.goto(f"{base}/jobs")
        page.get_by_role("treeitem", name=_spaced(lead)).first.click()
        page.locator(".peek").first.wait_for()
        page.wait_for_timeout(400)
        if composition.startswith("compare"):
            page.locator("#peek-tab-compare").click()
            original = page.locator(".pdfv-page img").first
            original.wait_for()
            page.wait_for_function(
                "img => img.complete && img.naturalWidth > 0",
                arg=original.element_handle(),
            )
            page.locator(".pdfv-page .omark").first.wait_for(state="attached")
            page.locator("#cmp-pane-rendered .doc-body").first.wait_for()
        else:
            page.locator("#peek-panel .doc-body").first.wait_for()
        page.wait_for_timeout(1200)
        if composition.startswith("compare") and highlight != "none":
            # Hovering a block's outline on the page outlines its Markdown
            # too, and the Markdown scrolls to center it. Then frame both
            # panes (FRAME_PANES) and hover again, until nothing moves.
            mark = page.locator(f".pdfv-page .omark.is-{highlight}").first
            for _ in range(5):
                mark.hover()
                page.wait_for_timeout(800)
                on = mark.evaluate("mark => mark.classList.contains('is-on')")
                if on and not page.evaluate(FRAME_PANES):
                    break
                page.wait_for_timeout(400)
            else:
                raise SystemExit(f"hovering the {highlight} did not outline it")
            page.locator("#cmp-pane-rendered .is-linked").first.wait_for()
            gap = page.evaluate(HEADING_GAP)
            if gap > 32:
                raise SystemExit(
                    f"the Markdown cannot scroll the {highlight}'s heading to the top "
                    f"({gap:.0f} px below it): the report needs more text after it"
                )
        else:
            page.mouse.move(1, HEIGHT - 1)
        if composition.endswith("-download"):
            page.locator(".peek-head .split-caret").click()
            page.locator(".peek-head .download-menu").wait_for()
        page.wait_for_timeout(600)
        suffix = ("" if lang == "en" else ".zh") + ("" if theme == "light" else ".dark")
        written += _save(page, out / f"workbench{suffix}.png", oxipng)
        context.close()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retake the website's workbench screenshots."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "website" / "images",
        help="where the eight files go",
    )
    parser.add_argument(
        "--markitai",
        type=Path,
        help="the markitai binary (default: target/release/markitai, else target/debug/markitai)",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=ROOT / ".local" / "website-shots",
        help="input cache (default .local/website-shots, git-ignored); delete a file to fetch it again",
    )
    parser.add_argument("--composition", choices=COMPOSITIONS, default="compare")
    parser.add_argument("--document", choices=sorted(REPORTS), default="overview")
    parser.add_argument(
        "--highlight",
        choices=HIGHLIGHTS,
        default="image",
        help="the block hovered in Compare",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="shoot even though the sidebar will show <commit>-dirty",
    )
    args = parser.parse_args()
    markitai = args.markitai or next(
        (
            path
            for kind in ("release", "debug")
            if (path := ROOT / "target" / kind / f"markitai{'.exe' if os.name == 'nt' else ''}").exists()
        ),
        None,
    )
    if markitai is None:
        raise SystemExit("build it first: cargo build --release --locked -p markitai-cli")
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    from playwright.sync_api import sync_playwright

    prepare_inputs(args.inputs)
    # The frames go to a staging folder and into --out only at the end, so
    # a run that stops halfway leaves the eight files as they were
    with (
        sync_playwright() as playwright,
        tempfile.TemporaryDirectory(
            prefix="markitai-shots-", dir=args.inputs
        ) as staging,
    ):
        browser = playwright.chromium.launch()
        try:
            for lang in ("en", "zh"):
                home = Path(staging) / lang
                home.mkdir()
                lead = home / REPORTS[args.document][lang]["file"]
                server, base = start_server(
                    markitai, home / "home", home / "serve.log"
                )
                try:
                    build = _get(base, "/api/capabilities").get("build", {})
                    print(
                        f"markitai serve {build.get('version')} "
                        f"({build.get('commit')}) at {base}"
                    )
                    # The sidebar shows the commit the binary was built from
                    if not args.allow_dirty:
                        if build.get("dirty"):
                            raise SystemExit(
                                "the binary was built from uncommitted changes (its sidebar "
                                "shows <commit>-dirty): commit, rebuild, or pass --allow-dirty"
                            )
                        if not head.startswith(build.get("commit") or "?"):
                            raise SystemExit(
                                f"{markitai} was built from {build.get('commit')}, not "
                                f"HEAD {head[:8]}: rebuild it, or pass --allow-dirty"
                            )
                    build_report(args.document, lang, lead, browser, base)
                    files = [lead] + [args.inputs / name for name in OTHERS]
                    convert(base, files, LABEL[lang])
                    shoot(
                        base,
                        browser,
                        Path(staging) / "out",
                        lang,
                        lead.name,
                        args.composition,
                        args.highlight,
                    )
                finally:
                    stop_server(server)
        finally:
            browser.close()
        args.out.mkdir(parents=True, exist_ok=True)
        for frame in sorted((Path(staging) / "out").iterdir()):
            shutil.copyfile(frame, args.out / frame.name)
            print(f"{args.out / frame.name} ({frame.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
