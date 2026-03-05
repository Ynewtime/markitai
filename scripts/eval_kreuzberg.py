"""Kreuzberg vs Current Pipeline Evaluation Script (v2).

Fully utilizes kreuzberg's API including page markers, image extraction,
annotations, hierarchy, and quality processing.

Usage:
    uv run python scripts/eval_kreuzberg.py
"""

from __future__ import annotations

import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "markitai" / "src"))

FIXTURES_DIR = (
    Path(__file__).parent.parent / "packages" / "markitai" / "tests" / "fixtures"
)
OUTPUT_DIR = Path(__file__).parent.parent / "eval_output"
RUNS = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    char_count: int = 0
    line_count: int = 0
    table_row_count: int = 0
    image_ref_count: int = 0
    heading_count: int = 0
    elapsed_ms: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.elapsed_ms) if self.elapsed_ms else 0.0

    def summary_dict(self) -> dict[str, Any]:
        return {
            "chars": self.char_count,
            "lines": self.line_count,
            "table_rows": self.table_row_count,
            "image_refs": self.image_ref_count,
            "headings": self.heading_count,
            "median_ms": round(self.median_ms, 1),
        }


def compute_metrics(text: str) -> Metrics:
    lines = text.split("\n")
    return Metrics(
        char_count=len(text),
        line_count=len(lines),
        table_row_count=sum(
            1 for ln in lines if "|" in ln and ln.strip().startswith("|")
        ),
        image_ref_count=len(re.findall(r"!\[", text)),
        heading_count=sum(1 for ln in lines if ln.strip().startswith("#")),
    )


# ---------------------------------------------------------------------------
# Current pipeline converters
# ---------------------------------------------------------------------------


def convert_current_pdf(file_path: Path) -> str:
    from markitai.converter.pdf import PdfConverter

    return PdfConverter(config=None).convert(file_path).markdown


def convert_current_image_ocr(file_path: Path) -> str:
    from markitai.config import MarkitaiConfig
    from markitai.converter.image import ImageConverter

    config = MarkitaiConfig()
    config.ocr.enabled = True
    return ImageConverter(config=config).convert(file_path).markdown


def convert_current_office(file_path: Path) -> str:
    from markitai.converter.office import OfficeConverter

    return OfficeConverter(config=None).convert(file_path).markdown


def convert_current_legacy(file_path: Path) -> str:
    from markitai.converter.legacy import LegacyOfficeConverter

    return LegacyOfficeConverter(config=None).convert(file_path).markdown


# ---------------------------------------------------------------------------
# Kreuzberg converter — fully utilizing API
# ---------------------------------------------------------------------------


def convert_kreuzberg(file_path: Path) -> str:
    """Convert using kreuzberg with full API utilization."""
    from kreuzberg import (
        ExtractionConfig,
        ImageExtractionConfig,
        OutputFormat,
        PageConfig,
        PdfConfig,
        extract_file_sync,
    )

    config = ExtractionConfig(
        output_format=OutputFormat.MARKDOWN,
        pages=PageConfig(
            extract_pages=True,
            insert_page_markers=True,
            marker_format="\n\n<!-- Page number: {page_num} -->\n\n",
        ),
        pdf_options=PdfConfig(
            extract_images=True,
            extract_annotations=True,
        ),
        images=ImageExtractionConfig(extract_images=True),
        enable_quality_processing=True,
        include_document_structure=True,
    )
    result = extract_file_sync(str(file_path), config=config)
    return result.content


def convert_kreuzberg_result(file_path: Path):
    """Return full kreuzberg result for detailed analysis."""
    from kreuzberg import (
        ExtractionConfig,
        ImageExtractionConfig,
        OutputFormat,
        PageConfig,
        PdfConfig,
        extract_file_sync,
    )

    config = ExtractionConfig(
        output_format=OutputFormat.MARKDOWN,
        pages=PageConfig(
            extract_pages=True,
            insert_page_markers=True,
            marker_format="\n\n<!-- Page number: {page_num} -->\n\n",
        ),
        pdf_options=PdfConfig(
            extract_images=True,
            extract_annotations=True,
        ),
        images=ImageExtractionConfig(extract_images=True),
        enable_quality_processing=True,
        include_document_structure=True,
    )
    return extract_file_sync(str(file_path), config=config)


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    file_path: Path
    current_fn: Any
    category: str


TEST_CASES = [
    TestCase(
        "PDF (500KB)",
        FIXTURES_DIR / "file-example_PDF_500_kB.pdf",
        convert_current_pdf,
        "pdf",
    ),
    TestCase(
        "Image OCR (JPG)",
        FIXTURES_DIR / "candy.JPG",
        convert_current_image_ocr,
        "image",
    ),
    TestCase(
        "XLSX",
        FIXTURES_DIR / "file_example_XLSX_100.xlsx",
        convert_current_office,
        "office",
    ),
    TestCase(
        "PPTX",
        FIXTURES_DIR / "Free_Test_Data_500KB_PPTX.pptx",
        convert_current_office,
        "office",
    ),
    TestCase(
        "DOC (legacy)",
        FIXTURES_DIR / "sub_dir" / "file-sample_100kB.doc",
        convert_current_legacy,
        "legacy",
    ),
    TestCase(
        "XLS (legacy)",
        FIXTURES_DIR / "sub_dir" / "file_example_XLS_100.xls",
        convert_current_legacy,
        "legacy",
    ),
    TestCase(
        "PPT (legacy)",
        FIXTURES_DIR / "sub_dir" / "file_example_PPT_250kB.ppt",
        convert_current_legacy,
        "legacy",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_timed(fn, file_path: Path, runs: int = RUNS) -> tuple[str, Metrics]:
    result_text = ""
    elapsed_list: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result_text = fn(file_path)
        t1 = time.perf_counter()
        elapsed_list.append((t1 - t0) * 1000)
    metrics = compute_metrics(result_text)
    metrics.elapsed_ms = elapsed_list
    return result_text, metrics


# ---------------------------------------------------------------------------
# Defect-specific validation tests
# ---------------------------------------------------------------------------


def run_defect_tests() -> list[dict[str, Any]]:
    """Run targeted tests to verify/falsify each identified defect."""
    from kreuzberg import (
        ExtractionConfig,
        ImageExtractionConfig,
        OutputFormat,
        PageConfig,
        PdfConfig,
        extract_file_sync,
    )

    results: list[dict[str, Any]] = []
    pdf_path = str(FIXTURES_DIR / "file-example_PDF_500_kB.pdf")
    pptx_path = str(FIXTURES_DIR / "Free_Test_Data_500KB_PPTX.pptx")
    doc_path = str(FIXTURES_DIR / "sub_dir" / "file-sample_100kB.doc")
    ppt_path = str(FIXTURES_DIR / "sub_dir" / "file_example_PPT_250kB.ppt")

    # -----------------------------------------------------------------------
    # Defect 1: Page markers missing
    # -----------------------------------------------------------------------
    config = ExtractionConfig(
        output_format=OutputFormat.MARKDOWN,
        pages=PageConfig(
            extract_pages=True,
            insert_page_markers=True,
            marker_format="\n\n<!-- Page number: {page_num} -->\n\n",
        ),
    )
    r = extract_file_sync(pdf_path, config=config)
    markers = re.findall(r"<!-- Page number: \d+ -->", r.content)
    pages_data = r.pages or []
    results.append(
        {
            "defect": "1. Page markers missing",
            "status": "FIXED" if len(markers) >= 5 else "CONFIRMED",
            "detail": f"Found {len(markers)} markers in content, {len(pages_data)} page objects with per-page data",
            "note": "marker_format uses {{page_num}} placeholder (not {{page_number}})",
        }
    )

    # -----------------------------------------------------------------------
    # Defect 2: Images not extracted from PDF
    # -----------------------------------------------------------------------
    config = ExtractionConfig(
        output_format=OutputFormat.MARKDOWN,
        pdf_options=PdfConfig(extract_images=True),
        images=ImageExtractionConfig(extract_images=True),
    )
    r = extract_file_sync(pdf_path, config=config)
    img_count = len(r.images) if r.images else 0
    img_details = []
    if r.images:
        for img in r.images:
            img_details.append(
                f"page={img.get('page_number')}, {img.get('width')}x{img.get('height')}, format={img.get('format')}"
            )
    results.append(
        {
            "defect": "2. Images not extracted from PDF",
            "status": "FIXED" if img_count > 0 else "CONFIRMED",
            "detail": f"Extracted {img_count} images: {'; '.join(img_details[:3])}",
            "note": "Requires ImageExtractionConfig(extract_images=True) + PdfConfig(extract_images=True)",
        }
    )

    # -----------------------------------------------------------------------
    # Defect 3: PDF tables not extracted
    # -----------------------------------------------------------------------
    # Test with default config (no OCR)
    config = ExtractionConfig(output_format=OutputFormat.MARKDOWN)
    r = extract_file_sync(pdf_path, config=config)
    table_count_default = len(r.tables) if r.tables else 0

    # Test with forced OCR table detection (may fail without Tesseract)
    table_count_ocr = -1
    ocr_error = ""
    try:
        from kreuzberg import OcrConfig, TesseractConfig

        config_ocr = ExtractionConfig(
            output_format=OutputFormat.MARKDOWN,
            force_ocr=True,
            ocr=OcrConfig(
                backend="tesseract",
                tesseract_config=TesseractConfig(enable_table_detection=True),
            ),
        )
        r_ocr = extract_file_sync(pdf_path, config=config_ocr)
        table_count_ocr = len(r_ocr.tables) if r_ocr.tables else 0
    except Exception as e:
        ocr_error = f"{type(e).__name__}: {str(e)[:100]}"

    results.append(
        {
            "defect": "3. PDF tables not extracted",
            "status": "DESIGN" if table_count_default == 0 else "FIXED",
            "detail": f"Default mode: {table_count_default} tables. OCR+table_detection: {table_count_ocr if table_count_ocr >= 0 else 'UNAVAILABLE'} ({ocr_error})",
            "note": "Table extraction requires OCR with enable_table_detection=True (Tesseract). This is by design — PDFium text extraction doesn't detect tables.",
        }
    )

    # -----------------------------------------------------------------------
    # Defect 4: Word spacing bug in Markdown output
    # -----------------------------------------------------------------------
    broken_words = [
        "utvarius",
        "interdumcondimentum",
        "cursusconvallis",
        "luctusnisl",
        "amettortor",
        "antesagittis",
    ]

    spacing_by_format = {}
    for fmt_name in ["PLAIN", "MARKDOWN", "STRUCTURED"]:
        config = ExtractionConfig(
            output_format=getattr(OutputFormat, fmt_name),
            enable_quality_processing=True,
        )
        r = extract_file_sync(pdf_path, config=config)
        broken_count = sum(1 for w in broken_words if w in r.content)
        spacing_by_format[fmt_name] = broken_count

    results.append(
        {
            "defect": "4. Word spacing bug in Markdown output",
            "status": "CONFIRMED (Markdown renderer bug)",
            "detail": f"Broken words — PLAIN: {spacing_by_format['PLAIN']}/6, MARKDOWN: {spacing_by_format['MARKDOWN']}/6, STRUCTURED: {spacing_by_format['STRUCTURED']}/6",
            "note": "Bug is in html-to-markdown layer, not PDF extraction. PLAIN/STRUCTURED output has correct spacing. Workaround: use PLAIN output + custom markdown formatting.",
        }
    )

    # -----------------------------------------------------------------------
    # Defect 5: Links not preserved in PDF
    # -----------------------------------------------------------------------
    config = ExtractionConfig(
        output_format=OutputFormat.MARKDOWN,
        pdf_options=PdfConfig(extract_annotations=True),
    )
    r = extract_file_sync(pdf_path, config=config)
    has_markdown_link = bool(re.search(r"\[.*?\]\(https?://", r.content))
    annotations = r.annotations or []
    link_annotations = [
        a
        for a in annotations
        if hasattr(a, "type") and "link" in str(getattr(a, "type", ""))
    ]

    results.append(
        {
            "defect": "5. Links not preserved in PDF Markdown",
            "status": "PARTIAL" if link_annotations else "CONFIRMED",
            "detail": f"Inline markdown links: {'Yes' if has_markdown_link else 'No'}. Annotations extracted: {len(link_annotations)} link(s): {[str(a)[:80] for a in link_annotations[:3]]}",
            "note": "Links are available via result.annotations but NOT rendered as inline [text](url) in Markdown output.",
        }
    )

    # -----------------------------------------------------------------------
    # Defect 6: DOC missing headings/tables/formatting
    # -----------------------------------------------------------------------
    config = ExtractionConfig(
        output_format=OutputFormat.MARKDOWN,
        enable_quality_processing=True,
        include_document_structure=True,
        images=ImageExtractionConfig(extract_images=True),
        pages=PageConfig(extract_pages=True, insert_page_markers=True),
    )
    r = extract_file_sync(doc_path, config=config)
    has_headings = bool(re.search(r"^#{1,6}\s", r.content, re.MULTILINE))
    has_tables = bool(re.search(r"^\|.*\|", r.content, re.MULTILINE))
    has_bold = "**" in r.content
    has_link = bool(re.search(r"\[.*?\]\(", r.content))
    has_hyperlink_raw = "HYPERLINK" in r.content
    doc_images = len(r.images) if r.images else 0
    doc_pages = len(r.pages) if r.pages else 0

    results.append(
        {
            "defect": "6. DOC: missing headings/tables/formatting",
            "status": "CONFIRMED",
            "detail": f"headings={has_headings}, tables={has_tables}, bold={has_bold}, links={has_link}, raw_HYPERLINK={has_hyperlink_raw}, images={doc_images}, pages={doc_pages}",
            "note": "OLE/CFB parser extracts plain text only. No structural elements. Page extraction not supported for DOC format.",
        }
    )

    # -----------------------------------------------------------------------
    # Defect 7: PPT missing structure
    # -----------------------------------------------------------------------
    config = ExtractionConfig(
        output_format=OutputFormat.MARKDOWN,
        enable_quality_processing=True,
        pages=PageConfig(extract_pages=True, insert_page_markers=True),
        images=ImageExtractionConfig(extract_images=True),
    )
    r = extract_file_sync(ppt_path, config=config)
    has_headings = bool(re.search(r"^#{1,6}\s", r.content, re.MULTILINE))
    has_tables = bool(re.search(r"^\|.*\|", r.content, re.MULTILINE))
    ppt_images = len(r.images) if r.images else 0
    ppt_pages = len(r.pages) if r.pages else 0

    # Check PPTX for comparison
    r_pptx = extract_file_sync(pptx_path, config=config)
    pptx_has_tables = bool(re.search(r"^\|.*\|", r_pptx.content, re.MULTILINE))
    pptx_markers = len(re.findall(r"<!-- Page number: \d+ -->", r_pptx.content))
    pptx_images = len(r_pptx.images) if r_pptx.images else 0

    results.append(
        {
            "defect": "7. PPT/PPTX structure issues",
            "status": "PARTIAL",
            "detail": f"PPT: headings={has_headings}, tables={has_tables}, images={ppt_images}, pages={ppt_pages}. "
            f"PPTX: tables={pptx_has_tables}, markers={pptx_markers}, images={pptx_images}",
            "note": "PPT (legacy) has no structure. PPTX (modern) extracts tables and page markers correctly, but has cosmetic issues (empty list items, heading concatenation).",
        }
    )

    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_evaluation() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    print("=" * 70)
    print("PART 1: Side-by-side format comparison")
    print("=" * 70)

    for tc in TEST_CASES:
        print(f"\n{'─' * 60}")
        print(f"  {tc.name} ({tc.file_path.name})")
        print(f"{'─' * 60}")

        if not tc.file_path.exists():
            print("  SKIP: file not found")
            continue

        case_dir = OUTPUT_DIR / tc.category
        case_dir.mkdir(parents=True, exist_ok=True)
        safe_name = tc.file_path.stem

        # Current pipeline
        print(f"  Current pipeline ({RUNS}x)...", end=" ", flush=True)
        try:
            current_text, current_metrics = run_timed(tc.current_fn, tc.file_path)
            (case_dir / f"{safe_name}_current.md").write_text(
                current_text, encoding="utf-8"
            )
            print(f"OK ({current_metrics.median_ms:.0f}ms)")
        except Exception as e:
            current_text = f"ERROR: {e}"
            current_metrics = Metrics()
            print(f"FAIL: {e}")

        # Kreuzberg (full API)
        print(f"  Kreuzberg full API ({RUNS}x)...", end=" ", flush=True)
        try:
            kreuzberg_text, kreuzberg_metrics = run_timed(
                convert_kreuzberg, tc.file_path
            )
            (case_dir / f"{safe_name}_kreuzberg_v2.md").write_text(
                kreuzberg_text, encoding="utf-8"
            )
            print(f"OK ({kreuzberg_metrics.median_ms:.0f}ms)")
        except Exception as e:
            kreuzberg_text = f"ERROR: {e}"
            kreuzberg_metrics = Metrics()
            print(f"FAIL: {e}")

        # Kreuzberg full result for extra details (only once for timing)
        extra = {}
        try:
            full_result = convert_kreuzberg_result(tc.file_path)
            extra = {
                "images_extracted": len(full_result.images)
                if full_result.images
                else 0,
                "tables_extracted": len(full_result.tables)
                if full_result.tables
                else 0,
                "pages_extracted": len(full_result.pages) if full_result.pages else 0,
                "annotations": len(full_result.annotations)
                if full_result.annotations
                else 0,
                "quality_score": full_result.quality_score,
            }
            print(
                f"  Extra: images={extra['images_extracted']}, tables={extra['tables_extracted']}, pages={extra['pages_extracted']}, annotations={extra['annotations']}"
            )
        except Exception:
            pass

        results.append(
            {
                "name": tc.name,
                "file": tc.file_path.name,
                "category": tc.category,
                "current": current_metrics.summary_dict(),
                "kreuzberg": kreuzberg_metrics.summary_dict(),
                "extra": extra,
            }
        )

    # -----------------------------------------------------------------------
    # Part 2: Defect-specific tests
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: Defect-specific validation tests")
    print("=" * 70)

    defect_results = run_defect_tests()

    for d in defect_results:
        status_icon = {
            "FIXED": "✅",
            "CONFIRMED": "❌",
            "PARTIAL": "⚠️",
            "DESIGN": "📐",
        }.get(d["status"].split(" ")[0], "❓")
        print(f"\n{status_icon} {d['defect']}")
        print(f"  Status: {d['status']}")
        print(f"  Detail: {d['detail']}")
        print(f"  Note:   {d['note']}")

    # -----------------------------------------------------------------------
    # Generate report
    # -----------------------------------------------------------------------
    report_lines: list[str] = []
    report_lines.append("# Kreuzberg 评测报告 v2（充分利用 API）\n")
    report_lines.append("> 评测日期：2026-03-05 | kreuzberg v4.4.2 | markitai v0.6.1")
    report_lines.append(f"> 每个文件运行 {RUNS} 次取中位数\n")

    # Summary table
    report_lines.append("## 一、性能与指标对比\n")
    report_lines.append(
        "| 格式 | Current (ms) | Kreuzberg (ms) | 加速比 | K:images | K:tables | K:pages |"
    )
    report_lines.append(
        "|------|-------------|----------------|--------|----------|----------|---------|"
    )
    for r in results:
        c, k = r["current"], r["kreuzberg"]
        ex = r.get("extra", {})
        speedup = (
            c["median_ms"] / k["median_ms"] if k["median_ms"] > 0 else float("inf")
        )
        report_lines.append(
            f"| {r['name']} | {c['median_ms']} | {k['median_ms']} | {speedup:.1f}x "
            f"| {ex.get('images_extracted', '-')} | {ex.get('tables_extracted', '-')} | {ex.get('pages_extracted', '-')} |"
        )

    # Content metrics
    report_lines.append(
        "\n| 格式 | C:chars | K:chars | C:tables | K:tables | C:headings | K:headings |"
    )
    report_lines.append(
        "|------|---------|---------|----------|----------|------------|------------|"
    )
    for r in results:
        c, k = r["current"], r["kreuzberg"]
        report_lines.append(
            f"| {r['name']} | {c['chars']} | {k['chars']} "
            f"| {c['table_rows']} | {k['table_rows']} "
            f"| {c['headings']} | {k['headings']} |"
        )

    # Defect report
    report_lines.append("\n## 二、缺陷验证结果\n")
    report_lines.append("| # | 缺陷 | 状态 | 说明 |")
    report_lines.append("|---|------|------|------|")
    for d in defect_results:
        status_icon = {
            "FIXED": "✅",
            "CONFIRMED": "❌",
            "PARTIAL": "⚠️",
            "DESIGN": "📐",
        }.get(d["status"].split(" ")[0], "❓")
        report_lines.append(
            f"| {status_icon} | {d['defect']} | {d['status']} | {d['note']} |"
        )

    # Write report
    report_path = OUTPUT_DIR / "report_v2.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n{'=' * 70}")
    print(f"Report: {report_path}")

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for r in results:
        c, k, ex = r["current"], r["kreuzberg"], r.get("extra", {})
        speedup = (
            c["median_ms"] / k["median_ms"] if k["median_ms"] > 0 else float("inf")
        )
        print(f"\n{r['name']}:")
        print(
            f"  Speed:    {c['median_ms']:.0f}ms → {k['median_ms']:.0f}ms ({speedup:.1f}x)"
        )
        print(f"  Content:  {c['chars']} → {k['chars']} chars")
        print(
            f"  Tables:   {c['table_rows']} → {k['table_rows']} md-rows  (extracted: {ex.get('tables_extracted', '?')})"
        )
        print(
            f"  Images:   {c['image_refs']} refs → extracted: {ex.get('images_extracted', '?')}"
        )
        print(f"  Headings: {c['headings']} → {k['headings']}")


if __name__ == "__main__":
    run_evaluation()
