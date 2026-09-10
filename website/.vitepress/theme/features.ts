/**
 * Homepage feature cards, shared by the English and Chinese home pages.
 *
 * One source for both locales: each entry carries its own `en` and `zh` copy
 * and its own SVG icon, so `index.md` and `zh/index.md` never drift apart and
 * an icon edit happens once.
 */

export interface HomeFeature {
  /** Inline SVG markup (stroke: currentColor), 24x24 viewBox. */
  icon: string
  en: { title: string; details: string }
  zh: { title: string; details: string }
}

const svg = (paths: string) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${paths}</svg>`

export const homeFeatures: HomeFeature[] = [
  {
    icon: svg(
      '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    ),
    en: {
      title: 'Multi-format Support',
      details:
        'Convert DOCX, PPTX, XLSX, PDF, EPUB, EML, TXT, MD, images (JPG/PNG/WebP) and URLs to Markdown. Legacy .doc/.ppt via the legacy extra.',
    },
    zh: {
      title: '多格式支持',
      details:
        '支持 DOCX、PPTX、XLSX、PDF、EPUB、EML、TXT、MD、图片（JPG/PNG/WebP）和 URL 转换为 Markdown；旧版 .doc/.ppt 需 legacy extra。',
    },
  },
  {
    icon: svg(
      '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
    ),
    en: {
      title: 'LLM Enhancement',
      details:
        'AI-powered format cleaning, metadata generation (frontmatter), and image analysis.',
    },
    zh: {
      title: 'LLM 增强',
      details: 'AI 驱动的格式清洗、元数据生成（frontmatter）和图片分析。',
    },
  },
  {
    icon: svg(
      '<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/><path d="m15 9 6-6"/>',
    ),
    en: {
      title: 'Batch Processing',
      details:
        'Concurrent conversion with progress display and resume capability for interrupted jobs.',
    },
    zh: {
      title: '批量处理',
      details: '并发转换，支持进度显示和断点恢复。',
    },
  },
  {
    icon: svg(
      '<path d="M15 3v4a2 2 0 0 0 2 2h4"/><path d="M12 17v-6"/><path d="M9.5 14.5 12 17l2.5-2.5"/><path d="M20 17.5a9 9 0 1 1-18 0V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2Z"/>',
    ),
    en: {
      title: 'OCR Recognition',
      details:
        'Extract text from scanned PDFs and images with local RapidOCR (markitai[ocr]), or let a vision model read the pages directly with --ocr --llm.',
    },
    zh: {
      title: 'OCR 识别',
      details:
        '通过可选的 markitai[ocr] extra 使用本地 RapidOCR，或用 --ocr --llm 让视觉模型直接读取页面图像（VLM-OCR）。',
    },
  },
  {
    icon: svg(
      '<path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8Z"/>',
    ),
    en: {
      title: 'MCP for AI Agents',
      details:
        'A bundled MCP server exposes convert_document, convert_url, batch_convert and job_status over stdio, so agents convert files without loading markitai into their context.',
    },
    zh: {
      title: '面向 AI Agent 的 MCP',
      details:
        '随包发布的 MCP 服务器通过 stdio 暴露 convert_document、convert_url、batch_convert 与 job_status，Agent 无需把 markitai 载入上下文即可转换文件。',
    },
  },
  {
    // Sixth card keeps the default theme's 3x2 grid: five cards fall into a
    // 4+1 row with one card rattling alone.
    icon: svg(
      '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
    ),
    en: {
      title: 'Output Profiles',
      details:
        'Shape the result for its consumer with --profile rag, obsidian or okf: visible assets, wikilinks or OKF frontmatter, without changing the conversion itself.',
    },
    zh: {
      title: '输出 Profile',
      details:
        '用 --profile rag、obsidian 或 okf 为下游消费者塑形输出：可见的 assets、wikilink 或 OKF frontmatter，且不改变转换本身。',
    },
  },
]
