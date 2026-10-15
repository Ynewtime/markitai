// Conversion warnings, read as notices: one line per kind of problem with a
// count, instead of one line per occurrence. Backends word warnings in free
// English text, so kinds are recognized by shape; anything else is grouped by
// its template (names, numbers and quoted values blanked out) and shown with
// its first wording.

export type NoticeKind =
  | "no_model"
  | "alt_kept"
  | "cost_incomplete"
  | "json_resent"
  | "image_requests_split"
  | "resent"
  | "deployment_skipped"
  | "answer_rejected"
  | "pdf_scanned"
  | "pdf_ocr_pages"
  | "ocr_blank"
  | "other";

export interface NoticeEntry {
  itemKey: string;
  itemName: string;
  text: string;
}

export interface NoticeGroup {
  key: string;
  kind: NoticeKind;
  /** Occurrences, over every covered item. */
  count: number;
  /** Item keys the group speaks for, in first-seen order. */
  items: string[];
  names: string[];
  /** The first wording, and up to 50 for the details. */
  sample: string;
  samples: string[];
  /** Model names a cost or deployment notice is about. */
  models: string[];
  /** v1.16, PDF kinds of one document: its pages without readable text, as ranges, how many, and of how many. */
  pages: PageSet | null;
}

export interface PageSet {
  /** `1–12, 15, 20–31`. */
  ranges: string;
  count: number;
  /** The document's pages, when the warning says. */
  total: number | null;
}

const SHAPES: [NoticeKind, RegExp][] = [
  // Every model request failed: no `.llm.md` was written (contract v1.6, both backends word it so).
  ["no_model", /^No model succeeded for this document\b/],
  ["alt_kept", /^Image analysis failed for [\s\S]+?existing alt text was retained/],
  ["cost_incomplete", /^Cost is incomplete: /],
  ["json_resent", /^LLM JSON-text answer was rejected\b/],
  ["image_requests_split", /^Structured image analysis failed\b[\s\S]*separate caption and description requests/],
  ["resent", /^LLM request to \S+ failed \([\s\S]*\) and was sent again$/],
  ["deployment_skipped", /^LLM deployment \S+ [\s\S]* is skipped for this run$/],
  // A model's answer that a guard refused, the base Markdown kept: Python's page drift guard,
  // Rust's cleanup check (its old and its plain wording).
  ["answer_rejected", /page drift guard|LLM answer discarded or replaced too much source content|answer left out or changed too much of the document/],
];

/** Page numbers a list names: `1, 2, 5`, `1–12, 15` (en dash or hyphen); at most 100 000 of them. */
export function parsePages(list: string): number[] {
  const pages = new Set<number>();
  for (const part of list.split(",")) {
    const match = /^\s*(\d{1,6})\s*(?:[–-]\s*(\d{1,6}))?\s*$/.exec(part);
    if (!match) continue;
    const from = Number(match[1]);
    const to = match[2] === undefined ? from : Number(match[2]);
    for (let page = Math.min(from, to); page <= Math.max(from, to) && pages.size < 100_000; page++) pages.add(page);
  }
  return [...pages].sort((a, b) => a - b);
}

/** Page numbers as ranges: `1–12, 15, 20–31` (sorted, each once; an en dash between the ends). */
export function pageRanges(pages: Iterable<number>): string {
  const sorted = [...new Set(pages)].filter((page) => Number.isInteger(page)).sort((a, b) => a - b);
  const parts: string[] = [];
  for (let index = 0; index < sorted.length; ) {
    const start = sorted[index] as number;
    let end = start;
    while (index + 1 < sorted.length && sorted[index + 1] === end + 1) end = sorted[++index] as number;
    parts.push(end === start ? String(start) : end === start + 1 ? `${start}, ${end}` : `${start}–${end}`);
    index++;
  }
  return parts.join(", ");
}

const PAGE_LIST = /\b(pages?)( |\u00a0)(\d{1,6}(?:\s*[–-]\s*\d{1,6})?(?:\s*,\s*\d{1,6}(?:\s*[–-]\s*\d{1,6})?)+)/gi;

/** A warning's lists of page numbers written as ranges, everywhere in its text. */
export const compressPageLists = (text: string): string => text.replace(PAGE_LIST, (_, word: string, space: string, list: string) => `${word}${space}${pageRanges(parsePages(list))}`);

/** What a PDF warning says about pages without readable text: Python's `[PDF] name: N [of M] pages look
 * scanned/garbled (pages …)`, Rust's `PDF is scanned: N of M pages …`, `PDF pages 3–5 (4 of 120): native text
 * was not recovered …` and its older one line per page `PDF page 7: native text was not recovered …`. */
export function pdfPageNotice(text: string): { pages: number[]; total: number | null } | null {
  let match = /^\[PDF\] [\s\S]*?: \d+(?: of (\d+))? pages? look scanned\/garbled \(pages? ([\d,\s–-]+)\)/.exec(text);
  if (match) return { pages: parsePages(match[2] ?? ""), total: match[1] ? Number(match[1]) : null };
  match = /^PDF is scanned: \d+ of (\d+) pages?[^(]*\(pages? ([\d,\s–-]+)\)/.exec(text);
  if (match) return { pages: parsePages(match[2] ?? ""), total: Number(match[1]) };
  match = /^PDF pages? ([\d,\s–-]+?)(?: \(\d+ of (\d+)\))?: native text was not recovered\b/.exec(text);
  if (match) return { pages: parsePages(match[1] ?? ""), total: match[2] ? Number(match[2]) : null };
  return null;
}

/** Pages OCR read but found no text on: Python 1.2.1's `[OCR] No text found on N pages of <name> (pages …); they are
 * blank in the output` (and its later `[OCR] <name>: N of M pages have no text (pages …); …`), Rust's `PDF pages 2, 245–246 (3 of 490): OCR found no text; they are blank in the output.`
 * (`Office pages …: OCR found no text; the OCR supplement leaves them out.` for an Office document). */
export function ocrBlankNotice(text: string): { pages: number[]; total: number | null; office: boolean } | null {
  let match = /^\[OCR\] No text found on \d+ pages? of [\s\S]*?\(pages? ([\d,\s–-]+)\)/.exec(text);
  if (match) return { pages: parsePages(match[1] ?? ""), total: null, office: false };
  // Python's later form: `[OCR] <name>: 2 of 491 pages have no text (pages 166, 360); they are left blank in the output`.
  match = /^\[OCR\] [\s\S]*?: \d+ of (\d+) pages? (?:has|have) no text \(pages? ([\d,\s–-]+)\)/.exec(text);
  if (match) return { pages: parsePages(match[2] ?? ""), total: Number(match[1]), office: false };
  match = /^(PDF|Office) pages? ([\d,\s–-]+?)(?: \(\d+ of (\d+)\))?: OCR found no text\b/.exec(text);
  if (match) return { pages: parsePages(match[2] ?? ""), total: match[3] ? Number(match[3]) : null, office: match[1] === "Office" };
  return null;
}

/** At least this share of a document's pages without readable text: the document is a scan. */
export const SCANNED_SHARE = 0.9;
/** Without the page count: pages 1 to at least this many, all without text, read as a scan. */
export const SCANNED_RUN = 10;

/** The text with its variable parts blanked, so repeats of one message group together. */
export function template(text: string): string {
  return text
    .replace(/https?:\/\/\S+/g, "<url>")
    .replace(/(['"`‘“])[^'"`’”]{0,200}(['"`’”])/g, "<q>")
    .replace(/[\w.-]*[\w-]\.[A-Za-z][A-Za-z0-9]{1,5}\b/g, "<file>")
    .replace(/\d+(?:[.,]\d+)*/g, "<n>")
    .replace(/\s+/g, " ")
    .trim();
}

function kindOf(text: string): NoticeKind {
  for (const [kind, pattern] of SHAPES) if (pattern.test(text)) return kind;
  return "other";
}

/** Models named by a cost or deployment warning. */
export function noticeModels(kind: NoticeKind, text: string): string[] {
  if (kind === "cost_incomplete") return [...text.matchAll(/\d+ requests? to (\S+) (?:has|have|reported)/g)].map((match) => match[1] ?? "").filter(Boolean);
  if (kind === "deployment_skipped") return [/^LLM deployment (\S+)/.exec(text)?.[1] ?? ""].filter(Boolean);
  if (kind === "resent") return [/^LLM request to (\S+)/.exec(text)?.[1] ?? ""].filter(Boolean);
  return [];
}

/** The cost warning is the Cost fact in other words: rows do not mark it again. */
export const isCostWarning = (text: string): boolean => /^Cost is incomplete: /.test(text);

/** Warnings a row marks and the Notices filter counts. */
export const rowWarnings = (warnings: string[]): string[] => warnings.filter((text) => !isCostWarning(text));

export function groupNotices(entries: NoticeEntry[]): NoticeGroup[] {
  const groups = new Map<string, NoticeGroup>();
  // PDF pages without readable text, per document: one warning per page (older Rust) or per
  // reason adds up to the document's set, judged once all are read.
  const pdf = new Map<string, { name: string; pages: Set<number>; total: number | null; texts: string[]; count: number }>();
  const paged = new Map<string, { pages: Set<number>; rest: string } | null>();
  for (const entry of entries) {
    const text = entry.text.trim();
    if (!text) continue;
    const pages = pdfPageNotice(text);
    if (pages) {
      let document = pdf.get(entry.itemKey);
      if (!document) {
        document = { name: entry.itemName, pages: new Set(), total: null, texts: [], count: 0 };
        pdf.set(entry.itemKey, document);
      }
      for (const page of pages.pages) document.pages.add(page);
      document.total = pages.total ?? document.total;
      document.count++;
      if (document.texts.length < 50) document.texts.push(compressPageLists(text));
      continue;
    }
    const blank = ocrBlankNotice(text);
    const kind = blank ? "ocr_blank" : kindOf(text);
    // A recognized kind is one group whatever its details; others group by template.
    const key = kind === "other" ? `other:${template(text)}` : kind;
    let group = groups.get(key);
    if (!group) {
      group = { key, kind, count: 0, items: [], names: [], sample: compressPageLists(text), samples: [], models: [], pages: null };
      groups.set(key, group);
    }
    group.count++;
    if (!group.items.includes(entry.itemKey)) {
      group.items.push(entry.itemKey);
      group.names.push(entry.itemName);
    }
    // One warning per page with the same words (`PDF page 3: …`, `PDF page 4: …`) reads as one with ranges.
    const perPage = /^PDF page (\d{1,6}): ([\s\S]+)$/.exec(text);
    const run = paged.get(key);
    if (run === undefined) paged.set(key, perPage ? { pages: new Set([Number(perPage[1])]), rest: perPage[2] ?? "" } : null);
    else if (run !== null && perPage && perPage[2] === run.rest) run.pages.add(Number(perPage[1]));
    else paged.set(key, null);
    const shown = compressPageLists(text);
    if (group.samples.length < 50 && !group.samples.includes(shown)) group.samples.push(shown);
    for (const model of noticeModels(kind, text)) if (!group.models.includes(model)) group.models.push(model);
    // One document's blank pages are told with their ranges; several documents only by count.
    if (blank) group.pages = group.items.length === 1 ? { ranges: pageRanges(blank.pages), count: blank.pages.length, total: blank.total } : null;
  }
  for (const [key, pages] of paged) {
    const group = groups.get(key);
    if (!group || pages === null || pages.pages.size < 2 || group.items.length > 1) continue;
    group.sample = `PDF pages ${pageRanges(pages.pages)}: ${pages.rest.replace(/\bthis page\b/, "these pages")}`;
    group.pages = { ranges: pageRanges(pages.pages), count: pages.pages.size, total: null };
  }
  for (const [itemKey, document] of pdf) {
    const scanned =
      document.total !== null
        ? document.total > 0 && document.pages.size >= SCANNED_SHARE * document.total
        : // Python 1.2.1 does not say how many pages the document has: an unbroken run from page 1 of at least
          // SCANNED_RUN pages reads as a scan (the warning's own text, behind Details, names the pages).
          document.pages.size >= SCANNED_RUN && [...document.pages].reduce((top, page) => Math.max(top, page), 0) === document.pages.size;
    const kind: NoticeKind = scanned ? "pdf_scanned" : "pdf_ocr_pages";
    let group = groups.get(kind);
    if (!group) {
      group = { key: kind, kind, count: 0, items: [], names: [], sample: document.texts[0] ?? "", samples: [], models: [], pages: null };
      groups.set(kind, group);
    }
    // A count of documents; the pages are told for one document.
    group.count++;
    group.items.push(itemKey);
    group.names.push(document.name);
    group.pages = group.count === 1 ? { ranges: pageRanges(document.pages), count: document.pages.size, total: document.total } : null;
    // Older Rust wrote one line per page: they read as one, with the ranges.
    const lines = document.count > 1 && document.texts.every((line) => /^PDF page \d+: /.test(line)) ? [`PDF pages ${pageRanges(document.pages)}: ${document.texts[0]?.replace(/^PDF page \d+: /, "").replace("for this page", "for these pages")}`] : document.texts;
    for (const line of lines) if (group.samples.length < 50 && !group.samples.includes(line)) group.samples.push(line);
  }
  // Recognized kinds first, then by how much they cover.
  return [...groups.values()].sort((left, right) => Number(left.kind === "other") - Number(right.kind === "other") || right.count - left.count);
}
