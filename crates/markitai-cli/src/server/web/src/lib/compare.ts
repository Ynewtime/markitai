// Where a heading, table or other block of the Markdown sits in its original.
// PDF pages report positioned text lines; a heading is the line (or two) that
// reads the same, a table is the run of lines from its header cells to its last
// row, a paragraph, list item or code block is the run of lines (or the part
// of a line) that spells its text, and a picture is the box its page draws it in
// (the page's pictures in reading order, when the page has as many as the
// Markdown places on it). Elements are matched in document order, each
// search starting after the previous match, so a repeated heading finds its own
// occurrence.

import type { CaptureQuality, OriginalDescriptor } from "../api/types.ts";

export type Box = [number, number, number, number];

export interface PageLines {
  page: number;
  lines: { text: string; bbox: Box }[];
  /** Where the page's pictures are drawn (servers that report them). */
  images?: Box[];
}

export interface Target {
  id: string;
  kind: "heading" | "table" | "block" | "image";
  /** Heading text, or a table's header cells. */
  text: string[];
  /** A picture's page. */
  page?: number;
  /** A table's last row cells. */
  last?: string[];
  /** A picture's address as the Markdown wrote it. */
  src?: string;
}

export interface Located {
  page: number;
  box: Box;
}

/** CJK radicals that NFKC leaves alone, as the characters they stand for: an
 * extracted PDF's text sometimes spells 人民币 with the radical ⺠ (U+2EA0). */
const RADICALS: Record<string, string> = { "\u2EA0": "民", "\u2EB0": "纟", "\u2EC4": "西", "\u2EC5": "见", "\u2EC6": "角", "\u2EC8": "讠", "\u2EC9": "贝", "\u2ECB": "车", "\u2ED0": "钅", "\u2ED3": "长", "\u2ED4": "门", "\u2ED8": "青", "\u2ED9": "韦", "\u2EDA": "页", "\u2EDB": "风", "\u2EDC": "飞", "\u2EDD": "食", "\u2EE0": "饣", "\u2EE2": "马", "\u2EE5": "鱼", "\u2EE6": "鸟", "\u2EE7": "卤", "\u2EE8": "麦", "\u2EE9": "黄", "\u2EEA": "黾", "\u2EEC": "齐", "\u2EEE": "齿", "\u2EF0": "龙" };

/** Letters and digits only, lower case, compatibility forms folded (full-width digits, ligatures, Kangxi radicals). */
export function norm(text: string): string {
  return text
    .normalize("NFKC")
    .replace(/[\u2E80-\u2EFF]/g, (radical) => RADICALS[radical] ?? radical)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "");
}

function union(boxes: Box[]): Box {
  return [
    Math.min(...boxes.map((box) => box[0])),
    Math.min(...boxes.map((box) => box[1])),
    Math.max(...boxes.map((box) => box[2])),
    Math.max(...boxes.map((box) => box[3])),
  ];
}

interface Cursor {
  pageIndex: number;
  line: number;
}

/** Walk lines in reading order from a cursor to the document's end. It never wraps back to the
 * start: a heading not found ahead (its page not read yet, or written otherwise) stays unlinked
 * rather than pairing with an earlier line that says the same (a running head, the contents). */
function* walk(pages: PageLines[], from: Cursor): Generator<Cursor> {
  let pageIndex = from.pageIndex;
  let line = from.line;
  for (;;) {
    while (pageIndex < pages.length && line >= (pages[pageIndex]?.lines.length ?? 0)) {
      pageIndex++;
      line = 0;
    }
    if (pageIndex >= pages.length) return;
    yield { pageIndex, line };
    line++;
  }
}

/** A line's text as `norm` gives it, worked out once: a long document is searched many times. */
const normed = new WeakMap<object, string>();
function normOf(line: { text: string } | undefined): string {
  if (!line) return "";
  let text = normed.get(line);
  if (text === undefined) {
    text = norm(line.text);
    normed.set(line, text);
  }
  return text;
}

const lineAt = (pages: PageLines[], at: Cursor) => pages[at.pageIndex]?.lines[at.line];

function findHeading(pages: PageLines[], wanted: string, from: Cursor): { at: Cursor; box: Box } | null {
  if (wanted.length < 2) return null;
  for (const at of walk(pages, from)) {
    const line = lineAt(pages, at);
    if (!line) continue;
    const text = normOf(line);
    if (!text) continue;
    if (text === wanted) return { at, box: line.bbox };
    // A heading wrapped over two lines, or set apart from a section number.
    if (wanted.startsWith(text) && text.length >= Math.min(8, wanted.length / 2)) {
      const next = pages[at.pageIndex]?.lines[at.line + 1];
      if (next && norm(line.text + next.text) === wanted) return { at, box: union([line.bbox, next.bbox]) };
    }
    if (text.endsWith(wanted) && wanted.length >= 4 && text.length - wanted.length <= 6) return { at, box: line.bbox };
  }
  return null;
}

const holds = (line: string, cell: string) => cell.length > 0 && (line === cell || line.includes(cell));

function findTable(pages: PageLines[], header: string[], last: string[], from: Cursor): { at: Cursor; box: Box } | null {
  const cells = header.map(norm).filter((cell) => cell.length > 0);
  const tail = last.map(norm).filter((cell) => cell.length > 0);
  if (!cells.length) return null;
  for (const at of walk(pages, from)) {
    const start = lineAt(pages, at);
    if (!start) continue;
    const text = normOf(start);
    if (!holds(text, cells[0] as string)) continue;
    // The header must show at least half its cells within the next few lines.
    const page = pages[at.pageIndex] as PageLines;
    const window = page.lines.slice(at.line, at.line + cells.length + 2).map(normOf).join("|");
    if (cells.filter((cell) => window.includes(cell)).length < Math.ceil(cells.length / 2)) continue;
    // The table ends at the line that holds its last row's first cell (or the page end).
    let end = page.lines.length - 1;
    if (tail.length) {
      const first = tail[0] as string;
      for (let index = at.line + 1; index < page.lines.length; index++) {
        if (holds(normOf(page.lines[index]), first)) {
          end = index;
          // A row set as several lines: take the lines on the same baseline.
          const baseline = page.lines[index]?.bbox[3] ?? 0;
          while (end + 1 < page.lines.length && Math.abs((page.lines[end + 1]?.bbox[3] ?? 0) - baseline) < 2) end++;
          break;
        }
      }
    } else end = Math.min(page.lines.length - 1, at.line + cells.length);
    return { at, box: union(page.lines.slice(at.line, end + 1).map((line) => line.bbox)) };
  }
  return null;
}

/** Lines a block is looked for ahead of the previous match (blocks do not wrap around). */
const BLOCK_REACH = 240;

/** The part of a line from `start` to `end` (of `length` letters and digits), by proportion. */
function part(box: Box, start: number, end: number, length: number): Box {
  if (length <= 0 || (start <= 0 && end >= length)) return box;
  const width = box[2] - box[0];
  return [box[0] + (width * start) / length, box[1], box[0] + (width * end) / length, box[3]];
}

/** A block of text: the run of lines that spells it, starting in a line and possibly ending in one. */
function findBlock(pages: PageLines[], wanted: string, from: Cursor): { at: Cursor; end: Cursor; box: Box } | null {
  if (wanted.length < 2) return null;
  const probe = wanted.slice(0, Math.min(wanted.length, 6));
  let seen = 0;
  for (let pageIndex = from.pageIndex; pageIndex < pages.length && seen < BLOCK_REACH; pageIndex++) {
    const lines = pages[pageIndex]?.lines ?? [];
    for (let line = pageIndex === from.pageIndex ? from.line : 0; line < lines.length && seen < BLOCK_REACH; line++, seen++) {
      const text = normOf(lines[line]);
      // A short block is a line of its own; a longer one may start or sit inside a line.
      if (wanted.length < 4) {
        if (text === wanted) return { at: { pageIndex, line }, end: { pageIndex, line }, box: lines[line]?.bbox as Box };
        continue;
      }
      for (let start = text.indexOf(probe); start >= 0; start = text.indexOf(probe, start + 1)) {
        const rest = text.slice(start);
        const box = lines[line]?.bbox as Box;
        if (rest.startsWith(wanted)) {
          return { at: { pageIndex, line }, end: { pageIndex, line }, box: part(box, start, start + wanted.length, text.length) };
        }
        if (!wanted.startsWith(rest)) continue;
        const boxes = [part(box, start, text.length, text.length)];
        let read = rest.length;
        let next = line + 1;
        let last = line;
        while (read < wanted.length && next < lines.length) {
          const more = normOf(lines[next]);
          if (!more) {
            next++;
            continue;
          }
          const remaining = wanted.slice(read);
          if (remaining.startsWith(more)) {
            boxes.push(lines[next]?.bbox as Box);
            read += more.length;
          } else if (more.startsWith(remaining)) {
            boxes.push(part(lines[next]?.bbox as Box, 0, remaining.length, more.length));
            read = wanted.length;
          } else break;
          last = next++;
        }
        // Whole, or nearly; or cut by the page's end (it goes on over the page).
        if (read >= wanted.length * 0.85 || (next >= lines.length && read >= Math.min(wanted.length, 12))) {
          return { at: { pageIndex, line }, end: { pageIndex, line: last }, box: union(boxes) };
        }
      }
    }
  }
  return null;
}

/** Locate every target that can be found; the rest are absent from the map. */
export function locate(pages: PageLines[], targets: Target[]): Map<string, Located> {
  const found = new Map<string, Located>();
  // Pictures: the Markdown's pictures of a page, in order, are the page's own in reading order.
  const pictures = new Map<number, Target[]>();
  for (const target of targets) if (target.kind === "image" && target.page) pictures.set(target.page, [...(pictures.get(target.page) ?? []), target]);
  for (const [page, wanted] of pictures) {
    const boxes = [...(pages.find((layer) => layer.page === page)?.images ?? [])].sort((a, b) => (Math.abs(a[1] - b[1]) > 4 ? a[1] - b[1] : a[0] - b[0]));
    if (boxes.length !== wanted.length) continue;
    for (const [index, target] of wanted.entries()) found.set(target.id, { page, box: boxes[index] as Box });
  }
  let cursor: Cursor = { pageIndex: 0, line: 0 };
  for (const target of targets) {
    if (target.kind === "image") continue;
    if (target.kind === "block") {
      const match = findBlock(pages, norm(target.text.join(" ")), cursor);
      if (!match) continue;
      found.set(target.id, { page: pages[match.at.pageIndex]?.page ?? 1, box: match.box });
      // The next block may start in the same line.
      cursor = match.end.line > match.at.line ? match.end : match.at;
      continue;
    }
    const match =
      target.kind === "heading"
        ? findHeading(pages, norm(target.text.join(" ")), cursor)
        : findTable(pages, target.text, target.last ?? [], cursor);
    if (!match) continue;
    found.set(target.id, { page: pages[match.at.pageIndex]?.page ?? 1, box: match.box });
    cursor = { pageIndex: match.at.pageIndex, line: match.at.line + 1 };
  }
  return found;
}

/** The elements Compare links: headings, tables and the text blocks between them. */
export const TARGET_SELECTOR = "h1, h2, h3, h4, h5, h6, table, p, li, pre, dt, dd, img";
/** What a list item says itself, without the lists, paragraphs and code it holds (those are their own blocks). */
const OWN_SKIP = new Set(["UL", "OL", "P", "PRE", "TABLE", "BLOCKQUOTE", "DL"]);

function ownText(element: Element): string {
  let text = "";
  for (const node of element.childNodes) {
    if (node.nodeType === 3) text += node.textContent ?? "";
    else if (node.nodeType === 1 && !OWN_SKIP.has((node as Element).tagName)) text += node.textContent ?? "";
  }
  return text;
}

/** A picture file's name as both versions of a result write it (`.markitai/assets/3f…9a.jpg` → `3f…9a.jpg`):
 * a PDF's pictures are named by their content, so the Base and LLM Markdown name each one alike. */
export function assetName(path: string): string {
  const bare = path.trim().replace(/^<|>$/g, "").split(/[?#]/)[0] ?? "";
  const name = bare.slice(bare.lastIndexOf("/") + 1);
  try {
    return decodeURIComponent(name);
  } catch {
    return name;
  }
}

const PAGE_MARK = /<!--\s*page:\s*(\d{1,5})\s*-->/;
const PAGE_SHOT = /!\[Page (\d{1,5})\]\([^)]*\)/;
const PICTURE = /!\[([^\]]*)\]\(\s*(<[^>]*>|[^)\s]+)[^)]*\)/g;

/** The page of each picture a PDF's Markdown shows, by file name, in the order the pictures come
 * (one file may stand on several pages: identical pictures share it). A Base alt (`Image on page 3`)
 * or a page-numbered name says it; otherwise the page marker or page screenshot before it does
 * (`<!-- page: 3 -->`, `<!-- ![Page 3](…) -->`); null when nothing does. The LLM version's alt
 * texts are the model's, so its pictures often take their pages from the Base version. */
export function imagePages(markdown: string): Map<string, (number | null)[]> {
  const pages = new Map<string, (number | null)[]>();
  let page: number | null = null;
  for (const line of markdown.split("\n")) {
    const marked = PAGE_MARK.exec(line) ?? PAGE_SHOT.exec(line);
    if (marked) page = Number(marked[1]);
    for (const [whole, alt, path] of line.matchAll(PICTURE)) {
      if (PAGE_SHOT.test(whole)) continue;
      const name = assetName(path ?? "");
      if (!name) continue;
      const own = /^(?:Image|Chart) on page (\d{1,5})$/.exec((alt ?? "").trim()) ?? /\.pdf-(\d{1,5})-\d+\.[a-z0-9]+$/i.exec(name);
      pages.set(name, [...(pages.get(name) ?? []), own ? Number(own[1]) : page]);
    }
  }
  return pages;
}

/** How many of a version's pictures it gives no page (see imagePages). */
export function unplacedPictures(pages: Map<string, (number | null)[]>): number {
  let count = 0;
  for (const list of pages.values()) count += list.filter((page) => page === null).length;
  return count;
}

/** Gives each rendered picture without a page (`data-page`) the page `pages` names for its file, in
 * order; `fallback` (the Base version's) fills the ones `pages` leaves open. */
export function placePictures(root: ParentNode, pages: Map<string, (number | null)[]>, fallback?: Map<string, (number | null)[]>): void {
  const seen = new Map<string, number>();
  for (const image of root.querySelectorAll<HTMLElement>("img[data-asset]")) {
    const name = assetName(image.dataset.asset ?? "");
    const index = seen.get(name) ?? 0;
    seen.set(name, index + 1);
    if (image.dataset.page) continue;
    const page = pages.get(name)?.[index] ?? fallback?.get(name)?.[index] ?? null;
    if (page !== null && page >= 1) image.dataset.page = String(page);
  }
}

/** Headings, tables and text blocks of a rendered Markdown body, in document order, with stable ids. */
export function targetsOf(root: ParentNode): { element: HTMLElement; target: Target }[] {
  const result: { element: HTMLElement; target: Target }[] = [];
  let index = 0;
  for (const element of root.querySelectorAll<HTMLElement>(TARGET_SELECTOR)) {
    if (element.tagName === "TABLE") {
      const table = element as HTMLTableElement;
      const head = table.tHead?.rows[0] ?? table.rows[0];
      const bodyRows = [...table.tBodies].flatMap((body) => [...body.rows]);
      const lastRow = bodyRows[bodyRows.length - 1];
      result.push({
        element,
        target: {
          id: `t${index++}`,
          kind: "table",
          text: head ? [...head.cells].map((cell) => cell.textContent ?? "") : [],
          last: lastRow ? [...lastRow.cells].map((cell) => cell.textContent ?? "") : [],
        },
      });
    } else if (/^H[1-6]$/.test(element.tagName)) result.push({ element, target: { id: `t${index++}`, kind: "heading", text: [element.textContent ?? ""] } });
    else if (element.tagName === "IMG") {
      const page = Number(element.dataset.page);
      const src = element.getAttribute("src") ?? "";
      result.push({ element, target: { id: `t${index++}`, kind: "image", text: [element.getAttribute("alt") ?? ""], ...(page >= 1 ? { page } : {}), ...(/^https:/i.test(src) ? { src } : {}) } });
    }
    else {
      // A block inside a table is part of the table; one without words (a picture) has nothing to find.
      if (element.closest("table")) continue;
      const text = element.tagName === "LI" ? ownText(element) : (element.textContent ?? "");
      if (norm(text).length < 2) continue;
      result.push({ element, target: { id: `t${index++}`, kind: "block", text: [text] } });
    }
  }
  return result;
}

/** HTML originals: headings pair by text in order, tables by position. */
export function pairElements(markdown: { element: HTMLElement; target: Target }[], original: ParentNode): Map<string, HTMLElement> {
  const pairs = new Map<string, HTMLElement>();
  const headings = [...original.querySelectorAll<HTMLElement>("h1, h2, h3, h4, h5, h6")];
  const tables = [...original.querySelectorAll<HTMLElement>("table")];
  let headingFrom = 0;
  let tableIndex = 0;
  for (const { target } of markdown) {
    if (target.kind === "block" || target.kind === "image") continue;
    if (target.kind === "table") {
      const table = tables[tableIndex++];
      if (table) pairs.set(target.id, table);
      continue;
    }
    const wanted = norm(target.text.join(" "));
    if (!wanted) continue;
    for (let index = headingFrom; index < headings.length; index++) {
      if (norm(headings[index]?.textContent ?? "") === wanted) {
        pairs.set(target.id, headings[index] as HTMLElement);
        headingFrom = index + 1;
        break;
      }
    }
  }
  return pairs;
}

/** One element of a framed web page, as the matcher reads it. */
export interface PageBlock {
  /** Lower-case tag name. */
  tag: string;
  /** The words it shows itself (a list item without its nested lists). */
  text: string;
  /** A picture's resolved address. */
  src?: string;
  /** A picture too small to stand for one of the Markdown's (an icon, an avatar). */
  small?: boolean;
}

/** Pictures narrower or lower than this (CSS pixels) are a page's furniture, not its content. */
const SMALL_PICTURE = 48;

/** A picture's size as the page states it, when it does. */
function statedSmall(element: Element): boolean {
  const width = Number(element.getAttribute("width"));
  const height = Number(element.getAttribute("height"));
  return (width > 0 && width < SMALL_PICTURE) || (height > 0 && height < SMALL_PICTURE);
}

/** Blocks read in the page when its screenshots were taken (contract v1.14 `…/original/blocks`). */
export function shotBlocks(blocks: { tag: string; text: string; src?: string; bbox: Box }[]): PageBlock[] {
  return blocks.map(({ tag, text, src, bbox }) => {
    const small = tag === "img" && (bbox[2] - bbox[0] < SMALL_PICTURE || bbox[3] - bbox[1] < SMALL_PICTURE);
    return { tag, text, ...(src !== undefined ? { src } : {}), ...(small ? { small } : {}) };
  });
}

/** One box of the page (CSS pixels) on the screenshots it was cut into, top to bottom: the
 * screenshot it starts on and the box there, in that screenshot's pixels, clipped to it. */
export function onShots(box: Box, pageWidth: number, shots: { width: number; height: number }[]): Located | null {
  const first = shots[0];
  if (!first || pageWidth <= 0) return null;
  const scale = first.width / pageWidth;
  let top = 0;
  for (const [index, shot] of shots.entries()) {
    const bottom = top + shot.height;
    const y0 = box[1] * scale;
    if (y0 < bottom) {
      return {
        page: index + 1,
        box: [box[0] * scale, Math.max(0, y0 - top), Math.min(shot.width, box[2] * scale), Math.min(shot.height, box[3] * scale - top)],
      };
    }
    top = bottom;
  }
  return null;
}

/** The frame's elements a block of the Markdown can stand for, in document order. */
export const PAGE_BLOCKS = "h1, h2, h3, h4, h5, h6, p, li, pre, dt, dd, td, th, blockquote, figcaption, caption, table, img, div, section, article";
/** Containers count only for the words they hold directly (a page that writes paragraphs as `<div>`s). */
const CONTAINERS = new Set(["div", "section", "article", "blockquote", "td", "th", "li", "dd"]);

/** A framed page's candidate blocks; `limit` bounds the work on a huge page. */
export function pageBlocks(root: ParentNode, limit = 20_000): { element: Element; block: PageBlock }[] {
  const result: { element: Element; block: PageBlock }[] = [];
  for (const element of root.querySelectorAll(PAGE_BLOCKS)) {
    if (result.length >= limit) break;
    const tag = element.tagName.toLowerCase();
    if (tag === "img") {
      const block: PageBlock = { tag, text: element.getAttribute("alt") ?? "", src: (element as HTMLImageElement).src || element.getAttribute("src") || "" };
      if (statedSmall(element)) block.small = true;
      result.push({ element, block });
      continue;
    }
    // Text in a hidden part of the page (menus, templates) is no match for what the reader sees.
    if (element.closest("script, style, template, noscript, [hidden], nav, header > nav")) continue;
    const text = CONTAINERS.has(tag) && tag !== "td" && tag !== "th" ? ownText(element) : (element.textContent ?? "");
    if (tag === "div" || tag === "section" || tag === "article") {
      // Only a container whose own words make a block (no block children holding them).
      if (norm(text).length < 8) continue;
    }
    result.push({ element, block: { tag, text } });
  }
  return result;
}

const HEADING = /^h[1-6]$/;

/**
 * Pair the Markdown's targets with a framed web page's blocks, in document order: a heading with a
 * heading that reads the same, a paragraph, list item or code block with the block that spells it
 * (the same words, or the page's block holding them, or the start of them when the page splits a
 * paragraph), a table with the page's tables in order, a picture with the page's picture of the same
 * address (else the same description). Each search starts after the previous match and falls back to
 * the whole page once, so one block the page moved does not lose the rest. Returns target id → block index.
 */
export function pairBlocks(targets: Target[], blocks: PageBlock[]): Map<string, number> {
  const pairs = new Map<string, number>();
  const words = blocks.map((block) => (block.tag === "img" ? "" : norm(block.text)));
  const tables = blocks.flatMap((block, index) => (block.tag === "table" ? [index] : []));
  const used = new Set<number>();
  let cursor = 0;
  let tableIndex = 0;
  const search = (test: (index: number) => boolean): number => {
    for (const [from, to] of [
      [cursor, blocks.length],
      [0, cursor],
    ] as const) {
      for (let index = from; index < to; index++) if (!used.has(index) && test(index)) return index;
    }
    return -1;
  };
  for (const target of targets) {
    let found = -1;
    if (target.kind === "table") {
      const index = tables[tableIndex++];
      if (index !== undefined) found = index;
    } else if (target.kind === "image") {
      const src = target.src;
      const alt = norm(target.text.join(" "));
      found = src ? search((index) => blocks[index]?.tag === "img" && blocks[index]?.src === src) : -1;
      if (found < 0 && alt.length >= 2) found = search((index) => blocks[index]?.tag === "img" && norm(blocks[index]?.text ?? "") === alt);
    } else {
      const wanted = norm(target.text.join(" "));
      if (wanted.length < 2) continue;
      const heading = target.kind === "heading";
      // The same words; for a heading a heading first.
      if (heading) found = search((index) => HEADING.test(blocks[index]?.tag ?? "") && words[index] === wanted);
      if (found < 0) found = search((index) => words[index] === wanted);
      // The smallest block that holds them (the page wraps a paragraph with more, or sets two of
      // the Markdown's list items in one paragraph): it may hold a block paired before, but not
      // a whole article around a few words.
      if (found < 0 && wanted.length >= 6) {
        let best = -1;
        const most = wanted.length * 4 + 400;
        for (const [from, to] of [
          [Math.max(0, cursor - 1), blocks.length],
          [0, Math.max(0, cursor - 1)],
        ] as const) {
          for (let index = from; index < to; index++) {
            const text = words[index] ?? "";
            if (text.length < wanted.length || text.length > most || !text.includes(wanted)) continue;
            if (best < 0 || text.length < (words[best] ?? "").length) best = index;
            if (text.length < wanted.length * 1.5) break;
          }
          if (best >= 0) break;
        }
        found = best;
      }
      // The block the paragraph starts with (the page split it in two).
      if (found < 0 && wanted.length >= 12) found = search((index) => (words[index]?.length ?? 0) >= Math.max(8, wanted.length / 2) && wanted.startsWith(words[index] ?? "\u0000"));
    }
    if (found < 0) continue;
    pairs.set(target.id, found);
    if (target.kind !== "table") used.add(found);
    if (found >= cursor) cursor = found + 1;
  }
  // Pictures the Markdown names otherwise (a downloaded copy, a description a model wrote): the
  // page's own pictures, in order, between the blocks paired before and after them.
  const anchor = (from: number, step: number): number | undefined => {
    for (let index = from; index >= 0 && index < targets.length; index += step) {
      const target = targets[index] as Target;
      const at = target.kind === "table" ? undefined : pairs.get(target.id);
      if (at !== undefined) return at;
    }
    return undefined;
  };
  for (const [position, target] of targets.entries()) {
    if (target.kind !== "image" || pairs.has(target.id)) continue;
    const low = anchor(position - 1, -1) ?? -1;
    const high = anchor(position + 1, 1) ?? blocks.length;
    for (let index = low + 1; index < high; index++) {
      const block = blocks[index];
      if (block?.tag !== "img" || block.small || used.has(index)) continue;
      pairs.set(target.id, index);
      used.add(index);
      break;
    }
  }
  return pairs;
}

/** What a web page's Compare can show (contract v1.14 `capture`): the kept page, its screenshots, or
 * neither; `failed` says why nothing usable was kept (a check page, a blank page) when that is known. */
export interface WebViews {
  page: boolean;
  shot: boolean;
  failed: CaptureQuality | null;
}

const usable = (quality: CaptureQuality | null | undefined) => !quality || quality.state === "ok";

export function webViews(descriptor: Pick<OriginalDescriptor, "kind" | "view" | "page_count" | "capture">): WebViews {
  const kept = descriptor.kind === "html" && typeof descriptor.view === "string";
  const shots = descriptor.page_count > 0 && (descriptor.kind === "html" || descriptor.kind === "screenshots");
  const page = kept && usable(descriptor.capture?.page);
  const shot = shots && usable(descriptor.capture?.screenshot);
  const quality = descriptor.capture ?? null;
  const failed = page || shot ? null : ((kept && !usable(quality?.page) ? quality?.page : null) ?? (shots && !usable(quality?.screenshot) ? quality?.screenshot : null) ?? null);
  return { page, shot, failed };
}
