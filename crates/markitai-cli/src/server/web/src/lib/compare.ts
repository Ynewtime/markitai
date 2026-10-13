// Where a heading, table or other block of the Markdown sits in its original.
// PDF pages report positioned text lines; a heading is the line (or two) that
// reads the same, a table is the run of lines from its header cells to its last
// row, a paragraph, list item or code block is the run of lines (or the part
// of a line) that spells its text, and a picture is the box its page draws it in
// (the page's pictures in reading order, when the page has as many as the
// Markdown places on it). Elements are matched in document order, each
// search starting after the previous match, so a repeated heading finds its own
// occurrence.

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

/** Walk lines in reading order from a cursor, wrapping once to the start. */
function* walk(pages: PageLines[], from: Cursor): Generator<Cursor> {
  const total = pages.reduce((sum, page) => sum + page.lines.length, 0);
  let pageIndex = from.pageIndex;
  let line = from.line;
  for (let seen = 0; seen < total; seen++) {
    while (pageIndex < pages.length && line >= (pages[pageIndex]?.lines.length ?? 0)) {
      pageIndex++;
      line = 0;
    }
    if (pageIndex >= pages.length) {
      pageIndex = 0;
      line = 0;
      while (pageIndex < pages.length && (pages[pageIndex]?.lines.length ?? 0) === 0) pageIndex++;
      if (pageIndex >= pages.length) return;
    }
    yield { pageIndex, line };
    line++;
  }
}

const lineAt = (pages: PageLines[], at: Cursor) => pages[at.pageIndex]?.lines[at.line];

function findHeading(pages: PageLines[], wanted: string, from: Cursor): { at: Cursor; box: Box } | null {
  if (wanted.length < 2) return null;
  for (const at of walk(pages, from)) {
    const line = lineAt(pages, at);
    if (!line) continue;
    const text = norm(line.text);
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
    const text = norm(start.text);
    if (!holds(text, cells[0] as string)) continue;
    // The header must show at least half its cells within the next few lines.
    const page = pages[at.pageIndex] as PageLines;
    const window = page.lines.slice(at.line, at.line + cells.length + 2).map((line) => norm(line.text)).join("|");
    if (cells.filter((cell) => window.includes(cell)).length < Math.ceil(cells.length / 2)) continue;
    // The table ends at the line that holds its last row's first cell (or the page end).
    let end = page.lines.length - 1;
    if (tail.length) {
      const first = tail[0] as string;
      for (let index = at.line + 1; index < page.lines.length; index++) {
        if (holds(norm(page.lines[index]?.text ?? ""), first)) {
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
      const text = norm(lines[line]?.text ?? "");
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
          const more = norm(lines[next]?.text ?? "");
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
export const TARGET_SELECTOR = "h1, h2, h3, h4, h5, h6, table, p, li, pre, dt, dd, img[data-page]";
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
    else if (element.tagName === "IMG") result.push({ element, target: { id: `t${index++}`, kind: "image", text: [], page: Number(element.dataset.page) } });
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
