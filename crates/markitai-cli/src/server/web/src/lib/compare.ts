// Where a heading or table of the Markdown sits in its original. PDF pages
// report positioned text lines; a heading is the line (or two) that reads the
// same, a table is the run of lines from its header cells to its last row.
// Elements are matched in document order, each search starting after the
// previous match, so a repeated heading finds its own occurrence.

export type Box = [number, number, number, number];

export interface PageLines {
  page: number;
  lines: { text: string; bbox: Box }[];
}

export interface Target {
  id: string;
  kind: "heading" | "table";
  /** Heading text, or a table's header cells. */
  text: string[];
  /** A table's last row cells. */
  last?: string[];
}

export interface Located {
  page: number;
  box: Box;
}

/** Letters and digits only, lower case, compatibility forms folded (full-width digits, ligatures). */
export function norm(text: string): string {
  return text.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");
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

/** Locate every target that can be found; the rest are absent from the map. */
export function locate(pages: PageLines[], targets: Target[]): Map<string, Located> {
  const found = new Map<string, Located>();
  let cursor: Cursor = { pageIndex: 0, line: 0 };
  for (const target of targets) {
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

/** Headings and tables of a rendered Markdown body, in document order, with stable ids. */
export function targetsOf(root: ParentNode): { element: HTMLElement; target: Target }[] {
  const result: { element: HTMLElement; target: Target }[] = [];
  let index = 0;
  for (const element of root.querySelectorAll<HTMLElement>("h1, h2, h3, h4, h5, h6, table")) {
    const id = `t${index++}`;
    if (element.tagName === "TABLE") {
      const table = element as HTMLTableElement;
      const head = table.tHead?.rows[0] ?? table.rows[0];
      const bodyRows = [...table.tBodies].flatMap((body) => [...body.rows]);
      const lastRow = bodyRows[bodyRows.length - 1];
      result.push({
        element,
        target: {
          id,
          kind: "table",
          text: head ? [...head.cells].map((cell) => cell.textContent ?? "") : [],
          last: lastRow ? [...lastRow.cells].map((cell) => cell.textContent ?? "") : [],
        },
      });
    } else result.push({ element, target: { id, kind: "heading", text: [element.textContent ?? ""] } });
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
