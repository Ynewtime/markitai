// Large tables. Parsing and drawing a 20,000-row table blocks the page for
// seconds, so the preview parses only the first rows of every long GFM table:
// the Markdown is cut before it reaches the parser, a marker paragraph takes
// the place of the rest, and the renderer turns the marker into a caption with
// "Show all rows". The download and the Source tab keep every row.

export const ROW_LIMIT = 500;
/** Rows drawn per step when the reader asks for the whole table. */
export const ROW_CHUNK = 250;
export const MARKER = "markitai-table-cap-";

export interface CappedTable {
  /** Header, delimiter and every data row, as written. */
  head: string[];
  rows: string[];
  /** Data rows in the whole table. */
  total: number;
}

const DELIMITER = /^\s{0,3}\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
const FENCE = /^\s{0,3}(`{3,}|~{3,})/;

const hasPipe = (line: string) => line.includes("|");

/** Characters that end a line for the parser's row pattern but not for its line splitting: inside a row they
 * end the table there and the rest of it becomes one paragraph (U+2028, U+2029). Vertical tab and NEL are a
 * manual line break in some exports; they all read as a line break inside a cell. */
const CELL_BREAK = /[\u2028\u2029\u000b\u0085]/g;

/** The cells a table row holds: its unescaped pipes outside code spans, less the outer ones. */
export function rowCells(line: string): number {
  const text = line.trim();
  let pipes = 0;
  let code = 0;
  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    if (char === "\\") index++;
    else if (char === "`") {
      let run = 1;
      while (text[index + run] === "`") run++;
      code = code === 0 ? run : code === run ? 0 : code;
      index += run - 1;
    } else if (char === "|" && code === 0) pipes++;
  }
  const leading = text.startsWith("|") ? 1 : 0;
  const trailing = text.length > 1 && text.endsWith("|") && !text.endsWith("\\|") ? 1 : 0;
  return pipes - leading - trailing + 1;
}

/**
 * A table that a line break inside a cell broke, put back together for the parser (v1.16): converters should
 * write `<br>` there, but a raw break still reaches it. Inside a table a line separator becomes `<br>`; in a table
 * whose rows start with `|`, a row that is still open (fewer cells than its header, or no closing `|` where the
 * header has one) takes the lines that follow and do not start with `|` as the rest of its last cell, joined by
 * `<br>` (one blank line between them too, when the line after it closes the row). The following rows then stay
 * rows of the same table. Fenced code is left alone; the Source view keeps the text as written.
 */
export function repairTables(markdown: string): string {
  if (!markdown.includes("|")) return markdown;
  const lines = markdown.split("\n");
  const out: string[] = [];
  let fence: string | null = null;
  let changed = false;
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index] as string;
    const opened = FENCE.exec(line);
    if (fence !== null) {
      out.push(line);
      if (opened && opened[1]?.[0] === fence[0] && (opened[1]?.length ?? 0) >= fence.length) fence = null;
      continue;
    }
    if (opened) {
      fence = opened[1] ?? "```";
      out.push(line);
      continue;
    }
    const next = lines[index + 1];
    if (!hasPipe(line) || next === undefined || !DELIMITER.test(next) || !hasPipe(next)) {
      out.push(line);
      continue;
    }
    const clean = (text: string) => {
      const value = text.replace(CELL_BREAK, "<br>");
      if (value !== text) changed = true;
      return value;
    };
    const head = clean(line);
    const columns = rowCells(head);
    const leading = head.trim().startsWith("|");
    const closing = head.trim().endsWith("|");
    out.push(head, next);
    let at = index + 2;
    while (at < lines.length) {
      let row = lines[at] as string;
      if (row.trim() === "" || !hasPipe(row)) break;
      row = clean(row);
      at++;
      if (leading) {
        const open = () => rowCells(row) < columns || (closing && !row.trim().endsWith("|"));
        while (open() && at < lines.length) {
          const following = lines[at] as string;
          if (following.trim() !== "" && !following.trimStart().startsWith("|")) {
            row = `${row.trimEnd()}<br>${clean(following.trim())}`;
            at++;
            changed = true;
            continue;
          }
          const after = lines[at + 1];
          if (following.trim() === "" && after !== undefined && after.trim() !== "" && !after.trimStart().startsWith("|") && (after.trimEnd().endsWith("|") || (lines[at + 2] ?? "").trimStart().startsWith("|"))) {
            row = `${row.trimEnd()}<br>${clean(after.trim())}`;
            at += 2;
            changed = true;
            continue;
          }
          break;
        }
      }
      out.push(row);
    }
    index = at - 1;
  }
  return changed ? out.join("\n") : markdown;
}

/** Cut every table over `limit` data rows; returns the cut Markdown and the tables, by marker index. */
export function capTables(markdown: string, limit = ROW_LIMIT): { markdown: string; tables: CappedTable[] } {
  // Fast path: short documents cannot hold a long table.
  if (markdown.length < limit * 4) return { markdown, tables: [] };
  const lines = markdown.split("\n");
  const out: string[] = [];
  const tables: CappedTable[] = [];
  let fence: string | null = null;
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index] as string;
    const opened = FENCE.exec(line);
    if (fence !== null) {
      out.push(line);
      if (opened && opened[1]?.[0] === fence[0] && (opened[1]?.length ?? 0) >= fence.length) fence = null;
      continue;
    }
    if (opened) {
      fence = opened[1] ?? "```";
      out.push(line);
      continue;
    }
    const next = lines[index + 1];
    if (!hasPipe(line) || next === undefined || !DELIMITER.test(next) || !hasPipe(next)) {
      out.push(line);
      continue;
    }
    // A table: header, delimiter, then rows until a blank line or a line without a pipe.
    let end = index + 2;
    while (end < lines.length && (lines[end] as string).trim() !== "" && hasPipe(lines[end] as string)) end++;
    const rows = lines.slice(index + 2, end);
    if (rows.length > limit) {
      out.push(line, next, ...rows.slice(0, limit), "", `${MARKER}${tables.length}`, "");
      tables.push({ head: [line, next], rows, total: rows.length });
    } else out.push(line, next, ...rows);
    index = end - 1;
  }
  return { markdown: out.join("\n"), tables };
}

/** The Markdown of one slice of a capped table's rows, with its header, for drawing in steps. */
export function tableSlice(table: CappedTable, from: number, to: number): string {
  return [...table.head, ...table.rows.slice(from, to)].join("\n");
}

/** A cell that reads as a number: right-aligned in tabular figures. */
export const NUMERIC = /^[-+−]?(?:[$€£¥]\s?)?\d[\d,.\s]*(?:%|[kKmMbB])?$/;

/** Keep the columns as wide as they are now, so adding rows lays out only the new ones. */
export function freezeColumns(table: HTMLTableElement): void {
  const head = table.tHead?.rows[0] ?? table.rows[0];
  if (!head || table.classList.contains("is-frozen")) return;
  const widths = [...head.cells].map((cell) => cell.getBoundingClientRect().width);
  widths.forEach((width, index) => {
    const cell = head.cells[index];
    if (cell) cell.style.width = `${Math.ceil(width)}px`;
  });
  table.style.width = `${Math.ceil(table.getBoundingClientRect().width)}px`;
  table.classList.add("is-frozen");
}

/** Mark the columns whose filled cells are all numeric (header included in the mark). */
export function markNumericColumns(table: HTMLTableElement, sample = 200): void {
  const rows = [...table.tBodies].flatMap((body) => [...body.rows]).slice(0, sample);
  if (!rows.length) return;
  const width = Math.max(...rows.map((row) => row.cells.length));
  const numeric: boolean[] = [];
  for (let column = 0; column < width; column++) {
    let filled = 0;
    let numbers = 0;
    for (const row of rows) {
      const text = row.cells[column]?.textContent?.trim() ?? "";
      if (!text) continue;
      filled++;
      if (NUMERIC.test(text)) numbers++;
    }
    numeric[column] = filled > 0 && numbers === filled;
  }
  const mark = (row: HTMLTableRowElement) => {
    for (let column = 0; column < row.cells.length; column++) if (numeric[column]) row.cells[column]?.classList.add("num");
  };
  for (const row of table.tHead?.rows ?? []) mark(row);
  for (const body of table.tBodies) for (const row of body.rows) mark(row);
}
