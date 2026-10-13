// The Edit config page keeps the file's own line endings. A textarea turns
// every CRLF and CR into LF, so the page edits and compares the text with LF
// only (and without a byte order mark, which the reader cannot see), then puts
// the file's endings back before it is sent: a line that was not changed keeps
// its own ending byte for byte, a changed line keeps the ending of the line it
// replaced, a new line gets the file's ending (the most common one when the
// file mixes them). The backends save the text exactly as sent.
import { compareLines } from "./diff.ts";

export type LineBreak = "\r\n" | "\n" | "\r";

export interface TextLayout {
  /** A U+FEFF byte order mark at the start. */
  bom: boolean;
  /** The ending of each line but the last, in order. */
  breaks: LineBreak[];
  /** How many of each. */
  counts: { crlf: number; lf: number; cr: number };
  /** "none": no line break at all; "mixed": more than one kind. */
  kind: "lf" | "crlf" | "cr" | "mixed" | "none";
  /** The ending a new line gets: the only or most common one (on a tie, the first
   * one in the file); LF when the file has none. */
  newline: LineBreak;
  /** The text ends with a line break. */
  finalBreak: boolean;
}

const BREAK = /\r\n|\r|\n/g;
const BOM = "\uFEFF";

/** How `raw` (the file's text as the server sent it) ends its lines. */
export function textLayout(raw: string): TextLayout {
  const bom = raw.startsWith(BOM);
  const breaks = (raw.match(BREAK) ?? []) as LineBreak[];
  const counts = { crlf: 0, lf: 0, cr: 0 };
  for (const item of breaks) counts[item === "\r\n" ? "crlf" : item === "\n" ? "lf" : "cr"]++;
  const kinds = (["crlf", "lf", "cr"] as const).filter((name) => counts[name] > 0);
  const kind = kinds.length === 0 ? "none" : kinds.length > 1 ? "mixed" : kinds[0];
  const most = Math.max(counts.crlf, counts.lf, counts.cr);
  const newline = breaks.find((item) => counts[item === "\r\n" ? "crlf" : item === "\n" ? "lf" : "cr"] === most) ?? "\n";
  return { bom, breaks, counts, kind, newline, finalBreak: /[\r\n]$/.test(raw) };
}

/** A line's JSON key (`"name": …`), else the line without its indentation. */
function lineKey(line: string): string {
  return /^\s*("(?:[^"\\]|\\.)*")\s*:/.exec(line)?.[1] ?? line.trim();
}

/** `raw` as the editor shows it: every line break LF, no byte order mark. */
export function editorText(raw: string): string {
  return (raw.startsWith(BOM) ? raw.slice(1) : raw).replace(BREAK, "\n");
}

/** The editor's `edited` text (LF only) with `raw`'s byte order mark and line
 * endings: `fileText(raw, editorText(raw)) === raw` for every text. */
export function fileText(raw: string, edited: string): string {
  const layout = textLayout(raw);
  const base = editorText(raw);
  const lines = edited.replace(BREAK, "\n").split("\n");
  // The ending each edited line gets, by its index (the last line has none).
  const endings: LineBreak[] = new Array(Math.max(0, lines.length - 1)).fill(layout.newline);
  if (layout.kind !== "none" && base !== edited) {
    const comparison = compareLines(base, lines.join("\n"));
    if (comparison.kind === "rows") {
      // Removed lines come before added ones in a change. A changed line takes
      // the ending of the removed line it replaces: by place when the change
      // has as many lines on both sides, else the removed line with the same
      // JSON key (in order); other added lines get the file's ending.
      let removed: LineBreak[] = [];
      let removedKeys: string[] = [];
      let added: { index: number; key: string }[] = [];
      const settle = () => {
        if (removed.length === added.length) {
          added.forEach((line, place) => {
            if (line.index < endings.length) endings[line.index] = removed[place] as LineBreak;
          });
        } else {
          let from = 0;
          removedKeys.forEach((key, place) => {
            const match = added.findIndex((line, at) => at >= from && line.key === key);
            if (match < 0) return;
            const index = (added[match] as { index: number }).index;
            if (index < endings.length) endings[index] = removed[place] as LineBreak;
            from = match + 1;
          });
        }
        removed = [];
        removedKeys = [];
        added = [];
      };
      for (const row of comparison.rows) {
        if (row.type === "del") {
          removed.push(layout.breaks[(row.aNo as number) - 1] ?? layout.newline);
          removedKeys.push(lineKey(row.text));
          continue;
        }
        const index = (row.bNo as number) - 1;
        if (row.type === "add") {
          added.push({ index, key: lineKey(row.text) });
          continue;
        }
        settle();
        const own = layout.breaks[(row.aNo as number) - 1];
        if (own && index < endings.length) endings[index] = own;
      }
      settle();
    } else {
      // Too large to compare: equal lines at the start and the end keep their
      // endings, and the lines between are one change.
      const old = base.split("\n");
      let head = 0;
      while (head < old.length && head < lines.length && old[head] === lines[head]) head++;
      let tail = 0;
      while (tail < old.length - head && tail < lines.length - head && old[old.length - 1 - tail] === lines[lines.length - 1 - tail]) tail++;
      for (let index = 0; index < endings.length; index++) {
        const from = index >= lines.length - tail ? old.length - (lines.length - index) : index < old.length - tail ? index : -1;
        const own = from >= 0 ? layout.breaks[from] : undefined;
        if (own) endings[index] = own;
      }
    }
  } else if (base === edited) {
    endings.splice(0, endings.length, ...layout.breaks);
  }
  let out = layout.bom ? BOM : "";
  lines.forEach((line, index) => {
    out += line;
    if (index < endings.length) out += endings[index];
  });
  return out;
}
