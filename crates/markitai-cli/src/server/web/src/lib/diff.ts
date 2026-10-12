// Line comparison of the base and LLM Markdown. Equal head and tail lines are
// trimmed first; the middle uses Myers' O(ND) difference algorithm, whose work
// grows with the number of changed lines rather than the document's length,
// bounded so a document that changed everywhere is refused instead of freezing
// the page. Long unchanged stretches are then folded around the changes.

/** `cells`: the work allowed in the middle, as (lines on both sides) × (edits). */
export const DIFF_LIMITS = Object.freeze({ lines: 200_000, characters: 16 * 1024 * 1024, cells: 40_000_000 });
/** Edits kept for the way back: their snapshots grow with the square of this. */
const MAX_EDITS = 4000;

export interface DiffRow {
  type: "ctx" | "add" | "del";
  text: string;
  /** Line number in the base text; null for added lines. */
  aNo: number | null;
  /** Line number in the LLM text; null for removed lines. */
  bNo: number | null;
}

export type Comparison = { kind: "rows"; rows: DiffRow[]; added: number; removed: number } | { kind: "too-large" };

export function compareLines(
  base: string,
  llm: string,
  limits: { lines: number; characters: number; cells: number } = DIFF_LIMITS,
): Comparison {
  if (base.length > limits.characters || llm.length > limits.characters) return { kind: "too-large" };
  const a = base.split("\n");
  const b = llm.split("\n");
  if (a.length + b.length > limits.lines) return { kind: "too-large" };
  let head = 0;
  while (head < a.length && head < b.length && a[head] === b[head]) head++;
  let tail = 0;
  while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++;
  // Lines as numbers, so the inner loop compares integers.
  const ids = new Map<string, number>();
  const intern = (lines: string[], from: number, to: number) => {
    const out = new Int32Array(Math.max(0, to - from));
    for (let index = from; index < to; index++) {
      const line = lines[index] ?? "";
      let id = ids.get(line);
      if (id === undefined) ids.set(line, (id = ids.size));
      out[index - from] = id;
    }
    return out;
  };
  const middle = editScript(intern(a, head, a.length - tail), intern(b, head, b.length - tail), limits.cells);
  if (middle === null) return { kind: "too-large" };

  const rows: DiffRow[] = [];
  let aNo = 1;
  let bNo = 1;
  for (let k = 0; k < head; k++) rows.push({ type: "ctx", text: a[k] ?? "", aNo: aNo++, bNo: bNo++ });
  // Within one change, removed lines come before added ones.
  let removedRun: DiffRow[] = [];
  let addedRun: DiffRow[] = [];
  const flush = () => {
    rows.push(...removedRun, ...addedRun);
    removedRun = [];
    addedRun = [];
  };
  for (const op of middle) {
    if (op === 0) {
      flush();
      rows.push({ type: "ctx", text: a[aNo - 1] ?? "", aNo: aNo++, bNo: bNo++ });
    } else if (op === 1) removedRun.push({ type: "del", text: a[aNo - 1] ?? "", aNo: aNo++, bNo: null });
    else addedRun.push({ type: "add", text: b[bNo - 1] ?? "", aNo: null, bNo: bNo++ });
  }
  flush();
  for (let k = a.length - tail; k < a.length; k++) rows.push({ type: "ctx", text: a[k] ?? "", aNo: aNo++, bNo: bNo++ });
  let plus = 0;
  let minus = 0;
  for (const row of rows) {
    if (row.type === "add") plus++;
    else if (row.type === "del") minus++;
  }
  return { kind: "rows", rows, added: plus, removed: minus };
}

/** The shortest edit script from `a` to `b` as 0 (same line), 1 (remove from
 * `a`) and 2 (add from `b`), or null when it would take more than `budget`
 * work or more than MAX_EDITS edits. */
function editScript(a: Int32Array, b: Int32Array, budget: number): Uint8Array | null {
  const n = a.length;
  const m = b.length;
  const most = Math.min(n + m, MAX_EDITS);
  const offset = most + 1;
  const v = new Int32Array(2 * most + 3);
  const trace: Int32Array[] = [];
  let work = 0;
  let found = -1;
  outer: for (let d = 0; d <= most; d++) {
    work += n + m;
    if (d > 0 && work > budget) return null;
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && (v[offset + k - 1] ?? 0) < (v[offset + k + 1] ?? 0)) ? (v[offset + k + 1] ?? 0) : (v[offset + k - 1] ?? 0) + 1;
      let y = x - k;
      while (x < n && y < m && a[x] === b[y]) {
        x++;
        y++;
      }
      v[offset + k] = x;
      if (x >= n && y >= m) {
        trace.push(v.slice(offset - d, offset + d + 1));
        found = d;
        break outer;
      }
    }
    trace.push(v.slice(offset - d, offset + d + 1));
  }
  if (found < 0) return null;
  const ops: number[] = [];
  let x = n;
  let y = m;
  for (let d = found; d > 0; d--) {
    const previous = trace[d - 1] as Int32Array;
    const at = (k: number) => previous[k + (d - 1)] ?? 0;
    const k = x - y;
    const down = k === -d || (k !== d && at(k - 1) < at(k + 1));
    const fromK = down ? k + 1 : k - 1;
    const fromX = at(fromK);
    const fromY = fromX - fromK;
    while (x > fromX + (down ? 0 : 1) && y > fromY + (down ? 1 : 0)) {
      ops.push(0);
      x--;
      y--;
    }
    if (down) {
      ops.push(2);
      y--;
    } else {
      ops.push(1);
      x--;
    }
  }
  while (x > 0 && y > 0) {
    ops.push(0);
    x--;
    y--;
  }
  return Uint8Array.from(ops.reverse());
}

/** A run of rows on screen, or unchanged rows folded into one "Show N lines" line. */
export type DiffSegment = { kind: "rows"; rows: DiffRow[] } | { kind: "fold"; rows: DiffRow[] };

/** Fold unchanged stretches longer than `2 × context + 4` lines, keeping
 * `context` lines on each side of every change; a document without changes
 * is one fold. */
export function foldRows(rows: DiffRow[], context = 3): DiffSegment[] {
  const segments: DiffSegment[] = [];
  let shown: DiffRow[] = [];
  let index = 0;
  while (index < rows.length) {
    if ((rows[index] as DiffRow).type !== "ctx") {
      shown.push(rows[index++] as DiffRow);
      continue;
    }
    let end = index;
    while (end < rows.length && (rows[end] as DiffRow).type === "ctx") end++;
    const before = index === 0 ? 0 : context;
    const after = end === rows.length ? 0 : context;
    if (end - index > before + after + 4) {
      shown.push(...rows.slice(index, index + before));
      if (shown.length) segments.push({ kind: "rows", rows: shown });
      segments.push({ kind: "fold", rows: rows.slice(index + before, end - after) });
      shown = rows.slice(end - after, end);
    } else shown.push(...rows.slice(index, end));
    index = end;
  }
  if (shown.length) segments.push({ kind: "rows", rows: shown });
  return segments;
}
