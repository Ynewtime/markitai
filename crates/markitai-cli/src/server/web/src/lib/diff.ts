// Line comparison of the base and LLM Markdown. Equal head and tail lines are
// trimmed first; the middle uses a longest-common-subsequence table, bounded so
// a large document is refused instead of freezing the page.

export const DIFF_LIMITS = Object.freeze({ lines: 5000, characters: 2 * 1024 * 1024, cells: 1_000_000 });

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
  const n = a.length - head - tail;
  const m = b.length - head - tail;
  if (n * m > limits.cells) return { kind: "too-large" };

  const rows: DiffRow[] = [];
  let aNo = 1;
  let bNo = 1;
  const same = (text: string) => rows.push({ type: "ctx", text, aNo: aNo++, bNo: bNo++ });
  const removed = (text: string) => rows.push({ type: "del", text, aNo: aNo++, bNo: null });
  const added = (text: string) => rows.push({ type: "add", text, aNo: null, bNo: bNo++ });

  for (let k = 0; k < head; k++) same(a[k] ?? "");
  // table[i][j] = common subsequence length of a[head+i..] and b[head+j..].
  const width = m + 1;
  const table = new Uint32Array((n + 1) * width);
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      table[i * width + j] =
        a[head + i] === b[head + j]
          ? (table[(i + 1) * width + j + 1] ?? 0) + 1
          : Math.max(table[(i + 1) * width + j] ?? 0, table[i * width + j + 1] ?? 0);
    }
  }
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[head + i] === b[head + j]) {
      same(a[head + i] ?? "");
      i++;
      j++;
    } else if (i < n && (j === m || (table[(i + 1) * width + j] ?? 0) >= (table[i * width + j + 1] ?? 0))) {
      removed(a[head + i] ?? "");
      i++;
    } else {
      added(b[head + j] ?? "");
      j++;
    }
  }
  for (let k = a.length - tail; k < a.length; k++) same(a[k] ?? "");
  let plus = 0;
  let minus = 0;
  for (const row of rows) {
    if (row.type === "add") plus++;
    else if (row.type === "del") minus++;
  }
  return { kind: "rows", rows, added: plus, removed: minus };
}
