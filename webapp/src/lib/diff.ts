/** Hand-rolled line-level LCS diff (no dependencies). Callers cap input at
 * MAX_DIFF_LINES total lines before calling; under that cap the DP table
 * stays small enough after common prefix/suffix trimming. */

export const MAX_DIFF_LINES = 5000;

export interface DiffLine {
  type: "ctx" | "add" | "del";
  text: string;
  /** 1-based line number in the old (base) text; null for added lines. */
  aNo: number | null;
  /** 1-based line number in the new (llm) text; null for removed lines. */
  bNo: number | null;
}

export function diffLines(aText: string, bText: string): DiffLine[] {
  const a = aText.split("\n");
  const b = bText.split("\n");

  // Trim the common prefix/suffix so the DP table only covers the middle.
  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start += 1;
  let endA = a.length;
  let endB = b.length;
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    endA -= 1;
    endB -= 1;
  }
  const midA = a.slice(start, endA);
  const midB = b.slice(start, endB);
  const n = midA.length;
  const m = midB.length;

  // LCS lengths, bottom-up: dp[i][j] = LCS(midA[i:], midB[j:]).
  const width = m + 1;
  const dp = new Uint32Array((n + 1) * width);
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      dp[i * width + j] =
        midA[i] === midB[j]
          ? (dp[(i + 1) * width + j + 1] ?? 0) + 1
          : Math.max(dp[(i + 1) * width + j] ?? 0, dp[i * width + j + 1] ?? 0);
    }
  }

  const out: DiffLine[] = [];
  let aNo = 1;
  let bNo = 1;
  const ctx = (text: string) => {
    out.push({ type: "ctx", text, aNo, bNo });
    aNo += 1;
    bNo += 1;
  };
  const del = (text: string) => {
    out.push({ type: "del", text, aNo, bNo: null });
    aNo += 1;
  };
  const add = (text: string) => {
    out.push({ type: "add", text, aNo: null, bNo });
    bNo += 1;
  };

  for (let k = 0; k < start; k += 1) ctx(a[k] ?? "");
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (midA[i] === midB[j]) {
      ctx(midA[i] ?? "");
      i += 1;
      j += 1;
    } else if ((dp[(i + 1) * width + j] ?? 0) >= (dp[i * width + j + 1] ?? 0)) {
      del(midA[i] ?? "");
      i += 1;
    } else {
      add(midB[j] ?? "");
      j += 1;
    }
  }
  while (i < n) {
    del(midA[i] ?? "");
    i += 1;
  }
  while (j < m) {
    add(midB[j] ?? "");
    j += 1;
  }
  for (let k = endA; k < a.length; k += 1) ctx(a[k] ?? "");
  return out;
}
