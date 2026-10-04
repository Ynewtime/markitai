import type { Dict } from "../i18n/index.ts";
import type { HistoryEntry, JobSnapshot } from "../api/types.ts";
import { snapshotStats } from "./session.ts";

/** History's API totals include requested stops as errors. Resolve only the
 * currently loaded web summaries, using their existing per-item snapshots.
 * Four workers bound requests; a newer refresh/unmount stops scheduling and
 * discards old results. Failures retain the unverified summary with a visible
 * row error, rather than guessing that conversion errors were user stops. */
export async function resolveHistoryPresentation(
  entries: readonly HistoryEntry[],
  fetch: (jobId: string) => Promise<JobSnapshot | null>,
  current: () => boolean = () => true,
): Promise<{ entries: HistoryEntry[]; errors: Record<string, unknown> }> {
  const resolved = [...entries];
  const errors: Record<string, unknown> = {};
  const work = entries.map((entry, index) => ({ entry, index })).filter(({ entry }) => entry.origin === "web" && entry.failed > 0);
  let cursor = 0;
  const worker = async () => {
    while (current()) {
      const next = work[cursor++];
      if (!next) return;
      const { entry, index } = next;
      try {
        const snapshot = await fetch(entry.job_id);
        if (!current()) return;
        if (!snapshot || snapshot.job_id !== entry.job_id || snapshot.status === "running") throw new Error("Saved item details are unavailable; refresh to verify the summary");
        const stats = snapshotStats(snapshot);
        resolved[index] = { ...entry, total: stats.total, done: stats.done + stats.skipped, failed: stats.failed, skipped: stats.skipped, finished_at: snapshot.finished_at };
      } catch (error) {
        if (current()) errors[entry.job_id] = new HistorySummaryError(error);
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(4, work.length) }, worker));
  return { entries: resolved, errors };
}

export class HistorySummaryError extends Error {
  readonly detail: string;
  constructor(reason: unknown) {
    super("History summary unavailable");
    this.name = "HistorySummaryError";
    this.detail = reason instanceof Error ? reason.message : String(reason);
  }
}

export function historySummaryProblem(error: unknown, t: Dict): { text: string; detail: string } | null {
  return error instanceof HistorySummaryError ? { text: t.historySummaryUnavailable, detail: error.detail } : null;
}

/** Cleared action errors must not hide unresolved summary errors. */
export function mergeHistoryErrors(summary: Record<string, unknown>, actions: Record<string, unknown>): Record<string, unknown> {
  return { ...summary, ...Object.fromEntries(Object.entries(actions).filter(([, value]) => value != null)) };
}
