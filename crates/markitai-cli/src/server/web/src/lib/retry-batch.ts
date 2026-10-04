import type { Dict } from "../i18n/index.ts";
import type { JobOptions } from "../api/types.ts";
import { publicOptions, withOcrFor } from "./options.ts";
import { failedToRetry, type SessionItem, type SessionJob } from "./session.ts";

export interface RetryBatchEntry { readonly item: SessionItem; readonly options: JobOptions }

/** The approval belongs to the exact retryable identities and options captured at this click. */
export function prepareRetryBatch(items: readonly SessionItem[], jobs: Readonly<Record<string, SessionJob>>, fallback: JobOptions): readonly RetryBatchEntry[] {
  return Object.freeze(failedToRetry([...items]).map((item) => {
    const base = publicOptions(item.options ?? jobs[item.jobId]?.options ?? fallback);
    const options = item.skipped && item.skipReason === "image_only" ? withOcrFor(base) : base;
    return Object.freeze({ item: Object.freeze({ ...item }), options: Object.freeze(options) });
  }));
}

/** Cancellation sends nothing, including native entries from the same mixed batch. */
export async function submitRetryBatch(batch: readonly RetryBatchEntry[], authorized: readonly JobOptions[] | null,
  send: (item: SessionItem, options: JobOptions) => Promise<unknown>): Promise<{ attempted: number; failed: number } | null> {
  if (authorized === null) return null;
  if (authorized.length !== batch.length) throw new Error("Retry approval does not match the captured batch");
  let failed = 0;
  for (let index = 0; index < batch.length; index++) {
    if ((await send(batch[index].item, authorized[index])) !== null) failed++;
  }
  return { attempted: batch.length, failed };
}

/** Announce the captured selection, even after rows have already been requeued. */
export function retryBatchAnnouncement(batch: readonly RetryBatchEntry[], result: { attempted: number; failed: number }, t: Dict): string {
  const stopped = batch.some(({ item }) => item.skipReason === "user_stopped");
  if (stopped) return result.failed ? t.announceRetryStoppedFailed(result.attempted, result.failed) : t.announceRetryStopped(result.attempted);
  return result.failed ? t.announceRetryAllFailed(result.attempted, result.failed) : t.announceRetryAll(result.attempted);
}
