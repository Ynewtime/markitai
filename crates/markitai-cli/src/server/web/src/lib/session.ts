// The session ledger: rows of every job submitted from this tab, merged with
// the service's saved jobs into one chronological list. Pure functions only.
import type {
  AttemptDiagnostics,
  RemoteProcessing,
  HistoryEntry,
  ItemKind,
  ItemPayload,
  ItemStatus,
  JobOptions,
  JobSnapshot,
  JobStatus,
  Pricing,
  RerunFailure,
} from "../api/types.ts";
import { publicOptions } from "./options.ts";
import { isUnsupported } from "../i18n/errors.ts";
import { displayName, timestampMs } from "./format.ts";

/** A terminal fetch can outlive its stream or a user's next attempt. */
export async function settleCurrentJobSnapshot(
  pending: Promise<JobSnapshot | null>,
  current: () => boolean,
  receive: (snapshot: JobSnapshot) => void,
  complete: (snapshot: JobSnapshot) => void,
): Promise<void> {
  const snapshot = await pending;
  if (!current() || snapshot === null) return;
  receive(snapshot);
  if (snapshot.status !== "running") complete(snapshot);
}

export interface SessionItem {
  /** `${jobId}/${itemId}` */
  key: string;
  jobId: string;
  itemId: string;
  name: string;
  kind: ItemKind;
  status: ItemStatus;
  error: string | null;
  errorCode: string | null;
  output: string | null;
  durationMs: number | null;
  finishedAt: string | null;
  costUsd: number | null;
  pricing: Pricing | null;
  diagnostics: AttemptDiagnostics | null;
  rerunFailure: RerunFailure | null;
  llmEnhanced: boolean;
  operation: "convert" | "retry" | "enhance";
  skipped: boolean;
  skipReason: string | null;
  retryable: boolean;
  warnings: string[];
  remoteProcessing?: RemoteProcessing | null;
  options?: JobOptions;
  /** Upload size known to this tab (files only). */
  sizeBytes: number | null;
  /** When this tab first saw the item running, for the live timer. */
  startedAt: number | null;
}

export interface SessionJob {
  jobId: string;
  status: JobStatus;
  createdAt: string | null;
  options: JobOptions;
  persistenceError: string | null;
  /** What the submission called itself (a folder or URL list name), for a group. */
  label: string | null;
}

export interface Seed {
  itemId: string;
  name: string;
  kind: ItemKind;
  sizeBytes: number | null;
}

export function seedItem(jobId: string, seed: Seed): SessionItem {
  return {
    key: `${jobId}/${seed.itemId}`,
    jobId,
    itemId: seed.itemId,
    name: seed.name,
    kind: seed.kind,
    status: "queued",
    error: null,
    errorCode: null,
    output: null,
    durationMs: null,
    finishedAt: null,
    costUsd: null,
    pricing: null,
    diagnostics: null,
    rerunFailure: null,
    llmEnhanced: false,
    operation: "convert",
    skipped: false,
    skipReason: null,
    retryable: true,
    warnings: [],
    sizeBytes: seed.sizeBytes,
    startedAt: null,
  };
}

/** Apply an item payload; unchanged rows keep their identity so memoized rows skip work. */
export function mergeItem(previous: SessionItem, payload: ItemPayload, now: number): SessionItem {
  // Presentation only: the service still persists error/cancelled. A stopped
  // item has no successful output and is counted apart from completed work.
  const stopped = payload.status === "error" && payload.output === null && !payload.rerun_failure &&
    (payload.error_code === "cancelled" || (!payload.error_code && payload.error === "cancelled (stopped by request)"));
  const startedAt =
    payload.status === "running" ? (previous.status === "running" ? (previous.startedAt ?? now) : now) : null;
  return {
    ...previous,
    name: payload.name,
    kind: payload.kind,
    status: stopped ? "done" : payload.status,
    error: payload.error,
    errorCode: payload.error_code ?? null,
    output: payload.output,
    durationMs: payload.duration_ms,
    finishedAt: payload.finished_at,
    costUsd: payload.cost_usd,
    pricing: payload.pricing ?? null,
    diagnostics: payload.diagnostics ?? null,
    remoteProcessing: payload.remote_processing ?? null,
    options: payload.options ? publicOptions(payload.options) : undefined,
    rerunFailure: payload.rerun_failure ?? null,
    llmEnhanced: payload.llm_enhanced,
    operation: payload.operation,
    skipped: stopped || payload.skipped,
    skipReason: stopped ? "user_stopped" : payload.skip_reason,
    retryable: payload.retryable,
    warnings: payload.warnings ?? [],
    startedAt,
  };
}

export function itemFromPayload(jobId: string, payload: ItemPayload, now: number, sizeBytes: number | null = null): SessionItem {
  return mergeItem(seedItem(jobId, { itemId: payload.item_id, name: payload.name, kind: payload.kind, sizeBytes }), payload, now);
}

/** A row queued again in place by retry or enhance. */
export function requeued(item: SessionItem, operation: "retry" | "enhance"): SessionItem {
  return {
    ...item,
    status: "queued",
    error: null,
    errorCode: null,
    output: null,
    durationMs: null,
    finishedAt: null,
    costUsd: null,
    pricing: null,
    diagnostics: null,
    rerunFailure: null,
    llmEnhanced: false,
    operation,
    skipped: false,
    skipReason: null,
    warnings: [],
    startedAt: null,
  };
}

/** The snapshot owns which items exist; local rows keep only what the page knows (sizes). */
export function reconcile(previous: SessionItem[], jobId: string, payloads: ItemPayload[], now: number): SessionItem[] {
  const local = new Map(previous.filter((item) => item.jobId === jobId).map((item) => [item.itemId, item]));
  const incoming = payloads.map((payload) => {
    const prior = local.get(payload.item_id);
    return prior ? mergeItem(prior, payload, now) : itemFromPayload(jobId, payload, now);
  });
  const first = previous.findIndex((item) => item.jobId === jobId);
  const others = previous.filter((item) => item.jobId !== jobId);
  others.splice(first < 0 ? others.length : first, 0, ...incoming);
  return others;
}

export interface SessionStats {
  /** Converted with a result; skips are counted apart. */
  done: number;
  skipped: number;
  failed: number;
  total: number;
  costTotal: number;
  hasCost: boolean;
  doneDurationMs: number;
}

export function sessionStats(items: SessionItem[]): SessionStats {
  const stats: SessionStats = { done: 0, skipped: 0, failed: 0, total: items.length, costTotal: 0, hasCost: false, doneDurationMs: 0 };
  for (const item of items) {
    if (item.status === "done") {
      if (item.skipped) stats.skipped++;
      else {
        stats.done++;
        if (item.durationMs !== null) stats.doneDurationMs += item.durationMs;
      }
    } else if (item.status === "error") stats.failed++;
    if (item.llmEnhanced) stats.hasCost = true;
    if (item.costUsd !== null && item.costUsd > 0) {
      stats.hasCost = true;
      stats.costTotal += item.costUsd;
    }
  }
  return stats;
}

/** Summary notifications use the same presentation as rows, not raw API error totals. */
export function snapshotStats(snapshot: JobSnapshot): SessionStats {
  return sessionStats(snapshot.items.map((item) => itemFromPayload(snapshot.job_id, item, 0)));
}

export const isSettled = (item: SessionItem): boolean => item.status === "done" || item.status === "error";
/** A reconnect can deliver a failed rerun directly as done → done. */
export const settledIdentity = (item: SessionItem): string | null =>
  isSettled(item) ? `${item.status}/${item.rerunFailure?.failed_at ?? item.finishedAt ?? ""}` : null;
export const isActive = (item: SessionItem): boolean => item.status === "queued" || item.status === "running";
export const isPreviewable = (item: SessionItem): boolean => item.status === "done" && item.output !== null && !item.skipped;

/** Retry helps a failed or skipped row, unless the source is gone, a batch is
 * pending, or the file type is simply not supported. */
export function canRetry(item: SessionItem): boolean {
  const eligible = item.status === "error" || (item.status === "done" && (item.skipped || item.rerunFailure !== null));
  return (
    eligible &&
    item.retryable &&
    item.skipReason !== "pending_batch" &&
    !isUnsupported({ error: item.error, error_code: item.errorCode })
  );
}

export const failedToRetry = (items: SessionItem[]): SessionItem[] => items.filter((item) => (item.status === "error" || item.status === "done" && item.skipped && item.skipReason === "user_stopped") && canRetry(item));

/** Original items still waiting for a conversion slot: what Stop remaining can stop. */
export function waitingJobs(items: SessionItem[], jobs: Record<string, SessionJob>): string[] {
  const ids = new Set<string>();
  for (const item of items) {
    if (item.status === "queued" && item.operation === "convert" && jobs[item.jobId]?.status === "running") ids.add(item.jobId);
  }
  return [...ids];
}

export type LedgerRow =
  | { kind: "session"; key: string; item: SessionItem }
  // One submission that carried several items: a folder, several files, a URL
  // list or a batch of URLs. It reads as one row and opens into its items.
  | { kind: "group"; key: string; jobId: string; items: SessionItem[]; label: string | null }
  | { kind: "archive"; key: string; entry: HistoryEntry };

/** The rows a group shows when it is open. */
export function groupItemRows(row: Extract<LedgerRow, { kind: "group" }>): LedgerRow[] {
  return row.items.map((item) => ({ kind: "session" as const, key: item.key, item }));
}

/** What a submitted batch is called in the ledger: the folder it came from, the
 * URL list it came from, or the number of items it holds. */
export function groupLabel(label: string | null | undefined, items: SessionItem[], t: { itemsNotice: (count: number) => string }): string {
  if (label) return label;
  const first = items[0] ? displayName(items[0].name) : "";
  return items.length > 1 ? `${first} +${items.length - 1}` : first || t.itemsNotice(items.length);
}

function activity(items: SessionItem[], createdAt: string | null | undefined): number {
  if (items.some(isActive)) return Number.POSITIVE_INFINITY;
  let latest: number | null = null;
  for (const item of items) {
    const at = timestampMs(item.finishedAt);
    if (at !== null && (latest === null || at > latest)) latest = at;
  }
  return latest ?? timestampMs(createdAt) ?? 0;
}

/** One chronological ledger: jobs ordered by their latest activity (running
 * first), items inside a job in input order, saved jobs folded in by the same rule. */
export function mergeLedger(items: SessionItem[], jobs: Record<string, SessionJob>, archive: HistoryEntry[]): LedgerRow[] {
  const groups = new Map<string, SessionItem[]>();
  for (const item of items) {
    const group = groups.get(item.jobId);
    if (group) group.push(item);
    else groups.set(item.jobId, [item]);
  }
  interface Group {
    jobId: string;
    at: number;
    created: number;
    rows: LedgerRow[];
  }
  const ordered: Group[] = [];
  for (const [jobId, group] of groups) {
    const createdAt = jobs[jobId]?.createdAt ?? null;
    const rows: LedgerRow[] =
      group.length > 1
        ? [{ kind: "group", key: `group:${jobId}`, jobId, items: group, label: jobs[jobId]?.label ?? null }]
        : group.map((item) => ({ kind: "session", key: item.key, item }));
    ordered.push({
      jobId,
      at: activity(group, createdAt),
      created: createdAt === null ? Number.POSITIVE_INFINITY : (timestampMs(createdAt) ?? 0),
      rows,
    });
  }
  for (const entry of archive) {
    // A job briefly in both stores during a retry shows once, as the live row.
    if (groups.has(entry.job_id)) continue;
    ordered.push({
      jobId: entry.job_id,
      at: timestampMs(entry.finished_at) ?? timestampMs(entry.created_at) ?? 0,
      created: timestampMs(entry.created_at) ?? 0,
      rows: [{ kind: "archive", key: `archive:${entry.job_id}`, entry }],
    });
  }
  ordered.sort((left, right) => right.at - left.at || right.created - left.created || left.jobId.localeCompare(right.jobId));
  return ordered.flatMap((group) => group.rows);
}

export type StatusFilter = "all" | "done" | "failed" | "skipped";
export const STATUS_FILTERS: StatusFilter[] = ["all", "done", "failed", "skipped"];

export function rowMatches(row: LedgerRow, filter: StatusFilter, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (row.kind === "group") {
    // A group matches when any of its items does, and opens to show them.
    return row.items.some((item) =>
      rowMatches({ kind: "session", key: item.key, item }, filter, query),
    ) || (!needle && filter === "all");
  }
  if (row.kind === "session") {
    const item = row.item;
    const status =
      filter === "all" ||
      (filter === "done" && item.status === "done" && !item.skipped) ||
      (filter === "failed" && item.status === "error") ||
      (filter === "skipped" && item.status === "done" && item.skipped);
    return status && (!needle || displayName(item.name).toLowerCase().includes(needle));
  }
  const entry = row.entry;
  const status =
    filter === "all" ||
    (filter === "done" && entry.done - entry.skipped > 0) ||
    (filter === "failed" && entry.failed > 0) ||
    (filter === "skipped" && entry.skipped > 0);
  return status && (!needle || entry.names_preview.some((name) => displayName(name).toLowerCase().includes(needle)));
}

/** Session seeds survive a reload in sessionStorage; the service is asked for the rest. */
export const SESSION_KEY = "markitai.session";

export interface StoredJob {
  jobId: string;
  /** What the submission called itself, so a reload keeps the group's name. */
  label?: string | null;
  items: Seed[];
}

interface SessionStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

function sessionStore(): SessionStore | null {
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

export function readSeeds(store: SessionStore | null = sessionStore()): StoredJob[] {
  try {
    const value: unknown = JSON.parse(store?.getItem(SESSION_KEY) ?? "[]");
    if (!Array.isArray(value)) return [];
    return value.filter(
      (job): job is StoredJob =>
        job !== null && typeof job === "object" && typeof job.jobId === "string" && Array.isArray(job.items),
    );
  } catch {
    return [];
  }
}

export function writeSeeds(jobs: StoredJob[], store: SessionStore | null = sessionStore()): void {
  try {
    if (jobs.length) store?.setItem(SESSION_KEY, JSON.stringify(jobs));
    else store?.removeItem(SESSION_KEY);
  } catch {
    /* The ledger lasts for this page only. */
  }
}
