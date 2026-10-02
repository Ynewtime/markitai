// The session ledger: rows of every job submitted from this tab, merged with
// the service's saved jobs into one chronological list. Pure functions only.
import type {
  AttemptDiagnostics,
  HistoryEntry,
  ItemKind,
  ItemPayload,
  ItemStatus,
  JobOptions,
  JobStatus,
  Pricing,
} from "../api/types.ts";
import { isUnsupported } from "../i18n/errors.ts";
import { displayName, timestampMs } from "./format.ts";

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
  llmEnhanced: boolean;
  operation: "convert" | "retry" | "enhance";
  skipped: boolean;
  skipReason: string | null;
  retryable: boolean;
  warnings: string[];
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
  const startedAt =
    payload.status === "running" ? (previous.status === "running" ? (previous.startedAt ?? now) : now) : null;
  return {
    ...previous,
    name: payload.name,
    kind: payload.kind,
    status: payload.status,
    error: payload.error,
    errorCode: payload.error_code ?? null,
    output: payload.output,
    durationMs: payload.duration_ms,
    finishedAt: payload.finished_at,
    costUsd: payload.cost_usd,
    pricing: payload.pricing ?? null,
    diagnostics: payload.diagnostics ?? null,
    llmEnhanced: payload.llm_enhanced,
    operation: payload.operation,
    skipped: payload.skipped,
    skipReason: payload.skip_reason,
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

export const isSettled = (item: SessionItem): boolean => item.status === "done" || item.status === "error";
export const isActive = (item: SessionItem): boolean => item.status === "queued" || item.status === "running";
export const isPreviewable = (item: SessionItem): boolean => item.status === "done" && item.output !== null && !item.skipped;

/** Retry helps a failed or skipped row, unless the source is gone, a batch is
 * pending, or the file type is simply not supported. */
export function canRetry(item: SessionItem): boolean {
  const eligible = item.status === "error" || (item.status === "done" && item.skipped);
  return (
    eligible &&
    item.retryable &&
    item.skipReason !== "pending_batch" &&
    !isUnsupported({ error: item.error, error_code: item.errorCode })
  );
}

export const failedToRetry = (items: SessionItem[]): SessionItem[] => items.filter((item) => item.status === "error" && canRetry(item));

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
  | { kind: "archive"; key: string; entry: HistoryEntry };

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
    ordered.push({
      jobId,
      at: activity(group, createdAt),
      created: createdAt === null ? Number.POSITIVE_INFINITY : (timestampMs(createdAt) ?? 0),
      rows: group.map((item) => ({ kind: "session", key: item.key, item })),
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
