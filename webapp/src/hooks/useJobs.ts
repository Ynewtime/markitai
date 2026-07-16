import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createJob,
  deleteJobItem,
  enhanceJobItem,
  fetchJobSnapshot,
  jobEventsUrl,
  retryJobItem,
} from "../api/client";
import type {
  CreateJobResponse,
  ItemKind,
  ItemPayload,
  ItemStatus,
  JobOptions,
  JobPayload,
  JobSnapshot,
  JobStatus,
} from "../api/types";
import { serverTimestampMs } from "../lib/format";
import { notifyJobDone, requestNotifyPermission } from "../lib/notify";

export { serverTimestampMs } from "../lib/format";

/** sessionStorage seeds so an F5 mid-job can rebuild the ledger and re-attach
 * to still-running jobs (the server replays a snapshot on connect). */
const SESSION_KEY = "markitai.session";

interface StoredSeed {
  itemId: string;
  name: string;
  kind: ItemKind;
  sizeBytes: number | null;
  /** Pre-in-place-retry clients marked the superseded source seed this way. */
  retried?: boolean;
}

interface StoredJob {
  jobId: string;
  items: StoredSeed[];
}

interface LegacyRetryRemoval {
  jobId: string;
  itemId: string;
}

/** Collapse session seeds written by the old "retry creates a job" client.
 * The marker is authoritative: it was only persisted after the replacement
 * job had been created and appended. New retries never write this marker. */
export function migrateLegacyRetrySeeds(stored: StoredJob[]): {
  jobs: StoredJob[];
  removals: LegacyRetryRemoval[];
  suppressedJobIds: Set<string>;
} {
  const removed = new Set<string>();
  const removals: LegacyRetryRemoval[] = [];

  stored.forEach((job, jobIndex) => {
    for (const seed of job.items) {
      if (seed.retried !== true) continue;
      const replacementExists = stored.slice(jobIndex + 1).some(
        (candidate) =>
          candidate.items.length === 1 &&
          candidate.items.some(
            (next) =>
              next.name === seed.name &&
              next.kind === seed.kind &&
              next.sizeBytes === seed.sizeBytes,
          ),
      );
      if (!replacementExists) continue;
      removed.add(`${job.jobId}/${seed.itemId}`);
      removals.push({ jobId: job.jobId, itemId: seed.itemId });
    }
  });

  const suppressedJobIds = new Set<string>();
  const jobs = stored.flatMap((job) => {
    const items = job.items.filter(
      (seed) => !removed.has(`${job.jobId}/${seed.itemId}`),
    );
    if (items.length > 0) return [{ ...job, items }];
    if (job.items.length > 0) suppressedJobIds.add(job.jobId);
    return [];
  });
  return { jobs, removals, suppressedJobIds };
}

function readStoredJobs(): StoredJob[] {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (raw === null) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as StoredJob[];
  } catch {
    return [];
  }
}

function writeStoredJobs(jobs: StoredJob[]): void {
  try {
    if (jobs.length === 0) sessionStorage.removeItem(SESSION_KEY);
    else sessionStorage.setItem(SESSION_KEY, JSON.stringify(jobs));
  } catch {
    /* sessionStorage unavailable */
  }
}

function seedItem(jobId: string, seed: StoredSeed): SessionItem {
  return {
    key: `${jobId}/${seed.itemId}`,
    jobId,
    itemId: seed.itemId,
    name: seed.name,
    kind: seed.kind,
    status: "queued",
    error: null,
    output: null,
    durationMs: null,
    finishedAt: null,
    costUsd: null,
    llmEnhanced: false,
    operation: "convert",
    skipped: false,
    skipReason: null,
    sizeBytes: seed.sizeBytes,
    startedAt: null,
  };
}

/** One row across the session ledger. `key` is `${jobId}/${itemId}`. */
export interface SessionItem {
  key: string;
  jobId: string;
  itemId: string;
  name: string;
  kind: ItemKind;
  status: ItemStatus;
  error: string | null;
  output: string | null;
  durationMs: number | null;
  /** Completion timestamp reported by the server. */
  finishedAt: string | null;
  costUsd: number | null;
  llmEnhanced: boolean;
  operation: "convert" | "retry" | "enhance";
  /** Completed as a skip (neutral chip, non-selectable). */
  skipped: boolean;
  skipReason: string | null;
  /** Client-known upload size (files only; URLs have none). */
  sizeBytes: number | null;
  /** Wall-clock ms when we first observed the item running (live timer). */
  startedAt: number | null;
}

export interface SessionJob {
  jobId: string;
  status: JobStatus;
  createdAt: string | null;
  options: JobOptions;
}

/** Keep each job's input order, but place the most recently created job first. */
export function newestSessionItemsFirst(
  items: SessionItem[],
  jobs: Record<string, SessionJob>,
): SessionItem[] {
  const itemPosition = new Map(items.map((item, index) => [item.key, index]));
  const firstJobPosition = new Map<string, number>();
  items.forEach((item, index) => {
    if (!firstJobPosition.has(item.jobId)) firstJobPosition.set(item.jobId, index);
  });
  const jobTime = (jobId: string) => {
    const createdAt = jobs[jobId]?.createdAt;
    if (createdAt === null || createdAt === undefined) return Number.POSITIVE_INFINITY;
    return serverTimestampMs(createdAt) ?? 0;
  };

  return [...items].sort((left, right) => {
    if (left.jobId === right.jobId) {
      return (itemPosition.get(left.key) ?? 0) - (itemPosition.get(right.key) ?? 0);
    }
    const leftTime = jobTime(left.jobId);
    const rightTime = jobTime(right.jobId);
    if (leftTime > rightTime) return -1;
    if (leftTime < rightTime) return 1;
    return (
      (firstJobPosition.get(right.jobId) ?? 0) -
      (firstJobPosition.get(left.jobId) ?? 0)
    );
  });
}

export interface SessionStats {
  /** Completed with a real result — skips are counted separately. */
  done: number;
  skipped: number;
  failed: number;
  total: number;
  costTotal: number;
  hasCost: boolean;
  doneDurationMs: number;
}

function mergeItem(prev: SessionItem, p: ItemPayload, now: number): SessionItem {
  return {
    ...prev,
    name: p.name,
    kind: p.kind,
    status: p.status,
    error: p.error,
    output: p.output,
    durationMs: p.duration_ms,
    finishedAt: p.finished_at,
    costUsd: p.cost_usd,
    llmEnhanced: p.llm_enhanced,
    operation: p.operation,
    skipped: p.skipped,
    skipReason: p.skip_reason,
    startedAt:
      p.status === "running"
        ? prev.status === "running"
          ? (prev.startedAt ?? now)
          : now
        : null,
  };
}

function itemFromPayload(jobId: string, payload: ItemPayload, now: number): SessionItem {
  return mergeItem(
    seedItem(jobId, {
      itemId: payload.item_id,
      name: payload.name,
      kind: payload.kind,
      sizeBytes: null,
    }),
    payload,
    now,
  );
}

/**
 * Session-level job state: every drop/convert action POSTs a new job; items
 * from all session jobs accumulate in one ledger. One EventSource per active
 * job, closed on the terminal `job` event. EventSource auto-reconnects and
 * the server replays a `snapshot` first — merges are idempotent.
 */
export function useJobs() {
  const [items, setItems] = useState<SessionItem[]>([]);
  const [jobs, setJobs] = useState<Record<string, SessionJob>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [suppressedHistoryIds, setSuppressedHistoryIds] = useState(
    () => new Set<string>(),
  );
  const [historyRevision, setHistoryRevision] = useState(0);
  const sourcesRef = useRef<Map<string, EventSource>>(new Map());
  // One completion notification per job; both the snapshot and the final
  // `job` frame can report the terminal state, so terminality is latched.
  const notifiedRef = useRef<Set<string>>(new Set());

  const closeEvents = useCallback((jobId: string) => {
    const es = sourcesRef.current.get(jobId);
    if (es !== undefined) {
      es.close();
      sourcesRef.current.delete(jobId);
    }
  }, []);

  /** Terminal state observed on a live event stream (submitted or re-attached
   * running jobs only; persisted history never opens a stream and cannot
   * notify. notifyJobDone itself checks hidden + permission. */
  const notifyTerminal = useCallback((jobId: string, done: number, failed: number) => {
    if (notifiedRef.current.has(jobId)) return;
    notifiedRef.current.add(jobId);
    notifyJobDone(failed > 0 ? `${done} done · ${failed} failed` : `${done} done`);
  }, []);

  const openEvents = useCallback(
    (jobId: string) => {
      closeEvents(jobId);
      const es = new EventSource(jobEventsUrl(jobId));
      sourcesRef.current.set(jobId, es);

      es.addEventListener("snapshot", (e) => {
        const snap = JSON.parse((e as MessageEvent<string>).data) as JobSnapshot;
        const now = Date.now();
        setJobs((prev) => ({
          ...prev,
          [jobId]: {
            jobId,
            status: snap.status,
            createdAt: snap.created_at,
            options: snap.options,
          },
        }));
        setItems((prev) =>
          prev.map((item) => {
            if (item.jobId !== jobId) return item;
            const p = snap.items.find((x) => x.item_id === item.itemId);
            return p === undefined ? item : mergeItem(item, p, now);
          }),
        );
        if (snap.status !== "running") {
          notifyTerminal(jobId, snap.done, snap.failed);
          closeEvents(jobId);
        }
      });

      es.addEventListener("item", (e) => {
        const p = JSON.parse((e as MessageEvent<string>).data) as ItemPayload;
        const now = Date.now();
        setItems((prev) =>
          prev.map((item) =>
            item.jobId === jobId && item.itemId === p.item_id
              ? mergeItem(item, p, now)
              : item,
          ),
        );
      });

      es.addEventListener("job", (e) => {
        const p = JSON.parse((e as MessageEvent<string>).data) as JobPayload;
        setJobs((prev) => {
          const job = prev[jobId];
          return job === undefined
            ? prev
            : { ...prev, [jobId]: { ...job, status: p.status } };
        });
        if (p.status !== "running") {
          notifyTerminal(jobId, p.done, p.failed);
          closeEvents(jobId);
        }
      });
    },
    [closeEvents, notifyTerminal],
  );

  /** Merge a freshly created POST /api/jobs response into the session:
   * seed rows, persist the seeds, attach the event stream. */
  const adopt = useCallback(
    (res: CreateJobResponse, sizes: (number | null)[], options: JobOptions) => {
      const jobId = res.job_id;
      setJobs((prev) => ({
        ...prev,
        [jobId]: { jobId, status: "running", createdAt: null, options },
      }));
      const seeds: StoredSeed[] = res.items.map((it, idx) => ({
        itemId: it.item_id,
        name: it.name,
        kind: it.kind,
        sizeBytes: sizes[idx] ?? null,
      }));
      setItems((prev) => [...prev, ...seeds.map((s) => seedItem(jobId, s))]);
      writeStoredJobs([...readStoredJobs(), { jobId, items: seeds }]);
      openEvents(jobId);
    },
    [openEvents],
  );

  /** POST a new job. Returns true when created (errors land in submitError). */
  const submit = useCallback(
    async (files: File[], urls: string[], options: JobOptions): Promise<boolean> => {
      requestNotifyPermission(); // first submit is the moment to ask, never load
      setSubmitError(null);
      try {
        const res = await createJob(files, urls, options);
        // Backend item order is files-then-urls, matching our FormData order.
        adopt(
          res,
          res.items.map((it, idx) =>
            it.kind === "file" ? (files[idx]?.size ?? null) : null,
          ),
          options,
        );
        return true;
      } catch (e) {
        setSubmitError(e instanceof Error ? e.message : String(e));
        return false;
      }
    },
    [adopt],
  );

  /** Queue a failed or skipped item in place, reset its row, and reattach to
   * the same job's SSE stream. No duplicate seed or session job is created. */
  const retry = useCallback(
    async (item: SessionItem, options?: JobOptions): Promise<string | null> => {
      requestNotifyPermission();
      try {
        if (options === undefined) await retryJobItem(item.jobId, item.itemId);
        else await retryJobItem(item.jobId, item.itemId, options);
        notifiedRef.current.delete(item.jobId);
        setJobs((prev) => {
          const job = prev[item.jobId];
          return job === undefined
            ? prev
            : { ...prev, [item.jobId]: { ...job, status: "running" } };
        });
        setItems((prev) =>
          prev.map((candidate) =>
            candidate.key === item.key
              ? {
                  ...candidate,
                  status: "queued",
                  error: null,
                  output: null,
                  durationMs: null,
                  finishedAt: null,
                  costUsd: null,
                  llmEnhanced: false,
                  operation: "retry",
                  skipped: false,
                  skipReason: null,
                  startedAt: null,
                }
              : candidate,
          ),
        );
        openEvents(item.jobId);
        return null;
      } catch (e) {
        return e instanceof Error ? e.message : String(e);
      }
    },
    [openEvents],
  );

  /** Re-run a completed base result with LLM required, in the same row. */
  const enhance = useCallback(
    async (item: SessionItem, options: JobOptions): Promise<string | null> => {
      requestNotifyPermission();
      try {
        await enhanceJobItem(item.jobId, item.itemId, options);
        notifiedRef.current.delete(item.jobId);
        setJobs((previous) => {
          const job = previous[item.jobId];
          return job === undefined
            ? previous
            : { ...previous, [item.jobId]: { ...job, status: "running" } };
        });
        setItems((previous) =>
          previous.map((candidate) =>
            candidate.key === item.key
              ? {
                  ...candidate,
                  status: "queued",
                  error: null,
                  output: null,
                  durationMs: null,
                  finishedAt: null,
                  costUsd: null,
                  llmEnhanced: false,
                  operation: "enhance",
                  skipped: false,
                  skipReason: null,
                  startedAt: null,
                }
              : candidate,
          ),
        );
        openEvents(item.jobId);
        return null;
      } catch (error) {
        return error instanceof Error ? error.message : String(error);
      }
    },
    [openEvents],
  );

  /** Reattach one archived failed/skipped item to the current ledger before
   * retrying it in place. The whole snapshot keeps sibling rows visible. */
  const retryArchived = useCallback(
    async (
      snapshot: JobSnapshot,
      itemId: string,
      retryOptions?: JobOptions,
    ): Promise<string | null> => {
      requestNotifyPermission();
      try {
        if (retryOptions === undefined) {
          await retryJobItem(snapshot.job_id, itemId);
        } else {
          await retryJobItem(snapshot.job_id, itemId, retryOptions);
        }
        const now = Date.now();
        const restored = snapshot.items.map((payload) => {
          const item = itemFromPayload(snapshot.job_id, payload, now);
          return payload.item_id === itemId
            ? {
                ...item,
                status: "queued" as const,
                error: null,
                output: null,
                durationMs: null,
                finishedAt: null,
                costUsd: null,
                llmEnhanced: false,
                operation: "retry" as const,
                skipped: false,
                skipReason: null,
                startedAt: null,
              }
            : item;
        });
        const options: JobOptions = retryOptions ?? {
          preset: snapshot.options.preset ?? null,
          llm: snapshot.options.llm ?? null,
          ocr: snapshot.options.ocr ?? null,
        };
        notifiedRef.current.delete(snapshot.job_id);
        setJobs((previous) => ({
          ...previous,
          [snapshot.job_id]: {
            jobId: snapshot.job_id,
            status: "running",
            createdAt: snapshot.created_at,
            options,
          },
        }));
        setItems((previous) => [
          ...previous.filter((item) => item.jobId !== snapshot.job_id),
          ...restored,
        ]);
        const seeds: StoredSeed[] = restored.map((item) => ({
          itemId: item.itemId,
          name: item.name,
          kind: item.kind,
          sizeBytes: null,
        }));
        writeStoredJobs([
          ...readStoredJobs().filter((job) => job.jobId !== snapshot.job_id),
          { jobId: snapshot.job_id, items: seeds },
        ]);
        openEvents(snapshot.job_id);
        return null;
      } catch (error) {
        return error instanceof Error ? error.message : String(error);
      }
    },
    [openEvents],
  );

  /** Adopt an archived base result, then enhance it in place with LLM. */
  const enhanceArchived = useCallback(
    async (
      snapshot: JobSnapshot,
      itemId: string,
      options: JobOptions,
    ): Promise<string | null> => {
      requestNotifyPermission();
      try {
        await enhanceJobItem(snapshot.job_id, itemId, options);
        const now = Date.now();
        const restored = snapshot.items.map((payload) => {
          const item = itemFromPayload(snapshot.job_id, payload, now);
          return payload.item_id === itemId
            ? {
                ...item,
                status: "queued" as const,
                error: null,
                output: null,
                durationMs: null,
                finishedAt: null,
                costUsd: null,
                llmEnhanced: false,
                operation: "enhance" as const,
                skipped: false,
                skipReason: null,
                startedAt: null,
              }
            : item;
        });
        notifiedRef.current.delete(snapshot.job_id);
        setJobs((previous) => ({
          ...previous,
          [snapshot.job_id]: {
            jobId: snapshot.job_id,
            status: "running",
            createdAt: snapshot.created_at,
            options: snapshot.options,
          },
        }));
        setItems((previous) => [
          ...previous.filter((item) => item.jobId !== snapshot.job_id),
          ...restored,
        ]);
        const seeds: StoredSeed[] = restored.map((item) => ({
          itemId: item.itemId,
          name: item.name,
          kind: item.kind,
          sizeBytes: null,
        }));
        writeStoredJobs([
          ...readStoredJobs().filter((job) => job.jobId !== snapshot.job_id),
          { jobId: snapshot.job_id, items: seeds },
        ]);
        openEvents(snapshot.job_id);
        return null;
      } catch (error) {
        return error instanceof Error ? error.message : String(error);
      }
    },
    [openEvents],
  );

  /** Permanently delete one terminal row. The server removes the whole job
   * directory when this was its last item. */
  const deleteItem = useCallback(
    async (item: SessionItem): Promise<string | null> => {
      try {
        await deleteJobItem(item.jobId, item.itemId);
        const wasLast = !items.some(
          (candidate) => candidate.jobId === item.jobId && candidate.key !== item.key,
        );
        setItems((prev) => prev.filter((candidate) => candidate.key !== item.key));
        if (wasLast) {
          closeEvents(item.jobId);
          setJobs((prev) => {
            const { [item.jobId]: _deleted, ...rest } = prev;
            return rest;
          });
        }
        writeStoredJobs(
          readStoredJobs().flatMap((job) => {
            if (job.jobId !== item.jobId) return [job];
            const seeds = job.items.filter((seed) => seed.itemId !== item.itemId);
            return seeds.length === 0 ? [] : [{ ...job, items: seeds }];
          }),
        );
        return null;
      } catch (e) {
        return e instanceof Error ? e.message : String(e);
      }
    },
    [closeEvents, items],
  );

  const clear = useCallback(() => {
    for (const es of sourcesRef.current.values()) es.close();
    sourcesRef.current.clear();
    setItems([]);
    setJobs({});
    setSubmitError(null);
    writeStoredJobs([]);
  }, []);

  /** Keep running jobs reachable; remove only whole terminal jobs. */
  const clearSettled = useCallback(() => {
    const terminalIds = new Set(
      Object.values(jobs)
        .filter((job) => job.status !== "running")
        .map((job) => job.jobId),
    );
    if (terminalIds.size === 0) return;
    setItems((previous) => previous.filter((item) => !terminalIds.has(item.jobId)));
    setJobs((previous) =>
      Object.fromEntries(
        Object.entries(previous).filter(([jobId]) => !terminalIds.has(jobId)),
      ),
    );
    setSubmitError(null);
    writeStoredJobs(readStoredJobs().filter((job) => !terminalIds.has(job.jobId)));
  }, [jobs]);

  // ---- session restore (F5 mid-job): seed the ledger from sessionStorage,
  // then reconcile each job against the server. 404 (server restarted) drops
  // the job silently; running jobs re-attach via EventSource snapshot replay.
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    const migration = migrateLegacyRetrySeeds(readStoredJobs());
    const stored = migration.jobs;
    if (migration.removals.length > 0) {
      writeStoredJobs(stored);
      setSuppressedHistoryIds(migration.suppressedJobIds);
      void (async () => {
        for (const removal of migration.removals) {
          try {
            await deleteJobItem(removal.jobId, removal.itemId);
          } catch {
            // Best effort: a missing old job is already the desired result,
            // and the superseded row stays hidden for this browser session.
          }
        }
        setHistoryRevision((revision) => revision + 1);
      })();
    }
    if (stored.length === 0) return;

    setItems((prev) =>
      prev.length > 0
        ? prev
        : stored.flatMap((j) => j.items.map((s) => seedItem(j.jobId, s))),
    );
    setJobs((prev) => {
      const next = { ...prev };
      for (const j of stored) {
        next[j.jobId] ??= {
          jobId: j.jobId,
          status: "running",
          createdAt: null,
          options: { preset: null, llm: null, ocr: null },
        };
      }
      return next;
    });

    const dropJob = (jobId: string) => {
      setItems((prev) => prev.filter((i) => i.jobId !== jobId));
      setJobs((prev) => {
        const { [jobId]: _gone, ...rest } = prev;
        return rest;
      });
      writeStoredJobs(readStoredJobs().filter((j) => j.jobId !== jobId));
    };

    for (const j of stored) {
      void fetchJobSnapshot(j.jobId).then(
        (snap) => {
          if (snap === null) {
            dropJob(j.jobId);
            return;
          }
          const now = Date.now();
          setJobs((prev) => ({
            ...prev,
            [j.jobId]: {
              jobId: j.jobId,
              status: snap.status,
              createdAt: snap.created_at,
              options: snap.options,
            },
          }));
          setItems((prev) =>
            prev.map((item) => {
              if (item.jobId !== j.jobId) return item;
              const p = snap.items.find((x) => x.item_id === item.itemId);
              return p === undefined ? item : mergeItem(item, p, now);
            }),
          );
          if (snap.status === "running") openEvents(j.jobId);
        },
        () => dropJob(j.jobId), // unreachable server state — treat like 404
      );
    }
  }, [openEvents]);

  useEffect(() => {
    const sources = sourcesRef.current;
    return () => {
      for (const es of sources.values()) es.close();
      sources.clear();
    };
  }, []);

  // Live timer for running rows: tick only while something is in flight.
  const anyActive = items.some(
    (i) => i.status === "running" || i.status === "queued",
  );
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!anyActive) return;
    const id = window.setInterval(() => setNow(Date.now()), 150);
    return () => window.clearInterval(id);
  }, [anyActive]);

  const stats = useMemo<SessionStats>(() => {
    let done = 0;
    let skipped = 0;
    let failed = 0;
    let costTotal = 0;
    let hasCost = false;
    let doneDurationMs = 0;
    for (const i of items) {
      if (i.status === "done") {
        if (i.skipped) skipped += 1;
        else {
          done += 1;
          if (i.durationMs !== null) doneDurationMs += i.durationMs;
        }
      }
      if (i.status === "error") failed += 1;
      if (i.llmEnhanced) hasCost = true;
      if (i.costUsd !== null && i.costUsd > 0) {
        hasCost = true;
        costTotal += i.costUsd;
      }
    }
    return {
      done,
      skipped,
      failed,
      total: items.length,
      costTotal,
      hasCost,
      doneDurationMs,
    };
  }, [items]);

  const running = useMemo(
    () => Object.values(jobs).some((j) => j.status === "running"),
    [jobs],
  );
  const terminalJobCount = useMemo(
    () => Object.values(jobs).filter((job) => job.status !== "running").length,
    [jobs],
  );
  const orderedItems = useMemo(
    () => newestSessionItemsFirst(items, jobs),
    [items, jobs],
  );

  /** Items still queued or running (drives the clear-confirm step). */
  const activeCount = useMemo(
    () =>
      items.reduce(
        (n, i) => (i.status === "running" || i.status === "queued" ? n + 1 : n),
        0,
      ),
    [items],
  );

  return {
    items: orderedItems,
    jobs,
    stats,
    running,
    activeCount,
    now,
    submit,
    retry,
    retryArchived,
    enhance,
    enhanceArchived,
    deleteItem,
    submitError,
    clear,
    clearSettled,
    terminalJobCount,
    suppressedHistoryIds,
    historyRevision,
  };
}
