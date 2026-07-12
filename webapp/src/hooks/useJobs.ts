import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createJob, fetchJobSnapshot, jobEventsUrl, retryJobItem } from "../api/client";
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
import { notifyJobDone, requestNotifyPermission } from "../lib/notify";

/** sessionStorage seeds so an F5 mid-job can rebuild the ledger and re-attach
 * to still-running jobs (the server replays a snapshot on connect). */
const SESSION_KEY = "markitai.session";

interface StoredSeed {
  itemId: string;
  name: string;
  kind: ItemKind;
  sizeBytes: number | null;
  /** The row was retried as a new job (ledger keeps the original row). */
  retried?: boolean;
}

interface StoredJob {
  jobId: string;
  items: StoredSeed[];
  /** Job was merged from the conversion history (server-side archive). */
  archived?: boolean;
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

function seedItem(jobId: string, seed: StoredSeed, archived: boolean): SessionItem {
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
    costUsd: null,
    skipped: false,
    skipReason: null,
    sizeBytes: seed.sizeBytes,
    startedAt: null,
    archived,
    retried: seed.retried === true,
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
  costUsd: number | null;
  /** Completed as a skip (neutral chip, non-selectable). */
  skipped: boolean;
  skipReason: string | null;
  /** Client-known upload size (files only; URLs have none). */
  sizeBytes: number | null;
  /** Wall-clock ms when we first observed the item running (live timer). */
  startedAt: number | null;
  /** Merged from the conversion history — rows carry a neutral badge. */
  archived: boolean;
  /** Retried into a new job — the original row keeps a neutral marker. */
  retried: boolean;
}

export interface SessionJob {
  jobId: string;
  status: JobStatus;
  createdAt: string | null;
  options: JobOptions;
  archived: boolean;
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
    costUsd: p.cost_usd,
    skipped: p.skipped,
    skipReason: p.skip_reason,
    startedAt: p.status === "running" ? (prev.startedAt ?? now) : prev.startedAt,
  };
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
   * running jobs only — history merges never open a stream, so old archived
   * jobs cannot notify). notifyJobDone itself checks hidden + permission. */
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
            archived: prev[jobId]?.archived ?? false,
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

  /** Merge a freshly created job (POST /api/jobs or retry) into the session:
   * seed rows, persist the seeds, attach the event stream. */
  const adopt = useCallback(
    (res: CreateJobResponse, sizes: (number | null)[], options: JobOptions) => {
      const jobId = res.job_id;
      setJobs((prev) => ({
        ...prev,
        [jobId]: { jobId, status: "running", createdAt: null, options, archived: false },
      }));
      const seeds: StoredSeed[] = res.items.map((it, idx) => ({
        itemId: it.item_id,
        name: it.name,
        kind: it.kind,
        sizeBytes: sizes[idx] ?? null,
      }));
      setItems((prev) => [...prev, ...seeds.map((s) => seedItem(jobId, s, false))]);
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

  /** Retry a failed (terminal) item as a new single-item job. The new job is
   * adopted through the normal submit path; the original row is only marked
   * `retried` — the ledger does not rewrite history. Returns the inline
   * error text (409/404 detail) or null on success. */
  const retry = useCallback(
    async (item: SessionItem): Promise<string | null> => {
      requestNotifyPermission();
      try {
        const res = await retryJobItem(item.jobId, item.itemId);
        adopt(
          res,
          res.items.map((it) => (it.kind === "file" ? item.sizeBytes : null)),
          jobs[item.jobId]?.options ?? { preset: null, llm: null },
        );
        setItems((prev) =>
          prev.map((i) => (i.key === item.key ? { ...i, retried: true } : i)),
        );
        writeStoredJobs(
          readStoredJobs().map((j) =>
            j.jobId === item.jobId
              ? {
                  ...j,
                  items: j.items.map((s) =>
                    s.itemId === item.itemId ? { ...s, retried: true } : s,
                  ),
                }
              : j,
          ),
        );
        return null;
      } catch (e) {
        return e instanceof Error ? e.message : String(e);
      }
    },
    [adopt, jobs],
  );

  const clear = useCallback(() => {
    for (const es of sourcesRef.current.values()) es.close();
    sourcesRef.current.clear();
    setItems([]);
    setJobs({});
    setSubmitError(null);
    writeStoredJobs([]);
  }, []);

  /** Merge an archived (terminal) job from the history into the session.
   * No event stream — the snapshot already is the final state. */
  const openArchived = useCallback((snap: JobSnapshot) => {
    const jobId = snap.job_id;
    const now = Date.now();
    setJobs((prev) =>
      prev[jobId] !== undefined
        ? prev
        : {
            ...prev,
            [jobId]: {
              jobId,
              status: snap.status,
              createdAt: snap.created_at,
              options: snap.options,
              archived: true,
            },
          },
    );
    setItems((prev) =>
      prev.some((i) => i.jobId === jobId)
        ? prev
        : [
            ...prev,
            ...snap.items.map((p) =>
              mergeItem(
                seedItem(
                  jobId,
                  { itemId: p.item_id, name: p.name, kind: p.kind, sizeBytes: null },
                  true,
                ),
                p,
                now,
              ),
            ),
          ],
    );
    const stored = readStoredJobs();
    if (!stored.some((j) => j.jobId === jobId)) {
      writeStoredJobs([
        ...stored,
        {
          jobId,
          items: snap.items.map((p) => ({
            itemId: p.item_id,
            name: p.name,
            kind: p.kind,
            sizeBytes: null,
          })),
          archived: true,
        },
      ]);
    }
  }, []);

  /** Drop one job (and its rows) from the session, e.g. after the history
   * entry backing it was deleted. */
  const removeJob = useCallback(
    (jobId: string) => {
      closeEvents(jobId);
      setItems((prev) => prev.filter((i) => i.jobId !== jobId));
      setJobs((prev) => {
        const { [jobId]: _gone, ...rest } = prev;
        return rest;
      });
      writeStoredJobs(readStoredJobs().filter((j) => j.jobId !== jobId));
    },
    [closeEvents],
  );

  // ---- session restore (F5 mid-job): seed the ledger from sessionStorage,
  // then reconcile each job against the server. 404 (server restarted) drops
  // the job silently; running jobs re-attach via EventSource snapshot replay.
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    const stored = readStoredJobs();
    if (stored.length === 0) return;

    setItems((prev) =>
      prev.length > 0
        ? prev
        : stored.flatMap((j) => j.items.map((s) => seedItem(j.jobId, s, j.archived === true))),
    );
    setJobs((prev) => {
      const next = { ...prev };
      for (const j of stored) {
        next[j.jobId] ??= {
          jobId: j.jobId,
          status: "running",
          createdAt: null,
          options: { preset: null, llm: null },
          archived: j.archived === true,
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
              archived: j.archived === true,
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
    items,
    jobs,
    jobCount: Object.keys(jobs).length,
    stats,
    running,
    activeCount,
    now,
    submit,
    retry,
    submitError,
    clear,
    openArchived,
    removeJob,
  };
}
