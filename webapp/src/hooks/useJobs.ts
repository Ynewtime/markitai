import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createJob, fetchJobSnapshot, jobEventsUrl } from "../api/client";
import type {
  ItemKind,
  ItemPayload,
  ItemStatus,
  JobOptions,
  JobPayload,
  JobSnapshot,
  JobStatus,
} from "../api/types";

/** sessionStorage seeds so an F5 mid-job can rebuild the ledger and re-attach
 * to still-running jobs (the server replays a snapshot on connect). */
const SESSION_KEY = "markitai.session";

interface StoredSeed {
  itemId: string;
  name: string;
  kind: ItemKind;
  sizeBytes: number | null;
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

  const closeEvents = useCallback((jobId: string) => {
    const es = sourcesRef.current.get(jobId);
    if (es !== undefined) {
      es.close();
      sourcesRef.current.delete(jobId);
    }
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
        if (snap.status !== "running") closeEvents(jobId);
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
        if (p.status !== "running") closeEvents(jobId);
      });
    },
    [closeEvents],
  );

  /** POST a new job. Returns true when created (errors land in submitError). */
  const submit = useCallback(
    async (files: File[], urls: string[], options: JobOptions): Promise<boolean> => {
      setSubmitError(null);
      try {
        const res = await createJob(files, urls, options);
        const jobId = res.job_id;
        setJobs((prev) => ({
          ...prev,
          [jobId]: { jobId, status: "running", createdAt: null, options, archived: false },
        }));
        // Backend item order is files-then-urls, matching our FormData order.
        const seeds: StoredSeed[] = res.items.map((it, idx) => ({
          itemId: it.item_id,
          name: it.name,
          kind: it.kind,
          sizeBytes: it.kind === "file" ? (files[idx]?.size ?? null) : null,
        }));
        setItems((prev) => [...prev, ...seeds.map((s) => seedItem(jobId, s, false))]);
        writeStoredJobs([...readStoredJobs(), { jobId, items: seeds }]);
        openEvents(jobId);
        return true;
      } catch (e) {
        setSubmitError(e instanceof Error ? e.message : String(e));
        return false;
      }
    },
    [openEvents],
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
    submitError,
    clear,
    openArchived,
    removeJob,
  };
}
