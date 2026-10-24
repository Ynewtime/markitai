// Every submission the service keeps, in one store: saved jobs from history
// (with their items) and the jobs this tab is watching. A running job has one
// event stream (read with fetch, so it carries the header token); the service
// replays a snapshot on every (re)connection, so merges are idempotent. Saved
// jobs refresh when a watched job ends, when the window regains focus (at most
// every 20 s) and on demand.
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import {
  ApiError,
  batchDelete,
  createJob,
  deleteHistoryJob,
  deleteItem as deleteJobItem,
  eventsPath,
  fetchHistory,
  fetchSnapshot,
  retryItem,
  stopJob,
  type UploadHooks,
} from "../api/client.ts";
import { EventStream, type StreamEvent } from "../api/events.ts";
import type { CreateJobResponse, Features, HistoryEntry, ItemPayload, JobOptions, JobProgress, JobSnapshot } from "../api/types.ts";
import { placeholderJob, type ConvJob } from "../lib/conversions.ts";
import { askNotifyPermission, notifyDone } from "../lib/notify.ts";
import { publicOptions } from "../lib/options.ts";
import { publicSettings } from "../lib/request.ts";
import {
  itemFromPayload,
  mergeItem,
  readSeeds,
  reconcile,
  requeued,
  seedItem,
  settleCurrentJobSnapshot,
  snapshotStats,
  writeSeeds,
  type Seed,
  type SessionItem,
} from "../lib/session.ts";

const terminal = (status: string) => status !== "running";
const FOCUS_REFRESH_MS = 20_000;
/** Snapshots fetched at once when the backend cannot list items with history. */
const SNAPSHOT_WORKERS = 8;

function fromSnapshot(snapshot: JobSnapshot, previous: ConvJob | undefined): ConvJob {
  return {
    jobId: snapshot.job_id,
    status: snapshot.status,
    createdAt: snapshot.created_at,
    finishedAt: snapshot.finished_at,
    options: publicOptions(snapshot.options),
    settings: publicSettings(snapshot.settings),
    // A label this tab gave the submission stays when the backend keeps none.
    label: snapshot.label ?? previous?.label ?? null,
    origin: previous?.origin ?? "web",
    persistenceError: snapshot.persistence_error ?? null,
    durationMs: previous?.durationMs ?? null,
    costUsd: previous?.costUsd ?? null,
    pricing: previous?.pricing ?? null,
  };
}

function fromHistory(entry: HistoryEntry, previous: ConvJob | undefined): ConvJob {
  return {
    jobId: entry.job_id,
    status: entry.status,
    createdAt: entry.created_at,
    finishedAt: entry.finished_at,
    options: entry.options ? publicOptions(entry.options) : (previous?.options ?? null),
    settings: entry.items ? publicSettings(entry.settings) : (previous?.settings ?? null),
    label: entry.label ?? previous?.label ?? null,
    origin: entry.origin ?? "web",
    persistenceError: null,
    durationMs: entry.duration_ms,
    costUsd: entry.cost_usd,
    pricing: entry.pricing ?? null,
  };
}

export interface ConversionsApi {
  jobs: Record<string, ConvJob>;
  items: SessionItem[];
  /** History has answered at least once. */
  loaded: boolean;
  loadError: unknown;
  submitError: unknown;
  restoreFailed: Set<string>;
  /** Jobs restored from this tab's seeds; their first outcome stays quiet. */
  restoredQuiet: Set<string>;
  activeCount: number;
  submit(files: File[], urls: string[], options: JobOptions, hooks?: UploadHooks, label?: string | null): Promise<string | null>;
  retry(item: SessionItem, options?: JobOptions): Promise<unknown>;
  enhance(item: SessionItem, options: JobOptions): Promise<unknown>;
  removeItem(item: SessionItem): Promise<unknown>;
  removeJob(jobId: string): Promise<unknown>;
  /** Delete whole submissions and single items at once; each one succeeds or fails on its own. */
  removeMany(jobIds: string[], items: SessionItem[]): Promise<RemoveManyResult>;
  stop(jobIds: string[]): Promise<{ stopping: number; error: unknown }>;
  refresh(): Promise<void>;
  retryRestore(): Promise<number>;
  clearSubmitError(): void;
}

export interface RemoveManyResult {
  /** Conversions (items) removed, counting those that were already gone. */
  removed: number;
  /** What could not be deleted: the item (or the submission's first item) and why. */
  failed: { item: SessionItem; error: unknown }[];
}

/** Single deletions at once when the backend has no batch route. */
const DELETE_WORKERS = 4;
/** Entries per batch request (the backends accept up to 1000). */
const BATCH_LIMIT = 1000;

/** Run `work` over `entries`, at most `limit` at a time, in order of start. */
export async function bounded<T, R>(entries: T[], limit: number, work: (entry: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(entries.length);
  let next = 0;
  const worker = async () => {
    while (next < entries.length) {
      const index = next++;
      results[index] = await work(entries[index] as T);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, entries.length) }, worker));
  return results;
}

/** Connectivity checks are owned by the app; a broken stream asks for one. */
const requestCheck = () => window.dispatchEvent(new CustomEvent("markitai:check"));

export function useConversions(
  features: Features | null,
  ready: boolean,
  notifyText: (done: number, failed: number, retained: number, skipped: number) => string,
  connLost: () => string,
): ConversionsApi {
  const [items, setItems] = useState<SessionItem[]>([]);
  const [jobs, setJobs] = useState<Record<string, ConvJob>>({});
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [submitError, setSubmitError] = useState<unknown>(null);
  const [restoreFailed, setRestoreFailed] = useState<Set<string>>(() => new Set());
  const sources = useRef(new Map<string, EventStream>());
  const generations = useRef(new Map<string, symbol>());
  const notified = useRef(new Set<string>());
  const restoredQuiet = useRef(new Set<string>());
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  const textRef = useRef({ notifyText, connLost });
  textRef.current = { notifyText, connLost };
  const featuresRef = useRef(features);
  featuresRef.current = features;

  const patchJob = useCallback((jobId: string, patch: Partial<ConvJob>) => {
    setJobs((previous) => {
      const job = previous[jobId];
      return job === undefined ? previous : { ...previous, [jobId]: { ...job, ...patch } };
    });
  }, []);

  const close = useCallback((jobId: string) => {
    sources.current.get(jobId)?.close();
    sources.current.delete(jobId);
  }, []);

  const finished = useCallback((jobId: string, snapshot: JobSnapshot) => {
    if (notified.current.has(jobId)) return;
    notified.current.add(jobId);
    const stats = snapshotStats(snapshot);
    notifyDone(textRef.current.notifyText(stats.done, stats.failed, snapshot.items.filter((item) => item.rerun_failure).length, stats.skipped));
  }, []);

  const applySnapshot = useCallback((snapshot: JobSnapshot) => {
    const now = Date.now();
    setJobs((previous) => ({ ...previous, [snapshot.job_id]: fromSnapshot(snapshot, previous[snapshot.job_id]) }));
    setItems((previous) => reconcile(previous, snapshot.job_id, snapshot.items, now));
    // Only a running job needs its seed: history lists it once it has finished.
    writeSeeds(
      snapshot.status === "running"
        ? readSeeds().map((job) => (job.jobId === snapshot.job_id ? { ...job, items: snapshot.items.map((item) => ({ itemId: item.item_id, name: item.name, kind: item.kind, sizeBytes: null })) } : job))
        : readSeeds().filter((job) => job.jobId !== snapshot.job_id),
    );
  }, []);

  // ---- saved jobs
  const historyRequest = useRef(0);
  const lastFetch = useRef(0);
  const refresh = useCallback(async () => {
    const id = ++historyRequest.current;
    const withItems = featuresRef.current?.history_items === true;
    try {
      const entries = await fetchHistory(withItems);
      if (id !== historyRequest.current) return;
      const known = new Set(entries.map((entry) => entry.job_id));
      let payloads = new Map<string, ItemPayload[]>();
      if (withItems) {
        for (const entry of entries) if (entry.items) payloads.set(entry.job_id, entry.items);
      } else payloads = await snapshotsOf(entries, jobsRef.current, itemsRef.current, () => id === historyRequest.current);
      if (id !== historyRequest.current) return;
      const watched = (jobId: string) => sources.current.has(jobId) || jobsRef.current[jobId]?.status === "running";
      const now = Date.now();
      setJobs((previous) => {
        const next: Record<string, ConvJob> = {};
        // Watched jobs stay; saved jobs that history no longer lists were deleted or expired.
        for (const [jobId, job] of Object.entries(previous)) if (watched(jobId)) next[jobId] = job;
        for (const entry of entries) if (!watched(entry.job_id)) next[entry.job_id] = fromHistory(entry, previous[entry.job_id]);
        return next;
      });
      setItems((previous) => {
        let next = previous.filter((item) => watched(item.jobId) || known.has(item.jobId));
        for (const entry of entries) {
          if (watched(entry.job_id)) continue;
          const list = payloads.get(entry.job_id);
          if (list) next = reconcile(next, entry.job_id, list, now);
        }
        return next;
      });
      for (const jobId of known) notified.current.add(jobId);
      setLoadError(null);
      setLoaded(true);
      lastFetch.current = Date.now();
    } catch (error) {
      if (id === historyRequest.current) setLoadError(error);
    }
  }, []);

  const listen = useCallback(
    (jobId: string) => {
      close(jobId);
      const generation = Symbol(jobId);
      generations.current.set(jobId, generation);
      const source = new EventStream(eventsPath(jobId));
      sources.current.set(jobId, source);
      const current = () => generations.current.get(jobId) === generation;
      const live = () => current() && sources.current.get(jobId) === source;
      const parse = <T,>(event: StreamEvent): T | null => {
        try {
          return JSON.parse(event.data) as T;
        } catch {
          return null;
        }
      };
      const ended = () => {
        close(jobId);
        void refresh();
      };
      source.addEventListener("snapshot", (event) => {
        if (!live()) return;
        const snapshot = parse<JobSnapshot>(event);
        if (snapshot === null) return;
        applySnapshot(snapshot);
        if (terminal(snapshot.status)) {
          finished(jobId, snapshot);
          ended();
        }
      });
      source.addEventListener("item", (event) => {
        if (!live()) return;
        const payload = parse<ItemPayload>(event);
        if (payload === null) return;
        const now = Date.now();
        setItems((previous) => previous.map((item) => (item.jobId === jobId && item.itemId === payload.item_id ? mergeItem(item, payload, now) : item)));
      });
      source.addEventListener("job", (event) => {
        if (!live()) return;
        const progress = parse<JobProgress & { persistence_error?: string }>(event);
        if (progress === null) return;
        patchJob(jobId, { status: progress.status, persistenceError: progress.persistence_error ?? null });
        if (terminal(progress.status)) {
          close(jobId);
          // The final snapshot carries finish times and durations of every row.
          void settleCurrentJobSnapshot(
            fetchSnapshot(jobId),
            current,
            (snapshot) => {
              applySnapshot(snapshot);
              if (snapshot.status === "running") listen(jobId);
            },
            (snapshot) => {
              finished(jobId, snapshot);
              void refresh();
            },
          ).catch(() => undefined);
        }
      });
      source.addEventListener("error", () => {
        if (!live()) return;
        requestCheck();
        // While CONNECTING the reader retries by itself and the service replays a
        // snapshot. CLOSED is final: reconcile by hand or rows would spin forever.
        if (source.readyState !== EventStream.CLOSED) return;
        const lost = () => {
          const message = textRef.current.connLost();
          setItems((previous) =>
            previous.map((item) =>
              item.jobId === jobId && (item.status === "queued" || item.status === "running") ? { ...item, status: "error", error: message, errorCode: null, startedAt: null } : item,
            ),
          );
          patchJob(jobId, { status: "done" });
          notified.current.add(jobId);
          close(jobId);
        };
        fetchSnapshot(jobId).then(
          (snapshot) => {
            if (sources.current.get(jobId) !== source) return;
            if (snapshot === null) {
              lost();
              return;
            }
            applySnapshot(snapshot);
            if (snapshot.status === "running") listen(jobId);
            else ended();
          },
          () => {
            if (sources.current.get(jobId) === source) lost();
          },
        );
      });
    },
    [applySnapshot, close, finished, patchJob, refresh],
  );

  const adoptCreated = useCallback(
    (created: CreateJobResponse, sizes: (number | null)[], options: JobOptions, label: string | null) => {
      const jobId = created.job_id;
      setJobs((previous) => ({ ...previous, [jobId]: { ...placeholderJob(jobId), createdAt: new Date().toISOString(), options, label } }));
      const seeds: Seed[] = created.items.map((item, index) => ({ itemId: item.item_id, name: item.name, kind: item.kind, sizeBytes: sizes[index] ?? null }));
      setItems((previous) => [...previous, ...seeds.map((seed) => seedItem(jobId, seed))]);
      writeSeeds([...readSeeds(), { jobId, label, items: seeds }]);
      listen(jobId);
    },
    [listen],
  );

  const submit = useCallback(
    async (files: File[], urls: string[], options: JobOptions, hooks: UploadHooks = {}, label: string | null = null) => {
      askNotifyPermission();
      setSubmitError(null);
      try {
        const created = await createJob(files, urls, options, hooks, featuresRef.current?.job_labels ? label : null);
        // The service lists files first, then URLs, in submission order.
        adoptCreated(created, created.items.map((item, index) => (item.kind === "file" ? (files[index]?.size ?? null) : null)), options, label);
        return created.job_id;
      } catch (error) {
        if (!hooks.signal?.aborted) setSubmitError(error);
        return null;
      }
    },
    [adoptCreated],
  );

  const requeue = useCallback(
    (item: SessionItem, operation: "retry" | "enhance") => {
      notified.current.delete(item.jobId);
      restoredQuiet.current.delete(item.jobId);
      patchJob(item.jobId, { status: "running" });
      setItems((previous) => previous.map((candidate) => (candidate.key === item.key ? requeued(candidate, operation) : candidate)));
      if (!readSeeds().some((job) => job.jobId === item.jobId)) {
        const jobItems = itemsRef.current.filter((candidate) => candidate.jobId === item.jobId);
        writeSeeds([...readSeeds(), { jobId: item.jobId, label: jobsRef.current[item.jobId]?.label ?? null, items: jobItems.map((row) => ({ itemId: row.itemId, name: row.name, kind: row.kind, sizeBytes: row.sizeBytes })) }]);
      }
      listen(item.jobId);
    },
    [listen, patchJob],
  );

  const rerun = useCallback(
    async (item: SessionItem, operation: "retry" | "enhance", options?: JobOptions) => {
      askNotifyPermission();
      try {
        await retryItem(item.jobId, item.itemId, options, operation);
        requeue(item, operation);
        return null;
      } catch (error) {
        return error;
      }
    },
    [requeue],
  );

  const forgetJob = useCallback((jobId: string) => {
    generations.current.delete(jobId);
    close(jobId);
    setJobs((previous) => {
      const next = { ...previous };
      delete next[jobId];
      return next;
    });
    setItems((previous) => previous.filter((item) => item.jobId !== jobId));
    writeSeeds(readSeeds().filter((job) => job.jobId !== jobId));
  }, [close]);

  const removeItem = useCallback(
    async (item: SessionItem) => {
      try {
        await deleteJobItem(item.jobId, item.itemId);
      } catch (error) {
        return error;
      }
      const last = !itemsRef.current.some((candidate) => candidate.jobId === item.jobId && candidate.key !== item.key);
      setItems((previous) => previous.filter((candidate) => candidate.key !== item.key));
      if (last) forgetJob(item.jobId);
      return null;
    },
    [forgetJob],
  );


  const removeJob = useCallback(
    async (jobId: string) => {
      try {
        await deleteHistoryJob(jobId);
      } catch (error) {
        return error;
      }
      forgetJob(jobId);
      return null;
    },
    [forgetJob],
  );

  const removeMany = useCallback(
    async (jobIds: string[], chosen: SessionItem[]): Promise<RemoveManyResult> => {
      const all = itemsRef.current;
      const firstOf = (jobId: string) => all.find((item) => item.jobId === jobId);
      const gone = (error: unknown) => error instanceof ApiError && error.status === 404;
      const jobOutcome: { jobId: string; error: unknown }[] = [];
      const itemOutcome: { item: SessionItem; error: unknown }[] = [];
      if (features?.batch_delete === true) {
        const entries = [...jobIds.map((jobId) => ({ jobId })), ...chosen.map((item) => ({ item }))];
        for (let start = 0; start < entries.length; start += BATCH_LIMIT) {
          const slice = entries.slice(start, start + BATCH_LIMIT);
          const jobs = slice.flatMap((entry) => ("jobId" in entry ? [entry.jobId] : []));
          const singles = slice.flatMap((entry) => ("item" in entry ? [entry.item] : []));
          try {
            const answer = await batchDelete(
              jobs,
              singles.map((item) => ({ job_id: item.jobId, item_id: item.itemId })),
            );
            answer.results.forEach((result, index) => {
              // Already gone counts as deleted, as the single routes' 404 does.
              const missing = result.reason === "job_not_found" || result.reason === "item_not_found";
              const error = result.ok || missing ? null : new ApiError(409, { detail: result.detail ?? "", code: "conflict", reason: result.reason ?? null });
              if (index < jobs.length) jobOutcome.push({ jobId: jobs[index] as string, error });
              else itemOutcome.push({ item: singles[index - jobs.length] as SessionItem, error });
            });
          } catch (error) {
            for (const jobId of jobs) jobOutcome.push({ jobId, error });
            for (const item of singles) itemOutcome.push({ item, error });
          }
        }
      } else {
        const settle = async (run: () => Promise<unknown>) => {
          try {
            await run();
            return null;
          } catch (error) {
            return gone(error) ? null : error;
          }
        };
        const jobErrors = await bounded(jobIds, DELETE_WORKERS, (jobId) => settle(() => deleteHistoryJob(jobId)));
        jobIds.forEach((jobId, index) => jobOutcome.push({ jobId, error: jobErrors[index] }));
        // Items of one submission one after another: the service locks the submission per deletion.
        const byJob = new Map<string, SessionItem[]>();
        for (const item of chosen) byJob.set(item.jobId, [...(byJob.get(item.jobId) ?? []), item]);
        const itemErrors = await bounded([...byJob.values()], DELETE_WORKERS, async (group) => {
          const errors: unknown[] = [];
          for (const item of group) errors.push(await settle(() => deleteJobItem(item.jobId, item.itemId)));
          return errors;
        });
        [...byJob.values()].forEach((group, index) => group.forEach((item, position) => itemOutcome.push({ item, error: itemErrors[index]?.[position] ?? null })));
      }
      let removed = 0;
      const failed: RemoveManyResult["failed"] = [];
      const deletedJobs = new Set<string>();
      const deletedItems = new Set<string>();
      for (const { jobId, error } of jobOutcome) {
        const members = all.filter((item) => item.jobId === jobId);
        if (error === null) {
          deletedJobs.add(jobId);
          removed += members.length;
        } else {
          const first = firstOf(jobId);
          if (first) failed.push({ item: first, error });
        }
      }
      for (const { item, error } of itemOutcome) {
        if (error === null) {
          deletedItems.add(item.key);
          removed += 1;
        } else failed.push({ item, error });
      }
      setItems((previous) => previous.filter((item) => !deletedItems.has(item.key) && !deletedJobs.has(item.jobId)));
      // A submission whose every item went is gone too.
      const emptied = new Set(deletedJobs);
      for (const jobId of new Set(chosen.map((item) => item.jobId))) {
        if (!all.some((item) => item.jobId === jobId && !deletedItems.has(item.key))) emptied.add(jobId);
      }
      for (const jobId of emptied) forgetJob(jobId);
      return { removed, failed };
    },
    [features, forgetJob],
  );

  /** Stop the original items still waiting for a slot. A 409 means nothing waits any more. */
  const stop = useCallback(async (jobIds: string[]) => {
    let stopping = 0;
    let error: unknown = null;
    for (const jobId of jobIds) {
      try {
        stopping += (await stopJob(jobId)).stopping;
      } catch (reason) {
        if ((reason as { status?: number })?.status !== 409) error = reason;
      }
    }
    return { stopping, error };
  }, []);

  // ---- restore after a reload: running jobs from this tab's seeds, saved ones from history.
  const reconcileJob = useCallback(
    async (jobId: string) => {
      const generation = Symbol(jobId);
      generations.current.set(jobId, generation);
      try {
        const snapshot = await fetchSnapshot(jobId);
        if (generations.current.get(jobId) !== generation) return true;
        setRestoreFailed((previous) => {
          if (!previous.has(jobId)) return previous;
          const next = new Set(previous);
          next.delete(jobId);
          return next;
        });
        if (snapshot === null) {
          forgetJob(jobId);
          return true;
        }
        applySnapshot(snapshot);
        if (snapshot.status === "running") listen(jobId);
        else notified.current.add(jobId);
        return true;
      } catch {
        if (generations.current.get(jobId) !== generation) return true;
        setRestoreFailed((previous) => new Set(previous).add(jobId));
        return false;
      }
    },
    [applySnapshot, forgetJob, listen],
  );

  const started = useRef(false);
  useEffect(() => {
    if (!ready || started.current) return;
    started.current = true;
    const stored = readSeeds();
    for (const job of stored) restoredQuiet.current.add(job.jobId);
    if (stored.length) {
      setItems((previous) => [...previous, ...stored.flatMap((job) => job.items.map((seed) => seedItem(job.jobId, seed)))]);
      setJobs((previous) => ({ ...previous, ...Object.fromEntries(stored.map((job) => [job.jobId, { ...placeholderJob(job.jobId), label: job.label ?? null }])) }));
      for (const job of stored) void reconcileJob(job.jobId);
    }
    void refresh();
  }, [ready, reconcileJob, refresh]);

  useEffect(() => {
    const onFocus = () => {
      if (started.current && Date.now() - lastFetch.current >= FOCUS_REFRESH_MS) void refresh();
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [refresh]);

  const retryRestore = useCallback(async () => {
    const pending = [...restoreFailed];
    const outcomes = await Promise.all(pending.map((jobId) => reconcileJob(jobId)));
    void refresh();
    return outcomes.filter((ok) => !ok).length;
  }, [reconcileJob, restoreFailed, refresh]);

  useEffect(() => {
    const all = sources.current;
    return () => {
      for (const source of all.values()) source.close();
      all.clear();
      generations.current.clear();
    };
  }, []);

  const activeCount = useMemo(() => items.filter((item) => item.status === "queued" || item.status === "running").length, [items]);

  return {
    jobs,
    items,
    loaded,
    loadError,
    submitError,
    restoreFailed,
    restoredQuiet: restoredQuiet.current,
    activeCount,
    submit,
    retry: (item, options) => rerun(item, "retry", options),
    enhance: (item, options) => rerun(item, "enhance", options),
    removeItem,
    removeJob,
    removeMany,
    stop,
    refresh,
    retryRestore,
    clearSubmitError: () => setSubmitError(null),
  };
}

/** Item lists for history entries when the backend cannot send them inline:
 * one snapshot per job not already known, a few at a time. */
async function snapshotsOf(
  entries: HistoryEntry[],
  jobs: Record<string, ConvJob>,
  items: SessionItem[],
  current: () => boolean,
): Promise<Map<string, ItemPayload[]>> {
  const result = new Map<string, ItemPayload[]>();
  const have = new Set(items.map((item) => item.jobId));
  const todo = entries.filter((entry) => !have.has(entry.job_id) || jobs[entry.job_id]?.finishedAt !== entry.finished_at);
  let cursor = 0;
  const worker = async () => {
    while (current()) {
      const entry = todo[cursor++];
      if (!entry) return;
      try {
        const snapshot = await fetchSnapshot(entry.job_id);
        if (snapshot) result.set(entry.job_id, snapshot.items);
      } catch {
        /* Kept as it was; the next refresh asks again. */
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(SNAPSHOT_WORKERS, todo.length) }, worker));
  return result;
}

export { itemFromPayload };
