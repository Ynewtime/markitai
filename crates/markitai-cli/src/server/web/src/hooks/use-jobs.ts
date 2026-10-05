// Session job state. Every submission creates a job; the rows of all of them
// form one ledger. Each running job has one event stream (read with fetch, so
// it carries the header token); the service replays a snapshot on every
// (re)connection, so merges are idempotent.
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import {
  createJob,
  deleteItem as deleteJobItem,
  eventsPath,
  fetchSnapshot,
  retryItem,
  stopJob,
  type UploadHooks,
} from "../api/client.ts";
import { EventStream, type StreamEvent } from "../api/events.ts";
import type { CreateJobResponse, ItemPayload, JobOptions, JobProgress, JobSnapshot } from "../api/types.ts";
import { askNotifyPermission, notifyDone } from "../lib/notify.ts";
import { emptyOptions, publicOptions } from "../lib/options.ts";
import {
  settleCurrentJobSnapshot,
  itemFromPayload,
  mergeItem,
  readSeeds,
  reconcile,
  requeued,
  seedItem,
  sessionStats,
  snapshotStats,
  writeSeeds,
  type Seed,
  type SessionItem,
  type SessionJob,
  type StoredJob,
} from "../lib/session.ts";

const terminal = (status: string) => status !== "running";

function jobRecord(snapshot: JobSnapshot, label: string | null = null): SessionJob {
  return {
    jobId: snapshot.job_id,
    status: snapshot.status,
    createdAt: snapshot.created_at,
    options: publicOptions(snapshot.options),
    persistenceError: snapshot.persistence_error ?? null,
    // The label belongs to the submission, not to the service's answer, so a
    // snapshot keeps the one this session recorded.
    label,
  };
}

function storeSnapshotSeeds(jobId: string, payloads: ItemPayload[]): void {
  writeSeeds(
    readSeeds().map((job) => {
      if (job.jobId !== jobId) return job;
      const sizes = new Map(job.items.map((seed) => [seed.itemId, seed.sizeBytes]));
      return {
        jobId,
        label: job.label ?? null,
        items: payloads.map((item) => ({ itemId: item.item_id, name: item.name, kind: item.kind, sizeBytes: sizes.get(item.item_id) ?? null })),
      };
    }),
  );
}

export interface JobsApi {
  items: SessionItem[];
  jobs: Record<string, SessionJob>;
  stats: ReturnType<typeof sessionStats>;
  running: boolean;
  activeCount: number;
  terminalJobCount: number;
  submitError: unknown;
  restoreFailed: Set<string>;
  /** Job ids restored from this tab's seeds; their first outcome stays quiet. */
  restoredQuiet: Set<string>;
  /** `label` names the submission in the ledger when it carries several items. */
  submit(files: File[], urls: string[], options: JobOptions, hooks?: UploadHooks, label?: string | null): Promise<boolean>;
  retry(item: SessionItem, options?: JobOptions): Promise<unknown>;
  enhance(item: SessionItem, options: JobOptions): Promise<unknown>;
  retryArchived(snapshot: JobSnapshot, itemId: string, options?: JobOptions, operation?: "retry" | "enhance"): Promise<unknown>;
  adopt(snapshot: JobSnapshot): void;
  remove(item: SessionItem): Promise<unknown>;
  stop(jobIds: string[]): Promise<{ stopping: number; error: unknown }>;
  clear(): void;
  clearSettled(): void;
  retryRestore(): Promise<number>;
  clearSubmitError(): void;
}

/** Connectivity checks are owned by the app; a broken stream asks for one. */
const requestCheck = () => window.dispatchEvent(new CustomEvent("markitai:check"));

export function useJobs(notifyText: (done: number, failed: number, retained: number, skipped: number) => string, connLost: () => string): JobsApi {
  const [items, setItems] = useState<SessionItem[]>([]);
  const [jobs, setJobs] = useState<Record<string, SessionJob>>({});
  const [submitError, setSubmitError] = useState<unknown>(null);
  const [restoreFailed, setRestoreFailed] = useState<Set<string>>(() => new Set());
  const sources = useRef(new Map<string, EventStream>());
  const generations = useRef(new Map<string, symbol>());
  const notified = useRef(new Set<string>());
  // Jobs hydrated from this tab's seeds: their first settled outcome is not news
  // the user just watched happen, so the ledger reports it without a notice.
  // Acting on one of them (retry or enhance) makes its outcome reportable again.
  const restoredQuiet = useRef(new Set<string>());
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const jobsRef = useRef(jobs);
  jobsRef.current = jobs;
  const textRef = useRef({ notifyText, connLost });
  textRef.current = { notifyText, connLost };

  const patchJob = useCallback((jobId: string, patch: Partial<SessionJob>) => {
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

  const applySnapshot = useCallback(
    (snapshot: JobSnapshot) => {
      const now = Date.now();
      setJobs((previous) => ({
        ...previous,
        [snapshot.job_id]: jobRecord(snapshot, previous[snapshot.job_id]?.label ?? null),
      }));
      setItems((previous) => reconcile(previous, snapshot.job_id, snapshot.items, now));
      storeSnapshotSeeds(snapshot.job_id, snapshot.items);
    },
    [],
  );

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
      source.addEventListener("snapshot", (event) => {
        if (!live()) return;
        const snapshot = parse<JobSnapshot>(event);
        if (snapshot === null) return;
        applySnapshot(snapshot);
        if (terminal(snapshot.status)) {
          finished(jobId, snapshot);
          close(jobId);
        }
      });
      source.addEventListener("item", (event) => {
        if (!live()) return;
        const payload = parse<ItemPayload>(event);
        if (payload === null) return;
        const now = Date.now();
        setItems((previous) =>
          previous.map((item) => (item.jobId === jobId && item.itemId === payload.item_id ? mergeItem(item, payload, now) : item)),
        );
      });
      source.addEventListener("job", (event) => {
        if (!live()) return;
        const progress = parse<JobProgress & { persistence_error?: string }>(event);
        if (progress === null) return;
        patchJob(jobId, { status: progress.status, persistenceError: progress.persistence_error ?? null });
        if (terminal(progress.status)) {
          close(jobId);
          // The final snapshot carries finish times and durations of every row.
          void settleCurrentJobSnapshot(fetchSnapshot(jobId), current,
            (snapshot) => {
              applySnapshot(snapshot);
              if (snapshot.status === "running") listen(jobId);
            },
            (snapshot) => finished(jobId, snapshot),
          ).catch(() => undefined);
        }
      });
      source.addEventListener("error", () => {
        if (!live()) return;
        requestCheck();
        // While CONNECTING the browser retries by itself and the service replays a
        // snapshot. CLOSED is final: reconcile by hand or rows would spin forever.
        if (source.readyState !== EventStream.CLOSED) return;
        const lost = () => {
          const message = textRef.current.connLost();
          setItems((previous) =>
            previous.map((item) =>
              item.jobId === jobId && (item.status === "queued" || item.status === "running")
                ? { ...item, status: "error", error: message, errorCode: null, startedAt: null }
                : item,
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
            else close(jobId);
          },
          () => {
            if (sources.current.get(jobId) === source) lost();
          },
        );
      });
    },
    [applySnapshot, close, finished, patchJob],
  );

  const adoptCreated = useCallback(
    (created: CreateJobResponse, sizes: (number | null)[], options: JobOptions, label?: string | null) => {
      const jobId = created.job_id;
      setJobs((previous) => ({ ...previous, [jobId]: { jobId, status: "running", createdAt: null, options, persistenceError: null, label: label ?? null } }));
      const seeds: Seed[] = created.items.map((item, index) => ({ itemId: item.item_id, name: item.name, kind: item.kind, sizeBytes: sizes[index] ?? null }));
      setItems((previous) => [...previous, ...seeds.map((seed) => seedItem(jobId, seed))]);
      writeSeeds([...readSeeds(), { jobId, label: label ?? null, items: seeds }]);
      listen(jobId);
    },
    [listen],
  );

  const submit = useCallback(
    async (files: File[], urls: string[], options: JobOptions, hooks: UploadHooks = {}, label?: string | null) => {
      askNotifyPermission();
      setSubmitError(null);
      try {
        const created = await createJob(files, urls, options, hooks);
        // The service lists files first, then URLs, in submission order.
        adoptCreated(created, created.items.map((item, index) => (item.kind === "file" ? (files[index]?.size ?? null) : null)), options, label);
        return true;
      } catch (error) {
        if (!hooks.signal?.aborted) setSubmitError(error);
        return false;
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
      listen(item.jobId);
    },
    [listen, patchJob],
  );

  const retry = useCallback(
    async (item: SessionItem, options?: JobOptions) => {
      askNotifyPermission();
      try {
        await retryItem(item.jobId, item.itemId, options);
        requeue(item, "retry");
        return null;
      } catch (error) {
        return error;
      }
    },
    [requeue],
  );

  const enhance = useCallback(
    async (item: SessionItem, options: JobOptions) => {
      askNotifyPermission();
      try {
        await retryItem(item.jobId, item.itemId, options, "enhance");
        requeue(item, "enhance");
        return null;
      } catch (error) {
        return error;
      }
    },
    [requeue],
  );

  /** Bring a saved job into the ledger as live rows (all of its items). */
  const adoptSnapshot = useCallback(
    (snapshot: JobSnapshot, transform: (item: SessionItem) => SessionItem = (item) => item) => {
      generations.current.delete(snapshot.job_id);
      close(snapshot.job_id);
      const now = Date.now();
      const rows = snapshot.items.map((payload) => transform(itemFromPayload(snapshot.job_id, payload, now)));
      setJobs((previous) => ({
        ...previous,
        [snapshot.job_id]: jobRecord(snapshot, previous[snapshot.job_id]?.label ?? null),
      }));
      setItems((previous) => [...previous.filter((item) => item.jobId !== snapshot.job_id), ...rows]);
      const stored: StoredJob = {
        jobId: snapshot.job_id,
        items: rows.map((item) => ({ itemId: item.itemId, name: item.name, kind: item.kind, sizeBytes: null })),
      };
      writeSeeds([...readSeeds().filter((job) => job.jobId !== snapshot.job_id), stored]);
    },
    [close],
  );

  const retryArchived = useCallback(
    async (snapshot: JobSnapshot, itemId: string, options?: JobOptions, operation: "retry" | "enhance" = "retry") => {
      askNotifyPermission();
      try {
        await retryItem(snapshot.job_id, itemId, options, operation);
        adoptSnapshot(snapshot, (item) => (item.itemId === itemId ? requeued(item, operation) : item));
        notified.current.delete(snapshot.job_id);
        restoredQuiet.current.delete(snapshot.job_id);
        patchJob(snapshot.job_id, { status: "running" });
        listen(snapshot.job_id);
        return null;
      } catch (error) {
        return error;
      }
    },
    [adoptSnapshot, listen, patchJob],
  );

  const remove = useCallback(
    async (item: SessionItem) => {
      try {
        await deleteJobItem(item.jobId, item.itemId);
      } catch (error) {
        return error;
      }
      const last = !itemsRef.current.some((candidate) => candidate.jobId === item.jobId && candidate.key !== item.key);
      setItems((previous) => previous.filter((candidate) => candidate.key !== item.key));
      if (last) {
        generations.current.delete(item.jobId);
        close(item.jobId);
        setJobs((previous) => {
          const next = { ...previous };
          delete next[item.jobId];
          return next;
        });
      }
      writeSeeds(
        readSeeds().flatMap((job) => {
          if (job.jobId !== item.jobId) return [job];
          const seeds = job.items.filter((seed) => seed.itemId !== item.itemId);
          return seeds.length ? [{ ...job, items: seeds }] : [];
        }),
      );
      return null;
    },
    [close],
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

  const clear = useCallback(() => {
    for (const source of sources.current.values()) source.close();
    sources.current.clear();
    generations.current.clear();
    setItems([]);
    setJobs({});
    setSubmitError(null);
    setRestoreFailed(new Set());
    writeSeeds([]);
  }, []);

  /** Keep running jobs; remove whole finished ones. */
  const clearSettled = useCallback(() => {
    const finishedIds = new Set(
      Object.values(jobsRef.current)
        .filter((job) => terminal(job.status))
        .map((job) => job.jobId),
    );
    if (!finishedIds.size) return;
    for (const jobId of finishedIds) {
      generations.current.delete(jobId);
      close(jobId);
    }
    setItems((rows) => rows.filter((item) => !finishedIds.has(item.jobId)));
    setJobs((previous) => Object.fromEntries(Object.entries(previous).filter(([jobId]) => !finishedIds.has(jobId))));
    writeSeeds(readSeeds().filter((job) => !finishedIds.has(job.jobId)));
    setSubmitError(null);
  }, [close]);

  // ---- restore after a reload: seed rows from sessionStorage, then ask the
  // service. A 404 drops the job; an unreachable service keeps the rows with a retry.
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
          setItems((previous) => previous.filter((item) => item.jobId !== jobId));
          setJobs((previous) => {
            const next = { ...previous };
            delete next[jobId];
            return next;
          });
          writeSeeds(readSeeds().filter((job) => job.jobId !== jobId));
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
    [applySnapshot, listen],
  );

  const restored = useRef(false);
  useEffect(() => {
    if (restored.current) return;
    restored.current = true;
    const stored = readSeeds();
    if (!stored.length) return;
    for (const job of stored) restoredQuiet.current.add(job.jobId);
    setItems(stored.flatMap((job) => job.items.map((seed) => seedItem(job.jobId, seed))));
    setJobs(
      Object.fromEntries(
        stored.map((job) => [job.jobId, { jobId: job.jobId, status: "running", createdAt: null, options: emptyOptions(), persistenceError: null, label: job.label ?? null }]),
      ),
    );
    for (const job of stored) void reconcileJob(job.jobId);
  }, [reconcileJob]);

  const retryRestore = useCallback(async () => {
    const pending = [...restoreFailed];
    const outcomes = await Promise.all(pending.map((jobId) => reconcileJob(jobId)));
    return outcomes.filter((ok) => !ok).length;
  }, [reconcileJob, restoreFailed]);

  useEffect(() => {
    const all = sources.current;
    return () => {
      for (const source of all.values()) source.close();
      all.clear();
      generations.current.clear();
    };
  }, []);

  const stats = useMemo(() => sessionStats(items), [items]);
  const running = useMemo(() => Object.values(jobs).some((job) => job.status === "running"), [jobs]);
  const terminalJobCount = useMemo(() => Object.values(jobs).filter((job) => terminal(job.status)).length, [jobs]);
  const activeCount = useMemo(() => items.filter((item) => item.status === "queued" || item.status === "running").length, [items]);

  return {
    items,
    jobs,
    stats,
    running,
    activeCount,
    terminalJobCount,
    submitError,
    restoreFailed,
    restoredQuiet: restoredQuiet.current,
    submit,
    retry,
    enhance,
    retryArchived,
    adopt: (snapshot) => {
      adoptSnapshot(snapshot);
      if (snapshot.status === "running") listen(snapshot.job_id);
      else notified.current.add(snapshot.job_id);
    },
    remove,
    stop,
    clear,
    clearSettled,
    retryRestore,
    clearSubmitError: () => setSubmitError(null),
  };
}
