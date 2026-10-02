// Saved jobs from `GET /api/history`. Jobs this tab already shows as live rows
// are filtered out; the list refreshes when a job finishes, on demand, and when
// the window regains focus (at most every 20 seconds).
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { deleteHistoryJob, fetchHistory, fetchSnapshot } from "../api/client.ts";
import type { HistoryEntry, JobSnapshot, JobStatus } from "../api/types.ts";

const FOCUS_REFRESH_MS = 20_000;

export type ArchiveAction = "open" | "delete";

export interface ArchiveApi {
  entries: HistoryEntry[] | null;
  error: unknown;
  actions: Record<string, ArchiveAction>;
  rowErrors: Record<string, unknown>;
  refresh(): Promise<void>;
  open(jobId: string): Promise<JobSnapshot | null>;
  remove(jobId: string): Promise<boolean>;
}

export function useArchive(jobs: Record<string, { status: JobStatus }>): ArchiveApi {
  const [all, setAll] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [actions, setActions] = useState<Record<string, ArchiveAction>>({});
  const actionsRef = useRef<Record<string, ArchiveAction>>({});
  const [rowErrors, setRowErrors] = useState<Record<string, unknown>>({});
  const gone = useRef(new Set<string>());
  const lastFetch = useRef(0);
  const request = useRef(0);

  const refresh = useCallback(async () => {
    const id = ++request.current;
    try {
      const next = await fetchHistory();
      if (id !== request.current) return;
      setAll(next.filter((entry) => !gone.current.has(entry.job_id)));
      setError(null);
      lastFetch.current = Date.now();
    } catch (reason) {
      if (id === request.current) setError(reason);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // A live job reaching its end appears in history: refresh then.
  const statuses = useRef<Record<string, JobStatus>>({});
  useEffect(() => {
    let ended = false;
    const next: Record<string, JobStatus> = {};
    for (const [jobId, job] of Object.entries(jobs)) {
      next[jobId] = job.status;
      if (statuses.current[jobId] === "running" && job.status !== "running") ended = true;
    }
    statuses.current = next;
    if (ended) void refresh();
  }, [jobs, refresh]);

  useEffect(() => {
    const onFocus = () => {
      if (Date.now() - lastFetch.current >= FOCUS_REFRESH_MS) void refresh();
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [refresh]);

  const begin = (jobId: string, action: ArchiveAction): boolean => {
    if (actionsRef.current[jobId] !== undefined) return false;
    actionsRef.current = { ...actionsRef.current, [jobId]: action };
    setActions(actionsRef.current);
    setRowErrors((previous) => {
      if (!(jobId in previous)) return previous;
      const next = { ...previous };
      delete next[jobId];
      return next;
    });
    return true;
  };
  const end = (jobId: string) => {
    const next = { ...actionsRef.current };
    delete next[jobId];
    actionsRef.current = next;
    setActions(next);
  };
  const forget = (jobId: string) => {
    gone.current.add(jobId);
    setAll((previous) => previous?.filter((entry) => entry.job_id !== jobId) ?? previous);
  };

  const open = useCallback(async (jobId: string) => {
    if (!begin(jobId, "open")) return null;
    try {
      const snapshot = await fetchSnapshot(jobId);
      if (snapshot === null) forget(jobId);
      return snapshot;
    } catch (reason) {
      setRowErrors((previous) => ({ ...previous, [jobId]: reason }));
      return null;
    } finally {
      end(jobId);
    }
  }, []);

  const remove = useCallback(async (jobId: string) => {
    if (!begin(jobId, "delete")) return false;
    try {
      await deleteHistoryJob(jobId);
      forget(jobId);
      return true;
    } catch (reason) {
      setRowErrors((previous) => ({ ...previous, [jobId]: reason }));
      return false;
    } finally {
      end(jobId);
    }
  }, []);

  const entries = useMemo(() => all?.filter((entry) => !(entry.job_id in jobs)) ?? null, [all, jobs]);

  return { entries, error, actions, rowErrors, refresh, open, remove };
}
