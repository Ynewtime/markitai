import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { deleteHistoryJob, fetchHistory, fetchJobSnapshot } from "../api/client";
import type { HistoryEntry, JobSnapshot, JobStatus } from "../api/types";

const FOCUS_REFRESH_MS = 20_000;

type ArchivedAction = "open" | "delete";

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Server-persisted terminal jobs. Current-session membership is filtered at
 * the boundary so archived job summaries never become item-ledger rows. */
export function useArchivedJobs(
  jobs: Record<string, { status: JobStatus }>,
) {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [actions, setActions] = useState<Record<string, ArchivedAction>>({});
  const actionsRef = useRef<Record<string, ArchivedAction>>({});
  const [rowErrors, setRowErrors] = useState<Record<string, string>>({});
  const staleIdsRef = useRef(new Set<string>());
  const lastFetchRef = useRef(0);
  const requestRef = useRef(0);

  const refresh = useCallback(async () => {
    const request = ++requestRef.current;
    setRefreshing(true);
    try {
      const next = await fetchHistory();
      if (request !== requestRef.current) return;
      setEntries(next.filter((entry) => !staleIdsRef.current.has(entry.job_id)));
      setError(null);
      lastFetchRef.current = Date.now();
    } catch (reason) {
      if (request === requestRef.current) setError(errorText(reason));
    } finally {
      if (request === requestRef.current) setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const previousStatusesRef = useRef<Record<string, JobStatus>>({});
  useEffect(() => {
    const previous = previousStatusesRef.current;
    const next: Record<string, JobStatus> = {};
    let reachedTerminal = false;
    for (const [jobId, job] of Object.entries(jobs)) {
      next[jobId] = job.status;
      if (previous[jobId] === "running" && job.status !== "running") {
        reachedTerminal = true;
      }
    }
    previousStatusesRef.current = next;
    if (reachedTerminal) void refresh();
  }, [jobs, refresh]);

  useEffect(() => {
    const onFocus = () => {
      if (Date.now() - lastFetchRef.current >= FOCUS_REFRESH_MS) void refresh();
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [refresh]);

  const clearRowError = useCallback((jobId: string) => {
    setRowErrors((previous) => {
      if (previous[jobId] === undefined) return previous;
      const next = { ...previous };
      delete next[jobId];
      return next;
    });
  }, []);

  const removeStale = useCallback((jobId: string) => {
    staleIdsRef.current.add(jobId);
    setEntries((previous) =>
      previous === null ? previous : previous.filter((entry) => entry.job_id !== jobId),
    );
  }, []);

  const beginAction = useCallback((jobId: string, action: ArchivedAction): boolean => {
    if (actionsRef.current[jobId] !== undefined) return false;
    actionsRef.current = { ...actionsRef.current, [jobId]: action };
    setActions(actionsRef.current);
    return true;
  }, []);

  const endAction = useCallback((jobId: string) => {
    const next = { ...actionsRef.current };
    delete next[jobId];
    actionsRef.current = next;
    setActions(next);
  }, []);

  const openJob = useCallback(
    async (jobId: string): Promise<JobSnapshot | null> => {
      if (!beginAction(jobId, "open")) return null;
      clearRowError(jobId);
      try {
        const snapshot = await fetchJobSnapshot(jobId);
        if (snapshot === null) removeStale(jobId);
        return snapshot;
      } catch (reason) {
        setRowErrors((previous) => ({ ...previous, [jobId]: errorText(reason) }));
        return null;
      } finally {
        endAction(jobId);
      }
    },
    [beginAction, clearRowError, endAction, removeStale],
  );

  const deleteJob = useCallback(
    async (jobId: string): Promise<boolean> => {
      if (!beginAction(jobId, "delete")) return false;
      clearRowError(jobId);
      try {
        await deleteHistoryJob(jobId);
        removeStale(jobId);
        return true;
      } catch (reason) {
        setRowErrors((previous) => ({ ...previous, [jobId]: errorText(reason) }));
        return false;
      } finally {
        endAction(jobId);
      }
    },
    [beginAction, clearRowError, endAction, removeStale],
  );

  const currentIds = useMemo(() => new Set(Object.keys(jobs)), [jobs]);
  const recent = useMemo(
    () => entries?.filter((entry) => !currentIds.has(entry.job_id)) ?? null,
    [currentIds, entries],
  );

  return {
    entries: recent,
    error,
    refreshing,
    actions,
    rowErrors,
    refresh,
    openJob,
    deleteJob,
  };
}
