import { useEffect, useRef, useState } from "react";
import { deleteHistoryJob, fetchHistory } from "../api/client";
import type { HistoryEntry } from "../api/types";
import type { Dict } from "../i18n";
import { fmtBytes } from "../lib/format";

const CONFIRM_RESET_MS = 4000;

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function localYmd(d: Date): string {
  return `${String(d.getFullYear())}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate(),
  ).padStart(2, "0")}`;
}

function dayLabel(iso: string, t: Dict): string {
  const d = new Date(iso);
  const now = new Date();
  const yest = new Date(now);
  yest.setDate(now.getDate() - 1);
  const ymd = localYmd(d);
  if (ymd === localYmd(now)) return t.histToday;
  if (ymd === localYmd(yest)) return t.histYesterday;
  return ymd;
}

function hhmm(iso: string): string {
  const d = new Date(iso);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function statsLine(e: HistoryEntry, t: Dict): string {
  const parts = [`${e.done} ${t.statDone}`];
  if (e.failed > 0) parts.push(`${e.failed} ${t.statFailed}`);
  if (e.skipped > 0) parts.push(`${e.skipped} ${t.statSkipped}`);
  return parts.join(" · ");
}

/** Conversion history: date-grouped rows (today / yesterday / YYYY-MM-DD)
 * from GET /api/history. open merges the archived job into the session
 * (App switches to the workspace); delete is a two-step inline confirm. */
export function HistoryView({
  t,
  onOpen,
  onDeleted,
}: {
  t: Dict;
  /** Resolves to an error message, or null when the job entered the session. */
  onOpen: (jobId: string) => Promise<string | null>;
  onDeleted: (jobId: string) => void;
}) {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [rowErr, setRowErr] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmDel, setConfirmDel] = useState<string | null>(null);
  const confirmTimerRef = useRef<number | null>(null);

  useEffect(() => {
    let stale = false;
    fetchHistory().then(
      (list) => {
        if (!stale) setEntries(list);
      },
      (e: unknown) => {
        if (!stale) setLoadError(errText(e));
      },
    );
    return () => {
      stale = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
    };
  }, []);

  const setRowError = (jobId: string, msg: string | null) => {
    setRowErr((prev) => {
      const next = { ...prev };
      if (msg === null) delete next[jobId];
      else next[jobId] = msg;
      return next;
    });
  };

  const open = async (jobId: string) => {
    if (busy !== null) return;
    setBusy(jobId);
    setRowError(jobId, null);
    const err = await onOpen(jobId);
    if (err !== null) setRowError(jobId, err);
    setBusy(null);
  };

  const remove = async (jobId: string) => {
    if (confirmDel !== jobId) {
      setConfirmDel(jobId);
      if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
      confirmTimerRef.current = window.setTimeout(() => setConfirmDel(null), CONFIRM_RESET_MS);
      return;
    }
    if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
    setConfirmDel(null);
    setRowError(jobId, null);
    try {
      await deleteHistoryJob(jobId);
      setEntries((prev) => (prev === null ? prev : prev.filter((e) => e.job_id !== jobId)));
      onDeleted(jobId); // drop from the session too, if it was merged
    } catch (e) {
      setRowError(jobId, errText(e));
    }
  };

  if (entries === null) {
    return (
      <p className={loadError === null ? "hist-empty mono" : "errline"}>
        {loadError ?? t.loading}
      </p>
    );
  }
  if (entries.length === 0) {
    return <p className="hist-empty mono">{t.histEmpty}</p>;
  }

  // Server order is time-descending — group consecutive same-day entries.
  const groups: { label: string; entries: HistoryEntry[] }[] = [];
  for (const e of entries) {
    const label = dayLabel(e.created_at, t);
    const last = groups[groups.length - 1];
    if (last !== undefined && last.label === label) last.entries.push(e);
    else groups.push({ label, entries: [e] });
  }

  return (
    <div className="hist">
      {groups.map((g) => (
        <section key={g.label}>
          <h3 className="hgroup">{g.label}</h3>
          {g.entries.map((e) => {
            const more = e.total - e.names_preview.length;
            const names =
              e.names_preview.join(", ") + (more > 0 ? ` ${t.histMore(more)}` : "");
            const firstName = e.names_preview[0] ?? e.job_id;
            return (
              <div className="hrow" key={e.job_id}>
                <span className="h-time mono">{hhmm(e.created_at)}</span>
                <span className="h-names mono" title={names}>
                  {names}
                </span>
                <span className="h-stats mono">{statsLine(e, t)}</span>
                <span className="h-size mono">{fmtBytes(e.size_bytes)}</span>
                <span className="h-acts">
                  <button
                    type="button"
                    className="rowact"
                    disabled={busy !== null}
                    aria-label={`${t.histOpen} ${firstName}`}
                    onClick={() => void open(e.job_id)}
                  >
                    {busy === e.job_id ? t.loading : t.histOpen}
                  </button>
                  <button
                    type="button"
                    className={confirmDel === e.job_id ? "rowact warn" : "rowact"}
                    aria-label={`${t.histDelete} ${firstName}`}
                    onClick={() => void remove(e.job_id)}
                  >
                    {confirmDel === e.job_id ? t.histConfirm : t.histDelete}
                  </button>
                </span>
                {rowErr[e.job_id] !== undefined && (
                  <span className="h-err errline">{rowErr[e.job_id]}</span>
                )}
              </div>
            );
          })}
        </section>
      ))}
    </div>
  );
}
