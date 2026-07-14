import { useEffect, useRef, useState } from "react";
import type { HistoryEntry, JobSnapshot } from "../api/types";
import type { Dict } from "../i18n";
import { fmtBytes } from "../lib/format";
import { FileTextIcon } from "./icons";

const CONFIRM_RESET_MS = 4000;

function localYmd(date: Date): string {
  return `${String(date.getFullYear())}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(
    date.getDate(),
  ).padStart(2, "0")}`;
}

function dayLabel(iso: string, t: Dict): string {
  const date = new Date(iso);
  const now = new Date();
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  const ymd = localYmd(date);
  if (ymd === localYmd(now)) return t.histToday;
  if (ymd === localYmd(yesterday)) return t.histYesterday;
  return ymd;
}

function hhmm(iso: string): string {
  const date = new Date(iso);
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function statsLine(entry: HistoryEntry, t: Dict): string {
  const succeeded = Math.max(0, entry.done - entry.skipped);
  const parts = [`${succeeded} ${t.statDone}`];
  if (entry.failed > 0) parts.push(`${entry.failed} ${t.statFailed}`);
  if (entry.skipped > 0) parts.push(`${entry.skipped} ${t.statSkipped}`);
  return parts.join(" · ");
}

/** Persisted jobs rendered as rows inside the main conversion ledger. */
export function ArchivedJobRows({
  t,
  entries,
  error,
  actions,
  rowErrors,
  showCost,
  startIndex,
  onRefresh,
  onOpen,
  onDelete,
  announce,
}: {
  t: Dict;
  entries: HistoryEntry[] | null;
  error: string | null;
  actions: Record<string, "open" | "delete">;
  rowErrors: Record<string, string>;
  showCost: boolean;
  startIndex: number;
  onRefresh: () => Promise<void>;
  onOpen: (jobId: string, opener: HTMLElement) => Promise<JobSnapshot | null>;
  onDelete: (jobId: string) => Promise<boolean>;
  announce: (message: string) => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const confirmTimerRef = useRef<number | null>(null);
  const rowRefs = useRef(new Map<string, HTMLDivElement>());

  useEffect(() => {
    return () => {
      if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (confirmDelete === null) return;
    const cancel = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setConfirmDelete(null);
    };
    document.addEventListener("keydown", cancel, true);
    return () => document.removeEventListener("keydown", cancel, true);
  }, [confirmDelete]);

  const requestDelete = async (entry: HistoryEntry, index: number) => {
    const jobId = entry.job_id;
    if (confirmDelete !== jobId) {
      setConfirmDelete(jobId);
      if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
      confirmTimerRef.current = window.setTimeout(
        () => setConfirmDelete(null),
        CONFIRM_RESET_MS,
      );
      return;
    }

    if (confirmTimerRef.current !== null) window.clearTimeout(confirmTimerRef.current);
    setConfirmDelete(null);
    const nextId = entries?.[index + 1]?.job_id ?? entries?.[index - 1]?.job_id ?? null;
    const deleted = await onDelete(jobId);
    if (!deleted) return;

    announce(t.histDeleted(entry.names_preview[0] ?? jobId));
    window.requestAnimationFrame(() => {
      const nextRow = nextId === null ? null : rowRefs.current.get(nextId);
      nextRow?.focus();
    });
  };

  if (error !== null) {
    return (
      <div
        className="lrow archive-state errline"
        role="option"
        aria-selected={false}
      >
        <span />
        <span>{error}</span>
        <span />
        {showCost && <span />}
        <button type="button" className="rowact" onClick={() => void onRefresh()}>
          {t.retryLoad}
        </button>
      </div>
    );
  }
  if (entries === null || entries.length === 0) return null;

  return (
    <>
      {entries.map((entry, index) => {
        const more = entry.total - entry.names_preview.length;
        const names =
          entry.names_preview.join(", ") + (more > 0 ? ` ${t.histMore(more)}` : "");
        const firstName = entry.names_preview[0] ?? entry.job_id;
        const busy = actions[entry.job_id];
        const confirming = confirmDelete === entry.job_id;
        const activate = (opener: HTMLElement) => {
          if (busy === undefined) void onOpen(entry.job_id, opener);
        };
        return (
          <div
            className="lrow archived-row"
            role="option"
            aria-selected={false}
            aria-disabled={busy !== undefined ? true : undefined}
            tabIndex={startIndex === 0 && index === 0 ? 0 : -1}
            aria-label={`${t.histOpen} ${firstName}`}
            key={entry.job_id}
            ref={(element) => {
              if (element === null) rowRefs.current.delete(entry.job_id);
              else rowRefs.current.set(entry.job_id, element);
            }}
            onClick={(event) => activate(event.currentTarget)}
            onKeyDown={(event) => {
              if (event.target !== event.currentTarget) return;
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                activate(event.currentTarget);
              }
            }}
          >
            <span className="c-num">{String(startIndex + index + 1).padStart(2, "0")}</span>
            <span className="c-name" title={names}>
              <FileTextIcon />
              <span className="archive-name">{names}</span>
              <span className="minibadge">archived</span>
            </span>
            <span className="c-time" title={dayLabel(entry.created_at, t)}>
              {hhmm(entry.created_at)}
            </span>
            {showCost && <span className="c-cost">—</span>}
            <span className="c-status archive-actions">
              <span className="minibadge archive-result">{statsLine(entry, t)}</span>
              <button
                type="button"
                className={confirming ? "rowact warn" : "rowact"}
                disabled={busy !== undefined}
                aria-label={
                  confirming
                    ? t.histConfirmDeleteAria(firstName)
                    : t.histDeleteAria(firstName)
                }
                onClick={(event) => {
                  event.stopPropagation();
                  void requestDelete(entry, index);
                }}
              >
                {busy === "delete"
                  ? t.deleting
                  : confirming
                    ? t.histConfirm
                    : t.histDelete}
              </button>
            </span>
            <span className="rowmeta">
              {statsLine(entry, t)} · {t.histStorageSize(fmtBytes(entry.size_bytes))}
            </span>
            {rowErrors[entry.job_id] !== undefined && (
              <span className="c-err errline">{rowErrors[entry.job_id]}</span>
            )}
          </div>
        );
      })}
    </>
  );
}
