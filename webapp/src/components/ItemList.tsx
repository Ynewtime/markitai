import { useEffect, useMemo, useRef, useState } from "react";
import type { HistoryEntry } from "../api/types";
import type {
  SessionItem,
  SessionJob,
  SessionStats,
} from "../hooks/useJobs";
import type { Dict } from "../i18n";
import { fmtCost, fmtDur, serverTimestampMs } from "../lib/format";
import {
  ArchivedJobRows,
  type ArchivedJobRowsProps,
} from "./ArchivedJobsSection";
import { ItemRow } from "./ItemRow";

/** Status facets share the chip vocabulary (mono words, English in both
 * locales — they mirror the status chips). */
const STATUS_FILTERS = ["all", "done", "failed", "skipped"] as const;
type StatusFilter = (typeof STATUS_FILTERS)[number];

/** Stable fallback: a fresh [] per render would invalidate the rows memo. */
const NO_ARCHIVE_ENTRIES: HistoryEntry[] = [];

type ArchiveRowProps = Omit<
  ArchivedJobRowsProps,
  | "t"
  | "entries"
  | "error"
  | "showCost"
  | "startIndex"
  | "llmAvailable"
  | "llmDisabledReason"
  | "tabbableJobId"
  | "onRowFocus"
>;

export interface ItemListArchive {
  entries: HistoryEntry[] | null;
  error: string | null;
  rowProps: ArchiveRowProps;
}

type LedgerRow =
  | { kind: "session"; key: string; item: SessionItem }
  | { kind: "archive"; key: string; entry: HistoryEntry };

type LedgerGroup =
  | {
      kind: "session";
      jobId: string;
      timestamp: number;
      createdTimestamp: number;
      items: SessionItem[];
    }
  | {
      kind: "archive";
      jobId: string;
      timestamp: number;
      createdTimestamp: number;
      entry: HistoryEntry;
    };

function matchesStatus(item: SessionItem, filter: StatusFilter): boolean {
  switch (filter) {
    case "all":
      return true;
    case "done":
      return item.status === "done" && !item.skipped;
    case "failed":
      return item.status === "error";
    case "skipped":
      return item.status === "done" && item.skipped;
  }
}

function matchesArchivedStatus(
  entry: HistoryEntry,
  filter: StatusFilter,
): boolean {
  switch (filter) {
    case "all":
      return true;
    case "done":
      return entry.done - entry.skipped > 0;
    case "failed":
      return entry.failed > 0;
    case "skipped":
      return entry.skipped > 0;
  }
}

function sessionActivityTimestamp(
  items: SessionItem[],
  createdAt: string | null | undefined,
): number {
  if (items.some((item) => item.status === "queued" || item.status === "running")) {
    return Number.POSITIVE_INFINITY;
  }
  const finished = items.flatMap((item) => {
    const value = item.finishedAt === null ? null : serverTimestampMs(item.finishedAt);
    return value === null ? [] : [value];
  });
  if (finished.length > 0) return Math.max(...finished);
  return createdAt === null || createdAt === undefined
    ? 0
    : (serverTimestampMs(createdAt) ?? 0);
}

/** Merge live and persisted jobs by latest activity. Retried/enhanced work
 * moves to the top deterministically, while items inside one job retain their
 * original input order instead of being scrambled by individual finish time. */
// eslint-disable-next-line react-refresh/only-export-components -- exported for unit tests; the merge logic belongs next to the list it feeds.
export function mergeLedgerRows(
  items: SessionItem[],
  jobs: Record<string, SessionJob>,
  archivedEntries: HistoryEntry[],
): LedgerRow[] {
  const sessionGroups = new Map<string, SessionItem[]>();
  for (const item of items) {
    const group = sessionGroups.get(item.jobId);
    if (group === undefined) sessionGroups.set(item.jobId, [item]);
    else group.push(item);
  }

  const groups: LedgerGroup[] = [];
  for (const [jobId, groupItems] of sessionGroups) {
    groups.push({
      kind: "session",
      jobId,
      timestamp: sessionActivityTimestamp(groupItems, jobs[jobId]?.createdAt),
      createdTimestamp:
        jobs[jobId]?.createdAt === null || jobs[jobId]?.createdAt === undefined
          ? Number.POSITIVE_INFINITY
          : (serverTimestampMs(jobs[jobId].createdAt) ?? 0),
      items: groupItems,
    });
  }
  for (const entry of archivedEntries) {
    // During an SSE retry/enhancement the same job can briefly exist in both
    // stores. The live row wins, preventing a duplicate and preserving its key.
    if (sessionGroups.has(entry.job_id)) continue;
    groups.push({
      kind: "archive",
      jobId: entry.job_id,
      timestamp:
        serverTimestampMs(entry.finished_at ?? entry.created_at) ??
        serverTimestampMs(entry.created_at) ??
        0,
      createdTimestamp: serverTimestampMs(entry.created_at) ?? 0,
      entry,
    });
  }

  groups.sort((left, right) => {
    if (left.timestamp > right.timestamp) return -1;
    if (left.timestamp < right.timestamp) return 1;
    if (left.createdTimestamp > right.createdTimestamp) return -1;
    if (left.createdTimestamp < right.createdTimestamp) return 1;
    return left.jobId.localeCompare(right.jobId);
  });

  return groups.flatMap((group): LedgerRow[] =>
    group.kind === "session"
      ? group.items.map((item) => ({ kind: "session", key: item.key, item }))
      : [
          {
            kind: "archive",
            key: `archive:${group.entry.job_id}`,
            entry: group.entry,
          },
        ],
  );
}

/** One chronological ledger for live and persisted jobs. */
export function ItemList({
  t,
  items,
  jobs,
  archive,
  showCost,
  now,
  stats,
  settled,
  selectedKey,
  onSelect,
  onPreview,
  focusKey,
  onFocusKeyHandled,
  onRetry,
  onEnhance = async () => null,
  onDelete,
  canDelete,
  llmAvailable = false,
  llmDisabledReason,
}: {
  t: Dict;
  items: SessionItem[];
  jobs: Record<string, SessionJob>;
  archive?: ItemListArchive;
  showCost: boolean;
  now: number;
  stats: SessionStats;
  settled: boolean;
  selectedKey: string | null;
  onSelect: (key: string | null) => void;
  onPreview: (key: string, opener: HTMLElement) => void;
  focusKey: string | null;
  onFocusKeyHandled: () => void;
  onRetry: (item: SessionItem) => Promise<string | null>;
  onEnhance?: (item: SessionItem) => Promise<string | null>;
  onDelete: (item: SessionItem) => Promise<string | null>;
  canDelete: (item: SessionItem) => boolean;
  llmAvailable?: boolean;
  llmDisabledReason?: string;
}) {
  const listRef = useRef<HTMLDivElement>(null);
  const archivedEntries = archive?.entries ?? NO_ARCHIVE_ENTRIES;
  const rows = useMemo(
    () => mergeLedgerRows(items, jobs, archivedEntries),
    [archivedEntries, items, jobs],
  );

  // ---- view-layer filter. It covers both live and archived rows while row
  // numbering remains anchored to the unfiltered chronological ledger.
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const filterable = rows.length > 10;
  useEffect(() => {
    if (!filterable) {
      setQuery("");
      setStatusFilter("all");
    }
  }, [filterable]);
  const filterOn =
    filterable && (query.trim() !== "" || statusFilter !== "all");
  const visibleRows = useMemo(() => {
    if (!filterOn) return rows;
    const normalizedQuery = query.trim().toLowerCase();
    return rows.filter((row) => {
      if (row.kind === "session") {
        return (
          matchesStatus(row.item, statusFilter) &&
          (normalizedQuery === "" ||
            row.item.name
              .replace(/^https?:\/\//, "")
              .toLowerCase()
              .includes(normalizedQuery))
        );
      }
      return (
        matchesArchivedStatus(row.entry, statusFilter) &&
        (normalizedQuery === "" ||
          row.entry.names_preview.some((name) =>
            name
              .replace(/^https?:\/\//, "")
              .toLowerCase()
              .includes(normalizedQuery),
          ))
      );
    });
  }, [filterOn, query, rows, statusFilter]);

  const indexByKey = useMemo(
    () => new Map(rows.map((row, index) => [row.key, index])),
    [rows],
  );

  // Roving tabindex spans both current and persisted rows.
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const visibleKeys = useMemo(
    () => new Set(visibleRows.map((row) => row.key)),
    [visibleRows],
  );
  const effectiveActive =
    (activeKey !== null && visibleKeys.has(activeKey) ? activeKey : null) ??
    (selectedKey !== null && visibleKeys.has(selectedKey) ? selectedKey : null) ??
    visibleRows[0]?.key ??
    null;

  useEffect(() => {
    if (focusKey === null) return;
    setQuery("");
    setStatusFilter("all");
    const id = window.requestAnimationFrame(() => {
      const optionId = `opt-${focusKey.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
      const option = document.getElementById(optionId);
      if (option !== null) {
        setActiveKey(focusKey);
        onSelect(focusKey);
        option.focus();
      }
      onFocusKeyHandled();
    });
    return () => window.cancelAnimationFrame(id);
  }, [focusKey, onFocusKeyHandled, onSelect]);

  const onListKeyDown = (event: React.KeyboardEvent) => {
    const keys = ["ArrowDown", "ArrowUp", "Home", "End"];
    if (!keys.includes(event.key)) return;
    const list = listRef.current;
    if (list === null) return;
    const options = Array.from(
      list.querySelectorAll<HTMLElement>('[role="option"][data-ledger-key]'),
    );
    if (options.length === 0) return;
    const current = options.findIndex(
      (element) =>
        element === document.activeElement ||
        element.contains(document.activeElement),
    );
    let next: number;
    if (event.key === "Home") next = 0;
    else if (event.key === "End") next = options.length - 1;
    else if (event.key === "ArrowDown") {
      next = current < 0 ? 0 : Math.min(current + 1, options.length - 1);
    } else next = current < 0 ? 0 : Math.max(current - 1, 0);
    event.preventDefault();
    const element = options[next];
    if (element === undefined) return;
    const ledgerKey = element.dataset.ledgerKey ?? null;
    setActiveKey(ledgerKey);
    element.focus();
    const sessionKey = element.dataset.sessionKey;
    if (sessionKey !== undefined && element.getAttribute("aria-disabled") !== "true") {
      onSelect(sessionKey);
    } else {
      onSelect(null);
    }
  };

  const totalNote = `${stats.done}/${stats.total} ${t.statDone}`;
  const totalTime = fmtDur(stats.doneDurationMs);
  const hasArchivedRows =
    archivedEntries.length > 0 || (archive?.error ?? null) !== null;

  return (
    <div className={`flist${showCost ? "" : " nocost"}`}>
      {filterable && (
        <div className="filterrow">
          <input
            type="text"
            className="fin"
            value={query}
            placeholder={t.filterPh}
            aria-label={t.filterAria}
            spellCheck={false}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                setQuery("");
              }
            }}
          />
          <div className="fchips" role="group" aria-label={t.filterStatusAria}>
            {STATUS_FILTERS.map((filter) => (
              <button
                key={filter}
                type="button"
                className={
                  statusFilter === filter ? `fchip on ${filter}` : "fchip"
                }
                aria-pressed={statusFilter === filter}
                onClick={() => setStatusFilter(filter)}
              >
                {filter}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="lrow lhead" aria-hidden="true">
        <span />
        <span>{t.colName}</span>
        <span className="c-duration">{t.colDuration}</span>
        <span className="c-finished">{t.colFinished}</span>
        {showCost && <span className="c-cost">{t.colCost}</span>}
        <span className="c-status">{t.colStatus}</span>
      </div>
      <div
        ref={listRef}
        role="listbox"
        aria-label={t.itemsAria}
        onKeyDown={onListKeyDown}
      >
        {visibleRows.map((row) => {
          const index = indexByKey.get(row.key) ?? 0;
          if (row.kind === "session") {
            return (
              <ItemRow
                key={row.key}
                t={t}
                item={row.item}
                index={index}
                showCost={showCost}
                now={now}
                selected={row.key === selectedKey}
                tabbable={row.key === effectiveActive}
                onPreview={onPreview}
                onRowFocus={(key) => {
                  setActiveKey(key);
                  onSelect(key);
                }}
                onRetry={onRetry}
                onEnhance={onEnhance}
                onDelete={onDelete}
                canDelete={canDelete(row.item)}
                llmAvailable={llmAvailable}
                llmDisabledReason={llmDisabledReason}
              />
            );
          }
          if (archive === undefined) return null;
          return (
            <ArchivedJobRows
              key={row.key}
              {...archive.rowProps}
              t={t}
              entries={[row.entry]}
              error={null}
              showCost={showCost}
              startIndex={index}
              llmAvailable={llmAvailable}
              llmDisabledReason={llmDisabledReason}
              tabbableJobId={
                row.key === effectiveActive ? row.entry.job_id : null
              }
              onRowFocus={(jobId) => {
                setActiveKey(`archive:${jobId}`);
                onSelect(null);
              }}
            />
          );
        })}
        {archive !== undefined && archive.error !== null && (
          <ArchivedJobRows
            {...archive.rowProps}
            t={t}
            entries={null}
            error={archive.error}
            showCost={showCost}
            startIndex={rows.length}
            llmAvailable={llmAvailable}
            llmDisabledReason={llmDisabledReason}
            tabbableJobId={null}
          />
        )}
      </div>
      {filterOn && visibleRows.length === 0 && (
        <p className="fempty">{t.filterNoMatch}</p>
      )}
      {rows.length === 0 && !hasArchivedRows && archive?.entries !== null && (
        <p className="fempty">{t.emptyWorkspace}</p>
      )}
      {settled && !hasArchivedRows && rows.length > 0 && (
        <div className="lrow totals">
          <span />
          <span className="t-lbl">
            {t.total}
            {filterOn && (
              <span className="t-shownote">
                {" · "}
                {t.filterShown(visibleRows.length, rows.length)}
              </span>
            )}
          </span>
          <span className="c-duration">{totalTime}</span>
          <span className="c-finished" />
          {showCost && <span className="c-cost">{fmtCost(stats.costTotal)}</span>}
          <span className="t-note">{totalNote}</span>
          <span className="totmeta">
            {totalTime}
            {showCost && ` · ${fmtCost(stats.costTotal)}`}
          </span>
        </div>
      )}
    </div>
  );
}
