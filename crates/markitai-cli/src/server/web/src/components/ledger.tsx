// The ledger: live and saved jobs in one chronological listbox, numbered in
// that order, with a name/status filter once it holds more than ten rows.
// Arrow keys, Home and End move between rows; one row is the tab stop.
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { HistoryEntry } from "../api/types.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { fmtCost, fmtDur } from "../lib/format.ts";
import type { ItemRequestFailure } from "../lib/pricing.ts";
import type { NotificationModel } from "./notification.tsx";
import { mergeLedger, rowMatches, STATUS_FILTERS, type SessionItem, type SessionJob, type SessionStats, type StatusFilter } from "../lib/session.ts";
import { ArchiveRow } from "./archive-row.tsx";
import { domKey, LedgerRow } from "./ledger-row.tsx";

const NO_ENTRIES: HistoryEntry[] = [];

export interface ArchiveProps {
  entries: HistoryEntry[] | null;
  error: string | null;
  busy: Record<string, unknown>;
  rowErrors: Record<string, unknown>;
  onRefresh: () => void;
  onOpen: (jobId: string, opener: HTMLElement) => void;
  onRetry: (jobId: string) => Promise<unknown>;
  onEnhance: (jobId: string) => Promise<unknown>;
  onDelete: (entry: HistoryEntry) => Promise<boolean>;
}

export function Ledger({
  t,
  locale,
  items,
  jobs,
  archive,
  stats,
  settled,
  selectedKey,
  focusKey,
  llmAvailable,
  llmDisabledReason,
  onSelect,
  onFocusHandled,
  onPreview,
  onRetry,
  onEnhance,
  onDelete,
  describe,
  onDownloadError,
  requestFailures,
  onItemNotice,
  onNotice,
}: {
  t: Dict;
  locale: Locale;
  items: SessionItem[];
  jobs: Record<string, SessionJob>;
  archive: ArchiveProps;
  stats: SessionStats;
  settled: boolean;
  selectedKey: string | null;
  focusKey: string | null;
  llmAvailable: boolean;
  llmDisabledReason: string;
  onSelect: (key: string | null) => void;
  onFocusHandled: () => void;
  onPreview: (key: string, opener: HTMLElement) => void;
  onRetry: (item: SessionItem) => Promise<unknown>;
  onEnhance: (item: SessionItem) => Promise<unknown>;
  onDelete: (item: SessionItem) => Promise<unknown>;
  describe: (error: unknown) => { text: string; detail: string };
  onDownloadError: (error: unknown) => void;
  requestFailures: Record<string, ItemRequestFailure>;
  onItemNotice: (item: SessionItem, opener?: HTMLElement) => void;
  onNotice: (note: NotificationModel, opener?: HTMLElement) => void;
}) {
  const list = useRef<HTMLDivElement>(null);
  const saved = archive.entries ?? NO_ENTRIES;
  const rows = useMemo(() => mergeLedger(items, jobs, saved), [items, jobs, saved]);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<StatusFilter>("all");
  const filterable = rows.length > 10;
  useEffect(() => {
    if (!filterable) {
      setQuery("");
      setStatus("all");
    }
  }, [filterable]);
  const filtering = filterable && (query.trim() !== "" || status !== "all");
  const visible = useMemo(() => (filtering ? rows.filter((row) => rowMatches(row, status, query)) : rows), [filtering, rows, status, query]);
  const position = useMemo(() => new Map(rows.map((row, index) => [row.key, index])), [rows]);

  const [activeKey, setActiveKey] = useState<string | null>(null);
  const visibleKeys = useMemo(() => new Set(visible.map((row) => row.key)), [visible]);
  const tabStop =
    (activeKey !== null && visibleKeys.has(activeKey) ? activeKey : null) ??
    (selectedKey !== null && visibleKeys.has(selectedKey) ? selectedKey : null) ??
    visible[0]?.key ??
    null;
  const focusRow = useCallback(
    (key: string) => {
      setActiveKey(key);
      onSelect(key);
    },
    [onSelect],
  );
  const focusArchive = useCallback(
    (jobId: string) => {
      setActiveKey(`archive:${jobId}`);
      onSelect(null);
    },
    [onSelect],
  );

  // Retrying a saved job adopts it; move focus to its row once it is rendered.
  useEffect(() => {
    if (focusKey === null) return;
    setQuery("");
    setStatus("all");
    const frame = requestAnimationFrame(() => {
      const node = document.getElementById(domKey(focusKey));
      if (node) {
        setActiveKey(focusKey);
        onSelect(focusKey);
        node.focus();
      }
      onFocusHandled();
    });
    return () => cancelAnimationFrame(frame);
  }, [focusKey, onFocusHandled, onSelect]);

  const onKeyDown = (event: KeyboardEvent) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key) || !list.current) return;
    const options = [...list.current.querySelectorAll<HTMLElement>('[role="option"][data-ledger-key]')];
    if (!options.length) return;
    const current = options.findIndex((node) => node === document.activeElement || node.contains(document.activeElement));
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? options.length - 1
          : event.key === "ArrowDown"
            ? current < 0
              ? 0
              : Math.min(current + 1, options.length - 1)
            : current < 0
              ? 0
              : Math.max(current - 1, 0);
    event.preventDefault();
    const node = options[next];
    if (!node) return;
    setActiveKey(node.dataset.ledgerKey ?? null);
    node.focus();
    const session = node.dataset.sessionKey;
    onSelect(session !== undefined && node.getAttribute("aria-disabled") !== "true" ? session : null);
  };

  const hasArchive = saved.length > 0 || archive.error !== null;
  const labels: Record<StatusFilter, string> = { all: t.filterAll, done: t.filterDone, failed: t.filterFailed, skipped: t.filterSkipped };
  const totalTime = fmtDur(stats.doneDurationMs);

  return (
    <div class="ledger">
      {filterable && (
        <div class="ledger-filter">
          <input
            type="text"
            class="filter-input"
            value={query}
            placeholder={t.filterPh}
            aria-label={t.filterAria}
            spellcheck={false}
            onInput={(event) => setQuery(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                setQuery("");
              }
            }}
          />
          <div class="filter-chips" role="group" aria-label={t.filterStatusAria}>
            {STATUS_FILTERS.map((value) => (
              <button
                key={value}
                type="button"
                class={status === value ? `filter-chip is-on is-${value}` : "filter-chip"}
                aria-pressed={status === value}
                onClick={() => setStatus(value)}
              >
                {labels[value]}
              </button>
            ))}
          </div>
        </div>
      )}
      <div class="lg-row lg-head" aria-hidden="true">
        <span />
        <span>{t.colName}</span>
        <span class="cell-time">{t.colDuration}</span>
        <span class="cell-done">{t.colFinished}</span>
        <span class="cell-cost">{t.colCost}</span>
        <span class="cell-state">{t.colStatus}</span>
      </div>
      <div ref={list} role="listbox" aria-label={t.itemsAria} onKeyDown={onKeyDown}>
        {visible.map((row) => {
          const index = position.get(row.key) ?? 0;
          if (row.kind === "session") {
            return (
              <LedgerRow
                key={row.key}
                t={t}
                locale={locale}
                item={row.item}
                index={index}
                selected={row.key === selectedKey}
                tabbable={row.key === tabStop}
                canDelete={jobs[row.item.jobId]?.status !== undefined && jobs[row.item.jobId]?.status !== "running"}
                llmAvailable={llmAvailable}
                llmDisabledReason={llmDisabledReason}
                onPreview={onPreview}
                onRowFocus={focusRow}
                onRetry={onRetry}
                onEnhance={onEnhance}
                onDelete={onDelete}
                describe={describe}
                onDownloadError={onDownloadError}
                requestFailure={requestFailures[row.key] ?? null}
                onItemNotice={onItemNotice}
              />
            );
          }
          const jobId = row.entry.job_id;
          return (
            <ArchiveRow
              key={row.key}
              t={t}
              locale={locale}
              entry={row.entry}
              index={index}
              tabbable={row.key === tabStop}
              busy={archive.busy[jobId] !== undefined}
              rowError={archive.rowErrors[jobId] ?? null}
              llmAvailable={llmAvailable}
              llmDisabledReason={llmDisabledReason}
              onOpen={archive.onOpen}
              onRetry={archive.onRetry}
              onEnhance={archive.onEnhance}
              onDelete={archive.onDelete}
              onRowFocus={focusArchive}
              describe={describe}
              onNotice={onNotice}
            />
          );
        })}
        {archive.error !== null && (
          <div class="lg-row ledger-state" role="option" aria-selected={false} data-ledger-key="archive:error" tabIndex={-1}>
            <span />
            <span class="line-error">{archive.error}</span>
            <button type="button" class="text-btn" onClick={archive.onRefresh}>
              {t.retryLoad}
            </button>
          </div>
        )}
      </div>
      {filtering && visible.length === 0 && <p class="ledger-empty">{t.filterNoMatch}</p>}
      {rows.length === 0 && !hasArchive && archive.entries !== null && <p class="ledger-empty">{t.emptyWorkspace}</p>}
      {settled && !hasArchive && rows.length > 0 && (
        <div class="lg-row lg-total">
          <span />
          <span class="total-label">
            {t.total}
            {filtering && <span class="total-shown"> · {t.filterShown(visible.length, rows.length)}</span>}
          </span>
          <span class="cell-time">{totalTime}</span>
          <span class="cell-done" />
          <span class="cell-cost">{fmtCost(stats.costTotal)}</span>
          <span class="total-note">
            {stats.done}/{stats.total} {t.statusDone}
          </span>
          <span class="total-facts">
            {totalTime} · {fmtCost(stats.costTotal)}
          </span>
        </div>
      )}
    </div>
  );
}
