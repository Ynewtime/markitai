import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { SessionItem, SessionStats } from "../hooks/useJobs";
import type { Dict } from "../i18n";
import { fmtCost, fmtDur } from "../lib/format";
import { ItemRow } from "./ItemRow";

/** Status facets share the chip vocabulary (mono words, English in both
 * locales — they mirror the status chips). */
const STATUS_FILTERS = ["all", "done", "failed", "skipped"] as const;
type StatusFilter = (typeof STATUS_FILTERS)[number];

function matchesStatus(i: SessionItem, f: StatusFilter): boolean {
  switch (f) {
    case "all":
      return true;
    case "done":
      return i.status === "done" && !i.skipped;
    case "failed":
      return i.status === "error";
    case "skipped":
      return i.status === "done" && i.skipped;
  }
}

/** The session ledger — one compact shape for the whole workspace lifetime
 * (converting and done states share it, so the grid never reflows mid-run).
 *
 * Real listbox contract: roving tabindex, ArrowUp/Down/Home/End move focus
 * through every row (failed/skipped rows stay reachable); selection follows
 * focus onto previewable rows. The accounting TOTAL double-rule appears only
 * once the session has settled.
 *
 * Above 10 items a compact filter row appears (name substring + status
 * chips). Filtering is pure view: row numbers and the TOTAL line keep the
 * full session; keyboard navigation walks the visible subset only. */
export function ItemList({
  t,
  items,
  showCost,
  now,
  stats,
  settled,
  llmConfigured,
  selectedKey,
  onSelect,
  onPreview,
  focusKey,
  onFocusKeyHandled,
  onOpenSettings,
  onRetry,
  archivedRows,
  hasArchivedRows = false,
}: {
  t: Dict;
  items: SessionItem[];
  showCost: boolean;
  now: number;
  stats: SessionStats;
  settled: boolean;
  llmConfigured: boolean;
  selectedKey: string | null;
  onSelect: (key: string) => void;
  onPreview: (key: string, opener: HTMLElement) => void;
  focusKey: string | null;
  onFocusKeyHandled: () => void;
  onOpenSettings: () => void;
  onRetry: (item: SessionItem) => Promise<string | null>;
  archivedRows?: ReactNode;
  hasArchivedRows?: boolean;
}) {
  const listRef = useRef<HTMLDivElement>(null);

  // ---- view-layer filter (session-scoped; resets when the ledger shrinks
  // back under the threshold so a stale query can never hide fresh rows).
  const [query, setQuery] = useState("");
  const [statusF, setStatusF] = useState<StatusFilter>("all");
  const filterable = items.length > 10;
  useEffect(() => {
    if (!filterable) {
      setQuery("");
      setStatusF("all");
    }
  }, [filterable]);
  const filterOn = filterable && (query.trim() !== "" || statusF !== "all");
  const visible = useMemo(() => {
    if (!filterOn) return items;
    const q = query.trim().toLowerCase();
    return items.filter(
      (i) =>
        matchesStatus(i, statusF) &&
        (q === "" || i.name.replace(/^https?:\/\//, "").toLowerCase().includes(q)),
    );
  }, [items, filterOn, query, statusF]);

  // Ledger numbering stays positional in the full session, filtered or not.
  const indexByKey = useMemo(() => {
    const m = new Map<string, number>();
    items.forEach((it, i) => m.set(it.key, i));
    return m;
  }, [items]);

  // Roving tabindex: the last-focused row stays the single Tab stop.
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const effectiveActive =
    (activeKey !== null && visible.some((i) => i.key === activeKey) ? activeKey : null) ??
    (selectedKey !== null && visible.some((i) => i.key === selectedKey)
      ? selectedKey
      : null) ??
    visible[0]?.key ??
    null;

  useEffect(() => {
    if (focusKey === null) return;
    setQuery("");
    setStatusF("all");
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

  const onListKeyDown = (e: React.KeyboardEvent) => {
    const keys = ["ArrowDown", "ArrowUp", "Home", "End"];
    if (!keys.includes(e.key)) return;
    const list = listRef.current;
    if (list === null) return;
    const options = Array.from(list.querySelectorAll<HTMLElement>('[role="option"]'));
    if (options.length === 0) return;
    // Arrows also work from nested controls (retry) — walk from their row.
    const current = options.findIndex(
      (el) => el === document.activeElement || el.contains(document.activeElement),
    );
    let next: number;
    if (e.key === "Home") next = 0;
    else if (e.key === "End") next = options.length - 1;
    else if (e.key === "ArrowDown") next = current < 0 ? 0 : Math.min(current + 1, options.length - 1);
    else next = current < 0 ? 0 : Math.max(current - 1, 0);
    e.preventDefault();
    const el = options[next];
    if (el === undefined) return;
    el.focus();
    // Selection follows focus for current-session rows. Archived options
    // own their activation and remain in place when opened.
    const sessionKey = el.dataset.sessionKey;
    const item = sessionKey === undefined ? undefined : visible.find((entry) => entry.key === sessionKey);
    if (item !== undefined && el.getAttribute("aria-disabled") !== "true") {
      onSelect(item.key);
    }
  };

  const totalNote = `${stats.done}/${stats.total} ${t.statDone}`;
  const totalTime = fmtDur(stats.doneDurationMs);
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
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                setQuery("");
              }
            }}
          />
          <div className="fchips" role="group" aria-label={t.filterStatusAria}>
            {STATUS_FILTERS.map((f) => (
              <button
                key={f}
                type="button"
                className={statusF === f ? `fchip on ${f}` : "fchip"}
                aria-pressed={statusF === f}
                onClick={() => setStatusF(f)}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="lrow lhead" aria-hidden="true">
        <span />
        <span>{t.colName}</span>
        <span className="c-time">{t.colTime}</span>
        {showCost && <span className="c-cost">{t.colCost}</span>}
        <span className="c-status">{t.colStatus}</span>
      </div>
      <div
        ref={listRef}
        role="listbox"
        aria-label={t.itemsAria}
        onKeyDown={onListKeyDown}
      >
        {visible.map((item) => (
          <ItemRow
            key={item.key}
            t={t}
            item={item}
            index={indexByKey.get(item.key) ?? 0}
            showCost={showCost}
            now={now}
            selected={item.key === selectedKey}
            tabbable={item.key === effectiveActive}
            llmConfigured={llmConfigured}
            onPreview={onPreview}
            onOpenSettings={onOpenSettings}
            onRowFocus={setActiveKey}
            onRetry={onRetry}
          />
        ))}
        {archivedRows}
      </div>
      {filterOn && visible.length === 0 && (
        <p className="fempty">{t.filterNoMatch}</p>
      )}
      {settled && !hasArchivedRows && (
        <div className="lrow totals">
          <span />
          <span className="t-lbl">
            {t.total}
            {filterOn && (
              <span className="t-shownote">
                {" · "}
                {t.filterShown(visible.length, items.length)}
              </span>
            )}
          </span>
          <span className="c-time">{totalTime}</span>
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
