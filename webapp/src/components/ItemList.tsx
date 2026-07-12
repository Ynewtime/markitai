import { useRef, useState } from "react";
import type { SessionItem, SessionStats } from "../hooks/useJobs";
import type { Dict } from "../i18n";
import { fmtCost, fmtDur } from "../lib/format";
import { ItemRow } from "./ItemRow";

/** The session ledger — one compact shape for the whole workspace lifetime
 * (converting and done states share it, so the grid never reflows mid-run).
 *
 * Real listbox contract: roving tabindex, ArrowUp/Down/Home/End move focus
 * through every row (failed/skipped rows stay reachable); selection follows
 * focus onto previewable rows. The accounting TOTAL double-rule appears only
 * once the session has settled. */
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
  onOpenSettings,
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
  onOpenSettings: () => void;
}) {
  const listRef = useRef<HTMLDivElement>(null);
  // Roving tabindex: the last-focused row stays the single Tab stop.
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const effectiveActive =
    (activeKey !== null && items.some((i) => i.key === activeKey) ? activeKey : null) ??
    selectedKey ??
    items[0]?.key ??
    null;

  const onListKeyDown = (e: React.KeyboardEvent) => {
    const keys = ["ArrowDown", "ArrowUp", "Home", "End"];
    if (!keys.includes(e.key)) return;
    const list = listRef.current;
    if (list === null) return;
    const options = Array.from(list.querySelectorAll<HTMLElement>('[role="option"]'));
    if (options.length === 0) return;
    const current = options.findIndex((el) => el === document.activeElement);
    let next: number;
    if (e.key === "Home") next = 0;
    else if (e.key === "End") next = options.length - 1;
    else if (e.key === "ArrowDown") next = current < 0 ? 0 : Math.min(current + 1, options.length - 1);
    else next = current < 0 ? 0 : Math.max(current - 1, 0);
    e.preventDefault();
    const el = options[next];
    if (el === undefined) return;
    el.focus();
    // Selection follows focus onto previewable rows only.
    const item = items[next];
    if (item !== undefined && el.getAttribute("aria-disabled") !== "true") {
      onSelect(item.key);
    }
  };

  const totalNote = `${stats.done}/${stats.total} ${t.statDone}`;
  const totalTime = fmtDur(stats.doneDurationMs);
  return (
    <div className={`flist${showCost ? "" : " nocost"}`}>
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
        {items.map((item, i) => (
          <ItemRow
            key={item.key}
            t={t}
            item={item}
            index={i}
            showCost={showCost}
            now={now}
            selected={item.key === selectedKey}
            tabbable={item.key === effectiveActive}
            llmConfigured={llmConfigured}
            onSelect={onSelect}
            onOpenSettings={onOpenSettings}
            onRowFocus={setActiveKey}
          />
        ))}
      </div>
      {settled && (
        <div className="lrow totals">
          <span />
          <span className="t-lbl">{t.total}</span>
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
