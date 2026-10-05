// A submission that carried several items — a folder, several files, a URL list
// or a batch of URLs — reads as one row in the ledger. It shows the same facts
// the other rows do, aggregated over its items, and opens into them.
import { memo } from "preact/compat";
import type { Dict } from "../i18n/index.ts";
import { fmtCost, fmtDateTime, fmtDur } from "../lib/format.ts";
import { groupItemRows, groupLabel, sessionStats, type LedgerRow, type SessionItem } from "../lib/session.ts";
import { Icon } from "./icons.tsx";

export interface GroupRowProps {
  t: Dict;
  row: Extract<LedgerRow, { kind: "group" }>;
  index: number;
  open: boolean;
  tabbable: boolean;
  onToggle: (key: string) => void;
  onRowFocus: (key: string) => void;
}

/** The items a group holds, in the order the submission sent them. */
export function groupRows(row: Extract<LedgerRow, { kind: "group" }>): LedgerRow[] {
  return groupItemRows(row);
}

export const GroupRow = memo(function GroupRow({
  t,
  row,
  index,
  open,
  tabbable,
  onToggle,
  onRowFocus,
}: GroupRowProps) {
  const items: SessionItem[] = row.items;
  const stats = sessionStats(items);
  const label = groupLabel(row.label, row.jobId);
  const done = Math.max(0, stats.done);
  const parts = [`${done} ${t.statusDone}`];
  if (stats.failed > 0) parts.push(`${stats.failed} ${t.statusFailed}`);
  if (stats.skipped > 0) parts.push(`${stats.skipped} ${t.statusSkipped}`);
  const counts = parts.join(" · ");
  const running = items.some((item) => item.status === "running" || item.status === "queued");
  const finishedAt = items.reduce<string | null>((latest, item) => {
    if (item.finishedAt === null) return latest;
    return latest === null || item.finishedAt > latest ? item.finishedAt : latest;
  }, null);
  const duration = stats.doneDurationMs > 0 ? fmtDur(stats.doneDurationMs) : "-";
  const finished = running ? t.statusRunning : fmtDateTime(finishedAt);
  const hasLlm = stats.hasCost || items.some((item) => item.llmEnhanced);
  const enhanced = items.filter((item) => item.llmEnhanced).length;
  const llmLabel = !hasLlm
    ? t.baseTag
    : enhanced === 0 || enhanced === items.length
      ? t.llmTag
      : `${t.llmTag} ${enhanced}/${items.length}`;
  const pill = stats.failed > 0 ? "state-pill is-err" : stats.skipped === items.length ? "state-pill is-skip" : "state-pill is-ok";
  const kind = items.every((item) => item.kind === "url") ? "Globe" : "FolderSimple";
  // A group whose items carry warnings speaks once, the way the notice does.
  const warnings = new Set(items.flatMap((item) => item.warnings));
  const label_aria = open ? t.groupCollapse(label) : t.groupExpand(label);

  return (
    <div
      class="lg-row is-actionable is-group"
      role="option"
      aria-selected={false}
      aria-expanded={open}
      aria-label={label_aria}
      title={label_aria}
      data-ledger-key={row.key}
      data-group-key={row.key}
      tabIndex={tabbable ? 0 : -1}
      onClick={() => onToggle(row.key)}
      onFocus={() => onRowFocus(row.key)}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onToggle(row.key);
        }
        if (event.key === "ArrowRight" && !open) onToggle(row.key);
        if (event.key === "ArrowLeft" && open) onToggle(row.key);
      }}
    >
      <span class="cell-num">{String(index + 1).padStart(2, "0")}</span>
      <span class="cell-name" title={label}>
        <span class={open ? "group-caret is-open" : "group-caret"} aria-hidden="true">
          <Icon name={open ? "CaretDown" : "CaretRight"} size={11} />
        </span>
        <Icon name={kind} size={14} />
        <span class="fname">{label}</span>
        <span class="tag tag-count">{t.itemsNotice(items.length)}</span>
        {warnings.size > 0 && (
          <span class="mark is-warn" aria-label={t.itemWarnings(warnings.size)} title={t.itemWarningsTitle}>
            <span aria-hidden="true">
              <Icon name="WarningFill" size={13} />
            </span>
          </span>
        )}
      </span>
      <span class="cell-time">{duration}</span>
      <span class="cell-done">{finished}</span>
      <span class="cell-cost" title={hasLlm && stats.costTotal > 0 ? fmtCost(stats.costTotal) : undefined}>
        <span class={hasLlm ? "tag is-llm" : "tag"}>{llmLabel}</span>
        {hasLlm && stats.costTotal > 0 && <span class="tag-price">{fmtCost(stats.costTotal)}</span>}
      </span>
      <span class="cell-state">
        <span class={pill} title={counts}>
          {counts}
        </span>
      </span>
      <span class="row-facts">
        <span class="fact">{duration}</span>
        <span class="fact is-time">{finished}</span>
        <span class="fact">
          {llmLabel}
          {hasLlm && stats.costTotal > 0 && ` ${fmtCost(stats.costTotal)}`}
        </span>
        <span class="fact">{t.itemsNotice(items.length)}</span>
        <span class="fact">{open ? t.groupCollapse(label) : t.groupExpand(label)}</span>
      </span>
    </div>
  );
});
