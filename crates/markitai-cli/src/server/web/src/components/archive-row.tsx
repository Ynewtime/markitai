// A saved job in the same ledger language as live rows: names, duration,
// finish time, Base/LLM, and a status mark (one item) or a dark count pill.
import { memo } from "preact/compat";
import { useRef, useState } from "preact/hooks";
import type { HistoryEntry } from "../api/types.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { displayName, fmtBytes, fmtCost, fmtDateTime, fmtDur } from "../lib/format.ts";
import { PRICE_WORDS, priceText } from "../lib/pricing.ts";
import { ConfirmPopover } from "./confirm-popover.tsx";
import { Icon } from "./icons.tsx";

export interface ArchiveRowProps {
  t: Dict;
  locale: Locale;
  entry: HistoryEntry;
  index: number;
  tabbable: boolean;
  busy: boolean;
  rowError: string | null;
  llmAvailable: boolean;
  llmDisabledReason: string;
  onOpen: (jobId: string, opener: HTMLElement) => void;
  onRetry: (jobId: string) => Promise<string | null>;
  onEnhance: (jobId: string) => Promise<string | null>;
  onDelete: (entry: HistoryEntry) => Promise<boolean>;
  onRowFocus: (jobId: string) => void;
}

function counts(entry: HistoryEntry, t: Dict): string {
  const parts = [`${Math.max(0, entry.done - entry.skipped)} ${t.statusDone}`];
  if (entry.failed > 0) parts.push(`${entry.failed} ${t.statusFailed}`);
  if (entry.skipped > 0) parts.push(`${entry.skipped} ${t.statusSkipped}`);
  return parts.join(" · ");
}

export const ArchiveRow = memo(function ArchiveRow({
  t,
  locale,
  entry,
  index,
  tabbable,
  busy,
  rowError,
  llmAvailable,
  llmDisabledReason,
  onOpen,
  onRetry,
  onEnhance,
  onDelete,
  onRowFocus,
}: ArchiveRowProps) {
  const row = useRef<HTMLDivElement>(null);
  const [working, setWorking] = useState<"retry" | "enhance" | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const more = entry.total - entry.names_preview.length;
  const names = entry.names_preview.map(displayName).join(", ") + (more > 0 ? ` ${t.histMore(more)}` : "");
  const first = displayName(entry.names_preview[0] ?? entry.job_id);
  const single = entry.total === 1;
  const result = single ? (entry.failed > 0 ? "err" : entry.skipped > 0 ? "skip" : "ok") : null;
  const resultWord = result === "err" ? t.statusFailed : result === "skip" ? t.statusSkipped : t.statusDone;
  const hasLlm = entry.llm_enhanced > 0 || (entry.cost_usd !== null && entry.cost_usd > 0);
  const llmLabel = !hasLlm
    ? t.baseTag
    : entry.llm_enhanced === 0 || entry.llm_enhanced === entry.total
      ? t.llmTag
      : `${t.llmTag} ${entry.llm_enhanced}/${entry.total}`;
  const duration = entry.duration_ms === null ? "-" : fmtDur(entry.duration_ms);
  const finished = fmtDateTime(entry.finished_at);
  const enhanceable = entry.retryable && single && entry.done === 1 && entry.failed === 0 && entry.skipped === 0 && entry.llm_enhanced === 0;
  const retryable = entry.retryable && single && (entry.failed === 1 || entry.skipped === 1);
  const blocked = busy || working !== null;
  const pill = entry.failed > 0 ? "state-pill is-err" : entry.skipped === entry.total ? "state-pill is-skip" : "state-pill is-ok";

  const act = async (kind: "retry" | "enhance") => {
    if (blocked) return;
    setWorking(kind);
    setActionError(null);
    const error = await (kind === "retry" ? onRetry(entry.job_id) : onEnhance(entry.job_id));
    if (error !== null) {
      setActionError(`${kind === "retry" ? t.retryFailed : t.llmEnhanceFailed}: ${error}`);
      setWorking(null);
    }
  };

  const remove = async () => {
    const options = [...(row.current?.closest('[role="listbox"]')?.querySelectorAll<HTMLElement>('[role="option"]') ?? [])];
    const at = row.current ? options.indexOf(row.current) : -1;
    const next = at < 0 ? null : (options[at + 1] ?? options[at - 1] ?? null);
    const removed = await onDelete(entry);
    if (removed) requestAnimationFrame(() => next?.isConnected && next.focus());
    return removed;
  };

  const open = (opener: HTMLElement) => {
    if (!blocked) onOpen(entry.job_id, opener);
  };
  const error = actionError ?? rowError;

  return (
    <div
      ref={row}
      class="lg-row is-actionable"
      role="option"
      aria-selected={false}
      aria-disabled={blocked || undefined}
      aria-label={`${t.histOpen} ${first}`}
      data-ledger-key={`archive:${entry.job_id}`}
      tabIndex={tabbable ? 0 : -1}
      onClick={(event) => {
        event.currentTarget.focus({ preventScroll: true });
        open(event.currentTarget);
      }}
      onFocus={() => onRowFocus(entry.job_id)}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          open(event.currentTarget);
        }
      }}
    >
      <span class="cell-num">{String(index + 1).padStart(2, "0")}</span>
      <span class="cell-name" title={names}>
        <Icon name={entry.kinds_preview[0] === "url" ? "Globe" : "FileText"} size={14} />
        <span class="fname">{names}</span>
        {entry.origin === "cli" && <span class="tag tag-origin">{t.originCli}</span>}
      </span>
      <span class="cell-time">{duration}</span>
      <span class="cell-done" title={entry.finished_at ?? undefined}>
        {finished}
      </span>
      <span class="cell-cost" title={priceText(entry.cost_usd, entry.pricing, PRICE_WORDS[locale]) || undefined}>
        <span class={hasLlm ? "tag is-llm" : "tag"}>{llmLabel}</span>
        {hasLlm && entry.cost_usd !== null && <span class="tag-price">{fmtCost(entry.cost_usd)}</span>}
      </span>
      <span class="cell-state">
        {result === null ? (
          <span class={pill} title={counts(entry, t)}>
            {counts(entry, t)}
          </span>
        ) : (
          <span
            class={`mark is-${result}${result === "skip" ? " has-tip" : ""}`}
            title={resultWord}
            data-tip={result === "skip" ? resultWord : undefined}
            aria-label={resultWord}
            tabIndex={result === "skip" ? 0 : undefined}
          >
            <span aria-hidden="true">{result === "err" ? "×" : result === "skip" ? <Icon name="WarningFill" size={17} /> : "✓"}</span>
            <span class="sr-only">{resultWord}</span>
          </span>
        )}
        <span class="row-tools">
          {enhanceable && (
            <button
              type="button"
              class={actionError ? "row-icon is-failed" : "row-icon"}
              aria-label={t.enhanceWithLlm(first)}
              title={actionError ?? (llmAvailable ? t.enhanceWithLlm(first) : llmDisabledReason)}
              disabled={!llmAvailable || blocked}
              onClick={(event) => {
                event.stopPropagation();
                void act("enhance");
              }}
            >
              {working === "enhance" ? <span class="spinner" aria-hidden="true" /> : <Icon name="MagicWand" size={14} />}
            </button>
          )}
          {retryable && (
            <button
              type="button"
              class="row-icon"
              aria-label={t.retryAria(first)}
              title={t.retryAria(first)}
              disabled={blocked}
              onClick={(event) => {
                event.stopPropagation();
                void act("retry");
              }}
            >
              {working === "retry" ? <span class="spinner" aria-hidden="true" /> : <Icon name="ArrowCounterClockwise" size={13} />}
            </button>
          )}
          <ConfirmPopover
            triggerLabel={t.histDeleteAria(first)}
            title={t.deleteItemTitle(first)}
            description={t.deleteItemDescription}
            confirmLabel={t.deletePermanently}
            cancelLabel={t.cancel}
            busyLabel={t.deleting}
            disabled={blocked}
            onConfirm={remove}
          />
        </span>
      </span>
      <span class="row-facts">
        <span class="fact">{duration}</span>
        <span class="fact is-time">{finished}</span>
        <span class="fact">
          {llmLabel}
          {hasLlm && entry.cost_usd !== null && ` ${fmtCost(entry.cost_usd)}`}
        </span>
        <span class="fact">{t.histStorageSize(fmtBytes(entry.size_bytes))}</span>
      </span>
      {error && (
        <span class="row-note is-err" role="alert">
          {error}
        </span>
      )}
    </div>
  );
});
