import { useRef, useState } from "react";
import type { HistoryEntry, JobSnapshot } from "../api/types";
import type { Dict } from "../i18n";
import { fmtBytes, fmtCost, fmtDateTime, fmtDur } from "../lib/format";
import { ConfirmDeletePopover } from "./ConfirmDeletePopover";
import {
  FileTextIcon,
  GlobeIcon,
  MagicWandIcon,
  RotateCcwIcon,
  WarningIcon,
} from "./icons";

function statsLine(entry: HistoryEntry, t: Dict): string {
  const succeeded = Math.max(0, entry.done - entry.skipped);
  const parts = [`${succeeded} ${t.statDone}`];
  if (entry.failed > 0) parts.push(`${entry.failed} ${t.statFailed}`);
  if (entry.skipped > 0) parts.push(`${entry.skipped} ${t.statSkipped}`);
  return parts.join(" · ");
}

export interface ArchivedJobRowsProps {
  t: Dict;
  entries: HistoryEntry[] | null;
  error: string | null;
  actions: Record<string, "open" | "delete">;
  rowErrors: Record<string, string>;
  showCost: boolean;
  startIndex: number;
  onRefresh: () => Promise<void>;
  onOpen: (jobId: string, opener: HTMLElement) => Promise<JobSnapshot | null>;
  onRetry: (jobId: string) => Promise<string | null>;
  onEnhance?: (jobId: string) => Promise<string | null>;
  onDelete: (jobId: string) => Promise<boolean>;
  announce: (message: string) => void;
  llmAvailable?: boolean;
  llmDisabledReason?: string;
  tabbableJobId?: string | null;
  onRowFocus?: (jobId: string) => void;
}

/** Persisted jobs use the same ledger row and terminal status language. */
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
  onRetry,
  onEnhance = async () => null,
  onDelete,
  announce,
  llmAvailable = false,
  llmDisabledReason,
  tabbableJobId,
  onRowFocus,
}: ArchivedJobRowsProps) {
  const rowRefs = useRef(new Map<string, HTMLDivElement>());
  const [retrying, setRetrying] = useState<string | null>(null);
  const [enhancing, setEnhancing] = useState<string | null>(null);
  const [retryErrors, setRetryErrors] = useState<Record<string, string>>({});

  const retry = async (entry: HistoryEntry) => {
    if (retrying !== null) return;
    setRetrying(entry.job_id);
    setRetryErrors((previous) => {
      const next = { ...previous };
      delete next[entry.job_id];
      return next;
    });
    const error = await onRetry(entry.job_id);
    if (error !== null) {
      setRetryErrors((previous) => ({
        ...previous,
        [entry.job_id]: `${t.retryFailed}: ${error}`,
      }));
      setRetrying(null);
    }
  };

  const enhance = async (entry: HistoryEntry) => {
    if (retrying !== null || enhancing !== null) return;
    setEnhancing(entry.job_id);
    setRetryErrors((previous) => {
      const next = { ...previous };
      delete next[entry.job_id];
      return next;
    });
    const error = await onEnhance(entry.job_id);
    if (error !== null) {
      setRetryErrors((previous) => ({
        ...previous,
        [entry.job_id]: `${t.llmEnhanceFailed}: ${error}`,
      }));
      setEnhancing(null);
    }
  };

  const requestDelete = async (entry: HistoryEntry) => {
    const jobId = entry.job_id;
    const row = rowRefs.current.get(jobId);
    const listOptions = row
      ?.closest('[role="listbox"]')
      ?.querySelectorAll<HTMLElement>('[role="option"]');
    const options =
      listOptions === undefined
        ? (entries ?? []).flatMap((candidate) => {
            const element = rowRefs.current.get(candidate.job_id);
            return element === undefined ? [] : [element];
          })
        : Array.from(listOptions);
    const rowIndex = row === undefined ? -1 : options.indexOf(row);
    const focusTarget =
      rowIndex < 0 ? null : (options[rowIndex + 1] ?? options[rowIndex - 1] ?? null);
    const deleted = await onDelete(jobId);
    if (!deleted) return false;

    announce(t.histDeleted(entry.names_preview[0] ?? jobId));
    window.requestAnimationFrame(() => {
      if (focusTarget?.isConnected) focusTarget.focus();
    });
    return true;
  };

  if (error !== null) {
    return (
      <div
        className="lrow archive-state errline"
        role="option"
        aria-selected={false}
        // Part of the ledger's roving listbox so arrow navigation can reach
        // the retry affordance instead of skipping over this row.
        data-ledger-key="archive:error"
        tabIndex={-1}
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
        const displayNames = entry.names_preview.map((name) =>
          name.replace(/^https?:\/\//, ""),
        );
        const names =
          displayNames.join(", ") + (more > 0 ? ` ${t.histMore(more)}` : "");
        const firstName = entry.names_preview[0] ?? entry.job_id;
        const firstKind = entry.kinds_preview[0] ?? "file";
        const busy = actions[entry.job_id];
        // The wand turns a base result into an LLM one; a job whose only
        // item is already enhanced has nothing left to offer.
        const enhanceable =
          entry.total === 1 &&
          entry.done === 1 &&
          entry.failed === 0 &&
          entry.skipped === 0 &&
          entry.llm_enhanced === 0;
        const hasLlm =
          entry.llm_enhanced > 0 ||
          (entry.cost_usd !== null && entry.cost_usd > 0);
        const llmLabel =
          !hasLlm
            ? "Base"
            : entry.llm_enhanced === 0 || entry.llm_enhanced === entry.total
              ? "LLM"
              : `LLM ${entry.llm_enhanced}/${entry.total}`;
        const duration =
          entry.duration_ms === null ? "-" : fmtDur(entry.duration_ms);
        const finished = fmtDateTime(entry.finished_at);
        const singleResult =
          entry.total === 1
            ? entry.failed > 0
              ? "error"
              : entry.skipped > 0
                ? "skip"
                : "ok"
            : null;
        const statusClass =
          entry.failed > 0
            ? "chip err history-result"
            : entry.skipped === entry.total
              ? "chip skip history-result"
              : "chip ok history-result";
        const activate = (opener: HTMLElement) => {
          if (busy === undefined) void onOpen(entry.job_id, opener);
        };
        return (
          <div
            className="lrow actionable"
            role="option"
            aria-selected={false}
            aria-disabled={busy !== undefined ? true : undefined}
            data-ledger-key={`archive:${entry.job_id}`}
            tabIndex={
              tabbableJobId === undefined
                ? startIndex === 0 && index === 0
                  ? 0
                  : -1
                : tabbableJobId === entry.job_id
                  ? 0
                  : -1
            }
            aria-label={`${t.histOpen} ${firstName}`}
            key={entry.job_id}
            ref={(element) => {
              if (element === null) rowRefs.current.delete(entry.job_id);
              else rowRefs.current.set(entry.job_id, element);
            }}
            onClick={(event) => {
              event.currentTarget.focus({ preventScroll: true });
              activate(event.currentTarget);
            }}
            onFocus={() => onRowFocus?.(entry.job_id)}
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
              {firstKind === "url" ? <GlobeIcon /> : <FileTextIcon />}
              <span className="fname">{names}</span>
              {entry.origin === "cli" && (
                <span className="origin-tag">{t.originCli}</span>
              )}
            </span>
            <span className="c-duration">{duration}</span>
            <span className="c-finished" title={entry.finished_at ?? undefined}>
              {finished}
            </span>
            {showCost && (
              <span className="c-cost llm-cell">
                <span
                  className={hasLlm ? "llm-tag on" : "llm-tag"}
                >
                  {llmLabel}
                </span>
                {hasLlm && entry.cost_usd !== null && (
                  <span className="llm-price">{fmtCost(entry.cost_usd)}</span>
                )}
              </span>
            )}
            <span className="c-status archive-actions">
              {singleResult === null ? (
                <span className={statusClass}>{statsLine(entry, t)}</span>
              ) : (
                <span
                  className={`item-result ${singleResult}${singleResult === "skip" ? " tooltip" : ""}`}
                  title={
                    singleResult === "error"
                      ? t.statFailed
                      : singleResult === "skip"
                        ? t.statSkipped
                        : t.done
                  }
                  data-tooltip={singleResult === "skip" ? t.statSkipped : undefined}
                  aria-label={
                    singleResult === "error"
                      ? t.statFailed
                      : singleResult === "skip"
                        ? t.statSkipped
                        : t.done
                  }
                  tabIndex={singleResult === "skip" ? 0 : undefined}
                >
                  <span aria-hidden="true">
                    {singleResult === "error" ? (
                      "×"
                    ) : singleResult === "skip" ? (
                      <WarningIcon size={17} />
                    ) : (
                      "✓"
                    )}
                  </span>
                  <span className="sr-only">
                    {singleResult === "error"
                      ? t.statFailed
                      : singleResult === "skip"
                        ? t.statSkipped
                        : t.done}
                  </span>
                </span>
              )}
              {/* same .item-actions slot as session rows: mobile CSS gives it
                  a fixed width so the status mark keeps one x down the list
                  whether or not wand/retry render */}
              <span className="item-actions">
                {enhanceable && (
                  <button
                    type="button"
                    className={
                      retryErrors[entry.job_id] === undefined
                        ? "rowicon enhance"
                        : "rowicon enhance fail tooltip"
                    }
                    aria-label={t.enhanceWithLlm(firstName)}
                    title={
                      retryErrors[entry.job_id] ??
                      (llmAvailable
                        ? t.enhanceWithLlm(firstName)
                        : (llmDisabledReason ?? t.llmEnhanceUnavailable))
                    }
                    data-tooltip={retryErrors[entry.job_id]}
                    disabled={
                      !llmAvailable ||
                      busy !== undefined ||
                      retrying !== null ||
                      enhancing !== null
                    }
                    onClick={(event) => {
                      event.stopPropagation();
                      void enhance(entry);
                    }}
                  >
                    {enhancing === entry.job_id ? (
                      <span className="spin" aria-hidden="true" />
                    ) : (
                      <MagicWandIcon />
                    )}
                  </button>
                )}
                {entry.total === 1 && (entry.failed === 1 || entry.skipped === 1) && (
                  <button
                    type="button"
                    className="rowicon retry"
                    aria-label={t.retryAria(firstName)}
                    title={t.retryAria(firstName)}
                    disabled={
                      busy !== undefined || retrying !== null || enhancing !== null
                    }
                    onClick={(event) => {
                      event.stopPropagation();
                      void retry(entry);
                    }}
                  >
                    {retrying === entry.job_id ? (
                      <span className="spin" aria-hidden="true" />
                    ) : (
                      <RotateCcwIcon />
                    )}
                  </button>
                )}
                <ConfirmDeletePopover
                  triggerLabel={t.histDeleteAria(firstName)}
                  title={t.deleteItemTitle(firstName)}
                  description={t.deleteItemDescription}
                  confirmLabel={t.deletePermanently}
                  cancelLabel={t.cancel}
                  busyLabel={t.deleting}
                  disabled={
                    busy !== undefined || retrying !== null || enhancing !== null
                  }
                  onConfirm={() => requestDelete(entry)}
                />
              </span>
            </span>
            {/* separate spans (not one joined string): the mobile meta line
                wraps between facts, never mid-fact; the timestamp bit is the
                first thing the tightest phones drop */}
            <span className="rowmeta">
              <span className="metabit">{duration}</span>
              <span className="metabit metabit-time">{finished}</span>
              <span className="metabit">
                {llmLabel}
                {hasLlm && entry.cost_usd !== null && ` ${fmtCost(entry.cost_usd)}`}
              </span>
              <span className="metabit">{t.histStorageSize(fmtBytes(entry.size_bytes))}</span>
            </span>
            {(rowErrors[entry.job_id] !== undefined ||
              retryErrors[entry.job_id] !== undefined) && (
              <span className="c-err errline">
                {retryErrors[entry.job_id] ?? rowErrors[entry.job_id]}
              </span>
            )}
          </div>
        );
      })}
    </>
  );
}
