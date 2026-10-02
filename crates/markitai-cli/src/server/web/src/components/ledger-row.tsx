// One ledger row of the session. Memoized: an `item` event re-renders only the
// row whose item object changed.
import { memo } from "preact/compat";
import { useEffect, useRef, useState } from "preact/hooks";
import { filePath } from "../api/client.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { itemErrorText } from "../i18n/errors.ts";
import { interceptDownload } from "../lib/download.ts";
import { basename, displayName, durParts, fmtBytes, fmtCost, fmtDateTime, fmtDur, splitName } from "../lib/format.ts";
import { attemptNotice, PRICE_WORDS, priceText } from "../lib/pricing.ts";
import { canRetry, isPreviewable, type SessionItem } from "../lib/session.ts";
import { ConfirmPopover } from "./confirm-popover.tsx";
import { Icon } from "./icons.tsx";

/** The clock of one running row; a tick repaints this span only. */
function Elapsed({ since }: { since: number | null }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (since === null) return;
    const timer = setInterval(() => setNow(Date.now()), 150);
    return () => clearInterval(timer);
  }, [since]);
  return <>{since === null ? "-" : fmtDur(Math.max(0, now - since))}</>;
}

export function FileName({ name, title }: { name: string; title: string }) {
  const parts = splitName(name);
  if (parts === null) {
    return (
      <span class="fname" title={title}>
        {name}
      </span>
    );
  }
  return (
    <span class="fname is-split" title={title}>
      <span class="fname-head">{parts[0]}</span>
      <span class="fname-tail">{parts[1]}</span>
    </span>
  );
}

export const domKey = (key: string): string => `row-${key.replace(/[^a-zA-Z0-9_-]/g, "-")}`;

function StatusMark({ t, item }: { t: Dict; item: SessionItem }) {
  if (item.status === "done") {
    if (!item.skipped) {
      return (
        <span class="mark is-ok" title={t.statusDone}>
          <span aria-hidden="true">✓</span>
          <span class="sr-only">{t.statusDone}</span>
        </span>
      );
    }
    const why =
      item.skipReason === "image_only" ? t.skipImageOnly : item.skipReason === "pending_batch" ? t.skipPendingBatch : t.statusSkipped;
    return (
      <span class="mark is-skip has-tip" data-tip={why} title={why} tabIndex={0} aria-label={why}>
        <span aria-hidden="true">
          <Icon name="WarningFill" size={17} />
        </span>
      </span>
    );
  }
  if (item.status === "error") {
    return (
      <span class="mark is-err" title={item.error ?? t.statusFailed} aria-label={t.statusFailed}>
        <span aria-hidden="true">×</span>
        <span class="sr-only">{t.statusFailed}</span>
      </span>
    );
  }
  if (item.status === "running") {
    return (
      <span class="run-state" title={t.statusRunning}>
        <span class="spinner" aria-hidden="true" />
        <span class="sr-only">{t.statusRunning}</span>
      </span>
    );
  }
  return <span class="run-state is-queued">{t.statusQueued}</span>;
}

export interface RowProps {
  t: Dict;
  locale: Locale;
  item: SessionItem;
  index: number;
  selected: boolean;
  tabbable: boolean;
  canDelete: boolean;
  llmAvailable: boolean;
  llmDisabledReason: string;
  onPreview: (key: string, opener: HTMLElement) => void;
  onRowFocus: (key: string) => void;
  onRetry: (item: SessionItem) => Promise<unknown>;
  onEnhance: (item: SessionItem) => Promise<unknown>;
  onDelete: (item: SessionItem) => Promise<unknown>;
  describe: (error: unknown) => { text: string; detail: string };
  onDownloadError: (error: unknown) => void;
}

export const LedgerRow = memo(function LedgerRow({
  t,
  locale,
  item,
  index,
  selected,
  tabbable,
  canDelete,
  llmAvailable,
  llmDisabledReason,
  onPreview,
  onRowFocus,
  onRetry,
  onEnhance,
  onDelete,
  describe,
  onDownloadError,
}: RowProps) {
  const row = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState<"retry" | "enhance" | "delete" | null>(null);
  const [actionError, setActionError] = useState<{ text: string; detail: string } | null>(null);
  const [enhanceFailed, setEnhanceFailed] = useState(false);

  const running = item.status === "running";
  const failed = item.status === "error";
  const skipped = item.status === "done" && item.skipped;
  const previewable = isPreviewable(item);
  const llmApplied = item.llmEnhanced || item.operation === "enhance" || (item.costUsd !== null && item.costUsd > 0);
  const enhanceable = previewable && canDelete && item.retryable;
  const retryable = canRetry(item);
  const name = displayName(item.name);
  const size = item.sizeBytes !== null ? fmtBytes(item.sizeBytes) : null;
  const words = PRICE_WORDS[locale];
  const costTitle = priceText(item.costUsd, item.pricing, words) || undefined;
  const cost = item.costUsd !== null ? fmtCost(item.costUsd) : null;
  const problem = failed && item.error !== null ? itemErrorText(locale, { error: item.error, error_code: item.errorCode, kind: item.kind }) : null;
  const attempt = attemptNotice(item.diagnostics ? { cost_usd: item.costUsd, error: item.error, diagnostics: item.diagnostics } : {}, words, (error) =>
    itemErrorText(locale, { error, kind: item.kind }),
  );
  const warnings = item.status === "done" || item.status === "error" ? item.warnings : [];
  // An image skipped for lack of text says so in a notification instead.
  const skipText =
    item.skipReason === "image_only"
      ? null
      : item.skipReason === "exists"
        ? t.skipExists
        : item.skipReason === "pending_batch"
          ? t.skipPendingBatch
          : t.statusSkipped;
  const rowId = domKey(item.key);

  const run = async (kind: "retry" | "enhance") => {
    if (busy) return;
    setBusy(kind);
    setActionError(null);
    setEnhanceFailed(false);
    const error = await (kind === "retry" ? onRetry(item) : onEnhance(item));
    if (error !== null && error !== undefined) {
      const text = describe(error);
      setActionError({ text: `${kind === "retry" ? t.retryFailed : t.llmEnhanceFailed}: ${text.text}`, detail: text.detail });
      if (kind === "enhance") setEnhanceFailed(true);
    }
    setBusy(null);
  };

  const remove = async () => {
    if (busy) return false;
    setBusy("delete");
    setActionError(null);
    // Pick the next focus target before this row unmounts.
    const options = [...(row.current?.closest('[role="listbox"]')?.querySelectorAll<HTMLElement>('[role="option"]') ?? [])];
    const at = row.current ? options.indexOf(row.current) : -1;
    const next = at < 0 ? null : (options[at + 1] ?? options[at - 1] ?? null);
    const error = await onDelete(item);
    setBusy(null);
    if (error !== null && error !== undefined) {
      setActionError(describe(error));
      return false;
    }
    requestAnimationFrame(() => next?.isConnected && next.focus());
    return true;
  };

  const activate = (opener: HTMLElement) => {
    if (previewable) onPreview(item.key, opener);
    else if (failed) setExpanded((value) => !value);
  };

  const facts: { text: string; time?: boolean }[] = [];
  if (size) facts.push({ text: size });
  if (running) facts.push({ text: t.statusRunning });
  else if (!skipped && item.durationMs !== null) facts.push({ text: fmtDur(item.durationMs) });
  if (item.finishedAt !== null) facts.push({ text: fmtDateTime(item.finishedAt), time: true });
  facts.push({ text: llmApplied ? `${t.llmTag}${cost === null ? "" : ` ${cost}`}` : t.baseTag });
  if (item.status === "queued") facts.push({ text: t.statusQueued });

  const spoken = [name];
  if (size) spoken.push(size);
  if (!skipped && item.durationMs !== null) {
    const { minutes, seconds } = durParts(item.durationMs);
    spoken.push(t.ariaDuration(minutes, seconds));
  }
  spoken.push(llmApplied ? t.llmTag : t.baseTag);
  spoken.push(skipped ? t.statusSkipped : { queued: t.statusQueued, running: t.statusRunning, done: t.statusDone, error: t.statusFailed }[item.status]);
  if (warnings.length) spoken.push(t.itemWarnings(warnings.length));
  const inert = !previewable && !failed && !retryable && !canDelete;
  const output = previewable ? item.output : null;

  return (
    <div
      ref={row}
      role="option"
      id={rowId}
      data-session-key={item.key}
      data-ledger-key={item.key}
      aria-selected={selected}
      aria-disabled={inert || undefined}
      aria-label={spoken.join(", ")}
      aria-describedby={[problem || (skipped && skipText) ? `${rowId}-note` : null, warnings.length ? `${rowId}-warn` : null].filter(Boolean).join(" ") || undefined}
      tabIndex={tabbable ? 0 : -1}
      class={`lg-row${selected ? " is-selected" : ""}${failed ? " is-actionable" : ""}`}
      onClick={(event) => {
        event.currentTarget.focus({ preventScroll: true });
        activate(event.currentTarget);
      }}
      onFocus={() => onRowFocus(item.key)}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate(event.currentTarget);
        }
      }}
    >
      <span class="cell-num">{String(index + 1).padStart(2, "0")}</span>
      <span class="cell-name">
        <Icon name={item.kind === "file" ? "FileText" : "Globe"} size={14} />
        <FileName name={name} title={size ? `${item.name} · ${size}` : item.name} />
      </span>
      <span class={running ? "cell-time is-live" : "cell-time"}>
        {running ? <Elapsed since={item.startedAt} /> : skipped ? "-" : item.durationMs !== null ? fmtDur(item.durationMs) : "-"}
      </span>
      <span class="cell-done" title={item.finishedAt ?? undefined}>
        {fmtDateTime(item.finishedAt)}
      </span>
      <span class="cell-cost" title={costTitle}>
        <span class={llmApplied ? "tag is-llm" : "tag"}>{llmApplied ? t.llmTag : t.baseTag}</span>
        {llmApplied && cost !== null && <span class="tag-price">{cost}</span>}
      </span>
      <span class="cell-state">
        <StatusMark t={t} item={item} />
        {(previewable || enhanceable || retryable || canDelete) && (
          <span class="row-tools">
            {output !== null && (
              <a
                class="row-icon"
                href={filePath(item.jobId, output)}
                download={basename(output)}
                aria-label={`${t.downloadMd}: ${name}`}
                title={`${t.downloadMd}: ${name}`}
                onClick={(event) => interceptDownload(event, filePath(item.jobId, output), basename(output), onDownloadError)}
                onAuxClick={(event) => interceptDownload(event, filePath(item.jobId, output), basename(output), onDownloadError)}
              >
                <Icon name="DownloadSimple" size={15} />
              </a>
            )}
            {enhanceable && (
              <button
                type="button"
                class={enhanceFailed ? "row-icon is-failed" : "row-icon"}
                aria-label={t.enhanceWithLlm(name)}
                title={enhanceFailed && actionError ? actionError.text : llmAvailable ? t.enhanceWithLlm(name) : llmDisabledReason}
                disabled={!llmAvailable || busy !== null}
                aria-busy={busy === "enhance" || undefined}
                onClick={(event) => {
                  event.stopPropagation();
                  void run("enhance");
                }}
              >
                {busy === "enhance" ? <span class="spinner" aria-hidden="true" /> : <Icon name="MagicWand" size={14} />}
              </button>
            )}
            {retryable && (
              <button
                type="button"
                class="row-icon"
                aria-label={t.retryAria(name)}
                title={t.retryAria(name)}
                disabled={busy !== null}
                aria-busy={busy === "retry" || undefined}
                onClick={(event) => {
                  event.stopPropagation();
                  void run("retry");
                }}
              >
                {busy === "retry" ? <span class="spinner" aria-hidden="true" /> : <Icon name="ArrowCounterClockwise" size={13} />}
              </button>
            )}
            {canDelete && (
              <ConfirmPopover
                triggerLabel={t.histDeleteAria(name)}
                title={t.deleteItemTitle(name)}
                description={t.deleteItemDescription}
                confirmLabel={t.deletePermanently}
                cancelLabel={t.cancel}
                busyLabel={t.deleting}
                disabled={busy !== null}
                onConfirm={remove}
              />
            )}
          </span>
        )}
      </span>
      <span class="row-facts">
        {facts.map((fact) => (
          <span key={fact.text} class={fact.time ? "fact is-time" : "fact"}>
            {fact.text}
          </span>
        ))}
      </span>
      {problem && (
        <span class={problem.hint ? "row-note is-quiet" : "row-note is-err"} id={`${rowId}-note`} title={problem.detail || undefined}>
          <span class="row-note-text" title={problem.detail ? t.errExpandTitle : undefined}>
            {problem.text}
            {expanded && problem.detail && (
              <span class="row-note-full">{problem.detail}</span>
            )}
          </span>
        </span>
      )}
      {skipped && skipText && (
        <span class="row-note is-quiet" id={`${rowId}-note`} title={item.skipReason === "pending_batch" ? (item.error ?? undefined) : undefined}>
          {skipText}
        </span>
      )}
      {attempt && (attempt.label || attempt.error) && (
        <span class="row-note is-warn" title={attempt.error?.detail || undefined}>
          <Icon name="WarningFill" size={13} />
          <span class="row-note-line">{[attempt.label, attempt.error?.text].filter(Boolean).join(" · ")}</span>
        </span>
      )}
      {warnings.length > 0 && (
        <span class="row-note is-warn" id={`${rowId}-warn`} title={warnings.join("\n")}>
          <Icon name="WarningFill" size={13} />
          <span class="row-note-line">
            {t.itemWarnings(warnings.length)}: {warnings[0]}
          </span>
        </span>
      )}
      {actionError && (
        <span class="row-note is-err" role="alert" title={actionError.detail || undefined}>
          {actionError.text}
        </span>
      )}
    </div>
  );
});
