import { memo, useEffect, useRef, useState } from "react";
import { jobFileUrl } from "../api/client";
import type { ItemStatus } from "../api/types";
import type { SessionItem } from "../hooks/useJobs";
import type { Dict } from "../i18n";
import {
  durParts,
  fmtBytes,
  fmtCost,
  fmtDateTime,
  fmtDur,
  shortError,
} from "../lib/format";
import { ConfirmDeletePopover } from "./ConfirmDeletePopover";
import {
  DownloadIcon,
  FileTextIcon,
  GlobeIcon,
  MagicWandIcon,
  RotateCcwIcon,
  WarningIcon,
} from "./icons";

/** Status words are UI labels, so they come from the locale dictionary. */
const statusText = (t: Dict): Record<ItemStatus | "skipped", string> => ({
  queued: t.statusQueued,
  running: t.statusRunning,
  done: t.statusDone,
  error: t.statusFailed,
  skipped: t.statusSkipped,
});
const DASH = "-";
const NAME_TAIL_CHARS = 12;

/** Owns the 150ms clock for one running row, so a tick repaints this span
 * instead of the whole ledger. Non-running rows never mount it. */
const Elapsed = memo(function Elapsed({ startedAt }: { startedAt: number | null }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (startedAt === null) return;
    const id = window.setInterval(() => setNow(Date.now()), 150);
    return () => window.clearInterval(id);
  }, [startedAt]);
  if (startedAt === null) return <>-</>;
  return <>{fmtDur(Math.max(0, now - startedAt))}</>;
});

/** Preserve the filename tail while the middle of a long name ellipsizes. */
function MidName({ name, title }: { name: string; title: string }) {
  if (name.length <= NAME_TAIL_CHARS + 4) {
    return (
      <span className="fname" title={title}>
        {name}
      </span>
    );
  }
  return (
    <span className="fname mid" title={title}>
      <span className="fhead">{name.slice(0, -NAME_TAIL_CHARS)}</span>
      <span className="ftail">{name.slice(-NAME_TAIL_CHARS)}</span>
    </span>
  );
}

/** Terminal rows use compact colored marks rather than textual `<done>` pills. */
function StatusMark({ item, t }: { item: SessionItem; t: Dict }) {
  if (item.status === "done") {
    const skipped = item.skipped;
    const tooltip =
      skipped && item.skipReason === "image_only" ? t.skipImageOnly : t.statusSkipped;
    return (
      <span
        className={skipped ? "item-result skip tooltip" : "item-result ok"}
        title={skipped ? tooltip : t.statusDone}
        data-tooltip={skipped ? tooltip : undefined}
        aria-label={skipped ? tooltip : t.statusDone}
        tabIndex={skipped ? 0 : undefined}
      >
        <span aria-hidden="true">
          {skipped ? <WarningIcon size={17} /> : "✓"}
        </span>
        <span className="sr-only">{skipped ? tooltip : t.statusDone}</span>
      </span>
    );
  }
  if (item.status === "error") {
    const detail = item.error ?? t.statusFailed;
    return (
      <span
        className="item-result error tooltip"
        title={detail}
        data-tooltip={detail}
        aria-label={`${t.statusFailed}: ${detail}`}
        tabIndex={0}
      >
        <span aria-hidden="true">×</span>
        <span className="sr-only">{t.statusFailed}</span>
      </span>
    );
  }
  if (item.status === "running") {
    return (
      <span className="runstat" title={t.statusRunning}>
        <span className="spin" aria-hidden="true" />
        <span className="sr-only">{t.statusRunning}</span>
      </span>
    );
  }
  return <span className="runstat queued">{t.statusQueued}</span>;
}

export const ItemRow = memo(function ItemRow({
  t,
  item,
  index,
  showCost,
  selected,
  tabbable,
  canDelete,
  onPreview,
  onRowFocus,
  onRetry,
  onEnhance = async () => null,
  onDelete,
  llmAvailable = false,
  llmDisabledReason,
}: {
  t: Dict;
  item: SessionItem;
  index: number;
  showCost: boolean;
  selected: boolean;
  tabbable: boolean;
  canDelete: boolean;
  onPreview: (key: string, opener: HTMLElement) => void;
  onRowFocus: (key: string) => void;
  onRetry: (item: SessionItem) => Promise<string | null>;
  onEnhance?: (item: SessionItem) => Promise<string | null>;
  onDelete: (item: SessionItem) => Promise<string | null>;
  llmAvailable?: boolean;
  llmDisabledReason?: string;
}) {
  const rowRef = useRef<HTMLDivElement>(null);
  const [errExpanded, setErrExpanded] = useState(false);
  const [retryBusy, setRetryBusy] = useState(false);
  const [enhanceBusy, setEnhanceBusy] = useState(false);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [actionErr, setActionErr] = useState<string | null>(null);
  const [enhanceErr, setEnhanceErr] = useState<string | null>(null);

  const running = item.status === "running";
  const failed = item.status === "error";
  const skipped = item.status === "done" && item.skipped;
  // Previewable = done with a fresh output; skips complete as "done" but carry
  // no fresh result. `outputPath` holds the narrowed path so the download link
  // does not repeat the null check.
  const outputPath =
    item.status === "done" && item.output !== null && !item.skipped
      ? item.output
      : null;
  const previewable = outputPath !== null;
  const llmApplied =
    item.llmEnhanced ||
    item.operation === "enhance" ||
    (item.costUsd !== null && item.costUsd > 0);
  const enhanceable = previewable && canDelete;

  const doRetry = async () => {
    if (retryBusy || enhanceBusy || deleteBusy) return;
    setRetryBusy(true);
    setActionErr(null);
    setEnhanceErr(null);
    const error = await onRetry(item);
    if (error !== null) setActionErr(`${t.retryFailed}: ${error}`);
    setRetryBusy(false);
  };

  const doEnhance = async () => {
    if (retryBusy || enhanceBusy || deleteBusy || !llmAvailable) return;
    setEnhanceBusy(true);
    setActionErr(null);
    setEnhanceErr(null);
    const error = await onEnhance(item);
    if (error !== null) {
      const detail = `${t.llmEnhanceFailed}: ${error}`;
      setActionErr(detail);
      setEnhanceErr(detail);
    }
    setEnhanceBusy(false);
  };

  const doDelete = async () => {
    if (retryBusy || enhanceBusy || deleteBusy) return false;
    setDeleteBusy(true);
    setActionErr(null);
    setEnhanceErr(null);
    // Pick the focus successor before the row (and the confirm popover's
    // trigger) unmounts, so deleting does not drop focus to <body> — the
    // same handoff ArchivedJobRows performs.
    const row = rowRef.current;
    const options = Array.from(
      row
        ?.closest('[role="listbox"]')
        ?.querySelectorAll<HTMLElement>('[role="option"]') ?? [],
    );
    const rowIndex = row === null ? -1 : options.indexOf(row);
    const focusTarget =
      rowIndex < 0 ? null : (options[rowIndex + 1] ?? options[rowIndex - 1] ?? null);
    const error = await onDelete(item);
    if (error !== null) {
      setActionErr(error);
      setDeleteBusy(false);
      return false;
    }
    window.requestAnimationFrame(() => {
      if (focusTarget?.isConnected) focusTarget.focus();
    });
    return true;
  };

  const timeText = skipped
    ? DASH
    : item.durationMs !== null
      ? fmtDur(item.durationMs)
      : null;
  const finishedText = fmtDateTime(item.finishedAt);
  const sizeText = item.sizeBytes !== null ? fmtBytes(item.sizeBytes) : null;
  const costText = item.costUsd !== null ? fmtCost(item.costUsd) : null;
  const displayName = item.name.replace(/^https?:\/\//, "");
  const nameTitle = sizeText !== null ? `${item.name} · ${sizeText}` : item.name;

  // Bits stay separate spans (not one joined string) so the mobile meta line
  // wraps between facts — never mid-fact, never stranding a separator. The
  // timestamp is tagged: it is the first bit the tightest phones drop.
  const metaParts: { text: string; time?: boolean }[] = [];
  if (sizeText !== null) metaParts.push({ text: sizeText });
  if (running) metaParts.push({ text: t.statusRunning });
  else if (!skipped && item.durationMs !== null)
    metaParts.push({ text: fmtDur(item.durationMs) });
  if (item.finishedAt !== null) metaParts.push({ text: finishedText, time: true });
  if (showCost) {
    metaParts.push({
      text: llmApplied ? `${t.llmTag}${costText === null ? "" : ` ${costText}`}` : t.baseTag,
    });
  }
  if (item.status === "queued") metaParts.push({ text: t.statusQueued });

  // A skipped row still hosts an enabled Retry (rendered on failed || skipped),
  // so it must not announce itself disabled to assistive tech.
  const inert = !previewable && !failed && !skipped && !canDelete;
  // Known reasons read as sentences; an unknown one keeps the generic label
  // and carries the raw wire value in the title/aria description instead.
  const skipKnown =
    item.skipReason === "image_only"
      ? t.skipImageOnly
      : item.skipReason === "exists"
        ? t.skipExists
        : null;
  const statusTexts = statusText(t);
  const skipText = skipKnown ?? t.statusSkipped;
  const skipTitle = skipKnown === null ? (item.skipReason ?? undefined) : undefined;
  const detailId =
    failed || (skipped && item.skipReason !== "image_only")
      ? `d-${item.key.replace(/[^a-zA-Z0-9_-]/g, "-")}`
      : undefined;

  const ariaParts = [displayName];
  if (sizeText !== null) ariaParts.push(sizeText);
  if (!skipped && item.durationMs !== null) {
    // Spoken form: a long job must not read "312.4 Seconds", and fmtDur's
    // clock form ("5:12") would be read out as a time of day.
    const { minutes, seconds } = durParts(item.durationMs);
    ariaParts.push(t.ariaDuration(minutes, seconds));
  }
  if (showCost) {
    ariaParts.push(
      llmApplied ? `${t.llmTag}${costText === null ? "" : ` ${costText}`}` : t.baseTag,
    );
  }
  ariaParts.push(skipped ? t.statusSkipped : statusTexts[item.status]);

  const activate = (opener: HTMLElement) => {
    if (previewable) onPreview(item.key, opener);
    else if (failed) setErrExpanded((value) => !value);
  };

  return (
    <div
      ref={rowRef}
      role="option"
      id={`opt-${item.key.replace(/[^a-zA-Z0-9_-]/g, "-")}`}
      data-session-key={item.key}
      data-ledger-key={item.key}
      aria-selected={selected}
      aria-disabled={inert ? true : undefined}
      aria-label={ariaParts.join(", ")}
      aria-describedby={detailId}
      tabIndex={tabbable ? 0 : -1}
      className={`lrow${selected ? " sel" : ""}${failed ? " actionable" : ""}`}
      title={failed ? (item.error ?? undefined) : undefined}
      onClick={(event) => {
        // Safari does not consistently focus generic tabindex elements on a
        // pointer click; make the listbox's roving focus browser-independent.
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
      <span className="c-num">{String(index + 1).padStart(2, "0")}</span>
      <span className="c-name">
        {item.kind === "file" ? <FileTextIcon /> : <GlobeIcon />}
        <MidName name={displayName} title={nameTitle} />
      </span>
      <span className={running ? "c-duration live" : "c-duration"}>
        {running ? <Elapsed startedAt={item.startedAt} /> : (timeText ?? DASH)}
      </span>
      <span className="c-finished" title={item.finishedAt ?? undefined}>
        {finishedText}
      </span>
      {showCost && (
        <span className="c-cost llm-cell">
          <span className={llmApplied ? "llm-tag on" : "llm-tag"}>
            {llmApplied ? t.llmTag : t.baseTag}
          </span>
          {llmApplied && costText !== null && (
            <span className="llm-price">{costText}</span>
          )}
        </span>
      )}
      <span className="c-status archive-actions">
        <StatusMark item={item} t={t} />
        {(previewable || enhanceable || failed || skipped || canDelete) && (
          <span className="item-actions">
            {outputPath !== null && (
              // The row's primary output is one click away; the preview modal
              // stays the way to read the file, not the way to fetch it.
              <a
                className="rowicon download"
                href={jobFileUrl(item.jobId, outputPath)}
                download
                aria-label={`${t.downloadMd}: ${displayName}`}
                title={`${t.downloadMd}: ${displayName}`}
                onClick={(event) => event.stopPropagation()}
              >
                <DownloadIcon size={15} />
              </a>
            )}
            {enhanceable && (
              <button
                type="button"
                className={
                  enhanceErr === null
                    ? "rowicon enhance"
                    : "rowicon enhance fail tooltip"
                }
                aria-label={t.enhanceWithLlm(displayName)}
                title={
                  enhanceErr ??
                  (llmAvailable
                    ? t.enhanceWithLlm(displayName)
                    : (llmDisabledReason ?? t.llmEnhanceUnavailable))
                }
                data-tooltip={enhanceErr ?? undefined}
                disabled={
                  !llmAvailable || retryBusy || enhanceBusy || deleteBusy
                }
                aria-busy={enhanceBusy || undefined}
                onClick={(event) => {
                  event.stopPropagation();
                  void doEnhance();
                }}
              >
                {enhanceBusy ? (
                  <span className="spin" aria-hidden="true" />
                ) : (
                  <MagicWandIcon />
                )}
              </button>
            )}
            {(failed || skipped) && (
              <button
                type="button"
                className="rowicon retry"
                aria-label={t.retryAria(displayName)}
                title={t.retryAria(displayName)}
                disabled={retryBusy || enhanceBusy || deleteBusy}
                aria-busy={retryBusy || undefined}
                onClick={(event) => {
                  event.stopPropagation();
                  void doRetry();
                }}
              >
                {retryBusy ? <span className="spin" aria-hidden="true" /> : <RotateCcwIcon />}
              </button>
            )}
            {canDelete && (
              <ConfirmDeletePopover
                triggerLabel={t.histDeleteAria(displayName)}
                title={t.deleteItemTitle(displayName)}
                description={t.deleteItemDescription}
                confirmLabel={t.deletePermanently}
                cancelLabel={t.cancel}
                busyLabel={t.deleting}
                disabled={retryBusy || enhanceBusy || deleteBusy}
                onConfirm={doDelete}
              />
            )}
          </span>
        )}
      </span>
      {metaParts.length > 0 && (
        <span className="rowmeta">
          {metaParts.map((bit) => (
            <span
              key={bit.text}
              className={bit.time === true ? "metabit metabit-time" : "metabit"}
            >
              {bit.text}
            </span>
          ))}
        </span>
      )}
      {failed && item.error !== null && (
        <span className="c-err" id={detailId}>
          <span className="errtext" title={t.errExpandTitle}>
            {errExpanded ? (
              <span className="err-full">{item.error}</span>
            ) : (
              shortError(item.error)
            )}
          </span>
        </span>
      )}
      {actionErr !== null && (
        <span className="c-err retryerr" role="alert">
          {actionErr}
        </span>
      )}
      {skipped && item.skipReason !== "image_only" && (
        <span className="c-skip" id={detailId} title={skipTitle}>
          {skipText}
        </span>
      )}
    </div>
  );
});
