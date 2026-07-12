import { useState } from "react";
import type { ItemStatus } from "../api/types";
import type { SessionItem } from "../hooks/useJobs";
import type { Dict } from "../i18n";
import { fmtBytes, fmtCost, fmtDur, shortError } from "../lib/format";
import { FileTextIcon, GlobeIcon, RotateCcwIcon } from "./icons";

/** CLI status words — lowercase mono, identical in both locales. */
const STATUS_TEXT: Record<ItemStatus | "skipped", string> = {
  queued: "queued",
  running: "converting",
  done: "done",
  error: "failed",
  skipped: "skipped",
};

/** Empty-cell placeholder — a plain hyphen (visible copy bans em/en dashes). */
const DASH = "-";
/** Kept tail of middle-truncated names — preserves “….txt” endings. */
const NAME_TAIL_CHARS = 12;

/** CLI word for history-merged rows — stays English in both locales. */
const ARCHIVED_TEXT = "archived";
/** CLI word for rows re-run as a new job — the ledger keeps the original. */
const RETRIED_TEXT = "retried";

function liveDuration(item: SessionItem, now: number): string | null {
  if (item.startedAt === null) return null;
  return fmtDur(Math.max(0, now - item.startedAt));
}

/** Middle truncation: head span ellipsizes, tail (extension) stays visible.
 * CSS alone can only end-truncate, so the split happens here. */
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

function StatusCell({ item }: { item: SessionItem }) {
  switch (item.status) {
    case "done":
      // Skips complete as "done" but carry no fresh result — neutral chip.
      return item.skipped ? (
        <span className="chip skip">{STATUS_TEXT.skipped}</span>
      ) : (
        <span className="chip ok">{STATUS_TEXT.done}</span>
      );
    case "error":
      return <span className="chip err">{STATUS_TEXT.error}</span>;
    case "running":
      // The compact column is too narrow for the word — spinner only.
      return (
        <span className="runstat" title={STATUS_TEXT.running}>
          <span className="spin" aria-hidden="true" />
          <span className="sr-only">{STATUS_TEXT.running}</span>
        </span>
      );
    case "queued":
      return <span className="runstat queued">{STATUS_TEXT.queued}</span>;
  }
}

export function ItemRow({
  t,
  item,
  index,
  showCost,
  now,
  selected,
  tabbable,
  llmConfigured,
  onSelect,
  onOpenSettings,
  onRowFocus,
  onRetry,
}: {
  t: Dict;
  item: SessionItem;
  index: number;
  showCost: boolean;
  now: number;
  selected: boolean;
  /** Roving tabindex: exactly one row in the listbox is tabbable. */
  tabbable: boolean;
  llmConfigured: boolean;
  onSelect: (key: string) => void;
  onOpenSettings: () => void;
  onRowFocus: (key: string) => void;
  onRetry: (item: SessionItem) => Promise<string | null>;
}) {
  const [errExpanded, setErrExpanded] = useState(false);
  const [retryBusy, setRetryBusy] = useState(false);
  const [retryErr, setRetryErr] = useState<string | null>(null);

  const running = item.status === "running";
  const failed = item.status === "error";
  const skipped = item.status === "done" && item.skipped;
  const previewable = item.status === "done" && item.output !== null && !item.skipped;
  // Failed rows (archived ones too) can be re-run; once retried the button
  // yields to the neutral marker (the new row carries its own retry).
  const canRetry = failed && !item.retried;
  const doRetry = async () => {
    if (retryBusy) return;
    setRetryBusy(true);
    setRetryErr(null);
    const err = await onRetry(item);
    setRetryBusy(false);
    if (err !== null) setRetryErr(err);
  };

  const live = running ? liveDuration(item, now) : null;
  const timeText = skipped
    ? DASH
    : item.durationMs !== null
      ? fmtDur(item.durationMs)
      : (live ?? DASH);
  const sizeText = item.sizeBytes !== null ? fmtBytes(item.sizeBytes) : null;
  const costText =
    item.costUsd !== null && item.costUsd > 0 ? fmtCost(item.costUsd) : DASH;
  // The mock strips the protocol from URLs in the compact list.
  const displayName = item.name.replace(/^https?:\/\//, "");
  const nameTitle = sizeText !== null ? `${item.name} · ${sizeText}` : item.name;

  // Size lost its column — it lives in the name title and the meta line.
  const metaParts: string[] = [];
  if (sizeText !== null) metaParts.push(sizeText);
  if (running) metaParts.push(`running ${live ?? DASH}`);
  else if (!skipped && item.durationMs !== null) metaParts.push(fmtDur(item.durationMs));
  if (showCost && item.costUsd !== null && item.costUsd > 0)
    metaParts.push(fmtCost(item.costUsd));
  if (item.status === "queued") metaParts.push(STATUS_TEXT.queued);

  // "image input — configure llm …" points at the settings panel while llm
  // is unconfigured; the row click opens it.
  const skipNeedsConfig = skipped && item.skipReason === "image_only" && !llmConfigured;
  // aria-disabled marks rows with no interaction at all (queued/running/plain
  // skips). Failed rows toggle their error text and host the enabled retry
  // button, and unconfigured image-skips open settings - announcing those as
  // disabled would contradict their working controls. Selection-follows-focus
  // is gated on the item data in ItemList, not on this attribute.
  const inert = !previewable && !failed && !skipNeedsConfig;
  const skipText =
    item.skipReason === "image_only"
      ? t.skipImageOnly
      : item.skipReason === "exists"
        ? t.skipExists
        : item.skipReason === null
          ? STATUS_TEXT.skipped
          : `${STATUS_TEXT.skipped} (${item.skipReason})`;

  const detailId =
    failed || skipped ? `d-${item.key.replace(/[^a-zA-Z0-9_-]/g, "-")}` : undefined;

  // Column context for screen readers ("sample.txt, 12 KB, 0.3 seconds, done").
  const ariaParts = [displayName];
  if (item.archived) ariaParts.push(ARCHIVED_TEXT);
  if (item.retried) ariaParts.push(RETRIED_TEXT);
  if (sizeText !== null) ariaParts.push(sizeText);
  if (!skipped && item.durationMs !== null)
    ariaParts.push(`${(item.durationMs / 1000).toFixed(1)} ${t.ariaSeconds}`);
  ariaParts.push(skipped ? STATUS_TEXT.skipped : STATUS_TEXT[item.status]);

  const activate = () => {
    if (previewable) onSelect(item.key);
    else if (failed) setErrExpanded((v) => !v);
    else if (skipNeedsConfig) onOpenSettings();
  };

  return (
    <div
      role="option"
      id={`opt-${item.key.replace(/[^a-zA-Z0-9_-]/g, "-")}`}
      aria-selected={selected}
      aria-disabled={inert ? true : undefined}
      aria-label={ariaParts.join(", ")}
      aria-describedby={detailId}
      tabIndex={tabbable ? 0 : -1}
      className={`lrow${selected ? " sel" : ""}${failed || skipNeedsConfig ? " actionable" : ""}`}
      title={failed ? (item.error ?? undefined) : undefined}
      onClick={activate}
      onFocus={() => onRowFocus(item.key)}
      onKeyDown={(e) => {
        // Keys on nested controls (the retry button) act on the control only.
        if (e.target !== e.currentTarget) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activate();
        }
      }}
    >
      <span className="c-num">{String(index + 1).padStart(2, "0")}</span>
      <span className="c-name">
        {item.kind === "file" ? <FileTextIcon /> : <GlobeIcon />}
        <MidName name={displayName} title={nameTitle} />
        {item.archived && <span className="minibadge">{ARCHIVED_TEXT}</span>}
        {item.retried && <span className="minibadge">{RETRIED_TEXT}</span>}
      </span>
      <span className={running ? "c-time live" : "c-time"}>{timeText}</span>
      {showCost && <span className="c-cost">{costText}</span>}
      <span className="c-status">
        <StatusCell item={item} />
      </span>
      {metaParts.length > 0 && <span className="rowmeta">{metaParts.join(" · ")}</span>}
      {failed && (item.error !== null || canRetry) && (
        <span className="c-err" id={detailId}>
          <span className="errtext" title={t.errExpandTitle}>
            {item.error === null ? null : errExpanded ? (
              <span className="err-full">{item.error}</span>
            ) : (
              shortError(item.error)
            )}
          </span>
          {canRetry && (
            <button
              type="button"
              className="rowact retry"
              aria-label={t.retryAria(displayName)}
              title={t.retryAria(displayName)}
              disabled={retryBusy}
              aria-busy={retryBusy || undefined}
              onClick={(e) => {
                e.stopPropagation(); // the row click toggles the error text
                void doRetry();
              }}
            >
              {retryBusy ? (
                <span className="spin" aria-hidden="true" />
              ) : (
                <RotateCcwIcon size={13} />
              )}
            </button>
          )}
        </span>
      )}
      {failed && retryErr !== null && (
        <span className="c-err retryerr" role="alert">
          {t.retryFailed}: {retryErr}
        </span>
      )}
      {skipped && (
        <span className="c-skip" id={detailId}>
          {skipNeedsConfig ? (
            <>
              {t.skipCfgPre}
              <span className="linkish">{t.skipCfgLink}</span>
              {t.skipCfgPost}
            </>
          ) : (
            skipText
          )}
        </span>
      )}
    </div>
  );
}
