// Small pieces every item view shares: the status glyph (one shape per state,
// so colour is never the only signal) and the live duration. Names are in
// file-name.tsx.
import { useEffect, useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { isFailed, isStopped } from "../lib/conversions.ts";
import { durParts, fmtDur } from "../lib/format.ts";
import type { SessionItem } from "../lib/session.ts";
import { Icon } from "./icons.tsx";

export type Glyph = "done" | "failed" | "stopped" | "skipped" | "running" | "queued" | "retained";

export function glyphOf(item: SessionItem): Glyph {
  if (item.status === "running") return "running";
  if (item.status === "queued") return "queued";
  if (isStopped(item)) return "stopped";
  if (isFailed(item)) return "failed";
  if (item.skipped) return "skipped";
  if (item.rerunFailure) return "retained";
  return "done";
}

export function statusWord(t: Dict, glyph: Glyph): string {
  return {
    done: t.statusDone,
    failed: t.statusFailed,
    stopped: t.statusStopped,
    skipped: t.statusSkipped,
    running: t.statusRunning,
    queued: t.statusQueued,
    retained: t.statusRetained,
  }[glyph];
}

/** The glyph; `label` names it for assistive technology when the row does not say it in words. */
export function StatusGlyph({ glyph, label }: { glyph: Glyph; label?: string }) {
  const icon = {
    done: <Icon name="Check" size={14} />,
    retained: <Icon name="Check" size={14} />,
    failed: <Icon name="X" size={14} />,
    stopped: <Icon name="MinusCircle" size={15} />,
    skipped: <Icon name="Warning" size={15} />,
    running: <Icon name="CircleNotch" size={14} />,
    queued: <span class="glyph-ring" />,
  }[glyph];
  return (
    <span class={`glyph is-${glyph}`} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : "true"}>
      {icon}
    </span>
  );
}

/** A running item's elapsed time, ticking once a second. */
export function useElapsed(startedAt: number | null): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (startedAt === null) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [startedAt]);
  return startedAt === null ? null : Math.max(0, now - startedAt);
}

/** "14.6 s" with a spoken form that says the unit (a label on a plain span is not read). */
export function Duration({ t, ms }: { t: Dict; ms: number }) {
  const { minutes, seconds } = durParts(ms);
  return (
    <span>
      <span aria-hidden="true">{fmtDur(ms)}</span>
      <span class="sr-only">{t.ariaDuration(minutes, seconds)}</span>
    </span>
  );
}
