// What the update entry and its card show for an update's state (contract
// v1.24, `GET /api/update`): the stepper's position, the status row's text
// and value, a failure's words and what it left, and the entry icon's look.
// Pure, so the rules are tested without a page.
import type { Dict } from "../i18n/index.ts";
import type { UpdateError, UpdateState, UpdateStatus, UpdateSummary } from "../api/types.ts";

/** Download, verify, install, restart. */
export const STEP_COUNT = 4;

/** The states in which the server works on an update. */
export const BUSY: ReadonlySet<UpdateState> = new Set(["downloading", "verifying", "waiting", "installing", "restarting"]);
/** The states in which new conversions are held (503 `updating`). */
export const HOLDING: ReadonlySet<UpdateState> = new Set(["waiting", "installing", "restarting"]);
/** The states `DELETE /api/update` may still cancel. */
export const CANCELLABLE: ReadonlySet<UpdateState> = new Set(["downloading", "verifying", "waiting"]);

/** The step a state is on: waiting is the first part of installing. */
export function stepOf(state: UpdateState): number {
  switch (state) {
    case "downloading":
      return 0;
    case "verifying":
      return 1;
    case "waiting":
    case "installing":
      return 2;
    case "restarting":
      return 3;
    default:
      return -1;
  }
}

/** The step a failure stopped on; -1 when none is red (cancelled, another update). */
export function failedStep(error: UpdateError): number {
  switch (error.code) {
    case "offline":
    case "disk_full":
    case "not_writable":
      return 0;
    case "checksum":
      return 1;
    case "file_busy":
    case "install_failed":
      return 2;
    case "start_failed":
      // The dry start belongs to verifying; a rollback, to the restart.
      return error.rolled_back ? 3 : 1;
    default:
      return -1;
  }
}

/** Download progress as a fraction, when the size is known. */
export function fraction(status: Pick<UpdateStatus, "progress">): number | null {
  const progress = status.progress;
  if (!progress || !progress.total || progress.total <= 0) return null;
  return Math.max(0, Math.min(1, progress.done / progress.total));
}

const mb = (bytes: number) => (bytes / 1_000_000).toFixed(1);

/** `12.1 / 31.2 MB`, or `12.1 MB` without a size. */
export function downloaded(progress: UpdateStatus["progress"]): string {
  if (!progress) return "";
  return progress.total ? `${mb(progress.done)} / ${mb(progress.total)} MB` : `${mb(progress.done)} MB`;
}

/** `0:42`, `12:05`, `1:02:03`. */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const pad = (value: number) => String(value).padStart(2, "0");
  return s >= 3600 ? `${Math.floor(s / 3600)}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}` : `${Math.floor(s / 60)}:${pad(s % 60)}`;
}

/** The status row of a busy state: its text and its grey value. */
export function statusRow(t: Dict, status: UpdateStatus, waitedSeconds: number): { text: string; value: string } {
  switch (status.state) {
    case "downloading":
      return { text: t.updDownloading, value: downloaded(status.progress) };
    case "verifying":
      return { text: t.updVerifying, value: "SHA-256" };
    case "waiting":
      return { text: status.waiting_jobs > 0 ? t.updWaiting(status.waiting_jobs) : t.updWaitingNone, value: clock(waitedSeconds) };
    case "installing":
      return { text: t.updInstalling, value: "" };
    case "restarting":
      return { text: t.updRestarting, value: "" };
    default:
      return { text: "", value: "" };
  }
}

/** A part of the service's English reason: `Can't reach HOST`, `Can't write to DIR`, … */
const pick = (pattern: RegExp, reason: string): string | null => pattern.exec(reason)?.[1]?.trim() || null;

/** A failure in the interface language, and the line under it (what it left). */
export function failureText(t: Dict, error: UpdateError, current: string, target: string): { title: string; left: string } {
  const left = error.rolled_back ? (error.code === "start_failed" ? t.updBackOn(current) : t.updRolledBack(current)) : t.updNothingChanged;
  const reason = error.reason ?? "";
  const title = (() => {
    switch (error.code) {
      case "offline":
        return t.updErrOffline(pick(/Can't reach ([^\s:]+)/, reason));
      case "checksum":
        return t.updErrChecksum;
      case "disk_full":
        return t.updErrDisk(pick(/needs ([\d.]+ [KMG]B)/, reason));
      case "file_busy":
        return t.updErrBusy(pick(/Can't replace (.+?): /, reason));
      case "not_writable":
        return t.updErrReadonly(pick(/Can't write to (.+?): /, reason));
      case "start_failed":
        return t.updErrStart(error.version ?? target);
      case "update_running":
        return t.updErrRunning;
      case "cancelled":
        return t.updCancelled;
      default:
        return t.updErrInstall;
    }
  })();
  return { title, left };
}

/** How the entry icon looks: an arrow with a dot, a spinner while it works, red after a failure. */
export type EntryLook = "available" | "busy" | "failed";

/** The failure the card shows: the last update's, else one the last start undid (unless dismissed here). */
export function shownError(status: UpdateStatus | null, dismissed: string | null): UpdateError | null {
  if (!status) return null;
  const error = status.state === "failed" ? status.error : status.state === "idle" ? status.error : null;
  if (!error) return null;
  return errorKey(error) === dismissed ? null : error;
}

/** A failure's identity, so "OK" hides it in this browser until another one. */
export const errorKey = (error: UpdateError): string => `${error.code}:${error.version ?? ""}:${error.reason}`;

/** The entry's look, or null when there is nothing to show. */
export function entryLook(summary: UpdateSummary | undefined, status: UpdateStatus | null, dismissed: string | null): EntryLook | null {
  if (!summary) return null;
  const state = status?.state ?? summary.state;
  if (BUSY.has(state)) return "busy";
  if (shownError(status, dismissed)) return "failed";
  const newer = status ? status.newer : summary.newer;
  return newer ? "available" : null;
}

/** How often to ask: every second while it works with the card open (or restarts), every 3 s while it
 * works or the card is open (another tab may start or fail it), else every 30 s. */
export function pollDelay(state: UpdateState | null, open: boolean): number {
  const busy = state !== null && BUSY.has(state);
  if (busy && (open || state === "restarting")) return 1000;
  if (busy || open) return 3000;
  return 30_000;
}

/** The card's "now": the version the page runs on (the one it loaded from, or the one it reconnected
 * to), not the server's `current`, which names the new version as soon as it is installed. */
export function runningVersion(running: string | null, status: Pick<UpdateStatus, "current">): string {
  return running ?? status.current ?? "";
}

/** The release notes of a version, on the server's source repository. */
export function notesFor(sourceUrl: string, version: string): string {
  return `${sourceUrl.replace(/\/+$/, "")}/releases/tag/v${version}`;
}
