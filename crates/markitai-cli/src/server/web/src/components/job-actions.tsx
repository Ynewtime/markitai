// The workspace heading with session counters, and its actions: Stop remaining,
// Retry all failed, Clear, and the archive download under the ledger.
import { useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { download } from "../lib/download.ts";
import { fmtCost } from "../lib/format.ts";
import type { SessionStats } from "../lib/session.ts";
import { Icon } from "./icons.tsx";

export function JobStats({ t, running, stats }: { t: Dict; running: boolean; stats: SessionStats }) {
  return (
    <div>
      <h2 class="work-title">{t.conversions}</h2>
      {stats.total > 0 && (
        <div class="work-stats">
          <strong>
            {t.currentSession} · {running ? `${t.statusRunning} · ` : ""}
            {stats.done}/{stats.total} {t.statusDone}
          </strong>
          {stats.skipped > 0 && ` · ${stats.skipped} ${t.statusSkipped}`}
          {stats.failed > 0 && ` · ${stats.failed} ${t.statusFailed}`}
          {stats.hasCost && ` · ${fmtCost(stats.costTotal)}`}
        </div>
      )}
    </div>
  );
}

export function ClearButton({
  t,
  activeCount,
  finishedJobs,
  onClear,
}: {
  t: Dict;
  activeCount: number;
  finishedJobs: number;
  onClear: () => void;
}) {
  const onlyFinished = activeCount > 0;
  const nothing = onlyFinished && finishedJobs === 0;
  return (
    <button type="button" class="btn btn-ghost" disabled={nothing} title={nothing ? t.nothingCompleted : undefined} onClick={onClear}>
      {onlyFinished ? t.clearCompleted : t.clearAll}
    </button>
  );
}

/** The archive of every saved job, streamed by the browser itself; with a
 * token, through a single-use download ticket. */
export function ZipButton({
  t,
  available,
  activeCount,
  onError,
}: {
  t: Dict;
  available: boolean;
  activeCount: number;
  onError: (error: unknown) => void;
}) {
  const [busy, setBusy] = useState(false);
  if (!available && activeCount === 0) return null;
  if (!available) {
    return (
      <button type="button" class="btn btn-primary zip-btn" disabled title={t.zipWhileRunning}>
        <Icon name="DownloadSimple" size={15} />
        {t.downloadAllZip}
      </button>
    );
  }
  const run = async () => {
    if (busy) return;
    setBusy(true);
    try {
      await download("/api/history/archive", "markitai-history.zip", true);
    } catch (error) {
      onError(error);
    } finally {
      setBusy(false);
    }
  };
  return (
    <button type="button" class="btn btn-primary zip-btn" disabled={busy} aria-busy={busy || undefined} onClick={() => void run()}>
      {busy ? <span class="spinner" aria-hidden="true" /> : <Icon name="DownloadSimple" size={15} />}
      {busy ? t.downloadingZip : t.downloadAllZip}
    </button>
  );
}
