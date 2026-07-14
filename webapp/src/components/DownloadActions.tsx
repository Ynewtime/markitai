import type { Dict } from "../i18n";
import { DownloadIcon } from "./icons";

/** Workspace job-header actions. Running jobs stay in the current session;
 * while any are active, clear removes terminal jobs only. */
export function DownloadActions({
  t,
  multiJob,
  zipHref,
  jobRunning,
  activeCount,
  clearableJobCount,
  onClear,
}: {
  t: Dict;
  multiJob: boolean;
  zipHref: string | null;
  jobRunning: boolean;
  activeCount: number;
  clearableJobCount: number;
  onClear: () => void;
}) {
  const zipLabel = multiJob ? t.downloadJobZip : t.downloadAllZip;
  const clearingCompleted = activeCount > 0;
  const clearDisabled = clearingCompleted && clearableJobCount === 0;
  return (
    <div className="actions">
      <button
        type="button"
        className="btn ghost"
        disabled={clearDisabled}
        title={clearDisabled ? t.nothingCompleted : undefined}
        onClick={onClear}
      >
        {clearingCompleted ? t.clearCompleted : t.clearAll}
      </button>
      {zipHref !== null &&
        (jobRunning ? (
          <button type="button" className="btn primary" disabled title={t.zipWhileRunning}>
            <DownloadIcon size={15} />
            {zipLabel}
          </button>
        ) : (
          <a className="btn primary" href={zipHref} download>
            <DownloadIcon size={15} />
            {zipLabel}
          </a>
        ))}
    </div>
  );
}
