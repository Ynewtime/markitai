import { useEffect, useRef, useState } from "react";
import type { Dict } from "../i18n";
import { DownloadIcon } from "./icons";

const CONFIRM_RESET_MS = 4000;

/** Workspace job-header actions.
 *
 * clear: always "clear all" (session scope); while items are still active it
 * becomes a two-step inline confirm ("discard n running?") — no modal.
 * zip: archive of the selected item's job; disabled until that job reaches a
 * terminal state (the server answers 409 while it runs). */
export function DownloadActions({
  t,
  multiJob,
  zipHref,
  jobRunning,
  activeCount,
  onClear,
}: {
  t: Dict;
  multiJob: boolean;
  zipHref: string | null;
  jobRunning: boolean;
  activeCount: number;
  onClear: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const timerRef = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, []);
  // Once everything settled, plain clear applies again.
  useEffect(() => {
    if (activeCount === 0) setConfirming(false);
  }, [activeCount]);

  const handleClear = () => {
    if (activeCount > 0 && !confirming) {
      setConfirming(true);
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      timerRef.current = window.setTimeout(() => setConfirming(false), CONFIRM_RESET_MS);
      return;
    }
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    setConfirming(false);
    onClear();
  };

  const zipLabel = multiJob ? t.downloadJobZip : t.downloadAllZip;
  return (
    <div className="actions">
      <button type="button" className="btn ghost" onClick={handleClear}>
        {confirming ? t.confirmClear(activeCount) : t.clearAll}
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
