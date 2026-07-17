import { useState } from "react";
import type { Dict } from "../i18n";
import { DownloadIcon } from "./icons";

/** Job-header clear. Session-scoped: while conversions run it only removes
 * settled rows, so the label and disabled state track that split. */
export function ClearJobsButton({
  t,
  activeCount,
  clearableJobCount,
  onClear,
}: {
  t: Dict;
  activeCount: number;
  clearableJobCount: number;
  onClear: () => void;
}) {
  const clearingCompleted = activeCount > 0;
  const clearDisabled = clearingCompleted && clearableJobCount === 0;
  return (
    <button
      type="button"
      className="btn ghost"
      disabled={clearDisabled}
      title={clearDisabled ? t.nothingCompleted : undefined}
      onClick={onClear}
    >
      {clearingCompleted ? t.clearCompleted : t.clearAll}
    </button>
  );
}

/** Composer-row ZIP download. The archive covers every completed server job
 * and stays disabled until active conversions settle; with nothing completed
 * and nothing running there is no archive to offer, so it renders nothing. */
export function DownloadArchiveButton({
  t,
  zipHref,
  activeCount,
  onDownloadError,
}: {
  t: Dict;
  zipHref: string | null;
  activeCount: number;
  onDownloadError: (message: string) => void;
}) {
  const [downloading, setDownloading] = useState(false);

  const downloadArchive = async () => {
    if (zipHref === null || downloading) return;
    setDownloading(true);
    try {
      const response = await fetch(zipHref);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const disposition = response.headers.get("content-disposition") ?? "";
      const encodedName = /filename\*=UTF-8''([^;]+)/i.exec(disposition)?.[1];
      const plainName = /filename="?([^";]+)"?/i.exec(disposition)?.[1];
      let filename = plainName ?? "markitai-all.zip";
      if (encodedName !== undefined) {
        try {
          filename = decodeURIComponent(encodedName);
        } catch {
          filename = encodedName;
        }
      }
      link.href = objectUrl;
      link.download = filename;
      link.style.display = "none";
      document.body.append(link);
      link.click();
      link.remove();
      // Safari may consume the object URL after the synthetic click returns.
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
    } catch (error) {
      onDownloadError(error instanceof Error ? error.message : String(error));
    } finally {
      setDownloading(false);
    }
  };

  return zipHref !== null ? (
    <button
      type="button"
      className="btn primary zipbtn"
      disabled={downloading}
      aria-busy={downloading || undefined}
      onClick={() => void downloadArchive()}
    >
      {downloading ? (
        <span className="spin" aria-hidden="true" />
      ) : (
        <DownloadIcon size={15} />
      )}
      {downloading ? t.downloadingZip : t.downloadAllZip}
    </button>
  ) : activeCount > 0 ? (
    <button type="button" className="btn primary zipbtn" disabled title={t.zipWhileRunning}>
      <DownloadIcon size={15} />
      {t.downloadAllZip}
    </button>
  ) : null;
}
