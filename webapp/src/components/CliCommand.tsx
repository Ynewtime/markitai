import { useEffect, useId, useState } from "react";
import type { Preset } from "../api/types";
import type { Dict } from "../i18n";
import { buildCliCommand } from "../lib/cli";
import { TerminalIcon } from "./icons";

/** Terminal icon at the end of the options row — expands to a one-line
 * always-dark mono command strip with a copy badge (same badge + live-region
 * voice as the Source card). The command tracks the composer state live:
 * typed URLs are listed verbatim; with no URLs a <your-files> placeholder
 * stands in and a hint explains the swap. */
export function CliCommand({
  t,
  urls,
  preset,
  llm,
  ocr,
  announce,
}: {
  t: Dict;
  urls: string[];
  preset: Preset;
  llm: boolean;
  ocr: boolean;
  announce: (msg: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const id = useId();

  useEffect(() => {
    if (!copied) return;
    const h = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(h);
  }, [copied]);

  const cmd = buildCliCommand(urls, preset, llm, ocr);
  const placeholder = urls.length === 0;

  const copy = () => {
    navigator.clipboard.writeText(cmd).then(
      () => {
        setCopied(true);
        announce(t.copied);
      },
      () => undefined,
    );
  };

  return (
    <>
      <button
        type="button"
        className={open ? "clibtn on" : "clibtn"}
        aria-label={t.cliToggle}
        title={t.cliToggle}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
      >
        <TerminalIcon size={15} />
      </button>
      {open && (
        <div className="cliwrap" id={id}>
          <div className="clicmd">
            <code className="clitext" tabIndex={0} aria-label={t.cliAria}>
              <span className="clidollar" aria-hidden="true">
                ${" "}
              </span>
              {cmd}
            </code>
            <button type="button" className="badge" onClick={copy}>
              {copied ? t.copied : t.copy}
            </button>
          </div>
          {placeholder && <p className="clihint">{t.cliHint}</p>}
        </div>
      )}
    </>
  );
}
