import { useEffect, useId, useState } from "react";
import type { Preset } from "../api/types";
import type { Dict } from "../i18n";
import { buildCliCommand } from "../lib/cli";
import { TerminalIcon } from "./icons";

export type CopyState = "idle" | "copied" | "failed";

/** navigator.clipboard only exists on secure origins — on LAN http the
 * unguarded call would throw synchronously in the click handler. Fall back
 * to a transient textarea + execCommand("copy"); resolves false when neither
 * path copied so callers can surface the failure. */
// eslint-disable-next-line react-refresh/only-export-components -- shared with MarkdownPreview; this file trades fast refresh for the one helper.
export async function copyTextToClipboard(text: string): Promise<boolean> {
  const clipboard: Clipboard | undefined = navigator.clipboard;
  if (clipboard !== undefined) {
    try {
      await clipboard.writeText(text);
      return true;
    } catch {
      // Permission denied or focus lost — the legacy path may still work.
    }
  }
  // select() moves focus to the textarea; restore it afterwards so a keyboard
  // user activating Copy keeps their place (and the button keeps focus for its
  // Copied/Copy failed badge swap).
  const previouslyFocused = document.activeElement;
  const host = document.createElement("textarea");
  host.value = text;
  host.setAttribute("readonly", "");
  // display:none would make the selection empty; park it off-view instead.
  host.style.position = "fixed";
  host.style.opacity = "0";
  document.body.append(host);
  host.select();
  let ok: boolean;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  host.remove();
  if (previouslyFocused instanceof HTMLElement) previouslyFocused.focus();
  return ok;
}

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
  const [copyState, setCopyState] = useState<CopyState>("idle");
  const id = useId();

  useEffect(() => {
    if (copyState === "idle") return;
    const h = window.setTimeout(() => setCopyState("idle"), 1500);
    return () => window.clearTimeout(h);
  }, [copyState]);

  const cmd = buildCliCommand(urls, preset, llm, ocr);
  const placeholder = urls.length === 0;

  const copy = () => {
    void copyTextToClipboard(cmd).then((ok) => {
      setCopyState(ok ? "copied" : "failed");
      announce(ok ? t.copied : t.copyFailed);
    });
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
        <TerminalIcon size={14} />
        {/* short technical token, same in both locales; the full name lives
            in aria-label/title */}
        <span aria-hidden="true">CLI</span>
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
              {copyState === "copied"
                ? t.copied
                : copyState === "failed"
                  ? t.copyFailed
                  : t.copy}
            </button>
          </div>
          {placeholder && <p className="clihint">{t.cliHint}</p>}
        </div>
      )}
    </>
  );
}
