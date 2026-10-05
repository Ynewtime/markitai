// The URL line: a textarea dressed as one input that grows a row per pasted URL
// (up to six), above a bar with the composer tools on the left and Convert on
// the right. Enter converts, Shift+Enter breaks the line, and confirming an
// IME composition never submits.
import type { ComponentChildren } from "preact";
import { useRef, useState } from "preact/hooks";
import { NARROW, useMedia } from "../hooks/use-media.ts";
import type { Dict } from "../i18n/index.ts";
import type { StagedSubmission } from "../lib/files.ts";
import { Icon } from "./icons.tsx";

export function UrlInput({
  t,
  text,
  onText,
  onConvert,
  busy = false,
  compact = false,
  staged = null,
  onClearStaged,
  tools,
}: {
  t: Dict;
  text: string;
  onText: (text: string) => void;
  onConvert: (text: string) => Promise<boolean>;
  busy?: boolean;
  compact?: boolean;
  /** A batch that is waiting for Convert: a folder, several files or a URL list. */
  staged?: StagedSubmission | null;
  onClearStaged?: () => void;
  /** Controls placed at the start of the bottom bar. */
  tools?: ComponentChildren;
}) {
  const [sending, setSending] = useState(false);
  const field = useRef<HTMLTextAreaElement>(null);
  const narrow = useMedia(NARROW);
  // A staged batch is something to convert even with no URL typed.
  const empty = text.trim() === "" && staged === null;
  const rows = Math.min(6, Math.max(1, text.split("\n").length));
  const submit = async () => {
    if (empty || sending) return;
    setSending(true);
    try {
      if (await onConvert(text)) onText("");
    } finally {
      setSending(false);
      field.current?.focus({ preventScroll: true });
    }
  };
  return (
    <div class={compact ? "url-row is-compact" : "url-row"}>
      {staged && (
        <div class="staged-row">
          <Icon name={staged.kind === "urls" ? "Globe" : staged.kind === "folder" ? "FolderSimple" : "FileText"} size={14} />
          {staged.label && (
            <span class="staged-name" title={staged.label}>
              {staged.label}
            </span>
          )}
          <span class={staged.label ? "staged-count" : "staged-name"}>{t.stagedCount(staged.files.length)}</span>
          <button
            type="button"
            class="staged-clear"
            aria-label={t.stagedClear(staged.label || t.stagedCount(staged.files.length))}
            title={t.stagedClear(staged.label || t.stagedCount(staged.files.length))}
            onClick={() => onClearStaged?.()}
          >
            <Icon name="X" size={12} />
          </button>
        </div>
      )}
      <div class="url-field">
        <textarea
          ref={field}
          class="url-box"
          rows={rows}
          value={text}
          placeholder={narrow ? t.urlPlaceholderShort : t.urlPlaceholder}
          spellcheck={false}
          aria-label={t.urlPlaceholder}
          onInput={(event) => onText(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
            event.preventDefault();
            void submit();
          }}
        />
      </div>
      {/* A press on the bar's empty space puts the caret back in the URL line. */}
      <div
        class="url-bar"
        onMouseDown={(event) => {
          if (event.target !== event.currentTarget) return;
          event.preventDefault();
          field.current?.focus();
        }}
      >
        {tools}
        <button
          type="button"
          class="tool tool-convert"
          aria-label={t.convert}
          disabled={busy || sending || empty}
          aria-busy={busy || sending || undefined}
          onClick={() => void submit()}
        >
          <Icon name="ArrowRight" size={14} />
          <span class="tool-text">{t.convert}</span>
        </button>
      </div>
    </div>
  );
}
