import { useMemo, useRef, useState } from "react";
import { useMediaQuery } from "../hooks/useMediaQuery";
import type { Dict } from "../i18n";
import { ArrowRightIcon } from "./icons";

/** App's mobile breakpoint (matches app.css) — below it the full en
 * placeholder wraps and the 1-row textarea clips it, so swap in the short
 * copy. aria-label keeps the full hint. */
const NARROW_Q = "(max-width: 780px)";

/** URL entry: a textarea styled as the mock's single input — pasting
 * multi-line text grows it one row per URL. Enter converts; Shift+Enter inserts a newline; Cmd/Ctrl+Enter still works.
 * `compact` is the slim-composer variant that lives in the workspace.
 * The draft is owned by App (the CLI-command line mirrors it live). */
export function UrlInput({
  t,
  text,
  onText,
  onConvert,
  busy: submitting = false,
  compact = false,
}: {
  t: Dict;
  text: string;
  onText: (text: string) => void;
  onConvert: (urls: string[]) => Promise<boolean>;
  busy?: boolean;
  compact?: boolean;
}) {
  const [sending, setSending] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const narrow = useMediaQuery(NARROW_Q);

  const urls = useMemo(
    () =>
      text
        .split("\n")
        .map((s) => s.trim())
        .filter((s) => s.length > 0),
    [text],
  );
  const rows = Math.min(6, Math.max(1, text.split("\n").length));

  const submit = async () => {
    if (urls.length === 0 || sending) return;
    setSending(true);
    try {
      const created = await onConvert(urls);
      if (created) onText("");
    } finally {
      setSending(false);
      inputRef.current?.focus({ preventScroll: true });
    }
  };

  return (
    <div className={compact ? "urlrow compact" : "urlrow"}>
      <div className="url-entry">
        <textarea
          ref={inputRef}
          className="urlin"
          rows={rows}
          value={text}
          placeholder={narrow ? t.urlPlaceholderShort : t.urlPlaceholder}
          spellCheck={false}
          aria-label={t.urlPlaceholder}
          onChange={(e) => onText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key !== "Enter" || e.shiftKey) return;
            if (e.nativeEvent.isComposing) return; // IME confirm, not submit
            e.preventDefault();
            void submit();
          }}
        />
      </div>
      <button
        type="button"
        className="srcact convert"
        disabled={submitting || sending || urls.length === 0}
        aria-busy={submitting || sending || undefined}
        onClick={() => void submit()}
      >
        <ArrowRightIcon size={14} />
        <span>{t.convert}</span>
      </button>
    </div>
  );
}
