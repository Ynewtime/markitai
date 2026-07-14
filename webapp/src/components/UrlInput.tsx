import { useEffect, useMemo, useRef, useState } from "react";
import type { Dict } from "../i18n";

/** App's mobile breakpoint (matches app.css) — below it the full en
 * placeholder wraps and the 1-row textarea clips it, so swap in the short
 * copy. aria-label keeps the full hint. */
const NARROW_Q = "(max-width: 780px)";

function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && window.matchMedia(NARROW_Q).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia(NARROW_Q);
    const onChange = () => setNarrow(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return narrow;
}

/** URL entry: a textarea styled as the mock's single input — pasting
 * multi-line text grows it one row per URL. Enter converts (the placeholder
 * says so); Shift+Enter inserts a newline; Cmd/Ctrl+Enter still works.
 * `compact` is the slim-composer variant that lives in the workspace.
 * The draft is owned by App (the CLI-command line mirrors it live). */
export function UrlInput({
  t,
  text,
  onText,
  onConvert,
  compact = false,
}: {
  t: Dict;
  text: string;
  onText: (text: string) => void;
  onConvert: (urls: string[]) => Promise<boolean>;
  compact?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const narrow = useNarrow();

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
    if (urls.length === 0 || busy) return;
    setBusy(true);
    try {
      const created = await onConvert(urls);
      if (created) onText("");
    } finally {
      setBusy(false);
      inputRef.current?.focus({ preventScroll: true });
    }
  };

  return (
    <div className={compact ? "urlrow compact" : "urlrow"}>
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
      <button
        type="button"
        className={compact ? "btn primary" : "btn primary lg"}
        disabled={busy || urls.length === 0}
        onClick={() => void submit()}
      >
        {t.convert}
      </button>
    </div>
  );
}
