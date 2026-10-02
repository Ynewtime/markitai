// The URL line: a textarea dressed as one input that grows a row per pasted URL
// (up to six). Enter converts, Shift+Enter breaks the line, and confirming an
// IME composition never submits.
import { useRef, useState } from "preact/hooks";
import { NARROW, useMedia } from "../hooks/use-media.ts";
import type { Dict } from "../i18n/index.ts";
import { Icon } from "./icons.tsx";

export function UrlInput({
  t,
  text,
  onText,
  onConvert,
  busy = false,
  compact = false,
}: {
  t: Dict;
  text: string;
  onText: (text: string) => void;
  onConvert: (text: string) => Promise<boolean>;
  busy?: boolean;
  compact?: boolean;
}) {
  const [sending, setSending] = useState(false);
  const field = useRef<HTMLTextAreaElement>(null);
  const narrow = useMedia(NARROW);
  const empty = text.trim() === "";
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
  );
}
