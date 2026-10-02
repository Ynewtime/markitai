// The access token, for a remote visitor whose link lost it. It is kept for
// this tab only and sent as a Bearer header.
import { useRef, useState } from "preact/hooks";
import { useModal } from "../hooks/use-modal.ts";
import type { Dict } from "../i18n/index.ts";
import { Icon } from "./icons.tsx";

export function TokenDialog({ t, onSubmit, onClose }: { t: Dict; onSubmit: (token: string) => void; onClose: () => void }) {
  const dialog = useRef<HTMLDivElement>(null);
  const [value, setValue] = useState("");
  useModal(dialog, onClose);
  return (
    <div
      class="veil"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div ref={dialog} class="dialog" role="dialog" aria-modal="true" aria-labelledby="token-title" tabIndex={-1}>
        <div class="dialog-head">
          <h2 id="token-title" class="dialog-title">
            {t.tokenTitle}
          </h2>
          <button type="button" class="icon-btn" aria-label={t.close} title={t.close} onClick={onClose}>
            <Icon name="X" size={16} />
          </button>
        </div>
        <form
          class="dialog-body"
          onSubmit={(event) => {
            event.preventDefault();
            if (value.trim()) onSubmit(value);
          }}
        >
          <p class="dialog-text">{t.tokenText}</p>
          <label class="field">
            <span class="field-label">{t.tokenLabel}</span>
            <input type="password" value={value} autoComplete="off" spellcheck={false} onInput={(event) => setValue(event.currentTarget.value)} />
          </label>
          <p class="dialog-dim">{t.tokenNote}</p>
          <div class="form-actions">
            <button type="button" class="btn btn-ghost" onClick={onClose}>
              {t.cancel}
            </button>
            <button type="submit" class="btn btn-primary" disabled={!value.trim()}>
              {t.connect}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
