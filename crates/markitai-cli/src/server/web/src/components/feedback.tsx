// Single-line monospaced feedback under the source card: the capability hint,
// the printed-red error line (the service's own wording on hover), and
// neutral notices such as upload progress.
import type { ComponentChildren } from "preact";
import type { Dict } from "../i18n/index.ts";

export function CapabilityHint({ t, onOpen }: { t: Dict; onOpen: () => void }) {
  return (
    <p class="line-hint">
      {t.capHintPre}
      <button type="button" class="text-link" onClick={onOpen}>
        {t.capHintLink}
      </button>
      {t.capHintPost}
    </p>
  );
}

export function ErrorLine({ text, detail, children }: { text: string; detail?: string; children?: ComponentChildren }) {
  return (
    <p class="line-error" role="alert" title={detail || undefined}>
      {text}
      {children}
    </p>
  );
}

export function NoticeLine({ children, busy = false }: { children: ComponentChildren; busy?: boolean }) {
  return (
    <p class={busy ? "line-note is-busy" : "line-note"} role="status">
      {busy && <span class="spinner" aria-hidden="true" />}
      {children}
    </p>
  );
}

export function InlineAction({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button type="button" class="inline-action" onClick={onClick}>
      {label}
    </button>
  );
}
