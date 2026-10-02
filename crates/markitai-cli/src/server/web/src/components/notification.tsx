// A flat notification card at the top right (docked to the bottom on phones):
// a coloured rule, an icon box, a title, one sentence, an optional action.
import { createPortal } from "preact/compat";
import { Icon } from "./icons.tsx";

export type Tone = "warning" | "success" | "error";

export interface NotificationModel {
  tone: Tone;
  title: string;
  message: string;
  detail?: string;
  action?: { label: string; run: () => void };
}

/** One shared column at the screen edge, so several cards stack instead of overlapping. */
function stack(): HTMLElement {
  let node = document.querySelector<HTMLElement>("body > .notice-stack");
  if (!node) {
    node = document.createElement("div");
    node.className = "notice-stack";
    document.body.append(node);
  }
  return node;
}

export function Notification({ note, closeLabel, onClose }: { note: NotificationModel; closeLabel: string; onClose: () => void }) {
  return createPortal(
    <aside
      class={`notice-card is-${note.tone}`}
      role={note.tone === "error" ? "alert" : "status"}
      aria-live={note.tone === "error" ? "assertive" : "polite"}
    >
      <span class="notice-icon" aria-hidden="true">
        <Icon name={note.tone === "success" ? "CheckBold" : "WarningFill"} size={18} />
      </span>
      <span class="notice-copy">
        <strong>{note.title}</strong>
        <span title={note.detail || undefined}>{note.message}</span>
        {note.action && (
          <button type="button" class="notice-action" onClick={note.action.run}>
            {note.action.label}
          </button>
        )}
      </span>
      <button type="button" class="notice-close" aria-label={closeLabel} title={closeLabel} onClick={onClose}>
        <Icon name="X" size={14} />
      </button>
    </aside>,
    stack(),
  );
}
