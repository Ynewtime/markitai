// A flat notification card at the top right (docked to the bottom on phones):
// a coloured rule, readable full details, and optional actions.
import { createPortal } from "preact/compat";
import { Icon } from "./icons.tsx";
import { dicts } from "../i18n/index.ts";

export type Tone = "warning" | "success" | "error";

export interface NotificationModel {
  tone: Tone;
  title: string;
  message: string;
  detail?: string;
  warnings?: string[];
  warningsTitle?: string;
  warningsContext?: string;
  cost?: string;
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

export function Notification({ note, replay = 0, closeLabel, detailsLabel, warningsLabel, onClose }: { note: NotificationModel; replay?: number; closeLabel: string; detailsLabel?: string; warningsLabel?: string; onClose: () => void }) {
  // The existing settings dialog supplies only closeLabel; use the active document language.
  const words = dicts[document.documentElement.lang.startsWith("zh") ? "zh" : "en"];
  const warningsTitle = note.warningsTitle ?? warningsLabel ?? words.itemWarningsTitle;
  return createPortal(
    <aside
      class={`notice-card is-${note.tone}`}
      aria-label={note.title}
    >
      <span key={replay} class="sr-only" role={note.tone === "error" ? "alert" : "status"} aria-live={note.tone === "error" ? "assertive" : "polite"}>
        {note.title}. {note.message} {note.cost}
      </span>
      <span class="notice-icon" aria-hidden="true">
        <Icon name={note.tone === "success" ? "CheckBold" : "WarningFill"} size={18} />
      </span>
      <div class="notice-copy">
        <div class="notice-content">
          <strong>{note.title}</strong>
          <p class="notice-message">{note.message}</p>
          {note.cost && <p class="notice-cost">{note.cost}</p>}
          {note.warnings && note.warnings.length > 0 && (
            <section class="notice-warnings" aria-label={warningsTitle}>
              <strong>{warningsTitle}</strong>
              {note.warningsContext && <p class="notice-cost">{note.warningsContext}</p>}
              <ul>{note.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>
            </section>
          )}
          {note.detail && (
            <details class="notice-details">
              <summary>{detailsLabel ?? words.notificationDetails}</summary>
              <div class="notice-raw">{note.detail}</div>
            </details>
          )}
        </div>
        {note.action && (
          <button type="button" class="notice-action" onClick={note.action.run}>
            {note.action.label}
          </button>
        )}
      </div>
      <button type="button" class="notice-close" aria-label={closeLabel} title={closeLabel} onClick={onClose}>
        <Icon name="X" size={14} />
      </button>
    </aside>,
    stack(),
  );
}
