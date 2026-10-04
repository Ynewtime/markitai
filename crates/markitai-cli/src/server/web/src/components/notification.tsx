// A flat notification card at the top right (docked to the bottom on phones):
// a coloured rule, readable full details, and optional actions.
import { createPortal } from "preact/compat";
import { useLayoutEffect, useRef } from "preact/hooks";
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

type NoticeEntry = { node: HTMLElement; close: () => void };
const notices = new Set<NoticeEntry>();
const escapeTargets = new WeakMap<KeyboardEvent, NoticeEntry>();
const HIGHER_LAYER = '[role="dialog"], [role="alertdialog"], .confirm-card, .pdf-card, .help-bubble';

function higherLayerOpen(): boolean {
  return [...document.querySelectorAll<HTMLElement>(HIGHER_LAYER)].some(
    (node) => !node.closest("[hidden]") && node.getClientRects().length > 0,
  );
}

// Observe before an inner layer closes. In particular, help bubbles consume
// Escape on window without preventDefault; checking only afterwards is too late.
function rememberEscape(event: KeyboardEvent): void {
  if (event.key !== "Escape" || event.isComposing || event.keyCode === 229 || event.repeat || higherLayerOpen()) return;
  // A native select owns Escape even when its OS popup does not cancel keydown.
  if (event.target instanceof Element && event.target.closest("select")) return;
  const latest = [...notices].reverse().find((entry) => entry.node.isConnected);
  if (latest) escapeTargets.set(event, latest);
}

function dismissNotice(event: KeyboardEvent): void {
  const entry = escapeTargets.get(event);
  escapeTargets.delete(event);
  // Target handlers (such as the ledger filter) and modal layers get first use.
  if (!entry || event.defaultPrevented || event.isComposing || !notices.has(entry) || !entry.node.isConnected || higherLayerOpen()) return;
  event.preventDefault();
  entry.close();
}

function registerNotice(entry: NoticeEntry): () => void {
  if (notices.size === 0) {
    window.addEventListener("keydown", rememberEscape, true);
    window.addEventListener("keydown", dismissNotice);
  }
  notices.add(entry);
  return () => {
    notices.delete(entry);
    if (notices.size === 0) {
      window.removeEventListener("keydown", rememberEscape, true);
      window.removeEventListener("keydown", dismissNotice);
    }
  };
}

export function Notification({ note, replay = 0, closeLabel, detailsLabel, warningsLabel, onClose }: { note: NotificationModel; replay?: number; closeLabel: string; detailsLabel?: string; warningsLabel?: string; onClose: () => void }) {
  const card = useRef<HTMLElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useLayoutEffect(() => {
    const node = card.current;
    if (!node) return;
    // A newly shown or replayed notice is the topmost non-modal notice.
    return registerNotice({ node, close: () => close.current() });
  }, [note, replay]);
  // The existing settings dialog supplies only closeLabel; use the active document language.
  const words = dicts[document.documentElement.lang.startsWith("zh") ? "zh" : "en"];
  const warningsTitle = note.warningsTitle ?? warningsLabel ?? words.itemWarningsTitle;
  return createPortal(
    <aside
      ref={card}
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
