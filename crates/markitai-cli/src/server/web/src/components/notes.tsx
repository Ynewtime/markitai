// Transient notices as toast cards in one stack at the bottom of the window: on a
// wide screen at the bottom right, on a phone centred directly above the tab bar
// (or the selection bar). Newest at the bottom, at most three; they never move
// the page. Each is a sentence with a mark (✕ error, ! warning, ✓ done), at most
// one text action, folded details when there are any, and a close button. Done
// and warning notices leave after a few seconds, but not while hovered,
// pressed or focused; errors stay until closed. A swipe down dismisses one.
// Lasting conditions are not notices: they live beside what they affect.
import { useEffect, useRef, useState } from "preact/hooks";
import { withLead } from "../lib/toast.ts";
import { Icon } from "./icons.tsx";

export interface Note {
  id: string;
  tone: "error" | "warning" | "success";
  text: string;
  title?: string;
  detail?: string;
  action?: { label: string; run: () => void };
  /** Busy notices (an upload) show an arrow and never leave by themselves. */
  busy?: boolean;
}

/** The most notices shown at once; a fourth removes the oldest at once. */
export const MAX_NOTES = 3;
/** How long a done or warning notice stays (errors stay until closed). */
const STAY_MS = 5000;
const LEAVE_MS = 140;
const MARK: Record<Note["tone"], string> = { error: "✕", warning: "!", success: "✓" };

function Toast({ note, closeLabel, detailsLabel, lead, onDismiss }: { note: Note; closeLabel: string; detailsLabel: string; lead: string; onDismiss: (id: string) => void }) {
  const [held, setHeld] = useState({ pointer: false, focus: false, press: false });
  const [leaving, setLeaving] = useState(false);
  const start = useRef<number | null>(null);
  const paused = held.pointer || held.focus || held.press;
  const leave = () => {
    if (leaving) return;
    setLeaving(true);
    setTimeout(() => onDismiss(note.id), LEAVE_MS);
  };
  useEffect(() => {
    if (note.busy || note.tone === "error" || paused || leaving) return;
    const timer = setTimeout(leave, STAY_MS);
    return () => clearTimeout(timer);
  }, [note, paused, leaving]);
  return (
    <li
      class={`toast is-${note.tone}${leaving ? " is-leaving" : ""}`}
      onPointerEnter={() => setHeld((state) => ({ ...state, pointer: true }))}
      onPointerLeave={() => setHeld((state) => ({ ...state, pointer: false, press: false }))}
      onPointerDown={(event) => {
        start.current = event.clientY;
        // Keep the gesture when the finger or pointer leaves the card downwards.
        if (!(event.target as Element).closest("button, summary")) event.currentTarget.setPointerCapture?.(event.pointerId);
        setHeld((state) => ({ ...state, press: true }));
      }}
      onPointerUp={(event) => {
        // A swipe down dismisses it.
        if (start.current !== null && event.clientY - start.current > 24) leave();
        start.current = null;
        setHeld((state) => ({ ...state, press: false }));
      }}
      onPointerCancel={() => {
        start.current = null;
        setHeld((state) => ({ ...state, press: false }));
      }}
      onFocusIn={() => setHeld((state) => ({ ...state, focus: true }))}
      onFocusOut={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setHeld((state) => ({ ...state, focus: false }));
      }}
    >
      <span class="note-mark" aria-hidden="true">
        {note.busy ? "↑" : MARK[note.tone]}
      </span>
      <div class="toast-copy">
        <p class="toast-text">
          {note.title && <strong>{note.text ? `${note.title}${lead}` : note.title}</strong>}
          {note.text}
        </p>
        {note.detail && (
          <details class="toast-detail">
            <summary>{detailsLabel}</summary>
            <p>{note.detail}</p>
          </details>
        )}
      </div>
      {note.action && (
        <button type="button" class="note-action" onClick={note.action.run}>
          {note.action.label}
        </button>
      )}
      {!note.busy && (
        <button type="button" class="icon-btn toast-close" aria-label={closeLabel} title={closeLabel} onClick={leave}>
          <Icon name="X" size={14} />
        </button>
      )}
    </li>
  );
}

/** The stack, and two live regions that read every new notice (errors at once,
 * the others politely). The regions are always in the page, apart from the
 * stack, so its visual order (newest last) never decides what is read. */
export function NoticeStack({ label, closeLabel, detailsLabel, lead, notes, onDismiss }: { label: string; closeLabel: string; detailsLabel: string; lead: string; notes: Note[]; onDismiss: (id: string) => void }) {
  const [said, setSaid] = useState({ alert: "", status: "" });
  const heard = useRef(new Set<string>());
  useEffect(() => {
    for (const note of notes) {
      const key = `${note.id}\u0000${note.text}`;
      if (heard.current.has(key)) continue;
      heard.current.add(key);
      const channel = note.tone === "error" ? "alert" : "status";
      const text = note.title ? withLead(note.title, note.text, lead) : note.text;
      setSaid((state) => ({ ...state, [channel]: state[channel] === text ? `${text} ` : text }));
    }
  }, [notes, lead]);
  const shown = notes.slice(-MAX_NOTES);
  return (
    <>
      {shown.length > 0 && (
        <section class="toasts" aria-label={label}>
          <ul>
            {shown.map((note) => (
              <Toast key={note.id} note={note} closeLabel={closeLabel} detailsLabel={detailsLabel} lead={lead} onDismiss={onDismiss} />
            ))}
          </ul>
        </section>
      )}
      <div class="sr-only" role="alert">
        {said.alert}
      </div>
      <div class="sr-only" role="status">
        {said.status}
      </div>
    </>
  );
}
