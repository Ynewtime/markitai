// Transient notices as minimal strips at the bottom: on a phone full width,
// directly above the tab bar (or an open sheet's action bar); on a wide screen a
// 380 px stack at the bottom right. Newest at the bottom, at most three. Each is
// one sentence with a mark (✕ error, ! warning, ✓ done) and at most one text
// action; it leaves after 4 s (6 s for an error), but not while it is pressed,
// hovered or focused, and a swipe down dismisses it. Lasting conditions are not
// notices: they live in the inspector of what they affect, or in a panel.
import { useEffect, useRef, useState } from "preact/hooks";

export interface Note {
  id: string;
  tone: "error" | "warning" | "success";
  text: string;
  action?: { label: string; run: () => void };
  /** Busy notices (an upload) show an arrow and never leave by themselves. */
  busy?: boolean;
}

/** The most notices shown at once; a fourth removes the oldest at once. */
export const MAX_NOTES = 3;
const STAY: Record<Note["tone"], number> = { success: 4000, warning: 4000, error: 6000 };
const LEAVE_MS = 140;
const MARK: Record<Note["tone"], string> = { error: "✕", warning: "!", success: "✓" };

function Strip({ note, onDismiss }: { note: Note; onDismiss: (id: string) => void }) {
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
    if (note.busy || paused || leaving) return;
    const timer = setTimeout(leave, STAY[note.tone]);
    return () => clearTimeout(timer);
  }, [note, paused, leaving]);
  return (
    <li
      class={`note-strip is-${note.tone}${leaving ? " is-leaving" : ""}`}
      onPointerEnter={() => setHeld((state) => ({ ...state, pointer: true }))}
      onPointerLeave={() => setHeld((state) => ({ ...state, pointer: false, press: false }))}
      onPointerDown={(event) => {
        start.current = event.clientY;
        // Keep the gesture when the finger or pointer leaves the strip downwards.
        if (!(event.target as Element).closest("button")) event.currentTarget.setPointerCapture?.(event.pointerId);
        setHeld((state) => ({ ...state, press: true }));
      }}
      onPointerUp={(event) => {
        // A swipe down (or a drag of a strip toward the edge) dismisses it.
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
      <span class="note-text" title={note.text}>
        {note.text}
      </span>
      {note.action && (
        <button type="button" class="note-action" onClick={note.action.run}>
          {note.action.label}
        </button>
      )}
    </li>
  );
}

/** The stack, and two live regions that read every new notice (errors at once,
 * the others politely). The regions are always in the page, apart from the
 * stack, so its visual order (newest last) never decides what is read. */
export function NoticeStack({ label, notes, onDismiss }: { label: string; notes: Note[]; onDismiss: (id: string) => void }) {
  const [said, setSaid] = useState({ alert: "", status: "" });
  const heard = useRef(new Set<string>());
  useEffect(() => {
    for (const note of notes) {
      const key = `${note.id}\u0000${note.text}`;
      if (heard.current.has(key)) continue;
      heard.current.add(key);
      const channel = note.tone === "error" ? "alert" : "status";
      setSaid((state) => ({ ...state, [channel]: state[channel] === note.text ? `${note.text} ` : note.text }));
    }
  }, [notes]);
  const shown = notes.slice(-MAX_NOTES);
  return (
    <>
      {shown.length > 0 && (
        <section class="note-strips" aria-label={label}>
          <ul>
            {shown.map((note) => (
              <Strip key={note.id} note={note} onDismiss={onDismiss} />
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
