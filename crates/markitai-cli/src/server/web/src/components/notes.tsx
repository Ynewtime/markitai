// Page notices, in the flow of the page instead of floating over it: they push
// the content down rather than cover a button. One line each, newest first,
// with an optional action and the service's full wording behind "Details".
// A notice that asks nothing (a confirmation) clears itself after a while, but
// not while the pointer or the keyboard is on it. The live regions are always
// in the page, so a notice that appears later is announced.
import { useEffect, useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { Icon } from "./icons.tsx";

export interface Note {
  id: string;
  tone: "error" | "warning" | "success" | "info";
  text: string;
  detail?: string;
  action?: { label: string; run: () => void };
  /** Busy notices (an upload) show a spinner and never expire. */
  busy?: boolean;
  /** Milliseconds before a notice without an action clears itself; 0 keeps it. */
  ttl?: number;
}

function NoteLine({ t, note, onDismiss }: { t: Dict; note: Note; onDismiss: (id: string) => void }) {
  const [held, setHeld] = useState({ pointer: false, focus: false });
  const paused = held.pointer || held.focus;
  useEffect(() => {
    const ttl = note.ttl ?? (note.tone === "success" || note.tone === "info" ? 6000 : 0);
    if (!ttl || note.busy || note.action || paused) return;
    const timer = setTimeout(() => onDismiss(note.id), ttl);
    return () => clearTimeout(timer);
  }, [note, onDismiss, paused]);
  const icon = note.tone === "success" ? "Check" : note.tone === "info" ? "Info" : "Warning";
  return (
    <li
      class={`note is-${note.tone}${note.action ? " has-action" : ""}`}
      onPointerEnter={() => setHeld((state) => ({ ...state, pointer: true }))}
      onPointerLeave={() => setHeld((state) => ({ ...state, pointer: false }))}
      onFocusIn={() => setHeld((state) => ({ ...state, focus: true }))}
      onFocusOut={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setHeld((state) => ({ ...state, focus: false }));
      }}
    >
      {note.busy ? <span class="spinner" aria-hidden="true" /> : <Icon name={icon} size={15} />}
      <div class="note-copy">
        <span class="note-text">{note.text}</span>
        {note.detail && (
          <details class="note-details">
            <summary>{t.notificationDetails}</summary>
            <span class="raw">{note.detail}</span>
          </details>
        )}
      </div>
      {note.action && (
        <button type="button" class="btn btn-sm note-action" onClick={note.action.run}>
          {note.action.label}
        </button>
      )}
      {!note.busy && (
        <button type="button" class="icon-btn note-close" aria-label={t.dismiss} title={t.dismiss} onClick={() => onDismiss(note.id)}>
          <Icon name="X" size={13} />
        </button>
      )}
    </li>
  );
}

export function Notes({ t, notes, onDismiss }: { t: Dict; notes: Note[]; onDismiss: (id: string) => void }) {
  const errors = notes.filter((note) => note.tone === "error");
  const rest = notes.filter((note) => note.tone !== "error");
  return (
    <div class={notes.length ? "notes" : "notes is-empty"}>
      {/* Errors are announced at once; the others politely. */}
      <div role="alert" class="notes-region">
        {errors.length > 0 && (
          <ul class="notes-list">
            {errors.map((note) => (
              <NoteLine key={note.id} t={t} note={note} onDismiss={onDismiss} />
            ))}
          </ul>
        )}
      </div>
      <div role="status" class="notes-region">
        {rest.length > 0 && (
          <ul class="notes-list">
            {rest.map((note) => (
              <NoteLine key={note.id} t={t} note={note} onDismiss={onDismiss} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
