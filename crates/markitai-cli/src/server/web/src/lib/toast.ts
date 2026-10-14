// Any part of the workbench can raise a transient notice; the app shows it as a
// toast card in the one stack at the bottom of the window (components/notes.tsx).

export interface ToastInput {
  /** A notice raised again with the same id replaces the shown one. */
  id?: string;
  tone: "error" | "warning" | "success";
  text: string;
  /** A short lead in bold before the text ("Model test passed"). */
  title?: string;
  /** The service's own wording or more detail, folded under "Details". */
  detail?: string;
  action?: { label: string; run: () => void };
}

/** A lead and the rest as one sentence pair, in the language's own
 * punctuation (`join` is the dictionary's `toastLead`: ". " or "。"). */
export function withLead(lead: string, text: string, join: string): string {
  return text ? `${lead}${join}${text}` : lead;
}

type Listener = (note: ToastInput) => void;
const listeners = new Set<Listener>();

export function toast(note: ToastInput): void {
  for (const listener of listeners) listener(note);
}

export function dismissToast(id: string): void {
  for (const listener of listeners) listener({ id, tone: "success", text: "", dismiss: true } as ToastInput & { dismiss: true });
}

/** The app's stack listens here; returns the unsubscribe. */
export function onToast(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
