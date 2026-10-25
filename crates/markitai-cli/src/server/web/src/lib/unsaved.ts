// Edits that a reload would lose, registered by the views that hold them (the
// config editor's text). The page asks before it reloads itself after an update.
const held = new Set<string>();

export function setUnsaved(key: string, on: boolean): void {
  if (on) held.add(key);
  else held.delete(key);
}

/** Registered edits, or an open dialog (a model form, the options) whose input would go. */
export function hasUnsaved(doc: Pick<Document, "querySelector"> | undefined = globalThis.document): boolean {
  return held.size > 0 || Boolean(doc?.querySelector(".veil:not(.upd-veil)"));
}
