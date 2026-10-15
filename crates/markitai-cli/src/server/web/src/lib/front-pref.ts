// Whether documents show their front matter as the full properties table: one
// choice for this tab's session, shared by Preview, Compare and the Details excerpt.
const KEY = "markitai.front";
const listeners = new Set<(open: boolean) => void>();

function store(): Storage | null {
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

export function frontOpen(): boolean {
  try {
    return store()?.getItem(KEY) === "open";
  } catch {
    return false;
  }
}

export function setFrontOpen(open: boolean): void {
  try {
    store()?.setItem(KEY, open ? "open" : "closed");
  } catch {
    /* The choice lasts until the page reloads. */
  }
  for (const listener of listeners) listener(open);
}

export function onFrontOpen(listener: (open: boolean) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
