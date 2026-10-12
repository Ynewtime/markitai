// The peek window's one per-browser choice: open at full screen or as the window.
const PEEK_KEY = "markitai.peek";
/** Whether this browser opens conversions at full screen. The docked layout
 * of earlier versions ("docked") is gone: it opens as the window again. */
export function readFull(storage: Pick<Storage, "getItem" | "setItem"> | null = safeStorage()): boolean {
  try {
    const value = storage?.getItem(PEEK_KEY) ?? null;
    if (value === "docked") storage?.setItem(PEEK_KEY, "window");
    return value === "full";
  } catch {
    return false;
  }
}
export function storeFull(full: boolean, storage: Pick<Storage, "setItem"> | null = safeStorage()): void {
  try {
    storage?.setItem(PEEK_KEY, full ? "full" : "window");
  } catch {
    // A browser without storage keeps the choice for this page only.
  }
}
function safeStorage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}
