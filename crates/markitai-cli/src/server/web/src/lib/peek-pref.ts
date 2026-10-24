// The peek window's per-browser choices: open at full screen or as the window, and Details' request options.
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

// Details' request options (the command line and its chips): open or folded, one choice for this browser.
const REQUEST_KEY = "markitai.request";
export function readRequestOpen(storage: Pick<Storage, "getItem"> | null = safeStorage()): boolean {
  try {
    return storage?.getItem(REQUEST_KEY) === "open";
  } catch {
    return false;
  }
}
export function storeRequestOpen(open: boolean, storage: Pick<Storage, "setItem"> | null = safeStorage()): void {
  try {
    storage?.setItem(REQUEST_KEY, open ? "open" : "closed");
  } catch {
    // A browser without storage keeps the choice for this page only.
  }
}
// The view Preview opens with: Details' result start in Source opens Preview's Source (owner,
// 2026-10-25), anything else Rendered. Kept for this page only and taken by the next Preview.
let nextView: "rendered" | "source" = "rendered";
export function openPreviewAs(view: "rendered" | "source"): void {
  nextView = view;
}
export function takePreviewView(): "rendered" | "source" {
  const view = nextView;
  nextView = "rendered";
  return view;
}
function safeStorage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}
