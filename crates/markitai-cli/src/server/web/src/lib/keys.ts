// Ctrl+B (Windows, Linux) / ⌘B (macOS) folds or unfolds the desktop sidebar, as its toggle does
// (owner, 2026-10-25). It never fires while typing or with a dialog open.

/** A Mac (or iPad) keyboard: ⌘ instead of Ctrl. */
export function isMacPlatform(platform: string): boolean {
  return /Mac|iPhone|iPad|iPod/i.test(platform);
}

/** This browser's platform, as the page can tell it. */
export function browserPlatform(): string {
  const nav = typeof navigator === "undefined" ? null : (navigator as Navigator & { userAgentData?: { platform?: string } });
  return nav?.userAgentData?.platform || nav?.platform || "";
}

/** The shortcut as it is written beside the toggle, and for `aria-keyshortcuts`. */
export function sidebarShortcut(mac: boolean): { label: string; aria: string } {
  return mac ? { label: "⌘B", aria: "Meta+B" } : { label: "Ctrl+B", aria: "Control+B" };
}

/** The key press is the shortcut: B (any layout's B key) with ⌘ on a Mac, Ctrl elsewhere, nothing else held. */
export function isSidebarShortcut(event: Pick<KeyboardEvent, "key" | "code" | "metaKey" | "ctrlKey" | "altKey" | "shiftKey" | "isComposing" | "repeat">, mac: boolean): boolean {
  if (event.isComposing || event.repeat || event.altKey || event.shiftKey) return false;
  if (mac ? !event.metaKey || event.ctrlKey : !event.ctrlKey || event.metaKey) return false;
  return event.code === "KeyB" || event.key === "b" || event.key === "B";
}

/** Typing here keeps Ctrl/⌘+B for itself (a text field, the config editor, editable text). */
export function isTypingTarget(target: { tagName?: string; isContentEditable?: boolean; closest?: (selector: string) => unknown } | null): boolean {
  if (!target) return false;
  const tag = (target.tagName ?? "").toUpperCase();
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (target.isContentEditable) return true;
  return Boolean(target.closest?.("[contenteditable]:not([contenteditable='false'])"));
}
