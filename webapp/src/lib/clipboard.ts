/** Copying text, with the fallbacks a non-secure context needs.
 *
 * Lives here rather than beside a component because three of them need
 * it, and hanging it off one made that file trade fast refresh for a
 * helper that has nothing to do with rendering.
 */

export type CopyState = "idle" | "copied" | "failed";

/** navigator.clipboard only exists on secure origins — on LAN http the
 * unguarded call would throw synchronously in the click handler. Fall back
 * to a transient textarea + execCommand("copy"); resolves false when neither
 * path copied so callers can surface the failure. */
export async function copyTextToClipboard(text: string): Promise<boolean> {
  const clipboard: Clipboard | undefined = navigator.clipboard;
  if (clipboard !== undefined) {
    try {
      await clipboard.writeText(text);
      return true;
    } catch {
      // Permission denied or focus lost — the legacy path may still work.
    }
  }
  // select() moves focus to the textarea; restore it afterwards so a keyboard
  // user activating Copy keeps their place (and the button keeps focus for its
  // Copied/Copy failed badge swap).
  const previouslyFocused = document.activeElement;
  const host = document.createElement("textarea");
  host.value = text;
  host.setAttribute("readonly", "");
  // display:none would make the selection empty; park it off-view instead.
  host.style.position = "fixed";
  host.style.opacity = "0";
  document.body.append(host);
  host.select();
  let ok: boolean;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  host.remove();
  if (previouslyFocused instanceof HTMLElement) previouslyFocused.focus();
  return ok;
}
