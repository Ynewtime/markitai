// navigator.clipboard exists only in secure contexts; a service reached over
// plain HTTP on a LAN address falls back to the selection-copy command.

interface CopyEnvironment {
  clipboard?: { writeText(text: string): Promise<void> } | undefined;
  doc?: Document;
}

export async function copyText(text: string, env: CopyEnvironment = {}): Promise<boolean> {
  const clipboard = "clipboard" in env ? env.clipboard : globalThis.navigator?.clipboard;
  const doc = env.doc ?? globalThis.document;
  if (clipboard?.writeText) {
    try {
      await clipboard.writeText(text);
      return true;
    } catch {
      /* Permission refused or focus lost: the fallback may still work. */
    }
  }
  const focused = doc.activeElement;
  const area = doc.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.className = "copy-buffer";
  doc.body.append(area);
  area.select();
  let copied = false;
  try {
    copied = doc.execCommand("copy");
  } catch {
    copied = false;
  } finally {
    area.remove();
  }
  if (focused && "focus" in focused) (focused as HTMLElement).focus();
  return copied;
}
