// Saving files without the token ever entering a URL. Without a token a plain
// link works (loopback visitors are trusted). With one, a file is fetched with
// the header token and handed to the browser as a Blob, so the page can report
// a refusal; an archive, which can be far larger than memory, is opened by the
// browser itself through a single-use ticket (`/api/download-tickets`) that is
// worthless once used or a minute old.
import { downloadTicket, fetchBlob } from "../api/client.ts";
import { hasToken } from "../api/token.ts";

export function saveURL(url: string, name: string, doc: Document = document): void {
  const link = doc.createElement("a");
  link.href = url;
  link.download = name;
  link.className = "copy-buffer";
  doc.body.append(link);
  link.click();
  link.remove();
}

export function saveBlob(blob: Blob, name: string, doc: Document = document): void {
  const url = URL.createObjectURL(blob);
  saveURL(url, name, doc);
  // Safari may read the object URL after the synthetic click returns.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export interface DownloadDeps {
  tokenHeld?: () => boolean;
  ticket?: (path: string) => Promise<string>;
  blob?: (path: string) => Promise<{ blob: Blob; name: string | null }>;
  saveURL?: (url: string, name: string) => void;
  saveBlob?: (blob: Blob, name: string) => void;
}

/** `native`: let the browser stream it (archives); otherwise read it as a Blob. */
export async function download(path: string, fallbackName: string, native = false, deps: DownloadDeps = {}): Promise<void> {
  const { tokenHeld = hasToken, ticket = downloadTicket, blob = fetchBlob } = deps;
  if (native) {
    (deps.saveURL ?? saveURL)(tokenHeld() ? await ticket(path) : path, fallbackName);
    return;
  }
  const file = await blob(path);
  (deps.saveBlob ?? saveBlob)(file.blob, file.name ?? fallbackName);
}

/** Click and middle-click handler for a download anchor whose href is the
 * plain path: with a token, the header download replaces the navigation. */
export function interceptDownload(event: MouseEvent, path: string, name: string, onError: (error: unknown) => void): void {
  if (event.type === "auxclick" && event.button !== 1) return;
  event.stopPropagation();
  if (!hasToken()) return;
  event.preventDefault();
  download(path, name).catch(onError);
}
