// Saving files. Without a token a plain link works (loopback visitors are
// trusted); with one, the file is fetched with the header token and handed to
// the browser as a Blob, so the token never appears in a download URL.
import { fetchBlob } from "../api/client.ts";
import { hasToken } from "../api/token.ts";

export function saveBlob(blob: Blob, name: string, doc: Document = document): void {
  const url = URL.createObjectURL(blob);
  const link = doc.createElement("a");
  link.href = url;
  link.download = name;
  link.className = "copy-buffer";
  doc.body.append(link);
  link.click();
  link.remove();
  // Safari may read the object URL after the synthetic click returns.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export async function download(path: string, fallbackName: string): Promise<void> {
  const { blob, name } = await fetchBlob(path);
  saveBlob(blob, name ?? fallbackName);
}

/** Click handler for a download anchor whose href has no token. */
export function interceptDownload(event: MouseEvent, path: string, name: string, onError: (error: unknown) => void): void {
  event.stopPropagation();
  if (!hasToken()) return;
  event.preventDefault();
  download(path, name).catch(onError);
}
