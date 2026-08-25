/** Jupyter-style serve access token.
 *
 * The server prints a `?token=` URL at startup; requests from other machines
 * must present that token. `initToken` captures it once from the boot URL into
 * sessionStorage and scrubs it from the address bar; API fetches then send it
 * as `Authorization: Bearer` and header-less browser requests (EventSource,
 * download links, inline images) append it back as `?token=`. Without a
 * stored token everything is sent bare — loopback sessions need none.
 */

const STORAGE_KEY = "markitai.serve.token";

/** Fallback when sessionStorage is unavailable (e.g. blocked storage). */
let memoryToken: string | null = null;

/** Capture `?token=` into sessionStorage and remove it from the URL bar. */
export function initToken(): void {
  const url = new URL(window.location.href);
  const token = url.searchParams.get("token");
  if (token === null || token === "") return;
  memoryToken = token;
  try {
    sessionStorage.setItem(STORAGE_KEY, token);
  } catch {
    /* memoryToken carries this tab through */
  }
  url.searchParams.delete("token");
  window.history.replaceState(window.history.state, "", url.toString());
}

export function getToken(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch {
    return memoryToken;
  }
}

/** Append `?token=` for URLs the browser requests without headers. */
export function withToken(url: string): string {
  const token = getToken();
  if (token === null) return url;
  return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
}

/** `Authorization: Bearer` header for API fetches; empty without a token. */
export function authHeaders(): Record<string, string> {
  const token = getToken();
  return token === null ? {} : { Authorization: `Bearer ${token}` };
}
