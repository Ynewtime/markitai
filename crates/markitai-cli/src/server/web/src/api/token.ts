// The service access token. It arrives in the URL fragment of the launch link
// (`#token=`; `?token=` is still accepted), is removed from the address bar at
// once and kept only for this tab's session. Local and remote API calls both
// authenticate unless the operator explicitly starts the service with --no-auth.

export const TOKEN_KEY = "markitai.service-token";

let current = "";

interface TokenStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function sessionStore(): TokenStore | null {
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

/** Read the launch token from `href`, return the address without it. */
export function takeToken(href: string): { token: string | null; cleaned: string } {
  const url = new URL(href);
  const fragment = new URLSearchParams(url.hash.slice(1));
  const supplied = fragment.get("token") ?? url.searchParams.get("token");
  if (supplied === null) return { token: null, cleaned: href };
  fragment.delete("token");
  url.searchParams.delete("token");
  const hash = fragment.toString();
  return { token: supplied, cleaned: url.pathname + url.search + (hash ? `#${hash}` : "") };
}

/** Capture the token before anything is fetched. */
export function initToken(
  href: string = globalThis.location?.href ?? "",
  replace: (path: string) => void = (path) => globalThis.history?.replaceState(null, "", path),
  store: TokenStore | null = sessionStore(),
): void {
  const { token, cleaned } = href ? takeToken(href) : { token: null, cleaned: href };
  if (token !== null) {
    current = token.trim();
    replace(cleaned);
    try {
      store?.setItem(TOKEN_KEY, current);
    } catch {
      /* The in-memory copy carries this tab. */
    }
    return;
  }
  try {
    current = store?.getItem(TOKEN_KEY) ?? "";
  } catch {
    current = "";
  }
}

export function setToken(value: string, store: TokenStore | null = sessionStore()): void {
  current = value.trim();
  try {
    store?.setItem(TOKEN_KEY, current);
  } catch {
    /* Memory only. */
  }
}

export function hasToken(): boolean {
  return current !== "";
}

export function bearer(): Record<string, string> {
  return current ? { Authorization: `Bearer ${current}` } : {};
}

/** A URL of this service only: no other origin, scheme or embedded credentials.
 * The token never goes into a URL: requests send it as a header, event streams
 * are read with fetch, images are fetched as Blobs, and a download the browser
 * must open itself uses a single-use ticket (`/api/download-tickets`). */
export function serviceURL(path: string, origin: string = globalThis.location?.origin ?? "http://localhost"): URL {
  const url = new URL(path, origin);
  if (url.origin !== origin || !["http:", "https:"].includes(url.protocol) || url.username || url.password) {
    throw new Error("External service URL rejected");
  }
  return url;
}
