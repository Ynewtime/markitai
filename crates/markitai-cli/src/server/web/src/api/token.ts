// The service access token. It arrives in the URL fragment of the launch link
// (`#token=`; `?token=` is still accepted), is removed from the address bar at
// once and kept only for this tab's session. Loopback visitors need none.

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

/** A URL of this service only: no other origin, scheme or embedded credentials. */
export function serviceURL(path: string, origin: string = globalThis.location?.origin ?? "http://localhost"): URL {
  const url = new URL(path, origin);
  if (url.origin !== origin || !["http:", "https:"].includes(url.protocol) || url.username || url.password) {
    throw new Error("External service URL rejected");
  }
  return url;
}

/** For requests that cannot carry a header (EventSource, images): the token
 * rides as `?token=`, and only on this service's `/api/` URLs. */
export function withToken(path: string, origin?: string): string {
  const url = serviceURL(path, origin);
  if (!url.pathname.startsWith("/api/")) throw new Error("Tokens are restricted to service API URLs");
  if (current) url.searchParams.set("token", current);
  return url.pathname + url.search;
}
