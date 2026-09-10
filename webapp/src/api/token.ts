/** Jupyter-style serve access token.
 *
 * The server prints a `#token=` URL at startup (fragment, so the credential
 * never reaches the server's access log); older banners used `?token=`, so
 * `initToken` accepts both, preferring the fragment. It captures the token
 * once into sessionStorage and scrubs both the hash and the query from the
 * address bar; API fetches then send it as `Authorization: Bearer` and
 * header-less browser requests (EventSource, download links, inline images)
 * append it back as `?token=`. Without a stored token everything is sent
 * bare — loopback sessions need none.
 */

const STORAGE_KEY = "markitai.serve.token";

/** Fallback when sessionStorage is unavailable (e.g. blocked storage). */
let memoryToken: string | null = null;

/** `#token=xxx`, `#token=xxx&other=1`, or a bare `#mk_…` fragment. The
 * `token=` prefix wins; a bare fragment is only a token when it carries the
 * server's `mk_` prefix, so ordinary anchors (`#section-2`) are left alone. */
function tokenFromHash(hash: string): string | null {
  const fragment = hash.startsWith("#") ? hash.slice(1) : hash;
  if (fragment === "") return null;
  const pair = /(?:^|&)token=([^&]*)/.exec(fragment);
  const raw = pair !== null ? pair[1] : /^mk_[\w-]+$/.test(fragment) ? fragment : undefined;
  if (raw === undefined) return null;
  let value = raw;
  try {
    value = decodeURIComponent(raw);
  } catch {
    /* keep the raw fragment when it is not valid percent-encoding */
  }
  return value === "" ? null : value;
}

/** Capture the boot token (fragment first, then `?token=`) and scrub the URL. */
export function initToken(): void {
  const url = new URL(window.location.href);
  const token = tokenFromHash(url.hash) ?? url.searchParams.get("token");
  if (token === null || token === "") return;
  memoryToken = token;
  try {
    sessionStorage.setItem(STORAGE_KEY, token);
  } catch {
    /* memoryToken carries this tab through */
  }
  url.hash = "";
  url.searchParams.delete("token");
  window.history.replaceState(window.history.state, "", url.toString());
}

export function getToken(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY) ?? memoryToken;
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
