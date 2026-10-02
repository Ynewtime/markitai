// The URL field: one address per line. A bare domain such as `example.com/page`
// gets https://; the first unusable line is reported by itself.

export interface ParsedUrls {
  urls: string[];
  invalid: { value: string; reason: "badUrl" | "badScheme" } | null;
}

const SCHEME = /^[a-z][a-z\d+.-]*:/i;
const BARE_DOMAIN = /^[\w-]+(\.[\w-]+)*\.[a-z]{2,}(?:[:/?#]|$)/i;

export function parseUrls(text: string): ParsedUrls {
  const urls: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    const value = line.trim();
    if (!value) continue;
    const candidate = !SCHEME.test(value) && BARE_DOMAIN.test(value) ? `https://${value}` : value;
    let parsed: URL;
    try {
      parsed = new URL(candidate);
    } catch {
      return { urls, invalid: { value, reason: "badUrl" } };
    }
    if (!["http:", "https:"].includes(parsed.protocol)) return { urls, invalid: { value, reason: "badScheme" } };
    if (!parsed.hostname) return { urls, invalid: { value, reason: "badUrl" } };
    urls.push(candidate);
  }
  return { urls, invalid: null };
}
