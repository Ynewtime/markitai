// The rows of a document's properties table, from its front matter. The table names
// markitai's own keys and values in the interface's language and leaves out what another
// row already says; the file itself keeps every key as written.
import type { Dict } from "../i18n/index.ts";
import type { FrontMatter } from "./frontmatter.ts";

export type FrontRow =
  | { key: string; kind: "text"; value: string }
  | { key: string; kind: "tags"; value: string[] }
  | { key: string; kind: "link"; value: string }
  | { key: string; kind: "time"; value: string };

const ISO_TIME = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/;
const WEB = /^https?:\/\/[^\s]+$/i;

/** The same address written twice (a readable and a percent-encoded form, a trailing slash). */
function sameUrl(a: string, b: string): boolean {
  const norm = (value: string) => {
    try {
      return new URL(value).href.replace(/\/$/, "");
    } catch {
      return value;
    }
  };
  return norm(a) === norm(b);
}

function hostOf(value: string | undefined): string | null {
  if (!value) return null;
  try {
    return new URL(value).hostname.toLowerCase();
  } catch {
    return null;
  }
}

/** What another row already says: the final address (`source_url`) or the canonical link when it is the
 * source's, the renderer when it is the fetch method, the domain when it is the source's host. An X
 * post's display name joins its `@handle` in the author row instead. */
function shownFields(fields: [string, string][]): [string, string][] {
  const get = (key: string) => fields.find(([name]) => name === key)?.[1];
  const source = get("source");
  const hosts = new Set([source, get("source_url")].map(hostOf).filter((host) => host !== null));
  const author = get("author");
  const name = get("author_name");
  const joined = author !== undefined && author.startsWith("@") && name !== undefined && name !== "";
  const out: [string, string][] = [];
  for (const [key, value] of fields) {
    if ((key === "source_url" || key === "canonical_url") && source !== undefined && sameUrl(value, source)) continue;
    if (key === "source_url" && out.some(([name, shown]) => name === "canonical_url" && sameUrl(shown, value))) continue;
    if (key === "renderer" && value === get("fetch_strategy")) continue;
    if (key === "domain" && hosts.has(value.toLowerCase())) continue;
    if (key === "author_name" && joined) continue;
    out.push(key === "author" && joined ? [key, `${name} (${value})`] : [key, value]);
  }
  return out;
}

/** markitai's own values by name: the fetch method and renderer as the URL option names them, the
 * other enumerations from the dictionary; anything else as written. */
export function frontValue(t: Dict, key: string, value: string): string {
  if (key === "fetch_strategy" || key === "renderer")
    return (
      ({ static: t.strategyStatic, playwright: t.strategyPlaywright, defuddle: t.strategyDefuddle, jina: t.strategyJina, cloudflare: t.strategyCloudflare }) as Record<string, string>
    )[value] ?? value;
  if (key === "ocr_path") return ({ vision: "Apple Vision", paddle: "PP-OCR" } as Record<string, string>)[value] ?? t.frontValues(value) ?? value;
  if (["content_profile", "type", "numbers_value_mode", "ocr_used"].includes(key)) return t.frontValues(value) ?? value;
  return value;
}

/** Summary, tags, then the other keys in the document's order. The title is
 * left to the body's own heading (`cardFacts` keeps it only when the body has none). */
export function frontRows(front: FrontMatter): FrontRow[] {
  const rows: FrontRow[] = [];
  if (front.title) rows.push({ key: "title", kind: "text", value: front.title });
  if (front.description) rows.push({ key: "description", kind: "text", value: front.description });
  if (front.tags.length) rows.push({ key: "tags", kind: "tags", value: front.tags });
  for (const [key, value] of shownFields(front.fields)) {
    if (WEB.test(value)) rows.push({ key, kind: "link", value });
    else if (ISO_TIME.test(value) && !Number.isNaN(Date.parse(value))) rows.push({ key, kind: "time", value });
    else rows.push({ key, kind: "text", value });
  }
  return rows;
}

/** What the folded line says: the tags, else the summary, else the first other value. */
export function frontLine(front: FrontMatter): { kind: "tags"; tags: string[] } | { kind: "text"; text: string } | null {
  if (front.tags.length) return { kind: "tags", tags: front.tags };
  if (front.description) return { kind: "text", text: front.description };
  const first = front.fields[0];
  return first ? { kind: "text", text: first[1] } : null;
}
