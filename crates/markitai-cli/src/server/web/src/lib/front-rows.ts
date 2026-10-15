// The rows of a document's properties table, from its front matter.
import type { FrontMatter } from "./frontmatter.ts";

export type FrontRow =
  | { key: string; kind: "text"; value: string }
  | { key: string; kind: "tags"; value: string[] }
  | { key: string; kind: "link"; value: string }
  | { key: string; kind: "time"; value: string };

const ISO_TIME = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/;
const WEB = /^https?:\/\/[^\s]+$/i;

/** Summary, tags, then the other keys in the document's order. The title is
 * left to the body's own heading (`cardFacts` keeps it only when the body has none). */
export function frontRows(front: FrontMatter): FrontRow[] {
  const rows: FrontRow[] = [];
  if (front.title) rows.push({ key: "title", kind: "text", value: front.title });
  if (front.description) rows.push({ key: "description", kind: "text", value: front.description });
  if (front.tags.length) rows.push({ key: "tags", kind: "tags", value: front.tags });
  for (const [key, value] of front.fields) {
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
