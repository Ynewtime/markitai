// Results read for the inspector and the preview, kept for this page. A retry
// or enhancement keeps an item's identity but replaces its output, so the key
// carries what changes with it. The inspector asks for an excerpt; the preview
// for the whole document, which then also serves later excerpts.
import { fetchResult } from "../api/client.ts";
import type { ItemResult } from "../api/types.ts";
import type { SessionItem } from "./session.ts";

/** Bytes the inspector asks for. */
export const EXCERPT_BYTES = 16 * 1024;
/** Results kept at most; the oldest go first. */
const KEEP = 24;

const whole = new Map<string, ItemResult>();
const excerpts = new Map<string, ItemResult>();
/** Requests in flight, so two views asking for one result share it. */
const pending = new Map<string, Promise<ItemResult>>();

function once(key: string, load: () => Promise<ItemResult>): Promise<ItemResult> {
  const known = pending.get(key);
  if (known) return known;
  const request = load().finally(() => pending.delete(key));
  pending.set(key, request);
  return request;
}

export const resultKey = (item: Pick<SessionItem, "key" | "finishedAt" | "llmEnhanced" | "operation">): string =>
  `${item.key}|${item.finishedAt ?? ""}|${item.llmEnhanced}|${item.operation}`;

function remember(store: Map<string, ItemResult>, key: string, value: ItemResult): void {
  store.delete(key);
  store.set(key, value);
  while (store.size > KEEP) store.delete(store.keys().next().value as string);
}

export function cachedResult(key: string): ItemResult | null {
  return whole.get(key) ?? null;
}

export function cachedExcerpt(key: string): ItemResult | null {
  return whole.get(key) ?? excerpts.get(key) ?? null;
}

export function loadResult(item: SessionItem): Promise<ItemResult> {
  const key = resultKey(item);
  const known = whole.get(key);
  if (known) return Promise.resolve(known);
  return once(`whole|${key}`, async () => {
    const value = await fetchResult(item.jobId, item.itemId);
    remember(whole, key, value);
    return value;
  });
}

/** The start of a result. A backend without excerpts sends everything, which is then kept whole. */
export function loadExcerpt(item: SessionItem, supported: boolean): Promise<ItemResult> {
  const key = resultKey(item);
  const known = cachedExcerpt(key);
  if (known) return Promise.resolve(known);
  return once(`excerpt|${key}`, async () => {
    const value = await fetchResult(item.jobId, item.itemId, supported ? EXCERPT_BYTES : undefined);
    remember(value.truncated ? excerpts : whole, key, value);
    return value;
  });
}

/** Lines of a text as the server counts `markdown_lines`: its line breaks, plus a last line without one. */
export const lineCount = (text: string): number => (text === "" ? 0 : text.split("\n").length - (text.endsWith("\n") ? 1 : 0));

/** Lines Details' Source shows. */
export const START_LINES = 12;

export interface SourceStart {
  /** The first lines, those of the front matter (from its `---` to its closing line) marked. */
  lines: { text: string; front: boolean }[];
  /** The whole result's lines; null for an excerpt whose server did not count them. */
  total: number | null;
}

/** The start of a result as Details' Source shows it. */
export function sourceStart(result: Pick<ItemResult, "markdown" | "truncated" | "markdown_lines">, count = START_LINES): SourceStart {
  const total = result.markdown_lines ?? (result.truncated ? null : lineCount(result.markdown));
  const head = result.markdown.split(/\r?\n/, Math.min(count, total ?? count));
  const opens = head[0]?.replace(/^\uFEFF/, "").trimEnd() === "---";
  const close = opens ? head.findIndex((line, index) => index > 0 && /^(---|\.\.\.)\s*$/.test(line)) : -1;
  // a front matter that closes after the shown lines covers all of them
  const last = opens && close === -1 ? head.length : close;
  return { lines: head.map((text, index) => ({ text, front: index <= last })), total };
}
