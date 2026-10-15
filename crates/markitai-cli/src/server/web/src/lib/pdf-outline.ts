// The PDF viewer's outline panel: the PDF's own bookmarks (contract v1.17
// `…/original/outline`), else the converted Markdown's headings with the page
// each sits on (from its `<!-- Page number: N -->` markers, else from where
// Compare found them on the pages), else page thumbnails. Which entries are
// folded, and which one holds the page being read.

import { fetchCapabilities, fetchOriginalOutline } from "../api/client.ts";
import type { PdfOutline } from "../api/types.ts";

export interface OutlineItem {
  title: string;
  /** 1-based. */
  page: number;
  /** 1-based; a child may sit more than one level below its parent. */
  level: number;
}

export type OutlineSource = "pdf" | "headings" | "pages";

/** Entries the panel lists at most, and characters a title keeps (the server's bounds). */
export const OUTLINE_MAX = 2000;
const TITLE_MAX = 300;

const MARKER = /<!--\s*Page number:\s*(\d+)\s*-->/;
const FENCE = /^ {0,3}(`{3,}|~{3,})/;
const ATX = /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/;
const SETEXT = /^ {0,3}(=+|-+)[ \t]*$/;

/** A heading's words without Markdown's inline marks: links and pictures as their text, code, emphasis and tags dropped. */
export function plainHeading(text: string): string {
  return text
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/!?\[([^\]]*)\]\[[^\]]*\]/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/`+([^`]*)`+/g, "$1")
    .replace(/(\*\*|__)(.+?)\1/g, "$2")
    .replace(/(^|[^\w*])[*_]([^*_]+)[*_](?=[^\w*]|$)/g, "$1$2")
    .replace(/\\([\\`*_{}[\]()#+\-.!|<>~])/g, "$1")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, TITLE_MAX);
}

/**
 * The Markdown's headings, each with the page its last page marker names (page 1 before the first
 * marker). `paged` is false when the Markdown has no markers: the pages are then not known.
 * Fenced code and front matter are skipped; levels start at 1 for the document's top heading level.
 */
export function markdownHeadings(markdown: string, pageCount: number): { items: OutlineItem[]; paged: boolean } {
  const lines = markdown.replace(/^﻿/, "").split(/\r\n|\r|\n/);
  const found: { title: string; page: number; depth: number }[] = [];
  let page = 1;
  let paged = false;
  let fence: string | null = null;
  let start = 0;
  // Front matter: a first line of `---` up to the next `---` or `...`.
  if (lines[0]?.trim() === "---") {
    const end = lines.findIndex((line, index) => index > 0 && (line.trim() === "---" || line.trim() === "..."));
    if (end > 0) start = end + 1;
  }
  for (let index = start; index < lines.length && found.length < OUTLINE_MAX; index++) {
    const line = lines[index] as string;
    const opened = FENCE.exec(line);
    if (fence !== null) {
      if (opened && (opened[1] as string)[0] === fence[0] && (opened[1] as string).length >= fence.length && line.trim() === opened[1]) fence = null;
      continue;
    }
    if (opened) {
      fence = opened[1] as string;
      continue;
    }
    const marker = MARKER.exec(line);
    if (marker && !/^ {4,}|^\t/.test(line)) {
      const number = Number(marker[1]);
      if (number >= 1) {
        page = pageCount > 0 ? Math.min(number, pageCount) : number;
        paged = true;
      }
      continue;
    }
    const atx = ATX.exec(line);
    if (atx) {
      const title = plainHeading(atx[2] ?? "");
      if (title) found.push({ title, page, depth: (atx[1] as string).length });
      continue;
    }
    // A setext heading: a paragraph line underlined with = or - (not a list, quote, table or rule).
    const next = lines[index + 1];
    const under = next !== undefined ? SETEXT.exec(next) : null;
    if (under && line.trim() && !/^ {0,3}([-*+>|]|\d+[.)])(\s|$)/.test(line) && !/^ {4,}|^\t/.test(line) && !(/^ {0,3}-+[ \t]*$/.test(line))) {
      const title = plainHeading(line);
      if (title) found.push({ title, page, depth: (under[1] as string)[0] === "=" ? 1 : 2 });
      index++;
    }
  }
  return { items: leveled(found), paged };
}

/** Headings Compare found on the pages (a Markdown without page markers): in document order, the ones it placed. */
export function placedHeadings(headings: readonly { title: string; depth: number; page: number | null }[]): OutlineItem[] {
  return leveled(
    headings.flatMap((heading) => {
      const title = plainHeading(heading.title);
      return heading.page !== null && heading.page >= 1 && title ? [{ title, page: heading.page, depth: heading.depth }] : [];
    }),
  ).slice(0, OUTLINE_MAX);
}

function leveled(found: readonly { title: string; page: number; depth: number }[]): OutlineItem[] {
  const top = found.reduce((least, heading) => Math.min(least, heading.depth), 6);
  return found.map((heading) => ({ title: heading.title, page: heading.page, level: heading.depth - top + 1 }));
}

/** A server's outline answer, checked: entries with a page from 1 to `pageCount` and a level of at least 1. */
export function outlineItems(answer: PdfOutline | null | undefined, pageCount: number): OutlineItem[] {
  if (!answer || !Array.isArray(answer.items)) return [];
  return answer.items
    .filter((item) => item && typeof item.title === "string" && Number.isInteger(item.page) && item.page >= 1 && Number.isInteger(item.level) && item.level >= 1)
    .slice(0, OUTLINE_MAX)
    .map((item) => ({ title: item.title.slice(0, TITLE_MAX), page: pageCount > 0 ? Math.min(item.page, pageCount) : item.page, level: Math.min(item.level, 64) }));
}

/** Which source the panel shows first: the PDF's bookmarks, else headings with pages, else thumbnails. */
export function outlineSource(bookmarks: readonly OutlineItem[], headings: readonly OutlineItem[]): OutlineSource {
  return bookmarks.length ? "pdf" : headings.length ? "headings" : "pages";
}

/** For each entry, the entry it is nested under (the nearest earlier one at a lower level), or -1. */
export function parents(items: readonly OutlineItem[]): number[] {
  const result: number[] = [];
  const stack: number[] = [];
  for (const [index, item] of items.entries()) {
    while (stack.length && (items[stack[stack.length - 1] as number] as OutlineItem).level >= item.level) stack.pop();
    result.push(stack.length ? (stack[stack.length - 1] as number) : -1);
    stack.push(index);
  }
  return result;
}

/** Entries that hold others. */
export function withChildren(items: readonly OutlineItem[]): Set<number> {
  return new Set(parents(items).filter((parent) => parent >= 0));
}

/** The entries shown while `folded` entries hide what they hold. */
export function shownEntries(items: readonly OutlineItem[], folded: ReadonlySet<number>): number[] {
  const up = parents(items);
  const hidden: boolean[] = [];
  const shown: number[] = [];
  for (let index = 0; index < items.length; index++) {
    const parent = up[index] as number;
    hidden[index] = parent >= 0 && (hidden[parent] === true || folded.has(parent));
    if (!hidden[index]) shown.push(index);
  }
  return shown;
}

/** A long outline opens with its top level only; a short one opens whole. */
export function firstFolded(items: readonly OutlineItem[]): Set<number> {
  return items.length > 60 ? withChildren(items) : new Set();
}

/** The entry that holds the page being read: the last one that starts on or before it. */
export function entryAt(items: readonly OutlineItem[], page: number): number {
  let found = -1;
  for (const [index, item] of items.entries()) if (item.page <= page) found = index;
  else if (found >= 0 && item.page > page) break;
  return found;
}

/** An entry's ancestors (to unfold them so the entry shows). */
export function ancestors(items: readonly OutlineItem[], index: number): number[] {
  const up = parents(items);
  const result: number[] = [];
  for (let at = up[index] ?? -1; at >= 0; at = up[at] ?? -1) result.push(at);
  return result;
}

let supported: Promise<boolean> | null = null;

/** Whether the backend serves outlines (`features.pdf_outline`), asked once per page load. */
export function outlineSupported(): Promise<boolean> {
  supported ??= fetchCapabilities().then(
    (capabilities) => capabilities.features?.pdf_outline === true,
    () => {
      supported = null;
      return false;
    },
  );
  return supported;
}

const outlines = new Map<string, Promise<PdfOutline | null>>();

/** An item's outline, read once per item (null when the backend has none to give). */
export function itemOutline(job: string, item: string): Promise<PdfOutline | null> {
  const key = `${job}\u0000${item}`;
  let found = outlines.get(key);
  if (!found) {
    found = outlineSupported().then((yes) => (yes ? fetchOriginalOutline(job, item) : null));
    found.catch(() => outlines.delete(key));
    outlines.set(key, found);
    while (outlines.size > 50) outlines.delete(outlines.keys().next().value as string);
  }
  return found;
}
