// The contents of a rendered document: which headings are listed, at which
// level, and which entry holds the heading being read.
import type { Heading } from "./markdown.ts";

/** Headings a document needs before it gets contents. */
export const TOC_MIN = 4;
/** Levels shown below the document's top level. */
const TOC_DEPTH = 3;

export interface TocEntry {
  index: number;
  level: number;
  text: string;
}

/** The entries worth listing: the top three levels present, without empty headings. */
export function tocEntries(headings: readonly Heading[]): TocEntry[] {
  const named = headings.map((heading, index) => ({ ...heading, index })).filter((heading) => heading.text);
  if (named.length < TOC_MIN) return [];
  // A single top-level title (the book's name) does not make a level of its own.
  const depths = [...new Set(named.map((heading) => heading.depth))].sort((a, b) => a - b);
  const titleOnly = depths.length > 1 && named.filter((heading) => heading.depth === depths[0]).length === 1;
  const levels = (titleOnly ? depths.slice(1) : depths).slice(0, TOC_DEPTH);
  const entries = named
    .filter((heading) => levels.includes(heading.depth))
    .map((heading) => ({ index: heading.index, level: levels.indexOf(heading.depth), text: heading.text }));
  return entries.length >= TOC_MIN ? entries : [];
}

/** The entry that holds the heading being read: the last entry at or before it. */
export function entryAt(entries: readonly TocEntry[], heading: number): number {
  let found = -1;
  for (const [position, entry] of entries.entries()) {
    if (entry.index > heading) break;
    found = position;
  }
  return found;
}
