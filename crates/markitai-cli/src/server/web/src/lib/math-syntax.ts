// Mathematics in Markdown: `$…$` inline and `$$…$$` as a display, as converted
// pages and most Markdown tools write them. The sanitized output holds each
// formula as its own source text inside a `.math` element (`$…$` or `$$…$$`),
// which is also what shows if the renderer never loads; `typeset` in math.ts
// draws them. Rules as in Pandoc, so prices do not turn into formulas: an
// inline formula's `$` is not followed or preceded by white space, and its
// closing `$` is not followed by a digit.

/** An inline formula, at the start of the text. */
const INLINE = /^(\$\$(?!\$)[\s\S]+?\$\$|\$(?![\s$])(?:\\[\s\S]|[^$\\`\n])+?(?<![\s\\])\$(?!\d))/;
/** A display on lines of its own, opened and closed with `$$` (may hold blank lines). */
const BLOCK = /^ {0,3}\$\$[ \t]*\n?([\s\S]+?)\n? {0,3}\$\$[ \t]*(?:\n|$)/;
/** Formulas in a line of text, for a heading's label. */
const ANYWHERE = /(?<!\\)(\$\$(?!\$)[\s\S]+?\$\$|\$(?![\s$])(?:\\[\s\S]|[^$\\`\n])+?(?<![\s\\])\$(?!\d))/g;

const escapeHtml = (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

interface MathToken {
  type: string;
  raw: string;
  text: string;
}

/** marked extensions: a display block, and a formula inside a line. */
export const mathExtensions = [
  {
    name: "mathBlock",
    level: "block" as const,
    start: (source: string) => source.search(/(?:^|\n) {0,3}\$\$/),
    tokenizer: (source: string): MathToken | undefined => {
      const match = BLOCK.exec(source);
      return match ? { type: "mathBlock", raw: match[0], text: `$$${(match[1] ?? "").trim()}$$` } : undefined;
    },
    renderer: (token: MathToken) => `<div class="math math-display">${escapeHtml(token.text)}</div>\n`,
  },
  {
    name: "mathInline",
    level: "inline" as const,
    start: (source: string) => source.indexOf("$"),
    tokenizer: (source: string): MathToken | undefined => {
      const match = INLINE.exec(source);
      return match ? { type: "mathInline", raw: match[0], text: match[0] } : undefined;
    },
    renderer: (token: MathToken) => `<span class="${token.text.startsWith("$$") ? "math math-display" : "math"}">${escapeHtml(token.text)}</span>`,
  },
];

/** A piece of a line: plain text, or a formula with its `$` signs. */
export interface MathPiece {
  text: string;
  math: boolean;
}

/** A heading's label cut into text and formulas, for the contents. */
export function splitMath(label: string): MathPiece[] {
  const pieces: MathPiece[] = [];
  let from = 0;
  for (const match of label.matchAll(ANYWHERE)) {
    const at = match.index ?? 0;
    if (at > from) pieces.push({ text: label.slice(from, at), math: false });
    pieces.push({ text: match[0], math: true });
    from = at + match[0].length;
  }
  if (from < label.length) pieces.push({ text: label.slice(from), math: false });
  return pieces;
}
