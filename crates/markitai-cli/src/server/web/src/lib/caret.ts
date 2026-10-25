// Edit config draws its own caret (owner report, 2026-10-25): browsers draw a textarea's caret
// differently once the line is taller than the font (WebKit as tall as the line and level with the
// text's bottom, Blink as tall as the font), so over the coloured layer it sat low or high. The
// page's caret is one font-high bar centred on its line in every browser.

/** The line (0-based) and column a caret offset falls on. */
export function lineColumn(text: string, offset: number): { line: number; column: number; lineStart: number } {
  const before = text.slice(0, Math.max(0, Math.min(offset, text.length)));
  const lineStart = before.lastIndexOf("\n") + 1;
  let line = 0;
  for (let index = before.indexOf("\n"); index >= 0; index = before.indexOf("\n", index + 1)) line++;
  return { line, column: before.length - lineStart, lineStart };
}

/** A caret's height: the font's own (its em box and a little), never taller than the line. */
export function caretHeight(fontSize: number, lineHeight: number): number {
  return Math.min(lineHeight, Math.round(fontSize * 1.2));
}

/** Where line `line`'s caret starts, from the top of the text's box (its padding edge, unscrolled):
 * the line's own box, the caret centred in it. */
export function caretTop(line: number, metrics: { paddingTop: number; lineHeight: number; fontSize: number }): number {
  const height = caretHeight(metrics.fontSize, metrics.lineHeight);
  return metrics.paddingTop + line * metrics.lineHeight + (metrics.lineHeight - height) / 2;
}
