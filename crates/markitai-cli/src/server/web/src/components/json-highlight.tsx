// The coloured layer under Edit config's textarea: the same text, drawn as JSON
// pieces, while the textarea above it (transparent text) stays the one control the
// keyboard, screen readers and the line/column check use. The caret is the page's
// own (lib/caret.ts): a font-high bar centred on its line, the same in every browser;
// the textarea's caret shows again without scripts and in forced colours.
import { useLayoutEffect, useRef } from "preact/hooks";
import type { RefObject } from "preact";
import { caretHeight, caretTop, lineColumn } from "../lib/caret.ts";
import { jsonTokens } from "../lib/json-tokens.ts";

/** Above this size the text is shown plain: colouring it on every keystroke would lag. */
const MAX_HIGHLIGHT = 256 * 1024;

/** The layer's text nodes with the text offset each starts at, for finding a caret's place. */
type TextIndex = { starts: number[]; nodes: Text[] };

function textIndex(root: Node): TextIndex {
  const index: TextIndex = { starts: [], nodes: [] };
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let offset = 0;
  for (let node = walker.nextNode() as Text | null; node; node = walker.nextNode() as Text | null) {
    index.starts.push(offset);
    index.nodes.push(node);
    offset += node.data.length;
  }
  return index;
}

/** The text node holding the character at `offset`, and the offset in it. */
function nodeAt(index: TextIndex, offset: number): { node: Text; at: number } | null {
  let low = 0;
  let high = index.starts.length - 1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if ((index.starts[middle] ?? 0) <= offset) low = middle;
    else high = middle - 1;
  }
  const node = index.nodes[low];
  return node ? { node, at: offset - (index.starts[low] ?? 0) } : null;
}

export function JsonHighlight({ text, editor }: { text: string; editor: RefObject<HTMLTextAreaElement | null> }) {
  const layer = useRef<HTMLPreElement>(null);
  const caret = useRef<HTMLSpanElement>(null);
  const index = useRef<TextIndex>({ starts: [], nodes: [] });
  const place = useRef<() => void>(() => undefined);
  useLayoutEffect(() => {
    const target = layer.current;
    if (!target) return;
    const code = document.createElement("code");
    if (text.length > MAX_HIGHLIGHT) code.textContent = text;
    else
      for (const [kind, piece] of jsonTokens(text)) {
        if (kind === null) {
          code.append(piece);
          continue;
        }
        const span = document.createElement("span");
        span.className = `json-${kind}`;
        span.textContent = piece;
        code.append(span);
      }
    // A final line break needs a character after it to take its line's height.
    code.append("\n ");
    target.replaceChildren(code);
    index.current = textIndex(code);
    const area = editor.current;
    if (area) {
      target.scrollTop = area.scrollTop;
      target.scrollLeft = area.scrollLeft;
    }
    place.current();
  }, [text]);
  useLayoutEffect(() => {
    const area = editor.current;
    const target = layer.current;
    const bar = caret.current;
    if (!area || !target || !bar) return;
    const host = area.parentElement;
    host?.classList.add("has-caret");
    // The caret: shown while the textarea has focus and nothing is selected, at the place of the
    // character before it (its right edge) or the line's start, centred on its line.
    let frame = 0;
    const draw = () => {
      frame = 0;
      const start = area.selectionStart;
      if (document.activeElement !== area || start !== area.selectionEnd) {
        bar.hidden = true;
        return;
      }
      const style = getComputedStyle(area);
      const fontSize = parseFloat(style.fontSize) || 12;
      const lineHeight = parseFloat(style.lineHeight) || fontSize * 1.65;
      const paddingTop = parseFloat(style.paddingTop) || 0;
      const paddingLeft = parseFloat(style.paddingLeft) || 0;
      const { line, column } = lineColumn(area.value, start);
      const height = caretHeight(fontSize, lineHeight);
      let x = paddingLeft - area.scrollLeft;
      let y = caretTop(line, { paddingTop, lineHeight, fontSize }) - area.scrollTop;
      // A character on the caret's line (the one before it, else the one after) gives its place:
      // its right or left edge, and the text's own box on that line (fonts sit their text a little
      // off the line's middle), the caret centred on that box.
      const before = column > 0;
      const next = area.value[start];
      const found = before ? nodeAt(index.current, start - 1) : next !== undefined && next !== "\n" ? nodeAt(index.current, start) : null;
      if (found) {
        const range = document.createRange();
        range.setStart(found.node, found.at);
        range.setEnd(found.node, Math.min(found.at + 1, found.node.data.length));
        const box = range.getBoundingClientRect();
        const origin = target.getBoundingClientRect();
        if (box.height > 0) {
          x = (before ? box.right : box.left) - origin.left;
          y = box.top - origin.top + (box.height - height) / 2;
        }
      }
      const visible = y + height > 0 && y < area.clientHeight && x >= paddingLeft - 1 && x <= area.clientWidth;
      bar.hidden = !visible;
      bar.style.transform = `translate(${area.offsetLeft + x}px, ${area.offsetTop + y}px)`;
      bar.style.height = `${height}px`;
      // a moving caret stays lit: its blink starts again
      bar.style.animation = "none";
      void bar.offsetWidth;
      bar.style.animation = "";
    };
    const later = () => {
      if (!frame) frame = requestAnimationFrame(draw);
    };
    place.current = later;
    const follow = () => {
      target.scrollTop = area.scrollTop;
      target.scrollLeft = area.scrollLeft;
      later();
    };
    const selection = () => {
      if (document.activeElement === area) later();
    };
    area.addEventListener("scroll", follow, { passive: true });
    for (const name of ["input", "keydown", "keyup", "pointerup", "focus", "blur", "select"]) area.addEventListener(name, later);
    document.addEventListener("selectionchange", selection);
    window.addEventListener("resize", later);
    later();
    return () => {
      cancelAnimationFrame(frame);
      host?.classList.remove("has-caret");
      area.removeEventListener("scroll", follow);
      for (const name of ["input", "keydown", "keyup", "pointerup", "focus", "blur", "select"]) area.removeEventListener(name, later);
      document.removeEventListener("selectionchange", selection);
      window.removeEventListener("resize", later);
      place.current = () => undefined;
    };
  }, [editor.current]);
  return (
    <>
      <pre ref={layer} class="config-highlight" aria-hidden="true" />
      <span ref={caret} class="config-caret" aria-hidden="true" hidden />
    </>
  );
}
