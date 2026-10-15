// The coloured layer under Edit config's textarea: the same text, drawn as JSON
// pieces, while the textarea above it (transparent text, visible caret) stays the
// one control the keyboard, screen readers and the line/column check use.
import { useLayoutEffect, useRef } from "preact/hooks";
import type { RefObject } from "preact";
import { jsonTokens } from "../lib/json-tokens.ts";

/** Above this size the text is shown plain: colouring it on every keystroke would lag. */
const MAX_HIGHLIGHT = 256 * 1024;

export function JsonHighlight({ text, editor }: { text: string; editor: RefObject<HTMLTextAreaElement> }) {
  const layer = useRef<HTMLPreElement>(null);
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
    const area = editor.current;
    if (area) {
      target.scrollTop = area.scrollTop;
      target.scrollLeft = area.scrollLeft;
    }
  }, [text]);
  useLayoutEffect(() => {
    const area = editor.current;
    const target = layer.current;
    if (!area || !target) return;
    const follow = () => {
      target.scrollTop = area.scrollTop;
      target.scrollLeft = area.scrollLeft;
    };
    area.addEventListener("scroll", follow, { passive: true });
    return () => area.removeEventListener("scroll", follow);
  }, [editor.current]);
  return <pre ref={layer} class="config-highlight" aria-hidden="true" />;
}
