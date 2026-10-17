// A line of text that may hold formulas (a heading in the contents): the text
// with each `$…$` drawn as a formula once the renderer has loaded, the source
// text meanwhile. The nodes are filled by hand, so the formulas the renderer
// builds are never fought over by the component's own updates.
import { useLayoutEffect, useRef } from "preact/hooks";
import { typeset } from "../lib/math.ts";
import { splitMath } from "../lib/math-syntax.ts";

export function MathText({ text, class: className }: { text: string; class?: string }) {
  const root = useRef<HTMLSpanElement>(null);
  const pieces = splitMath(text);
  const plain = !pieces.some((piece) => piece.math);
  useLayoutEffect(() => {
    const target = root.current;
    if (!target || plain) return;
    target.replaceChildren(
      ...pieces.map((piece) => {
        if (!piece.math) return document.createTextNode(piece.text);
        const formula = document.createElement("span");
        formula.className = "math";
        formula.textContent = piece.text;
        return formula;
      }),
    );
    void typeset(target);
  }, [text]);
  return (
    <span ref={root} class={className}>
      {plain ? text : null}
    </span>
  );
}
