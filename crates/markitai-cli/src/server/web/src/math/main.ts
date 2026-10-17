// Mathematics for the preview, built as its own file (`/ui/math.js`, with the
// stylesheet `/ui/math.css` and the KaTeX fonts) and loaded only when a
// rendered result holds a formula. KaTeX builds its output as DOM nodes, so
// the page's policy (no inline style markup) is never in play, and the
// formula's text is read from the node, never reparsed as HTML.
import katex from "katex";

/** Draw every untouched `.math` element below `root`: its text is `$…$` or `$$…$$`. */
export function typesetAll(root: ParentNode): void {
  for (const node of root.querySelectorAll<HTMLElement>(".math:not([data-typeset])")) {
    node.setAttribute("data-typeset", "");
    const source = node.textContent ?? "";
    const display = source.startsWith("$$");
    const tex = source.slice(display ? 2 : 1, source.length - (display ? 2 : 1)).trim();
    try {
      // Drawn aside, then swapped in: a formula that does not parse leaves its source as it was.
      const drawn = document.createElement("span");
      katex.render(tex, drawn, { displayMode: display, throwOnError: true, strict: "ignore", trust: false, maxExpand: 1000 });
      node.replaceChildren(...drawn.childNodes);
      node.classList.add("is-typeset");
    } catch {
      node.classList.add("is-invalid");
    }
  }
}
