// Formulas in a rendered document: `.math` elements (see math-syntax.ts) are
// drawn by the formula renderer, a separate same-origin file with its own
// stylesheet and fonts, fetched only when a document holds a formula.

let renderer: Promise<typeof import("/ui/math.js")> | null = null;

function loadRenderer(): Promise<typeof import("/ui/math.js")> {
  if (renderer === null) {
    if (!document.querySelector('link[data-math="true"]')) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "/ui/math.css";
      link.setAttribute("data-math", "true");
      document.head.append(link);
    }
    renderer = import("/ui/math.js");
    renderer.catch(() => {
      renderer = null;
    });
  }
  return renderer;
}

/** Draw the formulas below `root`; a document without any never loads the renderer. A failure leaves the source text. */
export async function typeset(root: ParentNode): Promise<void> {
  if (!root.querySelector(".math:not([data-typeset])")) return;
  try {
    (await loadRenderer()).typesetAll(root);
  } catch {
    // The formulas stay as the `$…$` source they were.
  }
}
