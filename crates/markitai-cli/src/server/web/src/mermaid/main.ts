// Mermaid diagrams for the preview, built as its own file (`/ui/mermaid.js`)
// and loaded only when a rendered result holds a ```mermaid block. Pele
// (https://pele.run) lays the diagram out and returns SVG markup; that markup
// is never inserted as it is: lib/diagram.ts rebuilds it element by element
// from an allowlist, so the page's policy (no inline style markup, no loads)
// holds whatever the diagram says.
import { PeleError, render } from "pele";

export type Drawn = { ok: true; svg: string } | { ok: false; code: string; line: number };

export interface DrawOptions {
  /** The label font, as the page draws it, so labels fit the boxes measured for them. */
  fontFamily: string;
  fontSize: number;
  /** Width of the column, in pixels: a narrow one gets a narrower (often vertical) layout. */
  maxWidth: number;
  /** Ids in the SVG start with it (letters, digits, `-`, `_`). */
  idPrefix: string;
  links: boolean;
}

/** Limits (characters of source and of SVG, edges of a flowchart) beyond which a diagram is not drawn. */
export const LIMITS = { source: 50_000, output: 2_000_000, edges: 2_000 };

/** One diagram drawn as SVG markup, or why it was not. Never throws for the diagram's own faults. */
export function drawDiagram(source: string, options: DrawOptions): Drawn {
  try {
    const draw = (directionBreakpoint?: number) =>
      render(source, {
        ...(directionBreakpoint === undefined ? {} : { directionBreakpoint }),
        fontFamily: options.fontFamily,
        fontSize: options.fontSize,
        maxWidth: options.maxWidth,
        idPrefix: options.idPrefix,
        limit: LIMITS.source,
        outputLimit: LIMITS.output,
        maxEdges: LIMITS.edges,
        // Pictures in a diagram would load from wherever it says: none are drawn.
        images: false,
        links: options.links,
        linkSchemes: ["http", "https", "mailto"],
        linkRel: "noopener noreferrer",
      });
    let drawn = draw();
    // Wider than the column, it would be shrunk (a long left-to-right flowchart's labels to 10 px or
    // less): the vertical layout Pele uses on narrow screens is kept when it is drawn larger.
    if (drawn.width > options.maxWidth) {
      const upright = draw(options.maxWidth + 1);
      if (upright.width < drawn.width) drawn = upright;
    }
    return { ok: true, svg: drawn.svg };
  } catch (error) {
    // An error at the end of the text is on the line after the last one: it is named as the last.
    if (error instanceof PeleError) return { ok: false, code: error.code, line: Math.min(error.line, source.trimEnd().split("\n").length) };
    return { ok: false, code: "failed", line: 0 };
  }
}
