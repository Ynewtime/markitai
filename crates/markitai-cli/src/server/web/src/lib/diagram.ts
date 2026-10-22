// Mermaid diagrams in a rendered document: a ```mermaid block is drawn by the
// diagram renderer (/ui/mermaid.js, Pele), a separate same-origin file fetched
// only when a document holds one. Its SVG markup is treated as untrusted: it is
// read here into a tree, every element and attribute is checked against an
// allowlist (drawing elements only; nothing that loads, embeds or runs; links
// as the document's own links; ids and references kept inside the drawing),
// and the drawing is built from that tree with DOM calls. Its `style`
// declarations (Pele's colour variables, corner radii, the diagram's own
// colours) are set through the CSS object model, which the page's policy
// (style-src 'self') allows, after the same checks as any drawing value.
// A block that cannot be drawn stays the code block it was, with a note.
import type { Dict } from "../i18n/index.ts";
import { drawingReference, drawingValue, webLink } from "./markdown.ts";

export interface DiagramElement {
  tag: string;
  attributes: Array<[string, string]>;
  /** Declarations of its `style`, set through the CSS object model. */
  styles: Array<[string, string]>;
  children: DiagramNode[];
}
export type DiagramNode = DiagramElement | string;

const SVG_NS = "http://www.w3.org/2000/svg";
const XML_NS = "http://www.w3.org/XML/1998/namespace";

const ENTITIES: Record<string, string> = { lt: "<", gt: ">", amp: "&", quot: '"', apos: "'" };

/** Character data with its entities decoded; null for an entity XML does not define. */
function decode(text: string): string | null {
  let bad = false;
  const out = text.replace(/&(#x[\da-f]+|#\d+|[a-z]+);|&/gi, (whole, name: string | undefined) => {
    if (name === undefined) {
      bad = true;
      return "";
    }
    if (name.startsWith("#")) {
      const code = name[1] === "x" || name[1] === "X" ? parseInt(name.slice(2), 16) : parseInt(name.slice(1), 10);
      if (!Number.isInteger(code) || code < 1 || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) {
        bad = true;
        return "";
      }
      return String.fromCodePoint(code);
    }
    const value = ENTITIES[name];
    if (value === undefined) bad = true;
    return value ?? whole;
  });
  return bad ? null : out;
}

const TAG = /<([A-Za-z][\w:.-]*)((?:\s+[^\s=/>]+\s*=\s*(?:"[^"]*"|'[^']*'))*)\s*(\/?)>/y;
const CLOSE = /<\/([A-Za-z][\w:.-]*)\s*>/y;
const ATTRIBUTE = /\s+([^\s=/>]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;

/** SVG markup (one element, as Pele writes it) read into a tree, every attribute and
 * all text as written; null when it is not well formed. Nothing is checked here. */
export function parseSvg(markup: string): DiagramElement | null {
  const root: DiagramElement = { tag: "", attributes: [], styles: [], children: [] };
  const stack: DiagramElement[] = [root];
  let at = 0;
  while (at < markup.length) {
    const parent = stack[stack.length - 1] as DiagramElement;
    const lt = markup.indexOf("<", at);
    if (lt !== at) {
      const end = lt < 0 ? markup.length : lt;
      const text = decode(markup.slice(at, end));
      if (text === null) return null;
      if (text) parent.children.push(text);
      at = end;
      continue;
    }
    if (markup.startsWith("<!--", at)) {
      const end = markup.indexOf("-->", at + 4);
      if (end < 0) return null;
      at = end + 3;
      continue;
    }
    if (markup.startsWith("<?", at)) {
      const end = markup.indexOf("?>", at + 2);
      if (end < 0) return null;
      at = end + 2;
      continue;
    }
    CLOSE.lastIndex = at;
    const close = CLOSE.exec(markup);
    if (close) {
      if (stack.length < 2 || parent.tag !== close[1]) return null;
      stack.pop();
      at = CLOSE.lastIndex;
      continue;
    }
    TAG.lastIndex = at;
    const open = TAG.exec(markup);
    // A declaration, CDATA or anything else this reader does not know ends the reading.
    if (!open) return null;
    const element: DiagramElement = { tag: open[1] as string, attributes: [], styles: [], children: [] };
    for (const attribute of (open[2] ?? "").matchAll(ATTRIBUTE)) {
      const value = decode(attribute[2] ?? attribute[3] ?? "");
      if (value === null) return null;
      element.attributes.push([attribute[1] as string, value]);
    }
    parent.children.push(element);
    if (!open[3]) stack.push(element);
    at = TAG.lastIndex;
  }
  const elements = root.children.filter((child): child is DiagramElement => typeof child !== "string");
  const stray = root.children.some((child) => typeof child === "string" && child.trim() !== "");
  return stack.length === 1 && elements.length === 1 && !stray ? (elements[0] as DiagramElement) : null;
}

/** Drawing elements a diagram may hold: nothing that loads (image, feImage), embeds a
 * document (foreignObject), runs (script), styles the page (style) or animates. */
const TAGS = new Set([
  "svg", "g", "defs", "symbol", "use", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon",
  "text", "tspan", "textPath", "title", "desc", "a", "marker", "clipPath", "mask", "pattern",
  "linearGradient", "radialGradient", "stop",
]);

/** Attributes of those elements: geometry, paint, type and accessibility. */
const ATTRIBUTES = new Set([
  "viewBox", "preserveAspectRatio", "width", "height", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
  "fx", "fy", "d", "points", "transform", "pathLength", "dx", "dy", "rotate", "textLength", "lengthAdjust", "startOffset",
  "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-opacity", "stroke-dasharray", "stroke-dashoffset",
  "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "opacity", "color", "visibility", "display", "paint-order",
  "vector-effect", "clip-path", "clip-rule", "mask", "marker-start", "marker-mid", "marker-end",
  "font-family", "font-size", "font-weight", "font-style", "font-variant", "letter-spacing", "word-spacing",
  "text-anchor", "text-decoration", "dominant-baseline", "alignment-baseline", "baseline-shift", "xml:space",
  "markerWidth", "markerHeight", "markerUnits", "refX", "refY", "orient", "clipPathUnits", "maskUnits",
  "maskContentUnits", "patternUnits", "patternContentUnits", "patternTransform", "gradientUnits", "gradientTransform",
  "spreadMethod", "offset", "stop-color", "stop-opacity",
  "role", "aria-label", "aria-roledescription", "aria-hidden", "aria-labelledby", "aria-describedby",
]);

/** Properties a `style` may set: Pele's own variables (`--_bg` …), and drawing and type properties. */
const STYLE_PROPERTIES = new Set([
  "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-opacity", "stroke-dasharray", "stroke-dashoffset",
  "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "opacity", "color", "visibility", "display", "paint-order",
  "rx", "ry", "max-width", "height", "font", "font-family", "font-size", "font-weight", "font-style", "font-variant",
  "letter-spacing", "word-spacing", "text-transform", "text-anchor", "text-decoration", "dominant-baseline", "direction",
  "stop-color", "stop-opacity",
]);
const PRIVATE_VARIABLE = /^--_[a-z\d]{1,8}$/;

/** A declaration's value when it is a plain value (no escapes, blocks, comments or
 * rules) that loads and runs nothing; `url(#id)` is pinned within the drawing. */
function styleValue(value: string, ids: ReadonlyMap<string, string>): string | null {
  if (!value || /[\\{}<>@;]|\/\*|[\u0000-\u001f\u007f]/.test(value)) return null;
  return drawingValue(value, ids);
}

/** The safe declarations of a `style` attribute. */
export function diagramStyles(style: string, ids: ReadonlyMap<string, string>): Array<[string, string]> {
  const result: Array<[string, string]> = [];
  for (const declaration of style.split(";")) {
    const colon = declaration.indexOf(":");
    if (colon < 0) continue;
    const raw = declaration.slice(0, colon).trim();
    const name = raw.startsWith("--") ? raw : raw.toLowerCase();
    if (!PRIVATE_VARIABLE.test(name) && !STYLE_PROPERTIES.has(name)) continue;
    const value = styleValue(declaration.slice(colon + 1).replace(/!\s*important\s*$/i, "").trim(), ids);
    if (value !== null) result.push([name, value]);
  }
  return result;
}

export interface CleanOptions {
  /** Ids are kept only when they start with it (the prefix Pele was given). */
  prefix: string;
  /** Keep the diagram's links (else they become plain groups). */
  links: boolean;
}

/** A parsed diagram reduced to the allowlist: unknown elements are dropped with
 * everything inside them, unknown attributes dropped, classes kept only when
 * they are Pele's (`pele`, `pele-…`, so a diagram cannot borrow the page's),
 * ids kept only when unique and under `prefix`, references (`href` of `use`,
 * `url(#…)`, `aria-labelledby`) only to ids of this drawing, links only to web
 * addresses (opened in a new tab without a referrer). Null when the root is not
 * an `svg`. */
export function cleanDiagram(root: DiagramElement, { prefix, links }: CleanOptions): DiagramElement | null {
  if (root.tag !== "svg") return null;
  const ids = new Map<string, string>();
  const repeated = new Set<string>();
  const collect = (element: DiagramElement) => {
    if (!TAGS.has(element.tag)) return;
    for (const [name, value] of element.attributes) {
      if (name !== "id") continue;
      if (ids.has(value)) repeated.add(value);
      else if (value.startsWith(prefix) && /^[\w-]+$/.test(value)) ids.set(value, value);
    }
    for (const child of element.children) if (typeof child !== "string") collect(child);
  };
  collect(root);
  for (const id of repeated) ids.delete(id);

  const clean = (element: DiagramElement): DiagramElement | null => {
    if (!TAGS.has(element.tag)) return null;
    const attributes: Array<[string, string]> = [];
    let styles: Array<[string, string]> = [];
    let link: string | null = null;
    for (const [name, value] of element.attributes) {
      if (name === "style") styles = diagramStyles(value, ids);
      else if (name === "id") {
        if (ids.has(value)) attributes.push(["id", value]);
      } else if (name === "class") {
        const kept = value.split(/\s+/).filter((token) => /^pele(?:-[\w-]+)?$/.test(token));
        if (kept.length) attributes.push(["class", kept.join(" ")]);
      } else if (name === "href" || name === "xlink:href") {
        if (element.tag === "a") link = links ? webLink(value) : null;
        else {
          const pinned = drawingReference(value, ids);
          if (pinned !== null) attributes.push(["href", pinned]);
        }
      } else if (name === "aria-labelledby" || name === "aria-describedby") {
        const kept = value.split(/\s+/).filter((id) => ids.has(id));
        if (kept.length) attributes.push([name, kept.join(" ")]);
      } else if (ATTRIBUTES.has(name)) {
        const pinned = drawingValue(value, ids);
        if (pinned !== null) attributes.push([name, pinned]);
      }
    }
    if (element.tag === "use" && !attributes.some(([name]) => name === "href")) return null;
    if (link !== null) attributes.push(["href", link], ["target", "_blank"], ["rel", "noopener noreferrer"], ["referrerpolicy", "no-referrer"]);
    const children: DiagramNode[] = [];
    for (const child of element.children) {
      if (typeof child === "string") children.push(child);
      else {
        const kept = clean(child);
        if (kept) children.push(kept);
      }
    }
    return { tag: element.tag === "a" && link === null ? "g" : element.tag, attributes, styles, children };
  };
  return clean(root);
}

/** Pele's SVG markup as a checked tree, or null when it cannot be read. */
export function readDiagram(markup: string, options: CleanOptions): DiagramElement | null {
  const parsed = parseSvg(markup);
  return parsed && cleanDiagram(parsed, options);
}

/** Build a checked tree as live SVG elements of `doc`: text as text nodes, styles through the CSS object model. */
export function buildDiagram(doc: Document, element: DiagramElement): SVGElement {
  const node = doc.createElementNS(SVG_NS, element.tag) as SVGElement;
  for (const [name, value] of element.attributes) {
    if (name === "xml:space") node.setAttributeNS(XML_NS, name, value);
    else node.setAttribute(name, value);
  }
  for (const [name, value] of element.styles) node.style.setProperty(name, value);
  for (const child of element.children) node.append(typeof child === "string" ? doc.createTextNode(child) : buildDiagram(doc, child));
  return node;
}

type Renderer = typeof import("/ui/mermaid.js");
let renderer: Promise<Renderer> | null = null;

function loadRenderer(): Promise<Renderer> {
  renderer ??= import("/ui/mermaid.js");
  renderer.catch(() => {
    renderer = null;
  });
  return renderer;
}

let diagrams = 0;

/** The note under a block that was not drawn. */
export function diagramNote(t: Dict, code: string, line: number): string {
  if (code === "syntax" && line > 0) return t.diagramSyntax(line);
  if (code === "limit") return t.diagramTooLarge;
  if (code === "unsupported-diagram") return t.diagramUnsupported;
  return t.diagramFailed;
}

const BLOCKS = "pre > code.language-mermaid";

/** Draw the Mermaid blocks below `root`, one at a time; a document without any never loads the renderer. */
export async function drawDiagrams(root: HTMLElement, t: Dict, { links = true }: { links?: boolean } = {}): Promise<void> {
  const codes = [...root.querySelectorAll<HTMLElement>(BLOCKS)].filter((code) => !code.parentElement?.hasAttribute("data-diagram"));
  if (!codes.length) return;
  for (const code of codes) code.parentElement?.setAttribute("data-diagram", "");
  const doc = root.ownerDocument;
  const style = doc.defaultView?.getComputedStyle(root);
  const fontFamily = style?.fontFamily || "sans-serif";
  let module: Renderer | null = null;
  try {
    module = await loadRenderer();
  } catch {
    module = null;
  }
  for (const code of codes) {
    const pre = code.parentElement as HTMLElement;
    const prefix = `md-diagram-${++diagrams}-`;
    // The drawing gets the code block's inner width (the frame keeps its padding); a chunk not laid out yet has none.
    const width = pre.clientWidth ? pre.clientWidth - 32 : (root.clientWidth || 760) - 34;
    const drawn = module?.drawDiagram(code.textContent ?? "", { fontFamily, fontSize: 14, maxWidth: Math.max(240, width), idPrefix: prefix, links }) ?? { ok: false, code: "failed", line: 0 };
    const tree = drawn.ok ? readDiagram(drawn.svg, { prefix, links }) : null;
    if (tree) {
      const figure = doc.createElement("figure");
      figure.className = "md-diagram";
      figure.append(buildDiagram(doc, tree));
      pre.replaceWith(figure);
    } else {
      const note = doc.createElement("p");
      note.className = "diagram-note";
      note.textContent = drawn.ok ? t.diagramFailed : diagramNote(t, drawn.code, drawn.line);
      pre.classList.add("has-diagram-note");
      pre.after(note);
    }
    // A large diagram takes a moment to lay out: the page answers between diagrams.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}
