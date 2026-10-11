// Safe rendering of converted Markdown. The vendored marked parser and DOMPurify
// sanitizer load on first use; the sanitized fragment is then rewritten so that
// images and attachments resolve only to files listed in the result itself.
import type { Artifact } from "../api/types.ts";

/** The leading YAML block; the rendered view leaves it to Source. */
export function splitFrontmatter(markdown: string): { frontmatter: string | null; body: string } {
  const match = /^---\n[\s\S]*?\n---(?=\n|$)/.exec(markdown);
  return match ? { frontmatter: match[0], body: markdown.slice(match[0].length) } : { frontmatter: null, body: markdown };
}

/** The exact base/LLM pair of one result; an ambiguous inventory has none. */
export function markdownPair(artifacts: Artifact[]): { base: string; llm: string } | null {
  const paths = new Set(artifacts.map((artifact) => artifact.relpath));
  const pairs = [...paths]
    .filter((path) => path.endsWith(".llm.md"))
    .map((llm) => ({ base: `${llm.slice(0, -".llm.md".length)}.md`, llm }))
    .filter((pair) => paths.has(pair.base));
  return pairs.length === 1 ? (pairs[0] ?? null) : null;
}

/** A reference inside the document resolved to a listed artifact path, or null.
 * Schemes, absolute paths and backslashes never resolve. */
export function artifactPath(target: string, documentPath: string, allowed: ReadonlySet<string>): string | null {
  if (/^[a-z][a-z\d+.-]*:/i.test(target) || target.startsWith("//") || target.startsWith("/") || target.includes("\\")) return null;
  let url: URL;
  try {
    url = new URL(target, `https://artifact.invalid/${documentPath.split("/").map(encodeURIComponent).join("/")}`);
  } catch {
    return null;
  }
  let path: string;
  try {
    path = decodeURIComponent(url.pathname.slice(1));
  } catch {
    return null;
  }
  return allowed.has(path) ? path : null;
}

const RASTER = /\.(png|jpe?g|gif|webp|avif)$/i;

export interface RenderContext {
  documentPath: string;
  artifacts: Artifact[];
  /** URL of an artifact for an <img>, or null when it must be fetched with the
   * header token (the image then waits in `data-artifact-src`; see loadArtifactImages). */
  imageURL: (path: string) => string | null;
  /** URL of an artifact for a download link (never carries the token). */
  fileURL: (path: string) => string;
  placeholder: (alt: string) => string;
  fallbackAlt: string;
  tableLabel: string;
  codeLabel: string;
}

interface Libraries {
  marked: { parse(markdown: string, options: { async: false; gfm: boolean }): string };
  purify: Purify;
}

interface AttributeHook {
  attrName: string;
  keepAttr: boolean;
}
interface Purify {
  sanitize(html: string, options: Record<string, unknown>): DocumentFragment;
  addHook(entry: "beforeSanitizeAttributes", hook: (node: Element) => void): void;
  addHook(entry: "uponSanitizeAttribute", hook: (node: Element, data: AttributeHook) => void): void;
  removeHooks(entry: string): void;
}

let libraries: Promise<Libraries> | null = null;

/** marked and DOMPurify are separate same-origin files, fetched the first time a preview opens. */
export function loadLibraries(): Promise<Libraries> {
  libraries ??= Promise.all([import("/ui/marked.js"), import("/ui/purify.js")]).then(([markedModule, purifyModule]) => ({
    marked: markedModule.marked,
    purify: purifyModule.default,
  }));
  libraries.catch(() => {
    libraries = null;
  });
  return libraries;
}

const FORBID_TAGS = [
  "style", "script", "iframe", "object", "embed", "form", "input", "button", "textarea", "select", "video", "audio", "source", "picture", "link", "meta", "base",
  // In drawings: anything that loads, embeds a document or animates attributes.
  "image", "feimage", "foreignobject", "animate", "animatecolor", "animatemotion", "animatetransform", "set", "mpath", "tref", "altglyph", "font-face-uri", "cursor",
];
// `id` is kept on drawing elements only (see `drawingIds`).
const FORBID_ATTR = ["style", "srcset", "poster", "ping", "target", "name"];
const SVG_NS = "http://www.w3.org/2000/svg";
const XLINK_NS = "http://www.w3.org/1999/xlink";

/** Parse, sanitize and rewrite. Raw HTML never reaches the live document. */
export async function renderMarkdown(markdown: string, context: RenderContext): Promise<DocumentFragment> {
  const { marked, purify } = await loadLibraries();
  const html = marked.parse(markdown, { async: false, gfm: true });
  // A drawing's paint often sits in `style`, which the page's policy never
  // applies: its drawing properties become attributes, then sanitized like
  // any other. An `id` survives only inside a drawing, where gradients and
  // clip paths need it; `pinDrawings` then gives it a prefix of its own.
  purify.addHook("beforeSanitizeAttributes", (node) => {
    if (node.namespaceURI !== SVG_NS) return;
    const style = node.getAttribute("style");
    if (!style) return;
    for (const [name, value] of presentationFromStyle(style)) if (!node.hasAttribute(name)) node.setAttribute(name, value);
  });
  purify.addHook("uponSanitizeAttribute", (node, data) => {
    if (data.attrName === "id" && node.namespaceURI !== SVG_NS) data.keepAttr = false;
  });
  let fragment: DocumentFragment;
  try {
    fragment = purify.sanitize(html, {
      RETURN_DOM_FRAGMENT: true,
      USE_PROFILES: { html: true, svg: true },
      ADD_TAGS: ["use"],
      FORBID_TAGS,
      FORBID_ATTR,
      ALLOW_DATA_ATTR: false,
      ALLOW_ARIA_ATTR: false,
    });
  } finally {
    purify.removeHooks("beforeSanitizeAttributes");
    purify.removeHooks("uponSanitizeAttribute");
  }
  rewrite(fragment, context);
  return fragment;
}

/** Drawing properties a `style` may set that SVG also takes as attributes. */
const PRESENTATION = new Set([
  "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-opacity", "stroke-dasharray", "stroke-dashoffset",
  "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "opacity", "color", "visibility", "display", "font-size",
  "font-family", "font-weight", "font-style", "letter-spacing", "word-spacing", "text-anchor", "text-decoration",
  "dominant-baseline", "stop-color", "stop-opacity", "clip-rule", "paint-order",
]);

/** A drawing element's `style` as attributes: drawing properties only, with
 * values that load or run nothing; a `url(#id)` paint stays for `pinDrawings`
 * to resolve within its drawing or drop. */
export function presentationFromStyle(style: string): Array<[string, string]> {
  const result: Array<[string, string]> = [];
  for (const declaration of style.split(";")) {
    const colon = declaration.indexOf(":");
    if (colon < 0) continue;
    const name = declaration.slice(0, colon).trim().toLowerCase();
    const value = declaration.slice(colon + 1).replace(/!\s*important\s*$/i, "").trim();
    const outside = value.replace(/url\(\s*(['"]?)#[\w.:-]+\1\s*\)/gi, "");
    if (!PRESENTATION.has(name) || !value || /url\s*\(/i.test(outside) || drawingValue(outside, new Map()) === null) continue;
    result.push([name, value]);
  }
  return result;
}

/** CSS functions that load, embed or run something. */
const UNSAFE_FUNCTION = /(?:image-set|image|cross-fade|element|src|expression|paint|attr|env)\s*\(/i;

/** An attribute value of a drawing with every `url(#id)` pointed at the
 * drawing's renamed element, or null when the value loads or runs anything:
 * a `url()` to anything but an element of the same drawing, an image or
 * script function, or a `javascript:` value. */
export function drawingValue(value: string, ids: ReadonlyMap<string, string>): string | null {
  if (UNSAFE_FUNCTION.test(value) || /^\s*(?:javascript|vbscript|data):/i.test(value)) return null;
  let safe = true;
  const pinned = value.replace(/url\(\s*(['"]?)([^'")]*)\1\s*\)/gi, (_, _quote: string, target: string) => {
    const id = target.trim().startsWith("#") ? ids.get(target.trim().slice(1)) : undefined;
    if (id === undefined) safe = false;
    return `url(#${id ?? ""})`;
  });
  return safe && !/url\s*\(/i.test(pinned.replace(/url\(#[^)]+\)/gi, "")) ? pinned : null;
}

/** A drawing reference (`href` of `use`, a gradient, a text path) pointed at
 * the renamed element of the same drawing, or null for anything else. */
export function drawingReference(value: string, ids: ReadonlyMap<string, string>): string | null {
  const target = value.trim();
  if (!target.startsWith("#")) return null;
  const id = ids.get(target.slice(1));
  return id === undefined ? null : `#${id}`;
}

let drawings = 0;

/** Every inline drawing references only itself: ids get a prefix of their
 * own drawing (two drawings cannot take each other's gradients, and no id can
 * shadow a name the page uses), `href`/`xlink:href` and `url()` resolve only
 * to elements of the same drawing, and a `use` that points elsewhere is
 * removed. Links (`a`) are left to the link pass, as in the text. */
function pinDrawings(fragment: DocumentFragment): void {
  for (const svg of [...fragment.querySelectorAll("svg")]) {
    if (svg.parentElement?.closest("svg")) continue;
    const prefix = `md-drawing-${++drawings}-`;
    const elements = [svg, ...svg.querySelectorAll("*")];
    const ids = new Map<string, string>();
    for (const element of elements) {
      const id = element.getAttribute("id");
      if (id === null) continue;
      if (element.namespaceURI !== SVG_NS || ids.has(id)) element.removeAttribute("id");
      else ids.set(id, `${prefix}${id}`);
    }
    for (const element of elements) {
      const id = element.getAttribute("id");
      if (id !== null) element.setAttribute("id", ids.get(id) ?? "");
      const linked = element.getAttributeNS(XLINK_NS, "href");
      element.removeAttributeNS(XLINK_NS, "href");
      const href = element.getAttribute("href") ?? linked;
      if (element.localName === "a") {
        if (href !== null) element.setAttribute("href", href);
      } else if (href !== null) {
        const pinned = drawingReference(href, ids);
        if (pinned === null) element.removeAttribute("href");
        else element.setAttribute("href", pinned);
      }
      if (element.localName === "use" && !element.hasAttribute("href")) {
        element.remove();
        continue;
      }
      for (const attribute of [...element.attributes]) {
        if (attribute.name === "href" || attribute.name === "id") continue;
        const value = drawingValue(attribute.value, ids);
        if (value === null) element.removeAttribute(attribute.name);
        else if (value !== attribute.value) element.setAttribute(attribute.name, value);
      }
    }
    svg.classList.add("md-drawing");
  }
}

/** Pin every reference to the result's own files; everything else is inert. */
export function rewrite(fragment: DocumentFragment, context: RenderContext): void {
  const doc = fragment.ownerDocument;
  const allowed = new Set(context.artifacts.map((artifact) => artifact.relpath));
  pinDrawings(fragment);
  for (const image of [...fragment.querySelectorAll("img")]) {
    const path = artifactPath(image.getAttribute("src") ?? "", context.documentPath, allowed);
    const alt = image.getAttribute("alt") || "";
    if (path && RASTER.test(path)) {
      const url = context.imageURL(path);
      if (url === null) {
        image.removeAttribute("src");
        image.setAttribute("data-artifact-src", path);
      } else image.setAttribute("src", url);
      image.setAttribute("loading", "lazy");
      image.setAttribute("referrerpolicy", "no-referrer");
      if (!alt) image.setAttribute("alt", context.fallbackAlt);
    } else {
      const span = doc.createElement("span");
      span.className = "img-blocked";
      span.textContent = context.placeholder(alt);
      image.replaceWith(span);
    }
  }
  for (const link of [...fragment.querySelectorAll("a")]) {
    const href = link.getAttribute("href") ?? "";
    // No address (the sanitizer may have removed a `javascript:` one): plain
    // text, never a link to the document itself.
    if (!href.trim()) {
      link.removeAttribute("href");
      continue;
    }
    if (href.startsWith("#")) continue;
    const path = artifactPath(href, context.documentPath, allowed);
    if (path) {
      link.setAttribute("href", context.fileURL(path));
      link.setAttribute("download", "");
      link.setAttribute("data-artifact", path);
    } else {
      let safe: string | null = null;
      try {
        const url = new URL(href);
        if (["https:", "http:", "mailto:"].includes(url.protocol) && !url.username && !url.password) safe = url.href;
      } catch {
        safe = null;
      }
      if (safe === null) {
        link.removeAttribute("href");
        continue;
      }
      link.setAttribute("href", safe);
    }
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
    link.setAttribute("referrerpolicy", "no-referrer");
  }
  // A header cell without text names nothing: it stays a plain cell in the header row.
  for (const cell of [...fragment.querySelectorAll("th")]) {
    if (cell.textContent?.trim() || cell.querySelector("img, svg")) continue;
    const plain = doc.createElement("td");
    for (const attribute of [...cell.attributes]) plain.setAttribute(attribute.name, attribute.value);
    plain.append(...cell.childNodes);
    cell.replaceWith(plain);
  }
  for (const table of [...fragment.querySelectorAll("table")]) {
    const wrap = doc.createElement("div");
    wrap.className = "table-scroll";
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", context.tableLabel);
    wrap.tabIndex = 0;
    table.replaceWith(wrap);
    wrap.append(table);
  }
  for (const block of [...fragment.querySelectorAll("pre")]) {
    block.tabIndex = 0;
    block.setAttribute("aria-label", context.codeLabel);
  }
}

export interface ImageLoads {
  /** Settles once every waiting image is shown or replaced. */
  done: Promise<void>;
  /** Stop loading and release the object URLs. */
  dispose(): void;
}

/** Show the images that wait in `data-artifact-src`: each is fetched with the
 * header token (at most `limit` at a time) and shown through an object URL.
 * One that cannot be loaded is handed to `failed` (the view puts a labelled
 * placeholder there). */
export function loadArtifactImages(
  root: ParentNode,
  load: (path: string) => Promise<Blob>,
  failed: (image: HTMLImageElement) => void,
  { limit = 4, objectURL = URL }: { limit?: number; objectURL?: Pick<typeof URL, "createObjectURL" | "revokeObjectURL"> } = {},
): ImageLoads {
  const images = [...root.querySelectorAll<HTMLImageElement>("img[data-artifact-src]")];
  const urls: string[] = [];
  let disposed = false;
  let next = 0;
  const worker = async () => {
    while (!disposed && next < images.length) {
      const image = images[next++] as HTMLImageElement;
      const path = image.getAttribute("data-artifact-src") ?? "";
      let blob: Blob | null = null;
      try {
        blob = await load(path);
      } catch {
        blob = null;
      }
      if (disposed) return;
      image.removeAttribute("data-artifact-src");
      if (blob === null) {
        failed(image);
        continue;
      }
      const url = objectURL.createObjectURL(blob);
      urls.push(url);
      image.setAttribute("src", url);
    }
  };
  const done = Promise.all(Array.from({ length: Math.min(limit, images.length) }, worker)).then(() => undefined);
  return {
    done,
    dispose() {
      disposed = true;
      for (const url of urls.splice(0)) objectURL.revokeObjectURL(url);
    },
  };
}
