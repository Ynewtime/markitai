// Safe rendering of converted Markdown. The vendored marked parser and DOMPurify
// sanitizer load on first use; the sanitized fragment is then rewritten so that
// images and attachments resolve only to files listed in the result itself.
// A long document is read in pieces (`splitMarkdown`, `readPiece`) into a
// plan of chunks and drawn a chunk at a time (`renderChunk`), so the first
// screen never waits for the last.
import type { Artifact } from "../api/types.ts";
import { mathExtensions } from "./math-syntax.ts";

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

/** An image reference whose folder does not match (a model rewrote `.markitai/assets/x.png`
 * as `assets/x.png`, an e-book kept its own `../Images/x.png`) resolved by its file name to
 * the one listed raster image of that name; nothing when the name is missing or not unique.
 * Schemes, absolute paths and backslashes never resolve. */
export function imageByName(target: string, images: readonly string[]): string | null {
  if (/^[a-z][a-z\d+.-]*:/i.test(target) || target.startsWith("//") || target.startsWith("/") || target.includes("\\")) return null;
  let name: string;
  try {
    name = decodeURIComponent(target.split(/[?#]/)[0]?.split("/").pop() ?? "");
  } catch {
    return null;
  }
  if (!name || !RASTER.test(name)) return null;
  const found = images.filter((path) => path === name || path.endsWith(`/${name}`));
  return found.length === 1 ? (found[0] ?? null) : null;
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

/** A block of marked's token list: only what the chunk plan reads. */
export interface Token {
  type: string;
  raw: string;
  depth?: number;
  text?: string;
  tokens?: Token[];
  items?: Token[];
  header?: Token[];
  rows?: Token[][];
}
type TokenList = Token[] & { links?: unknown };

export interface Libraries {
  marked: {
    use(options: { extensions: typeof mathExtensions }): void;
    /** The options `use` has changed: the static lexer and parser take only the options they are given. */
    defaults?: Record<string, unknown>;
    parse(markdown: string, options: { async: false; gfm: boolean }): string;
    lexer(markdown: string, options: { gfm: boolean }): TokenList;
    parser(tokens: TokenList, options: { async: false; gfm: boolean }): string;
  };
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
  libraries ??= Promise.all([import("/ui/marked.js"), import("/ui/purify.js")]).then(([markedModule, purifyModule]) => {
    // Formulas are read as `.math` elements (drawn later, by lib/math.ts), not as Markdown.
    markedModule.marked.use({ extensions: mathExtensions });
    return { marked: markedModule.marked, purify: purifyModule.default };
  });
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
  return sanitized(purify, marked.parse(markdown, { async: false, gfm: true }), context);
}

/** A heading of the document's outline, by its position among the top-level headings. */
export interface Heading {
  depth: number;
  text: string;
  /** The chunk that draws it. */
  chunk: number;
}

/** A document cut into chunks of about `size` characters of Markdown (the
 * first one smaller, so the first screen comes quickly); a chunk prefers to
 * end before a chapter heading. It grows as the document is read in pieces
 * (`splitMarkdown`), so reading never holds the page either. */
export interface DocumentPlan {
  chunks: TokenList[];
  /** Characters (UTF-16 units) of Markdown per chunk, for a size estimate before it is drawn. */
  sizes: number[];
  headings: Heading[];
  /** Tables in the whole document, counted before it is read. */
  tables: number;
}

export const FIRST_CHUNK = 3000;
export const CHUNK = 5000;

export const emptyPlan = (tables = 0): DocumentPlan => ({ chunks: [], sizes: [], headings: [], tables });

/** Add one piece's blocks to a plan; a piece always starts a chunk of its own. */
export function extendPlan(plan: DocumentPlan, tokens: TokenList, { first = FIRST_CHUNK, size = CHUNK }: { first?: number; size?: number } = {}): DocumentPlan {
  let current: TokenList = [];
  let length = 0;
  const close = () => {
    if (!current.length) return;
    current.links = tokens.links;
    plan.chunks.push(current);
    plan.sizes.push(length);
    current = [];
    length = 0;
  };
  for (const token of tokens) {
    const target = plan.chunks.length === 0 ? first : size;
    const chapter = token.type === "heading" && (token.depth ?? 6) <= 2;
    if (length >= target || (chapter && length >= target / 2)) close();
    if (token.type === "heading") plan.headings.push({ depth: token.depth ?? 1, text: headingLabel(token), chunk: plan.chunks.length });
    current.push(token);
    length += token.raw.length;
  }
  close();
  return plan;
}

/** A whole token list as one plan (tests, and documents read at once). */
export function planTokens(tokens: TokenList, sizes?: { first?: number; size?: number }): DocumentPlan {
  let tables = 0;
  for (const token of tokens) tables += countTables(token);
  return extendPlan(emptyPlan(tables), tokens, sizes);
}

const REFERENCE_DEFINITION = /^ {0,3}\[[^\]\n]+\]:/m;
const FENCE_LINE = /^ {0,3}(`{3,}|~{3,})/;
const RAW_BLOCK = /^ {0,3}<(pre|script|style|textarea)[\s>]|^ {0,3}<!--/i;
const TABLE_DELIMITER = /^ {0,3}\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$/;

/** Markdown cut into pieces of about `size` characters that read the same alone
 * as together: only at a blank line before a line that starts a new top-level
 * block (not indented, not a list item or quote that would join the one above),
 * never inside fenced code or a raw HTML block, and not at all when the document
 * defines reference links (each piece would lose the others' definitions). */
export function splitMarkdown(markdown: string, { first = FIRST_CHUNK, size = CHUNK }: { first?: number; size?: number } = {}): string[] {
  if (markdown.length <= first || REFERENCE_DEFINITION.test(markdown)) return [markdown];
  const pieces: string[] = [];
  let start = 0;
  let fence: string | null = null;
  let raw: RegExp | null = null;
  let blank = false;
  let at = 0;
  while (at < markdown.length) {
    const newline = markdown.indexOf("\n", at);
    const end = newline < 0 ? markdown.length : newline + 1;
    const line = markdown.slice(at, newline < 0 ? markdown.length : newline);
    if (fence === null && raw === null && blank && at - start >= (pieces.length ? size : first) && /^[^\s>*+\-\d|]/.test(line)) {
      pieces.push(markdown.slice(start, at));
      start = at;
    }
    const opened = FENCE_LINE.exec(line);
    if (fence !== null) {
      if (opened && opened[1]?.[0] === fence[0] && (opened[1]?.length ?? 0) >= fence.length && !line.trim().slice(opened[1]?.length ?? 0).trim()) fence = null;
    } else if (raw !== null) {
      if (raw.test(line)) raw = null;
    } else if (opened) fence = opened[1] ?? "```";
    else {
      const html = RAW_BLOCK.exec(line);
      if (html) {
        const close = html[1] ? new RegExp(`</${html[1]}>`, "i") : /-->/;
        if (!close.test(line.slice(html.index + 4))) raw = close;
      }
    }
    blank = line.trim() === "";
    at = end;
  }
  if (start < markdown.length) pieces.push(markdown.slice(start));
  return pieces;
}

/** Tables in a document, counted from their delimiter rows (outside fenced code). */
export function countTableRows(markdown: string): number {
  let count = 0;
  let fence = false;
  let previous = "";
  for (const line of markdown.split("\n")) {
    if (FENCE_LINE.test(line)) fence = !fence;
    else if (!fence && previous.includes("|") && line.includes("-") && TABLE_DELIMITER.test(line)) count++;
    previous = line;
  }
  return count;
}

/** Draw one chunk of a plan: parsed, sanitized and pinned like a whole document;
 * `tables` numbers its tables among the document's. */
export function renderChunk({ marked, purify }: Libraries, plan: DocumentPlan, index: number, context: RenderContext, tables: { offset: number }): DocumentFragment {
  const tokens = plan.chunks[index] ?? [];
  return sanitized(purify, marked.parser(tokens, { ...marked.defaults, async: false, gfm: true }), context, { offset: tables.offset, total: plan.tables }, tables);
}

/** Read one piece of a document into its plan. */
export function readPiece({ marked }: Libraries, plan: DocumentPlan, piece: string): void {
  extendPlan(plan, marked.lexer(piece, { ...marked.defaults, gfm: true }));
}

/** The text a heading shows, without its Markdown. */
export function plainText(token: Token): string {
  if (token.tokens?.length) return token.tokens.map(plainText).join("");
  if (token.type === "image") return "";
  return (token.text ?? "").replace(/&(amp|lt|gt|quot|#39);/g, (_, name: string) => ({ amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'" })[name] ?? "");
}

// Private-use code points (an extracted font's own glyphs) draw as empty boxes.
const PRIVATE_USE = /[\uE000-\uF8FF\u{F0000}-\u{FFFFD}\u{100000}-\u{10FFFD}\uFFFD]/gu;

/** A heading as the contents list it: its words as drawn, without inline HTML
 * tags, private-use glyphs or runs of white space. */
export function headingLabel(token: Token): string {
  const words = (part: Token): string => {
    if (part.tokens?.length) return part.tokens.map(words).join("");
    if (part.type === "html") return " ";
    return plainText(part);
  };
  return words(token).replace(PRIVATE_USE, " ").replace(/\s+/g, " ").trim();
}

function countTables(token: Token): number {
  if (token.type === "table") return 1;
  let count = 0;
  for (const child of token.tokens ?? []) count += countTables(child);
  for (const item of token.items ?? []) count += countTables(item);
  return count;
}

function sanitized(purify: Purify, html: string, context: RenderContext, numbering?: { offset: number; total: number }, counter?: { offset: number }): DocumentFragment {
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
  const drawn = rewrite(fragment, context, numbering);
  if (counter) counter.offset += drawn;
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

/** An absolute https address without user information, as the page may load it; else null. */
export function remotePicture(source: string): string | null {
  if (!/^https:\/\//i.test(source.trim())) return null;
  try {
    const url = new URL(source.trim());
    return url.protocol === "https:" && !url.username && !url.password && url.hostname ? url.href : null;
  } catch {
    return null;
  }
}

/** A picture's placeholder once it could not load, and back: Compare keeps
 * outlining the block that stands in for a picture it paired. */
const placeholders = new WeakMap<Element, Element>();
const pictures = new WeakMap<Element, Element>();
/** The element now showing `element`: its placeholder when it was a picture that failed. */
export const shownAs = (element: Element): Element => (element.isConnected ? element : (placeholders.get(element) ?? element));
/** The picture a placeholder replaced, or the element itself. */
export const shownFor = (element: Element): Element => pictures.get(element) ?? element;

/** Put a labelled placeholder where a picture was. */
export function replacePicture(image: Element, text: string): HTMLSpanElement {
  const span = image.ownerDocument.createElement("span");
  span.className = "img-blocked";
  span.textContent = text;
  image.replaceWith(span);
  placeholders.set(image, span);
  pictures.set(span, image);
  return span;
}

/** A video address the page may play: absolute `https://` without user information, naming a video file
 * (`.mp4`, `.m4v`, `.webm`, `.mov`, `.ogv`); else null. */
export function remoteVideo(address: string): string | null {
  const url = remotePicture(address);
  if (url === null) return null;
  return /\.(mp4|m4v|webm|mov|ogv)$/i.test(new URL(url).pathname) ? url : null;
}

/** The link a played-in-place video stands for, back in its place if the video cannot play. */
const videoLinks = new WeakMap<Element, Element>();

/** A remote picture that cannot load becomes its labelled placeholder, and a video that cannot
 * play its link (one listener for every rendered document, as error events do not bubble). */
export function swapFailedPicture(event: Event): void {
  const target = event.target as HTMLElement | null;
  if (target?.tagName === "VIDEO") {
    const link = videoLinks.get(target);
    if (link) target.replaceWith(link);
    return;
  }
  const image = target as HTMLImageElement | null;
  if (!image || image.tagName !== "IMG" || !image.hasAttribute("data-remote")) return;
  replacePicture(image, image.getAttribute("data-fallback") || image.alt);
}
if (typeof document !== "undefined") document.addEventListener("error", swapFailedPicture, true);

/** The page a PDF's extracted picture came from: Markitai's Rust converter
 * writes `![Image on page 3](…)`, the Python one names the file
 * `<name>.pdf-0003-01.png`; null for any other picture. */
export function pdfImagePage(alt: string, path: string): number | null {
  const found = /^Image on page (\d{1,5})$/.exec(alt.trim()) ?? /\.pdf-(\d{1,5})-\d+\.[a-z0-9]+$/i.exec(path);
  const page = found ? Number(found[1]) : 0;
  return page >= 1 ? page : null;
}

/** Pin every reference to the result's own files; everything else is inert.
 * Tables are named among `numbering.total` from `numbering.offset` on (a chunk
 * of a longer document); returns how many tables the fragment holds. */
export function rewrite(fragment: DocumentFragment, context: RenderContext, numbering?: { offset: number; total: number }): number {
  const doc = fragment.ownerDocument;
  const allowed = new Set(context.artifacts.map((artifact) => artifact.relpath));
  const images = context.artifacts.map((artifact) => artifact.relpath).filter((path) => RASTER.test(path));
  pinDrawings(fragment);
  for (const image of [...fragment.querySelectorAll("img")]) {
    const source = image.getAttribute("src") ?? "";
    const path = artifactPath(source, context.documentPath, allowed) ?? imageByName(source, images);
    const alt = image.getAttribute("alt") || "";
    if (path && RASTER.test(path)) {
      const url = context.imageURL(path);
      if (url === null) {
        image.removeAttribute("src");
        image.setAttribute("data-artifact-src", path);
      } else image.setAttribute("src", url);
      image.setAttribute("loading", "lazy");
      image.setAttribute("referrerpolicy", "no-referrer");
      // A PDF's picture names its page: Compare finds it there.
      const page = pdfImagePage(alt, path);
      if (page !== null) image.setAttribute("data-page", String(page));
      // A caption beside the picture that says exactly its alt text (an e-book's
      // figure: `![Fig. 2](…)Fig. 2`) is read once: the picture becomes decorative.
      const block = typeof image.closest === "function" ? image.closest("p, li, td, th") : null;
      if (alt && block?.textContent?.trim() === alt.trim()) image.setAttribute("alt", "");
      else if (!alt) image.setAttribute("alt", context.fallbackAlt);
    } else if (remotePicture(source)) {
      // A picture on the web (a fetched page's own): loaded from its site as the
      // page's policy allows (https only), without a referrer, when it comes near
      // the view; the placeholder takes its place only if it fails to load.
      image.setAttribute("src", remotePicture(source) as string);
      image.setAttribute("loading", "lazy");
      image.setAttribute("decoding", "async");
      image.setAttribute("referrerpolicy", "no-referrer");
      image.setAttribute("data-remote", "");
      image.setAttribute("data-fallback", context.placeholder(alt));
      image.removeAttribute("srcset");
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
  // A video written as a linked poster (`[![Video](poster)](clip.mp4)`, both backends' form) plays in place.
  for (const link of [...fragment.querySelectorAll("a[href]")]) {
    const source = remoteVideo(link.getAttribute("href") ?? "");
    const poster = link.querySelector("img");
    if (source === null || link.textContent?.trim() || link.querySelectorAll("img, .img-blocked").length > 1) continue;
    const video = doc.createElement("video");
    video.setAttribute("controls", "");
    video.setAttribute("preload", "none");
    video.setAttribute("playsinline", "");
    // The poster is the link's picture: one on the web (https) as the page's policy loads it, or the
    // item's own copy through its asset route (fetched with the token and shown as an object URL).
    const still = poster?.hasAttribute("data-remote") ? remotePicture(poster.getAttribute("src") ?? "") : null;
    if (still) video.setAttribute("poster", still);
    else if (poster?.getAttribute("data-artifact-src")) video.setAttribute("data-artifact-poster", poster.getAttribute("data-artifact-src") as string);
    else if (poster?.getAttribute("src")) video.setAttribute("poster", poster.getAttribute("src") as string);
    video.setAttribute("aria-label", poster?.getAttribute("alt") || link.querySelector(".img-blocked")?.textContent || context.fallbackAlt);
    video.setAttribute("src", source);
    video.className = "doc-video";
    link.replaceWith(video);
    videoLinks.set(video, link);
  }
  // A header cell without text names nothing: it stays a plain cell in the header row.
  for (const cell of [...fragment.querySelectorAll("th")]) {
    if (cell.textContent?.trim() || cell.querySelector("img, svg")) continue;
    const plain = doc.createElement("td");
    for (const attribute of [...cell.attributes]) plain.setAttribute(attribute.name, attribute.value);
    plain.append(...cell.childNodes);
    cell.replaceWith(plain);
  }
  // Each table is a scrollable region of its own name ("Table 2"), so a document with several reads them apart.
  const tables = [...fragment.querySelectorAll("table")];
  const total = numbering?.total ?? tables.length;
  for (const [index, table] of tables.entries()) {
    const wrap = doc.createElement("div");
    wrap.className = "table-scroll";
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", total > 1 ? `${context.tableLabel} ${(numbering?.offset ?? 0) + index + 1}` : context.tableLabel);
    wrap.tabIndex = 0;
    table.replaceWith(wrap);
    wrap.append(table);
  }
  for (const block of [...fragment.querySelectorAll("pre")]) {
    block.tabIndex = 0;
    block.setAttribute("aria-label", context.codeLabel);
  }
  return tables.length;
}

export interface ImageLoads {
  /** Also show the waiting images under `root` (a chunk drawn later). */
  add(root: ParentNode): void;
  /** Load every waiting image now (printing needs them all); settles once each is shown or replaced. */
  all(): Promise<void>;
  /** Stop loading and release the object URLs. */
  dispose(): void;
}

/** Show the images that wait in `data-artifact-src`: each is fetched with the
 * header token (at most `limit` at a time) and shown through an object URL.
 * `lazy` fetches an image only once it comes near the viewport, so a long
 * document never downloads and decodes every picture at once. One that cannot
 * be loaded is handed to `failed` (the view puts a labelled placeholder there). */
export function loadArtifactImages(
  root: ParentNode,
  load: (path: string) => Promise<Blob>,
  failed: (image: HTMLImageElement) => void,
  {
    limit = 4,
    lazy = false,
    margin = "1200px 0px",
    objectURL = URL,
  }: { limit?: number; lazy?: boolean; margin?: string; objectURL?: Pick<typeof URL, "createObjectURL" | "revokeObjectURL"> } = {},
): ImageLoads {
  // Pictures wait in `data-artifact-src`; a video's poster (the item's own picture) in `data-artifact-poster`.
  const attributeOf = (element: Element) => (element.tagName === "VIDEO" ? "data-artifact-poster" : "data-artifact-src");
  const urls: string[] = [];
  const known: HTMLElement[] = [];
  const seen = new Set<HTMLElement>();
  const queued = new Set<HTMLElement>();
  const queue: HTMLElement[] = [];
  let waiters: (() => void)[] = [];
  let active = 0;
  let disposed = false;
  const settle = () => {
    if (active > 0 || (queue.length > 0 && !disposed)) return;
    const ready = waiters;
    waiters = [];
    for (const resolve of ready) resolve();
  };
  const show = async (image: HTMLElement) => {
    const attribute = attributeOf(image);
    const path = image.getAttribute(attribute) ?? "";
    let blob: Blob | null = null;
    try {
      blob = await load(path);
    } catch {
      blob = null;
    }
    if (disposed) return;
    image.removeAttribute(attribute);
    const video = image.tagName === "VIDEO";
    if (blob === null) {
      // A poster that cannot load leaves the player as it is.
      if (!video) failed(image as HTMLImageElement);
      return;
    }
    const url = objectURL.createObjectURL(blob);
    urls.push(url);
    image.setAttribute(video ? "poster" : "src", url);
  };
  const pump = () => {
    while (!disposed && active < limit && queue.length > 0) {
      const image = queue.shift() as HTMLElement;
      active++;
      void show(image).finally(() => {
        active--;
        pump();
        settle();
      });
    }
  };
  const enqueue = (image: HTMLElement) => {
    if (disposed || queued.has(image) || image.getAttribute(attributeOf(image)) === null) return;
    queued.add(image);
    queue.push(image);
    pump();
  };
  const observer =
    lazy && typeof IntersectionObserver === "function"
      ? new IntersectionObserver(
          (entries) => {
            for (const entry of entries) {
              if (!entry.isIntersecting) continue;
              observer?.unobserve(entry.target);
              enqueue(entry.target as HTMLElement);
            }
          },
          { rootMargin: margin },
        )
      : null;
  const add = (scope: ParentNode) => {
    for (const image of scope.querySelectorAll<HTMLElement>("img[data-artifact-src], video[data-artifact-poster]")) {
      if (seen.has(image)) continue;
      seen.add(image);
      known.push(image);
      if (observer) observer.observe(image);
      else enqueue(image);
    }
  };
  add(root);
  return {
    add,
    all() {
      for (const image of known) {
        observer?.unobserve(image);
        enqueue(image);
      }
      return new Promise<void>((resolve) => {
        waiters.push(resolve);
        settle();
      });
    },
    dispose() {
      disposed = true;
      observer?.disconnect();
      for (const url of urls.splice(0)) objectURL.revokeObjectURL(url);
      settle();
    },
  };
}
