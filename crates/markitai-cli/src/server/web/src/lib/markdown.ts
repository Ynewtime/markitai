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
  /** URL of an artifact for an <img> (carries the token when one is held). */
  imageURL: (path: string) => string;
  /** URL of an artifact for a download link. */
  fileURL: (path: string) => string;
  placeholder: (alt: string) => string;
  fallbackAlt: string;
  tableLabel: string;
  codeLabel: string;
}

interface Libraries {
  marked: { parse(markdown: string, options: { async: false; gfm: boolean }): string };
  purify: { sanitize(html: string, options: Record<string, unknown>): DocumentFragment };
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

const FORBID_TAGS = ["style", "script", "iframe", "object", "embed", "form", "input", "button", "textarea", "select", "video", "audio", "source", "picture", "link", "meta", "base"];
const FORBID_ATTR = ["style", "srcset", "poster", "ping", "target", "id", "name"];

/** Parse, sanitize and rewrite. Raw HTML never reaches the live document. */
export async function renderMarkdown(markdown: string, context: RenderContext): Promise<DocumentFragment> {
  const { marked, purify } = await loadLibraries();
  const html = marked.parse(markdown, { async: false, gfm: true });
  const fragment = purify.sanitize(html, {
    RETURN_DOM_FRAGMENT: true,
    USE_PROFILES: { html: true },
    FORBID_TAGS,
    FORBID_ATTR,
    ALLOW_DATA_ATTR: false,
    ALLOW_ARIA_ATTR: false,
  });
  rewrite(fragment, context);
  return fragment;
}

/** Pin every reference to the result's own files; everything else is inert. */
export function rewrite(fragment: DocumentFragment, context: RenderContext): void {
  const doc = fragment.ownerDocument;
  const allowed = new Set(context.artifacts.map((artifact) => artifact.relpath));
  for (const image of [...fragment.querySelectorAll("img")]) {
    const path = artifactPath(image.getAttribute("src") ?? "", context.documentPath, allowed);
    const alt = image.getAttribute("alt") || "";
    if (path && RASTER.test(path)) {
      image.setAttribute("src", context.imageURL(path));
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
    if (href.startsWith("#")) continue;
    const path = artifactPath(href, context.documentPath, allowed);
    if (path) {
      link.setAttribute("href", context.fileURL(path));
      link.setAttribute("download", "");
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
