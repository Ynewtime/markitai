// The blocks of a loaded web page and where they are drawn, read when its
// screenshots are taken (workbench contract v1.14, GET …/original/blocks).
// One expression, evaluated in the page; it changes nothing. The rules are the
// workbench's own for a framed page (`pageBlocks` in the web UI's compare.ts):
// headings, paragraphs, list items, code, definitions, cells, quotes, captions,
// tables and pictures, and div/section/article for the words they hold
// directly; nothing inside a script, style, template, noscript, hidden part or
// navigation; at most 5000. Both backends evaluate this exact text (Python
// reads it from the workbench bundle as capture/page-blocks.js).
(() => {
  const SELECTOR = "h1, h2, h3, h4, h5, h6, p, li, pre, dt, dd, td, th, blockquote, figcaption, caption, table, img, div, section, article";
  const CONTAINERS = new Set(["div", "section", "article", "blockquote", "td", "th", "li", "dd"]);
  const SKIP = new Set(["UL", "OL", "P", "PRE", "TABLE", "BLOCKQUOTE", "DL"]);
  const LIMIT = 5000;
  const TEXT = 2000;
  const words = (text) => text.normalize("NFKC").replace(/[^\p{L}\p{N}]+/gu, "").length;
  const own = (element) => {
    let text = "";
    for (const node of element.childNodes) {
      if (node.nodeType === 3) text += node.textContent || "";
      else if (node.nodeType === 1 && !SKIP.has(node.tagName)) text += node.textContent || "";
    }
    return text;
  };
  const root = document.documentElement;
  const left = window.scrollX;
  const top = window.scrollY;
  const blocks = [];
  for (const element of document.querySelectorAll(SELECTOR)) {
    if (blocks.length >= LIMIT) break;
    const tag = element.tagName.toLowerCase();
    const rect = element.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) continue;
    const bbox = [rect.left + left, rect.top + top, rect.right + left, rect.bottom + top].map((value) => Math.round(value * 10) / 10);
    if (tag === "img") {
      blocks.push({ tag, text: (element.getAttribute("alt") || "").slice(0, TEXT), src: element.currentSrc || element.src || "", bbox });
      continue;
    }
    if (element.closest("script, style, template, noscript, [hidden], nav, header > nav")) continue;
    const text = CONTAINERS.has(tag) && tag !== "td" && tag !== "th" ? own(element) : element.textContent || "";
    if ((tag === "div" || tag === "section" || tag === "article") && words(text) < 8) continue;
    blocks.push({ tag, text: text.replace(/\s+/g, " ").trim().slice(0, TEXT), bbox });
  }
  return { version: 1, width: Math.max(root.scrollWidth, root.clientWidth), height: Math.max(root.scrollHeight, document.body ? document.body.scrollHeight : 0), blocks };
})()
