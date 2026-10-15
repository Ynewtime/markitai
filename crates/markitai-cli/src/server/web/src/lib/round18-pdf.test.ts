import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { setToken } from "../api/token.ts";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { locate, type PageLines, type Target } from "./compare.ts";
import { ancestors, entryAt, firstFolded, itemOutline, markdownHeadings, outlineItems, outlineSource, parents, placedHeadings, plainHeading, shownEntries, withChildren } from "./pdf-outline.ts";
import {
  anchorOf,
  layoutPages,
  PAGE_GAP,
  PAGE_PAD,
  PageQueue,
  pageAt,
  pageInView,
  pagesNear,
  percentOf,
  PT_TO_PX,
  readZoom,
  renderWidth,
  scalesFor,
  scrollFor,
  typedPage,
  usableSizes,
  zoomKey,
  zoomStep,
  type PageTask,
} from "./pdf-view.ts";

const letter = { width: 612, height: 792 };

test("zoom: percentages are the printed size; fit width and fit page fit each page", () => {
  assert.deepEqual(scalesFor(100, { width: 500, height: 500 }, [letter]), [PT_TO_PX]);
  assert.equal(percentOf((scalesFor(150, { width: 1, height: 1 }, [letter]) as number[])[0] as number), 150);
  const view = { width: 644 + 2 * PAGE_PAD, height: 600 };
  // A landscape page among portrait ones: each page fills the width (as pages fitted the pane before).
  const pages = [letter, { width: 792, height: 612 }, letter];
  assert.deepEqual(scalesFor("width", view, pages), [644 / 612, 644 / 792, 644 / 612]);
  const fit = scalesFor("page", view, pages);
  assert.deepEqual(fit, [Math.min(644 / 612, 568 / 792), Math.min(644 / 792, 568 / 612), Math.min(644 / 612, 568 / 792)]);
  // Every page fits whole at fit page.
  for (const [index, page] of pages.entries()) assert.ok(page.height * (fit[index] as number) <= 568 + 1e-9 && page.width * (fit[index] as number) <= 644 + 1e-9);
  const layout = layoutPages(pages, scalesFor("width", view, pages), view.width);
  assert.deepEqual(layout.widths, [644, 644, 644]);
  assert.deepEqual([zoomStep(100, 1), zoomStep(100, -1), zoomStep(87, 1), zoomStep(87, -1), zoomStep(200, 1), zoomStep(50, -1)], [125, 75, 100, 75, 200, 50]);
  // A page the server could not measure takes the usual size, so the layout has no hole.
  assert.deepEqual(usableSizes([letter, { width: 0, height: 0 }]), [letter, letter]);
});

test("layout: one centred column, pages found by position, the page in view and the pages near it", () => {
  const pages = [letter, letter, letter, letter];
  const layout = layoutPages(pages, 0.5, 600);
  assert.deepEqual(layout.widths, [306, 306, 306, 306]);
  assert.deepEqual(layout.tops, [PAGE_PAD, PAGE_PAD + 396 + PAGE_GAP, PAGE_PAD + 2 * (396 + PAGE_GAP), PAGE_PAD + 3 * (396 + PAGE_GAP)]);
  assert.equal(layout.lefts[0], Math.floor((600 - 306) / 2));
  assert.equal(layout.height, PAGE_PAD + 4 * 396 + 3 * PAGE_GAP + PAGE_PAD);
  // Wider than the view: pages start at the padding and the sheet scrolls sideways.
  const wide = layoutPages(pages, 2, 600);
  assert.equal(wide.lefts[0], PAGE_PAD);
  assert.equal(wide.width, 1224 + 2 * PAGE_PAD);
  assert.equal(pageAt(layout, 0), 0);
  assert.equal(pageAt(layout, layout.tops[2] as number), 2);
  assert.equal(pageAt(layout, (layout.tops[2] as number) - 1), 1);
  assert.equal(pageInView(layout, layout.tops[1] as number, 300), 2);
  // Mostly page 3 in view.
  assert.equal(pageInView(layout, (layout.tops[2] as number) - 50, 300), 3);
  // At the end, the last page even when it fills less of the view.
  assert.equal(pageInView(layout, layout.height - 300, 300), 4);
  assert.deepEqual(pagesNear(layout, layout.tops[1] as number, 300, 0), [1, 1]);
  assert.deepEqual(pagesNear(layout, layout.tops[1] as number, 300, 300), [0, 2]);
});

test("a zoom keeps the same point of the same page where it was in the view", () => {
  const pages = Array.from({ length: 30 }, () => letter);
  const view = { width: 700, height: 500 };
  const before = layoutPages(pages, scalesFor("width", view, pages), view.width);
  const scroll = { top: (before.tops[12] as number) + 140, left: 0 };
  for (const zoom of [50, 200, "page", 125] as const) {
    const anchor = anchorOf(before, scroll, view);
    assert.equal(anchor.page, 12);
    const after = layoutPages(pages, scalesFor(zoom, view, pages), view.width);
    const next = scrollFor(after, anchor, view);
    // The anchored point is back at the same height in the view, and page 13 is still the one being read.
    const y = (after.tops[12] as number) + anchor.fy * (after.heights[12] as number) - next.top;
    assert.ok(Math.abs(y - anchor.vy) <= 1, `${zoom}: ${y} vs ${anchor.vy}`);
    assert.equal(pageInView(after, next.top, view.height), 13, String(zoom));
  }
  // Around the pointer (Ctrl + wheel): the point under it stays under it.
  const wide = layoutPages(pages, scalesFor(200, view, pages), view.width);
  const point = anchorOf(before, scroll, view, { vx: 100, vy: 50 });
  const moved = scrollFor(wide, point, view);
  assert.ok(Math.abs((wide.lefts[12] as number) + point.fx * (wide.widths[12] as number) - moved.left - 100) <= 1);
});

test("pages are asked for at the device width, in steps of 200, within the server's bounds", () => {
  assert.deepEqual([renderWidth(300, 1), renderWidth(300, 2), renderWidth(816, 1), renderWidth(1632, 2), renderWidth(50, 1), renderWidth(0, 0)], [400, 600, 1000, 2400, 200, 200]);
});

test("keys: Ctrl or ⌘ with + − 0 zoom; typed pages and remembered zooms are checked", () => {
  const key = (key: string, extra: Partial<KeyboardEvent> = {}) => zoomKey({ key, code: "", ctrlKey: true, metaKey: false, altKey: false, ...extra });
  assert.deepEqual([key("+"), key("="), key("-"), key("_"), key("0"), key("a")], ["in", "in", "out", "out", "reset", null]);
  assert.equal(key("=", { ctrlKey: false, metaKey: true }), "in");
  assert.equal(key("=", { ctrlKey: false }), null);
  assert.equal(key("=", { altKey: true }), null);
  assert.equal(zoomKey({ key: "Unidentified", code: "NumpadAdd", ctrlKey: true, metaKey: false, altKey: false }), "in");
  assert.deepEqual([typedPage("12", 492), typedPage(" ４９２ ", 492), typedPage("0", 492), typedPage("493", 492), typedPage("1.5", 492), typedPage("", 492)], [12, 492, null, null, null, null]);
  assert.deepEqual([readZoom("width"), readZoom("page"), readZoom(125), readZoom(130), readZoom("2x")], ["width", "page", 125, null, null]);
});

test("the page queue asks two at a time, nearest first, retries a busy page and drops what is no longer wanted", async () => {
  const started: string[] = [];
  const finish = new Map<string, (outcome: "done" | "busy" | "failed") => void>();
  const given: string[] = [];
  const queue = new PageQueue<PageTask>(
    (task) => {
      started.push(task.key);
      return new Promise((resolve) => finish.set(task.key, resolve));
    },
    (task) => given.push(task.key),
    2,
    1,
    () => 0,
  );
  queue.want([
    { key: "p5", rank: 3 },
    { key: "p3", rank: 1 },
    { key: "p4", rank: 2 },
    { key: "p9", rank: 7 },
  ]);
  assert.deepEqual(started, ["p3", "p4"]);
  assert.equal(queue.active, 2);
  // The reader moved on: p5 is no longer wanted, p8 is.
  queue.want([
    { key: "p8", rank: 0 },
    { key: "p9", rank: 1 },
  ]);
  finish.get("p3")?.("done");
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(started, ["p3", "p4", "p8"]);
  finish.get("p8")?.("busy");
  await new Promise((resolve) => setTimeout(resolve, 5));
  assert.ok(started.filter((key) => key === "p8").length === 2 || started.includes("p9"), started.join());
  // A second busy answer is one too many with one retry: it gives up.
  finish.get("p4")?.("done");
  await new Promise((resolve) => setTimeout(resolve, 5));
  finish.get("p8")?.("busy");
  await new Promise((resolve) => setTimeout(resolve, 5));
  assert.ok(given.includes("p8"), given.join());
  assert.ok(!started.includes("p5"));
  queue.close();
});

test("headings from the Markdown take the page of the marker before them, outside code and front matter", () => {
  const markdown = [
    "---",
    "title: Seed catalogue",
    "# not a heading",
    "---",
    "<!-- Page number: 1 -->",
    "",
    "# Seed catalogue",
    "",
    "Intro text.",
    "",
    "<!-- Page number: 2 -->",
    "",
    "## Tomatoes *heirloom*",
    "```md",
    "# inside code",
    "<!-- Page number: 9 -->",
    "```",
    "Beans and [peas](https://example.invalid/)",
    "---------",
    "",
    "<!-- Page number: 4 -->",
    "### Squash ###",
    "- list item",
    "---",
  ].join("\n");
  const { items, paged } = markdownHeadings(markdown, 3);
  assert.equal(paged, true);
  assert.deepEqual(items, [
    { title: "Seed catalogue", page: 1, level: 1 },
    { title: "Tomatoes heirloom", page: 2, level: 2 },
    { title: "Beans and peas", page: 2, level: 2 },
    // The marker names a page past the PDF's end: the last page.
    { title: "Squash", page: 3, level: 3 },
  ]);
  const plain = markdownHeadings("## Only\n\ntext\n\n### Deeper", 0);
  assert.equal(plain.paged, false);
  assert.deepEqual(plain.items, [
    { title: "Only", page: 1, level: 1 },
    { title: "Deeper", page: 1, level: 2 },
  ]);
  assert.equal(plainHeading("**Bold** `code` <span>tag</span> [link](x) \\# and &amp;"), "Bold code tag link # and &");
  // Without markers, the headings Compare placed on the pages.
  assert.deepEqual(
    placedHeadings([
      { title: "Intro", depth: 2, page: 1 },
      { title: "Lost", depth: 2, page: null },
      { title: "Detail", depth: 3, page: 4 },
    ]),
    [
      { title: "Intro", page: 1, level: 1 },
      { title: "Detail", page: 4, level: 2 },
    ],
  );
});

test("an outline folds, marks the entry of the page being read and picks its source", () => {
  const items = [
    { title: "Part one", page: 1, level: 1 },
    { title: "Chapter 1", page: 2, level: 2 },
    { title: "Section", page: 3, level: 4 },
    { title: "Chapter 2", page: 6, level: 2 },
    { title: "Part two", page: 9, level: 1 },
  ];
  assert.deepEqual(parents(items), [-1, 0, 1, 0, -1]);
  assert.deepEqual([...withChildren(items)].sort(), [0, 1]);
  assert.deepEqual(shownEntries(items, new Set([1])), [0, 1, 3, 4]);
  assert.deepEqual(shownEntries(items, new Set([0])), [0, 4]);
  assert.deepEqual([entryAt(items, 1), entryAt(items, 4), entryAt(items, 8), entryAt(items, 20)], [0, 2, 3, 4]);
  assert.deepEqual(ancestors(items, 2), [1, 0]);
  assert.equal(firstFolded(items).size, 0);
  const long = Array.from({ length: 80 }, (_, index) => ({ title: `Entry ${index}`, page: index + 1, level: index % 4 === 0 ? 1 : 2 }));
  assert.equal(shownEntries(long, firstFolded(long)).length, 20);
  assert.equal(outlineSource(items, []), "pdf");
  assert.equal(outlineSource([], items), "headings");
  assert.equal(outlineSource([], []), "pages");
  // A server's answer is checked and its pages kept within the document.
  assert.deepEqual(
    outlineItems({ items: [{ title: "A", page: 3, level: 1 }, { title: "Bad", page: 0, level: 1 }, { title: "Far", page: 900, level: 2 }, { title: "x", page: 2, level: 0 }], truncated: false }, 10),
    [
      { title: "A", page: 3, level: 1 },
      { title: "Far", page: 10, level: 2 },
    ],
  );
  assert.deepEqual(outlineItems(null, 10), []);
});

test("the outline is read from its route only when the backend says it has one", async () => {
  setToken("t-admin", null);
  const original = globalThis.fetch;
  const asked: string[] = [];
  globalThis.fetch = (async (url: string | URL) => {
    const path = new URL(String(url)).pathname;
    asked.push(path);
    const body = path === "/api/capabilities" ? { features: { pdf_outline: true } } : { items: [{ title: "Beds", page: 2, level: 1 }], truncated: false };
    return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
  }) as typeof fetch;
  try {
    const first = await itemOutline("job 1", "i1");
    const again = await itemOutline("job 1", "i1");
    assert.deepEqual(first, { items: [{ title: "Beds", page: 2, level: 1 }], truncated: false });
    assert.equal(again, first);
    assert.deepEqual(asked, ["/api/capabilities", "/api/jobs/job%201/items/i1/original/outline"]);
  } finally {
    globalThis.fetch = original;
    setToken("", null);
  }
});

test("block outlines sit in page percentages, so linking holds at every zoom", () => {
  // The same located box drawn at three zooms covers the same share of the page.
  const pages: PageLines[] = [{ page: 1, lines: [{ text: "Planting calendar", bbox: [72, 90, 300, 110] }] }];
  const targets: Target[] = [{ id: "t0", kind: "heading", text: ["Planting calendar"] }];
  const spot = locate(pages, targets).get("t0");
  assert.ok(spot);
  for (const zoom of [50, "width", 200] as const) {
    const layout = layoutPages([letter], scalesFor(zoom, { width: 700, height: 500 }, [letter]), 700);
    const left = (spot.box[0] / letter.width) * (layout.widths[0] as number);
    assert.ok(Math.abs(left / (layout.widths[0] as number) - 72 / 612) < 1e-9);
  }
  const source = readFileSync(new URL("../components/pdf-viewer.tsx", import.meta.url), "utf8");
  assert.match(source, /left: `\$\{\(box\[0\] \/ size\.width\) \* 100\}%`/);
});

test("the viewer's words exist in every language and its stylesheet is pulled into app.css", () => {
  const keys = Object.keys(en).filter((key) => key.startsWith("pdf"));
  assert.ok(keys.length >= 25);
  for (const pack of [zh, ja]) for (const key of keys) assert.ok(key in pack, key);
  assert.equal(zh.pdfPageInput(492), "页码，1 至 492");
  const css = readFileSync(new URL("../styles/app.css", import.meta.url), "utf8");
  assert.match(css, /^@import "\.\/pdf-viewer\.css";$/m);
  const viewer = readFileSync(new URL("../styles/pdf-viewer.css", import.meta.url), "utf8");
  // Only the scale's tokens: no raw font sizes or durations.
  assert.ok(!/font-size:\s*\d|\d+m?s\b/.test(viewer.replace(/\/\*[\s\S]*?\*\//g, "")));
});
