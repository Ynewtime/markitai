import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { cliCommand, stagedPaths } from "./cli.ts";
import { onShots, PAGE_BLOCKS, pairBlocks, shotBlocks, webViews, type PageBlock, type Target } from "./compare.ts";
import { compareLines, sideBySide, wordDiff } from "./diff.ts";
import { ADVANCED_DEFAULTS, resolveOptions } from "./options.ts";
import { listQuery, parseListState } from "./route.ts";

const file = (name: string, path?: string): File => {
  const value = new File(["x"], name);
  if (path) Object.defineProperty(value, "webkitRelativePath", { value: path });
  return value;
};

test("staged files are named from the folder that holds them, a picked folder once", () => {
  const files = [file("notes.pdf"), file("a.md", "Field trip/a.md"), file("b.md", "Field trip/sub/b.md"), file("春季 计划.docx"), file("notes.pdf")];
  assert.deepEqual(stagedPaths(files), ["./notes.pdf", "./Field trip/", "./春季 计划.docx"]);
  const options = resolveOptions({ preset: "minimal", llm: false, ocr: false, profile: null, advanced: { ...ADVANCED_DEFAULTS }, format: null, ocrLang: null });
  assert.equal(
    cliCommand([...stagedPaths(files), "https://example.com/a"], options),
    "markitai ./notes.pdf './Field trip/' './春季 计划.docx' https://example.com/a -o output/ --preset minimal",
  );
  // A name with a quote stays one word for the shell.
  assert.equal(cliCommand(stagedPaths([file("it's.txt")]), options), "markitai './it'\\''s.txt' -o output/ --preset minimal");
});

test("a diff side by side puts a change's lines face to face, with the words that changed", () => {
  const comparison = compareLines("title\nkeep\nold line one\nold two\nend", "title\nkeep\nnew line one\nend\nextra");
  assert.equal(comparison.kind, "rows");
  if (comparison.kind !== "rows") return;
  const rows = sideBySide(comparison.rows);
  assert.deepEqual(
    rows.map((row) => [row.left?.type ?? null, row.left?.no ?? null, row.right?.type ?? null, row.right?.no ?? null]),
    [
      ["ctx", 1, "ctx", 1],
      ["ctx", 2, "ctx", 2],
      ["del", 3, "add", 3],
      ["del", 4, null, null],
      ["ctx", 5, "ctx", 4],
      [null, null, "add", 5],
    ],
  );
  assert.deepEqual(rows[2]?.left?.parts, [
    { text: "old", changed: true },
    { text: " line one", changed: false },
  ]);
  assert.deepEqual(rows[2]?.right?.parts, [
    { text: "new", changed: true },
    { text: " line one", changed: false },
  ]);
  // CJK is compared character by character; lines with little in common are marked whole.
  assert.deepEqual(wordDiff("出租方：____", "出租方：______")?.b.map((part) => part.changed), [false, true]);
  assert.equal(wordDiff("completely different words", "nothing alike here at all"), null);
});

const block = (tag: string, text: string, extra: Partial<PageBlock> = {}): PageBlock => ({ tag, text, ...extra });
const target = (id: string, kind: Target["kind"], text: string, extra: Partial<Target> = {}): Target => ({ id, kind, text: [text], ...extra });

test("a page's pictures pair by place when the Markdown names a downloaded copy", () => {
  const blocks = [
    block("img", "", { src: "https://cdn.example/logo.png", small: true }),
    block("h1", "A year of field notes"),
    block("img", "", { src: "https://cdn.example/photo-1.jpg" }),
    block("p", "The first paragraph after the photo."),
    block("p", "- Plan the route? - Pack the tent."),
    block("article", "A year of field notes The first paragraph after the photo. Plan the route? Pack the tent. And much more text that makes the article long enough to hold everything in it."),
    block("img", "", { src: "https://cdn.example/photo-2.jpg" }),
    block("p", "Closing words."),
  ];
  const targets = [
    target("t0", "heading", "A year of field notes"),
    target("t1", "image", "A lake at dawn, written by a model"),
    target("t2", "block", "The first paragraph after the photo."),
    target("t3", "block", "Plan the route?"),
    target("t4", "block", "Pack the tent."),
    target("t5", "image", "A tent on a hill"),
    target("t6", "block", "Closing words."),
  ];
  const pairs = pairBlocks(targets, blocks);
  assert.equal(pairs.get("t1"), 2, "the picture between its neighbours, not the small logo");
  assert.equal(pairs.get("t3"), 4, "a list item inside the page's paragraph");
  assert.equal(pairs.get("t4"), 4, "two list items may share one page paragraph, never the whole article");
  assert.equal(pairs.get("t5"), 6);
  assert.equal(pairs.get("t6"), 7);
});

test("block boxes land on the screenshot tile they start on, at its scale", () => {
  const shots = [
    { width: 1920, height: 2000 },
    { width: 1920, height: 2000 },
    { width: 1920, height: 500 },
  ];
  // Captured at 1920 CSS px: scale 1.
  assert.deepEqual(onShots([10, 100, 300, 160], 1920, shots), { page: 1, box: [10, 100, 300, 160] });
  assert.deepEqual(onShots([10, 2100, 300, 2160], 1920, shots), { page: 2, box: [10, 100, 300, 160] });
  // Clipped at the tile's bottom.
  assert.deepEqual(onShots([0, 1990, 50, 2050], 1920, shots), { page: 1, box: [0, 1990, 50, 2000] });
  // Screenshots half the page's width: half the size.
  assert.deepEqual(onShots([100, 100, 200, 300], 3840, shots), { page: 1, box: [50, 50, 100, 150] });
  assert.equal(onShots([0, 9000, 10, 9010], 1920, shots), null);
  assert.deepEqual(shotBlocks([{ tag: "img", text: "", src: "x", bbox: [0, 0, 32, 32] }]), [{ tag: "img", text: "", src: "x", small: true }]);
});

test("a web page shows what was kept usable: the page, else its screenshot, else why neither", () => {
  const ok = { state: "ok" as const, text_length: 900, title: "Notes", reason: null };
  const check = { state: "challenge" as const, text_length: 40, title: "Just a moment...", reason: "challenge" };
  const blank = { state: "empty" as const, text_length: 0, title: null, reason: "blank" };
  const html = { kind: "html" as const, view: "/v", page_count: 2 };
  assert.deepEqual(webViews(html), { page: true, shot: true, failed: null }, "an older backend says nothing: both usable");
  assert.deepEqual(webViews({ ...html, capture: { page: blank, screenshot: ok } }), { page: false, shot: true, failed: null });
  assert.deepEqual(webViews({ ...html, capture: { page: check, screenshot: check } }), { page: false, shot: false, failed: check });
  assert.deepEqual(webViews({ kind: "screenshots", view: null, page_count: 1, capture: { page: null, screenshot: blank } }), { page: false, shot: false, failed: blank });
  assert.deepEqual(webViews({ kind: "none", view: null, page_count: 0 }), { page: false, shot: false, failed: null });
});

test("a conversion opens on its Preview; Details has its own address", () => {
  assert.equal(parseListState("?sel=0123456789ab/i1").mode, "preview");
  assert.equal(parseListState("?sel=0123456789ab/i1&view=details").mode, "inspector");
  assert.equal(listQuery({ selected: "a/i1", mode: "compare" }), "?sel=a%2Fi1&view=compare");
});

test("the capture script reads a page's blocks by the workbench's own rules", () => {
  // Both backends evaluate this file when they take a page's screenshots (contract v1.14).
  const script = readFileSync(new URL("../../../../../../markitai-core/src/browser/page_blocks.js", import.meta.url), "utf8");
  assert.ok(script.includes(`const SELECTOR = "${PAGE_BLOCKS}";`));
  assert.ok(script.includes('"script, style, template, noscript, [hidden], nav, header > nav"'));
  assert.ok(script.includes('const CONTAINERS = new Set(["div", "section", "article", "blockquote", "td", "th", "li", "dd"]);'));
  assert.ok(script.includes('const SKIP = new Set(["UL", "OL", "P", "PRE", "TABLE", "BLOCKQUOTE", "DL"]);'));
});
