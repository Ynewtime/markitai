// Compare links a PDF's pictures to their boxes by page (owner report, 2026-10-25): in the LLM
// version the model rewrote the alt text, so a picture's page comes from page markers or from the
// Base version, matched by the picture's file, which both versions name alike.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { assetName, imagePages, locate, placePictures, targetsOf, unplacedPictures, type PageLines } from "./compare.ts";
import { rewrite } from "./markdown.ts";

const BASE = [
  "<!-- ![Page 1](.markitai/screenshots/orchard-report.pdf.page0001.jpg) -->",
  "",
  "# Orchard Cooperative Yearly Report",
  "",
  "![Image on page 1](.markitai/assets/e4b7a272af48e4ca2e7a6de9.jpg)",
  "",
  "<!-- ![Page 2](.markitai/screenshots/orchard-report.pdf.page0002.jpg) -->",
  "",
  "![Image on page 2](.markitai/assets/15634d49a36698d130cbd823.jpg)",
].join("\n");
// The LLM version: the model's alt texts, no screenshots or page markers.
const LLM = [
  "# Orchard Cooperative Yearly Report",
  "",
  "![Bar chart of three columns](.markitai/assets/e4b7a272af48e4ca2e7a6de9.jpg)",
  "",
  "Volunteer hours by quarter",
  "",
  "![Bar chart of three columns](.markitai/assets/15634d49a36698d130cbd823.jpg)",
].join("\n");

test("a picture's page: its Base alt, else the page marker or screenshot before it", () => {
  assert.deepEqual([...imagePages(BASE)], [
    ["e4b7a272af48e4ca2e7a6de9.jpg", [1]],
    ["15634d49a36698d130cbd823.jpg", [2]],
  ]);
  // The LLM text alone says nothing about pages.
  const llm = imagePages(LLM);
  assert.deepEqual([...llm.values()], [[null], [null]]);
  assert.equal(unplacedPictures(llm), 2);
  // With page markers (`--page-markers`) or page screenshots it says them itself.
  const marked = imagePages("<!-- page: 1 -->\n\n![A chart](a/x.jpg)\n\n<!-- page: 3 -->\n\nText ![inline](<a/y z.jpg>) and more\n\n![Page 4](s/doc.pdf.page0004.jpg)\n\n![last](a/x.jpg)");
  assert.deepEqual([...marked], [
    ["x.jpg", [1, 4]],
    ["y z.jpg", [3]],
  ]);
  assert.equal(unplacedPictures(marked), 0);
  // A page-numbered name says it too; names decode and lose their query.
  assert.deepEqual([...imagePages("![x](assets/doc.pdf-7-2.png)")], [["doc.pdf-7-2.png", [7]]]);
  assert.equal(assetName("<.markitai/assets/caf%C3%A9.png?v=1>"), "café.png");
});

/** A rendered picture as the workbench draws it: its file in data-asset, a page in data-page when known. */
function picture(asset: string, alt: string, page?: number) {
  const dataset: Record<string, string> = { asset };
  if (page) dataset.page = String(page);
  return { tagName: "IMG", dataset, getAttribute: (name: string) => (name === "alt" ? alt : name === "src" ? `blob:${asset}` : null) };
}

test("the LLM version's pictures take their pages from the Base version and find their boxes", () => {
  const images = [
    picture(".markitai/assets/e4b7a272af48e4ca2e7a6de9.jpg", "Bar chart of three columns"),
    picture(".markitai/assets/15634d49a36698d130cbd823.jpg", "Bar chart of three columns"),
  ];
  const root = { querySelectorAll: () => images };
  // Before: no page, so nothing to find (the owner's report).
  const pages: PageLines[] = [
    { page: 1, lines: [{ text: "Orchard Cooperative Yearly Report", bbox: [56, 40, 400, 60] }], images: [[56, 200, 476, 443]] },
    { page: 2, lines: [{ text: "Volunteer hours by quarter", bbox: [56, 40, 300, 56] }], images: [[56, 180, 476, 423]] },
  ];
  const before = targetsOf(root as never);
  assert.equal(locate(pages, before.map(({ target }) => target)).size, 0);
  // After: the LLM text says nothing, the Base text names each file's page.
  placePictures(root as never, imagePages(LLM), imagePages(BASE));
  assert.deepEqual(images.map((image) => image.dataset.page), ["1", "2"]);
  const targets = targetsOf(root as never).map(({ target }) => target);
  const found = locate(pages, targets);
  assert.deepEqual([...found.values()], [
    { page: 1, box: [56, 200, 476, 443] },
    { page: 2, box: [56, 180, 476, 423] },
  ]);
  // A page the picture already names is kept, and an identical picture on two pages (one file)
  // takes the pages in order.
  const twice = [picture("a/logo.png", "logo"), picture("a/logo.png", "logo", 5)];
  placePictures({ querySelectorAll: () => twice } as never, new Map([["logo.png", [2, 3]]]));
  assert.deepEqual(twice.map((image) => image.dataset.page), ["2", "5"]);
});

test("rendering records a picture's file, and Compare reads the Base version only when it has to", () => {
  const values = new Map<string, string>([["src", ".markitai/assets/e4b7a272af48e4ca2e7a6de9.jpg"], ["alt", "Bar chart of three columns"]]);
  const image = { getAttribute: (name: string) => values.get(name) ?? null, setAttribute: (name: string, value: string) => void values.set(name, value), removeAttribute: (name: string) => void values.delete(name), closest: () => null };
  rewrite({ ownerDocument: { createElement: () => ({}) }, querySelectorAll: (selector: string) => (selector === "img" ? [image] : []) } as never, {
    documentPath: "orchard-report.pdf.llm.md",
    artifacts: [{ relpath: ".markitai/assets/e4b7a272af48e4ca2e7a6de9.jpg", size: 1 }],
    imageURL: (path) => `/x/${path}`,
    fileURL: (path) => `/x/${path}`,
    placeholder: (alt) => `[${alt}]`,
    fallbackAlt: "figure",
    tableLabel: "table",
    codeLabel: "code",
  });
  assert.equal(values.get("data-asset"), ".markitai/assets/e4b7a272af48e4ca2e7a6de9.jpg");
  assert.equal(values.has("data-page"), false);
  const view = readFileSync(new URL("../components/compare-view.tsx", import.meta.url), "utf8");
  // only a PDF, only the LLM version of a pair, only while one of its pictures has no page
  assert.match(view, /const basePath = doc\.pair && doc\.shown === "llm" \? doc\.pair\.base : null;/);
  assert.match(view, /const wantBase = basePath !== null && shownPages !== null && unplacedPictures\(shownPages\) > 0;/);
  assert.match(view, /if \(root && shownPages\) placePictures\(root, shownPages, basePages\);\s*setTargets\(root \? targetsOf\(root\) : \[\]\);/);
});
