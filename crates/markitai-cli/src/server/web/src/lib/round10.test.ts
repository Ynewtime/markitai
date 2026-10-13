import assert from "node:assert/strict";
import test from "node:test";
import { locate, norm, type PageLines } from "./compare.ts";
import { addStaged, removeStaged } from "./files.ts";
import { cardFacts, openingHeading, parseFrontMatter } from "./frontmatter.ts";
import { headingLabel, pdfImagePage, planTokens, type Token } from "./markdown.ts";
// The vendored parser the page loads as /ui/marked.js.
import { marked } from "../../../../../../../vendor/web/marked.js";

const lex = (markdown: string) => marked.lexer(markdown, { gfm: true }) as unknown as Token[];

test("contents list a heading's words without inline tags, private-use glyphs or runs of space", () => {
  const markdown = [
    "# 示例条款",
    "## <sup>申请人享有的主要权利</sup>",
    "## **Bold**  and `code`  <span>inside</span>",
    "Line one\nwraps\n---",
    "## ",
  ].join("\n\n");
  const labels = planTokens(lex(markdown)).headings.map((heading) => heading.text);
  assert.deepEqual(labels, ["示例条款", "申请人享有的主要权利", "Bold and code inside", "Line one wraps", ""]);
  const [heading] = lex("## A&amp;B <sup>2</sup>");
  assert.equal(headingLabel(heading as Token), "A&B 2");
});

test("the card above a document leaves out a title the document opens with", () => {
  const front = parseFrontMatter('---\ntitle: "Reflections on the *team*"\ndescription: A note\n---');
  assert.equal(openingHeading("\n# Reflections on the team\n\nText"), "Reflections on the team");
  assert.equal(openingHeading("Reflections\n===\n"), "Reflections");
  assert.equal(openingHeading("## Not a title\n"), null);
  assert.equal(openingHeading("Text first\n\n# Later"), null);
  // Same words: no title, the description stays.
  assert.deepEqual(cardFacts(front, "# Reflections on the **team**\n\nBody"), { ...front, title: null });
  // Another first heading keeps the title.
  assert.equal(cardFacts(front, "# Something else\n")?.title, "Reflections on the *team*");
  // Nothing left to show: no card.
  assert.equal(cardFacts(parseFrontMatter("---\ntitle: Only\n---"), "# Only\n"), null);
  assert.equal(cardFacts(null, "# x"), null);
});

test("chosen files wait for Convert, join what is staged once each, and come off one at a time", () => {
  const file = (path: string, size = 1) => Object.assign(new File(["x".repeat(size)], path.split("/").pop() ?? path, { lastModified: 1 }), { markitaiPath: path });
  const a = file("a.pdf");
  const first = addStaged(null, [a], false, 10);
  assert.deepEqual(first, { staged: { files: [a], label: "", kind: "files" }, dropped: 0 });
  // The same file again adds nothing; another joins.
  assert.equal(addStaged(first.staged, [file("a.pdf")], false, 10).staged, first.staged);
  const b = file("b.png", 3);
  const both = addStaged(first.staged, [b], false, 10).staged;
  assert.deepEqual(both?.files, [a, b]);
  // A folder keeps its name until something else joins it.
  const folder = addStaged(null, [file("top/x.md"), file("top/y.md")], true, 10).staged;
  assert.equal(folder?.kind, "folder");
  assert.equal(folder?.label, "top");
  const mixed = addStaged(folder, [a], false, 10).staged;
  assert.deepEqual([mixed?.kind, mixed?.label, mixed?.files.length], ["files", "", 3]);
  // Over the limit, the rest are counted, not staged.
  const capped = addStaged(both, [file("c.md"), file("d.md")], false, 3);
  assert.deepEqual([capped.staged?.files.length, capped.dropped], [3, 1]);
  // Remove one; the last one leaves nothing staged.
  assert.deepEqual(removeStaged(both!, a)?.files, [b]);
  assert.equal(removeStaged(first.staged!, a), null);
  const list = addStaged(null, [file("links.urls"), b], false, 10).staged!;
  assert.equal(list.kind, "urls");
  assert.deepEqual([removeStaged(list, list.files[0]!)?.kind, removeStaged(list, list.files[0]!)?.label], ["files", ""]);
});

test("Compare locates paragraphs and list items too: across lines, inside a line, over a page's end", () => {
  const pages: PageLines[] = [
    {
      page: 1,
      lines: [
        { text: "示例数据服务股份有限公司 电子回单", bbox: [40, 30, 560, 60] },
        { text: "面积单位：平方千米", bbox: [40, 70, 200, 84] },
        { text: "办理须知", bbox: [40, 100, 120, 116] },
        { text: "1. 本登记每次收费标准变动将于本中心服务窗口", bbox: [40, 120, 520, 132] },
        { text: "公布，公布满十五日后正式生效。", bbox: [40, 134, 300, 146] },
        { text: "友情提醒", bbox: [40, 160, 120, 176] },
        { text: "本电子回单是申请用户向服务机构查询的依据，请", bbox: [40, 180, 520, 192] },
      ],
    },
    { page: 2, lines: [{ text: "妥善留存。", bbox: [40, 30, 120, 42] }] },
  ];
  const found = locate(pages, [
    { id: "title", kind: "block", text: ["电子回单"] },
    { id: "unit", kind: "block", text: ["面积单位：平方千米"] },
    { id: "h", kind: "heading", text: ["办理须知"] },
    { id: "item", kind: "block", text: ["本登记每次收费标准变动将于本中心服务窗口公布，公布满十五日后正式生效。"] },
    { id: "tip", kind: "block", text: ["本电子回单是申请用户向服务机构查询的依据，请妥善留存。"] },
    { id: "absent", kind: "block", text: ["这段文字在原件中不存在。"] },
  ]);
  // Part of a line: the end of the logo line, by proportion.
  const title = found.get("title");
  assert.equal(title?.page, 1);
  assert.ok((title?.box[0] ?? 0) > 300 && title?.box[2] === 560, JSON.stringify(title));
  assert.deepEqual(found.get("unit"), { page: 1, box: [40, 70, 200, 84] });
  assert.deepEqual(found.get("h"), { page: 1, box: [40, 100, 120, 116] });
  // Two lines, the first one after its list number.
  assert.deepEqual(found.get("item"), { page: 1, box: [40, 120, 520, 146] });
  // Cut by the page's end: the part on page 1.
  assert.deepEqual(found.get("tip"), { page: 1, box: [40, 180, 520, 192] });
  assert.equal(found.has("absent"), false);
});

test("a PDF's pictures are found on their page, in reading order, when the page has as many", () => {
  assert.equal(pdfImagePage("Image on page 3", ".markitai/assets/0a1b2c.jpg"), 3);
  assert.equal(pdfImagePage("", ".markitai/assets/report.pdf-0012-01.png"), 12);
  assert.equal(pdfImagePage("A chart", "assets/chart.png"), null);
  assert.equal(pdfImagePage("Image on page 0", "x.png"), null);
  const pages: PageLines[] = [
    { page: 1, lines: [], images: [[300, 40, 500, 90], [40, 40, 200, 90], [40, 400, 560, 700]] },
    { page: 2, lines: [], images: [[40, 40, 200, 90]] },
  ];
  const found = locate(pages, [
    { id: "a", kind: "image", text: [], page: 1 },
    { id: "b", kind: "image", text: [], page: 1 },
    { id: "c", kind: "image", text: [], page: 1 },
    // Page 2 draws one picture where the Markdown has two: neither is guessed.
    { id: "d", kind: "image", text: [], page: 2 },
    { id: "e", kind: "image", text: [], page: 2 },
    // A server that reports no pictures.
    { id: "f", kind: "image", text: [], page: 3 },
  ]);
  assert.deepEqual([found.get("a")?.box, found.get("b")?.box, found.get("c")?.box], [[40, 40, 200, 90], [300, 40, 500, 90], [40, 400, 560, 700]]);
  assert.deepEqual([found.has("d"), found.has("e"), found.has("f")], [false, false, false]);
});

test("Compare reads Kangxi and supplement radicals as the characters they stand for", () => {
  // As one PDF extraction spells 申请人：人民币 日期.
  assert.equal(norm("申请\u2F08：\u2F08\u2EA0币 \u2F47期"), "申请人人民币日期");
  assert.equal(norm("\u2F08\u2EA0币"), "人民币");
  assert.equal(norm("\u2ED4\u2EDA"), "门页");
});
