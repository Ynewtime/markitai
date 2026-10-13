import assert from "node:assert/strict";
import test from "node:test";
import { pairBlocks, type PageBlock, type Target } from "./compare.ts";
import { errorPlace, firstError, jsonProblem, lineColumn } from "./config-text.ts";
import { displayName, readable } from "./format.ts";
import { remotePicture } from "./markdown.ts";
import { ADMIN_SECTIONS, parseRoute, routePath } from "./route.ts";
import { bounded } from "../hooks/use-conversions.ts";

test("escapes read as their characters, for display only", () => {
  assert.equal(readable("20260314-%E5%B1%B1%E9%97%B4weather.md"), "20260314-山间weather.md");
  assert.equal(displayName("https://example.com/posts/20260314-%E5%B1%B1%E9%97%B4/"), "example.com/posts/20260314-山间/");
  // Lower-case hex, spaces and mixed text.
  assert.equal(readable("a%20b%e6%97%a5"), "a b日");
  // Invalid sequences stay as written: a lone continuation byte, a cut sequence, an overlong form, a surrogate.
  assert.equal(readable("x%80y"), "x%80y");
  assert.equal(readable("%E5%89"), "%E5%89");
  assert.equal(readable("%E5%B1%B1%E5"), "山%E5");
  assert.equal(readable("%C0%AF"), "%C0%AF");
  assert.equal(readable("%ED%A0%80"), "%ED%A0%80");
  // What would change how a path reads stays escaped: separators, the escape mark, controls, invisible and bidi marks.
  assert.equal(readable("a%2Fb%5Cc%3Fd%23e%25f"), "a%2Fb%5Cc%3Fd%23e%25f");
  assert.equal(readable("x%00%0A%7F"), "x%00%0A%7F");
  assert.equal(readable("file%E2%80%AEgnp.exe"), "file%E2%80%AEgnp.exe");
  assert.equal(readable("a%E2%80%8Bb%EF%BB%BF"), "a%E2%80%8Bb%EF%BB%BF");
  // No escapes, or a stray "%": unchanged.
  assert.equal(readable("100% sure.pdf"), "100% sure.pdf");
  assert.equal(readable("%zz%4"), "%zz%4");
});

test("only https pictures without user information load from the web", () => {
  assert.equal(remotePicture("https://example.com/a.png"), "https://example.com/a.png");
  assert.equal(remotePicture(" https://EXAMPLE.com/a b.png "), "https://example.com/a%20b.png");
  for (const source of ["http://example.com/a.png", "data:image/png;base64,AAAA", "//example.com/a.png", "assets/a.png", "https://u:p@example.com/a.png", "javascript:alert(1)", "https://"]) {
    assert.equal(remotePicture(source), null, source);
  }
});

const block = (tag: string, text: string, src?: string): PageBlock => (src === undefined ? { tag, text } : { tag, text, src });
const target = (id: string, kind: Target["kind"], text: string, extra: Partial<Target> = {}): Target => ({ id, kind, text: [text], ...extra });

test("a framed page's blocks pair with the Markdown's by their words, in order", () => {
  const blocks = [
    block("div", "Hillside Notes 首页 文章 关于"),
    block("h1", "山间小站的一年"),
    block("p", "2026-03-14 · 虚构作者"),
    block("blockquote", "这是一篇为测试编写的虚构文章。"),
    block("p", "这是一篇为测试编写的虚构文章。"),
    block("div", "去年秋天，我把一台闲置的旧笔记本电脑改造成了家庭气象站。它每五分钟读取一次温度。"),
    block("img", "山峦与小屋的插图", "https://cdn.example/pic1.png"),
    block("h2", "硬件清单"),
    block("li", "旧笔记本电脑一台"),
    block("li", "传感器两个"),
    block("table", "a b"),
    block("h2", "后记"),
    block("p", "第一段。第二段，在页面上与第一段同在一个元素里。"),
    block("img", "月度湿度柱状图", "https://cdn.example/pic2.png"),
  ];
  const targets = [
    target("t0", "heading", "山间小站的一年"),
    target("t1", "block", "这是一篇为测试编写的虚构文章。"),
    target("t2", "block", "去年秋天，我把一台闲置的旧笔记本电脑改造成了家庭气象站。它每五分钟读取一次温度。"),
    target("t3", "image", "插图", { src: "https://cdn.example/pic1.png" }),
    target("t4", "heading", "硬件清单"),
    target("t5", "block", "传感器两个"),
    target("t6", "table", "a"),
    target("t7", "block", "第二段，在页面上与第一段同在一个元素里。"),
    target("t8", "image", "月度湿度柱状图"),
    target("t9", "block", "页面上没有的一段话"),
    target("t10", "heading", "后记"),
  ];
  const pairs = pairBlocks(targets, blocks);
  assert.equal(pairs.get("t0"), 1, "heading by its words");
  assert.equal(pairs.get("t1"), 3, "the first block with the same words");
  assert.equal(pairs.get("t2"), 5, "a container with its own words");
  assert.equal(pairs.get("t3"), 6, "a picture by its address");
  assert.equal(pairs.get("t4"), 7);
  assert.equal(pairs.get("t5"), 9, "searched forward, not the earlier item");
  assert.equal(pairs.get("t6"), 10, "tables by order");
  assert.equal(pairs.get("t7"), 12, "the smallest block holding the words");
  assert.equal(pairs.get("t8"), 13, "a picture by its description when no address matches");
  assert.equal(pairs.has("t9"), false);
  assert.equal(pairs.get("t10"), 11, "found again from the start once the cursor passed it");
});

test("the config text check names the line and column of a mistake", () => {
  const words = { configNotJson: (line: number, column: number) => `L${line}C${column}`, configNotJsonAnywhere: "bad", configNotObject: "object" };
  assert.equal(jsonProblem('{\n  "a": 1\n}\n', words), null);
  assert.equal(jsonProblem("[]", words), "object");
  assert.equal(jsonProblem("null", words), "object");
  assert.match(jsonProblem('{\n  "a": 1\n  "b": 2\n}', words) ?? "", /^L3C\d+$/);
  assert.deepEqual(lineColumn("ab\ncd", 4), { line: 2, column: 2 });
  // Messages without a position still get one from the text itself.
  assert.equal(firstError('{"a": [1, 2], "b": {"c": "\\u00e9"}}'), -1);
  assert.equal(firstError('{\n  "enabled": tru\n}'), 18);
  assert.equal(firstError('{"a": 1,}'), 8);
  assert.equal(firstError('{"a": "x\ny"}'), 8);
  assert.equal(firstError("[1, 2] 3"), 7);
  assert.equal(firstError('{"a": 01}'), 7);
  assert.deepEqual(errorPlace("Unexpected token in JSON at position 4", "ab\ncd"), { line: 2, column: 2 });
  assert.deepEqual(errorPlace("Expected ',' (line 7 column 3)", ""), { line: 7, column: 3 });
});

test("bounded work keeps the limit and the order of results", async () => {
  let running = 0;
  let peak = 0;
  const results = await bounded([5, 1, 4, 2, 3, 0, 6], 3, async (n) => {
    running++;
    peak = Math.max(peak, running);
    await new Promise((resolve) => setTimeout(resolve, n));
    running--;
    return n * 10;
  });
  assert.deepEqual(results, [50, 10, 40, 20, 30, 0, 60]);
  assert.ok(peak <= 3);
  assert.deepEqual(await bounded([], 4, async () => 1), []);
});

test("settings: Appearance first, Edit config has its own address, the rest is the admin's", () => {
  assert.deepEqual(parseRoute("/settings/config"), { view: "settings", section: "config" });
  assert.equal(routePath({ view: "settings", section: "config" }), "/settings/config");
  assert.deepEqual([...ADMIN_SECTIONS].sort(), ["access", "config", "models"]);
});
