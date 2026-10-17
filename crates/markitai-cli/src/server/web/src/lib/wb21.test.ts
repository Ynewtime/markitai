import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { mathTitle } from "./math-syntax.ts";
import { entryOnly, listEntry } from "./route.ts";

const css = readFileSync(new URL("../styles/app.css", import.meta.url), "utf8");
const page = readFileSync(new URL("../components/conversions-page.tsx", import.meta.url), "utf8");

test("a history entry narrows the list only with a list of item keys", () => {
  assert.equal(listEntry(null), null);
  assert.equal(listEntry("dialog"), null);
  assert.deepEqual(entryOnly(listEntry({ wb: "only", only: ["a/i1", "b/i2"] })), ["a/i1", "b/i2"]);
  assert.equal(entryOnly(listEntry({ wb: "summary" })), null);
  assert.equal(entryOnly(listEntry({ only: ["a/i1", 3] })), null);
  assert.equal(entryOnly(null), null);
});

test("Show items narrows the list with a bar over it, not a chip in the filter row", () => {
  assert.ok(!css.includes(".subset-chip"));
  assert.match(css, /\n\.subset-bar \{/);
  // the bar counts the narrowed items still listed, not the keys it was given
  assert.match(page, /showingItems\(onlyLeft\)/);
  assert.ok(!page.includes("showingItems(props.only.length)"));
});

test("a row's notice mark stands on its name's line", () => {
  // a monospace figure rode ~3 px above the name; the Inter figure moves down the 1 px its smaller size leaves
  const mark = /\n\.warn-mark \{\n([^}]*)\}/.exec(css)?.[1] ?? "";
  assert.match(mark, /font-family: var\(--font-sans\);/);
  assert.match(/\n\.warn-wrap \{\n([^}]*)\}/.exec(css)?.[1] ?? "", /top: 1px;/);
});

test("contents entries keep to one line, the whole heading in their tooltip", () => {
  const text = /\n\.toc-text \{\n([^}]*)\}/.exec(css)?.[1] ?? "";
  assert.match(text, /white-space: nowrap;/);
  assert.match(text, /text-overflow: ellipsis;/);
  assert.ok(!text.includes("line-clamp"));
  assert.equal(mathTitle("The probability of getting $k$ heads when flipping $n$ coins"), "The probability of getting k heads when flipping n coins");
  assert.equal(mathTitle("Plain heading"), "Plain heading");
  assert.equal(mathTitle("Area $$\\pi r^2$$"), "Area \\pi r^2");
});
