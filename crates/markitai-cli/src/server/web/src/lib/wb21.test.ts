import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
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
