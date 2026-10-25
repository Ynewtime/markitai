// Compare's outline (2.1.0): a hovered block's outline paints inside its chunk, so moving it never changes a
// chunk's containment (lifting it while a block was outlined left the old outline's edges on screen).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const css = readFileSync(new URL("../styles/app.css", import.meta.url), "utf8");
const block = (selector: string) => new RegExp(`\\n${selector.replace(/[.+]/g, "\\$&")} \\{\\n([^}]*)\\}`).exec(css)?.[1] ?? "";
const px = (rules: string, property: string) => Number(new RegExp(`\\n?\\s*${property}: (-?\\d+)px`).exec(rules)?.[1] ?? NaN);

test("hovering never changes a chunk's containment", () => {
  assert.ok(!/:has\(\.is-linked\)/.test(css));
  assert.match(block(".doc-chunk,\n.source-block,\n.diff-block"), /content-visibility: auto;/);
});

test("a chunk's box holds the outlines of its blocks, and its blocks stay where they were", () => {
  const pad = px(block(".doc-chunk"), "padding");
  const outline = block(".doc-body .is-linked,\n.orig-html .is-linked");
  const reach = Number(/outline: (\d+)px/.exec(outline)?.[1]) + px(outline, "outline-offset");
  assert.ok(reach > 0 && pad >= reach, `padding ${pad}px holds an outline reaching ${reach}px`);
  assert.ok(px(block(".doc-body .table-scroll.is-linked"), "outline-offset") + 2 <= pad);
  // the padding is given back: -pad around a chunk, and -2 × pad between two (their margins collapse to the larger)
  assert.equal(px(block(".doc-chunk"), "margin"), -pad);
  assert.equal(px(block(".doc-chunk + .doc-chunk"), "margin-top"), -2 * pad);
});
