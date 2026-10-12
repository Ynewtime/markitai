import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const css = readFileSync(new URL("../styles/app.css", import.meta.url), "utf8");

/** The declarations of the first top-level rule with exactly this selector. */
function rule(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = new RegExp(`(?:^|\\n)${escaped} \\{\\n([^}]*)\\}`).exec(css);
  assert.ok(match, `no rule for ${selector}`);
  return match[1] ?? "";
}

test("the window's reading column is centred at its reading width", () => {
  // Rendered, the skeleton in its frame, Source, Files and versions share one column.
  for (const selector of [".doc", ".skeleton-doc", ".source", ".file-list", ".diff"]) {
    const body = rule(selector);
    assert.match(body, /max-width: 820px;/, selector);
    assert.match(body, /margin: 0 auto;/, selector);
  }
  // A pane's own note follows the column; the list's loading note stays where it is.
  assert.match(rule(".pane > .pane-note"), /margin-inline: auto;/);
  assert.ok(!/margin/.test(rule(".pane-note")));
});

test("the window header's counter centres its figures on the header's line", () => {
  // A platform monospace box centres its figures higher than Inter's; the counter uses Inter.
  const body = rule(".peek-pos");
  assert.ok(!/font-family/.test(body));
  assert.match(body, /font-variant-numeric: tabular-nums;/);
  assert.match(rule(".peek-head"), /align-items: center;/);
});
