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
  // Rendered, the skeleton in its frame, Source, Files and versions share one column: 820 px,
  // wider as the window grows.
  assert.match(css, /--read-width: clamp\(820px, 56vw, 1120px\);/);
  for (const selector of [".doc", ".skeleton-doc", ".source", ".file-list", ".diff"]) {
    const body = rule(selector);
    assert.match(body, /max-width: var\(--read-width\);/, selector);
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

test("the window header's name and icons sit on the arrows' line", () => {
  // 48 px with its 1 px rule: the line is on a half pixel, and text and icons are drawn from
  // whole ones. The name's box stays centred while its text rises 1.5 px (1 px at 1x); the
  // icons are odd sizes. Measured with Playwright ink boxes, en/zh, light/dark.
  const head = rule(".peek-head");
  assert.match(head, /min-height: 48px;/);
  assert.match(head, /border-bottom: 1px solid/);
  assert.match(rule(".peek:not(.is-phone) .peek-title"), /padding-bottom: 2px;/);
  assert.match(css, /@media \(min-resolution: 2dppx\) \{\n  \.peek:not\(\.is-phone\) \.peek-title > \.mid-name \{\n    transform: translateY\(-0\.5px\);/);
  assert.match(rule(".peek:not(.is-phone) .peek-head .action-menu > .icon-btn > svg"), /width: 17px;\n  height: 17px;/);
  const peek = readFileSync(new URL("../components/peek.tsx", import.meta.url), "utf8");
  const header = peek.slice(peek.indexOf('<header class="peek-head">'), peek.indexOf("</header>"));
  const desktop = [...header.matchAll(/!phone && \([\s\S]*?\n *\)\}/g)].map((block) => block[0]).join("");
  const sizes = [...desktop.matchAll(/size=\{(\d+)\}/g)].map((size) => Number(size[1]));
  assert.ok(sizes.length >= 4 && sizes.every((size) => size % 2 === 1), `desktop header icon sizes ${sizes}`);
});

test("on a phone the tap area of Full details stays below the clickable notice names", () => {
  // The names sit right above the summary: its extra area upward must fit in the gap between them.
  const gap = /\n  \.notice-details \{\n    margin-top: (\d+)px;/.exec(css);
  const area = /\.notice-details summary::before \{\n[^}]*inset: -(\d+)px -4px -(\d+)px;/.exec(css);
  assert.ok(gap && area, "the phone rules exist");
  assert.ok(Number(area[1]) <= Number(gap[1]), "the area above the summary reaches no further than the gap");
  // Together with the line itself it is still a finger's height (44 px): 18 px of text and the margins.
  assert.ok(Number(area[1]) + Number(area[2]) + 18 >= 44);
});
