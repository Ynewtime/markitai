// 2.0.1 owner items on the release candidate (2026-10-25): the options dialog's command scrolls
// instead of being cut.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), "utf8");
const css = read("../styles/app.css");
const block = (selector: string) => new RegExp(`\\n${selector.replace(/[.[\]()"=:>]/g, "\\$&")} \\{\\n([^}]*)\\}`).exec(css)?.[1] ?? "";

test("the dialog's command is one line that scrolls in its box, fading while there is more", () => {
  const line = block(".sheet-cli .cli-text");
  assert.match(line, /overflow-x: auto;/);
  assert.match(line, /scrollbar-width: none;/);
  assert.match(line, /white-space: nowrap;/);
  assert.ok(!/ellipsis|overflow: hidden/.test(line));
  assert.match(block(".sheet-cli .cli-text.is-more"), /mask-image: linear-gradient\(to right/);
  // Copy keeps its place at the box's right end
  assert.match(block(".sheet-cli .icon-btn"), /flex: none;/);
  const dialog = read("../components/options-dialog.tsx");
  assert.match(dialog, /<code ref=\{cli\} class="cli-text"[^>]*onScroll=\{\(event\) => markMore\(event\.currentTarget\)\}/);
});
