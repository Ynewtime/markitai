// Guard the colour pairs the workbench actually draws: every text colour on
// every background it sits on, in the light and dark palettes, must keep the
// WCAG 2.2 AA minimum (4.5:1 for text, 3:1 for large text and for icons and
// other glyphs that carry meaning). The pairs below were found by auditing the
// rendered page (every view, both themes, hover states, 1440 and 375 px); the
// palettes are read from the token blocks of app.css, so a token change that
// breaks a pair fails here. Disabled controls and the logo are exempt
// (WCAG 1.4.3) and not listed.
//
//   node scripts/check-contrast.mjs [--table] [path/to/app.css]
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const table = args.includes("--table");
const file = args.find((arg) => !arg.startsWith("--")) ?? fileURLToPath(new URL("../src/styles/app.css", import.meta.url));

/** The declarations of the first rule whose selector is exactly `selector`. */
function block(css, selector) {
  const at = css.indexOf(`${selector} {`);
  if (at < 0) throw new Error(`no ${selector} block in ${file}`);
  const start = css.indexOf("{", at) + 1;
  const body = css.slice(start, css.indexOf("}", start));
  return new Map([...body.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((match) => [match[1], match[2].trim().replace(/\s+/g, " ")]));
}

/** Light and dark token maps; dark inherits what it does not override. */
export function palettes(source) {
  const css = source.replace(/\/\*[\s\S]*?\*\//g, "");
  const light = block(css, ":root");
  const auto = block(css, ':root:not([data-theme="light"])');
  const dark = block(css, ':root[data-theme="dark"]');
  const differences = [...new Set([...auto.keys(), ...dark.keys()])].filter((name) => auto.get(name) !== dark.get(name));
  return { light, dark: new Map([...light, ...dark]), differences };
}

const hex = (value) => {
  const digits = value.slice(1);
  const full = digits.length === 3 ? [...digits].map((d) => d + d).join("") : digits;
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)).concat(1);
};

/** Resolve a colour value (hex, var(), color-mix in srgb, transparent) to RGBA 0–255/0–1. */
export function resolve(value, tokens, depth = 0) {
  if (depth > 16) throw new Error(`token cycle at ${value}`);
  const text = value.trim();
  if (text === "transparent") return [0, 0, 0, 0];
  if (/^#[\da-f]{3}([\da-f]{3})?$/i.test(text)) return hex(text);
  const variable = /^var\((--[\w-]+)\)$/.exec(text);
  if (variable) {
    const target = tokens.get(variable[1]);
    if (target === undefined) throw new Error(`undeclared ${variable[1]}`);
    return resolve(target, tokens, depth + 1);
  }
  const mix = /^color-mix\(in srgb, (.+?)(?: (\d+(?:\.\d+)?)%)?, (.+?)(?: (\d+(?:\.\d+)?)%)?\)$/.exec(text);
  if (mix) {
    const a = resolve(mix[1], tokens, depth + 1);
    const b = resolve(mix[3], tokens, depth + 1);
    const p = mix[2] !== undefined ? Number(mix[2]) / 100 : mix[4] !== undefined ? 1 - Number(mix[4]) / 100 : 0.5;
    // Premultiplied interpolation, as CSS Color 5 specifies.
    const alpha = a[3] * p + b[3] * (1 - p);
    if (alpha === 0) return [0, 0, 0, 0];
    const channel = (i) => (a[i] * a[3] * p + b[i] * b[3] * (1 - p)) / alpha;
    return [channel(0), channel(1), channel(2), alpha];
  }
  throw new Error(`unsupported colour ${text}`);
}

/** Paint layers (top first) over an opaque base. */
export function composite(layers, tokens) {
  let result = [255, 255, 255, 1];
  for (const layer of [...layers].reverse()) {
    const top = resolve(layer, tokens);
    result = [0, 1, 2].map((i) => top[i] * top[3] + result[i] * (1 - top[3])).concat(1);
  }
  return result;
}

const linear = (c) => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
const luminance = (c) => 0.2126 * linear(c[0]) + 0.7152 * linear(c[1]) + 0.0722 * linear(c[2]);
export function ratio(fg, bg) {
  const [high, low] = [luminance(fg), luminance(bg)].sort((x, y) => y - x);
  return (high + 0.05) / (low + 0.05);
}
const toHex = (c) => `#${c.slice(0, 3).map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;

// [text colour, background layers (top first; the last one opaque), minimum, where it is drawn].
// 4.5 is body text; 3 is large text (≥ 24 px, or ≥ 18.66 px bold) and meaningful glyphs.
const T = 4.5;
const G = 3;
export const PAIRS = [
  ["var(--text-1)", ["var(--bg)"], T, "page text, headline, ledger names"],
  ["var(--text-1)", ["var(--surface)"], T, "cards, dialogs, rendered preview"],
  ["var(--text-1)", ["var(--strip)"], T, "hovered tools and menu items, selected chips"],
  ["var(--text-1)", ["var(--row-hover)"], T, "hovered ledger rows"],
  ["var(--text-1)", ["color-mix(in srgb, var(--accent) 10%, var(--surface))"], T, "selected segment and chip"],
  ["var(--text-2)", ["var(--bg)"], T, "secondary text on the page"],
  ["var(--text-2)", ["var(--surface)"], T, "secondary text in cards and dialogs"],
  ["var(--text-2)", ["var(--row-hover)"], T, "secondary text in hovered rows"],
  ["var(--text-2)", ["color-mix(in srgb, var(--accent) 7%, transparent)", "var(--surface)"], T, "Convert button"],
  ["var(--text-3)", ["var(--bg)"], T, "ledger facts, counters, separators"],
  ["var(--text-3)", ["var(--surface)"], T, "hints, metadata, diff line numbers"],
  ["var(--text-3)", ["var(--strip)"], T, "provider card and link hover (was 4.40:1)"],
  ["var(--text-3)", ["var(--row-hover)"], T, "facts in hovered rows"],
  ["var(--text-3)", ["var(--diff-add-bg)"], T, "line numbers of added lines (was --text-4)"],
  ["var(--text-3)", ["var(--diff-del-bg)"], T, "line numbers and marks of removed lines"],
  ["var(--text-1)", ["var(--diff-add-bg)"], T, "added lines"],
  ["var(--text-1)", ["var(--diff-del-bg)"], T, "removed lines"],
  ["var(--accent-contrast)", ["var(--accent)"], T, "primary buttons"],
  ["var(--accent-contrast)", ["var(--accent-hover)"], T, "hovered primary buttons"],
  ["var(--selection-fg)", ["var(--selection-bg)"], T, "text selection"],
  ["var(--danger-on-fill)", ["var(--danger-fill)"], T, "Delete in a confirmation card"],
  ["var(--danger-on-fill)", ["var(--danger-fill-hover)"], T, "hovered Delete"],
  ["var(--err)", ["var(--bg)"], T, "error lines on the page"],
  ["var(--err)", ["var(--surface)"], T, "error lines in cards and dialogs"],
  ["var(--err)", ["var(--row-hover)"], T, "failed-row cause in a hovered row"],
  ["var(--warning)", ["var(--bg)"], T, "conversion warnings in the ledger"],
  ["var(--warning)", ["var(--surface)"], T, "warnings in the preview"],
  ["var(--warning)", ["var(--row-hover)"], T, "warnings in hovered rows"],
  ["var(--success)", ["var(--bg)"], T, "done mark, LLM tag"],
  ["var(--success)", ["var(--surface)"], T, "test passed, success notices"],
  ["var(--success)", ["var(--row-hover)"], T, "done mark and LLM tag in hovered rows"],
  ["var(--err)", ["var(--strip)"], G, "hovered delete icon"],
  ["var(--err)", ["color-mix(in srgb, var(--err) 10%, var(--surface))"], G, "confirmation card icon"],
  ["var(--text-1)", ["color-mix(in srgb, var(--text-1) 6%, transparent)", "var(--surface)"], T, "hovered text buttons (Test, Edit)"],
  ["var(--text-1)", ["color-mix(in srgb, var(--warning) 13%, var(--surface))"], T, "hovered notice action (warning)"],
  ["var(--text-1)", ["color-mix(in srgb, var(--err) 13%, var(--surface))"], T, "hovered notice action (error)"],
  ["color-mix(in srgb, var(--text-1) 70%, transparent)", ["var(--bg)"], T, "hovered brand name (70% opacity)"],
  ["var(--t-text)", ["var(--t-bg)"], T, "CLI line, Source card"],
  ["var(--t-dim)", ["var(--t-bg)"], T, "dimmed front matter, CLI prompt"],
  ["var(--t-badge-text)", ["var(--t-bg)"], T, "Copy pill, status pills, active filter chip"],
  ["var(--t-green)", ["var(--t-bg)"], T, "status pills: done"],
  ["var(--t-red)", ["var(--t-bg)"], T, "status pills: failed"],
  ["var(--t-amber)", ["var(--t-bg)"], T, "status pills: skipped"],
];

export function check(source) {
  const { light, dark, differences } = palettes(source);
  const problems = differences.map((name) => `${name} differs between the automatic and the explicit dark palette`);
  const rows = [];
  for (const [fg, layers, minimum, use] of PAIRS) {
    const cells = [];
    for (const [theme, tokens] of [
      ["light", light],
      ["dark", dark],
    ]) {
      const background = composite(layers, tokens);
      const foreground = composite([fg, ...layers], tokens);
      const value = ratio(foreground, background);
      if (value < minimum) problems.push(`${theme}: ${fg} on ${layers.join(" over ")} is ${value.toFixed(2)}:1, below ${minimum}:1 (${use})`);
      cells.push({ value, fg: toHex(foreground), bg: toHex(background) });
    }
    rows.push({ fg, layers, minimum, use, cells });
  }

  if (table) {
    const name = (value) => value.replace(/^var\((--[\w-]+)\)$/, "$1").replace(/var\((--[\w-]+)\)/g, "$1");
    console.log("| Text | Background | Minimum | Light | Dark | Where |");
    console.log("|---|---|---|---|---|---|");
    for (const row of rows) {
      const [l, d] = row.cells;
      console.log(
        `| \`${name(row.fg)}\` | ${row.layers.map((layer) => `\`${name(layer)}\``).join(" over ")} | ${row.minimum}:1 | ${l.value.toFixed(2)} (${l.fg} on ${l.bg}) | ${d.value.toFixed(2)} (${d.fg} on ${d.bg}) | ${row.use} |`,
      );
    }
  }
  if (problems.length) {
    console.error(`Contrast check failed:\n  ${problems.join("\n  ")}`);
    process.exit(1);
  }
  if (!table) console.log(`Contrast check passed (${PAIRS.length} pairs, light and dark)`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) check(readFileSync(file, "utf8"));
