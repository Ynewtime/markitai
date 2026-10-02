// Guard the stylesheet's visual scale: one type ladder, one radius family,
// three durations (plus the spinner loops), two shadows, and the reset.
// Values come from the token block itself; a new value needs a token first.
//
//   node scripts/check-css-scale.mjs [path/to/app.css]
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const file = process.argv[2] ?? fileURLToPath(new URL("../src/styles/app.css", import.meta.url));
const source = readFileSync(file, "utf8");
const css = source.replace(/\/\*[\s\S]*?\*\//g, (comment) => comment.replace(/[^\n]/g, " "));

const tokens = new Map([...css.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((match) => [match[1], match[2].trim()]));
const family = (prefix) => new Set([...tokens].filter(([name]) => name.startsWith(prefix)).map(([, value]) => value));

const TYPE = new Set([...family("--text-"), "9px"]);
for (const value of TYPE) if (!/px$/.test(value) && value !== "9px") TYPE.delete(value);
const DISPLAY = new Set(["30px", "44px"]);
const PRINT = new Set(["8pt", "9pt", "10.5pt", "12pt", "18pt"]);
const RELATIVE = new Set(["1em", "inherit"]);
const RADII = new Set([...family("--r-"), "0", "2px", "999px", "50%", "inherit"]);
const DURATIONS = new Set(family("--dur-"));
const LOOPS = new Set(["700ms", "1500ms"]);
const SHADOWS = new Set([...family("--shadow-"), "none"]);

const problems = [];
const lineOf = (index) => css.slice(0, index).split("\n").length;
const declared = new Set(tokens.keys());

for (const match of css.matchAll(/(?<![\w-])([a-z-]+)\s*:\s*([^;{}]+);/g)) {
  const [, property, raw] = match;
  const value = raw.trim().replace(/\s+/g, " ");
  const at = `${file}:${lineOf(match.index ?? 0)}`;
  for (const reference of value.matchAll(/var\((--[\w-]+)/g)) {
    if (!declared.has(reference[1])) problems.push(`${at}: ${property} uses undeclared ${reference[1]}`);
  }
  if (property === "font-size") {
    const clamp = /^clamp\(([^,]+),[^,]+,([^)]+)\)$/.exec(value);
    const ok =
      value.startsWith("var(--text-") ||
      (clamp !== null && DISPLAY.has(clamp[1].trim()) && DISPLAY.has(clamp[2].trim())) ||
      TYPE.has(value) ||
      PRINT.has(value) ||
      RELATIVE.has(value);
    if (!ok) problems.push(`${at}: font-size ${value} is off the type scale`);
  }
  if (property === "border-radius") {
    for (const part of value.split(" ")) {
      if (!part.startsWith("var(--r-") && !RADII.has(part)) problems.push(`${at}: border-radius ${part} is off the radius scale`);
    }
  }
  if (property === "box-shadow" && !value.startsWith("var(--shadow-") && !SHADOWS.has(value) && !/^(inset |0 0 0 )/.test(value)) {
    problems.push(`${at}: box-shadow "${value}" is not a shadow token`);
  }
  if (!property.startsWith("--")) {
    for (const time of value.matchAll(/(?<![\w.-])(\d*\.?\d+)(ms|s)\b/g)) {
      const ms = time[2] === "ms" ? `${Number(time[1])}ms` : `${Number(time[1]) * 1000}ms`;
      const allowed = property === "animation" || property === "animation-duration" ? LOOPS : DURATIONS;
      if (!allowed.has(ms)) problems.push(`${at}: ${property} duration ${time[0]} is off the motion scale`);
    }
  }
}

if (/@apply\b|@import\s+["']tailwindcss|@tailwind\b/.test(css)) problems.push(`${file}: utility-framework directives are not allowed`);
const reset = /(?:^|\})\s*\*\s*,[^{]*\{([^}]*)\}/m.exec(css);
if (!reset || !/box-sizing:\s*border-box/.test(reset[1]) || !/margin:\s*0/.test(reset[1])) {
  problems.push(`${file}: the universal reset (box-sizing and margin on *) is missing`);
}
for (const [pattern, label] of [
  [/\bhr\s*\{[^}]*border-top-width/, "hr rules"],
  [/\bsummary\s*\{[^}]*display:\s*list-item/, "the <summary> marker"],
  [/::placeholder\s*\{[^}]*opacity:\s*1/, "::placeholder opacity"],
  [/\[hidden\]\s*\{[^}]*display:\s*none/, "[hidden]"],
]) {
  if (!pattern.test(css)) problems.push(`${file}: the reset is missing ${label}`);
}

if (problems.length) {
  console.error(`CSS scale check failed:\n  ${problems.join("\n  ")}`);
  process.exit(1);
}
console.log(`CSS scale check passed (${TYPE.size} type sizes, ${RADII.size} radii, ${DURATIONS.size} durations, ${SHADOWS.size} shadows)`);
