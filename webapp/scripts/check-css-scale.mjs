/**
 * CSS scale guard.
 *
 * The stylesheet is the product's visual contract: one type ladder, one radius
 * family, three durations, two shadows, and the reset that replaced Tailwind's
 * preflight. Drift creeps in one hand-written value at a time, so this check
 * fails the build on any value outside the declared tokens. Add a token first,
 * then use it.
 *
 * Run: bun run check:css
 */
import { readFileSync } from "node:fs";

const FILE = new URL("../src/styles/app.css", import.meta.url);
const css = readFileSync(FILE, "utf8");

/** Declaration values are read from the token block, never hard-coded here. */
function tokenValues(names) {
  const out = new Map();
  for (const name of names) {
    const match = new RegExp(`--${name}:\\s*([^;]+);`).exec(css);
    if (match === null) throw new Error(`token --${name} is not declared`);
    out.set(`var(--${name})`, match[1].trim());
  }
  return out;
}

const TYPE_TOKENS = [
  "text-2xs",
  "text-xs",
  "text-sm",
  "text-base",
  "text-md",
  "text-lg",
  "text-xl",
  "text-2xl",
];
const RADIUS_TOKENS = ["r-xs", "r-sm", "r-term", "r-panel"];
const DURATION_TOKENS = ["dur-fast", "dur-base", "dur-slow"];
const SHADOW_TOKENS = ["shadow-pop", "shadow-modal"];

const typeScale = new Set([...tokenValues(TYPE_TOKENS).values(), "9px"]);
const radiusScale = new Set([...tokenValues(RADIUS_TOKENS).values(), "0", "2px", "999px", "50%", "inherit"]);
const durationScale = new Set(tokenValues(DURATION_TOKENS).values());
const shadowScale = new Set([...tokenValues(SHADOW_TOKENS).values(), "none"]);

/** Display type may exceed the body ladder, but only at these declared steps. */
const DISPLAY_SIZES = new Set(["30px", "44px"]);
/** The print sheet keeps its own pt ladder (physical media, not screen px). */
const PRINT_SIZES = new Set(["8pt", "9pt", "9.5pt", "10pt", "10.5pt", "12pt", "18pt", "24pt"]);
/** Relative sizes carried over verbatim from the preflight reset. */
const RESET_SIZES = new Set(["80%", "75%", "1em", "inherit"]);
/** Looping animation periods (spinners) are not transitions; declared here. */
const LOOP_DURATIONS = new Set(["700ms", "1500ms"]);

/** Values that are not lengths (e.g. a focus ring's 0 0 0 1px) stay allowed. */
const SHADOW_ALLOWED_EXTRA = /^(none|inset |0 0 0 )/;

const declaredTokens = new Set(
  [...css.matchAll(/^\s*(--[\w-]+)\s*:/gm)].map((match) => match[1]),
);
/** Token families whose members must exist: a typo would silently do nothing. */
const TOKEN_FAMILIES = ["--text-", "--r-", "--dur-", "--shadow-", "--ease-"];

/** Returns the offending token name, or null when every reference resolves. */
function undeclaredToken(value) {
  for (const family of TOKEN_FAMILIES) {
    const match = new RegExp(`${family}[\\w-]+`).exec(value);
    if (match !== null && !declaredTokens.has(match[0])) return match[0];
  }
  return null;
}

const problems = [];
const lines = css.split("\n");

/** A fluid size is fine only when both its bounds are declared steps. */
function fluidBounds(value) {
  const inner = value.slice("clamp(".length, -1);
  const parts = inner.split(",").map((part) => part.trim());
  return [parts[0], parts[parts.length - 1]];
}

lines.forEach((line, index) => {
  const at = `src/styles/app.css:${index + 1}`;
  const text = line.trim();

  const fontSize = /^font-size:\s*([^;]+);/.exec(text);
  if (fontSize !== null) {
    const value = fontSize[1].trim();
    if (value.startsWith("var(--text-")) {
      const bad = undeclaredToken(value);
      if (bad !== null) {
        problems.push(`${at}: font-size references undeclared token ${bad}`);
      }
    } else if (value.startsWith("clamp(")) {
      for (const bound of fluidBounds(value)) {
        if (!DISPLAY_SIZES.has(bound)) {
          problems.push(`${at}: fluid font-size bound ${bound} is not a declared display size`);
        }
      }
    } else if (!PRINT_SIZES.has(value) && !RESET_SIZES.has(value) && !typeScale.has(value)) {
      problems.push(`${at}: font-size ${value} is off the type scale`);
    }
  }

  const radius = /^border-radius:\s*([^;]+);/.exec(text);
  if (radius !== null) {
    const parts = radius[1].split(/\s+/);
    for (const part of parts) {
      if (part.startsWith("var(--r-")) {
        const bad = undeclaredToken(part);
        if (bad !== null) {
          problems.push(`${at}: border-radius references undeclared token ${bad}`);
        }
      } else if (!radiusScale.has(part)) {
        problems.push(`${at}: border-radius ${part} is off the radius scale`);
      }
    }
  }

  // `s` counts as well: a 0.7s spin is as much a motion value as 140ms.
  const duration = /(?<![\w.])([\d.]+)(ms|s)\b/.exec(text);
  if (duration !== null && !/^--(dur|ease)/.test(text)) {
    const ms = duration[2] === "ms" ? duration[1] : String(Number(duration[1]) * 1000);
    const allowed = text.startsWith("animation:") ? LOOP_DURATIONS : durationScale;
    if (!allowed.has(`${ms}ms`)) {
      problems.push(`${at}: duration ${duration[0]} is off the motion scale`);
    }
  }

  const shadow = /^box-shadow:\s*([^;]+);/.exec(text);
  if (shadow !== null) {
    const value = shadow[1].trim();
    if (value.startsWith("var(--shadow-")) {
      const bad = undeclaredToken(value);
      if (bad !== null) {
        problems.push(`${at}: box-shadow references undeclared token ${bad}`);
      }
    } else if (!shadowScale.has(value) && !SHADOW_ALLOWED_EXTRA.test(value)) {
      problems.push(`${at}: box-shadow "${value}" is not a shadow token`);
    }
  }
});

// The sheet must not reintroduce a utility framework (comments may name it).
const withoutComments = css.replace(/\/\*[\s\S]*?\*\//g, "");
if (/@import\s+["']tailwindcss["']/.test(withoutComments) || /@apply\b/.test(withoutComments)) {
  problems.push("src/styles/app.css: utility-framework directives are not allowed");
}

// Removing the framework also removed its preflight. The sheet must carry
// the reset itself: without box-sizing and a zeroed margin the UA defaults
// return and the shell overflows its container.
const universalReset = /(?:^|\})\s*(\*[^{}]*)\{([^{}]*)\}/m.exec(withoutComments);
if (
  universalReset === null ||
  !/box-sizing:\s*border-box/.test(universalReset[2]) ||
  !/margin:\s*0/.test(universalReset[2])
) {
  problems.push(
    "src/styles/app.css: the universal reset (box-sizing/margin on *) is missing",
  );
}
// The reset must also restore the native behaviours it deliberately zeroes.
const RESET_REQUIRED = [
  [/\bhr\s*\{[^}]*border-top-width/, "hr rules"],
  [/\bsummary\s*\{[^}]*display:\s*list-item/, "the <summary> marker"],
  [/::placeholder\s*\{[^}]*opacity:\s*1/, "::placeholder opacity"],
  [/\[hidden\][^{]*\{[^}]*display:\s*none/, "[hidden]"],
];
for (const [pattern, label] of RESET_REQUIRED) {
  if (!pattern.test(withoutComments)) {
    problems.push(`src/styles/app.css: the preflight reset is missing ${label}`);
  }
}

if (problems.length > 0) {
  console.error("CSS scale guard failed:\n" + problems.map((p) => `  - ${p}`).join("\n"));
  process.exit(1);
}
console.log(
  `CSS scale guard passed (${typeScale.size} type sizes, ${radiusScale.size} radii, ` +
    `${durationScale.size} durations, ${shadowScale.size} shadows, reset present)`,
);
