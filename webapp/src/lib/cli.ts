import type { JobOptions } from "../api/types";
import { BUILTIN_PRESET_OPTIONS, presetFeatures, type PresetOptions } from "./conversionOptions";

const SHELL_SAFE_RE = /^(?!~)[A-Za-z0-9_\-./:@%+=,~]+$/;

function shellQuote(s: string): string {
  if (SHELL_SAFE_RE.test(s)) return s;
  return `'${s.replaceAll("'", "'\\''")}'`;
}

/** Compact commands assume default non-preset configuration and the same preset
 * definitions as the server. Explicit mode preserves every non-null override,
 * including false; null still means inherit (not full config reproducibility).
 * Browser uploads expose no local paths, so their input remains a placeholder. */
export function buildCliCommand(
  urls: string[], options: JobOptions,
  presets: PresetOptions = BUILTIN_PRESET_OPTIONS,
  mode: "compact" | "explicit" = "compact",
): string {
  const compact = mode === "compact";
  const features = options.preset === null ? null : presetFeatures(options.preset, presets);
  const inputs = urls.length > 0 ? urls.map(shellQuote) : ["<your-files-or-url-or-url_files>"];
  const parts = ["markitai", ...inputs, "-o", "out/"];
  if (options.preset !== null) parts.push("--preset", options.preset);
  for (const key of ["llm", "ocr", "alt", "desc", "screenshot"] as const) {
    if (options[key] !== null && (!compact || options[key] !== features?.[key])) {
      parts.push(options[key] ? `--${key}` : `--no-${key}`);
    }
  }
  if (options.profile !== null) parts.push("--profile", options.profile);
  for (const [key, positive, negative] of [
    ["screenshot_only", "--screenshot-only", "--no-screenshot-only"],
    ["pure", "--pure", "--no-pure"],
    ["no_cache", "--no-cache", "--cache"],
    ["no_compress", "--no-compress", "--compress"],
  ] as const) {
    // These flags are NOT part of a preset. Omitting false is only valid under
    // compact mode's documented default-config assumption, never explicit mode.
    if (options[key] !== null && (!compact || options[key])) parts.push(options[key] ? positive : negative);
  }
  if (options.strategy !== null && (!compact || options.strategy !== "auto")) parts.push("--strategy", options.strategy);
  if (options.backend !== null && (!compact || options.backend !== "native")) parts.push("--backend", options.backend);
  return parts.join(" ");
}
