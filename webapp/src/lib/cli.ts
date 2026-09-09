import type { JobOptions } from "../api/types";
import { BUILTIN_PRESET_OPTIONS, presetFeatures, type PresetOptions } from "./conversionOptions";

const SHELL_SAFE_RE = /^(?!~)[A-Za-z0-9_\-./:@%+=,~]+$/;

function shellQuote(s: string): string {
  if (SHELL_SAFE_RE.test(s)) return s;
  return `'${s.replaceAll("'", "'\\''")}'`;
}

/** The command assumes default non-preset configuration and the same preset
 * definitions as the server, so only deviations from the bundle are spelled out.
 * Browser uploads expose no local paths, so their input remains a placeholder. */
export function buildCliCommand(
  urls: string[], options: JobOptions,
  presets: PresetOptions = BUILTIN_PRESET_OPTIONS,
): string {
  const features = options.preset === null ? null : presetFeatures(options.preset, presets);
  const inputs = urls.length > 0 ? urls.map(shellQuote) : ["<your-files-or-url-or-url_files>"];
  const parts = ["markitai", ...inputs, "-o", "out/"];
  if (options.preset !== null) parts.push("--preset", options.preset);
  for (const key of ["llm", "ocr", "alt", "desc", "screenshot"] as const) {
    if (options[key] !== null && options[key] !== features?.[key]) {
      parts.push(options[key] ? `--${key}` : `--no-${key}`);
    }
  }
  if (options.profile !== null) parts.push("--profile", options.profile);
  for (const [key, positive] of [
    ["screenshot_only", "--screenshot-only", "--no-screenshot-only"],
    ["pure", "--pure", "--no-pure"],
    ["no_cache", "--no-cache", "--cache"],
    ["no_compress", "--no-compress", "--compress"],
  ] as const) {
    // These flags are not part of a preset; false is the documented default and stays implicit.
    if (options[key]) parts.push(positive);
  }
  if (options.strategy !== null && options.strategy !== "auto") parts.push("--strategy", options.strategy);
  if (options.backend !== null && options.backend !== "native") parts.push("--backend", options.backend);
  return parts.join(" ");
}
