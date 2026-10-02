// The command line equivalent of the composer. It assumes default configuration
// and the server's preset definitions, so only deviations are spelled out.
// Uploads have no local path the page could know, so they stay a placeholder.
import type { JobOptions, Preset } from "../api/types.ts";
import { BUILTIN_PRESETS, featuresOf, type PresetTable } from "./options.ts";

const SAFE = /^(?!~)[A-Za-z0-9_\-./:@%+=,~]+$/;

export function shellQuote(word: string): string {
  return SAFE.test(word) ? word : `'${word.replaceAll("'", "'\\''")}'`;
}

export function cliCommand(urls: string[], options: JobOptions, table: PresetTable = BUILTIN_PRESETS): string {
  const preset = options.preset as Preset | null;
  const features = preset === null ? null : featuresOf(preset, table);
  const words = ["markitai", ...(urls.length ? urls.map(shellQuote) : ["<your-files-or-url-or-url_files>"]), "-o", "out/"];
  if (preset !== null) words.push("--preset", preset);
  for (const key of ["llm", "ocr", "alt", "desc", "screenshot"] as const) {
    const value = options[key];
    if (value !== null && value !== features?.[key]) words.push(value ? `--${key}` : `--no-${key}`);
  }
  if (options.profile !== null) words.push("--profile", options.profile);
  // Not part of a preset; false is the documented default and stays implicit.
  if (options.screenshot_only) words.push("--screenshot-only");
  if (options.pure) words.push("--pure");
  if (options.no_cache) words.push("--no-cache");
  if (options.no_compress) words.push("--no-compress");
  if (options.strategy !== null && options.strategy !== "auto") words.push("--strategy", options.strategy);
  if (options.backend !== null && options.backend !== "native") words.push("--backend", options.backend);
  return words.join(" ");
}
