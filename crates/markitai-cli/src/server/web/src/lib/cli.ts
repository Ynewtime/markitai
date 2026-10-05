// The command line equivalent of the composer. It assumes default configuration
// and the server's preset definitions, so only deviations are spelled out.
// Uploads have no local path the page could know, so they stay a placeholder.
import type { JobOptions, Preset, PresetFeatures } from "../api/types.ts";
import { BUILTIN_PRESETS, featuresOf, type PresetTable } from "./options.ts";

const SAFE = /^(?!~)[A-Za-z0-9_\-./:@%+=,~]+$/;
/** The five features a preset sets; every CLI default for them is false. */
const FEATURES = ["llm", "ocr", "alt", "desc", "screenshot"] as const;

export function shellQuote(word: string): string {
  return SAFE.test(word) ? word : `'${word.replaceAll("'", "'\\''")}'`;
}

/** A preset is named only while the request still equals it. Once the reader
 * changes one of its features the line spells the five features out, so it
 * never claims a preset it then overrides. */
function namedPreset(options: JobOptions, preset: Preset | null, table: PresetTable): PresetFeatures | null {
  if (preset === null) return null;
  const features = featuresOf(preset, table);
  return FEATURES.every((key) => options[key] === features[key]) ? features : null;
}

export function cliCommand(urls: string[], options: JobOptions, table: PresetTable = BUILTIN_PRESETS): string {
  const preset = options.preset as Preset | null;
  const named = namedPreset(options, preset, table);
  const features = named ?? { llm: false, ocr: false, alt: false, desc: false, screenshot: false };
  const words = ["markitai", ...(urls.length ? urls.map(shellQuote) : ["<files-or-urls>"]), "-o", "output/"];
  if (named !== null && preset !== null) words.push("--preset", preset);
  for (const key of FEATURES) {
    const value = options[key];
    if (value !== null && value !== features[key]) words.push(value ? `--${key}` : `--no-${key}`);
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
