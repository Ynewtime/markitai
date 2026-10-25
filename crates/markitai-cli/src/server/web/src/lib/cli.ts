// The command line equivalent of the composer. It assumes default configuration
// and the server's preset definitions, so only deviations are spelled out.
// A browser never tells a page where a chosen file lives (not even on the same
// machine), so staged files are named relative to the folder that holds them:
// `./name` for a file, `./folder/` for a picked folder.
import type { JobOptions, Preset, PresetFeatures } from "../api/types.ts";
import { relativePath } from "./files.ts";
import { BUILTIN_PRESETS, knownFeatures, type PresetTable } from "./options.ts";

const SAFE = /^(?!~)[A-Za-z0-9_\-./:@%+=,~]+$/;
/** The five features a preset sets; every CLI default for them is false. */
const FEATURES = ["llm", "ocr", "alt", "desc", "screenshot"] as const;

export function shellQuote(word: string): string {
  return SAFE.test(word) ? word : `'${word.replaceAll("'", "'\\''")}'`;
}

/** An input as the command line names it: quoted, and a name that starts with `-` as `./-name`, so
 * the shell's word is never read as an option. */
export function inputWord(input: string): string {
  return shellQuote(input.startsWith("-") ? `./${input}` : input);
}

/** A preset is named only while the request still equals it. Once the reader
 * changes one of its features the line spells the five features out, so it
 * never claims a preset it then overrides. A preset this page does not know (a
 * configured one the server does not list, a CLI run's, a Python job's) is never
 * named either: its features are unknown, so the switches are spelled out. */
export function namedPreset(options: JobOptions, preset: Preset | null, table: PresetTable): PresetFeatures | null {
  if (preset === null) return null;
  const features = knownFeatures(preset, table);
  if (features === null) return null;
  // A switch left null follows the preset (an older request names only its preset).
  return FEATURES.every((key) => (options[key] ?? features[key]) === features[key]) ? features : null;
}

/** Staged files as the command line names them: each file `./name`, a folder's files once as `./folder/`. */
export function stagedPaths(files: File[]): string[] {
  const paths: string[] = [];
  for (const file of files) {
    const path = relativePath(file);
    const slash = path.indexOf("/");
    const name = slash > 0 ? `./${path.slice(0, slash)}/` : `./${path.slice(slash + 1)}`;
    if (!paths.includes(name)) paths.push(name);
  }
  return paths;
}

/** The flags of a request (after `-o output/`), and its OCR settings that have no flag of their own. */
export function requestFlags(options: JobOptions, table: PresetTable = BUILTIN_PRESETS): { words: string[]; ocr: Record<string, string | boolean> } {
  const preset = options.preset as Preset | null;
  const named = namedPreset(options, preset, table);
  const features = named ?? { llm: false, ocr: false, alt: false, desc: false, screenshot: false };
  // a switch left null is the preset's, also when the line spells the switches out
  const own = preset !== null ? knownFeatures(preset, table) : null;
  const words: string[] = [];
  if (named !== null && preset !== null) words.push("--preset", shellQuote(preset));
  for (const key of FEATURES) {
    const value = options[key] ?? own?.[key] ?? null;
    if (value !== null && value !== undefined && value !== features[key]) words.push(value ? `--${key}` : `--no-${key}`);
  }
  if (options.profile !== null) words.push("--profile", shellQuote(options.profile));
  if (options.format === "json") words.push("-f", "json");
  if (options.page_markers) words.push("--page-markers");
  // Not part of a preset; false is the documented default and stays implicit.
  if (options.screenshot_only) words.push("--screenshot-only");
  if (options.pure) words.push("--pure");
  if (options.text_only) words.push("--text-only");
  if (options.no_cache) words.push("--no-cache");
  if (options.no_compress) words.push("--no-compress");
  if (options.strategy !== null && options.strategy !== "auto") words.push("--strategy", shellQuote(options.strategy));
  if (options.backend !== null && options.backend !== "native") words.push("--backend", shellQuote(options.backend));
  // OCR settings without a flag of their own; "en" is the default language policy.
  const ocr: Record<string, string | boolean> = {};
  if (options.ocr_lang && options.ocr_lang !== "en") ocr.lang = options.ocr_lang;
  if (options.ocr_all_pages) ocr.per_page_routing = false;
  return { words, ocr };
}

/** `inputs`: staged paths (see stagedPaths) and URLs, in that order. */
export function cliCommand(inputs: string[], options: JobOptions, table: PresetTable = BUILTIN_PRESETS): string {
  const words = ["markitai", ...(inputs.length ? inputs.map(inputWord) : ["<files-or-urls>"]), "-o", "output/"];
  const flags = requestFlags(options, table);
  words.push(...flags.words);
  if (Object.keys(flags.ocr).length > 0) words.push("--config-json", shellQuote(JSON.stringify({ ocr: flags.ocr })));
  return words.join(" ");
}
