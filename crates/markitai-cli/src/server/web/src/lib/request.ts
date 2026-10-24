// What a conversion was asked to do, for its Details: the equivalent command
// line, and the settings it ran with that no flag sets, as `key: value` chips.
// A backend that records what a job ran with (`settings`, contract v1.22)
// gives the whole picture, server configuration included; otherwise the
// request's own options are all that is known, and nothing else is claimed.
import type { JobOptions, JobSettings, Preset } from "../api/types.ts";
import { namedPreset, requestFlags, shellQuote } from "./cli.ts";
import { BUILTIN_PRESETS, type PresetTable } from "./options.ts";

export interface RequestChip {
  key: string;
  value: string;
}

export interface RequestView {
  command: string;
  /** The command's words, each quoted for the shell (`command` is them joined by spaces). */
  words: string[];
  chips: RequestChip[];
}

const NAME = /^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){1,3}$/;

/** Only dotted keys with plain values: anything else a backend sends is left out. */
export function publicSettings(raw: unknown): JobSettings | null {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) return null;
  const settings: JobSettings = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>).slice(0, 64)) {
    if (!NAME.test(key)) continue;
    if (typeof value === "boolean" || (typeof value === "number" && Number.isFinite(value)) || value === null) settings[key] = value;
    else if (typeof value === "string" && value.length <= 80) settings[key] = value;
  }
  return settings;
}

const on = (settings: JobSettings, key: string) => settings[key] === true;

/** The flags a job's settings stand for, and the keys they used. */
function settingsFlags(settings: JobSettings, options: JobOptions | null, url: boolean, table: PresetTable): { words: string[]; used: Set<string> } {
  const used = new Set<string>();
  const take = (key: string) => {
    used.add(key);
    return settings[key];
  };
  const features = {
    llm: on(settings, "llm.enabled"),
    ocr: on(settings, "ocr.enabled"),
    alt: on(settings, "image.alt_enabled"),
    desc: on(settings, "image.desc_enabled"),
    screenshot: on(settings, "screenshot.enabled"),
  };
  for (const key of ["llm.enabled", "ocr.enabled", "image.alt_enabled", "image.desc_enabled", "screenshot.enabled"]) used.add(key);
  const words: string[] = [];
  // The preset the request named, while the job's five switches still equal it.
  const preset = (options?.preset ?? null) as Preset | null;
  const named = namedPreset({ ...(options ?? ({} as JobOptions)), ...features }, preset, table);
  if (named !== null && preset !== null) words.push("--preset", shellQuote(preset));
  else for (const [key, value] of Object.entries(features)) if (value) words.push(`--${key}`);
  if (take("llm.keep_base") === true) words.push("--keep-base");
  if (take("llm.pure") === true) words.push("--pure");
  if (take("screenshot.screenshot_only") === true) words.push("--screenshot-only");
  const profile = take("output.profile");
  if (typeof profile === "string") words.push("--profile", shellQuote(profile));
  if (take("output.format") === "json") words.push("-f", "json");
  if (take("output.page_markers") === true) words.push("--page-markers");
  if (take("output.text_only") === true) words.push("--text-only");
  if (take("cache.no_cache") === true) words.push("--no-cache");
  if (take("image.compress") === false) words.push("--no-compress");
  // The URL strategy says nothing about a file.
  const strategy = take("fetch.strategy");
  if (url && typeof strategy === "string") words.push("--strategy", shellQuote(strategy));
  if (take("fetch.cloudflare.convert_enabled") === true) words.push("--backend", "cloudflare");
  return { words, used };
}

/** The words as the line wraps them: a flag stays with the value it takes (`-o output/`, `--preset rich`). */
export function wrapUnits(words: string[]): string[] {
  const units: string[] = [];
  for (let index = 0; index < words.length; index++) {
    const word = words[index] as string;
    const next = words[index + 1];
    if (word.startsWith("-") && next !== undefined && !next.startsWith("-")) {
      units.push(`${word} ${next}`);
      index++;
    } else units.push(word);
  }
  return units;
}

const shown = (value: string | number | boolean | null) => (value === null ? "null" : String(value));

/** `inputs`: what was converted, as names or URLs; `url`: the item is a URL. Null when nothing is known. */
export function requestView(
  inputs: string[],
  options: JobOptions | null | undefined,
  settings: JobSettings | null | undefined,
  url: boolean,
  table: PresetTable = BUILTIN_PRESETS,
): RequestView | null {
  const words = ["markitai", ...(inputs.length ? inputs.map(shellQuote) : ["<files-or-urls>"]), "-o", "output/"];
  const chips: RequestChip[] = [];
  if (settings) {
    const flags = settingsFlags(settings, options ?? null, url, table);
    words.push(...flags.words);
    for (const [key, value] of Object.entries(settings)) if (!flags.used.has(key)) chips.push({ key, value: shown(value) });
  } else if (options) {
    const flags = requestFlags(options, table);
    words.push(...flags.words);
    if (typeof flags.ocr.lang === "string") chips.push({ key: "ocr.lang", value: flags.ocr.lang });
    if (flags.ocr.per_page_routing === false) chips.push({ key: "ocr.per_page_routing", value: "false" });
  } else return null;
  return { command: words.join(" "), words, chips };
}
