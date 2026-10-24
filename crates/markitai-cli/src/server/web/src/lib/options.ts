// Conversion options as the composer holds them, and the request they become.
// A preset sets five features (llm, ocr, alt, desc, screenshot); the image
// switches can override it, and the advanced choices are independent.
import type {
  ConvertDefaults,
  ConversionBackend,
  DocumentFormat,
  FetchStrategy,
  JobOptions,
  OutputProfile,
  Preset,
  PresetFeatures,
} from "../api/types.ts";

export interface Advanced {
  /** null follows the preset; a boolean overrides it. */
  alt: boolean | null;
  desc: boolean | null;
  screenshot: boolean | null;
  screenshotOnly: boolean;
  pure: boolean;
  /** Images and PDFs: only the recognised text (`features.text_only` backends). */
  textOnly: boolean;
  noCache: boolean;
  noCompress: boolean;
  strategy: FetchStrategy;
  backend: ConversionBackend;
  /** v1.20 (`features.page_markers`): null leaves the server's configuration. */
  pageMarkers: boolean | null;
  /** v1.20 (`features.ocr_all_pages`): OCR every PDF page; sent only while OCR is on. */
  ocrAllPages: boolean | null;
}

export const ADVANCED_DEFAULTS: Advanced = {
  alt: null,
  desc: null,
  screenshot: null,
  screenshotOnly: false,
  pure: false,
  textOnly: false,
  noCache: false,
  noCompress: false,
  strategy: "auto",
  backend: "native",
  pageMarkers: null,
  ocrAllPages: null,
};

export interface Composer {
  preset: Preset;
  llm: boolean;
  ocr: boolean;
  profile: OutputProfile | null;
  advanced: Advanced;
  /** v1.20 (`features.document_json`): null leaves the server's configuration (Markdown). */
  format: DocumentFormat | null;
  /** v1.20 (`features.ocr_lang`): the local OCR language; null leaves the server's `ocr.lang`. */
  ocrLang: string | null;
}

export type PresetTable = Record<string, PresetFeatures>;

export const BUILTIN_PRESETS: Record<Preset, PresetFeatures> = {
  minimal: { llm: false, ocr: false, alt: false, desc: false, screenshot: false },
  standard: { llm: true, ocr: false, alt: true, desc: true, screenshot: false },
  rich: { llm: true, ocr: false, alt: true, desc: true, screenshot: true },
};

export const PRESETS: Preset[] = ["minimal", "standard", "rich"];
const FEATURES = ["llm", "ocr", "alt", "desc", "screenshot"] as const;

export function featuresOf(preset: Preset, table: PresetTable = BUILTIN_PRESETS): PresetFeatures {
  return table[preset] ?? BUILTIN_PRESETS[preset];
}

/** Choosing a preset resets its five features, never the output or source choices. */
export function applyPreset(state: Composer, preset: Preset, table: PresetTable = BUILTIN_PRESETS): Composer {
  const features = featuresOf(preset, table);
  return {
    ...state,
    preset,
    llm: features.llm,
    ocr: features.ocr,
    advanced: { ...state.advanced, alt: null, desc: null, screenshot: null },
  };
}

/** The effective request: what the UI shows, what is sent, what the CLI line prints. */
export function resolveOptions(state: Composer, table: PresetTable = BUILTIN_PRESETS): JobOptions {
  const { preset, llm, ocr, profile, advanced: a } = state;
  const features = featuresOf(preset, table);
  const analysis = llm && !a.pure;
  return {
    preset,
    llm,
    ocr,
    profile,
    alt: analysis && (a.alt ?? features.alt),
    desc: analysis && (a.desc ?? features.desc),
    screenshot: a.screenshotOnly || (a.screenshot ?? features.screenshot),
    screenshot_only: a.screenshotOnly,
    pure: a.pure,
    no_cache: a.noCache,
    no_compress: a.noCompress,
    strategy: a.strategy,
    backend: a.backend,
    // Sent only when chosen: a backend without the option never sees the key.
    ...(a.textOnly ? { text_only: true } : {}),
    // v1.20: set only on a backend that offers them (its defaults or its controls).
    ...(state.format !== null ? { format: state.format } : {}),
    // The language applies to local OCR, and every-page OCR to OCR at all.
    ...(ocr && !llm && state.ocrLang !== null ? { ocr_lang: state.ocrLang } : {}),
    ...(ocr && a.ocrAllPages !== null ? { ocr_all_pages: a.ocrAllPages } : {}),
    ...(a.pageMarkers !== null ? { page_markers: a.pageMarkers } : {}),
  };
}

/** The preset whose five features the effective options equal, if any. */
export function matchingPreset(state: Composer, table: PresetTable = BUILTIN_PRESETS): Preset | null {
  const effective = resolveOptions(state, table);
  for (const preset of [state.preset, ...PRESETS]) {
    const features = featuresOf(preset, table);
    if (FEATURES.every((key) => effective[key] === features[key])) return preset;
  }
  return null;
}

export function changeAdvanced<K extends keyof Advanced>(value: Advanced, key: K, next: Advanced[K]): Advanced {
  const changed = { ...value, [key]: next };
  // Plain mode and screenshots-as-source exclude each other.
  if (key === "pure" && next) changed.screenshotOnly = false;
  if (key === "screenshotOnly" && next) changed.pure = false;
  return changed;
}

export function emptyOptions(): JobOptions {
  return {
    preset: null,
    llm: null,
    ocr: null,
    profile: null,
    alt: null,
    desc: null,
    screenshot: null,
    screenshot_only: null,
    pure: null,
    no_cache: null,
    no_compress: null,
    strategy: null,
    backend: null,
  };
}

/** The fetch strategies the composer offers. A user token's conversions
 * never use a remote service (defuddle.md, Jina Reader, Cloudflare): both
 * backends refuse those strategies for it, so they are not offered. */
export function offeredStrategies(localOnly: boolean): FetchStrategy[] {
  const all: FetchStrategy[] = ["auto", "static", "playwright", "defuddle", "jina", "cloudflare"];
  return localOnly ? all.slice(0, 3) : all;
}

const PROFILES = ["rag", "obsidian", "okf"];
const STRATEGIES = ["auto", "static", "playwright", "defuddle", "jina", "cloudflare"];
const BACKENDS = ["native", "cloudflare"];
const BOOLEANS = ["llm", "ocr", "alt", "desc", "screenshot", "screenshot_only", "pure", "no_cache", "no_compress"] as const;
const FORMATS = ["markdown", "json"];
/** An `ocr.lang` code as both backends spell them (`zh_tw`, `pt-BR`). */
const LANGUAGE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$/;

/** Only the public option keys with valid values: stored job options carry
 * internal metadata (`origin`), and a retry body rejects unknown keys. */
export function publicOptions(raw: unknown): JobOptions {
  const options = emptyOptions();
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) return options;
  const value = raw as Record<string, unknown>;
  for (const key of BOOLEANS) if (typeof value[key] === "boolean") options[key] = value[key];
  if (typeof value.preset === "string" && value.preset && value.preset.length <= 64) options.preset = value.preset;
  if (typeof value.profile === "string" && PROFILES.includes(value.profile)) options.profile = value.profile as OutputProfile;
  if (typeof value.strategy === "string" && STRATEGIES.includes(value.strategy)) options.strategy = value.strategy as FetchStrategy;
  if (typeof value.backend === "string" && BACKENDS.includes(value.backend)) options.backend = value.backend as ConversionBackend;
  if (value.text_only === true) options.text_only = true;
  if (typeof value.format === "string" && FORMATS.includes(value.format)) options.format = value.format as DocumentFormat;
  if (typeof value.ocr_lang === "string" && LANGUAGE.test(value.ocr_lang)) options.ocr_lang = value.ocr_lang;
  if (typeof value.ocr_all_pages === "boolean") options.ocr_all_pages = value.ocr_all_pages;
  if (typeof value.page_markers === "boolean") options.page_markers = value.page_markers;
  return options;
}

export const OPTIONS_KEY = "markitai.options";
const OPTIONS_VERSION = 4;

export interface Remembered {
  preset: Preset | null;
  llm: boolean | null;
  ocr: boolean | null;
  profile: OutputProfile | null;
  images: Pick<Advanced, "alt" | "desc" | "screenshot">;
}

interface Store {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function localStore(): Store | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

const asPreset = (value: unknown): Preset | null =>
  value === "minimal" || value === "standard" || value === "rich" ? value : null;
const asBool = (value: unknown): boolean | null => (typeof value === "boolean" ? value : null);
const asProfile = (value: unknown): OutputProfile | null =>
  typeof value === "string" && PROFILES.includes(value) ? (value as OutputProfile) : null;

/** The last composer choice in this browser. Source, cache and remote choices are
 * deliberately not remembered: a revisit must not silently re-enable a remote service.
 * Both the current shape and the earlier flat shape are read. */
export function readRemembered(store: Store | null = localStore()): Remembered {
  const none: Remembered = { preset: null, llm: null, ocr: null, profile: null, images: { alt: null, desc: null, screenshot: null } };
  let parsed: unknown;
  try {
    parsed = JSON.parse(store?.getItem(OPTIONS_KEY) ?? "null");
  } catch {
    return none;
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return none;
  const value = parsed as Record<string, unknown>;
  const nested = value.imageOverrides !== null && typeof value.imageOverrides === "object" ? (value.imageOverrides as Record<string, unknown>) : value;
  return {
    preset: asPreset(value.preset),
    llm: asBool(value.llm),
    ocr: asBool(value.ocr),
    profile: asProfile(value.profile),
    images: { alt: asBool(nested.alt), desc: asBool(nested.desc), screenshot: asBool(nested.screenshot) },
  };
}

export function remember(state: Composer, store: Store | null = localStore()): void {
  try {
    store?.setItem(
      OPTIONS_KEY,
      JSON.stringify({
        version: OPTIONS_VERSION,
        preset: state.preset,
        llm: state.llm,
        ocr: state.ocr,
        profile: state.profile,
        imageOverrides: { alt: state.advanced.alt, desc: state.advanced.desc, screenshot: state.advanced.screenshot },
      }),
    );
  } catch {
    /* The choice lasts for this page only. */
  }
}

/** Composer state to start from: remembered choices over the defaults. */
export function initialComposer(saved: Remembered): Composer {
  return {
    preset: saved.preset ?? "minimal",
    llm: saved.llm ?? false,
    ocr: saved.ocr ?? false,
    profile: saved.profile,
    advanced: { ...ADVANCED_DEFAULTS, ...saved.images },
    format: null,
    ocrLang: null,
  };
}

/** Options for re-running an image skipped for lack of text: OCR on unless a
 * model or OCR already ran. */
export function withOcrFor(options: JobOptions): JobOptions {
  return !options.llm && !options.ocr ? { ...options, ocr: true } : options;
}

/** v1.17: the composer a set of saved defaults gives: the preset that differs least from them
 * (image switches only where they differ from it), and the rest as saved. */
export function composerFromDefaults(defaults: ConvertDefaults, table: PresetTable = BUILTIN_PRESETS): Composer {
  const misses = (preset: Preset) => {
    const features = featuresOf(preset, table);
    const images = defaults.llm ? (["alt", "desc"] as const).filter((key) => features[key] !== defaults[key]).length : 0;
    // The model matters most: a preset that agrees on it wins over one that agrees on two pictures switches.
    return (features.llm !== defaults.llm ? 3 : 0) + (features.screenshot !== defaults.screenshot ? 1 : 0) + images;
  };
  const preset = PRESETS.reduce((best, preset) => (misses(preset) < misses(best) ? preset : best), PRESETS[0] as Preset);
  const features = featuresOf(preset, table);
  const differs = (key: "alt" | "desc" | "screenshot") => (features[key] !== defaults[key] ? defaults[key] : null);
  return {
    preset,
    llm: defaults.llm,
    ocr: defaults.ocr,
    profile: defaults.profile,
    advanced: {
      // Kept while the model is off too: they apply once it is turned on.
      alt: differs("alt"),
      desc: differs("desc"),
      screenshot: differs("screenshot"),
      screenshotOnly: defaults.screenshot_only,
      pure: defaults.pure,
      textOnly: false,
      noCache: defaults.no_cache,
      noCompress: defaults.no_compress,
      strategy: defaults.strategy,
      backend: "native",
      pageMarkers: defaults.page_markers ?? null,
      ocrAllPages: defaults.ocr_all_pages ?? null,
    },
    format: defaults.format ?? null,
    ocrLang: defaults.ocr_lang ?? null,
  };
}

/** The defaults the composer's choices would save; null while they name Cloudflare (it asks each time). */
export function defaultsOf(state: Composer, table: PresetTable = BUILTIN_PRESETS): ConvertDefaults | null {
  const a = state.advanced;
  if (a.strategy === "cloudflare" || a.backend === "cloudflare") return null;
  const features = featuresOf(state.preset, table);
  return {
    llm: state.llm,
    ocr: state.ocr,
    // The image switches as shown, not as sent: without a model they are kept for when one runs.
    alt: a.alt ?? features.alt,
    desc: a.desc ?? features.desc,
    screenshot: a.screenshotOnly || (a.screenshot ?? features.screenshot),
    screenshot_only: a.screenshotOnly,
    pure: a.pure,
    no_cache: a.noCache,
    no_compress: a.noCompress,
    profile: state.profile,
    strategy: a.strategy,
    // v1.20: only what this backend offers (a choice is set only through its defaults or controls).
    ...(state.format !== null ? { format: state.format } : {}),
    ...(state.ocrLang !== null ? { ocr_lang: state.ocrLang } : {}),
    ...(a.ocrAllPages !== null ? { ocr_all_pages: a.ocrAllPages } : {}),
    ...(a.pageMarkers !== null ? { page_markers: a.pageMarkers } : {}),
  };
}

const DEFAULT_KEYS = ["llm", "ocr", "alt", "desc", "screenshot", "screenshot_only", "pure", "no_cache", "no_compress", "profile", "strategy"] as const;
/** v1.20 defaults a backend has only with their features. */
const OFFERED_KEYS = ["format", "ocr_lang", "ocr_all_pages", "page_markers"] as const;

/** Whether the composer's choices are these defaults (what a request would do, so the image
 * switches count only while a model analyses pictures). */
export function sameAsDefaults(state: Composer, defaults: ConvertDefaults, table: PresetTable = BUILTIN_PRESETS): boolean {
  const mine = defaultsOf(state, table);
  if (mine === null) return false;
  const analysis = mine.llm && !mine.pure;
  return (
    DEFAULT_KEYS.every((key) => ((key === "alt" || key === "desc") && !analysis) || mine[key] === defaults[key]) &&
    // A v1.20 default the composer has not set yet still holds.
    OFFERED_KEYS.every((key) => mine[key] === undefined || mine[key] === defaults[key])
  );
}

/** What a backend without `/api/convert-defaults` converts with: the minimal preset, nothing else on. */
export const BUILTIN_DEFAULTS: ConvertDefaults = {
  llm: false,
  ocr: false,
  alt: false,
  desc: false,
  screenshot: false,
  screenshot_only: false,
  pure: false,
  no_cache: false,
  no_compress: false,
  profile: null,
  strategy: "auto",
};

export type OptionKey = (typeof DEFAULT_KEYS)[number] | (typeof OFFERED_KEYS)[number] | "backend" | "text_only";

/** The options dialog's groups and the keys each one shows. */
export const OPTION_GROUPS = {
  enhance: ["llm", "ocr", "ocr_lang"],
  images: ["alt", "desc", "screenshot"],
  output: ["profile", "format"],
  url: ["strategy"],
  files: ["backend"],
  pages: ["page_markers", "ocr_all_pages"],
  other: ["screenshot_only", "pure", "text_only", "no_cache", "no_compress"],
} as const satisfies Record<string, readonly OptionKey[]>;
export type OptionGroup = keyof typeof OPTION_GROUPS;

/** The choices that differ from these defaults, as sameAsDefaults counts them, plus the per-request
 * ones no default holds (Cloudflare, Text only): the dot on Options and the bright values in its dialog. */
export function changedOptions(state: Composer, defaults: ConvertDefaults, table: PresetTable = BUILTIN_PRESETS): Set<OptionKey> {
  const a = state.advanced;
  const local = { ...state, advanced: { ...a, strategy: a.strategy === "cloudflare" ? defaults.strategy : a.strategy, backend: "native" as const } };
  const mine = defaultsOf(local, table) as ConvertDefaults;
  const analysis = mine.llm && !mine.pure;
  const changed = new Set<OptionKey>();
  for (const key of DEFAULT_KEYS) if (!((key === "alt" || key === "desc") && !analysis) && mine[key] !== defaults[key]) changed.add(key);
  for (const key of OFFERED_KEYS) if (mine[key] !== undefined && mine[key] !== defaults[key]) changed.add(key);
  if (a.strategy === "cloudflare") changed.add("strategy");
  if (a.backend !== "native") changed.add("backend");
  if (a.textOnly) changed.add("text_only");
  return changed;
}

/** A group's value on the dialog's first page: its single choice, or the switches that are on
 * joined with " · ", or Off. */
export function groupValue(parts: string[], off: string): string {
  return parts.length ? parts.join(" · ") : off;
}
