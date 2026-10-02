// Conversion options as the composer holds them, and the request they become.
// A preset sets five features (llm, ocr, alt, desc, screenshot); the image
// switches can override it, and the advanced choices are independent.
import type {
  ConversionBackend,
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
  noCache: boolean;
  noCompress: boolean;
  strategy: FetchStrategy;
  backend: ConversionBackend;
}

export const ADVANCED_DEFAULTS: Advanced = {
  alt: null,
  desc: null,
  screenshot: null,
  screenshotOnly: false,
  pure: false,
  noCache: false,
  noCompress: false,
  strategy: "auto",
  backend: "native",
};

export interface Composer {
  preset: Preset;
  llm: boolean;
  ocr: boolean;
  profile: OutputProfile | null;
  advanced: Advanced;
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

export function hasAdvancedChoices(value: Advanced): boolean {
  return (["screenshotOnly", "pure", "noCache", "noCompress", "strategy", "backend"] as const).some(
    (key) => value[key] !== ADVANCED_DEFAULTS[key],
  );
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

const PROFILES = ["rag", "obsidian", "okf"];
const STRATEGIES = ["auto", "static", "playwright", "defuddle", "jina", "cloudflare"];
const BACKENDS = ["native", "cloudflare"];
const BOOLEANS = ["llm", "ocr", "alt", "desc", "screenshot", "screenshot_only", "pure", "no_cache", "no_compress"] as const;

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
  };
}

/** Options for re-running an image skipped for lack of text: OCR on unless a
 * model or OCR already ran. */
export function withOcrFor(options: JobOptions): JobOptions {
  return !options.llm && !options.ocr ? { ...options, ocr: true } : options;
}
