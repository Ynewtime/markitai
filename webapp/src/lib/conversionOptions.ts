import type { OutputProfile, Preset, PresetFeatures } from "../api/types";
import { ADVANCED_DEFAULTS, type Advanced } from "./advanced";

export const BUILTIN_PRESET_OPTIONS: Record<Preset, PresetFeatures> = {
  minimal: { llm: false, ocr: false, alt: false, desc: false, screenshot: false },
  standard: { llm: true, ocr: false, alt: true, desc: true, screenshot: false },
  rich: { llm: true, ocr: false, alt: true, desc: true, screenshot: true },
};

export type PresetOptions = Record<string, PresetFeatures>;

export interface ComposerOptions {
  preset: Preset;
  llm: boolean;
  ocr: boolean;
  profile: OutputProfile | null;
  advanced: Advanced;
}

export function presetFeatures(preset: Preset, presets: PresetOptions) {
  return presets[preset] ?? BUILTIN_PRESET_OPTIONS[preset];
}

/** Selecting a bundle resets its five features, not unrelated output/source choices. */
export function applyPreset(
  options: ComposerOptions,
  preset: Preset,
  presets: PresetOptions = BUILTIN_PRESET_OPTIONS,
): ComposerOptions {
  const features = presetFeatures(preset, presets);
  return {
    ...options,
    preset,
    llm: features.llm,
    ocr: features.ocr,
    advanced: { ...options.advanced, alt: null, desc: null, screenshot: null },
  };
}

/** UI, requests and CLI previews must all describe these effective values. */
export function resolveOptions(
  options: ComposerOptions,
  presets: PresetOptions = BUILTIN_PRESET_OPTIONS,
) {
  const { preset, llm, ocr, profile, advanced: a } = options;
  const features = presetFeatures(preset, presets);
  const analysis = llm && !a.pure;
  return {
    preset,
    llm,
    ocr,
    profile,
    alt: analysis && (a.alt ?? features.alt),
    desc: analysis && (a.desc ?? features.desc),
    screenshot: a.screenshotOnly || (a.screenshot ?? features.screenshot),
    screenshot_only: a.screenshotOnly && !a.pure,
    pure: a.pure,
    no_cache: a.noCache,
    no_compress: a.noCompress,
    strategy: a.strategy,
    // The CLI's Cloudflare URL strategy also selects its file converter.
    backend: a.strategy === "cloudflare" ? "cloudflare" as const : a.backend,
  };
}

export function matchingPreset(
  options: ComposerOptions,
  presets: PresetOptions = BUILTIN_PRESET_OPTIONS,
): Preset | null {
  const effective = resolveOptions(options, presets);
  const candidates = [options.preset, ...(["minimal", "standard", "rich"] as const)];
  return candidates.find((preset) => {
    const features = presetFeatures(preset, presets);
    return (["llm", "ocr", "alt", "desc", "screenshot"] as const)
      .every((key) => effective[key] === features[key]);
  }) ?? null;
}

export function presetIsCustomized(
  options: ComposerOptions,
  presets: PresetOptions = BUILTIN_PRESET_OPTIONS,
): boolean {
  const effective = resolveOptions(options, presets);
  const features = presetFeatures(options.preset, presets);
  return (["llm", "ocr", "alt", "desc", "screenshot"] as const).some(
    (key) => effective[key] !== features[key],
  );
}

export function changeAdvanced<K extends keyof Advanced>(
  previous: Advanced,
  key: K,
  value: Advanced[K],
): Advanced {
  const next = { ...previous, [key]: value };
  // The file and URL pipelines give these modes different precedence.
  if (key === "pure" && value) next.screenshotOnly = false;
  if (key === "screenshotOnly" && value) next.pure = false;
  return next;
}

export function hasExtraOptions(a: Advanced): boolean {
  return (["screenshotOnly", "pure", "noCache", "noCompress", "strategy", "backend"] as const)
    .some((key) => a[key] !== ADVANCED_DEFAULTS[key]);
}
