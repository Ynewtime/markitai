import { describe, expect, it } from "vitest";
import { ADVANCED_DEFAULTS } from "./advanced";
import { applyPreset, BUILTIN_PRESET_OPTIONS, changeAdvanced, matchingPreset, resolveOptions, type ComposerOptions } from "./conversionOptions";
import { buildCliCommand } from "./cli";

const initial: ComposerOptions = {
  preset: "minimal", llm: false, ocr: false, profile: null, advanced: ADVANCED_DEFAULTS,
};

describe("conversion option dependencies", () => {
  it.each(["minimal", "standard", "rich"] as const)("applies all five %s features", (preset) => {
    const selected = applyPreset(initial, preset);
    expect(resolveOptions(selected)).toMatchObject(BUILTIN_PRESET_OPTIONS[preset]);
  });

  it("resets bundle overrides, including OCR, but retains independent choices", () => {
    const modified: ComposerOptions = {
      ...initial, llm: true, ocr: true, profile: "rag",
      advanced: { ...ADVANCED_DEFAULTS, alt: false, desc: false, screenshot: false, noCache: true },
    };
    const rich = applyPreset(modified, "rich");
    expect(rich.profile).toBe("rag");
    expect(rich.advanced.noCache).toBe(true);
    expect(resolveOptions(rich)).toMatchObject({ llm: true, ocr: false, alt: true, desc: true, screenshot: true });
    const minimal = applyPreset(rich, "minimal");
    expect(resolveOptions(minimal)).toMatchObject(BUILTIN_PRESET_OPTIONS.minimal);
  });

  it("uses server preset definitions, including customized built-in names", () => {
    const presets = { rich: { llm: true, ocr: true, alt: false, desc: true, screenshot: false } };
    const rich = applyPreset(initial, "rich", presets);
    expect(resolveOptions(rich, presets)).toMatchObject(presets.rich);
  });

  it("honors explicit false, reports customization, and allows restoring the same preset", () => {
    const rich = applyPreset(initial, "rich");
    const changed = { ...rich, advanced: changeAdvanced(rich.advanced, "alt", false) };
    expect(resolveOptions(changed)).toMatchObject({ alt: false, desc: true, screenshot: true });
    expect(resolveOptions(applyPreset(changed, "rich")).alt).toBe(true);
  });

  it("pauses image analysis when LLM is disabled without losing manual choices", () => {
    const rich = applyPreset(initial, "rich");
    rich.advanced.desc = false;
    const off = { ...rich, llm: false };
    expect(resolveOptions(off)).toMatchObject({ llm: false, alt: false, desc: false, screenshot: true });
    expect(resolveOptions({ ...off, llm: true })).toMatchObject({ alt: true, desc: false });
  });

  it("plain mode pauses analysis and excludes the screenshot content source", () => {
    const rich = applyPreset(initial, "rich");
    rich.advanced = changeAdvanced(rich.advanced, "screenshotOnly", true);
    rich.advanced = changeAdvanced(rich.advanced, "pure", true);
    expect(resolveOptions(rich)).toMatchObject({ pure: true, screenshot_only: false, alt: false, desc: false });
    rich.advanced = changeAdvanced(rich.advanced, "screenshotOnly", true);
    expect(resolveOptions(rich)).toMatchObject({ pure: false, screenshot_only: true, screenshot: true, alt: true });
  });

  it("screenshot source implies capture, never LLM, and restores the capture choice", () => {
    const state = { ...initial, advanced: { ...ADVANCED_DEFAULTS, screenshot: false, screenshotOnly: true } };
    expect(resolveOptions(state)).toMatchObject({ screenshot: true, screenshot_only: true, llm: false });
    expect(resolveOptions({ ...state, advanced: changeAdvanced(state.advanced, "screenshotOnly", false) }).screenshot).toBe(false);
  });

  it("Cloudflare strategy implies its backend without destroying the previous file choice", () => {
    const advanced = { ...ADVANCED_DEFAULTS, backend: "native" as const, strategy: "cloudflare" as const };
    expect(resolveOptions({ ...initial, advanced }).backend).toBe("cloudflare");
    expect(resolveOptions({ ...initial, advanced: changeAdvanced(advanced, "strategy", "auto") }).backend).toBe("native");
  });

  it("keeps CLI flags consistent with effective values throughout the feature matrix", () => {
    for (const preset of ["minimal", "standard", "rich"] as const) {
      for (const llm of [false, true]) {
        for (const pure of [false, true]) {
          for (const alt of [null, false, true]) {
            const state = applyPreset(initial, preset);
            const effective = resolveOptions({ ...state, llm, advanced: { ...state.advanced, pure, alt } });
            const command = buildCliCommand(["a.pdf"], effective).split(" ");
            for (const key of ["llm", "ocr", "alt", "desc", "screenshot"] as const) {
              const flag = effective[key] ? `--${key}` : `--no-${key}`;
              if (effective[key] === BUILTIN_PRESET_OPTIONS[preset][key]) expect(command).not.toContain(flag);
              else expect(command).toContain(flag);
              expect(command).not.toContain(effective[key] ? `--no-${key}` : `--${key}`);
            }
          }
        }
      }
    }
  });
});


describe("effective preset matching without rewriting options", () => {
  it("keeps minimal + LLM custom and never enables image analysis", () => {
    const options = { ...initial, llm: true };
    expect(matchingPreset(options)).toBeNull();
    expect(resolveOptions(options)).toMatchObject({ preset: "minimal", llm: true, alt: false, desc: false });
    expect(buildCliCommand([], resolveOptions(options))).toContain("--preset minimal --llm");
  });

  it("matches an exact different bundle while preserving the underlying inheritance", () => {
    const options = { ...initial, llm: true, advanced: { ...ADVANCED_DEFAULTS, alt: true, desc: true } };
    const snapshot = structuredClone(options);
    expect(matchingPreset(options)).toBe("standard");
    expect(options).toEqual(snapshot);
    expect(buildCliCommand([], resolveOptions(options))).toContain("--preset minimal --llm --alt --desc");
    expect(matchingPreset({ ...options, ocr: true })).toBeNull();
    expect(matchingPreset({ ...options, advanced: { ...options.advanced, screenshot: true } })).toBe("rich");
  });

  it("uses configured bundles and prefers the selected name if bundles coincide", () => {
    const presets = { ...BUILTIN_PRESET_OPTIONS, standard: BUILTIN_PRESET_OPTIONS.minimal };
    expect(matchingPreset(initial, presets)).toBe("minimal");
    expect(matchingPreset({ ...initial, preset: "standard" }, presets)).toBe("standard");
    const custom = { ...presets, rich: { ...BUILTIN_PRESET_OPTIONS.minimal, ocr: true } };
    expect(matchingPreset({ ...initial, ocr: true }, custom)).toBe("rich");
  });

  it("recognizes only exact effective bundles across all five-feature combinations", () => {
    for (const preset of ["minimal", "standard", "rich"] as const) {
      for (let mask = 0; mask < 32; mask++) {
        const options = { ...initial, preset, llm: !!(mask & 1), ocr: !!(mask & 2),
          advanced: { ...ADVANCED_DEFAULTS, alt: !!(mask & 4), desc: !!(mask & 8), screenshot: !!(mask & 16) } };
        const effective = resolveOptions(options);
        const matches = ([preset, "minimal", "standard", "rich"] as const).filter((name) =>
          (["llm", "ocr", "alt", "desc", "screenshot"] as const).every((key) => effective[key] === BUILTIN_PRESET_OPTIONS[name][key]));
        expect(matchingPreset(options)).toBe(matches[0] ?? null);
      }
    }
  });
});
