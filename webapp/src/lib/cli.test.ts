import { describe, expect, it } from "vitest";

import { buildCliCommand } from "./cli";
import type { OutputProfile, Preset } from "../api/types";
import { BUILTIN_PRESET_OPTIONS, type PresetOptions } from "./conversionOptions";
import { jobOptions } from "./jobOptions";

const command = (urls: string[], preset: Preset, llm: boolean, ocr: boolean, profile: OutputProfile | null) =>
  buildCliCommand(urls, jobOptions({ preset, llm, ocr, profile }));

describe("buildCliCommand", () => {
  it.each([
    [true, "--screenshot-only --pure --no-cache --no-compress"],
    [false, ""],
    [null, ""],
  ] as const)("renders advanced booleans as %s, leaving the default false implicit", (value, flags) => {
    expect(buildCliCommand([], jobOptions({
      screenshot_only: value, pure: value, no_cache: value, no_compress: value,
    }))).toBe(`markitai <your-files-or-url-or-url_files> -o out/${flags ? ` ${flags}` : ""}`);
  });
  it("includes advanced values and explicit image opt-outs", () => {
    expect(buildCliCommand([], jobOptions({
      preset: "rich", llm: true, ocr: true, profile: "obsidian",
      alt: false, desc: true, screenshot: true, screenshot_only: true,
      pure: false, no_cache: true, no_compress: true, strategy: "playwright", backend: "kreuzberg",
    }))).toBe("markitai <your-files-or-url-or-url_files> -o out/ --preset rich --ocr --no-alt --profile obsidian --screenshot-only --no-cache --no-compress --strategy playwright --backend kreuzberg");
  });
  it("quotes a URL with a query string without altering it", () => {
    expect(command(["https://example.com/page?a=1&b"], "standard", true, false, null)).toBe(
      "markitai 'https://example.com/page?a=1&b' -o out/ --preset standard",
    );
  });

  it("escapes embedded single quotes so the shell reassembles one word", () => {
    expect(command(["https://example.com/it's"], "minimal", false, false, null)).toBe(
      "markitai 'https://example.com/it'\\''s' -o out/ --preset minimal",
    );
  });

  it("quotes inputs containing spaces", () => {
    expect(command(["my file.pdf"], "rich", true, true, null)).toBe(
      "markitai 'my file.pdf' -o out/ --preset rich --ocr",
    );
  });

  it("quotes a leading tilde but leaves an interior tilde bare", () => {
    expect(command(["~/docs/report.pdf"], "standard", false, true, null)).toBe(
      "markitai '~/docs/report.pdf' -o out/ --preset standard --no-llm --ocr",
    );
    expect(command(["https://example.com/~user/page"], "standard", false, true, null)).toBe(
      "markitai https://example.com/~user/page -o out/ --preset standard --no-llm --ocr",
    );
  });

  it("appends the profile only when one is chosen", () => {
    expect(command(["a.pdf"], "standard", true, false, "rag")).toBe(
      "markitai a.pdf -o out/ --preset standard --profile rag",
    );
    expect(command(["a.pdf"], "standard", true, false, null)).toBe(
      "markitai a.pdf -o out/ --preset standard",
    );
  });

  it("falls back to the file placeholder when no URLs are typed", () => {
    expect(command([], "standard", true, false, null)).toBe(
      "markitai <your-files-or-url-or-url_files> -o out/ --preset standard",
    );
  });
});

const features = ["llm", "ocr", "alt", "desc", "screenshot"] as const;
const presetNames = ["minimal", "standard", "rich"] as const;
const customPresets: PresetOptions = {
  minimal: { llm: true, ocr: true, alt: true, desc: false, screenshot: true },
  standard: { llm: false, ocr: true, alt: false, desc: false, screenshot: true },
  rich: { llm: true, ocr: true, alt: false, desc: true, screenshot: false },
};
describe.each([BUILTIN_PRESET_OPTIONS, customPresets])("preset-aware compact commands", (presets) => {
  it.each(presetNames)("omits all matching %s bundle flags and default extras", (preset) => {
    const options = jobOptions({ preset, ...presets[preset], pure: false, screenshot_only: false,
      no_cache: false, no_compress: false, strategy: "auto", backend: "native" });
    expect(buildCliCommand([], options, presets)).toBe(`markitai <your-files-or-url-or-url_files> -o out/ --preset ${preset}`);
  });
  it.each(presetNames)("preserves every positive and negative %s deviation", (preset) => {
    for (const key of features) {
      const value = !presets[preset]![key];
      const options = jobOptions({ preset, ...presets[preset], [key]: value });
      expect(buildCliCommand([], options, presets)).toBe(`markitai <your-files-or-url-or-url_files> -o out/ --preset ${preset} ${value ? `--${key}` : `--no-${key}`}`);
    }
  });
});
it("falls back for a missing server preset, without dropping no-preset false features", () => {
  expect(buildCliCommand([], jobOptions({ preset: "rich", ...BUILTIN_PRESET_OPTIONS.rich }), {})).toBe("markitai <your-files-or-url-or-url_files> -o out/ --preset rich");
  expect(buildCliCommand([], jobOptions({ llm: false, alt: false }))).toContain("--no-llm --no-alt");
});
