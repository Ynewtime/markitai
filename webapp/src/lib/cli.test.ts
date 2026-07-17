import { describe, expect, it } from "vitest";

import { buildCliCommand } from "./cli";

describe("buildCliCommand", () => {
  it("quotes a URL with a query string without altering it", () => {
    expect(buildCliCommand(["https://example.com/page?a=1&b"], "standard", true, false)).toBe(
      "markitai 'https://example.com/page?a=1&b' -o out/ --preset standard --llm --no-ocr",
    );
  });

  it("escapes embedded single quotes so the shell reassembles one word", () => {
    expect(buildCliCommand(["https://example.com/it's"], "minimal", false, false)).toBe(
      "markitai 'https://example.com/it'\\''s' -o out/ --preset minimal --no-llm --no-ocr",
    );
  });

  it("quotes inputs containing spaces", () => {
    expect(buildCliCommand(["my file.pdf"], "rich", true, true)).toBe(
      "markitai 'my file.pdf' -o out/ --preset rich --llm --ocr",
    );
  });

  it("quotes a leading tilde but leaves an interior tilde bare", () => {
    expect(buildCliCommand(["~/docs/report.pdf"], "standard", false, true)).toBe(
      "markitai '~/docs/report.pdf' -o out/ --preset standard --no-llm --ocr",
    );
    expect(buildCliCommand(["https://example.com/~user/page"], "standard", false, true)).toBe(
      "markitai https://example.com/~user/page -o out/ --preset standard --no-llm --ocr",
    );
  });

  it("falls back to the file placeholder when no URLs are typed", () => {
    expect(buildCliCommand([], "standard", true, false)).toBe(
      "markitai <your-files> -o out/ --preset standard --llm --no-ocr",
    );
  });
});
