import { describe, expect, it } from "vitest";

import { countWords, fmtBytes, fmtDateTime, shortError } from "./format";

describe("countWords", () => {
  it("counts CJK per character and Latin per word", () => {
    expect(countWords("hello 世界 world")).toBe(4);
  });

  it("counts supplementary-plane CJK per character", () => {
    // U+2000B sits in CJK Unified Ideographs Extension B.
    expect(countWords("see \u{2000B}\u{2000B} here")).toBe(4);
  });

  it("counts kana per character", () => {
    expect(countWords("ひらがな")).toBe(4);
  });

  it("does not treat Hebrew as CJK", () => {
    expect(countWords("שלום עולם")).toBe(2);
    // U+FB20-FB22 are Hebrew presentation forms near the CJK compat block.
    expect(countWords("ﬠﬡﬢ")).toBe(1);
  });
});

describe("fmtBytes", () => {
  it("switches units at the KB and MB boundaries", () => {
    expect(fmtBytes(0)).toBe("0 B");
    expect(fmtBytes(1023)).toBe("1023 B");
    expect(fmtBytes(1024)).toBe("1.0 KB");
    expect(fmtBytes(1024 * 1024 - 1)).toBe("1024.0 KB");
    expect(fmtBytes(1024 * 1024)).toBe("1.0 MB");
  });
});

describe("shortError", () => {
  it("keeps only the first line and strips the exception-class prefix", () => {
    expect(shortError("RuntimeError: boom\nTraceback...")).toBe("boom");
  });

  it("strips the all-strategies preamble down to the detail", () => {
    expect(shortError("All fetch strategies failed for https://x.test/a - 403 Forbidden")).toBe(
      "403 Forbidden",
    );
  });

  it("caps long messages at 120 characters with an ellipsis", () => {
    const long = "x".repeat(200);
    const out = shortError(long);
    expect(out).toHaveLength(120);
    expect(out).toBe(`${"x".repeat(119)}…`);
    expect(shortError("y".repeat(120))).toBe("y".repeat(120));
  });
});

describe("fmtDateTime", () => {
  it("renders '-' for null or too-short input", () => {
    expect(fmtDateTime(null)).toBe("-");
    expect(fmtDateTime("2026-07-12")).toBe("-");
  });

  it("renders a compact date and time", () => {
    expect(fmtDateTime("2026-07-12T14:30:00")).toBe("07-12 14:30");
  });
});
