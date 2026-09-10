import { describe, expect, it } from "vitest";

import {
  countWords,
  durParts,
  fmtBytes,
  fmtCost,
  fmtDateTime,
  fmtDur,
  shortError,
} from "./format";

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

  it("renders a naive server-local timestamp as written", () => {
    expect(fmtDateTime("2026-07-12T14:30:00")).toBe("07-12 14:30");
  });

  it("converts an offset-aware timestamp into the reader's zone", () => {
    const iso = "2026-07-12T14:30:00Z";
    const local = new Date(iso);
    const pad = (n: number) => String(n).padStart(2, "0");
    const expected = `${pad(local.getMonth() + 1)}-${pad(local.getDate())} ${pad(local.getHours())}:${pad(local.getMinutes())}`;
    expect(fmtDateTime(iso)).toBe(expected);
  });
});

describe("fmtDur", () => {
  it("keeps one decimal under a minute", () => {
    expect(fmtDur(4200)).toBe("4.2s");
    expect(fmtDur(59_900)).toBe("59.9s");
  });

  it("switches to m:ss and h:mm:ss above a minute", () => {
    expect(fmtDur(60_000)).toBe("1:00");
    expect(fmtDur(83_000)).toBe("1:23");
    expect(fmtDur(3_723_000)).toBe("1:02:03");
  });

  it("never prints a rounded 60.0s", () => {
    // 59.96s rounds to 60.0 at one decimal; the label must roll over instead.
    expect(fmtDur(59_960)).toBe("1:00");
  });
});

describe("durParts", () => {
  it("keeps the tenths below a minute and flags them with minutes = 0", () => {
    expect(durParts(4200)).toEqual({ minutes: 0, seconds: 4.2 });
    expect(durParts(59_900)).toEqual({ minutes: 0, seconds: 59.9 });
  });

  it("splits whole minutes above a minute", () => {
    expect(durParts(60_000)).toEqual({ minutes: 1, seconds: 0 });
    expect(durParts(83_000)).toEqual({ minutes: 1, seconds: 23 });
    expect(durParts(3_723_000)).toEqual({ minutes: 62, seconds: 3 });
  });

  it("keeps both labels on one rounding step", () => {
    // The spoken label reads these parts, so a value fmtDur rolls over must
    // roll over here too, and a negative duration cannot reach the label.
    expect(durParts(59_960)).toEqual({ minutes: 1, seconds: 0 });
    expect(fmtDur(59_960)).toBe("1:00");
    expect(`${durParts(4200).seconds.toFixed(1)}s`).toBe(fmtDur(4200));
    expect(durParts(-5)).toEqual({ minutes: 0, seconds: 0 });
  });
});

describe("fmtCost", () => {
  it("drops meaningless trailing zeros", () => {
    expect(fmtCost(0)).toBe("$0");
    expect(fmtCost(0.0123)).toBe("$0.0123");
    expect(fmtCost(0.01)).toBe("$0.01");
    expect(fmtCost(1.5)).toBe("$1.5");
    expect(fmtCost(12)).toBe("$12");
  });
});
