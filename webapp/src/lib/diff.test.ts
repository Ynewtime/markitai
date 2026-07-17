import { describe, expect, it } from "vitest";

import { diffLines, type DiffLine } from "./diff";

describe("diffLines", () => {
  it("keeps shared prefix/suffix as context around an interior edit", () => {
    const a = ["top", "keep", "old line", "bottom"].join("\n");
    const b = ["top", "keep", "new line", "bottom"].join("\n");
    const expected: DiffLine[] = [
      { type: "ctx", text: "top", aNo: 1, bNo: 1 },
      { type: "ctx", text: "keep", aNo: 2, bNo: 2 },
      { type: "del", text: "old line", aNo: 3, bNo: null },
      { type: "add", text: "new line", aNo: null, bNo: 3 },
      { type: "ctx", text: "bottom", aNo: 4, bNo: 4 },
    ];
    expect(diffLines(a, b)).toEqual(expected);
  });

  it("returns only context lines for identical inputs", () => {
    const text = "alpha\nbeta";
    const expected: DiffLine[] = [
      { type: "ctx", text: "alpha", aNo: 1, bNo: 1 },
      { type: "ctx", text: "beta", aNo: 2, bNo: 2 },
    ];
    expect(diffLines(text, text)).toEqual(expected);
  });

  it("treats two empty inputs as one shared empty line", () => {
    // "" splits to [""], so both sides share a single empty context line.
    expect(diffLines("", "")).toEqual([{ type: "ctx", text: "", aNo: 1, bNo: 1 }]);
  });

  it("numbers a pure insert against both sides", () => {
    const expected: DiffLine[] = [
      { type: "ctx", text: "a", aNo: 1, bNo: 1 },
      { type: "add", text: "b", aNo: null, bNo: 2 },
      { type: "ctx", text: "c", aNo: 2, bNo: 3 },
    ];
    expect(diffLines("a\nc", "a\nb\nc")).toEqual(expected);
  });

  it("numbers a pure delete against both sides", () => {
    const expected: DiffLine[] = [
      { type: "ctx", text: "a", aNo: 1, bNo: 1 },
      { type: "del", text: "b", aNo: 2, bNo: null },
      { type: "ctx", text: "c", aNo: 3, bNo: 2 },
    ];
    expect(diffLines("a\nb\nc", "a\nc")).toEqual(expected);
  });
});
