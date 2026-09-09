import { describe, expect, it } from "vitest";
import type { JobOptions } from "../api/types";
import { emptyJobOptions, jobOptionsFromSnapshot } from "./jobOptions";

describe("jobOptionsFromSnapshot", () => {
  it("drops server bookkeeping keys and fills options the snapshot lacks", () => {
    const raw = {
      preset: "standard",
      llm: false,
      ocr: true,
      origin: "cli",
    } as unknown as JobOptions;

    const options = jobOptionsFromSnapshot(raw);

    expect(options).not.toHaveProperty("origin");
    expect(Object.keys(options).sort()).toEqual(Object.keys(emptyJobOptions()).sort());
    expect(options).toMatchObject({ preset: "standard", llm: false, ocr: true, alt: null });
  });

  it("keeps every declared option a full snapshot carries", () => {
    const full: JobOptions = {
      ...emptyJobOptions(),
      preset: "rich",
      llm: true,
      alt: true,
      strategy: "auto",
    };
    expect(jobOptionsFromSnapshot(full)).toEqual(full);
  });
});
