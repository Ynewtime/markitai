import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { OptionsBar } from "./OptionsBar";

const baseProps = {
  t: dicts.en,
  preset: "minimal" as const,
  urls: [],
  ocr: false,
  announce: vi.fn(),
  onPreset: vi.fn(),
  onLlm: vi.fn(),
  onOcr: vi.fn(),
};

describe("OptionsBar", () => {
  it("keeps OCR available when no LLM is configured", () => {
    const onOcr = vi.fn();
    render(
      <OptionsBar
        {...baseProps}
        llm={false}
        llmConfigured={false}
        onOcr={onOcr}
      />,
    );

    expect(screen.getByRole("switch", { name: "OCR" })).toBeVisible();
    fireEvent.click(screen.getByRole("switch", { name: "OCR" }));
    expect(onOcr).toHaveBeenCalledWith(true);
    expect(
      screen.queryByRole("switch", { name: "LLM enhancement" }),
    ).not.toBeInTheDocument();
  });

  it("shows presets only after LLM enhancement is enabled", () => {
    const onLlm = vi.fn();
    const { rerender } = render(
      <OptionsBar
        {...baseProps}
        llm={false}
        llmConfigured
        onLlm={onLlm}
      />,
    );

    expect(screen.getByRole("switch", { name: "LLM enhancement" })).toBeVisible();
    expect(screen.queryByRole("group", { name: "Preset" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("switch", { name: "LLM enhancement" }));
    expect(onLlm).toHaveBeenCalledWith(true);

    rerender(
      <OptionsBar {...baseProps} llm llmConfigured onLlm={onLlm} />,
    );
    const llmSwitch = screen.getByRole("switch", { name: "LLM enhancement" });
    const presets = screen.getByRole("group", { name: "Preset" });
    expect(presets).toBeVisible();
    expect(
      llmSwitch.compareDocumentPosition(presets) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "minimal" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("shortens the LLM label on the phone tier but keeps the full accessible name", () => {
    // The setup-file matchMedia stub always reports false; report true for
    // the phone query so the component takes its narrow branch.
    const original = window.matchMedia;
    window.matchMedia = ((query: string) => ({
      ...original(query),
      matches: query === "(max-width: 780px)",
    })) as typeof window.matchMedia;
    try {
      render(<OptionsBar {...baseProps} llm llmConfigured />);

      expect(screen.getByText("LLM")).toBeVisible();
      expect(screen.queryByText("LLM enhancement")).not.toBeInTheDocument();
      // aria-label keeps the full wording even while the visible label is short
      expect(screen.getByRole("switch", { name: "LLM enhancement" })).toBeVisible();
      // the visually hidden #preset-lbl still names the segment group
      expect(screen.getByRole("group", { name: "Preset" })).toBeInTheDocument();
    } finally {
      window.matchMedia = original;
    }
  });
});
