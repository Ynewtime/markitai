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
});
