import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { ADVANCED_DEFAULTS } from "../lib/advanced";
import { OptionsPanel } from "./OptionsPanel";

const t = dicts.en;
const base = {
  t,
  value: ADVANCED_DEFAULTS,
  preset: "minimal" as const,
  llm: true,
  ocr: false,
  profile: null,
  llmConfigured: true,
  urls: [] as string[],
  announce: vi.fn(),
  onChange: vi.fn(),
  onPreset: vi.fn(),
  onLlm: vi.fn(),
  onOcr: vi.fn(),
  onProfile: vi.fn(),
};

type Props = Parameters<typeof OptionsPanel>[0];

function panel(props: Partial<Props> = {}) {
  const handlers = {
    onChange: vi.fn(),
    onPreset: vi.fn(),
    onLlm: vi.fn(),
    onOcr: vi.fn(),
    onProfile: vi.fn(),
  };
  render(<OptionsPanel {...base} {...handlers} {...props} />);
  return handlers;
}

function open(props: Partial<Props> = {}) {
  const handlers = panel(props);
  fireEvent.click(screen.getByRole("button", { name: t.options }));
  return handlers;
}

describe("OptionsPanel", () => {
  it("asks nothing until opened", () => {
    panel();
    expect(screen.queryByLabelText(t.advAlt)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(t.llmEnhance)).not.toBeInTheDocument();
  });

  it("says what is in effect while collapsed", () => {
    // Every option lives behind the disclosure now, so without this the
    // screen would show nothing to say the next conversion costs money.
    panel({ llm: true, ocr: true, profile: "rag" });
    const summary = screen.getByLabelText(t.optionsChanged);
    expect(summary).toHaveTextContent(t.ocr);
    expect(summary).toHaveTextContent("rag");
  });

  it("says nothing when everything is at its default", () => {
    panel({ llm: false, ocr: false, profile: null });
    expect(screen.queryByLabelText(t.optionsChanged)).not.toBeInTheDocument();
  });

  it("marks the summary once an advanced option differs", () => {
    panel({
      llm: false,
      ocr: false,
      profile: null,
      value: { ...ADVANCED_DEFAULTS, strategy: "jina" },
    });
    expect(screen.getByLabelText(t.optionsChanged)).toBeInTheDocument();
  });

  it("carries the primary switches", () => {
    const { onLlm, onOcr, onProfile } = open();
    fireEvent.click(screen.getByLabelText(t.llmEnhance));
    expect(onLlm).toHaveBeenCalledWith(false);
    fireEvent.click(screen.getByLabelText(t.ocr));
    expect(onOcr).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByRole("button", { name: "rag" }));
    expect(onProfile).toHaveBeenCalledWith("rag");
  });

  it("keeps OCR reachable when no LLM is configured", () => {
    open({ llmConfigured: false, llm: false });
    expect(screen.getByLabelText(t.ocr)).toBeInTheDocument();
    // Promising LLM work the server cannot route would be worse than an
    // absent row, so the enhancement switch is left out entirely.
    expect(screen.queryByLabelText(t.llmEnhance)).not.toBeInTheDocument();
  });

  it("offers the preset only while LLM is on", () => {
    open({ llm: false });
    expect(
      screen.queryByRole("button", { name: "standard" }),
    ).not.toBeInTheDocument();
  });

  it("reports one changed option without touching the rest", () => {
    const { onChange } = open();
    fireEvent.click(screen.getByLabelText(t.advPure));
    expect(onChange).toHaveBeenCalledWith({ ...ADVANCED_DEFAULTS, pure: true });
  });

  it("disables image analysis when LLM is off", () => {
    // The CLI rejects --alt/--desc without --llm. Showing them disabled keeps
    // the capability discoverable; hiding them would not say why.
    open({ llm: false });
    expect(screen.getByLabelText(t.advAlt)).toBeDisabled();
    expect(screen.getByLabelText(t.advDesc)).toBeDisabled();
    expect(screen.getByLabelText(t.advPure)).not.toBeDisabled();
  });

  it("shows page screenshots as on once the source is screenshots", () => {
    // --screenshot-only implies --screenshot, so an unchecked box here would
    // contradict what the job actually runs with.
    open({ value: { ...ADVANCED_DEFAULTS, screenshotOnly: true } });
    const box = screen.getByLabelText(t.advScreenshot);
    expect(box).toBeChecked();
    expect(box).toBeDisabled();
  });

  it("shows the command those settings produce, in the panel", () => {
    // It is the settings spelled out, not a sibling of them — a second
    // disclosure made it look like an unrelated feature.
    open({ llm: true, ocr: false });
    expect(screen.getByLabelText(t.cliAria)).toHaveTextContent("--llm");
    expect(screen.getByRole("button", { name: t.copy })).toBeInTheDocument();
  });

  it("offers every fetch strategy and backend the CLI accepts", () => {
    open();
    for (const name of ["auto", "static", "playwright", "defuddle", "jina"]) {
      expect(
        screen.getByRole("option", { name, selected: name === "auto" }),
      ).toBeInTheDocument();
    }
    expect(
      screen.getByRole("option", { name: "kreuzberg" }),
    ).toBeInTheDocument();
  });
});
