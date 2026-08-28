import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { ADVANCED_DEFAULTS } from "../lib/advanced";
import { OptionsBar } from "./OptionsBar";

const t = dicts.en;
const baseProps = {
  t,
  preset: "minimal" as const,
  urls: [],
  llm: false,
  ocr: false,
  profile: null,
  advanced: ADVANCED_DEFAULTS,
  llmConfigured: false,
  announce: vi.fn(),
  onPreset: vi.fn(),
  onLlm: vi.fn(),
  onOcr: vi.fn(),
  onProfile: vi.fn(),
  onAdvanced: vi.fn(),
};

/** The row itself only composes: the options live in OptionsPanel and are
 * tested there. What matters here is that the default screen asks nothing. */
describe("OptionsBar", () => {
  it("asks nothing on the default screen", () => {
    render(<OptionsBar {...baseProps} />);

    // The product's promise is drop-a-file-get-markdown; every switch,
    // selector and the CLI command are one click away rather than in the way.
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.queryByText(/markitai/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: t.options })).toBeInTheDocument();
  });

  it("opens every option and the command they produce at once", () => {
    render(<OptionsBar {...baseProps} llm llmConfigured />);

    fireEvent.click(screen.getByRole("button", { name: t.options }));

    expect(screen.getByRole("switch", { name: t.llmEnhance })).toBeVisible();
    expect(screen.getByLabelText(t.cliAria)).toBeVisible();
  });

  it("renders the trailing slot after the disclosure", () => {
    render(<OptionsBar {...baseProps} trailing={<button>archive</button>} />);

    expect(screen.getByRole("button", { name: "archive" })).toBeInTheDocument();
  });
});
