import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { ADVANCED_DEFAULTS } from "../lib/advanced";
import { AdvancedOptions } from "./AdvancedOptions";

const base = { t: dicts.en, value: ADVANCED_DEFAULTS, llm: true };

function open(props: Partial<Parameters<typeof AdvancedOptions>[0]> = {}) {
  const onChange = vi.fn();
  render(<AdvancedOptions {...base} onChange={onChange} {...props} />);
  fireEvent.click(screen.getByRole("button", { name: dicts.en.advanced }));
  return onChange;
}

describe("AdvancedOptions", () => {
  it("stays collapsed until asked for", () => {
    render(<AdvancedOptions {...base} onChange={vi.fn()} />);
    expect(screen.queryByLabelText(dicts.en.advAlt)).not.toBeInTheDocument();
  });

  it("reports one changed option without touching the rest", () => {
    const onChange = open();
    fireEvent.click(screen.getByLabelText(dicts.en.advPure));
    expect(onChange).toHaveBeenCalledWith({ ...ADVANCED_DEFAULTS, pure: true });
  });

  it("disables image analysis when LLM is off", () => {
    // The CLI rejects --alt/--desc without --llm. Showing them disabled keeps
    // the capability discoverable; hiding them would not say why.
    open({ llm: false });
    expect(screen.getByLabelText(dicts.en.advAlt)).toBeDisabled();
    expect(screen.getByLabelText(dicts.en.advDesc)).toBeDisabled();
    expect(screen.getByLabelText(dicts.en.advPure)).not.toBeDisabled();
  });

  it("shows page screenshots as on once the source is screenshots", () => {
    // --screenshot-only implies --screenshot, so an unchecked box here would
    // contradict what the job actually runs with.
    open({ value: { ...ADVANCED_DEFAULTS, screenshotOnly: true } });
    const box = screen.getByLabelText(dicts.en.advScreenshot);
    expect(box).toBeChecked();
    expect(box).toBeDisabled();
  });

  it("marks the toggle once anything differs from the defaults", () => {
    const { rerender } = render(
      <AdvancedOptions {...base} onChange={vi.fn()} />,
    );
    expect(
      screen.queryByLabelText(dicts.en.advancedChanged),
    ).not.toBeInTheDocument();

    rerender(
      <AdvancedOptions
        {...base}
        value={{ ...ADVANCED_DEFAULTS, strategy: "jina" }}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByLabelText(dicts.en.advancedChanged)).toBeInTheDocument();
  });

  it("offers every fetch strategy and backend the CLI accepts", () => {
    open();
    for (const name of ["auto", "static", "playwright", "defuddle", "jina"]) {
      expect(
        screen.getByRole("option", { name, selected: name === "auto" }),
      ).toBeInTheDocument();
    }
    expect(screen.getByRole("option", { name: "kreuzberg" })).toBeInTheDocument();
  });
});
