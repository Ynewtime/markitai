import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { UrlInput } from "./UrlInput";
import { readFileSync } from "node:fs";

const appStyles = readFileSync("src/styles/app.css", "utf8");

function ControlledInput({ onConvert }: { onConvert: (urls: string[]) => Promise<boolean> }) {
  const [text, setText] = useState("");
  return (
    <UrlInput
      t={dicts.en}
      text={text}
      onText={setText}
      onConvert={onConvert}
    />
  );
}

describe("UrlInput", () => {
  it.each([false, true])("keeps bottom-only focus styling for multiline input (compact=%s)", (compact) => {
    render(<UrlInput t={dicts.en} text={"https://one.test\nhttps://two.test"}
      onText={vi.fn()} onConvert={vi.fn()} compact={compact} />);
    const input = screen.getByRole("textbox");
    input.focus();
    expect(input).toHaveFocus();
    expect(input).toHaveAttribute("rows", "2");
    // jsdom does not lay out CSS layers: guard both focus selectors and against
    // reintroducing the later shared focus-visible outline that caused the box.
    const rules = [...appStyles.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
      .filter((match) => match[1]!.includes(".urlin:focus"));
    expect(rules).toHaveLength(1);
    expect(rules[0]![1]).toContain(".urlrow .urlin:focus,");
    expect(rules[0]![1]).toContain(".urlrow .urlin:focus-visible");
    expect(rules[0]![2]).toContain("border: 0;");
    expect(rules[0]![2]).toContain("outline: none;");
    expect(rules[0]![2]).toContain("box-shadow: inset 0 -1px 0 var(--text-2);");
  });
  it("keeps only the textarea and Convert in the input row", () => {
    render(<ControlledInput onConvert={vi.fn().mockResolvedValue(true)} />);
    const input = screen.getByRole("textbox");
    const convert = screen.getByRole("button", { name: "Convert" });
    expect(input.closest(".urlrow")).toContainElement(convert);
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.queryByText("Upload")).not.toBeInTheDocument();
    expect(convert).toHaveClass("srcact", "convert");
    expect(convert).toBeDisabled();
  });

  it("converts by button and keeps a failed draft", async () => {
    const onConvert = vi.fn().mockResolvedValue(false);
    render(<ControlledInput onConvert={onConvert} />);
    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "https://example.com" } });
    fireEvent.click(screen.getByRole("button", { name: "Convert" }));
    await waitFor(() => expect(onConvert).toHaveBeenCalledWith(["https://example.com"]));
    expect(input).toHaveValue("https://example.com");
  });

  it("trims multiline URLs, clears a successful draft, and restores focus", async () => {
    const onConvert = vi.fn().mockResolvedValue(true);
    render(<ControlledInput onConvert={onConvert} />);

    const input = screen.getByRole("textbox", { name: dicts.en.urlPlaceholder });
    fireEvent.change(input, {
      target: { value: " https://example.com/a \n\nhttps://example.com/b  " },
    });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => {
      expect(onConvert).toHaveBeenCalledWith([
        "https://example.com/a",
        "https://example.com/b",
      ]);
      expect(input).toHaveValue("");
      expect(input).toHaveFocus();
    });
  });
});
