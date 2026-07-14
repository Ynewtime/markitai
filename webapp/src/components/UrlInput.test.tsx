import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { UrlInput } from "./UrlInput";

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
