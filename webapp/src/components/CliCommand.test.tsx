import { afterEach, describe, expect, it, vi } from "vitest";
import { copyTextToClipboard } from "./CliCommand";

describe("copyTextToClipboard legacy fallback", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("restores focus to the element focused before the copy", async () => {
    // Force the execCommand path: navigator.clipboard is absent on insecure
    // (LAN http) origins, exactly where this fallback runs.
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: undefined,
    });
    // Real browsers move focus to the textarea on select(); jsdom does not, so
    // emulate the focus theft to exercise the restore path.
    vi.spyOn(HTMLTextAreaElement.prototype, "select").mockImplementation(
      function (this: HTMLTextAreaElement) {
        this.focus();
      },
    );
    // jsdom does not implement execCommand at all; define a stub for the path.
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: () => true,
    });

    const button = document.createElement("button");
    document.body.append(button);
    button.focus();
    expect(document.activeElement).toBe(button);

    const ok = await copyTextToClipboard("markitai run");

    expect(ok).toBe(true);
    expect(document.activeElement).toBe(button);
    button.remove();
  });
});
