import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { DownloadActions } from "./DownloadActions";

afterEach(() => vi.unstubAllGlobals());

describe("DownloadActions", () => {
  it("shows progress while the history archive is being prepared", async () => {
    let finish: ((response: Response) => void) | undefined;
    const pending = new Promise<Response>((resolve) => {
      finish = resolve;
    });
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(pending));
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn().mockReturnValue("blob:archive"),
      revokeObjectURL: vi.fn(),
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    const user = userEvent.setup();

    render(
      <DownloadActions
        t={dicts.en}
        zipHref="/api/history/archive"
        showClear={false}
        activeCount={0}
        clearableJobCount={0}
        onClear={() => undefined}
        onDownloadError={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: dicts.en.downloadAllZip }));
    expect(
      screen.getByRole("button", { name: dicts.en.downloadingZip }),
    ).toHaveAttribute("aria-busy", "true");

    finish?.(new Response(new Blob(["zip"]), { status: 200 }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: dicts.en.downloadAllZip }),
      ).toBeEnabled(),
    );
    expect(click).toHaveBeenCalledOnce();
    click.mockRestore();
  });
});
