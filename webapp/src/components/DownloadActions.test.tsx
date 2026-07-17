import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { ClearJobsButton, DownloadArchiveButton } from "./DownloadActions";

// jsdom does not implement object URLs — give vi.spyOn real methods to wrap
// so the URL constructor itself stays intact.
if (typeof URL.createObjectURL !== "function") {
  Object.assign(URL, {
    createObjectURL: (): string => "",
    revokeObjectURL: (): void => undefined,
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function renderArchiveButton(onDownloadError = vi.fn()) {
  render(
    <DownloadArchiveButton
      t={dicts.en}
      zipHref="/api/history/archive"
      activeCount={0}
      onDownloadError={onDownloadError}
    />,
  );
  return onDownloadError;
}

describe("DownloadArchiveButton", () => {
  it("shows progress while the history archive is being prepared", async () => {
    let finish: ((response: Response) => void) | undefined;
    const pending = new Promise<Response>((resolve) => {
      finish = resolve;
    });
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(pending));
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:archive");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    const user = userEvent.setup();

    renderArchiveButton();

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
  });

  it("reports a non-OK response and re-enables the button", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(new Response(null, { status: 403, statusText: "Forbidden" })),
    );
    const user = userEvent.setup();

    const onDownloadError = renderArchiveButton();

    await user.click(screen.getByRole("button", { name: dicts.en.downloadAllZip }));
    await waitFor(() => expect(onDownloadError).toHaveBeenCalledWith("403 Forbidden"));
    expect(
      screen.getByRole("button", { name: dicts.en.downloadAllZip }),
    ).toBeEnabled();
  });

  it("decodes the RFC 5987 filename* content-disposition variant", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(new Blob(["zip"]), {
          status: 200,
          headers: {
            "content-disposition":
              "attachment; filename*=UTF-8''%E6%96%87%E6%A1%A3%20final.zip",
          },
        }),
      ),
    );
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:archive");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
    let downloadName: string | null = null;
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {
      // The component appends the anchor before the synthetic click, so the
      // suggested filename is observable from the document.
      downloadName = document.querySelector("a[download]")?.getAttribute("download") ?? null;
    });
    const user = userEvent.setup();

    renderArchiveButton();

    await user.click(screen.getByRole("button", { name: dicts.en.downloadAllZip }));
    await waitFor(() => expect(downloadName).toBe("文档 final.zip"));
  });

  it("stays visible but disabled while conversions run", () => {
    render(
      <DownloadArchiveButton
        t={dicts.en}
        zipHref={null}
        activeCount={2}
        onDownloadError={vi.fn()}
      />,
    );

    const button = screen.getByRole("button", { name: dicts.en.downloadAllZip });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("title", dicts.en.zipWhileRunning);
  });

  it("renders nothing when there is no archive and nothing is running", () => {
    render(
      <DownloadArchiveButton
        t={dicts.en}
        zipHref={null}
        activeCount={0}
        onDownloadError={vi.fn()}
      />,
    );

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});

describe("ClearJobsButton", () => {
  it("clears everything when idle and only settled rows while running", async () => {
    const onClear = vi.fn();
    const user = userEvent.setup();
    const { rerender } = render(
      <ClearJobsButton t={dicts.en} activeCount={0} clearableJobCount={0} onClear={onClear} />,
    );

    await user.click(screen.getByRole("button", { name: dicts.en.clearAll }));
    expect(onClear).toHaveBeenCalledOnce();

    // Running with nothing settled: the completed-only variant has no work.
    rerender(
      <ClearJobsButton t={dicts.en} activeCount={1} clearableJobCount={0} onClear={onClear} />,
    );
    const completedOnly = screen.getByRole("button", { name: dicts.en.clearCompleted });
    expect(completedOnly).toBeDisabled();
    expect(completedOnly).toHaveAttribute("title", dicts.en.nothingCompleted);

    rerender(
      <ClearJobsButton t={dicts.en} activeCount={1} clearableJobCount={1} onClear={onClear} />,
    );
    expect(
      screen.getByRole("button", { name: dicts.en.clearCompleted }),
    ).toBeEnabled();
  });
});
