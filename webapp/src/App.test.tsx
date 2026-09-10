import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { dicts } from "./i18n";

const mocks = vi.hoisted(() => ({
  submit: vi.fn(),
  openJob: vi.fn(),
  enhanceArchived: vi.fn(),
  retry: vi.fn(),
  // Job cap served by the mocked /api/capabilities below.
  maxJobItems: 50,
  // Per-test create-job failure, surfaced through the mocked useJobs below.
  submitError: null as null | { status: number; message: string },
  // Per-test ledger rows; null keeps the default single completed row.
  failedItems: null as null | Array<Record<string, unknown>>,
  // Per-test session-restore failures and running batch size.
  restoreFailed: null as null | string[],
  retryRestore: vi.fn(),
  activeCount: 0,
}));
const MAX_JOB_ITEMS = mocks.maxJobItems;

/** One failed ledger row for the retry-all action. */
function failedRow(itemId: string): Record<string, unknown> {
  return {
    key: `job-1/${itemId}`,
    jobId: "job-1",
    itemId,
    name: `${itemId}.pdf`,
    kind: "file",
    status: "error",
    error: "conversion failed",
    output: null,
    durationMs: 100,
    finishedAt: "2026-07-13T10:00:01Z",
    costUsd: null,
    llmEnhanced: false,
    operation: "convert",
    skipped: false,
    skipReason: null,
    sizeBytes: null,
    startedAt: null,
  };
}

function archivedSnapshot() {
  return {
    job_id: "job-2",
    status: "done",
    done: 1,
    failed: 0,
    total: 1,
    created_at: "2026-07-12T10:00:00Z",
    finished_at: "2026-07-12T10:01:00Z",
    options: { preset: "standard", llm: false, ocr: false },
    items: [
      {
        item_id: "item-2",
        name: "archived.pdf",
        kind: "file",
        status: "done",
        error: null,
        output: "archived.md",
        output_name: null,
        duration_ms: 100,
        finished_at: "2026-07-12T10:01:00Z",
        cost_usd: null,
        llm_enhanced: false,
        operation: "convert",
        skipped: false,
        skip_reason: null,
      },
    ],
  };
}

vi.mock("./api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api/client")>();
  return {
    ...actual,
    fetchCapabilities: vi.fn().mockResolvedValue({
      version: "test",
      llm: { configured: true, routable: true, effective: true, models: [] },
      presets: ["minimal", "standard", "rich"],
      preset_options: (await import("./lib/conversionOptions")).BUILTIN_PRESET_OPTIONS,
      extras: { browser: false, svg: false },
      limits: { max_job_items: mocks.maxJobItems },
    }),
  };
});

vi.mock("./hooks/useArchivedJobs", () => ({
  useArchivedJobs: () => ({
    entries: [
      {
        job_id: "job-2",
        created_at: "2026-07-12T10:00:00Z",
        finished_at: "2026-07-12T10:01:00Z",
        status: "done",
        total: 1,
        done: 1,
        failed: 0,
        skipped: 0,
        llm_enhanced: 0,
        cost_usd: 0,
        names_preview: ["archived.pdf"],
        kinds_preview: ["file"],
        duration_ms: 60_000,
        size_bytes: 100,
        origin: "web",
      },
    ],
    error: null,
    refreshing: false,
    actions: {},
    rowErrors: {},
    refresh: vi.fn().mockResolvedValue(undefined),
    openJob: mocks.openJob,
    deleteJob: vi.fn().mockResolvedValue(true),
  }),
}));

vi.mock("./hooks/useJobs", () => ({
  useJobs: () => ({
    items: mocks.failedItems ?? [
      {
        key: "job-1/item-1",
        jobId: "job-1",
        itemId: "item-1",
        name: "result.md",
        kind: "file",
        status: "done",
        error: null,
        output: "result.md",
        durationMs: 120,
        finishedAt: "2026-07-13T10:00:01Z",
        costUsd: null,
        llmEnhanced: false,
        operation: "convert",
        skipped: false,
        skipReason: null,
        sizeBytes: 8,
        startedAt: null,
      },
    ],
    jobs: {
      "job-1": {
        jobId: "job-1",
        status: "done",
        createdAt: "2026-07-13T10:00:00Z",
        options: { preset: "standard", llm: false, ocr: false },
      },
    },
    stats: {
      done: 1,
      skipped: 0,
      failed: 0,
      total: 1,
      costTotal: 0,
      hasCost: false,
      doneDurationMs: 120,
    },
    running: false,
    activeCount: mocks.activeCount,
    submit: mocks.submit,
    retry: mocks.retry,
    enhance: vi.fn().mockResolvedValue(null),
    enhanceArchived: mocks.enhanceArchived,
    retryArchived: vi.fn().mockResolvedValue(null),
    deleteItem: vi.fn().mockResolvedValue(null),
    submitError: mocks.submitError,
    clear: vi.fn(),
    clearSettled: vi.fn(),
    terminalJobCount: 1,
    suppressedHistoryIds: new Set<string>(),
    historyRevision: 0,
    restoreFailedJobs: new Set(mocks.restoreFailed ?? []),
    retryRestore: mocks.retryRestore,
  }),
}));

describe("App workspace", () => {
  afterEach(() => vi.unstubAllGlobals());

  beforeEach(() => {
    window.history.replaceState(null, "", "/");
    mocks.submit.mockReset();
    mocks.submit.mockResolvedValue(true);
    mocks.openJob.mockReset();
    mocks.openJob.mockResolvedValue(archivedSnapshot());
    mocks.enhanceArchived.mockReset();
    mocks.enhanceArchived.mockResolvedValue(null);
    mocks.retry.mockReset();
    mocks.retry.mockResolvedValue(null);
    mocks.submitError = null;
    mocks.failedItems = null;
    mocks.restoreFailed = null;
    mocks.retryRestore.mockReset();
    mocks.retryRestore.mockResolvedValue(0);
    mocks.activeCount = 0;
  });

  it("names an unreachable server instead of printing HTTP 0", async () => {
    mocks.submitError = { status: 0, message: "Failed to fetch" };
    render(<App />);

    expect(await screen.findByText(dicts.en.submitNetworkFailed)).toBeVisible();
    expect(screen.queryByText(/HTTP 0/)).toBeNull();
  });

  it("keeps the status line for a known HTTP failure", async () => {
    mocks.submitError = { status: 413, message: "too large" };
    render(<App />);

    expect(await screen.findByText(dicts.en.submitTooLarge)).toBeVisible();
  });

  it("starts at home on / and navigates the task list to /jobs", async () => {
    render(<App />);

    expect(
      await screen.findByRole("heading", {
        name: "Drop files. Paste URLs. Get Markdown.",
      }),
    ).toBeVisible();
    // Every conversion option lives behind the Options disclosure now, and
    // the LLM row only appears once /api/capabilities reports a routable
    // deployment — so wait for the session link, which lands with it.
    await screen.findByRole("button", { name: /item in session/ });
    fireEvent.click(screen.getByRole("button", { name: "Options" }));
    const llmSwitch = await screen.findByRole("switch", {
      name: "LLM Enhancement",
    });
    expect(llmSwitch).toHaveAttribute("aria-checked", "false");
    expect(screen.getByRole("button", { name: dicts.en.presetMinimal })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /item in session/ }));
    const listbox = await screen.findByRole("listbox");
    expect(window.location.pathname).toBe("/jobs");
    const composer = screen.getByRole("textbox");
    expect(
      composer.compareDocumentPosition(listbox) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    const source = composer.closest(".convert-source");
    expect(source).not.toBeNull();
    expect(source?.querySelector(".file-picker")).toBeNull();
    expect(screen.getByRole("button", { name: "Options" }).closest(".jobhead")).not.toBeNull();
    expect(screen.getByLabelText("Upload", { selector: "input" }).closest(".jobhead")).not.toBeNull();
    const currentRow = screen.getByRole("option", { name: /result\.md/ });
    expect(currentRow.querySelector(".c-finished")).toHaveTextContent(
      new Date("2026-07-13T10:00:01Z").toLocaleString("en-CA", {
        month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
      }).replace(",", ""),
    );
    const enhance = screen.getByRole("button", {
      name: "Enhance result.md with LLM",
    });
    expect(enhance).toBeDisabled();
    // The workspace has its own options row, so its panel starts collapsed.
    fireEvent.click(screen.getByRole("button", { name: "Options" }));
    fireEvent.click(screen.getByRole("switch", { name: "LLM Enhancement" }));
    expect(enhance).toBeEnabled();
    expect(currentRow.querySelector(".c-status.archive-actions")).not.toBeNull();
    const archivedRow = screen.getByRole("option", { name: "Open archived.pdf" });
    expect(listbox.contains(archivedRow)).toBe(true);
    expect(archivedRow).not.toHaveClass("archived-row");
    expect(archivedRow.querySelector(".c-status.archive-actions")).not.toBeNull();
    const zipButtons = screen.getAllByRole("button", { name: /download all/i });
    expect(zipButtons).toHaveLength(1);
    const [listZip] = zipButtons;
    expect(listZip).toBeEnabled();
    expect(listZip!.closest(".composer")).toBeNull();
    expect(listZip!.closest(".list-zip")).not.toBeNull();
    expect(listbox.compareDocumentPosition(listZip!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // The phone footer mirrors the header's external links.
    const footer = document.querySelector("footer.app-footer");
    expect(footer?.querySelector('a[href="https://github.com/Ynewtime/markitai"]')).not.toBeNull();
    const clearButton = screen.getByRole("button", { name: "Clear all" });
    expect(clearButton.closest(".jobhead-r")).not.toBeNull();
    expect(clearButton.closest(".convert-source")).toBeNull();
    const historyButton = screen.getByRole("button", { name: "View conversions" });
    expect(historyButton).toBeVisible();
    expect(historyButton).toHaveAttribute("aria-current", "page");
    expect(historyButton).toHaveClass("on");
    historyButton.focus();
    fireEvent.click(historyButton);
    await waitFor(() => expect(currentRow).toHaveFocus());
  });

  it.each(["/", "/jobs"])("uploads from the top toolbar on %s", async (path) => {
    window.history.replaceState(null, "", path);
    render(<App />);
    const picker = await screen.findByLabelText("Upload", { selector: "input" });
    expect(picker.closest(path === "/" ? ".composer-toolbar" : ".jobhead")).not.toBeNull();
    expect(picker.closest(".urlrow")).toBeNull();
    const file = new File(["hello"], "hello.txt", { type: "text/plain" });
    fireEvent.change(picker, { target: { files: [file] } });
    await waitFor(() => expect(mocks.submit).toHaveBeenCalled());
    expect(mocks.submit).toHaveBeenCalledWith([file], [], expect.any(Object), expect.any(AbortSignal));
  });

  it("shows an abortable busy state while the create-job POST is in flight", async () => {
    let resolveSubmit: (value: boolean) => void = () => undefined;
    const seenSignal: { current: AbortSignal | null } = { current: null };
    mocks.submit.mockImplementation(
      (_files: File[], _urls: string[], _options: unknown, signal?: AbortSignal) => {
        seenSignal.current = signal ?? null;
        return new Promise<boolean>((resolve) => {
          resolveSubmit = resolve;
        });
      },
    );
    render(<App />);
    const picker = await screen.findByLabelText("Upload", { selector: "input" });
    const file = new File(["hello"], "hello.txt", { type: "text/plain" });

    fireEvent.change(picker, { target: { files: [file] } });

    // Deterministic feedback: a status line, a disabled input, and a cancel.
    const status = await screen.findByText(dicts.en.submitting);
    expect(status.closest("[role='status']")).not.toBeNull();
    expect(picker).toBeDisabled();
    expect(seenSignal.current?.aborted).toBe(false);
    const cancel = screen.getByRole("button", { name: dicts.en.cancelSubmit });
    fireEvent.click(cancel);
    // Cancel must actually abort the in-flight request, not just hide the row.
    expect(seenSignal.current?.aborted).toBe(true);
    expect(screen.queryByText(dicts.en.submitting)).not.toBeInTheDocument();

    await act(async () => {
      resolveSubmit(false);
    });
  });

  it("re-queues every failed row from one action", async () => {
    window.history.replaceState(null, "", "/jobs");
    mocks.failedItems = [failedRow("item-a"), failedRow("item-b")];
    render(<App />);

    const retryAll = await screen.findByRole("button", {
      name: dicts.en.retryAllFailed(2),
    });
    fireEvent.click(retryAll);

    await waitFor(() => expect(mocks.retry).toHaveBeenCalledTimes(2));
  });

  it("shows a failed session restore under the home composer too", async () => {
    mocks.restoreFailed = ["job-9"];
    render(<App />);

    // The restore notice used to live only inside the workspace, so a user who
    // never left the home view never learned the task list was not restored.
    const notice = await screen.findByText(dicts.en.restoreFailed);
    expect(notice.closest(".drop-main")).not.toBeNull();
    fireEvent.click(screen.getByRole("button", { name: dicts.en.restoreRetry }));
    await waitFor(() => expect(mocks.retryRestore).toHaveBeenCalledOnce());
    // A full recovery is announced through the polite live region.
    await waitFor(() =>
      expect(document.querySelector(".sr-only[role='status']")).toHaveTextContent(
        dicts.en.sessResults(1),
      ),
    );
  });

  it("retries only the failed rows and disables the action while it runs", async () => {
    window.history.replaceState(null, "", "/jobs");
    mocks.failedItems = [failedRow("item-a"), failedRow("item-b")];
    const gate: { release: (value: string | null) => void } = {
      release: () => undefined,
    };
    mocks.retry.mockImplementation(
      () =>
        new Promise<string | null>((resolve) => {
          gate.release = resolve;
        }),
    );
    render(<App />);

    // Running rows are not failed rows: the label counts what the click sends.
    const retryAll = await screen.findByRole("button", {
      name: dicts.en.retryAllFailed(2),
    });
    fireEvent.click(retryAll);
    await waitFor(() => expect(retryAll).toBeDisabled());
    expect(retryAll).toHaveAttribute("aria-busy", "true");
    // Rows re-queue one at a time, in list order.
    expect(mocks.retry).toHaveBeenCalledTimes(1);
    expect(mocks.retry.mock.calls[0]![0]).toMatchObject({ itemId: "item-a" });

    await act(async () => {
      gate.release(null);
    });
    await act(async () => {
      gate.release(null);
    });
    await waitFor(() => expect(mocks.retry).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(retryAll).toBeEnabled());
    expect(retryAll).not.toHaveAttribute("aria-busy");
    expect(document.querySelector(".sr-only[role='status']")).toHaveTextContent(
      dicts.en.announceRetryAll(2),
    );
  });

  it("names the view and the running count in the document title", async () => {
    mocks.activeCount = 3;
    window.history.replaceState(null, "", "/jobs");
    render(<App />);

    await waitFor(() =>
      expect(document.title).toBe(`3 · ${dicts.en.titleWorkspace}`),
    );
    fireEvent.click(await screen.findByRole("link", { name: dicts.en.homeAria }));
    await waitFor(() => expect(document.title).toBe(dicts.en.titleHome));
  });

  it("restores the task-list view when /jobs is refreshed", async () => {
    window.history.replaceState(null, "", "/jobs");

    render(<App />);

    expect(await screen.findByRole("listbox")).toBeVisible();
    expect(
      screen.queryByRole("heading", {
        name: "Drop files. Paste URLs. Get Markdown.",
      }),
    ).not.toBeInTheDocument();
    expect(window.location.pathname).toBe("/jobs");
    expect(screen.getByRole("button", { name: "View conversions" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    const firstRow = screen.getByRole("option", { name: /result\.md/ });
    await waitFor(() => expect(firstRow).not.toHaveFocus());
    expect(firstRow).toHaveAttribute("aria-selected", "false");

    fireEvent.focus(firstRow);
    expect(firstRow).toHaveAttribute("aria-selected", "true");
    fireEvent.pointerDown(screen.getByRole("main"));
    await waitFor(() => expect(firstRow).toHaveAttribute("aria-selected", "false"));
    expect(firstRow).not.toHaveFocus();
  });

  it("opens an archived option in place without moving it into the current session", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "View conversions" }));
    const listbox = await screen.findByRole("listbox");
    const archivedOption = screen.getByRole("option", { name: "Open archived.pdf" });

    fireEvent.click(archivedOption);

    expect(await screen.findByRole("dialog", { name: "archived.pdf" })).toBeVisible();
    expect(listbox.contains(archivedOption)).toBe(true);
  });

  it("reports a job-load failure inline when an archived enhance cannot fetch the job", async () => {
    window.history.replaceState(null, "", "/jobs");
    mocks.openJob.mockResolvedValue(null);
    render(<App />);

    await screen.findByRole("listbox");
    fireEvent.click(await screen.findByRole("button", { name: "Options" }));
    fireEvent.click(
      await screen.findByRole("switch", { name: "LLM Enhancement" }),
    );
    const wand = screen.getByRole("button", {
      name: "Enhance archived.pdf with LLM",
    });
    expect(wand).toBeEnabled();
    fireEvent.click(wand);

    expect(
      await screen.findByText("LLM enhancement failed: Could not load this job"),
    ).toBeVisible();
  });

  it("enhances an archived job without replaying the snapshot's bookkeeping keys", async () => {
    // Rehydrated and CLI-recorded jobs carry `origin` in their snapshot
    // options; the retry endpoint rejects unknown keys, so replaying the
    // snapshot verbatim failed with "Extra inputs are not permitted".
    window.history.replaceState(null, "", "/jobs");
    const snapshot = archivedSnapshot();
    snapshot.options = { ...snapshot.options, origin: "cli" } as typeof snapshot.options;
    mocks.openJob.mockResolvedValue(snapshot);
    render(<App />);

    await screen.findByRole("listbox");
    fireEvent.click(await screen.findByRole("button", { name: "Options" }));
    fireEvent.click(
      await screen.findByRole("switch", { name: "LLM Enhancement" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Enhance archived.pdf with LLM" }),
    );

    await waitFor(() => expect(mocks.enhanceArchived).toHaveBeenCalled());
    const sent = mocks.enhanceArchived.mock.calls[0]?.[2];
    expect(sent).not.toHaveProperty("origin");
    expect(sent).toMatchObject({ preset: "standard", llm: true, ocr: false });
  });

  it("keeps preset UI, manual overrides, and submitted options in sync", async () => {
    render(<App />);
    await act(async () => {});
    fireEvent.click(screen.getByRole("button", { name: "Options" }));
    const rich = screen.getByRole("button", { name: dicts.en.presetRich });
    await waitFor(() => expect(rich).toBeEnabled());
    fireEvent.click(rich);
    expect(screen.getByRole("switch", { name: "LLM Enhancement" })).toBeChecked();
    expect(screen.getByRole("switch", { name: "Alt Text" })).toBeChecked();
    expect(screen.getByRole("switch", { name: "Description JSON" })).toBeChecked();
    expect(screen.getByRole("switch", { name: "Page Screenshots" })).toBeChecked();
    fireEvent.click(screen.getByRole("switch", { name: "Alt Text" }));
    expect(screen.getByText("Custom")).toBeVisible();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "https://example.com/article" } });
    fireEvent.click(screen.getByRole("button", { name: "Convert" }));
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledOnce());
    expect(mocks.submit.mock.calls[0]![2]).toMatchObject({
      preset: "rich", llm: true, ocr: false, alt: false, desc: true, screenshot: true,
    });
  });

  it("restores image overrides with the preset but not session-only source settings", async () => {
    const entries = new Map<string, string>();
    const storage = {
      getItem: (key: string) => entries.get(key) ?? null,
      setItem: (key: string, value: string) => entries.set(key, value),
      clear: () => entries.clear(),
    };
    vi.stubGlobal("localStorage", storage);
    storage.setItem("markitai.options", JSON.stringify({
      version: 4, preset: "rich", llm: true, ocr: false, profile: null,
      imageOverrides: { alt: false, desc: null, screenshot: false },
    }));
    render(<App />);
    await act(async () => {});
    fireEvent.click(screen.getByRole("button", { name: "Options" }));
    expect(screen.getByRole("switch", { name: "Alt Text" })).not.toBeChecked();
    expect(screen.getByRole("switch", { name: "Description JSON" })).toBeChecked();
    expect(screen.getByRole("switch", { name: "Page Screenshots" })).not.toBeChecked();
    fireEvent.click(screen.getByRole("switch", { name: "Alt Text" }));
    const stored = JSON.parse(storage.getItem("markitai.options")!);
    expect(stored.imageOverrides).toEqual({ alt: true, desc: null, screenshot: false });
    expect(stored).not.toHaveProperty("pure");
    expect(stored).not.toHaveProperty("strategy");
  });

  it("reselects a preset as a bundle and keeps the output profile independent", async () => {
    render(<App />);
    await act(async () => {});
    fireEvent.click(screen.getByRole("button", { name: "Options" }));
    await waitFor(() => expect(screen.getByRole("button", { name: dicts.en.presetRich })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: dicts.en.presetRich }));
    fireEvent.click(screen.getByRole("switch", { name: "OCR" }));
    fireEvent.click(screen.getByRole("button", { name: dicts.en.profileRag }));
    fireEvent.click(screen.getByRole("button", { name: dicts.en.presetMinimal }));
    for (const name of ["LLM Enhancement", "OCR", "Alt Text", "Description JSON", "Page Screenshots"]) {
      expect(screen.getByRole("switch", { name })).not.toBeChecked();
    }
    expect(screen.getByRole("button", { name: dicts.en.profileRag })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: dicts.en.presetRich }));
    fireEvent.click(screen.getByRole("switch", { name: "LLM Enhancement" }));
    expect(screen.getByRole("switch", { name: "Alt Text" })).not.toBeChecked();
    expect(screen.getByRole("switch", { name: "Alt Text" })).toBeDisabled();
    fireEvent.click(screen.getByRole("switch", { name: "LLM Enhancement" }));
    expect(screen.getByRole("switch", { name: "Alt Text" })).toBeChecked();
  });

  it("caps a pasted URL batch at the job limit and says so", async () => {
    render(<App />);
    const urls = Array.from(
      { length: MAX_JOB_ITEMS + 1 },
      (_, index) => `https://example.com/page-${index}`,
    );

    const input = await screen.findByRole("textbox");
    fireEvent.change(input, { target: { value: urls.join("\n") } });
    fireEvent.click(screen.getByRole("button", { name: "Convert" }));

    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1));
    expect(mocks.submit.mock.calls[0]![1]).toEqual(urls.slice(0, MAX_JOB_ITEMS));
    expect(
      screen.getByText(`${MAX_JOB_ITEMS} of ${MAX_JOB_ITEMS + 1} files added (job limit)`),
    ).toBeVisible();
  });

  it("truncates an oversized folder drop at the job limit and says so", async () => {
    render(<App />);
    await screen.findByRole("textbox");
    // Let the mocked capabilities land: the drop listener truncates against
    // the server-provided limits.max_job_items.
    await act(async () => {});

    const fileEntries = Array.from({ length: MAX_JOB_ITEMS + 1 }, (_, index) => ({
      name: `doc-${index}.txt`,
      isFile: true,
      isDirectory: false,
      file: (resolve: (file: File) => void) =>
        resolve(new File(["x"], `doc-${index}.txt`)),
    }));
    let served = false;
    const directoryEntry = {
      name: "dropped-folder",
      isFile: false,
      isDirectory: true,
      createReader: () => ({
        readEntries: (resolve: (batch: unknown[]) => void) => {
          const batch = served ? [] : fileEntries;
          served = true;
          resolve(batch);
        },
      }),
    };
    const drop = new Event("drop", { bubbles: true, cancelable: true });
    Object.defineProperty(drop, "dataTransfer", {
      value: {
        types: ["Files"],
        items: [{ webkitGetAsEntry: () => directoryEntry }],
        files: [],
      },
    });

    fireEvent(window, drop);

    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1));
    expect(mocks.submit.mock.calls[0]![0]).toHaveLength(MAX_JOB_ITEMS);
    expect(
      await screen.findByText(
        `${MAX_JOB_ITEMS} of ${MAX_JOB_ITEMS + 1} files added (job limit)`,
      ),
    ).toBeVisible();
  });
});
