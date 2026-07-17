import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MAX_JOB_ITEMS } from "./api/types";
import App from "./App";

const mocks = vi.hoisted(() => ({
  submit: vi.fn(),
  openJob: vi.fn(),
}));

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
      extras: { browser: false, svg: false, kreuzberg: false },
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
    items: [
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
    activeCount: 0,
    now: Date.now(),
    submit: mocks.submit,
    retry: vi.fn().mockResolvedValue(null),
    enhance: vi.fn().mockResolvedValue(null),
    enhanceArchived: vi.fn().mockResolvedValue(null),
    retryArchived: vi.fn().mockResolvedValue(null),
    deleteItem: vi.fn().mockResolvedValue(null),
    submitError: null,
    clear: vi.fn(),
    clearSettled: vi.fn(),
    terminalJobCount: 1,
    suppressedHistoryIds: new Set<string>(),
    historyRevision: 0,
  }),
}));

describe("App workspace", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/");
    mocks.submit.mockReset();
    mocks.submit.mockResolvedValue(true);
    mocks.openJob.mockReset();
    mocks.openJob.mockResolvedValue(archivedSnapshot());
  });

  it("starts at home on / and navigates the task list to /jobs", async () => {
    render(<App />);

    expect(
      await screen.findByRole("heading", {
        name: "Drop files. Paste URLs. Get Markdown.",
      }),
    ).toBeVisible();
    const llmSwitch = await screen.findByRole("switch", { name: "LLM enhancement" });
    expect(llmSwitch).toHaveAttribute("aria-checked", "false");
    expect(screen.queryByRole("button", { name: "minimal" })).not.toBeInTheDocument();
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
    expect(source?.querySelector(".url-entry .file-picker")).not.toBeNull();
    const currentRow = screen.getByRole("option", { name: /result\.md/ });
    expect(currentRow.querySelector(".c-finished")).toHaveTextContent("07-13 10:00");
    const enhance = screen.getByRole("button", {
      name: "Enhance result.md with LLM",
    });
    expect(enhance).toBeDisabled();
    fireEvent.click(screen.getByRole("switch", { name: "LLM enhancement" }));
    expect(enhance).toBeEnabled();
    expect(currentRow.querySelector(".c-status.archive-actions")).not.toBeNull();
    const archivedRow = screen.getByRole("option", { name: "Open archived.pdf" });
    expect(listbox.contains(archivedRow)).toBe(true);
    expect(archivedRow).not.toHaveClass("archived-row");
    expect(archivedRow.querySelector(".c-status.archive-actions")).not.toBeNull();
    // The zip action renders in both breakpoint slots — the composer options
    // row (desktop) and below the ledger (phones); CSS shows one at a time.
    // Clear stays in the job header.
    const zipButtons = screen.getAllByRole("button", { name: /download all/i });
    expect(zipButtons).toHaveLength(2);
    const [rowZip, listZip] = zipButtons;
    expect(rowZip).toBeEnabled();
    expect(rowZip!.closest(".composer .options")).not.toBeNull();
    expect(rowZip!.closest(".jobhead")).toBeNull();
    expect(listZip).toBeEnabled();
    expect(listZip!.closest(".list-zip")).not.toBeNull();
    expect(
      listbox.compareDocumentPosition(listZip!) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    // The phone footer mirrors the header's external links.
    const footer = document.querySelector("footer.app-footer");
    expect(footer?.querySelector('a[href="https://github.com/Ynewtime/markitai"]')).not.toBeNull();
    const clearButton = screen.getByRole("button", { name: "Clear all" });
    expect(clearButton.closest(".jobhead-r")).not.toBeNull();
    expect(clearButton.closest(".composer")).toBeNull();
    const historyButton = screen.getByRole("button", { name: "View conversions" });
    expect(historyButton).toBeVisible();
    expect(historyButton).toHaveAttribute("aria-current", "page");
    expect(historyButton).toHaveClass("on");
    historyButton.focus();
    fireEvent.click(historyButton);
    await waitFor(() => expect(currentRow).toHaveFocus());
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
    fireEvent.click(await screen.findByRole("switch", { name: "LLM enhancement" }));
    const wand = screen.getByRole("button", {
      name: "Enhance archived.pdf with LLM",
    });
    expect(wand).toBeEnabled();
    fireEvent.click(wand);

    expect(
      await screen.findByText("LLM enhancement failed: Could not load this job"),
    ).toBeVisible();
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
