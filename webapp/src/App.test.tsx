import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

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
    openJob: vi.fn().mockResolvedValue({
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
    }),
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
    submit: vi.fn().mockResolvedValue(true),
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
    expect(currentRow.querySelector(".c-status.archive-actions")).not.toBeNull();
    const archivedRow = screen.getByRole("option", { name: "Open archived.pdf" });
    expect(listbox.contains(archivedRow)).toBe(true);
    expect(archivedRow).not.toHaveClass("archived-row");
    expect(archivedRow.querySelector(".c-status.archive-actions")).not.toBeNull();
    expect(screen.getByRole("button", { name: /download all/i })).toBeEnabled();
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
});
