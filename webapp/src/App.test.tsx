import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
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
        names_preview: ["archived.pdf"],
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
      options: { preset: "standard", llm: false },
      items: [
        {
          item_id: "item-2",
          name: "archived.pdf",
          kind: "file",
          status: "done",
          error: null,
          output: "archived.md",
          duration_ms: 100,
          cost_usd: null,
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
        costUsd: null,
        skipped: false,
        skipReason: null,
        sizeBytes: 8,
        startedAt: null,
        archived: false,
        retried: false,
      },
    ],
    jobs: {
      "job-1": {
        jobId: "job-1",
        status: "done",
        createdAt: "2026-07-13T10:00:00Z",
        options: { preset: "standard", llm: false },
        archived: false,
      },
    },
    jobCount: 1,
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
    submitError: null,
    clear: vi.fn(),
    clearSettled: vi.fn(),
    terminalJobCount: 1,
    openArchived: vi.fn().mockReturnValue("job-1/item-1"),
  }),
}));

describe("App workspace", () => {
  it("starts at home after a reload and keeps restored tasks behind an explicit navigation", async () => {
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Drop files. Get Markdown." })).toBeVisible();
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /item in session/ }));
    const listbox = await screen.findByRole("listbox");
    const composer = screen.getByRole("textbox");
    expect(
      composer.compareDocumentPosition(listbox) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    const archivedRow = screen.getByRole("option", { name: "open archived.pdf" });
    expect(listbox.contains(archivedRow)).toBe(true);
    expect(document.querySelector(".archived-rows")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "view conversions" })).toBeVisible();
  });

  it("opens an archived option in place without moving it into the current session", async () => {
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "view conversions" }));
    const listbox = await screen.findByRole("listbox");
    const archivedOption = screen.getByRole("option", { name: "open archived.pdf" });

    fireEvent.click(archivedOption);

    expect(await screen.findByRole("dialog", { name: "archived.pdf" })).toBeVisible();
    expect(listbox.contains(archivedOption)).toBe(true);
  });
});
