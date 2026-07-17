import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchItemResult, fetchJobFileText } from "../api/client";
import type { ItemResult } from "../api/types";
import type { SessionItem } from "../hooks/useJobs";
import { dicts } from "../i18n";
import { MAX_DIFF_LINES } from "../lib/diff";
import { MarkdownPreview } from "./MarkdownPreview";

vi.mock("../api/client", () => ({
  encodeRelPath: (path: string) => path.split("/").map(encodeURIComponent).join("/"),
  fetchItemResult: vi.fn(),
  fetchJobFileText: vi.fn(),
  jobFileUrl: () => "/doc.llm.md",
}));

const item: SessionItem = {
  key: "job-1/item-1",
  jobId: "job-1",
  itemId: "item-1",
  name: "doc.md",
  kind: "file",
  status: "done",
  error: null,
  output: "doc.llm.md",
  durationMs: 120,
  finishedAt: "2026-07-13T10:00:01Z",
  costUsd: null,
  llmEnhanced: true,
  operation: "convert",
  skipped: false,
  skipReason: null,
  sizeBytes: 8,
  startedAt: null,
};

const llmResult: ItemResult = {
  name: "doc.llm.md",
  variant: "llm",
  markdown: "# Doc",
  artifacts: [
    { relpath: "doc.md", size: 10 },
    { relpath: "doc.llm.md", size: 12 },
  ],
};

const baseResult: ItemResult = {
  name: "doc.md",
  variant: "base",
  markdown: "# Doc",
  artifacts: [{ relpath: "doc.md", size: 10 }],
};

function renderPreview(announce = vi.fn()) {
  render(
    <MarkdownPreview t={dicts.en} item={item} createdAt={null} announce={announce} />,
  );
  return announce;
}

describe("MarkdownPreview diff tab", () => {
  beforeEach(() => {
    vi.mocked(fetchItemResult).mockReset();
    vi.mocked(fetchJobFileText).mockReset();
  });

  it("appears when both .md and .llm.md artifacts exist and shows +N -N", async () => {
    vi.mocked(fetchItemResult).mockResolvedValue(llmResult);
    vi.mocked(fetchJobFileText).mockImplementation((_jobId, rel) =>
      Promise.resolve(rel === "doc.llm.md" ? "line1\nline2\nline3" : "line1\nold2"),
    );
    renderPreview();

    fireEvent.click(await screen.findByRole("tab", { name: dicts.en.diffTab }));
    expect(await screen.findByText(/\+2 -1/)).toHaveTextContent(
      "doc.md → doc.llm.md · +2 -1",
    );
    expect(vi.mocked(fetchJobFileText)).toHaveBeenCalledWith("job-1", "doc.md");
    expect(vi.mocked(fetchJobFileText)).toHaveBeenCalledWith("job-1", "doc.llm.md");
  });

  it("stays hidden when only the base artifact exists", async () => {
    vi.mocked(fetchItemResult).mockResolvedValue(baseResult);
    renderPreview();

    await screen.findByRole("heading", { name: "Doc" });
    expect(screen.queryByRole("tab", { name: dicts.en.diffTab })).not.toBeInTheDocument();
  });

  it("shows the too-large note above MAX_DIFF_LINES", async () => {
    vi.mocked(fetchItemResult).mockResolvedValue(llmResult);
    vi.mocked(fetchJobFileText).mockResolvedValue(
      "x\n".repeat(MAX_DIFF_LINES / 2 + 1),
    );
    renderPreview();

    fireEvent.click(await screen.findByRole("tab", { name: dicts.en.diffTab }));
    expect(await screen.findByText(dicts.en.diffTooLarge)).toBeVisible();
  });

  it("surfaces a diff fetch failure as an error line", async () => {
    vi.mocked(fetchItemResult).mockResolvedValue(llmResult);
    vi.mocked(fetchJobFileText).mockRejectedValue(new Error("network down"));
    renderPreview();

    fireEvent.click(await screen.findByRole("tab", { name: dicts.en.diffTab }));
    expect(await screen.findByText("network down")).toHaveClass("errline");
  });
});

describe("MarkdownPreview tab roving", () => {
  beforeEach(() => {
    vi.mocked(fetchItemResult).mockReset();
    vi.mocked(fetchJobFileText).mockReset();
  });

  it("wraps ArrowLeft/ArrowRight across two tabs", async () => {
    vi.mocked(fetchItemResult).mockResolvedValue(baseResult);
    renderPreview();
    const rendered = await screen.findByRole("tab", { name: dicts.en.rendered });
    const source = screen.getByRole("tab", { name: dicts.en.source });

    fireEvent.keyDown(rendered, { key: "ArrowRight" });
    expect(source).toHaveAttribute("aria-selected", "true");
    expect(source).toHaveFocus();

    fireEvent.keyDown(source, { key: "ArrowRight" });
    expect(rendered).toHaveAttribute("aria-selected", "true");
    expect(rendered).toHaveFocus();

    fireEvent.keyDown(rendered, { key: "ArrowLeft" });
    expect(source).toHaveAttribute("aria-selected", "true");
    expect(source).toHaveFocus();
  });

  it("wraps ArrowLeft/ArrowRight across three tabs", async () => {
    vi.mocked(fetchItemResult).mockResolvedValue(llmResult);
    vi.mocked(fetchJobFileText).mockResolvedValue("# Doc");
    renderPreview();
    const diff = await screen.findByRole("tab", { name: dicts.en.diffTab });
    const rendered = screen.getByRole("tab", { name: dicts.en.rendered });
    const source = screen.getByRole("tab", { name: dicts.en.source });

    fireEvent.keyDown(rendered, { key: "ArrowLeft" });
    expect(diff).toHaveAttribute("aria-selected", "true");
    expect(diff).toHaveFocus();

    fireEvent.keyDown(diff, { key: "ArrowRight" });
    expect(rendered).toHaveAttribute("aria-selected", "true");
    expect(rendered).toHaveFocus();

    fireEvent.keyDown(rendered, { key: "ArrowRight" });
    expect(source).toHaveAttribute("aria-selected", "true");
    expect(source).toHaveFocus();

    fireEvent.keyDown(source, { key: "ArrowRight" });
    expect(diff).toHaveAttribute("aria-selected", "true");
    expect(diff).toHaveFocus();
  });
});

describe("MarkdownPreview copy fallback", () => {
  const execCommand = vi.fn();

  beforeEach(() => {
    vi.mocked(fetchItemResult).mockReset();
    vi.mocked(fetchJobFileText).mockReset();
    execCommand.mockReset();
    // jsdom has neither navigator.clipboard nor document.execCommand — the
    // absent clipboard is exactly the non-secure-origin case under test.
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: execCommand,
    });
  });

  afterEach(() => {
    delete (document as { execCommand?: unknown }).execCommand;
  });

  it("copies via execCommand when navigator.clipboard is unavailable", async () => {
    expect(navigator.clipboard).toBeUndefined();
    execCommand.mockReturnValue(true);
    vi.mocked(fetchItemResult).mockResolvedValue(baseResult);
    const announce = renderPreview();

    fireEvent.click(await screen.findByRole("tab", { name: dicts.en.source }));
    fireEvent.click(screen.getByRole("button", { name: dicts.en.copy }));

    await waitFor(() => expect(announce).toHaveBeenCalledWith(dicts.en.copied));
    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(screen.getByRole("button", { name: dicts.en.copied })).toBeVisible();
  });

  it("surfaces a failed copy instead of swallowing it", async () => {
    execCommand.mockReturnValue(false);
    vi.mocked(fetchItemResult).mockResolvedValue(baseResult);
    const announce = renderPreview();

    fireEvent.click(await screen.findByRole("tab", { name: dicts.en.source }));
    fireEvent.click(screen.getByRole("button", { name: dicts.en.copy }));

    await waitFor(() => expect(announce).toHaveBeenCalledWith(dicts.en.copyFailed));
    expect(screen.getByRole("button", { name: dicts.en.copyFailed })).toBeVisible();
  });
});
