import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import type { SessionItem } from "../hooks/useJobs";
import { dicts } from "../i18n";
import { ItemRow } from "./ItemRow";

function item(status: SessionItem["status"]): SessionItem {
  return {
    key: "job-1/i1",
    jobId: "job-1",
    itemId: "i1",
    name: "doc.pdf",
    kind: "file",
    status,
    error: status === "error" ? "conversion failed" : null,
    output: status === "done" ? "doc.md" : null,
    durationMs: 120,
    finishedAt: "2026-07-15T10:00:00Z",
    costUsd: null,
    llmEnhanced: false,
    operation: "convert",
    skipped: false,
    skipReason: null,
    sizeBytes: 100,
    startedAt: null,
  };
}

function renderRow(
  value: SessionItem,
  overrides: Partial<ComponentProps<typeof ItemRow>> = {},
) {
  const props: ComponentProps<typeof ItemRow> = {
    t: dicts.en,
    item: value,
    index: 0,
    showCost: false,
    now: Date.now(),
    selected: false,
    tabbable: true,
    canDelete: true,
    onPreview: vi.fn(),
    onRowFocus: vi.fn(),
    onRetry: vi.fn().mockResolvedValue(null),
    onDelete: vi.fn().mockResolvedValue(null),
    ...overrides,
  };
  return { ...render(<ItemRow {...props} />), props };
}

describe("ItemRow terminal actions", () => {
  it("takes roving focus when its row is clicked", async () => {
    const user = userEvent.setup();
    const onRowFocus = vi.fn();
    renderRow(item("done"), { onRowFocus });

    const row = screen.getByRole("option");
    await user.click(row);

    expect(row).toHaveFocus();
    expect(onRowFocus).toHaveBeenCalledWith("job-1/i1");
  });

  it("shows a colored success mark and confirms row deletion", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn().mockResolvedValue(null);
    const { container } = renderRow(item("done"), { onDelete });

    expect(container.querySelector(".item-result.ok")).toHaveTextContent("✓");
    expect(container).not.toHaveTextContent("<done>");
    await user.click(screen.getByRole("button", { name: "Permanently delete doc.pdf" }));
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Delete doc.pdf?");
    await user.click(screen.getByRole("button", { name: "Delete permanently" }));
    expect(onDelete).toHaveBeenCalledWith(expect.objectContaining({ itemId: "i1" }));
  });

  it("shows a warning tooltip and allows an image-only skip to retry", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn().mockResolvedValue(null);
    const skipped = {
      ...item("done"),
      output: null,
      skipped: true,
      skipReason: "image_only",
    };
    const { container } = renderRow(skipped, { onRetry });

    const mark = container.querySelector(".item-result.skip.tooltip");
    expect(mark).toHaveAttribute("title", dicts.en.skipImageOnly);
    expect(mark).toHaveAttribute("data-tooltip", dicts.en.skipImageOnly);
    expect(mark?.querySelector("svg")).not.toBeNull();
    expect(container.querySelector(".c-skip")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Retry doc.pdf" }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it("labels base output and can enhance it in place when LLM is available", async () => {
    const user = userEvent.setup();
    const onEnhance = vi.fn().mockResolvedValue(null);
    renderRow(item("done"), {
      showCost: true,
      llmAvailable: true,
      onEnhance,
    });

    expect(screen.getByText("Base")).toHaveClass("llm-tag");
    const action = screen.getByRole("button", {
      name: "Enhance doc.pdf with LLM",
    });
    expect(action).toBeEnabled();
    await user.click(action);
    expect(onEnhance).toHaveBeenCalledWith(
      expect.objectContaining({ itemId: "i1" }),
    );
  });

  it("keeps re-enhancement available for an existing LLM result", async () => {
    const user = userEvent.setup();
    const onEnhance = vi.fn().mockResolvedValue(null);
    renderRow(
      {
        ...item("done"),
        output: "doc.llm.md",
        costUsd: 0.0123,
        llmEnhanced: true,
        operation: "enhance",
      },
      { showCost: true, llmAvailable: true, onEnhance },
    );

    expect(screen.getByText("LLM")).toHaveClass("llm-tag", "on");
    expect(screen.getByText("$0.0123")).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Enhance doc.pdf with LLM" }),
    );
    expect(onEnhance).toHaveBeenCalledOnce();
  });

  it("cannot trigger enhancement while the workspace LLM switch is off", async () => {
    const user = userEvent.setup();
    const onEnhance = vi.fn().mockResolvedValue(null);
    renderRow(item("done"), {
      llmAvailable: false,
      llmDisabledReason: "Turn on LLM enhancement above to use this action",
      onEnhance,
    });

    const action = screen.getByRole("button", {
      name: "Enhance doc.pdf with LLM",
    });
    expect(action).toBeDisabled();
    expect(action).toHaveAttribute(
      "title",
      "Turn on LLM enhancement above to use this action",
    );
    await user.click(action);
    expect(onEnhance).not.toHaveBeenCalled();
  });

  it("shows enhancement failures inline and on the failed-status tooltip", async () => {
    const user = userEvent.setup();
    renderRow(item("done"), {
      llmAvailable: true,
      onEnhance: vi.fn().mockResolvedValue("provider timed out"),
    });

    await user.click(
      screen.getByRole("button", { name: "Enhance doc.pdf with LLM" }),
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "LLM enhancement failed: provider timed out",
    );
    expect(
      screen.getByRole("button", { name: "Enhance doc.pdf with LLM" }),
    ).toHaveAttribute(
      "data-tooltip",
      "LLM enhancement failed: provider timed out",
    );

    const { container } = renderRow(item("error"));
    expect(container.querySelector(".item-result.error.tooltip")).toHaveAttribute(
      "data-tooltip",
      "conversion failed",
    );
  });

  it("keeps retry and delete aligned in the terminal action cell", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn().mockResolvedValue(null);
    const { container } = renderRow(item("error"), { onRetry });

    expect(container.querySelector(".item-result.error")).toHaveTextContent("×");
    const actions = container.querySelector(".c-status .item-actions");
    expect(actions).not.toBeNull();
    expect(actions?.querySelectorAll("button")).toHaveLength(2);
    await user.click(screen.getByRole("button", { name: "Retry doc.pdf" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
