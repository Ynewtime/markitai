import { render, screen, waitFor } from "@testing-library/react";
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

  it("renders the meta line as separate facts with a tagged timestamp", () => {
    const { container } = renderRow(item("done"), { showCost: true });

    const meta = container.querySelector(".rowmeta");
    const bits = Array.from(meta?.querySelectorAll(".metabit") ?? []).map(
      (bit) => bit.textContent,
    );
    expect(bits).toEqual(["100 B", "0.1s", "07-15 10:00", "Base"]);
    // the timestamp bit is tagged: it is what the tightest phones drop
    expect(meta?.querySelector(".metabit-time")).toHaveTextContent("07-15 10:00");
    // separators are CSS ::before — a joined string would strand a "/" on wrap
    expect(meta?.textContent).not.toContain("/");
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

  it("is not aria-disabled while it hosts an enabled Retry (skipped, job running)", () => {
    const skipped = {
      ...item("done"),
      output: null,
      skipped: true,
      skipReason: "image_only",
    };
    // canDelete false models a still-running job; the row is not previewable
    // and not failed, but its Retry is enabled — so it must not read disabled
    renderRow(skipped, { canDelete: false });

    const row = screen.getByRole("option");
    expect(row).not.toHaveAttribute("aria-disabled");
    expect(screen.getByRole("button", { name: "Retry doc.pdf" })).toBeEnabled();
  });

  it("labels base output and can enhance it in place when LLM is available", async () => {
    const user = userEvent.setup();
    const onEnhance = vi.fn().mockResolvedValue(null);
    const { container } = renderRow(item("done"), {
      showCost: true,
      llmAvailable: true,
      onEnhance,
    });

    // scoped to the cost cell: the meta line repeats "Base" in its own span
    expect(container.querySelector(".llm-cell .llm-tag")).toHaveTextContent("Base");
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

  it("toggles the full error detail on the failed row", async () => {
    const user = userEvent.setup();
    const { container } = renderRow(item("error"));

    const row = screen.getByRole("option");
    // role="option" does not support aria-expanded; the error text the row
    // describes (aria-describedby) is what conveys state to screen readers.
    expect(row).not.toHaveAttribute("aria-expanded");
    expect(container.querySelector(".err-full")).toBeNull();
    await user.click(row);
    expect(container.querySelector(".err-full")).not.toBeNull();
  });

  it("hands focus to the neighboring row after a delete", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn().mockResolvedValue(null);
    const second = { ...item("done"), key: "job-1/i2", itemId: "i2", name: "next.pdf" };
    const rowProps = {
      t: dicts.en,
      index: 0,
      showCost: false,
      now: Date.now(),
      selected: false,
      tabbable: false,
      canDelete: true,
      onPreview: vi.fn(),
      onRowFocus: vi.fn(),
      onRetry: vi.fn().mockResolvedValue(null),
      onDelete,
    };
    const list = (items: SessionItem[]) => (
      <div role="listbox">
        {items.map((value) => (
          <ItemRow key={value.key} {...rowProps} item={value} />
        ))}
      </div>
    );
    const { rerender } = render(list([item("done"), second]));

    await user.click(
      screen.getByRole("button", { name: "Permanently delete doc.pdf" }),
    );
    await user.click(screen.getByRole("button", { name: "Delete permanently" }));
    rerender(list([second]));

    await waitFor(() => {
      expect(onDelete).toHaveBeenCalledWith(expect.objectContaining({ itemId: "i1" }));
      expect(screen.getByRole("option", { name: /next\.pdf/ })).toHaveFocus();
    });
  });
});
