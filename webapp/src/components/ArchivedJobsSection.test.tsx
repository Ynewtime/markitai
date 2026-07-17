import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { HistoryEntry } from "../api/types";
import { dicts } from "../i18n";
import { ArchivedJobRows } from "./ArchivedJobsSection";

const entries: HistoryEntry[] = [
  {
    job_id: "job-1",
    created_at: "2026-07-13T10:00:00Z",
    finished_at: "2026-07-13T10:01:00Z",
    status: "done",
    total: 1,
    done: 1,
    failed: 0,
    skipped: 0,
    llm_enhanced: 0,
    cost_usd: 0,
    names_preview: ["first.pdf"],
    kinds_preview: ["file"],
    duration_ms: 60_000,
    size_bytes: 100,
  },
  {
    job_id: "job-2",
    created_at: "2026-07-13T09:00:00Z",
    finished_at: "2026-07-13T09:01:00Z",
    status: "done",
    total: 1,
    done: 1,
    failed: 0,
    skipped: 0,
    llm_enhanced: 0,
    cost_usd: 0,
    names_preview: ["second.pdf"],
    kinds_preview: ["file"],
    duration_ms: 30_000,
    size_bytes: 200,
  },
];

describe("ArchivedJobRows", () => {
  it("uses an explicit permanent-delete confirmation and hands focus to the next row", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn().mockResolvedValue(true);
    const { rerender } = render(
      <ArchivedJobRows
        t={dicts.en}
        entries={entries}
        error={null}
        actions={{}}
        rowErrors={{}}
        showCost={false}
        startIndex={1}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
        onOpen={vi.fn().mockResolvedValue(null)}
        onRetry={vi.fn().mockResolvedValue(null)}
        onDelete={onDelete}
        announce={() => undefined}
      />,
    );

    const first = screen.getByRole("option", { name: "Open first.pdf" });
    expect(first).toHaveClass("lrow", "actionable");
    expect(first).not.toHaveClass("archived-row");
    expect(first.querySelector(".c-duration")).toHaveTextContent("60.0s");
    expect(first.querySelector(".c-finished")).toHaveTextContent("07-13 10:01");
    const resultMark = first.querySelector(".c-status.archive-actions .item-result.ok");
    expect(resultMark).toHaveTextContent("✓");
    expect(resultMark).toHaveAttribute("title", "Done");
    // actions live in the shared .item-actions slot (mobile CSS fixes its
    // width so status marks align down the list) and the meta line is
    // separate facts with a tagged timestamp, mirroring session rows
    expect(
      first.querySelector(".archive-actions .item-actions .rowicon.danger"),
    ).not.toBeNull();
    const metaBits = Array.from(first.querySelectorAll(".rowmeta .metabit")).map(
      (bit) => bit.textContent,
    );
    expect(metaBits).toEqual(["60.0s", "07-13 10:01", "Base", "Storage 100 B"]);
    expect(
      first.querySelector(".rowmeta .metabit-time"),
    ).toHaveTextContent("07-13 10:01");

    await user.click(screen.getByRole("button", { name: "Permanently delete first.pdf" }));
    expect(screen.getByRole("alertdialog")).toHaveTextContent("Delete first.pdf?");
    await user.click(screen.getByRole("button", { name: "Delete permanently" }));

    rerender(
      <ArchivedJobRows
        t={dicts.en}
        entries={[entries[1]!]}
        error={null}
        actions={{}}
        rowErrors={{}}
        showCost={false}
        startIndex={1}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
        onOpen={vi.fn().mockResolvedValue(null)}
        onRetry={vi.fn().mockResolvedValue(null)}
        onDelete={onDelete}
        announce={() => undefined}
      />,
    );

    await waitFor(() => {
      expect(onDelete).toHaveBeenCalledWith("job-1");
      expect(screen.getByRole("option", { name: "Open second.pdf" })).toHaveFocus();
    });
  });

  it("enhances a persisted base result and keeps failures on the action tooltip", async () => {
    const user = userEvent.setup();
    const onEnhance = vi.fn().mockResolvedValue("provider timed out");
    render(
      <ArchivedJobRows
        t={dicts.en}
        entries={[entries[0]!]}
        error={null}
        actions={{}}
        rowErrors={{}}
        showCost
        startIndex={0}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
        onOpen={vi.fn().mockResolvedValue(null)}
        onRetry={vi.fn().mockResolvedValue(null)}
        onEnhance={onEnhance}
        onDelete={vi.fn().mockResolvedValue(true)}
        announce={() => undefined}
        llmAvailable
      />,
    );

    const action = screen.getByRole("button", {
      name: "Enhance first.pdf with LLM",
    });
    await user.click(action);
    expect(onEnhance).toHaveBeenCalledWith("job-1");
    expect(screen.getByText(/LLM enhancement failed: provider timed out/)).toBeVisible();
    expect(action).toHaveAttribute(
      "data-tooltip",
      "LLM enhancement failed: provider timed out",
    );
  });

  it("does not offer enhancement for an already-enhanced persisted result", () => {
    render(
      <ArchivedJobRows
        t={dicts.en}
        entries={[
          { ...entries[0]!, llm_enhanced: 1, cost_usd: 0.01 },
        ]}
        error={null}
        actions={{}}
        rowErrors={{}}
        showCost
        startIndex={0}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
        onOpen={vi.fn().mockResolvedValue(null)}
        onRetry={vi.fn().mockResolvedValue(null)}
        onEnhance={vi.fn().mockResolvedValue(null)}
        onDelete={vi.fn().mockResolvedValue(true)}
        announce={() => undefined}
        llmAvailable
      />,
    );

    expect(
      screen.queryByRole("button", { name: "Enhance first.pdf with LLM" }),
    ).not.toBeInTheDocument();
  });

  it("shows a warning icon and retry for a persisted skip", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn().mockResolvedValue(null);
    const { container } = render(
      <ArchivedJobRows
        t={dicts.en}
        entries={[
          {
            ...entries[0]!,
            skipped: 1,
          },
        ]}
        error={null}
        actions={{}}
        rowErrors={{}}
        showCost={false}
        startIndex={0}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
        onOpen={vi.fn().mockResolvedValue(null)}
        onRetry={onRetry}
        onDelete={vi.fn().mockResolvedValue(true)}
        announce={() => undefined}
      />,
    );

    expect(container.querySelector(".item-result.skip svg")).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "Retry first.pdf" }));
    expect(onRetry).toHaveBeenCalledWith("job-1");
  });

  it("restores retry for a persisted single-item failure", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn().mockResolvedValue(null);
    render(
      <ArchivedJobRows
        t={dicts.en}
        entries={[
          {
            ...entries[0]!,
            done: 0,
            failed: 1,
          },
        ]}
        error={null}
        actions={{}}
        rowErrors={{}}
        showCost={false}
        startIndex={0}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
        onOpen={vi.fn().mockResolvedValue(null)}
        onRetry={onRetry}
        onDelete={vi.fn().mockResolvedValue(true)}
        announce={() => undefined}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Retry first.pdf" }));
    expect(onRetry).toHaveBeenCalledWith("job-1");
  });
});
