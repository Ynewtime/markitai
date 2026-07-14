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
    names_preview: ["first.pdf"],
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
    names_preview: ["second.pdf"],
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
        onDelete={onDelete}
        announce={() => undefined}
      />,
    );

    expect(screen.getByRole("option", { name: "open first.pdf" })).toHaveClass(
      "lrow",
      "archived-row",
    );
    expect(document.querySelector(".archived-rows")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "permanently delete first.pdf" }));
    const confirm = screen.getByRole("button", {
      name: "confirm permanent deletion of first.pdf",
    });
    expect(confirm).toHaveTextContent("delete permanently?");
    await user.click(confirm);

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
        onDelete={onDelete}
        announce={() => undefined}
      />,
    );

    await waitFor(() => {
      expect(onDelete).toHaveBeenCalledWith("job-1");
      expect(screen.getByRole("option", { name: "open second.pdf" })).toHaveFocus();
    });
  });
});
