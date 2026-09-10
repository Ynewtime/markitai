import { render, screen, within } from "@testing-library/react";
import { jobOptions } from "../lib/jobOptions";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import type { HistoryEntry } from "../api/types";
import type { SessionItem, SessionJob } from "../hooks/useJobs";
import { dicts } from "../i18n";
import { ItemList, mergeLedgerRows } from "./ItemList";

function item(itemId: string, skipped = false): SessionItem {
  return {
    key: `job-1/${itemId}`,
    jobId: "job-1",
    itemId,
    name: `${itemId}.png`,
    kind: "file",
    status: "done",
    error: null,
    output: skipped ? null : `${itemId}.md`,
    durationMs: skipped ? null : 10,
    finishedAt: "2026-07-16T03:52:00.000+08:00",
    costUsd: null,
    llmEnhanced: false,
    operation: "convert",
    skipped,
    skipReason: skipped ? "image_only" : null,
    sizeBytes: 10,
    startedAt: null,
  };
}

function Harness() {
  const items = [item("rendered"), item("skipped", true)];
  const [selected, setSelected] = useState<string | null>(items[0]!.key);
  return (
    <ItemList
      t={dicts.en}
      items={items}
      jobs={{
        "job-1": {
          jobId: "job-1",
          status: "done",
          createdAt: "2026-07-16T03:52:00.000+08:00",
          options: jobOptions({ preset: "minimal", llm: false, ocr: false }),
        },
      }}
      showCost={false}
      stats={{
        done: 1,
        skipped: 1,
        failed: 0,
        total: 2,
        costTotal: 0,
        hasCost: false,
        doneDurationMs: 10,
      }}
      settled
      selectedKey={selected}
      onSelect={setSelected}
      onPreview={vi.fn()}
      focusKey={null}
      onFocusKeyHandled={vi.fn()}
      onRetry={vi.fn().mockResolvedValue(null)}
      onDelete={vi.fn().mockResolvedValue(null)}
    />
  );
}

function history(jobId: string, createdAt: string): HistoryEntry {
  return {
    job_id: jobId,
    created_at: createdAt,
    finished_at: createdAt,
    status: "done",
    total: 1,
    done: 1,
    failed: 0,
    skipped: 0,
    llm_enhanced: 0,
    cost_usd: null,
    names_preview: [`${jobId}.pdf`],
    kinds_preview: ["file"],
    duration_ms: 10,
    size_bytes: 10,
    origin: "web",
  };
}

describe("merged task ordering", () => {
  it("moves enhancement to the top by latest activity without duplicating it", () => {
    const entries = [
      history("new", "2026-07-16T12:00:00.000+08:00"),
      history("target", "2026-07-16T11:00:00.000+08:00"),
      history("old", "2026-07-16T10:00:00.000+08:00"),
    ];
    expect(mergeLedgerRows([], {}, entries).map((row) => row.key)).toEqual([
      "archive:new",
      "archive:target",
      "archive:old",
    ]);

    const adopted = {
      ...item("target-item"),
      key: "target/target-item",
      jobId: "target",
      status: "queued" as const,
      finishedAt: null,
    };
    const jobs: Record<string, SessionJob> = {
      target: {
        jobId: "target",
        status: "running",
        createdAt: "2026-07-16T11:00:00.000+08:00",
        options: jobOptions({ preset: "minimal", llm: false, ocr: false }),
      },
    };
    expect(mergeLedgerRows([adopted], jobs, entries).map((row) => row.key)).toEqual([
      "target/target-item",
      "archive:new",
      "archive:old",
    ]);

    const completed = {
      ...adopted,
      status: "done" as const,
      finishedAt: "2026-07-16T13:00:00.000+08:00",
    };
    expect(
      mergeLedgerRows(
        [completed],
        { ...jobs, target: { ...jobs.target!, status: "done" } },
        entries,
      ).map((row) => row.key),
    ).toEqual(["target/target-item", "archive:new", "archive:old"]);
  });

  it("preserves input order inside a multi-item job", () => {
    const first = { ...item("first"), key: "batch/first", jobId: "batch" };
    const second = { ...item("second"), key: "batch/second", jobId: "batch" };
    const jobs: Record<string, SessionJob> = {
      batch: {
        jobId: "batch",
        status: "done",
        createdAt: "2026-07-16T11:00:00.000+08:00",
        options: jobOptions({ preset: "minimal", llm: false, ocr: false }),
      },
    };
    expect(
      mergeLedgerRows([first, second], jobs, []).map((row) => row.key),
    ).toEqual(["batch/first", "batch/second"]);
  });
});

describe("ItemList status filters", () => {
  it("narrows the ledger to the clicked status without renumbering it", async () => {
    const user = userEvent.setup();
    const items = [
      ...Array.from({ length: 11 }, (_, index) => item(`row-${index}`)),
      item("skipped-row", true),
    ];
    const stats = {
      done: 11,
      skipped: 1,
      failed: 0,
      total: 12,
      costTotal: 0,
      hasCost: false,
      doneDurationMs: 10,
    };
    render(
      <ItemList
        t={dicts.en}
        items={items}
        jobs={{}}
        showCost={false}
        stats={stats}
        settled
        selectedKey={null}
        onSelect={vi.fn()}
        onPreview={vi.fn()}
        focusKey={null}
        onFocusKeyHandled={vi.fn()}
        onRetry={vi.fn().mockResolvedValue(null)}
        onDelete={vi.fn().mockResolvedValue(null)}
      />,
    );

    const group = screen.getByRole("group", { name: dicts.en.filterStatusAria });
    const chip = (label: string) => within(group).getByRole("button", { name: label });

    await user.click(chip(dicts.en.filterSkipped));
    expect(chip(dicts.en.filterSkipped)).toHaveAttribute("aria-pressed", "true");
    expect(screen.getAllByRole("option")).toHaveLength(1);
    // The row number stays anchored to the unfiltered ledger.
    expect(screen.getByRole("option").querySelector(".c-num")).toHaveTextContent("12");

    await user.click(chip(dicts.en.filterDone));
    expect(screen.getAllByRole("option")).toHaveLength(11);
    await user.click(chip(dicts.en.filterFailed));
    expect(screen.queryAllByRole("option")).toHaveLength(0);
    // An empty result explains itself; the no-conversions copy would be wrong.
    expect(screen.getByText(dicts.en.filterNoMatch)).toBeVisible();

    await user.click(chip(dicts.en.filterAll));
    expect(screen.getAllByRole("option")).toHaveLength(12);
  });

  it("labels the filter chips in the active locale", () => {
    const items = Array.from({ length: 12 }, (_, index) => item(`row-${index}`));
    render(
      <ItemList
        t={dicts.zh}
        items={items}
        jobs={{}}
        showCost={false}
        stats={{
          done: items.length,
          skipped: 0,
          failed: 0,
          total: items.length,
          costTotal: 0,
          hasCost: false,
          doneDurationMs: 10,
        }}
        settled
        selectedKey={null}
        onSelect={vi.fn()}
        onPreview={vi.fn()}
        focusKey={null}
        onFocusKeyHandled={vi.fn()}
        onRetry={vi.fn().mockResolvedValue(null)}
        onDelete={vi.fn().mockResolvedValue(null)}
      />,
    );

    const group = screen.getByRole("group", { name: dicts.zh.filterStatusAria });
    for (const label of [dicts.zh.filterAll, dicts.zh.filterDone, dicts.zh.filterFailed, dicts.zh.filterSkipped]) {
      expect(within(group).getByRole("button", { name: label })).toBeInTheDocument();
    }
  });
});

describe("ItemList roving selection", () => {
  it("lets a skipped row take focus and selection on click", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const skipped = screen.getByRole("option", { name: /skipped\.png/i });
    await user.click(skipped);

    expect(skipped).toHaveFocus();
    expect(skipped).toHaveAttribute("aria-selected", "true");
  });
});

describe("ItemList empty workspace", () => {
  it("shows an explicit empty state instead of a zeroed totals row", () => {
    const { container } = render(
      <ItemList
        t={dicts.en}
        items={[]}
        jobs={{}}
        showCost={false}
        stats={{
          done: 0,
          skipped: 0,
          failed: 0,
          total: 0,
          costTotal: 0,
          hasCost: false,
          doneDurationMs: 0,
        }}
        settled
        selectedKey={null}
        onSelect={vi.fn()}
        onPreview={vi.fn()}
        focusKey={null}
        onFocusKeyHandled={vi.fn()}
        onRetry={vi.fn().mockResolvedValue(null)}
        onDelete={vi.fn().mockResolvedValue(null)}
      />,
    );

    expect(screen.getByText(dicts.en.emptyWorkspace)).toBeVisible();
    expect(container.querySelector(".lrow.totals")).toBeNull();
  });

  it("suppresses the empty state while archived history is still loading", () => {
    render(
      <ItemList
        t={dicts.en}
        items={[]}
        jobs={{}}
        archive={{
          // null entries = fetchHistory still in flight; the empty message
          // must wait so it never flashes over about-to-arrive archived rows.
          entries: null,
          error: null,
          rowProps: {
            actions: {},
            rowErrors: {},
            onRefresh: vi.fn().mockResolvedValue(undefined),
            onOpen: vi.fn().mockResolvedValue(null),
            onRetry: vi.fn().mockResolvedValue(null),
            onDelete: vi.fn().mockResolvedValue(true),
            announce: vi.fn(),
          },
        }}
        showCost={false}
        stats={{
          done: 0,
          skipped: 0,
          failed: 0,
          total: 0,
          costTotal: 0,
          hasCost: false,
          doneDurationMs: 0,
        }}
        settled
        selectedKey={null}
        onSelect={vi.fn()}
        onPreview={vi.fn()}
        focusKey={null}
        onFocusKeyHandled={vi.fn()}
        onRetry={vi.fn().mockResolvedValue(null)}
        onDelete={vi.fn().mockResolvedValue(null)}
      />,
    );

    expect(screen.queryByText(dicts.en.emptyWorkspace)).not.toBeInTheDocument();
  });
});
