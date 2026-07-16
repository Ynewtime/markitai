import { render, screen } from "@testing-library/react";
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
          options: { preset: "minimal", llm: false, ocr: false },
        },
      }}
      showCost={false}
      now={Date.now()}
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
      canDelete={() => true}
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
        options: { preset: "minimal", llm: false, ocr: false },
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
        options: { preset: "minimal", llm: false, ocr: false },
      },
    };
    expect(
      mergeLedgerRows([first, second], jobs, []).map((row) => row.key),
    ).toEqual(["batch/first", "batch/second"]);
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
