import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({
  createJob: vi.fn(),
  retryJobItem: vi.fn(),
  enhanceJobItem: vi.fn(),
}));

vi.mock("../api/client", () => ({
  createJob: api.createJob,
  deleteJobItem: vi.fn().mockResolvedValue(undefined),
  enhanceJobItem: api.enhanceJobItem,
  fetchJobSnapshot: vi.fn().mockResolvedValue(null),
  jobEventsUrl: (jobId: string) => `/api/jobs/${jobId}/events`,
  retryJobItem: api.retryJobItem,
}));

vi.mock("../lib/notify", () => ({
  notifyJobDone: vi.fn(),
  requestNotifyPermission: vi.fn(),
}));

import {
  migrateLegacyRetrySeeds,
  newestSessionItemsFirst,
  serverTimestampMs,
  useJobs,
  type SessionItem,
  type SessionJob,
} from "./useJobs";

class FakeEventSource {
  addEventListener() {}
  close() {}
}

function sessionItem(jobId: string, itemId: string, name: string): SessionItem {
  return {
    key: `${jobId}/${itemId}`,
    jobId,
    itemId,
    name,
    kind: "file",
    status: "done",
    error: null,
    output: `${name}.md`,
    durationMs: 1,
    finishedAt: null,
    costUsd: null,
    llmEnhanced: false,
    operation: "convert",
    skipped: false,
    skipReason: null,
    sizeBytes: 1,
    startedAt: null,
  };
}

function seed(
  itemId: string,
  name: string,
  options: { size?: number | null; retried?: boolean } = {},
) {
  return {
    itemId,
    name,
    kind: "file" as const,
    sizeBytes: options.size ?? 12,
    ...(options.retried === undefined ? {} : { retried: options.retried }),
  };
}

describe("useJobs retry identity", () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.stubGlobal("EventSource", FakeEventSource);
    api.createJob.mockReset().mockResolvedValue({
      job_id: "job-1",
      items: [{ item_id: "i1", name: "doc.pdf", kind: "file" }],
    });
    // Deliberately return a different ID: even a stale backend response must
    // not be adopted as a second foreground row.
    api.retryJobItem.mockReset().mockResolvedValue({
      job_id: "legacy-retry-job",
      items: [{ item_id: "i1", name: "doc.pdf", kind: "file" }],
    });
    api.enhanceJobItem.mockReset().mockResolvedValue({
      job_id: "job-1",
      items: [{ item_id: "i1", name: "doc.pdf", kind: "file" }],
    });
  });

  it("queues explicit LLM enhancement with the original row identity", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    const value = sessionItem("job-1", "i1", "doc.pdf");
    const options = { preset: "minimal" as const, llm: true, ocr: false };

    let error: string | null | undefined;
    await act(async () => {
      error = await result.current.enhance(value, options);
    });

    expect(error).toBeNull();
    expect(api.enhanceJobItem).toHaveBeenCalledWith("job-1", "i1", options);
    unmount();
  });

  it("adopts an archived failure and requeues it with the same identity", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    await act(async () => {
      await result.current.retryArchived(
        {
          job_id: "archived-job",
          status: "done",
          done: 0,
          failed: 1,
          total: 1,
          created_at: "2026-07-15T10:00:00Z",
          finished_at: "2026-07-15T10:00:01Z",
          options: { preset: "minimal", llm: false, ocr: false },
          items: [
            {
              item_id: "i1",
              name: "broken.pdf",
              kind: "file",
              status: "error",
              error: "failed",
              output: null,
              duration_ms: 10,
              finished_at: "2026-07-15T10:00:01Z",
              cost_usd: null,
              llm_enhanced: false,
              operation: "convert",
              skipped: false,
              skip_reason: null,
            },
          ],
        },
        "i1",
        { preset: "minimal", llm: false, ocr: true },
      );
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0]).toMatchObject({
      key: "archived-job/i1",
      status: "queued",
      error: null,
    });
    expect(result.current.jobs["archived-job"]).toMatchObject({
      status: "running",
      options: { preset: "minimal", llm: false, ocr: true },
    });
    expect(api.retryJobItem).toHaveBeenCalledWith("archived-job", "i1", {
      preset: "minimal",
      llm: false,
      ocr: true,
    });
    unmount();
  });

  it("shows newer jobs first while preserving item order within a job", async () => {
    api.createJob
      .mockResolvedValueOnce({
        job_id: "job-old",
        items: [{ item_id: "old-1", name: "old.pdf", kind: "file" }],
      })
      .mockResolvedValueOnce({
        job_id: "job-new",
        items: [
          { item_id: "new-1", name: "first.pdf", kind: "file" },
          { item_id: "new-2", name: "second.pdf", kind: "file" },
        ],
      });
    const { result, unmount } = renderHook(() => useJobs());

    await act(async () => {
      await result.current.submit(
        [new File(["old"], "old.pdf")],
        [],
        { preset: "minimal", llm: false, ocr: false },
      );
      await result.current.submit(
        [
          new File(["first"], "first.pdf"),
          new File(["second"], "second.pdf"),
        ],
        [],
        { preset: "minimal", llm: false, ocr: false },
      );
    });

    expect(result.current.items.map((item) => item.name)).toEqual([
      "first.pdf",
      "second.pdf",
      "old.pdf",
    ]);
    unmount();
  });

  it("sorts Python microsecond timestamps consistently across browsers", () => {
    const options = { preset: "minimal", llm: false, ocr: false } as const;
    const jobs: Record<string, SessionJob> = {
      old: {
        jobId: "old",
        status: "done",
        createdAt: "2026-07-16T03:51:00.999999+08:00",
        options,
      },
      newest: {
        jobId: "newest",
        status: "done",
        createdAt: "2026-07-16T03:52:00.000001+08:00",
        options,
      },
    };
    const items = [
      sessionItem("old", "old-1", "old.pdf"),
      sessionItem("newest", "new-1", "first.pdf"),
      sessionItem("newest", "new-2", "second.pdf"),
    ];

    expect(serverTimestampMs(jobs.newest!.createdAt!)).toBeGreaterThan(
      serverTimestampMs(jobs.old!.createdAt!)!,
    );
    expect(newestSessionItemsFirst(items, jobs).map((item) => item.name)).toEqual([
      "first.pdf",
      "second.pdf",
      "old.pdf",
    ]);
  });

  it("requeues the original item without appending a row", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    await act(async () => {
      await result.current.submit(
        [new File(["pdf"], "doc.pdf")],
        [],
        { preset: "minimal", llm: false, ocr: false },
      );
    });
    const original = result.current.items[0];
    expect(original).toBeDefined();

    await act(async () => {
      await result.current.retry(original!);
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0]?.key).toBe("job-1/i1");
    expect(Object.keys(result.current.jobs)).toEqual(["job-1"]);
    expect(api.retryJobItem).toHaveBeenCalledWith("job-1", "i1");
    unmount();
  });
});

describe("legacy retry seed migration", () => {
  it("keeps only the replacement row and schedules the old item for cleanup", () => {
    const migrated = migrateLegacyRetrySeeds([
      { jobId: "old", items: [seed("i1", "doc.pdf", { retried: true })] },
      { jobId: "retry", items: [seed("i1", "doc.pdf")] },
    ]);

    expect(migrated.jobs).toEqual([
      { jobId: "retry", items: [seed("i1", "doc.pdf")] },
    ]);
    expect(migrated.removals).toEqual([{ jobId: "old", itemId: "i1" }]);
    expect([...migrated.suppressedJobIds]).toEqual(["old"]);
  });

  it("collapses a retry chain to its final attempt", () => {
    const migrated = migrateLegacyRetrySeeds([
      { jobId: "first", items: [seed("i1", "doc.pdf", { retried: true })] },
      { jobId: "second", items: [seed("i1", "doc.pdf", { retried: true })] },
      { jobId: "third", items: [seed("i1", "doc.pdf")] },
    ]);

    expect(migrated.jobs.map((job) => job.jobId)).toEqual(["third"]);
    expect(migrated.removals).toEqual([
      { jobId: "first", itemId: "i1" },
      { jobId: "second", itemId: "i1" },
    ]);
  });

  it("does not infer retries from filenames without the legacy marker", () => {
    const stored = [
      { jobId: "first", items: [seed("i1", "doc.pdf")] },
      { jobId: "second", items: [seed("i1", "doc.pdf")] },
    ];

    const migrated = migrateLegacyRetrySeeds(stored);
    expect(migrated.jobs).toEqual(stored);
    expect(migrated.removals).toEqual([]);
  });

  it("retains unrelated items from a source batch", () => {
    const migrated = migrateLegacyRetrySeeds([
      {
        jobId: "batch",
        items: [
          seed("i1", "doc.pdf", { retried: true }),
          seed("i2", "other.pdf"),
        ],
      },
      { jobId: "retry", items: [seed("i1", "doc.pdf")] },
    ]);

    expect(migrated.jobs[0]).toEqual({
      jobId: "batch",
      items: [seed("i2", "other.pdf")],
    });
    expect(migrated.suppressedJobIds.size).toBe(0);
  });
});
