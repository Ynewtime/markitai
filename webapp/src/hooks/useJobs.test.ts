import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type {
  ItemPayload,
  ItemStatus,
  JobSnapshot,
  JobStatus,
} from "../api/types";

const api = vi.hoisted(() => ({
  createJob: vi.fn(),
  deleteJobItem: vi.fn(),
  enhanceJobItem: vi.fn(),
  fetchJobSnapshot: vi.fn(),
  retryJobItem: vi.fn(),
}));

vi.mock("../api/client", () => ({
  createJob: api.createJob,
  deleteJobItem: api.deleteJobItem,
  enhanceJobItem: api.enhanceJobItem,
  fetchJobSnapshot: api.fetchJobSnapshot,
  jobEventsUrl: (jobId: string) => `/api/jobs/${jobId}/events`,
  retryJobItem: api.retryJobItem,
}));

const notify = vi.hoisted(() => ({
  notifyJobDone: vi.fn(),
  requestNotifyPermission: vi.fn(),
}));

vi.mock("../lib/notify", () => notify);

import {
  migrateLegacyRetrySeeds,
  newestSessionItemsFirst,
  serverTimestampMs,
  useJobs,
  type SessionItem,
  type SessionJob,
} from "./useJobs";

type FrameListener = (event: MessageEvent<string>) => void;

/** Controllable EventSource: tests drive frames and connection failures. */
class FakeEventSource {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;
  static instances: FakeEventSource[] = [];

  readonly url: string;
  readyState: number = FakeEventSource.CONNECTING;
  closed = false;
  private listeners = new Map<string, FrameListener[]>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: FrameListener): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }

  close(): void {
    this.closed = true;
    this.readyState = FakeEventSource.CLOSED;
  }

  emit(type: string, payload: unknown): void {
    this.readyState = FakeEventSource.OPEN;
    for (const listener of this.listeners.get(type) ?? []) {
      listener(new MessageEvent(type, { data: JSON.stringify(payload) }));
    }
  }

  fail(readyState: number): void {
    this.readyState = readyState;
    for (const listener of this.listeners.get("error") ?? []) {
      listener(new Event("error") as MessageEvent<string>);
    }
  }
}

function latestSource(): FakeEventSource {
  const source = FakeEventSource.instances.at(-1);
  if (source === undefined) throw new Error("no EventSource was opened");
  return source;
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

function itemPayload(
  itemId: string,
  status: ItemStatus,
  overrides: Partial<ItemPayload> = {},
): ItemPayload {
  return {
    item_id: itemId,
    name: "doc.pdf",
    kind: "file",
    status,
    error: status === "error" ? "boom" : null,
    output: status === "done" ? "doc.md" : null,
    duration_ms: status === "done" ? 10 : null,
    finished_at: null,
    cost_usd: null,
    llm_enhanced: false,
    operation: "convert",
    skipped: false,
    skip_reason: null,
    ...overrides,
  };
}

function jobSnapshot(
  jobId: string,
  status: JobStatus,
  items: ItemPayload[],
): JobSnapshot {
  return {
    job_id: jobId,
    status,
    done: items.filter((item) => item.status === "done").length,
    failed: items.filter((item) => item.status === "error").length,
    total: items.length,
    created_at: "2026-07-16T10:00:00Z",
    finished_at: status === "running" ? null : "2026-07-16T10:00:05Z",
    options: { preset: "minimal", llm: false, ocr: false },
    items,
  };
}

async function submitJob(result: { current: ReturnType<typeof useJobs> }) {
  await act(async () => {
    await result.current.submit([new File(["pdf"], "doc.pdf")], [], {
      preset: "minimal",
      llm: false,
      ocr: false,
    });
  });
}

beforeEach(() => {
  sessionStorage.clear();
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
  notify.notifyJobDone.mockReset();
  notify.requestNotifyPermission.mockReset();
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
  api.fetchJobSnapshot.mockReset().mockResolvedValue(null);
  api.deleteJobItem.mockReset().mockResolvedValue(undefined);
});

describe("useJobs retry identity", () => {
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

describe("useJobs event stream", () => {
  it("merges snapshot and item frames and notifies once at the terminal state", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    await submitJob(result);
    const source = latestSource();
    expect(source.url).toBe("/api/jobs/job-1/events");

    act(() => {
      source.emit(
        "snapshot",
        jobSnapshot("job-1", "running", [itemPayload("i1", "running")]),
      );
    });
    expect(result.current.items[0]).toMatchObject({ status: "running" });
    expect(result.current.jobs["job-1"]).toMatchObject({
      status: "running",
      createdAt: "2026-07-16T10:00:00Z",
    });

    act(() => {
      source.emit("item", itemPayload("i1", "done"));
    });
    expect(result.current.items[0]).toMatchObject({
      status: "done",
      output: "doc.md",
    });

    act(() => {
      source.emit("job", { status: "done", done: 1, failed: 0, total: 1 });
      // A reconnect replay can restate the terminal status; the latch must
      // keep it to a single notification.
      source.emit(
        "snapshot",
        jobSnapshot("job-1", "done", [itemPayload("i1", "done")]),
      );
    });

    expect(result.current.jobs["job-1"]).toMatchObject({ status: "done" });
    expect(source.closed).toBe(true);
    expect(notify.notifyJobDone).toHaveBeenCalledTimes(1);
    expect(notify.notifyJobDone).toHaveBeenCalledWith("1 done");
    unmount();
  });

  it("reconciles a permanently failed stream against a terminal snapshot", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    await submitJob(result);
    const source = latestSource();
    api.fetchJobSnapshot.mockResolvedValue(
      jobSnapshot("job-1", "done", [itemPayload("i1", "error")]),
    );

    await act(async () => {
      source.fail(FakeEventSource.CLOSED);
    });

    expect(api.fetchJobSnapshot).toHaveBeenCalledWith("job-1");
    expect(result.current.items[0]).toMatchObject({
      status: "error",
      error: "boom",
    });
    expect(result.current.jobs["job-1"]).toMatchObject({ status: "done" });
    expect(source.closed).toBe(true);
    expect(notify.notifyJobDone).toHaveBeenCalledTimes(1);
    unmount();
  });

  it("marks unfinished rows as connection-lost when no snapshot remains", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    await submitJob(result);
    const source = latestSource();
    api.fetchJobSnapshot.mockResolvedValue(null);

    await act(async () => {
      source.fail(FakeEventSource.CLOSED);
    });

    expect(result.current.items[0]).toMatchObject({
      status: "error",
      error: "Lost connection to the server",
    });
    expect(result.current.jobs["job-1"]).toMatchObject({ status: "done" });
    expect(source.closed).toBe(true);
    expect(notify.notifyJobDone).not.toHaveBeenCalled();
    unmount();
  });

  it("marks unfinished rows as connection-lost when the snapshot fetch fails", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    await submitJob(result);
    const source = latestSource();
    api.fetchJobSnapshot.mockRejectedValue(new Error("fetch failed"));

    await act(async () => {
      source.fail(FakeEventSource.CLOSED);
    });

    expect(result.current.items[0]).toMatchObject({
      status: "error",
      error: "Lost connection to the server",
    });
    expect(source.closed).toBe(true);
    unmount();
  });

  it("leaves a CONNECTING stream to the browser's own reconnect", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    await submitJob(result);
    const source = latestSource();

    await act(async () => {
      source.fail(FakeEventSource.CONNECTING);
    });

    expect(api.fetchJobSnapshot).not.toHaveBeenCalled();
    expect(result.current.items[0]).toMatchObject({ status: "queued" });
    expect(source.closed).toBe(false);
    unmount();
  });

  it("reattaches a fresh stream when the server still reports the job running", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    await submitJob(result);
    const source = latestSource();
    api.fetchJobSnapshot.mockResolvedValue(
      jobSnapshot("job-1", "running", [itemPayload("i1", "running")]),
    );

    await act(async () => {
      source.fail(FakeEventSource.CLOSED);
    });

    expect(result.current.items[0]).toMatchObject({ status: "running" });
    const reopened = latestSource();
    expect(reopened).not.toBe(source);
    expect(reopened.url).toBe("/api/jobs/job-1/events");
    expect(reopened.closed).toBe(false);
    unmount();
  });

  it("removes the job, stream, and seed when the last row is deleted", async () => {
    const { result, unmount } = renderHook(() => useJobs());
    await submitJob(result);
    const source = latestSource();
    expect(sessionStorage.getItem("markitai.session")).not.toBeNull();

    let error: string | null | undefined;
    await act(async () => {
      error = await result.current.deleteItem(result.current.items[0]!);
    });

    expect(error).toBeNull();
    expect(api.deleteJobItem).toHaveBeenCalledWith("job-1", "i1");
    expect(result.current.items).toHaveLength(0);
    expect(result.current.jobs).toEqual({});
    expect(source.closed).toBe(true);
    expect(sessionStorage.getItem("markitai.session")).toBeNull();
    unmount();
  });
});

describe("useJobs session restore", () => {
  it("re-attaches running jobs, merges terminal ones, and drops unknown ones", async () => {
    sessionStorage.setItem(
      "markitai.session",
      JSON.stringify([
        { jobId: "job-running", items: [seed("i1", "run.pdf")] },
        { jobId: "job-finished", items: [seed("i1", "done.pdf")] },
        { jobId: "job-gone", items: [seed("i1", "gone.pdf")] },
      ]),
    );
    api.fetchJobSnapshot.mockImplementation((jobId: string) => {
      if (jobId === "job-running") {
        return Promise.resolve(
          jobSnapshot("job-running", "running", [
            itemPayload("i1", "running", { name: "run.pdf" }),
          ]),
        );
      }
      if (jobId === "job-finished") {
        return Promise.resolve(
          jobSnapshot("job-finished", "done", [
            itemPayload("i1", "done", { name: "done.pdf", output: "done.md" }),
          ]),
        );
      }
      return Promise.resolve(null);
    });

    const { result, unmount } = renderHook(() => useJobs());
    await act(async () => {});

    expect(result.current.items).toHaveLength(2);
    expect(result.current.items.map((item) => item.key)).toEqual(
      expect.arrayContaining(["job-running/i1", "job-finished/i1"]),
    );
    expect(result.current.jobs["job-running"]).toMatchObject({ status: "running" });
    expect(result.current.jobs["job-finished"]).toMatchObject({ status: "done" });
    expect(result.current.jobs["job-gone"]).toBeUndefined();
    expect(FakeEventSource.instances.map((source) => source.url)).toEqual([
      "/api/jobs/job-running/events",
    ]);
    const stored = JSON.parse(
      sessionStorage.getItem("markitai.session") ?? "[]",
    ) as { jobId: string }[];
    expect(stored.map((job) => job.jobId)).toEqual(["job-running", "job-finished"]);
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
