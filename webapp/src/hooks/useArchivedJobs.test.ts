import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HistoryEntry, JobSnapshot } from "../api/types";

const api = vi.hoisted(() => ({
  deleteHistoryJob: vi.fn(),
  fetchHistory: vi.fn(),
  fetchJobSnapshot: vi.fn(),
}));

vi.mock("../api/client", () => ({
  deleteHistoryJob: api.deleteHistoryJob,
  fetchHistory: api.fetchHistory,
  fetchJobSnapshot: api.fetchJobSnapshot,
}));

import { useArchivedJobs } from "./useArchivedJobs";

function entry(jobId: string): HistoryEntry {
  return {
    job_id: jobId,
    created_at: "2026-07-16T10:00:00Z",
    finished_at: "2026-07-16T10:00:05Z",
    status: "done",
    total: 1,
    done: 1,
    failed: 0,
    skipped: 0,
    llm_enhanced: 0,
    cost_usd: null,
    names_preview: ["doc.pdf"],
    kinds_preview: ["file"],
    duration_ms: 10,
    size_bytes: 100,
  };
}

function snapshot(jobId: string): JobSnapshot {
  return {
    job_id: jobId,
    status: "done",
    done: 1,
    failed: 0,
    total: 1,
    created_at: "2026-07-16T10:00:00Z",
    finished_at: "2026-07-16T10:00:05Z",
    options: { preset: "minimal", llm: false, ocr: false },
    items: [],
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("useArchivedJobs", () => {
  beforeEach(() => {
    api.fetchHistory.mockReset().mockResolvedValue([]);
    api.fetchJobSnapshot.mockReset().mockResolvedValue(null);
    api.deleteHistoryJob.mockReset().mockResolvedValue(true);
  });

  it("discards a stale refresh response that resolves after a newer one", async () => {
    const first = deferred<HistoryEntry[]>();
    const second = deferred<HistoryEntry[]>();
    api.fetchHistory
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    const { result, unmount } = renderHook(() => useArchivedJobs({}));
    await act(async () => {
      void result.current.refresh();
    });
    await act(async () => {
      second.resolve([entry("fresh")]);
    });
    await act(async () => {
      first.resolve([entry("stale")]);
    });

    expect(result.current.entries?.map((item) => item.job_id)).toEqual(["fresh"]);
    expect(result.current.refreshing).toBe(false);
    expect(result.current.error).toBeNull();
    unmount();
  });

  it("removes the entry when openJob finds no snapshot and keeps it out on refresh", async () => {
    api.fetchHistory.mockResolvedValue([entry("a"), entry("b")]);
    const { result, unmount } = renderHook(() => useArchivedJobs({}));
    await act(async () => {});
    expect(result.current.entries?.map((item) => item.job_id)).toEqual(["a", "b"]);

    let opened: JobSnapshot | null | undefined;
    await act(async () => {
      opened = await result.current.openJob("a");
    });

    expect(opened).toBeNull();
    expect(result.current.entries?.map((item) => item.job_id)).toEqual(["b"]);

    // The server may still list the deleted-under-us job until it settles.
    await act(async () => {
      await result.current.refresh();
    });
    expect(result.current.entries?.map((item) => item.job_id)).toEqual(["b"]);
    unmount();
  });

  it("records a delete failure on the row and keeps the entry", async () => {
    api.fetchHistory.mockResolvedValue([entry("a")]);
    api.deleteHistoryJob.mockRejectedValue(new Error("disk gone"));
    const { result, unmount } = renderHook(() => useArchivedJobs({}));
    await act(async () => {});

    let deleted: boolean | undefined;
    await act(async () => {
      deleted = await result.current.deleteJob("a");
    });

    expect(deleted).toBe(false);
    expect(result.current.rowErrors).toEqual({ a: "disk gone" });
    expect(result.current.entries?.map((item) => item.job_id)).toEqual(["a"]);
    expect(result.current.actions).toEqual({});
    unmount();
  });

  it("blocks a second action while a row is busy", async () => {
    api.fetchHistory.mockResolvedValue([entry("a")]);
    const pending = deferred<JobSnapshot | null>();
    api.fetchJobSnapshot.mockReturnValue(pending.promise);
    const { result, unmount } = renderHook(() => useArchivedJobs({}));
    await act(async () => {});

    let openPromise: Promise<JobSnapshot | null> | undefined;
    act(() => {
      openPromise = result.current.openJob("a");
    });
    expect(result.current.actions).toEqual({ a: "open" });

    let deleted: boolean | undefined;
    await act(async () => {
      deleted = await result.current.deleteJob("a");
    });
    expect(deleted).toBe(false);
    expect(api.deleteHistoryJob).not.toHaveBeenCalled();

    let opened: JobSnapshot | null | undefined;
    await act(async () => {
      pending.resolve(snapshot("a"));
      opened = await openPromise!;
    });
    expect(opened).toEqual(snapshot("a"));
    expect(result.current.actions).toEqual({});
    expect(result.current.entries?.map((item) => item.job_id)).toEqual(["a"]);
    unmount();
  });
});
