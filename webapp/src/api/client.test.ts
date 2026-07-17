import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  createJob,
  deleteHistoryJob,
  deleteLLMProvider,
  enhanceJobItem,
  fetchCapabilities,
  fetchJobSnapshot,
  retryJobItem,
} from "./client";
import type { JobOptions } from "./types";

type FetchMock = ReturnType<typeof vi.fn<typeof fetch>>;

function stubFetch(response: Response): FetchMock {
  const mock = vi.fn<typeof fetch>().mockResolvedValue(response);
  vi.stubGlobal("fetch", mock);
  return mock;
}

function jsonResponse(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function sentRequest(mock: FetchMock): { url: string; init: RequestInit } {
  const call = mock.mock.calls[0];
  if (!call) throw new Error("fetch was not called");
  const [input, init] = call;
  if (init === undefined) throw new Error("fetch was called without init");
  return { url: String(input), init };
}

async function expectApiError(promise: Promise<unknown>): Promise<ApiError> {
  const outcome = await promise.then(
    () => null,
    (error: unknown) => error,
  );
  expect(outcome).toBeInstanceOf(ApiError);
  return outcome as ApiError;
}

afterEach(() => vi.unstubAllGlobals());

describe("error responses", () => {
  it("joins FastAPI 422 validation messages with '; '", async () => {
    stubFetch(
      jsonResponse(
        { detail: [{ msg: "field required" }, { msg: "value too long" }] },
        422,
      ),
    );

    const error = await expectApiError(fetchCapabilities());
    expect(error.message).toBe("field required; value too long");
    expect(error.status).toBe(422);
    expect(error.code).toBeNull();
    expect(error.currentRevision).toBeNull();
  });

  it("uses a plain string detail as the message", async () => {
    stubFetch(jsonResponse({ detail: "Job not found" }, 400));

    const error = await expectApiError(fetchCapabilities());
    expect(error.message).toBe("Job not found");
    expect(error.status).toBe(400);
    expect(error.code).toBeNull();
  });

  it("exposes code and currentRevision from a stale_revision 409", async () => {
    // the server's current_revision is a sha256 hex string, not a number
    const revision = "a".repeat(64);
    stubFetch(
      jsonResponse(
        { detail: { code: "stale_revision", current_revision: revision } },
        409,
      ),
    );

    const error = await expectApiError(deleteLLMProvider("openai", "6"));
    expect(error.message).toBe("Settings were changed in another window");
    expect(error.status).toBe(409);
    expect(error.code).toBe("stale_revision");
    expect(error.currentRevision).toBe(revision);
  });

  it("humanizes other machine codes by replacing underscores", async () => {
    stubFetch(jsonResponse({ detail: { code: "provider_in_use" } }, 409));

    const error = await expectApiError(deleteLLMProvider("openai", "6"));
    expect(error.message).toBe("provider in use");
    expect(error.code).toBe("provider_in_use");
    expect(error.currentRevision).toBeNull();
  });

  it("falls back to status and statusText for non-JSON bodies", async () => {
    stubFetch(
      new Response("<html>Bad Gateway</html>", {
        status: 502,
        statusText: "Bad Gateway",
      }),
    );

    const error = await expectApiError(fetchCapabilities());
    expect(error.message).toBe("502 Bad Gateway");
    expect(error.status).toBe(502);
    expect(error.code).toBeNull();
  });
});

describe("404 semantics", () => {
  it("fetchJobSnapshot resolves null for a forgotten job", async () => {
    stubFetch(new Response(null, { status: 404 }));

    await expect(fetchJobSnapshot("gone")).resolves.toBeNull();
  });

  it("deleteHistoryJob resolves false for a missing job", async () => {
    stubFetch(new Response(null, { status: 404 }));

    await expect(deleteHistoryJob("gone")).resolves.toBe(false);
  });

  it("deleteHistoryJob throws on a 500", async () => {
    stubFetch(jsonResponse({ detail: "History pruning failed" }, 500));

    const error = await expectApiError(deleteHistoryJob("j1"));
    expect(error.message).toBe("History pruning failed");
    expect(error.status).toBe(500);
  });
});

describe("job-creation request bodies", () => {
  const options: JobOptions = { preset: "standard", llm: true, ocr: null };
  const created = { job_id: "job-1", items: [] };

  it("createJob posts multipart form data with files, urls, and options", async () => {
    const mock = stubFetch(jsonResponse(created, 200));
    const file = new File(["hello"], "notes.txt", { type: "text/plain" });

    await createJob([file], ["https://example.com"], options);

    const { url, init } = sentRequest(mock);
    expect(url).toBe("/api/jobs");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    const form = init.body as FormData;
    const files = form.getAll("files");
    expect(files).toHaveLength(1);
    expect(files[0]).toBeInstanceOf(File);
    expect((files[0] as File).name).toBe("notes.txt");
    expect(form.get("urls")).toBe(JSON.stringify(["https://example.com"]));
    expect(form.get("options")).toBe(JSON.stringify(options));
  });

  it("retryJobItem without options posts an empty request", async () => {
    const mock = stubFetch(jsonResponse(created, 200));

    await retryJobItem("job 1", "item#2");

    const { url, init } = sentRequest(mock);
    expect(url).toBe("/api/jobs/job%201/items/item%232/retry");
    expect(init).toEqual({ method: "POST" });
  });

  it("retryJobItem with options posts only the options", async () => {
    const mock = stubFetch(jsonResponse(created, 200));

    await retryJobItem("j1", "i1", options);

    const { url, init } = sentRequest(mock);
    expect(url).toBe("/api/jobs/j1/items/i1/retry");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(String(init.body))).toEqual({ options });
  });

  it("enhanceJobItem posts the enhance operation with options", async () => {
    const mock = stubFetch(jsonResponse(created, 200));

    await enhanceJobItem("j1", "i1", options);

    const { url, init } = sentRequest(mock);
    expect(url).toBe("/api/jobs/j1/items/i1/retry");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(String(init.body))).toEqual({ operation: "enhance", options });
  });
});
