import { beforeEach, describe, expect, it, vi } from "vitest";

// The module keeps an in-memory copy of the token, so each test gets a fresh import.
async function load() {
  vi.resetModules();
  return import("./token");
}

beforeEach(() => {
  sessionStorage.clear();
  window.history.replaceState(null, "", "/");
});

describe("initToken", () => {
  it("captures the fragment token and scrubs the hash", async () => {
    const { initToken } = await load();
    window.history.replaceState(null, "", "/#token=mk_secret");

    initToken();

    expect(sessionStorage.getItem("markitai.serve.token")).toBe("mk_secret");
    expect(window.location.hash).toBe("");
  });

  it("reads the fragment token from a multi-parameter fragment", async () => {
    const { initToken } = await load();
    window.history.replaceState(null, "", "/#token=mk_secret&view=ledger");

    initToken();

    expect(sessionStorage.getItem("markitai.serve.token")).toBe("mk_secret");
    expect(window.location.hash).toBe("");
  });

  it("reads a bare fragment as the token", async () => {
    const { initToken } = await load();
    window.history.replaceState(null, "", "/#mk_secret");

    initToken();

    expect(sessionStorage.getItem("markitai.serve.token")).toBe("mk_secret");
    expect(window.location.hash).toBe("");
  });

  it("decodes a percent-encoded fragment token", async () => {
    const { initToken } = await load();
    window.history.replaceState(null, "", "/#token=mk_a%2Bb");

    initToken();

    expect(sessionStorage.getItem("markitai.serve.token")).toBe("mk_a+b");
  });

  it("prefers the fragment over a query token", async () => {
    const { initToken } = await load();
    window.history.replaceState(null, "", "/?token=old#token=mk_new");

    initToken();

    expect(sessionStorage.getItem("markitai.serve.token")).toBe("mk_new");
  });

  it("captures ?token= into sessionStorage and scrubs the URL bar", async () => {
    const { initToken } = await load();
    window.history.replaceState(null, "", "/?token=mk_secret&lang=en");

    initToken();

    expect(sessionStorage.getItem("markitai.serve.token")).toBe("mk_secret");
    expect(window.location.search).toBe("?lang=en");
  });

  it("scrubs a query token and a fragment token at once", async () => {
    const { initToken } = await load();
    window.history.replaceState(null, "", "/?token=mk_secret&lang=en#token=mk_secret");

    initToken();

    expect(window.location.search).toBe("?lang=en");
    expect(window.location.hash).toBe("");
  });

  it("leaves the URL untouched without a token parameter", async () => {
    const { initToken } = await load();
    window.history.replaceState(null, "", "/?lang=en");

    initToken();

    expect(window.location.search).toBe("?lang=en");
  });

  it("leaves a non-token fragment untouched", async () => {
    const { initToken } = await load();
    window.history.replaceState(null, "", "/?lang=en#section-2");

    initToken();

    expect(window.location.search).toBe("?lang=en");
    expect(window.location.hash).toBe("#section-2");
  });

  it("keeps the captured token when storage accepts reads but rejects writes", async () => {
    const { getToken, initToken } = await load();
    window.history.replaceState(null, "", "/?token=mk_secret");
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });
    try {
      initToken();
      expect(getToken()).toBe("mk_secret");
    } finally {
      setItem.mockRestore();
    }
  });
});

describe("token transport", () => {
  it("is a no-op everywhere while no token is stored (loopback sessions)", async () => {
    const { authHeaders, getToken, withToken } = await load();
    expect(getToken()).toBeNull();
    expect(withToken("/api/history/archive")).toBe("/api/history/archive");
    expect(authHeaders()).toEqual({});
  });

  it("appends ?token= (encoded) and builds the bearer header", async () => {
    const { authHeaders, withToken } = await load();
    sessionStorage.setItem("markitai.serve.token", "mk_a+b");

    expect(withToken("/api/jobs/j1/events")).toBe(
      "/api/jobs/j1/events?token=mk_a%2Bb",
    );
    expect(withToken("/api/x?y=1")).toBe("/api/x?y=1&token=mk_a%2Bb");
    expect(authHeaders()).toEqual({ Authorization: "Bearer mk_a+b" });
  });
});
