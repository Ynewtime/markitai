import { afterEach, describe, expect, it } from "vitest";
import { authHeaders, getToken, initToken, withToken } from "./token";

afterEach(() => {
  sessionStorage.clear();
  window.history.replaceState(null, "", "/");
});

describe("initToken", () => {
  it("captures ?token= into sessionStorage and scrubs the URL bar", () => {
    window.history.replaceState(null, "", "/?token=mk_secret&lang=en");

    initToken();

    expect(sessionStorage.getItem("markitai.serve.token")).toBe("mk_secret");
    expect(window.location.search).toBe("?lang=en");
  });

  it("leaves the URL untouched without a token parameter", () => {
    window.history.replaceState(null, "", "/?lang=en");

    initToken();

    expect(window.location.search).toBe("?lang=en");
  });
});

describe("token transport", () => {
  it("is a no-op everywhere while no token is stored (loopback sessions)", () => {
    expect(getToken()).toBeNull();
    expect(withToken("/api/history/archive")).toBe("/api/history/archive");
    expect(authHeaders()).toEqual({});
  });

  it("appends ?token= (encoded) and builds the bearer header", () => {
    sessionStorage.setItem("markitai.serve.token", "mk_a+b");

    expect(withToken("/api/jobs/j1/events")).toBe(
      "/api/jobs/j1/events?token=mk_a%2Bb",
    );
    expect(withToken("/api/x?y=1")).toBe("/api/x?y=1&token=mk_a%2Bb");
    expect(authHeaders()).toEqual({ Authorization: "Bearer mk_a+b" });
  });
});
