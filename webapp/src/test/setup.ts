import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
  // App persists options (e.g. the LLM toggle) to localStorage; without this,
  // that state leaks into later tests in the same file and, depending on when
  // the persistence effect flushes, flips assumptions (a toggle-on click
  // becomes toggle-off). Reset storage so every test starts from defaults.
  // Use window.* explicitly: the bare `localStorage` global is undefined in
  // this setup module under Node's jsdom, while window.localStorage is the
  // jsdom Storage the app actually reads and writes.
  window.localStorage?.clear();
  window.sessionStorage?.clear();
});

Object.defineProperty(window, "matchMedia", {
  configurable: true,
  writable: true,
  value: (query: string): MediaQueryList => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  }),
});
