import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { guideLinks, runtimeGuide } from "./guides.ts";

test("setup guides open markitai.dev in the page's language at the section for the cause", () => {
  assert.equal(runtimeGuide("zh", "claude-agent", "not_signed_in"), "https://markitai.dev/zh/guide/subscriptions#sign-in");
  assert.equal(runtimeGuide("en", "copilot", "runtime_missing"), "https://markitai.dev/guide/subscriptions#install");
  assert.equal(runtimeGuide("en", "chatgpt", "runtime_unsupported"), "https://markitai.dev/guide/subscription-chatgpt#install");
  // The site has no Japanese pages: English.
  assert.equal(runtimeGuide("ja", "chatgpt", "runtime_failed"), "https://markitai.dev/guide/subscription-chatgpt#troubleshooting");
  // website/tests checks each of these against the site's pages and headings.
  assert.deepEqual(
    guideLinks(),
    ["", "/zh"]
      .flatMap((locale) =>
        ["subscription-chatgpt", "subscriptions"].flatMap((page) =>
          ["install", "sign-in", "troubleshooting"].map((anchor) => `https://markitai.dev${locale}/guide/${page}#${anchor}`),
        ),
      )
      .sort(),
  );
});

test("a runtime notice links the markitai.dev guide, not the backend's docs", () => {
  const notice = readFileSync(new URL("../components/runtime-notice.tsx", import.meta.url), "utf8");
  assert.ok(!notice.includes("docsPage") && notice.includes("runtimeGuide(locale, provider, failure.reason)"));
});
