// The markitai.dev guide sections the provider pages link to ("Setup guide"):
// the live site in the page's language (Chinese under /zh/, English otherwise:
// the site has no Japanese pages), whatever backend serves the workbench. The
// sections carry explicit anchors on both locales' pages (`## 登录 {#sign-in}`),
// and website/tests checks every address `guideLinks` lists against the site.
import type { Locale } from "../i18n/locale.ts";
import type { RuntimeReason } from "./model-test.ts";
import { RUNTIME_REASONS } from "./model-test.ts";

export const SITE = "https://markitai.dev";

/** A guide page on markitai.dev, with a section anchor. */
export const sitePage = (locale: Locale, page: string, anchor = ""): string =>
  `${SITE}${locale === "zh" ? "/zh" : ""}/guide/${page}${anchor ? `#${anchor}` : ""}`;

/** Each subscription runtime's page. */
const RUNTIME_PAGES: Record<string, string> = {
  "claude-agent": "subscriptions",
  copilot: "subscriptions",
  chatgpt: "subscription-chatgpt",
};

/** The section of that page for each reason a runtime lists no models. */
const RUNTIME_SECTIONS: Record<RuntimeReason, string> = {
  runtime_missing: "install",
  runtime_unsupported: "install",
  not_signed_in: "sign-in",
  runtime_failed: "troubleshooting",
};

/** The setup guide for a subscription runtime that lists no models, at the section for its reason. */
export const runtimeGuide = (locale: Locale, provider: string, reason: RuntimeReason): string =>
  sitePage(locale, RUNTIME_PAGES[provider] ?? "subscriptions", RUNTIME_SECTIONS[reason]);

/** Every guide address the provider pages can show, for the site test. */
export const guideLinks = (): string[] => {
  const links = new Set<string>();
  for (const locale of ["en", "zh", "ja"] as const)
    for (const provider of Object.keys(RUNTIME_PAGES)) for (const reason of RUNTIME_REASONS) links.add(runtimeGuide(locale, provider, reason));
  return [...links].sort();
};
