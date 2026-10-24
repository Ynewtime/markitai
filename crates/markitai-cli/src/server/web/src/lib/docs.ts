// Addresses in the backend's own documentation, from `docs_url`: a page beside an `index.md` address (the
// Markdown docs Rust servers before 2.0.1 named), or under `guide/` of a site root such as `https://markitai.dev` or `https://markitai.dev/`
// (the VitePress site markitai.dev). Any other address is kept as it is.
import type { Locale } from "../i18n/locale.ts";

/** The docs in the page's language: markitai.dev keeps its Chinese pages under `/zh/` (it has no Japanese
 * ones); any other address is kept as it is. */
export const docsHome = (docsUrl: string, locale: Locale): string =>
  locale === "zh" && /^https:\/\/markitai\.dev\/?$/.test(docsUrl) ? "https://markitai.dev/zh/" : docsUrl;

export const docsPage = (docsUrl: string, page: string): string => {
  if (/\/index\.md$/.test(docsUrl)) return docsUrl.replace(/index\.md$/, `${page}.md`);
  const root = /^(https:\/\/[^/?#]+)(\/[^?#]*)?$/.exec(docsUrl);
  if (!root) return docsUrl;
  const path = root[2] ?? "";
  if (path !== "" && !path.endsWith("/")) return docsUrl;
  return `${root[1]}${path || "/"}guide/${page}`;
};
