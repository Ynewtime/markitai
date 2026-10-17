// Addresses in the backend's own documentation, from `docs_url`: a page beside an `index.md` address (Rust's
// Markdown docs), or under `guide/` of a site root such as `https://markitai.dev` or `https://markitai.dev/`
// (Python's VitePress site). Any other address is kept as it is.
export const docsPage = (docsUrl: string, page: string): string => {
  if (/\/index\.md$/.test(docsUrl)) return docsUrl.replace(/index\.md$/, `${page}.md`);
  const root = /^(https:\/\/[^/?#]+)(\/[^?#]*)?$/.exec(docsUrl);
  if (!root) return docsUrl;
  const path = root[2] ?? "";
  if (path !== "" && !path.endsWith("/")) return docsUrl;
  return `${root[1]}${path || "/"}guide/${page}`;
};
