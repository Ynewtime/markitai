// Addresses in the backend's own documentation, from `docs_url` (an `index.md` address): a page beside it.
export const docsPage = (docsUrl: string, page: string): string => (/\/index\.md$/.test(docsUrl) ? docsUrl.replace(/index\.md$/, `${page}.md`) : docsUrl);
