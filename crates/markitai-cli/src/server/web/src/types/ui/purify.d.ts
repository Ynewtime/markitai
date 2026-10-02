// Type of the vendored DOMPurify module served at /ui/purify.js (vendor/web/purify.js).
declare const purify: { sanitize(html: string, options: Record<string, unknown>): DocumentFragment };
export default purify;
