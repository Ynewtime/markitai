// Type of the vendored DOMPurify module served at /ui/purify.js (vendor/web/purify.js).
declare const purify: {
  sanitize(html: string, options: Record<string, unknown>): DocumentFragment;
  addHook(entry: string, hook: (node: Element, data: { attrName: string; keepAttr: boolean }) => void): void;
  removeHooks(entry: string): void;
};
export default purify;
