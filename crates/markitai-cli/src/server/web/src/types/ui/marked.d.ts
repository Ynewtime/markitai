// Type of the vendored marked module served at /ui/marked.js (vendor/web/marked.js).
type Tokens = { type: string; raw: string }[] & { links?: unknown };
export const marked: {
  use(options: { extensions: unknown[] }): void;
  defaults: Record<string, unknown>;
  parse(markdown: string, options: { async: false; gfm: boolean }): string;
  lexer(markdown: string, options: { gfm: boolean }): Tokens;
  parser(tokens: Tokens, options: { async: false; gfm: boolean }): string;
};
