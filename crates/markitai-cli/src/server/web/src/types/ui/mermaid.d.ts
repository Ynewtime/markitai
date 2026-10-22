// Type of the diagram renderer built from src/mermaid/main.ts and served at /ui/mermaid.js.
export type Drawn = { ok: true; svg: string } | { ok: false; code: string; line: number };
export interface DrawOptions {
  fontFamily: string;
  fontSize: number;
  maxWidth: number;
  idPrefix: string;
  links: boolean;
}
export function drawDiagram(source: string, options: DrawOptions): Drawn;
