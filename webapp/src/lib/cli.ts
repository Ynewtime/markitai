import type { Preset } from "../api/types";

/** Safe-bare shell charset; anything else (?, &, spaces, quotes) is wrapped
 * in single quotes so pasted commands survive a real shell. */
const SHELL_SAFE_RE = /^[A-Za-z0-9_\-./:@%+=,~]+$/;

function shellQuote(s: string): string {
  if (SHELL_SAFE_RE.test(s)) return s;
  return `'${s.replaceAll("'", "'\\''")}'`;
}

/** Equivalent `markitai` invocation for the current composer state. Flags
 * mirror the real CLI option declarations in cli/main.py (`-o/--output`,
 * `--preset`, `--llm/--no-llm`, `--ocr/--no-ocr`). The web options payload
 * carries explicit values, so the equivalent flags are always emitted. Browser file drops
 * carry no local paths, so a <your-files> placeholder stands in whenever no
 * URLs are typed (the UI shows a replace-me hint next to it). */
export function buildCliCommand(
  urls: string[],
  preset: Preset,
  llm: boolean,
  ocr: boolean,
): string {
  const inputs = urls.length > 0 ? urls.map(shellQuote) : ["<your-files>"];
  return [
    "markitai",
    ...inputs,
    "-o",
    "out/",
    "--preset",
    preset,
    llm ? "--llm" : "--no-llm",
    ocr ? "--ocr" : "--no-ocr",
  ].join(" ");
}
