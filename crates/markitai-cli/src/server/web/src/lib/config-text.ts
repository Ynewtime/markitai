// The Edit config page's check of the text as it is typed: JSON, an object at
// the top, and where the first mistake is ("Line 3, column 9"), in the reader's
// language. The server checks the configuration itself when it is saved.

/** Line and column (1-based) of a character offset. */
export function lineColumn(text: string, offset: number): { line: number; column: number } {
  const before = text.slice(0, Math.max(0, Math.min(offset, text.length)));
  const lines = before.split("\n");
  return { line: lines.length, column: (lines[lines.length - 1]?.length ?? 0) + 1 };
}

/** Where a JSON.parse error points: its own "line L column C", else "position N". */
export function errorPlace(message: string, text: string): { line: number; column: number } | null {
  const named = /line (\d+) column (\d+)/i.exec(message);
  if (named) return { line: Number(named[1]), column: Number(named[2]) };
  const position = /position (\d+)/i.exec(message);
  if (position) return lineColumn(text, Number(position[1]));
  if (/unexpected end/i.test(message)) return lineColumn(text, text.length);
  return null;
}

/** Where JSON text first goes wrong (a character offset), found by reading it; -1 when it is valid.
 * Used when the browser's own message gives no position. */
export function firstError(text: string): number {
  let at = 0;
  const space = () => {
    while (at < text.length && " \t\n\r".includes(text[at] as string)) at++;
  };
  const fail = (): never => {
    throw at;
  };
  const literal = (word: string) => {
    for (const char of word) if (text[at++] !== char) (at--, fail());
  };
  const string = () => {
    at++;
    while (at < text.length) {
      const char = text[at] as string;
      if (char === '"') return void at++;
      if (char < " ") fail();
      if (char === "\\") {
        at++;
        const escape = text[at] ?? "";
        if (escape === "u") {
          if (!/^[0-9a-fA-F]{4}$/.test(text.slice(at + 1, at + 5))) (at++, fail());
          at += 5;
        } else if ('"\\/bfnrt'.includes(escape) && escape) at++;
        else fail();
      } else at++;
    }
    fail();
  };
  const number = () => {
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(at));
    if (!match) fail();
    at += (match as RegExpExecArray)[0].length;
  };
  const value = (depth: number): void => {
    if (depth > 512) fail();
    space();
    const char = text[at];
    if (char === "{") {
      at++;
      space();
      if (text[at] === "}") return void at++;
      for (;;) {
        space();
        if (text[at] !== '"') fail();
        string();
        space();
        if (text[at] !== ":") fail();
        at++;
        value(depth + 1);
        space();
        if (text[at] === ",") at++;
        else if (text[at] === "}") return void at++;
        else fail();
      }
    }
    if (char === "[") {
      at++;
      space();
      if (text[at] === "]") return void at++;
      for (;;) {
        value(depth + 1);
        space();
        if (text[at] === ",") at++;
        else if (text[at] === "]") return void at++;
        else fail();
      }
    }
    if (char === '"') return string();
    if (char === "t") return literal("true");
    if (char === "f") return literal("false");
    if (char === "n") return literal("null");
    return number();
  };
  try {
    value(0);
    space();
    if (at < text.length) fail();
    return -1;
  } catch (offset) {
    return typeof offset === "number" ? offset : 0;
  }
}

/** null when the text is a JSON object; else what is wrong, in the reader's words. */
export function jsonProblem(text: string, t: { configNotJson: (line: number, column: number) => string; configNotJsonAnywhere: string; configNotObject: string }): string | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    const offset = firstError(text);
    const place = errorPlace(error instanceof Error ? error.message : String(error), text) ?? (offset >= 0 ? lineColumn(text, offset) : null);
    return place ? t.configNotJson(place.line, place.column) : t.configNotJsonAnywhere;
  }
  return value !== null && typeof value === "object" && !Array.isArray(value) ? null : t.configNotObject;
}
