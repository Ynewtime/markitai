// JSON split into coloured pieces for the Edit config editor's highlight layer.
// Every character of the input is in exactly one piece, in order, so the layer
// lines up with the textarea above it whatever the text holds (even invalid JSON).

export type JsonToken = "key" | "string" | "number" | "literal" | "punct" | null;

const PATTERN = /("(?:[^"\\\n]|\\.)*"?)|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false|null)\b|([{}[\]:,])|(\s+)|([^"\s{}[\]:,\d-]+|[\d-])/gy;

/** The pieces of `text`; a string followed by `:` is a key. */
export function jsonTokens(text: string): [JsonToken, string][] {
  const pieces: [JsonToken, string][] = [];
  PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while (PATTERN.lastIndex < text.length && (match = PATTERN.exec(text)) !== null) {
    const [whole, string, number, literal, punct] = match;
    if (string !== undefined) {
      let next = PATTERN.lastIndex;
      while (next < text.length && /[ \t\r\n]/.test(text[next] as string)) next++;
      pieces.push([text[next] === ":" ? "key" : "string", whole]);
    } else if (number !== undefined) pieces.push(["number", whole]);
    else if (literal !== undefined) pieces.push(["literal", whole]);
    else if (punct !== undefined) pieces.push(["punct", whole]);
    else pieces.push([null, whole]);
  }
  return pieces;
}
