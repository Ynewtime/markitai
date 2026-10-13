// The YAML front matter a converted document starts with, read for its card:
// title, description and tags, plus the other plain fields. Only the shapes
// Markitai writes are read (scalars, quoted strings, inline and block lists);
// anything else stays in the Source tab.

export interface FrontMatter {
  title: string | null;
  description: string | null;
  tags: string[];
  /** Remaining scalar fields, in order. */
  fields: [string, string][];
}

function scalar(raw: string): string {
  const value = raw.trim();
  if (value.length >= 2 && ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'")))) {
    const inner = value.slice(1, -1);
    return value.startsWith('"') ? inner.replace(/\\(["\\nt])/g, (_m, c: string) => ({ n: "\n", t: "\t" })[c] ?? c) : inner.replace(/''/g, "'");
  }
  return value;
}

function inlineList(raw: string): string[] | null {
  const value = raw.trim();
  if (!value.startsWith("[") || !value.endsWith("]")) return null;
  const body = value.slice(1, -1).trim();
  if (!body) return [];
  return body.split(",").map(scalar).filter(Boolean);
}

/** Read the block between the leading `---` lines; null when there is none. */
export function parseFrontMatter(block: string | null): FrontMatter | null {
  if (block === null) return null;
  const lines = block.split("\n").slice(1, -1);
  const result: FrontMatter = { title: null, description: null, tags: [], fields: [] };
  const lists = new Map<string, string[]>();
  let listKey: string | null = null;
  let folded: { key: string; parts: string[] } | null = null;
  const assign = (key: string, value: string | string[]) => {
    const lower = key.toLowerCase();
    if (Array.isArray(value)) {
      if (lower === "tags" || lower === "keywords") result.tags.push(...value);
      else if (value.length) result.fields.push([key, value.join(", ")]);
      return;
    }
    if (lower === "title") result.title = value;
    else if (lower === "description" || lower === "summary") result.description = value;
    else if (lower === "tags" || lower === "keywords") result.tags.push(...value.split(",").map((tag) => tag.trim()).filter(Boolean));
    else if (value) result.fields.push([key, value]);
  };
  const finishFolded = () => {
    if (folded) assign(folded.key, folded.parts.join(" ").trim());
    folded = null;
  };
  for (const line of lines) {
    const item = /^\s+-\s+(.*)$/.exec(line) ?? /^-\s+(.*)$/.exec(line);
    if (item && listKey !== null) {
      lists.get(listKey)?.push(scalar(item[1] ?? ""));
      continue;
    }
    if (folded && /^\s+\S/.test(line)) {
      folded.parts.push(line.trim());
      continue;
    }
    finishFolded();
    if (listKey !== null) {
      assign(listKey, lists.get(listKey) ?? []);
      listKey = null;
    }
    const pair = /^([A-Za-z_][\w.-]*)\s*:\s*(.*)$/.exec(line);
    if (!pair) continue;
    const key = pair[1] ?? "";
    const rest = pair[2] ?? "";
    if (rest === "") {
      listKey = key;
      lists.set(key, []);
      continue;
    }
    if (rest === ">" || rest === "|" || rest === ">-" || rest === "|-") {
      folded = { key, parts: [] };
      continue;
    }
    const list = inlineList(rest);
    assign(key, list ?? scalar(rest));
  }
  finishFolded();
  if (listKey !== null) assign(listKey, lists.get(listKey) ?? []);
  return result;
}

const sameWords = (text: string) =>
  text
    .replace(/[*_`~]|<[^>]*>|\\(?=[^\s])/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();

/** The document's first `#` heading, when the body opens with one (ATX or setext). */
export function openingHeading(body: string): string | null {
  const lines = body.replace(/^\s*\n/, "").split("\n", 2);
  const first = lines[0] ?? "";
  const atx = /^ {0,3}#(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/.exec(first);
  if (atx) return atx[1] ?? "";
  if (first.trim() && /^ {0,3}=+[ \t]*$/.test(lines[1] ?? "")) return first.trim();
  return null;
}

/** What the card above a rendered document shows: no title when the document
 * opens with the same heading; null when nothing is left to show. */
export function cardFacts(front: FrontMatter | null, body: string): FrontMatter | null {
  if (!front) return null;
  const opening = openingHeading(body);
  const shown = front.title && opening !== null && sameWords(opening) === sameWords(front.title) ? { ...front, title: null } : front;
  return shown.title || shown.description || shown.tags.length > 0 ? shown : null;
}
