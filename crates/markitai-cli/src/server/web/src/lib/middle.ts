// A long name cut in the middle to fit a width, with the "…" as part of the
// text (so it sits right against what follows). A path or web address gives
// up whole folders first ("slow/annual-reports/…/s5.html"), keeping its host
// and its last segment; a query goes before the path. Only when that is not
// enough is the last segment cut by characters, keeping its extension and the
// four characters before it ("年度报告…披露全文.pdf"), and the host last of all.

export type NameKind = "file" | "url";

export interface ShortName {
  /** A web address's scheme and host (shown apart, dimmed in lists); "" for files. */
  host: string;
  body: string;
}

const ELLIPSIS = "…";

/** Where a name's always-visible end starts: its extensions (".pdf.llm.md") and the four characters before them. */
export function tailLength(name: string): number {
  const chars = Array.from(name);
  const extensions = /(?:\.[A-Za-z0-9]{1,8}){1,3}$/.exec(name);
  const extension = extensions ? Array.from(extensions[0]).length : 0;
  return Math.min(chars.length, extension + 4);
}

/** A name as host, path and query. A web address keeps its scheme when it has one. */
export function nameShape(name: string, kind: NameKind): { host: string; path: string; query: string } {
  if (kind === "url") {
    const match = /^((?:[a-z][a-z\d+.-]*:\/\/)?[^/?#]+)([^?#]*)(.*)$/is.exec(name);
    if (match) return { host: match[1] ?? "", path: match[2] ?? "", query: match[3] ?? "" };
  }
  return { host: "", path: name, query: "" };
}

/** `head…tail` with as much of the head as fits; the tail is never cut. */
function cutChars(text: string, keep: number, fits: (text: string) => boolean, around: (cut: string) => string): string {
  const chars = Array.from(text);
  if (keep >= chars.length) return text;
  const tail = chars.slice(chars.length - keep).join("");
  let low = 0;
  let high = chars.length - keep;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (fits(around(chars.slice(0, middle).join("") + ELLIPSIS + tail))) low = middle;
    else high = middle - 1;
  }
  // A head that ends in a separator or a space reads as cut there.
  const head = chars.slice(0, low).join("").replace(/[\s_-]+$/u, "");
  return head + ELLIPSIS + tail;
}

/** The longest form of `name` that `fits`, cut in the middle as described above. */
export function shortenName(name: string, kind: NameKind, fits: (text: string) => boolean): ShortName {
  return shortenShape(nameShape(name, kind), fits);
}

/** The same for a name already split; a host left empty (hidden) takes no room. */
export function shortenShape({ host, path, query }: { host: string; path: string; query: string }, fits: (text: string) => boolean): ShortName {
  const whole = (body: string) => host + body;
  if (fits(host + path + query)) return { host, body: path + query };
  if (query && fits(whole(`${path}?${ELLIPSIS}`))) return { host, body: `${path}?${ELLIPSIS}` };
  const parts = path.split("/");
  const last = parts.pop() ?? "";
  // Folders go from the end of the path towards its start; the first ones stay longest.
  for (let keep = parts.length - 1; keep >= 0; keep--) {
    if (keep === 0 && parts[0] === "") continue;
    const body = `${parts.slice(0, keep).join("/")}${keep > 0 ? "/" : ""}${ELLIPSIS}/${last}`;
    if (fits(whole(body))) return { host, body };
  }
  // Then the last segment itself, behind the shortest folder form.
  // (The folders' "…/" goes too when it leaves no room for the segment's start.)
  const lead = parts.length === 0 ? "" : parts[0] === "" && parts.length === 1 ? "/" : parts[0] === "" ? `/${ELLIPSIS}/` : `${ELLIPSIS}/`;
  const keep = tailLength(last);
  const minimal = Array.from(last).length > keep ? ELLIPSIS + Array.from(last).slice(-keep).join("") : last;
  const leads = host || !lead ? [lead] : [lead, ""];
  for (const before of leads) {
    const body = before + cutChars(last, keep, (text) => fits(whole(text)), (cut) => before + cut);
    // "…/…inal.docx" says nothing more than "…inal.docx": without a host, try that next.
    const bare = leads.length > 1 && before !== "" && body === before + minimal;
    if (fits(whole(body)) && !bare) return { host, body };
  }
  if (!host) return { host, body: minimal };
  // Last of all the host gives up its middle too (keeping its start and its port or end).
  const end = lead + minimal;
  const shortHost = cutChars(host, Math.min(Array.from(host).length, 6), (text) => fits(text + end), (cut) => cut);
  return { host: shortHost, body: end };
}
