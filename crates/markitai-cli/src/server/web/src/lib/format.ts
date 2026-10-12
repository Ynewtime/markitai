// Display formatting. Every output reads well in the monospaced, tabular-figure columns.

export function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Duration split once, so the printed and the spoken forms cannot disagree:
 * under a minute tenths of a second survive; above it, whole minutes and seconds. */
export function durParts(ms: number): { minutes: number; seconds: number } {
  const tenths = Math.round(Math.max(0, ms) / 100) / 10;
  if (tenths < 60) return { minutes: 0, seconds: tenths };
  const whole = Math.round(tenths);
  return { minutes: Math.floor(whole / 60), seconds: whole % 60 };
}

/** "240 ms", "4.2 s", then "1:23", then "1:02:03". */
export function fmtDur(ms: number): string {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`;
  const { minutes, seconds } = durParts(ms);
  if (minutes === 0) return `${seconds.toFixed(1)} s`;
  const s = String(seconds).padStart(2, "0");
  if (minutes < 60) return `${minutes}:${s}`;
  return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${s}`;
}

/** "$0.0123"; trailing zeros dropped, zero reads "$0". */
export function fmtCost(usd: number): string {
  const text = usd.toFixed(4).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  return `$${text}`;
}

/** Milliseconds of an RFC 3339 timestamp, parsed by hand: some engines reject
 * long fractional seconds. null when it is not one. */
export function timestampMs(value: string | null | undefined): number | null {
  if (!value) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|([+-])(\d{2}):(\d{2}))$/.exec(value);
  if (match === null) return null;
  const [, y, mo, d, h, mi, s, fraction = "", zone, sign, zh, zm] = match;
  const utc = Date.UTC(Number(y), Number(mo) - 1, Number(d), Number(h), Number(mi), Number(s), Number((fraction + "000").slice(0, 3)));
  if (!Number.isFinite(utc)) return null;
  if (zone === "Z") return utc;
  const offset = (Number(zh) * 60 + Number(zm)) * 60_000;
  return sign === "+" ? utc - offset : utc + offset;
}

const pad = (n: number) => String(n).padStart(2, "0");

/** "2026-10-02" in the reader's zone. */
export function fmtDate(value: string): string {
  const ms = timestampMs(value);
  if (ms === null) return value.slice(0, 10);
  const date = new Date(ms);
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** "10-02 14:30" in the reader's zone; "-" when absent. */
export function fmtDateTime(value: string | null): string {
  if (value === null || value.length < 16) return "-";
  const ms = timestampMs(value);
  const date = ms === null ? new Date(value) : new Date(ms);
  if (Number.isNaN(date.getTime())) return `${value.slice(5, 10)} ${value.slice(11, 16)}`;
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

// Kana, CJK Extension A, Unified Ideographs, compatibility ideographs and the
// supplementary planes; neighbouring scripts still count per word.
const CJK = /[぀-ヿ㐀-䶿一-鿿豈-﫿\u{20000}-\u{3ffff}]/gu;

/** Latin words plus CJK characters, so Chinese documents count sensibly. */
export function countWords(text: string): number {
  const cjk = text.match(CJK)?.length ?? 0;
  const words = text.replace(CJK, " ").match(/\S+/g)?.length ?? 0;
  return cjk + words;
}

export function utf8Bytes(text: string): number {
  return new TextEncoder().encode(text).length;
}

/** Long names keep their tail (the extension) visible while the head ellipsizes. */
export function splitName(name: string, tail = 12): [string, string] | null {
  return name.length <= tail + 4 ? null : [name.slice(0, -tail), name.slice(-tail)];
}

/** Display name of an item: a URL without its scheme. */
export const displayName = (name: string): string => name.replace(/^https?:\/\//, "");

export function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

const LOCALE_TAGS: Record<string, string> = { en: "en", zh: "zh-CN", ja: "ja" };
export const localeTag = (locale: string): string => LOCALE_TAGS[locale] ?? "en";

/** "22:51" today, "Oct 9, 22:51" (in the reader's language) on another day; "" when absent. */
export function fmtClock(value: string | null, locale: string, now: Date = new Date()): string {
  const ms = timestampMs(value);
  if (ms === null) return "";
  const date = new Date(ms);
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
  if (date.toDateString() === now.toDateString()) return time;
  const day = new Intl.DateTimeFormat(localeTag(locale), { month: "short", day: "numeric", ...(date.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }) }).format(date);
  return `${day} ${time}`;
}

/** A whole date in the reader's language: "Oct 18". */
export function fmtDay(ms: number, locale: string): string {
  return new Intl.DateTimeFormat(localeTag(locale), { month: "short", day: "numeric" }).format(new Date(ms));
}

/** Where the always-visible end of a long file name starts: its extensions
 * (".pdf.llm.md") and the four characters before them. */
export function nameTail(name: string): number {
  const extensions = /(?:\.[A-Za-z0-9]{1,8}){1,3}$/.exec(name);
  const start = extensions ? extensions.index : name.length;
  return Math.max(0, start - 4);
}

/** A long web address cut like a file name: the head keeps the scheme and
 * host and gives up the middle of the path first, the tail keeps the end of
 * the path (from its last "/", at most 20 characters, from a word break when
 * one is near), and a query or
 * fragment after it gives way before either. The three parts join to `url`. */
export function urlParts(url: string): [head: string, tail: string, rest: string] {
  const match = /^([a-z][a-z\d+.-]*:\/\/[^/?#]*)([^?#]*)(.*)$/is.exec(url);
  if (!match) {
    const cut = nameTail(url);
    return [url.slice(0, cut), url.slice(cut), ""];
  }
  const [, host, path, rest] = match;
  const slash = path.lastIndexOf("/", path.length - 2);
  let start = Math.max(slash, 0, path.length - 20);
  // A long last segment is cut at a word break ("-to-markdown.html").
  if (start > slash) {
    const word = path.slice(start).search(/[-_ ]/);
    if (word > 0 && word < 12) start += word;
  }
  return [host + path.slice(0, start), path.slice(start), rest];
}
