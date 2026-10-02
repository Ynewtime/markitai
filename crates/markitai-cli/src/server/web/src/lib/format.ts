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

/** "4.2s", then "1:23", then "1:02:03". */
export function fmtDur(ms: number): string {
  const { minutes, seconds } = durParts(ms);
  if (minutes === 0) return `${seconds.toFixed(1)}s`;
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
