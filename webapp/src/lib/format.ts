/** Small display formatters. All output is mono/tabular-nums friendly. */

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** A duration split for display: under a minute the tenths are kept and
 * `minutes` is 0; at a minute and above it is whole minutes plus the leftover
 * seconds. One rounding step behind both the printed and the spoken label, so
 * the two can never disagree. */
export function durParts(ms: number): { minutes: number; seconds: number } {
  // Round to the displayed tenth first, so 59.96s becomes a full minute rather
  // than a value the label renders as "60.0s".
  const tenths = Math.round(Math.max(0, ms) / 100) / 10;
  if (tenths < 60) return { minutes: 0, seconds: tenths };
  const whole = Math.round(tenths);
  return { minutes: Math.floor(whole / 60), seconds: whole % 60 };
}

/** "4.2s" under a minute, "1:23" / "1:02:03" above it — a long batch should
 * not print a four-digit second count. */
export function fmtDur(ms: number): string {
  const { minutes, seconds } = durParts(ms);
  if (minutes === 0) return `${seconds.toFixed(1)}s`;
  const s = String(seconds).padStart(2, "0");
  if (minutes < 60) return `${minutes}:${s}`;
  return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${s}`;
}

/** "$0.0123" — trailing zeros are noise; a zero cost reads as "$0". */
export function fmtCost(usd: number): string {
  const trimmed = usd.toFixed(4).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  return `$${trimmed}`;
}

/** ISO timestamp -> "2026-07-12". */
export function fmtDate(iso: string): string {
  return iso.slice(0, 10);
}

/** Server ISO timestamp -> compact "07-12 14:30" in the reader's own zone.
 * Offset-aware values are converted; a naive timestamp is already server-local
 * and is shown as written. Unparseable input falls back to the raw slice. */
export function fmtDateTime(iso: string | null): string {
  if (iso === null || iso.length < 16) return "-";
  const pad = (n: number) => String(n).padStart(2, "0");
  const zoneAware = /(Z|[+-]\d{2}:\d{2})$/.test(iso);
  const date = zoneAware ? new Date(serverTimestampMs(iso) ?? Number.NaN) : new Date(iso);
  if (Number.isNaN(date.getTime())) return `${iso.slice(5, 10)} ${iso.slice(11, 16)}`;
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** Parse Python's offset-aware ISO timestamps without Date.parse quirks.
 * Safari rejects some otherwise valid six-digit fractional timestamps. */
export function serverTimestampMs(value: string): number | null {
  const match =
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|([+-])(\d{2}):(\d{2}))$/.exec(
      value,
    );
  if (match === null) return null;
  const [
    ,
    year,
    month,
    day,
    hour,
    minute,
    second,
    fraction = "",
    zone,
    sign,
    zoneHour,
    zoneMinute,
  ] = match;
  const milliseconds = Number((fraction + "000").slice(0, 3));
  const utc = Date.UTC(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second),
    milliseconds,
  );
  if (!Number.isFinite(utc)) return null;
  if (zone === "Z") return utc;
  const offset = (Number(zoneHour) * 60 + Number(zoneMinute)) * 60_000;
  return sign === "+" ? utc - offset : utc + offset;
}

/** Latin words + CJK chars, so zh documents count sensibly too. Kept to real
 * CJK blocks — kana, Extension A, Unified Ideographs, the F900-FAFF
 * compatibility block, and the plane-2/3 supplementary extensions (B and up,
 * hence the u flag) — so nearby scripts such as Hebrew/Arabic presentation
 * forms still count per word, not per character. */
export function countWords(s: string): number {
  const cjkRe = /[぀-ヿ㐀-䶿一-鿿豈-﫿\u{20000}-\u{3FFFF}]/gu;
  const cjk = s.match(cjkRe)?.length ?? 0;
  const words = s.replace(cjkRe, " ").match(/\S+/g)?.length ?? 0;
  return cjk + words;
}

export function utf8Bytes(s: string): number {
  return new TextEncoder().encode(s).length;
}

/** First line, exception-class prefix stripped, capped — failed items print
 * a short reason inline (mock register: "fetch failed: 403"). The row label
 * already shows the URL, so the "All fetch strategies failed for <url>:"
 * preamble is dropped down to the per-strategy detail. */
export function shortError(err: string): string {
  const line = (err.split("\n", 1)[0] ?? err)
    .replace(/^[A-Za-z]*Error:\s*/, "")
    .replace(/^All fetch strategies failed for \S+\s*(?:-\s*)?/, "");
  return line.length > 120 ? `${line.slice(0, 119)}…` : line;
}
