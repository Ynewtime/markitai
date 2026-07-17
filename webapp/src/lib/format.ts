/** Small display formatters. All output is mono/tabular-nums friendly. */

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** "4.2s" — the mock's duration language (seconds, one decimal). */
export function fmtDur(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

export function fmtCost(usd: number): string {
  return `$${usd.toFixed(4)}`;
}

/** ISO timestamp -> "2026-07-12". */
export function fmtDate(iso: string): string {
  return iso.slice(0, 10);
}

/** Server-local ISO timestamp -> compact "07-12 14:30". */
export function fmtDateTime(iso: string | null): string {
  if (iso === null || iso.length < 16) return "-";
  return `${iso.slice(5, 10)} ${iso.slice(11, 16)}`;
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
