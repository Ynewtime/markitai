// The sidebar's version line (v1.16): `1.2.1 a1b2c3d 10-14 22:48`, the version,
// the commit it was built from and when, in the viewer's own time zone. A build
// from a tree with uncommitted changes marks its commit `-dirty` (git's own
// word); a development version says `-dev` in the version itself. An older
// backend without `build` shows its version alone.
import type { Capabilities } from "../api/types.ts";

const pad = (value: number) => String(value).padStart(2, "0");

/** A build time as digits in local time: `10-14 22:48` this year, `2025-10-14` before. */
export function buildTime(value: string, now: Date = new Date()): string | null {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  const day = `${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  if (date.getFullYear() !== now.getFullYear()) return `${date.getFullYear()}-${day}`;
  return `${day} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** The build's date and time in local time, `20261017 19:54`: the sidebar's "Build:" row and the phone Settings list. */
export function buildDate(value: string): string | null {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** The full local time, with the offset, for the line's tooltip. */
export function buildStamp(value: string): string | null {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  const offset = -date.getTimezoneOffset();
  const sign = offset < 0 ? "-" : "+";
  const zone = `UTC${sign}${pad(Math.floor(Math.abs(offset) / 60))}:${pad(Math.abs(offset) % 60)}`;
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())} ${zone}`;
}

/** The commit as shown beside GitHub: short, `-dirty` for a build with uncommitted changes; null when unknown. */
export function commitLabel(line: VersionLine): string | null {
  return line.parts.commit === null ? null : line.parts.dirty ? `${line.parts.commit}-dirty` : line.parts.commit;
}

export interface VersionLine {
  /** The sidebar's line. */
  text: string;
  /** The phone header's: version and commit. */
  short: string;
  /** Spelled out, for the tooltip. */
  parts: { version: string; commit: string | null; dirty: boolean; built: string | null; date: string | null };
}

export function versionLine(caps: Pick<Capabilities, "version" | "build"> | null, now: Date = new Date()): VersionLine | null {
  if (!caps) return null;
  const build = caps.build;
  const version = (build?.version || caps.version || "").trim();
  if (!version) return null;
  const commit = build?.commit?.trim() ? build.commit.trim().slice(0, 12) : null;
  const dirty = build?.dirty === true;
  const shownCommit = commit === null ? null : dirty ? `${commit}-dirty` : commit;
  const when = build?.built_at ? buildTime(build.built_at, now) : null;
  const short = [version, shownCommit].filter(Boolean).join(" ");
  return {
    text: [short, when].filter(Boolean).join(" "),
    short,
    parts: { version, commit, dirty, built: build?.built_at ? buildStamp(build.built_at) : null, date: build?.built_at ? buildDate(build.built_at) : null },
  };
}
