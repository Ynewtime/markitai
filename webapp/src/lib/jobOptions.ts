import type { JobOptions } from "../api/types";

/** Every option a job can carry, all unset.
 *
 * One place, because the field list kept going stale: each time an option was
 * added, the hand-written literals in the retry path, the placeholder job and
 * every test fixture had to be found and edited, and a missed one silently
 * dropped that option.
 */
export function emptyJobOptions(): JobOptions {
  return {
    preset: null,
    llm: null,
    ocr: null,
    profile: null,
    alt: null,
    desc: null,
    screenshot: null,
    screenshot_only: null,
    pure: null,
    no_cache: null,
    no_compress: null,
    strategy: null,
    backend: null,
  };
}

/** The same, with only what the caller cares about set. */
export function jobOptions(over: Partial<JobOptions> = {}): JobOptions {
  return { ...emptyJobOptions(), ...over };
}

/** Reduce a job snapshot's options to the request shape.
 *
 * Snapshot options are a pass-through: the server keeps extra bookkeeping
 * there (``origin: "cli" | "web"`` on rehydrated and CLI-recorded jobs) and
 * older jobs lack options added since. ``POST .../retry`` rejects unknown
 * keys, so replaying a snapshot's options verbatim failed every enhance of a
 * rehydrated job with "Extra inputs are not permitted".
 */
export function jobOptionsFromSnapshot(raw: JobOptions): JobOptions {
  const options: Record<string, unknown> = { ...emptyJobOptions() };
  const source: Record<string, unknown> = { ...raw };
  for (const key of Object.keys(options)) {
    if (source[key] !== undefined) options[key] = source[key];
  }
  return options as unknown as JobOptions;
}
