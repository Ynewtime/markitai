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
