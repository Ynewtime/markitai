import type { ConversionBackend, FetchStrategy } from "../api/types";

/** Everything the CLI can ask of a single conversion that the main options
 * row has no space for. Kept beside the types rather than in the component so
 * the panel file exports only its component (fast refresh). */
export interface Advanced {
  alt: boolean;
  desc: boolean;
  screenshot: boolean;
  screenshotOnly: boolean;
  pure: boolean;
  noCache: boolean;
  noCompress: boolean;
  strategy: FetchStrategy;
  backend: ConversionBackend;
}

export const ADVANCED_DEFAULTS: Advanced = {
  alt: false,
  desc: false,
  screenshot: false,
  screenshotOnly: false,
  pure: false,
  noCache: false,
  noCompress: false,
  strategy: "auto",
  backend: "native",
};

/** Whether anything differs from the defaults. The panel is collapsed by
 * default, so a setting made once and forgotten would otherwise shape every
 * later conversion invisibly; the toggle wears a dot when this is true. */
export function advancedIsCustom(a: Advanced): boolean {
  return (Object.keys(ADVANCED_DEFAULTS) as (keyof Advanced)[]).some(
    (key) => a[key] !== ADVANCED_DEFAULTS[key],
  );
}
