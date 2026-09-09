import type { ConversionBackend, FetchStrategy } from "../api/types";

/** Everything the CLI can ask of a single conversion that the main options
 * row has no space for. Kept beside the types rather than in the component so
 * the panel file exports only its component (fast refresh). */
export interface Advanced {
  /** null inherits the selected preset; false is an explicit override. */
  alt: boolean | null;
  desc: boolean | null;
  screenshot: boolean | null;
  screenshotOnly: boolean;
  pure: boolean;
  noCache: boolean;
  noCompress: boolean;
  strategy: FetchStrategy;
  backend: ConversionBackend;
}

export const ADVANCED_DEFAULTS: Advanced = {
  alt: null,
  desc: null,
  screenshot: null,
  screenshotOnly: false,
  pure: false,
  noCache: false,
  noCompress: false,
  strategy: "auto",
  backend: "native",
};
