// Interface language. An explicit choice is remembered in this browser; without
// one, a browser whose first language starts with `zh` gets Chinese.

export type Locale = "en" | "zh";
export const LANG_KEY = "markitai.lang";

function storage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export function detectLocale(
  stored: string | null = readStored(),
  language: string | undefined = globalThis.navigator?.language,
): Locale {
  if (stored === "en" || stored === "zh") return stored;
  return String(language ?? "").toLowerCase().startsWith("zh") ? "zh" : "en";
}

function readStored(): string | null {
  try {
    return storage()?.getItem(LANG_KEY) ?? null;
  } catch {
    return null;
  }
}

export function storeLocale(locale: Locale): void {
  try {
    storage()?.setItem(LANG_KEY, locale);
  } catch {
    /* The choice lasts for this page only. */
  }
}
