// Interface language. An explicit choice is remembered in this browser; without
// one, the browser's first language picks Chinese (`zh…`), Japanese (`ja…`)
// or English.

export type Locale = "en" | "zh" | "ja";
export const LOCALES: Locale[] = ["en", "zh", "ja"];
export const LANG_KEY = "markitai.lang";
/** The `lang` attribute of the document for each interface language. */
export const LANG_TAGS: Record<Locale, string> = { en: "en", zh: "zh-CN", ja: "ja" };

function storage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

const isLocale = (value: string | null): value is Locale => value === "en" || value === "zh" || value === "ja";

export function detectLocale(
  stored: string | null = readStored(),
  language: string | undefined = globalThis.navigator?.language,
): Locale {
  if (isLocale(stored)) return stored;
  const first = String(language ?? "").toLowerCase();
  if (first.startsWith("zh")) return "zh";
  if (first.startsWith("ja")) return "ja";
  return "en";
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
