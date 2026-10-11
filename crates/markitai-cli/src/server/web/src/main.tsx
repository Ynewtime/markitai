import { render } from "preact";
import { initToken } from "./api/token.ts";
import { App } from "./app.tsx";
import { CrashBoundary } from "./components/crash.tsx";
import { detectLocale, loadLocale } from "./i18n/index.ts";

// Before anything is fetched: take the launch token and clear it from the address bar.
initToken();
// The reader's language is in place before the first paint (boot.js preloads its pack).
await loadLocale(detectLocale()).catch(() => undefined);

const root = document.getElementById("root");
if (root !== null) {
  render(
    <CrashBoundary>
      <App />
    </CrashBoundary>,
    root,
  );
}
