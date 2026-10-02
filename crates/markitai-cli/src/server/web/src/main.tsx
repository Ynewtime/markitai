import { render } from "preact";
import { initToken } from "./api/token.ts";
import { App } from "./app.tsx";
import { CrashBoundary } from "./components/crash.tsx";

// Before anything is fetched: take the launch token and clear it from the address bar.
initToken();

const root = document.getElementById("root");
if (root !== null) {
  render(
    <CrashBoundary>
      <App />
    </CrashBoundary>,
    root,
  );
}
