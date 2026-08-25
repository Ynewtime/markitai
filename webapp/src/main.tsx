import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { initToken } from "./api/token";
import "./styles/app.css";

// Before anything fetches: capture ?token= and scrub it from the URL bar.
initToken();

const rootEl = document.getElementById("root");
if (rootEl !== null) {
  createRoot(rootEl).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
