# markitai webapp

Web UI for `markitai serve` — Vite + React 19 + TypeScript + Tailwind CSS v4.

## Dev

Run the API and the dev server side by side:

```sh
uv run markitai serve --port 3611 --no-open   # API on 127.0.0.1:3611
pnpm install
pnpm dev                                      # Vite proxies /api -> 127.0.0.1:3611
```

## Build

From the repository root, build the app and sync it into the Python package:

```sh
scripts/sync_webapp_static.sh
```

Use `scripts/sync_webapp_static.sh --check` to fail when the committed package
assets are stale. `markitai serve` serves the bundled `serve/static/` directory
when installed, or falls back to this repo's `webapp/dist/` during development.

## Layout

- `src/styles/app.css` — all brand tokens (Tailwind `@theme`) + component CSS
- `src/api/` — typed client mirroring the serve API contract
- `src/hooks/useJobs.ts` — session state: jobs, SSE item/job events
- `src/i18n.ts` — en/zh dictionaries (auto-detected, default en)
