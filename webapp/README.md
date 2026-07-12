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

```sh
pnpm build   # tsc + vite build -> dist/
```

`markitai serve` auto-detects the built UI: it serves the package's bundled
`serve/static/` if present, else this repo's `webapp/dist/`. So after
`pnpm build`, plain `uv run markitai serve` hosts the app on its own port —
no proxy needed.

## Layout

- `src/styles/app.css` — all brand tokens (Tailwind `@theme`) + component CSS
- `src/api/` — typed client mirroring the serve API contract
- `src/hooks/useJobs.ts` — session state: jobs, SSE item/job events
- `src/i18n.ts` — en/zh dictionaries (auto-detected, default en)
