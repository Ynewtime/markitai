# markitai webapp

Web UI for `markitai serve` — Vite + React 19 + TypeScript + Tailwind CSS v4.

## Dev

Run the API and the dev server side by side:

```sh
uv run markitai serve --port 3611 --no-open   # API on 127.0.0.1:3611
bun install
bun run dev                                   # Vite proxies /api -> 127.0.0.1:3611
```

## Build

From the repository root, build the app and sync it into the Python package:

```sh
scripts/sync_webapp_static.sh
```

Use `scripts/sync_webapp_static.sh --check` to fail when the committed package
assets are stale. `markitai serve` serves the bundled `serve/static/` directory
when installed, or falls back to this repo's `webapp/dist/` during development.

## Conversion option rules

`src/lib/conversionOptions.ts` resolves the values shown by the panel, sent to the
API and rendered by the CLI preview. Preset definitions come from
`/api/capabilities.preset_options`, including config-file overrides; the built-in
map is only a fallback while capabilities load or when using an older server.

| Built-in preset | LLM | OCR | alt / description | Page screenshots |
| --- | --- | --- | --- | --- |
| minimal | off | off | off | off |
| standard | on | off | on | off |
| rich | on | off | on | on |

- Selecting or reselecting a preset resets these five features. Output profile,
  source mode, cache and converter choices are independent and remain unchanged.
- Image overrides are tri-state: `null` inherits the preset; `false` explicitly
  disables a feature. An exact effective bundle highlights its matching preset;
  otherwise **Custom** is shown and no preset is pressed. Matching is display-only:
  the original preset inheritance and all overrides remain intact. Enabling LLM
  on minimal does not enable alt/description or force standard. Image overrides are
  persisted with the preset, but source, remote-service and cache choices are not.
- LLM off or plain mode pauses alt/description analysis without losing the user's
  choices. Re-enabling normal LLM processing restores them. No configured model
  means LLM-dependent presets are disabled, not silently enabled.
- Screenshot source implies screenshot capture, never LLM. Without LLM, URLs
  return screenshot references in a Markdown wrapper for web preview (not extracted
  text), and files keep normal conversion. Screenshot source and plain
  mode are mutually exclusive because the file and URL pipelines give conflicting
  combinations different precedence.
- LLM + OCR uses vision-model OCR; the panel explains the potential for more model costs. OCR alone uses
  the optional local engine. An output profile remains available with LLM off.
- The Cloudflare URL strategy implies its file backend, matching the CLI; the
  manual backend choice is restored when leaving that strategy. Explicit file
  backends replace both inherited converter flags. Separate notices identify the
  selected URL service (Defuddle, Jina or Cloudflare Browser Rendering) and any
  Cloudflare Workers AI file upload. Auto discloses policy-gated remote fallbacks.
- Skip-cache bypasses reads rather than disabling cache storage, matching CLI
  `--no-cache`.
- The CLI preview is compact: it omits features already supplied by the
  selected preset and retains explicit deviations, including image opt-outs. It
  assumes default local CLI configuration and matching preset definitions.
  Minimal defaults therefore need only `--preset minimal`. While no URLs are
  typed, `<your-files-or-url-or-url_files>` stands in for paths, URLs or URL-list
  files. There is no CLI comment; hover/focus help explains the placeholder and
  default-configuration assumption. `buildCliCommand` still accepts an `"explicit"` mode (every
  non-null value, including `--no-pure`, `--cache`, `--compress`) for callers
  that need it; the panel no longer exposes a toggle for it.
- The options disclosure and active summary stay available at all viewport
  widths. Mobile uses an integrated input area above one options/upload/convert
  action row with 44px targets. Shrinkable layout tracks prevent long commands
  from pushing controls out of view; commands wrap at spaces on desktop and mobile.
- Every option group and all 25 choices have bilingual accessible descriptions.
  Portal tooltips open on hover/focus or tap, stay within the viewport, and dismiss
  on Escape, outside interaction or focus departure. Disabled choices retain a
  keyboard-focusable help target.
- Download all ZIP has one rendered instance below the ledger, right-aligned on
  desktop and full-width on mobile; it never shares the expanded options row.
- Settings retain centered modal positioning, 5px WebKit scrollbars (thin on other
  engines), and right-aligned nonshrinking actions: at least 36px high on desktop
  and 44px on mobile.

## Layout

- `src/styles/app.css` — all brand tokens (Tailwind `@theme`) + component CSS
- `src/api/` — typed client mirroring the serve API contract
- `src/hooks/useJobs.ts` — session state: jobs, SSE item/job events
- `src/i18n.ts` — en/zh dictionaries (auto-detected, default en)
