# markitai webapp

Web UI for `markitai serve` — Vite + React 19 + TypeScript + plain CSS (no utility framework; the sheet carries its own preflight reset).

## Dev

Run the API and the dev server side by side:

```sh
uv run markitai serve --port 3611 --no-open   # API on 127.0.0.1:3611
bun install --frozen-lockfile
bun run dev                                   # Vite proxies /api -> 127.0.0.1:3611
```

Keep TypeScript on 6.x while `typescript-eslint` requires its compiler API.
After dependency updates, run `bun run test`, `bun run lint`,
`bun run typecheck`, and `bun run check:css`, then rebuild the packaged assets.

## Build

From the repository root, build the app and sync it into the Python package:

```sh
scripts/sync_webapp_static.sh
```

Use `scripts/sync_webapp_static.sh --check` to fail when the committed package
assets are stale. `markitai serve` serves the bundled `serve/static/` directory
when installed, or falls back to this repo's `webapp/dist/` during development.

## Conversion option rules

`src/lib/conversionOptions.ts` resolves the values the panel shows, the API
receives and the CLI preview renders. Preset definitions come from
`/api/capabilities.preset_options`, config-file overrides included;
`BUILTIN_PRESET_OPTIONS` in that module is only the fallback while capabilities
load, or when talking to an older server.

| Built-in preset | LLM | OCR | alt / description | Page screenshots |
| --- | --- | --- | --- | --- |
| minimal | off | off | off | off |
| standard | on | off | on | off |
| rich | on | off | on | on |

Presets and overrides:

- Selecting a preset, including reselecting the current one, resets those five
  features. Output profile, source mode, cache and converter choices are
  independent and survive it.
- Image overrides are tri-state (`src/lib/advanced.ts`): `null` inherits the
  preset, `false` is an explicit off. An effective bundle that matches a preset
  highlights it, anything else shows **Custom**. The match is display only —
  inheritance and overrides stay as they are, so turning LLM on under `minimal`
  neither enables alt/description nor promotes the run to `standard`.
- Image overrides are persisted with the preset; source, remote-service and
  cache choices are not.
- LLM off, or plain mode, suspends alt/description analysis without discarding
  the user's choices, and re-enabling normal LLM processing restores them. With
  no model configured, the LLM-dependent presets are disabled rather than
  silently enabled.

Options that imply each other:

- Screenshot source implies screenshot capture, never LLM. Without LLM, URLs
  return screenshot references in a Markdown wrapper for web preview rather
  than extracted text, and files convert normally. Screenshot source and plain
  mode are mutually exclusive, because the file and URL pipelines resolve that
  combination with different precedence.
- LLM + OCR uses vision-model OCR, and the panel flags the added model cost;
  OCR alone uses the optional local engine. An output profile stays available
  with LLM off.
- The Cloudflare URL strategy implies its file backend, matching the CLI, and
  the manual backend choice is restored on leaving that strategy. An explicit
  file backend replaces both inherited converter flags. Separate notices name
  the selected URL service (Defuddle, Jina or Cloudflare Browser Rendering) and
  any Cloudflare Workers AI file upload; `auto` discloses policy-gated remote
  fallbacks.
- Skip-cache bypasses reads and keeps writing, matching CLI `--no-cache`.

CLI preview (`src/lib/cli.ts`):

- It is opt-in and compact: the toolbar's terminal button, or opening the
  Options panel, reveals it, so the first screen stays on the input.
- It prints only the deviations from the selected preset, image opt-outs
  included, and assumes a default local CLI configuration with the same preset
  definitions — minimal defaults therefore render as just `--preset minimal`.
  The flags that belong to no preset (`--screenshot-only`, `--pure`,
  `--no-cache`, `--no-compress`) appear only when switched on.
- While no URLs are typed, `<your-files-or-url-or-url_files>` stands in for
  paths, URLs or URL-list files. The command carries no explanatory comment;
  hover and focus help carry the placeholder and the configuration assumption.

Layout and accessibility of the panel:

- The options disclosure and the active summary stay available at every
  viewport width. Mobile puts the input area above a single
  options/upload/convert row with 44px targets. Shrinkable layout tracks keep a
  long command from pushing controls out of view; commands wrap at spaces on
  desktop and mobile alike.
- Every option group and every choice carries a bilingual accessible
  description. Portal tooltips open on hover, focus or tap, stay inside the
  viewport, and dismiss on Escape, on outside interaction, or when focus
  leaves. A disabled choice keeps a keyboard-focusable help target.

## Layout

- `src/styles/app.css` — all design tokens + component CSS (plain CSS, no
  utility framework; every size, radius, duration and shadow is a token)
- `src/api/` — typed client mirroring the serve API contract
- `src/hooks/useJobs.ts` — session state: jobs, SSE item/job events
- `src/i18n.ts` — en/zh dictionaries (auto-detected, default en)
- `scripts/check-css-scale.mjs` — the scale guard (`bun run check:css`)

## Design scale

One ladder, enforced by `bun run check:css`:

| Family | Allowed values | Tokens |
| --- | --- | --- |
| Type | 10, 11, 12, 14, 16, 18, 20, 24px; 9px exception | `--text-2xs` … `--text-2xl` |
| Radius | 0, 2, 6, 8, 14, 16, 999px | `--r-xs`, `--r-sm`, `--r-term`, `--r-panel` |
| Motion | 120, 140, 160ms | `--dur-fast`, `--dur-base`, `--dur-slow` |
| Shadow | popover, modal only | `--shadow-pop`, `--shadow-modal` |

The guard fails the build on any value outside these sets, so a new step is a
token change first. Every exemption is declared in
`scripts/check-css-scale.mjs`: `DISPLAY_SIZES` for the fluid hero `clamp()`,
`PRINT_SIZES` for the print sheet's own pt ladder, `RESET_SIZES` for the
preflight reset's relative sizes, and `LOOP_DURATIONS` for spinner periods. The
same guard fails when the preflight reset
(`box-sizing`/`margin` on `*`) goes missing. Dark tokens have one definition shared by the
`prefers-color-scheme` block and `[data-theme="dark"]`; `--font-sans` carries a
CJK fallback chain because the UI ships a Chinese dictionary.
