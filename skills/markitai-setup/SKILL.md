---
name: markitai-setup
description: "Install, configure, and heal a markitai (mkai) installation. Use when markitai is missing or misbehaving, an LLM provider must be wired up for --llm (API key or Claude/Copilot/ChatGPT subscription auth), Playwright browser rendering or LibreOffice capability is needed, or markitai doctor reports failures."
---

# Set up and heal markitai

Target state: `markitai doctor` exits 0 and shows ✓ on every capability the user's workflow needs. Missing optional tools that the workflow doesn't use are warnings, not failures — stop when the needed rows are green, not when every row is.

## Workflow

1. **Install the package** (skip if `markitai -V` works):

   ```bash
   uv tool install markitai        # isolated tool env (recommended); pipx install markitai also works
   ```

   Humans setting up interactively can instead use the guided installer, which handles Python/uv, extras, optional components, and mirrors: `curl -fsSL https://markitai.dev/setup.sh | sh` (Windows: `irm https://markitai.dev/setup.ps1 | iex`). Requires Python 3.11–3.13. Both `markitai` and the `mkai` alias are installed.

2. **Add extras only when the workflow needs them** — reinstall with the extra spelled into the requirement:

   | Extra | Unlocks | Install (uv tool) |
   |---|---|---|
   | `browser` | Playwright rendering for JS-heavy pages, URL screenshots | `uv tool install 'markitai[browser]' --force` |
   | `claude-agent` / `copilot` | Claude Agent SDK / GitHub Copilot SDK as LLM providers | `uv tool install 'markitai[claude-agent]' --force` |
   | `extra-fetch` | curl-cffi TLS-impersonating static fetch | same pattern |
   | `kreuzberg` | `.xml` `.tsv` `.rtf` `.rst` `.org` `.tex` `.odt` `.ods` conversion, `-b kreuzberg` | same pattern |
   | `svg` / `heif` | SVG rasterization / HEIC-HEIF-AVIF input | same pattern |
   | `all` | everything above | `uv tool install 'markitai[all]' --force` |

   `markitai doctor --suggest-extras` prints the comma-separated extras the current environment would benefit from.

3. **Create config**: `markitai init` (interactive wizard: detects providers, checks dependencies) or `markitai init --yes` for defaults without prompts; `--local` writes `./markitai.json` instead of `~/.markitai/config.json`.

4. **Run the doctor loop** until the needed rows are green — run `markitai doctor`, fix the first failing line, re-run:
   - **Playwright/Chromium missing**: `markitai doctor --fix` installs Chromium when the Playwright package is present; with a core-only install it exits safely and names the extra to add first (step 2, `browser`). Every doctor run smoke-tests a real Chromium launch, so a green result is trustworthy. Linux launch failures name the `playwright install-deps chromium` recovery.
   - **LibreOffice missing** (legacy `.doc`/`.ppt` / slide rendering; `.xls` converts in pure Python and needs neither): `sudo apt-get install libreoffice` / `brew install --cask libreoffice`. On macOS without LibreOffice, installed MS Office is driven via AppleScript instead — first conversion triggers a one-time consent dialog per app; headless machines should set `{"office": {"macos_fallback": false}}`.
   - **LLM/auth rows failing**: wire a provider (step 5).
   - `--json` gives a machine-readable snapshot (mutually exclusive with `--fix`).

5. **Wire an LLM provider** (needed for `--llm`, `--alt`, `--desc`, presets `rich`/`standard`). Two routes:
   - **API key**: set the provider env var (e.g. `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`) and configure a model named `provider/model`, e.g. `gemini/gemini-3.1-flash-lite-preview`. Reference env vars in config as `"api_key": "env:GEMINI_API_KEY"`.
   - **Subscription (no API key)**: `claude-agent/sonnet` (Claude Code CLI login), `copilot/gpt-5.4` (Copilot CLI login), `chatgpt/gpt-5.4` (OAuth on first use). Check and repair auth with `markitai auth` (overview) and `markitai auth copilot|claude|chatgpt status|login`.
   - Model lists, `api_base`/Azure/Ollama examples, vision requirements, and provider error table: [references/providers.md](references/providers.md).

6. **Verify end-to-end** — the setup is done only when both hold:
   - `markitai doctor` exits 0 with the workflow's rows green.
   - A real conversion succeeds: `markitai https://example.com --pure` (plain), plus `markitai <sample> --llm` if LLM was configured.

## Quick diagnosis table

| Symptom | Fix |
|---|---|
| "SDK not installed" for a local provider | reinstall with the `claude-agent` / `copilot` extra (step 2) |
| "CLI not found" | install the provider CLI: `curl -fsSL https://claude.ai/install.sh \| bash` (Claude) / `curl -fsSL https://gh.io/copilot-install \| bash` (Copilot) |
| "Not authenticated" | `markitai auth <provider> login`; CI alternatives: `COPILOT_GITHUB_TOKEN`/`GH_TOKEN` for Copilot, `CLAUDE_CODE_USE_BEDROCK=1`/`VERTEX`/`FOUNDRY` for Claude |
| Active model references a missing env var | export the variable or fix the `env:` reference; doctor treats it as a failure, not a warning |
| Rate limit / timeout on `--llm` | retry later or lower `--llm-concurrency`; timeouts adapt to document size |
| Secrets needed for URL strategies | `JINA_API_KEY`; `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` (token permissions: Browser Rendering Edit, Workers AI Read) |

Config debugging: `markitai config list` (secrets redacted; never paste `--show-secrets` output into shared channels), `markitai config path`, `markitai config validate`. Resolution order: CLI args > env vars > config file (`--config` > `MARKITAI_CONFIG` > `./markitai.json` > `~/.markitai/config.json`) > defaults. `.env` files load from `./.env` then `~/.markitai/.env`.
