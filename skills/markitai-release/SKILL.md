---
name: markitai-release
description: "Cut or re-publish a markitai release to PyPI. Use when bumping the markitai version, tagging vX.Y.Z, writing the bilingual changelog sections, or diagnosing a failed Release workflow run."
metadata:
  internal: true
---

# Releasing markitai

Releases are manual and **tag-driven**: pushing a `vX.Y.Z` tag runs `.github/workflows/publish.yml`, which gates on lint plus the full test matrix (Linux/macOS/Windows × Python 3.11–3.13), builds, verifies the tag against `__version__`, publishes to PyPI via trusted publishing, creates the GitHub Release, and then publishes `server.json` to the MCP Registry (GitHub OIDC, no stored token). Nothing publishes until you push a tag.

## Steps

1. **Preflight** on a clean `main` checkout: all Python gates in `markitai-dev` green locally, plus frontend tests, `scripts/sync_webapp_static.sh --check` and `bun run --cwd website docs:build` (CI re-runs the full matrix, but a red tag run wastes a cycle). Confirm the target version does not already exist on PyPI.

2. **Run the end-to-end check** on a clean `main`: `scripts/e2e_release_check.sh`. It builds the wheel, installs it as a tool into a throwaway `HOME`, and drives the public CLI through install → doctor → zero-config convert → real LLM enhancement → URL → batch → interrupt → resume → cache → failure messages → every input format → presets and profiles → run-shape flags → refusals → `.urls` list → config layers → Python API, then adds the serve/mcp/legacy extras and drives doctor, legacy Office, the serve API and `markitai-mcp` over stdio, then the browser/extra-fetch extras (Chromium via `doctor --fix`, Playwright rendering, screenshots, screenshot-only reading), the Cloudflare file backend, the remote strategies on a public page (skipped, with the reason, on fake-IP VPN resolvers), and a real two-phase Batch API job. It costs a few cents in real LLM calls and it catches the class of defect the suite cannot see: a flag that silently does nothing without a config file, a `--resume` that restarts, an install hint naming a command that does not work. It writes `WORKDIR/report.html` (verdict, per-step results in plain language, and the conversions themselves inline so you can judge quality by eye), with one numbered directory per step beside it holding that step's inputs, outputs and logs. It keeps the artifacts for inspection; `CLEANUP_ON_SUCCESS=1` removes them on a clean pass, and a failed run always keeps them. The top of the script documents every knob.

3. **Bump the version everywhere it is written**; they must match (`test_release_versions_match_the_package` fails otherwise):
   - `packages/markitai/src/markitai/__init__.py` `__version__`, the single source of truth: hatch reads it at build time and the workflow fails loudly if the tag disagrees with it.
   - Root `pyproject.toml` `version`, workspace identity only (unpublished), kept in sync, and the `markitai-workspace` entry in `uv.lock` (`uv lock --check` must pass).
   - Both `version` fields in the root `server.json`, the MCP Registry entry the workflow publishes after PyPI.

4. **Write both changelog sections** for `## [X.Y.Z] - YYYY-MM-DD` (Keep a Changelog format):
   - `CHANGELOG.md`: English; this section becomes the GitHub Release notes **verbatim**.
   - `CHANGELOG.zh.md`: the Chinese mirror of the same section.
   - Keep entries short: describe user-visible additions, behavior changes and fixes; keep implementation logs in review notes.
   - The website copies both files from the repo root at docs build time; rebuild to validate Vue/Markdown rendering.
   - Recheck CLI `--help`, both website languages and `skills/` whenever a flag, default, output layout or extra changes.

5. **Commit and push to main**: `chore(release): X.Y.Z`.

6. **Tag and push the tag.** This is the publish trigger:

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

7. **Verify. The release is done only when all of these hold:**
   - The Release workflow run is green end to end.
   - `https://pypi.org/project/markitai/` shows X.Y.Z (`uv tool install markitai==X.Y.Z` resolves).
   - The GitHub Release exists with notes matching the `CHANGELOG.md` section.
   - The MCP Registry lists X.Y.Z as latest: `curl -s https://registry.modelcontextprotocol.io/v0/servers/io.github.Ynewtime%2Fmarkitai/versions/X.Y.Z` answers 200.

8. **The MCP Registry entry is published by the workflow's last job** (`mcp-registry`, after PyPI, because the registry fetches that exact version's README and looks for the `mcp-name: io.github.Ynewtime/markitai` marker in it). It logs in with GitHub OIDC (the registry grants `io.github.<repo owner>/*`), publishes the tagged `server.json` with its versions set to the tag (warning if the committed ones were stale), and skips a version the registry already lists, so re-publishing a tag is safe. `mcp-publisher` is pinned by version and SHA-256 in the job's `env`; bump both together. By hand, if ever needed: `mcp-publisher validate server.json`, `mcp-publisher login github` (device code in the browser, once per machine), `mcp-publisher publish server.json`. PulseMCP and other aggregators ingest the official registry on their own. Skills need no registration: skills.sh indexes the public repo (`npx skills add Ynewtime/markitai --list`).

## Re-publishing an existing tag

After a transient failure (PyPI hiccup, runner outage): Actions → **Release** workflow → *Run workflow* with the existing `vX.Y.Z` tag as input. It re-runs the same gates against that tag. A code fix needs a new patch version rather than a moved tag.

## Failure modes

| Symptom | Cause and fix |
|---|---|
| Workflow fails at the tag/version check | `__version__` was not bumped (or mismatches the tag). Fix both version files on `main`, then move the tag: `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`, re-tag the fixed commit, push again |
| One OS/Python cell red, rest green | By design this blocks PyPI. Fix the platform failure on `main` and re-tag (new patch version if the tag already published anything) |
| Only `mcp-registry` is red (PyPI and the GitHub Release are out) | Read the job log. The registry was down or PyPI lagged: *Re-run failed jobs* on that run (nothing else is redone). The tagged `server.json` itself was refused (the job publishes the tag's copy): fix it on `main` and publish by hand (step 8) |
| GitHub Release has generated notes instead of the changelog | The `## [X.Y.Z]` section is missing from `CHANGELOG.md`; add it and re-run the workflow via the tag input |
