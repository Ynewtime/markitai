---
name: markitai-release
description: "Cut or re-publish a markitai release to PyPI. Use when bumping the markitai version, tagging vX.Y.Z, writing the bilingual changelog sections, or diagnosing a failed Release workflow run."
metadata:
  internal: true
---

# Releasing markitai

Releases are manual and **tag-driven**: pushing a `vX.Y.Z` tag runs `.github/workflows/publish.yml`, which gates on lint plus the full test matrix (Linux/macOS/Windows × Python 3.11–3.13), builds, verifies the tag against `__version__`, publishes to PyPI via trusted publishing, and creates the GitHub Release. Nothing publishes until you push a tag.

## Steps

1. **Preflight** on a clean `main` checkout: `uv run pytest -q`, `uv run ruff check`, and `uv run pyright packages/markitai/src` all green locally (CI re-runs the full matrix, but a red tag run wastes a cycle). Confirm the target version does not already exist on PyPI.

2. **Run the end-to-end check** on a clean `main`: `scripts/e2e_release_check.sh`. It builds the wheel, installs it as a tool into a throwaway `HOME`, and drives the public CLI through install → doctor → zero-config convert → real LLM enhancement → URL → batch → interrupt → resume → cache → failure messages. It costs a few cents in real LLM calls and it catches the class of defect the suite cannot see: a flag that silently does nothing without a config file, a `--resume` that restarts, an install hint naming a command that does not work. It writes `WORKDIR/report.html` — verdict, per-step results in plain language, and the conversions themselves inline so quality can be judged by eye — with one numbered directory per step beside it holding that step's inputs, outputs and logs. Artifacts are kept for inspection; `CLEANUP_ON_SUCCESS=1` removes them on a clean pass, and a failed run always keeps them. Every knob is documented at the top of the script.

3. **Bump the version in both files** — they must match:
   - `packages/markitai/src/markitai/__init__.py` `__version__` — the single source of truth; hatch reads it at build time and the workflow fails loudly if the tag disagrees with it.
   - Root `pyproject.toml` `version` — workspace identity only (unpublished), kept in sync.

4. **Write both changelog sections** for `## [X.Y.Z] - YYYY-MM-DD` (Keep a Changelog format):
   - `CHANGELOG.md` — English; this section becomes the GitHub Release notes **verbatim**.
   - `CHANGELOG.zh.md` — the Chinese mirror of the same section.
   - The website copies both files from the repo root at docs build time — no separate website step.

5. **Commit and push to main**: `chore(release): X.Y.Z`.

6. **Tag and push the tag** — this is the publish trigger:

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

7. **Verify — the release is done only when all of these hold:**
   - The Release workflow run is green end to end.
   - `https://pypi.org/project/markitai/` shows X.Y.Z (`uv tool install markitai==X.Y.Z` resolves).
   - The GitHub Release exists with notes matching the `CHANGELOG.md` section.

## Re-publishing an existing tag

After a transient failure (PyPI hiccup, runner outage): Actions → **Release** workflow → *Run workflow* with the existing `vX.Y.Z` tag as input. It re-runs the same gates against that tag; fixing code requires a new patch version, not a moved tag.

## Failure modes

| Symptom | Cause and fix |
|---|---|
| Workflow fails at the tag/version check | `__version__` was not bumped (or mismatches the tag). Fix both version files on `main`, then move the tag: `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`, re-tag the fixed commit, push again |
| One OS/Python cell red, rest green | By design this blocks PyPI. Fix the platform failure on `main` and re-tag (new patch version if the tag already published anything) |
| GitHub Release has generated notes instead of the changelog | The `## [X.Y.Z]` section is missing from `CHANGELOG.md` — add it and re-run the workflow via the tag input |
