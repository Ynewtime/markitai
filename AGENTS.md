# Markitai Rust engineering protocol

Work in this Rust repository. A separately supplied reference checkout is a read-only
behavioral reference, not a build dependency. Contributor commands are in
[development](docs/development.md); product guides start at [docs](docs/index.md).

## Coordination and Git

- The coordinator owns Git, manifests and integration builds. Workers edit only
  assigned paths; freeze tracked files during gates, builds and packaging.
- Preserve unexpected changes. Never reset, clean, stash or switch branches to resolve concurrent work.
  History rewriting requires an explicit user request, an isolated copy and
  expected remote-ref checks; ordinary pushes must not force-update history.
  Commit only explicit reviewed paths.
- Keep the current version (the last release's) until a new release is
  authorized. Deliver to `main` for user verification; do not create
  releases/tags, publish packages or change visibility without explicit
  authorization.
- Pick a release's version bump by [versioning](docs/versioning.md); a change to
  a stored format follows its stored-data rules.
- Record test commands, source/artifact identities, fixture provenance and scope
  in ignored `.local/` evidence. Use Git history for completed work; do not keep
  duplicate status, handoff or completed-task documents in the user guides.
- Keep project-local memory and pending design tasks in ignored `.local/memory/`.
- Distinguish source checks, hosted CI, installed packages and real-user tests.
  A passing build or smoke test does not establish feature parity.

## Test isolation and privacy

- Never commit personal home paths, machine names, private email addresses,
  credentials or local environment dumps. Use repository-relative paths,
  environment variables or clearly synthetic examples; keep raw evidence local.
- Never print, commit or replace credentials. Every CLI test isolates `HOME`
  and `MARKITAI_HOME`, plus Windows user state when applicable.
- Use repository fixtures, authored `.local/` samples and reference fixtures.
  Do not read personal folders or real `~/.markitai`, or send user identity
  information in test requests. Live-provider tests need explicitly scoped test
  credentials and authorization.
- Start long tasks with tool background/session mode, never `cmd &`.
- Clean only regenerable idle caches; retain evidence and cited artifacts.

## Validation

Every commit requires a fresh round and the checks below, with tracked bytes
and modes unchanged during verification. Check existing `.local/` records before
choosing the round. A round is a `.local/desktop-integration-rNN/` folder: copy
the latest one's `gate.sh` and `freeze.py`, which run the checks with an
isolated `HOME` and `MARKITAI_HOME`, compare the tracked files before and
after, and keep the logs.

```sh
python3.13 scripts/check.py
python3.13 scripts/desktop_web_build.py vet
sh docs/validation/drivers/windows-check-round38/check-windows.sh clippy -- -D warnings
```

- For pdf-inspector changes, rsync `vendor/pdf-inspector` to a copy under
  `.local/`, touch each source file and run its own suite there before and
  after the change: the failed names must be the same set. Do not accept stale
  Cargo results.
- The shared workbench design lives here (owner decision, 2026-10-11):
  `crates/markitai-cli/src/server/web/` is the one frontend both this server and
  Python markitai serve, through the API contract in [web UI](docs/web-ui.md)
  and `openapi.json`. UI changes keep that design and contract working against
  both backends. Run `npm ci` and `node build.mjs`, commit generated `dist/`
  with the change, and package it for the other backend with
  `node scripts/package.mjs OUTDIR`.
- Retrieve full CI logs with `gh api repos/Ynewtime/markitai/actions/jobs/<id>/logs
  --allow-escape-sequences`, never truncated `gh run view --log`. Verify the exact
  commit, attempt and expected jobs. Do not restart a watcher or dispatch another
  run merely because an observation timed out.

## Implementation and documentation

- Core code stays independent of Python/Node/Go runtimes. Bindings call Rust in
  process; unsupported capabilities return explicit errors. macOS Vision is the
  one exception to reading in process: it reads in a helper program,
  `markitai-vision-reader`, that can be killed and restarted when a recognition
  stops ([local OCR](docs/ocr.md)).
- Keep user guides concise and current. Put applicable limits in the relevant
  topic guide, and public changes in the unreleased notes; avoid duplicated work logs.
- After each user-reported fix, check all project documentation for affected
  claims, including help, examples, configuration, format limits and Agent guides.
  Refresh affected content in the same change; leave unrelated documents alone.
- `CHANGELOG.md` and `CHANGELOG.zh.md` hold released versions only. Append
  unreleased notes to `.local/changelog-unreleased.md` and
  `.local/changelog-unreleased.zh.md` in the main checkout (shared by every
  worktree), in the CHANGELOG format, keeping English and Chinese in step; the
  release commit moves them into the CHANGELOG. Corrections to released
  sections may land at any time.
