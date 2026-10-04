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
- Keep **1.3.0-dev** until a new release is authorized. Deliver to `main` for user
  verification; do not create releases/tags, publish packages or change visibility
  without explicit authorization.
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

Every commit requires a fresh round and both checks below, with tracked bytes
and modes unchanged during verification. Check existing `.local/` records before
choosing the round; the local `run_checked_gate.py` wrapper supplies isolation.

```sh
python3.13 .local/integration-round36/run-gate2.py rNNN
sh docs/validation/drivers/windows-check-round38/check-windows.sh clippy -- -D warnings
```

- For pdf-inspector changes, rsync to `.local/pdf-backlog-r1/vendor-copy`, touch
  each source file, run its own suite and match the exact 21 failed names in
  `.local/pdf-backlog-r1/failures-r129.txt`. Do not accept stale Cargo results.
- UI changes preserve the reference webapp design. Run `npm ci` and
  `node build.mjs`, and commit generated `dist/` with the change.
- Retrieve full CI logs with `gh api repos/Ynewtime/markitai-rust/actions/jobs/<id>/logs
  --allow-escape-sequences`, never truncated `gh run view --log`. Verify the exact
  commit, attempt and expected jobs. Do not restart a watcher or dispatch another
  run merely because an observation timed out.

## Implementation and documentation

- Core code stays independent of Python/Node/Go runtimes. Bindings call Rust in
  process; unsupported capabilities return explicit errors.
- Keep user guides concise and current. Put applicable limits in the relevant
  topic guide, and public changes in the changelog; avoid duplicated work logs.
- After each user-reported fix, check all project documentation for affected
  claims, including help, examples, configuration, format limits and Agent guides.
  Refresh affected content in the same change; leave unrelated documents alone.
- Keep `CHANGELOG.md` and `CHANGELOG.zh.md` synchronized under undated `[1.3.0]`.
