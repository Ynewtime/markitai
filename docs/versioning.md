# Versioning

Markitai versions are `MAJOR.MINOR.PATCH`; the highest change in a release picks the number.

| Raise | When |
|---|---|
| Major | Something programs rely on stops working as before: the `-f json`, `--json`, report and `images.json` shapes; the HTTP API (`openapi.json`); MCP tool schemas; exit codes; output layout and file names; configuration keys and their meaning; environment variables; binding APIs; stored data an older minor cannot read |
| Minor | New features, options, configuration keys or values, API or JSON fields; any additive change to the above; changed messages, help or startup lines (with a **Changed** note when scripts could notice) |
| Patch | Fixes only: behaviour that was wrong according to the docs, with no new options or fields |

The HTTP API contract number `v1.N` rises with each additive API change (minor
releases or later); a breaking API change needs a major release.

## Deprecation

To remove or rename something, deprecate it in a minor release (it keeps
working and warns, naming the replacement), then remove it in the next major
release, where the old spelling is refused with a hint.

## Stored data

A newer markitai reads what older versions saved. An older markitai that finds
data a newer one wrote refuses it with a message naming the file, and leaves the
file unchanged.
