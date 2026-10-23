// Copies the release notes in changelog/ into the site with released versions only.
//
// The changelog pages change only with a release: a `## [X.Y.Z]` section is
// published once its vX.Y.Z tag exists, so an unreleased section never shows.
// Old versions (0.6–0.14) were never tagged, so the cut is "every section up to
// the newest version that has both a section here and a tag". Tags of other
// release lines in the repository (the Rust v2.x tags) have no section in these
// files and so never move the cut. Anything above it, or a heading that is not a
// version, stays off the site. Without readable tags the build fails rather than
// publish a section nobody has released.
//
// Usage: bun scripts/changelog.mjs           (docs:build and docs:dev run it)
//        bun scripts/changelog.mjs --check   (fails if a section would be dropped)

import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const SITE = join(dirname(fileURLToPath(import.meta.url)), '..')

const COPIES = [
  ['changelog/python-1.x.md', 'changelog.md'],
  ['changelog/python-1.x.zh.md', 'zh/changelog.md'],
]

const VERSION = /^(\d+)\.(\d+)\.(\d+)$/

function parse(version) {
  const m = VERSION.exec(version)
  return m ? m.slice(1).map(Number) : null
}

function compare(a, b) {
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] - b[i]
  return 0
}

function tags() {
  let out
  try {
    out = execFileSync('git', ['tag', '--list', 'v*'], { cwd: SITE, encoding: 'utf8' })
  } catch (err) {
    throw new Error(`cannot read the release tags (${err.message.split('\n')[0]})`)
  }
  return new Set(
    out
      .split('\n')
      .map((tag) => tag.trim().replace(/^v/, ''))
      .filter((version) => parse(version) !== null),
  )
}

/** Every `## ` heading outside code fences: its `[name]`, or the whole line. */
function sections(text) {
  const names = []
  let fenced = false
  for (const line of text.split('\n')) {
    if (line.trimStart().startsWith('```')) fenced = !fenced
    if (!fenced && line.startsWith('## ')) names.push(/^## \[([^\]]+)\]/.exec(line)?.[1] ?? line)
  }
  return names
}

/** The newest version that has a section in `text` and a release tag. */
function newestReleased(text, released, from) {
  const versions = sections(text)
    .filter((name) => released.has(name))
    .map(parse)
  if (versions.length === 0) {
    throw new Error(
      `${from} has no section with a vX.Y.Z tag (a shallow clone? check out with fetch-depth: 0)`,
    )
  }
  return versions.reduce((a, b) => (compare(a, b) >= 0 ? a : b))
}

function isReleased(name, newest) {
  const version = parse(name)
  return version !== null && compare(version, newest) <= 0
}

/** Drops every `## ` section whose version is not a released one. */
function releasedOnly(text, newest) {
  const kept = []
  let keep = true
  let fenced = false
  for (const line of text.split('\n')) {
    if (line.trimStart().startsWith('```')) fenced = !fenced
    if (!fenced && line.startsWith('## ')) {
      keep = isReleased(/^## \[([^\]]+)\]/.exec(line)?.[1] ?? '', newest)
    }
    if (keep) kept.push(line)
  }
  return kept.join('\n')
}

try {
  const released = tags()
  const check = process.argv.includes('--check')
  const problems = []
  for (const [from, to] of COPIES) {
    const text = readFileSync(join(SITE, from), 'utf8')
    const newest = newestReleased(text, released, from)
    if (check) {
      for (const name of sections(text).filter((name) => !isReleased(name, newest))) {
        problems.push(`${from}: section ${JSON.stringify(name)} has no release tag (newest v${newest.join('.')})`)
      }
      continue
    }
    const out = releasedOnly(text, newest)
    writeFileSync(join(SITE, to), out)
    const top = /^## \[([^\]]+)\]/m.exec(out)?.[1] ?? 'none'
    console.log(`${to}: released sections up to v${newest.join('.')}, newest shown ${top}`)
  }
  if (problems.length > 0) {
    for (const problem of problems) console.error(`::error::${problem}`)
    console.error('changelog: add a section to changelog/ only once its vX.Y.Z tag exists.')
    process.exit(1)
  }
  if (check) console.log('changelog: every section has a release tag.')
} catch (err) {
  console.error(`changelog: ${err.message}; refusing to publish unreleased sections.`)
  process.exit(1)
}
