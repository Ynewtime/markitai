// Copies CHANGELOG.md / CHANGELOG.zh.md into the site with released versions only.
//
// The changelog pages change only with a release: a `## [X.Y.Z]` section is
// published once its vX.Y.Z tag exists, so an unreleased section on main never
// shows. Old versions (0.6–0.14) were never tagged, so the cut is "every section
// up to the newest tagged version"; anything above it, or a heading that is not
// a version, stays off the site. Without readable tags the build fails rather
// than publish a section nobody has released.
//
// Usage: bun scripts/changelog.mjs   (docs:build and docs:dev run it)

import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const SITE = join(dirname(fileURLToPath(import.meta.url)), '..')

const COPIES = [
  ['../CHANGELOG.md', 'changelog.md'],
  ['../CHANGELOG.zh.md', 'zh/changelog.md'],
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

function newestTag() {
  let out
  try {
    out = execFileSync('git', ['tag', '--list', 'v*'], { cwd: SITE, encoding: 'utf8' })
  } catch (err) {
    throw new Error(`cannot read the release tags (${err.message.split('\n')[0]})`)
  }
  const versions = out
    .split('\n')
    .map((tag) => parse(tag.trim().replace(/^v/, '')))
    .filter(Boolean)
  if (versions.length === 0) {
    throw new Error('no vX.Y.Z tag found (a shallow clone? check out with fetch-depth: 0)')
  }
  return versions.reduce((a, b) => (compare(a, b) >= 0 ? a : b))
}

/** Drops every `## ` section whose version is not a released one. */
function releasedOnly(text, newest) {
  const kept = []
  let keep = true
  let fenced = false
  for (const line of text.split('\n')) {
    if (line.trimStart().startsWith('```')) fenced = !fenced
    if (!fenced && line.startsWith('## ')) {
      const version = parse(/^## \[([^\]]+)\]/.exec(line)?.[1] ?? '')
      keep = version !== null && compare(version, newest) <= 0
    }
    if (keep) kept.push(line)
  }
  return kept.join('\n')
}

try {
  const newest = newestTag()
  for (const [from, to] of COPIES) {
    const out = releasedOnly(readFileSync(join(SITE, from), 'utf8'), newest)
    writeFileSync(join(SITE, to), out)
    const top = /^## \[([^\]]+)\]/m.exec(out)?.[1] ?? 'none'
    console.log(`${to}: released sections up to v${newest.join('.')}, newest shown ${top}`)
  }
} catch (err) {
  console.error(`changelog: ${err.message}; refusing to publish unreleased sections.`)
  process.exit(1)
}
