// Builds /changelog and /zh/changelog from the release notes, released versions only.
//
// Each page is the 2.x notes of the repository's CHANGELOG(.zh).md followed by
// the Python 1.x notes in changelog/python-1.x(.zh).md, newest first, under the
// root file's intro and one heading that marks where 1.x starts.
//
// The changelog pages change only with a release: a `## [X.Y.Z]` section is
// published once its vX.Y.Z tag exists, so an unreleased section never shows.
// Old versions (0.6–0.14) were never tagged, so the cut is "every section up to
// the newest version that has both a section in that file and a tag". Tags of
// the other release line have no section in a file and so never move its cut.
// Anything above it, or a heading that is not a version, stays off the site.
// Without readable tags the build fails rather than publish a section nobody
// has released.
//
// The root files gain a section in the release commit, before its tag exists,
// so --check lets an unreleased section stand at their top (it stays off the
// site until the tag). The 1.x files only change with corrections, so there any
// untagged section fails. Both check that English and Chinese publish the same
// versions.
//
// Usage: bun scripts/changelog.mjs           (docs:build and docs:dev run it)
//        bun scripts/changelog.mjs --check   (fails if a section would be lost)

import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const SITE = join(dirname(fileURLToPath(import.meta.url)), '..')

// The anchor of the heading where the 1.x notes start (a link target, keep it).
const ANCHOR_1X = 'python-1x'

const PAGES = [
  {
    to: 'changelog.md',
    current: '../CHANGELOG.md',
    python: 'changelog/python-1.x.md',
    heading: `## Markitai 1.x (Python) {#${ANCHOR_1X}}`,
    lead: 'Python markitai, which 2.0.0 replaced.',
    pointer: '1.x part below',
  },
  {
    to: 'zh/changelog.md',
    current: '../CHANGELOG.zh.md',
    python: 'changelog/python-1.x.zh.md',
    heading: `## Markitai 1.x（Python 版） {#${ANCHOR_1X}}`,
    lead: 'Python 版 markitai，已由 2.0.0 取代。',
    pointer: '下方 1.x 部分',
  },
]

// The root intro links the 1.x notes on GitHub; on the site they are in-page.
const LINK_1X = /\[[^\]]*\]\(https:\/\/github\.com\/Ynewtime\/markitai\/blob\/v1\.[^)]*\/CHANGELOG(?:\.zh)?\.md\)/g

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

/** The text before the first `## ` heading, and each `## ` section as { name, text }. */
function split(text) {
  const parts = { intro: [], sections: [] }
  let lines = parts.intro
  let fenced = false
  for (const line of text.split('\n')) {
    if (line.trimStart().startsWith('```')) fenced = !fenced
    if (!fenced && line.startsWith('## ')) {
      lines = []
      parts.sections.push({ name: /^## \[([^\]]+)\]/.exec(line)?.[1] ?? line, lines })
    }
    lines.push(line)
  }
  for (const section of parts.sections) section.text = section.lines.join('\n').trimEnd()
  parts.intro = parts.intro.join('\n').trimEnd()
  return parts
}

/** The newest version that has a section and a release tag. */
function newestReleased(sections, released, from) {
  const versions = sections
    .map((section) => section.name)
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

/** A release-notes file: its intro, published sections, and what --check reports. */
function notes(from, released, { pendingAtTop }) {
  const { intro, sections } = split(readFileSync(join(SITE, from), 'utf8'))
  const newest = newestReleased(sections, released, from)
  const published = sections.filter((section) => isReleased(section.name, newest))
  const problems = []
  const pending = []
  let seenPublished = false
  for (const section of sections) {
    if (isReleased(section.name, newest)) {
      seenPublished = true
    } else if (pendingAtTop && !seenPublished) {
      pending.push(section.name)
    } else {
      problems.push(
        `${from}: section ${JSON.stringify(section.name)} has no release tag (newest v${newest.join('.')})`,
      )
    }
  }
  return { from, intro, published, newest, problems, pending }
}

function versions(file) {
  return file.published.map((section) => section.name).join(', ')
}

try {
  const released = tags()
  const check = process.argv.includes('--check')
  const problems = []
  const files = []
  for (const page of PAGES) {
    const current = notes(page.current, released, { pendingAtTop: true })
    const python = notes(page.python, released, { pendingAtTop: false })
    files.push([current, python])
    problems.push(...current.problems, ...python.problems)
    for (const name of current.pending) {
      console.log(`${page.current}: ${JSON.stringify(name)} stays off the site until its tag exists`)
    }
    if (check) continue
    const intro = current.intro.replace(LINK_1X, `[${page.pointer}](#${ANCHOR_1X})`)
    const body = [
      intro,
      ...current.published.map((section) => section.text),
      page.heading,
      page.lead,
      ...python.published.map((section) => section.text),
    ]
    writeFileSync(join(SITE, page.to), `${body.join('\n\n')}\n`)
    console.log(
      `${page.to}: ${current.published[0].name} … ${python.published.at(-1).name} ` +
        `(released up to v${current.newest.join('.')} and v${python.newest.join('.')})`,
    )
  }
  // English and Chinese publish the same versions, file by file.
  for (let i = 0; i < 2; i++) {
    const [en, zh] = [files[0][i], files[1][i]]
    if (versions(en) !== versions(zh)) {
      problems.push(
        `${en.from} publishes [${versions(en)}] but ${zh.from} publishes [${versions(zh)}]`,
      )
    }
  }
  // The build leaves such sections off the site; only --check fails on them.
  if (check && problems.length > 0) {
    for (const problem of problems) console.error(`::error::${problem}`)
    console.error(
      'changelog: a section is published once its vX.Y.Z tag exists, in both languages; ' +
        'only the top of CHANGELOG*.md may hold a section still waiting for its tag.',
    )
    process.exit(1)
  }
  if (check) console.log('changelog: every published section has a release tag, in both languages.')
} catch (err) {
  console.error(`changelog: ${err.message}; refusing to publish unreleased sections.`)
  process.exit(1)
}
