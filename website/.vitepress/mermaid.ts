/**
 * ```mermaid fences, drawn at build time with Pele (https://pele.run, MIT).
 *
 * Pele is synchronous and needs no DOM, so the diagram is SVG in the page's
 * HTML: no renderer is shipped to the browser, nothing moves when the page
 * loads, and a page without a diagram carries nothing for it. Colours are
 * Pele's CSS variables, which custom.css maps onto the theme's, so one
 * drawing follows light and dark mode.
 *
 * A diagram wider than a phone's column is drawn twice: the layout for the
 * desktop column and Pele's narrow (usually top-to-bottom) layout, and CSS
 * shows one of them. Shrinking a left-to-right flowchart onto a 390 px screen
 * would make its labels unreadable.
 *
 * A diagram Pele cannot draw fails the build with the page and line, rather
 * than shipping a broken figure.
 */
import { PeleError, render } from 'pele'
import type MarkdownIt from 'markdown-it'

/** The doc column beside the sidebar and outline from 960 to 1439 px (688 px from 1440). */
const WIDE = 624
/** A 390 px phone minus the doc's 24 px side padding. */
const NARROW = 342
/** Below this scale a wide drawing's labels get too small: draw it upright. */
const MIN_SCALE = 0.85
const FONT_SIZE = 14
/** The smallest label size on a phone: a wider drawing scrolls sideways instead. */
const MIN_PHONE_FONT = 11
/** The fallback in the SVG; custom.css sets the theme's font through `--pele-font`. */
const FONT_FAMILY = 'Inter, ui-sans-serif, system-ui, sans-serif'

function draw(source: string, maxWidth: number, idPrefix: string, upright: boolean) {
  return render(source, {
    fontFamily: FONT_FAMILY,
    fontSize: FONT_SIZE,
    maxWidth,
    idPrefix,
    // The direction is chosen here (fit), not by Pele's own breakpoint.
    ...(upright ? { directionBreakpoint: maxWidth + 1 } : { autoDirection: false }),
    images: false,
    links: true,
    linkSchemes: ['http', 'https'],
    linkRel: 'noopener noreferrer',
  })
}

/** The best drawing for a column: as written, or upright when that is much larger. */
function fit(source: string, maxWidth: number, idPrefix: string) {
  const drawn = draw(source, maxWidth, idPrefix, false)
  if (drawn.width <= maxWidth / MIN_SCALE) return drawn
  const upright = draw(source, maxWidth, idPrefix, true)
  return upright.width < drawn.width ? upright : drawn
}

/** The phone's drawing: whichever layout keeps the labels largest. */
function phoneFit(source: string, idPrefix: string) {
  const scale = (width: number) => Math.min(1, NARROW / width)
  const drawn = draw(source, NARROW, idPrefix, false)
  const upright = draw(source, NARROW, idPrefix, true)
  return scale(upright.width) > scale(drawn.width) ? upright : drawn
}

/**
 * Markup for Vue's `v-html`. Written into the template as elements, a drawing
 * is compiled into the page's render function and its hydration loses the
 * figure's second drawing; as `v-html` the server's markup is kept as it is.
 */
function html(markup: string): string {
  const js = markup.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n')
  return `v-html="'${js.replace(/&/g, '&amp;').replace(/"/g, '&quot;')}'"`
}

/** A short, stable id for a page, so drawings on one page never share ids. */
function pageKey(path: string): string {
  let hash = 0
  for (const char of path) hash = (hash * 31 + char.charCodeAt(0)) | 0
  return (hash >>> 0).toString(36)
}

export function mermaid(md: MarkdownIt): void {
  const fence = md.renderer.rules.fence!
  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    if (token.info.trim().split(/\s+/)[0] !== 'mermaid') {
      return fence(tokens, idx, options, env, self)
    }
    const path: string = env?.relativePath ?? env?.path ?? ''
    const id = `mk${pageKey(path)}-${idx}`
    try {
      const wide = fit(token.content, WIDE, `${id}w-`)
      // One drawing is enough when it already fits a phone at a readable size.
      if (wide.width <= NARROW / MIN_SCALE) {
        return `<figure class="mk-diagram" ${html(wide.svg)}></figure>\n`
      }
      const narrow = phoneFit(token.content, `${id}n-`)
      // Shrunk to the column its labels would drop below MIN_PHONE_FONT: it
      // keeps that size and its box scrolls sideways (custom.css).
      const least = Math.ceil((narrow.width * MIN_PHONE_FONT) / FONT_SIZE)
      const scroll = least > NARROW ? ` style="--mk-diagram-min:${least}px"` : ''
      return (
        `<figure class="mk-diagram has-narrow">` +
        `<div class="mk-diagram-wide" ${html(wide.svg)}></div>` +
        `<div class="mk-diagram-narrow"${scroll} ${html(narrow.svg)}></div>` +
        `</figure>\n`
      )
    } catch (error) {
      const where = error instanceof PeleError ? ` (diagram line ${error.line}: ${error.code})` : ''
      const line = token.map ? token.map[0] + 1 : 0
      throw new Error(`${path}:${line}: mermaid diagram could not be drawn${where}: ${error}`)
    }
  }
}
