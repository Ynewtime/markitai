// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import MarkitaiFeatures from './MarkitaiFeatures.vue'
import StrategyChain from './StrategyChain.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  // Registered globally so both locales can drop <StrategyChain /> straight
  // into a Markdown body. The component reads its own locale, so the fetch
  // order lives in one file instead of being hand-copied into en + zh pages.
  enhanceApp({ app }) {
    app.component('StrategyChain', StrategyChain)
  },
  // Home-page feature grid, shared by both locales (see MarkitaiFeatures.vue).
  // Rendered through the layout's home-features-before slot — right after the
  // hero — instead of inside the markdown body, because the body lives in
  // .vp-doc whose typography rules (list markers, heading borders) leak into
  // VPFeatures and break the card grid.
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      'home-features-before': () => h(MarkitaiFeatures),
    }),
} satisfies Theme
