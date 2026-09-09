// https://vitepress.dev/guide/custom-theme
import { defineComponent, h, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vitepress'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './custom.css'

// The home hero fills the first screen down to the fold, and its height
// depends on the feature block's content-driven height, which CSS cannot
// know. Publish that height as --vp-features-h (see custom.css); the static
// fallback in CSS only covers the paint before this runs.
function useFeaturesHeight() {
  const root = document.documentElement
  const route = useRoute()
  let observer: ResizeObserver | undefined
  let observed: HTMLElement | undefined

  const publish = (el: HTMLElement) => {
    const value = `${el.offsetHeight}px`
    if (root.style.getPropertyValue('--vp-features-h') !== value) {
      root.style.setProperty('--vp-features-h', value)
    }
  }

  const sync = () => {
    const el = document.querySelector<HTMLElement>('.VPHome > .VPFeatures')
    if (el) {
      if (!observer) {
        // The callback receives resize entries, not the element itself.
        observer = new ResizeObserver((entries) => {
          for (const entry of entries) publish(entry.target as HTMLElement)
        })
      }
      // SPA navigation and hydration can replace the node; keep watching the
      // live one. Observing the element (not the body) also catches height
      // changes that don't move the overall page, like web fonts swapping in.
      if (observed !== el) {
        observer.disconnect()
        observer.observe(el)
        observed = el
      }
      publish(el)
    } else {
      observer?.disconnect()
      observer = undefined
      observed = undefined
      root.style.removeProperty('--vp-features-h')
    }
  }

  const onResize = () => {
    const el = document.querySelector<HTMLElement>('.VPHome > .VPFeatures')
    if (el) publish(el)
  }

  onMounted(() => {
    // flush: post puts the callback after the new page's DOM is mounted, so
    // SPA navigations home -> guide -> home re-attach cleanly. The resize
    // listener is a belt-and-braces path for viewport-driven height changes.
    watch(() => route.path, sync, { immediate: true, flush: 'post' })
    window.addEventListener('resize', onResize)
  })

  onUnmounted(() => {
    window.removeEventListener('resize', onResize)
    observer?.disconnect()
  })
}

export default {
  extends: DefaultTheme,
  Layout: defineComponent({
    setup() {
      if (typeof document !== 'undefined') {
        useFeaturesHeight()
      }
      return () => h(DefaultTheme.Layout)
    },
  }),
  enhanceApp() {
    // ...
  },
} satisfies Theme
