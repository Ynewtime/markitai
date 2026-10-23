<script setup lang="ts">
/**
 * Feature grid for the home pages. Wraps the default theme's VPFeatures with
 * the shared bilingual data in `features.ts`, so `index.md` and `zh/index.md`
 * render identical markup from one source.
 */
import { computed } from 'vue'
import { useData } from 'vitepress'
import { VPFeatures } from 'vitepress/theme'
import { homeFeatures } from './features'

const { lang } = useData()

const features = computed(() =>
  homeFeatures.map((feature) => {
    const copy = lang.value.startsWith('zh') ? feature.zh : feature.en
    return { icon: feature.icon, title: copy.title, details: copy.details }
  }),
)
</script>

<template>
  <VPFeatures :features="features" />
</template>
