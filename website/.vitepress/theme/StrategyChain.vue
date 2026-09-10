<script setup lang="ts">
/**
 * The fetch-strategy fallback chain, drawn instead of described.
 *
 * Both guides used to spell the order out as an ASCII arrow line
 * (`Static → Playwright → Defuddle → Jina → Cloudflare`) and then spend a
 * paragraph explaining which of those five run on your machine and which
 * send the URL away.
 *
 * Consecutive strategies of the same kind are drawn as one labelled group, so
 * the privacy boundary is the gap between two cards rather than a hairline
 * nobody notices — and "on your machine" is said once per group instead of
 * once per pill. Both orders happen to group cleanly: the default chain is
 * local→remote, and the SPA chain is local→remote→local.
 *
 * Bilingual like `features.ts`: the strategy names are proper nouns and
 * identical in both locales, so only the group labels switch on `lang`. That
 * keeps the chain single-sourced rather than hand-copied into en + zh pages.
 */
import { computed } from 'vue'
import { useData } from 'vitepress'

const props = withDefaults(
  defineProps<{
    /** `default` = standard domains, `spa` = known JS-heavy domains. */
    mode?: 'default' | 'spa'
  }>(),
  { mode: 'default' },
)

const { lang } = useData()
const zh = computed(() => lang.value.startsWith('zh'))

/** Strategies that never send the URL off-machine. */
const LOCAL = new Set(['Static', 'Playwright'])

const ORDER = {
  default: ['Static', 'Playwright', 'Defuddle', 'Jina', 'Cloudflare'],
  spa: ['Playwright', 'Defuddle', 'Jina', 'Cloudflare', 'Static'],
} as const

const copy = computed(() =>
  zh.value
    ? { local: '本机', remote: '远程服务', order: '抓取策略回退顺序' }
    : {
        local: 'On your machine',
        remote: 'Remote services',
        order: 'Fetch strategy fallback order',
      },
)

/** Consecutive same-kind strategies collapse into one labelled card. */
const groups = computed(() => {
  const out: { local: boolean; steps: { name: string; step: number }[] }[] = []
  ORDER[props.mode].forEach((name, index) => {
    const local = LOCAL.has(name)
    const tail = out[out.length - 1]
    const step = { name, step: index + 1 }
    if (tail && tail.local === local) tail.steps.push(step)
    else out.push({ local, steps: [step] })
  })
  return out
})

const label = computed(() =>
  [
    copy.value.order,
    ...groups.value.map(
      (g) =>
        `${g.local ? copy.value.local : copy.value.remote}: ${g.steps
          .map((s) => `${s.step} ${s.name}`)
          .join(', ')}`,
    ),
  ].join('. '),
)
</script>

<template>
  <div class="mk-chain" role="img" :aria-label="label">
    <div class="mk-chain-track" aria-hidden="true">
      <template v-for="(group, index) in groups" :key="index">
        <span v-if="index" class="mk-chain-link" />
        <div class="mk-chain-group" :class="group.local ? 'is-local' : 'is-remote'">
          <span class="mk-chain-label">
            {{ group.local ? copy.local : copy.remote }}
          </span>
          <div class="mk-chain-steps">
            <span v-for="item in group.steps" :key="item.name" class="mk-chain-step">
              <span class="mk-chain-num">{{ item.step }}</span>
              <span class="mk-chain-name">{{ item.name }}</span>
            </span>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>
