<script setup lang="ts">
/**
 * AnimatedIcon.vue
 *
 * A thin wrapper around any lucide-vue-next icon component that
 * applies the animation CSS classes from icon-animations.css.
 *
 * Usage:
 *   <AnimatedIcon :icon="RefreshCw" animation="spin" :active="isLoading" />
 *   <AnimatedIcon :icon="Radio" animation="glow-teal" :size="18" />
 *   <AnimatedIcon :icon="UploadCloud" animation="draw-in" />
 *
 * Props:
 *   icon        — Any lucide-vue-next icon component (required)
 *   animation   — One of the animation names below, or omit for no animation
 *   size        — Icon size in px, passed directly to lucide (default: 18)
 *   strokeWidth — Stroke width (default: 1.75 — slightly lighter than lucide's 2)
 *   active      — Adds .is-active modifier class on the icon
 *   class       — Any additional classes are forwarded to the icon element
 */

import type { Component } from 'vue'

export type IconAnimation =
  | 'draw-in'
  | 'pulse'
  | 'spin'
  | 'bounce'
  | 'fade-in'
  | 'glow-teal'
  | 'glow-amber'

const props = withDefaults(defineProps<{
  /** Any lucide-vue-next component reference */
  icon: Component
  /** Animation variant to apply */
  animation?: IconAnimation
  /** Icon size in pixels (forwarded to lucide) */
  size?: number
  /** Stroke width (forwarded to lucide) */
  strokeWidth?: number
  /** Adds .is-active modifier to amplify certain animations */
  active?: boolean
}>(), {
  animation: undefined,
  size: 18,
  strokeWidth: 1.75,
  active: false,
})

function animationClass(animation: IconAnimation | undefined): string {
  if (!animation) return ''
  return `icon-${animation}`
}
</script>

<template>
  <!--
    We use <component :is="icon"> so this wrapper is tree-shakable:
    only imported lucide icons end up in the bundle.
  -->
  <component
    :is="icon"
    :size="size"
    :stroke-width="strokeWidth"
    :class="[
      'animated-icon',
      animationClass(animation),
      { 'is-active': active },
    ]"
  />
</template>
