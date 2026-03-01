<script setup lang="ts">
/**
 * ProcessingVisualization
 *
 * Real-time stream processing visualization.
 * Renders a filmstrip of frame bars that fill left-to-right as frames are
 * decoded, a sliding processing-window box, a live-cursor triangle, and a
 * time axis with second markers.
 *
 * Usage:
 *   <ProcessingVisualization
 *     :total-frames="600"
 *     :processed-frames="processedFrames"
 *     :keyframe-count="keyframes.length"
 *     :event-count="events.length"
 *     :current-chunk-start="chunkStart"
 *     :current-chunk-end="chunkEnd"
 *     :fps="5"
 *     :is-processing="true"
 *   />
 */

import { computed, ref, watch, onMounted, onUnmounted } from 'vue'

// ── Props ──────────────────────────────────────────────────────────────────────

interface Props {
  totalFrames: number        // total frames in video
  processedFrames: number    // frames processed so far
  keyframeCount: number      // keyframes selected
  eventCount: number         // events emitted
  currentChunkStart: number  // start frame of current processing window
  currentChunkEnd: number    // end frame of current processing window
  fps: number               // video fps
  isProcessing: boolean
  /** Frame indices that were selected for VLM inference (amber bars) */
  vlmFrames?: number[]
  /** Frame indices that are keyframes (taller teal bars) */
  keyframeIndices?: number[]
}

const props = withDefaults(defineProps<Props>(), {
  vlmFrames: () => [],
  keyframeIndices: () => [],
})

// ── Constants ─────────────────────────────────────────────────────────────────

const BAR_W = 2        // px per bar
const BAR_GAP = 1      // px gap between bars
const BAR_STRIDE = BAR_W + BAR_GAP   // 3 px total per frame slot

const COLOR_PENDING   = '#1e2633'
const COLOR_PROCESSED = '#2dd4bf'
const COLOR_VLM       = '#f59e0b'

// ── Strip geometry ────────────────────────────────────────────────────────────

/** The effective total frames we visualize — at least 1 to avoid division-by-zero */
const effectiveTotal = computed(() => Math.max(props.totalFrames, 1))

/** Strip total width in px */
const stripWidthPx = computed(() => effectiveTotal.value * BAR_STRIDE)

/** Convert a frame index to a pixel offset (left edge of that bar) */
function frameToX(frame: number): number {
  return frame * BAR_STRIDE
}

/** Pixel left of the processing window */
const windowLeft = computed(() => frameToX(props.currentChunkStart))

/** Pixel width of the processing window */
const windowWidth = computed(() => {
  const span = Math.max(props.currentChunkEnd - props.currentChunkStart, 1)
  return span * BAR_STRIDE
})

/** Pixel left of the live cursor (right edge of last processed frame) */
const cursorLeft = computed(() => frameToX(props.processedFrames))

// ── Time axis ─────────────────────────────────────────────────────────────────

/** Second markers for the time axis */
const secondMarkers = computed<{ label: string; x: number }[]>(() => {
  const fpsSafe = Math.max(props.fps, 1)
  const durationSec = Math.ceil(effectiveTotal.value / fpsSafe)
  const markers: { label: string; x: number }[] = []
  for (let s = 0; s <= durationSec; s++) {
    const frame = s * fpsSafe
    if (frame <= effectiveTotal.value) {
      markers.push({ label: `${s}s`, x: frameToX(frame) })
    }
  }
  return markers
})

// ── Auto-scroll the strip to keep the cursor visible ─────────────────────────

const scrollEl = ref<HTMLDivElement | null>(null)

watch(
  () => props.processedFrames,
  () => {
    if (!scrollEl.value) return
    const el = scrollEl.value
    const cursor = cursorLeft.value
    const viewLeft = el.scrollLeft
    const viewRight = viewLeft + el.clientWidth
    // Scroll only if cursor is outside the right 80 % of the visible area
    if (cursor > viewRight - el.clientWidth * 0.2 || cursor < viewLeft) {
      el.scrollTo({
        left: Math.max(0, cursor - el.clientWidth * 0.5),
        behavior: 'smooth',
      })
    }
  },
)

// ── Animation tick for subtle shimmer on the active window ────────────────────

const shimmerOffset = ref(0)
let rafId: number | null = null

function tick() {
  shimmerOffset.value = (shimmerOffset.value + 0.5) % 100
  if (props.isProcessing) rafId = requestAnimationFrame(tick)
}

watch(
  () => props.isProcessing,
  (active) => {
    if (active && rafId === null) rafId = requestAnimationFrame(tick)
    else if (!active && rafId !== null) {
      cancelAnimationFrame(rafId)
      rafId = null
    }
  },
  { immediate: true },
)

onUnmounted(() => {
  if (rafId !== null) cancelAnimationFrame(rafId)
})

// ── Bar color logic ───────────────────────────────────────────────────────────

const vlmSet = computed(() => new Set(props.vlmFrames ?? []))
const kfSet  = computed(() => new Set(props.keyframeIndices ?? []))

function barColor(index: number): string {
  if (vlmSet.value.has(index)) return COLOR_VLM
  if (index < props.processedFrames) return COLOR_PROCESSED
  return COLOR_PENDING
}

function barHeight(index: number): number {
  return kfSet.value.has(index) ? 32 : 24
}

function barGlow(index: number): string {
  if (vlmSet.value.has(index)) return `0 0 4px ${COLOR_VLM}88`
  if (index < props.processedFrames) return `0 0 3px ${COLOR_PROCESSED}55`
  return 'none'
}

// ── Stats ─────────────────────────────────────────────────────────────────────

const processPct = computed(() =>
  Math.min(100, Math.round((props.processedFrames / effectiveTotal.value) * 100))
)

const durationLabel = computed(() => {
  const sec = effectiveTotal.value / Math.max(props.fps, 1)
  if (sec < 60) return `${sec.toFixed(1)}s`
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${String(s).padStart(2, '0')}`
})
</script>

<template>
  <div
    class="rounded-[12px] overflow-hidden"
    style="
      background: linear-gradient(145deg, #0a0c12 0%, #0f1117 100%);
      border: 1px solid #1e2633;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 4px 12px rgba(0,0,0,0.5);
    "
    role="region"
    aria-label="Stream processing visualization"
  >
    <!-- Header bar ──────────────────────────────────────────────────────── -->
    <div
      class="flex items-center justify-between px-4 py-3"
      style="border-bottom: 1px solid #1e2633;"
    >
      <!-- Left: label + live indicator -->
      <div class="flex items-center gap-2.5">
        <div
          v-if="isProcessing"
          class="pv-live-dot"
          aria-hidden="true"
        />
        <span
          class="text-xs font-semibold uppercase tracking-widest"
          :style="isProcessing ? 'color: #2dd4bf;' : 'color: #475569;'"
        >
          {{ isProcessing ? 'Processing' : 'Stream' }}
        </span>
        <span class="mono text-[10px] text-[#475569]">
          {{ processPct }}%
        </span>
      </div>

      <!-- Right: stats chips -->
      <div class="flex items-center gap-3">
        <span class="mono text-[11px]" style="color: #2dd4bf;" :title="`${processedFrames} frames decoded`">
          {{ processedFrames.toLocaleString() }}
          <span class="text-[#475569] font-normal"> frames</span>
        </span>
        <span class="mono text-[11px]" style="color: #f59e0b;" :title="`${keyframeCount} keyframes`">
          {{ keyframeCount }}
          <span class="text-[#475569] font-normal"> kf</span>
        </span>
        <span class="mono text-[11px]" style="color: #94a3b8;" :title="`${eventCount} events`">
          {{ eventCount }}
          <span class="text-[#475569] font-normal"> ev</span>
        </span>
        <span class="mono text-[10px] text-[#334155]">{{ durationLabel }}</span>
      </div>
    </div>

    <!-- Strip scroll container ───────────────────────────────────────────── -->
    <div
      ref="scrollEl"
      class="pv-scroll"
      aria-hidden="true"
    >
      <!-- Inner strip — actual pixel-exact width -->
      <div
        class="relative"
        :style="`width: ${stripWidthPx}px; height: 72px; min-width: 100%;`"
      >

        <!-- Processing window (sliding teal box) ─────────────────────────── -->
        <div
          v-if="isProcessing && currentChunkEnd > currentChunkStart"
          class="pv-window"
          :style="`
            left: ${windowLeft}px;
            width: ${windowWidth}px;
          `"
        >
          <!-- "PROCESSING" label above the window -->
          <span class="pv-window-label">PROCESSING</span>
          <!-- Shimmer streak inside window -->
          <div
            class="pv-window-shimmer"
            :style="`left: ${shimmerOffset}%;`"
          />
        </div>

        <!-- Frame bars ──────────────────────────────────────────────────── -->
        <div class="pv-bar-row" aria-hidden="true">
          <div
            v-for="i in effectiveTotal"
            :key="i - 1"
            class="pv-bar"
            :style="`
              background: ${barColor(i - 1)};
              height: ${barHeight(i - 1)}px;
              box-shadow: ${barGlow(i - 1)};
            `"
            :title="`Frame ${i - 1}`"
          />
        </div>

        <!-- Live cursor (teal triangle) ─────────────────────────────────── -->
        <div
          v-if="isProcessing || processedFrames > 0"
          class="pv-cursor"
          :style="`left: ${cursorLeft}px;`"
          aria-label="Current processing position"
        >
          <!-- Vertical line -->
          <div class="pv-cursor-line" />
          <!-- Triangle head pointing down -->
          <div class="pv-cursor-head" />
        </div>

      </div><!-- /inner strip -->
    </div><!-- /scroll container -->

    <!-- Time axis ───────────────────────────────────────────────────────── -->
    <div
      class="pv-time-axis"
      aria-hidden="true"
    >
      <div
        class="relative"
        :style="`width: ${stripWidthPx}px; height: 20px; min-width: 100%;`"
      >
        <span
          v-for="m in secondMarkers"
          :key="m.label"
          class="pv-time-label"
          :style="`left: ${m.x}px;`"
        >
          {{ m.label }}
        </span>
      </div>
    </div>

    <!-- Progress rail ───────────────────────────────────────────────────── -->
    <div
      class="pv-progress-rail"
      role="progressbar"
      :aria-valuenow="processPct"
      :aria-valuemin="0"
      :aria-valuemax="100"
      :aria-label="`Processing ${processPct}% complete`"
    >
      <div
        class="pv-progress-fill"
        :class="isProcessing ? 'pv-progress-fill--active' : ''"
        :style="`width: ${processPct}%;`"
      />
    </div>

  </div>
</template>

<style scoped>
/* ── Live dot ──────────────────────────────────────────────────────────────── */
.pv-live-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #2dd4bf;
  flex-shrink: 0;
  animation: pv-pulse 2s ease-in-out infinite;
}

@keyframes pv-pulse {
  0%, 100% {
    opacity: 1;
    box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.5);
  }
  50% {
    opacity: 0.7;
    box-shadow: 0 0 0 5px rgba(45, 212, 191, 0);
  }
}

/* ── Scroll container ──────────────────────────────────────────────────────── */
.pv-scroll {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 12px 16px 0;
  /* Hide scrollbar visually but keep it functional */
  scrollbar-width: thin;
  scrollbar-color: #1e2633 transparent;
}
.pv-scroll::-webkit-scrollbar {
  height: 4px;
}
.pv-scroll::-webkit-scrollbar-track {
  background: transparent;
}
.pv-scroll::-webkit-scrollbar-thumb {
  background: #1e2633;
  border-radius: 2px;
}

/* ── Bar row ───────────────────────────────────────────────────────────────── */
.pv-bar-row {
  position: absolute;
  bottom: 8px;
  left: 0;
  display: flex;
  align-items: flex-end;
  gap: 1px;
}

.pv-bar {
  width: 2px;
  border-radius: 1px 1px 0 0;
  flex-shrink: 0;
  transition: background 120ms ease, height 120ms ease, box-shadow 120ms ease;
}

/* ── Processing window ─────────────────────────────────────────────────────── */
.pv-window {
  position: absolute;
  bottom: 0;
  height: 48px;
  border: 1px solid rgba(45, 212, 191, 0.5);
  border-radius: 3px;
  background: rgba(45, 212, 191, 0.06);
  box-shadow:
    0 0 12px rgba(45, 212, 191, 0.12),
    inset 0 1px 0 rgba(45, 212, 191, 0.08);
  overflow: hidden;
  pointer-events: none;
  transition: left 400ms cubic-bezier(0.4, 0, 0.2, 1),
              width 300ms cubic-bezier(0.4, 0, 0.2, 1);
}

.pv-window-label {
  position: absolute;
  top: -18px;
  left: 50%;
  transform: translateX(-50%);
  font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.12em;
  color: #2dd4bf;
  white-space: nowrap;
  text-shadow: 0 0 8px rgba(45, 212, 191, 0.5);
}

.pv-window-shimmer {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 12px;
  background: linear-gradient(90deg, transparent 0%, rgba(45, 212, 191, 0.25) 50%, transparent 100%);
  pointer-events: none;
  transform: translateX(-50%);
}

/* ── Live cursor ───────────────────────────────────────────────────────────── */
.pv-cursor {
  position: absolute;
  top: 4px;
  bottom: 0;
  width: 1px;
  pointer-events: none;
  transition: left 300ms cubic-bezier(0.4, 0, 0.2, 1);
}

.pv-cursor-line {
  position: absolute;
  top: 8px;
  bottom: 0;
  left: 0;
  width: 1px;
  background: #2dd4bf;
  box-shadow: 0 0 6px #2dd4bf, 0 0 12px rgba(45, 212, 191, 0.4);
}

/* Downward-pointing triangle at the top of the cursor */
.pv-cursor-head {
  position: absolute;
  top: 2px;
  left: 50%;
  transform: translateX(-50%);
  width: 0;
  height: 0;
  border-left: 4px solid transparent;
  border-right: 4px solid transparent;
  border-top: 6px solid #2dd4bf;
  filter: drop-shadow(0 0 4px #2dd4bf);
}

/* ── Time axis ─────────────────────────────────────────────────────────────── */
.pv-time-axis {
  overflow: hidden;
  padding: 2px 16px 0;
  border-top: 1px solid #131820;
}

.pv-time-label {
  position: absolute;
  top: 3px;
  font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
  font-size: 9px;
  color: #334155;
  letter-spacing: 0.04em;
  transform: translateX(-50%);
  white-space: nowrap;
  user-select: none;
}

/* ── Progress rail ─────────────────────────────────────────────────────────── */
.pv-progress-rail {
  height: 2px;
  background: #0f1117;
  overflow: hidden;
}

.pv-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #0d9488 0%, #2dd4bf 100%);
  transition: width 400ms cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 0 1px 1px 0;
}

.pv-progress-fill--active {
  animation: pv-progress-shimmer 2s ease-in-out infinite;
}

@keyframes pv-progress-shimmer {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.75; }
}
</style>
