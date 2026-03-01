<script setup lang="ts">
import { computed } from 'vue'
import { Cpu, SlidersHorizontal, Zap, Database } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'
import type { MetricsData } from '@/composables/useMetrics'

const props = defineProps<{
  metrics: MetricsData
}>()

// ─── Derived values ───────────────────────────────────────────────────────────

const passRatePct = computed(() =>
  (props.metrics.gatePassRate * 100).toFixed(1)
)

const vlmQueueFill = computed(() =>
  Math.min(100, (props.metrics.vlmQueueDepth / 16) * 100)
)

const decodeLatencyColor = computed(() => {
  const ms = props.metrics.decodeLatencyMs
  if (ms > 5) return '#ef4444'
  if (ms > 2) return '#f59e0b'
  return '#2dd4bf'
})

const vlmLatencyColor = computed(() => {
  const ms = props.metrics.vlmLatencyMs
  if (ms > 800) return '#ef4444'
  if (ms > 500) return '#f59e0b'
  return '#f59e0b'  // amber is the "data number" color for VLM
})

function fmt(n: number, decimals = 1): string {
  return n.toFixed(decimals)
}

function fmtInt(n: number): string {
  return Math.floor(n).toLocaleString()
}

function formatLatency(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`
  if (ms >= 1)    return `${ms.toFixed(1)}ms`
  return `${(ms * 1000).toFixed(0)}µs`
}
</script>

<template>
  <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">

    <!-- Card 1: Decode Stats -->
    <div class="card-skeuo p-5 flex flex-col gap-4">
      <div class="flex items-center gap-2">
        <div class="w-7 h-7 rounded-lg flex items-center justify-center"
             style="background: rgba(45,212,191,0.1); border: 1px solid rgba(45,212,191,0.2);">
          <AnimatedIcon :icon="Cpu" :size="14" :stroke-width="1.75"
                        animation="glow-teal" class="text-[#2dd4bf]" />
        </div>
        <h4 class="text-[#94a3b8] text-xs font-medium uppercase tracking-wider">Decode</h4>
        <span class="badge badge-teal ml-auto text-[9px]">GPU</span>
      </div>

      <!-- Primary metric -->
      <div>
        <div class="text-[#475569] text-[10px] uppercase tracking-wider mb-1">Throughput</div>
        <div class="mono text-3xl font-semibold" :style="{ color: decodeLatencyColor }">
          {{ fmt(metrics.decodeFps) }}
          <span class="text-sm font-normal text-[#475569]">fps</span>
        </div>
      </div>

      <!-- Secondary rows -->
      <div class="space-y-2 border-t border-[#1e2633] pt-3">
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Latency</span>
          <span class="mono text-xs" :style="{ color: decodeLatencyColor }">
            {{ formatLatency(metrics.decodeLatencyMs) }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Total frames</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.decodeFramesTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Sessions</span>
          <span class="mono text-xs"
                :class="metrics.activeSessions > 0 ? 'text-[#2dd4bf]' : 'text-[#475569]'">
            {{ metrics.activeSessions }}
          </span>
        </div>
      </div>

      <!-- Latency bar -->
      <div>
        <div class="flex justify-between text-[10px] text-[#475569] mb-1">
          <span>Latency budget</span>
          <span class="mono">{{ formatLatency(metrics.decodeLatencyMs) }} / 5ms</span>
        </div>
        <div class="h-1 rounded-full" style="background: #1e2633;">
          <div
            class="h-full rounded-full transition-all duration-500"
            :style="{
              width: `${Math.min(100, (metrics.decodeLatencyMs / 5) * 100)}%`,
              background: decodeLatencyColor,
              boxShadow: `0 0 8px ${decodeLatencyColor}66`,
            }"
          />
        </div>
      </div>
    </div>

    <!-- Card 2: Gate Engine -->
    <div class="card-skeuo-warm p-5 flex flex-col gap-4">
      <div class="flex items-center gap-2">
        <div class="w-7 h-7 rounded-lg flex items-center justify-center"
             style="background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.2);">
          <AnimatedIcon :icon="SlidersHorizontal" :size="14" :stroke-width="1.75"
                        animation="glow-amber" class="text-[#f59e0b]" />
        </div>
        <h4 class="text-[#94a3b8] text-xs font-medium uppercase tracking-wider">Gate Engine</h4>
      </div>

      <!-- Primary metric -->
      <div>
        <div class="text-[#475569] text-[10px] uppercase tracking-wider mb-1">Pass Rate</div>
        <div class="mono text-3xl font-semibold text-[#f59e0b]">
          {{ passRatePct }}
          <span class="text-sm font-normal text-[#475569]">%</span>
        </div>
      </div>

      <!-- Stats -->
      <div class="space-y-2 border-t border-[#2a2010] pt-3">
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Analyzed</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.gateFramesTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Keyframes</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.gateKeyframesTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Scene cuts</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.sceneCuts) }}</span>
        </div>
      </div>

      <!-- Pass rate bar -->
      <div>
        <div class="flex justify-between text-[10px] text-[#475569] mb-1">
          <span>Keyframe selectivity</span>
          <span class="mono text-[#f59e0b]">{{ passRatePct }}%</span>
        </div>
        <div class="h-1 rounded-full" style="background: #1e2633;">
          <div
            class="h-full rounded-full transition-all duration-500"
            :style="{
              width: `${Math.min(100, metrics.gatePassRate * 100)}%`,
              background: 'linear-gradient(90deg, #b45309, #f59e0b)',
              boxShadow: '0 0 8px rgba(245,158,11,0.4)',
            }"
          />
        </div>
      </div>
    </div>

    <!-- Card 3: VLM Inference -->
    <div class="card-skeuo p-5 flex flex-col gap-4">
      <div class="flex items-center gap-2">
        <div class="w-7 h-7 rounded-lg flex items-center justify-center"
             style="background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.2);">
          <AnimatedIcon :icon="Zap" :size="14" :stroke-width="1.75"
                        class="text-[#22c55e]" />
        </div>
        <h4 class="text-[#94a3b8] text-xs font-medium uppercase tracking-wider">VLM Inference</h4>
      </div>

      <!-- Primary metric -->
      <div>
        <div class="text-[#475569] text-[10px] uppercase tracking-wider mb-1">Avg Latency</div>
        <div class="mono text-3xl font-semibold" :style="{ color: vlmLatencyColor }">
          {{ Math.floor(metrics.vlmLatencyMs) }}
          <span class="text-sm font-normal text-[#475569]">ms</span>
        </div>
      </div>

      <!-- Stats -->
      <div class="space-y-2 border-t border-[#1e2633] pt-3">
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Total inferences</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.vlmInferencesTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Queue depth</span>
          <span class="mono text-xs"
                :class="metrics.vlmQueueDepth > 8 ? 'text-[#ef4444]' : metrics.vlmQueueDepth > 4 ? 'text-[#f59e0b]' : 'text-[#22c55e]'">
            {{ metrics.vlmQueueDepth }} / 16
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Workers</span>
          <span class="mono text-[#22c55e] text-xs">4 active</span>
        </div>
      </div>

      <!-- Queue depth bar -->
      <div>
        <div class="flex justify-between text-[10px] text-[#475569] mb-1">
          <span>Queue utilization</span>
          <span class="mono">{{ metrics.vlmQueueDepth }}/16</span>
        </div>
        <div class="h-1 rounded-full" style="background: #1e2633;">
          <div
            class="h-full rounded-full transition-all duration-500"
            :style="{
              width: `${vlmQueueFill}%`,
              background: vlmQueueFill > 75 ? '#ef4444' : vlmQueueFill > 50 ? '#f59e0b' : '#22c55e',
              boxShadow: `0 0 8px ${vlmQueueFill > 75 ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.3)'}`,
            }"
          />
        </div>
      </div>
    </div>

    <!-- Card 4: SpacetimeDB -->
    <div class="card-skeuo p-5 flex flex-col gap-4">
      <div class="flex items-center gap-2">
        <div class="w-7 h-7 rounded-lg flex items-center justify-center"
             style="background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.15);">
          <AnimatedIcon :icon="Database" :size="14" :stroke-width="1.75"
                        class="text-[#2dd4bf]" />
        </div>
        <h4 class="text-[#94a3b8] text-xs font-medium uppercase tracking-wider">SpacetimeDB</h4>
        <span class="badge badge-teal ml-auto text-[9px]">LIVE</span>
      </div>

      <!-- Primary metric -->
      <div>
        <div class="text-[#475569] text-[10px] uppercase tracking-wider mb-1">Emit Rate</div>
        <div class="mono text-3xl font-semibold text-[#2dd4bf]">
          {{ fmt(metrics.spacetimeEmitRate) }}
          <span class="text-sm font-normal text-[#475569]">evt/s</span>
        </div>
      </div>

      <!-- Stats -->
      <div class="space-y-2 border-t border-[#1e2633] pt-3">
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Total events</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.spacetimeEventsTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Connection</span>
          <span class="mono text-[#2dd4bf] text-xs">STABLE</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Table</span>
          <span class="mono text-[#475569] text-xs">agent_events</span>
        </div>
      </div>

      <!-- Emit rate bar -->
      <div>
        <div class="flex justify-between text-[10px] text-[#475569] mb-1">
          <span>Event throughput</span>
          <span class="mono text-[#2dd4bf]">{{ fmt(metrics.spacetimeEmitRate) }}/s</span>
        </div>
        <div class="h-1 rounded-full" style="background: #1e2633;">
          <div
            class="h-full rounded-full transition-all duration-500 progress-bar-active"
            :style="{
              width: `${Math.min(100, (metrics.spacetimeEmitRate / 10) * 100)}%`,
              background: 'linear-gradient(90deg, #0d9488, #2dd4bf)',
              boxShadow: '0 0 8px rgba(45,212,191,0.35)',
            }"
          />
        </div>
      </div>
    </div>

  </div>
</template>
