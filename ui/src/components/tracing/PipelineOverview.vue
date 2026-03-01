<script setup lang="ts">
import { computed, type Component } from 'vue'
import { Radio, Cpu, SlidersHorizontal, Zap, Database } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'
import type { PipelineStageMetrics, HistogramPercentiles } from '@/composables/useMetrics'

const props = defineProps<{
  stages: PipelineStageMetrics[]
}>()

// ─── Stage config ─────────────────────────────────────────────────────────────

interface StageConfig {
  icon: Component
}

const STAGE_CONFIG: Record<string, StageConfig> = {
  WebRTC:      { icon: Radio },
  Decode:      { icon: Cpu },
  Gate:        { icon: SlidersHorizontal },
  VLM:         { icon: Zap },
  SpacetimeDB: { icon: Database },
}

// ─── Status helpers ───────────────────────────────────────────────────────────

function statusColor(status: PipelineStageMetrics['status']): string {
  switch (status) {
    case 'healthy': return '#2dd4bf'
    case 'slow':    return '#f59e0b'
    case 'error':   return '#ef4444'
    case 'idle':    return '#475569'
  }
}

function statusGlow(status: PipelineStageMetrics['status']): string {
  switch (status) {
    case 'healthy': return '0 0 16px rgba(45,212,191,0.25)'
    case 'slow':    return '0 0 16px rgba(245,158,11,0.25)'
    case 'error':   return '0 0 16px rgba(239,68,68,0.25)'
    case 'idle':    return 'none'
  }
}

function iconAnimation(status: PipelineStageMetrics['status']) {
  if (status === 'healthy') return 'glow-teal'
  if (status === 'slow')    return 'glow-amber'
  return undefined
}

const connectorActive = computed(() =>
  props.stages.some(s => s.status === 'healthy' || s.status === 'slow')
)

function formatFps(fps: number): string {
  return fps.toFixed(1)
}

function formatLatency(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`
  if (ms >= 1)    return `${ms.toFixed(0)}ms`
  return `${(ms * 1000).toFixed(0)}µs`
}

/**
 * Return the best latency value to display for a stage card.
 * Prefers p50 from real histogram data; falls back to the mean/estimate.
 */
function displayLatency(stage: PipelineStageMetrics): string {
  if (stage.percentiles) {
    return formatLatency(stage.percentiles.p50)
  }
  return formatLatency(stage.latencyMs)
}

/**
 * Return a compact p95 label for the stage card subtitle when histogram
 * data is present (e.g. "p95 2.1ms").
 */
function p95Label(percentiles: HistogramPercentiles | undefined): string | null {
  if (!percentiles) return null
  return `p95 ${formatLatency(percentiles.p95)}`
}
</script>

<template>
  <div class="card-skeuo p-5">
    <!-- Header -->
    <div class="flex items-center gap-3 mb-6">
      <div class="live-dot" />
      <h3 class="text-[#e2e8f0] font-medium text-sm">Pipeline Flow</h3>
      <span class="badge badge-teal ml-1">LIVE</span>
      <span class="ml-auto text-[#475569] text-xs mono">
        {{ stages.find(s => s.status !== 'idle') ? 'STREAMING' : 'IDLE' }}
      </span>
    </div>

    <!-- Pipeline diagram — horizontally scrollable on small screens -->
    <div class="overflow-x-auto pb-2">
      <div class="flex items-stretch gap-0 min-w-max">
        <template v-for="(stage, idx) in stages" :key="stage.name">
          <!-- Stage card -->
          <div
            class="relative flex flex-col items-center gap-2 px-4 py-3.5 rounded-[10px] w-[130px] transition-all duration-200"
            :style="{
              background: 'linear-gradient(145deg, #0f1117 0%, #151a22 100%)',
              border: `1px solid ${statusColor(stage.status)}33`,
              boxShadow: `inset 0 1px 0 rgba(255,255,255,0.03), ${statusGlow(stage.status)}`,
            }"
          >
            <!-- Status indicator bar at top -->
            <div
              class="absolute top-0 left-4 right-4 h-0.5 rounded-full"
              :style="{ background: statusColor(stage.status), opacity: stage.status === 'idle' ? 0.3 : 0.9 }"
            />

            <!-- Icon -->
            <div
              class="w-9 h-9 rounded-lg flex items-center justify-center"
              :style="{
                background: `${statusColor(stage.status)}14`,
                border: `1px solid ${statusColor(stage.status)}30`,
              }"
            >
              <AnimatedIcon
                :icon="STAGE_CONFIG[stage.name]?.icon ?? Zap"
                :size="16"
                :stroke-width="1.75"
                :animation="iconAnimation(stage.status)"
                :style="{ color: statusColor(stage.status) }"
              />
            </div>

            <!-- Stage name -->
            <div class="text-[#94a3b8] text-xs font-medium text-center leading-tight">
              {{ stage.name }}
            </div>

            <!-- Throughput -->
            <div class="text-center">
              <div
                class="mono font-semibold text-sm"
                :style="{ color: statusColor(stage.status) }"
              >
                {{ formatFps(stage.throughputFps) }}
                <span class="text-[10px] font-normal text-[#475569]">fps</span>
              </div>
            </div>

            <!-- Latency (p50 from histogram when available, else mean/estimate) -->
            <div class="text-center">
              <div class="mono text-[#f59e0b] text-xs font-medium">
                {{ displayLatency(stage) }}
              </div>
              <div class="text-[#475569] text-[10px]">
                {{ stage.percentiles ? 'p50' : 'latency' }}
              </div>
              <!-- p95 sub-label, only shown when real data present -->
              <div
                v-if="p95Label(stage.percentiles)"
                class="mono text-[#2d3748] text-[9px] mt-0.5"
              >
                {{ p95Label(stage.percentiles) }}
              </div>
            </div>

            <!-- Total frames (compact) -->
            <div class="text-center">
              <div class="mono text-[#475569] text-[10px]">
                {{ stage.totalFrames.toLocaleString() }}
              </div>
              <div class="text-[#2d3748] text-[10px]">frames</div>
            </div>

            <!-- Status badge -->
            <span
              class="badge text-[9px] px-1.5 py-0.5"
              :class="{
                'badge-teal':  stage.status === 'healthy',
                'badge-amber': stage.status === 'slow',
                'badge-red':   stage.status === 'error',
                'badge-muted': stage.status === 'idle',
              }"
            >
              {{ stage.status.toUpperCase() }}
            </span>
          </div>

          <!-- Animated connector (between stages) -->
          <div
            v-if="idx < stages.length - 1"
            class="flex items-center self-center flex-shrink-0 w-10"
          >
            <div class="relative w-full h-0.5 overflow-hidden rounded-full"
                 style="background: #1e2633;">
              <!-- Animated flow sweep -->
              <div
                v-if="connectorActive"
                class="connector-flow absolute inset-0"
              />
              <!-- Arrow tip -->
            </div>
            <!-- Arrow head -->
            <div
              class="flex-shrink-0 w-0 h-0"
              :style="{
                borderTop: '3px solid transparent',
                borderBottom: '3px solid transparent',
                borderLeft: `4px solid ${connectorActive ? '#2dd4bf' : '#1e2633'}`,
                opacity: connectorActive ? 0.7 : 0.3,
              }"
            />
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
@keyframes connector-sweep {
  0%   { transform: translateX(-100%); }
  100% { transform: translateX(300%); }
}

.connector-flow {
  background: linear-gradient(90deg, transparent 0%, #2dd4bf 50%, transparent 100%);
  animation: connector-sweep 1.8s ease-in-out infinite;
}
</style>
