<script setup lang="ts">
import { computed } from 'vue'
import { Activity, CircleStop, ShieldAlert, TimerOff } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'
import type { MetricsData } from '@/composables/useMetrics'

const props = defineProps<{
  metrics: MetricsData
}>()

const status = computed(() => {
  const states = {
    healthy: {
      label: 'Healthy',
      detail: 'All active pipeline generations are supervised.',
      color: '#2dd4bf',
    },
    faulted: {
      label: 'Faulted',
      detail: props.metrics.detachedWorkersTotal > 0
        ? `${props.metrics.detachedWorkersTotal.toLocaleString()} worker threads missed shutdown and still hold reserved capacity.`
        : 'A worker fault occurred in the last minute.',
      color: '#ef4444',
    },
    saturated: {
      label: 'Saturated',
      detail: 'Inference work is waiting for admitted capacity.',
      color: '#f59e0b',
    },
    idle: {
      label: 'Idle',
      detail: 'No media-pipeline generation is active.',
      color: '#64748b',
    },
  }
  return states[props.metrics.pipelineStatus]
})

const fmt = (value: number) => value.toLocaleString()
</script>

<template>
  <div
    class="card-skeuo p-5"
    :style="{ borderColor: `${status.color}33` }"
  >
    <div class="flex flex-col gap-4 lg:flex-row lg:items-center">
      <div class="flex items-center gap-3 lg:min-w-[310px]">
        <div
          class="w-10 h-10 rounded-[10px] flex items-center justify-center"
          :style="{ background: `${status.color}14`, border: `1px solid ${status.color}33` }"
        >
          <AnimatedIcon
            :icon="metrics.pipelineStatus === 'faulted' ? ShieldAlert : Activity"
            :size="18"
            :stroke-width="1.8"
            :animation="metrics.pipelineStatus === 'healthy' ? 'glow-teal' : undefined"
            :style="{ color: status.color }"
          />
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h3 class="text-[#e2e8f0] text-sm font-medium">Generation health</h3>
            <span
              class="mono text-[10px] font-semibold uppercase tracking-wider"
              :style="{ color: status.color }"
            >{{ status.label }}</span>
          </div>
          <p class="text-[#64748b] text-xs mt-1">{{ status.detail }}</p>
        </div>
      </div>

      <div class="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-5 gap-2 flex-1">
        <div class="rounded-[9px] px-3 py-2.5 bg-white/[0.025] border border-[#1e2633]">
          <div class="flex items-center gap-1.5 text-[#64748b] text-[10px] uppercase tracking-wider">
            <Activity :size="12" /> Active
          </div>
          <div class="mono text-[#e2e8f0] text-base mt-1">{{ fmt(metrics.activeGenerations) }}</div>
        </div>
        <div class="rounded-[9px] px-3 py-2.5 bg-white/[0.025] border border-[#1e2633]">
          <div class="flex items-center gap-1.5 text-[#64748b] text-[10px] uppercase tracking-wider">
            <CircleStop :size="12" /> Clean stops
          </div>
          <div class="mono text-[#2dd4bf] text-base mt-1">{{ fmt(metrics.cleanShutdownsTotal) }}</div>
        </div>
        <div class="rounded-[9px] px-3 py-2.5 bg-white/[0.025] border border-[#1e2633]">
          <div class="flex items-center gap-1.5 text-[#64748b] text-[10px] uppercase tracking-wider">
            <ShieldAlert :size="12" /> Worker faults
          </div>
          <div
            class="mono text-base mt-1"
            :class="metrics.workerFaultsTotal ? 'text-[#ef4444]' : 'text-[#94a3b8]'"
          >{{ fmt(metrics.workerFaultsTotal) }}</div>
        </div>
        <div class="rounded-[9px] px-3 py-2.5 bg-white/[0.025] border border-[#1e2633]">
          <div class="flex items-center gap-1.5 text-[#64748b] text-[10px] uppercase tracking-wider">
            <TimerOff :size="12" /> Forced stops
          </div>
          <div
            class="mono text-base mt-1"
            :class="metrics.forcedShutdownsTotal ? 'text-[#f59e0b]' : 'text-[#94a3b8]'"
          >{{ fmt(metrics.forcedShutdownsTotal) }}</div>
        </div>
        <div class="rounded-[9px] px-3 py-2.5 bg-white/[0.025] border border-[#1e2633]">
          <div class="flex items-center gap-1.5 text-[#64748b] text-[10px] uppercase tracking-wider">
            <TimerOff :size="12" /> Detached
          </div>
          <div
            class="mono text-base mt-1"
            :class="metrics.detachedWorkersTotal ? 'text-[#ef4444]' : 'text-[#94a3b8]'"
          >{{ fmt(metrics.detachedWorkersTotal) }}</div>
        </div>
      </div>
    </div>
  </div>
</template>
