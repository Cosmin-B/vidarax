<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { RefreshCw, Activity, AlertCircle } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'
import PipelineOverview from '@/components/tracing/PipelineOverview.vue'
import MetricsGrid from '@/components/tracing/MetricsGrid.vue'
import { useMetrics } from '@/composables/useMetrics'

// ─── Metrics polling ──────────────────────────────────────────────────────────

const { metrics, error, loading, start, refetch } = useMetrics(2000)

// ─── Last updated display ─────────────────────────────────────────────────────

const lastUpdatedLabel = computed(() => {
  if (!metrics.value) return '—'
  const diff = Date.now() - metrics.value.lastUpdated
  if (diff < 2000) return 'just now'
  return `${Math.floor(diff / 1000)}s ago`
})

// ─── Lifecycle ────────────────────────────────────────────────────────────────

onMounted(() => {
  start()
})
</script>

<template>
  <div class="p-4 lg:p-6 space-y-6">

    <!-- Page header -->
    <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
      <div>
        <div class="flex items-center gap-2.5 mb-1">
          <div
            class="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
            style="background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.2);"
          >
            <AnimatedIcon
              :icon="Activity"
              :size="16"
              :stroke-width="1.75"
              animation="glow-teal"
              class="text-[#2dd4bf]"
            />
          </div>
          <h2 class="text-[#e2e8f0] font-semibold text-xl">Pipeline Observability</h2>
          <span v-if="metrics && !error" class="badge badge-teal">LIVE</span>
          <span v-else-if="metrics" class="badge badge-amber">STALE</span>
          <span v-else class="badge badge-muted">UNAVAILABLE</span>
        </div>
        <p class="text-[#475569] text-sm">
          Pipeline telemetry — live streams and recorded-file runs
        </p>
      </div>

      <div class="flex items-center gap-3">
        <!-- Last updated -->
        <div class="hidden sm:flex items-center gap-1.5 text-xs text-[#475569]">
          <div
            class="w-1.5 h-1.5 rounded-full"
            :class="loading ? 'bg-[#f59e0b]' : 'bg-[#2dd4bf]'"
            :style="loading ? '' : 'animation: pulse-teal 2s infinite;'"
          />
          <span class="mono">{{ lastUpdatedLabel }}</span>
        </div>

        <!-- Refresh button -->
        <button
          class="w-11 h-11 flex items-center justify-center rounded-[8px] text-[#64748b] transition-colors duration-200 hover:text-[#94a3b8] icon-hover-parent"
          style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
          :disabled="loading"
          :class="loading ? 'opacity-50' : ''"
          title="Refresh metrics"
          aria-label="Refresh metrics"
          @click="refetch"
        >
          <AnimatedIcon
            :icon="RefreshCw"
            :size="14"
            :stroke-width="2"
            :animation="loading ? 'spin' : undefined"
          />
        </button>
      </div>
    </div>

    <!-- Error banner -->
    <div
      v-if="error"
      class="flex items-center gap-3 p-3 rounded-[10px] text-sm"
      style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);"
    >
      <AnimatedIcon
        :icon="AlertCircle"
        :size="14"
        :stroke-width="2"
        animation="fade-in"
        class="text-[#ef4444] shrink-0"
      />
      <span class="text-[#ef4444] flex-1">{{ error }}</span>
      <button class="min-h-11 px-2 text-[#ef4444] text-xs underline hover:text-[#fca5a5]" @click="refetch">
        Retry
      </button>
    </div>

    <!-- Loading skeleton (first load) -->
    <div v-if="!metrics && loading" class="space-y-4">
      <!-- Pipeline skeleton -->
      <div class="card-skeuo p-5">
        <div class="flex items-center gap-2 mb-6">
          <div class="w-2 h-2 rounded-full bg-[#1e2633]" />
          <div class="h-3 w-32 rounded bg-[#1e2633]" />
        </div>
        <div class="flex gap-4">
          <div v-for="i in 5" :key="i"
               class="w-[130px] h-36 rounded-[10px] animate-pulse"
               style="background: #151a22;" />
        </div>
      </div>
      <!-- Grid skeleton -->
      <div class="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <div v-for="i in 4" :key="i"
             class="h-48 rounded-[12px] animate-pulse"
             style="background: #151a22;" />
      </div>
    </div>

    <!-- Main content -->
    <template v-else-if="metrics">

      <!-- 1. Pipeline Overview -->
      <section aria-label="Pipeline flow diagram">
        <PipelineOverview :stages="metrics.stages" />
      </section>

      <!-- 2. Live Metrics Grid -->
      <section aria-label="Live pipeline metrics">
        <div class="flex items-center gap-2 mb-3">
          <h3 class="text-[#64748b] text-xs font-medium uppercase tracking-wider">Pipeline Metrics</h3>
          <div class="flex-1 h-px" style="background: #1e2633;" />
          <span class="mono text-[#2d3748] text-[10px]">2s refresh</span>
        </div>
        <MetricsGrid :metrics="metrics" />
      </section>

    </template>

    <div
      v-else
      class="card-skeuo flex flex-col items-center justify-center gap-2 px-6 py-16 text-center"
    >
      <AnimatedIcon :icon="AlertCircle" :size="22" :stroke-width="1.5" class="text-[#475569]" />
      <p class="text-[#94a3b8] text-sm">No telemetry available</p>
      <p class="text-[#475569] text-xs">Connect the UI to a running Vidarax API and retry.</p>
    </div>

  </div>
</template>
