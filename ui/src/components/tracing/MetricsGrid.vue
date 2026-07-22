<script setup lang="ts">
import { computed } from 'vue'
import { Cpu, SlidersHorizontal, Zap, Database, Send } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'
import type { MetricsData, HistogramPercentiles } from '@/composables/useMetrics'

const props = defineProps<{
  metrics: MetricsData
}>()

// ─── Derived values ───────────────────────────────────────────────────────────

const passRatePct = computed(() =>
  (props.metrics.gatePassRate * 100).toFixed(1)
)

const noveltyReusePct = computed(() => props.metrics.noveltyReuseRatio * 100)

const blobSuccessPct = computed(() => {
  const successes = props.metrics.keyframeBlobsWrittenTotal + props.metrics.keyframeBlobsReusedTotal
  const attempts = successes + props.metrics.keyframeBlobFailuresTotal
  return attempts > 0 ? (successes / attempts) * 100 : 0
})

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

function fmtInt(n: number): string {
  return Math.floor(n).toLocaleString()
}

function formatLatency(ms: number): string {
  if (ms === 0) return '—'
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`
  if (ms >= 1)    return `${ms.toFixed(1)}ms`
  return `${(ms * 1000).toFixed(0)}µs`
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${fmtInt(bytes)} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`
}

/** Shorthand formatter for histogram percentile rows. */
function fmtPct(p: HistogramPercentiles | undefined, key: 'p50' | 'p95' | 'p99'): string {
  if (!p) return '—'
  return formatLatency(p[key])
}
</script>

<template>
  <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-4">

    <!-- Card 1: Decode Stats -->
    <div class="card-skeuo p-5 flex flex-col gap-4">
      <div class="flex items-center gap-2">
        <div class="w-7 h-7 rounded-lg flex items-center justify-center"
             style="background: rgba(45,212,191,0.1); border: 1px solid rgba(45,212,191,0.2);">
          <AnimatedIcon :icon="Cpu" :size="14" :stroke-width="1.75"
                        animation="glow-teal" class="text-[#2dd4bf]" />
        </div>
        <h4 class="text-[#94a3b8] text-xs font-medium uppercase tracking-wider">Decode</h4>
        <span class="badge badge-teal ml-auto text-[9px]">MEASURED</span>
      </div>

      <!-- Primary metric -->
      <div>
        <div class="text-[#475569] text-[10px] uppercase tracking-wider mb-1">Mean latency</div>
        <div class="mono text-3xl font-semibold" :style="{ color: decodeLatencyColor }">
          {{ formatLatency(metrics.decodeLatencyMs) }}
        </div>
      </div>

      <!-- Secondary rows -->
      <div class="space-y-2 border-t border-[#1e2633] pt-3">
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Dropped frames</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.droppedFramesTotal) }}</span>
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

      <!-- Percentile rows (real histogram data) -->
      <div
        v-if="metrics.histograms['Decode']"
        class="space-y-1.5 border-t border-[#1e2633] pt-3"
      >
        <div class="text-[#2d3748] text-[10px] uppercase tracking-wider mb-1">Percentiles</div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p50</span>
          <span class="mono text-[11px]" :style="{ color: decodeLatencyColor }">
            {{ fmtPct(metrics.histograms['Decode'], 'p50') }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p95</span>
          <span class="mono text-[11px]" :style="{ color: decodeLatencyColor }">
            {{ fmtPct(metrics.histograms['Decode'], 'p95') }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p99</span>
          <span class="mono text-[11px]" :style="{ color: decodeLatencyColor }">
            {{ fmtPct(metrics.histograms['Decode'], 'p99') }}
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
          <span class="text-[#475569] text-xs">Dropped keyframes</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.keyframesDroppedTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Loops detected</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.loopDetectedTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Zone assertions</span>
          <span class="mono text-[#38bdf8] text-xs">{{ fmtInt(metrics.restrictedZoneAssertionsTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Zone evidence failures</span>
          <span class="mono text-xs" :class="metrics.restrictedZoneEvidenceFailuresTotal ? 'text-[#ef4444]' : 'text-[#2dd4bf]'">
            {{ fmtInt(metrics.restrictedZoneEvidenceFailuresTotal) }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Zone queue drops</span>
          <span class="mono text-xs" :class="metrics.restrictedZoneQueueDroppedTotal ? 'text-[#f59e0b]' : 'text-[#2dd4bf]'">
            {{ fmtInt(metrics.restrictedZoneQueueDroppedTotal) }}
          </span>
        </div>
      </div>

      <!-- Percentile rows (real histogram data) -->
      <div
        v-if="metrics.histograms['Gate']"
        class="space-y-1.5 border-t border-[#2a2010] pt-3"
      >
        <div class="text-[#2d3748] text-[10px] uppercase tracking-wider mb-1">Latency percentiles</div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p50</span>
          <span class="mono text-[11px] text-[#f59e0b]">
            {{ fmtPct(metrics.histograms['Gate'], 'p50') }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p95</span>
          <span class="mono text-[11px] text-[#f59e0b]">
            {{ fmtPct(metrics.histograms['Gate'], 'p95') }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p99</span>
          <span class="mono text-[11px] text-[#f59e0b]">
            {{ fmtPct(metrics.histograms['Gate'], 'p99') }}
          </span>
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
          <span class="text-[#475569] text-xs">Novelty evaluated</span>
          <span class="mono text-[#22c55e] text-xs">{{ fmtInt(metrics.noveltyEvaluatedTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Descriptions reused</span>
          <span class="mono text-[#22c55e] text-xs">{{ fmtInt(metrics.noveltyReusedTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Forced refreshes</span>
          <span class="mono text-[#22c55e] text-xs">{{ fmtInt(metrics.noveltyForcedRefreshTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Embedding unavailable</span>
          <span class="mono text-xs" :class="metrics.noveltyEmbeddingUnavailableTotal ? 'text-[#f59e0b]' : 'text-[#22c55e]'">
            {{ fmtInt(metrics.noveltyEmbeddingUnavailableTotal) }}
          </span>
        </div>
      </div>

      <!-- Percentile rows (real histogram data) -->
      <div
        v-if="metrics.histograms['VLM']"
        class="space-y-1.5 border-t border-[#1e2633] pt-3"
      >
        <div class="text-[#2d3748] text-[10px] uppercase tracking-wider mb-1">Latency percentiles</div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p50</span>
          <span class="mono text-[11px]" :style="{ color: vlmLatencyColor }">
            {{ fmtPct(metrics.histograms['VLM'], 'p50') }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p95</span>
          <span class="mono text-[11px]" :style="{ color: vlmLatencyColor }">
            {{ fmtPct(metrics.histograms['VLM'], 'p95') }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p99</span>
          <span class="mono text-[11px]" :style="{ color: vlmLatencyColor }">
            {{ fmtPct(metrics.histograms['VLM'], 'p99') }}
          </span>
        </div>
      </div>

      <!-- Semantic reuse bar -->
      <div>
        <div class="flex justify-between text-[10px] text-[#475569] mb-1">
          <span>Semantic reuse</span>
          <span class="mono">{{ noveltyReusePct.toFixed(1) }}%</span>
        </div>
        <div class="h-1 rounded-full" style="background: #1e2633;">
          <div
            class="h-full rounded-full transition-all duration-500"
            :style="{
              width: `${Math.min(100, noveltyReusePct)}%`,
              background: '#22c55e',
              boxShadow: '0 0 8px rgba(34,197,94,0.3)',
            }"
          />
        </div>
      </div>
    </div>

    <!-- Card 4: durable binary keyframe sidecar -->
    <div class="card-skeuo p-5 flex flex-col gap-4">
      <div class="flex items-center gap-2">
        <div class="w-7 h-7 rounded-lg flex items-center justify-center"
             style="background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.15);">
          <AnimatedIcon :icon="Database" :size="14" :stroke-width="1.75"
                        class="text-[#2dd4bf]" />
        </div>
        <h4 class="text-[#94a3b8] text-xs font-medium uppercase tracking-wider">Keyframe sidecar</h4>
        <span class="badge badge-teal ml-auto text-[9px]">BINARY</span>
      </div>

      <!-- Primary metric -->
      <div>
        <div class="text-[#475569] text-[10px] uppercase tracking-wider mb-1">Bytes stored</div>
        <div class="mono text-3xl font-semibold text-[#2dd4bf]">
          {{ formatBytes(metrics.keyframeBlobBytesTotal) }}
        </div>
      </div>

      <!-- Stats -->
      <div class="space-y-2 border-t border-[#1e2633] pt-3">
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Blobs written</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.keyframeBlobsWrittenTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Content-addressed reuse</span>
          <span class="mono text-[#2dd4bf] text-xs">{{ fmtInt(metrics.keyframeBlobsReusedTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Write failures</span>
          <span class="mono text-xs" :class="metrics.keyframeBlobFailuresTotal ? 'text-[#ef4444]' : 'text-[#2dd4bf]'">
            {{ fmtInt(metrics.keyframeBlobFailuresTotal) }}
          </span>
        </div>
      </div>

      <!-- Percentile rows (real histogram data) -->
      <div
        v-if="metrics.histograms['Keyframe store']"
        class="space-y-1.5 border-t border-[#1e2633] pt-3"
      >
        <div class="text-[#2d3748] text-[10px] uppercase tracking-wider mb-1">Write latency</div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p50</span>
          <span class="mono text-[11px] text-[#2dd4bf]">
            {{ fmtPct(metrics.histograms['Keyframe store'], 'p50') }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p95</span>
          <span class="mono text-[11px] text-[#2dd4bf]">
            {{ fmtPct(metrics.histograms['Keyframe store'], 'p95') }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-[11px]">p99</span>
          <span class="mono text-[11px] text-[#2dd4bf]">
            {{ fmtPct(metrics.histograms['Keyframe store'], 'p99') }}
          </span>
        </div>
      </div>

      <!-- Sidecar success bar -->
      <div>
        <div class="flex justify-between text-[10px] text-[#475569] mb-1">
          <span>Write success</span>
          <span class="mono text-[#2dd4bf]">{{ blobSuccessPct.toFixed(1) }}%</span>
        </div>
        <div class="h-1 rounded-full" style="background: #1e2633;">
          <div
            class="h-full rounded-full transition-all duration-500 progress-bar-active"
            :style="{
              width: `${Math.min(100, blobSuccessPct)}%`,
              background: 'linear-gradient(90deg, #0d9488, #2dd4bf)',
              boxShadow: '0 0 8px rgba(45,212,191,0.35)',
            }"
          />
        </div>
      </div>
    </div>

    <!-- Card 5: durable event delivery -->
    <div class="card-skeuo p-5 flex flex-col gap-4">
      <div class="flex items-center gap-2">
        <div class="w-7 h-7 rounded-lg flex items-center justify-center"
             style="background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.18);">
          <AnimatedIcon :icon="Send" :size="14" :stroke-width="1.75" class="text-[#38bdf8]" />
        </div>
        <h4 class="text-[#94a3b8] text-xs font-medium uppercase tracking-wider">Delivery</h4>
        <span class="badge badge-teal ml-auto text-[9px]">DURABLE</span>
      </div>

      <div>
        <div class="text-[#475569] text-[10px] uppercase tracking-wider mb-1">Events sent</div>
        <div class="mono text-3xl font-semibold text-[#38bdf8]">
          {{ fmtInt(metrics.sseEventsTotal + metrics.webhookDeliveredTotal) }}
        </div>
      </div>

      <div class="space-y-2 border-t border-[#1e2633] pt-3">
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">SSE subscribers</span>
          <span class="mono text-[#38bdf8] text-xs">{{ fmtInt(metrics.sseSubscribersActive) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">WAL replays</span>
          <span class="mono text-[#2dd4bf] text-xs">{{ fmtInt(metrics.sseReplayedEventsTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Queue stalls</span>
          <span class="mono text-xs" :class="metrics.sseQueueStallsTotal ? 'text-[#f59e0b]' : 'text-[#2dd4bf]'">
            {{ fmtInt(metrics.sseQueueStallsTotal) }}
          </span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Active webhooks</span>
          <span class="mono text-[#38bdf8] text-xs">{{ fmtInt(metrics.webhooksConfigured) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Webhook delivered</span>
          <span class="mono text-[#2dd4bf] text-xs">{{ fmtInt(metrics.webhookDeliveredTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Retries</span>
          <span class="mono text-[#f59e0b] text-xs">{{ fmtInt(metrics.webhookRetriesTotal) }}</span>
        </div>
        <div class="flex items-center justify-between">
          <span class="text-[#475569] text-xs">Dead letters</span>
          <span class="mono text-xs" :class="metrics.webhookDeadLettersTotal ? 'text-[#ef4444]' : 'text-[#2dd4bf]'">
            {{ fmtInt(metrics.webhookDeadLettersTotal) }}
          </span>
        </div>
      </div>
    </div>

  </div>
</template>
