<script setup lang="ts">
import { ref, computed } from 'vue'
import { ChevronDown, ChevronRight } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'
import type { Trace } from '@/composables/useMetrics'

const props = defineProps<{
  traces: Trace[]
}>()

// ─── Stage colors ─────────────────────────────────────────────────────────────

const STAGE_COLOR: Record<string, string> = {
  decode: '#2dd4bf',
  gate:   '#f59e0b',
  vlm:    '#22c55e',
}

const STAGE_LABEL: Record<string, string> = {
  decode: 'Decode',
  gate:   'Gate',
  vlm:    'VLM',
}

// ─── Expanded row state ───────────────────────────────────────────────────────

const expandedId = ref<string | null>(null)

function toggleExpand(id: string) {
  expandedId.value = expandedId.value === id ? null : id
}

// ─── Timeline scale ───────────────────────────────────────────────────────────

// Use p95 of totalMs as chart scale reference so most bars look full
const scaleMs = computed(() => {
  if (!props.traces.length) return 400
  const sorted = [...props.traces].map(t => t.totalMs).sort((a, b) => a - b)
  const p95idx = Math.floor(sorted.length * 0.95)
  return sorted[p95idx] ?? sorted[sorted.length - 1] ?? 400
})

function spanLeft(startMs: number): string {
  return `${(startMs / scaleMs.value) * 100}%`
}

function spanWidth(durationMs: number): string {
  return `${Math.max(0.5, (durationMs / scaleMs.value) * 100)}%`
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatTime(ts: number): string {
  const d = new Date(ts)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

function formatMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`
  if (ms >= 1)    return `${ms.toFixed(1)}ms`
  return `${(ms * 1000).toFixed(0)}µs`
}

function confidenceColor(c: number): string {
  if (c >= 0.85) return '#22c55e'
  if (c >= 0.70) return '#f59e0b'
  return '#ef4444'
}

// Sort traces newest first
const sortedTraces = computed(() =>
  [...props.traces].sort((a, b) => b.timestamp - a.timestamp)
)
</script>

<template>
  <div class="card-skeuo overflow-hidden">
    <!-- Header -->
    <div class="px-5 py-4 border-b border-[#1e2633] flex items-center gap-3 flex-wrap">
      <h3 class="text-[#e2e8f0] font-medium text-sm">Trace Timeline</h3>
      <span class="text-[#475569] text-xs mono">{{ traces.length }} keyframe traces</span>

      <!-- Legend -->
      <div class="flex items-center gap-4 ml-auto">
        <template v-for="(color, stage) in STAGE_COLOR" :key="stage">
          <div class="flex items-center gap-1.5">
            <div
              class="w-2.5 h-2.5 rounded-sm"
              :style="{ background: color, boxShadow: `0 0 6px ${color}66` }"
            />
            <span class="text-[#64748b] text-[10px] uppercase tracking-wider">
              {{ STAGE_LABEL[stage] }}
            </span>
          </div>
        </template>
      </div>
    </div>

    <!-- Empty state -->
    <div v-if="!traces.length" class="flex items-center justify-center py-16">
      <div class="text-center">
        <div class="text-[#2d3748] text-xs mono mb-2">No traces yet</div>
        <div class="text-[#475569] text-xs">Waiting for keyframe inference...</div>
      </div>
    </div>

    <!-- Trace rows -->
    <div v-else class="divide-y divide-[#1e2633]">
      <template v-for="trace in sortedTraces" :key="trace.id">
        <!-- Main row -->
        <div
          class="group cursor-pointer transition-colors duration-150"
          :class="expandedId === trace.id
            ? 'bg-[rgba(45,212,191,0.04)]'
            : 'hover:bg-[rgba(255,255,255,0.015)]'"
          role="button"
          :aria-expanded="expandedId === trace.id"
          @click="toggleExpand(trace.id)"
        >
          <div class="flex items-center gap-3 px-5 py-2.5">
            <!-- Expand toggle -->
            <AnimatedIcon
              :icon="expandedId === trace.id ? ChevronDown : ChevronRight"
              :size="12"
              :stroke-width="2"
              class="text-[#475569] shrink-0 transition-colors group-hover:text-[#64748b]"
            />

            <!-- Frame index + time -->
            <div class="w-32 shrink-0">
              <div class="mono text-xs text-[#e2e8f0] font-medium">
                #{{ trace.frameIndex.toLocaleString() }}
              </div>
              <div class="mono text-[10px] text-[#475569]">{{ formatTime(trace.timestamp) }}</div>
            </div>

            <!-- Waterfall bar -->
            <div class="flex-1 min-w-0 relative h-5 rounded-sm overflow-hidden"
                 style="background: #0a0c12;">
              <template v-for="span in trace.spans" :key="span.stage">
                <div
                  class="absolute top-0 bottom-0 rounded-sm"
                  :style="{
                    left: spanLeft(span.startMs),
                    width: spanWidth(span.durationMs),
                    background: STAGE_COLOR[span.stage],
                    opacity: span.status === 'slow' ? 0.7 : 0.85,
                    boxShadow: `0 0 4px ${STAGE_COLOR[span.stage]}55`,
                  }"
                />
              </template>
            </div>

            <!-- Total duration -->
            <div class="w-20 shrink-0 text-right">
              <div class="mono text-xs text-[#f59e0b] font-medium">
                {{ formatMs(trace.totalMs) }}
              </div>
            </div>

            <!-- Confidence -->
            <div class="w-12 shrink-0 text-right">
              <div
                class="mono text-xs font-medium"
                :style="{ color: confidenceColor(trace.confidence) }"
              >
                {{ (trace.confidence * 100).toFixed(0) }}%
              </div>
            </div>
          </div>
        </div>

        <!-- Expanded detail panel -->
        <div
          v-if="expandedId === trace.id"
          class="px-5 py-3"
          style="background: rgba(8,9,13,0.6); border-top: 1px solid #1e2633;"
        >
          <div class="flex flex-col sm:flex-row gap-4">
            <!-- Span breakdown table -->
            <div class="flex-1">
              <div class="text-[#475569] text-[10px] uppercase tracking-wider mb-2">Span Breakdown</div>
              <table class="w-full text-xs">
                <thead>
                  <tr class="text-[#475569] text-[10px] uppercase tracking-wider">
                    <th class="text-left pb-1.5 font-medium">Stage</th>
                    <th class="text-right pb-1.5 font-medium">Start</th>
                    <th class="text-right pb-1.5 font-medium">Duration</th>
                    <th class="text-right pb-1.5 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-[#1e2633]">
                  <tr
                    v-for="span in trace.spans"
                    :key="span.stage"
                    class="group/row"
                  >
                    <td class="py-1.5 pr-3">
                      <div class="flex items-center gap-2">
                        <div
                          class="w-2 h-2 rounded-sm shrink-0"
                          :style="{ background: STAGE_COLOR[span.stage] }"
                        />
                        <span class="mono text-[#94a3b8]">{{ STAGE_LABEL[span.stage] }}</span>
                      </div>
                    </td>
                    <td class="py-1.5 text-right mono text-[#475569]">+{{ formatMs(span.startMs) }}</td>
                    <td class="py-1.5 text-right mono" :style="{ color: STAGE_COLOR[span.stage] }">
                      {{ formatMs(span.durationMs) }}
                    </td>
                    <td class="py-1.5 text-right">
                      <span
                        class="badge text-[9px] px-1.5 py-0.5"
                        :class="span.status === 'slow' ? 'badge-amber' : 'badge-teal'"
                      >
                        {{ span.status.toUpperCase() }}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- Trace metadata -->
            <div class="sm:w-48 space-y-2">
              <div class="text-[#475569] text-[10px] uppercase tracking-wider mb-2">Metadata</div>
              <div class="flex justify-between text-xs">
                <span class="text-[#475569]">Trace ID</span>
                <span class="mono text-[#64748b]">{{ trace.id }}</span>
              </div>
              <div class="flex justify-between text-xs">
                <span class="text-[#475569]">Frame #</span>
                <span class="mono text-[#94a3b8]">{{ trace.frameIndex.toLocaleString() }}</span>
              </div>
              <div class="flex justify-between text-xs">
                <span class="text-[#475569]">Total time</span>
                <span class="mono text-[#f59e0b] font-medium">{{ formatMs(trace.totalMs) }}</span>
              </div>
              <div class="flex justify-between text-xs">
                <span class="text-[#475569]">Confidence</span>
                <span
                  class="mono font-medium"
                  :style="{ color: confidenceColor(trace.confidence) }"
                >
                  {{ (trace.confidence * 100).toFixed(1) }}%
                </span>
              </div>
              <div class="flex justify-between text-xs">
                <span class="text-[#475569]">Timestamp</span>
                <span class="mono text-[#475569]">{{ formatTime(trace.timestamp) }}</span>
              </div>
            </div>
          </div>
        </div>
      </template>
    </div>

    <!-- Footer: scale indicator -->
    <div
      v-if="traces.length"
      class="px-5 py-2.5 border-t border-[#1e2633] flex items-center justify-between"
      style="background: rgba(8,9,13,0.4);"
    >
      <span class="text-[#2d3748] text-[10px] mono">chart scale: {{ formatMs(scaleMs) }}</span>
      <span class="text-[#2d3748] text-[10px]">showing last {{ traces.length }} traces · click to expand</span>
    </div>
  </div>
</template>
