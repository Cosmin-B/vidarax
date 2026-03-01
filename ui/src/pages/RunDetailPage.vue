<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useRunsStore } from '@/stores/runs'
import { useEventsStore } from '@/stores/events'
import type { RunStatus } from '@/stores/runs'

const route = useRoute()
const router = useRouter()
const runsStore = useRunsStore()
const eventsStore = useEventsStore()

const runId = computed(() => route.params.runId as string)
const activeTab = ref<'timeline' | 'keyframes' | 'metadata'>('timeline')

const run = computed(() => runsStore.activeRun ?? runsStore.runs.find(r => r.run_id === runId.value))
const events = computed(() => eventsStore.eventsForRun(runId.value))
const keyframes = computed(() => eventsStore.keyframesForRun(runId.value))

onMounted(() => {
  eventsStore.setActiveRunId(runId.value)
  if (!run.value) {
    // In production: fetch from API
    // For scaffold: find from store or redirect
    router.push('/')
  }
})

function statusConfig(status: RunStatus) {
  const map = {
    processing: { label: 'Processing', cls: 'badge-amber' },
    completed: { label: 'Completed', cls: 'badge-green' },
    failed: { label: 'Failed', cls: 'badge-red' },
    pending: { label: 'Pending', cls: 'badge-muted' },
    stopped: { label: 'Stopped', cls: 'badge-muted' },
  }
  return map[status] ?? { label: status, cls: 'badge-muted' }
}

function eventDotColor(type: string): string {
  const map: Record<string, string> = {
    scene_cut: '#2dd4bf',
    loop_detected: '#f59e0b',
    vlm_description: '#22c55e',
    artifact_suspected: '#ef4444',
  }
  return map[type] ?? '#64748b'
}
</script>

<template>
  <div class="p-4 lg:p-6 space-y-6">
    <!-- Not found -->
    <div v-if="!run" class="flex items-center justify-center py-20">
      <div class="text-center">
        <div class="text-[#475569] text-sm">Run not found</div>
        <RouterLink to="/" class="text-[#2dd4bf] text-sm mt-2 inline-block">Back to Dashboard</RouterLink>
      </div>
    </div>

    <template v-else>
      <!-- Header -->
      <div class="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <div class="flex items-center gap-2 mb-1">
            <RouterLink to="/" class="text-[#475569] hover:text-[#64748b] transition-colors" aria-label="Back">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M10 3L6 8L10 13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </RouterLink>
            <span :class="['badge', statusConfig(run.status as RunStatus).cls]">
              {{ statusConfig(run.status as RunStatus).label }}
            </span>
          </div>
          <h2 class="mono text-[#e2e8f0] font-semibold text-lg">{{ run.run_id }}</h2>
          <div class="flex flex-wrap items-center gap-3 mt-2">
            <span class="text-[#64748b] text-xs">
              Model: <span class="text-[#94a3b8] mono">{{ run.model }}</span>
            </span>
            <span class="text-[#64748b] text-xs">
              Mode: <span class="text-[#94a3b8] mono">{{ run.mode }}</span>
            </span>
            <span class="text-[#64748b] text-xs">
              Events: <span class="text-[#f59e0b] mono">{{ run.event_count ?? 0 }}</span>
            </span>
          </div>
        </div>

        <!-- Actions -->
        <div class="flex gap-2 shrink-0">
          <button
            v-if="run.status === 'processing'"
            class="px-3 py-1.5 rounded-[8px] text-xs font-medium text-[#ef4444] transition-all duration-200"
            style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);"
          >
            Stop
          </button>
          <button
            class="px-3 py-1.5 rounded-[8px] text-xs font-medium text-[#64748b] transition-all duration-200"
            style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
          >
            Delete
          </button>
        </div>
      </div>

      <!-- Tabs -->
      <div class="flex gap-1 p-1 rounded-[10px] w-fit" style="background: #0f1117; border: 1px solid #1e2633;">
        <button
          v-for="tab in ['timeline', 'keyframes', 'metadata'] as const"
          :key="tab"
          class="px-4 py-1.5 rounded-[8px] text-sm font-medium transition-all duration-200 capitalize"
          :class="activeTab === tab
            ? 'text-[#e2e8f0] bg-[#151a22]'
            : 'text-[#475569] hover:text-[#64748b]'"
          :style="activeTab === tab
            ? 'box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 2px 8px rgba(0,0,0,0.4);'
            : ''"
          @click="activeTab = tab"
        >
          {{ tab }}
        </button>
      </div>

      <!-- Tab: Timeline -->
      <div v-if="activeTab === 'timeline'" class="card-skeuo p-5">
        <h3 class="text-[#e2e8f0] font-medium text-sm mb-4">Event Timeline</h3>
        <div v-if="events.length === 0" class="py-10 text-center text-[#475569] text-sm">
          No events recorded for this run.
        </div>
        <div v-else class="space-y-2">
          <div
            v-for="evt in events"
            :key="`${evt.frame_index}-${evt.event_type}`"
            class="flex items-center gap-3 p-3 rounded-[8px] transition-colors duration-200 hover:bg-[rgba(255,255,255,0.02)]"
          >
            <div
              class="w-2.5 h-2.5 rounded-full shrink-0"
              :style="`background: ${eventDotColor(evt.event_type)}; box-shadow: 0 0 6px ${eventDotColor(evt.event_type)}66`"
            />
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2">
                <span class="mono text-xs text-[#475569]">
                  frame {{ evt.frame_index }}
                </span>
                <span class="mono text-xs" :style="`color: ${eventDotColor(evt.event_type)}`">
                  {{ evt.event_type }}
                </span>
                <span class="mono text-xs text-[#f59e0b]">
                  {{ (evt.confidence * 100).toFixed(0) }}%
                </span>
              </div>
              <p class="text-[#94a3b8] text-xs mt-0.5 truncate">{{ evt.description }}</p>
            </div>
            <span class="mono text-xs text-[#475569] shrink-0 hidden sm:block">
              {{ evt.pts_ms }}ms
            </span>
          </div>
        </div>
      </div>

      <!-- Tab: Keyframes -->
      <div v-if="activeTab === 'keyframes'" class="card-skeuo p-5">
        <h3 class="text-[#e2e8f0] font-medium text-sm mb-4">Keyframe Gallery</h3>
        <div v-if="keyframes.length === 0" class="py-10 text-center text-[#475569] text-sm">
          No keyframes stored for this run.
        </div>
        <div v-else class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          <div
            v-for="kf in keyframes"
            :key="kf.id"
            class="relative rounded-[10px] overflow-hidden group"
            style="background: #050507; border: 1px solid #1e2633; aspect-ratio: 16/9;"
          >
            <img
              v-if="kf.jpeg_b64"
              :src="`data:image/jpeg;base64,${kf.jpeg_b64}`"
              :alt="`Keyframe ${kf.frame_index}`"
              class="w-full h-full object-cover"
            />
            <div
              class="absolute bottom-0 left-0 right-0 p-2"
              style="background: linear-gradient(0deg, rgba(5,5,7,0.9) 0%, transparent 100%);"
            >
              <p class="text-[#94a3b8] text-xs line-clamp-2">{{ kf.description }}</p>
              <span class="mono text-[#475569] text-xs">f{{ kf.frame_index }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab: Metadata -->
      <div v-if="activeTab === 'metadata'" class="card-skeuo p-5">
        <h3 class="text-[#e2e8f0] font-medium text-sm mb-4">Raw Metadata</h3>
        <pre
          class="mono text-xs text-[#94a3b8] bg-[#050507] rounded-[8px] p-4 overflow-x-auto border border-[#1e2633]"
          style="line-height: 1.7;"
        >{{ JSON.stringify(run, null, 2) }}</pre>
      </div>
    </template>
  </div>
</template>
