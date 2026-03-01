<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useRunsStore } from '@/stores/runs'
import { useEventsStore } from '@/stores/events'
import type { RunSummary, RunStatus } from '@/stores/runs'

const router = useRouter()
const runsStore = useRunsStore()
const eventsStore = useEventsStore()

const loading = ref(false)

// Placeholder data so page renders without API
const mockRuns: RunSummary[] = [
  {
    run_id: 'run_8f3a2c1d',
    status: 'processing',
    mode: 'webrtc',
    model: 'qwen2.5-vl-7b',
    created_at: new Date(Date.now() - 1000 * 60 * 3).toISOString(),
    event_count: 42,
  },
  {
    run_id: 'run_7e2b1c0a',
    status: 'completed',
    mode: 'batch',
    model: 'qwen2.5-vl-7b',
    created_at: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
    event_count: 187,
  },
  {
    run_id: 'run_6d1a0b9f',
    status: 'failed',
    mode: 'batch',
    model: 'llava-1.6-7b',
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
    event_count: 0,
  },
  {
    run_id: 'run_5c0b9a8e',
    status: 'completed',
    mode: 'stream',
    model: 'qwen2.5-vl-2b',
    created_at: new Date(Date.now() - 1000 * 60 * 60 * 5).toISOString(),
    event_count: 321,
  },
]

onMounted(() => {
  if (runsStore.runs.length === 0) {
    runsStore.setRuns(mockRuns)
  }
})

const runs = computed(() => runsStore.sortedRuns)
const recentEvents = computed(() => eventsStore.latestEvents)

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

function formatRelative(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function goToRun(runId: string) {
  router.push(`/runs/${runId}`)
}
</script>

<template>
  <div class="p-4 lg:p-6 space-y-6">
    <!-- Page header -->
    <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
      <div>
        <h2 class="text-[#e2e8f0] font-semibold text-xl">Command Center</h2>
        <p class="text-[#64748b] text-sm mt-1">Monitor runs, streams, and events</p>
      </div>
      <div class="flex gap-3">
        <RouterLink
          to="/stream"
          class="flex items-center gap-2 px-4 py-2 rounded-[10px] text-sm font-medium text-[#08090d] transition-all duration-200"
          style="background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
                 box-shadow: 0 0 20px rgba(45,212,191,0.2);"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="2" fill="currentColor"/>
            <path d="M3.5 3.5C2.3 4.7 2.3 9.3 3.5 10.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M10.5 3.5C11.7 4.7 11.7 9.3 10.5 10.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
          New Stream
        </RouterLink>
        <RouterLink
          to="/upload"
          class="flex items-center gap-2 px-4 py-2 rounded-[10px] text-sm font-medium text-[#94a3b8] transition-all duration-200"
          style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M7 1L7 9M7 1L4 4M7 1L10 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 11V12C2 12.6 2.4 13 3 13H11C11.6 13 12 12.6 12 12V11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
          Upload Video
        </RouterLink>
      </div>
    </div>

    <!-- Stats row -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <div
        v-for="stat in [
          { label: 'Total Runs', value: String(runs.length), unit: '', accent: 'teal' },
          { label: 'Processing', value: String(runs.filter(r => r.status === 'processing').length), unit: '', accent: 'amber' },
          { label: 'Events', value: String(runs.reduce((a, r) => a + (r.event_count ?? 0), 0)), unit: '', accent: 'teal' },
          { label: 'Completed', value: String(runs.filter(r => r.status === 'completed').length), unit: '', accent: 'green' },
        ]"
        :key="stat.label"
        class="card-skeuo p-4"
      >
        <div class="text-[#64748b] text-xs font-medium uppercase tracking-wider mb-2">
          {{ stat.label }}
        </div>
        <div
          class="mono text-2xl font-semibold"
          :class="{
            'text-[#2dd4bf]': stat.accent === 'teal',
            'text-[#f59e0b]': stat.accent === 'amber',
            'text-[#22c55e]': stat.accent === 'green',
          }"
        >
          {{ stat.value }}
        </div>
      </div>
    </div>

    <!-- Run list -->
    <div class="card-skeuo overflow-hidden">
      <div class="px-5 py-4 border-b border-[#1e2633] flex items-center justify-between">
        <h3 class="text-[#e2e8f0] font-medium text-sm">Recent Runs</h3>
        <span class="text-[#475569] text-xs mono">{{ runs.length }} total</span>
      </div>

      <div v-if="loading" class="flex items-center justify-center py-12">
        <div class="w-6 h-6 rounded-full border-2 border-[#1e2633] border-t-[#2dd4bf] animate-spin" />
      </div>

      <div v-else-if="runs.length === 0" class="py-12 text-center">
        <p class="text-[#475569] text-sm">No runs yet.</p>
        <RouterLink to="/stream" class="text-[#2dd4bf] text-sm mt-2 inline-block hover:text-[#5eead4]">
          Start your first stream
        </RouterLink>
      </div>

      <div v-else class="divide-y divide-[#1e2633]">
        <button
          v-for="run in runs"
          :key="run.run_id"
          class="w-full flex items-center gap-4 px-5 py-3.5 text-left transition-colors duration-200 hover:bg-[rgba(255,255,255,0.02)] group"
          @click="goToRun(run.run_id)"
        >
          <!-- Status dot -->
          <div
            class="w-2 h-2 rounded-full shrink-0"
            :class="{
              'bg-[#f59e0b]': run.status === 'processing',
              'bg-[#22c55e]': run.status === 'completed',
              'bg-[#ef4444]': run.status === 'failed',
              'bg-[#475569]': run.status === 'pending' || run.status === 'stopped',
            }"
            :style="run.status === 'processing' ? 'animation: pulse-teal 2s infinite; box-shadow: 0 0 6px rgba(245,158,11,0.5)' : ''"
          />

          <!-- Run ID + mode -->
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2">
              <span class="mono text-[#e2e8f0] text-sm font-medium truncate">
                {{ run.run_id }}
              </span>
              <span class="badge badge-muted hidden sm:inline-flex">{{ run.mode }}</span>
            </div>
            <div class="text-[#475569] text-xs mt-0.5 truncate">
              {{ run.model }} &middot; {{ formatRelative(run.created_at) }}
            </div>
          </div>

          <!-- Event count -->
          <div class="text-right shrink-0 hidden sm:block">
            <div class="mono text-[#f59e0b] text-sm">{{ run.event_count ?? 0 }}</div>
            <div class="text-[#475569] text-xs">events</div>
          </div>

          <!-- Status badge -->
          <span :class="['badge shrink-0', statusConfig(run.status).cls]">
            {{ statusConfig(run.status).label }}
          </span>

          <!-- Chevron -->
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" class="text-[#475569] shrink-0 group-hover:text-[#64748b] transition-colors">
            <path d="M5 3L9 7L5 11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- Recent events (when connected) -->
    <div v-if="recentEvents.length > 0" class="card-skeuo overflow-hidden">
      <div class="px-5 py-4 border-b border-[#1e2633]">
        <h3 class="text-[#e2e8f0] font-medium text-sm">Live Events</h3>
      </div>
      <div class="divide-y divide-[#1e2633] max-h-64 overflow-y-auto">
        <div
          v-for="evt in recentEvents"
          :key="`${evt.run_id}-${evt.frame_index}`"
          class="px-5 py-3 flex items-center gap-3"
        >
          <div
            class="w-2 h-2 rounded-full shrink-0"
            :class="{
              'bg-[#2dd4bf]': evt.event_type === 'scene_cut',
              'bg-[#f59e0b]': evt.event_type === 'loop_detected',
              'bg-[#22c55e]': evt.event_type === 'vlm_description',
              'bg-[#ef4444]': evt.event_type === 'artifact_suspected',
            }"
          />
          <span class="text-[#94a3b8] text-xs flex-1 truncate">{{ evt.description }}</span>
          <span class="mono text-[#475569] text-xs shrink-0">{{ evt.event_type }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
