<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useRunsStore } from '@/stores/runs'
import { useEventsStore } from '@/stores/events'
import { api, ApiError } from '@/lib/api'
import type { RunSummary, RunStatus } from '@/stores/runs'
import { RefreshCw, Radio, Upload, AlertCircle, ChevronRight, Zap } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'

const router = useRouter()
const runsStore = useRunsStore()
const eventsStore = useEventsStore()

const loading = ref(false)
const fetchError = ref<string | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null

// ── Data ──────────────────────────────────────────────────────────────────────

const runs = computed(() => runsStore.sortedRuns)
const recentEvents = computed(() => eventsStore.latestEvents)
const hasProcessing = computed(() => runs.value.some(r => r.status === 'processing'))

// ── API calls ─────────────────────────────────────────────────────────────────

async function fetchRuns(): Promise<void> {
  loading.value = true
  fetchError.value = null
  try {
    const list = await api.runs.list()
    // api returns RunResponse[] — map to RunSummary
    runsStore.setRuns(
      list.map(r => ({
        run_id: r.run_id,
        status: r.status as RunSummary['status'],
        mode: r.mode as RunSummary['mode'],
        model: r.model,
        created_at: r.created_at,
      }))
    )
  } catch (err) {
    fetchError.value = err instanceof ApiError
      ? `API error ${err.status}: ${err.message}`
      : err instanceof Error ? err.message : 'Failed to fetch runs'
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  await fetchRuns()
  // Poll every 8s while any run is processing
  pollTimer = setInterval(() => {
    if (hasProcessing.value) fetchRuns()
  }, 8000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})

// ── Helpers ───────────────────────────────────────────────────────────────────

function statusConfig(status: RunStatus) {
  const map: Record<RunStatus, { label: string; cls: string }> = {
    processing: { label: 'Processing', cls: 'badge-amber' },
    completed:  { label: 'Completed',  cls: 'badge-green' },
    failed:     { label: 'Failed',     cls: 'badge-red'   },
    pending:    { label: 'Pending',    cls: 'badge-muted' },
    stopped:    { label: 'Stopped',   cls: 'badge-muted' },
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

const EVENT_COLORS: Record<string, string> = {
  scene_cut: '#2dd4bf', loop_detected: '#f59e0b',
  vlm_description: '#22c55e', artifact_suspected: '#ef4444',
  exposure_shift: '#a78bfa', flicker: '#fb7185', keyframe: '#22c55e',
}
function eventColor(type: string) { return EVENT_COLORS[type] ?? '#64748b' }
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
        <!-- Refresh -->
        <button
          class="w-9 h-9 flex items-center justify-center rounded-[8px] text-[#64748b] transition-colors duration-200 hover:text-[#94a3b8] icon-hover-parent"
          style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
          :disabled="loading"
          :class="loading ? 'opacity-50' : ''"
          title="Refresh"
          aria-label="Refresh runs"
          @click="fetchRuns"
        >
          <AnimatedIcon
            :icon="RefreshCw"
            :size="14"
            :stroke-width="2"
            :animation="loading ? 'spin' : undefined"
          />
        </button>
        <RouterLink
          to="/stream"
          class="flex items-center gap-2 px-4 py-2 rounded-[10px] text-sm font-medium text-[#08090d] transition-all duration-200 icon-hover-parent"
          style="background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
                 box-shadow: 0 0 20px rgba(45,212,191,0.2);"
        >
          <AnimatedIcon :icon="Radio" :size="14" :stroke-width="2" />
          New Stream
        </RouterLink>
        <RouterLink
          to="/upload"
          class="flex items-center gap-2 px-4 py-2 rounded-[10px] text-sm font-medium text-[#94a3b8] transition-all duration-200 icon-hover-parent"
          style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
        >
          <AnimatedIcon :icon="Upload" :size="14" :stroke-width="2" />
          Upload Video
        </RouterLink>
      </div>
    </div>

    <!-- API error banner -->
    <div
      v-if="fetchError"
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
      <span class="text-[#ef4444] flex-1">{{ fetchError }}</span>
      <button class="text-[#ef4444] text-xs underline" @click="fetchRuns">Retry</button>
    </div>

    <!-- Stats row -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <div
        v-for="stat in [
          { label: 'Total Runs',  value: String(runs.length),                                                  accent: 'teal'  },
          { label: 'Processing',  value: String(runs.filter(r => r.status === 'processing').length),           accent: 'amber' },
          { label: 'Events',      value: String(runs.reduce((a, r) => a + (r.event_count ?? 0), 0)),           accent: 'warm'  },
          { label: 'Completed',   value: String(runs.filter(r => r.status === 'completed').length),            accent: 'green' },
        ]"
        :key="stat.label"
        class="card-skeuo p-4"
      >
        <div class="text-[#64748b] text-xs font-medium uppercase tracking-wider mb-2">{{ stat.label }}</div>
        <div
          class="mono text-2xl font-semibold"
          :class="{
            'text-[#2dd4bf]': stat.accent === 'teal',
            'text-[#f59e0b]': stat.accent === 'amber' || stat.accent === 'warm',
            'text-[#22c55e]': stat.accent === 'green',
          }"
        >{{ stat.value }}</div>
      </div>
    </div>

    <!-- Run list -->
    <div class="card-skeuo overflow-hidden">
      <div class="px-5 py-4 border-b border-[#1e2633] flex items-center justify-between">
        <h3 class="text-[#e2e8f0] font-medium text-sm">Recent Runs</h3>
        <span class="text-[#475569] text-xs mono">{{ runs.length }} total</span>
      </div>

      <div v-if="loading && runs.length === 0" class="flex items-center justify-center py-12">
        <div class="w-6 h-6 rounded-full border-2 border-[#1e2633] border-t-[#2dd4bf] animate-spin" />
      </div>

      <div v-else-if="!fetchError && runs.length === 0" class="py-12 text-center">
        <p class="text-[#475569] text-sm">No runs yet.</p>
        <RouterLink to="/stream" class="text-[#2dd4bf] text-sm mt-2 inline-block hover:text-[#5eead4]">
          Start your first stream →
        </RouterLink>
      </div>

      <div v-else class="divide-y divide-[#1e2633]">
        <button
          v-for="run in runs"
          :key="run.run_id"
          class="w-full flex items-center gap-4 px-5 py-3.5 text-left transition-colors duration-200 hover:bg-[rgba(255,255,255,0.02)] group"
          @click="router.push(`/runs/${run.run_id}`)"
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
            :style="run.status === 'processing'
              ? 'animation: pulse-teal 2s infinite; box-shadow: 0 0 6px rgba(245,158,11,0.5)'
              : ''"
          />
          <!-- Run ID + mode -->
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2">
              <span class="mono text-[#e2e8f0] text-sm font-medium truncate">{{ run.run_id }}</span>
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
          <AnimatedIcon
            :icon="ChevronRight"
            :size="14"
            :stroke-width="2"
            class="text-[#475569] shrink-0 group-hover:text-[#64748b] transition-colors"
          />
        </button>
      </div>
    </div>

    <!-- Recent SpacetimeDB events -->
    <div v-if="recentEvents.length > 0" class="card-skeuo overflow-hidden">
      <div class="px-5 py-4 border-b border-[#1e2633] flex items-center gap-2">
        <AnimatedIcon
          :icon="Zap"
          :size="12"
          :stroke-width="2"
          animation="pulse"
          class="text-[#2dd4bf] icon-glow-teal shrink-0"
        />
        <h3 class="text-[#e2e8f0] font-medium text-sm">Live Events</h3>
        <span class="badge badge-teal mono ml-auto">{{ recentEvents.length }}</span>
      </div>
      <div class="divide-y divide-[#1e2633] max-h-56 overflow-y-auto">
        <div
          v-for="evt in recentEvents"
          :key="`${evt.run_id}-${evt.frame_index}-${evt.event_type}`"
          class="px-5 py-2.5 flex items-center gap-3 cursor-pointer hover:bg-[rgba(255,255,255,0.02)] transition-colors"
          @click="router.push(`/runs/${evt.run_id}`)"
        >
          <div
            class="w-1.5 h-1.5 rounded-full shrink-0"
            :style="{ background: eventColor(evt.event_type) }"
          />
          <span class="text-[#94a3b8] text-xs flex-1 truncate">{{ evt.description || evt.event_type }}</span>
          <span class="mono text-[#475569] text-xs shrink-0">{{ evt.event_type.replace(/_/g, ' ') }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
