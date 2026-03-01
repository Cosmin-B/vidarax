<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useRunsStore } from '@/stores/runs'
import { useEventsStore } from '@/stores/events'
import { useEventStream } from '@/composables/useEventStream'
import { api, ApiError } from '@/lib/api'
import type { RunStatus } from '@/stores/runs'

const route = useRoute()
const router = useRouter()
const runsStore = useRunsStore()
const eventsStore = useEventsStore()
const { connect: connectEvents, disconnect: disconnectEvents } = useEventStream()

const runId = computed(() => route.params.runId as string)
const activeTab = ref<'timeline' | 'keyframes' | 'metadata'>('timeline')
const loading = ref(false)
const fetchError = ref<string | null>(null)
const actionLoading = ref<'stop' | 'delete' | 'keepalive' | null>(null)
const confirmDelete = ref(false)
let keepaliveTimer: ReturnType<typeof setInterval> | null = null

// ── Run data ──────────────────────────────────────────────────────────────────

const run = computed(() =>
  runsStore.activeRun ?? runsStore.runs.find(r => r.run_id === runId.value)
)
const events = computed(() => eventsStore.eventsForRun(runId.value))
const keyframes = computed(() => eventsStore.keyframesForRun(runId.value))

const isProcessing = computed(() =>
  run.value?.status === 'processing' || run.value?.status === 'pending'
)

// ── Lifecycle ─────────────────────────────────────────────────────────────────

onMounted(async () => {
  eventsStore.setActiveRunId(runId.value)

  // Fetch full run details from API
  loading.value = true
  fetchError.value = null
  try {
    const data = await api.runs.get(runId.value)
    runsStore.setActiveRun({
      run_id: data.run_id,
      status: data.status as RunStatus,
      mode: data.mode as 'batch' | 'stream' | 'webrtc',
      model: data.model,
      source_uri: data.source_uri,
      created_at: data.created_at,
      updated_at: data.updated_at,
    })
  } catch (err) {
    fetchError.value = err instanceof ApiError
      ? `API ${err.status}: ${err.message}`
      : err instanceof Error ? err.message : 'Failed to load run'
  } finally {
    loading.value = false
  }

  // Subscribe to SpacetimeDB events for this run
  connectEvents(runId.value)

  // Keepalive for processing runs (every 20s)
  if (isProcessing.value) {
    keepaliveTimer = setInterval(async () => {
      if (!isProcessing.value) {
        clearInterval(keepaliveTimer!)
        keepaliveTimer = null
        return
      }
      try { await api.runs.keepalive(runId.value) } catch { /* non-fatal */ }
    }, 20000)
  }
})

onUnmounted(() => {
  disconnectEvents()
  runsStore.setActiveRun(null)
  eventsStore.setActiveRunId(null)
  if (keepaliveTimer) clearInterval(keepaliveTimer)
})

// ── Actions ───────────────────────────────────────────────────────────────────

async function stopRun(): Promise<void> {
  if (actionLoading.value) return
  actionLoading.value = 'stop'
  try {
    await api.runs.stop(runId.value)
    runsStore.updateRunStatus(runId.value, 'stopped')
  } catch (err) {
    console.error('[RunDetail] stop failed:', err)
  } finally {
    actionLoading.value = null
  }
}

async function deleteRun(): Promise<void> {
  if (actionLoading.value) return
  actionLoading.value = 'delete'
  try {
    await api.runs.delete(runId.value)
    runsStore.removeRun(runId.value)
    router.push('/')
  } catch (err) {
    console.error('[RunDetail] delete failed:', err)
    actionLoading.value = null
  }
}

// ── Display helpers ───────────────────────────────────────────────────────────

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

const EVENT_COLORS: Record<string, string> = {
  scene_cut: '#2dd4bf', loop_detected: '#f59e0b',
  vlm_description: '#22c55e', artifact_suspected: '#ef4444',
  exposure_shift: '#a78bfa', flicker: '#fb7185', keyframe: '#22c55e',
}
function eventColor(type: string) { return EVENT_COLORS[type] ?? '#64748b' }

function formatPts(ms: number): string {
  const s = Math.floor(ms / 1000)
  const rem = ms % 1000
  return `${s}.${String(rem).padStart(3, '0')}s`
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: 'medium', timeStyle: 'short',
  })
}
</script>

<template>
  <div class="p-4 lg:p-6 space-y-6">

    <!-- Loading skeleton -->
    <div v-if="loading" class="flex items-center justify-center py-20">
      <div class="w-6 h-6 rounded-full border-2 border-[#1e2633] border-t-[#2dd4bf] animate-spin" />
    </div>

    <!-- Fetch error -->
    <div v-else-if="fetchError && !run" class="flex items-center justify-center py-20">
      <div class="text-center space-y-3">
        <div class="text-[#ef4444] text-sm">{{ fetchError }}</div>
        <RouterLink to="/" class="text-[#2dd4bf] text-sm inline-block hover:text-[#5eead4]">
          ← Back to Dashboard
        </RouterLink>
      </div>
    </div>

    <template v-else-if="run">

      <!-- Header ─────────────────────────────────────────────────────────── -->
      <div class="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <div class="flex items-center gap-2 mb-1">
            <RouterLink
              to="/"
              class="text-[#475569] hover:text-[#64748b] transition-colors"
              aria-label="Back to Dashboard"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M10 3L6 8L10 13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </RouterLink>
            <span :class="['badge', statusConfig(run.status as RunStatus).cls]">
              {{ statusConfig(run.status as RunStatus).label }}
            </span>
            <span v-if="isProcessing" class="live-dot" style="width:6px; height:6px;" />
          </div>
          <h2 class="mono text-[#e2e8f0] font-semibold text-lg">{{ run.run_id }}</h2>
          <div class="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-xs text-[#64748b]">
            <span>Model: <span class="text-[#94a3b8] mono">{{ run.model }}</span></span>
            <span>Mode: <span class="text-[#94a3b8] mono">{{ run.mode }}</span></span>
            <span>Events: <span class="text-[#f59e0b] mono">{{ events.length }}</span></span>
            <span v-if="run.created_at">
              Created: <span class="text-[#94a3b8]">{{ formatDate(run.created_at) }}</span>
            </span>
          </div>
        </div>

        <!-- Actions -->
        <div class="flex gap-2 shrink-0 items-start">
          <!-- Stop -->
          <button
            v-if="isProcessing"
            class="px-3 py-1.5 rounded-[8px] text-xs font-medium transition-all duration-200"
            style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); color: #ef4444;"
            :disabled="actionLoading !== null"
            @click="stopRun"
          >
            <span v-if="actionLoading === 'stop'">Stopping…</span>
            <span v-else>Stop</span>
          </button>

          <!-- Delete with confirm -->
          <template v-if="!confirmDelete">
            <button
              class="px-3 py-1.5 rounded-[8px] text-xs font-medium text-[#64748b] transition-all duration-200"
              style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
              :disabled="actionLoading !== null"
              @click="confirmDelete = true"
            >
              Delete
            </button>
          </template>
          <template v-else>
            <span class="text-xs text-[#94a3b8] self-center">Confirm?</span>
            <button
              class="px-3 py-1.5 rounded-[8px] text-xs font-medium transition-all duration-200"
              style="background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); color: #ef4444;"
              :disabled="actionLoading !== null"
              @click="deleteRun"
            >
              <span v-if="actionLoading === 'delete'">Deleting…</span>
              <span v-else>Yes, Delete</span>
            </button>
            <button
              class="px-3 py-1.5 rounded-[8px] text-xs font-medium text-[#64748b]"
              style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
              @click="confirmDelete = false"
            >
              Cancel
            </button>
          </template>
        </div>
      </div>

      <!-- Tabs ────────────────────────────────────────────────────────────── -->
      <div class="flex gap-1 p-1 rounded-[10px] w-fit" style="background: #0f1117; border: 1px solid #1e2633;">
        <button
          v-for="tab in ['timeline', 'keyframes', 'metadata'] as const"
          :key="tab"
          class="px-4 py-1.5 rounded-[8px] text-sm font-medium transition-all duration-200 capitalize"
          :class="activeTab === tab ? 'text-[#e2e8f0]' : 'text-[#475569] hover:text-[#64748b]'"
          :style="activeTab === tab
            ? 'background: #151a22; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 2px 8px rgba(0,0,0,0.4);'
            : ''"
          @click="activeTab = tab"
        >
          {{ tab }}
          <span v-if="tab === 'timeline' && events.length > 0" class="ml-1 badge badge-muted">
            {{ events.length }}
          </span>
          <span v-if="tab === 'keyframes' && keyframes.length > 0" class="ml-1 badge badge-teal">
            {{ keyframes.length }}
          </span>
        </button>
      </div>

      <!-- Tab: Timeline ───────────────────────────────────────────────────── -->
      <div v-if="activeTab === 'timeline'" class="card-skeuo overflow-hidden">
        <div class="px-5 py-4 border-b border-[#1e2633] flex items-center justify-between">
          <h3 class="text-[#e2e8f0] font-medium text-sm">Event Timeline</h3>
          <span v-if="isProcessing" class="flex items-center gap-1.5 text-xs text-[#f59e0b]">
            <span class="live-dot" style="background: #f59e0b; width:6px; height:6px;" />
            Live
          </span>
        </div>

        <div v-if="events.length === 0" class="py-12 text-center text-[#475569] text-sm">
          <span v-if="isProcessing">Waiting for events…</span>
          <span v-else>No events recorded for this run.</span>
        </div>

        <div v-else class="divide-y divide-[#1e2633] max-h-[480px] overflow-y-auto">
          <div
            v-for="evt in events"
            :key="`${evt.frame_index}-${evt.event_type}-${evt.pts_ms}`"
            class="flex items-center gap-3 px-5 py-3 transition-colors hover:bg-[rgba(255,255,255,0.02)]"
          >
            <!-- Color dot with glow -->
            <div
              class="w-2.5 h-2.5 rounded-full shrink-0"
              :style="`background: ${eventColor(evt.event_type)}; box-shadow: 0 0 6px ${eventColor(evt.event_type)}66`"
            />
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2 flex-wrap">
                <span class="mono text-xs text-[#475569]">f{{ evt.frame_index }}</span>
                <span class="mono text-xs font-medium" :style="`color: ${eventColor(evt.event_type)}`">
                  {{ evt.event_type.replace(/_/g, ' ') }}
                </span>
                <span class="mono text-xs text-[#f59e0b]">{{ (evt.confidence * 100).toFixed(0) }}%</span>
              </div>
              <p v-if="evt.description" class="text-[#94a3b8] text-xs mt-0.5 truncate">
                {{ evt.description }}
              </p>
            </div>
            <span class="mono text-xs text-[#475569] shrink-0 hidden sm:block">
              {{ formatPts(evt.pts_ms) }}
            </span>
          </div>
        </div>
      </div>

      <!-- Tab: Keyframes ──────────────────────────────────────────────────── -->
      <div v-if="activeTab === 'keyframes'" class="card-skeuo p-5">
        <h3 class="text-[#e2e8f0] font-medium text-sm mb-4">
          Keyframe Gallery
          <span v-if="keyframes.length > 0" class="ml-2 badge badge-teal">{{ keyframes.length }}</span>
        </h3>

        <div v-if="keyframes.length === 0" class="py-10 text-center text-[#475569] text-sm">
          <span v-if="isProcessing">Keyframes will appear here as analysis runs…</span>
          <span v-else>No keyframes stored for this run.</span>
        </div>

        <div v-else class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          <div
            v-for="kf in keyframes"
            :key="kf.id"
            class="relative rounded-[10px] overflow-hidden group cursor-pointer"
            style="background: #050507; border: 1px solid #1e2633; aspect-ratio: 16/9;"
          >
            <img
              v-if="kf.jpeg_b64"
              :src="`data:image/jpeg;base64,${kf.jpeg_b64}`"
              :alt="`Keyframe at frame ${kf.frame_index}`"
              class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              loading="lazy"
            />
            <!-- Frame placeholder when no image -->
            <div
              v-else
              class="absolute inset-0 flex items-center justify-center text-[#1e2633]"
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <path d="M3 9l4-4 4 4 4-4 4 4"/>
              </svg>
            </div>
            <!-- Overlay -->
            <div
              class="absolute bottom-0 left-0 right-0 p-2"
              style="background: linear-gradient(0deg, rgba(5,5,7,0.9) 0%, transparent 100%);"
            >
              <p v-if="kf.description" class="text-[#94a3b8] text-xs line-clamp-2 leading-snug">
                {{ kf.description }}
              </p>
              <div class="flex items-center justify-between mt-1">
                <span class="mono text-[#475569] text-[10px]">f{{ kf.frame_index }}</span>
                <span
                  v-if="kf.event_type"
                  class="mono text-[10px]"
                  :style="`color: ${eventColor(kf.event_type)}`"
                >
                  {{ kf.event_type.replace(/_/g, ' ') }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab: Metadata ───────────────────────────────────────────────────── -->
      <div v-if="activeTab === 'metadata'" class="card-skeuo p-5">
        <h3 class="text-[#e2e8f0] font-medium text-sm mb-4">Raw Metadata</h3>
        <pre
          class="mono text-xs text-[#94a3b8] rounded-[8px] p-4 overflow-x-auto"
          style="background: #050507; border: 1px solid #1e2633; line-height: 1.7;"
        >{{ JSON.stringify(run, null, 2) }}</pre>
      </div>

    </template>
  </div>
</template>
