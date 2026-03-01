<script setup lang="ts">
import {
  ref,
  computed,
  onMounted,
  onUnmounted,
  watch,
  nextTick,
} from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useRunsStore } from '@/stores/runs'
import { useEventsStore } from '@/stores/events'
import { useEventStream } from '@/composables/useEventStream'
import { api, ApiError } from '@/lib/api'
import { useAuthStore } from '@/stores/auth'
import type { FeedbackRequest, RawRunEvent } from '@/lib/api'
import type { RunStatus } from '@/stores/runs'
import type { AgentEvent } from '@/stores/events'
import { ChevronLeft, Image, Radio, Zap } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'
import ProcessingVisualization from '@/components/stream/ProcessingVisualization.vue'

const route = useRoute()
const router = useRouter()
const runsStore = useRunsStore()
const eventsStore = useEventsStore()
const authStore = useAuthStore()
const { connect: connectEvents, disconnect: disconnectEvents, isConnected: stdbConnected } = useEventStream()

const runId = computed(() => route.params.runId as string)
const activeTab = ref<'player' | 'keyframes' | 'metadata'>('player')
const loading = ref(false)
const fetchError = ref<string | null>(null)
const actionLoading = ref<'stop' | 'delete' | 'keepalive' | null>(null)
const confirmDelete = ref(false)
let keepaliveTimer: ReturnType<typeof setInterval> | null = null
const eventsPollingTimer = ref<ReturnType<typeof setInterval> | null>(null)

// ── Video player state ────────────────────────────────────────────────────────

const videoEl = ref<HTMLVideoElement | null>(null)
const timelineEl = ref<HTMLDivElement | null>(null)
const videoDuration = ref(0)
const videoCurrentTime = ref(0)
const videoReady = ref(false)
const videoError = ref<string | null>(null)

/** Source URI from the run, mapped to an HTTP URL for the browser */
const videoSrc = computed((): string | null => {
  const src = run.value?.source_uri
  if (!src) return null
  // Already an HTTP(S) URL — use directly.
  if (src.startsWith('http://') || src.startsWith('https://')) return src
  // file:// path from Hetzner server — map to the API file serving endpoint.
  // Strip any leading "file://" prefix and extract the bare filename.
  const stripped = src.replace(/^file:\/\//, '')
  const filename = stripped.split('/').pop()
  if (!filename) return null
  return `${authStore.apiEndpoint}/v1/files/${encodeURIComponent(filename)}`
})

function onVideoMetadata() {
  if (!videoEl.value) return
  videoDuration.value = videoEl.value.duration * 1000 // store in ms
  videoReady.value = true
}

function onVideoTimeUpdate() {
  if (!videoEl.value) return
  videoCurrentTime.value = videoEl.value.currentTime * 1000 // in ms
}

function onVideoError() {
  videoError.value = 'Video could not be loaded. The file may not be accessible from the server.'
}

/** Seek the video element to a given time in milliseconds. */
function seekTo(ms: number) {
  if (!videoEl.value || !videoReady.value) return
  videoEl.value.currentTime = ms / 1000
}

// ── Timeline scrubber ─────────────────────────────────────────────────────────

/** Events sorted by pts_ms for timeline rendering. */
const timelineEvents = computed<AgentEvent[]>(() =>
  [...eventsStore.eventsForRun(runId.value)].sort((a, b) => a.pts_ms - b.pts_ms)
)

/**
 * Effective duration used for the scrubber.
 * Prefer the actual video duration; fall back to the last event's pts_ms.
 */
const effectiveDuration = computed<number>(() => {
  if (videoDuration.value > 0) return videoDuration.value
  const last = timelineEvents.value[timelineEvents.value.length - 1]
  return last ? last.pts_ms + 2000 : 60000
})

function markerLeftPct(pts: number): number {
  const dur = effectiveDuration.value
  if (dur <= 0) return 0
  return Math.min(100, Math.max(0, (pts / dur) * 100))
}

function playheadLeftPct(): number {
  return markerLeftPct(videoCurrentTime.value)
}

/** Clicking on the timeline scrubber bar seeks proportionally. */
function onTimelineClick(evt: MouseEvent) {
  if (!timelineEl.value) return
  const rect = timelineEl.value.getBoundingClientRect()
  const pct = (evt.clientX - rect.left) / rect.width
  const targetMs = pct * effectiveDuration.value
  seekTo(Math.max(0, targetMs))
}

// ── Event highlighting sync ───────────────────────────────────────────────────

/** The event closest to the current video time (within 2 s tolerance). */
const activeEventIndex = computed<number>(() => {
  const t = videoCurrentTime.value
  let bestIdx = -1
  let bestDiff = Infinity
  timelineEvents.value.forEach((evt, i) => {
    const diff = Math.abs(evt.pts_ms - t)
    if (diff < bestDiff && diff < 5000) {
      bestDiff = diff
      bestIdx = i
    }
  })
  return bestIdx
})

const activeEvent = computed<AgentEvent | null>(() =>
  activeEventIndex.value >= 0 ? timelineEvents.value[activeEventIndex.value] : null
)

// Auto-scroll the event list to keep the active event visible.
const eventListEl = ref<HTMLDivElement | null>(null)
watch(activeEventIndex, async (newIdx) => {
  if (newIdx < 0 || !eventListEl.value) return
  await nextTick()
  const row = eventListEl.value.querySelectorAll<HTMLElement>('[data-event-row]')[newIdx]
  if (row) {
    row.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }
})

// ── Keyframe canvas extraction ────────────────────────────────────────────────

interface KeyframeThumbnail {
  pts_ms: number
  event_type: string
  description: string
  dataUrl: string
}

const extractedKeyframes = ref<KeyframeThumbnail[]>([])
const extracting = ref(false)

/**
 * Seek the hidden video element to each marker's pts_ms,
 * capture the frame via an off-screen canvas, and build thumbnail data URLs.
 */
async function extractKeyframes(): Promise<void> {
  if (!videoEl.value || !videoReady.value || extracting.value) return
  extracting.value = true
  extractedKeyframes.value = []

  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    extracting.value = false
    return
  }

  // Use a copy of the video element's src on a hidden video for frame extraction
  // to avoid interrupting playback.
  const scratchVideo = document.createElement('video')
  scratchVideo.src = videoEl.value.src
  scratchVideo.muted = true
  scratchVideo.preload = 'auto'
  scratchVideo.crossOrigin = 'anonymous'

  await new Promise<void>((resolve) => {
    scratchVideo.onloadedmetadata = () => resolve()
    scratchVideo.onerror = () => resolve()
    scratchVideo.load()
  })

  canvas.width = 320
  canvas.height = 180

  // Collect unique pts_ms values from all events (cap at 30 to avoid long extraction).
  const markers = timelineEvents.value.slice(0, 30)
  for (const evt of markers) {
    try {
      scratchVideo.currentTime = evt.pts_ms / 1000
      await new Promise<void>((resolve) => {
        const handler = () => {
          scratchVideo.removeEventListener('seeked', handler)
          resolve()
        }
        scratchVideo.addEventListener('seeked', handler)
        setTimeout(resolve, 1000) // timeout guard
      })
      ctx.drawImage(scratchVideo, 0, 0, 320, 180)
      const dataUrl = canvas.toDataURL('image/jpeg', 0.8)
      extractedKeyframes.value.push({
        pts_ms: evt.pts_ms,
        event_type: evt.event_type,
        description: evt.description,
        dataUrl,
      })
    } catch {
      // Skip frames that can't be captured
    }
  }
  extracting.value = false
}

// ── Run data ──────────────────────────────────────────────────────────────────

const run = computed(() =>
  runsStore.activeRun ?? runsStore.runs.find(r => r.run_id === runId.value)
)
const events = computed(() => eventsStore.eventsForRun(runId.value))
const keyframes = computed(() => eventsStore.keyframesForRun(runId.value))

const isProcessing = computed(() =>
  run.value?.status === 'processing' || run.value?.status === 'pending'
)

// ── Visualization props derived from run + event data ─────────────────────

const vizFps = computed(() => {
  const stored = Number(localStorage.getItem('vidarax_fps') ?? '5')
  return stored > 0 ? stored : 5
})

const vizTotalFrames = computed(() => {
  // activeRun (type Run) has frame_count; RunSummary does not
  const fc = runsStore.activeRun?.frame_count
  if (fc && fc > 0) return fc
  if (events.value.length === 0) return 0
  const maxFrame = Math.max(...events.value.map(e => e.frame_index))
  return maxFrame + 1
})

const vizProcessedFrames = computed(() => {
  if (!isProcessing.value && events.value.length > 0) return vizTotalFrames.value
  if (events.value.length === 0) return 0
  return Math.max(...events.value.map(e => e.frame_index))
})

const vizChunkStart = computed(() => {
  if (events.value.length === 0) return 0
  const sorted = [...events.value].sort((a, b) => a.frame_index - b.frame_index)
  const last = sorted[sorted.length - 1]
  const chunkSize = Number(localStorage.getItem('vidarax_chunk_size') ?? '30')
  return Math.max(0, (last?.frame_index ?? 0) - chunkSize)
})

const vizChunkEnd = computed(() => {
  if (events.value.length === 0) return 0
  return Math.max(...events.value.map(e => e.frame_index))
})

const vizVlmFrames = computed(() =>
  events.value
    .filter(e => e.event_type === 'vlm_description')
    .map(e => e.frame_index)
)

const vizKfIndices = computed(() =>
  events.value
    .filter(e => e.event_type === 'keyframe')
    .map(e => e.frame_index)
)

// ── Lifecycle ─────────────────────────────────────────────────────────────────

onMounted(async () => {
  eventsStore.setActiveRunId(runId.value)

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

  connectEvents(runId.value)

  setTimeout(() => {
    if (!stdbConnected.value) {
      startEventsPolling(runId.value)
    }
  }, 200)

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
  if (eventsPollingTimer.value) clearInterval(eventsPollingTimer.value)
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

// ── HTTP events fallback ─────────────────────────────────────────────────────

function mapRawEvent(raw: RawRunEvent, runIdVal: string): AgentEvent | null {
  if (raw.kind !== 'marker_emitted') return null
  const p = raw.payload
  const eventType = (p.event_type as string | undefined)
    ?? (p.kind as string | undefined)
    ?? 'vlm_description'
  return {
    run_id: runIdVal,
    session_id: (p.stream_id as string | undefined) ?? '',
    frame_index: Number((p.start_frame as number | undefined) ?? 0),
    pts_ms: Number(raw.pts_ms ?? 0),
    event_type: eventType as AgentEvent['event_type'],
    confidence: Number((p.confidence as number | undefined) ?? 0),
    description: (p.description as string | undefined) ?? (p.summary as string | undefined) ?? '',
    timestamp_ms: 0,
  }
}

function startEventsPolling(id: string): void {
  const TERMINAL = new Set<string>(['completed', 'failed', 'stopped'])

  async function poll(): Promise<void> {
    try {
      const res = await api.runs.events(id)
      const mapped = res.events
        .map(e => mapRawEvent(e, id))
        .filter((e): e is AgentEvent => e !== null)
      const existing = new Set(
        eventsStore.eventsForRun(id).map(e => `${e.frame_index}:${e.event_type}`)
      )
      for (const evt of mapped) {
        const key = `${evt.frame_index}:${evt.event_type}`
        if (!existing.has(key)) {
          eventsStore.addEvent(evt)
          existing.add(key)
        }
      }
      const runData = await api.runs.get(id)
      runsStore.updateRunStatus(id, runData.status as RunStatus)
      if (TERMINAL.has(runData.status) && eventsPollingTimer.value !== null) {
        clearInterval(eventsPollingTimer.value)
        eventsPollingTimer.value = null
      }
    } catch {
      // transient — keep polling
    }
  }

  poll()
  eventsPollingTimer.value = setInterval(poll, 3000)
}

// ── Feedback ──────────────────────────────────────────────────────────────────

const showFeedback = ref(false)
const feedbackRating = ref(-1)
const feedbackCategory = ref('quality')
const feedbackText = ref('')
const feedbackSubmitting = ref(false)
const feedbackSubmitted = ref(false)

const FEEDBACK_CATEGORIES = ['accuracy', 'latency', 'quality'] as const

async function submitFeedback(): Promise<void> {
  if (feedbackSubmitting.value) return
  feedbackSubmitting.value = true
  try {
    const data: FeedbackRequest = {
      rating: feedbackRating.value,
      category: feedbackCategory.value,
      feedback: feedbackText.value || undefined,
    }
    await api.runs.feedback(runId.value, data)
    feedbackSubmitted.value = true
    showFeedback.value = false
  } catch (err) {
    console.error('[RunDetail] feedback submit failed:', err)
  } finally {
    feedbackSubmitting.value = false
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
  scene_cut:          '#2dd4bf',
  loop_detected:      '#f59e0b',
  vlm_description:    '#22c55e',
  artifact_suspected: '#ef4444',
  exposure_shift:     '#a78bfa',
  flicker:            '#fb7185',
  keyframe:           '#22c55e',
}
function eventColor(type: string) { return EVENT_COLORS[type] ?? '#64748b' }

function formatPts(ms: number): string {
  if (ms > 1_577_836_800_000) {
    return new Date(ms).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }
  const totalSec = Math.floor(ms / 1000)
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  const frac = Math.floor((ms % 1000) / 100)
  return min > 0 ? `${min}:${String(sec).padStart(2, '0')}.${frac}` : `${sec}.${frac}s`
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: 'medium', timeStyle: 'short',
  })
}

function formatTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000)
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  return `${min}:${String(sec).padStart(2, '0')}`
}
</script>

<template>
  <div class="p-4 lg:p-6 space-y-4">

    <!-- Loading skeleton -->
    <div v-if="loading" class="flex items-center justify-center py-20">
      <div class="w-6 h-6 rounded-full border-2 border-[#1e2633] border-t-[#2dd4bf] animate-spin" />
    </div>

    <!-- Fetch error -->
    <div v-else-if="fetchError && !run" class="flex items-center justify-center py-20">
      <div class="text-center space-y-3">
        <div class="text-[#ef4444] text-sm">{{ fetchError }}</div>
        <RouterLink to="/" class="text-[#2dd4bf] text-sm inline-block hover:text-[#5eead4]">
          &larr; Back to Dashboard
        </RouterLink>
      </div>
    </div>

    <template v-else-if="run">

      <!-- Header ────────────────────────────────────────────────────────────── -->
      <div class="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <div class="flex items-center gap-2 mb-1">
            <RouterLink
              to="/"
              class="text-[#475569] hover:text-[#64748b] transition-colors icon-hover-parent"
              aria-label="Back to Dashboard"
            >
              <AnimatedIcon :icon="ChevronLeft" :size="16" :stroke-width="2" />
            </RouterLink>
            <span :class="['badge', statusConfig(run.status as RunStatus).cls]">
              {{ statusConfig(run.status as RunStatus).label }}
            </span>
            <AnimatedIcon
              v-if="isProcessing"
              :icon="Radio"
              :size="14"
              :stroke-width="2"
              animation="pulse"
              class="text-[#f59e0b] icon-glow-amber"
            />
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

      <!-- Tabs ─────────────────────────────────────────────────────────────── -->
      <div class="flex gap-1 p-1 rounded-[10px] w-fit" style="background: #0f1117; border: 1px solid #1e2633;">
        <button
          v-for="tab in ['player', 'keyframes', 'metadata'] as const"
          :key="tab"
          class="px-4 py-1.5 rounded-[8px] text-sm font-medium transition-all duration-200 capitalize"
          :class="activeTab === tab ? 'text-[#e2e8f0]' : 'text-[#475569] hover:text-[#64748b]'"
          :style="activeTab === tab
            ? 'background: #151a22; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 2px 8px rgba(0,0,0,0.4);'
            : ''"
          @click="activeTab = tab"
        >
          {{ tab === 'player' ? 'Player' : tab === 'keyframes' ? 'Keyframes' : 'Metadata' }}
          <span v-if="tab === 'player' && events.length > 0" class="ml-1 badge badge-muted">
            {{ events.length }}
          </span>
          <span v-if="tab === 'keyframes' && (keyframes.length > 0 || extractedKeyframes.length > 0)" class="ml-1 badge badge-teal">
            {{ keyframes.length > 0 ? keyframes.length : extractedKeyframes.length }}
          </span>
        </button>
      </div>

      <!-- Tab: Player (video + marker timeline + event list) ─────────────── -->
      <div v-if="activeTab === 'player'" class="space-y-3">

        <!-- Stream processing visualization (shown while processing, collapses when done) -->
        <Transition name="pv-collapse">
          <ProcessingVisualization
            v-if="isProcessing && vizTotalFrames > 0"
            :total-frames="vizTotalFrames"
            :processed-frames="vizProcessedFrames"
            :keyframe-count="keyframes.length"
            :event-count="events.length"
            :current-chunk-start="vizChunkStart"
            :current-chunk-end="vizChunkEnd"
            :fps="vizFps"
            :is-processing="isProcessing"
            :vlm-frames="vizVlmFrames"
            :keyframe-indices="vizKfIndices"
          />
        </Transition>

        <!-- Video player -->
        <div class="card-skeuo overflow-hidden" style="background: #050507;">
          <!-- No source_uri -->
          <div v-if="!videoSrc" class="flex items-center justify-center py-16 text-[#475569] text-sm">
            <div class="text-center space-y-2">
              <AnimatedIcon :icon="Image" :size="32" :stroke-width="1" class="mx-auto text-[#1e2633]" />
              <p>No video source available for this run.</p>
              <p class="text-xs text-[#334155]">Upload a video and run analysis to see it here.</p>
            </div>
          </div>

          <!-- Video element -->
          <div v-else>
            <video
              ref="videoEl"
              :src="videoSrc"
              controls
              class="w-full block"
              style="max-height: 480px; background: #000;"
              preload="metadata"
              @loadedmetadata="onVideoMetadata"
              @timeupdate="onVideoTimeUpdate"
              @error="onVideoError"
            />

            <!-- Video load error -->
            <div
              v-if="videoError"
              class="px-5 py-3 text-xs text-[#f59e0b]"
              style="background: rgba(245,158,11,0.06); border-top: 1px solid rgba(245,158,11,0.15);"
            >
              {{ videoError }}
            </div>

            <!-- Marker timeline scrubber -->
            <div class="px-4 py-3" style="border-top: 1px solid #1e2633;">
              <div class="flex items-center justify-between mb-2 text-xs text-[#475569]">
                <span class="mono">{{ formatTime(videoCurrentTime) }}</span>
                <span class="text-[#334155] text-[10px] tracking-wider uppercase">Marker Timeline</span>
                <span class="mono">{{ formatTime(effectiveDuration) }}</span>
              </div>

              <!-- Timeline bar -->
              <div
                ref="timelineEl"
                class="relative w-full cursor-crosshair select-none"
                style="height: 32px;"
                role="slider"
                :aria-valuenow="videoCurrentTime"
                :aria-valuemin="0"
                :aria-valuemax="effectiveDuration"
                aria-label="Video timeline scrubber"
                @click="onTimelineClick"
              >
                <!-- Track -->
                <div
                  class="absolute inset-x-0 top-1/2 -translate-y-1/2 rounded-full"
                  style="height: 4px; background: #1e2633;"
                />

                <!-- Progress fill -->
                <div
                  class="absolute left-0 top-1/2 -translate-y-1/2 rounded-full"
                  style="height: 4px; background: rgba(45,212,191,0.3); transition: width 0.1s linear;"
                  :style="{ width: `${playheadLeftPct()}%` }"
                />

                <!-- Marker dots -->
                <button
                  v-for="(evt, i) in timelineEvents"
                  :key="`marker-${i}`"
                  class="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full transition-all duration-100 focus:outline-none"
                  :class="activeEventIndex === i ? 'w-3.5 h-3.5 z-20' : 'w-2 h-2 z-10 hover:w-3 hover:h-3'"
                  :style="{
                    left: `${markerLeftPct(evt.pts_ms)}%`,
                    background: eventColor(evt.event_type),
                    boxShadow: activeEventIndex === i
                      ? `0 0 8px ${eventColor(evt.event_type)}`
                      : `0 0 4px ${eventColor(evt.event_type)}66`,
                    border: activeEventIndex === i ? `2px solid #e2e8f0` : 'none',
                  }"
                  :title="`${evt.event_type.replace(/_/g, ' ')} @ ${formatPts(evt.pts_ms)}`"
                  :aria-label="`Seek to ${evt.event_type} at ${formatPts(evt.pts_ms)}`"
                  @click.stop="seekTo(evt.pts_ms)"
                />

                <!-- Playhead -->
                <div
                  v-if="videoReady"
                  class="absolute top-0 bottom-0 w-px pointer-events-none z-30"
                  style="background: #2dd4bf; box-shadow: 0 0 6px #2dd4bf;"
                  :style="{ left: `${playheadLeftPct()}%` }"
                >
                  <div
                    class="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full"
                    style="background: #2dd4bf; box-shadow: 0 0 6px #2dd4bf;"
                  />
                </div>
              </div>

              <!-- Legend -->
              <div class="flex flex-wrap gap-x-4 gap-y-1 mt-2">
                <div
                  v-for="(color, type) in EVENT_COLORS"
                  :key="type"
                  class="flex items-center gap-1.5"
                >
                  <div
                    class="w-2 h-2 rounded-full shrink-0"
                    :style="`background: ${color}; box-shadow: 0 0 4px ${color}66;`"
                  />
                  <span class="text-[10px] text-[#475569]">{{ type.replace(/_/g, ' ') }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Bottom split: event list + current frame detail -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-3">

          <!-- Event list (synced with video time) -->
          <div class="card-skeuo overflow-hidden" style="min-height: 300px;">
            <div class="px-5 py-3 border-b border-[#1e2633] flex items-center justify-between">
              <h3 class="text-[#e2e8f0] font-medium text-sm">Events</h3>
              <div class="flex items-center gap-2">
                <span
                  v-if="!stdbConnected && eventsPollingTimer !== null"
                  class="text-xs text-[#475569]"
                  title="SpacetimeDB not connected — polling REST API every 3s"
                >
                  polling
                </span>
                <span v-if="isProcessing" class="flex items-center gap-1.5 text-xs text-[#f59e0b]">
                  <AnimatedIcon
                    :icon="Zap"
                    :size="12"
                    :stroke-width="2"
                    animation="pulse"
                    class="icon-glow-amber"
                  />
                  {{ stdbConnected ? 'Live' : 'Live (HTTP)' }}
                </span>
              </div>
            </div>

            <div v-if="events.length === 0" class="py-12 text-center text-[#475569] text-sm">
              <span v-if="isProcessing">Waiting for events…</span>
              <span v-else>No events recorded for this run.</span>
            </div>

            <div
              v-else
              ref="eventListEl"
              class="divide-y divide-[#1e2633] overflow-y-auto"
              style="max-height: 380px;"
            >
              <button
                v-for="(evt, i) in timelineEvents"
                :key="`evt-${i}`"
                data-event-row
                class="w-full flex items-center gap-3 px-5 py-2.5 text-left transition-colors focus:outline-none"
                :class="activeEventIndex === i
                  ? 'bg-[rgba(45,212,191,0.06)]'
                  : 'hover:bg-[rgba(255,255,255,0.02)]'"
                :style="activeEventIndex === i ? 'border-left: 2px solid #2dd4bf;' : 'border-left: 2px solid transparent;'"
                @click="seekTo(evt.pts_ms)"
              >
                <!-- Color dot -->
                <div
                  class="w-2 h-2 rounded-full shrink-0"
                  :style="`background: ${eventColor(evt.event_type)}; box-shadow: 0 0 6px ${eventColor(evt.event_type)}66;`"
                />
                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-2 flex-wrap">
                    <span class="mono text-xs text-[#475569]">f{{ evt.frame_index }}</span>
                    <span class="mono text-xs font-medium" :style="`color: ${eventColor(evt.event_type)}`">
                      {{ evt.event_type.replace(/_/g, ' ') }}
                    </span>
                    <span class="mono text-xs text-[#f59e0b]">{{ (evt.confidence * 100).toFixed(0) }}%</span>
                  </div>
                  <p v-if="evt.description" class="text-[#94a3b8] text-xs mt-0.5 truncate leading-snug">
                    {{ evt.description }}
                  </p>
                </div>
                <span class="mono text-xs text-[#475569] shrink-0 hidden sm:block">
                  {{ formatPts(evt.pts_ms) }}
                </span>
              </button>
            </div>
          </div>

          <!-- Current frame detail panel -->
          <div class="card-skeuo p-5" style="min-height: 300px;">
            <h3 class="text-[#e2e8f0] font-medium text-sm mb-4">Current Frame</h3>

            <!-- No active event -->
            <div v-if="!activeEvent" class="flex flex-col items-center justify-center h-48 text-center">
              <div class="text-[#1e2633] mb-3">
                <AnimatedIcon :icon="Image" :size="40" :stroke-width="1" />
              </div>
              <p class="text-[#475569] text-sm">
                {{ videoReady ? 'Play the video to see frame details.' : 'Load a video to explore frame data.' }}
              </p>
            </div>

            <!-- Active event details -->
            <div v-else class="space-y-4">
              <!-- Event type badge + timestamp -->
              <div class="flex items-center justify-between">
                <div
                  class="inline-flex items-center gap-2 px-3 py-1.5 rounded-[8px]"
                  :style="`background: ${eventColor(activeEvent.event_type)}14; border: 1px solid ${eventColor(activeEvent.event_type)}33;`"
                >
                  <div
                    class="w-2 h-2 rounded-full"
                    :style="`background: ${eventColor(activeEvent.event_type)}; box-shadow: 0 0 6px ${eventColor(activeEvent.event_type)};`"
                  />
                  <span
                    class="mono text-xs font-semibold"
                    :style="`color: ${eventColor(activeEvent.event_type)}`"
                  >
                    {{ activeEvent.event_type.replace(/_/g, ' ') }}
                  </span>
                </div>
                <span class="mono text-xs text-[#475569]">{{ formatPts(activeEvent.pts_ms) }}</span>
              </div>

              <!-- Confidence bar -->
              <div>
                <div class="flex items-center justify-between mb-1.5">
                  <span class="text-xs text-[#64748b]">Confidence</span>
                  <span class="mono text-xs text-[#f59e0b]">{{ (activeEvent.confidence * 100).toFixed(1) }}%</span>
                </div>
                <div class="w-full rounded-full overflow-hidden" style="height: 4px; background: #1e2633;">
                  <div
                    class="h-full rounded-full transition-all duration-300"
                    style="background: #f59e0b;"
                    :style="{ width: `${activeEvent.confidence * 100}%` }"
                  />
                </div>
              </div>

              <!-- Frame index -->
              <div class="grid grid-cols-2 gap-3">
                <div class="rounded-[8px] p-3" style="background: #050507; border: 1px solid #1e2633;">
                  <div class="text-[10px] text-[#475569] uppercase tracking-wider mb-1">Frame</div>
                  <div class="mono text-sm text-[#94a3b8]">{{ activeEvent.frame_index }}</div>
                </div>
                <div class="rounded-[8px] p-3" style="background: #050507; border: 1px solid #1e2633;">
                  <div class="text-[10px] text-[#475569] uppercase tracking-wider mb-1">PTS</div>
                  <div class="mono text-sm text-[#94a3b8]">{{ activeEvent.pts_ms.toLocaleString() }} ms</div>
                </div>
              </div>

              <!-- VLM description -->
              <div v-if="activeEvent.description">
                <div class="text-[10px] text-[#475569] uppercase tracking-wider mb-2">Description</div>
                <p
                  class="text-[#94a3b8] text-xs leading-relaxed rounded-[8px] p-3"
                  style="background: #050507; border: 1px solid #1e2633;"
                >
                  {{ activeEvent.description }}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Tab: Keyframes ───────────────────────────────────────────────────── -->
      <div v-if="activeTab === 'keyframes'" class="card-skeuo p-5 space-y-4">
        <div class="flex items-center justify-between">
          <h3 class="text-[#e2e8f0] font-medium text-sm">
            Keyframe Gallery
            <span v-if="keyframes.length > 0" class="ml-2 badge badge-teal">{{ keyframes.length }}</span>
            <span v-else-if="extractedKeyframes.length > 0" class="ml-2 badge badge-teal">{{ extractedKeyframes.length }}</span>
          </h3>
          <!-- Extract button: uses the video element to grab canvas snapshots -->
          <button
            v-if="videoReady && events.length > 0 && extractedKeyframes.length === 0 && keyframes.length === 0"
            class="px-3 py-1.5 rounded-[8px] text-xs font-medium transition-all duration-200"
            style="background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.2); color: #2dd4bf;"
            :disabled="extracting"
            @click="extractKeyframes"
          >
            <span v-if="extracting">Extracting…</span>
            <span v-else>Extract Frames</span>
          </button>
        </div>

        <!-- SpacetimeDB keyframes (real jpeg_b64 data) -->
        <div v-if="keyframes.length > 0" class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          <div
            v-for="kf in keyframes"
            :key="kf.id"
            class="relative rounded-[10px] overflow-hidden group cursor-pointer"
            style="background: #050507; border: 1px solid #1e2633; aspect-ratio: 16/9;"
            @click="seekTo(kf.pts_ms); activeTab = 'player'"
          >
            <img
              v-if="kf.jpeg_b64"
              :src="`data:image/jpeg;base64,${kf.jpeg_b64}`"
              :alt="`Keyframe at frame ${kf.frame_index}`"
              class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              loading="lazy"
            />
            <div
              v-else
              class="absolute inset-0 flex items-center justify-center text-[#1e2633]"
            >
              <AnimatedIcon :icon="Image" :size="24" :stroke-width="1" />
            </div>
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

        <!-- Canvas-extracted keyframe thumbnails -->
        <div v-else-if="extractedKeyframes.length > 0" class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          <div
            v-for="(kf, i) in extractedKeyframes"
            :key="`extracted-${i}`"
            class="relative rounded-[10px] overflow-hidden group cursor-pointer"
            style="background: #050507; border: 1px solid #1e2633; aspect-ratio: 16/9;"
            @click="seekTo(kf.pts_ms); activeTab = 'player'"
          >
            <img
              :src="kf.dataUrl"
              :alt="`Frame at ${formatPts(kf.pts_ms)}`"
              class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              loading="lazy"
            />
            <div
              class="absolute bottom-0 left-0 right-0 p-2"
              style="background: linear-gradient(0deg, rgba(5,5,7,0.9) 0%, transparent 100%);"
            >
              <p v-if="kf.description" class="text-[#94a3b8] text-xs line-clamp-2 leading-snug">
                {{ kf.description }}
              </p>
              <div class="flex items-center justify-between mt-1">
                <span class="mono text-[#475569] text-[10px]">{{ formatPts(kf.pts_ms) }}</span>
                <span
                  class="mono text-[10px]"
                  :style="`color: ${eventColor(kf.event_type)}`"
                >
                  {{ kf.event_type.replace(/_/g, ' ') }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Empty state -->
        <div v-else class="py-10 text-center text-[#475569] text-sm space-y-2">
          <div v-if="isProcessing">Keyframes will appear here as analysis runs…</div>
          <template v-else-if="events.length > 0 && !videoReady">
            <p>Load a video in the Player tab to enable keyframe extraction.</p>
          </template>
          <template v-else-if="events.length === 0">
            <p>No events recorded for this run.</p>
          </template>
          <template v-else>
            <p>Click <strong class="text-[#94a3b8]">Extract Frames</strong> to capture thumbnails from the video.</p>
            <p class="text-xs text-[#334155]">Opens the Player tab and seeks to each event timestamp.</p>
          </template>
          <div v-if="extracting" class="flex justify-center pt-2">
            <div class="w-4 h-4 rounded-full border-2 border-[#1e2633] border-t-[#2dd4bf] animate-spin" />
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

      <!-- Feedback ─────────────────────────────────────────────────────────── -->
      <div v-if="!isProcessing && !feedbackSubmitted" class="flex justify-end">
        <button
          class="px-4 py-2 rounded-[8px] text-xs font-medium transition-all duration-200"
          style="background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.2); color: #2dd4bf;"
          @click="showFeedback = !showFeedback"
        >
          {{ showFeedback ? 'Cancel' : 'Rate this analysis' }}
        </button>
      </div>

      <div v-if="feedbackSubmitted" class="flex justify-end">
        <span class="text-xs text-[#2dd4bf]">Feedback submitted — thank you</span>
      </div>

      <div v-if="showFeedback" class="card-skeuo p-5 space-y-4">
        <h3 class="text-[#e2e8f0] font-medium text-sm">Rate this analysis</h3>

        <div>
          <label class="text-xs text-[#64748b] block mb-2">Rating</label>
          <div class="flex gap-1 items-center">
            <button
              v-for="n in 11"
              :key="n - 1"
              class="w-7 h-7 rounded-[6px] text-xs font-medium mono transition-all duration-150"
              :style="feedbackRating === n - 1
                ? 'background: rgba(45,212,191,0.15); border: 1px solid #2dd4bf; color: #2dd4bf; box-shadow: 0 0 8px rgba(45,212,191,0.2);'
                : 'background: rgba(255,255,255,0.04); border: 1px solid #1e2633; color: #475569;'"
              @click="feedbackRating = n - 1"
            >
              {{ n - 1 }}
            </button>
          </div>
        </div>

        <div>
          <label class="text-xs text-[#64748b] block mb-2">Category</label>
          <select
            v-model="feedbackCategory"
            class="rounded-[8px] px-3 py-1.5 text-xs mono w-full max-w-[200px]"
            style="background: #050507; border: 1px solid #1e2633; color: #e2e8f0; outline: none;"
          >
            <option v-for="cat in FEEDBACK_CATEGORIES" :key="cat" :value="cat">{{ cat }}</option>
          </select>
        </div>

        <div>
          <label class="text-xs text-[#64748b] block mb-2">Comments (optional)</label>
          <textarea
            v-model="feedbackText"
            rows="3"
            placeholder="Additional feedback…"
            class="rounded-[8px] px-3 py-2 text-xs w-full resize-none"
            style="background: #050507; border: 1px solid #1e2633; color: #e2e8f0; outline: none; font-family: 'JetBrains Mono', monospace;"
          />
        </div>

        <div class="flex justify-end">
          <button
            class="px-4 py-2 rounded-[8px] text-xs font-medium transition-all duration-200"
            style="background: rgba(45,212,191,0.12); border: 1px solid rgba(45,212,191,0.3); color: #2dd4bf;"
            :disabled="feedbackSubmitting || feedbackRating < 0"
            @click="submitFeedback"
          >
            <span v-if="feedbackSubmitting">Submitting…</span>
            <span v-else>Submit feedback</span>
          </button>
        </div>
      </div>

    </template>
  </div>
</template>

<style scoped>
/* Collapse transition for the processing visualization */
.pv-collapse-enter-active,
.pv-collapse-leave-active {
  transition: opacity 300ms ease, max-height 400ms cubic-bezier(0.4, 0, 0.2, 1), margin-bottom 300ms ease;
  max-height: 200px;
  overflow: hidden;
}
.pv-collapse-enter-from,
.pv-collapse-leave-to {
  opacity: 0;
  max-height: 0;
  margin-bottom: 0;
}
</style>
