<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useStreamStore } from '@/stores/stream'
import { useEventsStore } from '@/stores/events'
import { useWhip } from '@/composables/useWhip'
import { useEventStream } from '@/composables/useEventStream'
import type { StreamSourceType } from '@/stores/stream'

const route = useRoute()
const router = useRouter()
const streamStore = useStreamStore()
const eventsStore = useEventsStore()

const { localStream, startStream, stopStream: stopWhip } = useWhip()
const { isConnected: stdbConnected, connect: connectEvents, disconnect: disconnectEvents } = useEventStream()

// ── Local state ───────────────────────────────────────────────────────────────

interface SourceTile {
  id: StreamSourceType
  label: string
  description: string
  available: boolean
  icon: string
}

const sources: SourceTile[] = [
  { id: 'screen',  label: 'Screen',   description: 'Capture your entire screen or a window', available: true,  icon: 'monitor'   },
  { id: 'camera',  label: 'Camera',   description: 'Use your webcam or external camera',      available: true,  icon: 'camera'    },
  { id: 'livecam', label: 'Live Cam', description: 'Connect to an IP camera via WHIP',        available: true,  icon: 'broadcast' },
  { id: 'file',    label: 'File',     description: 'Analyse a local video file',               available: true,  icon: 'file'      },
  { id: 'livekit', label: 'LiveKit',  description: 'Join a LiveKit room',                      available: false, icon: 'generic'   },
  { id: 'rtsp',    label: 'RTSP',     description: 'Connect to an RTSP stream',                available: false, icon: 'generic'   },
]

const selectedSource = ref<StreamSourceType | null>(null)
const videoEl = ref<HTMLVideoElement | null>(null)

// ── Computed ──────────────────────────────────────────────────────────────────

const isStreaming = computed(() => streamStore.isActive)
const isNegotiating = computed(() => streamStore.sessionState === 'negotiating')
const hasError = computed(() => streamStore.isError)
const activeEvents = computed(() => eventsStore.latestEvents.slice(0, 12))

const statusLabel = computed(() => {
  switch (streamStore.sessionState) {
    case 'negotiating': return 'Negotiating…'
    case 'connected':   return 'LIVE'
    case 'error':       return 'Error'
    default:            return 'Idle'
  }
})

const bitrateKbps = computed(() => {
  // rough estimate from bytes sent
  return Math.round(streamStore.bytesTransferred * 8 / 1000)
})

// ── Actions ───────────────────────────────────────────────────────────────────

function selectSource(source: SourceTile) {
  if (!source.available) return
  if (source.id === 'file') { router.push('/upload'); return }
  selectedSource.value = source.id
  streamStore.setSource(source.id)
}

async function handleStart() {
  if (!selectedSource.value) return
  await startStream(selectedSource.value)
}

async function handleStop() {
  disconnectEvents()
  await stopWhip()
  selectedSource.value = null
}

// ── Video preview ─────────────────────────────────────────────────────────────

watch(localStream, (stream) => {
  if (videoEl.value) {
    videoEl.value.srcObject = stream
    if (stream) videoEl.value.play().catch(() => { /* autoplay blocked */ })
  }
})

// ── SpacetimeDB subscription ──────────────────────────────────────────────────

// When a WHIP session gets a runId, subscribe to events for that run
watch(
  () => streamStore.activeSession?.runId,
  (runId) => {
    if (runId) connectEvents(runId)
    else disconnectEvents()
  },
)

// ── Rejoin session from URL param ─────────────────────────────────────────────

onMounted(() => {
  const sessionId = route.params.sessionId as string | undefined
  if (sessionId) {
    // TODO: rejoin an existing session by fetching its run state
    console.info('[StreamPage] Session rejoin requested:', sessionId)
  }
})

// ── Helpers ───────────────────────────────────────────────────────────────────

const EVENT_COLORS: Record<string, string> = {
  scene_cut:          '#2dd4bf',
  vlm_description:    '#f59e0b',
  loop_detected:      '#ef4444',
  artifact_suspected: '#f97316',
  exposure_shift:     '#a78bfa',
  flicker:            '#fb7185',
  keyframe:           '#22c55e',
}

function eventColor(type: string): string {
  return EVENT_COLORS[type] ?? '#64748b'
}

function formatPts(pts: number): string {
  const s = Math.floor(pts / 1000)
  const ms = pts % 1000
  return `${s}.${String(ms).padStart(3, '0')}s`
}
</script>

<template>
  <div class="p-4 lg:p-6 space-y-4">

    <!-- ── Active stream view ────────────────────────────────────────────── -->
    <template v-if="isStreaming || isNegotiating">

      <!-- Header bar -->
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <span
            class="live-dot"
            :style="isNegotiating ? 'background: #f59e0b;' : ''"
          />
          <h2 class="text-[#e2e8f0] font-semibold text-lg">
            {{ isNegotiating ? 'Connecting…' : 'Live Stream' }}
          </h2>
          <span v-if="isStreaming" class="badge badge-teal mono">
            {{ streamStore.fps.toFixed(0) }} fps
          </span>
          <span
            v-if="stdbConnected"
            class="badge badge-green mono"
            title="SpacetimeDB connected"
          >
            ●&nbsp;Events
          </span>
        </div>

        <button
          class="px-4 py-2 rounded-[10px] text-sm font-medium transition-all duration-200"
          style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); color: #ef4444;"
          :disabled="isNegotiating"
          @click="handleStop"
        >
          Stop Stream
        </button>
      </div>

      <!-- Error banner -->
      <div
        v-if="hasError && streamStore.error"
        class="rounded-[10px] p-3 text-sm text-[#ef4444]"
        style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);"
      >
        {{ streamStore.error }}
      </div>

      <!-- Main layout: video + events side-by-side on large screens -->
      <div class="grid grid-cols-1 xl:grid-cols-[1fr_320px] gap-4">

        <!-- Video preview -->
        <div
          class="relative rounded-[12px] overflow-hidden"
          style="background: #050507;
                 border: 1px solid #1e2633;
                 box-shadow: inset 0 2px 4px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.6);
                 aspect-ratio: 16/9;"
        >
          <!-- Local video element -->
          <video
            ref="videoEl"
            class="absolute inset-0 w-full h-full object-cover"
            muted
            playsinline
            autoplay
          />

          <!-- Negotiating spinner overlay -->
          <div
            v-if="isNegotiating"
            class="absolute inset-0 flex items-center justify-center"
            style="background: rgba(5,5,7,0.85);"
          >
            <div class="text-center">
              <div class="w-10 h-10 rounded-full border-2 border-[#1e2633] border-t-[#2dd4bf] animate-spin mx-auto mb-3" />
              <p class="text-[#475569] text-sm">Negotiating WHIP session…</p>
            </div>
          </div>

          <!-- HUD overlay -->
          <div v-if="isStreaming" class="absolute top-3 left-3 flex items-center gap-2 pointer-events-none">
            <span
              class="px-2 py-1 rounded text-xs mono font-semibold"
              :style="statusLabel === 'LIVE'
                ? 'background: rgba(0,0,0,0.75); border: 1px solid rgba(45,212,191,0.4); color: #2dd4bf;'
                : 'background: rgba(0,0,0,0.75); color: #64748b;'"
            >
              {{ statusLabel }}
            </span>
            <span
              class="px-2 py-1 rounded text-xs mono"
              style="background: rgba(0,0,0,0.7); color: #f59e0b;"
            >
              {{ streamStore.frameCount.toLocaleString() }} frames
            </span>
            <span
              v-if="bitrateKbps > 0"
              class="px-2 py-1 rounded text-xs mono"
              style="background: rgba(0,0,0,0.7); color: #94a3b8;"
            >
              {{ bitrateKbps }} kbps
            </span>
          </div>

          <!-- Session ID badge bottom-right -->
          <div
            v-if="streamStore.activeSession?.sessionId"
            class="absolute bottom-3 right-3 pointer-events-none"
          >
            <span
              class="px-2 py-1 rounded text-[10px] mono"
              style="background: rgba(0,0,0,0.7); color: #475569;"
            >
              {{ streamStore.activeSession.sessionId.slice(0, 8) }}…
            </span>
          </div>
        </div>

        <!-- Event feed panel -->
        <div
          class="rounded-[12px] flex flex-col"
          style="background: linear-gradient(145deg, #0f1117 0%, #151a22 100%);
                 border: 1px solid #1e2633;
                 box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 4px 12px rgba(0,0,0,0.5);"
        >
          <!-- Panel header -->
          <div
            class="flex items-center justify-between px-4 py-3"
            style="border-bottom: 1px solid #1e2633;"
          >
            <span class="text-xs font-semibold text-[#94a3b8] uppercase tracking-widest">Events</span>
            <span class="badge badge-muted mono">
              {{ eventsStore.activeEvents.length }}
            </span>
          </div>

          <!-- Event list -->
          <div class="flex-1 overflow-y-auto p-2 space-y-1" style="max-height: 420px;">
            <div v-if="activeEvents.length === 0" class="px-2 py-8 text-center">
              <p class="text-[#475569] text-xs">
                {{ stdbConnected ? 'Waiting for events…' : 'SpacetimeDB not connected' }}
              </p>
            </div>

            <div
              v-for="(evt, i) in activeEvents"
              :key="i"
              class="flex items-start gap-2 px-3 py-2 rounded-[8px] transition-all duration-150"
              style="background: rgba(255,255,255,0.02);"
            >
              <!-- Color dot -->
              <span
                class="mt-[3px] w-2 h-2 rounded-full flex-shrink-0"
                :style="{ background: eventColor(evt.event_type) }"
              />
              <div class="min-w-0 flex-1">
                <div class="flex items-center justify-between gap-2">
                  <span
                    class="text-[10px] font-semibold uppercase tracking-wide"
                    :style="{ color: eventColor(evt.event_type) }"
                  >
                    {{ evt.event_type.replace(/_/g, ' ') }}
                  </span>
                  <span class="text-[10px] mono text-[#475569] flex-shrink-0">
                    {{ formatPts(evt.pts_ms) }}
                  </span>
                </div>
                <p class="text-[11px] text-[#94a3b8] mt-0.5 leading-snug truncate">
                  {{ evt.description || '—' }}
                </p>
                <div class="flex items-center gap-2 mt-1">
                  <div
                    class="h-1 rounded-full"
                    :style="{
                      width: `${Math.round(evt.confidence * 100)}%`,
                      background: eventColor(evt.event_type),
                      opacity: 0.5,
                      minWidth: '4px',
                      maxWidth: '100%',
                    }"
                  />
                  <span class="text-[10px] mono text-[#475569]">
                    {{ Math.round(evt.confidence * 100) }}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>

    </template>

    <!-- ── Source picker ─────────────────────────────────────────────────── -->
    <template v-else>
      <div>
        <h2 class="text-[#e2e8f0] font-semibold text-xl">Select Source</h2>
        <p class="text-[#64748b] text-sm mt-1">Choose a video source to begin analysis</p>
      </div>

      <!-- Error from previous attempt -->
      <div
        v-if="hasError && streamStore.error"
        class="rounded-[10px] p-3 text-sm text-[#ef4444]"
        style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);"
      >
        {{ streamStore.error }}
        <button class="ml-3 underline text-xs" @click="streamStore.reset()">Dismiss</button>
      </div>

      <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
        <button
          v-for="source in sources"
          :key="source.id"
          class="relative flex flex-col items-start gap-3 p-4 rounded-[12px] text-left transition-all duration-200"
          :class="[
            source.available ? 'cursor-pointer' : 'cursor-not-allowed opacity-50',
          ]"
          :style="selectedSource === source.id
            ? 'background: rgba(45,212,191,0.06); border: 1px solid #2dd4bf; box-shadow: 0 0 20px rgba(45,212,191,0.15), inset 0 1px 0 rgba(255,255,255,0.05);'
            : 'background: linear-gradient(145deg, #0f1117 0%, #151a22 100%); border: 1px solid #1e2633; box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 4px 12px rgba(0,0,0,0.5);'"
          :disabled="!source.available"
          :aria-pressed="selectedSource === source.id"
          @click="selectSource(source)"
        >
          <!-- Selected checkmark -->
          <div
            v-if="selectedSource === source.id"
            class="absolute top-3 right-3 w-5 h-5 rounded-full flex items-center justify-center"
            style="background: #2dd4bf;"
          >
            <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
              <path d="M1 4L3.5 6.5L9 1" stroke="#08090d" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>

          <!-- Soon badge -->
          <span v-if="!source.available" class="absolute top-3 right-3 badge badge-muted">Soon</span>

          <!-- Icon -->
          <div
            class="w-10 h-10 rounded-[10px] flex items-center justify-center"
            :style="selectedSource === source.id
              ? 'background: rgba(45,212,191,0.15);'
              : 'background: rgba(255,255,255,0.04); border: 1px solid #1e2633;'"
          >
            <svg v-if="source.icon === 'monitor'" width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <rect x="2" y="3" width="16" height="11" rx="2"/>
              <path d="M7 17H13M10 14V17"/>
            </svg>
            <svg v-else-if="source.icon === 'camera'" width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <path d="M2 7C2 5.9 2.9 5 4 5H6L7.5 3H12.5L14 5H16C17.1 5 18 5.9 18 7V15C18 16.1 17.1 17 16 17H4C2.9 17 2 16.1 2 15V7Z"/>
              <circle cx="10" cy="11" r="3"/>
            </svg>
            <svg v-else-if="source.icon === 'broadcast'" width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <circle cx="10" cy="10" r="2.5" stroke="none"
                      :fill="selectedSource === source.id ? '#2dd4bf' : '#64748b'"/>
              <path d="M6 6C4.3 7.7 4.3 12.3 6 14M14 6C15.7 7.7 15.7 12.3 14 14"/>
              <path d="M3.5 3.5C1 6 1 14 3.5 16.5M16.5 3.5C19 6 19 14 16.5 16.5" opacity="0.5"/>
            </svg>
            <svg v-else-if="source.icon === 'file'" width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <path d="M4 3H12L16 7V17C16 17.6 15.6 18 15 18H5C4.4 18 4 17.6 4 17V3Z"/>
              <path d="M12 3V7H16"/>
              <path d="M7 11H13M7 14H11"/>
            </svg>
            <svg v-else width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <circle cx="10" cy="10" r="7"/>
              <path d="M10 7V10L12 12"/>
            </svg>
          </div>

          <!-- Text -->
          <div>
            <div class="text-sm font-medium" :class="selectedSource === source.id ? 'text-[#2dd4bf]' : 'text-[#e2e8f0]'">
              {{ source.label }}
            </div>
            <div class="text-xs text-[#475569] mt-0.5 leading-relaxed">{{ source.description }}</div>
          </div>
        </button>
      </div>

      <!-- Start button -->
      <div v-if="selectedSource" class="flex justify-end">
        <button
          class="flex items-center gap-2 px-6 py-2.5 rounded-[10px] text-sm font-semibold text-[#08090d] transition-all duration-200"
          style="background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
                 box-shadow: 0 0 20px rgba(45,212,191,0.3);"
          @click="handleStart"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M3 2L11 7L3 12V2Z" fill="currentColor"/>
          </svg>
          Start Analysis
        </button>
      </div>
    </template>

  </div>
</template>
