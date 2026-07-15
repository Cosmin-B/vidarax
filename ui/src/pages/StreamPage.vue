<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useStreamStore } from '@/stores/stream'
import { useEventsStore } from '@/stores/events'
import { useAuthStore } from '@/stores/auth'
import { useWhip } from '@/composables/useWhip'
import { useEventStream } from '@/composables/useEventStream'
import { api } from '@/lib/api'
import type { StreamSourceType } from '@/stores/stream'
import type { AgentEvent } from '@/stores/events'
import type { ModelInfo } from '@/lib/api'
import {
  Monitor, Camera, Radio, Square, Code2, ChevronDown,
  AlertCircle, Copy, Check, X, Sliders,
} from 'lucide-vue-next'
import ProcessingVisualization from '@/components/stream/ProcessingVisualization.vue'
import type { Component } from 'vue'

const route  = useRoute()
const streamStore  = useStreamStore()
const eventsStore  = useEventsStore()
const authStore    = useAuthStore()

const { localStream, startStream, stopStream: stopWhip } = useWhip()
const { isConnected: timelineConnected, connect: connectEvents, disconnect: disconnectEvents } = useEventStream()

// ── Types ────────────────────────────────────────────────────────────────────

type OutputMode = 'text' | 'json'
type ProcessingMode = 'clip' | 'frame'

interface SourceTile {
  id: StreamSourceType
  label: string
  description: string
  available: boolean
  icon: Component
}

interface VlmResult {
  id: number
  timestamp: string
  latencyMs: number
  text: string
  isJson: boolean
}

// ── Source tiles ──────────────────────────────────────────────────────────────

const sources: SourceTile[] = [
  { id: 'screen',  label: 'Screen',   description: 'Screen or window capture', available: true,  icon: Monitor },
  { id: 'camera',  label: 'Camera',   description: 'Webcam or external camera', available: true,  icon: Camera  },
  { id: 'livecam', label: 'Live Cam', description: 'IP camera via WHIP',        available: true,  icon: Radio   },
]

// ── Setup state ───────────────────────────────────────────────────────────────

const selectedSource  = ref<StreamSourceType | null>(null)
const analysisPrompt  = ref('')
const selectedModel   = ref('')
const processingMode  = ref<ProcessingMode>('clip')
const outputMode      = ref<OutputMode>('text')
const clipFps         = ref(1)
const clipLengthSec   = ref(5)
const intervalSec     = ref(2)
const jsonSchema      = ref('{\n  "type": "object",\n  "properties": {\n    "description": { "type": "string" },\n    "objects": { "type": "array", "items": { "type": "string" } }\n  }\n}')
const advancedOpen    = ref(false)
const availableModels = ref<ModelInfo[]>([])
const modelsLoading   = ref(false)

// ── Streaming state ───────────────────────────────────────────────────────────

const videoEl         = ref<HTMLVideoElement | null>(null)
const vlmResults      = ref<VlmResult[]>([])
const resultCounter   = ref(0)
const livePrompt      = ref('')
const promptDebounce  = ref<ReturnType<typeof setTimeout> | null>(null)
const showExportModal = ref(false)
const codeCopied      = ref(false)

// ── Computed ──────────────────────────────────────────────────────────────────

const isStreaming    = computed(() => streamStore.isActive)
const isNegotiating  = computed(() => streamStore.sessionState === 'negotiating')
const isConnected    = computed(() => streamStore.sessionState === 'connected')
const hasError       = computed(() => streamStore.isError)

const readyModels = computed(() =>
  availableModels.value.filter(m => m.availability !== 'unavailable')
)

const avgLatency = computed(() => {
  if (vlmResults.value.length === 0) return 0
  const sum = vlmResults.value.reduce((acc, r) => acc + r.latencyMs, 0)
  return Math.round(sum / vlmResults.value.length)
})

// ── Visualization computed ────────────────────────────────────────────────────

const streamVizFps        = computed(() => Math.max(Math.round(streamStore.fps), 1) || 30)
const streamVizProcessed  = computed(() => streamStore.frameCount)
const streamVizTotal      = computed(() => Math.max(streamVizProcessed.value + streamVizFps.value * 10, 60))
const streamVizChunkStart = computed(() => Math.max(0, streamVizProcessed.value - 30))
const streamVizChunkEnd   = computed(() => streamVizProcessed.value)
const streamVizEvents     = computed(() => eventsStore.activeEvents)
const streamVizVlmFrames  = ref<number[]>([])
const streamVizKfIndices  = ref<number[]>([])
const seenVizVlmFrames = new Set<number>()
const seenVizKfFrames = new Set<number>()
const seenVlmResultEvents = new Set<string>()
const seenStreamEvents = new Set<string>()

// ── Export code snippet ───────────────────────────────────────────────────────

const exportCode = computed(() => {
  const sourceUri = selectedSource.value === 'screen'
    ? 'screen://'
    : selectedSource.value === 'camera'
      ? 'camera://'
      : 'whip://<session_id>'

  const modelVal  = selectedModel.value || 'vidarax-default'
  const promptVal = (livePrompt.value || analysisPrompt.value).trim() || 'Describe what you see.'

  return `import { Vidarax } from 'vidarax'

const v = new Vidarax('${authStore.apiEndpoint}')
const run = await v.createRun({ mode: 'balanced' })
const result = await v.reason(run.run_id, {
  source_uri: '${sourceUri}',
  model: '${modelVal}',
  semantic_prompt: '${promptVal}',${processingMode.value === 'clip' ? `
  fps: ${clipFps.value},
  clip_length_seconds: ${clipLengthSec.value},` : `
  interval_seconds: ${intervalSec.value},`}${outputMode.value === 'json' ? `
  output_format: 'json',` : ''}
})`
})

// ── Actions ───────────────────────────────────────────────────────────────────

function selectSource(source: SourceTile) {
  if (!source.available) return
  selectedSource.value = source.id
  streamStore.setSource(source.id)
}

async function handleStart() {
  if (!selectedSource.value) return
  livePrompt.value = analysisPrompt.value
  await startStream(selectedSource.value, {
    prompt: analysisPrompt.value.trim() || 'Describe what is happening in this video frame.',
  })
}

async function handleStop() {
  disconnectEvents()
  await stopWhip()
  selectedSource.value = null
  vlmResults.value = []
  resultCounter.value = 0
  livePrompt.value = ''
  resetEventDerivedState()
}

async function patchPrompt(value: string) {
  const sessionId = streamStore.activeSession?.sessionId
  if (!sessionId) return
  try {
    await api.stream.whipUpdatePrompt(sessionId, { prompt: value })
  } catch (err) {
    streamStore.setError(err instanceof Error ? err.message : 'Prompt update failed')
  }
}

function onLivePromptInput(value: string) {
  livePrompt.value = value
  if (promptDebounce.value) clearTimeout(promptDebounce.value)
  promptDebounce.value = setTimeout(() => patchPrompt(value), 500)
}

async function copyExportCode() {
  try {
    await navigator.clipboard.writeText(exportCode.value)
    codeCopied.value = true
    setTimeout(() => { codeCopied.value = false }, 2000)
  } catch {
    // fallback: select
  }
}

async function fetchModels() {
  modelsLoading.value = true
  try {
    const data = await api.models()
    availableModels.value = data.models ?? []
    if (availableModels.value.length > 0 && !selectedModel.value) {
      const first = readyModels.value[0]
      if (first) selectedModel.value = first.id
    }
  } catch {
    // non-fatal
  } finally {
    modelsLoading.value = false
  }
}

// ── Collect visualization frames + VLM results from events ───────────────────

function resetEventDerivedState(): void {
  streamVizVlmFrames.value = []
  streamVizKfIndices.value = []
  seenVizVlmFrames.clear()
  seenVizKfFrames.clear()
  seenVlmResultEvents.clear()
  seenStreamEvents.clear()
}

function eventResultKey(event: { frame_index: number; timestamp_ms: number; pts_ms: number; event_type: string }): string {
  return `${event.event_type}:${event.frame_index}:${event.pts_ms}:${event.timestamp_ms}`
}

function streamEventKey(event: AgentEvent): string {
  return `${event.run_id}:${event.session_id}:${event.event_type}:${event.frame_index}:${event.pts_ms}:${event.timestamp_ms}:${event.description}`
}

function eventLatencyMs(timestampMs: number): number {
  return timestampMs > 1_577_836_800_000 ? Math.max(0, Date.now() - timestampMs) : 0
}

watch(
  () => [streamStore.activeSession?.runId, eventsStore.eventVersion] as const,
  () => {
    const events = eventsStore.activeEvents

    for (const event of events) {
      const processedKey = streamEventKey(event)
      if (seenStreamEvents.has(processedKey)) continue
      seenStreamEvents.add(processedKey)

      if (event.event_type === 'vlm_description' && !seenVizVlmFrames.has(event.frame_index)) {
        seenVizVlmFrames.add(event.frame_index)
        streamVizVlmFrames.value = [...streamVizVlmFrames.value, event.frame_index]
      } else if (event.event_type === 'keyframe' && !seenVizKfFrames.has(event.frame_index)) {
        seenVizKfFrames.add(event.frame_index)
        streamVizKfIndices.value = [...streamVizKfIndices.value, event.frame_index]
      }

      if (event.event_type !== 'vlm_description') continue

      const key = eventResultKey(event)
      if (seenVlmResultEvents.has(key)) continue
      seenVlmResultEvents.add(key)

      const result: VlmResult = {
        id: resultCounter.value++,
        timestamp: formatTimestamp(event.timestamp_ms),
        latencyMs: eventLatencyMs(event.timestamp_ms),
        text: event.description,
        isJson: outputMode.value === 'json',
      }

      vlmResults.value.unshift(result)
      if (vlmResults.value.length > 200) {
        vlmResults.value = vlmResults.value.slice(0, 200)
      }
    }
  }
)

// ── Video preview ─────────────────────────────────────────────────────────────

watch(localStream, (stream) => {
  if (videoEl.value) {
    videoEl.value.srcObject = stream
    if (stream) videoEl.value.play().catch(() => {})
  }
})

// ── Local event timeline ─────────────────────────────────────────────────────

watch(
  () => streamStore.activeSession?.runId,
  (runId) => {
    resetEventDerivedState()
    if (runId) connectEvents(runId)
    else disconnectEvents()
  },
)

// ── Init ──────────────────────────────────────────────────────────────────────

onMounted(() => {
  fetchModels()
  const sessionId = route.params.sessionId as string | undefined
  if (sessionId) {
    console.info('[StreamPage] Session rejoin requested:', sessionId)
  }
})

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatTimestamp(ms: number): string {
  if (!ms) {
    const now = new Date()
    return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`
  }
  const d = new Date(ms)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

function tryParseJson(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2)
  } catch {
    return text
  }
}

function modelDisplayName(id: string): string {
  const m = availableModels.value.find(x => x.id === id)
  return m?.name ?? id
}
</script>

<template>
  <!-- ══════════════════════════════════════════════════════════════
       SETUP MODE
       ══════════════════════════════════════════════════════════════ -->
  <div v-if="!isStreaming && !isNegotiating" class="sp-root">

    <!-- Two-column layout: left preview placeholder | right config -->
    <div class="sp-layout">

      <!-- ── Left: placeholder ──────────────────────────────────── -->
      <div class="sp-preview-placeholder">
        <div class="sp-preview-placeholder-inner">
          <div class="sp-preview-placeholder-icon">
            <Monitor :size="40" :stroke-width="1.25" class="text-[#1e2633]" />
          </div>
          <p class="sp-preview-placeholder-label">Configure your stream to get started</p>
          <p class="sp-preview-placeholder-sub">Select a source and set your prompt</p>
        </div>
      </div>

      <!-- ── Right: config sidebar ─────────────────────────────── -->
      <aside class="sp-sidebar">

        <!-- Error from previous attempt -->
        <div
          v-if="hasError && streamStore.error"
          class="sp-error-banner"
        >
          <AlertCircle :size="13" :stroke-width="2" class="flex-shrink-0" />
          <span>{{ streamStore.error }}</span>
          <button class="ml-auto text-[#ef4444] hover:text-white transition-colors" @click="streamStore.reset()">
            <X :size="13" :stroke-width="2" />
          </button>
        </div>

        <!-- Source picker -->
        <section class="sp-section">
          <label class="sp-label">Source</label>
          <div class="sp-source-grid">
            <button
              v-for="source in sources"
              :key="source.id"
              class="sp-source-tile"
              :class="selectedSource === source.id ? 'sp-source-tile--active' : ''"
              :disabled="!source.available"
              :aria-pressed="selectedSource === source.id"
              @click="selectSource(source)"
            >
              <div class="sp-source-tile-icon">
                <component :is="source.icon" :size="16" :stroke-width="1.75" />
              </div>
              <span class="sp-source-tile-label">{{ source.label }}</span>
            </button>
          </div>
        </section>

        <!-- Prompt -->
        <section class="sp-section">
          <label class="sp-label" for="setup-prompt">
            Prompt
            <span class="sp-label-required">*</span>
          </label>
          <textarea
            id="setup-prompt"
            v-model="analysisPrompt"
            rows="3"
            placeholder="Tell the model what to look for"
            class="sp-textarea"
            aria-label="Analysis prompt"
          />
        </section>

        <!-- Model -->
        <section class="sp-section">
          <label class="sp-label" for="setup-model">
            Model
            <span class="sp-label-required">*</span>
          </label>
          <div class="relative">
            <select
              id="setup-model"
              v-model="selectedModel"
              class="sp-select"
              :disabled="modelsLoading"
            >
              <option value="" disabled>{{ modelsLoading ? 'Loading models…' : 'Select a model' }}</option>
              <option
                v-for="m in availableModels"
                :key="m.id"
                :value="m.id"
                :disabled="m.availability === 'unavailable'"
              >
                {{ m.name ?? m.id }}{{ m.availability === 'unavailable' ? ' (unavailable)' : '' }}
              </option>
            </select>
            <!-- Ready indicator -->
            <div
              v-if="selectedModel && readyModels.some(m => m.id === selectedModel)"
              class="absolute right-8 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-[#22c55e]"
              title="Model ready"
            />
          </div>
        </section>

        <!-- Advanced settings (collapsible) -->
        <section class="sp-section">
          <button
            class="sp-advanced-toggle"
            :aria-expanded="advancedOpen"
            @click="advancedOpen = !advancedOpen"
          >
            <Sliders :size="13" :stroke-width="2" class="text-[#475569]" />
            <span>Advanced Settings</span>
            <ChevronDown
              :size="13"
              :stroke-width="2"
              class="ml-auto text-[#475569] transition-transform duration-200"
              :class="advancedOpen ? 'rotate-180' : ''"
            />
          </button>

          <div v-if="advancedOpen" class="sp-advanced-body">

            <!-- Mode toggle -->
            <div class="sp-advanced-row">
              <span class="sp-advanced-label">Mode</span>
              <div class="sp-toggle-group" role="group" aria-label="Processing mode">
                <button
                  class="sp-toggle-btn"
                  :class="processingMode === 'clip' ? 'sp-toggle-btn--active' : ''"
                  @click="processingMode = 'clip'"
                >Clip</button>
                <button
                  class="sp-toggle-btn"
                  :class="processingMode === 'frame' ? 'sp-toggle-btn--active' : ''"
                  @click="processingMode = 'frame'"
                >Frame</button>
              </div>
            </div>

            <!-- Clip settings -->
            <template v-if="processingMode === 'clip'">
              <div class="sp-advanced-row">
                <label class="sp-advanced-label" for="adv-fps">FPS</label>
                <div class="flex items-center gap-2">
                  <input
                    id="adv-fps"
                    v-model.number="clipFps"
                    type="range" min="1" max="30" step="1"
                    class="sp-range"
                  />
                  <span class="mono text-xs text-[#2dd4bf] w-6 text-right">{{ clipFps }}</span>
                </div>
              </div>
              <div class="sp-advanced-row">
                <label class="sp-advanced-label" for="adv-clip-len">Clip length (s)</label>
                <input
                  id="adv-clip-len"
                  v-model.number="clipLengthSec"
                  type="number" min="1" max="60"
                  class="sp-num-input"
                />
              </div>
            </template>

            <!-- Frame settings -->
            <template v-if="processingMode === 'frame'">
              <div class="sp-advanced-row">
                <label class="sp-advanced-label" for="adv-interval">Interval (s)</label>
                <input
                  id="adv-interval"
                  v-model.number="intervalSec"
                  type="number" min="0.5" max="60" step="0.5"
                  class="sp-num-input"
                />
              </div>
            </template>

            <!-- Output toggle -->
            <div class="sp-advanced-row">
              <span class="sp-advanced-label">Output</span>
              <div class="sp-toggle-group" role="group" aria-label="Output format">
                <button
                  class="sp-toggle-btn"
                  :class="outputMode === 'text' ? 'sp-toggle-btn--active' : ''"
                  @click="outputMode = 'text'"
                >Text</button>
                <button
                  class="sp-toggle-btn"
                  :class="outputMode === 'json' ? 'sp-toggle-btn--active' : ''"
                  @click="outputMode = 'json'"
                >JSON</button>
              </div>
            </div>

            <!-- JSON Schema editor -->
            <template v-if="outputMode === 'json'">
              <div class="sp-advanced-col">
                <label class="sp-advanced-label" for="adv-schema">JSON Schema</label>
                <textarea
                  id="adv-schema"
                  v-model="jsonSchema"
                  rows="7"
                  spellcheck="false"
                  class="sp-textarea sp-textarea--mono"
                  aria-label="JSON schema"
                />
              </div>
            </template>

          </div><!-- /advancedBody -->
        </section>

        <!-- Start button -->
        <button
          class="sp-start-btn"
          :disabled="!selectedSource"
          @click="handleStart"
        >
          Start Stream
        </button>

      </aside>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════════
       NEGOTIATING / STREAMING MODE
       ══════════════════════════════════════════════════════════════ -->
  <div v-else class="sp-root">
    <div class="sp-layout">

      <!-- ── Left: video + visualization ───────────────────────── -->
      <div class="sp-video-col">

        <!-- Video container -->
        <div class="sp-video-wrap">
          <video
            ref="videoEl"
            class="sp-video"
            muted
            playsinline
            autoplay
            aria-label="Local stream preview"
          />

          <!-- Negotiating overlay -->
          <div
            v-if="isNegotiating && !hasError"
            class="sp-video-overlay"
          >
            <div class="text-center">
              <div class="sp-spinner" />
              <p class="text-[#475569] text-sm mt-3">Negotiating WHIP session…</p>
            </div>
          </div>

          <!-- Error overlay -->
          <div
            v-if="hasError && streamStore.error"
            class="sp-video-overlay"
          >
            <div class="text-center space-y-3 max-w-xs">
              <AlertCircle :size="28" :stroke-width="1.75" class="text-[#ef4444] mx-auto" />
              <p class="text-[#ef4444] text-sm font-medium">Stream negotiation failed</p>
              <p class="text-[#64748b] text-xs leading-relaxed">{{ streamStore.error }}</p>
              <button class="sp-dismiss-btn" @click="handleStop">Dismiss</button>
            </div>
          </div>

          <!-- HUD: top-left -->
          <div v-if="isConnected" class="sp-hud-tl">
            <span class="sp-hud-badge sp-hud-badge--live">
              <span class="live-dot" style="width:6px;height:6px;" />
              LIVE
            </span>
            <span class="sp-hud-badge">
              {{ streamStore.frameCount.toLocaleString() }} frames
            </span>
            <span v-if="streamStore.fps > 0" class="sp-hud-badge">
              {{ streamStore.fps.toFixed(0) }} fps
            </span>
          </div>

          <!-- HUD: session ID bottom-right -->
          <div v-if="streamStore.activeSession?.sessionId" class="sp-hud-br">
            <span class="sp-hud-badge sp-hud-badge--dim">
              {{ streamStore.activeSession.sessionId.slice(0, 8) }}…
            </span>
          </div>
        </div>

        <!-- Processing visualization -->
        <ProcessingVisualization
          v-if="isConnected"
          :total-frames="streamVizTotal"
          :processed-frames="streamVizProcessed"
          :keyframe-count="streamVizKfIndices.length"
          :event-count="streamVizEvents.length"
          :current-chunk-start="streamVizChunkStart"
          :current-chunk-end="streamVizChunkEnd"
          :fps="streamVizFps"
          :is-processing="isConnected"
          :vlm-frames="streamVizVlmFrames"
          :keyframe-indices="streamVizKfIndices"
        />

      </div><!-- /video col -->

      <!-- ── Right: control sidebar ─────────────────────────────── -->
      <aside class="sp-sidebar">

        <!-- Header row -->
        <div class="sp-streaming-header">
          <div class="flex items-center gap-2 min-w-0">
            <span class="live-dot" />
            <span class="text-sm font-semibold text-[#e2e8f0] truncate">
              {{ isNegotiating ? 'Connecting…' : (selectedModel ? modelDisplayName(selectedModel) : 'Live Stream') }}
            </span>
          </div>
          <div class="flex items-center gap-2 flex-shrink-0">
            <button
              class="sp-export-btn"
              title="Export SDK code"
              @click="showExportModal = true"
            >
              <Code2 :size="13" :stroke-width="2" />
              <span>Export</span>
            </button>
            <button
              class="sp-stop-btn"
              :disabled="isNegotiating"
              @click="handleStop"
            >
              <Square :size="11" :stroke-width="2" style="fill: currentColor;" />
              Stop
            </button>
          </div>
        </div>

        <!-- Error banner -->
        <div v-if="hasError && streamStore.error" class="sp-error-banner">
          <AlertCircle :size="13" :stroke-width="2" class="flex-shrink-0" />
          <span>{{ streamStore.error }}</span>
        </div>

        <!-- Live prompt (editable while streaming) -->
        <section class="sp-section">
          <label class="sp-label" for="live-prompt">Prompt</label>
          <textarea
            id="live-prompt"
            :value="livePrompt"
            rows="3"
            placeholder="Describe what to look for…"
            class="sp-textarea"
            aria-label="Live analysis prompt — editable while streaming"
            @input="onLivePromptInput(($event.target as HTMLTextAreaElement).value)"
          />
          <p class="sp-hint">Changes apply within ~500 ms</p>
        </section>

        <!-- Results panel -->
        <section class="sp-section sp-results-section">
          <div class="sp-results-header">
            <span class="sp-label" style="margin-bottom:0;">Results</span>
            <span class="badge badge-muted mono">{{ vlmResults.length }}</span>
          </div>

          <!-- Stats bar -->
          <div v-if="vlmResults.length > 0" class="sp-stats-bar">
            <span class="mono text-[11px] text-[#475569]">
              avg latency:
              <span class="text-[#2dd4bf]">{{ avgLatency }}ms</span>
            </span>
            <span v-if="timelineConnected" class="badge badge-green" style="font-size:10px;padding:1px 6px;">
              Live
            </span>
          </div>

          <!-- Results list -->
          <div
            ref="resultsEl"
            class="sp-results-list"
            role="log"
            aria-live="polite"
            aria-label="VLM analysis results"
          >
            <div
              v-if="vlmResults.length === 0"
              class="sp-results-empty"
            >
              <p>{{ isConnected ? 'Waiting for results…' : 'Stream starting…' }}</p>
            </div>

            <div
              v-for="result in vlmResults"
              :key="result.id"
              class="sp-result-card"
            >
              <div class="sp-result-meta">
                <span class="mono sp-result-timestamp">{{ result.timestamp }}</span>
                <span class="mono sp-result-latency">{{ result.latencyMs }}ms</span>
              </div>
              <pre
                v-if="result.isJson"
                class="sp-result-json"
              >{{ tryParseJson(result.text) }}</pre>
              <p v-else class="sp-result-text">{{ result.text }}</p>
            </div>
          </div>
        </section>

      </aside>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════════
       EXPORT CODE MODAL
       ══════════════════════════════════════════════════════════════ -->
  <Teleport to="body">
    <div
      v-if="showExportModal"
      class="sp-modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Export SDK code"
      @click.self="showExportModal = false"
    >
      <div class="sp-modal">
        <div class="sp-modal-header">
          <span class="text-sm font-semibold text-[#e2e8f0]">SDK Snippet</span>
          <div class="flex items-center gap-2">
            <button
              class="sp-export-btn"
              @click="copyExportCode"
            >
              <component :is="codeCopied ? Check : Copy" :size="13" :stroke-width="2" />
              {{ codeCopied ? 'Copied!' : 'Copy' }}
            </button>
            <button
              class="sp-modal-close"
              aria-label="Close modal"
              @click="showExportModal = false"
            >
              <X :size="15" :stroke-width="2" />
            </button>
          </div>
        </div>
        <pre class="sp-modal-code"><code>{{ exportCode }}</code></pre>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
/* ── Root & layout ─────────────────────────────────────────────────────────── */

.sp-root {
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 0;
  overflow: hidden;
}

.sp-layout {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 16px;
  flex: 1;
  min-height: 0;
}

@media (max-width: 900px) {
  .sp-layout {
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr;
    overflow-y: auto;
  }
  .sp-root {
    overflow: auto;
    height: auto;
  }
}

/* ── Video column ──────────────────────────────────────────────────────────── */

.sp-video-col {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
}

.sp-video-wrap {
  position: relative;
  border-radius: 12px;
  overflow: hidden;
  background: #050507;
  border: 1px solid #1e2633;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.6);
  aspect-ratio: 16 / 9;
  flex-shrink: 0;
}

.sp-video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.sp-video-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(5, 5, 7, 0.88);
}

.sp-spinner {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 2px solid #1e2633;
  border-top-color: #2dd4bf;
  animation: sp-spin 0.8s linear infinite;
}

@keyframes sp-spin {
  to { transform: rotate(360deg); }
}

/* ── HUD badges ────────────────────────────────────────────────────────────── */

.sp-hud-tl {
  position: absolute;
  top: 10px;
  left: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
  pointer-events: none;
}

.sp-hud-br {
  position: absolute;
  bottom: 10px;
  right: 10px;
  pointer-events: none;
}

.sp-hud-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 8px;
  border-radius: 5px;
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
  font-size: 10px;
  font-weight: 600;
  background: rgba(0,0,0,0.75);
  color: #94a3b8;
  border: 1px solid rgba(255,255,255,0.06);
}

.sp-hud-badge--live {
  color: #2dd4bf;
  border-color: rgba(45, 212, 191, 0.3);
}

.sp-hud-badge--dim {
  color: #475569;
}

/* ── Preview placeholder ───────────────────────────────────────────────────── */

.sp-preview-placeholder {
  border-radius: 12px;
  background: linear-gradient(145deg, #0a0c12 0%, #0f1117 100%);
  border: 1px dashed #1e2633;
  aspect-ratio: 16 / 9;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sp-preview-placeholder-inner {
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.sp-preview-placeholder-icon {
  width: 72px;
  height: 72px;
  border-radius: 16px;
  background: rgba(255,255,255,0.02);
  border: 1px solid #1e2633;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sp-preview-placeholder-label {
  color: #475569;
  font-size: 13px;
  font-weight: 500;
  margin: 0;
}

.sp-preview-placeholder-sub {
  color: #334155;
  font-size: 11px;
  margin: 0;
}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */

.sp-sidebar {
  display: flex;
  flex-direction: column;
  gap: 0;
  overflow-y: auto;
  border-radius: 12px;
  background: linear-gradient(145deg, #0f1117 0%, #151a22 100%);
  border: 1px solid #1e2633;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 4px 12px rgba(0,0,0,0.5);
  scrollbar-width: thin;
  scrollbar-color: #1e2633 transparent;
}

.sp-sidebar::-webkit-scrollbar { width: 4px; }
.sp-sidebar::-webkit-scrollbar-track { background: transparent; }
.sp-sidebar::-webkit-scrollbar-thumb { background: #1e2633; border-radius: 2px; }

/* ── Sidebar sections ──────────────────────────────────────────────────────── */

.sp-section {
  padding: 14px 16px;
  border-bottom: 1px solid #111827;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sp-section:last-child {
  border-bottom: none;
}

.sp-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #64748b;
  display: block;
}

.sp-label-required {
  color: #2dd4bf;
  margin-left: 2px;
}

.sp-hint {
  font-size: 10px;
  color: #334155;
  margin: 0;
}

/* ── Source tiles ──────────────────────────────────────────────────────────── */

.sp-source-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}

.sp-source-tile {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 10px 6px;
  border-radius: 9px;
  border: 1px solid #1e2633;
  background: rgba(255,255,255,0.02);
  cursor: pointer;
  transition: border-color 150ms, background 150ms, box-shadow 150ms;
  color: #64748b;
}

.sp-source-tile:hover:not(:disabled) {
  border-color: rgba(45,212,191,0.3);
  background: rgba(45,212,191,0.04);
  color: #94a3b8;
}

.sp-source-tile--active {
  border-color: #2dd4bf !important;
  background: rgba(45,212,191,0.08) !important;
  color: #2dd4bf !important;
  box-shadow: 0 0 12px rgba(45,212,191,0.12);
}

.sp-source-tile-icon {
  width: 30px;
  height: 30px;
  border-radius: 7px;
  background: rgba(255,255,255,0.03);
  border: 1px solid #1e2633;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 150ms, border-color 150ms;
}

.sp-source-tile--active .sp-source-tile-icon {
  background: rgba(45,212,191,0.12);
  border-color: rgba(45,212,191,0.3);
}

.sp-source-tile-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.03em;
  white-space: nowrap;
}

/* ── Inputs ────────────────────────────────────────────────────────────────── */

.sp-textarea {
  width: 100%;
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 12px;
  color: #e2e8f0;
  background: #080a0f;
  border: 1px solid #1e2633;
  outline: none;
  resize: none;
  line-height: 1.5;
  transition: border-color 150ms;
  font-family: inherit;
}

.sp-textarea:focus {
  border-color: rgba(45,212,191,0.4);
}

.sp-textarea::placeholder {
  color: #334155;
}

.sp-textarea--mono {
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
  tab-size: 2;
}

.sp-select {
  width: 100%;
  min-height: 44px;
  padding: 7px 10px;
  padding-right: 28px;
  border-radius: 8px;
  font-size: 12px;
  color: #e2e8f0;
  background: #080a0f;
  border: 1px solid #1e2633;
  outline: none;
  appearance: none;
  cursor: pointer;
  transition: border-color 150ms;
}

.sp-select:focus {
  border-color: rgba(45,212,191,0.4);
}

.sp-select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ── Advanced settings ─────────────────────────────────────────────────────── */

.sp-advanced-toggle {
  display: flex;
  align-items: center;
  min-height: 44px;
  gap: 7px;
  font-size: 11px;
  font-weight: 500;
  color: #64748b;
  cursor: pointer;
  padding: 5px 0;
  transition: color 150ms;
  background: none;
  border: none;
  width: 100%;
  text-align: left;
}

.sp-advanced-toggle:hover {
  color: #94a3b8;
}

.sp-advanced-body {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-top: 4px;
}

.sp-advanced-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.sp-advanced-col {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.sp-advanced-label {
  font-size: 11px;
  color: #64748b;
  flex-shrink: 0;
}

.sp-range {
  -webkit-appearance: none;
  appearance: none;
  height: 3px;
  border-radius: 2px;
  background: #1e2633;
  outline: none;
  cursor: pointer;
  flex: 1;
}

.sp-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #2dd4bf;
  cursor: pointer;
  box-shadow: 0 0 6px rgba(45,212,191,0.4);
}

.sp-num-input {
  width: 64px;
  min-height: 44px;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 12px;
  color: #e2e8f0;
  background: #080a0f;
  border: 1px solid #1e2633;
  outline: none;
  text-align: right;
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
}

.sp-num-input:focus {
  border-color: rgba(45,212,191,0.4);
}

/* ── Toggle group ──────────────────────────────────────────────────────────── */

.sp-toggle-group {
  display: flex;
  border-radius: 7px;
  overflow: hidden;
  border: 1px solid #1e2633;
}

.sp-toggle-btn {
  min-height: 44px;
  padding: 4px 12px;
  font-size: 11px;
  font-weight: 500;
  color: #475569;
  background: transparent;
  border: none;
  cursor: pointer;
  transition: background 150ms, color 150ms;
}

.sp-toggle-btn:first-child {
  border-right: 1px solid #1e2633;
}

.sp-toggle-btn--active {
  background: rgba(45,212,191,0.12);
  color: #2dd4bf;
}

.sp-toggle-btn:hover:not(.sp-toggle-btn--active) {
  color: #94a3b8;
  background: rgba(255,255,255,0.03);
}

/* ── Start button ──────────────────────────────────────────────────────────── */

.sp-start-btn {
  margin: 14px 16px;
  min-height: 44px;
  padding: 10px 16px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 600;
  color: #08090d;
  background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
  border: none;
  cursor: pointer;
  transition: opacity 150ms, box-shadow 150ms;
  box-shadow: 0 0 20px rgba(45,212,191,0.25);
  text-align: center;
}

.sp-start-btn:hover:not(:disabled) {
  box-shadow: 0 0 28px rgba(45,212,191,0.4);
  opacity: 0.95;
}

.sp-start-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
  box-shadow: none;
}

/* ── Streaming header ──────────────────────────────────────────────────────── */

.sp-streaming-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 12px 14px;
  border-bottom: 1px solid #111827;
  flex-shrink: 0;
}

/* ── Action buttons ────────────────────────────────────────────────────────── */

.sp-export-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 10px;
  border-radius: 7px;
  font-size: 11px;
  font-weight: 500;
  color: #94a3b8;
  background: rgba(255,255,255,0.04);
  border: 1px solid #1e2633;
  cursor: pointer;
  transition: border-color 150ms, color 150ms, background 150ms;
}

.sp-export-btn:hover {
  border-color: rgba(45,212,191,0.3);
  color: #2dd4bf;
  background: rgba(45,212,191,0.05);
}

.sp-stop-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 10px;
  border-radius: 7px;
  font-size: 11px;
  font-weight: 600;
  color: #ef4444;
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.2);
  cursor: pointer;
  transition: background 150ms, border-color 150ms;
}

.sp-stop-btn:hover:not(:disabled) {
  background: rgba(239,68,68,0.14);
  border-color: rgba(239,68,68,0.4);
}

.sp-stop-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.sp-dismiss-btn {
  padding: 5px 14px;
  border-radius: 7px;
  font-size: 11px;
  font-weight: 500;
  color: #94a3b8;
  background: rgba(255,255,255,0.05);
  border: 1px solid #1e2633;
  cursor: pointer;
  transition: background 150ms;
}

.sp-dismiss-btn:hover {
  background: rgba(255,255,255,0.08);
}

/* ── Error banner ──────────────────────────────────────────────────────────── */

.sp-error-banner {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  border-radius: 8px;
  margin: 10px 14px;
  font-size: 12px;
  color: #ef4444;
  background: rgba(239,68,68,0.07);
  border: 1px solid rgba(239,68,68,0.18);
}

/* ── Results ───────────────────────────────────────────────────────────────── */

.sp-results-section {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  padding-bottom: 0 !important;
}

.sp-results-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.sp-stats-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0;
}

.sp-results-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-bottom: 12px;
  scrollbar-width: thin;
  scrollbar-color: #1e2633 transparent;
  max-height: calc(100vh - 420px);
  min-height: 120px;
}

.sp-results-list::-webkit-scrollbar { width: 3px; }
.sp-results-list::-webkit-scrollbar-track { background: transparent; }
.sp-results-list::-webkit-scrollbar-thumb { background: #1e2633; border-radius: 2px; }

.sp-results-empty {
  padding: 24px 0;
  text-align: center;
  color: #334155;
  font-size: 12px;
}

.sp-result-card {
  padding: 9px 10px;
  border-radius: 8px;
  background: rgba(255,255,255,0.02);
  border: 1px solid #13192280;
  display: flex;
  flex-direction: column;
  gap: 5px;
  transition: border-color 150ms;
}

.sp-result-card:hover {
  border-color: rgba(45,212,191,0.12);
}

.sp-result-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.sp-result-timestamp {
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
  font-size: 10px;
  color: #475569;
}

.sp-result-latency {
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
  font-size: 10px;
  color: #2dd4bf;
}

.sp-result-text {
  font-size: 11px;
  color: #94a3b8;
  line-height: 1.5;
  margin: 0;
  word-break: break-word;
}

.sp-result-json {
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
  font-size: 10px;
  color: #64748b;
  line-height: 1.55;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-x: auto;
}

/* ── Export modal ──────────────────────────────────────────────────────────── */

.sp-modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(5,5,7,0.82);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  padding: 24px;
}

.sp-modal {
  width: 100%;
  max-width: 560px;
  border-radius: 14px;
  background: linear-gradient(145deg, #0f1117 0%, #151a22 100%);
  border: 1px solid #1e2633;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 24px 60px rgba(0,0,0,0.7);
  overflow: hidden;
}

.sp-modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid #1e2633;
}

.sp-modal-close {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 6px;
  color: #475569;
  background: rgba(255,255,255,0.04);
  border: 1px solid #1e2633;
  cursor: pointer;
  transition: color 150ms, background 150ms;
}

.sp-modal-close:hover {
  color: #e2e8f0;
  background: rgba(255,255,255,0.07);
}

.sp-modal-code {
  padding: 16px;
  margin: 0;
  font-family: ui-monospace, 'SFMono-Regular', Menlo, Monaco, Consolas, monospace;
  font-size: 11.5px;
  line-height: 1.65;
  color: #94a3b8;
  overflow-x: auto;
  white-space: pre;
  background: #080a0f;
  max-height: 400px;
  overflow-y: auto;
}

.sp-modal-code code {
  font-family: inherit;
}

/* ── Scrollbar for results list ────────────────────────────────────────────── */
.sp-results-list { scrollbar-gutter: stable; }
</style>
