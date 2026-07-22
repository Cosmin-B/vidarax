<script setup lang="ts">
import { ref, computed, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useRunsStore } from '@/stores/runs'
import { useEventsStore } from '@/stores/events'
import { useEventStream } from '@/composables/useEventStream'
import { useAuthStore } from '@/stores/auth'
import { api, ApiError } from '@/lib/api'
import { ls, lsBool, lsNum, STORAGE_KEYS, UI_DEFAULTS } from '@/lib/config'
import type { RunStatus } from '@/stores/runs'
import { ChevronLeft, UploadCloud, CheckCircle, AlertCircle, Upload, Zap } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'
import ProcessingVisualization from '@/components/stream/ProcessingVisualization.vue'

const router = useRouter()
const runsStore = useRunsStore()
const eventsStore = useEventsStore()
const authStore = useAuthStore()
const { connect: connectEvents, disconnect: disconnectEvents, isConnected: timelineConnected } = useEventStream()

// ── State ─────────────────────────────────────────────────────────────────────

const isDragging = ref(false)
const selectedFile = ref<File | null>(null)
const uploadProgress = ref(0)
const uploadState = ref<'idle' | 'uploading' | 'creating' | 'analyzing' | 'done' | 'error'>('idle')
const uploadError = ref<string | null>(null)

const DEFAULT_ANALYSIS_PROMPT = 'Describe what is happening in this video frame.'

// Config for the run
const selectedModel = ref(ls(STORAGE_KEYS.defaultModel, UI_DEFAULTS.defaultModel))
const prompt = ref('')
const semanticInference = ref(lsBool(STORAGE_KEYS.semanticInference, UI_DEFAULTS.semanticInference))

// Created run
const createdRunId = ref<string | null>(null)
const runStatus = ref<RunStatus | null>(null)

// ── Visualization state ────────────────────────────────────────────────────
const vizTotalFrames    = ref(0)
const vizProcessed      = ref(0)
const vizChunkStart     = ref(0)
const vizChunkEnd       = ref(0)
const vizFps            = ref(5)
const vizVlmFrames      = ref<number[]>([])
const vizKfIndices      = ref<number[]>([])
let   vizPollTimer: ReturnType<typeof setInterval> | null = null

/** Called once we know the run's fps + chunk_size to set up the strip. */
function initViz(fps: number, chunkSize: number) {
  vizFps.value = fps
  // We don't know total frames yet — seed with one chunk width so the strip
  // expands naturally as events arrive.
  vizTotalFrames.value = chunkSize * 4
}

/** Poll GET /v1/runs/:id/events every second to update the visualization. */
function startVizPolling(runId: string): void {
  if (vizPollTimer !== null) return

  async function poll() {
    try {
      const res = await api.runs.events(runId)
      let maxFrame = 0
      const newVlm: number[] = []
      const newKf: number[]  = []

      for (const raw of res.events) {
        const p = raw.payload
        const startFrame = Number(p.start_frame ?? 0)
        const endFrame   = Number(p.end_frame   ?? startFrame)

        maxFrame = Math.max(maxFrame, endFrame)

        if (raw.kind === 'chunk_processed' || raw.kind === 'frame_decoded') {
          vizProcessed.value = Math.max(vizProcessed.value, endFrame)
          vizChunkStart.value = startFrame
          vizChunkEnd.value   = endFrame
        }

        if (raw.kind === 'marker_emitted') {
          const evtType = String(p.event_type ?? p.kind ?? '')
          if (evtType === 'keyframe') newKf.push(startFrame)
          if (evtType === 'vlm_description') newVlm.push(startFrame)
        }
      }

      if (maxFrame > 0) {
        vizTotalFrames.value = Math.max(vizTotalFrames.value, maxFrame + 1)
      }

      // Derive processed count from event count when no explicit frame events
      if (vizProcessed.value === 0 && activeEvents.value.length > 0) {
        const lastEvt = activeEvents.value[activeEvents.value.length - 1]
        if (lastEvt) vizProcessed.value = lastEvt.frame_index
      }

      if (newVlm.length) vizVlmFrames.value = newVlm
      if (newKf.length)  vizKfIndices.value  = newKf
    } catch {
      // transient — keep polling
    }
  }

  poll()
  vizPollTimer = setInterval(poll, 1000)
}

function stopVizPolling(): void {
  if (vizPollTimer !== null) {
    clearInterval(vizPollTimer)
    vizPollTimer = null
  }
}

const MODELS = [
  { id: 'Qwen/Qwen3-VL-8B-Instruct',    label: 'Qwen3-VL 8B'    },
  { id: 'Qwen/Qwen3-VL-4B-Instruct',    label: 'Qwen3-VL 4B'    },
  { id: 'Qwen/Qwen3-VL-2B-Instruct',    label: 'Qwen3-VL 2B'    },
  { id: 'OpenGVLab/InternVL3_5-4B',     label: 'InternVL3.5 4B' },
  { id: 'LiquidAI/LFM2.5-VL-1.6B',     label: 'LFM2.5-VL 1.6B' },
]

const acceptedTypes = ['video/mp4', 'video/webm', 'video/quicktime']

// ── Computed ──────────────────────────────────────────────────────────────────

const fileSize = computed(() => {
  if (!selectedFile.value) return ''
  const bytes = selectedFile.value.size
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
})

const isProcessing = computed(() =>
  ['uploading', 'creating', 'analyzing'].includes(uploadState.value)
)

const activeEvents = computed(() =>
  createdRunId.value ? eventsStore.eventsForRun(createdRunId.value) : []
)

const activeKeyframes = computed(() =>
  createdRunId.value ? eventsStore.keyframesForRun(createdRunId.value) : []
)

function parseSemanticFramesPerChunk(raw: string | null): number | undefined {
  if (raw === null) return undefined
  const trimmed = raw.trim()
  if (trimmed === '') return undefined
  const parsed = Number(trimmed)
  return Number.isInteger(parsed) && parsed >= 1 && parsed <= 4 ? parsed : undefined
}

const statusLabel = computed(() => {
  switch (uploadState.value) {
    case 'uploading':  return 'Uploading file…'
    case 'creating':   return 'Creating run…'
    case 'analyzing':  return 'Analysing frames…'
    case 'done':       return 'Analysis complete'
    case 'error':      return 'Error'
    default:           return ''
  }
})

// ── Drop zone ─────────────────────────────────────────────────────────────────

const fileInputRef = ref<HTMLInputElement | null>(null)

function onDragover(e: DragEvent) {
  e.preventDefault()
  isDragging.value = true
}
function onDragleave() { isDragging.value = false }
function onDrop(e: DragEvent) {
  e.preventDefault()
  isDragging.value = false
  const file = e.dataTransfer?.files[0]
  if (file) validateAndSet(file)
}
function onFileInput(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (file) validateAndSet(file)
}

function validateAndSet(file: File) {
  uploadError.value = null
  if (!acceptedTypes.includes(file.type) && !file.name.match(/\.(mp4|webm|mov)$/i)) {
    uploadError.value = 'Only .mp4, .webm, and .mov files are supported.'
    return
  }
  selectedFile.value = file
  uploadState.value = 'idle'
  uploadProgress.value = 0
}

// ── Upload + analysis flow ────────────────────────────────────────────────────

async function startUpload(): Promise<void> {
  if (!selectedFile.value) return
  uploadError.value = null

  try {
    // ── Step 1: Upload file via XHR for progress ──────────────────────────
    uploadState.value = 'uploading'
    uploadProgress.value = 0
    const filePath = await uploadFileWithProgress(selectedFile.value)

    // ── Step 2: Create run ────────────────────────────────────────────────
    uploadState.value = 'creating'
    uploadProgress.value = 0

    const fps = lsNum(STORAGE_KEYS.fps, UI_DEFAULTS.fps)
    const chunkSize = lsNum(STORAGE_KEYS.chunkSize, UI_DEFAULTS.chunkSize)
    const firstPassModel = ls(STORAGE_KEYS.firstPassModel, UI_DEFAULTS.firstPassModel).trim()
    const secondPassModel = ls(STORAGE_KEYS.secondPassModel, UI_DEFAULTS.secondPassModel).trim()
    const semanticFramesPerChunk = parseSemanticFramesPerChunk(
      localStorage.getItem(STORAGE_KEYS.semanticFramesPerChunk)
        ?? String(UI_DEFAULTS.semanticFramesPerChunk),
    )

    const effectivePrompt = prompt.value.trim() || DEFAULT_ANALYSIS_PROMPT

    const run = await api.runs.create({
      mode: 'balanced',
      model: selectedModel.value,
    })

    createdRunId.value = run.run_id
    runStatus.value = run.status as RunStatus

    runsStore.upsertRun({
      run_id: run.run_id,
      status: run.status as RunStatus,
      mode: run.mode as 'balanced',
      model: run.model,
      created_at: run.created_at,
    })

    // ── Step 3: Follow the local event timeline ───────────────────────────
    eventsStore.setActiveRunId(run.run_id)
    connectEvents(run.run_id)

    // ── Initialize visualization ──────────────────────────────────────────
    initViz(fps, chunkSize)
    startVizPolling(run.run_id)

    uploadState.value = 'analyzing'

    // ── Step 4: Run analysis (blocks until complete) ───────────────────────
    await api.runs.reason(run.run_id, {
      source_uri: filePath,
      model: selectedModel.value,
      prompt: effectivePrompt,
      semantic_inference: semanticInference.value,
      fps,
      chunk_size: chunkSize,
      firstPassModel: firstPassModel || undefined,
      secondPassModel: secondPassModel || undefined,
      semanticFramesPerChunk,
    })

    stopVizPolling()
    uploadState.value = 'done'

  } catch (err) {
    stopVizPolling()
    const msg = err instanceof ApiError
      ? `API ${err.status}: ${err.message}`
      : err instanceof Error ? err.message : 'Upload failed'
    uploadError.value = msg
    uploadState.value = 'error'
  }
}

/** Upload using XHR so we get progress events. */
function uploadFileWithProgress(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const url = `${authStore.apiEndpoint}/v1/upload`
    xhr.open('POST', url)
    if (authStore.apiKey) xhr.setRequestHeader('x-api-key', authStore.apiKey)

    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) {
        uploadProgress.value = Math.round((ev.loaded / ev.total) * 100)
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText) as { file_path: string }
          resolve(data.file_path)
        } catch {
          reject(new Error('Invalid upload response'))
        }
      } else {
        reject(new ApiError(xhr.status, xhr.responseText || xhr.statusText))
      }
    }

    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.ontimeout = () => reject(new Error('Upload timed out'))

    const formData = new FormData()
    formData.append('file', file)
    xhr.send(formData)
  })
}

function reset(): void {
  disconnectEvents()
  stopVizPolling()
  selectedFile.value = null
  uploadState.value = 'idle'
  uploadProgress.value = 0
  uploadError.value = null
  createdRunId.value = null
  runStatus.value = null
  vizProcessed.value = 0
  vizChunkStart.value = 0
  vizChunkEnd.value = 0
  vizVlmFrames.value = []
  vizKfIndices.value = []
  if (fileInputRef.value) fileInputRef.value.value = ''
}

onUnmounted(() => {
  disconnectEvents()
  stopVizPolling()
})

// ── Helpers ───────────────────────────────────────────────────────────────────

const EVENT_COLORS: Record<string, string> = {
  scene_cut: '#2dd4bf', loop_detected: '#f59e0b',
  vlm_description: '#22c55e', artifact_suspected: '#ef4444',
  exposure_shift: '#a78bfa', flicker: '#fb7185', keyframe: '#22c55e',
  restricted_zone_activity: '#38bdf8',
  trigger_assertion: '#38bdf8',
}
function eventColor(t: string) { return EVENT_COLORS[t] ?? '#64748b' }
function formatPts(ms: number) {
  return `${Math.floor(ms / 1000)}.${String(ms % 1000).padStart(3, '0')}s`
}
</script>

<template>
  <div class="p-4 lg:p-6 space-y-6">

    <!-- Header -->
    <div class="flex items-center gap-3">
      <RouterLink
        to="/"
        class="-ml-3 flex h-11 w-11 items-center justify-center text-[#475569] hover:text-[#64748b] transition-colors icon-hover-parent"
        aria-label="Back to Dashboard"
      >
        <AnimatedIcon :icon="ChevronLeft" :size="16" :stroke-width="2" />
      </RouterLink>
      <div>
        <h2 class="text-[#e2e8f0] font-semibold text-xl">Upload Video</h2>
        <p class="text-[#64748b] text-sm mt-0.5">Upload a video file for batch semantic analysis</p>
      </div>
    </div>

    <!-- ── Done state ──────────────────────────────────────────────────────── -->
    <template v-if="uploadState === 'done'">
      <div class="card-skeuo p-6 text-center space-y-4">
        <div
          class="w-12 h-12 rounded-full flex items-center justify-center mx-auto"
          style="background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.2);"
        >
          <AnimatedIcon
            :icon="CheckCircle"
            :size="24"
            :stroke-width="2"
            animation="draw-in"
            class="text-[#22c55e]"
          />
        </div>
        <div>
          <div class="text-[#e2e8f0] font-semibold">Analysis Complete</div>
          <div v-if="createdRunId" class="mono text-[#475569] text-xs mt-1">{{ createdRunId }}</div>
        </div>
        <div class="flex gap-3 justify-center">
          <button
            v-if="createdRunId"
            class="min-h-11 px-4 py-2 rounded-[10px] text-sm font-medium text-[#08090d]"
            style="background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);"
            @click="router.push(`/runs/${createdRunId}`)"
          >
            View Results
          </button>
          <RouterLink
            to="/"
            class="inline-flex min-h-11 items-center px-4 py-2 rounded-[10px] text-sm font-medium text-[#94a3b8]"
            style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
          >
            Dashboard
          </RouterLink>
          <button
            class="min-h-11 px-4 py-2 rounded-[10px] text-sm font-medium text-[#64748b]"
            style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
            @click="reset"
          >
            Upload Another
          </button>
        </div>
      </div>

      <!-- Inline results on done -->
      <div v-if="activeEvents.length > 0 || activeKeyframes.length > 0" class="space-y-4">

        <!-- Event timeline -->
        <div v-if="activeEvents.length > 0" class="card-skeuo overflow-hidden">
          <div class="px-5 py-4 border-b border-[#1e2633] flex items-center justify-between">
            <h3 class="text-[#e2e8f0] font-medium text-sm">Event Timeline</h3>
            <span class="badge badge-amber mono">{{ activeEvents.length }}</span>
          </div>
          <div class="divide-y divide-[#1e2633] max-h-64 overflow-y-auto">
            <div
              v-for="evt in activeEvents"
              :key="`${evt.frame_index}-${evt.event_type}`"
              class="flex items-center gap-3 px-5 py-2.5"
            >
              <div
                class="w-2 h-2 rounded-full shrink-0"
                :style="`background: ${eventColor(evt.event_type)}; box-shadow: 0 0 5px ${eventColor(evt.event_type)}55`"
              />
              <span class="mono text-xs text-[#475569] shrink-0">f{{ evt.frame_index }}</span>
              <span class="mono text-xs font-medium shrink-0" :style="`color: ${eventColor(evt.event_type)}`">
                {{ evt.event_type.replace(/_/g, ' ') }}
              </span>
              <span class="text-[#94a3b8] text-xs flex-1 truncate">{{ evt.description }}</span>
              <span class="mono text-[10px] text-[#475569] shrink-0">{{ formatPts(evt.pts_ms) }}</span>
            </div>
          </div>
        </div>

        <!-- Keyframe gallery -->
        <div v-if="activeKeyframes.length > 0" class="card-skeuo p-5">
          <h3 class="text-[#e2e8f0] font-medium text-sm mb-4">
            Keyframes
            <span class="ml-2 badge badge-teal">{{ activeKeyframes.length }}</span>
          </h3>
          <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            <div
              v-for="kf in activeKeyframes"
              :key="kf.id"
              class="relative rounded-[10px] overflow-hidden"
              style="background: #050507; border: 1px solid #1e2633; aspect-ratio: 16/9;"
            >
              <img
                v-if="kf.image_url"
                :src="kf.image_url"
                :alt="`Keyframe ${kf.frame_index}`"
                class="w-full h-full object-cover"
                loading="lazy"
              />
              <div
                class="absolute bottom-0 left-0 right-0 p-2"
                style="background: linear-gradient(0deg, rgba(5,5,7,0.9) 0%, transparent 100%);"
              >
                <p v-if="kf.description" class="text-[#94a3b8] text-xs line-clamp-2 leading-snug">
                  {{ kf.description }}
                </p>
                <span class="mono text-[#475569] text-[10px]">f{{ kf.frame_index }}</span>
              </div>
            </div>
          </div>
        </div>

      </div>
    </template>

    <!-- ── Upload form ─────────────────────────────────────────────────────── -->
    <template v-else>

      <!-- Dropzone -->
      <div
        ref="dropzone"
        class="relative rounded-[12px] transition-all duration-200 cursor-pointer"
        :style="isDragging
          ? 'background: rgba(45,212,191,0.04); border: 2px dashed #2dd4bf; box-shadow: 0 0 20px rgba(45,212,191,0.1); min-height: 200px;'
          : 'background: linear-gradient(145deg, #0f1117, #151a22); border: 2px dashed #1e2633; min-height: 200px;'"
        role="button"
        tabindex="0"
        :aria-disabled="isProcessing"
        aria-label="Drop video file here or click to browse"
        @dragover="onDragover"
        @dragleave="onDragleave"
        @drop="onDrop"
        @click="!isProcessing && fileInputRef?.click()"
        @keydown.enter="!isProcessing && fileInputRef?.click()"
        @keydown.space.prevent="!isProcessing && fileInputRef?.click()"
      >
        <input
          ref="fileInputRef"
          type="file"
          accept=".mp4,.webm,.mov"
          class="sr-only"
          aria-hidden="true"
          @change="onFileInput"
        />

        <div class="flex flex-col items-center justify-center py-12 px-6 text-center">
          <div
            class="w-14 h-14 rounded-[12px] flex items-center justify-center mb-4 transition-colors duration-200"
            :style="isDragging
              ? 'background: rgba(45,212,191,0.15);'
              : 'background: rgba(255,255,255,0.04); border: 1px solid #1e2633;'"
          >
            <AnimatedIcon
              :icon="UploadCloud"
              :size="24"
              :stroke-width="1.75"
              :animation="isDragging ? 'bounce' : undefined"
              :class="isDragging ? 'text-[#2dd4bf]' : 'text-[#64748b]'"
            />
          </div>

          <template v-if="!selectedFile">
            <div class="text-[#e2e8f0] font-medium mb-1">Drop your video here</div>
            <div class="text-[#475569] text-sm mb-3">or click to browse files</div>
            <div class="text-[#475569] text-xs">Supports .mp4, .webm, .mov</div>
          </template>
          <template v-else>
            <div class="text-[#2dd4bf] font-medium">{{ selectedFile.name }}</div>
            <div class="mono text-[#64748b] text-sm mt-1">{{ fileSize }}</div>
            <button
              v-if="!isProcessing"
              class="min-h-11 px-2 text-xs text-[#475569] hover:text-[#64748b] mt-2 underline"
              @click.stop="reset"
            >
              Remove
            </button>
          </template>
        </div>
      </div>

      <!-- Error banner -->
      <div
        v-if="uploadError"
        class="flex items-center gap-2 p-3 rounded-[10px] text-sm"
        style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);"
      >
        <AnimatedIcon
          :icon="AlertCircle"
          :size="14"
          :stroke-width="2"
          animation="fade-in"
          class="text-[#ef4444] shrink-0"
        />
        <span class="text-[#ef4444] flex-1">{{ uploadError }}</span>
        <button class="min-h-11 px-2 text-xs text-[#ef4444] underline ml-2" @click="reset">Reset</button>
      </div>

      <!-- Settings panel -->
      <div v-if="selectedFile && !isProcessing" class="card-skeuo p-5 space-y-4">
        <h3 class="text-[#e2e8f0] font-medium text-sm">Analysis Settings</h3>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <!-- Model selector -->
          <div class="space-y-1.5">
            <label class="text-xs text-[#64748b] uppercase tracking-wide">Model</label>
            <select
              v-model="selectedModel"
              class="w-full min-h-11 px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] transition-colors duration-200"
              style="background: #0f1117; border: 1px solid #1e2633; outline: none;"
            >
              <option v-for="m in MODELS" :key="m.id" :value="m.id">{{ m.label }}</option>
            </select>
          </div>

          <!-- Prompt -->
          <div class="space-y-1.5 sm:col-span-2">
            <label class="text-xs text-[#64748b] uppercase tracking-wide">Analysis Prompt</label>
            <textarea
              v-model="prompt"
              rows="2"
              :placeholder="DEFAULT_ANALYSIS_PROMPT"
              class="w-full px-3 py-2 rounded-[8px] text-sm text-[#e2e8f0] placeholder-[#475569] resize-none transition-colors duration-200"
              style="background: #0f1117; border: 1px solid #1e2633; outline: none; font-family: inherit;"
              aria-label="Analysis prompt for VLM"
            />
            <p class="text-[#475569] text-xs">Sent as <span class="mono">semantic_prompt</span>. Leave blank to use the default.</p>
          </div>
        </div>

        <!-- Semantic inference toggle -->
        <button
          type="button"
          role="switch"
          :aria-checked="semanticInference"
          class="flex min-h-11 items-center gap-3 cursor-pointer w-fit text-left"
          @click="semanticInference = !semanticInference"
        >
          <span
            aria-hidden="true"
            class="relative w-9 h-5 rounded-full transition-colors duration-200"
            :style="semanticInference
              ? 'background: #0d9488;'
              : 'background: #1e2633;'"
          >
            <span
              class="absolute top-0.5 w-4 h-4 rounded-full transition-all duration-200"
              style="background: #e2e8f0;"
              :style="semanticInference ? 'left: 18px;' : 'left: 2px;'"
            />
          </span>
          <span class="text-sm text-[#94a3b8]">Semantic inference</span>
        </button>
      </div>

      <!-- Progress card -->
      <div v-if="isProcessing" class="card-skeuo p-5 space-y-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <div class="w-4 h-4 rounded-full border-2 border-[#1e2633] border-t-[#2dd4bf] animate-spin" />
            <span class="text-[#e2e8f0] text-sm font-medium">{{ statusLabel }}</span>
          </div>
          <span
            v-if="uploadState === 'uploading'"
            class="mono text-[#f59e0b] text-sm"
          >
            {{ uploadProgress }}%
          </span>
          <span
            v-else-if="uploadState === 'analyzing' && createdRunId"
            class="mono text-[#f59e0b] text-sm"
          >
            {{ activeEvents.length }} events
          </span>
        </div>

        <!-- Upload progress bar (only during upload phase) -->
        <div v-if="uploadState === 'uploading'" class="h-1.5 rounded-full overflow-hidden" style="background: #1e2633;">
          <div
            class="h-full rounded-full progress-bar-active transition-all duration-300"
            style="background: linear-gradient(90deg, #0d9488, #2dd4bf);"
            :style="{ width: `${uploadProgress}%` }"
          />
        </div>

        <p class="text-[#475569] text-xs">
          <span v-if="uploadState === 'uploading'">Transferring to server…</span>
          <span v-else-if="uploadState === 'creating'">Setting up analysis run…</span>
          <span v-else>
            Running semantic analysis.
            {{ timelineConnected ? 'Events are updating from the local timeline.' : 'Connecting to the local timeline…' }}
          </span>
        </p>

        <!-- Stream processing visualization -->
        <ProcessingVisualization
          v-if="uploadState === 'analyzing'"
          :total-frames="vizTotalFrames"
          :processed-frames="vizProcessed"
          :keyframe-count="activeKeyframes.length"
          :event-count="activeEvents.length"
          :current-chunk-start="vizChunkStart"
          :current-chunk-end="vizChunkEnd"
          :fps="vizFps"
          :is-processing="true"
          :vlm-frames="vizVlmFrames"
          :keyframe-indices="vizKfIndices"
        />

        <!-- Inline live event feed during analysis -->
        <div v-if="uploadState === 'analyzing' && activeEvents.length > 0">
          <div
            class="rounded-[10px] overflow-hidden"
            style="border: 1px solid #1e2633;"
          >
            <div class="px-4 py-2.5 flex items-center gap-2" style="border-bottom: 1px solid #1e2633;">
              <AnimatedIcon
                :icon="Zap"
                :size="12"
                :stroke-width="2"
                animation="pulse"
                class="text-[#2dd4bf] icon-glow-teal shrink-0"
              />
              <span class="text-xs text-[#94a3b8] uppercase tracking-wide">Live Events</span>
            </div>
            <div class="divide-y divide-[#1e2633] max-h-48 overflow-y-auto">
              <div
                v-for="(evt, i) in activeEvents.slice(-10)"
                :key="i"
                class="flex items-center gap-3 px-4 py-2.5"
              >
                <div
                  class="w-1.5 h-1.5 rounded-full shrink-0"
                  :style="{ background: eventColor(evt.event_type) }"
                />
                <span
                  class="mono text-[10px] shrink-0"
                  :style="{ color: eventColor(evt.event_type) }"
                >
                  {{ evt.event_type.replace(/_/g, ' ') }}
                </span>
                <span class="text-[#94a3b8] text-xs flex-1 truncate">{{ evt.description }}</span>
                <span class="mono text-[#475569] text-[10px] shrink-0">{{ formatPts(evt.pts_ms) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Submit button -->
      <div v-if="selectedFile && !isProcessing && uploadState !== 'error'" class="flex justify-end">
        <button
          class="flex min-h-11 items-center gap-2 px-6 py-2.5 rounded-[10px] text-sm font-semibold text-[#08090d] transition-all duration-200 icon-hover-parent"
          style="background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
                 box-shadow: 0 0 20px rgba(45,212,191,0.2);"
          @click="startUpload"
        >
          <AnimatedIcon :icon="Upload" :size="14" :stroke-width="2" />
          Start Analysis
        </button>
      </div>

    </template>
  </div>
</template>
