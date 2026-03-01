<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { api } from '@/lib/api'
import { Check, ChevronDown, RefreshCw } from 'lucide-vue-next'
import AnimatedIcon from '@/components/icons/AnimatedIcon.vue'

const authStore = useAuthStore()

// ── Defaults ──────────────────────────────────────────────────────────────────

const DEFAULTS = {
  apiEndpoint: 'http://localhost:8080',
  apiKey: '',
  spacetimeEndpoint: 'http://localhost:3000',
  turnUrl: '',
  fps: 5,
  chunkSize: 30,
  tokenRateCap: 0,
  clipMode: false,
  semanticInference: true,
  semanticFramesPerChunk: 4,
  sceneCutHammingThreshold: 10,
  lumaShiftThreshold: 0.15,
  loopDetection: true,
  loopRepeatTrigger: 3,
  gpuDecode: 'auto',
  vlmWorkers: 4,
  defaultModel: 'Qwen/Qwen3-VL-4B-Instruct',
  firstPassModel: 'Qwen/Qwen3-VL-2B-Instruct',
  secondPassModel: 'Qwen/Qwen3-VL-4B-Instruct',
} as const

function ls(key: string, fallback: string): string {
  return localStorage.getItem(key) ?? fallback
}
function lsNum(key: string, fallback: number): number {
  const v = localStorage.getItem(key)
  return v !== null ? Number(v) : fallback
}
function lsBool(key: string, fallback: boolean): boolean {
  const v = localStorage.getItem(key)
  return v !== null ? v !== 'false' : fallback
}

// ── Reactive form ─────────────────────────────────────────────────────────────

const form = reactive({
  // Connection
  apiEndpoint:       authStore.apiEndpoint,
  apiKey:            authStore.apiKey,
  spacetimeEndpoint: authStore.spacetimeEndpoint,
  turnUrl:           ls('vidarax_turn_url', DEFAULTS.turnUrl),
  // Models
  defaultModel:      ls('vidarax_default_model',    DEFAULTS.defaultModel),
  firstPassModel:    ls('vidarax_first_pass_model',  DEFAULTS.firstPassModel),
  secondPassModel:   ls('vidarax_second_pass_model', DEFAULTS.secondPassModel),
  // Stream
  fps:                    lsNum('vidarax_fps', DEFAULTS.fps),
  chunkSize:              lsNum('vidarax_chunk_size', DEFAULTS.chunkSize),
  tokenRateCap:           lsNum('vidarax_token_rate_cap', DEFAULTS.tokenRateCap),
  clipMode:               lsBool('vidarax_clip_mode', DEFAULTS.clipMode),
  semanticInference:      lsBool('vidarax_semantic_inference', DEFAULTS.semanticInference),
  semanticFramesPerChunk: lsNum('vidarax_semantic_frames_per_chunk', DEFAULTS.semanticFramesPerChunk),
  // Gate engine
  sceneCutHammingThreshold: lsNum('vidarax_scene_cut_hamming_threshold', DEFAULTS.sceneCutHammingThreshold),
  lumaShiftThreshold:       lsNum('vidarax_luma_shift_threshold', DEFAULTS.lumaShiftThreshold),
  loopDetection:            lsBool('vidarax_loop_detection', DEFAULTS.loopDetection),
  loopRepeatTrigger:        lsNum('vidarax_loop_repeat_trigger', DEFAULTS.loopRepeatTrigger),
  // Advanced
  gpuDecode:  ls('vidarax_gpu_decode', DEFAULTS.gpuDecode),
  vlmWorkers: lsNum('vidarax_vlm_workers', DEFAULTS.vlmWorkers),
})

// ── Models fetched from API ───────────────────────────────────────────────────

const availableModels = ref<Array<{ id: string; name: string }>>([
  { id: 'Qwen/Qwen3-VL-8B-Instruct',  name: 'Qwen3-VL 8B'    },
  { id: 'Qwen/Qwen3-VL-4B-Instruct',  name: 'Qwen3-VL 4B'    },
  { id: 'Qwen/Qwen3-VL-2B-Instruct',  name: 'Qwen3-VL 2B'    },
  { id: 'OpenGVLab/InternVL3_5-4B',   name: 'InternVL3.5 4B' },
  { id: 'LiquidAI/LFM2.5-VL-1.6B',   name: 'LFM2.5-VL 1.6B' },
])
const modelsLoading = ref(false)

async function fetchModels(): Promise<void> {
  modelsLoading.value = true
  try {
    const res = await api.models()
    if (res.models?.length) {
      availableModels.value = res.models.map(m => ({
        id: m.id,
        name: m.name ?? m.id,
      }))
    }
  } catch { /* use defaults */ } finally {
    modelsLoading.value = false
  }
}

onMounted(fetchModels)

// ── Connection test ───────────────────────────────────────────────────────────

type TestState = 'idle' | 'testing' | 'ok' | 'error'
const apiTestState  = ref<TestState>('idle')
const stdbTestState = ref<TestState>('idle')
const testError = ref<string | null>(null)

async function testConnection(): Promise<void> {
  apiTestState.value  = 'testing'
  stdbTestState.value = 'testing'
  testError.value = null

  // Run both pings in parallel
  const [apiResult, stdbResult] = await Promise.allSettled([
    pingApi(),
    pingSpacetime(),
  ])

  apiTestState.value  = apiResult.status  === 'fulfilled' ? 'ok' : 'error'
  stdbTestState.value = stdbResult.status === 'fulfilled' ? 'ok' : 'error'

  if (apiResult.status === 'rejected') {
    const err = apiResult.reason
    testError.value = err instanceof Error ? err.message : 'API unreachable'
  }
}

async function pingApi(): Promise<void> {
  const res = await fetch(`${form.apiEndpoint}/v1/health`, {
    headers: form.apiKey ? { 'x-api-key': form.apiKey } : {},
    signal: AbortSignal.timeout(5000),
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
}

function pingSpacetime(): Promise<void> {
  return new Promise((resolve, reject) => {
    const url = new URL(form.spacetimeEndpoint.replace(/\/$/, ''))
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
    url.pathname = '/v1/database/vidarax/subscribe'
    const ws = new WebSocket(url.toString(), ['v1.json.spacetimedb'])
    const t = setTimeout(() => { ws.close(); reject(new Error('SpacetimeDB timeout')) }, 4000)
    ws.onopen = () => { clearTimeout(t); ws.close(1000); resolve() }
    ws.onerror = () => { clearTimeout(t); reject(new Error('SpacetimeDB unreachable')) }
  })
}

// ── Save / Reset ──────────────────────────────────────────────────────────────

const saved = ref(false)

function saveSettings(): void {
  // Auth store (also persists to localStorage)
  authStore.setApiEndpoint(form.apiEndpoint)
  authStore.setApiKey(form.apiKey)
  authStore.setSpacetimeEndpoint(form.spacetimeEndpoint)
  localStorage.setItem('vidarax_turn_url', form.turnUrl)

  // Models
  localStorage.setItem('vidarax_default_model',    form.defaultModel)
  localStorage.setItem('vidarax_first_pass_model',  form.firstPassModel)
  localStorage.setItem('vidarax_second_pass_model', form.secondPassModel)

  // Stream
  localStorage.setItem('vidarax_fps',                      String(form.fps))
  localStorage.setItem('vidarax_chunk_size',               String(form.chunkSize))
  localStorage.setItem('vidarax_token_rate_cap',           String(form.tokenRateCap))
  localStorage.setItem('vidarax_clip_mode',                String(form.clipMode))
  localStorage.setItem('vidarax_semantic_inference',       String(form.semanticInference))
  localStorage.setItem('vidarax_semantic_frames_per_chunk', String(form.semanticFramesPerChunk))

  // Gate engine
  localStorage.setItem('vidarax_scene_cut_hamming_threshold', String(form.sceneCutHammingThreshold))
  localStorage.setItem('vidarax_luma_shift_threshold',        String(form.lumaShiftThreshold))
  localStorage.setItem('vidarax_loop_detection',              String(form.loopDetection))
  localStorage.setItem('vidarax_loop_repeat_trigger',         String(form.loopRepeatTrigger))

  // Advanced
  localStorage.setItem('vidarax_gpu_decode',   form.gpuDecode)
  localStorage.setItem('vidarax_vlm_workers',  String(form.vlmWorkers))

  saved.value = true
  setTimeout(() => { saved.value = false }, 2000)
}

function resetToDefaults(): void {
  Object.assign(form, {
    apiEndpoint:               DEFAULTS.apiEndpoint,
    apiKey:                    DEFAULTS.apiKey,
    spacetimeEndpoint:         DEFAULTS.spacetimeEndpoint,
    turnUrl:                   DEFAULTS.turnUrl,
    defaultModel:              DEFAULTS.defaultModel,
    firstPassModel:            DEFAULTS.firstPassModel,
    secondPassModel:           DEFAULTS.secondPassModel,
    fps:                       DEFAULTS.fps,
    chunkSize:                 DEFAULTS.chunkSize,
    tokenRateCap:              DEFAULTS.tokenRateCap,
    clipMode:                  DEFAULTS.clipMode,
    semanticInference:         DEFAULTS.semanticInference,
    semanticFramesPerChunk:    DEFAULTS.semanticFramesPerChunk,
    sceneCutHammingThreshold:  DEFAULTS.sceneCutHammingThreshold,
    lumaShiftThreshold:        DEFAULTS.lumaShiftThreshold,
    loopDetection:             DEFAULTS.loopDetection,
    loopRepeatTrigger:         DEFAULTS.loopRepeatTrigger,
    gpuDecode:                 DEFAULTS.gpuDecode,
    vlmWorkers:                DEFAULTS.vlmWorkers,
  })
}

// ── Section accordion ─────────────────────────────────────────────────────────

type SectionKey = 'connection' | 'models' | 'stream' | 'gate' | 'advanced'
const openSection = ref<SectionKey>('connection')

function toggle(section: SectionKey): void {
  openSection.value = openSection.value === section ? 'connection' : section
}
</script>

<template>
  <div class="p-4 lg:p-6 space-y-4 max-w-2xl">

    <!-- Header -->
    <div class="flex items-center justify-between flex-wrap gap-3">
      <div>
        <h2 class="text-[#e2e8f0] font-semibold text-xl">Settings</h2>
        <p class="text-[#64748b] text-sm mt-0.5">Configure your Vidarax environment</p>
      </div>
      <div class="flex gap-2">
        <button
          class="px-4 py-2 rounded-[10px] text-sm font-medium text-[#64748b] transition-all duration-200"
          style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
          @click="resetToDefaults"
        >
          Reset Defaults
        </button>
        <button
          data-testid="save-button"
          class="flex items-center gap-2 px-5 py-2 rounded-[10px] text-sm font-semibold transition-all duration-200"
          :class="saved ? 'text-[#22c55e]' : 'text-[#08090d]'"
          :style="saved
            ? 'background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.2);'
            : 'background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%); box-shadow: 0 0 20px rgba(45,212,191,0.2);'"
          @click="saveSettings"
        >
          <AnimatedIcon
            v-if="saved"
            :icon="Check"
            :size="14"
            :stroke-width="2"
            animation="draw-in"
            class="icon-glow-teal"
          />
          {{ saved ? 'Saved!' : 'Save' }}
        </button>
      </div>
    </div>

    <!-- ── Connection ──────────────────────────────────────────────────────── -->
    <div class="card-skeuo overflow-hidden">
      <button
        class="w-full flex items-center justify-between px-5 py-4 text-left"
        @click="toggle('connection')"
      >
        <div class="flex items-center gap-2">
          <span class="text-[#e2e8f0] font-medium text-sm">Connection</span>
          <!-- Status pills -->
          <span
            v-if="apiTestState !== 'idle'"
            class="badge"
            :class="apiTestState === 'ok' ? 'badge-green' : apiTestState === 'error' ? 'badge-red' : 'badge-muted'"
          >
            API {{ apiTestState === 'ok' ? '✓' : apiTestState === 'error' ? '✗' : '…' }}
          </span>
          <span
            v-if="stdbTestState !== 'idle'"
            class="badge"
            :class="stdbTestState === 'ok' ? 'badge-teal' : stdbTestState === 'error' ? 'badge-red' : 'badge-muted'"
          >
            STDB {{ stdbTestState === 'ok' ? '✓' : stdbTestState === 'error' ? '✗' : '…' }}
          </span>
        </div>
        <AnimatedIcon
          :icon="ChevronDown"
          :size="14"
          :stroke-width="2"
          animation="draw-in"
          class="text-[#475569] transition-transform duration-200"
          :class="openSection === 'connection' ? 'rotate-180' : ''"
        />
      </button>

      <div v-if="openSection === 'connection'"
           class="px-5 pb-5 pt-4 space-y-4 border-t border-[#1e2633]">

        <!-- API endpoint -->
        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="api-endpoint">Vidarax API URL</label>
          <input
            id="api-endpoint"
            v-model="form.apiEndpoint"
            data-testid="api-endpoint"
            type="url"
            placeholder="http://localhost:8080"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] placeholder-[#475569] outline-none transition-colors duration-200 focus:border-[#2dd4bf44]"
            style="background: #050507; border: 1px solid #1e2633;"
          />
        </div>

        <!-- API key -->
        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="api-key">API Key</label>
          <input
            id="api-key"
            v-model="form.apiKey"
            data-testid="api-key"
            type="password"
            placeholder="Leave blank if VIDARAX_REQUIRE_API_KEY=false"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] placeholder-[#475569] outline-none transition-colors duration-200"
            style="background: #050507; border: 1px solid #1e2633;"
          />
        </div>

        <!-- SpacetimeDB URL -->
        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="spacetime-endpoint">SpacetimeDB URL</label>
          <input
            id="spacetime-endpoint"
            v-model="form.spacetimeEndpoint"
            data-testid="spacetime-endpoint"
            type="url"
            placeholder="http://localhost:3000"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] placeholder-[#475569] outline-none transition-colors duration-200"
            style="background: #050507; border: 1px solid #1e2633;"
          />
        </div>

        <!-- TURN server URL -->
        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="turn-url">TURN Server URL</label>
          <div class="text-[#475569] text-xs">Optional — required behind symmetric NATs</div>
          <input
            id="turn-url"
            v-model="form.turnUrl"
            data-testid="turn-url"
            type="url"
            placeholder="turn:turn.example.com:3478?transport=tcp"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] placeholder-[#475569] outline-none transition-colors duration-200"
            style="background: #050507; border: 1px solid #1e2633;"
          />
        </div>

        <div class="flex items-center gap-3 flex-wrap">
          <button
            data-testid="test-connection"
            class="px-4 py-2 rounded-[8px] text-sm font-medium transition-all duration-200"
            style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
            :class="{
              'text-[#94a3b8]': apiTestState === 'idle' || apiTestState === 'testing',
              'text-[#22c55e]': apiTestState === 'ok' && stdbTestState === 'ok',
              'text-[#ef4444]': apiTestState === 'error' || stdbTestState === 'error',
            }"
            :disabled="apiTestState === 'testing'"
            @click="testConnection"
          >
            <span v-if="apiTestState === 'testing'">Testing…</span>
            <span v-else-if="apiTestState === 'ok' && stdbTestState === 'ok'">All Connected ✓</span>
            <span v-else-if="apiTestState === 'error' || stdbTestState === 'error'">Connection Failed</span>
            <span v-else>Test Connection</span>
          </button>
          <span v-if="testError" class="text-[#ef4444] text-xs">{{ testError }}</span>
        </div>
      </div>
    </div>

    <!-- ── Models ──────────────────────────────────────────────────────────── -->
    <div class="card-skeuo overflow-hidden">
      <button
        class="w-full flex items-center justify-between px-5 py-4 text-left"
        @click="toggle('models')"
      >
        <div class="flex items-center gap-2">
          <span class="text-[#e2e8f0] font-medium text-sm">Models</span>
          <span v-if="modelsLoading" class="badge badge-muted">Fetching…</span>
        </div>
        <AnimatedIcon
          :icon="ChevronDown"
          :size="14"
          :stroke-width="2"
          animation="draw-in"
          class="text-[#475569] transition-transform duration-200"
          :class="openSection === 'models' ? 'rotate-180' : ''"
        />
      </button>

      <div v-if="openSection === 'models'"
           class="px-5 pb-5 pt-4 space-y-4 border-t border-[#1e2633]">

        <!-- Default model -->
        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="default-model">Default Model</label>
          <div class="text-[#475569] text-xs mb-1">Used for new runs from Upload and Stream pages</div>
          <select
            id="default-model"
            v-model="form.defaultModel"
            data-testid="default-model"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] outline-none"
            style="background: #050507; border: 1px solid #1e2633;"
          >
            <option v-for="m in availableModels" :key="m.id" :value="m.id">{{ m.name }}</option>
          </select>
        </div>

        <!-- Tiered routing -->
        <div
          class="p-4 rounded-[10px] space-y-4"
          style="background: rgba(245,158,11,0.04); border: 1px solid rgba(245,158,11,0.12);"
        >
          <div class="flex items-center gap-2">
            <span class="text-[#f59e0b] text-xs font-semibold uppercase tracking-wide">Tiered Routing</span>
            <span class="badge badge-amber">2-pass</span>
          </div>
          <div class="text-[#64748b] text-xs leading-relaxed">
            First pass runs a lighter model on all frames. Second pass runs the full model only on high-confidence keyframes.
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div class="space-y-1.5">
              <label class="text-[#94a3b8] text-xs font-medium" for="first-pass">First Pass Model</label>
              <select
                id="first-pass"
                v-model="form.firstPassModel"
                data-testid="first-pass-model"
                class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] outline-none"
                style="background: #050507; border: 1px solid #1e2633;"
              >
                <option v-for="m in availableModels" :key="m.id" :value="m.id">{{ m.name }}</option>
              </select>
            </div>

            <div class="space-y-1.5">
              <label class="text-[#94a3b8] text-xs font-medium" for="second-pass">Second Pass Model</label>
              <select
                id="second-pass"
                v-model="form.secondPassModel"
                data-testid="second-pass-model"
                class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] outline-none"
                style="background: #050507; border: 1px solid #1e2633;"
              >
                <option v-for="m in availableModels" :key="m.id" :value="m.id">{{ m.name }}</option>
              </select>
            </div>
          </div>
        </div>

        <button
          class="flex items-center gap-1.5 text-xs text-[#2dd4bf] hover:text-[#5eead4] transition-colors icon-scale-hover"
          :disabled="modelsLoading"
          @click="fetchModels"
        >
          <AnimatedIcon
            :icon="RefreshCw"
            :size="12"
            :stroke-width="2"
            :animation="modelsLoading ? 'spin' : undefined"
          />
          {{ modelsLoading ? 'Fetching models…' : 'Refresh from API' }}
        </button>
      </div>
    </div>

    <!-- ── Stream ──────────────────────────────────────────────────────────── -->
    <div class="card-skeuo overflow-hidden">
      <button
        class="w-full flex items-center justify-between px-5 py-4 text-left"
        @click="toggle('stream')"
      >
        <span class="text-[#e2e8f0] font-medium text-sm">Stream</span>
        <AnimatedIcon
          :icon="ChevronDown"
          :size="14"
          :stroke-width="2"
          animation="draw-in"
          class="text-[#475569] transition-transform duration-200"
          :class="openSection === 'stream' ? 'rotate-180' : ''"
        />
      </button>

      <div v-if="openSection === 'stream'"
           class="px-5 pb-5 pt-4 space-y-5 border-t border-[#1e2633]">

        <!-- FPS -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">Analysis FPS</label>
            <span class="mono text-[#f59e0b] text-sm" data-testid="fps-value">{{ form.fps }}</span>
          </div>
          <input
            v-model.number="form.fps"
            data-testid="fps-slider"
            type="range" min="0.5" max="30" step="0.5"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
          <div class="flex justify-between text-xs text-[#475569] mono">
            <span>0.5 fps</span><span>30 fps</span>
          </div>
        </div>

        <!-- Chunk size -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">Chunk Size <span class="text-[#475569]">(frames)</span></label>
            <span class="mono text-[#f59e0b] text-sm">{{ form.chunkSize }}</span>
          </div>
          <input
            v-model.number="form.chunkSize"
            type="range" min="5" max="120" step="5"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
          <div class="flex justify-between text-xs text-[#475569] mono">
            <span>5</span><span>120</span>
          </div>
        </div>

        <!-- Token rate cap -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">Token Rate Cap <span class="text-[#475569]">(tokens/s, 0 = unlimited)</span></label>
            <span class="mono text-[#f59e0b] text-sm" data-testid="token-rate-cap-value">{{ form.tokenRateCap === 0 ? '∞' : form.tokenRateCap }}</span>
          </div>
          <input
            v-model.number="form.tokenRateCap"
            data-testid="token-rate-cap"
            type="range" min="0" max="500" step="10"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
          <div class="flex justify-between text-xs text-[#475569] mono">
            <span>unlimited</span><span>500 t/s</span>
          </div>
        </div>

        <!-- Clip mode toggle -->
        <div class="flex items-center justify-between">
          <div>
            <div class="text-[#94a3b8] text-xs font-medium">Clip Mode</div>
            <div class="text-[#475569] text-xs mt-0.5">Send temporal multi-frame clips to VLM</div>
          </div>
          <button
            data-testid="clip-mode-toggle"
            class="relative rounded-full transition-colors duration-200 shrink-0"
            style="width:40px;height:22px;"
            :style="form.clipMode ? 'background:#2dd4bf;' : 'background:#1e2633;'"
            :aria-pressed="form.clipMode"
            role="switch"
            :aria-checked="form.clipMode"
            @click="form.clipMode = !form.clipMode"
          >
            <div
              class="absolute rounded-full bg-white transition-transform duration-200"
              style="top:3px;width:16px;height:16px;"
              :style="form.clipMode ? 'transform:translateX(21px)' : 'transform:translateX(3px)'"
            />
          </button>
        </div>

        <!-- Semantic inference toggle -->
        <div class="flex items-center justify-between">
          <div>
            <div class="text-[#94a3b8] text-xs font-medium">Semantic Inference</div>
            <div class="text-[#475569] text-xs mt-0.5">Run VLM on sampled keyframes</div>
          </div>
          <button
            data-testid="semantic-inference-toggle"
            class="relative rounded-full transition-colors duration-200 shrink-0"
            style="width:40px;height:22px;"
            :style="form.semanticInference ? 'background:#2dd4bf;' : 'background:#1e2633;'"
            :aria-pressed="form.semanticInference"
            role="switch"
            :aria-checked="form.semanticInference"
            @click="form.semanticInference = !form.semanticInference"
          >
            <div
              class="absolute rounded-full bg-white transition-transform duration-200"
              style="top:3px;width:16px;height:16px;"
              :style="form.semanticInference ? 'transform:translateX(21px)' : 'transform:translateX(3px)'"
            />
          </button>
        </div>

        <!-- Semantic frames per chunk -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">
              Semantic Frames / Chunk
              <span class="text-[#475569]">(passed to VLM)</span>
            </label>
            <span class="mono text-[#f59e0b] text-sm">{{ form.semanticFramesPerChunk }}</span>
          </div>
          <input
            v-model.number="form.semanticFramesPerChunk"
            type="range" min="1" max="4" step="1"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
          <div class="flex justify-between text-xs text-[#475569] mono">
            <span>1</span><span>4</span>
          </div>
        </div>
      </div>
    </div>

    <!-- ── Gate Engine ─────────────────────────────────────────────────────── -->
    <div class="card-skeuo overflow-hidden">
      <button
        class="w-full flex items-center justify-between px-5 py-4 text-left"
        @click="toggle('gate')"
      >
        <span class="text-[#e2e8f0] font-medium text-sm">Gate Engine</span>
        <AnimatedIcon
          :icon="ChevronDown"
          :size="14"
          :stroke-width="2"
          animation="draw-in"
          class="text-[#475569] transition-transform duration-200"
          :class="openSection === 'gate' ? 'rotate-180' : ''"
        />
      </button>

      <div v-if="openSection === 'gate'"
           class="px-5 pb-5 pt-4 space-y-5 border-t border-[#1e2633]">

        <!-- Scene cut Hamming threshold -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <div>
              <label class="text-[#94a3b8] text-xs font-medium">Scene Cut Hamming Threshold</label>
              <div class="text-[#475569] text-xs mt-0.5">pHash distance — lower = more sensitive</div>
            </div>
            <span class="mono text-[#f59e0b] text-sm">{{ form.sceneCutHammingThreshold }}</span>
          </div>
          <input
            v-model.number="form.sceneCutHammingThreshold"
            type="range" min="1" max="32" step="1"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
          <div class="flex justify-between text-xs text-[#475569] mono">
            <span>1 (strict)</span><span>32 (loose)</span>
          </div>
        </div>

        <!-- Luma shift threshold -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <div>
              <label class="text-[#94a3b8] text-xs font-medium">Luma Shift Threshold</label>
              <div class="text-[#475569] text-xs mt-0.5">Exposure change fraction to trigger event</div>
            </div>
            <span class="mono text-[#f59e0b] text-sm">{{ form.lumaShiftThreshold.toFixed(2) }}</span>
          </div>
          <input
            v-model.number="form.lumaShiftThreshold"
            type="range" min="0.01" max="1.0" step="0.01"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
          <div class="flex justify-between text-xs text-[#475569] mono">
            <span>0.01</span><span>1.0</span>
          </div>
        </div>

        <!-- Loop detection toggle -->
        <div class="flex items-center justify-between">
          <div>
            <div class="text-[#94a3b8] text-xs font-medium">Loop Detection</div>
            <div class="text-[#475569] text-xs mt-0.5">Detect repeating screen states via pHash ring</div>
          </div>
          <button
            class="relative rounded-full transition-colors duration-200 shrink-0"
            style="width:40px;height:22px;"
            :style="form.loopDetection ? 'background:#2dd4bf;' : 'background:#1e2633;'"
            :aria-pressed="form.loopDetection"
            role="switch"
            :aria-checked="form.loopDetection"
            @click="form.loopDetection = !form.loopDetection"
          >
            <div
              class="absolute rounded-full bg-white transition-transform duration-200"
              style="top:3px;width:16px;height:16px;"
              :style="form.loopDetection ? 'transform:translateX(21px)' : 'transform:translateX(3px)'"
            />
          </button>
        </div>

        <!-- Loop repeat trigger -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <div>
              <label class="text-[#94a3b8] text-xs font-medium">Loop Repeat Trigger</label>
              <div class="text-[#475569] text-xs mt-0.5">N identical frames before firing event</div>
            </div>
            <span class="mono text-[#f59e0b] text-sm">{{ form.loopRepeatTrigger }}</span>
          </div>
          <input
            v-model.number="form.loopRepeatTrigger"
            type="range" min="1" max="8" step="1"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
          <div class="flex justify-between text-xs text-[#475569] mono">
            <span>1</span><span>8</span>
          </div>
        </div>
      </div>
    </div>

    <!-- ── Advanced ────────────────────────────────────────────────────────── -->
    <div class="card-skeuo overflow-hidden">
      <button
        class="w-full flex items-center justify-between px-5 py-4 text-left"
        @click="toggle('advanced')"
      >
        <span class="text-[#e2e8f0] font-medium text-sm">Advanced</span>
        <AnimatedIcon
          :icon="ChevronDown"
          :size="14"
          :stroke-width="2"
          animation="draw-in"
          class="text-[#475569] transition-transform duration-200"
          :class="openSection === 'advanced' ? 'rotate-180' : ''"
        />
      </button>

      <div v-if="openSection === 'advanced'"
           class="px-5 pb-5 pt-4 space-y-5 border-t border-[#1e2633]">

        <!-- GPU decode -->
        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="gpu-decode">GPU Decode</label>
          <select
            id="gpu-decode"
            v-model="form.gpuDecode"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] outline-none"
            style="background: #050507; border: 1px solid #1e2633;"
          >
            <option value="auto">auto (detect nvidia-smi)</option>
            <option value="true">Enabled (NVDEC)</option>
            <option value="false">Disabled (CPU)</option>
          </select>
        </div>

        <!-- VLM workers -->
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <div>
              <label class="text-[#94a3b8] text-xs font-medium">VLM Workers</label>
              <div class="text-[#475569] text-xs mt-0.5">Saturate vLLM continuous batching — N=4 recommended</div>
            </div>
            <span class="mono text-[#f59e0b] text-sm">{{ form.vlmWorkers }}</span>
          </div>
          <input
            v-model.number="form.vlmWorkers"
            type="range" min="1" max="8" step="1"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
          <div class="flex justify-between text-xs text-[#475569] mono">
            <span>1</span><span>8</span>
          </div>
        </div>
      </div>
    </div>

  </div>
</template>
