<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useAuthStore } from '@/stores/auth'

const authStore = useAuthStore()

const form = reactive({
  apiEndpoint: authStore.apiEndpoint,
  apiKey: authStore.apiKey,
  spacetimeEndpoint: authStore.spacetimeEndpoint,

  // Stream settings
  fps: Number(localStorage.getItem('vidarax_fps') ?? '5'),
  chunkSize: Number(localStorage.getItem('vidarax_chunk_size') ?? '30'),
  semanticInference: localStorage.getItem('vidarax_semantic_inference') !== 'false',
  semanticFramesPerChunk: Number(localStorage.getItem('vidarax_semantic_frames_per_chunk') ?? '4'),

  // Gate engine
  sceneCutThreshold: Number(localStorage.getItem('vidarax_scene_cut_threshold') ?? '0.35'),
  exposureShiftThreshold: Number(localStorage.getItem('vidarax_exposure_shift_threshold') ?? '0.15'),
  loopDetection: localStorage.getItem('vidarax_loop_detection') !== 'false',
  loopRepeatTrigger: Number(localStorage.getItem('vidarax_loop_repeat_trigger') ?? '3'),

  // Advanced
  gpuDecode: localStorage.getItem('vidarax_gpu_decode') ?? 'auto',
  vlmWorkers: Number(localStorage.getItem('vidarax_vlm_workers') ?? '4'),
})

const testState = ref<'idle' | 'testing' | 'ok' | 'error'>('idle')
const testError = ref<string | null>(null)
const saved = ref(false)

async function testConnection() {
  testState.value = 'testing'
  testError.value = null
  try {
    const res = await fetch(`${form.apiEndpoint}/health`, {
      headers: form.apiKey ? { 'x-api-key': form.apiKey } : {},
      signal: AbortSignal.timeout(5000),
    })
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    testState.value = 'ok'
  } catch (err) {
    testState.value = 'error'
    testError.value = err instanceof Error ? err.message : 'Connection failed'
  }
}

function saveSettings() {
  authStore.setApiEndpoint(form.apiEndpoint)
  authStore.setApiKey(form.apiKey)
  authStore.setSpacetimeEndpoint(form.spacetimeEndpoint)

  localStorage.setItem('vidarax_fps', String(form.fps))
  localStorage.setItem('vidarax_chunk_size', String(form.chunkSize))
  localStorage.setItem('vidarax_semantic_inference', String(form.semanticInference))
  localStorage.setItem('vidarax_semantic_frames_per_chunk', String(form.semanticFramesPerChunk))
  localStorage.setItem('vidarax_scene_cut_threshold', String(form.sceneCutThreshold))
  localStorage.setItem('vidarax_exposure_shift_threshold', String(form.exposureShiftThreshold))
  localStorage.setItem('vidarax_loop_detection', String(form.loopDetection))
  localStorage.setItem('vidarax_loop_repeat_trigger', String(form.loopRepeatTrigger))
  localStorage.setItem('vidarax_gpu_decode', form.gpuDecode)
  localStorage.setItem('vidarax_vlm_workers', String(form.vlmWorkers))

  saved.value = true
  setTimeout(() => { saved.value = false }, 2000)
}

type SectionKey = 'connection' | 'stream' | 'gate' | 'advanced'
const openSection = ref<SectionKey>('connection')

function toggle(section: SectionKey) {
  openSection.value = openSection.value === section ? 'connection' : section
}
</script>

<template>
  <div class="p-4 lg:p-6 space-y-4 max-w-2xl">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-[#e2e8f0] font-semibold text-xl">Settings</h2>
        <p class="text-[#64748b] text-sm mt-1">Configure your Vidarax environment</p>
      </div>
      <button
        class="flex items-center gap-2 px-5 py-2 rounded-[10px] text-sm font-semibold transition-all duration-200"
        :class="saved
          ? 'text-[#22c55e]'
          : 'text-[#08090d]'"
        :style="saved
          ? 'background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.2);'
          : 'background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%); box-shadow: 0 0 20px rgba(45,212,191,0.2);'"
        @click="saveSettings"
      >
        <svg v-if="saved" width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M2 7L5.5 10.5L12 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        {{ saved ? 'Saved' : 'Save' }}
      </button>
    </div>

    <!-- Connection section -->
    <div class="card-skeuo overflow-hidden">
      <button
        class="w-full flex items-center justify-between px-5 py-4 text-left"
        @click="toggle('connection')"
      >
        <span class="text-[#e2e8f0] font-medium text-sm">Connection</span>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"
             class="text-[#475569] transition-transform duration-200"
             :class="openSection === 'connection' ? 'rotate-180' : ''">
          <path d="M3 5L7 9L11 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </button>

      <div v-if="openSection === 'connection'" class="px-5 pb-5 space-y-4 border-t border-[#1e2633]" style="padding-top: 16px;">
        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="api-endpoint">Vidarax API URL</label>
          <input
            id="api-endpoint"
            v-model="form.apiEndpoint"
            type="url"
            placeholder="http://localhost:8080"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] placeholder-[#475569] outline-none transition-all duration-200"
            style="background: #050507; border: 1px solid #1e2633;"
            @focus="($event.target as HTMLInputElement).style.borderColor = '#2dd4bf44'"
            @blur="($event.target as HTMLInputElement).style.borderColor = '#1e2633'"
          />
        </div>

        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="api-key">API Key</label>
          <input
            id="api-key"
            v-model="form.apiKey"
            type="password"
            placeholder="Leave blank if VIDARAX_REQUIRE_API_KEY=false"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] placeholder-[#475569] outline-none transition-all duration-200"
            style="background: #050507; border: 1px solid #1e2633;"
            @focus="($event.target as HTMLInputElement).style.borderColor = '#2dd4bf44'"
            @blur="($event.target as HTMLInputElement).style.borderColor = '#1e2633'"
          />
        </div>

        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="spacetime-endpoint">SpacetimeDB URL</label>
          <input
            id="spacetime-endpoint"
            v-model="form.spacetimeEndpoint"
            type="url"
            placeholder="http://localhost:3000"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] placeholder-[#475569] outline-none transition-all duration-200"
            style="background: #050507; border: 1px solid #1e2633;"
            @focus="($event.target as HTMLInputElement).style.borderColor = '#2dd4bf44'"
            @blur="($event.target as HTMLInputElement).style.borderColor = '#1e2633'"
          />
        </div>

        <div class="flex items-center gap-3">
          <button
            class="px-4 py-2 rounded-[8px] text-sm font-medium transition-all duration-200"
            :class="testState === 'ok' ? 'text-[#22c55e]' : testState === 'error' ? 'text-[#ef4444]' : 'text-[#94a3b8]'"
            style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
            :disabled="testState === 'testing'"
            @click="testConnection"
          >
            <span v-if="testState === 'testing'">Testing...</span>
            <span v-else-if="testState === 'ok'">Connected</span>
            <span v-else-if="testState === 'error'">Failed</span>
            <span v-else>Test Connection</span>
          </button>
          <span v-if="testError" class="text-[#ef4444] text-xs">{{ testError }}</span>
        </div>
      </div>
    </div>

    <!-- Stream section -->
    <div class="card-skeuo overflow-hidden">
      <button
        class="w-full flex items-center justify-between px-5 py-4 text-left"
        @click="toggle('stream')"
      >
        <span class="text-[#e2e8f0] font-medium text-sm">Stream</span>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"
             class="text-[#475569] transition-transform duration-200"
             :class="openSection === 'stream' ? 'rotate-180' : ''">
          <path d="M3 5L7 9L11 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </button>

      <div v-if="openSection === 'stream'" class="px-5 pb-5 space-y-5 border-t border-[#1e2633]" style="padding-top: 16px;">
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">Analysis FPS</label>
            <span class="mono text-[#f59e0b] text-sm">{{ form.fps }}</span>
          </div>
          <input
            v-model.number="form.fps"
            type="range" min="0.5" max="30" step="0.5"
            class="w-full accent-[#2dd4bf] cursor-pointer"
            style="height: 4px;"
          />
          <div class="flex justify-between text-xs text-[#475569] mono">
            <span>0.5</span><span>30</span>
          </div>
        </div>

        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">Chunk Size (frames)</label>
            <span class="mono text-[#f59e0b] text-sm">{{ form.chunkSize }}</span>
          </div>
          <input
            v-model.number="form.chunkSize"
            type="range" min="5" max="120" step="5"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
        </div>

        <div class="flex items-center justify-between">
          <div>
            <div class="text-[#94a3b8] text-xs font-medium">Semantic Inference</div>
            <div class="text-[#475569] text-xs mt-0.5">Run VLM on keyframes</div>
          </div>
          <button
            class="relative w-10 h-5.5 rounded-full transition-colors duration-200"
            :style="form.semanticInference
              ? 'background: #2dd4bf;'
              : 'background: #1e2633;'"
            style="width:40px;height:22px;"
            :aria-pressed="form.semanticInference"
            @click="form.semanticInference = !form.semanticInference"
          >
            <div
              class="absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-200"
              style="top:3px;width:16px;height:16px;"
              :style="form.semanticInference ? 'transform: translateX(21px)' : 'transform: translateX(3px)'"
            />
          </button>
        </div>
      </div>
    </div>

    <!-- Gate Engine section -->
    <div class="card-skeuo overflow-hidden">
      <button
        class="w-full flex items-center justify-between px-5 py-4 text-left"
        @click="toggle('gate')"
      >
        <span class="text-[#e2e8f0] font-medium text-sm">Gate Engine</span>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"
             class="text-[#475569] transition-transform duration-200"
             :class="openSection === 'gate' ? 'rotate-180' : ''">
          <path d="M3 5L7 9L11 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </button>

      <div v-if="openSection === 'gate'" class="px-5 pb-5 space-y-5 border-t border-[#1e2633]" style="padding-top: 16px;">
        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">Scene Cut Threshold</label>
            <span class="mono text-[#f59e0b] text-sm">{{ form.sceneCutThreshold.toFixed(2) }}</span>
          </div>
          <input
            v-model.number="form.sceneCutThreshold"
            type="range" min="0.1" max="0.9" step="0.05"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
        </div>

        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">Exposure Shift Threshold</label>
            <span class="mono text-[#f59e0b] text-sm">{{ form.exposureShiftThreshold.toFixed(2) }}</span>
          </div>
          <input
            v-model.number="form.exposureShiftThreshold"
            type="range" min="0.05" max="0.5" step="0.05"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
        </div>

        <div class="flex items-center justify-between">
          <div>
            <div class="text-[#94a3b8] text-xs font-medium">Loop Detection</div>
            <div class="text-[#475569] text-xs mt-0.5">Detect repeating screen states</div>
          </div>
          <button
            class="relative rounded-full transition-colors duration-200"
            style="width:40px;height:22px;"
            :style="form.loopDetection ? 'background: #2dd4bf;' : 'background: #1e2633;'"
            :aria-pressed="form.loopDetection"
            @click="form.loopDetection = !form.loopDetection"
          >
            <div
              class="absolute rounded-full bg-white transition-transform duration-200"
              style="top:3px;width:16px;height:16px;"
              :style="form.loopDetection ? 'transform: translateX(21px)' : 'transform: translateX(3px)'"
            />
          </button>
        </div>

        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">Loop Repeat Trigger</label>
            <span class="mono text-[#f59e0b] text-sm">{{ form.loopRepeatTrigger }}</span>
          </div>
          <input
            v-model.number="form.loopRepeatTrigger"
            type="range" min="2" max="10" step="1"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
        </div>
      </div>
    </div>

    <!-- Advanced section -->
    <div class="card-skeuo overflow-hidden">
      <button
        class="w-full flex items-center justify-between px-5 py-4 text-left"
        @click="toggle('advanced')"
      >
        <span class="text-[#e2e8f0] font-medium text-sm">Advanced</span>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"
             class="text-[#475569] transition-transform duration-200"
             :class="openSection === 'advanced' ? 'rotate-180' : ''">
          <path d="M3 5L7 9L11 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </button>

      <div v-if="openSection === 'advanced'" class="px-5 pb-5 space-y-5 border-t border-[#1e2633]" style="padding-top: 16px;">
        <div class="space-y-1.5">
          <label class="text-[#94a3b8] text-xs font-medium" for="gpu-decode">GPU Decode</label>
          <select
            id="gpu-decode"
            v-model="form.gpuDecode"
            class="w-full px-3 py-2 rounded-[8px] mono text-sm text-[#e2e8f0] outline-none transition-all duration-200"
            style="background: #050507; border: 1px solid #1e2633;"
          >
            <option value="auto">auto (detect nvidia-smi)</option>
            <option value="true">Enabled (NVDEC)</option>
            <option value="false">Disabled (CPU)</option>
          </select>
        </div>

        <div class="space-y-2">
          <div class="flex items-center justify-between">
            <label class="text-[#94a3b8] text-xs font-medium">VLM Workers</label>
            <span class="mono text-[#f59e0b] text-sm">{{ form.vlmWorkers }}</span>
          </div>
          <input
            v-model.number="form.vlmWorkers"
            type="range" min="1" max="8" step="1"
            class="w-full accent-[#2dd4bf] cursor-pointer"
          />
          <div class="text-[#475569] text-xs">
            N=4 recommended to saturate vLLM continuous batching
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
