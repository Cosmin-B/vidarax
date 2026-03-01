<script setup lang="ts">
import { ref, computed } from 'vue'

const isDragging = ref(false)
const selectedFile = ref<File | null>(null)
const uploadProgress = ref(0)
const uploadState = ref<'idle' | 'uploading' | 'analyzing' | 'done' | 'error'>('idle')
const error = ref<string | null>(null)

const acceptedTypes = ['video/mp4', 'video/webm', 'video/quicktime']
const acceptedExtensions = '.mp4,.webm,.mov'

const fileSize = computed(() => {
  if (!selectedFile.value) return ''
  const bytes = selectedFile.value.size
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
})

const isProcessing = computed(() =>
  uploadState.value === 'uploading' || uploadState.value === 'analyzing'
)

function onDragover(e: DragEvent) {
  e.preventDefault()
  isDragging.value = true
}

function onDragleave() {
  isDragging.value = false
}

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
  error.value = null
  if (!acceptedTypes.includes(file.type) && !file.name.match(/\.(mp4|webm|mov)$/i)) {
    error.value = 'Only .mp4, .webm, and .mov files are supported.'
    return
  }
  selectedFile.value = file
  uploadState.value = 'idle'
  uploadProgress.value = 0
}

async function startUpload() {
  if (!selectedFile.value) return
  uploadState.value = 'uploading'
  uploadProgress.value = 0
  error.value = null

  // Simulated progress for scaffold — real implementation in x05.5
  const interval = setInterval(() => {
    uploadProgress.value = Math.min(uploadProgress.value + Math.random() * 15, 95)
  }, 300)

  await new Promise(r => setTimeout(r, 2000))
  clearInterval(interval)
  uploadProgress.value = 100
  uploadState.value = 'analyzing'

  await new Promise(r => setTimeout(r, 1500))
  uploadState.value = 'done'
}

function reset() {
  selectedFile.value = null
  uploadState.value = 'idle'
  uploadProgress.value = 0
  error.value = null
}
</script>

<template>
  <div class="p-4 lg:p-6 space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-[#e2e8f0] font-semibold text-xl">Upload Video</h2>
      <p class="text-[#64748b] text-sm mt-1">Upload a video file for batch semantic analysis</p>
    </div>

    <!-- Done state -->
    <div v-if="uploadState === 'done'" class="card-skeuo p-8 text-center space-y-4">
      <div
        class="w-12 h-12 rounded-full flex items-center justify-center mx-auto"
        style="background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.2);"
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
          <path d="M5 12L9 16L19 6" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div>
        <div class="text-[#e2e8f0] font-semibold">Analysis Started</div>
        <div class="text-[#64748b] text-sm mt-1">Your video is being processed. Check the dashboard for results.</div>
      </div>
      <div class="flex gap-3 justify-center">
        <RouterLink
          to="/"
          class="px-4 py-2 rounded-[10px] text-sm font-medium text-[#08090d]"
          style="background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);"
        >
          View Dashboard
        </RouterLink>
        <button
          class="px-4 py-2 rounded-[10px] text-sm font-medium text-[#94a3b8]"
          style="background: rgba(255,255,255,0.04); border: 1px solid #1e2633;"
          @click="reset"
        >
          Upload Another
        </button>
      </div>
    </div>

    <!-- Upload form -->
    <template v-else>
      <!-- Dropzone -->
      <div
        class="relative rounded-[12px] transition-all duration-200 cursor-pointer"
        :class="isDragging ? 'border-[#2dd4bf]' : 'border-[#1e2633] hover:border-[rgba(45,212,191,0.3)]'"
        :style="isDragging
          ? 'background: rgba(45,212,191,0.04); border: 2px dashed #2dd4bf; box-shadow: 0 0 20px rgba(45,212,191,0.1);'
          : 'background: linear-gradient(145deg, #0f1117, #151a22); border: 2px dashed #1e2633;'"
        style="min-height: 200px;"
        role="button"
        tabindex="0"
        aria-label="Drop video file here or click to browse"
        @dragover="onDragover"
        @dragleave="onDragleave"
        @drop="onDrop"
        @click="($refs.fileInput as HTMLInputElement).click()"
        @keydown.enter="($refs.fileInput as HTMLInputElement).click()"
      >
        <input
          ref="fileInput"
          type="file"
          :accept="acceptedExtensions"
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
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                 :stroke="isDragging ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15V19C21 20.1 20.1 21 19 21H5C3.9 21 3 20.1 3 19V15"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>

          <div v-if="!selectedFile">
            <div class="text-[#e2e8f0] font-medium mb-1">
              Drop your video here
            </div>
            <div class="text-[#475569] text-sm mb-3">
              or click to browse files
            </div>
            <div class="text-[#475569] text-xs">
              Supports .mp4, .webm, .mov
            </div>
          </div>

          <div v-else class="space-y-1">
            <div class="text-[#2dd4bf] font-medium">{{ selectedFile.name }}</div>
            <div class="mono text-[#64748b] text-sm">{{ fileSize }}</div>
            <button
              class="text-xs text-[#475569] hover:text-[#64748b] mt-2 underline"
              @click.stop="reset"
            >
              Remove
            </button>
          </div>
        </div>
      </div>

      <!-- Error -->
      <div
        v-if="error"
        class="flex items-center gap-2 p-3 rounded-[8px]"
        style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="#ef4444" flex-shrink="0">
          <path d="M7 1C3.7 1 1 3.7 1 7s2.7 6 6 6 6-2.7 6-6-2.7-6-6-6zm-.5 3h1v4H6.5V4zm0 5h1v1h-1V9z"/>
        </svg>
        <span class="text-[#ef4444] text-sm">{{ error }}</span>
      </div>

      <!-- Progress -->
      <div v-if="isProcessing" class="card-skeuo p-5 space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-[#e2e8f0] text-sm font-medium">
            {{ uploadState === 'uploading' ? 'Uploading...' : 'Analyzing frames...' }}
          </span>
          <span class="mono text-[#f59e0b] text-sm">{{ Math.round(uploadProgress) }}%</span>
        </div>
        <div class="h-1.5 rounded-full overflow-hidden" style="background: #1e2633;">
          <div
            class="h-full rounded-full progress-bar-active transition-all duration-300"
            style="background: linear-gradient(90deg, #0d9488, #2dd4bf);"
            :style="{ width: `${uploadProgress}%` }"
          />
        </div>
        <p class="text-[#475569] text-xs">
          {{ uploadState === 'uploading'
            ? 'Transferring to server...'
            : 'Running semantic analysis. Events will appear in real-time.' }}
        </p>
      </div>

      <!-- Submit button -->
      <div v-if="selectedFile && !isProcessing" class="flex justify-end">
        <button
          class="flex items-center gap-2 px-6 py-2.5 rounded-[10px] text-sm font-semibold text-[#08090d] transition-all duration-200"
          style="background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
                 box-shadow: 0 0 20px rgba(45,212,191,0.2);"
          @click="startUpload"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M7 1L7 9M7 1L4 4M7 1L10 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          Start Analysis
        </button>
      </div>
    </template>
  </div>
</template>
