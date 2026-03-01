<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useStreamStore } from '@/stores/stream'
import type { StreamSourceType } from '@/stores/stream'

const route = useRoute()
const router = useRouter()
const streamStore = useStreamStore()

interface SourceTile {
  id: StreamSourceType
  label: string
  description: string
  available: boolean
  icon: string
}

const sources: SourceTile[] = [
  { id: 'screen', label: 'Screen', description: 'Capture your entire screen or a window', available: true, icon: 'monitor' },
  { id: 'camera', label: 'Camera', description: 'Use your webcam or external camera', available: true, icon: 'camera' },
  { id: 'livecam', label: 'Live Cam', description: 'Connect to an IP camera via WHIP', available: true, icon: 'broadcast' },
  { id: 'file', label: 'File', description: 'Analyse a local video file', available: true, icon: 'file' },
  { id: 'livekit', label: 'LiveKit', description: 'Join a LiveKit room', available: false, icon: 'livekit' },
  { id: 'rtsp', label: 'RTSP', description: 'Connect to an RTSP stream', available: false, icon: 'rtsp' },
]

const selectedSource = ref<StreamSourceType | null>(null)

// sessionId used for session rejoin in x05.2
void route.params.sessionId
const isStreaming = computed(() => streamStore.isActive)

function selectSource(source: SourceTile) {
  if (!source.available) return
  if (source.id === 'file') {
    router.push('/upload')
    return
  }
  selectedSource.value = source.id
  streamStore.setSource(source.id)
}

function startStream() {
  if (!selectedSource.value) return
  // Actual WHIP negotiation wired in x05.2
  streamStore.setState('negotiating')
  setTimeout(() => {
    streamStore.setState('connected')
  }, 1200)
}

function stopStream() {
  streamStore.reset()
  selectedSource.value = null
}

// Stream state persists across navigation — reset is done only on explicit stop
</script>

<template>
  <div class="p-4 lg:p-6">
    <!-- Active stream view -->
    <div v-if="isStreaming" class="space-y-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <span class="live-dot" />
          <h2 class="text-[#e2e8f0] font-semibold text-lg">Live Stream</h2>
          <span class="badge badge-teal mono">
            {{ streamStore.fps.toFixed(0) }}fps
          </span>
        </div>
        <button
          class="px-4 py-2 rounded-[10px] text-sm font-medium text-[#ef4444] transition-all duration-200"
          style="background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);"
          @click="stopStream"
        >
          Stop Stream
        </button>
      </div>

      <!-- Stream viewer placeholder (wired in x05.2) -->
      <div
        class="relative rounded-[12px] overflow-hidden"
        style="background: #050507;
               border: 1px solid #1e2633;
               box-shadow: inset 0 2px 4px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.6);
               aspect-ratio: 16/9;"
      >
        <div class="absolute inset-0 flex items-center justify-center">
          <div class="text-center">
            <div class="w-12 h-12 rounded-full border-2 border-[#1e2633] border-t-[#2dd4bf] animate-spin mx-auto mb-3" />
            <p class="text-[#475569] text-sm">Stream viewer connects in x05.2</p>
          </div>
        </div>
        <!-- HUD overlay -->
        <div class="absolute top-3 left-3 flex items-center gap-2">
          <span
            class="px-2 py-1 rounded text-xs mono text-[#2dd4bf]"
            style="background: rgba(0,0,0,0.7); border: 1px solid rgba(45,212,191,0.2);"
          >
            LIVE
          </span>
          <span
            class="px-2 py-1 rounded text-xs mono text-[#f59e0b]"
            style="background: rgba(0,0,0,0.7);"
          >
            {{ streamStore.frameCount }} frames
          </span>
        </div>
      </div>
    </div>

    <!-- Source picker -->
    <div v-else class="space-y-6">
      <div>
        <h2 class="text-[#e2e8f0] font-semibold text-xl">Select Source</h2>
        <p class="text-[#64748b] text-sm mt-1">Choose a video source to begin analysis</p>
      </div>

      <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
        <button
          v-for="source in sources"
          :key="source.id"
          class="relative flex flex-col items-start gap-3 p-4 rounded-[12px] text-left transition-all duration-200"
          :class="[
            source.available ? 'cursor-pointer' : 'cursor-not-allowed opacity-50',
            selectedSource === source.id
              ? 'border-[#2dd4bf]'
              : 'border-[#1e2633] hover:border-[rgba(45,212,191,0.3)]',
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
          <span
            v-if="!source.available"
            class="absolute top-3 right-3 badge badge-muted"
          >
            Soon
          </span>

          <!-- Icon -->
          <div
            class="w-10 h-10 rounded-[10px] flex items-center justify-center"
            :style="selectedSource === source.id
              ? 'background: rgba(45,212,191,0.15);'
              : 'background: rgba(255,255,255,0.04); border: 1px solid #1e2633;'"
          >
            <!-- Monitor icon -->
            <svg v-if="source.icon === 'monitor'" width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <rect x="2" y="3" width="16" height="11" rx="2"/>
              <path d="M7 17H13M10 14V17"/>
            </svg>
            <!-- Camera icon -->
            <svg v-else-if="source.icon === 'camera'" width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <path d="M2 7C2 5.9 2.9 5 4 5H6L7.5 3H12.5L14 5H16C17.1 5 18 5.9 18 7V15C18 16.1 17.1 17 16 17H4C2.9 17 2 16.1 2 15V7Z"/>
              <circle cx="10" cy="11" r="3"/>
            </svg>
            <!-- Broadcast icon -->
            <svg v-else-if="source.icon === 'broadcast'" width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <circle cx="10" cy="10" r="2.5" stroke="none"
                      :fill="selectedSource === source.id ? '#2dd4bf' : '#64748b'"/>
              <path d="M6 6C4.3 7.7 4.3 12.3 6 14M14 6C15.7 7.7 15.7 12.3 14 14"/>
              <path d="M3.5 3.5C1 6 1 14 3.5 16.5M16.5 3.5C19 6 19 14 16.5 16.5" opacity="0.5"/>
            </svg>
            <!-- File icon -->
            <svg v-else-if="source.icon === 'file'" width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <path d="M4 3H12L16 7V17C16 17.6 15.6 18 15 18H5C4.4 18 4 17.6 4 17V3Z"/>
              <path d="M12 3V7H16"/>
              <path d="M7 11H13M7 14H11"/>
            </svg>
            <!-- Generic icon for livekit/rtsp -->
            <svg v-else width="20" height="20" viewBox="0 0 20 20" fill="none"
                 :stroke="selectedSource === source.id ? '#2dd4bf' : '#64748b'"
                 stroke-width="1.5" stroke-linecap="round">
              <circle cx="10" cy="10" r="7"/>
              <path d="M10 7V10L12 12"/>
            </svg>
          </div>

          <!-- Text -->
          <div>
            <div
              class="text-sm font-medium"
              :class="selectedSource === source.id ? 'text-[#2dd4bf]' : 'text-[#e2e8f0]'"
            >
              {{ source.label }}
            </div>
            <div class="text-xs text-[#475569] mt-0.5 leading-relaxed">
              {{ source.description }}
            </div>
          </div>
        </button>
      </div>

      <!-- Start button -->
      <div v-if="selectedSource" class="flex justify-end">
        <button
          class="flex items-center gap-2 px-6 py-2.5 rounded-[10px] text-sm font-semibold text-[#08090d] transition-all duration-200"
          style="background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
                 box-shadow: 0 0 20px rgba(45,212,191,0.3);"
          @click="startStream"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M3 2L11 7L3 12V2Z" fill="currentColor"/>
          </svg>
          Start Analysis
        </button>
      </div>
    </div>
  </div>
</template>
