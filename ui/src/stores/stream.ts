import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export type StreamSourceType = 'camera' | 'screen' | 'livecam' | 'file' | 'livekit' | 'rtsp'
export type SessionState = 'idle' | 'negotiating' | 'connected' | 'error' | 'stopped'

export interface WhipSession {
  sessionId: string
  runId: string
  locationUrl: string
  createdAt: string
}

export const useStreamStore = defineStore('stream', () => {
  const sessionState = ref<SessionState>('idle')
  const sourceType = ref<StreamSourceType | null>(null)
  const activeSession = ref<WhipSession | null>(null)
  const mediaStream = ref<MediaStream | null>(null)
  const runId = ref<string | null>(null)
  const error = ref<string | null>(null)

  // Runtime metrics
  const fps = ref(0)
  const frameCount = ref(0)
  const latencyMs = ref(0)
  const bytesTransferred = ref(0)

  const isActive = computed(
    () => sessionState.value === 'connected' || sessionState.value === 'negotiating'
  )
  const isIdle = computed(() => sessionState.value === 'idle')
  const isError = computed(() => sessionState.value === 'error')

  function setSource(type: StreamSourceType) {
    sourceType.value = type
  }

  function setMediaStream(stream: MediaStream | null) {
    mediaStream.value = stream
  }

  function setSession(session: WhipSession | null) {
    activeSession.value = session
    if (session) {
      runId.value = session.runId
    }
  }

  function setState(state: SessionState) {
    sessionState.value = state
  }

  function setError(msg: string | null) {
    error.value = msg
    if (msg) sessionState.value = 'error'
  }

  function updateMetrics(data: {
    fps?: number
    frameCount?: number
    latencyMs?: number
    bytesTransferred?: number
  }) {
    if (data.fps !== undefined) fps.value = data.fps
    if (data.frameCount !== undefined) frameCount.value = data.frameCount
    if (data.latencyMs !== undefined) latencyMs.value = data.latencyMs
    if (data.bytesTransferred !== undefined) bytesTransferred.value = data.bytesTransferred
  }

  function reset() {
    sessionState.value = 'idle'
    sourceType.value = null
    activeSession.value = null
    if (mediaStream.value) {
      mediaStream.value.getTracks().forEach(t => t.stop())
      mediaStream.value = null
    }
    runId.value = null
    error.value = null
    fps.value = 0
    frameCount.value = 0
    latencyMs.value = 0
    bytesTransferred.value = 0
  }

  return {
    sessionState,
    sourceType,
    activeSession,
    mediaStream,
    runId,
    error,
    fps,
    frameCount,
    latencyMs,
    bytesTransferred,
    isActive,
    isIdle,
    isError,
    setSource,
    setMediaStream,
    setSession,
    setState,
    setError,
    updateMetrics,
    reset,
  }
})
