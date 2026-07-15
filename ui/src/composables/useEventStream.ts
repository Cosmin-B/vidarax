import { ref } from 'vue'
import { api, type RawRunEvent } from '@/lib/api'
import { useEventsStore, type AgentEvent, type KeyframeEntry } from '@/stores/events'
import { logger } from '@/lib/logger'

const POLL_INTERVAL_MS = 1000

function mapEvent(raw: RawRunEvent, runId: string): AgentEvent | null {
  if (raw.kind === 'keyframe_stored') return null
  const payload = raw.payload
  const workerKinds = new Set([
    'vlm',
    'vlm_tiered',
    'clip_vlm',
    'clip_vlm_tiered',
    'state_transition',
    'loop_detected',
  ])
  if (raw.kind !== 'marker_emitted' && !workerKinds.has(raw.kind)) return null

  const workerDescription = workerKinds.has(raw.kind)
  const eventType = raw.kind === 'marker_emitted'
    ? String(payload.event_type ?? 'context_observation')
    : raw.kind === 'loop_detected'
      ? 'loop_detected'
      : 'vlm_description'

  return {
    run_id: runId,
    session_id: String(payload.session_id ?? payload.stream_id ?? ''),
    frame_index: Number(payload.frame_index ?? payload.start_frame ?? 0),
    pts_ms: Number(payload.pts_ms ?? payload.start_pts_ms ?? 0),
    event_type: eventType as AgentEvent['event_type'],
    confidence: Number(payload.confidence ?? 0),
    description: workerDescription
      ? String(payload.description ?? '')
      : String(payload.description ?? payload.summary ?? ''),
    timestamp_ms: raw.pts_ms,
  }
}

async function loadKeyframe(raw: RawRunEvent, runId: string): Promise<KeyframeEntry | null> {
  if (raw.kind !== 'keyframe_stored') return null
  const sha256 = raw.payload.image_sha256
  if (typeof sha256 !== 'string' || !/^[0-9a-fA-F]{64}$/.test(sha256)) return null

  const blob = await api.runs.keyframe(runId, sha256)
  return {
    id: raw.seq,
    run_id: runId,
    frame_index: Number(raw.payload.frame_index ?? 0),
    pts_ms: Number(raw.payload.pts_ms ?? 0),
    event_type: String(raw.payload.event_type ?? ''),
    description: String(raw.payload.description ?? ''),
    image_sha256: sha256.toLowerCase(),
    image_url: URL.createObjectURL(blob),
    timestamp_ms: raw.pts_ms,
  }
}

export function useEventStream() {
  const eventsStore = useEventsStore()
  const isConnected = ref(false)
  const connectionError = ref<string | null>(null)

  let activeRunId: string | null = null
  let pollTimer: ReturnType<typeof setInterval> | null = null
  let pollInFlight = false
  const seenSequences = new Set<number>()
  const loadedKeyframes = new Set<string>()

  async function poll(): Promise<void> {
    if (!activeRunId || pollInFlight) return
    pollInFlight = true
    try {
      const response = await api.runs.events(activeRunId)
      for (const raw of response.events) {
        if (seenSequences.has(raw.seq)) continue
        const event = mapEvent(raw, activeRunId)
        if (event) eventsStore.addEvent(event)

        if (raw.kind === 'keyframe_stored') {
          const sha = raw.payload.image_sha256
          if (typeof sha === 'string' && !loadedKeyframes.has(sha)) {
            const keyframe = await loadKeyframe(raw, activeRunId)
            if (keyframe) {
              eventsStore.addKeyframe(keyframe)
              loadedKeyframes.add(keyframe.image_sha256)
            }
          }
        }
        seenSequences.add(raw.seq)
      }
      isConnected.value = true
      connectionError.value = null
      eventsStore.setConnectionStatus(true)
    } catch (error) {
      isConnected.value = false
      connectionError.value = error instanceof Error ? error.message : 'Timeline polling failed'
      eventsStore.setConnectionStatus(false, connectionError.value)
      logger.warn('[Timeline] Poll failed:', connectionError.value)
    } finally {
      pollInFlight = false
    }
  }

  async function connect(runId: string): Promise<void> {
    if (!/^[a-zA-Z0-9_-]+$/.test(runId)) {
      connectionError.value = 'Invalid run ID format'
      return
    }
    disconnect()
    activeRunId = runId
    seenSequences.clear()
    loadedKeyframes.clear()
    eventsStore.setActiveRunId(runId)
    await poll()
    pollTimer = setInterval(poll, POLL_INTERVAL_MS)
  }

  function disconnect(): void {
    if (pollTimer) clearInterval(pollTimer)
    pollTimer = null
    activeRunId = null
    isConnected.value = false
  }

  return {
    isConnected,
    connectionError,
    connect,
    disconnect,
  }
}
