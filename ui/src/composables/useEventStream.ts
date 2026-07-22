import { ref } from 'vue'
import { api, followRunEvents, type RawRunEvent } from '@/lib/api'
import { useEventsStore, type AgentEvent, type KeyframeEntry } from '@/stores/events'
import { logger } from '@/lib/logger'

const INITIAL_RECONNECT_MS = 250
const MAX_RECONNECT_MS = 5000

function objectField(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' ? value as Record<string, unknown> : {}
}

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
  const isRestrictedZone = raw.kind === 'restricted_zone_activity_entered'
  if (raw.kind !== 'marker_emitted' && !workerKinds.has(raw.kind) && !isRestrictedZone) return null

  const assertion = objectField(payload.assertion)
  const trigger = objectField(assertion.trigger)
  const workerDescription = workerKinds.has(raw.kind)
  const eventType = isRestrictedZone
    ? 'restricted_zone_activity'
    : raw.kind === 'marker_emitted'
      ? String(payload.event_type ?? 'context_observation')
      : raw.kind === 'loop_detected'
        ? 'loop_detected'
        : 'vlm_description'
  const zoneDescription = `Activity entered restricted zone ${String(assertion.policy_id ?? '')}`.trim()

  return {
    run_id: runId,
    session_id: String(payload.session_id ?? payload.stream_id ?? ''),
    frame_index: Number(payload.frame_index ?? payload.start_frame ?? 0),
    pts_ms: Number(payload.pts_ms ?? payload.start_pts_ms ?? 0),
    event_type: eventType as AgentEvent['event_type'],
    confidence: Number(payload.confidence ?? trigger.score ?? 0),
    description: isRestrictedZone
      ? zoneDescription
      : workerDescription
        ? String(payload.description ?? '')
        : String(payload.description ?? payload.summary ?? ''),
    timestamp_ms: raw.pts_ms,
  }
}

function keyframeEvidence(raw: RawRunEvent): Record<string, unknown> | null {
  if (raw.kind === 'keyframe_stored') return raw.payload
  if (raw.kind === 'restricted_zone_activity_entered') return objectField(raw.payload.evidence)
  return null
}

async function loadKeyframe(raw: RawRunEvent, runId: string): Promise<KeyframeEntry | null> {
  const evidence = keyframeEvidence(raw)
  if (!evidence) return null
  const sha256 = evidence.image_sha256
  if (typeof sha256 !== 'string' || !/^[0-9a-fA-F]{64}$/.test(sha256)) return null

  const blob = await api.runs.keyframe(runId, sha256)
  const assertion = objectField(raw.payload.assertion)
  return {
    id: raw.seq,
    run_id: runId,
    frame_index: Number(raw.payload.frame_index ?? 0),
    pts_ms: Number(raw.payload.pts_ms ?? 0),
    event_type: String(raw.payload.event_type ?? raw.kind),
    description: String(
      raw.payload.description
      ?? (raw.kind === 'restricted_zone_activity_entered'
        ? `Activity entered restricted zone ${String(assertion.policy_id ?? '')}`.trim()
        : ''),
    ),
    image_sha256: sha256.toLowerCase(),
    image_url: URL.createObjectURL(blob),
    timestamp_ms: raw.pts_ms,
  }
}

function reconnectDelay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise(resolve => {
    const timer = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => {
      clearTimeout(timer)
      resolve()
    }, { once: true })
  })
}

export function useEventStream() {
  const eventsStore = useEventsStore()
  const isConnected = ref(false)
  const connectionError = ref<string | null>(null)

  let activeRunId: string | null = null
  let streamAbort: AbortController | null = null
  let streamGeneration = 0
  let lastSequence = 0
  const loadedKeyframes = new Set<string>()

  function processEvent(raw: RawRunEvent, runId: string): void {
    if (raw.seq <= lastSequence) return
    const event = mapEvent(raw, runId)
    if (event) eventsStore.addEvent(event)
    lastSequence = raw.seq

    // Blob loading is independent of cursor progress. A slow or missing image
    // must not stall the SSE reader and cause a server-side subscriber queue
    // to fill; the assertion itself is already visible from the WAL event.
    const evidence = keyframeEvidence(raw)
    const sha = evidence?.image_sha256
    if (typeof sha !== 'string' || !/^[0-9a-fA-F]{64}$/.test(sha)) return
    const normalizedSha = sha.toLowerCase()
    if (loadedKeyframes.has(normalizedSha)) return
    loadedKeyframes.add(normalizedSha)

    void loadKeyframe(raw, runId)
      .then(keyframe => {
        if (keyframe && activeRunId === runId) eventsStore.addKeyframe(keyframe)
      })
      .catch(error => {
        loadedKeyframes.delete(normalizedSha)
        logger.warn('[Timeline] Evidence load failed:', error)
      })
  }

  async function follow(runId: string, generation: number, signal: AbortSignal): Promise<void> {
    let reconnectMs = INITIAL_RECONNECT_MS
    while (!signal.aborted && activeRunId === runId && streamGeneration === generation) {
      try {
        await followRunEvents(
          runId,
          lastSequence,
          signal,
          () => {
            isConnected.value = true
            connectionError.value = null
            eventsStore.setConnectionStatus(true)
            reconnectMs = INITIAL_RECONNECT_MS
          },
          raw => processEvent(raw, runId),
        )
        if (signal.aborted) return
        throw new Error('Timeline stream closed')
      } catch (error) {
        if (signal.aborted) return
        isConnected.value = false
        connectionError.value = error instanceof Error ? error.message : 'Timeline stream failed'
        eventsStore.setConnectionStatus(false, connectionError.value)
        logger.warn('[Timeline] Stream reconnect:', connectionError.value)
      }
      await reconnectDelay(reconnectMs, signal)
      reconnectMs = Math.min(reconnectMs * 2, MAX_RECONNECT_MS)
    }
  }

  async function connect(runId: string): Promise<void> {
    if (!/^[a-zA-Z0-9_-]+$/.test(runId)) {
      connectionError.value = 'Invalid run ID format'
      return
    }
    disconnect()
    activeRunId = runId
    lastSequence = 0
    loadedKeyframes.clear()
    eventsStore.setActiveRunId(runId)
    streamAbort = new AbortController()
    const generation = ++streamGeneration
    void follow(runId, generation, streamAbort.signal)
  }

  function disconnect(): void {
    streamGeneration++
    streamAbort?.abort()
    streamAbort = null
    activeRunId = null
    isConnected.value = false
    eventsStore.setConnectionStatus(false)
  }

  return {
    isConnected,
    connectionError,
    connect,
    disconnect,
  }
}
