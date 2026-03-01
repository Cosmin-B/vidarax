import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export type EventType =
  | 'scene_cut'
  | 'loop_detected'
  | 'vlm_description'
  | 'artifact_suspected'
  | 'exposure_shift'
  | 'flicker'
  | 'keyframe'

export interface AgentEvent {
  run_id: string
  session_id: string
  frame_index: number
  pts_ms: number
  event_type: EventType
  confidence: number
  description: string
  timestamp_ms: number
}

export interface KeyframeEntry {
  id: number
  run_id: string
  frame_index: number
  pts_ms: number
  event_type: string
  description: string
  jpeg_b64: string
  timestamp_ms: number
}

const MAX_EVENTS = 500

export const useEventsStore = defineStore('events', () => {
  const events = ref<AgentEvent[]>([])
  const keyframes = ref<KeyframeEntry[]>([])
  const isConnected = ref(false)
  const connectionError = ref<string | null>(null)
  const activeRunId = ref<string | null>(null)

  const eventsByRunId = computed(() => {
    const map = new Map<string, AgentEvent[]>()
    for (const evt of events.value) {
      if (!map.has(evt.run_id)) map.set(evt.run_id, [])
      map.get(evt.run_id)!.push(evt)
    }
    return map
  })

  const keyframesByRunId = computed(() => {
    const map = new Map<string, KeyframeEntry[]>()
    for (const kf of keyframes.value) {
      if (!map.has(kf.run_id)) map.set(kf.run_id, [])
      map.get(kf.run_id)!.push(kf)
    }
    return map
  })

  const activeEvents = computed(() =>
    activeRunId.value ? (eventsByRunId.value.get(activeRunId.value) ?? []) : []
  )

  const activeKeyframes = computed(() =>
    activeRunId.value ? (keyframesByRunId.value.get(activeRunId.value) ?? []) : []
  )

  const latestEvents = computed(() =>
    [...events.value].sort((a, b) => b.timestamp_ms - a.timestamp_ms).slice(0, 20)
  )

  function addEvent(event: AgentEvent) {
    events.value.push(event)
    // Rolling buffer — drop oldest when over limit
    if (events.value.length > MAX_EVENTS) {
      events.value = events.value.slice(events.value.length - MAX_EVENTS)
    }
  }

  function addKeyframe(kf: KeyframeEntry) {
    keyframes.value.push(kf)
  }

  function setConnectionStatus(connected: boolean, err?: string) {
    isConnected.value = connected
    connectionError.value = err ?? null
  }

  function setActiveRunId(runId: string | null) {
    activeRunId.value = runId
  }

  function clearEventsForRun(runId: string) {
    events.value = events.value.filter(e => e.run_id !== runId)
    keyframes.value = keyframes.value.filter(k => k.run_id !== runId)
  }

  function clearAll() {
    events.value = []
    keyframes.value = []
  }

  function eventsForRun(runId: string): AgentEvent[] {
    return eventsByRunId.value.get(runId) ?? []
  }

  function keyframesForRun(runId: string): KeyframeEntry[] {
    return keyframesByRunId.value.get(runId) ?? []
  }

  return {
    events,
    keyframes,
    isConnected,
    connectionError,
    activeRunId,
    eventsByRunId,
    keyframesByRunId,
    activeEvents,
    activeKeyframes,
    latestEvents,
    addEvent,
    addKeyframe,
    setConnectionStatus,
    setActiveRunId,
    clearEventsForRun,
    clearAll,
    eventsForRun,
    keyframesForRun,
  }
})
