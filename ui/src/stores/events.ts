import { defineStore } from 'pinia'
import { computed, markRaw, ref, shallowRef, triggerRef } from 'vue'

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
  const events = shallowRef<AgentEvent[]>([])
  const keyframes = shallowRef<KeyframeEntry[]>([])
  const isConnected = ref(false)
  const connectionError = ref<string | null>(null)
  const activeRunId = ref<string | null>(null)
  const eventVersion = ref(0)
  const keyframeVersion = ref(0)
  const eventsByRunId = shallowRef<Map<string, AgentEvent[]>>(markRaw(new Map()))
  const keyframesByRunId = shallowRef<Map<string, KeyframeEntry[]>>(markRaw(new Map()))
  const latestEventsIndex = shallowRef<AgentEvent[]>([])

  const activeEvents = computed(() => {
    eventVersion.value
    return activeRunId.value ? (eventsByRunId.value.get(activeRunId.value) ?? []) : []
  })

  const activeKeyframes = computed(() => {
    keyframeVersion.value
    return activeRunId.value ? (keyframesByRunId.value.get(activeRunId.value) ?? []) : []
  })

  const latestEvents = computed(() => {
    eventVersion.value
    return latestEventsIndex.value
  })

  function eventOrder(a: AgentEvent, b: AgentEvent): number {
    return (a.timestamp_ms - b.timestamp_ms)
      || (a.pts_ms - b.pts_ms)
      || (a.frame_index - b.frame_index)
  }

  function latestOrder(a: AgentEvent, b: AgentEvent): number {
    return -eventOrder(a, b)
  }

  function keyframeOrder(a: KeyframeEntry, b: KeyframeEntry): number {
    return (a.timestamp_ms - b.timestamp_ms)
      || (a.pts_ms - b.pts_ms)
      || (a.frame_index - b.frame_index)
      || (a.id - b.id)
  }

  function binaryInsert<T>(arr: T[], item: T, compare: (a: T, b: T) => number): void {
    // SSE events arrive in order almost always, so the tail append is the
    // common case and stays O(1); only an out-of-order event pays the O(n) splice.
    if (arr.length === 0 || compare(arr[arr.length - 1]!, item) <= 0) {
      arr.push(item)
      return
    }
    let lo = 0
    let hi = arr.length
    while (lo < hi) {
      const mid = (lo + hi) >>> 1
      if (compare(arr[mid]!, item) <= 0) lo = mid + 1
      else hi = mid
    }
    arr.splice(lo, 0, item)
  }

  function removeByIdentity<T>(arr: T[], item: T): void {
    const idx = arr.indexOf(item)
    if (idx >= 0) arr.splice(idx, 1)
  }

  function touchEvents(): void {
    eventVersion.value++
    triggerRef(events)
    triggerRef(eventsByRunId)
    triggerRef(latestEventsIndex)
  }

  function touchKeyframes(): void {
    keyframeVersion.value++
    triggerRef(keyframes)
    triggerRef(keyframesByRunId)
  }

  function addEvent(event: AgentEvent) {
    const stored = markRaw(event)
    binaryInsert(events.value, stored, eventOrder)

    let runEvents = eventsByRunId.value.get(stored.run_id)
    if (!runEvents) {
      runEvents = markRaw([])
      eventsByRunId.value.set(stored.run_id, runEvents)
    }
    binaryInsert(runEvents, stored, eventOrder)
    binaryInsert(latestEventsIndex.value, stored, latestOrder)
    if (latestEventsIndex.value.length > 20) latestEventsIndex.value.length = 20

    if (events.value.length > MAX_EVENTS) {
      const dropped = events.value.splice(0, events.value.length - MAX_EVENTS)
      for (const evt of dropped) {
        const indexed = eventsByRunId.value.get(evt.run_id)
        if (indexed) {
          removeByIdentity(indexed, evt)
          if (indexed.length === 0) eventsByRunId.value.delete(evt.run_id)
        }
        removeByIdentity(latestEventsIndex.value, evt)
      }
    }
    touchEvents()
  }

  function addKeyframe(kf: KeyframeEntry) {
    const stored = markRaw(kf)
    binaryInsert(keyframes.value, stored, keyframeOrder)

    let runKeyframes = keyframesByRunId.value.get(stored.run_id)
    if (!runKeyframes) {
      runKeyframes = markRaw([])
      keyframesByRunId.value.set(stored.run_id, runKeyframes)
    }
    binaryInsert(runKeyframes, stored, keyframeOrder)
    touchKeyframes()
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
    latestEventsIndex.value = latestEventsIndex.value.filter(e => e.run_id !== runId)
    eventsByRunId.value.delete(runId)
    keyframes.value = keyframes.value.filter(k => k.run_id !== runId)
    keyframesByRunId.value.delete(runId)
    touchEvents()
    touchKeyframes()
  }

  function clearAll() {
    events.value = []
    keyframes.value = []
    eventsByRunId.value = markRaw(new Map())
    keyframesByRunId.value = markRaw(new Map())
    latestEventsIndex.value = []
    touchEvents()
    touchKeyframes()
  }

  function eventsForRun(runId: string): AgentEvent[] {
    eventVersion.value
    return eventsByRunId.value.get(runId) ?? []
  }

  function keyframesForRun(runId: string): KeyframeEntry[] {
    keyframeVersion.value
    return keyframesByRunId.value.get(runId) ?? []
  }

  return {
    events,
    keyframes,
    isConnected,
    connectionError,
    activeRunId,
    eventVersion,
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
