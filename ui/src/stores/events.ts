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
export const MAX_KEYFRAMES = 256

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
  const eventIdentityIndex = markRaw(new Map<string, AgentEvent>())

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

  function eventIdentityMatches(a: AgentEvent, b: AgentEvent): boolean {
    return a.run_id === b.run_id
      && a.session_id === b.session_id
      && a.frame_index === b.frame_index
      && a.pts_ms === b.pts_ms
      && a.event_type === b.event_type
      && a.timestamp_ms === b.timestamp_ms
  }

  function eventIdentityKey(event: AgentEvent): string {
    // Keep this exact rather than delimiter-based; run and session ids are caller data.
    return JSON.stringify([
      event.run_id,
      event.session_id,
      event.frame_index,
      event.pts_ms,
      event.event_type,
      event.timestamp_ms,
    ])
  }

  function eventContentMatches(a: AgentEvent, b: AgentEvent): boolean {
    return eventIdentityMatches(a, b)
      && a.confidence === b.confidence
      && a.description === b.description
  }

  function keyframeIdentityMatches(a: KeyframeEntry, b: KeyframeEntry): boolean {
    if (a.run_id !== b.run_id) return false
    if (a.id !== 0 || b.id !== 0) return a.id === b.id
    return a.timestamp_ms === b.timestamp_ms
      && a.frame_index === b.frame_index
      && a.pts_ms === b.pts_ms
  }

  function keyframeContentMatches(a: KeyframeEntry, b: KeyframeEntry): boolean {
    return keyframeIdentityMatches(a, b)
      && a.event_type === b.event_type
      && a.description === b.description
      && a.jpeg_b64 === b.jpeg_b64
      && a.timestamp_ms === b.timestamp_ms
  }

  function removeMatching<T>(arr: T[], predicate: (item: T) => boolean): T | null {
    const idx = arr.findIndex(predicate)
    if (idx < 0) return null
    const [removed] = arr.splice(idx, 1)
    return removed ?? null
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
    const identityKey = eventIdentityKey(event)
    const existing = eventIdentityIndex.get(identityKey)
    if (existing) {
      if (eventContentMatches(existing, event)) return
      removeByIdentity(events.value, existing)
      const indexed = eventsByRunId.value.get(existing.run_id)
      if (indexed) {
        removeByIdentity(indexed, existing)
        if (indexed.length === 0) eventsByRunId.value.delete(existing.run_id)
      }
      removeByIdentity(latestEventsIndex.value, existing)
      eventIdentityIndex.delete(identityKey)
    }

    const stored = markRaw(event)
    eventIdentityIndex.set(identityKey, stored)
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
        eventIdentityIndex.delete(eventIdentityKey(evt))
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
    const existing = keyframes.value.find(keyframe => keyframeIdentityMatches(keyframe, kf))
    if (existing) {
      if (keyframeContentMatches(existing, kf)) return
      removeMatching(keyframes.value, keyframe => keyframeIdentityMatches(keyframe, kf))
      const indexed = keyframesByRunId.value.get(existing.run_id)
      if (indexed) {
        removeMatching(indexed, keyframe => keyframeIdentityMatches(keyframe, kf))
        if (indexed.length === 0) keyframesByRunId.value.delete(existing.run_id)
      }
    }

    const stored = markRaw(kf)
    binaryInsert(keyframes.value, stored, keyframeOrder)

    let runKeyframes = keyframesByRunId.value.get(stored.run_id)
    if (!runKeyframes) {
      runKeyframes = markRaw([])
      keyframesByRunId.value.set(stored.run_id, runKeyframes)
    }
    binaryInsert(runKeyframes, stored, keyframeOrder)

    if (keyframes.value.length > MAX_KEYFRAMES) {
      const dropped = keyframes.value.splice(0, keyframes.value.length - MAX_KEYFRAMES)
      for (const keyframe of dropped) {
        const indexed = keyframesByRunId.value.get(keyframe.run_id)
        if (indexed) {
          removeByIdentity(indexed, keyframe)
          if (indexed.length === 0) keyframesByRunId.value.delete(keyframe.run_id)
        }
      }
    }
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
    for (const event of events.value) {
      if (event.run_id === runId) eventIdentityIndex.delete(eventIdentityKey(event))
    }
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
    eventIdentityIndex.clear()
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
