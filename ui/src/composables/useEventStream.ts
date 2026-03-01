/**
 * useEventStream — SpacetimeDB real-time subscription composable.
 *
 * Connects to the SpacetimeDB module via WebSocket (v1.json.spacetimedb protocol)
 * and subscribes to AgentEvents + KeyframeStore for the active run.
 *
 * No generated bindings required — rows are mapped directly from the JSON
 * representation produced by the SATS type system (snake_case field names).
 *
 * Usage:
 *   const { connect, disconnect, isConnected, connectionError } = useEventStream()
 *   await connect(runId)
 */

import { ref, onUnmounted } from 'vue'
import { useEventsStore } from '@/stores/events'
import { useAuthStore } from '@/stores/auth'
import { logger } from '@/lib/logger'
import type { AgentEvent, KeyframeEntry } from '@/stores/events'

// ── SpacetimeDB v1 JSON protocol types ────────────────────────────────────────

interface StdbTableUpdate {
  table_id?: number
  table_name: string
  updates: {
    deletes: Array<{ row: unknown }>
    inserts: Array<{ row: unknown }>
  }
}

interface StdbDatabaseUpdate {
  tables: StdbTableUpdate[]
}

type ServerMessage =
  | { IdentityToken: { identity: string; token: string; address: string } }
  | {
      InitialSubscription: {
        database_update: StdbDatabaseUpdate
        request_id: number
        total_host_execution_duration_micros: number
      }
    }
  | {
      TransactionUpdate: {
        status: string | { Committed: StdbDatabaseUpdate }
        timestamp: { microseconds: number }
        caller_identity: string
        caller_address: string
        reducer_call?: unknown
        database_update?: StdbDatabaseUpdate
      }
    }
  | { SubscribeError: { request_id: number; message: string } }
  | { ErrorMessage: { message: string } }

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Build the SpacetimeDB WebSocket URL. */
function buildWsUrl(baseUrl: string, moduleName: string): string {
  // Normalise trailing slash + http → ws scheme
  const url = new URL(baseUrl.replace(/\/$/, ''))
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  // SpacetimeDB v1 subscribe endpoint
  url.pathname = `/v1/database/${moduleName}/subscribe`
  return url.toString()
}

/** Extract database_update from TransactionUpdate (handles both shapes). */
function extractDbUpdate(msg: ServerMessage & { TransactionUpdate: unknown }): StdbDatabaseUpdate | null {
  const tu = (msg as { TransactionUpdate: { status: unknown; database_update?: StdbDatabaseUpdate } })
    .TransactionUpdate
  // Shape A: database_update is a top-level field
  if (tu.database_update) return tu.database_update
  // Shape B: status.Committed holds the tables
  const committed = (tu.status as { Committed?: StdbDatabaseUpdate })?.Committed
  if (committed) return committed
  return null
}

/** Safely parse a raw row object into AgentEvent. */
function parseAgentEvent(row: unknown): AgentEvent | null {
  if (!row || typeof row !== 'object') return null
  const r = row as Record<string, unknown>
  if (typeof r.run_id !== 'string') return null
  return {
    run_id: r.run_id as string,
    session_id: (r.session_id as string) ?? '',
    frame_index: Number(r.frame_index ?? 0),
    pts_ms: Number(r.pts_ms ?? 0),
    event_type: (r.event_type as AgentEvent['event_type']) ?? 'vlm_description',
    confidence: Number(r.confidence ?? 0),
    description: (r.description as string) ?? '',
    timestamp_ms: Number(r.timestamp_ms ?? 0),
  }
}

/** Safely parse a raw row object into KeyframeEntry. */
function parseKeyframeEntry(row: unknown): KeyframeEntry | null {
  if (!row || typeof row !== 'object') return null
  const r = row as Record<string, unknown>
  if (typeof r.run_id !== 'string') return null
  return {
    id: Number(r.id ?? 0),
    run_id: r.run_id as string,
    frame_index: Number(r.frame_index ?? 0),
    pts_ms: Number(r.pts_ms ?? 0),
    event_type: (r.event_type as string) ?? '',
    description: (r.description as string) ?? '',
    jpeg_b64: (r.jpeg_b64 as string) ?? '',
    timestamp_ms: Number(r.timestamp_ms ?? 0),
  }
}

// ── Module name used in the SpacetimeDB module ────────────────────────────────
const MODULE_NAME = 'vidarax'

// ── Composable ────────────────────────────────────────────────────────────────

export function useEventStream() {
  const eventsStore = useEventsStore()
  const authStore = useAuthStore()

  const isConnected = ref(false)
  const connectionError = ref<string | null>(null)

  let ws: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let activeRunId: string | null = null
  let reqId = 1

  function applyTableInserts(dbUpdate: StdbDatabaseUpdate, runIdFilter?: string): void {
    for (const table of dbUpdate.tables) {
      for (const { row } of table.updates.inserts) {
        if (table.table_name === 'AgentEvents' || table.table_name === 'agent_events') {
          const event = parseAgentEvent(row)
          if (event && (!runIdFilter || event.run_id === runIdFilter)) {
            eventsStore.addEvent(event)
          }
        } else if (table.table_name === 'KeyframeStore' || table.table_name === 'keyframe_store') {
          const kf = parseKeyframeEntry(row)
          if (kf && (!runIdFilter || kf.run_id === runIdFilter)) {
            eventsStore.addKeyframe(kf)
          }
        }
      }
    }
  }

  function handleMessage(raw: string): void {
    let msg: ServerMessage
    try {
      msg = JSON.parse(raw) as ServerMessage
    } catch {
      logger.warn('[SpacetimeDB] Failed to parse message:', raw.slice(0, 200))
      return
    }

    if ('IdentityToken' in msg) {
      const { token } = msg.IdentityToken
      authStore.setSpacetimeToken(token)
      // Send subscription query once authenticated
      sendSubscribe()
      return
    }

    if ('InitialSubscription' in msg) {
      const { database_update } = msg.InitialSubscription
      applyTableInserts(database_update, activeRunId ?? undefined)
      return
    }

    if ('TransactionUpdate' in msg) {
      // Cast to avoid TS narrowing issues with discriminated union here
      const dbUpdate = extractDbUpdate(msg as Parameters<typeof extractDbUpdate>[0])
      if (dbUpdate) applyTableInserts(dbUpdate, activeRunId ?? undefined)
      return
    }

    if ('SubscribeError' in msg) {
      console.error('[SpacetimeDB] SubscribeError:', msg.SubscribeError.message)
      connectionError.value = msg.SubscribeError.message
      return
    }

    if ('ErrorMessage' in msg) {
      console.error('[SpacetimeDB] ErrorMessage:', msg.ErrorMessage.message)
      connectionError.value = msg.ErrorMessage.message
    }
  }

  function sendSubscribe(): void {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const runFilter = activeRunId ? ` WHERE run_id = '${activeRunId}'` : ''
    const payload = {
      Subscribe: {
        query_strings: [
          `SELECT * FROM AgentEvents${runFilter}`,
          `SELECT * FROM KeyframeStore${runFilter}`,
        ],
        request_id: reqId++,
      },
    }
    ws.send(JSON.stringify(payload))
  }

  async function connect(runId: string): Promise<void> {
    // C-1: Validate runId to prevent SQL injection in SpacetimeDB subscription queries.
    if (!/^[a-zA-Z0-9_-]+$/.test(runId)) {
      connectionError.value = 'Invalid run ID format'
      return
    }
    activeRunId = runId
    eventsStore.setActiveRunId(runId)

    if (ws && ws.readyState === WebSocket.OPEN) {
      // Already connected — re-subscribe for the new runId
      sendSubscribe()
      return
    }

    _openWebSocket()
  }

  function _openWebSocket(): void {
    if (ws) {
      ws.onclose = null
      ws.onerror = null
      ws.close()
      ws = null
    }

    const wsUrl = buildWsUrl(authStore.spacetimeEndpoint, MODULE_NAME)
    const protocols: string[] = ['v1.json.spacetimedb']

    try {
      ws = new WebSocket(wsUrl, protocols)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'WebSocket open failed'
      connectionError.value = msg
      eventsStore.setConnectionStatus(false, msg)
      return
    }

    ws.onopen = () => {
      isConnected.value = true
      connectionError.value = null
      eventsStore.setConnectionStatus(true)
      logger.info('[SpacetimeDB] Connected to', wsUrl)

      // If the server doesn't send IdentityToken (no-auth mode), subscribe now
      // A real server will send IdentityToken first, which triggers sendSubscribe
      if (!authStore.spacetimeToken) {
        sendSubscribe()
      }
    }

    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        handleMessage(ev.data)
      }
      // Binary BSATN frames are ignored (should not arrive with json protocol)
    }

    ws.onerror = (ev) => {
      console.error('[SpacetimeDB] WebSocket error', ev)
    }

    ws.onclose = (ev) => {
      isConnected.value = false
      eventsStore.setConnectionStatus(false)
      logger.info(`[SpacetimeDB] Disconnected (code=${ev.code})`)

      // Auto-reconnect after 3 s if we still have an active run
      if (activeRunId) {
        reconnectTimer = setTimeout(() => {
          logger.info('[SpacetimeDB] Attempting reconnect…')
          _openWebSocket()
        }, 3000)
      }
    }
  }

  function disconnect(): void {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    activeRunId = null
    eventsStore.setActiveRunId(null)

    if (ws) {
      ws.onclose = null
      ws.close(1000, 'User disconnected')
      ws = null
    }

    isConnected.value = false
    eventsStore.setConnectionStatus(false)
  }

  onUnmounted(disconnect)

  return { isConnected, connectionError, connect, disconnect }
}
