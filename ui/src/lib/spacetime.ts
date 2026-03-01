/**
 * SpacetimeDB connection singleton.
 *
 * The useEventStream composable (src/composables/useEventStream.ts) owns the
 * reactive WebSocket lifecycle for component trees.  This singleton acts as a
 * lighter convenience layer used by one-shot callers (e.g. app boot, auth flow)
 * that don't live inside a Vue component.
 *
 * When the full @clockworklabs/spacetimedb-sdk generated bindings are ready
 * (after `spacetime generate` runs against the deployed Vidarax module), replace
 * the WebSocket block inside SpacetimeConnection.connect() with the SDK builder.
 */

import { useEventsStore } from '@/stores/events'
import { useAuthStore } from '@/stores/auth'
import { logger } from '@/lib/logger'
import type { AgentEvent, KeyframeEntry } from '@/stores/events'

const MODULE_NAME = 'vidarax'

function toWsUrl(baseUrl: string): string {
  const url = new URL(baseUrl.replace(/\/$/, ''))
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  url.pathname = `/v1/database/${MODULE_NAME}/subscribe`
  return url.toString()
}

export class SpacetimeConnection {
  private wsUrl: string
  private ws: WebSocket | null = null
  private _connected = false

  constructor(baseUrl: string) {
    this.wsUrl = toWsUrl(baseUrl)
  }

  async connect(token?: string): Promise<void> {
    const eventsStore = useEventsStore()
    const authStore = useAuthStore()

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.wsUrl, ['v1.json.spacetimedb'])
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'WS open failed'
        eventsStore.setConnectionStatus(false, msg)
        reject(new Error(msg))
        return
      }

      this.ws.onopen = () => {
        this._connected = true
        eventsStore.setConnectionStatus(true)
        if (token) authStore.setSpacetimeToken(token)
        logger.info('[SpacetimeDB] Connected to', this.wsUrl)
        resolve()
      }

      this.ws.onerror = () => {
        const msg = 'WebSocket connection error'
        this._connected = false
        eventsStore.setConnectionStatus(false, msg)
        reject(new Error(msg))
      }

      this.ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return
        try {
          const msg = JSON.parse(ev.data) as Record<string, unknown>
          if ('TransactionUpdate' in msg || 'InitialSubscription' in msg) {
            this._dispatchTableRows(msg)
          }
        } catch { /* ignore parse errors */ }
      }

      this.ws.onclose = () => {
        this._connected = false
        eventsStore.setConnectionStatus(false)
      }
    })
  }

  disconnect(): void {
    this._connected = false
    const eventsStore = useEventsStore()
    eventsStore.setConnectionStatus(false)
    if (this.ws) {
      this.ws.onclose = null
      this.ws.close(1000, 'Singleton disconnect')
      this.ws = null
    }
    logger.info('[SpacetimeDB] Disconnected')
  }

  isConnected(): boolean {
    return this._connected && this.ws?.readyState === WebSocket.OPEN
  }

  /** Direct injection for SDK callbacks (used by useEventStream). */
  handleEventInsert(event: AgentEvent): void {
    useEventsStore().addEvent(event)
  }

  handleKeyframeInsert(kf: KeyframeEntry): void {
    useEventsStore().addKeyframe(kf)
  }

  private _dispatchTableRows(msg: Record<string, unknown>): void {
    const eventsStore = useEventsStore()
    const tables: Array<{ table_name: string; updates: { inserts: Array<{ row: unknown }> } }> = []

    // Extract from InitialSubscription
    const initial = msg['InitialSubscription'] as
      | { database_update?: { tables: typeof tables } }
      | undefined
    if (initial?.database_update?.tables) {
      tables.push(...initial.database_update.tables)
    }

    // Extract from TransactionUpdate (both status shapes)
    const tu = msg['TransactionUpdate'] as
      | { database_update?: { tables: typeof tables }; status?: { Committed?: { tables: typeof tables } } }
      | undefined
    if (tu?.database_update?.tables) tables.push(...tu.database_update.tables)
    if (tu?.status && typeof tu.status === 'object') {
      const committed = (tu.status as { Committed?: { tables: typeof tables } }).Committed
      if (committed?.tables) tables.push(...committed.tables)
    }

    for (const table of tables) {
      if (!table.updates?.inserts) continue
      for (const { row } of table.updates.inserts) {
        if (!row || typeof row !== 'object') continue
        const r = row as Record<string, unknown>
        if (table.table_name === 'AgentEvents' || table.table_name === 'agent_events') {
          if (typeof r.run_id === 'string') {
            eventsStore.addEvent(r as unknown as AgentEvent)
          }
        } else if (table.table_name === 'KeyframeStore' || table.table_name === 'keyframe_store') {
          if (typeof r.run_id === 'string') {
            eventsStore.addKeyframe(r as unknown as KeyframeEntry)
          }
        }
      }
    }
  }
}

// ── Singleton management ──────────────────────────────────────────────────────

let connectionInstance: SpacetimeConnection | null = null

export function getSpacetimeConnection(url?: string): SpacetimeConnection {
  const authStore = useAuthStore()
  const targetUrl = url ?? authStore.spacetimeEndpoint

  if (!connectionInstance || !connectionInstance.isConnected()) {
    connectionInstance = new SpacetimeConnection(targetUrl)
  }
  return connectionInstance
}

export function resetSpacetimeConnection(): void {
  if (connectionInstance) {
    connectionInstance.disconnect()
    connectionInstance = null
  }
}
