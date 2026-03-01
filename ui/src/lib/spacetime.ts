/**
 * SpacetimeDB connection singleton placeholder.
 * Will be wired to @clockworklabs/spacetimedb-sdk in x05.3.
 */

import { useEventsStore } from '@/stores/events'
import { useAuthStore } from '@/stores/auth'
import type { AgentEvent, KeyframeEntry } from '@/stores/events'

let connectionInstance: SpacetimeConnection | null = null

export class SpacetimeConnection {
  private url: string
  private connected = false

  constructor(url: string) {
    this.url = url
  }

  async connect(token?: string): Promise<void> {
    const eventsStore = useEventsStore()
    const authStore = useAuthStore()

    try {
      // Placeholder: real SDK connection wired in x05.3
      console.info('[SpacetimeDB] Connecting to', this.url)
      this.connected = true
      eventsStore.setConnectionStatus(true)

      if (token) {
        authStore.setSpacetimeToken(token)
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Connection failed'
      eventsStore.setConnectionStatus(false, msg)
      throw err
    }
  }

  disconnect(): void {
    this.connected = false
    const eventsStore = useEventsStore()
    eventsStore.setConnectionStatus(false)
    console.info('[SpacetimeDB] Disconnected')
  }

  isConnected(): boolean {
    return this.connected
  }

  // Called by useEventStream composable when SDK fires onInsert
  handleEventInsert(event: AgentEvent): void {
    const store = useEventsStore()
    store.addEvent(event)
  }

  handleKeyframeInsert(kf: KeyframeEntry): void {
    const store = useEventsStore()
    store.addKeyframe(kf)
  }
}

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
