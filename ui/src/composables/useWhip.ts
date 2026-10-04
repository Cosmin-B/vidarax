/**
 * useWhip — WHIP WebRTC streaming composable.
 *
 * Handles the full lifecycle:
 *   1. Acquire local media (camera / screen)
 *   2. Create RTCPeerConnection + offer SDP
 *   3. POST offer to /v1/stream/whip (WHIP protocol)
 *   4. Set remote answer, resolve ICE
 *   5. Poll getStats() for live metrics
 *   6. Terminate session on stop
 *
 * Usage:
 *   const { localStream, startStream, stopStream } = useWhip()
 */

import { ref, onUnmounted } from 'vue'
import { api } from '@/lib/api'
import { iceServersWithTurn, ls, STORAGE_KEYS, UI_DEFAULTS } from '@/lib/config'
import { useStreamStore } from '@/stores/stream'
import type { StreamSourceType } from '@/stores/stream'
import { stopDurableWhipSession, type DurableWhipSession } from '@/lib/whipLifecycle'

/** Wait for ICE gathering, component cancellation, or a 3 s timeout. */
function waitForIceComplete(pc: RTCPeerConnection, signal: AbortSignal): Promise<string> {
  if (signal.aborted) return Promise.resolve('')
  if (pc.iceGatheringState === 'complete') return Promise.resolve(pc.localDescription?.sdp ?? '')
  return new Promise(resolve => {
    const finish = () => {
      clearTimeout(timer)
      pc.removeEventListener('icegatheringstatechange', check)
      signal.removeEventListener('abort', finish)
      resolve(signal.aborted ? '' : pc.localDescription?.sdp ?? '')
    }
    const check = () => {
      if (pc.iceGatheringState === 'complete') finish()
    }
    const timer = setTimeout(finish, 3000)
    pc.addEventListener('icegatheringstatechange', check)
    signal.addEventListener('abort', finish, { once: true })
  })
}

interface WhipConnection {
  stream: MediaStream | null
  peer: RTCPeerConnection | null
  session: DurableWhipSession | null
  timer: ReturnType<typeof setInterval> | null
  abort: AbortController
  offerPending: boolean
  polling: boolean
  prevFrames: number
  prevTs: number
}

export interface WhipStartOptions {
  /** Optional analysis prompt sent with the WHIP offer attach config. */
  prompt?: string
  /** Analyze inbound audio with the local sidecar. */
  localAudio?: boolean
}

export function useWhip() {
  const streamStore = useStreamStore()

  const localStream = ref<MediaStream | null>(null)
  let active: WhipConnection | null = null

  /** Acquire local media based on source type. */
  async function acquireMedia(sourceType: StreamSourceType): Promise<MediaStream> {
    if (sourceType === 'screen') {
      return navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30 },
        audio: true,
      })
    }
    if (sourceType === 'camera') {
      return navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: true,
      })
    }
    throw new Error(`Source type "${sourceType}" is not supported by useWhip`)
  }

  async function startStream(sourceType: StreamSourceType, options: WhipStartOptions = {}): Promise<void> {
    const previous = active
    const connection: WhipConnection = {
      stream: null,
      peer: null,
      session: null,
      timer: null,
      abort: new AbortController(),
      offerPending: false,
      polling: false,
      prevFrames: 0,
      prevTs: 0,
    }
    active = connection
    localStream.value = null
    if (previous) void stopConnection(previous)
    streamStore.setMediaStream(null)
    streamStore.reset()
    streamStore.setSource(sourceType)
    streamStore.setState('negotiating')
    let connected = false

    try {
      connection.stream = await acquireMedia(sourceType)
      if (active !== connection) return
      localStream.value = connection.stream
      streamStore.setMediaStream(connection.stream)

      const turnUrl = ls(STORAGE_KEYS.turnUrl, UI_DEFAULTS.turnUrl)
      const peer = new RTCPeerConnection({ iceServers: iceServersWithTurn(turnUrl) })
      connection.peer = peer
      connection.stream.getTracks().forEach(track => peer.addTrack(track, connection.stream!))

      const offer = await peer.createOffer()
      if (active !== connection) return
      await peer.setLocalDescription(offer)
      if (active !== connection) return
      const offerSdp = await waitForIceComplete(peer, connection.abort.signal)
      if (active !== connection) return
      if (!offerSdp) throw new Error('Failed to build local SDP')

      const prompt = options.prompt?.trim()
      const attachConfig = {
        ...(prompt ? { prompt } : {}),
        ...(options.localAudio ? {
          local_audio: {
            profile: sourceType === 'screen' ? 'screen_recording' as const : 'physical_world' as const,
            speech_engine: 'whisper' as const,
            min_confidence: 0.35,
            max_events: 32,
          },
        } : {}),
      }
      connection.offerPending = true
      let whipResult: Awaited<ReturnType<typeof api.stream.whipOffer>>
      try {
        whipResult = await api.stream.whipOffer(
          offerSdp,
          Object.keys(attachConfig).length > 0 ? attachConfig : undefined,
        )
      } finally {
        connection.offerPending = false
      }
      const { answer_sdp, session_id, location, run_id } = whipResult
      // Own the server session before applying its answer. A failed answer or
      // a stop during the request must still release the returned session.
      connection.session = { sessionId: session_id, runId: run_id ?? '' }
      if (active !== connection) return
      await peer.setRemoteDescription({ type: 'answer', sdp: answer_sdp })
      if (active !== connection) return

      streamStore.setSession({
        ...connection.session,
        locationUrl: location,
        createdAt: new Date().toISOString(),
      })
      streamStore.setState('connected')
      connected = true
      connection.prevTs = Date.now()
      connection.timer = setInterval(() => { void pollMetrics(connection) }, 1000)
      peer.onconnectionstatechange = () => {
        if (active === connection && (peer.connectionState === 'failed' || peer.connectionState === 'disconnected')) {
          streamStore.setError('WebRTC connection lost')
        }
      }
    } catch (err) {
      if (active === connection) {
        streamStore.setError(err instanceof Error ? err.message : 'Stream failed')
      }
    } finally {
      if (!connected || active !== connection) {
        if (active === connection) {
          active = null
          localStream.value = null
          streamStore.setMediaStream(null)
        }
        await stopConnection(connection)
      }
    }
  }

  async function stopStream(): Promise<void> {
    const connection = active
    active = null
    localStream.value = null
    streamStore.setMediaStream(null)
    streamStore.reset()
    if (connection) await stopConnection(connection)
  }

  async function pollMetrics(connection: WhipConnection): Promise<void> {
    if (connection.polling || !connection.peer) return
    connection.polling = true
    try {
      const stats = await connection.peer.getStats()
      if (active !== connection) return
      const now = Date.now()
      const elapsed = (now - connection.prevTs) / 1000
      let bytesSent = 0
      let framesEncoded = 0
      stats.forEach(report => {
        if (report.type === 'outbound-rtp' && report.kind === 'video') {
          bytesSent += (report as RTCOutboundRtpStreamStats).bytesSent ?? 0
          framesEncoded = (report as RTCOutboundRtpStreamStats).framesEncoded ?? 0
        }
      })
      const fps = elapsed > 0 ? Math.round((framesEncoded - connection.prevFrames) / elapsed) : 0
      streamStore.updateMetrics({
        bytesTransferred: bytesSent,
        frameCount: framesEncoded,
        fps: Math.max(0, fps),
        latencyMs: 0,
      })
      connection.prevFrames = framesEncoded
      connection.prevTs = now
    } catch {
      // A closed peer can reject a pending stats request.
    } finally {
      connection.polling = false
    }
  }

  async function stopConnection(connection: WhipConnection): Promise<void> {
    connection.abort.abort()
    if (connection.timer !== null) {
      clearInterval(connection.timer)
      connection.timer = null
    }
    if (connection.peer) connection.peer.onconnectionstatechange = null
    if (connection.stream) {
      connection.stream.getTracks().forEach(track => track.stop())
      connection.stream = null
    }
    // An offer can return a durable session after stop. Its pending caller
    // retains the peer until it can stop that run before closing transport.
    if (connection.offerPending) return
    let peer = connection.peer
    connection.peer = null
    const closePeer = () => {
      peer?.close()
      peer = null
    }
    const session = connection.session
    connection.session = null
    try {
      if (session) {
        await stopDurableWhipSession(session, api.runs.stop, async sessionId => {
          closePeer()
          await api.stream.whipTerminate(sessionId)
        })
      }
    } finally {
      closePeer()
    }
  }

  onUnmounted(() => { void stopStream() })

  return { localStream, startStream, stopStream }
}
